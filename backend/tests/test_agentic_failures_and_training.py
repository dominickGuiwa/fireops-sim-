from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agents import QLearningAgent
from app.main import app
from app.models.config import SimulationConfig
from app.models.enums import (
    Action,
    AgentFailureType,
    ChaosFailure,
    LearningEpisodeSource,
    PolicyName,
    TrialOutcome,
)
from app.models.schemas import LearningStatsResponse, TrainResponse
from app.services.failure_analysis import detect_step_failures, sector_burning_counts
from app.services.session import SimulationSession
from app.simulation.engine import BURNING, UNBURNED, UNKNOWN, FireSimulation


client = TestClient(app)


def small_config(**updates: Any) -> SimulationConfig:
    values: dict[str, Any] = {
        "grid_size": 5,
        "ignition_count": 1,
        "base_spread_probability": 0.0,
        "wind_strength": 0.0,
        "sensor_reliability": 1.0,
        "observation_noise": 0.0,
        "initial_resources": 3,
        "burn_duration": 20,
        "max_steps": 5,
    }
    values.update(updates)
    return SimulationConfig.model_validate(values)


def test_q_update_math_epsilon_decay_and_history_are_exact() -> None:
    agent = QLearningAgent(seed=3, learning_rate=0.5, discount=0.9)
    state = ("north", "clear", "ok")
    next_state = ("none", "clear", "ok")
    agent.q_table[next_state] = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])

    agent.learn(state, Action.PROTECT_NORTH, reward=-1.0, next_state=next_state, done=False)
    # target = -1 + 0.9 * max(next) = 0.8; q = 0 + 0.5 * (0.8 - 0) = 0.4
    assert agent.q_table[state][int(Action.PROTECT_NORTH)] == pytest.approx(0.4)

    agent.learn(state, Action.PROTECT_NORTH, reward=-1.0, next_state=next_state, done=True)
    # Terminal target = -1; q = 0.4 + 0.5 * (-1 - 0.4) = -0.3
    assert agent.q_table[state][int(Action.PROTECT_NORTH)] == pytest.approx(-0.3)
    assert agent.updates_applied == 2

    epsilon_before = agent.epsilon
    agent.finish_episode(-2.5, TrialOutcome.PARTIAL, LearningEpisodeSource.INTERACTIVE)
    assert agent.episodes_trained == 1
    assert agent.epsilon == pytest.approx(epsilon_before * 0.93)
    record = agent.reward_history[-1]
    assert record["reward"] == pytest.approx(-2.5)
    assert record["epsilon"] == pytest.approx(epsilon_before)
    assert record["source"] is LearningEpisodeSource.INTERACTIVE


def test_headless_training_actually_learns_to_stop_wasting_resources() -> None:
    session = SimulationSession(small_config(), seed=200)
    before = session.learning_stats()
    assert before["episodes_trained"] == 0
    assert before["improvement"] is None

    payload = session.train(episodes=60, seed=555)
    TrainResponse.model_validate(payload)
    assert payload["episodes_run"] == 60
    assert len(payload["results"]) == 60
    stats = payload["stats"]
    assert stats["episodes_trained"] == 60
    assert stats["epsilon"] < before["epsilon"]
    assert stats["updates_applied"] >= 60
    assert stats["states_visited"] >= 1
    assert all(
        record["source"] is LearningEpisodeSource.TRAINING
        for record in stats["reward_history"]
    )
    assert stats["improvement"] is not None

    # With zero spread and a fire that outlives max_steps, every action's only
    # effect is its resource penalty, so the greedy learned move must be HOLD.
    probe = FireSimulation(small_config(), seed=556)
    agent = session.agents[PolicyName.LEARNING]
    agent.epsilon = 0.0
    assert Action(agent.act(probe.gym_observation())) is Action.HOLD

    # Training must not touch the interactive engine.
    assert session.engine.step_count == 0
    assert not session.engine.done


def test_learning_policy_survives_reset_and_records_interactive_episodes() -> None:
    session = SimulationSession(small_config(), seed=210)
    session.train(episodes=5, seed=99)
    trained_before_reset = session.learning_stats()["episodes_trained"]

    session.reset(seed=211, policy=PolicyName.LEARNING)
    assert session.learning_stats()["episodes_trained"] == trained_before_reset

    state = session.step()
    decision = state.agent_decision
    assert decision is not None
    assert decision.explored is not None
    assert decision.exploration_rate is not None
    assert decision.q_values is not None
    assert "learning policy" in decision.reason.lower()

    while not state.done:
        state = session.step()
    assert state.learning.episodes_trained == trained_before_reset + 1
    assert state.learning.reward_history[-1].source is LearningEpisodeSource.INTERACTIVE
    assert any(event.type == "learning" for event in state.events)

    catalog = session.policies()
    learning_entry = next(
        policy for policy in catalog.policies if policy.id is PolicyName.LEARNING
    )
    assert str(trained_before_reset + 1) in learning_entry.description


def test_manual_step_with_learning_policy_updates_table_and_says_so() -> None:
    session = SimulationSession(small_config(), seed=220)
    session.reset(seed=221, policy=PolicyName.LEARNING)
    updates_before = session.learning_stats()["updates_applied"]
    state = session.step(action=Action.HOLD)
    assert session.learning_stats()["updates_applied"] == updates_before + 1
    assert state.agent_decision is not None
    assert "off-policy" in state.agent_decision.reason


def test_detect_idle_perception_and_exhaustion_failures_in_session() -> None:
    session = SimulationSession(small_config(), seed=41)
    session.chaos(ChaosFailure.SENSOR_OFFLINE)
    state = session.step(action=Action.HOLD)
    kinds = {failure.failure_type for failure in state.agent_failures}
    assert AgentFailureType.IDLE_WHILE_BURNING in kinds
    assert AgentFailureType.PERCEPTION_GAP in kinds
    assert any(event.type == "agent_failure" for event in state.events)

    exhausted = SimulationSession(small_config(initial_resources=0), seed=42)
    exhausted_state = exhausted.step(action=Action.PROTECT_NORTH)
    kinds = {failure.failure_type for failure in exhausted_state.agent_failures}
    assert AgentFailureType.RESOURCE_EXHAUSTED_REQUEST in kinds


def test_misdirected_protection_uses_unique_true_fire_maximum() -> None:
    session = SimulationSession(small_config(), seed=43)
    session.engine.fire_state.fill(UNBURNED)
    session.engine.burn_age.fill(0)
    session.engine.fire_state[0, 0] = BURNING
    session.engine.fire_state[0, 4] = BURNING
    session.engine.refresh_observation()
    counts = sector_burning_counts(session.engine.fire_state)
    assert counts == {"north": 2, "south": 0, "east": 1, "west": 1}

    state = session.step(action=Action.PROTECT_SOUTH)
    kinds = {failure.failure_type for failure in state.agent_failures}
    assert AgentFailureType.MISDIRECTED_PROTECTION in kinds
    misdirected = next(
        failure
        for failure in state.agent_failures
        if failure.failure_type is AgentFailureType.MISDIRECTED_PROTECTION
    )
    assert misdirected.true_burning_cells == 2
    assert "south" in misdirected.message and "north" in misdirected.message


def test_wind_misread_flags_wrong_reading_but_not_unknown() -> None:
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[0, 0] = BURNING
    mask = np.ones((5, 5), dtype=np.int8)
    common = {
        "step": 3,
        "policy": "rule",
        "action": Action.PROTECT_NORTH,
        "true_fire_before": grid,
        "observed_fire_before": grid,
        "observed_mask_before": mask,
        "true_wind_code": 0,
        "resources_before": 2,
    }
    wrong = detect_step_failures(observed_wind_before=2, **common)
    assert AgentFailureType.WIND_MISREAD in {item["failure_type"] for item in wrong}
    unknown = detect_step_failures(observed_wind_before=UNKNOWN, **common)
    assert AgentFailureType.WIND_MISREAD not in {item["failure_type"] for item in unknown}


def test_run_summary_failure_counts_and_coaching_line() -> None:
    session = SimulationSession(small_config(max_steps=2), seed=44)
    session.step(action=Action.HOLD)
    state = session.step(action=Action.HOLD)
    assert state.done
    assert state.run_summary is not None
    counts = state.run_summary.failure_counts
    assert counts[AgentFailureType.IDLE_WHILE_BURNING] == 2
    assert sum(counts.values()) == len(state.agent_failures)
    assert any("Failure analysis" in line for line in state.run_summary.improvements)


def test_train_and_learning_endpoints_return_valid_payloads() -> None:
    reset = client.post(
        "/api/reset",
        json={
            "seed": 7001,
            "policy": "learning",
            "config": {
                "grid_size": 5,
                "base_spread_probability": 0.0,
                "sensor_reliability": 1.0,
                "observation_noise": 0.0,
                "burn_duration": 20,
                "max_steps": 5,
            },
        },
    )
    assert reset.status_code == 200

    trained = client.post("/api/train", json={"episodes": 3, "seed": 12})
    assert trained.status_code == 200
    payload = TrainResponse.model_validate(trained.json())
    assert payload.episodes_run == 3
    assert len(payload.results) == 3
    assert payload.stats.episodes_trained >= 3

    stats_response = client.get("/api/learning")
    assert stats_response.status_code == 200
    stats = LearningStatsResponse.model_validate(stats_response.json())
    assert stats.episodes_trained == payload.stats.episodes_trained

    state_response = client.get("/api/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert "agent_failures" in state_payload
    assert state_payload["learning"]["episodes_trained"] == stats.episodes_trained
    assert any(event["type"] == "training" for event in state_payload["events"])

    invalid = client.post("/api/train", json={"episodes": 0})
    assert invalid.status_code == 422
