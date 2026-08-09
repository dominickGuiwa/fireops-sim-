from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.agents import PPOAgent
from app.models.config import SimulationConfig
from app.models.enums import Action, ChaosFailure, PolicyName
from app.models.schemas import ConfigUpdate, StateResponse
from app.services.session import SimulationSession
from app.simulation.engine import BURNING, UNBURNED


def learning_config(**updates: Any) -> SimulationConfig:
    values: dict[str, Any] = {
        "grid_size": 5,
        "ignition_count": 1,
        "base_spread_probability": 0.0,
        "wind_strength": 0.0,
        "sensor_reliability": 1.0,
        "observation_noise": 0.0,
        "initial_resources": 3,
        "burn_duration": 1,
        "max_steps": 5,
    }
    values.update(updates)
    return SimulationConfig.model_validate(values)


def test_initial_learning_state_and_abandoned_trial_semantics() -> None:
    session = SimulationSession(learning_config(), seed=1)

    initial = session.state()
    assert initial.agent_decision is None
    assert initial.run_summary is None
    assert initial.last_reward == 0.0
    assert initial.max_steps == 5
    assert initial.trial_stats.model_dump() == {
        "scope": "server_session",
        "trials_started": 1,
        "trials_completed": 0,
        "successes": 0,
        "partials": 0,
        "needs_improvement": 0,
        "total_experience_points": 0,
        "current_level": 1,
        "average_completed_reward": 0.0,
    }

    for _ in range(5):
        assert session.state().trial_stats == initial.trial_stats

    reset = session.reset(seed=2)
    assert reset.trial_stats.trials_started == 2
    assert reset.trial_stats.trials_completed == 0
    configured = session.update_config(ConfigUpdate(max_steps=2))
    assert configured.trial_stats.trials_started == 3
    assert configured.trial_stats.trials_completed == 0
    assert configured.run_summary is None


def test_manual_decision_uses_pre_action_observation_and_reward_audit() -> None:
    session = SimulationSession(
        learning_config(sensor_reliability=0.25, burn_duration=5),
        seed=31,
    )
    before = session.state()
    visible_before = sum(cell.observed and cell.fire.value == "burning" for cell in before.cells)
    hidden_before = sum(not cell.observed for cell in before.cells)

    state = session.step(action=Action.PROTECT_EAST)
    decision = state.agent_decision
    assert decision is not None
    assert decision.chosen_action.value == "PROTECT_EAST"
    assert decision.action_label == "Protect east"
    assert decision.decision_source.value == "manual"
    assert "person chose" in decision.reason.lower()
    assert "no policy chose it" in decision.reason.lower()
    assert decision.visible_fire_count == visible_before
    assert decision.hidden_cell_count == hidden_before
    assert decision.resources_before_action == before.resources.available
    assert decision.reward_breakdown.resource_use_penalty == -0.3
    breakdown = decision.reward_breakdown
    component_total = (
        breakdown.new_spread_penalty
        + breakdown.infrastructure_loss_penalty
        + breakdown.resource_use_penalty
        + breakdown.active_fire_penalty
    )
    assert breakdown.total == pytest.approx(component_total)
    assert decision.reward_change == pytest.approx(breakdown.total)
    assert state.last_reward == pytest.approx(breakdown.total)
    assert state.metrics.cumulative_reward == pytest.approx(state.last_reward)


def test_policy_explanations_are_honest_about_random_ppo_and_fallback() -> None:
    empty_session = SimulationSession(
        learning_config(initial_resources=0, burn_duration=5),
        seed=9,
    )
    empty_state = empty_session.step(policy=PolicyName.RULE)
    assert empty_state.agent_decision is not None
    assert empty_state.agent_decision.chosen_action.value == "HOLD"
    assert "no response resources" in empty_state.agent_decision.reason

    hidden_session = SimulationSession(learning_config(burn_duration=5), seed=9)
    hidden_session.chaos(ChaosFailure.SENSOR_OFFLINE)
    hidden_state = hidden_session.step(policy=PolicyName.RULE)
    assert hidden_state.agent_decision is not None
    assert hidden_state.agent_decision.chosen_action.value == "RECON"
    assert hidden_state.agent_decision.visible_fire_count == 0
    assert hidden_state.agent_decision.hidden_cell_count == 25
    assert "no visible burning cells" in hidden_state.agent_decision.reason

    random_session = SimulationSession(learning_config(burn_duration=5), seed=9)
    random_state = random_session.step(policy=PolicyName.RANDOM)
    assert random_state.agent_decision is not None
    assert "by chance" in random_state.agent_decision.reason
    assert "hidden fire values" in random_state.agent_decision.reason
    assert "learn" in random_state.agent_decision.reason

    fallback_session = SimulationSession(learning_config(burn_duration=5), seed=9)
    fallback_session.agents[PolicyName.PPO_NOMINAL] = PPOAgent()
    fallback_state = fallback_session.step(policy=PolicyName.PPO_NOMINAL)
    assert fallback_state.agent_decision is not None
    assert "rule fallback was used" in fallback_state.agent_decision.reason
    assert "hidden fire values" in fallback_state.agent_decision.reason
    assert "learn online" in fallback_state.agent_decision.reason

    class AcceptingSpace:
        @staticmethod
        def contains(_observation: object) -> bool:
            return True

    class SavedModel:
        observation_space = AcceptingSpace()

        @staticmethod
        def predict(_observation: object, deterministic: bool = True) -> tuple[int, None]:
            assert deterministic
            return int(Action.HOLD), None

    native_session = SimulationSession(learning_config(burn_duration=5), seed=9)
    native_ppo = PPOAgent()
    native_ppo.model = SavedModel()
    native_session.agents[PolicyName.PPO_ROBUST] = native_ppo
    native_state = native_session.step(policy=PolicyName.PPO_ROBUST)
    assert native_state.agent_decision is not None
    assert "saved PPO model" in native_state.agent_decision.reason
    assert "opaque" in native_state.agent_decision.reason
    assert "does not learn" in native_state.agent_decision.reason
    assert "fallback" not in native_state.agent_decision.reason


def test_success_partial_and_needs_improvement_boundaries_count_once() -> None:
    session = SimulationSession(learning_config(), seed=71)

    success = session.step(action=Action.HOLD)
    assert success.run_summary is not None
    assert success.run_summary.fire_extinguished
    assert success.run_summary.mission_completed
    assert success.run_summary.success
    assert success.run_summary.outcome.value == "success"
    assert success.run_summary.termination_reason.value == "fire_extinguished"
    assert success.run_summary.xp_earned == 100
    assert success.run_summary.action_counts.hold == 1
    assert success.run_summary.final_metrics == success.metrics
    assert success.trial_stats.trials_completed == 1
    assert success.trial_stats.successes == 1
    success_id = success.run_summary.trial_id
    success_reward = success.metrics.cumulative_reward

    for _ in range(8):
        cached = session.state()
        assert cached.run_summary is not None
        assert cached.run_summary.trial_id == success_id
        assert cached.trial_stats == success.trial_stats
    with pytest.raises(RuntimeError, match="reset before stepping"):
        session.step(action=Action.HOLD)
    assert session.state().trial_stats == success.trial_stats

    reset = session.reset(seed=72)
    assert reset.run_summary is None
    assert reset.trial_stats.trials_started == 2
    assert reset.trial_stats.trials_completed == 1
    session.engine.fire_state.fill(UNBURNED)
    session.engine.burn_age.fill(0)
    site = next(iter(session.engine.infrastructure_locations))
    session.engine.fire_state[site] = BURNING
    session.engine.refresh_observation()

    partial = session.step(action=Action.HOLD)
    assert partial.run_summary is not None
    assert partial.run_summary.fire_extinguished
    assert not partial.run_summary.mission_completed
    assert not partial.run_summary.success
    assert partial.run_summary.outcome.value == "partial"
    assert partial.run_summary.termination_reason.value == "fire_extinguished"
    assert partial.run_summary.final_metrics.infrastructure_lost == 1
    assert partial.run_summary.xp_earned == 50
    assert partial.run_summary.trial_id != success_id
    assert partial.trial_stats.trials_completed == 2
    assert partial.trial_stats.partials == 1
    assert partial.trial_stats.total_experience_points == 150
    partial_reward = partial.metrics.cumulative_reward

    next_trial = session.update_config(ConfigUpdate(burn_duration=5, max_steps=1))
    assert next_trial.run_summary is None
    assert next_trial.trial_stats.trials_started == 3
    needs_work = session.step(action=Action.HOLD)
    assert needs_work.run_summary is not None
    assert not needs_work.run_summary.fire_extinguished
    assert not needs_work.run_summary.mission_completed
    assert not needs_work.run_summary.success
    assert needs_work.run_summary.outcome.value == "needs_improvement"
    assert needs_work.run_summary.termination_reason.value == "max_steps"
    assert needs_work.step_count == needs_work.max_steps == 1
    assert needs_work.run_summary.xp_earned == 20
    assert needs_work.trial_stats.trials_completed == 3
    assert needs_work.trial_stats.successes == 1
    assert needs_work.trial_stats.partials == 1
    assert needs_work.trial_stats.needs_improvement == 1
    assert needs_work.trial_stats.total_experience_points == 170
    assert needs_work.trial_stats.current_level == 1
    expected_average = round(
        (success_reward + partial_reward + needs_work.metrics.cumulative_reward) / 3,
        4,
    )
    assert needs_work.trial_stats.average_completed_reward == pytest.approx(expected_average)


def test_learning_response_models_forbid_extra_and_coerced_nested_fields() -> None:
    session = SimulationSession(learning_config(), seed=91)
    completed = session.step(action=Action.HOLD)
    payload = completed.model_dump(mode="json")
    payload["trial_stats"]["global_total"] = 99
    with pytest.raises(ValidationError):
        StateResponse.model_validate(payload)

    payload = completed.model_dump(mode="json")
    payload["trial_stats"]["trials_started"] = True
    with pytest.raises(ValidationError):
        StateResponse.model_validate(payload)

    payload = completed.model_dump(mode="json")
    payload["run_summary"]["action_counts"]["hold"] = "1"
    with pytest.raises(ValidationError):
        StateResponse.model_validate(payload)

    payload = completed.model_dump(mode="json")
    payload["run_summary"]["final_metrics"]["burned_area"] = "1"
    with pytest.raises(ValidationError):
        StateResponse.model_validate(payload)


def test_rejected_steps_do_not_change_policy_or_completed_trial_stats() -> None:
    session = SimulationSession(learning_config(burn_duration=5), seed=101)
    before = session.state()
    with pytest.raises(ValueError, match="Unknown action"):
        session.step(action="BAD", policy=PolicyName.RANDOM)
    rejected = session.state()
    assert rejected.policy is PolicyName.RULE
    assert rejected.step_count == before.step_count
    assert rejected.agent_decision is None
    assert rejected.trial_stats == before.trial_stats

    completed_session = SimulationSession(learning_config(), seed=102)
    completed = completed_session.step(action=Action.HOLD)
    with pytest.raises(RuntimeError, match="reset before stepping"):
        completed_session.step(policy=PolicyName.RANDOM)
    after_completed_rejection = completed_session.state()
    assert after_completed_rejection.policy is PolicyName.RULE
    assert after_completed_rejection.run_summary == completed.run_summary
    assert after_completed_rejection.trial_stats == completed.trial_stats
