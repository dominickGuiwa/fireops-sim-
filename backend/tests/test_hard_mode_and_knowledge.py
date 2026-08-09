from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.config import SimulationConfig
from app.models.enums import Action, TrainingDifficulty, WindDirection
from app.models.schemas import ConfigUpdate, TrainResponse
from app.services.knowledge import answer_query, get_index
from app.services.session import SimulationSession
from app.simulation.engine import BURNED, BURNING, WIND_NEIGHBOURS, FireSimulation

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.dqn_core import (  # noqa: E402
    ReplayBuffer,
    flatten_observation,
    linear_epsilon,
)


client = TestClient(app)


def hard_config(**updates: Any) -> SimulationConfig:
    values: dict[str, Any] = {
        "grid_size": 7,
        "ignition_count": 1,
        "base_spread_probability": 0.0,
        "wind_strength": 0.5,
        "sensor_reliability": 1.0,
        "observation_noise": 0.0,
        "initial_resources": 5,
        "burn_duration": 20,
        "max_steps": 30,
    }
    values.update(updates)
    return SimulationConfig.model_validate(values)


def test_ember_jump_ignites_downwind_spot_fire_and_logs_event() -> None:
    engine = FireSimulation(hard_config(ember_jump_probability=0.5), seed=5)
    engine.fire_state.fill(0)
    engine.burn_age.fill(0)
    engine.fire_state[3, 1] = BURNING
    engine.fuel.fill(1.0)

    for _ in range(10):
        engine.step(Action.HOLD)
        if any(event["type"] == "ember_jump" for event in engine.events):
            break
    assert any(event["type"] == "ember_jump" for event in engine.events)
    # Wind is east by default, so spot fires appear strictly east of sources.
    burning_columns = sorted(int(col) for _, col in np.argwhere(engine.fire_state == BURNING))
    assert burning_columns[-1] >= 3


def test_wind_drift_rotates_ninety_degrees_and_logs_event() -> None:
    engine = FireSimulation(hard_config(wind_drift_probability=1.0), seed=6)
    before = engine.wind_direction
    engine.step(Action.HOLD)
    assert engine.wind_direction in WIND_NEIGHBOURS[before]
    assert any(event["type"] == "wind_drift" for event in engine.events)
    assert 0.0 <= engine.wind_strength <= 1.0


def test_observation_delay_reports_stale_fire_frames() -> None:
    engine = FireSimulation(
        hard_config(observation_delay=2, burn_duration=1), seed=7
    )
    ignition = tuple(int(value) for value in np.argwhere(engine.fire_state == BURNING)[0])
    engine.step(Action.HOLD)
    # The cell burned out in truth, but delayed sensors still report it burning.
    assert engine.fire_state[ignition] == BURNED
    observation = engine.observation
    assert observation.fire[ignition] == BURNING


def test_sensor_dropout_blanks_the_frame_and_logs_event() -> None:
    engine = FireSimulation(hard_config(sensor_dropout_probability=1.0), seed=8)
    observation = engine.refresh_observation()
    assert not observation.observed_mask.any()
    assert observation.effective_reliability == 0.0
    assert any(event["type"] == "sensor_dropout" for event in engine.events)


def test_default_config_never_enables_hard_mode_features() -> None:
    config = SimulationConfig()
    assert config.ember_jump_probability == 0.0
    assert config.wind_drift_probability == 0.0
    assert config.observation_delay == 0
    assert config.sensor_dropout_probability == 0.0
    with pytest.raises(ValidationError):
        ConfigUpdate(observation_delay=6)
    with pytest.raises(ValidationError):
        ConfigUpdate(ember_jump_probability=0.9)


def test_training_difficulty_presets_randomize_reliability() -> None:
    session = SimulationSession(hard_config(max_steps=5), seed=300)
    brutal = session.train(episodes=3, seed=41, difficulty=TrainingDifficulty.BRUTAL)
    TrainResponse.model_validate(brutal)
    assert brutal["difficulty"] is TrainingDifficulty.BRUTAL
    reliabilities = [result["sensor_reliability"] for result in brutal["results"]]
    assert all(0.25 <= value <= 0.9 for value in reliabilities)
    assert len(set(reliabilities)) > 1

    standard = session.train(episodes=2, seed=42, difficulty=TrainingDifficulty.STANDARD)
    assert all(
        result["sensor_reliability"] == session.config.sensor_reliability
        for result in standard["results"]
    )
    # The interactive engine still runs the untouched baseline configuration.
    assert session.engine.config.ember_jump_probability == 0.0


def test_train_endpoint_accepts_difficulty_and_rejects_unknown() -> None:
    response = client.post("/api/train", json={"episodes": 2, "difficulty": "hard"})
    assert response.status_code == 200
    payload = TrainResponse.model_validate(response.json())
    assert payload.difficulty is TrainingDifficulty.HARD
    assert "hard" in payload.note

    invalid = client.post("/api/train", json={"episodes": 2, "difficulty": "impossible"})
    assert invalid.status_code == 422


def test_knowledge_retrieval_finds_relevant_chunks_with_citations() -> None:
    payload = answer_query("how does wind affect fire spread", k=3)
    assert payload["results"]
    assert payload["results"][0]["title"] == "Wind and fire spread"
    assert 0.0 < payload["results"][0]["score"] <= 1.0
    assert payload["answer"] is not None
    assert payload["answer"] in payload["results"][0]["text"]
    assert "Extractive" in payload["note"]

    empty = answer_query("zzzz qqqq", k=3)
    assert empty["results"] == []
    assert empty["answer"] is None


def test_knowledge_endpoint_validates_query_parameters() -> None:
    response = client.get("/api/knowledge", params={"query": "sensor dropout"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    titles = [result["title"] for result in body["results"]]
    assert any("Hard-mode" in title or "Sensors" in title for title in titles)

    assert client.get("/api/knowledge", params={"query": "a"}).status_code == 422
    assert (
        client.get("/api/knowledge", params={"query": "wind", "k": 9}).status_code == 422
    )
    assert client.get("/api/knowledge").status_code == 422


def test_knowledge_index_scores_are_normalized() -> None:
    index = get_index()
    for chunk in index.corpus:
        results = index.search(chunk["title"], k=1)
        assert results
        assert results[0]["score"] <= 1.0


def test_policy_registration_prototype_returns_501_and_app_survives() -> None:
    response = client.post("/api/policies/register")
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()
    assert "act(observation)" in response.json()["detail"]
    # "Program still in program": the simulator keeps serving normally.
    assert client.get("/api/state").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_dqn_replay_buffer_ring_semantics_and_seeded_sampling() -> None:
    def transition(value: float) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
        state = np.full(4, value, dtype=np.float32)
        return state, int(value) % 6, value, state + 1.0, False

    buffer = ReplayBuffer(capacity=3, seed=11)
    for value in range(5):
        buffer.push(*transition(float(value)))
    assert len(buffer) == 3

    twin = ReplayBuffer(capacity=3, seed=11)
    for value in range(5):
        twin.push(*transition(float(value)))
    states_a, actions_a, rewards_a, _, _ = buffer.sample(2)
    states_b, actions_b, rewards_b, _, _ = twin.sample(2)
    np.testing.assert_array_equal(states_a, states_b)
    np.testing.assert_array_equal(actions_a, actions_b)
    np.testing.assert_array_equal(rewards_a, rewards_b)

    with pytest.raises(ValueError, match="not enough transitions"):
        buffer.sample(10)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)


def test_dqn_epsilon_schedule_and_observation_flattening() -> None:
    assert linear_epsilon(0) == 1.0
    assert linear_epsilon(10_000) == pytest.approx(0.05)
    assert linear_epsilon(20_000) == pytest.approx(0.05)
    values = [linear_epsilon(step) for step in range(0, 10_001, 1_000)]
    assert values == sorted(values, reverse=True)
    with pytest.raises(ValueError):
        linear_epsilon(1, decay_steps=0)

    engine = FireSimulation(hard_config(), seed=12)
    features = flatten_observation(engine.gym_observation())
    grid_cells = engine.config.grid_size**2
    assert features.shape == (4 * grid_cells + 4,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_wind_direction_neighbours_never_reverse() -> None:
    opposites = {
        WindDirection.NORTH: WindDirection.SOUTH,
        WindDirection.SOUTH: WindDirection.NORTH,
        WindDirection.EAST: WindDirection.WEST,
        WindDirection.WEST: WindDirection.EAST,
    }
    for direction, choices in WIND_NEIGHBOURS.items():
        assert opposites[direction] not in choices
        assert direction not in choices
