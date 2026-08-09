from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import PoliciesResponse, StateResponse


client = TestClient(app)


def assert_valid_state(response_data: dict) -> StateResponse:
    state = StateResponse.model_validate(response_data)
    assert len(state.cells) == state.grid_size**2
    assert state.resources.available >= 0
    assert state.metrics.infrastructure_total == 3
    assert state.trial_stats.trials_started >= 1
    assert state.trial_stats.trials_completed <= state.trial_stats.trials_started
    return state


def test_health_and_default_state_are_public_20_by_20() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "fireops-sim", "version": "0.1.0"}

    client.post(
        "/api/reset",
        json={
            "seed": 42,
            "sensor_reliability": 1.0,
            "config": {"grid_size": 20, "observation_noise": 0.0},
        },
    )
    response = client.get("/api/state")
    assert response.status_code == 200
    state = assert_valid_state(response.json())
    assert state.grid_size == 20
    assert state.step_count == 0
    assert state.max_steps >= 1
    assert state.agent_decision is None
    assert state.run_summary is None
    assert state.last_reward == 0.0
    assert state.wind.direction.value == "east"
    assert state.wind.observed_direction is not None


def test_reset_and_step_return_valid_schema_and_deterministic_state() -> None:
    request = {"seed": 1234, "policy": "rule", "sensor_reliability": 1.0}
    first = client.post("/api/reset", json=request)
    second = client.post("/api/reset", json=request)
    assert first.status_code == second.status_code == 200
    first_data = first.json()
    second_data = second.json()
    first_stats = first_data.pop("trial_stats")
    second_stats = second_data.pop("trial_stats")
    assert first_data == second_data
    assert second_stats["trials_started"] == first_stats["trials_started"] + 1
    assert {
        key: value for key, value in second_stats.items() if key != "trials_started"
    } == {key: value for key, value in first_stats.items() if key != "trials_started"}

    step = client.post("/api/step", json={"action": "PROTECT_EAST"})
    assert step.status_code == 200
    state = assert_valid_state(step.json())
    assert state.step_count == 1
    assert state.current_action == "PROTECT_EAST"
    assert state.resources.consumed == 1
    assert state.agent_decision is not None
    assert state.agent_decision.chosen_action.value == "PROTECT_EAST"
    assert state.agent_decision.decision_source.value == "manual"
    assert "person chose" in state.agent_decision.reason.lower()
    assert state.agent_decision.reward_change == state.last_reward
    assert state.agent_decision.reward_breakdown.total == state.last_reward


def test_policy_driven_step_and_policy_catalog() -> None:
    client.post("/api/reset", json={"seed": 90, "policy": "random"})
    step = client.post("/api/step", json={"policy": "rule"})
    assert step.status_code == 200
    state = assert_valid_state(step.json())
    assert state.policy.value == "rule"

    response = client.get("/api/policies")
    assert response.status_code == 200
    policies = PoliciesResponse.model_validate(response.json())
    assert {policy.id.value for policy in policies.policies} == {
        "random",
        "rule",
        "learning",
        "ppo_nominal",
        "ppo_robust",
    }
    assert [action.value for action in policies.actions] == list(range(6))


def test_chaos_and_config_endpoints() -> None:
    client.post("/api/reset", json={"seed": 5})
    chaos = client.post("/api/chaos", json={"failure": "sensor_offline"})
    assert chaos.status_code == 200
    chaos_state = assert_valid_state(chaos.json())
    assert chaos_state.sensor.status.value == "offline"
    assert chaos_state.sensor.observed_fraction == 0.0
    assert all(cell.fire.value == "unknown" for cell in chaos_state.cells)
    assert all(cell.true_fire.value != "unknown" for cell in chaos_state.cells)
    assert any(cell.true_fire.value == "burning" for cell in chaos_state.cells)
    assert chaos_state.events[-1].type == "chaos"

    configured = client.put(
        "/api/config",
        json={"sensor_reliability": 0.25, "wind_direction": "north", "initial_resources": 4},
    )
    assert configured.status_code == 200
    configured_state = assert_valid_state(configured.json())
    assert configured_state.step_count == 0
    assert configured_state.sensor.configured_reliability == 0.25
    assert configured_state.wind.direction.value == "north"
    assert configured_state.resources.initial == 4


def test_api_rejects_invalid_actions_and_configuration() -> None:
    client.post("/api/reset", json={"seed": 2, "policy": "rule"})
    rejected_step = client.post(
        "/api/step",
        json={"action": "NOT_AN_ACTION", "policy": "random"},
    )
    assert rejected_step.status_code == 422
    assert client.get("/api/state").json()["policy"] == "rule"
    assert client.post("/api/step", json={"action": True}).status_code == 422
    assert client.post("/api/step", json={"action": 3.0}).status_code == 422
    assert client.post("/api/reset", json={"seed": True}).status_code == 422
    assert client.post("/api/reset", json={"sensor_reliability": "0.5"}).status_code == 422
    assert client.put("/api/config", json={"sensor_reliability": 0.1}).status_code == 422
    assert client.put("/api/config", json={"initial_resources": "4"}).status_code == 422
    assert client.post("/api/reset", json={"unexpected": True}).status_code == 422


def test_cors_allows_vite_development_origin() -> None:
    response = client.options(
        "/api/state",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_completed_trial_api_contract_is_cached_and_counted_once() -> None:
    reset = client.post(
        "/api/reset",
        json={
            "seed": 808,
            "policy": "rule",
            "config": {
                "grid_size": 5,
                "base_spread_probability": 0.0,
                "sensor_reliability": 1.0,
                "observation_noise": 0.0,
                "burn_duration": 5,
                "max_steps": 1,
            },
        },
    )
    assert reset.status_code == 200
    before = assert_valid_state(reset.json())
    assert before.run_summary is None

    completed_response = client.post("/api/step", json={"action": "HOLD"})
    assert completed_response.status_code == 200
    completed_data = completed_response.json()
    completed = assert_valid_state(completed_data)
    assert completed.done
    assert set(completed_data) >= {
        "max_steps",
        "last_reward",
        "agent_decision",
        "trial_stats",
        "run_summary",
    }
    assert set(completed_data["agent_decision"]) == {
        "chosen_action",
        "action_label",
        "decision_source",
        "reason",
        "visible_fire_count",
        "hidden_cell_count",
        "resources_before_action",
        "reward_change",
        "reward_breakdown",
        "explored",
        "exploration_rate",
        "q_values",
    }
    assert set(completed_data["trial_stats"]) == {
        "scope",
        "trials_started",
        "trials_completed",
        "successes",
        "partials",
        "needs_improvement",
        "total_experience_points",
        "current_level",
        "average_completed_reward",
    }
    assert set(completed_data["run_summary"]) == {
        "trial_id",
        "fire_extinguished",
        "mission_completed",
        "success",
        "outcome",
        "title",
        "what_happened",
        "reward_lesson",
        "improvements",
        "xp_earned",
        "action_counts",
        "termination_reason",
        "final_metrics",
        "failure_counts",
    }
    assert completed.run_summary is not None
    assert not completed.run_summary.fire_extinguished
    assert not completed.run_summary.mission_completed
    assert completed.run_summary.outcome.value == "needs_improvement"
    assert completed.run_summary.termination_reason.value == "max_steps"
    assert completed.trial_stats.trials_completed == before.trial_stats.trials_completed + 1

    for _ in range(4):
        cached = assert_valid_state(client.get("/api/state").json())
        assert cached.run_summary == completed.run_summary
        assert cached.trial_stats == completed.trial_stats
