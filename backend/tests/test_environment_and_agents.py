from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
from pytest import MonkeyPatch

from app.agents import PPOAgent, RandomAgent, RuleBasedAgent
from app.models.config import SimulationConfig
from app.models.enums import Action
from app.simulation.engine import BURNING, UNKNOWN
from app.simulation.environment import FireOpsEnv


def test_gymnasium_contract_and_spaces() -> None:
    env = FireOpsEnv(seed=10)
    observation, info = env.reset(seed=10)

    assert env.action_space.n == 6
    assert env.observation_space.contains(observation)
    assert observation["fire"].shape == (20, 20)
    assert "metrics" in info

    observation, reward, terminated, truncated, info = env.step(Action.HOLD)
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cumulative_reward" in info["metrics"]


def test_gym_reset_is_deterministic_and_supports_robust_options() -> None:
    env_a = FireOpsEnv(seed=1)
    env_b = FireOpsEnv(seed=999)
    obs_a, info_a = env_a.reset(seed=123, options={"sensor_reliability": 0.25, "inject_failure": True})
    obs_b, info_b = env_b.reset(seed=123, options={"sensor_reliability": 0.25, "inject_failure": True})

    for key in obs_a:
        assert np.array_equal(obs_a[key], obs_b[key])
    assert info_a["injected_failure"] == info_b["injected_failure"]
    assert obs_a["sensor_reliability"][0] <= 0.95


def test_gym_reset_cannot_change_the_declared_observation_shape() -> None:
    env = FireOpsEnv(seed=1)
    declared_space = env.observation_space

    with np.testing.assert_raises_regex(ValueError, "grid_size cannot change"):
        env.reset(options={"config": {"grid_size": 5}})

    observation, _ = env.reset(seed=2)
    assert env.observation_space is declared_space
    assert declared_space.contains(observation)


def test_gym_max_steps_override_stays_inside_declared_space() -> None:
    env = FireOpsEnv(
        SimulationConfig(max_steps=1, burn_duration=5, base_spread_probability=0.0),
        seed=3,
    )
    declared_space = env.observation_space
    env.reset(seed=3, options={"config": {"max_steps": 2}})

    observation, _, _, _, _ = env.step(Action.HOLD)
    assert declared_space.contains(observation)
    observation, _, _, truncated, _ = env.step(Action.HOLD)
    assert truncated
    assert declared_space.contains(observation)


def test_rule_agent_uses_partial_observation_and_resource_limit() -> None:
    fire = np.full((5, 5), UNKNOWN, dtype=np.int8)
    observation = {"fire": fire, "resources": np.array([2], dtype=np.int32)}
    agent = RuleBasedAgent()
    assert agent.act(observation) == Action.RECON

    fire[0, 2] = BURNING
    assert agent.act(observation) == Action.PROTECT_NORTH

    observation["resources"] = np.array([0], dtype=np.int32)
    assert agent.act(observation) == Action.HOLD


def test_random_agent_is_seeded_and_ppo_has_rule_fallback() -> None:
    first = RandomAgent(seed=44)
    second = RandomAgent(seed=44)
    observation = {"fire": np.zeros((5, 5), dtype=np.int8), "resources": np.array([1])}
    assert [first.act(observation) for _ in range(12)] == [second.act(observation) for _ in range(12)]

    ppo = PPOAgent()
    assert not ppo.native_available
    assert ppo.act(observation) == Action.RECON


def test_ppo_incompatible_observation_space_falls_back_safely() -> None:
    class RejectingSpace:
        @staticmethod
        def contains(_observation: object) -> bool:
            return False

    class IncompatibleModel:
        observation_space = RejectingSpace()

        @staticmethod
        def predict(_observation: object, deterministic: bool = True) -> tuple[int, None]:
            raise AssertionError(f"predict should not run: deterministic={deterministic}")

    observation = {"fire": np.zeros((5, 5), dtype=np.int8), "resources": np.array([1])}
    ppo = PPOAgent()
    ppo.model = IncompatibleModel()

    assert ppo.act(observation) == Action.RECON
    assert not ppo.native_available
    assert "incompatible" in (ppo.load_error or "")


def test_ppo_corrupt_model_load_uses_fallback(monkeypatch: MonkeyPatch) -> None:
    class BrokenPPO:
        @staticmethod
        def load(_path: str) -> object:
            raise RuntimeError("synthetic corrupt model")

    fake_sb3 = ModuleType("stable_baselines3")
    setattr(fake_sb3, "PPO", BrokenPPO)
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)

    ppo = PPOAgent("synthetic-corrupt.zip")

    assert not ppo.native_available
    assert "synthetic corrupt model" in (ppo.load_error or "")


def test_max_steps_is_reported_as_truncation() -> None:
    env = FireOpsEnv(
        SimulationConfig(max_steps=1, burn_duration=5, base_spread_probability=0.0), seed=3
    )
    env.reset(seed=3)
    _, _, terminated, truncated, info = env.step(Action.HOLD)
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "max_steps"
