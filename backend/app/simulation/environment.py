"""Gymnasium adapter for the FIREOPS synthetic simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.models.config import SimulationConfig
from app.models.enums import Action
from app.simulation.engine import BURNED, UNKNOWN, FireSimulation


class FireOpsEnv(gym.Env[dict[str, np.ndarray[Any, Any]], int]):
    """Gymnasium-compatible partially observable response environment.

    The observation contains only the degraded sensor frame. Aggregate metrics
    in ``info`` are evaluation outputs and should not be fed to a policy unless
    an experiment explicitly chooses to do so.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: SimulationConfig | dict[str, Any] | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.render_mode = render_mode
        self.base_config = config if isinstance(config, SimulationConfig) else SimulationConfig.model_validate(config or {})
        self._constructor_seed = int(seed if seed is not None else 42)
        self._first_reset = True
        self._episode_seed_rng = np.random.default_rng(self._constructor_seed)
        self.engine = FireSimulation(self.base_config, seed=self._constructor_seed)
        self.action_space = spaces.Discrete(len(Action))
        self._set_observation_space(self.base_config)

    def _set_observation_space(self, config: SimulationConfig) -> None:
        shape = (config.grid_size, config.grid_size)
        self.observation_space = spaces.Dict(
            {
                "fire": spaces.Box(low=UNKNOWN, high=BURNED, shape=shape, dtype=np.int8),
                "terrain": spaces.Box(low=0, high=2, shape=shape, dtype=np.int8),
                "infrastructure": spaces.Box(low=0, high=3, shape=shape, dtype=np.int8),
                "observed_mask": spaces.Box(low=0, high=1, shape=shape, dtype=np.int8),
                "wind": spaces.Box(low=UNKNOWN, high=3, shape=(1,), dtype=np.int8),
                "sensor_reliability": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "resources": spaces.Box(
                    low=0, high=max(config.initial_resources, 1_000), shape=(1,), dtype=np.int32
                ),
                # Use the validated global limit so per-episode max_steps overrides
                # never invalidate the environment's declared observation space.
                "step": spaces.Box(low=0, high=5_000, shape=(1,), dtype=np.int32),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            episode_seed = int(seed)
            self._episode_seed_rng = np.random.default_rng(episode_seed)
            self._first_reset = False
        elif self._first_reset:
            episode_seed = self._constructor_seed
            self._first_reset = False
        else:
            episode_seed = int(self._episode_seed_rng.integers(0, 2**32 - 1))

        updates: dict[str, Any] = {}
        if options:
            config_options = options.get("config", {})
            if not isinstance(config_options, dict):
                raise ValueError("options['config'] must be a mapping")
            updates.update(config_options)
            if "sensor_reliability" in options:
                updates["sensor_reliability"] = options["sensor_reliability"]
        episode_config = self.base_config.model_copy(update=updates)
        # model_copy does not revalidate; round-trip through validation for safe experiment input.
        episode_config = SimulationConfig.model_validate(episode_config.model_dump())
        if episode_config.grid_size != self.base_config.grid_size:
            raise ValueError("grid_size cannot change during reset; create a new FireOpsEnv instead")
        if episode_config != self.engine.config:
            self.engine = FireSimulation(episode_config, seed=episode_seed)
        else:
            self.engine.reset(seed=episode_seed)

        injected_failure = None
        if options and options.get("inject_failure"):
            requested = options.get("failure")
            injected_failure = self.engine.inject_chaos(requested).value

        observation = self.engine.gym_observation()
        info = self._info()
        info["seed"] = episode_seed
        if injected_failure:
            info["injected_failure"] = injected_failure
        return observation, info

    def step(
        self, action: int | Action
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        _, reward, done, _ = self.engine.step(action)
        terminated = bool(done and self.engine.termination_reason == "fire_extinguished")
        truncated = bool(done and self.engine.termination_reason == "max_steps")
        return self.engine.gym_observation(), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        observation = self.engine.observation
        return {
            "metrics": self.engine.metrics(),
            "events": list(self.engine.events[-10:]),
            "action": self.engine.current_action.name,
            "sensor": {
                "reliability": observation.effective_reliability,
                "observed_fraction": float(np.mean(observation.observed_mask)),
                "status": observation.sensor_status.value,
            },
            "termination_reason": self.engine.termination_reason,
        }

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        fire = self.engine.observation.fire
        symbols = {UNKNOWN: "?", 0: ".", 1: "F", 2: "x"}
        return "\n".join("".join(symbols[int(value)] for value in row) for row in fire)


__all__ = ["FireOpsEnv"]
