"""Optional Stable-Baselines3 PPO adapter with a functional rule fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.agents.base import Agent
from app.agents.rule_agent import RuleBasedAgent
from app.models.enums import Action


class PPOAgent:
    """Load an SB3 PPO model lazily when one is supplied.

    Stable-Baselines3 and PyTorch are optional heavyweight dependencies. The
    API policy remains usable in the base install by explicitly reporting that
    it is using the deterministic rule fallback.
    """

    def __init__(self, model_path: str | Path | None = None, fallback: Agent | None = None) -> None:
        self.model: Any | None = None
        self.model_path = Path(model_path) if model_path else None
        self.fallback = fallback or RuleBasedAgent()
        self.load_error: str | None = None
        if self.model_path is not None:
            self.load(self.model_path)

    @property
    def native_available(self) -> bool:
        return self.model is not None

    def load(self, model_path: str | Path) -> bool:
        self.model_path = Path(model_path)
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(self.model_path))
            action_count = getattr(getattr(self.model, "action_space", None), "n", None)
            if action_count != len(Action):
                raise ValueError(
                    f"PPO model action space has {action_count!r} actions; expected {len(Action)}"
                )
            self.load_error = None
            return True
        # Optional model artifacts must never prevent the baseline API from
        # starting. Stable-Baselines3/PyTorch may surface several exception
        # types for corrupt, incompatible, or partially written zip files.
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            return False

    def act(self, observation: dict[str, Any]) -> int:
        if self.model is None:
            return self.fallback.act(observation)
        try:
            observation_space = getattr(self.model, "observation_space", None)
            if observation_space is not None and not observation_space.contains(observation):
                raise ValueError("PPO model observation space is incompatible with this simulation")
            action, _ = self.model.predict(observation, deterministic=True)
            selected = int(np.asarray(action).reshape(-1)[0])
            if selected not in range(len(Action)):
                raise ValueError(f"PPO model returned out-of-range action {selected}")
            return selected
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            # A mismatched or corrupt optional artifact must not take down the
            # functional baseline API. Disable it and use the declared fallback.
            self.load_error = str(exc)
            self.model = None
            return self.fallback.act(observation)
