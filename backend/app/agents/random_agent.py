"""Seedable random baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models.enums import Action


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> int:
        del observation
        return int(self.rng.integers(0, len(Action)))

