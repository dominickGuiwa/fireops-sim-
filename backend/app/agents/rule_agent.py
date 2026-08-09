"""Auditable heuristic policy that consumes only the partial observation."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.agents.base import scalar
from app.models.enums import Action
from app.simulation.engine import BURNING


class RuleBasedAgent:
    """Protect the sector containing the largest visible fire front.

    When no fire is visible, the policy spends a resource on reconnaissance.
    It never reads true state or evaluation metrics.
    """

    def act(self, observation: dict[str, Any]) -> int:
        resources = scalar(observation.get("resources", 0))
        if resources <= 0:
            return int(Action.HOLD)

        fire = np.asarray(observation["fire"])
        burning = np.argwhere(fire == BURNING)
        if burning.size == 0:
            return int(Action.RECON)

        n_rows, n_cols = fire.shape
        rows = burning[:, 0].astype(np.float64)
        cols = burning[:, 1].astype(np.float64)
        # Distance-to-edge scores emphasize an advancing front near that sector.
        scores = {
            Action.PROTECT_NORTH: float(np.sum(n_rows - rows)),
            Action.PROTECT_SOUTH: float(np.sum(rows + 1)),
            Action.PROTECT_WEST: float(np.sum(n_cols - cols)),
            Action.PROTECT_EAST: float(np.sum(cols + 1)),
        }
        return int(max(scores, key=scores.get))

