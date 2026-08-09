"""Detect auditable agentic failure modes for one executed step.

The detector deliberately uses researcher truth (the full fire grid and true
wind) to judge the action an agent chose from its degraded observation. That
truth is analysis-only: it is never fed back into any policy. Every rule below
is a small, explicit predicate so a reader can verify exactly why a step was
flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.models.enums import Action, AgentFailureType
from app.simulation.engine import BURNING, UNKNOWN

PROTECT_SECTORS = {
    Action.PROTECT_NORTH: "north",
    Action.PROTECT_SOUTH: "south",
    Action.PROTECT_EAST: "east",
    Action.PROTECT_WEST: "west",
}


def sector_burning_counts(fire_state: NDArray[np.int8]) -> dict[str, int]:
    """True burning-cell counts per protect sector, using the engine's masks."""

    n_rows, n_cols = fire_state.shape
    burning = fire_state == BURNING
    return {
        "north": int(np.count_nonzero(burning[: (n_rows + 1) // 2, :])),
        "south": int(np.count_nonzero(burning[n_rows // 2 :, :])),
        "east": int(np.count_nonzero(burning[:, n_cols // 2 :])),
        "west": int(np.count_nonzero(burning[:, : (n_cols + 1) // 2])),
    }


def detect_step_failures(
    *,
    step: int,
    policy: str,
    action: Action,
    true_fire_before: NDArray[np.int8],
    observed_fire_before: NDArray[np.int8],
    observed_mask_before: NDArray[np.int8],
    observed_wind_before: int,
    true_wind_code: int,
    resources_before: int,
) -> list[dict[str, Any]]:
    """Return zero or more failure records for the step that just executed."""

    true_burning = int(np.count_nonzero(true_fire_before == BURNING))
    visible_burning = int(
        np.count_nonzero((observed_fire_before == BURNING) & (observed_mask_before != 0))
    )
    hidden_burning = int(
        np.count_nonzero((true_fire_before == BURNING) & (observed_mask_before == 0))
    )
    context = {
        "hidden_fire_cells": hidden_burning,
        "visible_fire_cells": visible_burning,
        "true_burning_cells": true_burning,
    }

    failures: list[dict[str, Any]] = []

    def record(failure_type: AgentFailureType, message: str) -> None:
        failures.append(
            {
                "step": step,
                "policy": policy,
                "failure_type": failure_type,
                "message": message,
                **context,
            }
        )

    if true_burning > 0 and hidden_burning * 2 > true_burning:
        record(
            AgentFailureType.PERCEPTION_GAP,
            f"Perception failure: {hidden_burning} of {true_burning} burning cell(s) "
            "were invisible to the agent when it chose this move.",
        )

    if action in PROTECT_SECTORS and true_burning > 0:
        counts = sector_burning_counts(true_fire_before)
        chosen_sector = PROTECT_SECTORS[action]
        best_count = max(counts.values())
        best_sectors = [name for name, count in counts.items() if count == best_count]
        if len(best_sectors) == 1 and counts[chosen_sector] < best_count:
            record(
                AgentFailureType.MISDIRECTED_PROTECTION,
                f"Decision failure: the agent protected the {chosen_sector} sector "
                f"({counts[chosen_sector]} true burning cell(s)) while the "
                f"{best_sectors[0]} sector held {best_count}.",
            )

    if action is Action.HOLD and resources_before > 0 and true_burning > 0:
        record(
            AgentFailureType.IDLE_WHILE_BURNING,
            f"Decision failure: the agent held with {resources_before} resource(s) "
            f"available while {true_burning} cell(s) were truly burning.",
        )

    if action is not Action.HOLD and resources_before <= 0:
        record(
            AgentFailureType.RESOURCE_EXHAUSTED_REQUEST,
            f"Resource failure: {action.name} was requested with no response "
            "resources remaining, so the move had no effect.",
        )

    if (
        action in PROTECT_SECTORS
        and true_burning > 0
        and observed_wind_before != UNKNOWN
        and observed_wind_before != true_wind_code
    ):
        record(
            AgentFailureType.WIND_MISREAD,
            "Sensor failure: the reported wind direction was wrong when the agent "
            "committed a protect action, so its spread estimate was misled.",
        )

    return failures
