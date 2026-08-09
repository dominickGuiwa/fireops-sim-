"""Shared policy protocol."""

from __future__ import annotations

from typing import Any, Protocol


class Agent(Protocol):
    def act(self, observation: dict[str, Any]) -> int:
        """Select an integer Gymnasium action from a degraded observation."""


def scalar(value: Any) -> int:
    """Read an int from either a scalar or one-element NumPy-style value."""

    try:
        return int(value.item())
    except AttributeError:
        if isinstance(value, (list, tuple)):
            return int(value[0])
        return int(value)

