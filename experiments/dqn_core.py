"""Torch-free building blocks for the DQN experiment.

Everything here is deliberately plain Python and NumPy so the replay buffer,
exploration schedule, and observation encoding can be unit tested without the
optional PyTorch dependency. ``train_dqn.py`` composes these pieces with the
actual network and optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def flatten_observation(observation: dict[str, Any]) -> np.ndarray:
    """Encode the degraded dict observation as one float32 feature vector.

    Grid layers are scaled to small ranges and concatenated with the scalar
    readings. Only observation keys are used — never engine truth.
    """

    fire = np.asarray(observation["fire"], dtype=np.float32).reshape(-1) / 2.0
    terrain = np.asarray(observation["terrain"], dtype=np.float32).reshape(-1) / 2.0
    infrastructure = (
        np.asarray(observation["infrastructure"], dtype=np.float32).reshape(-1) / 3.0
    )
    mask = np.asarray(observation["observed_mask"], dtype=np.float32).reshape(-1)
    wind = np.asarray(observation["wind"], dtype=np.float32).reshape(-1) / 3.0
    reliability = np.asarray(observation["sensor_reliability"], dtype=np.float32).reshape(-1)
    resources = np.asarray(observation["resources"], dtype=np.float32).reshape(-1) / 100.0
    step = np.asarray(observation["step"], dtype=np.float32).reshape(-1) / 5_000.0
    return np.concatenate(
        [fire, terrain, infrastructure, mask, wind, reliability, resources, step]
    ).astype(np.float32)


def linear_epsilon(
    step: int,
    *,
    start: float = 1.0,
    end: float = 0.05,
    decay_steps: int = 10_000,
) -> float:
    """Linearly anneal exploration from ``start`` to ``end`` over ``decay_steps``."""

    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return float(start + fraction * (end - start))


@dataclass(slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity ring buffer with seeded uniform sampling."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[Transition] = []
        self._cursor = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        transition = Transition(state, int(action), float(reward), next_state, bool(done))
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._storage):
            raise ValueError("not enough transitions stored to sample this batch")
        indices = self._rng.choice(len(self._storage), size=batch_size, replace=False)
        batch = [self._storage[int(index)] for index in indices]
        return (
            np.stack([item.state for item in batch]),
            np.array([item.action for item in batch], dtype=np.int64),
            np.array([item.reward for item in batch], dtype=np.float32),
            np.stack([item.next_state for item in batch]),
            np.array([item.done for item in batch], dtype=np.float32),
        )
