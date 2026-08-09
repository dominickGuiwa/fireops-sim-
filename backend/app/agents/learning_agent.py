"""Transparent tabular Q-learning policy that actually improves online.

Unlike the frozen PPO artifacts, this agent updates a small, fully inspectable
value table from step rewards during the server session. The table is keyed by
a coarse summary of the degraded observation, so every learned preference can
be printed and audited. Learning state lives in process memory only: it lasts
for the current server session and is never a claim about general robustness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.agents.base import scalar
from app.models.enums import Action, LearningEpisodeSource, TrialOutcome
from app.simulation.engine import BURNING

# The discrete state key is (visible-fire sector, visibility bucket, resource
# bucket): 5 x 2 x 3 = 30 states x 6 actions stays trivially auditable.
SECTOR_NONE = "none"
SECTORS = (SECTOR_NONE, "north", "south", "east", "west")
VISIBILITY_BUCKETS = ("clear", "murky")
RESOURCE_BUCKETS = ("none", "low", "ok")

HISTORY_LIMIT = 200
EPSILON_START = 1.0
EPSILON_DECAY = 0.93
EPSILON_FLOOR = 0.05

StateKey = tuple[str, str, str]


class QLearningAgent:
    """Seeded epsilon-greedy tabular Q-learning over the degraded observation."""

    def __init__(
        self,
        seed: int = 0,
        learning_rate: float = 0.20,
        discount: float = 0.95,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.epsilon = EPSILON_START
        self.q_table: dict[StateKey, np.ndarray] = {}
        self.episodes_trained = 0
        self.updates_applied = 0
        self.reward_history: list[dict[str, Any]] = []
        self.last_explored: bool | None = None
        self.last_state: StateKey | None = None
        self.last_q_values: dict[str, float] | None = None

    @staticmethod
    def encode(observation: dict[str, Any]) -> StateKey:
        """Summarize a degraded observation into an auditable discrete state.

        Only visible fire cells, the observed mask, and the resource count are
        used; the encoding never touches true world state.
        """

        fire = np.asarray(observation["fire"])
        mask = np.asarray(observation["observed_mask"])
        n_rows, n_cols = fire.shape
        burning = np.argwhere(fire == BURNING)
        if burning.size == 0:
            sector = SECTOR_NONE
        else:
            rows = burning[:, 0]
            cols = burning[:, 1]
            counts = {
                "north": int(np.count_nonzero(rows < (n_rows + 1) // 2)),
                "south": int(np.count_nonzero(rows >= n_rows // 2)),
                "east": int(np.count_nonzero(cols >= n_cols // 2)),
                "west": int(np.count_nonzero(cols < (n_cols + 1) // 2)),
            }
            sector = max(counts, key=lambda name: (counts[name], -SECTORS.index(name)))
        visibility = "clear" if float(np.mean(mask != 0)) >= 0.5 else "murky"
        resources = scalar(observation.get("resources", 0))
        if resources <= 0:
            resource_bucket = "none"
        elif resources <= 3:
            resource_bucket = "low"
        else:
            resource_bucket = "ok"
        return (sector, visibility, resource_bucket)

    def _q_values(self, state: StateKey) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(Action), dtype=np.float64)
        return self.q_table[state]

    def act(self, observation: dict[str, Any]) -> int:
        state = self.encode(observation)
        q_values = self._q_values(state)
        explored = bool(self.rng.random() < self.epsilon)
        if explored:
            action = int(self.rng.integers(0, len(Action)))
        else:
            # Deterministic tie-break keeps greedy choices reproducible.
            action = int(np.argmax(q_values))
        self.last_explored = explored
        self.last_state = state
        self.last_q_values = {
            Action(index).name: round(float(value), 4) for index, value in enumerate(q_values)
        }
        return action

    def learn(
        self,
        state: StateKey,
        action: Action | int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        """Standard off-policy Q-learning update from one executed step."""

        q_values = self._q_values(state)
        future = 0.0 if done else float(np.max(self._q_values(next_state)))
        index = int(action)
        target = float(reward) + self.discount * future
        q_values[index] += self.learning_rate * (target - q_values[index])
        self.updates_applied += 1

    def finish_episode(
        self,
        episode_reward: float,
        outcome: TrialOutcome,
        source: LearningEpisodeSource,
    ) -> None:
        self.episodes_trained += 1
        self.reward_history.append(
            {
                "episode": self.episodes_trained,
                "reward": round(float(episode_reward), 4),
                "outcome": outcome,
                "epsilon": round(float(self.epsilon), 4),
                "source": source,
            }
        )
        if len(self.reward_history) > HISTORY_LIMIT:
            del self.reward_history[:-HISTORY_LIMIT]
        self.epsilon = max(EPSILON_FLOOR, self.epsilon * EPSILON_DECAY)

    def stats(self) -> dict[str, Any]:
        """Honest training telemetry for the API and UI."""

        rewards = [record["reward"] for record in self.reward_history]
        window = min(10, len(rewards) // 2)
        early_average = round(float(np.mean(rewards[:window])), 4) if window else None
        recent_average = round(float(np.mean(rewards[-window:])), 4) if window else None
        improvement = (
            round(recent_average - early_average, 4)
            if early_average is not None and recent_average is not None
            else None
        )
        return {
            "policy": "learning",
            "episodes_trained": self.episodes_trained,
            "epsilon": round(float(self.epsilon), 4),
            "learning_rate": round(self.learning_rate, 4),
            "discount": round(self.discount, 4),
            "updates_applied": self.updates_applied,
            "states_visited": len(self.q_table),
            "reward_history": list(self.reward_history),
            "early_average": early_average,
            "recent_average": recent_average,
            "improvement": improvement,
            "history_limit": HISTORY_LIMIT,
            "note": (
                "Real tabular Q-learning over the degraded observation. Values update "
                "from step rewards and persist only for this server session process."
            ),
        }

    def q_snapshot(self) -> dict[str, dict[str, float]]:
        """Full value table for auditing; keys are readable state summaries."""

        return {
            "|".join(state): {
                Action(index).name: round(float(value), 4)
                for index, value in enumerate(values)
            }
            for state, values in sorted(self.q_table.items())
        }
