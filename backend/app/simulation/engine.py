"""Transparent, synthetic wildfire propagation and degraded sensor model.

This is intentionally not a physical fire model. Each burning cell gets one
independent chance to ignite each cardinal neighbour. Edge probabilities are
combined for targets with multiple burning neighbours. The probability is:

    base * target fuel * (1 - target moisture) * wind * protection

The model is compact enough to audit, deterministic when seeded, and useful for
comparing agents under the same synthetic conditions. It must not be used for
real emergency-response or life-safety decisions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.models.config import SimulationConfig
from app.models.enums import (
    Action,
    ChaosFailure,
    EventSeverity,
    FireStatus,
    Infrastructure,
    SensorStatus,
    Terrain,
    WindDirection,
    parse_action,
)


UNBURNED = 0
BURNING = 1
BURNED = 2
UNKNOWN = -1

TERRAIN_TO_CODE = {Terrain.VEGETATION: 0, Terrain.OPEN_GROUND: 1, Terrain.ROAD: 2}
CODE_TO_TERRAIN = {value: key for key, value in TERRAIN_TO_CODE.items()}
INFRA_TO_CODE = {
    Infrastructure.NONE: 0,
    Infrastructure.MEDICAL: 1,
    Infrastructure.POWER: 2,
    Infrastructure.COMMUNICATIONS: 3,
}
CODE_TO_INFRA = {value: key for key, value in INFRA_TO_CODE.items()}
CODE_TO_FIRE = {UNBURNED: FireStatus.UNBURNED, BURNING: FireStatus.BURNING, BURNED: FireStatus.BURNED}
WIND_TO_CODE = {
    WindDirection.NORTH: 0,
    WindDirection.SOUTH: 1,
    WindDirection.EAST: 2,
    WindDirection.WEST: 3,
}
CODE_TO_WIND = {value: key for key, value in WIND_TO_CODE.items()}
WIND_VECTOR = {
    WindDirection.NORTH: (-1, 0),
    WindDirection.SOUTH: (1, 0),
    WindDirection.EAST: (0, 1),
    WindDirection.WEST: (0, -1),
}
# A drifting wind rotates 90 degrees, never reverses in one step.
WIND_NEIGHBOURS = {
    WindDirection.NORTH: (WindDirection.EAST, WindDirection.WEST),
    WindDirection.SOUTH: (WindDirection.EAST, WindDirection.WEST),
    WindDirection.EAST: (WindDirection.NORTH, WindDirection.SOUTH),
    WindDirection.WEST: (WindDirection.NORTH, WindDirection.SOUTH),
}


@dataclass(slots=True)
class ObservationSnapshot:
    fire: NDArray[np.int8]
    observed_mask: NDArray[np.bool_]
    observed_wind: int
    effective_reliability: float
    sensor_status: SensorStatus


class FireSimulation:
    """Stateful synthetic world with separately seeded dynamics and sensors."""

    def __init__(self, config: SimulationConfig | dict[str, Any] | None = None, seed: int | None = None) -> None:
        self.config = config if isinstance(config, SimulationConfig) else SimulationConfig.model_validate(config or {})
        self.seed = int(seed if seed is not None else 42)
        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> ObservationSnapshot:
        if seed is not None:
            self.seed = int(seed)
        seed_sequence = np.random.SeedSequence(self.seed)
        dynamics_seed, observation_seed, chaos_seed = seed_sequence.spawn(3)
        self.rng = np.random.default_rng(dynamics_seed)
        self.observation_rng = np.random.default_rng(observation_seed)
        self.chaos_rng = np.random.default_rng(chaos_seed)

        n = self.config.grid_size
        self.step_count = 0
        self.done = False
        self.termination_reason: str | None = None
        self.current_action = Action.HOLD
        self.wind_direction = self.config.wind_direction
        self.wind_strength = self.config.wind_strength
        self.resources_available = self.config.initial_resources
        self.resources_consumed = 0
        self.resources_unavailable = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.last_reward_breakdown = {
            "new_spread_penalty": 0.0,
            "infrastructure_loss_penalty": 0.0,
            "resource_use_penalty": 0.0,
            "active_fire_penalty": 0.0,
            "total": 0.0,
        }
        self.events: list[dict[str, Any]] = []

        self._sensor_offline_until = 0
        self._sensor_degraded_until = 0
        self._communications_degraded_until = 0
        self._recon_until = 0

        self.terrain = self.rng.choice(3, size=(n, n), p=(0.78, 0.16, 0.06)).astype(np.int8)
        # A synthetic cross-road gives the map readable structure without using
        # any real facility or road layout.
        self.terrain[n // 2, :] = TERRAIN_TO_CODE[Terrain.ROAD]
        self.terrain[:, n // 2] = TERRAIN_TO_CODE[Terrain.ROAD]

        self.infrastructure = np.zeros((n, n), dtype=np.int8)
        infrastructure_cells = [
            (max(1, n // 5), max(1, n // 5), Infrastructure.MEDICAL),
            (min(n - 2, (4 * n) // 5), max(1, n // 4), Infrastructure.POWER),
            (n // 2, min(n - 2, (4 * n) // 5), Infrastructure.COMMUNICATIONS),
        ]
        self.infrastructure_locations: dict[tuple[int, int], Infrastructure] = {}
        for row, col, kind in infrastructure_cells:
            self.infrastructure[row, col] = INFRA_TO_CODE[kind]
            self.terrain[row, col] = TERRAIN_TO_CODE[Terrain.VEGETATION]
            self.infrastructure_locations[(row, col)] = kind

        fuel_by_terrain = np.array(
            [self.config.vegetation_fuel, self.config.open_ground_fuel, self.config.road_fuel], dtype=np.float64
        )
        self.fuel = fuel_by_terrain[self.terrain]
        # Facilities have surrounding combustible material, but less than dense vegetation.
        self.fuel[self.infrastructure > 0] *= 0.70
        self.cell_moisture = np.clip(
            self.config.moisture + self.rng.normal(0.0, 0.06, size=(n, n)), 0.0, 1.0
        )

        self.fire_state = np.zeros((n, n), dtype=np.int8)
        self.burn_age = np.zeros((n, n), dtype=np.int16)
        self.protection = np.zeros((n, n), dtype=np.float64)
        self.protection_ttl = np.zeros((n, n), dtype=np.int16)
        self.infrastructure_lost: set[tuple[int, int]] = set()

        candidates = np.argwhere((self.terrain == TERRAIN_TO_CODE[Terrain.VEGETATION]) & (self.infrastructure == 0))
        if len(candidates) < self.config.ignition_count:
            candidates = np.argwhere(self.infrastructure == 0)
        chosen = self.rng.choice(len(candidates), size=self.config.ignition_count, replace=False)
        for candidate_index in np.atleast_1d(chosen):
            row, col = (int(value) for value in candidates[int(candidate_index)])
            self.fire_state[row, col] = BURNING
            self._add_event("ignition", f"Synthetic ignition detected at ({row}, {col}).", EventSeverity.WARNING)

        # One frame per step; with observation_delay the oldest retained frame
        # is what sensors report, so the agent acts on stale telemetry.
        self._fire_frame_history: deque[NDArray[np.int8]] = deque(
            maxlen=self.config.observation_delay + 1
        )
        self._fire_frame_history.append(self.fire_state.copy())

        self._last_observation = self._generate_observation()
        return self._last_observation

    @property
    def shape(self) -> tuple[int, int]:
        return self.fire_state.shape

    def neighbours(self, row: int, col: int) -> Iterator[tuple[int, int]]:
        """Yield only bounds-safe cardinal neighbours."""

        n = self.config.grid_size
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target_row, target_col = row + delta_row, col + delta_col
            if 0 <= target_row < n and 0 <= target_col < n:
                yield target_row, target_col

    def spread_probability(self, source: tuple[int, int], target: tuple[int, int]) -> float:
        """Return the auditable edge ignition probability for two neighbours."""

        source_row, source_col = source
        target_row, target_col = target
        if abs(source_row - target_row) + abs(source_col - target_col) != 1:
            return 0.0
        n = self.config.grid_size
        if not (0 <= source_row < n and 0 <= source_col < n and 0 <= target_row < n and 0 <= target_col < n):
            return 0.0
        if self.fire_state[target_row, target_col] != UNBURNED:
            return 0.0

        delta = (target_row - source_row, target_col - source_col)
        wind_delta = WIND_VECTOR[self.wind_direction]
        alignment = delta[0] * wind_delta[0] + delta[1] * wind_delta[1]
        if alignment > 0:
            wind_multiplier = 1.0 + 1.50 * self.wind_strength
        elif alignment < 0:
            wind_multiplier = max(0.10, 1.0 - 0.75 * self.wind_strength)
        else:
            wind_multiplier = 1.0

        protection_multiplier = 1.0 - 0.90 * self.protection[target_row, target_col]
        probability = (
            self.config.base_spread_probability
            * self.fuel[target_row, target_col]
            * (1.0 - self.cell_moisture[target_row, target_col])
            * wind_multiplier
            * protection_multiplier
        )
        return float(np.clip(probability, 0.0, 0.98))

    def step(self, action: Action | int | str) -> tuple[ObservationSnapshot, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is complete; reset before stepping again")

        parsed_action = parse_action(action)
        self.current_action = parsed_action
        resource_used = self._apply_action(parsed_action)

        # Non-stationary wind: only enabled configs draw from the RNG, so the
        # default dynamics stream stays identical to earlier releases.
        if self.config.wind_drift_probability > 0 and self.rng.random() < self.config.wind_drift_probability:
            choices = WIND_NEIGHBOURS[self.wind_direction]
            self.wind_direction = choices[int(self.rng.integers(0, len(choices)))]
            self.wind_strength = float(
                np.clip(self.wind_strength + self.rng.uniform(-0.15, 0.15), 0.0, 1.0)
            )
            self._add_event(
                "wind_drift",
                f"Wind drifted {self.wind_direction.value} at strength {self.wind_strength:.2f}.",
                EventSeverity.WARNING,
            )

        burning_at_start = self.fire_state == BURNING
        ignition_probabilities = np.zeros(self.shape, dtype=np.float64)
        for source_row, source_col in np.argwhere(burning_at_start):
            source = (int(source_row), int(source_col))
            for target in self.neighbours(*source):
                probability = self.spread_probability(source, target)
                if probability:
                    # P(any neighbour ignites target) = 1 - product(1 - p_i).
                    ignition_probabilities[target] = 1.0 - (
                        (1.0 - ignition_probabilities[target]) * (1.0 - probability)
                    )

        # Dynamics use only the dynamics RNG; sensor queries cannot perturb fire spread.
        new_burning = (self.rng.random(self.shape) < ignition_probabilities) & (self.fire_state == UNBURNED)

        self.burn_age[burning_at_start] += 1
        newly_burned = burning_at_start & (self.burn_age >= self.config.burn_duration)
        self.fire_state[newly_burned] = BURNED
        self.fire_state[new_burning] = BURNING
        self.burn_age[new_burning] = 0

        new_fire_count = int(np.count_nonzero(new_burning))
        # Ember jumps: burning cells can loft a spot fire 2-3 cells downwind,
        # skipping roads and protection, which defeats purely local defenses.
        if self.config.ember_jump_probability > 0:
            ember_count = 0
            wind_delta = WIND_VECTOR[self.wind_direction]
            n = self.config.grid_size
            for source_row, source_col in np.argwhere(burning_at_start):
                if self.rng.random() >= self.config.ember_jump_probability:
                    continue
                distance = int(self.rng.integers(2, 4))
                target_row = int(source_row) + wind_delta[0] * distance
                target_col = int(source_col) + wind_delta[1] * distance
                if not (0 <= target_row < n and 0 <= target_col < n):
                    continue
                if self.fire_state[target_row, target_col] != UNBURNED:
                    continue
                if self.fuel[target_row, target_col] <= 0.05:
                    continue
                self.fire_state[target_row, target_col] = BURNING
                self.burn_age[target_row, target_col] = 0
                ember_count += 1
            if ember_count:
                new_fire_count += ember_count
                self._add_event(
                    "ember_jump",
                    f"Ember jump ignited {ember_count} spot fire(s) downwind.",
                    EventSeverity.CRITICAL,
                )
        if new_fire_count:
            self._add_event(
                "fire_spread", f"Fire spread to {new_fire_count} additional cell(s).", EventSeverity.WARNING
            )

        new_infrastructure_losses = self._record_infrastructure_losses()
        reward_components = {
            "new_spread_penalty": -0.75 * new_fire_count,
            "infrastructure_loss_penalty": -30.0 * new_infrastructure_losses,
            "resource_use_penalty": -0.30 * resource_used,
            "active_fire_penalty": -0.03 * int(np.count_nonzero(self.fire_state == BURNING)),
        }
        reward = sum(reward_components.values())
        self.last_reward = round(float(reward), 4)
        self.last_reward_breakdown = {
            **{name: round(float(value), 4) or 0.0 for name, value in reward_components.items()},
            "total": self.last_reward,
        }
        self.cumulative_reward += self.last_reward

        self._age_temporary_effects()
        self.step_count += 1
        self._fire_frame_history.append(self.fire_state.copy())
        burning_remaining = int(np.count_nonzero(self.fire_state == BURNING))
        if burning_remaining == 0:
            self.done = True
            self.termination_reason = "fire_extinguished"
            self._add_event("episode_complete", "No active fire remains.", EventSeverity.INFO)
        elif self.step_count >= self.config.max_steps:
            self.done = True
            self.termination_reason = "max_steps"
            self._add_event("episode_complete", "Maximum episode length reached.", EventSeverity.INFO)

        self._last_observation = self._generate_observation()
        info = {
            "metrics": self.metrics(),
            "events": list(self.events[-10:]),
            "action": parsed_action.name,
            "termination_reason": self.termination_reason,
        }
        return self._last_observation, self.last_reward, self.done, info

    def _apply_action(self, action: Action) -> int:
        if action == Action.HOLD:
            self._add_event("action", "Agent held current response posture.", EventSeverity.INFO)
            return 0
        if self.resources_available <= 0:
            self._add_event(
                "resource", f"{action.name} requested, but no response resources are available.", EventSeverity.WARNING
            )
            return 0

        self.resources_available -= 1
        self.resources_consumed += 1
        if action == Action.RECON:
            self._recon_until = max(self._recon_until, self.step_count + 4)
            self._add_event("action", "Recon temporarily improved the degraded observation.", EventSeverity.INFO)
            return 1

        n = self.config.grid_size
        mask = np.zeros(self.shape, dtype=np.bool_)
        if action == Action.PROTECT_NORTH:
            mask[: (n + 1) // 2, :] = True
        elif action == Action.PROTECT_SOUTH:
            mask[n // 2 :, :] = True
        elif action == Action.PROTECT_EAST:
            mask[:, n // 2 :] = True
        elif action == Action.PROTECT_WEST:
            mask[:, : (n + 1) // 2] = True
        self.protection[mask] = np.maximum(self.protection[mask], self.config.resource_effectiveness)
        self.protection_ttl[mask] = np.maximum(self.protection_ttl[mask], self.config.protection_duration)
        direction = action.name.removeprefix("PROTECT_").lower()
        self._add_event("action", f"A response resource protected the {direction} sector.", EventSeverity.INFO)
        return 1

    def _age_temporary_effects(self) -> None:
        active = self.protection_ttl > 0
        self.protection_ttl[active] -= 1
        self.protection[self.protection_ttl <= 0] = 0.0

    def _record_infrastructure_losses(self) -> int:
        new_losses = 0
        for location, kind in self.infrastructure_locations.items():
            if location not in self.infrastructure_lost and self.fire_state[location] != UNBURNED:
                self.infrastructure_lost.add(location)
                new_losses += 1
                self._add_event(
                    "infrastructure_loss",
                    f"Synthetic {kind.value} infrastructure was affected by fire.",
                    EventSeverity.CRITICAL,
                )
        return new_losses

    def inject_chaos(self, failure: ChaosFailure | str | None = None) -> ChaosFailure:
        if failure is None:
            failures = list(ChaosFailure)
            if self.resources_available == 0:
                failures.remove(ChaosFailure.RESOURCE_UNAVAILABLE)
            failure = failures[int(self.chaos_rng.integers(0, len(failures)))]
        else:
            failure = ChaosFailure(failure)

        severity = EventSeverity.CRITICAL

        if failure == ChaosFailure.SENSOR_OFFLINE:
            duration = int(self.chaos_rng.integers(3, 7))
            self._sensor_offline_until = max(self._sensor_offline_until, self.step_count + duration)
            message = f"Sensor network offline for approximately {duration} step(s)."
        elif failure == ChaosFailure.SENSOR_DEGRADED:
            duration = int(self.chaos_rng.integers(4, 9))
            self._sensor_degraded_until = max(self._sensor_degraded_until, self.step_count + duration)
            message = f"Sensor reliability degraded for approximately {duration} step(s)."
        elif failure == ChaosFailure.COMMUNICATIONS_DEGRADED:
            duration = int(self.chaos_rng.integers(4, 9))
            self._communications_degraded_until = max(
                self._communications_degraded_until, self.step_count + duration
            )
            message = f"Synthetic communications degraded for approximately {duration} step(s)."
        elif failure == ChaosFailure.WIND_CHANGE:
            choices = [direction for direction in WindDirection if direction != self.wind_direction]
            self.wind_direction = choices[int(self.chaos_rng.integers(0, len(choices)))]
            self.wind_strength = float(self.chaos_rng.uniform(0.25, 1.0))
            message = f"Wind shifted {self.wind_direction.value} at strength {self.wind_strength:.2f}."
        else:
            unavailable = min(self.resources_available, int(self.chaos_rng.integers(1, 4)))
            self.resources_available -= unavailable
            self.resources_unavailable += unavailable
            if unavailable:
                message = f"{unavailable} response resource(s) became unavailable."
            else:
                message = "Resource failure requested, but no response resources remained."
                severity = EventSeverity.WARNING

        self._add_event("chaos", message, severity)
        self._last_observation = self._generate_observation()
        return failure

    def effective_sensor_reliability(self) -> float:
        reliability = self.config.sensor_reliability
        if self.step_count < self._sensor_offline_until:
            reliability = 0.0
        else:
            if self.step_count < self._sensor_degraded_until:
                reliability *= 0.45
            if self.step_count < self._communications_degraded_until:
                reliability *= 0.60
        if self.step_count < self._recon_until:
            reliability = max(reliability, 0.95)
        return float(np.clip(reliability, 0.0, 1.0))

    def _generate_observation(self) -> ObservationSnapshot:
        reliability = self.effective_sensor_reliability()
        # Transient dropout: a whole sensor frame can arrive empty in hard mode.
        if (
            self.config.sensor_dropout_probability > 0
            and self.observation_rng.random() < self.config.sensor_dropout_probability
        ):
            reliability = 0.0
            self._add_event(
                "sensor_dropout",
                "Transient sensor dropout: this observation frame arrived empty.",
                EventSeverity.WARNING,
            )
        # With observation_delay, sensors report the oldest retained fire frame.
        reported_fire = (
            self._fire_frame_history[0]
            if self.config.observation_delay > 0
            else self.fire_state
        )
        mask = self.observation_rng.random(self.shape) < reliability
        observed_fire = np.full(self.shape, UNKNOWN, dtype=np.int8)
        observed_fire[mask] = reported_fire[mask]

        # Reliability affects availability; configured observation noise affects
        # the small number of visible but incorrect readings.
        noise_probability = self.config.observation_noise * (1.0 + (1.0 - reliability))
        noisy = mask & (self.observation_rng.random(self.shape) < noise_probability)
        if np.any(noisy):
            observed_fire[noisy] = self.observation_rng.integers(UNBURNED, BURNED + 1, size=int(noisy.sum()))

        if reliability == 0.0:
            status = SensorStatus.OFFLINE
        elif reliability < 0.75:
            status = SensorStatus.DEGRADED
        else:
            status = SensorStatus.OPERATIONAL

        observed_wind = UNKNOWN
        if self.observation_rng.random() < reliability:
            true_wind = WIND_TO_CODE[self.wind_direction]
            observed_wind = true_wind
            if self.observation_rng.random() < noise_probability:
                observed_wind = (true_wind + int(self.observation_rng.integers(1, 4))) % 4

        return ObservationSnapshot(
            fire=observed_fire,
            observed_mask=mask,
            observed_wind=observed_wind,
            effective_reliability=reliability,
            sensor_status=status,
        )

    def refresh_observation(self) -> ObservationSnapshot:
        """Explicitly generate a new sensor frame; ordinary state reads are cached."""

        self._last_observation = self._generate_observation()
        return self._last_observation

    @property
    def observation(self) -> ObservationSnapshot:
        return self._last_observation

    def metrics(self) -> dict[str, int | float]:
        affected = int(np.count_nonzero(self.fire_state != UNBURNED))
        total = self.config.grid_size**2
        infrastructure_total = len(self.infrastructure_locations)
        infrastructure_lost = len(self.infrastructure_lost)
        return {
            "cumulative_reward": round(float(self.cumulative_reward), 4),
            "burned_area": affected,
            "burned_fraction": round(affected / total, 6),
            "burning_cells": int(np.count_nonzero(self.fire_state == BURNING)),
            "infrastructure_preserved": infrastructure_total - infrastructure_lost,
            "infrastructure_total": infrastructure_total,
            "infrastructure_lost": infrastructure_lost,
            "resources_consumed": self.resources_consumed,
        }

    def gym_observation(self) -> dict[str, NDArray[Any]]:
        observation = self.observation
        return {
            "fire": observation.fire.copy(),
            "terrain": self.terrain.copy(),
            "infrastructure": self.infrastructure.copy(),
            "observed_mask": observation.observed_mask.astype(np.int8, copy=True),
            "wind": np.array([observation.observed_wind], dtype=np.int8),
            "sensor_reliability": np.array([observation.effective_reliability], dtype=np.float32),
            "resources": np.array([self.resources_available], dtype=np.int32),
            "step": np.array([self.step_count], dtype=np.int32),
        }

    def state_dict(self, policy: str = "rule", event_limit: int = 100) -> dict[str, Any]:
        observation = self.observation
        cells: list[dict[str, Any]] = []
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                observed = bool(observation.observed_mask[row, col])
                fire_code = int(observation.fire[row, col])
                cells.append(
                    {
                        "row": row,
                        "col": col,
                        "terrain": CODE_TO_TERRAIN[int(self.terrain[row, col])],
                        "fire": CODE_TO_FIRE[fire_code] if observed else FireStatus.UNKNOWN,
                        # Researcher/display truth overlay; never included in gym_observation().
                        "true_fire": CODE_TO_FIRE[int(self.fire_state[row, col])],
                        # Infrastructure and terrain are static map layers known in advance.
                        "infrastructure": CODE_TO_INFRA[int(self.infrastructure[row, col])],
                        "observed": observed,
                        "protection": round(float(self.protection[row, col]), 3),
                    }
                )
        observed_fraction = float(np.mean(observation.observed_mask))
        return {
            "grid_size": self.config.grid_size,
            "seed": self.seed,
            "step_count": self.step_count,
            "max_steps": self.config.max_steps,
            "cells": cells,
            "wind": {
                "direction": self.wind_direction,
                "observed_direction": (
                    CODE_TO_WIND[observation.observed_wind]
                    if observation.observed_wind != UNKNOWN
                    else None
                ),
                "strength": round(float(self.wind_strength), 4),
            },
            "sensor": {
                "reliability": round(observation.effective_reliability, 4),
                "configured_reliability": self.config.sensor_reliability,
                "status": observation.sensor_status,
                "observed_fraction": round(observed_fraction, 4),
                "communication_degraded": self.step_count < self._communications_degraded_until,
                "recon_active": self.step_count < self._recon_until,
            },
            "resources": {
                "available": self.resources_available,
                "initial": self.config.initial_resources,
                "consumed": self.resources_consumed,
                "unavailable": self.resources_unavailable,
            },
            "metrics": self.metrics(),
            "policy": policy,
            "current_action": self.current_action.name,
            "last_reward": round(float(self.last_reward), 4),
            "events": list(self.events[-event_limit:]),
            "done": self.done,
            "termination_reason": self.termination_reason,
        }

    def record_event(self, event_type: str, message: str, severity: EventSeverity) -> None:
        """Public hook for session-level analysis (failure/training) events."""

        self._add_event(event_type, message, severity)

    def _add_event(self, event_type: str, message: str, severity: EventSeverity) -> None:
        self.events.append(
            {"step": self.step_count, "type": event_type, "message": message, "severity": severity}
        )
        if len(self.events) > 500:
            del self.events[:-500]
