"""Validated simulator configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import WindDirection


class SimulationConfig(BaseModel):
    """Parameters for the deliberately simplified stochastic fire model.

    Values are bounded so API callers cannot create invalid probabilities or
    impractically large worlds. The public MVP uses the default 20 x 20 grid;
    smaller grids remain useful for deterministic unit tests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_size: int = Field(default=20, ge=5, le=50)
    base_spread_probability: float = Field(default=0.30, ge=0.0, le=1.0)
    wind_direction: WindDirection = WindDirection.EAST
    wind_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    moisture: float = Field(default=0.25, ge=0.0, le=1.0)
    sensor_reliability: float = Field(default=0.90, ge=0.25, le=1.0)
    observation_noise: float = Field(default=0.03, ge=0.0, le=0.5)
    max_steps: int = Field(default=150, ge=1, le=5_000)
    initial_resources: int = Field(default=12, ge=0, le=1_000)
    resource_effectiveness: float = Field(default=0.70, ge=0.0, le=1.0)
    protection_duration: int = Field(default=4, ge=1, le=50)
    ignition_count: int = Field(default=2, ge=1, le=20)
    burn_duration: int = Field(default=3, ge=1, le=20)
    vegetation_fuel: float = Field(default=1.0, ge=0.0, le=2.0)
    open_ground_fuel: float = Field(default=0.18, ge=0.0, le=2.0)
    road_fuel: float = Field(default=0.04, ge=0.0, le=2.0)

    # Hard-mode features. All default to off, and the engine draws random
    # numbers for them only when enabled, so default seeded trajectories stay
    # bit-identical to earlier releases.
    ember_jump_probability: float = Field(default=0.0, ge=0.0, le=0.5)
    wind_drift_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    observation_delay: int = Field(default=0, ge=0, le=5)
    sensor_dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ignition_fits_grid(self) -> "SimulationConfig":
        if self.ignition_count > self.grid_size * self.grid_size - 3:
            raise ValueError("ignition_count leaves no room for critical infrastructure")
        return self

