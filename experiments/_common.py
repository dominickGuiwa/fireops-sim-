"""Shared runtime helpers for standalone experiment scripts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment cannot be run as requested."""


def add_backend_to_path() -> None:
    """Make the local backend package importable without requiring installation."""

    backend = str(BACKEND_ROOT)
    if backend not in sys.path:
        sys.path.insert(0, backend)


def backend_types() -> tuple[type[Any], type[Any]]:
    """Load the simulator types with a useful error for incomplete installs."""

    add_backend_to_path()
    try:
        from app.models.config import SimulationConfig
        from app.simulation.environment import FireOpsEnv
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise RuntimeError(
            "Could not import the FIREOPS-SIM backend. Run this command from the "
            "repository root after installing the backend dependencies."
        ) from exc
    return SimulationConfig, FireOpsEnv


def validate_reliability(value: float) -> float:
    """Validate the sensor range supported by the v0.1 simulator."""

    reliability = float(value)
    if not 0.25 <= reliability <= 1.0:
        raise ExperimentConfigurationError(
            f"sensor reliability must be between 0.25 and 1.0, got {value}"
        )
    return reliability


def create_environment(*, reliability: float, seed: int) -> Any:
    """Create a simulator configured for an evaluation or training episode."""

    SimulationConfig, FireOpsEnv = backend_types()
    config = SimulationConfig(sensor_reliability=validate_reliability(reliability))
    return FireOpsEnv(config=config, seed=seed)


def reset_environment(
    environment: Any,
    *,
    reliability: float,
    seed: int | None,
    inject_failure: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Reset through the environment's public episode-randomization contract."""

    options: dict[str, Any] = {
        "sensor_reliability": validate_reliability(reliability),
    }
    if inject_failure:
        options["inject_failure"] = True
    observation, info = environment.reset(seed=seed, options=options)
    if not isinstance(info, dict):
        raise RuntimeError("FireOpsEnv.reset() returned a non-dictionary info value")
    return observation, info


def sha256_file(path: Path) -> str:
    """Return a model artifact digest without loading the artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_artifact_path(path: Path) -> str:
    """Return a provenance label that cannot leak a developer's home directory."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"external:{resolved.name}"
