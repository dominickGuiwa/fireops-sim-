#!/usr/bin/env python3
"""Train a real PPO policy against FIREOPS-SIM.

Stable-Baselines3 and PyTorch are imported only after argument validation. The
base simulator and API therefore remain usable without the optional PPO stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments._common import (  # noqa: E402
    ExperimentConfigurationError,
    create_environment,
    public_artifact_path,
    reset_environment,
    sha256_file,
    validate_reliability,
)


def randomization_wrapper(
    env: Any,
    *,
    gym_module: Any,
    numpy_module: Any,
    minimum_reliability: float,
    maximum_reliability: float,
    failure_probability: float,
    randomization_seed: int,
) -> Any:
    """Build the Gym wrapper after dependencies have been loaded lazily."""

    class EpisodeRandomizationWrapper(gym_module.Wrapper):
        """Randomize observation reliability and failures per episode."""

        def __init__(self, wrapped: Any) -> None:
            super().__init__(wrapped)
            self.minimum_reliability = validate_reliability(minimum_reliability)
            self.maximum_reliability = validate_reliability(maximum_reliability)
            if self.minimum_reliability > self.maximum_reliability:
                raise ExperimentConfigurationError(
                    "minimum reliability cannot exceed maximum reliability"
                )
            if not 0.0 <= failure_probability <= 1.0:
                raise ExperimentConfigurationError(
                    "failure probability must be between 0.0 and 1.0"
                )
            self.failure_probability = float(failure_probability)
            self._rng = numpy_module.random.default_rng(randomization_seed)
            self._has_received_seed = False

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            # Gymnasium/SB3 seeds the wrapper once. Re-seeding on every internal
            # reset would repeat one reliability instead of randomizing it.
            if seed is not None and not self._has_received_seed:
                self._rng = numpy_module.random.default_rng(seed)
                self._has_received_seed = True

            reliability = float(
                self._rng.uniform(self.minimum_reliability, self.maximum_reliability)
            )
            inject_failure = bool(self._rng.random() < self.failure_probability)
            reset_options = dict(options or {})
            reset_options.update(
                {
                    "sensor_reliability": reliability,
                    "inject_failure": inject_failure,
                }
            )
            observation, info = self.env.reset(seed=seed, options=reset_options)
            info = dict(info)
            info["training_randomization"] = {
                "sensor_reliability": reliability,
                "failure_injected": inject_failure,
            }
            return observation, info

    return EpisodeRandomizationWrapper(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a nominal or domain-randomized FIREOPS-SIM PPO policy."
    )
    parser.add_argument("--mode", choices=("nominal", "robust"), default="nominal")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total PPO environment steps (default: 50000).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        help="Model path, with or without the .zip suffix.",
    )
    parser.add_argument("--minimum-reliability", type=float, default=0.25)
    parser.add_argument("--maximum-reliability", type=float, default=1.0)
    parser.add_argument(
        "--failure-probability",
        type=float,
        default=0.20,
        help="Per-episode synthetic failure probability in robust mode.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.timesteps <= 0:
        raise ExperimentConfigurationError("timesteps must be positive")
    if args.seed < 0:
        raise ExperimentConfigurationError("seed must be non-negative")
    if args.n_steps < 2:
        raise ExperimentConfigurationError("n-steps must be at least 2")
    if not 2 <= args.batch_size <= args.n_steps:
        raise ExperimentConfigurationError("batch-size must be between 2 and n-steps")
    if args.learning_rate <= 0.0:
        raise ExperimentConfigurationError("learning-rate must be positive")
    if not 0.0 < args.gamma <= 1.0:
        raise ExperimentConfigurationError("gamma must be in (0, 1]")
    if not 0.0 <= args.failure_probability <= 1.0:
        raise ExperimentConfigurationError(
            "failure-probability must be between 0.0 and 1.0"
        )
    validate_reliability(args.minimum_reliability)
    validate_reliability(args.maximum_reliability)
    if args.minimum_reliability > args.maximum_reliability:
        raise ExperimentConfigurationError(
            "minimum-reliability cannot exceed maximum-reliability"
        )


def resolved_model_paths(mode: str, requested: Path | None) -> tuple[Path, Path]:
    model_base = requested or (REPOSITORY_ROOT / "models" / f"ppo_{mode}")
    model_base = model_base.expanduser().resolve()
    model_file = (
        model_base if model_base.suffix.lower() == ".zip" else Path(f"{model_base}.zip")
    )
    return model_base, model_file


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ExperimentConfigurationError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    try:
        import gymnasium as gym
        import numpy as np
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "PPO training dependencies are optional. Install them with "
            "`python -m pip install -e \"./backend[ppo]\"` and retry."
        ) from exc

    if args.mode == "nominal":
        minimum_reliability = maximum_reliability = 1.0
        failure_probability = 0.0
    else:
        minimum_reliability = args.minimum_reliability
        maximum_reliability = args.maximum_reliability
        failure_probability = args.failure_probability

    base_environment = create_environment(
        reliability=maximum_reliability,
        seed=args.seed,
    )
    randomized_environment = randomization_wrapper(
        base_environment,
        gym_module=gym,
        numpy_module=np,
        minimum_reliability=minimum_reliability,
        maximum_reliability=maximum_reliability,
        failure_probability=failure_probability,
        randomization_seed=args.seed,
    )
    environment = Monitor(randomized_environment)

    model_base, model_file = resolved_model_paths(args.mode, args.output)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    policy = (
        "MultiInputPolicy"
        if isinstance(environment.observation_space, gym.spaces.Dict)
        else "MlpPolicy"
    )
    model = PPO(
        policy,
        environment,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    try:
        # Exercise the public reset contract before committing to a long run.
        reset_environment(
            base_environment,
            reliability=maximum_reliability,
            seed=args.seed,
            inject_failure=False,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(str(model_base))
    finally:
        environment.close()

    if not model_file.is_file():
        raise RuntimeError(f"PPO did not create the expected model artifact: {model_file}")

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO",
        "mode": args.mode,
        "policy": policy,
        "requested_timesteps": args.timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "seed": args.seed,
        "sensor_reliability": {
            "distribution": (
                "constant" if minimum_reliability == maximum_reliability else "uniform"
            ),
            "minimum": minimum_reliability,
            "maximum": maximum_reliability,
        },
        "synthetic_failure_probability": failure_probability,
        "base_environment_config": base_environment.base_config.model_dump(mode="json"),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "device": args.device,
        },
        "software": {
            "python": sys.version.split()[0],
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "model": {
            "path": public_artifact_path(model_file),
            "sha256": sha256_file(model_file),
        },
    }
    metadata_path = model_file.with_suffix(".metadata.json")
    write_metadata(metadata_path, metadata)

    print(f"Saved PPO model: {model_file}")
    print(f"Saved training metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
