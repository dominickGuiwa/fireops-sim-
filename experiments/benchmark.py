#!/usr/bin/env python3
"""Run reproducible FIREOPS-SIM policy evaluations and save real results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments._common import (  # noqa: E402
    ExperimentConfigurationError,
    add_backend_to_path,
    create_environment,
    public_artifact_path,
    reset_environment,
    sha256_file,
    validate_reliability,
)


DEFAULT_RELIABILITIES = (1.0, 0.75, 0.50, 0.25)
METRIC_NAMES = (
    "cumulative_reward",
    "infrastructure_preserved",
    "burned_area",
    "resources_consumed",
)
FAILURE_TYPES = (
    "sensor_offline",
    "sensor_degraded",
    "communications_degraded",
    "wind_change",
    "resource_unavailable",
)


class Policy(Protocol):
    def act(self, observation: Any) -> int:
        """Choose one discrete abstract response action."""


class StableBaselinesPolicy:
    """Strict PPO adapter that never substitutes a fallback policy."""

    def __init__(
        self,
        model_path: Path,
        *,
        deterministic: bool,
        device: str,
        seed: int,
    ) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "A PPO model was requested, but Stable-Baselines3 is not installed. "
                "Install it with `python -m pip install -e \"./backend[ppo]\"` "
                "before benchmarking trained models."
            ) from exc
        self._model = PPO.load(str(model_path), device=device)
        self._model.set_random_seed(seed)
        self._deterministic = deterministic

    def act(self, observation: Any) -> int:
        action, _state = self._model.predict(
            observation,
            deterministic=self._deterministic,
        )
        values = np.asarray(action).reshape(-1)
        if values.size != 1:
            raise RuntimeError(f"PPO returned a non-scalar action with shape {values.shape}")
        return int(values[0])

    def for_episode(self, seed: int) -> "StableBaselinesPolicy":
        """Reset action-sampling RNG so paired runs use the same PPO draws."""

        self._model.set_random_seed(seed)
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FIREOPS-SIM policies under degraded sensors."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--reliabilities",
        type=float,
        nargs="+",
        default=list(DEFAULT_RELIABILITIES),
    )
    parser.add_argument("--nominal-model", type=Path)
    parser.add_argument("--robust-model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--name", default="benchmark")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5_000,
        help="Safety cap; exceeding it aborts rather than saving partial episodes.",
    )
    parser.add_argument(
        "--failure-probability",
        type=float,
        default=0.0,
        help="Seeded probability of one reset-time chaos event per episode.",
    )
    parser.add_argument(
        "--stochastic-ppo",
        action="store_true",
        help="Sample PPO actions instead of deterministic inference.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ExperimentConfigurationError("episodes must be positive")
    if args.seed < 0:
        raise ExperimentConfigurationError("seed must be non-negative")
    if args.max_steps <= 0:
        raise ExperimentConfigurationError("max-steps must be positive")
    if not 0.0 <= args.failure_probability <= 1.0:
        raise ExperimentConfigurationError(
            "failure-probability must be between 0.0 and 1.0"
        )
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        raise ExperimentConfigurationError("name must be a filename stem, not a path")
    if len(set(args.reliabilities)) != len(args.reliabilities):
        raise ExperimentConfigurationError("reliability values must be unique")
    for reliability in args.reliabilities:
        validate_reliability(reliability)


def resolve_model_path(requested: Path) -> Path:
    candidate = requested.expanduser().resolve()
    if not candidate.is_file() and candidate.suffix.lower() != ".zip":
        zipped = Path(f"{candidate}.zip")
        if zipped.is_file():
            candidate = zipped
    if not candidate.is_file():
        raise ExperimentConfigurationError(f"model artifact does not exist: {requested}")
    return candidate


def backend_policy_factories() -> dict[str, Callable[[int], Policy]]:
    add_backend_to_path()
    try:
        from app.agents import RandomAgent, RuleBasedAgent
    except ImportError as exc:
        raise RuntimeError("Could not import the backend policy implementations") from exc
    return {
        "random": lambda seed: RandomAgent(seed=seed),
        "rule": lambda _seed: RuleBasedAgent(),
    }


def extract_metrics(info: dict[str, Any], episode_return: float) -> dict[str, float]:
    source = info.get("metrics")
    if not isinstance(source, dict):
        raise RuntimeError("FireOpsEnv.step() info is missing its metrics dictionary")

    required = ("infrastructure_preserved", "burned_area", "resources_consumed")
    missing = [name for name in required if name not in source]
    if missing:
        raise RuntimeError(f"episode metrics are missing: {', '.join(missing)}")

    values: dict[str, float] = {"cumulative_reward": float(episode_return)}
    for name in required:
        value = source[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"episode metric {name!r} is not numeric")
        values[name] = float(value)

    reported_return = source.get("cumulative_reward")
    if isinstance(reported_return, (int, float)) and not isinstance(reported_return, bool):
        # The public info metric is rounded to four decimals for serialization.
        if not np.isclose(float(reported_return), episode_return, rtol=1e-6, atol=5e-4):
            raise RuntimeError(
                "environment cumulative_reward disagrees with the sum of step rewards"
            )
    return values


def run_episode(
    policy: Policy,
    *,
    reliability: float,
    episode_seed: int,
    inject_failure: bool,
    max_steps: int,
) -> dict[str, Any]:
    environment = create_environment(reliability=reliability, seed=episode_seed)
    try:
        observation, info = reset_environment(
            environment,
            reliability=reliability,
            seed=episode_seed,
            inject_failure=inject_failure,
        )
        injected_failure = info.get("injected_failure")
        if inject_failure and injected_failure not in FAILURE_TYPES:
            raise RuntimeError(
                "FireOpsEnv did not report the requested reset-time failure type"
            )
        if not inject_failure and injected_failure is not None:
            raise RuntimeError("FireOpsEnv reported an unexpected reset-time failure")
        episode_return = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            if steps >= max_steps:
                raise RuntimeError(
                    f"episode seed {episode_seed} exceeded the {max_steps}-step safety cap"
                )
            action = int(policy.act(observation))
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_return += float(reward)
            steps += 1

        result: dict[str, Any] = {
            "seed": episode_seed,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_requested": inject_failure,
            "injected_failure": injected_failure,
        }
        result.update(extract_metrics(info, episode_return))
        return result
    finally:
        environment.close()


def summarize(
    policy_name: str,
    reliability: float,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "policy": policy_name,
        "sensor_reliability": reliability,
        "episodes": len(episodes),
    }
    for metric in METRIC_NAMES:
        values = [float(episode[metric]) for episode in episodes]
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = statistics.pstdev(values)
    summary["failure_episodes"] = sum(
        episode["injected_failure"] is not None for episode in episodes
    )
    for failure_type in FAILURE_TYPES:
        summary[f"failure_{failure_type}_episodes"] = sum(
            episode["injected_failure"] == failure_type for episode in episodes
        )
    return summary


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty benchmark CSV")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ExperimentConfigurationError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    factories = backend_policy_factories()
    import gymnasium
    from app import __version__ as fireops_version
    from app.models.config import SimulationConfig
    model_metadata: dict[str, dict[str, str]] = {}
    ppo_models: dict[str, Path] = {}
    for label, requested in (
        ("ppo_nominal", args.nominal_model),
        ("ppo_robust", args.robust_model),
    ):
        if requested is None:
            continue
        try:
            model_path = resolve_model_path(requested)
        except ExperimentConfigurationError as exc:
            raise SystemExit(f"configuration error: {exc}") from exc
        ppo_models[label] = model_path
        model_metadata[label] = {
            "path": public_artifact_path(model_path),
            "sha256": sha256_file(model_path),
        }

    episodes_by_cell: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    reliability_values = [validate_reliability(value) for value in args.reliabilities]

    # Every reliability reuses the same episode seeds, failure schedule, and
    # policy-side seed. This paired design changes observation reliability while
    # holding the initial world and random action stream constant.
    episode_seeds = [args.seed + episode_index for episode_index in range(args.episodes)]
    failure_rng = np.random.default_rng(args.seed + 1_000_000)
    failure_schedule = [
        bool(failure_rng.random() < args.failure_probability)
        for _ in range(args.episodes)
    ]

    policy_factories = dict(factories)
    for name, model_path in ppo_models.items():
        adapter = StableBaselinesPolicy(
            model_path,
            deterministic=not args.stochastic_ppo,
            device=args.device,
            seed=args.seed,
        )
        policy_factories[name] = lambda seed, policy=adapter: policy.for_episode(seed)

    for reliability in reliability_values:
        for policy_name, policy_factory in policy_factories.items():
            cell_episodes = [
                run_episode(
                    policy_factory(episode_seed),
                    reliability=reliability,
                    episode_seed=episode_seed,
                    inject_failure=inject_failure,
                    max_steps=args.max_steps,
                )
                for episode_seed, inject_failure in zip(
                    episode_seeds,
                    failure_schedule,
                    strict=True,
                )
            ]
            episodes_by_cell.append(
                {
                    "policy": policy_name,
                    "sensor_reliability": reliability,
                    "episodes": cell_episodes,
                }
            )
            summaries.append(summarize(policy_name, reliability, cell_episodes))

    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{args.name}.json"
    csv_path = output_directory / f"{args.name}.csv"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "fireops_sim": fireops_version,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
        },
        "configuration": {
            "base_seed": args.seed,
            "episodes_per_cell": args.episodes,
            "sensor_reliabilities": reliability_values,
            "failure_probability": args.failure_probability,
            "max_steps": args.max_steps,
            "ppo_deterministic": not args.stochastic_ppo,
            "base_environment_config": SimulationConfig().model_dump(mode="json"),
            "paired_design": {
                "same_episode_seeds_across_reliabilities": True,
                "same_failure_schedule_across_reliabilities": True,
                "policy_rng_reseeded_per_episode": True,
            },
        },
        "models": model_metadata,
        "summaries": summaries,
        "episode_results": episodes_by_cell,
    }
    atomic_write_json(json_path, payload)
    atomic_write_csv(csv_path, summaries)

    for row in summaries:
        print(
            f"{row['policy']:>11} | reliability={row['sensor_reliability']:.2f} | "
            f"reward={row['cumulative_reward_mean']:.2f} +/- "
            f"{row['cumulative_reward_std']:.2f} | "
            f"infra={row['infrastructure_preserved_mean']:.2f} | "
            f"burned={row['burned_area_mean']:.2f} | "
            f"resources={row['resources_consumed_mean']:.2f}"
        )
    print(f"Saved episode data and summaries: {json_path}")
    print(f"Saved flat summaries: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
