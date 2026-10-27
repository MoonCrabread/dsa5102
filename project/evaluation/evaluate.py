"""Evaluate project agents with fixed seeds and export reproducible CSV results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from project.agents.afterstate import (
    AfterstateNetwork,
    DeepAfterstateAgent,
    HeuristicAgent,
    RandomLegalAgent,
)
from project.environment.environment_factory import make_project_env


PER_EPISODE_FIELDS = (
    "model",
    "train_seed",
    "eval_seed",
    "lines_cleared",
    "environment_reward",
    "episode_length",
    "terminated",
    "truncated",
    "cap_reached",
)
SUMMARY_METRICS = ("lines_cleared", "environment_reward", "episode_length")
SUMMARY_STATISTICS = ("mean", "sd", "median", "max")
AGGREGATE_FIELDS = (
    "model",
    "train_seed",
    "episodes",
    *(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in SUMMARY_STATISTICS
    ),
    "terminated_count",
    "truncated_count",
    "cap_reached_count",
    "cap_reached_rate",
)


class EvaluationAgent(Protocol):
    """Common action-selection interface used by all project agents."""

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Choose an action from an observation and action-mask info."""
        ...


class EvaluationEnvironment(Protocol):
    """Minimal Gymnasium environment interface required by the evaluator."""

    def reset(self, *, seed: int):
        """Reset the environment to a deterministic episode seed."""
        ...

    def step(self, action: int):
        """Apply one grouped action and return the Gymnasium step tuple."""
        ...

    def close(self) -> None:
        """Release resources held by the environment."""
        ...


@dataclass(frozen=True)
class EpisodeResult:
    """Raw metrics from one fixed-seed evaluation episode."""

    model: str
    train_seed: int | None
    eval_seed: int
    lines_cleared: int
    environment_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    cap_reached: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one evaluation target."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", choices=("random", "heuristic", "checkpoint"), required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--eval-seeds", nargs="+", type=int)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-cap", type=int, default=2_000)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("project/results/summary")
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Reject incomplete or invalid evaluation settings before creating output."""
    if args.agent == "checkpoint" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --agent=checkpoint")
    if args.agent != "checkpoint" and args.checkpoint is not None:
        raise ValueError("--checkpoint can only be used with --agent=checkpoint")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_cap <= 0:
        raise ValueError("--episode-cap must be positive")
    if args.eval_seeds is not None and not args.eval_seeds:
        raise ValueError("--eval-seeds cannot be empty")
    if args.eval_seeds is not None and len(set(args.eval_seeds)) != len(
        args.eval_seeds
    ):
        raise ValueError("--eval-seeds must not contain duplicates")


def resolve_eval_seeds(args: argparse.Namespace) -> list[int]:
    """Return the explicit or sequential fixed seeds for this evaluation."""
    if args.eval_seeds is not None:
        return list(args.eval_seeds)
    return list(range(args.eval_seed_start, args.eval_seed_start + args.episodes))


def load_checkpoint_agent(
    checkpoint_path: Path, device: torch.device
) -> tuple[DeepAfterstateAgent, dict[str, Any]]:
    """Load a trained afterstate agent and its non-secret checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no state_dict: {checkpoint_path}")
    network = AfterstateNetwork(hidden_size=int(checkpoint.get("hidden_size", 128)))
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    network.to(device).eval()
    return DeepAfterstateAgent(network, device), checkpoint


def build_agent(
    args: argparse.Namespace,
) -> tuple[EvaluationAgent, str, int | None]:
    """Construct the requested baseline or checkpoint evaluation target."""
    if args.agent == "random":
        return RandomLegalAgent(), args.model_name or "random", None
    if args.agent == "heuristic":
        return HeuristicAgent(), args.model_name or "heuristic", None

    device = torch.device(args.device)
    agent, checkpoint = load_checkpoint_agent(args.checkpoint, device)
    train_seed = args.train_seed
    if train_seed is None and checkpoint.get("seed") is not None:
        train_seed = int(checkpoint["seed"])
    model_name = args.model_name or str(checkpoint.get("algorithm", "checkpoint"))
    return agent, model_name, train_seed


def _reset_agent_rng(agent: EvaluationAgent, eval_seed: int) -> None:
    """Reset stochastic baseline state without imposing it on deterministic agents."""
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(eval_seed)


def _put_network_in_eval_mode(agent: EvaluationAgent) -> None:
    """Disable training behavior for agents backed by a Torch network."""
    network = getattr(agent, "network", None)
    if network is not None and isinstance(network, nn.Module):
        network.eval()


def _validate_action(action: int, info: dict) -> None:
    """Fail before stepping when an agent selects an illegal grouped action."""
    mask = np.asarray(info["action_mask"], dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"action mask must be one-dimensional, got {mask.shape}")
    if not 0 <= action < mask.size or not mask[action]:
        raise ValueError(f"agent selected illegal action {action}")


def evaluate_agent(
    agent: EvaluationAgent,
    *,
    model: str,
    train_seed: int | None,
    eval_seeds: Sequence[int],
    episode_cap: int,
    env_factory: Callable[..., EvaluationEnvironment] = make_project_env,
) -> list[EpisodeResult]:
    """Evaluate an agent without exploration, learning, or parameter updates."""
    if not eval_seeds:
        raise ValueError("at least one evaluation seed is required")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("evaluation seeds must not contain duplicates")
    if episode_cap <= 0:
        raise ValueError("episode_cap must be positive")

    _put_network_in_eval_mode(agent)
    env = env_factory(seed=eval_seeds[0], record_episode_statistics=False)
    results: list[EpisodeResult] = []
    try:
        for eval_seed in eval_seeds:
            _reset_agent_rng(agent, eval_seed)
            observation, info = env.reset(seed=eval_seed)
            reward_total = 0.0
            lines_total = 0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated) and episode_length < episode_cap:
                action = int(agent.select_action(np.asarray(observation), dict(info)))
                _validate_action(action, info)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_total += float(reward)
                lines_total += int(info.get("lines_cleared", 0))
                episode_length += 1

            cap_reached = bool(
                episode_length >= episode_cap and not terminated and not truncated
            )
            results.append(
                EpisodeResult(
                    model=model,
                    train_seed=train_seed,
                    eval_seed=int(eval_seed),
                    lines_cleared=lines_total,
                    environment_reward=reward_total,
                    episode_length=episode_length,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    cap_reached=cap_reached,
                )
            )
    finally:
        env.close()

    return results


def _summary(values: Iterable[int | float]) -> dict[str, float]:
    """Calculate mean, sample SD, median, and maximum."""
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": statistics.mean(numeric),
        "sd": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
        "max": max(numeric),
    }


def aggregate_results(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    """Aggregate episode metrics separately for each model and training seed."""
    if not results:
        raise ValueError("cannot aggregate empty evaluation results")
    grouped: dict[tuple[str, int | None], list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[(result.model, result.train_seed)].append(result)

    rows: list[dict[str, Any]] = []
    for model, train_seed in sorted(
        grouped, key=lambda key: (key[0], -1 if key[1] is None else key[1])
    ):
        episodes = grouped[(model, train_seed)]
        row: dict[str, Any] = {
            "model": model,
            "train_seed": train_seed,
            "episodes": len(episodes),
        }
        for metric in SUMMARY_METRICS:
            summary = _summary(getattr(episode, metric) for episode in episodes)
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
        row["terminated_count"] = sum(episode.terminated for episode in episodes)
        row["truncated_count"] = sum(episode.truncated for episode in episodes)
        row["cap_reached_count"] = sum(episode.cap_reached for episode in episodes)
        row["cap_reached_rate"] = row["cap_reached_count"] / len(episodes)
        rows.append(row)
    return rows


def write_results(
    results: Sequence[EpisodeResult], output_dir: Path
) -> tuple[Path, Path]:
    """Write raw per-episode and grouped aggregate evaluation CSV files."""
    if not results:
        raise ValueError("cannot write empty evaluation results")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = output_dir / "per_episode_final.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"

    with per_episode_path.open("w", newline="", encoding="utf-8") as file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            file, fieldnames=PER_EPISODE_FIELDS
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    with aggregate_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(aggregate_results(results))

    return per_episode_path, aggregate_path


def main() -> None:
    """Run one fixed-seed evaluation target and export its CSV results."""
    args = parse_args()
    validate_args(args)
    agent, model, train_seed = build_agent(args)
    eval_seeds = resolve_eval_seeds(args)
    results = evaluate_agent(
        agent,
        model=model,
        train_seed=train_seed,
        eval_seeds=eval_seeds,
        episode_cap=args.episode_cap,
    )
    per_episode_path, aggregate_path = write_results(results, args.output_dir)
    print(f"evaluated_model={model} episodes={len(results)}")
    print(f"per_episode_csv={per_episode_path}")
    print(f"aggregate_csv={aggregate_path}")


if __name__ == "__main__":
    main()
