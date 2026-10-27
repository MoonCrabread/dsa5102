"""Record comparable grouped-action gameplay from training checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from project.environment.environment_factory import make_project_env
from project.evaluation.evaluate import load_checkpoint_agent


def record_gameplay(
    checkpoints: list[Path], output_dir: Path, seed: int, episode_cap: int
) -> list[Path]:
    """Record each checkpoint with the same seed, cap, and grouped environment."""
    if episode_cap <= 0 or not checkpoints:
        raise ValueError("checkpoints and positive episode_cap are required")
    recordings = []
    for checkpoint in checkpoints:
        agent, metadata = load_checkpoint_agent(checkpoint, torch.device("cpu"))
        label = (
            f"{metadata['algorithm']}_train_{int(metadata['seed'])}"
            f"_step_{int(metadata['steps']):06d}_eval_{seed}"
        )
        destination = output_dir / f"{label}.gif"
        if any(existing[2] == destination for existing in recordings):
            raise ValueError(f"checkpoint filename collision: {destination}")
        if destination.exists() or destination.with_suffix(".json").exists():
            raise FileExistsError(destination)
        recordings.append((agent, metadata, destination, checkpoint))
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for agent, metadata, destination, checkpoint in recordings:
        env = make_project_env(
            seed=seed, render_mode="rgb_array", record_episode_statistics=False
        )
        label = destination.stem
        try:
            observation, info = env.reset(seed=seed)
            frames = [np.asarray(env.render())]
            lines = 0
            reward_total = 0.0
            terminated = truncated = False
            steps = 0
            while steps < episode_cap and not (terminated or truncated):
                action = agent.select_action(np.asarray(observation), dict(info))
                if not np.asarray(info["action_mask"], dtype=bool)[action]:
                    raise ValueError(f"illegal action {action}")
                observation, reward, terminated, truncated, info = env.step(action)
                frames.append(np.asarray(env.render()))
                lines += int(info.get("lines_cleared", 0))
                reward_total += float(reward)
                steps += 1
            imageio.mimsave(destination, frames, duration=0.12, loop=0)
            (output_dir / f"{label}.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "algorithm": metadata["algorithm"],
                        "training_step": metadata["steps"],
                        "train_seed": metadata["seed"],
                        "eval_seed": seed,
                        "episode_cap": episode_cap,
                        "episode_length": steps,
                        "lines_cleared": lines,
                        "environment_reward": reward_total,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "cap_reached": steps == episode_cap
                        and not (terminated or truncated),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            outputs.append(destination)
        finally:
            env.close()
    return outputs


def main() -> None:
    """Record initial, middle, and final checkpoint gameplay."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs=3,
        type=Path,
        required=True,
        metavar=("EARLY", "MIDDLE", "FINAL"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--episode-cap", type=int, default=2_000)
    args = parser.parse_args()
    for path in record_gameplay(
        args.checkpoints, args.output_dir, args.eval_seed, args.episode_cap
    ):
        print(f"video={path}")


if __name__ == "__main__":
    main()
