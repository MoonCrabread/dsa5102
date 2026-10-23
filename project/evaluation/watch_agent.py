"""Evaluate an agent and optionally watch it play in an OpenCV window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from project.agents.afterstate import AfterstateNetwork, DeepAfterstateAgent, HeuristicAgent
from project.environment.environment_factory import make_project_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("heuristic", "model"), default="heuristic")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def load_agent(args: argparse.Namespace):
    if args.agent == "heuristic":
        return HeuristicAgent()
    if args.model is None:
        raise SystemExit("--model is required when --agent=model")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    network = AfterstateNetwork(hidden_size=int(checkpoint.get("hidden_size", 128)))
    network.load_state_dict(checkpoint["state_dict"])
    network.to(device).eval()
    return DeepAfterstateAgent(network, device)


def main() -> None:
    args = parse_args()
    agent = load_agent(args)
    env = make_project_env(
        seed=args.seed,
        render_mode=None if args.no_render else "rgb_array",
        record_episode_statistics=False,
    )
    results = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, info = env.reset(seed=args.seed + episode - 1)
            reward_total = lines_total = steps = 0
            done = False

            while not done and steps < args.max_steps:
                action = agent.select_action(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_total += float(reward)
                lines_total += int(info.get("lines_cleared", 0))
                steps += 1
                done = bool(terminated or truncated)

                if not args.no_render:
                    frame = env.render()
                    if frame is not None:
                        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                        cv2.imshow("Tetris AI", frame)
                        if cv2.waitKey(max(1, int(args.delay * 1000))) & 0xFF == 27:
                            return
                    else:
                        time.sleep(args.delay)

            results.append((lines_total, reward_total, steps))
            print(
                f"episode={episode} lines={lines_total} "
                f"reward={reward_total:.1f} steps={steps}"
            )
    finally:
        env.close()
        cv2.destroyAllWindows()

    values = np.asarray(results, dtype=np.float64)
    print(
        f"mean_lines={values[:, 0].mean():.2f} "
        f"mean_reward={values[:, 1].mean():.2f} "
        f"mean_steps={values[:, 2].mean():.2f}"
    )


if __name__ == "__main__":
    main()
