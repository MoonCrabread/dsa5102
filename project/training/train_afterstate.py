"""Train a DQN or Double-DQN agent on grouped Tetris afterstates."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from project.agents.afterstate import AfterstateNetwork, HeuristicAgent
from project.environment.environment_factory import make_project_env


@dataclass
class Transition:
    selected_features: np.ndarray
    reward: float
    next_candidates: np.ndarray
    next_mask: np.ndarray
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("dqn", "ddqn"), default="ddqn")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--line-bonus", type=float, default=10.0)
    parser.add_argument(
        "--imitation-steps",
        type=int,
        default=2_000,
        help="Warm-start the network by imitating the heuristic agent.",
    )
    parser.add_argument(
        "--imitation-weight",
        type=float,
        default=1.0,
        help="Weight of heuristic regularization during TD learning.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("project/results"))
    return parser.parse_args()


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(step / max(decay_steps, 1), 1.0)
    return start + fraction * (end - start)


def masked_values(network: nn.Module, candidates: torch.Tensor, mask: torch.Tensor):
    values = network(candidates)
    return values.masked_fill(~mask, -torch.inf)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run_name = args.run_name or f"{args.algorithm}_{args.seed}_{int(time.time())}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    csv_file = (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=("episode", "step", "reward", "lines", "length", "epsilon"),
    )
    csv_writer.writeheader()

    env = make_project_env(seed=args.seed, record_episode_statistics=False)
    online = AfterstateNetwork().to(device)
    target = AfterstateNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate)
    replay: deque[Transition] = deque(maxlen=args.buffer_size)

    observation, info = env.reset(seed=args.seed)
    teacher = HeuristicAgent()

    if args.imitation_steps > 0:
        for imitation_step in range(1, args.imitation_steps + 1):
            action = teacher.select_action(observation, info)
            candidates = torch.as_tensor(
                observation[None, ...], dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(
                np.asarray(info["action_mask"], dtype=bool)[None, ...],
                dtype=torch.bool,
                device=device,
            )
            logits = masked_values(online, candidates, mask)
            target_action = torch.as_tensor([action], dtype=torch.long, device=device)
            imitation_loss = nn.functional.cross_entropy(logits, target_action)
            optimizer.zero_grad()
            imitation_loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            writer.add_scalar("pretraining/imitation_loss", imitation_loss.item(), imitation_step)

            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()

        target.load_state_dict(online.state_dict())
        print(f"imitation_pretraining_steps={args.imitation_steps}")

    episode = episode_reward = episode_lines = episode_length = 0

    try:
        for step in range(1, args.steps + 1):
            epsilon = epsilon_at(
                step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            )
            legal = np.flatnonzero(np.asarray(info["action_mask"]) == 1)

            if random.random() < epsilon:
                action = int(np.random.choice(legal))
            else:
                with torch.no_grad():
                    candidates = torch.as_tensor(
                        observation[legal], dtype=torch.float32, device=device
                    )
                    action = int(legal[int(torch.argmax(online(candidates)).item())])

            selected_features = np.asarray(observation[action], dtype=np.float32)
            next_observation, env_reward, terminated, truncated, next_info = env.step(action)
            lines = int(next_info.get("lines_cleared", 0))
            reward = float(env_reward) + args.line_bonus * lines
            done = bool(terminated or truncated)

            replay.append(
                Transition(
                    selected_features,
                    reward,
                    np.asarray(next_observation, dtype=np.float32),
                    np.asarray(next_info["action_mask"], dtype=bool),
                    done,
                )
            )

            episode_reward += float(env_reward)
            episode_lines += lines
            episode_length += 1

            if step >= args.learning_starts and len(replay) >= args.batch_size:
                batch = random.sample(replay, args.batch_size)
                chosen = torch.as_tensor(
                    np.stack([item.selected_features for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                rewards = torch.as_tensor(
                    [item.reward for item in batch], dtype=torch.float32, device=device
                )
                next_candidates = torch.as_tensor(
                    np.stack([item.next_candidates for item in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                next_masks = torch.as_tensor(
                    np.stack([item.next_mask for item in batch]),
                    dtype=torch.bool,
                    device=device,
                )
                dones = torch.as_tensor(
                    [item.done for item in batch], dtype=torch.float32, device=device
                )

                predicted = online(chosen)
                with torch.no_grad():
                    if args.algorithm == "ddqn":
                        next_actions = masked_values(
                            online, next_candidates, next_masks
                        ).argmax(dim=1)
                        next_values = target(next_candidates).gather(
                            1, next_actions[:, None]
                        ).squeeze(1)
                    else:
                        next_values = masked_values(
                            target, next_candidates, next_masks
                        ).max(dim=1).values
                    expected = rewards + args.gamma * (1.0 - dones) * next_values

                td_loss = nn.functional.smooth_l1_loss(predicted, expected)
                teacher_actions = torch.as_tensor(
                    [
                        teacher.select_action(
                            item.next_candidates,
                            {"action_mask": item.next_mask},
                        )
                        for item in batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
                imitation_loss = nn.functional.cross_entropy(
                    masked_values(online, next_candidates, next_masks),
                    teacher_actions,
                )
                loss = td_loss + args.imitation_weight * imitation_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
                writer.add_scalar("training/loss", loss.item(), step)
                writer.add_scalar("training/td_loss", td_loss.item(), step)
                writer.add_scalar(
                    "training/imitation_regularization", imitation_loss.item(), step
                )

            if step % args.target_update == 0:
                target.load_state_dict(online.state_dict())

            if done:
                episode += 1
                row = {
                    "episode": episode,
                    "step": step,
                    "reward": episode_reward,
                    "lines": episode_lines,
                    "length": episode_length,
                    "epsilon": epsilon,
                }
                csv_writer.writerow(row)
                csv_file.flush()
                for name in ("reward", "lines", "length"):
                    writer.add_scalar(f"episode/{name}", row[name], step)
                writer.add_scalar("training/epsilon", epsilon, step)
                if episode == 1 or episode % args.print_every == 0:
                    print(
                        f"episode={episode} step={step} reward={episode_reward:.1f} "
                        f"lines={episode_lines} length={episode_length} epsilon={epsilon:.3f}"
                    )
                observation, info = env.reset()
                episode_reward = episode_lines = episode_length = 0
            else:
                observation, info = next_observation, next_info

        checkpoint = {
            "algorithm": args.algorithm,
            "hidden_size": 128,
            "state_dict": online.state_dict(),
            "steps": args.steps,
            "seed": args.seed,
        }
        torch.save(checkpoint, run_dir / "model.pt")
        print(f"saved_model={run_dir / 'model.pt'}")
    finally:
        env.close()
        csv_file.close()
        writer.close()


if __name__ == "__main__":
    main()
