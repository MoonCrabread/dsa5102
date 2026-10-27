"""Train a DQN or Double-DQN agent on grouped Tetris afterstates."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from project.agents.afterstate import AfterstateNetwork, HeuristicAgent
from project.environment.environment_factory import make_project_env


@dataclass
class Transition:
    """One replay transition with separate Gymnasium ending signals."""

    selected_features: np.ndarray
    reward: float
    next_candidates: np.ndarray
    next_mask: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class OptimizationResult:
    """Scalar diagnostics produced by one optimizer update."""

    td_loss: float
    imitation_loss: float | None
    total_loss: float
    grad_norm: float


class ActionSelector(Protocol):
    """Structural type for agents that choose a masked grouped action."""

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Choose an action from an observation and its action-mask info."""
        ...


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for one DQN or DDQN training run."""
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
    parser.add_argument("--line-bonus", type=float, default=0.0)
    parser.add_argument(
        "--imitation-steps",
        type=int,
        default=0,
        help="Warm-start the network by imitating the heuristic agent.",
    )
    parser.add_argument(
        "--imitation-weight",
        type=float,
        default=0.0,
        help="Weight of heuristic regularization during TD learning.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("project/results"))
    return parser.parse_args()


def epsilon_at(step: int, start: float, end: float, decay_steps: int) -> float:
    """Return linearly decayed epsilon at an environment step."""
    fraction = min(step / max(decay_steps, 1), 1.0)
    return start + fraction * (end - start)


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid training configurations before creating run artifacts."""
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be positive and finite")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.buffer_size < args.batch_size:
        raise ValueError("--buffer-size must be at least --batch-size")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts cannot be negative")
    if args.target_update <= 0:
        raise ValueError("--target-update must be positive")
    if not 0.0 <= args.epsilon_start <= 1.0:
        raise ValueError("--epsilon-start must be in [0, 1]")
    if not 0.0 <= args.epsilon_end <= 1.0:
        raise ValueError("--epsilon-end must be in [0, 1]")
    if args.epsilon_decay_steps < 0:
        raise ValueError("--epsilon-decay-steps cannot be negative")
    if not np.isfinite(args.line_bonus):
        raise ValueError("--line-bonus must be finite")
    if args.imitation_steps < 0:
        raise ValueError("--imitation-steps cannot be negative")
    if args.imitation_weight < 0 or not np.isfinite(args.imitation_weight):
        raise ValueError("--imitation-weight must be non-negative and finite")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")


def masked_values(
    network: nn.Module, candidates: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Evaluate candidates and mask illegal actions with negative infinity."""
    values = network(candidates)
    if values.shape != mask.shape:
        raise ValueError(
            f"network values shape {values.shape} does not match mask shape {mask.shape}"
        )
    return values.masked_fill(~mask, -torch.inf)


def require_finite(name: str, value: torch.Tensor) -> None:
    """Fail immediately when a value used for optimization is NaN or infinite."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")


@torch.no_grad()
def compute_td_targets(
    *,
    online: nn.Module,
    target: nn.Module,
    rewards: torch.Tensor,
    next_candidates: torch.Tensor,
    next_masks: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    algorithm: str,
) -> torch.Tensor:
    """Compute finite DQN/DDQN targets for transitions with legal successors.

    Gymnasium termination ends the underlying task and therefore disables
    bootstrapping. Truncation does not: a truncated transition bootstraps when
    its successor still has a legal action. Rows with an all-false action mask
    never enter max/argmax, avoiding ``0 * -inf`` and invalid DDQN actions.
    """
    if algorithm not in {"dqn", "ddqn"}:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if rewards.ndim != 1 or terminated.shape != rewards.shape:
        raise ValueError("rewards and terminated must be one-dimensional and aligned")
    if next_candidates.shape[:2] != next_masks.shape:
        raise ValueError("next candidate and action-mask shapes do not align")
    if next_candidates.shape[0] != rewards.shape[0]:
        raise ValueError("transition batch dimensions do not align")

    require_finite("rewards", rewards)
    next_values = torch.zeros_like(rewards)
    can_bootstrap = ~terminated.bool() & next_masks.bool().any(dim=1)

    if can_bootstrap.any():
        candidates = next_candidates[can_bootstrap]
        masks = next_masks[can_bootstrap].bool()
        if algorithm == "ddqn":
            next_actions = masked_values(online, candidates, masks).argmax(dim=1)
            bootstrap_values = (
                target(candidates).gather(1, next_actions[:, None]).squeeze(1)
            )
        else:
            bootstrap_values = (
                masked_values(target, candidates, masks).max(dim=1).values
            )
        require_finite("bootstrap values", bootstrap_values)
        next_values[can_bootstrap] = bootstrap_values

    expected = rewards + gamma * next_values
    require_finite("TD targets", expected)
    return expected


def optimize_batch(
    *,
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Transition],
    algorithm: str,
    gamma: float,
    imitation_weight: float,
    teacher: ActionSelector | None,
    device: torch.device,
) -> OptimizationResult:
    """Apply one finite TD update and optional heuristic regularization."""
    if not batch:
        raise ValueError("cannot optimize an empty batch")

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
    terminated = torch.as_tensor(
        [item.terminated for item in batch], dtype=torch.bool, device=device
    )

    predicted = online(chosen)
    require_finite("predicted values", predicted)
    expected = compute_td_targets(
        online=online,
        target=target,
        rewards=rewards,
        next_candidates=next_candidates,
        next_masks=next_masks,
        terminated=terminated,
        gamma=gamma,
        algorithm=algorithm,
    )
    td_loss = nn.functional.smooth_l1_loss(predicted, expected)
    require_finite("TD loss", td_loss)

    imitation_loss: torch.Tensor | None = None
    loss = td_loss
    if imitation_weight > 0.0:
        if teacher is None:
            raise ValueError("a teacher is required when imitation is enabled")
        eligible_indices = torch.nonzero(
            ~terminated & next_masks.any(dim=1), as_tuple=False
        ).flatten()
        if eligible_indices.numel() > 0:
            teacher_actions = torch.as_tensor(
                [
                    teacher.select_action(
                        batch[index].next_candidates,
                        {"action_mask": batch[index].next_mask},
                    )
                    for index in eligible_indices.tolist()
                ],
                dtype=torch.long,
                device=device,
            )
            imitation_logits = masked_values(
                online,
                next_candidates[eligible_indices],
                next_masks[eligible_indices],
            )
            computed_imitation_loss = nn.functional.cross_entropy(
                imitation_logits, teacher_actions
            )
            imitation_loss = computed_imitation_loss
            require_finite("imitation loss", imitation_loss)
            loss = loss + imitation_weight * computed_imitation_loss

    require_finite("total loss", loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    require_finite("gradient norm", torch.as_tensor(grad_norm))
    optimizer.step()

    return OptimizationResult(
        td_loss=float(td_loss.item()),
        imitation_loss=(
            None if imitation_loss is None else float(imitation_loss.item())
        ),
        total_loss=float(loss.item()),
        grad_norm=float(grad_norm.item()),
    )


def main() -> None:
    """Train an afterstate value network and save the final checkpoint."""
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run_name = args.run_name or f"{args.algorithm}_{args.seed}_{int(time.time())}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    csv_file = (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8")
    csv_writer: csv.DictWriter[str] = csv.DictWriter(
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

    reset_observation, reset_info = env.reset(seed=args.seed)
    observation = np.asarray(reset_observation, dtype=np.float32)
    info = dict(reset_info)
    teacher = (
        HeuristicAgent()
        if args.imitation_steps > 0 or args.imitation_weight > 0.0
        else None
    )

    if args.imitation_steps > 0:
        assert teacher is not None
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
            require_finite("pretraining imitation loss", imitation_loss)
            optimizer.zero_grad(set_to_none=True)
            imitation_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            require_finite("pretraining gradient norm", torch.as_tensor(grad_norm))
            optimizer.step()
            writer.add_scalar(
                "pretraining/imitation_loss", imitation_loss.item(), imitation_step
            )

            next_observation, _, terminated, truncated, next_info = env.step(action)
            observation = np.asarray(next_observation, dtype=np.float32)
            info = dict(next_info)
            if terminated or truncated:
                reset_observation, reset_info = env.reset()
                observation = np.asarray(reset_observation, dtype=np.float32)
                info = dict(reset_info)

        target.load_state_dict(online.state_dict())
        print(f"imitation_pretraining_steps={args.imitation_steps}")

    episode = episode_reward = episode_lines = episode_length = 0

    try:
        for step in range(1, args.steps + 1):
            epsilon = epsilon_at(
                step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps
            )
            legal = np.flatnonzero(np.asarray(info["action_mask"]) == 1)
            if legal.size == 0:
                raise RuntimeError(
                    "environment returned no legal action before episode end"
                )

            if random.random() < epsilon:
                action = int(np.random.choice(legal))
            else:
                with torch.no_grad():
                    candidates = torch.as_tensor(
                        observation[legal], dtype=torch.float32, device=device
                    )
                    action = int(legal[int(torch.argmax(online(candidates)).item())])

            selected_features = np.asarray(observation[action], dtype=np.float32)
            raw_next_observation, env_reward, terminated, truncated, raw_next_info = (
                env.step(action)
            )
            next_observation = np.asarray(raw_next_observation, dtype=np.float32)
            next_info = dict(raw_next_info)
            lines = int(next_info.get("lines_cleared", 0))
            reward = float(env_reward) + args.line_bonus * lines
            episode_finished = bool(terminated or truncated)

            replay.append(
                Transition(
                    selected_features=selected_features,
                    reward=reward,
                    next_candidates=np.asarray(next_observation, dtype=np.float32),
                    next_mask=np.asarray(next_info["action_mask"], dtype=bool),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )

            episode_reward += float(env_reward)
            episode_lines += lines
            episode_length += 1

            if step >= args.learning_starts and len(replay) >= args.batch_size:
                batch = random.sample(replay, args.batch_size)
                try:
                    result = optimize_batch(
                        online=online,
                        target=target,
                        optimizer=optimizer,
                        batch=batch,
                        algorithm=args.algorithm,
                        gamma=args.gamma,
                        imitation_weight=args.imitation_weight,
                        teacher=teacher,
                        device=device,
                    )
                except FloatingPointError as error:
                    print(f"non_finite_training_value step={step}: {error}")
                    raise
                writer.add_scalar("training/loss", result.total_loss, step)
                writer.add_scalar("training/td_loss", result.td_loss, step)
                writer.add_scalar("training/grad_norm", result.grad_norm, step)
                if result.imitation_loss is not None:
                    writer.add_scalar(
                        "training/imitation_regularization",
                        result.imitation_loss,
                        step,
                    )

            if step % args.target_update == 0:
                target.load_state_dict(online.state_dict())

            if episode_finished:
                episode += 1
                row: dict[str, Any] = {
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
                reset_observation, reset_info = env.reset()
                observation = np.asarray(reset_observation, dtype=np.float32)
                info = dict(reset_info)
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
