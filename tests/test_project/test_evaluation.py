"""Tests for the project-wide baseline and checkpoint evaluator."""

from __future__ import annotations

import csv
from dataclasses import asdict

import numpy as np
import pytest
import torch

from project.agents.afterstate import (
    AfterstateNetwork,
    DeepAfterstateAgent,
    RandomLegalAgent,
)
from project.evaluation.evaluate import (
    EpisodeResult,
    evaluate_agent,
    load_checkpoint_agent,
    write_results,
)


class SeededEvaluationEnv:
    """Small deterministic environment with changing legal-action masks."""

    def __init__(self, *, terminate_after: int | None = 3):
        self.terminate_after = terminate_after
        self.actions: list[int] = []
        self.closed = False
        self._seed = 0
        self._step = 0
        self._mask = np.ones(4, dtype=bool)

    def _observation(self) -> np.ndarray:
        observation = np.zeros((4, 13), dtype=np.float32)
        observation[:, 0] = np.arange(4, dtype=np.float32)
        observation[:, 10] = float(self._step)
        return observation

    def _action_mask(self) -> np.ndarray:
        first = (self._seed + self._step) % 4
        mask = np.zeros(4, dtype=bool)
        mask[first] = True
        mask[(first + 2) % 4] = True
        return mask

    def reset(self, *, seed: int):
        self._seed = seed
        self._step = 0
        self._mask = self._action_mask()
        return self._observation(), {"action_mask": self._mask.copy()}

    def step(self, action: int):
        assert self._mask[action], f"illegal action selected: {action}"
        self.actions.append(action)
        reward = float((self._seed % 7) + action)
        lines = int(action == 2)
        self._step += 1
        terminated = (
            self.terminate_after is not None and self._step >= self.terminate_after
        )
        self._mask = self._action_mask()
        return (
            self._observation(),
            reward,
            terminated,
            False,
            {"action_mask": self._mask.copy(), "lines_cleared": lines},
        )

    def close(self) -> None:
        self.closed = True


class IllegalAgent:
    """Always return an action that the fake environment marks illegal."""

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        del observation, info
        return 1


def make_env_factory(created: list[SeededEvaluationEnv], *, terminate_after=3):
    """Return an injectable environment factory and retain its instances."""

    def factory(**kwargs):
        del kwargs
        env = SeededEvaluationEnv(terminate_after=terminate_after)
        created.append(env)
        return env

    return factory


def test_random_legal_agent_selects_only_masked_actions():
    agent = RandomLegalAgent(seed=42)
    observation = np.zeros((4, 13), dtype=np.float32)
    info = {"action_mask": np.array([False, True, False, True])}

    actions = [agent.select_action(observation, info) for _ in range(100)]

    assert set(actions) == {1, 3}


def test_random_evaluation_is_reproducible_for_fixed_eval_seeds():
    created: list[SeededEvaluationEnv] = []
    factory = make_env_factory(created)
    seeds = [101, 202, 303]

    first = evaluate_agent(
        RandomLegalAgent(),
        model="random",
        train_seed=None,
        eval_seeds=seeds,
        episode_cap=10,
        env_factory=factory,
    )
    second = evaluate_agent(
        RandomLegalAgent(),
        model="random",
        train_seed=None,
        eval_seeds=seeds,
        episode_cap=10,
        env_factory=factory,
    )

    assert first == second
    assert [result.eval_seed for result in first] == seeds
    assert all(env.closed for env in created)


def test_evaluation_never_updates_checkpoint_network():
    network = AfterstateNetwork()
    agent = DeepAfterstateAgent(network, torch.device("cpu"))
    parameters_before = {
        name: value.detach().clone() for name, value in network.state_dict().items()
    }

    results = evaluate_agent(
        agent,
        model="dqn",
        train_seed=42,
        eval_seeds=[500, 501],
        episode_cap=10,
        env_factory=make_env_factory([]),
    )

    assert len(results) == 2
    assert all(
        torch.equal(parameters_before[name], value)
        for name, value in network.state_dict().items()
    )


def test_checkpoint_loader_restores_eval_network_and_metadata(tmp_path):
    network = AfterstateNetwork(hidden_size=8)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "algorithm": "dqn",
            "hidden_size": 8,
            "state_dict": network.state_dict(),
            "steps": 100,
            "seed": 42,
        },
        checkpoint_path,
    )

    agent, metadata = load_checkpoint_agent(checkpoint_path, torch.device("cpu"))

    assert metadata["algorithm"] == "dqn"
    assert metadata["seed"] == 42
    assert agent.network.training is False
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            network.state_dict().values(),
            agent.network.state_dict().values(),
            strict=True,
        )
    )


def test_evaluator_rejects_an_illegal_agent_action():
    with pytest.raises(ValueError, match="illegal action"):
        evaluate_agent(
            IllegalAgent(),
            model="invalid",
            train_seed=None,
            eval_seeds=[42],
            episode_cap=3,
            env_factory=make_env_factory([]),
        )


def test_episode_cap_is_not_reported_as_normal_game_over():
    results = evaluate_agent(
        RandomLegalAgent(),
        model="random",
        train_seed=None,
        eval_seeds=[42],
        episode_cap=2,
        env_factory=make_env_factory([], terminate_after=None),
    )

    assert results == [
        EpisodeResult(
            model="random",
            train_seed=None,
            eval_seed=42,
            lines_cleared=0,
            environment_reward=3.0,
            episode_length=2,
            terminated=False,
            truncated=False,
            cap_reached=True,
        )
    ]


def test_write_results_preserves_episode_rows_and_groups_training_seeds(tmp_path):
    results = [
        EpisodeResult("dqn", 42, 10, 1, 3.0, 5, True, False, False),
        EpisodeResult("dqn", 42, 11, 3, 7.0, 9, True, False, False),
        EpisodeResult("dqn", 123, 10, 2, 5.0, 7, False, False, True),
    ]

    per_episode_path, aggregate_path = write_results(results, tmp_path)

    with per_episode_path.open(newline="", encoding="utf-8") as file:
        episode_rows = list(csv.DictReader(file))
    with aggregate_path.open(newline="", encoding="utf-8") as file:
        aggregate_rows = list(csv.DictReader(file))

    assert len(episode_rows) == 3
    assert episode_rows[0] == {
        key: str(value) for key, value in asdict(results[0]).items()
    }
    assert [(row["model"], row["train_seed"]) for row in aggregate_rows] == [
        ("dqn", "42"),
        ("dqn", "123"),
    ]
    first = aggregate_rows[0]
    assert first["episodes"] == "2"
    assert float(first["lines_cleared_mean"]) == pytest.approx(2.0)
    assert float(first["lines_cleared_sd"]) == pytest.approx(2**0.5)
    assert float(first["lines_cleared_median"]) == pytest.approx(2.0)
    assert float(first["lines_cleared_max"]) == pytest.approx(3.0)
    assert first["cap_reached_count"] == "0"
    assert aggregate_rows[1]["cap_reached_count"] == "1"
