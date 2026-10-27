"""Task C checkpoint, scheduling, plotting, and recording checks."""

import argparse
import csv
import json

import imageio.v2 as imageio
import numpy as np
import torch
import pytest

from project.agents.afterstate import AfterstateNetwork
from project.evaluation.evaluate import load_checkpoint_agent
from project.evaluation.record_gameplay import record_gameplay
from project.training.learning_curves import plot_learning_curves
from project.training.train_afterstate import due, evaluate_snapshot, save_checkpoint


def test_schedule_includes_intervals_and_final():
    assert [step for step in range(1, 11) if due(step, 4, 10)] == [4, 8, 10]


def test_checkpoint_remains_compatible_with_task_b(tmp_path):
    model = AfterstateNetwork()
    optimizer = torch.optim.Adam(model.parameters())
    args = argparse.Namespace(algorithm="dqn", seed=42, output_dir=tmp_path)
    path = tmp_path / "step_000010.pt"
    save_checkpoint(path, model, optimizer, args, 10)
    agent, metadata = load_checkpoint_agent(path, torch.device("cpu"))
    assert metadata["steps"] == 10
    assert metadata["config"]["algorithm"] == "dqn"
    assert "optimizer_state_dict" in metadata
    assert torch.isfinite(agent.network(torch.zeros(1, 13))).all()


def test_plots_can_be_reopened(tmp_path):
    with (tmp_path / "eval_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("step", "lines_cleared_mean", "environment_reward_mean")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"step": 0, "lines_cleared_mean": 1, "environment_reward_mean": 2},
                {"step": 10, "lines_cleared_mean": 3, "environment_reward_mean": 4},
            ]
        )
    with (tmp_path / "training_steps.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "td_loss", "epsilon"))
        writer.writeheader()
        writer.writerow({"step": 1, "td_loss": 0.5, "epsilon": 0.9})
    paths = plot_learning_curves(tmp_path)
    assert len(paths) == 4
    assert all(imageio.imread(path).size > 0 for path in paths)


def test_plot_marks_missing_td_updates(tmp_path, monkeypatch):
    import matplotlib.axes

    labels = []
    original = matplotlib.axes.Axes.text

    def record_label(self, *args, **kwargs):
        labels.append(args[2])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", record_label)
    with (tmp_path / "eval_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("step", "lines_cleared_mean", "environment_reward_mean")
        )
        writer.writeheader()
        writer.writerow(
            {"step": 0, "lines_cleared_mean": 0, "environment_reward_mean": 0}
        )
    with (tmp_path / "training_steps.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "td_loss", "epsilon"))
        writer.writeheader()
        writer.writerow({"step": 1, "td_loss": "", "epsilon": 1})
    paths = plot_learning_curves(tmp_path)
    assert len(paths) == 4
    assert "No TD updates recorded" in labels
    assert all(imageio.imread(path).size for path in paths)


@pytest.mark.parametrize("initial_mode", [True, False])
def test_periodic_evaluation_preserves_parameters_and_mode(monkeypatch, initial_mode):
    from project.evaluation.evaluate import EpisodeResult
    from io import StringIO

    model = AfterstateNetwork()
    model.train(initial_mode)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    def fake_evaluate(agent, **kwargs):
        agent.network.eval()
        assert agent.network.training is False
        return [EpisodeResult("dqn", 42, 7001, 0, 1.0, 1, True, False, False)]

    monkeypatch.setattr(
        "project.training.train_afterstate.evaluate_agent", fake_evaluate
    )
    args = argparse.Namespace(
        algorithm="dqn", seed=42, eval_seed_start=7001, eval_episodes=1, eval_cap=10
    )
    from project.evaluation.evaluate import AGGREGATE_FIELDS, PER_EPISODE_FIELDS

    summary = csv.DictWriter(
        StringIO(), fieldnames=("step", "eval_seeds", "episode_cap", *AGGREGATE_FIELDS)
    )
    raw = csv.DictWriter(StringIO(), fieldnames=("step", *PER_EPISODE_FIELDS))
    evaluate_snapshot(model, args, 0, summary, raw)
    assert model.training is initial_mode
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_recording_uses_checkpoint_metadata_and_reopens_gif(tmp_path, monkeypatch):
    class FakeEnv:
        def reset(self, *, seed):
            assert seed == 77
            return np.zeros((40, 13), dtype=np.float32), {
                "action_mask": np.ones(40, dtype=bool)
            }

        def step(self, action):
            return (
                np.zeros((40, 13), dtype=np.float32),
                1.0,
                True,
                False,
                {"lines_cleared": 1, "action_mask": np.ones(40, dtype=bool)},
            )

        def render(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(
        "project.evaluation.record_gameplay.make_project_env",
        lambda **kwargs: FakeEnv(),
    )
    checkpoint = tmp_path / "model.pt"
    model = AfterstateNetwork()
    torch.save(
        {
            "algorithm": "dqn",
            "steps": 10,
            "seed": 42,
            "hidden_size": 128,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    paths = record_gameplay([checkpoint], tmp_path / "videos", 77, 5)
    assert len(paths) == 1
    assert imageio.mimread(paths[0])
    metadata = json.loads(paths[0].with_suffix(".json").read_text())
    assert metadata["eval_seed"] == 77
    assert metadata["lines_cleared"] == 1
    assert metadata["cap_reached"] is False
    assert "train_42" in paths[0].name


def test_recording_rejects_checkpoint_filename_collisions(tmp_path):
    model = AfterstateNetwork()
    first, second = tmp_path / "early.pt", tmp_path / "duplicate.pt"
    for path in (first, second):
        torch.save(
            {
                "algorithm": "dqn",
                "steps": 10,
                "seed": 42,
                "hidden_size": 128,
                "state_dict": model.state_dict(),
            },
            path,
        )
    with pytest.raises(ValueError, match="collision"):
        record_gameplay([first, second], tmp_path / "videos", 77, 5)
    assert not (tmp_path / "videos").exists()
