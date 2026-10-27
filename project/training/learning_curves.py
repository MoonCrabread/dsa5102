"""Plot evaluation and training diagnostics from a Task C run."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsa5102-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(run_dir: Path) -> list[Path]:
    """Write four labeled, high resolution curves from recorded CSV data."""
    with (run_dir / "eval_history.csv").open(newline="", encoding="utf-8") as file:
        evaluations = list(csv.DictReader(file))
    with (run_dir / "training_steps.csv").open(newline="", encoding="utf-8") as file:
        training = list(csv.DictReader(file))
    if not evaluations or not training:
        raise ValueError("evaluation and training CSV data are required")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    specs = (
        ("evaluation_lines", evaluations, "lines_cleared_mean", "Mean lines cleared"),
        (
            "evaluation_reward",
            evaluations,
            "environment_reward_mean",
            "Mean environment reward",
        ),
        ("training_td_loss", training, "td_loss", "TD loss"),
        ("exploration_epsilon", training, "epsilon", "Epsilon"),
    )
    paths = []
    for filename, rows, field, ylabel in specs:
        points = [
            (int(row["step"]), float(row[field])) for row in rows if row[field] != ""
        ]
        figure, axes = plt.subplots(figsize=(7, 4), layout="constrained")
        if points:
            axes.plot(*zip(*points), linewidth=1.8)
        elif field == "td_loss":
            axes.text(
                0.5,
                0.5,
                "No TD updates recorded",
                ha="center",
                va="center",
                transform=axes.transAxes,
            )
        else:
            plt.close(figure)
            raise ValueError(f"no data for {field}")
        axes.set(xlabel="Environment steps", ylabel=ylabel, title=ylabel)
        axes.grid(alpha=0.25)
        path = figures / f"{filename}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        paths.append(path)
    return paths
