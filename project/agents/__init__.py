"""Agents that share the project-wide Tetris environment interface."""

from .afterstate import (
    AfterstateNetwork,
    DeepAfterstateAgent,
    HeuristicAgent,
    RandomLegalAgent,
)

__all__ = [
    "AfterstateNetwork",
    "DeepAfterstateAgent",
    "HeuristicAgent",
    "RandomLegalAgent",
]
