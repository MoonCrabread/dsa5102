"""Agents for the grouped-action, feature-vector Tetris environment."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RandomLegalAgent:
    """Sample uniformly from the legal grouped actions."""

    def __init__(self, seed: int | None = None):
        """Initialize the independent action RNG."""
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the action RNG for an independently reproducible episode."""
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select one legal action without using the observation values."""
        del observation
        mask = np.asarray(info["action_mask"], dtype=bool)
        if mask.ndim != 1:
            raise ValueError(f"action mask must be one-dimensional, got {mask.shape}")
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError("cannot select an action from an all-false action mask")
        return int(self.rng.choice(legal))


class AfterstateNetwork(nn.Module):
    """Estimate the value of a 13-dimensional candidate afterstate."""

    def __init__(self, feature_count: int = 13, hidden_size: int = 128):
        """Build the value network for candidate afterstate features."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Estimate one scalar value for each candidate afterstate."""
        return self.network(features.float()).squeeze(-1)


class HeuristicAgent:
    """Fast baseline that prefers low, flat boards with few holes."""

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Choose the highest-valued legal action using board heuristics."""
        legal = np.flatnonzero(np.asarray(info["action_mask"]) == 1)
        candidates = np.asarray(observation, dtype=np.float32)[legal]

        aggregate_height = candidates[:, :10].sum(axis=1)
        maximum_height = candidates[:, 10]
        holes = candidates[:, 11]
        bumpiness = candidates[:, 12]

        values = (
            -0.510066 * aggregate_height
            - 0.35663 * maximum_height
            - 0.760666 * holes
            - 0.184483 * bumpiness
        )
        return int(legal[int(np.argmax(values))])


class DeepAfterstateAgent:
    """Choose the legal candidate with the largest learned value."""

    def __init__(self, network: AfterstateNetwork, device: torch.device):
        """Store an afterstate network and the device used for inference."""
        self.network = network
        self.device = device

    @torch.no_grad()
    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Choose the legal candidate with the highest network value."""
        legal = np.flatnonzero(np.asarray(info["action_mask"]) == 1)
        features = torch.as_tensor(
            np.asarray(observation)[legal], dtype=torch.float32, device=self.device
        )
        values = self.network(features)
        return int(legal[int(torch.argmax(values).item())])
