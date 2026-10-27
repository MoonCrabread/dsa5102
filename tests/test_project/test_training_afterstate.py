"""Tests for the project DQN/DDQN training calculations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from project.training.train_afterstate import (
    Transition,
    compute_td_targets,
    masked_values,
    optimize_batch,
    parse_args,
    validate_args,
)


class FeatureValueNetwork(nn.Module):
    """Return one selected candidate feature as its value."""

    def __init__(self, feature_index: int):
        super().__init__()
        self.feature_index = feature_index

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[..., self.feature_index]


class TrainableValueNetwork(nn.Module):
    """Small value network used to exercise a real optimizer update."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class FailingTeacher:
    """Fail if disabled imitation attempts to query the teacher."""

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        raise AssertionError("disabled imitation queried the teacher")


class NonFiniteNetwork(nn.Module):
    """Return NaN values to verify numerical safety checks."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full(features.shape[:-1], torch.nan, device=features.device)


def make_transition(
    reward: float = 1.0,
    *,
    terminated: bool = False,
    truncated: bool = False,
    next_mask: tuple[bool, ...] = (True, True, True),
) -> Transition:
    return Transition(
        selected_features=np.array([0.25, -0.5], dtype=np.float32),
        reward=reward,
        next_candidates=np.array(
            [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]], dtype=np.float32
        ),
        next_mask=np.asarray(next_mask, dtype=bool),
        terminated=terminated,
        truncated=truncated,
    )


def test_masked_values_excludes_illegal_actions():
    candidates = torch.tensor([[[1.0], [2.0], [3.0]]])
    mask = torch.tensor([[True, False, True]])

    values = masked_values(FeatureValueNetwork(0), candidates, mask)

    assert values[0, 0].item() == 1.0
    assert torch.isneginf(values[0, 1])
    assert values[0, 2].item() == 3.0


@pytest.mark.parametrize("algorithm, expected", [("dqn", 17.0), ("ddqn", 12.0)])
def test_td_target_matches_hand_calculation(algorithm: str, expected: float):
    transition = make_transition(reward=2.0)

    targets = compute_td_targets(
        online=FeatureValueNetwork(0),
        target=FeatureValueNetwork(1),
        rewards=torch.tensor([transition.reward]),
        next_candidates=torch.as_tensor(transition.next_candidates[None, ...]),
        next_masks=torch.as_tensor(transition.next_mask[None, ...]),
        terminated=torch.tensor([transition.terminated]),
        gamma=0.5,
        algorithm=algorithm,
    )

    assert targets.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "transition",
    [
        make_transition(reward=7.0, terminated=True),
        make_transition(reward=7.0, next_mask=(False, False, False)),
    ],
    ids=("terminated", "all-invalid-mask"),
)
@pytest.mark.parametrize("algorithm", ["dqn", "ddqn"])
def test_td_target_does_not_bootstrap_without_valid_successor(
    transition: Transition, algorithm: str
):
    targets = compute_td_targets(
        online=FeatureValueNetwork(0),
        target=FeatureValueNetwork(1),
        rewards=torch.tensor([transition.reward]),
        next_candidates=torch.as_tensor(transition.next_candidates[None, ...]),
        next_masks=torch.as_tensor(transition.next_mask[None, ...]),
        terminated=torch.tensor([transition.terminated]),
        gamma=0.99,
        algorithm=algorithm,
    )

    assert targets.item() == pytest.approx(transition.reward)
    assert torch.isfinite(targets).all()


@pytest.mark.parametrize("algorithm", ["dqn", "ddqn"])
def test_truncation_bootstraps_when_a_valid_successor_exists(algorithm: str):
    transition = make_transition(reward=2.0, truncated=True)

    targets = compute_td_targets(
        online=FeatureValueNetwork(0),
        target=FeatureValueNetwork(1),
        rewards=torch.tensor([transition.reward]),
        next_candidates=torch.as_tensor(transition.next_candidates[None, ...]),
        next_masks=torch.as_tensor(transition.next_mask[None, ...]),
        terminated=torch.tensor([transition.terminated]),
        gamma=0.5,
        algorithm=algorithm,
    )

    expected = 17.0 if algorithm == "dqn" else 12.0
    assert targets.item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "algorithm, expected",
    [
        ("dqn", [17.0, 7.0, 16.0, 5.0]),
        ("ddqn", [12.0, 7.0, 11.0, 5.0]),
    ],
)
def test_td_targets_for_mixed_transition_batch(algorithm: str, expected: list[float]):
    transitions = [
        make_transition(reward=2.0),
        make_transition(reward=7.0, terminated=True),
        make_transition(reward=1.0, truncated=True),
        make_transition(reward=5.0, next_mask=(False, False, False)),
    ]

    targets = compute_td_targets(
        online=FeatureValueNetwork(0),
        target=FeatureValueNetwork(1),
        rewards=torch.tensor([item.reward for item in transitions]),
        next_candidates=torch.as_tensor(
            np.stack([item.next_candidates for item in transitions])
        ),
        next_masks=torch.as_tensor(np.stack([item.next_mask for item in transitions])),
        terminated=torch.tensor([item.terminated for item in transitions]),
        gamma=0.5,
        algorithm=algorithm,
    )

    torch.testing.assert_close(targets, torch.tensor(expected), rtol=0.0, atol=0.0)
    assert torch.isfinite(targets).all()


def test_td_target_rejects_non_finite_bootstrap_values():
    transition = make_transition()

    with pytest.raises(FloatingPointError, match="bootstrap values"):
        compute_td_targets(
            online=FeatureValueNetwork(0),
            target=NonFiniteNetwork(),
            rewards=torch.tensor([transition.reward]),
            next_candidates=torch.as_tensor(transition.next_candidates[None, ...]),
            next_masks=torch.as_tensor(transition.next_mask[None, ...]),
            terminated=torch.tensor([transition.terminated]),
            gamma=0.99,
            algorithm="dqn",
        )


def test_optimize_batch_has_finite_loss_and_skips_disabled_imitation():
    online = TrainableValueNetwork()
    target = TrainableValueNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    parameters_before = [
        parameter.detach().clone() for parameter in online.parameters()
    ]
    terminal = replace(
        make_transition(reward=0.5, terminated=True),
        next_mask=np.zeros(3, dtype=bool),
    )

    result = optimize_batch(
        online=online,
        target=target,
        optimizer=optimizer,
        batch=[make_transition(), terminal],
        algorithm="ddqn",
        gamma=0.99,
        imitation_weight=0.0,
        teacher=FailingTeacher(),
        device=torch.device("cpu"),
    )

    assert np.isfinite(result.td_loss)
    assert np.isfinite(result.total_loss)
    assert np.isfinite(result.grad_norm)
    assert result.imitation_loss is None
    parameters_after = [parameter.detach() for parameter in online.parameters()]
    assert all(torch.isfinite(parameter).all() for parameter in parameters_after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after, strict=True)
    )


def test_pure_dqn_defaults_disable_shaping_and_imitation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_afterstate.py"])

    args = parse_args()

    assert args.line_bonus == 0.0
    assert args.imitation_steps == 0
    assert args.imitation_weight == 0.0


def test_invalid_replay_configuration_is_rejected_before_training(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train_afterstate.py", "--batch-size", "16", "--buffer-size", "8"],
    )

    with pytest.raises(ValueError, match="buffer-size"):
        validate_args(parse_args())
