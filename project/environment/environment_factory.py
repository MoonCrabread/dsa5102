"""
DSA5102 Tetris RL - Shared Environment Factory

Purpose
-------
Every baseline / RL agent should create the project environment through this
module so that Random, Heuristic, DQN and Double DQN are compared under the
same environment configuration.

Frozen training configuration
-----------------------------
- Tetris board: default 20 x 10
- gravity=False
- Grouped actions: 10 columns x 4 rotations = 40 actions
- Action mask enabled through GroupedActionsObservations
- 13-dimensional FeatureVectorObservation
- Illegal grouped actions terminate the episode
- Optional RecordEpisodeStatistics wrapper

Important
---------
This file is for TRAINING / EVALUATION.

The pretty Human Demo uses the same underlying Tetris game, but it has a
different interaction style (keyboard controls + gravity=True). Do not use
the Human Demo interface for DQN/DDQN training.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Importing envs also registers "tetris_gymnasium/Tetris" with Gymnasium.
from tetris_gymnasium.envs import Tetris  # noqa: F401
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


ENV_ID = "tetris_gymnasium/Tetris"

# Project-wide frozen settings.
PROJECT_GRAVITY = False
PROJECT_TERMINATE_ON_ILLEGAL_ACTION = True

EXPECTED_ACTIONS = 40
EXPECTED_FEATURES = 13
EXPECTED_OBSERVATION_SHAPE = (
    EXPECTED_ACTIONS,
    EXPECTED_FEATURES,
)


def make_project_env(
    seed: int | None = None,
    *,
    render_mode: str | None = None,
    record_episode_statistics: bool = True,
):
    """
    Create the shared environment used by all project agents.

    Parameters
    ----------
    seed:
        Seed for the action space. The caller should still use
        env.reset(seed=seed) at the beginning of each run.
    render_mode:
        None for normal training/evaluation.
        "rgb_array" can be used later for video capture.
    record_episode_statistics:
        If True, Gymnasium adds episode return/length statistics to info.

    Returns
    -------
    gym.Env
        Wrapped grouped-action Tetris environment.

    Observation
    -----------
    obs.shape == (40, 13)

    Each of the 40 rows describes the 13-dimensional board feature vector
    resulting from one candidate placement.

    Action
    ------
    Discrete integer 0..39.

    Action mapping:
        action // 4 = target column (0..9)
        action % 4  = number of rotations (0..3)

    Legal actions
    -------------
    info["action_mask"] is a length-40 vector:
        1 = legal
        0 = illegal
    """

    # 1) Base Tetris environment.
    base_env = gym.make(
        ENV_ID,
        render_mode=render_mode,
        gravity=PROJECT_GRAVITY,
    )

    # 2) Feature extractor used by the grouped wrapper.
    #    The default configuration returns:
    #    10 column heights + max height + holes + bumpiness = 13 features.
    feature_wrapper = FeatureVectorObservation(
        base_env
    )

    # 3) Convert frame-by-frame actions into one placement decision:
    #    column + rotation.
    env = GroupedActionsObservations(
        base_env,
        observation_wrappers=[
            feature_wrapper
        ],
        terminate_on_illegal_action=PROJECT_TERMINATE_ON_ILLEGAL_ACTION,
    )

    # 4) Episode statistics are useful for training logs/evaluation.
    if record_episode_statistics:
        env = gym.wrappers.RecordEpisodeStatistics(
            env
        )

    if seed is not None:
        env.action_space.seed(seed)

    return env


def legal_actions_from_info(
    info: dict,
) -> np.ndarray:
    """
    Return the integer IDs of currently legal actions.

    All project agents should respect the action mask.
    """
    if "action_mask" not in info:
        raise KeyError(
            "info does not contain 'action_mask'. "
            "Make sure the shared grouped environment is being used."
        )

    mask = np.asarray(
        info["action_mask"]
    )

    if mask.shape != (
        EXPECTED_ACTIONS,
    ):
        raise ValueError(
            "Unexpected action-mask shape: "
            f"{mask.shape}; expected ({EXPECTED_ACTIONS},)"
        )

    return np.flatnonzero(
        mask == 1
    )


def decode_grouped_action(
    action: int,
) -> tuple[int, int]:
    """
    Human-readable interpretation of a grouped action.

    Returns
    -------
    (column, rotations)
    """
    if not (
        0
        <= int(action)
        < EXPECTED_ACTIONS
    ):
        raise ValueError(
            f"Action must be in 0..{EXPECTED_ACTIONS - 1}"
        )

    action = int(action)

    return (
        action // 4,
        action % 4,
    )
