r"""
Smoke test for the DSA5102 shared Tetris environment.

Run from the repository root:

    python project\environment\smoke_test_environment.py

Expected final message:

    PROJECT ENVIRONMENT SMOKE TEST: PASS
"""

import numpy as np

from environment_factory import (
    EXPECTED_ACTIONS,
    EXPECTED_FEATURES,
    EXPECTED_OBSERVATION_SHAPE,
    decode_grouped_action,
    legal_actions_from_info,
    make_project_env,
)


def main():
    seed = 42

    env = make_project_env(
        seed=seed
    )

    try:
        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------
        obs, info = env.reset(
            seed=seed
        )

        assert (
            obs.shape
            == EXPECTED_OBSERVATION_SHAPE
        ), (
            f"Observation shape is {obs.shape}, "
            f"expected {EXPECTED_OBSERVATION_SHAPE}"
        )

        assert (
            env.action_space.n
            == EXPECTED_ACTIONS
        ), (
            f"Action count is {env.action_space.n}, "
            f"expected {EXPECTED_ACTIONS}"
        )

        assert (
            "action_mask"
            in info
        ), "Missing action_mask after reset"

        mask = np.asarray(
            info["action_mask"]
        )

        assert mask.shape == (
            EXPECTED_ACTIONS,
        )

        assert set(
            np.unique(mask)
        ).issubset(
            {0, 1}
        ), "Action mask should contain only 0/1"

        legal = legal_actions_from_info(
            info
        )

        assert (
            len(legal) > 0
        ), "No legal actions found"

        # The grouped wrapper also exposes the current board feature vector.
        assert (
            "board"
            in info
        ), "Missing current board feature vector in info"

        board_features = np.asarray(
            info["board"]
        )

        assert board_features.shape == (
            EXPECTED_FEATURES,
        ), (
            f"Current board feature shape is {board_features.shape}, "
            f"expected ({EXPECTED_FEATURES},)"
        )

        print("[PASS] reset")
        print(
            "[PASS] observation shape:",
            obs.shape,
        )
        print(
            "[PASS] action space:",
            env.action_space,
        )
        print(
            "[PASS] legal actions:",
            len(legal),
            "/",
            EXPECTED_ACTIONS,
        )
        print(
            "[PASS] current feature vector:",
            board_features.shape,
        )

        # ----------------------------------------------------
        # ONE LEGAL STEP
        # ----------------------------------------------------
        action = int(
            legal[0]
        )

        column, rotations = (
            decode_grouped_action(
                action
            )
        )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(
            action
        )

        assert (
            next_obs.shape
            == EXPECTED_OBSERVATION_SHAPE
        )

        assert (
            "action_mask"
            in next_info
        )

        assert (
            np.asarray(
                next_info[
                    "action_mask"
                ]
            ).shape
            == (
                EXPECTED_ACTIONS,
            )
        )

        assert np.isscalar(
            reward
        )

        print(
            "[PASS] legal step:",
            f"action={action},",
            f"column={column},",
            f"rotations={rotations}",
        )
        print(
            "[PASS] reward:",
            float(reward),
        )
        print(
            "[PASS] terminated:",
            bool(terminated),
        )
        print(
            "[PASS] truncated:",
            bool(truncated),
        )

        # ----------------------------------------------------
        # REPRODUCIBILITY CHECK
        # ----------------------------------------------------
        env2 = make_project_env(
            seed=seed
        )

        try:
            obs2, info2 = env2.reset(
                seed=seed
            )

            assert np.array_equal(
                obs,
                obs2,
            ), (
                "Same reset seed did not produce "
                "the same grouped observation"
            )

            assert np.array_equal(
                info["action_mask"],
                info2["action_mask"],
            ), (
                "Same reset seed did not produce "
                "the same action mask"
            )

        finally:
            env2.close()

        print(
            "[PASS] seed reproducibility"
        )

        print()
        print(
            "PROJECT ENVIRONMENT SMOKE TEST: PASS"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
