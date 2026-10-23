# Shared Tetris Environment

This folder contains the project-wide environment configuration.

## Training / evaluation configuration

All baseline and RL agents should use `make_project_env()` from
`environment_factory.py`.

Frozen settings:

- Board: 20 x 10 (Tetris-Gymnasium default)
- Gravity: `False`
- Grouped Actions: 40 candidate placements
- Action Mask: required
- State representation: 13-dimensional feature vector
- Observation shape: `(40, 13)`
- Illegal grouped action policy: terminate episode
- Same environment configuration for Random, Heuristic, DQN and Double DQN

## Important distinction

`project/visualization/pretty_tetris_v3.py` is the **Human Demo / presentation
interface**. It uses keyboard controls and is not the model-training interface.

The RL models should use `environment_factory.py`.

## Smoke test

From the repository root:

```bat
python project\environment\smoke_test_environment.py
```

Expected:

```text
PROJECT ENVIRONMENT SMOKE TEST: PASS
```
