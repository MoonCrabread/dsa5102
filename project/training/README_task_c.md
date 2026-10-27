# Task C: checkpoints, evaluation, curves, and gameplay

Install the Poetry `dev` and `examples` groups before running training (Torch,
TensorBoard, Matplotlib, and ImageIO are in the examples group). Run commands
from the repository root. DQN and DDQN must use the same options except
`--algorithm` and `--run-name`.

```bash
poetry run python project/training/train_afterstate.py \
  --algorithm dqn --seed 42 --steps 100000 \
  --checkpoint-every 10000 --eval-every 10000 \
  --eval-episodes 10 --eval-seed-start 10000 --eval-cap 2000 \
  --line-bonus 0 --imitation-steps 0 --imitation-weight 0 \
  --run-name dqn_seed_42
```

Use `--algorithm ddqn --run-name ddqn_seed_42` for the corresponding DDQN run.
The run name must be new; the trainer refuses to overwrite an existing run.
This example defines the workflow, not the formal experiment budget or frozen
evaluation protocol.

Each run writes `config.json`, `episodes.csv` (exploratory training episodes),
`training_steps.csv`, `eval_episodes.csv`, `eval_history.csv`, TensorBoard
events, four PNG figures, and checkpoints at step 0, the configured intervals,
and the final step. `model.pt` remains readable by Task B's evaluator. Evaluation
uses a separate environment, fixed seeds starting at `--eval-seed-start`, greedy
network actions, and the same episode cap at every snapshot. The plots use
evaluation means for lines and environment reward; TD loss and epsilon come
from training steps. A single training seed is one run; compare seeds only
after combining their independent results.

To record early, middle, and final gameplay with the same test sequence:

```bash
poetry run python project/evaluation/record_gameplay.py \
  --checkpoints \
  project/results/dqn_seed_42/checkpoints/step_000000.pt \
  project/results/dqn_seed_42/checkpoints/step_050000.pt \
  project/results/dqn_seed_42/checkpoints/final.pt \
  --eval-seed 10000 --episode-cap 2000 \
  --output-dir project/results/dqn_seed_42/videos
```

Choose a middle checkpoint that exists for the run. The command writes labeled
GIF files and JSON sidecars containing the seed, step, lines, reward, length,
and cap/termination status. It uses the grouped RL environment with
`render_mode="rgb_array"`; the keyboard human demo is not involved.
