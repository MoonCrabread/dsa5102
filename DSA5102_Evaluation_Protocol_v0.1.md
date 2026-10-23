# DSA5102 Tetris RL Project
## Evaluation Protocol v0.1 — Candidate for Group Approval

> Status: **Candidate / Not Frozen**
>
> Purpose: define a fair and reproducible evaluation protocol before formal training begins.
> After group approval, this document should be renamed or marked as **Frozen v1.0**.

---

## 1. Current Environment Status

### Completed
- Tetris-Gymnasium runs successfully in Python 3.12.
- NumPy compatibility issue fixed: `np.integer -> np.int64`.
- Core gameplay test passed.
- Grouped environment reset test passed.
- Grouped actions test passed.
- 13-dimensional feature-vector test passed.
- Shared environment smoke test passed.
- Human Demo visualization v3 completed.
- Seed reproducibility smoke test passed.

### Candidate shared training configuration
- Board: 20 x 10.
- `gravity=False`.
- Grouped Actions.
- Action space: 40 candidate actions.
- Action Mask enabled.
- Feature Vector Observation: 13 dimensions.
- Observation shape: `(40, 13)`.
- Shared environment factory used by all agents.

### Important distinction
`pretty_tetris_v3.py` is for **Human Demo / presentation visualization**.

Formal Random / Heuristic / DQN / Double DQN training and evaluation should use the shared grouped-action environment.

---

# 2. Proposed Model Line

Candidate research progression:

1. Random Legal Agent
2. Heuristic Agent
3. Basic DQN
4. Double DQN
5. Double DQN + Reward Shaping

Optional extension only if time/resources allow:
- PPO or another cross-algorithm comparison

The group should confirm this line before formal training.

---

# 3. Fair Comparison Rules

## Environment variables that should remain identical
For Random / Heuristic / DQN / Double DQN comparisons:

- Same Tetris environment version
- Same board size
- Same observation representation
- Same grouped action space
- Same action-mask logic
- Same illegal-action handling
- Same base reward when comparing DQN vs Double DQN

## DQN vs Double DQN
The following should be identical unless explicitly studied:

- Network architecture
- Training budget
- Replay-buffer size
- Batch size
- Learning rate
- Gamma
- Target-network update schedule
- Exploration schedule
- Evaluation frequency
- Random seeds

Only the DQN update rule should change in the main DQN-vs-DDQN comparison.

## Reward study
Reward Shaping should be treated as a separate experiment.

Do **not** compare:
- DQN + default reward
versus
- Double DQN + shaped reward

and then attribute the entire improvement to Double DQN.

Prefer:

- Basic DQN + default reward
- Double DQN + default reward
- Double DQN + shaped reward

This isolates algorithm improvement from reward-design improvement.

---

# 4. Evaluation Metrics — Candidate

## Primary gameplay metric
### Lines Cleared
Recommended as the main gameplay-performance metric because it directly reflects Tetris task performance.

Report:
- Mean
- Standard deviation
- Median
- Maximum

## Additional gameplay metrics
- Episode Reward
- Survival Steps
- Holes
- Maximum Height
- Game Score, if a common score definition is frozen

### Important
`game_score` and `episode_reward` must be treated as different variables.

The Human Demo score is a display metric and should not automatically be used as the main scientific evaluation metric.

---

# 5. Training Metrics for Learning Curves

DQN / Double DQN training logs should preferably contain:

- Training timestep or episode
- Episode reward
- Lines cleared
- Episode length / survival steps
- Training loss
- Epsilon

Optional:
- Mean Q-value
- Holes
- Maximum height

The group should use consistent column names across learning agents.

---

# 6. Formal Evaluation Protocol — Candidate

## Random seeds
Discussion-report recommendation:
- Minimum: 3 seeds
- Preferred if resources allow: 5 seeds

Group decision required:
- Final number of seeds: __________

## Final test episodes
Discussion-report recommendation:
- At least 100 evaluation episodes per final model.

Candidate:
- 100 episodes per model per seed.

Group decision required:
- Final evaluation episodes: __________

## Exploration during evaluation
Candidate:
- DQN / DDQN evaluation uses `epsilon = 0` or a fixed near-zero value agreed by the group.
- Evaluation policy should not keep learning.

Group decision required:
- Final evaluation epsilon: __________

## Evaluation environment
All final models must be tested through the same shared environment factory.

---

# 7. Proposed Result Table

| Model | Mean Lines | SD Lines | Median Lines | Max Lines | Mean Reward | Mean Steps | Mean Holes | Mean Max Height |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Legal | | | | | | | | |
| Heuristic | | | | | | | | |
| DQN | | | | | | | | |
| Double DQN | | | | | | | | |
| DDQN + Reward Shaping | | | | | | | | |

---

# 8. Shared Agent Interface — Draft

To make final evaluation easier, each agent should expose a simple common interface.

Suggested minimum:

```python
action = agent.select_action(observation, info)
```

The evaluator should not need to know the internal implementation of the agent.

For learning agents, also provide:

```python
agent.load(model_path)
```

Recommended deliverables from each model owner:

1. Agent code
2. Saved model / checkpoint if applicable
3. Hyperparameter configuration
4. Training history CSV
5. Short run instructions
6. Best/final model filename
7. Seed information

---

# 9. Suggested Shared File Naming

```text
project/
├── environment/
│   ├── environment_factory.py
│   └── smoke_test_environment.py
│
├── visualization/
│   └── pretty_tetris_v3.py
│
├── agents/
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── dqn_agent.py
│   └── ddqn_agent.py
│
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
│
├── results/
│   ├── training/
│   └── evaluation/
│
└── models/
```

---

# 10. Second Group Meeting — Decisions Required

The meeting should end with explicit decisions for the following items.

## A. Environment Freeze
Confirm:
- [ ] 20 x 10 board
- [ ] `gravity=False`
- [ ] Grouped Actions
- [ ] 40 actions
- [ ] Action Mask
- [ ] 13-dimensional Feature Vector
- [ ] Shared `make_project_env()` required for all agents
- [ ] Illegal-action policy: ______________________

Once approved:
**Environment v0.1 Candidate -> Environment v1.0 Frozen**

---

## B. Model Line
Confirm:
- [ ] Random Legal
- [ ] Heuristic
- [ ] Basic DQN
- [ ] Double DQN
- [ ] DDQN + Reward Shaping
- [ ] Optional PPO? Yes / No

---

## C. DQN / DDQN Fairness
Confirm:
- Training budget: ______________________
- Network architecture: __________________
- Replay buffer: ________________________
- Batch size: ___________________________
- Learning rate: ________________________
- Gamma: _______________________________
- Target update frequency: ______________
- Epsilon schedule: ______________________
- Evaluation frequency: _________________

The same values should be used for DQN and DDQN unless one variable is intentionally studied.

---

## D. Reward Study
Confirm:
- Base/default reward definition: ______________________
- Reward-shaping variables: ___________________________
- Candidate variables:
  - holes
  - maximum / aggregate height
  - bumpiness
  - line clear
- Coefficients: to be determined by controlled experiment.

---

## E. Formal Evaluation
Confirm:
- Number of seeds: 3 / 5 / other ______
- Final episodes per model: 100 / other ______
- Evaluation epsilon: __________________
- Primary metric: Lines Cleared? Yes / No
- Other required metrics: ____________________________

---

## F. Collaboration
Confirm:
- GitHub repository owner: __________________
- Repository visibility: Private / other
- Branch / pull-request policy: ______________
- Final Notebook integrator: _________________
- Drive / checkpoint storage owner: __________

---

# 11. Suggested Deliverables for the Next Development Cycle

### Environment / Evaluation owner
- Freeze shared environment after group approval.
- Maintain environment README.
- Prepare Evaluation Framework v0.1.
- Test evaluator first using Random Legal Agent.

### Baseline owner
- Implement Random Legal Agent.
- Implement Heuristic Agent.
- Produce initial baseline results.

### DQN owner
- Implement corrected Basic DQN.
- Ensure target-network logic is correct.
- Export training logs and checkpoint.

### Improvement owner
- Implement Double DQN.
- Design reward-shaping experiment.
- Keep DDQN-vs-DQN comparison controlled.

### Group
- Confirm experiment protocol before expensive training.
- Record model configurations and seeds.
- Keep environment version fixed during comparable experiments.

---

# 12. Current Milestone Summary

**Completed milestone: Environment Integration & Validation**

Next milestone after group approval:

**Environment Frozen + Baselines/DQN/DDQN Development + Evaluation Framework Preparation**
