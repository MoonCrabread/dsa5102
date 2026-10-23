#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

MODEL="$PROJECT_DIR/project/results/ddqn_regularized_10k/model.pt"

if [ -f "$MODEL" ]; then
  exec "$PROJECT_DIR/.venv-macos/bin/python" -m project.evaluation.watch_agent \
    --agent model --model "$MODEL" --episodes 3 --delay 0.12
fi

exec "$PROJECT_DIR/.venv-macos/bin/python" -m project.evaluation.watch_agent \
  --agent heuristic --episodes 3 --delay 0.12
