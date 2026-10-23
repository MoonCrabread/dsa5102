#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/.venv-macos/bin/python" -m project.training.train_afterstate \
  --algorithm ddqn \
  --imitation-steps 5000 \
  --steps 100000 \
  --learning-starts 1000 \
  --epsilon-decay-steps 75000
