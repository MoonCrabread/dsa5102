#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/.venv-macos/bin/tensorboard" \
  --logdir_spec="ExampleRuns:$PROJECT_DIR/runs,Afterstate:$PROJECT_DIR/project/results" \
  --host 127.0.0.1 \
  --port 6006
