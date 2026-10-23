#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv-macos/bin/python" \
  "$PROJECT_DIR/project/visualization/pretty_tetris_v3.py"
