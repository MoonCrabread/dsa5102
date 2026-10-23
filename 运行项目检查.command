#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROJECT_DIR"

"$PROJECT_DIR/.venv-macos/bin/python" -m pytest -q
"$PROJECT_DIR/.venv-macos/bin/python" \
  "$PROJECT_DIR/project/environment/smoke_test_environment.py"

printf '\n全部检查完成。按 Enter 关闭窗口。\n'
read answer
