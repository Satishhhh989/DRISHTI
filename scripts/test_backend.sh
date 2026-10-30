#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
  echo "Virtual environment (.venv) not found. Run scripts/setup.sh first."
  exit 1
fi

export PYTHONPATH="$DIR/services/api"
echo "Running backend test suite (pytest)..."
.venv/bin/pytest services/api/tests -v

