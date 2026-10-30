#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Please run bash scripts/setup.sh first."
  exit 1
fi

export PYTHONPATH="$DIR/services/api"
echo "Starting PHOTO → 3D ENGINE API on 127.0.0.1:8000..."
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

