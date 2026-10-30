#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=================================================="
echo "Starting PHOTO → 3D ENGINE (Phase 01: Foundation)"
echo "=================================================="

# Cleanup handler
cleanup() {
  echo ""
  echo "Shutting down services..."
  kill $(jobs -p) 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Check python virtualenv
if [ ! -d ".venv" ]; then
  echo "Virtual environment (.venv) not found. Initializing..."
  bash scripts/setup.sh
fi

# Ensure storage directory exists
mkdir -p data/storage

# Start Backend API
export PYTHONPATH="$DIR/services/api"
echo "Starting API service on http://127.0.0.1:8000..."
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait briefly for backend to initialize
sleep 1

# Start Frontend Dev Server
echo "Starting Web Frontend on http://localhost:3000..."
pnpm --filter @photo-to-3d/web dev &
FRONTEND_PID=$!

echo ""
echo "PHOTO → 3D ENGINE is running!"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://127.0.0.1:8000"
echo "  API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services."

wait

