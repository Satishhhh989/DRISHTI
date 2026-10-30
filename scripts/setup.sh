#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=================================================="
echo "Setting up PHOTO → 3D ENGINE Environment"
echo "=================================================="

# 1. Python Environment Setup
echo ">>> Setting up Python virtual environment with uv..."
if [ ! -d ".venv" ]; then
  uv venv --python 3.11 .venv
fi

echo ">>> Installing backend Python dependencies..."
uv pip install --python .venv/bin/python -r services/api/requirements.txt

# 2. Node.js & pnpm Workspace Setup
echo ">>> Installing Node.js workspace dependencies..."
pnpm install

echo ">>> Building shared packages..."
pnpm --filter @photo-to-3d/scene-schema build
pnpm --filter @photo-to-3d/shared-types build

# 3. Environment configuration
if [ ! -f ".env" ]; then
  echo ">>> Initializing default .env from .env.example..."
  cp .env.example .env
fi

# 4. Storage directory initialization
mkdir -p data/storage/photos
mkdir -p data/storage/masks
mkdir -p data/storage/depth
mkdir -p data/storage/meshes
mkdir -p data/storage/textures
mkdir -p data/storage/skeletons
mkdir -p data/storage/exports

echo ""
echo "Setup complete! Run 'pnpm dev' or 'bash scripts/dev.sh' to start."

