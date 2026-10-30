#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=================================================="
echo "Running All PHOTO → 3D ENGINE Automated Tests"
echo "=================================================="

echo ""
echo ">>> [1/5] Testing @photo-to-3d/scene-schema (Vitest)..."
pnpm --filter @photo-to-3d/scene-schema test

echo ""
echo ">>> [2/5] Testing @photo-to-3d/shared-types (Vitest)..."
pnpm --filter @photo-to-3d/shared-types test

echo ""
echo ">>> [3/5] Testing @photo-to-3d/web (Vitest)..."
pnpm --filter @photo-to-3d/web test

echo ""
echo ">>> [4/5] Testing API Backend (Pytest)..."
bash scripts/test_backend.sh

echo ""
echo ">>> [5/5] Testing Browser E2E Suite (Playwright)..."
pnpm test:e2e

echo ""
echo "=================================================="
echo "All DRISHTI automated tests passed successfully!"
echo "=================================================="
