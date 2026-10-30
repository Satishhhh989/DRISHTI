# DRISHTI Development & Testing Guide

## 1. Prerequisites

- **Node.js**: `v20+` (tested on Node `v26.8.2`)
- **Package Manager**: `pnpm` (tested on `11.3.0`)
- **Python**: `3.11+` (tested on `3.11.17`)
- **Hardware Acceleration**:
  - Apple Silicon Mac with Metal Performance Shaders (`mps`)
  - Linux/Windows PC with NVIDIA GPU (`cuda`)
  - Multi-core CPU fallback (`cpu`)

---

## 2. Quick Start

### Initial Setup
Run the setup script to create the Python virtual environment, install backend and frontend dependencies (including PyTorch and Ultralytics), compile shared packages, and initialize local directories:

```bash
bash scripts/setup.sh
```

### Model Checkpoints
Phase 02 uses **YOLOv8 Nano Segmentation** (`yolov8n-seg.pt`, ~6.7MB).
Check model status or download the checkpoint locally:

```bash
# Check status via API
curl http://127.0.0.1:8000/api/v1/pipeline/models

# Explicit download via API
curl -X POST http://127.0.0.1:8000/api/v1/pipeline/models/download
```

Weights are stored in `data/models/yolov8n-seg.pt` and are preserved offline.

### Launch Development Environment
To launch both the backend API and frontend workspace concurrently:

```bash
bash scripts/dev.sh
```

- **Frontend Workspace**: `http://localhost:3000`
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`

---

## 3. Automated Testing

### Complete Test Suite
To run the complete test suite across all packages, services, and browsers:

```bash
bash scripts/test.sh
```

This executes:
1. `packages/scene-schema`: Vitest schema validation and contract tests.
2. `packages/shared-types`: Vitest event contract and DTO tests.
3. `apps/web`: Vitest component, store, and visualization tests.
4. `services/api`: Pytest backend test suite (unit tests, mock pipeline, and real model MPS smoke test).
5. `tests/e2e`: Playwright browser-level end-to-end tests.

### Running Individual Test Suites

```bash
# Backend test suite (pytest)
bash scripts/test_backend.sh

# Or directly in virtualenv
source .venv/bin/activate
pytest services/api/tests -v

# Frontend tests (vitest)
pnpm --filter @photo-to-3d/web test

# Schema package tests
pnpm --filter @photo-to-3d/scene-schema test

# Shared types tests
pnpm --filter @photo-to-3d/shared-types test

# Playwright E2E tests
pnpm test:e2e
```

---

## 4. Hardware & Pipeline Endpoints

DRISHTI exposes dedicated endpoints for inspecting hardware acceleration and pipeline status:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/pipeline/hardware` | Returns detected compute device (`mps`, `cuda`, `cpu`), device name, and allocation test status. |
| `GET` | `/api/v1/pipeline/models` | Lists local model files, sizes, and readiness. |
| `POST` | `/api/v1/pipeline/models/download` | Explicitly downloads default model weights with verification. |
| `POST` | `/api/v1/pipeline/analyze` | Initiates the 9-stage image intelligence pipeline for an uploaded photograph. |
| `GET` | `/api/v1/jobs/{job_id}` | Polls status, current stage, and progress of an active job. |
| `GET` | `/api/v1/jobs/{job_id}/events` | Retrieves ordered, sequenced events since a given sequence number. |
| `GET` | `/api/v1/assets/masks/{filename}` | Serves individual 8-bit binary PNG segmentation masks. |

---

## 5. Environment Variables

Configuration is driven by `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Application runtime environment |
| `API_HOST` | `127.0.0.1` | Loopback IP address for local privacy |
| `API_PORT` | `8000` | Local port for FastAPI service |
| `API_LOG_LEVEL` | `info` | Structured logging verbosity |
| `STORAGE_ROOT` | `./data/storage` | Local directory for photographs and masks |
| `MODELS_DIR` | `./data/models` | Local directory for neural network weights |
| `MAX_UPLOAD_SIZE_MB`| `50` | Maximum file upload limit in megabytes |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for web client |
| `NEXT_PUBLIC_API_URL`| `http://127.0.0.1:8000` | Backend API URL accessed by web client |
