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
DRISHTI relies on local neural network models stored in `data/models/`:
1. **YOLOv8 Nano Segmentation** (`yolov8n-seg.pt`, ~6.7MB): Fast person detection and binary instance segmentation.
2. **Depth Anything V2 Small** (`depth-anything-v2-small/`, ~99.2MB Safetensors): High-resolution monocular relative depth estimation.
3. **SPIN Human Mesh Reconstruction** (`human_reconstruction/spin_model_checkpoint.pth.tar`, ~103MB + `human_reconstruction/SMPL_N_model.npz`, ~88MB): Parametric 3D human mesh estimation and SMPL neutral body linear blend skinning.
4. **YOLOv8 Nano Pose** (`yolov8n-pose.pt`, ~6.5MB): 2D body pose estimation and 17-keypoint extraction.

Check model status or trigger local download via the API:

```bash
# Check status via API
curl http://127.0.0.1:8000/api/v1/pipeline/models

# Explicit download via API
curl -X POST http://127.0.0.1:8000/api/v1/pipeline/models/download
```

Weights are stored in `data/models/` and are preserved offline.

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
1. `packages/scene-schema`: Vitest schema validation, representation availability, contract, and representation registry tests (14 tests).
2. `packages/shared-types`: Vitest event contract and DTO tests (2 tests).
3. `apps/web`: Vitest component, store, event reducer, 3D mesh rendering, assembly inspector, skeleton overlay, motion controls, asset cache, and unified hybrid renderer tests (32 tests).
4. `services/api`: Pytest backend test suite (unit tests, mock pipeline, real model MPS smoke tests, geometry validation, progressive event stream tests, scene assembly engine, skeleton validation, pose estimation, rigging, procedural animation, and disk persistence tests) (88 tests).
5. `tests/e2e`: Playwright browser-level end-to-end tests including Phase 08 unified renderer representation verification (12 tests).

**Total Verified Tests: 148 passing.**

### Running Individual Test Suites

```bash
# Backend test suite (pytest)
bash scripts/test_backend.sh

# Or directly in virtualenv
source .venv/bin/activate
PYTHONPATH=services/api pytest services/api/tests -v

# Motion Subsystem backend tests (Phase 07)
PYTHONPATH=services/api pytest services/api/tests/test_skeleton_schema.py -v
PYTHONPATH=services/api pytest services/api/tests/test_pose_estimator.py -v
PYTHONPATH=services/api pytest services/api/tests/test_mesh_rigging.py -v
PYTHONPATH=services/api pytest services/api/tests/test_procedural_animation.py -v
PYTHONPATH=services/api pytest services/api/tests/test_pipeline_motion.py -v

# Frontend tests (vitest)
pnpm --filter @photo-to-3d/web test

# Unified Hybrid Renderer UI tests (Phase 08)
pnpm --filter @photo-to-3d/web test apps/web/tests/unified-renderer.test.tsx

# Schema package tests (Phase 08 Registry)
pnpm --filter @photo-to-3d/scene-schema test

# Shared types tests
pnpm --filter @photo-to-3d/shared-types test

# Playwright E2E tests
pnpm test:e2e
```

---

## 4. Hardware, Pipeline, Scene & Streaming Endpoints

DRISHTI exposes dedicated endpoints for inspecting hardware acceleration, pipeline status, scene persistence, and real-time streaming:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/pipeline/hardware` | Returns detected compute device (`mps`, `cuda`, `cpu`), device name, and allocation test status. |
| `GET` | `/api/v1/pipeline/models` | Lists local model files, sizes, and readiness. |
| `POST` | `/api/v1/pipeline/models/download` | Explicitly downloads default model weights with verification. |
| `POST` | `/api/v1/pipeline/analyze` | Initiates the 18-stage image intelligence, depth, 3D reconstruction, rigging, motion, and scene assembly pipeline. |
| `GET` | `/api/v1/jobs/{job_id}` | Polls status, current stage, and progress of an active job. |
| `GET` | `/api/v1/jobs/{job_id}/events` | Retrieves ordered, sequenced events since a given sequence number (`?since=...`). |
| `GET` | `/api/v1/jobs/{job_id}/stream` | Server-Sent Events (SSE) stream (`text/event-stream`) with automatic `Last-Event-ID` catchup. |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cooperatively cancels an active job without discarding already produced assets. |
| `GET` | `/api/v1/scenes` | Lists all saved scenes found locally in `data/storage/scenes/`. |
| `GET` | `/api/v1/scenes/{scene_id}` | Retrieves full persisted scene including entities, scene graph, and assembly report. |
| `POST` | `/api/v1/scenes` | Creates and persists a new scene definition. |
| `PUT` | `/api/v1/scenes/{scene_id}` | Updates and persists scene modifications to disk. |
| `GET` | `/api/v1/assets/masks/{filename}` | Serves individual 8-bit binary PNG segmentation masks. |
| `GET` | `/api/v1/assets/depth/{filename}` | Serves 8-bit grayscale depth visualization PNGs. |
| `GET` | `/api/v1/assets/meshes/{filename}` | Serves binary GLTF (.glb) 3D human meshes. |
| `GET` | `/api/v1/assets/photos/{filename}` | Serves sanitized original user photographs. |

---

## 5. Environment Variables

Configuration is driven by `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Application runtime environment |
| `API_HOST` | `127.0.0.1` | Loopback IP address for local privacy |
| `API_PORT` | `8000` | Local port for FastAPI service |
| `API_LOG_LEVEL` | `info` | Structured logging verbosity |
| `STORAGE_ROOT` | `./data/storage` | Local directory for photographs, masks, and depth maps |
| `MODELS_DIR` | `./data/models` | Local directory for neural network weights |
| `MAX_UPLOAD_SIZE_MB`| `50` | Maximum file upload limit in megabytes |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for web client |
| `NEXT_PUBLIC_API_URL`| `http://127.0.0.1:8000` | Backend API URL accessed by web client |
