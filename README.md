# DRISHTI: PHOTO → 3D ENGINE

A local-first AI-powered spatial reconstruction engine that transforms ordinary multi-person photographs into interactive, animated 3D scenes.

Current Status: **Phase 04: Live Visualization & Progressive Processing Engine (Completed)**

---

## Capabilities (Phase 04)

- **Local-First AI Architecture**: Runs entirely offline on the user's machine without any external cloud vision APIs or telemetry.
- **Hardware Acceleration**: Automatic capability discovery and execution on Apple Silicon Metal Performance Shaders (`mps`), NVIDIA GPU (`cuda`), and optimized CPU fallback.
- **Progressive Visual Reveal Pipeline**:
  - **Instant Photo Display**: Original image renders on the Three.js viewport immediately upon upload and server validation.
  - **Live Detection Boxes**: 3D bounding boxes render the moment person detection completes with stable entity IDs (`person-001`, `person-002`, ...).
  - **Individual Mask Streaming**: Segmentation masks stream and attach progressively to each person entity as they are written to disk.
  - **Dynamic Depth Map Activation**: 8-bit normalized grayscale depth visualization unlocks the moment relative proximity computation completes.
  - **Dynamic Spatial Photo Activation**: Interactive 3D vertex-displaced mesh surface unlocks seamlessly once depth and photographic textures are available.
  - **Honest Future Placeholders**: Wireframe and 3D solid representations remain disabled with clear explanatory tooltips until implemented in Phase 06.
- **Server-Sent Events (SSE) Push Architecture**: Real-time streaming (`GET /api/v1/jobs/{job_id}/stream`) with bounded 500-event FIFO buffer and automatic HTTP polling fallback for maximum reliability.
- **Idempotent Frontend Event Reducer (`reduceProcessingEvent`)**: Pure state machine guaranteeing deterministic scene assembly from events, preserving camera positions, entity selections, and viewport modes across reconnections.
- **Dynamic Representation Availability (`computeRepresentationAvailability`)**: Evaluates readiness across all 7 representations (`photo`, `detection`, `segmentation`, `depth`, `spatial_photo`, `wireframe`, `solid_3d`), displaying live status badges (`LIVE`, `ERR`, `P06`).
- **Live Pipeline Inspector**: Interactive panel reflecting real-time stage states, entity segmentation progress counters ("3 of 5 people segmented"), and partial failure alerts.
- **Cooperative Job Cancellation**: Clean cancellation via `POST /api/v1/jobs/{job_id}/cancel` and UI "Cancel Analysis" button, halting backend worker loops at cooperative checkpoints while retaining completed intermediate assets.
- **Non-Fatal Partial Failure Isolation**: Gracefully catches and isolates non-fatal stage errors (e.g. depth failure) while preserving and displaying successful upstream detection and segmentation results.
- **Monocular Depth Estimation**: Local Depth Anything V2 Small neural network producing dense relative depth fields $[0.0, 1.0]$ with robust 2nd/98th percentile normalization.
- **Interactive 3D Spatial Photography**: High-resolution $128 \times 128$ vertex displacement mesh with bounded parallax orbit controls (azimuth $\pm 18^\circ$, polar $75^\circ \text{--} 105^\circ$, damping $0.06$).
- **Local Asset Storage**: Lossless float32 machine-readable `.npy` depth maps, grayscale visualization PNGs, 8-bit binary instance masks, and sanitized photos stored safely in `data/storage/`.
- **Automated Verification**: **89 automated tests passing** across 5 test suites (backend pytest, frontend vitest, schema vitest, and Playwright E2E).

---

## Project Structure

```
DRISHTI/
│
├── apps/
│   └── web/                   # Next.js 15, React 19, Three.js, React Three Fiber, Zustand
│       ├── app/               # App Router pages and layout
│       ├── components/        # Viewport, controls, scene elements, contextual panel
│       │   ├── controls/      # RepresentationSelector, ProcessingIndicator, UploadButton
│       │   ├── layout/        # TopBar, ContextualPanel (with Live Pipeline Inspector)
│       │   ├── scene/         # PhotoPlane, DetectionBox (3D), SegmentationMaskLayer (3D)
│       │   └── viewport/      # ThreeViewport, GridFloor, ErrorBoundary
│       ├── hooks/             # Custom hooks (health monitoring, viewport resize)
│       ├── lib/               # API client (SSE + polling fallback) and image utilities
│       ├── stores/            # Zustand scene store and idempotent event-reducer
│       └── tests/             # Vitest frontend unit, reducer, inspector, and visualization tests
│
├── services/
│   └── api/                   # Python 3.11, FastAPI, PyTorch, Ultralytics, Pydantic
│       ├── app/
│       │   ├── api/           # Health, version, assets, scenes, events (SSE & cancel), pipeline routes
│       │   ├── core/          # Validated config, structured logging, custom errors
│       │   ├── schemas/       # Scene, event, asset, detection, and segmentation schemas
│       │   ├── scene/         # Scene manager and state validation
│       │   ├── assets/        # Storage manager and strict image validation
│       │   ├── pipeline/      # DeviceManager, ModelRegistry, YoloDetector, YoloSegmentor, DepthEstimator, Coordinator
│       │   └── events/        # Sequenced event emitter, bounded FIFO buffer, pub/sub SSE stream
│       └── tests/             # Pytest backend test suite (unit, mock, real MPS smoke, progressive events)
│
├── packages/
│   ├── scene-schema/          # TypeScript scene models, representation availability, coordinate conventions
│   └── shared-types/          # TypeScript processing event contracts and API DTOs
│
├── data/
│   ├── models/                # Local neural network checkpoints (yolov8n-seg.pt, depth-anything-v2-small/)
│   └── storage/               # Local user photos, segmentation masks, and depth maps (.npy / .png)
│
├── docs/                      # ARCHITECTURE, SCENE_SCHEMA, EVENT_PROTOCOL, DEVELOPMENT, ROADMAP
├── scripts/                   # dev.sh, test.sh, test_backend.sh, setup.sh, run_api.sh
└── tests/                     # Playwright browser-level E2E tests
```

---

## Quick Start

### 1. Initial Setup
```bash
bash scripts/setup.sh
```

### 2. Start Application
```bash
bash scripts/dev.sh
```
- Web Application: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Automated Tests
```bash
# Complete test suite across all packages (89 tests)
bash scripts/test.sh

# Individual suites:
bash scripts/test_backend.sh              # Backend pytest suite (48 tests)
pnpm --filter @photo-to-3d/web test       # Frontend vitest suite (22 tests)
pnpm --filter @photo-to-3d/scene-schema test # Scene schema tests (9 tests)
pnpm --filter @photo-to-3d/shared-types test # Shared type tests (2 tests)
pnpm test:e2e                             # Playwright E2E browser tests (8 tests)
```

---

## Engineering Rules

1. **No Cloud Vision APIs**: All neural inference runs locally on the host hardware.
2. **Local Pretrained Models Only**: Lightweight, high-performance YOLOv8 nano segmentation (~6.7MB) and Depth Anything V2 Small (~99.2MB), explicitly managed in `data/models/`.
3. **No Fake AI or Synthetic Meshes**: Detection boxes, masks, confidence scores, depth maps, and processing stages reflect actual model outputs. 3D polygon meshes and skeletons remain honest placeholders until Phases 06 and 08.
4. **Progressive Visibility**: Reveal intermediate assets as they are computed; never block the UI behind a monolithic opaque spinner.
5. **Strict Coordinate System**: Right-handed Cartesian coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward, metric units).
6. **Security & Privacy**: Enforced loopback binding (`127.0.0.1`), safe asset path sanitization preventing directory traversal, and strict decompression bomb protection.
