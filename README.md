# DRISHTI: PHOTO → 3D ENGINE

A local-first AI-powered spatial reconstruction engine that transforms ordinary multi-person photographs into interactive, animated 3D scenes.

Current Status: **Phase 05: AI Human 3D Reconstruction Engine (Completed)**

---

## Capabilities (Phase 05)

- **Local-First AI Architecture**: Runs entirely offline on the user's machine without any external cloud vision APIs or telemetry.
- **Hardware Acceleration**: Automatic capability discovery and execution on Apple Silicon Metal Performance Shaders (`mps`), NVIDIA GPU (`cuda`), and optimized CPU fallback.
- **Genuine 3D Human Mesh Reconstruction (Phase 05)**:
  - **Local Model Architecture**: Integrates SPIN (SMPL Optimization in the Loop of Neural Networks, Kolotouros et al., ICCV 2019) with a ResNet-50 feature extractor and a 3-iteration iterative 6D rotation & shape parameter regressor.
  - **Pure PyTorch SMPL Linear Blend Skinning (LBS)**: Eliminates legacy C++ and `chumpy` dependencies, evaluating canonical template deformation, PCA shape blend shapes ($\beta$), and joint kinematic rotations in pure PyTorch on Apple Silicon `mps` or CUDA.
  - **Standardized Anatomical Topology**: Reconstructs real volumetric humanoid meshes consisting of $6,890$ vertices and $13,776$ triangular faces.
  - **Geometry Validation & Sanitization**: Comprehensive validation rejecting NaNs, infinities, and out-of-range indices; removes degenerate zero-area faces; computes smooth vertex normals; and verifies real-world bounding dimensions ($[0.2\text{m}, 2.6\text{m}]$).
  - **Standard Binary GLTF (.glb) Export**: Writes binary GLTF models to local disk (`data/storage/meshes/{scene_id}_{entity_id}.glb`) with neutral studio clay PBR material (`#d0d0d8`, roughness 0.65).
  - **Dual Viewport Visualizations**:
    - **Wireframe Mode**: Displays genuine reconstructed polygon edge topology in technical cyan/white lines.
    - **Solid 3D Mode**: Renders smooth studio clay meshes under ambient and directional lighting with raycast selection highlighting.
  - **Multi-Person & Fault Tolerance**: Robust per-person bounding crop extraction with 15% context padding and aspect-ratio letterboxing. Partial failure isolation guarantees that failure to reconstruct one subject never aborts other subjects.
  - **Entity Inspector Integration**: Displays 3D Mesh Status (`Ready`), Vertex Count ($6,890$), Face Count ($13,776$), Dimensions ($W \times H \times D$ in meters), Model Provenance (`SPIN + SMPL LBS (Pretrained)`), and Texture Status (`Geometry Only - Neutral Studio Clay`).
- **Progressive Visual Reveal Pipeline**:
  - **Instant Photo Display**: Original image renders on the Three.js viewport immediately upon upload and server validation.
  - **Live Detection Boxes**: 3D bounding boxes render the moment person detection completes with stable entity IDs (`person-001`, `person-002`, ...).
  - **Individual Mask Streaming**: Segmentation masks stream and attach progressively to each person entity as they are written to disk.
  - **Dynamic Depth Map Activation**: 8-bit normalized grayscale depth visualization unlocks the moment relative proximity computation completes.
  - **Dynamic Spatial Photo Activation**: Interactive 3D vertex-displaced mesh surface unlocks seamlessly once depth and photographic textures are available.
  - **Reconstructed Human Meshes**: Wireframe and Solid 3D representations unlock progressively as human mesh GLB assets become ready.
- **Server-Sent Events (SSE) Push Architecture**: Real-time streaming (`GET /api/v1/jobs/{job_id}/stream`) with bounded 500-event FIFO buffer and automatic HTTP polling fallback for maximum reliability.
- **Idempotent Frontend Event Reducer (`reduceProcessingEvent`)**: Pure state machine guaranteeing deterministic scene assembly from events, preserving camera positions, entity selections, and viewport modes across reconnections.
- **Dynamic Representation Availability (`computeRepresentationAvailability`)**: Evaluates readiness across all 7 representations (`photo`, `detection`, `segmentation`, `depth`, `spatial_photo`, `wireframe`, `solid_3d`), displaying live status badges (`LIVE`, `ERR`).
- **Live Pipeline Inspector**: Interactive panel reflecting real-time stage states, entity segmentation/reconstruction progress counters ("2 of 2 meshes reconstructed"), and partial failure alerts.
- **Cooperative Job Cancellation**: Clean cancellation via `POST /api/v1/jobs/{job_id}/cancel` and UI "Cancel Analysis" button, halting backend worker loops at cooperative checkpoints while retaining completed intermediate assets.
- **Non-Fatal Partial Failure Isolation**: Gracefully catches and isolates non-fatal stage errors (e.g. depth failure, single-person crop failure) while preserving and displaying successful upstream and sibling results.
- **Monocular Depth Estimation**: Local Depth Anything V2 Small neural network producing dense relative depth fields $[0.0, 1.0]$ with robust 2nd/98th percentile normalization.
- **Interactive 3D Spatial Photography**: High-resolution $128 \times 128$ vertex displacement mesh with bounded parallax orbit controls (azimuth $\pm 18^\circ$, polar $75^\circ \text{--} 105^\circ$, damping $0.06$).
- **Local Asset Storage**: Lossless float32 machine-readable `.npy` depth maps, grayscale visualization PNGs, 8-bit binary instance masks, binary GLTF meshes (`.glb`), and sanitized photos stored safely in `data/storage/`.
- **Automated Verification**: **103 automated tests passing** across 5 test suites (backend pytest, frontend vitest, schema vitest, and Playwright E2E).

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
│       │   ├── layout/        # TopBar, ContextualPanel (with Live Pipeline & Entity Inspector)
│       │   ├── scene/         # PhotoPlane, DetectionBox (3D), SegmentationMaskLayer, HumanMeshLayer (3D GLTF)
│       │   └── viewport/      # ThreeViewport, GridFloor, ErrorBoundary
│       ├── hooks/             # Custom hooks (health monitoring, viewport resize)
│       ├── lib/               # API client (SSE + polling fallback) and image utilities
│       ├── stores/            # Zustand scene store and idempotent event-reducer
│       └── tests/             # Vitest frontend unit, reducer, inspector, 3D mesh, and visualization tests
│
├── services/
│   └── api/                   # Python 3.11, FastAPI, PyTorch, Ultralytics, Trimesh, Pydantic
│       ├── app/
│       │   ├── api/           # Health, version, assets (photos, masks, depth, meshes), scenes, events, pipeline routes
│       │   ├── core/          # Validated config, structured logging, custom errors
│       │   ├── schemas/       # Scene, event, asset, detection, segmentation, and reconstruction schemas
│       │   ├── scene/         # Scene manager and state validation
│       │   ├── assets/        # Storage manager (register_photo, register_mask, register_depth, register_mesh)
│       │   ├── pipeline/      # DeviceManager, ModelRegistry, YoloDetector, YoloSegmentor, DepthEstimator, Reconstructor, Coordinator
│       │   └── events/        # Sequenced event emitter, bounded FIFO buffer, pub/sub SSE stream
│       └── tests/             # Pytest backend test suite (unit, geometry, mock pipeline, real MPS smoke, progressive events)
│
├── packages/
│   ├── scene-schema/          # TypeScript scene models, representation availability, coordinate conventions
│   └── shared-types/          # TypeScript processing event contracts and API DTOs
│
├── data/
│   ├── models/                # Local neural network checkpoints
│   │   ├── yolov8n-seg.pt     # YOLOv8 nano instance segmentation
│   │   ├── depth-anything-v2-small/ # Depth Anything V2 Small Safetensors
│   │   └── human_reconstruction/    # SPIN checkpoint (.pth.tar) + SMPL Neutral model (.npz)
│   └── storage/               # Local user photos, segmentation masks, depth maps, and meshes (.glb)
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
# Complete test suite across all packages (103 tests)
bash scripts/test.sh

# Individual suites:
bash scripts/test_backend.sh              # Backend pytest suite (59 tests)
pnpm --filter @photo-to-3d/web test       # Frontend vitest suite (23 tests)
pnpm --filter @photo-to-3d/scene-schema test # Scene schema tests (10 tests)
pnpm --filter @photo-to-3d/shared-types test # Shared type tests (2 tests)
pnpm test:e2e                             # Playwright E2E browser tests (9 tests)
```

---

## Engineering Rules

1. **No Cloud Vision APIs**: All neural inference runs locally on the host hardware.
2. **Local Pretrained Models Only**: Lightweight, high-performance YOLOv8 nano segmentation (~6.7MB), Depth Anything V2 Small (~99.2MB), and SPIN + SMPL Neutral (~191MB), explicitly managed in `data/models/`.
3. **No Fake AI or Synthetic Meshes**: Detection boxes, masks, confidence scores, depth maps, and 3D polygon meshes reflect actual model inference outputs. Skeletal rigging and character animation remain honest future milestones (Phases 08 and 09).
4. **Progressive Visibility**: Reveal intermediate assets as they are computed; never block the UI behind a monolithic opaque spinner.
5. **Strict Coordinate System**: Right-handed Cartesian coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward, metric units).
6. **Security & Privacy**: Enforced loopback binding (`127.0.0.1`), safe asset path sanitization preventing directory traversal, and strict decompression bomb protection.
