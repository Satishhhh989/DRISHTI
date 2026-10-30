# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 04 implements the **Live Visualization & Progressive Processing Engine**, connecting real-time backend processing events via Server-Sent Events (SSE) and resilient fallback polling to a progressive frontend interface. DRISHTI reveals intermediate results incrementally as they become available (photo upload -> detection overlay -> per-person segmentation masks -> depth map -> spatial photo), while providing bounded event buffering, idempotent state reduction, live stage inspection, and honest job cancellation without discarding validated work.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js)                              |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation / Vis Modes     |  | Pipeline &     |  |  |
|  |  |  (Status & Reset)   |  | (Availability-Driven Selector) |  | Inspector Pane |  |  |
|  |  |                     |  | (Orig / Detect / Seg / Depth / |  | (Stages, Cancel|  |  |
|  |  |                     |  |  Spatial Photo / Wire / 3D)    |  |  People, Depth)|  |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |                        Three.js Viewport (React Three Fiber)                      |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  |  | Perspective Camera |  | Upright PhotoPlane /             |  | Parallax Cam  |  |  |
|  |  | & Lighting         |  | SpatialPhotoSurface (128x128 mesh|  | (Bounded orbit|  |  |
|  |  |                    |  |   + Vertex Displacement Map      |  |  az/pol limits|  |  |
|  |  |                    |  |   + 3D Detection Boxes & Masks)  |  |  damping 0.06)|  |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |  Zustand Stores: useSceneStore (Scene, Reducer, Stages) / useUIStore              |  |
|  |  Event Reducer: Idempotent sequence ordering, partial failure isolation           |  |
|  +-----------------------------------------------------------------------------------+  |
+--------------------------------------------|--------------------------------------------+
                                             | HTTP / SSE Stream (127.0.0.1:8000)
                                             v
+-----------------------------------------------------------------------------------------+
|                             Local Backend Service (FastAPI)                             |
|                                                                                         |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Health & Version APIs |  | Scene Lifecycle API  |  | Asset Manager API            |  |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Event Emitter (Bounded 500-event FIFO)          |  | Job State Tracker            |  |
|  | Real-Time SSE Stream (/jobs/{id}/stream)        |  | Cancellation Handler         |  |
|  +-------------------------------------------------+  +------------------------------+  |
|  | Progressive Multi-Stage Intelligence Pipeline (Phase 01 - 04)                     |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU fallback                  |  |
|  |  - ModelRegistry: Local cache in data/models/                                     |  |
|  |      * yolov8n-seg.pt (6.7MB)                                                     |  |
|  |      * depth-anything-v2-small/ (99.2MB safetensors)                             |  |
|  |  - Progressive Entity Commit: Entities committed immediately after detection     |  |
|  |  - Per-Person Mask Commit: Individual masks committed and broadcast upon save     |  |
|  |  - Progressive Depth Commit: Depth map and spatial photo committed upon save      |  |
|  |  - Cancellation Checkpoints: Clean early termination preserving prior assets     |  |
|  |  - AnalysisPipelineCoordinator: 12 sequential stages with live event stream       |  |
|  +-----------------------------------------------------------------------------------+  |
|  | Local Filesystem Storage (photos/, masks/, depth/, meshes/, skeletons/)           |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```
|  |  - Depth Normalizer: Robust percentile clamping (p2, p98) to canonical proximity |  |
|  |  - Entity Depth Associate: Per-person median, min, max proximity & depth stats   |  |
|  |  - AnalysisPipelineCoordinator: 12 sequential stages with monotonic event stream  |  |
|  +-----------------------------------------------------------------------------------+  |
|  | Local Filesystem Storage (photos/, masks/, depth/, meshes/, skeletons/)           |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

---

## 1. Monorepo Organization

The project is structured as a pnpm workspace:

- `apps/web`: Next.js 15 App Router frontend using React 19, Three.js, React Three Fiber, and Zustand.
- `services/api`: FastAPI backend service exposing REST endpoints on the loopback address (`127.0.0.1:8000`).
- `packages/scene-schema`: Shared TypeScript scene schema models, validation routines, detection/segmentation contracts, and coordinate system definitions.
- `packages/shared-types`: Shared TypeScript definitions for processing events, asset contracts, and API DTOs.
- `docs/`: Technical specifications, schema references, development instructions, and the 10-phase roadmap.
- `scripts/`: Development and automated verification scripts (`dev.sh`, `test_backend.sh`, `setup.sh`).

---

## 2. Frontend Architecture (Phase 04)

### 2.1 UI Layer
The UI provides an interactive workspace focused on the 3D viewport:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 04`), backend connectivity status indicator, grid floor toggle, camera reset button, and file upload trigger.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid, and camera controls.
- **Representation & Visualization Selector**: Computes availability dynamically using `computeRepresentationAvailability` based on real asset existence and active job stage. Each mode displays its live readiness (`Original`, `Detection`, `Segmentation`, `Depth Map`, `Spatial Photo`, `Wireframe [P06]`, `3D Solid [P06]`).
- **Progressive Processing Inspector (`ContextualPanel.tsx`)**: Compact stage-by-stage pipeline inspector displaying real-time state for each operation:
  - *Image Validation*: Validated on upload.
  - *Person Detection*: Reports count of people found immediately after detection.
  - *Segmentation*: Displays honest per-person progress (`X of Y masks ready`).
  - *Scene Depth*: Shows status (`Estimating...`, `Ready`, `Failed`).
  - *Spatial Photo*: Shows readiness.
  - *Cancellation*: "Cancel Analysis" button when a job is active, stopping subsequent computation safely without losing completed assets.
  - *Partial Failures*: Unobtrusive warning banner highlighting non-fatal errors while keeping successful layers usable.
- **ProcessingIndicator**: Status bar showing live engine state, current stage message, and pulsing progress during neural network analysis.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto an upright 3D plane at world coordinates `(0, height/2, 0)` with aspect ratio preserved.
- **Progressive Overlays**:
  - Person bounding boxes appear in 3D space immediately after detection completes, making people selectable right away.
  - Segmentation masks appear progressively as each individual mask asset is persisted.
  - Depth map visualization texture becomes active as soon as normalized depth is stored.
  - Spatial photo vertex-displaced mesh becomes active as soon as spatial photo preparation completes.
- **Camera & Selection Preservation**: Incoming processing events are processed idempotently without resetting camera position or deselecting the active entity.

### 2.3 State Management & Event Reducer
- `event-reducer.ts`: Pure, deterministic state reducer `reduceProcessingEvent(state, event)` enforcing strict sequence numbers, duplicate event rejection, wrong-scene filtering, and isolated partial failure handling.
- `api-client.ts`: Live EventSource subscription to `GET /api/v1/jobs/{id}/stream` with automatic fallback to polling, and cancellation via `POST /api/v1/jobs/{id}/cancel`.
- `useSceneStore`: Authoritative scene graph, live stage tracker (`stages`), partial failures list (`partialFailures`), visualization mode, and actions (`applyEvent`, `cancelActiveJob`, `reconcileAuthoritativeScene`).
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, backend connectivity status, and error alerts.

---

## 3. Backend Architecture (Phase 04)

### 3.1 Network & Privacy Boundary
- The API binds exclusively to loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, depth maps, model checkpoints, and scene metadata remain strictly on the local filesystem.
- No external cloud AI vision APIs are invoked.

### 3.2 Real-Time SSE Stream & Bounded Emitter (`emitter.py`)
- **Bounded Event Retention**: `EventEmitter` maintains at most 500 events per job in a FIFO buffer, preventing unbounded memory growth during long sessions.
- **Server-Sent Events (SSE)**: `GET /api/v1/jobs/{job_id}/stream` delivers real-time notifications to subscribers via `asyncio.Queue` pub/sub, with automatic replay of historical events since `since` sequence or `Last-Event-ID`.
- **Honest Progress Reporting**: Events report exact per-entity progress (e.g., `Segmentation mask ready for Person 1 (1 of 2)`) rather than fabricated percentage estimates.

### 3.3 Job Tracker & Cancellation (`tracker.py`)
- `JobTracker` records active job states: `pending`, `running`, `completed`, `failed`, `cancelled`.
- `cancel_job(job_id)` marks the job cancelled and broadcasts `job_cancelled`.
- Coordinator checks `job_tracker.is_cancelled(job_id)` at stage boundaries and exits cleanly without discarding previously persisted assets.

### 3.4 Progressive Coordinator Execution (`coordinator.py`)
- **Progressive Entity Commit**: Commits initial `SceneEntity` objects to `scene_manager` immediately after detection validation, so client queries reflect candidates before segmentation starts.
- **Progressive Mask Commit**: As each individual mask is persisted to disk, the entity's segmentation metadata is updated and committed to `scene_manager`.
- **Progressive Depth Commit**: Persists machine-readable `.npy` float32 depth map and grayscale visualization PNG, and updates `scene.depth` immediately upon normalization.
- **Non-Fatal Error Isolation**: If depth estimation fails (e.g. missing weights), records error in `scene.depth`, but preserves detection and segmentation results.

### 3.2 Hardware Detection & Acceleration (`device.py`)
- `DeviceManager` automatically detects local compute accelerators:
  - **Apple Silicon MPS** (`mps`): Verified on macOS with Metal Performance Shaders.
  - **NVIDIA CUDA** (`cuda`): Verified on Linux/Windows with CUDA-compatible GPUs.
  - **CPU Fallback** (`cpu`): Always available fallback with multithreading support.
- Performs actual tensor allocation verification before accepting a device as healthy.
- Exposes device details via `GET /api/v1/pipeline/hardware`.

### 3.3 Model Weight Management (`models.py`)
- Designated local cache directory: `data/models/`.
- Pretrained models:
  - **YOLOv8 Nano Segmentation** (`yolov8n-seg.pt`, 6.7 MB) for person detection and instance segmentation.
  - **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`, 99.2 MB Safetensors) for high-resolution relative depth estimation.
- Verifies local file existence and file size before execution.
- Downloads only upon explicit user request or setup command with progress reporting; never downloads silently.
- Exposes model status via `GET /api/v1/pipeline/models` and explicit fetch via `POST /api/v1/pipeline/models/download`.

### 3.4 Strict Image Validation (`validation.py`)
- **Decompression Bomb Protection**: Enforces PIL `Image.MAX_IMAGE_PIXELS = 50_000_000` (rejects excessive memory allocation).
- **Dimension Validation**: Width and height must be within $[32, 8192]$ pixels.
- **Format Verification**: Supports JPEG, PNG, and non-animated WebP. Animated GIFs/WebPs are rejected.
- **Integrity Check**: Re-encodes and validates file integrity before saving.
- **EXIF Orientation**: Transposes orientation according to EXIF tags before storage to prevent rotated bounding boxes.

### 3.5 Depth Estimation & Spatial Pipeline (`depth.py`)
- **Estimator (`DepthAnythingV2Estimator`)**: Monocular depth inference using HuggingFace `AutoModelForDepthEstimation` with a lightweight ViT-Small backbone and DPT neck. Resizes raw output back to native image dimensions with bilinear interpolation.
- **Canonical Proximity Convention**: Defines canonical relative proximity $P \in [0.0, 1.0]$ where $1.0$ is the nearest foreground and $0.0$ is the farthest background.
- **Robust Percentile Normalization (`normalize_depth_map`)**: Inverts distance-like raw output and clamps extreme outliers using 2nd and 98th percentiles:
  $$P(x, y) = \mathrm{clip}\left(\frac{p_{98} - d(x, y)}{p_{98} - p_2}, 0.0, 1.0\right)$$
  Protects against uniform/constant depth fields, zero variance, NaNs, and infinities.
- **Per-Person Depth Association (`compute_entity_depth_stats`)**: Indexes depth map using each person's instance segmentation mask (or bounding box if mask is absent) to calculate median proximity, min/max depth, depth variation ($\sigma$), and pixel coverage.
- **Persistence**: Writes float32 NumPy array (`.npy`) for lossless machine consumption and 8-bit single-channel grayscale PNG (`.png`) for rendering textures and visual inspection.

### 3.6 Multi-Stage Pipeline Coordinator (`coordinator.py`)
Orchestrates 12 sequential stages with monotonic sequence numbers and event emission:
1. `image_validation`: Validates file format, decompression bomb limits, and data integrity.
2. `image_preparation`: Normalizes EXIF orientation and extracts dimensions.
3. `person_detection`: Runs local YOLOv8 neural network inference to locate person bounding boxes.
4. `detection_validation`: Filters low-confidence detections and assigns stable entity IDs.
5. `instance_segmentation`: Generates individual instance masks resized to native photo resolution.
6. `mask_validation`: Validates binary mask boundaries and pixel area statistics.
7. `depth_estimation`: Runs local Depth Anything V2 monocular depth model across the entire photo.
8. `depth_normalization`: Performs robust 2nd/98th percentile normalization to canonical proximity $[0, 1]$.
9. `depth_association`: Computes per-entity depth statistics using segmentation masks.
10. `spatial_photo_preparation`: Configures 3D spatial mesh parameters and parallax camera boundaries.
11. `asset_persistence`: Encodes and writes PNG masks, `.npy` raw depth arrays, and depth visualization PNGs.
12. `scene_update`: Atomically updates the scene graph with entities, depth metadata, and spatial representations.

### 3.7 Asset Manager & Storage Layout
- Storage root: `data/storage/`
  - `photos/`: Source photographs validated and sanitized.
  - `masks/`: 8-bit binary PNG instance masks named `{uuid}_{entity_id}_mask.png`.
  - `depth/`: Machine-readable float32 `.npy` depth maps and 8-bit grayscale `.png` visualizations.
  - `meshes/`: Reserved for Phase 06 3D polygon meshes.
  - `skeletons/`: Reserved for Phase 08 kinematic skeletons.
- Path traversal protection (`StorageSecurityError`) on all file reads and writes.
