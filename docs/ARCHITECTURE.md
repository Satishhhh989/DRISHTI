# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 05 implements the **AI Human 3D Reconstruction Engine**, turning detected and segmented people into genuine 3D volumetric human meshes with real polygon geometry (6,890 vertices, 13,776 faces). DRISHTI integrates the SPIN (SMPL Optimization in the Loop of Neural Networks) model and SMPL Neutral Body linear blend skinning running locally on Apple Silicon MPS / CUDA / CPU, exports standard binary GLTF (`.glb`) assets, and renders them interactively in Three.js under both wireframe topology and neutral clay solid 3D modes with full 360-degree rotation and inspection.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js)                              |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation / Vis Modes     |  | Pipeline &     |  |  |
|  |  |  (Status & Reset)   |  | (Availability-Driven Selector) |  | Inspector Pane |  |  |
|  |  |                     |  | (Orig / Detect / Seg / Depth / |  | (Stages, Mesh, |  |  |
|  |  |                     |  |  Spatial Photo / Wire / Solid) |  |  People, Depth)|  |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |                        Three.js Viewport (React Three Fiber)                      |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  |  | Perspective Camera |  | Upright PhotoPlane /             |  | HumanMeshLayer|  |  |
|  |  | & Lighting         |  | SpatialPhotoSurface (128x128)   |  | - Wireframe   |  |  |
|  |  |                    |  |   + Vertex Displacement Map      |  | - Solid Clay  |  |  |
|  |  |                    |  |   + 3D Detection Boxes & Masks   |  | - Raycast Sel |  |  |
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
|  |                       |  |                      |  | (/photos, /masks, /meshes)   |  |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Event Emitter (Bounded 500-event FIFO)          |  | Job State Tracker            |  |
|  | Real-Time SSE Stream (/jobs/{id}/stream)        |  | Cancellation Handler         |  |
|  +-------------------------------------------------+  +------------------------------+  |
|  | Progressive Multi-Stage Intelligence Pipeline (Phase 01 - 05)                     |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU fallback                  |  |
|  |  - ModelRegistry: Local cache in data/models/                                     |  |
|  |      * yolov8n-seg.pt (6.7MB)                                                     |  |
|  |      * depth-anything-v2-small/ (99.2MB safetensors)                             |  |
|  |      * spin_model_checkpoint.pth.tar (103MB) + SMPL_N_model.npz (88MB)          |  |
|  |  - Progressive Entity Commit: Entities committed immediately after detection     |  |
|  |  - Per-Person Mask Commit: Individual masks committed and broadcast upon save     |  |
|  |  - Progressive Depth Commit: Depth map and spatial photo committed upon save      |  |
|  |  - Human 3D Reconstruction: SPIN + pure PyTorch SMPL LBS forward deformation      |  |
|  |  - Mesh Validation & Sanitization: Finite values, manifold bounds, GLB export     |  |
|  |  - Progressive Mesh Commit: Per-person GLB committed and broadcast upon save     |  |
|  |  - Cancellation Checkpoints: Clean early termination preserving prior assets     |  |
|  |  - AnalysisPipelineCoordinator: 12 sequential stages with live event stream       |  |
|  +-----------------------------------------------------------------------------------+  |
|  | Local Filesystem Storage (photos/, masks/, depth/, meshes/, skeletons/)           |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+

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

## 2. Frontend Architecture (Phase 05)

### 2.1 UI Layer
The UI provides an interactive workspace focused on the 3D viewport:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 05`), backend connectivity status indicator, grid floor toggle, camera reset button, and file upload trigger.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid, and camera controls.
- **Representation & Visualization Selector**: Computes availability dynamically using `computeRepresentationAvailability` based on real asset existence and active job stage. Wireframe and Solid 3D representations unlock immediately as per-person 3D meshes are reconstructed. Available modes: `Original`, `Detection`, `Segmentation`, `Depth Map`, `Spatial Photo`, `Wireframe`, `Solid 3D`.
- **HumanMeshLayer (`HumanMeshLayer.tsx`)**: Renders reconstructed volumetric 3D human meshes using Three.js `GLTFLoader`:
  - *Wireframe Mode*: Highlights genuine polygon edge topology in technical cyan/white lines.
  - *Solid 3D Mode*: Renders a neutral studio clay PBR material (`#d0d0d8`, roughness 0.65, metalness 0.05) with soft ambient and directional lighting.
  - *Entity Selection*: Clicking on a human mesh in the viewport selects the entity and highlights it with an emissive studio rim outline.
  - *Resource Management*: Disposes geometries, materials, and textures cleanly on unmount or mesh replacement.
- **Progressive Processing Inspector (`ContextualPanel.tsx`)**: Compact stage-by-stage pipeline inspector displaying real-time state for each operation:
  - *Image Validation*: Validated on upload.
  - *Person Detection*: Reports count of people found immediately after detection.
  - *Segmentation*: Displays honest per-person progress (`X of Y masks ready`).
  - *Scene Depth*: Shows status (`Estimating...`, `Ready`, `Failed`).
  - *3D Human Mesh*: Shows status (`Reconstructing...`, `Ready`, `Failed`).
  - *Spatial Photo*: Shows readiness.
  - *Entity Inspector*: Displays 3D Mesh Status (`Ready` / `Not Reconstructed`), Vertex Count (e.g. 6,890), Face Count (e.g. 13,776), Dimensions (Width, Height, Depth in meters), Model Provenance (`SPIN + SMPL LBS (Pretrained)`), and Texture Status (`Geometry Only - Neutral Studio Clay`).
  - *Cancellation*: "Cancel Analysis" button when a job is active, stopping subsequent computation safely without losing completed assets.
  - *Partial Failures*: Unobtrusive warning banner highlighting non-fatal errors while keeping successful layers usable.
- **ProcessingIndicator**: Status bar showing live engine state, current stage message, and pulsing progress during neural network analysis.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto an upright 3D plane at world coordinates `(0, height/2, 0)` with aspect ratio preserved.
- **Progressive Overlays & Meshes**:
  - Person bounding boxes appear in 3D space immediately after detection completes, making people selectable right away.
  - Segmentation masks appear progressively as each individual mask asset is persisted.
  - Depth map visualization texture becomes active as soon as normalized depth is stored.
  - Spatial photo vertex-displaced mesh becomes active as soon as spatial photo preparation completes.
  - Reconstructed 3D human meshes load and display progressively in wireframe or solid clay mode as each person finishes inference.
- **Camera & Selection Preservation**: Incoming processing events are processed idempotently without resetting camera position or deselecting the active entity. Controls permit full 360-degree orbit, pan, zoom, and instant camera reset.

### 2.3 State Management & Event Reducer
- `event-reducer.ts`: Pure, deterministic state reducer `reduceProcessingEvent(state, event)` enforcing strict sequence numbers, duplicate event rejection, wrong-scene filtering, mesh asset tracking (`asset_ready` with `asset_type === "mesh"`), and isolated partial failure handling.
- `api-client.ts`: Live EventSource subscription to `GET /api/v1/jobs/{id}/stream` with automatic fallback to polling, and cancellation via `POST /api/v1/jobs/{id}/cancel`.
- `useSceneStore`: Authoritative scene graph, live stage tracker (`stages`), partial failures list (`partialFailures`), visualization mode, and actions (`applyEvent`, `cancelActiveJob`, `reconcileAuthoritativeScene`).
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, backend connectivity status, and error alerts.

---

## 3. Backend Architecture (Phase 05)

### 3.1 Network & Privacy Boundary
- The API binds exclusively to loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, depth maps, 3D meshes, model checkpoints, and scene metadata remain strictly on the local filesystem.
- No external cloud AI vision APIs are invoked.

### 3.2 Real-Time SSE Stream & Bounded Emitter (`emitter.py`)
- **Bounded Event Retention**: `EventEmitter` maintains at most 500 events per job in a FIFO buffer, preventing unbounded memory growth during long sessions.
- **Server-Sent Events (SSE)**: `GET /api/v1/jobs/{job_id}/stream` delivers real-time notifications to subscribers via `asyncio.Queue` pub/sub, with automatic replay of historical events since `since` sequence or `Last-Event-ID`.
- **Honest Progress Reporting**: Events report exact per-entity progress (e.g., `3D human mesh reconstructed for Person 1 (1 of 2)`) rather than fabricated percentage estimates.

### 3.3 Job Tracker & Cancellation (`tracker.py`)
- `JobTracker` records active job states: `pending`, `running`, `completed`, `failed`, `cancelled`.
- `cancel_job(job_id)` marks the job cancelled and broadcasts `job_cancelled`.
- Coordinator checks `job_tracker.is_cancelled(job_id)` at stage boundaries and exits cleanly without discarding previously persisted assets.

### 3.4 Progressive Coordinator Execution (`coordinator.py`)
- **Progressive Entity Commit**: Commits initial `SceneEntity` objects to `scene_manager` immediately after detection validation, so client queries reflect candidates before segmentation starts.
- **Progressive Mask Commit**: As each individual mask is persisted to disk, the entity's segmentation metadata is updated and committed to `scene_manager`.
- **Progressive Depth Commit**: Persists machine-readable `.npy` float32 depth map and grayscale visualization PNG, and updates `scene.depth` immediately upon normalization.
- **Progressive Mesh Commit**: As each individual person's 3D mesh is reconstructed and validated, it is registered under `meshes/` as a GLB binary asset, committed to the authoritative scene, and broadcast via `asset_ready`.
- **Non-Fatal Error Isolation**: If one person's reconstruction fails (e.g. invalid crop or out-of-bounds coordinates), records warning and continues reconstructing other persons.

### 3.5 Hardware Detection & Acceleration (`device.py`)
- `DeviceManager` automatically detects local compute accelerators:
  - **Apple Silicon MPS** (`mps`): Verified on macOS with Metal Performance Shaders.
  - **NVIDIA CUDA** (`cuda`): Verified on Linux/Windows with CUDA-compatible GPUs.
  - **CPU Fallback** (`cpu`): Always available fallback with multithreading support.
- Performs actual tensor allocation verification before accepting a device as healthy.
- Exposes device details via `GET /api/v1/pipeline/hardware`.

### 3.6 Model Weight Management (`models.py`)
- Designated local cache directory: `data/models/`.
- Pretrained models:
  - **YOLOv8 Nano Segmentation** (`yolov8n-seg.pt`, 6.7 MB) for person detection and instance segmentation.
  - **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`, 99.2 MB Safetensors) for high-resolution relative depth estimation.
  - **SPIN Human Mesh Reconstruction** (`spin_model_checkpoint.pth.tar`, 103 MB) with **SMPL Neutral Model** (`SMPL_N_model.npz`, 88 MB) for parametric 3D body estimation.
- Verifies local file existence and file size before execution.
- Downloads only upon explicit user request or setup command with progress reporting; never downloads silently.
- Exposes model status via `GET /api/v1/pipeline/models` and explicit fetch via `POST /api/v1/pipeline/models/download`.

### 3.7 Strict Image Validation (`validation.py`)
- **Decompression Bomb Protection**: Enforces PIL `Image.MAX_IMAGE_PIXELS = 50_000_000` (rejects excessive memory allocation).
- **Dimension Validation**: Width and height must be within $[32, 8192]$ pixels.
- **Format Verification**: Supports JPEG, PNG, and non-animated WebP. Animated GIFs/WebPs are rejected.
- **Integrity Check**: Re-encodes and validates file integrity before saving.
- **EXIF Orientation**: Transposes orientation according to EXIF tags before storage to prevent rotated bounding boxes.

### 3.8 Depth Estimation & Spatial Pipeline (`depth.py`)
- **Estimator (`DepthAnythingV2Estimator`)**: Monocular depth inference using HuggingFace `AutoModelForDepthEstimation` with a lightweight ViT-Small backbone and DPT neck. Resizes raw output back to native image dimensions with bilinear interpolation.
- **Canonical Proximity Convention**: Defines canonical relative proximity $P \in [0.0, 1.0]$ where $1.0$ is the nearest foreground and $0.0$ is the farthest background.
- **Robust Percentile Normalization (`normalize_depth_map`)**: Inverts distance-like raw output and clamps extreme outliers using 2nd and 98th percentiles:
  $$P(x, y) = \mathrm{clip}\left(\frac{p_{98} - d(x, y)}{p_{98} - p_2}, 0.0, 1.0\right)$$
  Protects against uniform/constant depth fields, zero variance, NaNs, and infinities.
- **Per-Person Depth Association (`compute_entity_depth_stats`)**: Indexes depth map using each person's instance segmentation mask (or bounding box if mask is absent) to calculate median proximity, min/max depth, depth variation ($\sigma$), and pixel coverage.
- **Persistence**: Writes float32 NumPy array (`.npy`) for lossless machine consumption and 8-bit single-channel grayscale PNG (`.png`) for rendering textures and visual inspection.

### 3.9 Human 3D Reconstruction Subsystem (`reconstructor.py`)
- **SPIN (SMPL Optimization in the Loop of Neural Networks)**:
  - Architecture: ResNet-50 convolutional feature extractor with a 3-step iterative feedback loop regressing 6D continuous rotation representations and SMPL shape ($\beta$) parameters.
  - Forward Layer: Pure PyTorch implementation of Linear Blend Skinning (LBS) that eliminates fragile legacy C++ dependencies (`chumpy`).
  - Standard SMPL Neutral topology: produces $N = 6,890$ 3D vertices and $M = 13,776$ triangular faces.
- **Input Preprocessing**:
  - Extracts person bounding box with 15% contextual padding.
  - Applies aspect-ratio-preserving square letterbox resizing to $224 \times 224$ pixels.
  - Normalizes using standard ImageNet mean and standard deviation.
  - Detects image boundary clipping and flags uncertain truncated limbs.
- **Geometry Validation & Sanitization**:
  - Rejects meshes containing NaNs, infinities, or out-of-range face indices.
  - Filters degenerate zero-area faces.
  - Computes vertex surface normals for smooth lighting.
  - Validates physical bounding boxes against realistic human scale ($[0.2\text{m}, 2.6\text{m}]$).
- **Coordinate System Alignment**:
  - SMPL coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward) align with DRISHTI's metric world space.
  - Positions each human entity in scene space based on 2D box centroid and estimated relative scene depth.
- **Binary GLTF (.glb) Export**:
  - Exports standard binary GLTF (`.glb`) containing vertex positions, face indices, and normals.
  - Assigns a clean neutral studio clay PBR material (`#d0d0d8`, roughness 0.65).
  - Explicitly documents texture status: "Geometry Only - Neutral Studio Clay".

### 3.10 Multi-Stage Pipeline Coordinator (`coordinator.py`)
Orchestrates 13 sequential stages with monotonic sequence numbers and event emission:
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
11. `human_reconstruction`: Runs local SPIN + SMPL LBS model to reconstruct 3D human meshes per person.
12. `asset_persistence`: Encodes and writes PNG masks, `.npy` raw depth arrays, depth visualization PNGs, and GLB meshes.
13. `scene_update`: Atomically updates the scene graph with entities, depth metadata, spatial representations, and mesh paths.

### 3.11 Asset Manager & Storage Layout
- Storage root: `data/storage/`
  - `photos/`: Source photographs validated and sanitized.
  - `masks/`: 8-bit binary PNG instance masks named `{uuid}_{entity_id}_mask.png`.
  - `depth/`: Machine-readable float32 `.npy` depth maps and 8-bit grayscale `.png` visualizations.
  - `meshes/`: Binary GLTF human meshes named `{scene_id}_{entity_id}.glb`.
  - `skeletons/`: Reserved for Phase 08 kinematic skeletons.
- Path traversal protection (`StorageSecurityError`) on all file reads and writes.
- Endpoints:
  - `GET /api/v1/assets/photos/{filename}`
  - `GET /api/v1/assets/masks/{filename}`
  - `GET /api/v1/assets/depth/{filename}`
  - `GET /api/v1/assets/meshes/{filename}`
