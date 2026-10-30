# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 03 implements the **Depth Estimation & Spatial Photography** pipeline, empowering DRISHTI to transform 2D multi-person photographs into depth-aware, explorable 3D spatial photographs through genuine local neural network inference, robust percentile depth normalization, per-person depth statistics, and Three.js vertex-displaced mesh rendering with controlled parallax.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js)                              |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation / Vis Modes     |  | Inspector Pane |  |  |
|  |  |  (Status & Reset)   |  | (Orig / Detect / Seg / Depth / |  | (People, Depth |  |  |
|  |  |                     |  |  Spatial Photo / Wire / 3D)    |  |  Stats & Tuner)|  |  |
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
|  |  Zustand Stores: useSceneStore (Scene, Detections, Masks, Depth) / useUIStore     |  |
|  +-----------------------------------------------------------------------------------+  |
+--------------------------------------------|--------------------------------------------+
                                             | HTTP / JSON (127.0.0.1:8000)
                                             v
+-----------------------------------------------------------------------------------------+
|                             Local Backend Service (FastAPI)                             |
|                                                                                         |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Health & Version APIs |  | Scene Lifecycle API  |  | Asset Manager API            |  |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Event Emitter & Catch-up Engine                 |  | Job State Tracker            |  |
|  +-------------------------------------------------+  +------------------------------+  |
|  | Multi-Stage Intelligence Pipeline (Phase 01 - 03)                                 |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU fallback                  |  |
|  |  - ModelRegistry: Local cache in data/models/                                     |  |
|  |      * yolov8n-seg.pt (6.7MB)                                                     |  |
|  |      * depth-anything-v2-small/ (99.2MB safetensors)                             |  |
|  |  - Validation: Decompression bomb protection (50M px), corrupt check, EXIF orient |  |
|  |  - YoloPersonDetector: Multi-person bounding box detection (class 0)              |  |
|  |  - YoloPersonSegmentor: Instance masks, full-res bilinear resize, 8-bit binary PNG|  |
|  |  - DepthAnythingV2Estimator: Monocular depth estimation via local ViT-Small model  |  |
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

## 2. Frontend Architecture (Phase 03)

### 2.1 UI Layer
The UI provides an interactive workspace focused on the 3D viewport:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 03: Depth & Spatial Photo`), backend connectivity status indicator, grid floor toggle, camera reset button, and file upload trigger.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid, and camera controls.
- **Representation & Visualization Selector**: Allows instant switching between:
  - `Original`: Clean photograph in 3D space.
  - `Detection`: 3D bounding boxes and entity name tags rendered around detected people.
  - `Segmentation`: Semi-transparent colored alpha overlays highlighting individual person silhouettes.
  - `Depth Map`: Normalized 8-bit grayscale depth visualization ($255 = \text{closest foreground}$, $0 = \text{farthest background}$).
  - `Spatial Photo`: Vertex-displaced 3D photo mesh surface with interactive parallax camera movement.
  - `Wireframe`: Reconstructed 3D geometry (honest placeholder for Phase 06).
  - `3D Solid`: Animatable 3D human meshes (honest placeholder for Phase 06/07).
- **ContextualPanel**: Collapsible inspector displaying scene identifier, image resolution, aspect ratio, depth metadata (model name, canonical proximity convention, device), depth strength tuning slider ($0.00 - 0.25$, default $0.08$), camera reset action, and detected people list with per-person median/min/max proximity stats and pixel coverage.
- **ProcessingIndicator**: Status bar showing live engine state, current pipeline stage message, and pulsing progress during neural network analysis.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto an upright 3D plane at world coordinates `(0, height/2, 0)` with aspect ratio preserved.
- **DetectionBox**: Three.js line-segment wireframe planes placed at a slight $Z$-offset ($+0.005$) directly on the photo surface, with a rendered HTML badge showing person label and confidence score. Clicking a bounding box selects the entity in the inspector.
- **SegmentationMaskLayer**: Three.js colored planes textured with the person's binary mask (using alpha channel mapping) placed at $Z$-offset ($+0.002$) to eliminate Z-fighting.
- **SpatialPhotoSurface**: In `spatial_photo` mode, the flat plane is replaced by a high-resolution $128 \times 128$ subdivided mesh with `MeshStandardMaterial` vertex displacement mapped to the normalized depth texture. Foreground elements protrude along $+Z$ towards the viewer proportional to their relative proximity $P$.
- **Bounded Parallax Camera (`CameraController.tsx`)**: When in `spatial_photo` mode, camera orbit limits are constrained to realistic parallax ranges (azimuth angle $\pm 18^\circ$, polar elevation $75^\circ \text{--} 105^\circ$, camera distance $1.8 \text{--} 4.5\text{m}$) with smooth damping ($0.06$). This avoids disocclusion tearing and perspective breakdown while enabling rich depth exploration.

### 2.3 State Management
State is divided across two Zustand stores:
- `useSceneStore`: Authoritative scene graph, entities, detection info, segmentation masks, depth metadata, active visualization mode (`original` | `detection` | `segmentation` | `depth` | `spatial_photo` | `wireframe` | `solid3d`), depth strength scale, selected entity ID, and active pipeline analysis status.
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, backend connectivity status, and error alerts.

---

## 3. Backend Architecture (Phase 03)

### 3.1 Network & Privacy Boundary
- The API binds exclusively to loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, depth maps, model checkpoints, and scene metadata remain strictly on the local filesystem.
- No external cloud AI vision APIs are invoked.

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
