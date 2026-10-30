# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 02 implements the **Image Intelligence & Person Segmentation** pipeline, endowing DRISHTI with genuine understanding of multi-person photographs through local neural network inference, instance segmentation, and spatial viewport overlay.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js)                              |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation / Vis Modes     |  | Inspector Pane |  |  |
|  |  |  (Status & Reset)   |  | (Orig / Detect / Seg / Wire/3D)|  | (People & Dets)|  |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |                        Three.js Viewport (React Three Fiber)                      |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  |  | Perspective Camera |  | Upright PhotoPlane (Aspect locked|  | OrbitControls |  |  |
|  |  | & Lighting         |  |   + 3D Detection Bounding Boxes  |  | (Pan/Rot/Zoom)|  |  |
|  |  |                    |  |   + 3D Segmentation Overlays)    |  |               |  |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |  Zustand Stores: useSceneStore (Scene, Detections, Masks) / useUIStore (UI state)  |  |
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
|  | Image Intelligence Pipeline (Phase 02)                                            |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU fallback                  |  |
|  |  - ModelRegistry: Local cache in data/models/ (yolov8n-seg.pt, 6.7MB)             |  |
|  |  - Validation: Decompression bomb protection (50M px), corrupt check, EXIF orient |  |
|  |  - YoloPersonDetector: Multi-person bounding box detection (class 0)              |  |
|  |  - YoloPersonSegmentor: Instance masks, full-res bilinear resize, 8-bit binary PNG|  |
|  |  - AnalysisPipelineCoordinator: 9 sequential stages with monotonic event stream   |  |
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

## 2. Frontend Architecture (Phase 02)

### 2.1 UI Layer
The UI provides an interactive workspace focused on the 3D viewport:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 02: Image Intelligence`), backend connectivity status indicator, grid floor toggle, camera reset button, and file upload trigger.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid, and camera controls.
- **Representation & Visualization Selector**: Allows instant switching between:
  - `Original`: Clean photograph in 3D space.
  - `Detection`: 3D bounding boxes and entity name tags rendered around detected people.
  - `Segmentation`: Semi-transparent colored alpha overlays highlighting individual person silhouettes.
  - `Wireframe`: Reconstructed 3D geometry (honest placeholder for Phase 06).
  - `3D Solid`: Animatable 3D human meshes (honest placeholder for Phase 06/07).
- **ContextualPanel**: Collapsible inspector displaying scene identifier, image resolution, aspect ratio, detected people list with color indicators, and interactive selection. Selecting a person highlights their card and focuses their 3D bounding box.
- **ProcessingIndicator**: Status bar showing live engine state, current pipeline stage message, and pulsing progress during neural network analysis.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto an upright 3D plane at world coordinates `(0, height/2, 0)` with aspect ratio preserved.
- **DetectionBox**: Three.js line-segment wireframe planes placed at a slight $Z$-offset ($+0.005$) directly on the photo surface, with a rendered HTML badge showing person label and confidence score. Clicking a bounding box selects the entity in the inspector.
- **SegmentationMaskLayer**: Three.js colored planes textured with the person's binary mask (using alpha channel mapping) placed at $Z$-offset ($+0.002$) to eliminate Z-fighting.
- Camera rotation, panning, and zooming preserve pixel-perfect alignment between the photo, bounding boxes, and segmentation masks.

### 2.3 State Management
State is divided across two Zustand stores:
- `useSceneStore`: Authoritative scene graph, entities, detection info, segmentation masks, active visualization mode (`original` | `detection` | `segmentation` | `wireframe` | `solid3d`), selected entity ID, and active pipeline analysis status.
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, backend connectivity status, and error alerts.

---

## 3. Backend Architecture (Phase 02)

### 3.1 Network & Privacy Boundary
- The API binds exclusively to loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, model checkpoints, and scene metadata remain strictly on the local filesystem.
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
- Pretrained model: **YOLOv8 Nano Segmentation** (`yolov8n-seg.pt`, 6.7 MB).
  - Selected for balance of low latency (<50ms on MPS), small disk footprint, simultaneous bounding box and instance mask generation, and offline execution.
- Verifies local file existence and file size before execution.
- Downloads only upon explicit user request or setup command with progress reporting; never downloads silently.
- Exposes model status via `GET /api/v1/pipeline/models` and explicit fetch via `POST /api/v1/pipeline/models/download`.

### 3.4 Strict Image Validation (`validation.py`)
- **Decompression Bomb Protection**: Enforces PIL `Image.MAX_IMAGE_PIXELS = 50_000_000` (rejects excessive memory allocation).
- **Dimension Validation**: Width and height must be within $[32, 8192]$ pixels.
- **Format Verification**: Supports JPEG, PNG, and non-animated WebP. Animated GIFs/WebPs are rejected.
- **Integrity Check**: Re-encodes and validates file integrity before saving.
- **EXIF Orientation**: Transposes orientation according to EXIF tags before storage to prevent rotated bounding boxes.

### 3.5 Person Detection & Segmentation Pipeline
- **Detector (`detector.py`)**: `YoloPersonDetector` filters class 0 (`person`), extracts normalized bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$, computes exact integer pixel boxes, and assigns stable entity IDs (`person-001`, `person-002`, ...).
- **Segmentor (`segmentor.py`)**: `YoloPersonSegmentor` performs bilinear interpolation of the raw mask tensor to the source image dimensions, thresholds binary activations, and saves single-channel 8-bit PNG masks ($255$ foreground, $0$ background) to `data/storage/masks/`.
- **Coordinator (`coordinator.py`)**: Orchestrates 9 sequential stages with monotonic sequence numbers and event emission:
  1. `image_validation`
  2. `image_preparation`
  3. `person_detection`
  4. `detection_validation`
  5. `instance_segmentation`
  6. `mask_validation`
  7. `asset_persistence`
  8. `scene_update`
  9. `completed`

### 3.6 Asset Manager & Storage Layout
- Storage root: `data/storage/`
  - `photos/`: Source photographs validated and sanitized.
  - `masks/`: 8-bit binary PNG instance masks named `{uuid}_{entity_id}_mask.png`.
  - `depth/`: Reserved for Phase 04 depth maps.
  - `meshes/`: Reserved for Phase 06 3D polygon meshes.
  - `skeletons/`: Reserved for Phase 08 kinematic skeletons.
- Path traversal protection (`StorageSecurityError`) on all file reads and writes.
