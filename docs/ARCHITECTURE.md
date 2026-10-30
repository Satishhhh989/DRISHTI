# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 10 completes the full engineering roadmap, delivering the complete, integrated, security-hardened **DRISHTI v1.0.0** release across four primary subsystems:
1. **DRISHTI Engine**: Local vision inference (YOLOv8-seg, Depth Anything V2, SPIN + SMPL LBS, YOLOv8-pose) executing on Apple Silicon MPS, NVIDIA CUDA, or CPU fallback.
2. **DRISHTI Studio**: Unified 9-representation interactive 3D workspace (Next.js 15, React 19, Three.js, React Three Fiber, Zustand) with camera state preservation and progressive asset reveals.
3. **DRISHTI Scene**: Canonical scene graph, metric spatial coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward), ground contact alignment, and persistent asset manifests.
4. **DRISHTI Motion**: Canonical 24-joint kinematic skeleton, normalized Linear Blend Skinning weights, and deterministic procedural animations.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js 15)                           |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Unified Representation Switcher|  | Contextual &   |  |  |
|  |  |  (Fit Scene, Reset, |  | (9 Modes: Orig / Detect / Seg / |  | Transform       |  |  |
|  |  |   Save, Export)     |  |  Depth / Spatial / Wire / Solid|  | Inspector Pane |  |  |
|  |  |                     |  |  Articulated / Combined)       |  | (Live Stages)  |  |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |                        Three.js Viewport (React Three Fiber)                      |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  |  | Perspective Camera |  | Upright PhotoPlane /             |  | HumanMeshLayer|  |  |
|  |  | & Lighting         |  | SpatialPhotoSurface (128x128)   |  | - Wireframe   |  |  |
|  |  | - Single Contract  |  |   + Vertex Displacement Map      |  | - Solid Clay  |  |  |
|  |  | - State Preserved  |  |   + 3D Detection Boxes & Masks   |  | - Skeleton Rig|  |  |
|  |  | - Orbit / Pan/Zoom |  |   + Photographic Reference Plane |  | - Procedural  |  |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |  Zustand Stores: useSceneStore (Scene, Reducer, Stages) / useUIStore              |  |
|  |  Event Reducer: Idempotent sequence ordering, partial failure isolation           |  |
|  |  Asset Cache: Direct geometry and texture reuse across representation switches    |  |
|  +-----------------------------------------------------------------------------------+  |
+--------------------------------------------|--------------------------------------------+
                                             | HTTP Security Headers / SSE Stream (127.0.0.1:8000)
                                             v
+-----------------------------------------------------------------------------------------+
|                             Local Backend Service (FastAPI)                             |
|                                                                                         |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Health & Version APIs |  | Scene Lifecycle API  |  | Asset & Export Engine API    |  |
|  | (127.0.0.1 Loopback)  |  | (/scenes, /projects) |  | (/export: glb, scene, zip)   |  |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Event Emitter (Bounded 500-event FIFO)          |  | Anti-Zip-Slip & Bomb Defense |  |
|  | Real-Time SSE Stream (/jobs/{id}/stream)        |  | Safe Project Archive Import  |  |
|  +-------------------------------------------------+  +------------------------------+  |
|  | Progressive Multi-Stage Intelligence Pipeline                                     |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU SIMD fallback             |  |
|  |  - ModelRegistry: Local cache in data/models/                                     |  |
|  |      * yolov8n-seg.pt (7.1MB) & yolov8n-pose.pt (6.8MB)                           |  |
|  |      * depth-anything-v2-small/ (95MB safetensors)                                |  |
|  |      * spin_model_checkpoint.pth.tar (103MB) + SMPL_N_model.npz (84MB)          |  |
|  |  - Input Validation: 25MP decompression bomb check, 25MB upload cap, no anims    |  |
|  |  - Depth Estimation: Relative proximity [0, 1] with 2nd/98th percentile scaling   |  |
|  |  - Human 3D Reconstruction: SPIN + pure PyTorch SMPL LBS (6,890 verts, 13,776 f)  |  |
|  |  - Multi-Person Scene Assembly Engine: Ground contact, horizontal ordering, depth |  |
|  |  - DRISHTI Motion: 24-joint skeleton, SMPL 3D regression, LBS rigging weights,   |  |
|  |    deterministic procedural animation (idle, twist, wave, walk)                   |  |
|  |  - Atomic Disk Persistence: .tmp buffer & atomic replace for projects and scenes   |  |
|  +-----------------------------------------------------------------------------------+  |
|  | Local Sandboxed Storage (photos/, masks/, depth/, meshes/, scenes/, skeletons/)   |  |
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

## 2. Frontend Architecture & Unified Hybrid Renderer (Phase 08)

### 2.1 UI Layer & Representation System
The UI provides an interactive workspace centered on a unified rendering architecture:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 08`), backend connectivity status indicator, grid floor toggle, "Fit Scene" button, camera reset button, and file upload trigger.
- **Representation Registry (`packages/scene-schema/src/registry.ts`)**:
  - Central source of truth for all 9 supported representations:
    1. `original`: Original source photograph (2D photographic aspect ratio preserved without geometric warping).
    2. `detection`: 2D bounding boxes with person label tags and detection confidences.
    3. `segmentation`: Per-person 8-bit alpha instance masks projected over photograph.
    4. `depth`: Normalized grayscale relative depth map $[0.0, 1.0]$.
    5. `spatial_photo`: 128x128 vertex-displaced mesh surface with restrained parallax orbit.
    6. `wireframe`: Genuine reconstructed polygon mesh wireframe topology.
    7. `solid3d`: Reconstructed mesh with neutral studio clay material under 3-point lighting.
    8. `articulated`: Rigged mesh with 24-joint kinematic skeleton overlay and procedural animation tracks.
    9. `combined`: Hybrid spatial composition rendering assembled 3D characters in front of an upright photographic reference plane.
- **Unified Asset Cache (`apps/web/lib/asset-cache.ts`)**:
  - Deduplicated in-memory template caching for GLTF models (`.glb`) and Three.js textures.
  - Eliminates duplicate network requests and re-parsing when switching between representations.
  - Reuses underlying geometry buffers across solid, wireframe, articulated, and combined views.
  - Explicit resource disposal on scene change or unmount via `assetCache.disposeAll()`.
- **HumanMeshLayer (`HumanMeshLayer.tsx`)**:
  - Unified 3D mesh presentation supporting `wireframe`, `solid3d`, `articulated`, and `combined` modes.
  - Reconstructed characters positioned via authoritative scene-space transforms ($T_{\text{scene}}$).
  - Unreconstructed characters displayed honestly as 3D wireframe bounding indicators.
  - Integrated `SkeletonOverlay` displaying 24 joint spheres (pelvis in amber `#f59e0b`, limbs in emerald `#10b981`) and 23 bone line segments with X-ray depth testing.
- **Photographic & Combined Reference Plane (`PhotoPlane.tsx`)**:
  - In 2D/inspection modes, renders photograph or depth map at $Z = 0$.
  - In `combined` mode, renders as an upright photographic reference plane at $Z = -0.6\text{m}$ (85% opacity, slate boundary `#6366f1`), documenting the spatial correlation between the original photo and the assembled 3D characters.
- **Camera & Navigation Contract (`CameraController.tsx`)**:
  - **2D Navigation**: Used in `original`, `detection`, `segmentation`, `depth`; rotation locked to face plane frontally, permitting pure pan and zoom without accidental 3D tilting.
  - **3D Navigation**: Used in `wireframe`, `solid3d`, `articulated`, `combined`; unconstrained 360-degree orbit, pan, zoom, fit-to-scene, and focus-on-person.
  - **Restrained Parallax Navigation**: Used in `spatial_photo`; azimuth locked to $\pm 18^\circ$ and polar $75^\circ \text{--} 105^\circ$ to minimize disocclusion voids.
  - **Camera State Preservation**: Switching between 3D representations (`solid3d` $\leftrightarrow$ `wireframe` $\leftrightarrow$ `articulated` $\leftrightarrow$ `combined`) **preserves** the user's active camera position, target, and zoom level.
  - **Fit-to-View**: In 2D mode, computes exact distance to fit the photo plane dimensions ($W, H$) into camera FOV with comfortable padding; in 3D mode, frames visible entities using actual mesh bounding dimensions.
  - *Detected People List*: Per-entity visibility toggle button (show/hide), status tag (`Mesh` vs `In-Scene`), and detection confidence.
  - *Entity Inspector*: Displays 3D Mesh Status, Vertex & Face counts, Bounding Dimensions, Model Provenance, Texture Status, 3D Scene Placement Method (`Depth-Projected Composition`), Scene Coordinates $(X, Y, Z)$, Depth Source, Visibility, Uncertainty Flags, and "Focus Camera on Person" action.
- **CameraController (`CameraController.tsx`)**: Smooth OrbitControls with damping ($0.06$), camera reset, "Fit Scene" bounding box framing, and "Focus on Person" centering.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto an upright 3D plane at world coordinates `(0, height/2, 0)` with aspect ratio preserved.
- **Progressive Overlays & Assembled Meshes**:
  - Person bounding boxes appear in 3D space immediately after detection completes.
  - Segmentation masks appear progressively as each individual mask asset is persisted.
  - Depth map visualization texture becomes active as soon as normalized depth is stored.
  - Spatial photo vertex-displaced mesh becomes active as soon as spatial photo preparation completes.
  - Reconstructed 3D human meshes and unreconstructed entities render in their assembled scene-space positions.
- **Camera & Selection Preservation**: Incoming processing events are processed idempotently without resetting camera position or deselecting the active entity. Controls permit full 360-degree orbit, pan, zoom, fit scene, and focus.

### 2.3 State Management & Event Reducer
- `event-reducer.ts`: Pure, deterministic state reducer `reduceProcessingEvent(state, event)` handling sequence numbers, mesh asset tracking, and `scene_assembly` stage progression.
- `api-client.ts`: Live EventSource subscription to `GET /api/v1/jobs/{id}/stream` with automatic fallback to polling, and cancellation via `POST /api/v1/jobs/{id}/cancel`.
- `useSceneStore`: Authoritative scene graph, live stage tracker (`stages`), partial failures list (`partialFailures`), visualization mode, and actions (`applyEvent`, `toggleEntityVisibility`, `cancelActiveJob`, `reconcileAuthoritativeScene`).
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, fit-scene trigger, focus-entity trigger, backend connectivity status, and error alerts.

---

## 3. Backend Architecture (Phase 06)

### 3.1 Network & Privacy Boundary
- The API binds exclusively to loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, depth maps, 3D meshes, scenes, model checkpoints, and scene metadata remain strictly on the local filesystem.
- No external cloud AI vision APIs are invoked.

### 3.2 Real-Time SSE Stream & Bounded Emitter (`emitter.py`)
- **Bounded Event Retention**: `EventEmitter` maintains at most 500 events per job in a FIFO buffer, preventing unbounded memory growth during long sessions.
- **Server-Sent Events (SSE)**: `GET /api/v1/jobs/{job_id}/stream` delivers real-time notifications to subscribers via `asyncio.Queue` pub/sub, with automatic replay of historical events.
- **Honest Progress Reporting**: Events report exact per-entity progress rather than fabricated percentage estimates.

### 3.3 Job Tracker & Cancellation (`tracker.py`)
- `JobTracker` records active job states: `pending`, `running`, `completed`, `failed`, `cancelled`.
- `cancel_job(job_id)` marks the job cancelled and broadcasts `job_cancelled`.
- Coordinator checks `job_tracker.is_cancelled(job_id)` at stage boundaries and exits cleanly without discarding previously persisted assets.

### 3.4 Progressive Coordinator Execution (`coordinator.py`)
- **Progressive Entity Commit**: Commits initial `SceneEntity` objects immediately after detection validation.
- **Progressive Mask Commit**: As each individual mask is persisted to disk, the entity's segmentation metadata is updated and committed.
- **Progressive Depth Commit**: Persists machine-readable `.npy` float32 depth map and grayscale visualization PNG, and updates `scene.depth`.
- **Progressive Mesh Commit**: As each individual person's 3D mesh is reconstructed and validated, it is registered under `meshes/` as a GLB binary asset, committed to the authoritative scene, and broadcast via `asset_ready`.
- **Scene Assembly Commit**: Builds the full 3D `SceneGraph`, updates entity scene-space transforms and placement provenance, and persists to disk.
- **Non-Fatal Error Isolation**: Partial failures during reconstruction or depth estimation do not abort downstream scene assembly.

### 3.5 Hardware Detection & Acceleration (`device.py`)
- Automatically detects Apple Silicon MPS (`mps`), NVIDIA CUDA (`cuda`), or multithreaded CPU fallback.

### 3.6 Model Weight Management (`models.py`)
- Designated local cache directory: `data/models/`.
- Pretrained models: YOLOv8 Nano Segmentation (`yolov8n-seg.pt`), Depth Anything V2 Small (`depth-anything-v2-small/`), and SPIN Human Mesh Reconstruction (`spin_model_checkpoint.pth.tar` + `SMPL_N_model.npz`).

### 3.7 Strict Image Validation (`validation.py`)
- Decompression bomb protection (`MAX_IMAGE_PIXELS = 50_000_000`), dimension checks ($[32, 8192]$ px), format verification (JPEG, PNG, WebP), and EXIF normalization.

### 3.8 Depth Estimation & Spatial Pipeline (`depth.py`)
- Monocular depth inference using Depth Anything V2 Small with canonical relative proximity $P \in [0.0, 1.0]$ ($1.0$ = nearest foreground, $0.0$ = farthest background) and 2nd/98th percentile clamping.

### 3.9 Human 3D Reconstruction Subsystem (`reconstructor.py`)
- SPIN architecture with ResNet-50 backbone, iterative 6D feedback regressor, and pure PyTorch SMPL Linear Blend Skinning forward layer ($6,890$ vertices, $13,776$ faces).
- Standard binary GLTF (`.glb`) export with neutral studio clay PBR material (`#d0d0d8`).

### 3.10 Scene Assembly Engine (`assembly.py`)
- **Model Space vs Scene Space**:
  - Model space represents canonical mesh geometry ($+X$ right, $+Y$ up, $+Z$ forward) centered at pelvis.
  - Explicit transform decomposition:
    $$\text{sceneTransform} = \text{placementTransform} \times \text{orientationCorrection} \times \text{modelNormalization}$$
- **Horizontal Placement ($X$)**:
  - Maps image-space normalized centroid $c_x$ across the reference plane width $W_{\text{plane}}$:
    $$X_{\text{scene}} = (c_x - 0.5) \cdot W_{\text{plane}}$$
  - Preserves original horizontal ordering (left, center, right).
- **Vertical Placement ($Y$)**:
  - Grounds subjects relative to the floor reference grid ($Y = 0$), mapping vertical centroid $(1.0 - c_y) \cdot H_{\text{plane}}$.
- **Depth Placement ($Z$)**:
  - Uses relative proximity $P_{\text{median}} \in [0.0, 1.0]$ across a calibrated scene depth range ($1.6\text{m}$):
    $$Z_{\text{scene}} = (P_{\text{median}} - 0.5) \cdot Z_{\text{range}}$$
  - Missing depth gracefully falls back to $Z = 0.0$ with uncertainty flag `"missing_depth"`.
- **Perspective & Scale Normalization**:
  - Canonical humanoid scale normalized to avoid double-perspective distortion; metric accuracy is explicitly documented as uncalibrated.
- **Occlusion & Overlap Analysis**:
  - Computes pairwise 2D bounding box IoU and horizontal overlap.
  - Detects proximity differences $|\Delta P| < 0.05$ and emits structured occlusion warnings without destroying composition.
- **Partial Reconstruction Resilience**:
  - Reconstructed entities receive mesh asset references.
  - Entities whose reconstruction failed or is pending remain in the scene graph with their placement transform and detection context.
- **Scene Graph Construction (`SceneGraph`)**:
  - Builds root, camera node, ambient & directional light nodes, reference plane node, ground grid node, and per-person nodes.

### 3.11 DRISHTI Motion Subsystem (`motion.py`)

Phase 07 introduces the **DRISHTI Motion** subsystem, transitioning static reconstructed meshes into fully articulated, animatable 3D human characters:

- **Canonical 24-Joint Kinematic Hierarchy**:
  - Aligned with the SMPL standard topology: `pelvis` (root, index 0), `left_hip`, `right_hip`, `spine1`, `left_knee`, `right_knee`, `spine2`, `left_ankle`, `right_ankle`, `spine3`, `left_foot`, `right_foot`, `neck`, `left_collar`, `right_collar`, `head`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hand`, `right_hand`.
  - Rigorous structural validation (`validate_skeleton_hierarchy`): cycle detection, duplicate index detection, missing parent checks, and non-finite coordinate assertions.
- **2D Pose Estimation**:
  - `YoloPoseEstimator` using local `yolov8n-pose.pt` (6.5MB) with automatic multi-person matching via bounding box IoU.
  - `MockPoseEstimator` fallback ensuring testability without GPU acceleration.
  - Preserves per-keypoint detection confidence mapped from COCO-17 to canonical joints.
- **3D Skeleton Fitting (`SkeletonFitter`)**:
  - Reconstructs 3D joint centers via SMPL joint regression matrix ($J = J_{regressor} \cdot V_{6890}$) or anatomical bounding box landmarks.
  - Rest pose centered with pelvis at origin $(0, 0, 0)$ in local human mesh space (meters, Y-up).
- **Linear Blend Skinning Rigging Engine (`RiggingEngine`)**:
  - SMPL native skinning weights ($6890 \times 24$) from `SMPL_N_model.npz` or inverse distance proximity heat weights for arbitrary geometry.
  - Pruned to the top-4 bone influences per vertex and normalized so weights sum exactly to $1.0$ ($\sum w_i = 1.0$), complying with GPU vertex shader limits (`vec4` weights/indices).
- **Procedural Animation Engine (`ProceduralAnimationEngine`)**:
  - Deterministic procedural clips with quaternion rotations: `idle_breathing` (3.0s), `torso_twist` (4.0s), `arm_wave` (2.5s), `gentle_walk` (2.0s).
  - Explicitly labeled with `provenance: "procedural"`, communicating honestly that motion is procedurally synthesized and not historical motion recovered from the photograph.
  - Independent per-person animation state (`isPlaying`, `currentClip`, `playbackSpeed`, `loop`).

### 3.12 Multi-Stage Pipeline Coordinator (`coordinator.py`)
Orchestrates 18 sequential stages with monotonic sequence numbers:
1. `image_validation`
2. `image_preparation`
3. `person_detection`
4. `detection_validation`
5. `instance_segmentation`
6. `mask_validation`
7. `depth_estimation`
8. `depth_normalization`
9. `depth_association`
10. `spatial_photo_preparation`
11. `human_reconstruction`
12. `pose_estimation`
13. `skeleton_fitting`
14. `mesh_rigging`
15. `animation_preparation`
16. `scene_assembly`
17. `scene_update`
18. `completed`

### 3.13 Asset Manager & Storage Layout
- Storage root: `data/storage/`
  - `photos/`: Source photographs validated and sanitized.
  - `masks/`: 8-bit binary PNG instance masks named `{uuid}_{entity_id}_mask.png`.
  - `depth/`: Machine-readable float32 `.npy` depth maps and 8-bit grayscale `.png` visualizations.
  - `meshes/`: Binary GLTF human meshes named `{scene_id}_{entity_id}.glb`.
  - `scenes/`: Persisted JSON scene graphs named `{scene_id}.json`.
  - `projects/`: Persisted JSON project metadata and manifests named `{project_id}.json`.
  - `skeletons/`: JSON kinematic skeleton definitions named `{scene_id}_{entity_id}_skeleton.json`.
  - `rigs/`: Rig and skinning metadata JSON files.
  - `animations/`: Procedural animation clip JSON files.
- Path traversal protection (`StorageSecurityError`) on all file operations.

### 3.14 Interaction, Persistence & Real Export Engine (Phase 09)

Phase 09 transitions DRISHTI from viewing reconstructed scenes to actively manipulating, persisting, and exporting them:

- **Project Model & Asset Manifest (`ProjectManager`)**:
  - `Project` entity encapsulating stable ID, name, timestamps, scene references, workspace preferences, processing state, and structured asset manifest.
  - Asset manifest tracks all referenced files with relative paths, MIME types, and SHA-256 checksums without embedding giant binary data inside JSON metadata.
- **Atomic Local Persistence**:
  - All project and scene updates write to temporary `.tmp` files first, flush buffers, and atomically replace targets to eliminate corruption from interrupted writes.
  - Resilient project recovery: missing assets are flagged without corrupting valid scene entities or fabricating fake geometry.
- **Scene Interaction & Transform Constraints**:
  - Direct selection resolves to stable entity IDs.
  - Interactive transform editing: Position ($X, Y, Z$ in meters), Rotation (Euler $Y$), and Scale ($X, Y, Z$).
  - Server-side validation: asserts finite numbers, bounds positions within $[-50\text{m}, 50\text{m}]$, and restricts scale to $[0.05, 10.0]$.
  - Reliable Reset Transform restoring initial assembled depth-projected coordinates.
  - Workspace dirty tracking: unsaved changes indicator appears on edits; cleared upon saving.
- **Real Local Export Engine (`ExportEngine`)**:
  - **Single Entity Mesh (GLB)**: Validates geometry via `trimesh`, applies node transforms if requested, and exports standard binary GLTF. Rejects export if entity has no reconstructed mesh (honest reporting, no fake mannequins).
  - **Assembled Multi-Person Scene (GLB)**: Exports multi-person 3D scene preserving relative positions, rotations, and node transforms.
  - **Scene Metadata (JSON)**: Exports canonical scene metadata.
  - **Portable Project Archive (.zip)**: Bundles `project.json`, `scene.json`, `manifest.json`, and all referenced local assets into a self-contained ZIP archive.
  - **Anti-Zip-Slip Security**: Archive import strictly checks all entry paths, rejecting absolute paths, path traversal (`..`), and illegal characters. Reimports safely into isolated local project storage.
