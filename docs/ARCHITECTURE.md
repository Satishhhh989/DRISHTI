# DRISHTI System Architecture

## Overview

**DRISHTI** is a local-first AI-powered spatial reconstruction engine that transforms ordinary photographs containing multiple people into interactive, animated 3D scenes. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute strictly on the user's local machine.

Phase 06 implements the **Multi-Person Scene Assembly Engine**, transforming individually reconstructed people into a unified, interactive, persistent 3D scene. DRISHTI preserves the original photograph's spatial composition (left, center, right horizontal ordering and relative depth layers) while building a complete, versioned 3D scene graph (`SceneGraph`). Entities whose mesh reconstruction is incomplete remain represented in-scene alongside successfully reconstructed subjects. The scene is saved locally to disk (`data/storage/scenes/{scene_id}.json`) and rendered with camera controls (orbit, pan, zoom, fit-to-scene, focus person) and per-person visibility toggling.

```
+-----------------------------------------------------------------------------------------+
|                                     Browser Client                                      |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  |                            Workspace Shell (Next.js)                              |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation / Vis Modes     |  | Pipeline &     |  |  |
|  |  |  (Fit Scene, Reset) |  | (Availability-Driven Selector) |  | Inspector Pane |  |  |
|  |  |                     |  | (Orig / Detect / Seg / Depth / |  | (Stages, Mesh, |  |  |
|  |  |                     |  |  Spatial Photo / Wire / Solid) |  |  SceneAssembly)|  |  |
|  |  +---------------------+  +--------------------------------+  +----------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
|  |                        Three.js Viewport (React Three Fiber)                      |  |
|  |  +--------------------+  +----------------------------------+  +---------------+  |  |
|  |  | Perspective Camera |  | Upright PhotoPlane /             |  | HumanMeshLayer|  |  |
|  |  | & Lighting         |  | SpatialPhotoSurface (128x128)   |  | - Wireframe   |  |  |
|  |  | - Fit-to-Scene     |  |   + Vertex Displacement Map      |  | - Solid Clay  |  |  |
|  |  | - Focus-on-Person  |  |   + 3D Detection Boxes & Masks   |  | - In-Scene Box|  |  |
|  |  | - Orbit / Pan/Zoom |  |   + Reference Photo Plane        |  | - Raycast Sel |  |  |
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
|  |                       |  | (/scenes, /save)     |  | (/photos, /masks, /meshes)   |  |
|  +-----------------------+  +----------------------+  +------------------------------+  |
|  | Event Emitter (Bounded 500-event FIFO)          |  | Job State Tracker            |  |
|  | Real-Time SSE Stream (/jobs/{id}/stream)        |  | Cancellation Handler         |  |
|  +-------------------------------------------------+  +------------------------------+  |
|  | Progressive Multi-Stage Intelligence Pipeline (Phase 01 - 06)                     |  |
|  |  - DeviceManager: Apple Silicon MPS / NVIDIA CUDA / CPU fallback                  |  |
|  |  - ModelRegistry: Local cache in data/models/                                     |  |
|  |      * yolov8n-seg.pt (6.7MB)                                                     |  |
|  |      * depth-anything-v2-small/ (99.2MB safetensors)                             |  |
|  |      * spin_model_checkpoint.pth.tar (103MB) + SMPL_N_model.npz (88MB)          |  |
|  |  - Progressive Entity Commit: Entities committed immediately after detection     |  |
|  |  - Per-Person Mask Commit: Individual masks committed and broadcast upon save     |  |
|  |  - Progressive Depth Commit: Depth map and spatial photo committed upon save      |  |
|  |  - Human 3D Reconstruction: SPIN + pure PyTorch SMPL LBS forward deformation      |  |
|  |  - Scene Assembly Engine: Deterministic multi-person composition & SceneGraph    |  |
|  |      * Model vs Scene Space: sceneTransform = placement x orient x modelNorm      |  |
|  |      * Horizontal & depth alignment preserving composition and relative proximity  |  |
|  |      * Occlusion & overlap analysis with single-view ambiguity reporting          |  |
|  |      * Partial reconstruction tolerance: unreconstructed entities remain in-scene |  |
|  |  - Local Disk Persistence: JSON scene graph saved to data/storage/scenes/         |  |
|  |  - Cancellation Checkpoints: Clean early termination preserving prior assets     |  |
|  |  - AnalysisPipelineCoordinator: 14 sequential stages with live event stream       |  |
|  +-----------------------------------------------------------------------------------+  |
|  | Local Filesystem Storage (photos/, masks/, depth/, meshes/, scenes/, skeletons/)  |  |
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

## 2. Frontend Architecture (Phase 06)

### 2.1 UI Layer
The UI provides an interactive workspace focused on the 3D viewport:
- **TopBar**: Identity (`DRISHTI`), phase badge (`Phase 06`), backend connectivity status indicator, grid floor toggle, "Fit Scene" button to frame all assembled subjects, camera reset button, and file upload trigger.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid ($Y = 0$), and camera controls.
- **Representation & Visualization Selector**: Available modes: `Original`, `Detection`, `Segmentation`, `Depth Map`, `Spatial Photo`, `Wireframe`, `Solid 3D`. Wireframe and Solid 3D representations render the fully assembled multi-person 3D scene.
- **HumanMeshLayer (`HumanMeshLayer.tsx`)**: Renders the complete assembled multi-person scene:
  - *Reconstructed Subjects*: Loaded from binary GLTF meshes and positioned at their computed scene-space transforms.
  - *Unreconstructed Subjects*: Displayed honestly as distinct in-scene volumetric wireframe markers with status indicators, preventing subjects from vanishing.
  - *Per-Entity Visibility*: Toggling visibility hides the subject without deleting the entity or its transform.
  - *Wireframe Mode*: Highlights genuine polygon edge topology in technical cyan/white lines.
  - *Solid 3D Mode*: Renders neutral studio clay PBR material (`#d0d0d8`, roughness 0.65, metalness 0.05).
  - *Entity Selection*: Raycasting allows clicking on any person in the 3D scene to highlight them and inspect their placement metadata.
- **Progressive Processing Inspector (`ContextualPanel.tsx`)**:
  - *Pipeline Inspector*: Real-time stage rows including `Image Validation`, `Person Detection`, `Segmentation`, `Scene Depth`, `Spatial Photo`, `3D Human Mesh`, and `Scene Assembly`.
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
  - `skeletons/`: JSON kinematic skeleton definitions named `{scene_id}_{entity_id}_skeleton.json`.
  - `rigs/`: Rig and skinning metadata JSON files.
  - `animations/`: Procedural animation clip JSON files.
- Path traversal protection (`StorageSecurityError`) on all file operations.
