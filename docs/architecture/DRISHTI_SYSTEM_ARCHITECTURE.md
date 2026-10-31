# DRISHTI: Complete System Architecture, Component Responsibility & Pipeline Audit

**System Version:** 1.2.0  
**Repository:** [Satishhhh989/DRISHTI](https://github.com/Satishhhh989/DRISHTI)  
**Classification:** Technical Architecture Specification & Systems Engineering Audit  
**Date:** October 31, 2026  

---

## 1. System Architecture Overview

DRISHTI is an on-device, local-first artificial intelligence platform designed to transform single monocular photographs containing multiple human subjects into metric, articulated, and spatially-composed 3D scenes.

### 1.1 High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Client["Web Client (Next.js 15 / React 19 / Three.js)"]
        UI["User Interface (TopBar / ContextualPanel)"]
        Store["State Stores (Zustand: SceneStore, UIStore)"]
        EventRed["SSE Event Reducer"]
        Viewport["ThreeViewport (React Three Fiber / Canvas)"]
        PhotoPlane["PhotoPlane (2D / 2.5D Reference)"]
        MeshLayer["HumanMeshLayer (GLTF / MeshStandardMaterial)"]
        SkelLayer["SkeletonOverlay (24 Joints / 23 Bones)"]
    end

    subgraph API["FastAPI Backend Service (Python 3.11, Uvicorn on 127.0.0.1:8000)"]
        Router["API v1 Router (/api/v1)"]
        HealthEp["/health & /version"]
        UploadEp["/projects & /scenes (Photo Ingestion)"]
        PipelineEp["/pipeline/analyze & /jobs (Orchestration)"]
        AssetEp["/assets (Static File & GLB Streaming)"]
        ExportEp["/export (Scene GLB, Entity GLB, ZIP)"]
    end

    subgraph PipelineEngine["Progressive Processing Pipeline (PipelineCoordinator)"]
        Stage1["1. Image Validation"]
        Stage2["2. Image Preparation"]
        Stage3["3. Person Detection (YOLOv8m-seg)"]
        Stage4["4. Entity Registration"]
        Stage5["5. Instance Segmentation (YOLOv8m-seg)"]
        Stage6["6. Mask Persistence"]
        Stage7["7. Depth Estimation (Depth Anything V2)"]
        Stage8["8. Depth Normalization & Stats"]
        Stage9["9. Spatial Photo Surface"]
        Stage10["10. 2D Pose Estimation (YOLOv8m-pose)"]
        Stage11["11. 3D Human Mesh Recovery (HMR 2.0 / 4DHumans)"]
        Stage12["12. Gradient Pose & Silhouette Fitting (SMPLPoseFitter)"]
        Stage13["13. 3D Skeleton Fitting (SMPL J_regressor)"]
        Stage14["14. Skeletal Rigging (SMPL LBS Weights)"]
        Stage15["15. Procedural Animation (Idle Breathing)"]
        Stage16["16. Multi-Person Scene Assembly (SceneGraph)"]
    end

    subgraph InferenceHW["Hardware Acceleration Subsystem"]
        DeviceMgr["DeviceManager (device.py)"]
        MPS["Apple Silicon MPS (Metal Performance Shaders)"]
        CUDA["NVIDIA CUDA (Fallback)"]
        CPU["CPU Tensor Core (Universal Fallback)"]
    end

    subgraph Storage["Local Storage Repository (data/storage/)"]
        Photos["photos/ (Original source images)"]
        Masks["masks/ (PNG alpha segmentation masks)"]
        Depth["depth/ (Raw NPY disparity + PNG previews)"]
        Meshes["meshes/ (PBR binary GLB models)"]
        Scenes["scenes/ (Authoritative JSON metadata)"]
        Projects["projects/ (Portable project state)"]
        Exports["exports/ (Packaged scene bundles)"]
    end

    UI --> Store
    Store --> Viewport
    Viewport --> PhotoPlane
    Viewport --> MeshLayer
    Viewport --> SkelLayer
    UI --> Router
    Router --> PipelineEp
    PipelineEp --> PipelineEngine
    PipelineEngine --> DeviceMgr
    DeviceMgr --> MPS
    DeviceMgr --> CUDA
    DeviceMgr --> CPU
    PipelineEngine --> Storage
    Storage --> AssetEp
    AssetEp --> MeshLayer
    EventRed --> Store
```

---

## 2. Component Responsibility Matrix

| Component | Repository Path | Responsibility | Input | Output | Dependencies | Runtime Location | Status | Known Limitations |
|---|---|---|---|---|---|---|---|---|
| **API Entrypoint** | [`services/api/app/main.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/main.py) | Application startup, CORS configuration, exception handling, and storage validation. | HTTP requests | JSON / Binary streams | FastAPI, Starlette | Localhost (`127.0.0.1:8000`) | **Active** | Bound to loopback interface by design. |
| **Pipeline Coordinator** | [`services/api/app/pipeline/coordinator.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/coordinator.py) | Master orchestrator of progressive 16-stage reconstruction pipeline; manages cancellation and event streaming. | `scene_id`, `job_id`, quality preset | Updated `Scene` record | All pipeline subsystems | Backend async task | **Active** | Sequential execution per job. |
| **Person Detector** | [`services/api/app/pipeline/detector.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/detector.py) | Multi-person localization; predicts 2D bounding boxes and confidence scores. | Image file path / RGB array | `List[DetectionResult]` | Ultralytics YOLOv8, PyTorch | MPS / CUDA / CPU | **Active** | Heavily occluded or tiny figures (<30px) may be missed. |
| **Instance Segmentor** | [`services/api/app/pipeline/segmentor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/segmentor.py) | Extracts individual binary and alpha pixel masks for each detected person. | Image + Bounding boxes | `List[SegmentationResult]` | Ultralytics YOLOv8-seg, OpenCV | MPS / CUDA / CPU | **Active** | Shared clothing boundaries can blend overlapping persons. |
| **Monocular Depth Estimator** | [`services/api/app/pipeline/depth.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/depth.py) | Predicts dense monocular scene disparity and computes per-person relative depth. | Image file path | `DepthEstimationResult` | Transformers / Depth Anything V2 | MPS / CUDA / CPU | **Active** | Relative proximity only; uncalibrated in metric meters. |
| **2D Pose Estimator** | [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L318-L414) | Detects 17 COCO 2D keypoints; maps keypoints 1-to-1 to entities via bipartite IoU matching. | Image + Entity boxes | `List[PoseEstimationResult]` | YOLOv8-pose | MPS / CUDA / CPU | **Active** | Hand fingers and facial detail are omitted in COCO format. |
| **HMR 2.0 Reconstructor** | [`services/api/app/pipeline/reconstructor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reconstructor.py#L1286-L1560) | Primary production 3D human mesh recovery engine; predicts SMPL pose, shape, and camera. | Image crop, bounding box, 2D pose | `HumanReconstructionResult` | PyTorch, ViT-Huge, SMPL LBS | MPS / CUDA / CPU | **Active** | Requires 4GB RAM/VRAM; predicts bare body surface (no clothes). |
| **HMR 2.0 Neural Model** | [`services/api/app/pipeline/hmr2_model.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/hmr2_model.py) | Pure-PyTorch implementation of 4DHumans architecture (ViT-Huge + Cross-Attention SMPL Transformer Decoder). | Normalized tensor `(1, 3, 256, 256)` | Pose 6D `(1, 144)`, Shape `(1, 10)`, Cam `(1, 3)` | PyTorch, `timm`, `einops` | MPS / CUDA / CPU | **Active** | 503/503 state dict keys matched to official checkpoint. |
| **SMPL Forward Layer** | [`services/api/app/pipeline/reconstructor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reconstructor.py#L244-L342) | Pure-PyTorch Linear Blend Skinning forward deformation for neutral SMPL body model. | Pose (24x6 rot6d), Betas (10) | Vertices `(6890, 3)` | PyTorch | MPS / CUDA / CPU | **Active** | Neutral adult body model only; genderless prior. |
| **SMPL Pose Fitter** | [`services/api/app/pipeline/reconstructor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reconstructor.py#L387-L838) | Gradient optimizer refining pose, shape, and camera against 2D keypoints and silhouette constraints. | Initial parameters, 2D keypoints, mask | Optimized parameters + metrics | PyTorch Adam optimizer | MPS / CUDA / CPU | **Active** | Optional refinement in HMR 2.0; primary optimizer in SPIN. |
| **Reprojection Diagnostic** | [`services/api/app/pipeline/reprojection.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reprojection.py) | Generates multi-layer visual diagnosis (crop, 2D keypoints, reprojected 3D skeleton, silhouette). | Image, 3D joints, camera, keypoints | PNG overlay bytes + error stats | OpenCV, NumPy | CPU | **Active** | Depends on weak-perspective camera scale. |
| **3D Skeleton Fitter** | [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L420-L536) | Regresses 24 3D joint locations from 6,890 SMPL vertices using official $J$-regressor matrix. | Mesh vertices `(6890, 3)` | Validated `SkeletonData` | NumPy | CPU | **Active** | Requires exact SMPL 6,890 vertex indexing. |
| **Mesh Rigging Engine** | [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L566-L650) | Assigns vertex skinning weights (top 4 influences per vertex) using official SMPL LBS matrix. | Mesh vertices, `SkeletonData` | Validated `RigData` | NumPy | CPU | **Active** | Top 4 normalized weights format. |
| **Procedural Animation Engine** | [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L670-L815) | Generates deterministic trigonometric animation curves (breathing, chest sway). | Skeleton ID, Entity ID | `List[AnimationClipData]` | Python math | CPU | **Active** | Synthetic procedural loops; not captured motion. |
| **Scene Assembly Engine** | [`services/api/app/pipeline/assembly.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/assembly.py) | Composes individual entities into unified 3D scene graph with camera, lights, and ground plane. | `Scene` with entities | `AssemblyResult` (SceneGraph + Report) | Python math | CPU | **Active** | Relative depth spread ($1.6\text{m}$) is heuristic. |
| **Export Engine** | [`services/api/app/export/engine.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/export/engine.py) | Serializes validated 3D models and scenes to GLB, JSON, and portable ZIP archives. | `scene_id`, `entity_id` | Binary GLB / ZIP bytes | `trimesh`, `zipfile` | CPU | **Active** | Strips textures if none are present. |
| **Model Manager** | [`services/api/app/pipeline/models.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/models.py) | Catalog, discovery, verification, and HTTP/HuggingFace downloads for model weights. | Model keys | Local file paths | `urllib`, `huggingface_hub` | CPU | **Active** | Relies on external network if weights are missing. |
| **Asset Manager** | [`services/api/app/assets/manager.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/assets/manager.py) | Content-addressable storage, path traversal protection, MIME validation. | Raw bytes, category | `AssetMetadata` | Standard library | CPU | **Active** | Local disk persistence only. |
| **Web 3D Viewport** | [`apps/web/components/viewport/ThreeViewport.tsx`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/viewport/ThreeViewport.tsx) | Renders interactive 3D scene, photographic reference plane, grid floor, and lights. | Scene state, mode selection | WebGL Canvas frame | React Three Fiber, Three.js | Browser WebGL | **Active** | Multi-Three.js import warnings in test environment. |
| **Human Mesh Layer** | [`apps/web/components/scene/HumanMeshLayer.tsx`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/HumanMeshLayer.tsx) | Fetches and renders GLB meshes, updates PBR clay materials, and draws 3D skeleton overlay. | Entity reconstructions, mode | 3D meshes & line segments | Three.js, React Three Fiber | Browser WebGL | **Active** | Clay shading only (no texture mapping). |
| **Representation Selector** | [`apps/web/components/controls/RepresentationSelector.tsx`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/controls/RepresentationSelector.tsx) | Navigation toolbar allowing switching between 9 visual representations with live readiness badges. | Scene availability state | Mode change actions | React | Browser DOM | **Active** | Disables unavailable modes honestly. |

---

## 3. Complete Photograph-to-3D Data Flow

The complete pipeline execution is traced below through every stage:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Web as Web Frontend (Next.js)
    participant API as FastAPI Router
    participant Coord as PipelineCoordinator
    participant Det as YOLOv8m-seg (Detector)
    participant Seg as YOLOv8m-seg (Segmentor)
    participant Depth as Depth Anything V2
    participant Pose as YOLOv8m-pose
    participant HMR as HMR 2.0 (4DHumans)
    participant Fit as SMPLPoseFitter
    participant Rig as Skeleton & Rigging Engine
    participant Asm as SceneAssemblyEngine
    participant Store as Asset & Scene Storage

    User->>Web: Drops Photograph (JPEG/PNG)
    Web->>API: POST /api/v1/projects (Upload Image)
    API->>Store: Persist original photo to disk
    API-->>Web: Return scene_id & photo metadata
    Web->>API: POST /api/v1/pipeline/analyze (scene_id)
    API->>Coord: execute_pipeline(scene_id, job_id)
    
    rect rgb(30, 41, 59)
        note over Coord, Det: Stages 1-4: Detection & Entity Creation
        Coord->>Det: detect_people(photo_path, conf=0.35)
        Det-->>Coord: Return 5 BoundingBox2D results
        Coord->>Store: Save initial SceneEntity records
        Coord-->>Web: SSE: stage_completed (person_detection)
    end

    rect rgb(15, 23, 42)
        note over Coord, Seg: Stages 5-6: Instance Segmentation
        Coord->>Seg: segment_people(photo_path, detections)
        Seg-->>Coord: 5 binary alpha mask arrays
        Coord->>Store: Register mask PNG assets
        Coord-->>Web: SSE: asset_ready (segmentation_mask)
    end

    rect rgb(30, 41, 59)
        note over Coord, Depth: Stages 7-9: Monocular Depth & Spatial Photo
        Coord->>Depth: estimate_depth(photo_path)
        Depth-->>Coord: Relative disparity (H, W) float32
        Coord->>Store: Save depth.npy & depth_vis.png
        Coord->>Coord: compute_entity_depth_stats(depth, masks)
        Coord-->>Web: SSE: asset_ready (spatial_photo)
    end

    rect rgb(15, 23, 42)
        note over Coord, Pose: Stage 10: 2D Keypoint Extraction
        Coord->>Pose: estimate_poses(image, entity_boxes)
        Pose->>Pose: Bipartite IoU matching (>0.15)
        Pose-->>Coord: 5 1-to-1 matched 17-keypoint poses
        Coord-->>Web: SSE: stage_completed (pose_estimation)
    end

    rect rgb(30, 41, 59)
        note over Coord, Fit: Stages 11-12: HMR 2.0 Reconstruction & Fitting
        loop For Each Person Entity (1 to 5)
            Coord->>HMR: reconstruct(crop, box, mask, pose)
            HMR->>HMR: ViT-Huge backbone inference on MPS (~250ms)
            HMR->>HMR: Cross-Attention SMPL Transformer Decoder
            HMR->>Fit: evaluate(pose_6d, betas, cam, keypoints, mask)
            Fit-->>HMR: Reprojection error, silhouette IoU, stance
            HMR->>HMR: Forward SMPL LBS (6,890 vertices, 13,776 faces)
            HMR->>HMR: Export binary GLB with neutral clay PBR
            HMR-->>Coord: HumanReconstructionResult (GLB, vertices, metrics)
            Coord->>Store: Register mesh GLB asset
            Coord-->>Web: SSE: asset_ready (mesh)
        end
    end

    rect rgb(15, 23, 42)
        note over Coord, Rig: Stages 13-15: Skeleton Fitting, Rigging & Animation
        loop For Each Reconstructed Entity
            Coord->>Rig: build_skeleton(entity, vertices, pose)
            Rig-->>Coord: 24-joint SkeletonData
            Coord->>Rig: rig_mesh(entity, skeleton, vertices)
            Rig-->>Coord: RigData (SMPL LBS top-4 weights)
            Coord->>Rig: generate_clips(entity_id, skeleton_id)
            Rig-->>Coord: AnimationClipData (idle_breathing)
        end
        Coord-->>Web: SSE: stage_completed (animation_preparation)
    end

    rect rgb(30, 41, 59)
        note over Coord, Asm: Stage 16: Multi-Person Scene Assembly
        Coord->>Asm: assemble(scene)
        Asm->>Asm: Calculate horizontal spacing, relative Z depth, scene graph
        Asm-->>Coord: AssemblyResult (SceneGraph + SceneAssemblyReport)
        Coord->>Store: Update authoritative Scene JSON
        Coord-->>Web: SSE: job_completed (100% progress)
    end

    Web->>API: GET /api/v1/scenes/{scene_id}
    API-->>Web: Complete Scene JSON
    Web->>API: GET /api/v1/assets/meshes/{filename}.glb
    API-->>Web: Stream GLB binary assets
    Web->>Web: Render 3D Scene in Three.js Viewport
```

---

## 4. Stage-by-Stage Implementation Details

### Stage 1: Image Validation
* **Source:** [`services/api/app/assets/validation.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/assets/validation.py)
* **Function:** `validate_image_file(content: bytes) -> ImageValidationResult`
* **Input:** Raw binary file bytes.
* **Output:** Image metadata (dimensions, format, channel depth, aspect ratio).
* **Device:** CPU (Pillow/PIL).
* **Tests:** [`services/api/tests/test_image_validation.py`](file:///Users/satish/Desktop/DRISHTI/services/api/tests/test_image_validation.py).

### Stage 2: Image Preparation
* **Source:** [`services/api/app/pipeline/coordinator.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/coordinator.py#L237-L260)
* **Function:** In-line coordinate reference setup.
* **Input:** Validated image file path.
* **Output:** Instantiated `Scene` with initial status `processing`.
* **Device:** CPU.

### Stage 3: Person Detection
* **Source:** [`services/api/app/pipeline/detector.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/detector.py#L76-L140)
* **Class:** `YoloPersonDetector.detect_people(photo_path, conf=0.35)`
* **Input:** RGB image array $H \times W \times 3$.
* **Output:** `List[DetectionResult]` with normalized coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max}) \in [0, 1]^4$.
* **Model:** Ultralytics YOLOv8m-seg (`yolov8m-seg.pt`, 52.4 MB).
* **Device:** Apple Silicon `mps` or CUDA.
* **Fallback:** `MockPersonDetector` if weights are absent.

### Stage 4: Progressive Entity Creation
* **Source:** [`services/api/app/pipeline/coordinator.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/coordinator.py#L294-L373)
* **Function:** Immediate persistence of `SceneEntity` records to allow the web frontend to display detection bounding boxes while segmentation is running.

### Stage 5: Instance Segmentation
* **Source:** [`services/api/app/pipeline/segmentor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/segmentor.py#L82-L165)
* **Class:** `YoloPersonSegmentor.segment_people(photo_path, detections)`
* **Input:** Full image path + detection bounding boxes.
* **Output:** `List[SegmentationResult]` with $H \times W$ uint8 binary masks and compressed PNG bytes.
* **Model:** Ultralytics YOLOv8m-seg (`yolov8m-seg.pt`).
* **Device:** Apple Silicon `mps` or CUDA.

### Stage 6: Mask Persistence
* **Source:** [`services/api/app/assets/manager.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/assets/manager.py#L110-L150)
* **Function:** `asset_manager.register_pipeline_asset(...)`
* **Input:** PNG bytes.
* **Output:** Stored PNG in `data/storage/masks/{entity_id}_mask.png`.

### Stages 7 & 8: Depth Estimation & Normalization
* **Source:** [`services/api/app/pipeline/depth.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/depth.py#L140-L260)
* **Class:** `DepthAnythingV2Estimator.estimate_depth(photo_path)`
* **Input:** Image path.
* **Output:** `DepthEstimationResult` with raw disparity float32 array, normalized proximity array $[0, 1]$, and Turbo colormapped PNG bytes.
* **Model:** Depth Anything V2 Small (`depth-anything-v2-small`, 99.2 MB).
* **Device:** Apple Silicon `mps` or CUDA.
* **Fallback:** `MockDepthEstimator` generating synthetic linear gradient depth.

### Stage 9: Depth Association
* **Source:** [`services/api/app/pipeline/depth.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/depth.py#L280-L365)
* **Function:** `compute_entity_depth_stats(...)`
* **Input:** Normalized depth array + person segmentation mask.
* **Output:** `EntityDepthStats` (median, mean, min, max, standard deviation of proximity values under mask).

### Stage 10: 2D Pose Estimation
* **Source:** [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L318-L414)
* **Class:** `YoloPoseEstimator.estimate_poses(image, entity_boxes)`
* **Input:** BGR image array + list of entity bounding boxes.
* **Output:** `List[PoseEstimationResult]` guaranteed 1-to-1 matched to entity boxes via bipartite Hungarian IoU matching ($>0.15$).
* **Model:** Ultralytics YOLOv8m-pose (`yolov8m-pose.pt`, 50.8 MB).
* **Device:** Apple Silicon `mps` or CUDA.

### Stage 11: Human 3D Reconstruction
* **Source:** [`services/api/app/pipeline/reconstructor.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reconstructor.py#L1286-L1560)
* **Class:** `Hmr2HumanReconstructor.reconstruct(...)`
* **Input:** RGB image, bounding box, alpha mask, 2D pose result.
* **Output:** `HumanReconstructionResult` containing 6,890 vertices, 13,776 faces, weak-perspective camera params, metric dimensions, and GLB bytes.
* **Model:** HMR 2.0 / 4DHumans (`epoch=35-step=1000000.ckpt`, 2.58 GB).
* **Device:** Apple Silicon `mps` (~250ms per person).

### Stage 12: Reprojection & Quality Evaluation
* **Source:** [`services/api/app/pipeline/reprojection.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/reprojection.py)
* **Function:** `generate_reprojection_overlay(...)`
* **Input:** Source image, 3D joints, camera scale/translation, 2D keypoints.
* **Output:** 4-layer diagnostic overlay image (crop + 2D pose + 3D projected skeleton + mask boundary) and mean reprojection error in pixels.

### Stage 13: 3D Skeleton Fitting
* **Source:** [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L420-L536)
* **Class:** `SkeletonFitter.build_skeleton(...)`
* **Input:** Mesh vertices `(6890, 3)` + SMPL $J$-regressor matrix `(24, 6890)`.
* **Output:** 24-joint `SkeletonData` structure with verified parent-child topology and 23 bone definitions.

### Stage 14: Mesh Rigging
* **Source:** [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L566-L650)
* **Class:** `RiggingEngine.rig_mesh(...)`
* **Input:** Mesh vertices, skeleton, and SMPL skinning weights matrix `(6890, 24)`.
* **Output:** `RigData` with top-4 bone indices and normalized skinning weights per vertex.

### Stage 15: Procedural Motion Generation
* **Source:** [`services/api/app/pipeline/motion.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/motion.py#L670-L815)
* **Class:** `ProceduralAnimationEngine.generate_clips_for_entity(...)`
* **Input:** Entity ID, Skeleton ID.
* **Output:** 30 FPS deterministic animation tracks for idle breathing (`AnimationClipData`).

### Stage 16: Multi-Person Scene Assembly
* **Source:** [`services/api/app/pipeline/assembly.py`](file:///Users/satish/Desktop/DRISHTI/services/api/app/pipeline/assembly.py)
* **Class:** `SceneAssemblyEngine.assemble(scene)`
* **Input:** Unassembled `Scene` record with all entity meshes and depths.
* **Output:** Fully assembled `SceneGraph` with root, camera, lights, reference plane at $Z = -0.6\text{m}$, and positioned entity nodes.

---

## 5. AI Model and Algorithm Inventory

| Model / Subsystem | Variant / Checkpoint | Checkpoint Path | Size | Upstream Source | License | Active / Fallback |
|---|---|---|---|---|---|---|
| **YOLOv8 Detection & Segmentation** | `yolov8m-seg.pt` | `data/models/yolov8m-seg.pt` | 52.4 MB | Ultralytics YOLOv8 | AGPL-3.0 | **Active Primary** |
| **YOLOv8 Pose Estimation** | `yolov8m-pose.pt` | `data/models/yolov8m-pose.pt` | 50.8 MB | Ultralytics YOLOv8 | AGPL-3.0 | **Active Primary** |
| **Depth Anything V2** | `small` (`model.safetensors`) | `data/models/depth-anything-v2-small/model.safetensors` | 99.2 MB | ByteDance / TikTok | Apache-2.0 | **Active Primary** |
| **HMR 2.0 (4DHumans)** | `ViT-Huge` (`epoch=35-step=1000000.ckpt`) | `data/models/hmr2/epoch=35-step=1000000.ckpt` | 2.58 GB | Goel et al., CVPR 2023 | CC-BY-NC 4.0 | **Active Primary** |
| **SMPL Neutral Model** | `SMPL_N_model.npz` | `data/models/human_reconstruction/SMPL_N_model.npz` | 38.6 MB | Max Planck Institute (MPI) | Non-Commercial Research | **Active Core** |
| **SPIN Checkpoint** | `spin_model_checkpoint.pth.tar` | `data/models/human_reconstruction/spin_model_checkpoint.pth.tar` | 108.2 MB | Kolotouros et al., ICCV 2019 | MIT | Standby Fallback |
| **YOLOv8 Nano Detection** | `yolov8n-seg.pt` | `data/models/yolov8n-seg.pt` | 7.1 MB | Ultralytics | AGPL-3.0 | Preview Fallback |
| **YOLOv8 Nano Pose** | `yolov8n-pose.pt` | `data/models/yolov8n-pose.pt` | 6.8 MB | Ultralytics | AGPL-3.0 | Preview Fallback |

---

## 6. Frontend and Visualization Architecture

The DRISHTI web frontend is implemented with **Next.js 15.5.28, React 19, Zustand, and React Three Fiber (@react-three/fiber with Three.js r183)**.

### 6.1 Visualization Modes Breakdown

| Mode ID | Display Label | Rendering Component | Scene Data Consumed | Representation Type | Underlying Technology |
|---|---|---|---|---|---|
| `original` | **Original** | [`PhotoPlane`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/PhotoPlane.tsx) | `localPhotoUrl`, dimensions | 2D Plane | Three.js `PlaneGeometry` + `MeshBasicMaterial` |
| `detection` | **Detection** | `PhotoPlane` + [`DetectionBox`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/DetectionBox.tsx) | `entity.detection.boundingBox` | 2D Vector Overlay | HTML5 Canvas / SVG overlay projected onto plane |
| `segmentation` | **Segmentation** | `PhotoPlane` + [`SegmentationMaskLayer`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/SegmentationMaskLayer.tsx) | `entity.segmentation.maskUri` | Colorized Mask Overlay | PNG Alpha Canvas compositing with unique entity hue |
| `depth` | **Depth Map** | `PhotoPlane` | `scene.depth.visualizationUri` | Colormapped 2D Plane | Turbo colormap PNG rendered on plane surface |
| `spatial_photo` | **Spatial Photo** | [`SpatialPhotoSurface`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/SpatialPhotoSurface.tsx) | `scene.depth.depthUri`, photo URL | Displaced 2.5D Mesh | Custom GLSL vertex displacement shader ($128 \times 128$) |
| `wireframe` | **Wireframe** | [`HumanMeshLayer`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/HumanMeshLayer.tsx) | `entity.reconstruction.meshUrl` | 3D Wireframe Mesh | Three.js `MeshStandardMaterial({ wireframe: true })` |
| `solid3d` | **3D Solid** | `HumanMeshLayer` | `entity.reconstruction.meshUrl` | 3D Solid Surface | Three.js `MeshStandardMaterial({ roughness: 0.55 })` clay |
| `articulated` | **Articulated** | `HumanMeshLayer` + [`SkeletonOverlay`](file:///Users/satish/Desktop/DRISHTI/apps/web/components/scene/HumanMeshLayer.tsx#L40-L106) | `entity.skeleton`, clips | Rigged Character + Bones | `LineSegments` (23 bones) + `SphereGeometry` (24 joints) |
| `combined` | **Combined** | `PhotoPlane` + `HumanMeshLayer` | Reference plane + all entity meshes | Hybrid Spatial Composition | Background photo plane ($Z = -0.6\text{m}$) + foreground meshes |

---

## 7. Coordinate Systems & Geometry Pipeline

```mermaid
flowchart LR
    A["Image Space<br/>Top-Left (0,0) to (1,1)<br/>Normalized UV"] --> B["OpenCV Camera Frame<br/>+X Right, +Y Down, +Z Forward<br/>Weak Perspective (s, tx, ty)"]
    B --> C["SMPL Model Space<br/>Pelvis Origin (0,0,0)<br/>+X Right, +Y Up, +Z Forward"]
    C --> D["Scene World Space<br/>Right-Handed Metric (m)<br/>+X Right, +Y Up, +Z Out"]
    D --> E["Three.js Viewport<br/>Camera at (0, 1.2, 3.8)<br/>Reference Plane at Z=-0.6"]
```

### 7.1 Coordinate Transformations
1. **Image Space to Crop Frame:**
   The image is padded by 20% and letterboxed to a square:
   $$\text{crop\_box} = [x_{\min} - 0.2\Delta x, y_{\min} - 0.2\Delta y, x_{\max} + 0.2\Delta x, y_{\max} + 0.2\Delta y]$$
2. **OpenCV Camera Frame to Three.js Frame:**
   Because PyTorch/OpenCV conventions treat $+Y$ as downward and $+Z$ as away into depth, the raw reconstructed SMPL vertices are inverted:
   $$Y_{\text{Three.js}} = -Y_{\text{SMPL}}, \quad Z_{\text{Three.js}} = -Z_{\text{SMPL}}$$
3. **World Coordinate Placement:**
   Entities are placed such that their 3D pelvis projects directly to their 2D image pelvis position:
   $$X_{\text{world}} = (c_x - 0.5) \times W_{\text{plane}} - (\text{scale} \times \text{pelvis}_{3D, x})$$
   $$Y_{\text{world}} = (1.0 - c_y) \times H_{\text{plane}} - (\text{scale} \times \text{pelvis}_{3D, y})$$
   $$Z_{\text{world}} = (\text{proximity} - 0.5) \times 1.6\text{m}$$

---

## 8. Test Suite Verification & Coverage Matrix

| Test Suite | Path | Cases | Passed | Hardware / Mock | Quality Assertions |
|---|---|---|---|---|---|
| **Reconstruction Quality** | `test_reconstruction_quality.py` | 31 | **31** | **Real MPS Inference** | Standing/seated height discrimination, reprojection error, 6D rotation math. |
| **Pipeline Real Execution** | `test_pipeline_real.py` | 8 | **8** | Real Models | End-to-end pipeline run from image file to scene graph. |
| **Scene Assembly Engine** | `test_scene_assembly_engine.py` | 7 | **7** | Unit / Pure Logic | Horizontal placement ordering, depth proximity, occlusion ambiguity warnings. |
| **Mesh Rigging Engine** | `test_mesh_rigging.py` | 6 | **6** | Unit / SMPL Data | Skinning weight normalization ($L_1 = 1.0$), bone index bounds. |
| **Export System** | `test_export_system.py` | 7 | **7** | Unit / File IO | GLB binary structure, zip-slip path traversal protection. |
| **Image Validation** | `test_image_validation.py` | 6 | **6** | Unit / Bytes | Empty file, corrupted file, excessive dimensions, format whitelist. |
| **Depth Estimation** | `test_depth.py` | 9 | **9** | Unit / Math | NaN/Inf handling, disparity normalization, person mask stats. |
| **Security & Hardening** | `test_security_hardening.py` | 9 | **9** | Unit / HTTP | Security headers, zip bombs, decompression bombs, loopback CORS. |
| **Web Frontend Tests** | `apps/web/tests/*.test.tsx` | 39 | **39** | Vitest / jsdom | 11 test suites verifying representation switching, store updates, and Three.js canvas. |
| **Total Test Count** | **All Suites** | **182** | **182** | **100% Green** | Zero failures across backend and frontend. |

