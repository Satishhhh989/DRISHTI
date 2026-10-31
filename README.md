# DRISHTI: Photo to 3D Spatial Reconstruction Engine

DRISHTI is a local-first spatial reconstruction application that reconstructs multi-person 3D scenes from a single photograph. It combines deep learning models for person detection, instance segmentation, monocular metric depth estimation, human mesh recovery, and skeletal kinematics into a coordinated local pipeline.

The application runs entirely on local hardware, communicates over loopback network interfaces, and requires no external cloud vision APIs.

---

## Architecture and Subsystems

DRISHTI is structured as a monorepo consisting of a FastAPI backend service, a Next.js 15 web client, and shared TypeScript schema packages.

```
DRISHTI/
├── apps/
│   └── web/                   # Next.js 15, React 19, Three.js, React Three Fiber, Zustand
├── services/
│   └── api/                   # Python 3.11, FastAPI, PyTorch, Ultralytics, Trimesh
├── packages/
│   ├── scene-schema/          # Canonical scene schema and representation registry
│   └── shared-types/          # Shared TypeScript DTOs and event models
├── data/
│   ├── models/                # Local neural network checkpoints
│   └── storage/               # Local SQLite/JSON persistence and asset storage
├── docs/                      # Architectural specifications and design records
└── scripts/                   # Local setup, run, and test scripts
```

### Pipeline Flow

1. **Ingestion and Image Validation**: Validates image format (JPEG, PNG, WebP), dimensions, and metadata. Enforces limits to protect against decompression bombs. Maps the source photograph to a 3D image plane in metric space.
2. **Person Detection (YOLOv8m)**: Detects human candidates, generates 2D bounding boxes in normalized image coordinates, filters non-person classes, and assigns stable entity identifiers.
3. **Instance Segmentation (YOLOv8m-seg)**: Generates 8-bit binary silhouette masks for each detected person. Calculates pixel area coverage and boundaries.
4. **Scene Depth Estimation (Depth Anything V2 Small)**: Computes a dense relative depth field across the full image. Disparity maps are normalized using percentile boundaries (2nd to 98th) to eliminate outlier artifacts. Outputs machine-readable 32-bit floating point arrays and visualization textures.
5. **Spatial Photo Relief**: Projects the photograph onto a 128x128 vertex grid displaced along the Z axis by the estimated depth map.
6. **Human Mesh Recovery (HMR 2.0 / 4DHumans)**: Extracts cropped and letterboxed image patches for each person. A Vision Transformer (ViT) predicts 3D human pose, joint rotations (6D continuous representations), shape parameters (SMPL beta coefficients), and weak perspective camera parameters.
7. **SMPL Forward Kinematics (PyTorch LBS)**: Evaluates canonical SMPL linear blend skinning entirely in PyTorch to produce articulated humanoid meshes with 6,890 vertices and 13,776 faces. No external C++ or legacy dependencies are required.
8. **Surface Appearance & Projective Colors**: Samples RGB colors from the source photograph and maps them onto visible front-facing mesh vertices. Non-visible or obscured vertices receive neutral studio clay shading.
9. **2D/3D Skeleton Fitting**: Derives a canonical 24-joint human skeleton from SMPL joint regression matrices. Normalizes bone skinning weights to the top four influences per vertex ($L_1 = 1.0$) for hardware shader compatibility.
10. **Multi-Person Spatial Scene Assembly**: Grounds each character to the horizontal floor plane ($Y = 0$), preserves lateral photographic ordering on the X axis, and stages subjects along the Z axis based on relative depth measurements. Overlapping subjects are evaluated for occlusion boundaries.
11. **Procedural Motion Subsystem**: Evaluates character posture (standing vs. seated) and binds stance-appropriate skeletal animation tracks (e.g., seated idle breathing with locked hip/knee flexion, or standing idle cycles). All synthesized movement is explicitly flagged with procedural provenance.

---

## User Experience and Viewport Controls

### Reconstruction Trigger

Reconstruction is opt-in. When an image is uploaded or dropped into the viewport:
- The photograph is displayed cleanly on the 3D photo plane in original mode.
- The camera remains stationary.
- No automated transformations or perspective shifts occur until the user clicks **Reconstruct Scene** in the top toolbar.
- Once completed, the button allows replaying the sequence without re-running backend model inference.

### Choreographed Transformation Sequence

When **Reconstruct Scene** is clicked, the viewport executes a nine-stage sequence synchronized with real processing events:

- **Stage 1 (Photograph Introduction)**: Subdued push-in framing the source photograph.
- **Stage 2 (Person Detection)**: Sequential focus on detected candidates, highlighting bounding boxes and temporary identification tags with actual confidence scores.
- **Stage 3 (Segmentation)**: Silhouette contour reveal per person using the neural segmentation masks.
- **Stage 4 (Depth Map)**: Horizontal sweep across the photograph revealing Depth Anything V2's disparity field.
- **Stage 5 (Spatial Photo)**: Controlled 18-degree perspective tilt revealing 3D depth displacement.
- **Stage 6 (Individual Wireframe Extraction)**: Each person is focused sequentially. Their reconstructed 3D mesh lifts forward ($Z: 0 \to 0.65\text{m}$) from the photo plane with genuine 6,890-vertex SMPL topology visible.
- **Stage 7 (3D Solid Mesh)**: Wireframe edges dissolve into solid volumetric surfaces with photograph-projected vertex colors.
- **Stage 8 (Articulation & Rigging)**: Illuminates the 24 internal joints and 23 bones inside translucent humanoid bodies.
- **Stage 9 (Final Combined Scene)**: Characters step forward into their room depth positions in front of the reference photograph, handing control directly over to OrbitControls.

### Manual Inspection Modes

Users can toggle between representation modes at any time using the mode selector in the top bar:
- **Original**: Flat photograph plane.
- **Detection**: 2D bounding boxes and confidence tags.
- **Segmentation**: Color-coded instance mask overlays.
- **Depth Map**: Normalized grayscale depth field.
- **Spatial Photo**: Interactive depth relief mesh with bounded parallax controls.
- **Wireframe**: Reconstructed SMPL polygon edge topology.
- **3D Solid**: Matte studio clay or projective vertex-colored volumetric meshes.
- **Articulated**: Character meshes with internal joint and bone visualization.
- **Combined**: Assembled 3D human meshes positioned in front of the photographic reference plane.

### Contextual Inspector

The panel on the right provides:
- Live pipeline stage status and duration metrics.
- Active entity metadata (bounding box coordinates, 3D dimensions, vertex/face count, posture description, reprojection error).
- Transform adjustment tools: Position ($X, Y, Z$ in meters), Rotation ($Y$ yaw), and Scale.
- Procedural animation controls: Play, Pause, Reset Pose, playback speed selectors (0.5x, 1.0x, 1.5x, 2.0x), and clip selector.
- Export actions for individual meshes (GLB), assembled scenes (GLB), metadata (JSON), and complete portable project archives (ZIP).

---

## Hardware Acceleration

The backend automatically detects and leverages available hardware:

- **Apple Silicon (macOS)**: Uses Metal Performance Shaders (`mps`). Model inference, SMPL blend skinning, and depth estimation execute directly on Apple unified memory.
- **NVIDIA GPU (Linux / Windows WSL2)**: Uses CUDA acceleration (`cuda:0`).
- **CPU Fallback**: Vectorized multi-threaded PyTorch fallback if no GPU acceleration is present.

---

## Prerequisites

Ensure the following tools are installed on your host system:

- **Python**: Version 3.11 (Python 3.12 or newer is not supported by several computer vision dependencies).
- **Node.js**: Version 18.18 or newer (Node 20 or Node 22 LTS recommended).
- **Package Manager**: `pnpm` (version 9+) or `npm` (version 10+).
- **Git**: Version 2.30+.
- **uv** (Optional but recommended): High-speed Python package and environment manager.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Satishhhh989/DRISHTI.git
cd DRISHTI
```

### 2. Backend Environment Setup

Create a Python 3.11 virtual environment and install the required dependencies:

Using standard virtual environment:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r services/api/requirements.txt
```

Or using `uv`:
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r services/api/requirements.txt
```

### 3. Frontend & Shared Packages Setup

Install workspace dependencies and build the shared packages:

```bash
pnpm install
pnpm --filter @photo-to-3d/scene-schema build
pnpm --filter @photo-to-3d/shared-types build
```

If using standard `npm`:
```bash
npm install
npm --prefix packages/scene-schema run build
npm --prefix packages/shared-types run build
```

### 4. Storage Directory Initialization

Ensure local asset storage directories exist:

```bash
mkdir -p data/storage/{photos,masks,depth,meshes,textures,skeletons,exports,scenes}
```

### 5. Check Model Checkpoints

Verify that pretrained model checkpoints are present in `data/models/`:

- `data/models/yolov8m-seg.pt` (YOLOv8 instance segmentation)
- `data/models/yolov8m-pose.pt` (YOLOv8 2D pose estimation)
- `data/models/depth-anything-v2-small/model.safetensors` (Depth Anything V2 Small)
- `data/models/hmr2/epoch=35-step=1000000.ckpt` (HMR 2.0 4DHumans checkpoint)
- `data/models/human_reconstruction/SMPL_N_model.npz` (SMPL Neutral body model)

---

## Running Locally

### Option A: Automated Launch Script

The root workspace includes a runner script that starts both the backend API and frontend dev server simultaneously:

```bash
bash scripts/dev.sh
```

Press `Ctrl+C` to terminate both services cleanly.

### Option B: Manual Service Startup

To run the services independently in separate terminal windows:

#### Terminal 1: Backend API Service

```bash
cd /path/to/DRISHTI
source .venv/bin/activate
export PYTHONPATH="services/api"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API service will start on `http://127.0.0.1:8000`.
- API Health Check: `http://127.0.0.1:8000/api/v1/health`
- OpenAPI Documentation: `http://127.0.0.1:8000/docs`

#### Terminal 2: Frontend Web Application

```bash
cd /path/to/DRISHTI
pnpm --filter @photo-to-3d/web dev
# or: npm --prefix apps/web run dev
```

The web application will compile and launch on `http://localhost:3000`.

---

## Testing and Verification

The repository maintains an automated test suite across backend and frontend codebases.

### Run Backend Pytest Suite

```bash
bash scripts/test_backend.sh
```
Or directly via pytest:
```bash
source .venv/bin/activate
PYTHONPATH="services/api" pytest services/api/tests -v
```

This verifies 146 tests covering:
- Image validation and anti-decompression bomb protection.
- Monocular depth normalization and percentile scaling.
- SMPL 6D rotation conversions and kinematic skeleton validation.
- HMR 2.0 neural inference and pose fidelity.
- Multi-person scene assembly, ground alignment, and occlusion handling.
- Asset storage, atomic writes, and anti-path-traversal defenses.

### Run Frontend Vitest Suite

```bash
npm --prefix apps/web test -- --run
# or: pnpm --filter @photo-to-3d/web test -- --run
```

This verifies 44 tests covering:
- Complete removal of the bottom timeline panel from the viewport.
- TopBar "Reconstruct Scene" execution and replay handling.
- Individual wireframe extraction state and progressive lift logic.
- Idempotent SSE event reduction into client scene state.
- Depth inspection and procedural animation playback controls.

---

## Security and Privacy Protections

- **Strict Loopback Binding**: The FastAPI backend explicitly binds to `127.0.0.1` and will not bind to external interfaces (`0.0.0.0`) by default.
- **Restricted CORS**: Cross-origin requests are limited strictly to local development origins (`http://localhost:3000`, `http://127.0.0.1:3000`).
- **Decompression Bomb Protection**: Pillow's `MAX_IMAGE_PIXELS` is capped at 25,000,000, and uploads exceeding 25 MB or 8192x8192 pixels are rejected.
- **Anti-Zip-Slip Traversal Prevention**: Project archive imports strictly inspect file paths, rejecting any archive member with directory traversal sequences (`..`), leading slashes, or drive letters.
- **Atomic Persistence**: Scene and project files are written to isolated `.tmp` files and committed via atomic renames to prevent partial-write corruption during unexpected shutdowns.
- **No External Telemetry**: The application transmits zero data outside your local machine.

---

## API Reference Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service status, version, and storage availability. |
| `GET` | `/api/v1/pipeline/hardware` | Hardware discovery (device type, memory, acceleration status). |
| `POST` | `/api/v1/photos/upload` | Upload and sanitize a photograph; registers a new scene. |
| `POST` | `/api/v1/pipeline/analyze` | Initiates asynchronous multi-stage reconstruction job. |
| `GET` | `/api/v1/jobs/{job_id}/stream` | Server-Sent Events (SSE) push stream for job progress. |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cooperatively cancels an active analysis job. |
| `GET` | `/api/v1/scenes` | Lists all locally persisted scenes. |
| `GET` | `/api/v1/scenes/{scene_id}` | Retrieves full canonical scene JSON with entity graph. |
| `PATCH`| `/api/v1/scenes/{scene_id}/entities/{entity_id}` | Updates entity transforms, visibility, or animation settings. |
| `GET` | `/api/v1/export/scene/{scene_id}/glb` | Exports the assembled 3D scene as a binary GLB file. |
| `GET` | `/api/v1/export/project/{project_id}/archive`| Exports the complete project and binary assets as a ZIP archive. |
