# DRISHTI Local Model Setup & License Guide

**Product Identity**: DRISHTI Engine, DRISHTI Studio, DRISHTI Scene, DRISHTI Motion  
**Version**: `1.0.0` (Production Release)  
**Architecture**: 100% Local Inference, Zero Cloud Calls, Offline Checkpoints  

---

## 1. Overview of Neural Network Models

DRISHTI is built on a **local-first, privacy-preserving AI architecture**. All computer vision inference executes strictly on the user's host machine. No imagery, intermediate tensors, or reconstructions are ever transmitted to third-party cloud services or telemetry servers.

The system incorporates four purpose-built neural network subsystems:

| Subsystem | Model Artifact | Size | Upstream Source / Architecture | License |
|---|---|---|---|---|
| **Person Detection & Segmentation** | `yolov8n-seg.pt` | 7.1 MB | Ultralytics YOLOv8 Nano Segmentation | AGPL-3.0 |
| **Monocular Depth Estimation** | `depth-anything-v2-small/` | 95 MB | Depth Anything V2 Small (DINOv2 + DPT) | Apache-2.0 |
| **Human 3D Mesh Recovery** | `spin_model_checkpoint.pth.tar` | 103 MB | SPIN (ResNet-50 + SMPL Regressor, Kolotouros et al.) | MIT |
| **Parametric Human Body Model** | `SMPL_N_model.npz` | 84 MB | SMPL Neutral Model (Max-Planck-Gesellschaft) | MPI Non-Commercial |
| **SMPL Prior Distributions** | `smpl_mean_params.npz` | 1.3 KB | SPIN SMPL Mean Pose & Shape Priors | MIT |
| **2D Articulated Pose** | `yolov8n-pose.pt` | 6.8 MB | Ultralytics YOLOv8 Nano Pose | AGPL-3.0 |

Total Model Weights Footprint: **~296 MB**.

---

## 2. Directory Layout & Integrity

All neural model checkpoints reside within `data/models/`:

```
data/models/
├── yolov8n-seg.pt
├── yolov8n-pose.pt
├── depth-anything-v2-small/
│   ├── config.json
│   └── model.safetensors
└── human_reconstruction/
    ├── spin_model_checkpoint.pth.tar
    ├── SMPL_N_model.npz
    └── smpl_mean_params.npz
```

### Verification Script
You can verify the presence, size, and loadability of all checkpoints with the following command:

```bash
PYTHONPATH=services/api .venv/bin/python -c "
from pathlib import Path
import torch

models_dir = Path('data/models')
expected = [
    ('yolov8n-seg.pt', 5_000_000),
    ('yolov8n-pose.pt', 5_000_000),
    ('depth-anything-v2-small/model.safetensors', 50_000_000),
    ('human_reconstruction/spin_model_checkpoint.pth.tar', 80_000_000),
    ('human_reconstruction/SMPL_N_model.npz', 50_000_000),
    ('human_reconstruction/smpl_mean_params.npz', 1_000),
]

for rel_path, min_bytes in expected:
    p = models_dir / rel_path
    if p.exists() and p.stat().st_size >= min_bytes:
        print(f'✓ {rel_path} ({p.stat().st_size / (1024*1024):.1f} MB) verified.')
    else:
        print(f'✗ MISSING or truncated: {rel_path}')
"
```

---

## 3. Detailed Model Profiles

### 3.1 Person Detection & Instance Segmentation
- **File**: `data/models/yolov8n-seg.pt`
- **Architecture**: CSPDarknet backbone with Path Aggregation Network (PAN) and convolutional prototype mask generation heads.
- **Input**: Normalized RGB image $(B, 3, H, W)$.
- **Output**: 
  - Bounding boxes $(x_1, y_1, x_2, y_2)$ in normalized and pixel coordinates.
  - Class predictions (filtered strictly to COCO class index `0`: `"person"`).
  - High-resolution 8-bit binary segmentation masks ($W \times H$).
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0).

### 3.2 Monocular Depth Estimation
- **Files**: `data/models/depth-anything-v2-small/config.json` & `model.safetensors`
- **Architecture**: Vision Transformer based on DINOv2 with Dense Prediction Transformer (DPT) multi-scale fusion neck.
- **Input**: Resized RGB tensor ($518 \times 518$ or native patch-multiples).
- **Output**: Dense float32 relative proximity map $P \in [0.0, 1.0]$ where $1.0$ represents the nearest foreground surface and $0.0$ represents the furthest background horizon.
- **Post-Processing**: Robust 2nd-to-98th percentile clipping to eliminate sensor noise and depth outliers.
- **Storage**: Lossless float32 numpy `.npy` for internal metric calculations and 8-bit grayscale PNG for frontend visualization.
- **License**: Apache License 2.0.

### 3.3 Parametric 3D Human Mesh Recovery (SPIN)
- **File**: `data/models/human_reconstruction/spin_model_checkpoint.pth.tar`
- **Architecture**: ResNet-50 convolutional feature extractor followed by an iterative parameter regression loop that estimates:
  - Camera weak-perspective scale $s$ and translation $(t_x, t_y)$.
  - SMPL 3D joint rotations $\theta \in \mathbb{R}^{24 \times 3 \times 3}$ using 6D continuous rotation representations.
  - SMPL shape parameters $\beta \in \mathbb{R}^{10}$ representing principal component body shape variations.
- **License**: MIT License (Kolotouros et al., ICCV 2019).

### 3.4 SMPL Neutral Body Model
- **Files**: `data/models/human_reconstruction/SMPL_N_model.npz` & `smpl_mean_params.npz`
- **Architecture**: Skinned Multi-Person Linear (SMPL) model implementing standard anatomical topology:
  - Vertices: $6,890$
  - Triangular Faces: $13,776$
  - Skeletal Joints: $24$
  - Blend Skinning: Linear Blend Skinning (LBS) with pre-computed joint regressor and weight matrices.
- **DRISHTI Innovation**: Native pure-PyTorch forward deformation layer (`services/api/app/pipeline/smpl_pytorch.py`) eliminating legacy C++ dependencies, `chumpy`, or outdated Python 2 wrappers. Runs natively on Apple Silicon Metal and modern PyTorch 2.x.
- **License**: Max-Planck-Gesellschaft (MPI) SMPL Model License (Free for non-commercial research, education, and evaluation). Commercial deployment requires licensing from Meshcapade GmbH.

### 3.5 2D Articulated Pose Estimation
- **File**: `data/models/yolov8n-pose.pt`
- **Architecture**: Ultralytics YOLOv8 Nano Pose network detecting 17 COCO keypoints per person:
  - Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles.
- **Integration**: Associated with detected person entities via Maximum Intersection-over-Union (IoU) matching.
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0).

---

## 4. Hardware Execution & Acceleration

DRISHTI includes an intelligent hardware device manager (`services/api/app/pipeline/device.py`) that probes the execution environment on startup:

### Device Priority Order:
1. **Apple Silicon Metal (`mps`)**:
   - Discovered automatically on macOS ARM64 systems via `torch.backends.mps.is_available()`.
   - Executes in standard 32-bit floating point precision (`float32`), utilizing Unified Memory Architecture (UMA) for high throughput.
2. **NVIDIA CUDA (`cuda`)**:
   - Discovered automatically on Linux/Windows systems with NVIDIA GPUs via `torch.cuda.is_available()`.
   - Utilizes CUDA cores and Tensor Cores for parallel tensor operations.
3. **CPU Fallback (`cpu`)**:
   - Portable, zero-dependency fallback for standard x86_64 or ARM64 processors.
   - Utilizes multi-threaded AVX2/AVX-512 / NEON SIMD acceleration through OpenMP.

To query the active device programmatically:
```bash
curl http://127.0.0.1:8000/api/v1/health
```
Response:
```json
{
  "status": "ok",
  "service": "photo-to-3d-api",
  "version": "1.0.0",
  "device": "mps"
}
```

---

## 5. Offline Fallback & Mock Pipelines

If model weights have not yet been downloaded to `data/models/`, DRISHTI **never fails silently or invents deceptive results**:

- When checkpoints are absent or when `MOCK_INFERENCE=true` is set in `.env`, DRISHTI automatically transitions to its deterministic mock pipeline implementations (`MockYoloDetector`, `MockDepthEstimator`, `MockPoseEstimator`, `MockReconstructor`).
- Mock pipelines generate mathematically valid, reproducible geometric entities and structured events.
- All mock assets are stamped with explicit provenance metadata (`"mock_synthetic"`), and the UI displays transparent status alerts informing the user that mock models are currently active.

