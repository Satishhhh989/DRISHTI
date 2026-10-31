# DRISHTI Model Setup & Installation Guide

## 1. Overview & Directory Structure

DRISHTI is built on a **local-first, offline architecture**. All neural network weights and parametric model templates reside locally in `data/models/`. Once downloaded, DRISHTI operates with zero outbound internet network traffic.

```
data/models/
├── yolo/
│   ├── yolov8m-seg.pt            # Person detection & instance segmentation
│   └── yolov8m-pose.pt           # 2D human pose estimation (17 keypoints)
├── depth/
│   └── depth_anything_v2_vits.pth # Depth Anything V2 metric depth
├── human_reconstruction/         # Production 3D Human Recovery Engine (SPIN)
│   ├── spin_model_checkpoint.pt  # SPIN iterative regression checkpoint (~130 MB)
│   ├── SMPL_N_model.npz          # Canonical SMPL neutral body model (6,890 verts)
│   └── smpl_mean_params.npz      # Neutral body shape and camera prior
└── hmr2/                         # Optional High-Fidelity 3D Human Backend (HMR 2.0)
    ├── epoch=35-step=1000000.ckpt # 4DHumans / HMR 2.0 ViT checkpoint (~2.1 GB)
    └── smpl/                     # SMPL neutral body model directory
```

---

## 2. Production Models (Standard Setup)

The standard DRISHTI configuration provides high source-image fidelity, seated-pose resolution, and real-time execution (~25–45 ms/person on Apple Silicon MPS):

### 2.1 Automated Download Script
Run the automated model fetch script from the workspace root:

```bash
bash scripts/download_models.sh
```

### 2.2 Manual Verification
Verify that required model weights exist:

```bash
ls -lh data/models/human_reconstruction/
```
Expected files:
- `spin_model_checkpoint.pt` (~130 MB)
- `SMPL_N_model.npz` (~38 MB)
- `smpl_mean_params.npz` (~2 KB)

---

## 3. Optional High-Fidelity Backend: HMR 2.0 (4DHumans)

DRISHTI includes a modular adapter `Hmr2HumanReconstructor` for **HMR 2.0 (4DHumans)**, which leverages a Vision Transformer (ViT) backbone and cross-attention SMPL decoder.

### 3.1 Hardware Requirements
- **Apple Silicon**: M1 Pro/Max, M2, M3, or M4 with at least 16 GB unified memory.
- **NVIDIA GPU**: RTX 3070, 3080, 4080, 4090, or A100 with at least 8 GB VRAM.
- **Disk Space**: ~2.5 GB additional free storage.

### 3.2 Obtaining HMR 2.0 Weights
1. Create the HMR 2.0 model directory:
   ```bash
   mkdir -p data/models/hmr2/
   ```

2. Download the official 4DHumans / HMR 2.0 checkpoint:
   ```bash
   # From the official 4DHumans release:
   wget https://people.eecs.berkeley.edu/~shubham-goel/projects/4dhumans/hmr2_data.tar.gz
   tar -xvf hmr2_data.tar.gz -C data/models/hmr2/
   ```

3. Ensure the checkpoint is placed at:
   ```
   data/models/hmr2/epoch=35-step=1000000.ckpt
   ```

4. Link or copy the SMPL neutral body model into the HMR 2.0 directory:
   ```bash
   cp data/models/human_reconstruction/SMPL_N_model.npz data/models/hmr2/SMPL_NEUTRAL.npz
   ```

---

## 4. Backend Selection & Graceful Fallback Sequence

DRISHTI dynamically detects available weights through `ReconstructionBackendRegistry`.

### 4.1 Fallback Hierarchy
1. If `backend="hmr2"` is requested:
   - If `data/models/hmr2/epoch=35-step=1000000.ckpt` is present, executes via **HMR 2.0**.
   - If checkpoint is absent, logs:
     `"HMR 2.0 weights not found; executing through SPIN production reconstructor."`
     and automatically processes the entity through the **SPIN production engine**.
2. If `backend="spin"` (default):
   - Executes via **SPIN + SMPLPoseFitter**.
   - If SMPL files are absent, falls back to **MockHumanReconstructor** (for CI and mock testing).

### 4.2 Quality Presets
The human reconstruction pipeline supports three runtime quality presets:
- **`preview`**: 15 optimization steps, 128x128 silhouette grid, lowest latency (~15 ms).
- **`high`** (Default): 45 optimization steps, 128x128 silhouette grid, balanced fidelity and speed (~35 ms).
- **`maximum`**: 80 optimization steps, 256x256 silhouette grid, highest contour accuracy (~75 ms).

