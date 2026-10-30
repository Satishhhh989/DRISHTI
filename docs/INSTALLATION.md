# DRISHTI Installation & Setup Guide

**Product Identity**: DRISHTI Engine, DRISHTI Studio, DRISHTI Scene, DRISHTI Motion  
**Version**: `1.0.0` (Production Release)  
**Tagline**: *From Pixels to Perception.*  
**Architecture**: 100% Local-First, Zero Cloud Dependencies, Zero Telemetry  

---

## 1. System Requirements

### Supported Operating Systems
- **macOS**: Sonoma 14+ or Sequoia 15+ on Apple Silicon (M1/M2/M3/M4) recommended for native Metal Performance Shaders (`mps`) acceleration. macOS on Intel x86_64 supported with CPU execution.
- **Linux**: Ubuntu 22.04 LTS / 24.04 LTS, Debian 12+, Arch Linux, Fedora 39+. NVIDIA CUDA 11.8+ or 12.1+ supported for GPU acceleration.
- **Windows**: Windows 11 with WSL2 (Windows Subsystem for Linux 2, Ubuntu 22.04/24.04). Direct native Windows execution requires WSL2 for optimal PyTorch and POSIX path handling.

### Hardware Prerequisites
| Resource | Minimum Specification | Recommended Specification |
|---|---|---|
| **CPU** | 4 cores (x86_64 or ARM64) | 8+ cores (Apple Silicon or modern x86_64) |
| **System Memory (RAM)** | 8 GB | 16 GB+ unified memory or DDR5 |
| **GPU / Accelerator** | None (CPU fallback active) | Apple Silicon MPS or NVIDIA GPU with $\ge 6\text{ GB}$ VRAM |
| **Disk Storage** | 3 GB free disk space | 10 GB free disk space (models + workspace) |

### Runtime Tooling
- **Python**: `3.11.x` (recommended `3.11.7`+; Python 3.12+ also compatible)
- **Node.js**: `>= 20.0.0` (tested with Node `v20.x`, `v22.x`, and `v26.x`)
- **Package Manager**: `pnpm >= 9.0.0` (recommended `pnpm 11.x`)
- **Git**: `2.30+`

---

## 2. Quick-Start Setup (macOS / Linux)

DRISHTI provides an automated bootstrap script that verifies tooling, configures the Python virtual environment, installs monorepo dependencies, and checks for model checkpoints:

```bash
# 1. Clone the repository
git clone https://github.com/drishti-engine/drishti.git
cd drishti

# 2. Run the automated setup script
bash scripts/setup.sh

# 3. Launch the complete local stack
bash scripts/dev.sh
```

Once running:
- **DRISHTI Studio Frontend**: [http://localhost:3000](http://localhost:3000)
- **DRISHTI Engine REST API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3. Step-by-Step Manual Setup

If you prefer full control over environment configuration, follow these step-by-step instructions.

### Step 3.1: Repository & Environment Configuration
```bash
git clone https://github.com/drishti-engine/drishti.git
cd drishti

# Copy the example environment configuration
cp .env.example .env
```

Ensure `.env` contains the required loopback defaults:
```ini
ENVIRONMENT=production
HOST=127.0.0.1
PORT=8000
DATA_DIR=data
STORAGE_DIR=data/storage
MODELS_DIR=data/models
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
MAX_UPLOAD_SIZE_MB=25
MAX_IMAGE_PIXELS=25000000
```

### Step 3.2: Python Virtual Environment Setup
DRISHTI isolates its backend inside a local `.venv`:

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install backend dependencies
pip install -r services/api/requirements.txt
```

Verify backend installation:
```bash
PYTHONPATH=services/api .venv/bin/python -c "
import torch
from app.pipeline.device import DeviceManager
device = DeviceManager.get_device()
print(f'PyTorch {torch.__version__} initialized on device: {device}')
"
```

### Step 3.3: Node.js Monorepo Dependencies
DRISHTI uses a `pnpm` monorepo containing:
- `apps/web`: Next.js 15, React 19, Three.js, React Three Fiber, Zustand.
- `packages/scene-schema`: Canonical TypeScript scene models and representation registry.
- `packages/shared-types`: Server-Sent Events, processing contracts, and API DTOs.

Install monorepo dependencies and link packages:
```bash
pnpm install
```

Build packages and frontend:
```bash
# Build TypeScript packages and Next.js production bundle
pnpm --filter @photo-to-3d/scene-schema build
pnpm --filter @photo-to-3d/shared-types build
pnpm --filter @photo-to-3d/web build
```

---

## 4. Model Checkpoint Installation

DRISHTI requires four neural network checkpoints located in `data/models/`:

```
data/models/
├── yolov8n-seg.pt                     # Person Detection & Instance Segmentation (7.1 MB)
├── yolov8n-pose.pt                    # 2D Pose Estimation (6.8 MB)
├── depth-anything-v2-small/
│   ├── config.json                    # Depth Anything V2 architecture config
│   └── model.safetensors              # Depth Anything V2 model weights (95 MB)
└── human_reconstruction/
    ├── spin_model_checkpoint.pth.tar  # SPIN ResNet-50 + Parameter Regressor (103 MB)
    ├── SMPL_N_model.npz               # Neutral SMPL Human Body Model (84 MB)
    └── smpl_mean_params.npz           # SMPL Mean Pose & Shape Priors (1.3 KB)
```

Refer to [`docs/MODEL_SETUP.md`](file:///Users/satish/Desktop/DRISHTI/docs/MODEL_SETUP.md) for full download instructions, hash verification, licensing information, and architecture details.

---

## 5. Running the Application

### Option A: Unified Launcher
```bash
bash scripts/dev.sh
```
This script launches both the FastAPI backend and Next.js development server simultaneously and monitors both processes.

### Option B: Independent Processes
In Terminal 1 (Backend Engine):
```bash
bash scripts/run_api.sh
```
Or manually:
```bash
source .venv/bin/activate
PYTHONPATH=services/api uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In Terminal 2 (Frontend Studio):
```bash
pnpm --filter @photo-to-3d/web dev
```

---

## 6. Verification & Automated Testing

Verify your complete installation by executing the 182-test automated verification suite:

```bash
# Execute all 5 automated test suites
bash scripts/test.sh
```

Expected output:
```
==================================================
Running All PHOTO → 3D ENGINE Automated Tests
==================================================

>>> [1/5] Testing @photo-to-3d/scene-schema (Vitest)...
✓ 18 passed in 4 files

>>> [2/5] Testing @photo-to-3d/shared-types (Vitest)...
✓ 2 passed in 1 file

>>> [3/5] Testing @photo-to-3d/web (Vitest)...
✓ 37 passed in 11 files

>>> [4/5] Testing API Backend (Pytest)...
✓ 112 passed in 4.61s

>>> [5/5] Testing Browser E2E Suite (Playwright)...
✓ 13 passed in 4.1s

==================================================
All DRISHTI automated tests passed successfully!
==================================================
```

---

## 7. Windows (WSL2) Setup Notes

1. **Install WSL2**: Open PowerShell as Administrator and run `wsl --install -d Ubuntu-24.04`.
2. **GPU Pass-Through**: Install the latest NVIDIA Windows display drivers (WSL2 includes direct CUDA pass-through).
3. **Clone Inside WSL2**: Always clone the repository into the Linux filesystem (e.g. `~/projects/drishti`), not `/mnt/c/`, to ensure native POSIX I/O performance.
4. **Follow Section 3**: Run the standard Linux installation steps inside your WSL2 bash terminal.

