# DRISHTI Troubleshooting & Diagnostics Guide

**Product Identity**: DRISHTI Engine, DRISHTI Studio, DRISHTI Scene, DRISHTI Motion  
**Version**: `1.0.0` (Production Release)  
**Architecture**: Local-First Python FastAPI Backend + Next.js 15 / Three.js Frontend  

---

## 1. Network & Port Conflicts

### Issue: Port 8000 or Port 3000 is already in use
**Symptom**: 
- Backend fails to start with `[Errno 48] Address already in use`
- Frontend fails to start with `Error: listen EADDRINUSE: address already in use :::3000`

**Diagnosis**:
Identify which process is currently occupying the ports:
```bash
# Check port 8000 (Backend API)
lsof -i :8000

# Check port 3000 (Frontend Studio)
lsof -i :3000
```

**Resolution**:
1. Terminate conflicting processes:
   ```bash
   kill -9 <PID>
   ```
2. Or change the port in `.env` and `apps/web/package.json`:
   - In `.env`: `PORT=8001`
   - In `apps/web/package.json`: `"dev": "next dev -p 3001"`

---

## 2. Neural Model Checkpoints & Inference

### Issue: Model checkpoints are missing from `data/models/`
**Symptom**:
- Log message: `WARNING: Model checkpoint not found. Falling back to deterministic mock pipeline.`
- Studio displays badge: `MOCK INFERENCE ACTIVE`

**Diagnosis**:
Run the model integrity verification script:
```bash
PYTHONPATH=services/api .venv/bin/python -c "
from pathlib import Path
for p in ['yolov8n-seg.pt', 'yolov8n-pose.pt', 'depth-anything-v2-small/model.safetensors', 'human_reconstruction/spin_model_checkpoint.pth.tar']:
    print(p, 'EXISTS' if (Path('data/models') / p).exists() else 'MISSING')
"
```

**Resolution**:
- If weights are missing, place the required model files into `data/models/` as detailed in [`docs/MODEL_SETUP.md`](file:///Users/satish/Desktop/DRISHTI/docs/MODEL_SETUP.md).
- If you intend to run in lightweight offline mock mode for UI development or automated testing, you can safely ignore this warning; DRISHTI's mock pipelines generate valid deterministic geometries and scene events.

---

## 3. Hardware Acceleration & PyTorch Devices

### Issue: Apple Silicon Metal (`mps`) falls back to CPU
**Symptom**:
- Log message indicates `device: cpu` on an M-series Mac.

**Diagnosis**:
Check whether PyTorch detects Metal acceleration:
```bash
PYTHONPATH=services/api .venv/bin/python -c "
import torch
print('MPS Available:', torch.backends.mps.is_available())
print('MPS Built:', torch.backends.mps.is_built())
"
```

**Resolution**:
1. Ensure you are running a native ARM64 Python interpreter, not an x86_64 translation running under Rosetta 2:
   ```bash
   python -c "import platform; print(platform.machine())"  # Must output 'arm64'
   ```
2. Enable MPS operator fallback for any experimental PyTorch operations:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

### Issue: NVIDIA CUDA is not detected on Linux/Windows WSL2
**Symptom**:
- `device: cpu` on a system with an NVIDIA GPU.

**Diagnosis**:
```bash
PYTHONPATH=services/api .venv/bin/python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())
"
```

**Resolution**:
1. Check that `nvidia-smi` functions correctly in your terminal.
2. Ensure you installed PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 4. Image Upload & Validation Errors

### Issue: Upload fails with `400 Bad Request: Image pixel count exceeds maximum safe limit`
**Symptom**:
- High-resolution panoramic or medium-format photos are rejected during upload.

**Explanation**:
To prevent decompression bomb denial-of-service vulnerabilities, DRISHTI enforces an upper limit of 25 megapixels (`MAX_IMAGE_PIXELS = 25,000,000`) and dimension limits of $8192 \times 8192$ px.

**Resolution**:
If you need to process larger legitimate photographs, adjust the setting in `.env`:
```ini
MAX_IMAGE_PIXELS=50000000
```
Then restart the backend API.

### Issue: Upload fails with `400 Bad Request: Animated or multi-frame images are not supported`
**Symptom**:
- Uploading an animated GIF or APNG fails.

**Explanation**:
DRISHTI reconstructs 3D spatial scenes from single photographic frames. Multi-frame animations are rejected by security policy to avoid uncontrolled CPU/GPU memory consumption.

**Resolution**:
Convert the desired single frame of the animation to a standard JPEG or PNG prior to upload.

---

## 5. Project Import & Archive Security Errors

### Issue: Project import fails with `400 Bad Request: Path traversal detected in zip archive`
**Symptom**:
- Uploading a `.zip` project fails with an Anti-Zip-Slip security error.

**Explanation**:
The imported ZIP archive contains entries with parent directory navigation (`..`), absolute root paths (`/`), or Windows drive prefixes. This is a severe security vulnerability that DRISHTI blocks immediately.

**Resolution**:
Only import archives created via DRISHTI Studio's native **Export > Portable Project Archive (.zip)** feature, which guarantees safe relative paths.

### Issue: Project import fails with `400 Bad Request: Zip bomb detected`
**Symptom**:
- Archive is rejected with entry count or uncompressed size limits exceeded.

**Explanation**:
Imported archives are capped at 250 files and 500 MB total uncompressed size to prevent zip bomb attacks.

---

## 6. Real-Time Streaming & SSE Connection

### Issue: Pipeline Inspector stalls or does not update during processing
**Symptom**:
- Upload succeeds, but progress bars remain static in DRISHTI Studio.

**Diagnosis**:
Open the browser developer tools (F12) $\to$ **Network** tab $\to$ filter for `stream`:
- Check if `GET /api/v1/jobs/{job_id}/stream` returned HTTP 200 with `text/event-stream`.
- Look for CORS or proxy connection terminations.

**Resolution**:
1. DRISHTI's frontend includes an **automatic polling fallback** that polls `GET /api/v1/jobs/{job_id}` every 1000ms if SSE drops.
2. If using a local corporate VPN or proxy, ensure `127.0.0.1` and `localhost` are added to your bypass list.
3. Check backend uvicorn terminal logs for unhandled pipeline exceptions.

