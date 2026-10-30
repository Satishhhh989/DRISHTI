# Ten-Phase Project Roadmap

**DRISHTI** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | **Image Intelligence & Person Segmentation** | **COMPLETED** | Local multi-person detection (YOLOv8), instance segmentation (8-bit binary PNG masks), hardware-accelerated inference (Apple Silicon MPS / CUDA), 3D viewport overlays, entity inspector, sequenced event streaming. |
| **Phase 03** | **Depth Estimation & Spatial Photography** | **COMPLETED** | Local monocular depth estimation (Depth Anything V2), canonical relative proximity $[0, 1]$, robust percentile normalization, per-person depth statistics, 3D vertex displacement mesh, bounded parallax camera controls. |
| **Phase 04** | **Background Inpainting & Separation** | **PLANNED** | High-resolution alpha matting boundary refinement, depth-guided hole filling, and background inpainting to create clean separate image layers. |
| **Phase 05** | **Multiplane Spatial Photography** | Planned | Multiplane image (MPI) layers and depth displacement mesh creating multi-layer 2.5D parallax spatial photos. |
| **Phase 06** | **3D Human Mesh Reconstruction** | Planned | Volumetric human mesh generation producing genuine 3D polygon geometry and wireframes. |
| **Phase 07** | **Texture & Material Synthesis** | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 08** | **Human Skeleton & Rigging** | Planned | 3D kinematic skeleton fitting and skinning weight attachment for human subjects. |
| **Phase 09** | **Scene Assembly & Animation** | Planned | Interactive 3D scene compositing with independent character animation controls. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Project persistence, GLTF/USDZ export, performance optimization, and packaging. |

---

## Current Status: Phase 03 Completed

Phase 03 delivers genuine monocular depth estimation and interactive spatial photography:
- **Local Pretrained Depth Model**: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`, ~99.2MB) executing locally without external API dependencies.
- **Hardware Acceleration**: Automatic tensor placement and inference on Apple Silicon MPS (`mps`), NVIDIA CUDA (`cuda`), or multithreaded CPU.
- **Canonical Relative Proximity**: Standardized proximity representation $P \in [0.0, 1.0]$ where $1.0$ is the nearest foreground and $0.0$ is the farthest background.
- **Robust Percentile Normalization**: 2nd and 98th percentile clamping eliminating outliers, sensor artifacts, and pathological cases (uniform depth, zero variance, NaNs/Infs).
- **Per-Person Depth Association**: Evaluates depth distribution per detected entity using instance segmentation masks, extracting median proximity, min/max bounds, depth variation, and coverage.
- **Asset Persistence**: Lossless float32 machine-readable `.npy` depth arrays and 8-bit single-channel grayscale PNG visualizations stored in `data/storage/depth/`.
- **Interactive 3D Spatial Photo Surface**: High-resolution $128 \times 128$ subdivided mesh with `MeshStandardMaterial` vertex displacement mapped to depth texture.
- **Controlled Parallax Exploration**: Bounded orbit controls (azimuth $\pm 18^\circ$, polar elevation $75^\circ \text{--} 105^\circ$, distance $1.8 \text{--} 4.5\text{m}$, damping $0.06$) ensuring realistic depth perception while preventing disocclusion tears.
- **Depth Map Inspection Mode**: Real-time switching between clean photo, detection boxes, segmentation overlays, 8-bit grayscale depth visualization, and 3D spatial photo.
- **Interactive Inspector**: Depth strength tuning slider ($0.00 \text{--} 0.25$), camera reset action, and per-person depth statistics.
- **12-Stage Streaming Pipeline**: Sequenced progress events with non-fatal error isolation and catch-up reconciliation.
- **Automated Verification**: 43 backend pytest tests, 19 vitest tests, and 7 Playwright browser tests passing (69 total).

---

## Next Mission: Phase 04 — Background Inpainting & Separation

Phase 04 will focus on:
1. High-resolution matting refinement along fine boundaries (hair, clothing edges).
2. Foreground person extraction and layer composition.
3. Background inpainting to fill occlusion voids behind detected persons.
4. Independent foreground and background plane separation in 3D space.
