# Ten-Phase Project Roadmap

**DRISHTI** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | **Image Intelligence & Person Segmentation** | **COMPLETED** | Local multi-person detection (YOLOv8), instance segmentation (8-bit binary PNG masks), hardware-accelerated inference (Apple Silicon MPS / CUDA), 3D viewport overlays, entity inspector, sequenced event streaming. |
| **Phase 03** | **Person & Background Separation** | **PLANNED** | High-resolution alpha matting boundary refinement, depth-guided hole filling, and background inpainting to create clean separate image layers. |
| **Phase 04** | **Scene Depth Estimation** | Planned | Monocular metric and relative depth estimation generating depth maps for spatial positioning. |
| **Phase 05** | **Spatial Photo Reconstruction** | Planned | Multiplane image (MPI) layers and depth displacement mesh creating 2.5D parallax spatial photos. |
| **Phase 06** | **3D Human Mesh Reconstruction** | Planned | Volumetric human mesh generation producing genuine 3D polygon geometry and wireframes. |
| **Phase 07** | **Texture & Material Synthesis** | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 08** | **Human Skeleton & Rigging** | Planned | 3D kinematic skeleton fitting and skinning weight attachment for human subjects. |
| **Phase 09** | **Scene Assembly & Animation** | Planned | Interactive 3D scene compositing with independent character animation controls. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Project persistence, GLTF/USDZ export, performance optimization, and packaging. |

---

## Current Status: Phase 02 Completed

Phase 02 delivers genuine image intelligence:
- **Local Neural Inference**: Pretrained YOLOv8 nano segmentation model running locally with zero cloud API dependencies.
- **Hardware Acceleration**: Automatic detection and hardware acceleration on Apple Silicon MPS (`mps`), NVIDIA CUDA (`cuda`), and CPU fallback.
- **Multi-Person Detection**: Detects all visible people, generating normalized coordinates, integer pixel bounding boxes, confidence scores, and stable entity IDs (`person-001`, `person-002`, ...).
- **Instance Segmentation Masks**: Generates individual 8-bit single-channel binary PNG masks ($255$ foreground, $0$ background) bilinearly interpolated to the native photograph resolution, stored safely in `data/storage/masks/`.
- **Interactive 3D Viewport**: Seamless visualization switching between `Original`, `Detection` (3D line bounding boxes with name chips), and `Segmentation` (semi-transparent colored alpha overlays), perfectly locked to the photograph plane during 3D camera navigation.
- **Interactive Inspector**: Inspect individual detected persons, view metrics (confidence, pixel box, mask pixel count), and highlight selections bidirectionally.
- **Streaming Event Pipeline**: 9 sequential stages with monotonic sequence numbers, background execution, and catch-up reconciliation.
- **Robust Security**: Decompression bomb protection (50M pixel limit), format validation (JPEG/PNG/non-animated WebP), and directory traversal rejection.
- **Verification**: 30 backend tests, 8 web tests, 5 schema tests, 2 shared type tests, and 6 Playwright browser tests passing.

---

## Next Mission: Phase 03 — Person & Background Separation

Phase 03 will focus on:
1. High-resolution matting refinement along fine boundaries (hair, clothing edges).
2. Foreground person extraction and layer composition.
3. Background inpainting to fill occlusion voids behind detected persons.
4. Independent foreground and background plane separation in 3D space.
