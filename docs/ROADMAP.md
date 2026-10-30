# Ten-Phase Project Roadmap

**DRISHTI** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | **Image Intelligence & Person Segmentation** | **COMPLETED** | Local multi-person detection (YOLOv8), instance segmentation (8-bit binary PNG masks), hardware-accelerated inference (Apple Silicon MPS / CUDA), 3D viewport overlays, entity inspector, sequenced event streaming. |
| **Phase 03** | **Depth Estimation & Spatial Photography** | **COMPLETED** | Local monocular depth estimation (Depth Anything V2), canonical relative proximity $[0, 1]$, robust percentile normalization, per-person depth statistics, 3D vertex displacement mesh, bounded parallax camera controls. |
| **Phase 04** | **Live Visualization & Progressive Processing Engine** | **COMPLETED** | Server-Sent Events (SSE) streaming, bounded event retention (500 FIFO), idempotent event reducer, progressive visual reveals (instant photo, detection boxes, per-person masks, depth map, spatial photo), live pipeline inspector, job cancellation, non-fatal partial failure isolation. |
| **Phase 05** | **AI Human 3D Reconstruction Engine** | **COMPLETED** | Local parametric 3D human mesh reconstruction (SPIN + SMPL LBS forward layer), standard topology (6,890 vertices, 13,776 faces), geometry validation, binary GLTF (.glb) export with neutral studio clay PBR material, Three.js wireframe and solid 3D rendering with 360-degree inspection, multi-person partial failure resilience. |
| **Phase 06** | **Multi-Person 3D Scene Assembly & Kinematic Skeleton Fitting** | **PLANNED** | Multi-person spatial scene assembly, relative depth & scale positioning, occlusion reconciliation, and 3D kinematic skeleton fitting. |
| **Phase 07** | **Texture & Material Synthesis** | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 08** | **Human Skeleton & Rigging** | Planned | 3D kinematic skeleton fitting and skinning weight attachment for human subjects. |
| **Phase 09** | **Scene Assembly & Animation** | Planned | Interactive 3D scene compositing with independent character animation controls. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Project persistence, GLTF/USDZ export, performance optimization, and packaging. |

---

## Current Status: Phase 05 Completed

Phase 05 establishes genuine local 3D human mesh reconstruction from photographs:
- **Local Model Architecture**: Integrates SPIN (SMPL Optimization in the Loop of Neural Networks, Kolotouros et al., ICCV 2019) with a ResNet-50 backbone and iterative 6D parameter regressor.
- **Pure PyTorch SMPL Linear Blend Skinning (LBS)**: Eliminates legacy C++ and `chumpy` dependencies, performing canonical template deformation, PCA shape blending ($\beta$), and joint kinematic rotation in pure PyTorch running on Apple Silicon MPS / CUDA / CPU.
- **Parametric 3D Geometry**: Reconstructs standardized anatomical meshes with 6,890 vertices and 13,776 triangular faces.
- **Geometry Validation & Sanitization**: Strict checks for NaN/infinite values, degenerate zero-area faces, index boundary validity, vertex normal computation, and real-world human bounds ($[0.2\text{m}, 2.6\text{m}]$).
- **Binary GLTF (.glb) Assets**: Exports binary GLTF models stored under `data/storage/meshes/{scene_id}_{entity_id}.glb` with neutral studio clay PBR material (`#d0d0d8`, roughness 0.65).
- **Interactive Three.js Viewport**:
  - `wireframe`: Renders authentic polygon wireframe topology in technical cyan/white lines.
  - `solid3d`: Renders solid clay meshes with studio lighting and raycast selection highlight.
  - 360-degree orbit, pan, zoom, and camera reset controls.
- **Multi-Person & Fault-Tolerant Pipeline**: Independent per-person crop preparation with aspect-ratio letterboxing, safe padding, and clipping boundary detection. Non-fatal isolation guarantees failure of one person does not abort other people.
- **Full Test Coverage**: **103 automated tests passing** across 5 test suites (59 pytest backend tests, 35 vitest frontend and schema tests, 9 Playwright browser E2E tests).

---

## Next Mission: Phase 06 — Multi-Person 3D Scene Assembly & Kinematic Skeleton Fitting

Phase 06 will focus on:
1. Multi-person spatial positioning and relative depth scaling in 3D scene coordinates.
2. Ground plane alignment and contact resolution.
3. Multi-person occlusion reconciliation and depth-consistent sorting.
4. Kinematic skeleton extraction and joint hierarchy fitting for reconstructed meshes.
