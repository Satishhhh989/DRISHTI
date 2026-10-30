# Ten-Phase Project Roadmap

**DRISHTI** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | **Image Intelligence & Person Segmentation** | **COMPLETED** | Local multi-person detection (YOLOv8), instance segmentation (8-bit binary PNG masks), hardware-accelerated inference (Apple Silicon MPS / CUDA), 3D viewport overlays, entity inspector, sequenced event streaming. |
| **Phase 03** | **Depth Estimation & Spatial Photography** | **COMPLETED** | Local monocular depth estimation (Depth Anything V2), canonical relative proximity $[0, 1]$, robust percentile normalization, per-person depth statistics, 3D vertex displacement mesh, bounded parallax camera controls. |
| **Phase 04** | **Live Visualization & Progressive Processing Engine** | **COMPLETED** | Server-Sent Events (SSE) streaming, bounded event retention (500 FIFO), idempotent event reducer, progressive visual reveals (instant photo, detection boxes, per-person masks, depth map, spatial photo), live pipeline inspector, job cancellation, non-fatal partial failure isolation. |
| **Phase 05** | **AI Human 3D Reconstruction Engine** | **COMPLETED** | Local parametric 3D human mesh reconstruction (SPIN + SMPL LBS forward layer), standard topology (6,890 vertices, 13,776 faces), geometry validation, binary GLTF (.glb) export with neutral studio clay PBR material, Three.js wireframe and solid 3D rendering with 360-degree inspection, multi-person partial failure resilience. |
| **Phase 06** | **Multi-Person Scene Assembly** | **COMPLETED** | Multi-person spatial scene assembly, relative depth & horizontal positioning, ground contact alignment, occlusion reconciliation, partial reconstruction resilience, camera auto-framing ("Fit Scene" & "Focus Person"), visibility toggle, local scene JSON persistence. |
| **Phase 07** | **DRISHTI Motion: Skeleton, Rigging & Animation** | **COMPLETED** | Canonical 24-joint kinematic skeleton, 2D pose estimation (YOLOv8-pose), SMPL 3D joint regression, normalized linear blend skinning weights (top-4 influences), deterministic procedural animation clips (idle, twist, wave, walk), interactive viewport controls, independent character playback. |
| **Phase 08** | **Unified Hybrid Renderer & Neural Appearance** | **PLANNED** | High-fidelity texture mapping with unseen-surface inpainting, hybrid neural appearance, and unified multi-person rendering. |
| **Phase 09** | **Scene Editing & Interactive Directing** | Planned | Interactive 3D scene compositing, spatial staging, lighting controls, and multi-character timelines. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Production packaging, GLTF/USDZ export, performance optimization, and offline deployment. |

---

## Current Status: Phase 07 Completed

Phase 07 implements **DRISHTI Motion**, transitioning reconstructed static human meshes into articulated 3D characters:
- **Canonical 24-Joint Kinematic Skeleton (`services/api/app/pipeline/motion.py`)**:
  - Aligned with SMPL standard topology: 24 canonical joints, 23 bones, pelvis at root.
  - Complete structural validation (`validate_skeleton_hierarchy`): cycle detection, duplicate checks, parent validity, coordinate checks.
- **2D Pose Estimation & Matching**:
  - `YoloPoseEstimator` powered by local `yolov8n-pose.pt` (6.5MB) with multi-person IoU matching and confidence preservation.
  - `MockPoseEstimator` fallback for deterministic offline execution.
- **3D Skeleton Fitting (`SkeletonFitter`)**:
  - SMPL 3D joint regression ($J = J_{regressor} \cdot V_{6890}$) or anatomical bounding box landmarks.
  - Aligned rest positions centered at $(0, 0, 0)$ in local humanoid space.
- **Mesh Rigging Engine (`RiggingEngine`)**:
  - SMPL native LBS skinning weights ($6890 \times 24$) from `SMPL_N_model.npz` or proximity heat skinning.
  - Normalized to top-4 influences summing to 1.0 ($L_1 = 1.0$), satisfying `validate_skinning_weights`.
- **Procedural Animation Engine (`ProceduralAnimationEngine`)**:
  - Deterministic procedural clips with quaternion rotations: `idle_breathing` (3.0s), `torso_twist` (4.0s), `arm_wave` (2.5s), `gentle_walk` (2.0s).
  - Explicit `provenance: "procedural"` badge and schema label.
  - Independent per-person playback state (`isPlaying`, `currentClip`, `playbackSpeed`, `loop`).
- **Interactive Viewport & Controls**:
  - `SkeletonOverlay`: Renders 24 joint spheres (pelvis highlighted in amber, limbs in emerald) and 23 bone line segments with X-ray visibility.
  - Playback controls: Play/Pause, Reset Pose, speed chips (0.5x, 1x, 1.5x), clip dropdown.
  - Pipeline Inspector stage rows: `Pose Estimation`, `Skeleton Fitting`, `Mesh Rigging`, `Procedural Motion`.
- **Full Test Coverage**: **138 automated tests passing** across 5 test suites (88 backend pytest, 27 web vitest, 10 schema vitest, 2 shared-types vitest, and 11 Playwright E2E).

---

## Next Mission: Phase 08 — Unified Hybrid Renderer & Neural Appearance

Phase 08 will focus on:
1. High-fidelity texture mapping with unseen-surface inpainting and PBR shaders.
2. Hybrid neural appearance synthesis.
3. Multi-character unified scene lighting and compositing.
