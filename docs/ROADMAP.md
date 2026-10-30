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
| **Phase 07** | **Skeleton, Rigging & Animation** | **PLANNED** | 3D kinematic skeleton fitting, bone hierarchy attachment, skinning weights, and independent character pose/animation playback. |
| **Phase 08** | **Texture & Material Synthesis** | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 09** | **Scene Editing & Interactive Directing** | Planned | Interactive 3D scene compositing, spatial staging, lighting controls, and multi-character timelines. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Production packaging, GLTF/USDZ export, performance optimization, and offline deployment. |

---

## Current Status: Phase 06 Completed

Phase 06 establishes genuine multi-person 3D scene assembly from individually reconstructed human meshes:
- **Deterministic Scene Assembly Engine (`services/api/app/pipeline/assembly.py`)**:
  - Implements coordinate transformation separating canonical model space from scene space:
    $$\mathbf{T}_{\text{scene}} = \mathbf{T}_{\text{placement}} \times \mathbf{R}_{\text{orientation}} \times \mathbf{S}_{\text{normalization}}$$
  - **Horizontal Ordering ($X$)**: Maps bounding box lateral centers to the reference plane, ensuring person A on the left remains on the left in 3D.
  - **Vertical Grounding ($Y$)**: Mathematically grounds characters to the $Y = 0$ floor plane.
  - **Relative Depth Ordering ($Z$)**: Maps canonical monocular depth proximity $P \in [0.0, 1.0]$ into scene depth with calibrated spread factors, placing closer foreground people nearer to the camera ($+Z$) without inventing fake metric units.
  - **Scale Bounding**: Scales meshes proportionally to detected bounding box height, clamped to anatomical bounds $[0.8\text{m}, 2.2\text{m}]$.
  - **Occlusion Reconciliation**: Detects overlapping 2D boxes ($\text{IoU} > 0.12$) with close proximity ($|\Delta P| < 0.05$) and flags `"occlusion_ambiguity"` without destroying composition.
  - **Partial Reconstruction Resilience**: Entities with pending or failed mesh reconstructions are maintained as valid in-scene nodes and rendered as wireframe spatial bounding volumes.
- **Hierarchical Scene Graph & Provenance**:
  - `SceneGraph` and `SceneGraphNode` schemas fully integrated in TypeScript and Pydantic.
  - `PlacementProvenance` records image center, bounding box, depth source, relative proximity, scale, orientation, and uncertainty flags.
  - `SceneAssemblyReport` summarizes assembly metrics and occlusion warnings.
- **Local Scene Persistence**:
  - Scenes automatically serialized and recovered from `data/storage/scenes/{scene_id}.json`.
  - Atomic temporary file writes and path-traversal sanitization.
- **Interactive Three.js Viewport & Controls**:
  - **Camera Controls**: "Fit Scene" dynamically bounds all visible assembled entities and frames them; "Focus Person" centers on selected entity.
  - **Visibility Toggling**: Per-entity visibility toggle hides/shows meshes and bounding boxes interactively.
  - **Unreconstructed Subject Indicators**: Renders 3D wireframe bounding boxes with identity badges for subjects without 3D meshes.
  - **Pipeline & Entity Inspector**: Dedicated Scene Assembly stage row in pipeline inspector; comprehensive 3D Scene Placement inspector displaying position, depth source, visibility, and uncertainty warnings.
- **Full Test Coverage**: **116 automated tests passing** across 5 test suites (backend pytest, frontend vitest, schema vitest, and Playwright E2E).

---

## Next Mission: Phase 07 — Skeleton, Rigging & Animation

Phase 07 will focus on:
1. Extracting 3D kinematic humanoid skeletons aligned with reconstructed human meshes.
2. Generating skinning weights and binding bone hierarchies.
3. Enabling independent character posing and animation playback in the assembled scene.
