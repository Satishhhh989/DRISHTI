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
| **Phase 08** | **Unified Hybrid Renderer** | **COMPLETED** | Single-scene multi-representation architecture, formal representation registry (9 modes), shared scene state and camera contract, unified asset cache, 2D/3D navigation policy, articulated mode, and hybrid combined scene viewing. |
| **Phase 09** | **Scene Editing & Interactive Directing** | **PLANNED** | Interactive 3D scene compositing, spatial staging, lighting controls, multi-character timelines, and scene export. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Production packaging, GLTF/USDZ export, performance optimization, and offline deployment. |

---

## Current Status: Phase 08 Completed

Phase 08 implements the **Unified Hybrid Renderer**, integrating all capabilities across Phases 01 through 07 into a unified, consistent, interactive rendering architecture:
- **Representation Registry (`packages/scene-schema/src/registry.ts`)**:
  - Central registry declaring 9 canonical representations (`original`, `detection`, `segmentation`, `depth`, `spatial_photo`, `wireframe`, `solid3d`, `articulated`, `combined`).
  - Formal declaration of display names, categories, required assets, supported scene types, rendering behaviors, known limitations, and interactive capabilities.
- **Unified Asset Cache (`apps/web/lib/asset-cache.ts`)**:
  - Deduplicated in-memory template caching for GLTF models and textures.
  - Eliminates duplicate network requests and re-parsing across mode switches.
  - Shared geometry buffer reuse across solid, wireframe, articulated, and combined views.
  - Explicit resource disposal on scene change or unmount.
- **Single Camera & Navigation Contract (`CameraController.tsx`)**:
  - Clear separation between 2D photographic inspection (pure pan/zoom with rotation locked) and 3D scene navigation (unconstrained 360-degree orbit, pan, zoom, fit, focus).
  - Restrained parallax orbit in Spatial Photo mode ($\pm 18^\circ$ azimuth, $75^\circ \text{--} 105^\circ$ polar).
  - 3D camera state preservation: switching between `solid3d` $\leftrightarrow$ `wireframe` $\leftrightarrow$ `articulated` $\leftrightarrow$ `combined` maintains the user's active camera orbit and zoom.
  - Reliable fit-to-view computing bounds for 2D photographs and 3D multi-person meshes.
- **Unified Person Selection & Visibility**:
  - Stable person entity IDs and selected entity state preserved across all representations.
  - Per-entity visibility state maintained across representations.
  - Focus Camera on Person centers the camera on the selected entity's 3D mesh volume.
- **Hybrid Combined Scene Representation**:
  - Unifies 3D assembled human meshes with an upright photographic spatial reference plane ($Z = -0.6\text{m}$, 85% opacity, locked aspect ratio).
- **Full Test Coverage**: **148 automated tests passing** across 5 test suites (88 backend pytest, 32 web vitest, 14 schema vitest, 2 shared-types vitest, and 12 Playwright E2E).

---

## Next Mission: Phase 09 — Scene Editing & Interactive Directing

Phase 09 will focus on:
1. Interactive 3D scene compositing and spatial staging.
2. Dynamic lighting controls and multi-character timelines.
3. Scene export (GLTF / USDZ) and persistent directing tools.
