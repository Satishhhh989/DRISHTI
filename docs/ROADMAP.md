# Ten-Phase Project Roadmap

**DRISHTI** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | **Image Intelligence & Person Segmentation** | **COMPLETED** | Local multi-person detection (YOLOv8), instance segmentation (8-bit binary PNG masks), hardware-accelerated inference (Apple Silicon MPS / CUDA), 3D viewport overlays, entity inspector, sequenced event streaming. |
| **Phase 03** | **Depth Estimation & Spatial Photography** | **COMPLETED** | Local monocular depth estimation (Depth Anything V2), canonical relative proximity $[0, 1]$, robust percentile normalization, per-person depth statistics, 3D vertex displacement mesh, bounded parallax camera controls. |
| **Phase 04** | **Live Visualization & Progressive Processing Engine** | **COMPLETED** | Server-Sent Events (SSE) streaming, bounded event retention (500 FIFO), idempotent event reducer, progressive visual reveals (instant photo, detection boxes, per-person masks, depth map, spatial photo), live pipeline inspector, job cancellation, non-fatal partial failure isolation. |
| **Phase 05** | **Multiplane Spatial Photography & Background Separation** | **PLANNED** | High-resolution matting boundary refinement, foreground extraction, depth-guided hole filling, background inpainting, and layered multiplane 2.5D parallax spatial photo rendering. |
| **Phase 06** | **3D Human Mesh Reconstruction** | Planned | Volumetric human mesh generation producing genuine 3D polygon geometry and wireframes. |
| **Phase 07** | **Texture & Material Synthesis** | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 08** | **Human Skeleton & Rigging** | Planned | 3D kinematic skeleton fitting and skinning weight attachment for human subjects. |
| **Phase 09** | **Scene Assembly & Animation** | Planned | Interactive 3D scene compositing with independent character animation controls. |
| **Phase 10** | **Polish, Export & Performance** | Planned | Project persistence, GLTF/USDZ export, performance optimization, and packaging. |

---

## Current Status: Phase 04 Completed

Phase 04 transforms DRISHTI from an opaque batch processor into a live, transparent, progressive reconstruction engine:
- **Server-Sent Events (SSE) Push Architecture**: High-efficiency unidirectional `GET /api/v1/jobs/{job_id}/stream` endpoint delivering real-time processing events directly to the frontend with sub-millisecond latency.
- **Bounded Event Retention Buffer**: Thread-safe FIFO ring buffer capped at 500 events per job preventing memory bloat during long sessions while allowing seamless reconnection replay.
- **Progressive Scene Commits**: Coordinator commits intermediate results to the authoritative scene manager at each milestone (image validation, detection completion, per-person mask persistence, depth calculation), enabling live queries mid-pipeline.
- **Pure Idempotent Event Reducer (`reduceProcessingEvent`)**: Deterministic client-side state machine ensuring identical scene state regardless of network packet chunking or replay order, while strictly preserving active camera coordinates, selection state, and viewport modes.
- **Granular Representation Availability (`computeRepresentationAvailability`)**: Dynamic status tracking across all 7 scene representations (`unavailable | pending | processing | ready | failed`) with live badges (`LIVE`, `ERR`, `P06`).
- **Progressive Viewport Reveal**:
  - Original photographic plane appears immediately upon image validation.
  - 3D detection bounding boxes render the instant person detection completes.
  - Segmentation masks appear progressively as each person's mask is saved to disk and committed.
  - Depth map visualization unlocks dynamically as soon as relative proximity computation concludes.
  - Spatial photo vertex displacement surface activates seamlessly when depth and photo assets are verified.
  - Advanced wireframe and mesh representations remain cleanly disabled with honest explanatory messages until implemented in Phase 06.
- **Live Pipeline Inspector**: Contextual panel displaying stage-by-stage progression, entity counters ("3 of 5 people segmented"), and real-time step statuses (`pending | running | complete | failed`).
- **Cooperative Job Cancellation**: Clean cancellation via `POST /api/v1/jobs/{job_id}/cancel` and frontend "Cancel Analysis" button, terminating background thread processing cleanly while preserving completed intermediate assets.
- **Non-Fatal Partial Failure Isolation**: Gracefully isolates failures (e.g. depth estimation failure), recording warnings while preserving interactive detection and segmentation results.
- **Automated Verification**: **89 automated tests passing** across 5 test suites (48 pytest backend tests, 33 vitest frontend and schema tests, 8 Playwright browser E2E tests).

---

## Next Mission: Phase 05 — Multiplane Spatial Photography & Background Separation

Phase 05 will focus on:
1. High-resolution alpha matting refinement along fine boundaries (hair, clothing edges).
2. Foreground person extraction and layer composition.
3. Background inpainting and depth-guided hole filling to fill occlusion voids behind detected persons.
4. Independent foreground and background plane separation in 3D space with clean multiplane parallax.
