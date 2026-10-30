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
| **Phase 09** | **Interaction, Persistence & Export** | **COMPLETED** | Local project management, atomic disk persistence, asset manifests, interactive transform editing (position, rotation, scale, bounds validation, reset), scene hierarchy & inspector, single entity GLB export, multi-person assembled GLB export with node transforms, metadata JSON export, portable project ZIP archives with anti-zip-slip protection, archive reimport. |
| **Phase 10** | **Polish, Production & Release** | **PLANNED** | Production packaging, performance tuning, offline deployment, and final release. |

---

## Current Status: Phase 09 Completed

Phase 09 transforms DRISHTI into a fully functional, persistent scene-editing workspace with real local asset export:
- **Project Model & Asset Manifest (`packages/scene-schema`, `services/api/app/schemas/project.py`)**:
  - Stable project identifier, versioned metadata, creation and last-modified timestamps, scene references, workspace settings, processing state, and structured asset manifest.
  - Manifest tracks all referenced source photos, masks, depth maps, and reconstructed GLB meshes with relative paths and SHA-256 checksums.
- **Reliable Local Project Persistence (`services/api/app/project/manager.py`)**:
  - Atomic saves via temporary files (`.tmp` write, flush, atomic replace) to prevent corruption during interrupted writes.
  - Resilient project loading that reports missing assets without deleting or clobbering valid project data.
  - Recent projects discovery and sorting.
- **Scene Interaction & Transform Controls (`apps/web`, `services/api/app/api/scenes.py`)**:
  - User can select individual reconstructed people and edit Position ($X, Y, Z$), Rotation (Euler $Y$), and Uniform Scale.
  - Server-side constraint validation: rejects non-finite numbers, bounds position to $[-50\text{m}, 50\text{m}]$, and validates scale within $[0.05, 10.0]$.
  - Reliable Reset Transform operation restoring initial depth-projected placement coordinates.
  - Workspace dirty tracking: unsaved changes indicator appears on edits; cleared upon saving.
- **Real Local Export Engine (`services/api/app/export/engine.py`, `services/api/app/api/export.py`)**:
  - **Single Entity Mesh (GLB)**: Exports validated 3D mesh with neutral studio clay material; rejects export of unreconstructed entities without fake placeholder fabrication.
  - **Assembled Multi-Person Scene (GLB)**: Exports multi-person 3D scene preserving relative positions, rotations, and node transforms.
  - **Scene Metadata (JSON)**: Exports canonical scene metadata.
  - **Portable Project Archive (.zip)**: Exports self-contained ZIP archive containing `project.json`, `scene.json`, `manifest.json`, and all referenced local assets.
  - **Safe Project Archive Reimport**: Imports portable project ZIP archives with strict Path Traversal (Anti-Zip-Slip) defense and schema validation.
- **Full Test Coverage**: **172 automated tests passing** across 5 test suites (102 backend pytest, 37 web vitest, 18 schema vitest, 2 shared-types vitest, and 13 Playwright E2E).

---

## Next Mission: Phase 10 — Polish, Production & Release

Phase 10 will focus on:
1. End-to-end user experience polish and cross-platform desktop packaging.
2. Production builds, runtime performance optimizations, and final release verification.
