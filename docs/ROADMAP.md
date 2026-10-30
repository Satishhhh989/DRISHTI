# Ten-Phase Project Roadmap

**PHOTO → 3D ENGINE** is delivered across ten focused engineering phases.

| Phase | Title | Status | Core Objective |
|---|---|---|---|
| **Phase 01** | **Foundation & Architecture** | **COMPLETED** | Monorepo setup, versioned Scene & Event schemas, FastAPI backend, Three.js viewport, representation switcher, local asset interface, automated test suite. |
| **Phase 02** | Multi-Person Detection | Planned | Local multi-person detector to isolate 2D bounding boxes and subject counts. |
| **Phase 03** | Person & Background Segmentation | Planned | High-resolution alpha matting to separate subjects and inpaint background planes. |
| **Phase 04** | Scene Depth Estimation | Planned | Monocular depth estimation generating metric depth maps for camera-relative positioning. |
| **Phase 05** | Spatial Photo Reconstruction | Planned | Multiplane image (MPI) or mesh displacement creating 2.5D parallax spatial photos. |
| **Phase 06** | 3D Human Mesh Reconstruction | Planned | Volumetric human mesh generation producing genuine 3D polygon wireframes. |
| **Phase 07** | Texture & Material Synthesis | Planned | High-fidelity texture mapping with unseen-surface inpainting and PBR shaders. |
| **Phase 08** | Human Skeleton & Rigging | Planned | 3D kinematic skeleton fitting and skinning weight attachment for human subjects. |
| **Phase 09** | Scene Assembly & Animation | Planned | Interactive 3D scene compositing with independent character animation controls. |
| **Phase 10** | Polish, Export & Performance | Planned | Project persistence, GLTF/USDZ export, performance optimization, and packaging. |

---

## Current Status: Phase 01 Completed

Phase 01 establishes the foundation:
- Validated contracts for scenes, transforms, entities, and events.
- Working Three.js viewport rendering photographs with preserved aspect ratios.
- Representation mode selector (`Photo` / `Wireframe` / `3D`) with honest placeholders for future modes.
- FastAPI backend running on loopback (`127.0.0.1:8000`) with safe local filesystem storage.
- Unit, schema, backend, and browser tests passing across the entire workspace.

Phase 02 will integrate the first local multi-person detection model.
