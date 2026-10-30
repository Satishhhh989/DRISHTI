# System Architecture

## Overview

**PHOTO → 3D ENGINE** is an AI-assisted application designed to reconstruct interactive, animatable 3D scenes from multi-person 2D photographs. Normal operations—including model inference, asset processing, scene assembly, and project persistence—execute directly on the user's local machine.

Phase 01 establishes the foundational monorepo structure, typed contracts, modular backend service, Three.js 3D viewport, and state management.

```
+-------------------------------------------------------------------------------+
|                               Browser Client                                  |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  |                       Workspace Shell (Next.js)                         |  |
|  |  +---------------------+  +----------------------+  +----------------+  |  |
|  |  |  TopBar Navigation  |  | Representation Modes |  | Inspector Pane |  |  |
|  |  +---------------------+  +----------------------+  +----------------+  |  |
|  +-------------------------------------------------------------------------+  |
|  |                   Three.js Viewport (React Three Fiber)                 |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  |  | Perspective Camera |  | Textured 3D Plane  |  | Orbit Controls    |  |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  +-------------------------------------------------------------------------+  |
|  |  Zustand Stores: useSceneStore (Scene data) / useUIStore (UI state)      |  |
|  +-------------------------------------------------------------------------+  |
+---------------------------------------|---------------------------------------+
                                        | HTTP / JSON (127.0.0.1:8000)
                                        v
+-------------------------------------------------------------------------------+
|                       Local Backend Service (FastAPI)                         |
|                                                                               |
|  +-----------------------+  +----------------------+  +--------------------+  |
|  | Health & Version APIs |  | Scene Lifecycle API  |  | Asset Manager API  |  |
|  +-----------------------+  +----------------------+  +--------------------+  |
|  | Event Emitter & Catch-up Query Engine           |  | Job State Tracker  |  |
|  +-------------------------------------------------+  +--------------------+  |
|  | Modular AI Pipeline Interfaces (Phase 02 - 10)                           |  |
|  |   IPersonDetector | ISegmentor | IDepthEstimator | IMeshReconstructor    |  |
|  +-------------------------------------------------------------------------+  |
|  | Local Filesystem Storage (photos/, masks/, depth/, meshes/, skeletons/) |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## 1. Monorepo Organization

The project is structured as a workspace:

- `apps/web`: Next.js 15 App Router frontend using React 19, Three.js, React Three Fiber, and Zustand.
- `services/api`: FastAPI backend service exposing REST endpoints on the loopback address (`127.0.0.1:8000`).
- `packages/scene-schema`: Shared TypeScript scene schema models, validation routines, and coordinate system definitions.
- `packages/shared-types`: Shared TypeScript definitions for processing events, asset contracts, and API DTOs.
- `docs/`: Technical specifications, schema references, development instructions, and the 10-phase roadmap.
- `scripts/`: Development and automated verification scripts (`dev.sh`, `test.sh`, `setup.sh`).

---

## 2. Frontend Architecture

### 2.1 UI Layer
The UI provides a dark workspace focused on the 3D viewport:
- **TopBar**: Identity (`PHOTO → 3D ENGINE`), phase status (`Phase 01`), backend connection indicator, representation tabs, grid toggle, and camera reset.
- **ThreeViewport**: React Three Fiber canvas with WebGL renderer, perspective camera, ambient & directional lighting, 3D coordinate floor grid, and camera controls.
- **Empty State**: Drag-and-drop file upload target when no photograph is loaded.
- **ContextualPanel**: Collapsible inspector displaying scene identifier, image resolution, aspect ratio, detected people list, and selected entity properties.
- **ProcessingIndicator**: Status bar showing engine state, upload progress, and non-blocking error notifications.

### 2.2 Viewport & 3D Spatial Rendering
- The uploaded photograph is mapped onto a 3D textured rectangular plane positioned upright at world coordinates `(0, height/2, 0)`.
- Image aspect ratio ($W/H$) is preserved to prevent distortion.
- Orbit controls provide 3D camera rotation, panning, and zooming.
- Clean WebGL resource allocation with texture disposal on unmount.

### 2.3 State Management
State is divided across two Zustand stores:
- `useSceneStore`: Authoritative scene graph, active representation mode (`photo` | `wireframe` | `solid3d`), image dimensions, and local photo asset URL.
- `useUIStore`: UI-specific state: inspector drawer open/close, grid visibility, camera reset triggers, backend connectivity status, and error alerts.

---

## 3. Backend Architecture

### 3.1 Network & Privacy Boundary
- The API binds exclusively to the loopback address `127.0.0.1` by default to prevent external network exposure.
- Photographs, segmentation masks, depth maps, and 3D meshes remain strictly on the local filesystem.
- No third-party cloud AI inference is used.

### 3.2 Asset Manager
- Assets are stored under the configurable root directory (`data/storage/`).
- Separation by asset category:
  - `photos/`: Original photograph files.
  - `masks/`: Alpha mattes and instance segmentation masks.
  - `depth/`: Metric and relative scene depth maps.
  - `meshes/`: Extracted 3D polygon meshes.
  - `textures/`: Material texture maps.
  - `skeletons/`: Kinematic skeleton and joint hierarchies.
  - `exports/`: Scene packages and exported files.
- **Security**: Path traversal attacks (`../`) are detected and rejected with `StorageSecurityError`.
- SHA-256 checksums are calculated upon storage to guarantee asset integrity.

### 3.3 Modular AI Pipeline Interfaces
In Phase 01, abstract interfaces define the boundary for future inference engines without bundling heavy weights:
- `IPersonDetector`: Multi-person 2D bounding box detection (Phase 02).
- `ISegmentor`: Person instance segmentation and alpha matting (Phase 03).
- `IDepthEstimator`: Monocular scene depth estimation (Phase 04).
- `IMeshReconstructor`: Multi-person 3D mesh geometry reconstruction (Phase 06).
- `ISkeletonFitter`: Humanoid skeletal rigging and joint fitting (Phase 08).
- `IAnimator`: Character animation binding and playback (Phase 09).
