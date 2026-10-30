# DRISHTI: PHOTO → 3D ENGINE

A local-first AI-powered spatial reconstruction engine that transforms ordinary multi-person photographs into interactive, animated 3D scenes.

Current Status: **Phase 03: Depth Estimation & Spatial Photography (Completed)**

---

## Capabilities (Phase 03)

- **Local-First AI Architecture**: Runs entirely offline on the user's machine without any external cloud vision APIs or telemetry.
- **Hardware Acceleration**: Automatic capability discovery and execution on Apple Silicon Metal Performance Shaders (`mps`), NVIDIA GPU (`cuda`), and optimized CPU fallback.
- **Multi-Person Detection**: Detects all people in uploaded photographs using YOLOv8, calculating normalized coordinates, exact pixel boxes, confidence metrics, and stable entity IDs (`person-001`, `person-002`, ...).
- **Instance Segmentation**: Produces individual single-channel 8-bit binary PNG masks ($255$ foreground, $0$ background) bilinearly resized to the photo's original resolution.
- **Monocular Depth Estimation**: Local Depth Anything V2 Small neural network producing dense depth fields across the entire photograph without cloud round-trips.
- **Canonical Relative Proximity**: Standardized proximity convention $P \in [0.0, 1.0]$ where $1.0 = \text{closest foreground}$ and $0.0 = \text{farthest background}$, normalized with robust 2nd and 98th percentile clamping.
- **Interactive 3D Spatial Photography**:
  - **Spatial Photo Mode**: High-resolution $128 \times 128$ vertex-displaced 3D mesh surface with smooth parallax navigation.
  - **Bounded Parallax Camera**: Realistic camera orbit limits (azimuth $\pm 18^\circ$, polar $75^\circ \text{--} 105^\circ$, damping $0.06$) preventing disocclusion tears and extreme distortion.
  - **Depth Map Mode**: Normalized 8-bit grayscale inspection view.
  - **Original, Detection, and Segmentation Modes**: Instant switching between raw photo, 3D bounding boxes, and colored alpha masks.
  - **Wireframe & 3D Solid Modes**: Honest placeholder states communicating Phase 06 availability.
- **Contextual Inspector**: Interactive drawer providing depth metadata, depth displacement strength slider ($0.00 \text{--} 0.25$), camera reset action, and per-person median/min/max proximity stats.
- **12-Stage Streaming Event Pipeline**: Asynchronous background pipeline emitting monotonic sequenced events with catch-up reconciliation and non-fatal error isolation.
- **Local Asset Storage**: Lossless float32 machine-readable `.npy` depth maps, grayscale visualization PNGs, instance masks, and sanitized photos in `data/storage/`.
- **Automated Verification**: Comprehensive test coverage across backend (`pytest`), frontend (`vitest`), schemas (`vitest`), and browser automation (`playwright`).

---

## Project Structure

```
DRISHTI/
│
├── apps/
│   └── web/                   # Next.js 15, React 19, Three.js, React Three Fiber, Zustand
│       ├── app/               # App Router pages and layout
│       ├── components/        # Viewport, controls, scene elements, contextual panel
│       │   ├── controls/      # RepresentationSelector, ProcessingIndicator, UploadButton
│       │   ├── layout/        # TopBar, ContextualPanel
│       │   ├── scene/         # PhotoPlane, DetectionBox (3D), SegmentationMaskLayer (3D)
│       │   └── viewport/      # ThreeViewport, GridFloor, ErrorBoundary
│       ├── hooks/             # Custom hooks (health monitoring, viewport resize)
│       ├── lib/               # API client and image utilities
│       ├── stores/            # Zustand scene and UI state stores
│       └── tests/             # Vitest frontend unit and component tests
│
├── services/
│   └── api/                   # Python 3.11, FastAPI, PyTorch, Ultralytics, Pydantic
│       ├── app/
│       │   ├── api/           # Health, version, assets, scenes, events, pipeline routes
│       │   ├── core/          # Validated config, structured logging, custom errors
│       │   ├── schemas/       # Scene, event, asset, detection, and segmentation schemas
│       │   ├── scene/         # Scene manager and state validation
│       │   ├── assets/        # Storage manager and strict image validation
│       │   ├── pipeline/      # DeviceManager, ModelRegistry, YoloDetector, YoloSegmentor, Coordinator
│       │   └── events/        # Sequenced event emitter and catch-up query engine
│       └── tests/             # Pytest backend test suite (unit, mock, real MPS smoke tests)
│
├── packages/
│   ├── scene-schema/          # TypeScript scene models, validation, coordinate conventions
│   └── shared-types/          # TypeScript processing event contracts and API DTOs
│
├── data/
│   ├── models/                # Local neural network checkpoints (yolov8n-seg.pt, depth-anything-v2-small/)
│   └── storage/               # Local user photos, segmentation masks, and depth maps (.npy / .png)
│
├── docs/                      # ARCHITECTURE, SCENE_SCHEMA, EVENT_PROTOCOL, DEVELOPMENT, ROADMAP
├── scripts/                   # dev.sh, test_backend.sh, setup.sh, run_api.sh
└── tests/                     # Playwright browser-level E2E tests
```

---

## Quick Start

### 1. Initial Setup
```bash
bash scripts/setup.sh
```

### 2. Start Application
```bash
bash scripts/dev.sh
```
- Web Application: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Automated Tests
```bash
# Backend pytest suite
bash scripts/test_backend.sh

# Frontend vitest suite
pnpm --filter @photo-to-3d/web test

# Shared schema & type tests
pnpm --filter @photo-to-3d/scene-schema test
pnpm --filter @photo-to-3d/shared-types test

# Playwright E2E browser tests
pnpm test:e2e
```

---

## Engineering Rules

1. **No Cloud Vision APIs**: All neural inference runs locally on the host hardware.
2. **No Large Model Downloads**: Phase 02 uses lightweight, high-performance YOLOv8 nano segmentation (~6.7MB), explicitly managed in `data/models/`.
3. **No Fake AI or Synthetic Meshes**: Detection boxes, masks, confidence scores, and processing stages reflect actual model outputs. 3D polygon meshes and skeletons remain honest placeholders until Phases 06 and 08.
4. **Strict Coordinate System**: Right-handed Cartesian coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward, metric units).
5. **Security & Privacy**: Enforced loopback binding (`127.0.0.1`), safe asset path sanitization preventing directory traversal, and strict decompression bomb protection.
