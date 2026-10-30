# DRISHTI: PHOTO → 3D ENGINE

A local-first AI-powered spatial reconstruction engine that transforms ordinary multi-person photographs into interactive, animated 3D scenes.

Current Status: **Phase 02: Image Intelligence & Person Segmentation (Completed)**

---

## Capabilities (Phase 02)

- **Local-First AI Architecture**: Runs entirely offline on the user's machine without any external cloud vision APIs or telemetry.
- **Hardware Acceleration**: Automatic capability discovery for Apple Silicon Metal Performance Shaders (`mps`), NVIDIA GPU (`cuda`), and optimized CPU fallback.
- **Multi-Person Detection**: Detects all people in uploaded photographs using YOLOv8, calculating normalized coordinates, exact pixel boxes, confidence metrics, and stable entity IDs (`person-001`, `person-002`, ...).
- **Instance Segmentation**: Produces individual single-channel 8-bit binary PNG masks ($255$ foreground, $0$ background) bilinearly resized to the photo's original resolution.
- **Interactive 3D Viewport**:
  - **Original Mode**: Clean photograph rendered on an upright aspect-ratio-locked 3D plane.
  - **Detection Mode**: 3D wireframe bounding box outlines with floating name/confidence badges.
  - **Segmentation Mode**: Semi-transparent colored alpha overlays mapped onto person silhouettes.
  - **Wireframe & 3D Solid Modes**: Honest placeholder states communicating Phase 06 availability.
  - **3D Spatial Lock**: Bounding boxes and masks remain pixel-perfect during camera orbit, pan, and zoom.
- **Contextual Inspector**: Click any person in the 3D viewport or select from the sidebar to inspect confidence scores, exact pixel coordinates, and mask pixel areas.
- **9-Stage Streaming Event Pipeline**: Asynchronous background pipeline emitting monotonic sequenced events (`image_validation`, `image_preparation`, `person_detection`, `detection_validation`, `instance_segmentation`, `mask_validation`, `asset_persistence`, `scene_update`, `completed`) with catch-up reconciliation.
- **Decompression Bomb & Input Protection**: Strict image validation enforcing PIL limits (50M pixels max), corrupt file detection, non-animated format verification, and EXIF orientation normalization.
- **Local Asset Storage**: Segregated storage in `data/storage/` (`photos/`, `masks/`) with strict path traversal prevention.
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
│   ├── models/                # Local neural network checkpoints (yolov8n-seg.pt)
│   └── storage/               # Local user photos and segmentation masks
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
