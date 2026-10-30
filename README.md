# PHOTO → 3D ENGINE

A local-first application transforming ordinary multi-person photographs into interactive, spatially reconstructed, animatable 3D scenes.

Current Phase: **Phase 01: Foundation & Architecture**

---

## Capabilities (Phase 01)

- **Local-First Architecture**: AI inference boundaries, photographs, 3D assets, and scene metadata remain strictly on the user's computer.
- **Three.js Spatial Viewport**: Integrated React Three Fiber viewport with perspective camera, 3D floor grid, orbital navigation, and responsive canvas sizing.
- **Aspect-Ratio Preserved Photographic Plane**: Loads ordinary photographs onto an upright 3D textured plane in real-world metric coordinates without distortion.
- **Representation State Machine**: Switcher for **Photo**, **Wireframe**, and **3D** representations. Unavailable modes display honest placeholders describing the future phase that produces them.
- **Typed Scene & Event Schemas**: Formally validated contracts for scenes, entities, transforms, representations, and sequenced background processing events.
- **FastAPI Backend**: Modular service running on loopback (`127.0.0.1:8000`) with health checks, version endpoints, and safe local asset storage with directory traversal protection.
- **Comprehensive Automated Tests**: 32 unit, integration, and Playwright browser tests covering backend APIs, schemas, UI stores, and end-to-end workspace workflows.

---

## Project Structure

```
photo-to-3d-engine/
│
├── apps/
│   └── web/                   # Next.js 15, React 19, Three.js, React Three Fiber, Zustand
│       ├── app/               # App Router pages and layout
│       ├── components/        # Viewport, controls, scene elements, contextual panel
│       ├── hooks/             # Custom hooks (health monitoring, viewport resize)
│       ├── lib/               # API client and image dimension utility
│       ├── stores/            # Zustand scene and UI state stores
│       └── tests/             # Vitest frontend unit and component tests
│
├── services/
│   └── api/                   # Python 3.11, FastAPI, Pydantic, Uvicorn, Pytest
│       ├── app/
│       │   ├── api/           # Health, version, assets, scenes, events endpoints
│       │   ├── core/          # Validated config, structured logging, custom errors
│       │   ├── schemas/       # Scene, event, asset, and spatial Pydantic models
│       │   ├── scene/         # In-memory scene manager and validation
│       │   ├── assets/        # Safe local filesystem storage engine
│       │   ├── pipeline/      # Modular abstract AI pipeline interfaces
│       │   └── events/        # Event sequencer and catch-up emitter
│       └── tests/             # Pytest backend test suite
│
├── packages/
│   ├── scene-schema/          # TypeScript scene models, validation, coordinate conventions
│   └── shared-types/          # TypeScript processing event contracts and API DTOs
│
├── docs/                      # Architectural specs, schema guide, event protocol, roadmap
├── scripts/                   # dev.sh, test.sh, setup.sh, run_api.sh
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
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Run Automated Tests
```bash
bash scripts/test.sh
```

---

## Engineering Rules Followed in Phase 01

1. **No Cloud Inference**: Strictly local operations; no external AI services or image uploads.
2. **No Large Model Downloads**: Phase 01 establishes interfaces and architecture only.
3. **No Fake Progress or Synthetic 3D Meshes**: The application does not simulate AI progress or fabricate 3D humans.
4. **Strict Coordinate System**: Right-handed Cartesian coordinates ($+X$ Right, $+Y$ Up, $+Z$ Forward, metric units).
5. **Security & Privacy**: Enforced loopback binding (`127.0.0.1`) and safe asset path sanitization preventing directory traversal.
