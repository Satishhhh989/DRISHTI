# DRISHTI Automated Testing & Quality Assurance Guide

**Product Identity**: DRISHTI Engine, DRISHTI Studio, DRISHTI Scene, DRISHTI Motion  
**Version**: `1.0.0` (Production Release)  
**Total Automated Tests**: **182 Tests Across 5 Test Suites (100% Passing)**  

---

## 1. Testing Philosophy & Architecture

DRISHTI enforces a rigorous, multi-tiered testing strategy designed to verify functionality, numerical stability, local security boundaries, and user experience across all packages:

```
[Level 5: Browser E2E Tests] ────────── 13 Tests (Playwright / Chromium)
      │
[Level 4: Frontend Component Tests] ──── 37 Tests (Vitest / React Testing Library / Three.js)
      │
[Level 3: Shared Types & Schema] ────── 20 Tests (Vitest / TypeScript)
      │
[Level 2: Backend API & Pipeline] ───── 112 Tests (Pytest / FastAPI / PyTorch / Trimesh)
      │
[Level 1: Security Hardening] ────────── Dedicated Suite (Anti-Zip-Slip, Bomb Defense, Traversal)
```

---

## 2. Test Execution Commands

### 2.1 Complete System Verification (Master Suite)
Execute all 182 automated tests across all monorepo packages with a single command:

```bash
bash scripts/test.sh
```

### 2.2 Individual Package Test Suites

#### 1. Backend Python API & Pipeline Tests (112 tests)
```bash
bash scripts/test_backend.sh
# Or directly via pytest:
PYTHONPATH=services/api .venv/bin/pytest services/api/tests -v
```

#### 2. Frontend React & Three.js Studio Tests (37 tests)
```bash
pnpm --filter @photo-to-3d/web test
```

#### 3. Scene Schema & Registry Tests (18 tests)
```bash
pnpm --filter @photo-to-3d/scene-schema test
```

#### 4. Shared Types & Event Contracts (2 tests)
```bash
pnpm --filter @photo-to-3d/shared-types test
```

#### 5. Playwright Browser E2E Tests (13 tests)
```bash
pnpm test:e2e
```

---

## 3. Test Suites Breakdown

### 3.1 Backend Pytest Suite (112 Tests)

| Test File | Tests | Focus Area |
|---|---|---|
| `test_assets.py` | 3 | Photo, mask, depth, and mesh storage and registration |
| `test_config.py` | 2 | Environment variable parsing and directory validation |
| `test_depth.py` | 9 | Relative depth estimation, 2nd/98th percentile normalization, per-person statistics |
| `test_device.py` | 2 | DeviceManager hardware discovery (`mps`, `cuda`, `cpu`) |
| `test_e2e_integration.py` | 1 | Complete full-lifecycle integration test (upload $\to$ reconstruction $\to$ assembly $\to$ export $\to$ reimport) |
| `test_errors.py` | 2 | Structured error models and status codes |
| `test_event_schema.py` | 2 | Event schema validation and serialization |
| `test_export_system.py` | 6 | Single entity GLB, assembled GLB, JSON metadata, portable ZIP export, and safe import |
| `test_health.py` | 1 | Health check endpoint, service name, version (`1.0.0`), storage availability |
| `test_image_validation.py` | 6 | Dimension bounds, MIME whitelisting, decompression bomb rejection |
| `test_mesh_rigging.py` | 6 | RiggingEngine skinning weights, normalization ($L_1 = 1.0$), bone associations |
| `test_pipeline_api.py` | 3 | Background job submission, status polling, and result retrieval |
| `test_pipeline_assembly.py` | 1 | Scene assembly stage execution and SceneGraph generation |
| `test_pipeline_depth_mock.py` | 2 | Mock depth pipeline deterministic outputs |
| `test_pipeline_depth_real.py` | 1 | Real Depth Anything V2 inference smoke test |
| `test_pipeline_mock.py` | 3 | Mock pipeline detection, segmentation, and event emissions |
| `test_pipeline_motion.py` | 1 | End-to-end motion stage execution and skeleton export |
| `test_pipeline_real.py` | 1 | Real YOLOv8 segmentation inference smoke test |
| `test_pipeline_reconstruction_mock.py` | 2 | Mock reconstruction geometry output |
| `test_pipeline_reconstruction_real.py` | 1 | Real SPIN + SMPL neural reconstruction smoke test |
| `test_pose_estimator.py` | 3 | YOLOv8-pose keypoints and multi-person IoU matching |
| `test_procedural_animation.py` | 3 | Deterministic procedural animation clips (idle, twist, wave, walk) |
| `test_progressive_events.py` | 5 | Sequenced event emitter, bounded FIFO buffer (500 events), SSE stream pub/sub |
| `test_project_persistence.py` | 4 | Atomic project saves, manifest hashing, loading with missing assets |
| `test_reconstruction_geometry.py` | 8 | Mesh validation, face area calculation, vertex normals, dimension bounds |
| `test_scene_assembly_engine.py` | 7 | Ground contact, horizontal ordering, depth positioning, occlusion reconciliation |
| `test_scene_interaction.py` | 4 | Transform validation, bounding constraints, reset transform |
| `test_scene_persistence.py` | 2 | Scene JSON atomic disk save and discovery |
| `test_scene_schema.py` | 5 | Scene schema validation, scale bounds, bounding boxes, depth metadata |
| `test_security_hardening.py` | 9 | Security headers, upload bounds, anti-zip-slip, zip bomb limits, CORS |
| `test_skeleton_schema.py` | 6 | Kinematic hierarchy, cycle detection, duplicate joint checks, coordinate validation |
| `test_version.py` | 1 | Version endpoint reporting (`1.0.0`) |

### 3.2 Frontend Vitest Suite (37 Tests)

| Test File | Tests | Focus Area |
|---|---|---|
| `app.test.tsx` | 3 | Main workspace shell, topbar, and error boundaries |
| `assembly-view.test.tsx` | 2 | Multi-person scene assembly, ground plane, and visibility toggles |
| `depth.test.tsx` | 3 | Depth map visualization and spatial photo mesh displacement |
| `interaction-persistence.test.tsx` | 4 | Transform editing, numeric inputs, bounds validation, dirty indicators |
| `motion-view.test.tsx` | 2 | Skeleton overlay rendering, animation clip selector, playback controls |
| `progressive-inspector.test.tsx` | 3 | Real-time stage progress, entity counters, non-fatal partial error alerts |
| `reconstruction-view.test.tsx` | 1 | Human 3D wireframe and solid clay mesh rendering |
| `representation.test.tsx` | 3 | Representation selector availability logic and honest placeholders |
| `unified-renderer.test.tsx` | 5 | Multi-representation switching, camera state preservation, combined mode |
| `event-reducer.test.ts` | 6 | Idempotent event reduction, sequence ordering, stage updates |
| `image-utils.test.ts` | 5 | Image aspect ratio calculations and texture coordinates |

### 3.3 Scene Schema Vitest Suite (18 Tests)

| Test File | Tests | Focus Area |
|---|---|---|
| `schema.test.ts` | 6 | Scene validation, entity schemas, coordinate bounds |
| `registry.test.ts` | 4 | Representation registry, capabilities, and asset requirements |
| `availability.test.ts` | 4 | Dynamic representation readiness evaluation |
| `project.test.ts` | 4 | Project model, manifest schemas, and migration versions |

### 3.4 Playwright Browser E2E Suite (13 Tests)
Verifies browser-level user workflows under realistic rendering conditions:
1. Workspace shell layout and initial state rendering.
2. Upload control availability in top bar and empty viewport.
3. Three.js WebGL canvas initialization.
4. Representation switcher and honest placeholders for unavailable modes.
5. Photographic visual layers (original, detection boxes, segmentation masks).
6. Contextual inspector empty state handling.
7. Depth map and spatial photo 3D displaced surface rendering.
8. Live pipeline inspector real-time stage states.
9. Human 3D wireframe and solid clay modes.
10. Multi-person scene assembly and camera controls.
11. DRISHTI Motion skeleton overlays and procedural animation playback.
12. Unified Hybrid Renderer representations and view transitions.
13. Interaction, persistence, and asset export controls.

---

## 4. Mock vs. Real Model Testing Strategy

DRISHTI provides dual testing modes:

1. **Fast Deterministic Mode (Default)**:
   - Uses mock neural pipelines to execute tests rapidly without requiring model checkpoint downloads or GPU resources.
   - Entire 182-test suite completes in **under 15 seconds**.
2. **Real Model Smoke Tests**:
   - `test_pipeline_real.py`, `test_pipeline_depth_real.py`, and `test_pipeline_reconstruction_real.py` test actual neural network inference against real image crops when checkpoints are present in `data/models/`.
   - Automatically skipped if weights are absent, ensuring CI portability.

