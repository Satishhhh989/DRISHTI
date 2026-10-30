# DRISHTI v1.0.0 Release Notes

**Product**: DRISHTI (PHOTO → 3D ENGINE)  
**Release Version**: `v1.0.0` (Production Release)  
**Release Date**: October 2026  
**Tagline**: *From Pixels to Perception.*  
**Engineering Milestone**: Completion of all 10 phases in the engineering roadmap.  

---

## 1. Executive Summary

DRISHTI v1.0.0 marks the production delivery of a **local-first AI spatial reconstruction engine** that transforms ordinary photographs containing multiple people into interactive, explorable 3D scenes.

Built around absolute user privacy and data sovereignty, DRISHTI executes 100% of its computer vision, depth estimation, parametric human reconstruction, rigging, and animation locally on host hardware—delivering native hardware acceleration on Apple Silicon Metal Performance Shaders (`mps`) and NVIDIA CUDA (`cuda`), with zero cloud Vision API dependencies, zero external telemetry, and zero user data exfiltration.

---

## 2. Key Subsystems Delivered

### 2.1 DRISHTI Engine (Local Vision & Reconstruction Backend)
- **Local Neural Inference**: Integrates lightweight, state-of-the-art models for 2D detection, instance segmentation, depth estimation, and 3D human mesh recovery.
- **Hardware Acceleration**: Automatic discovery and device management supporting Apple Silicon Metal (`mps`), NVIDIA CUDA (`cuda`), and multi-threaded CPU SIMD fallback.
- **Event-Driven Streaming**: High-performance Server-Sent Events (SSE) streaming engine with a bounded 500-event FIFO buffer and automatic HTTP polling fallback.
- **Storage Subsystem**: Lossless disk persistence for photos, 8-bit binary instance masks, float32 `.npy` depth maps, and binary GLTF (`.glb`) meshes in `data/storage/`.

### 2.2 DRISHTI Studio (Interactive 3D Workspace)
- **Next.js 15 & React 19 Workspace Shell**: Modern, responsive dark-themed user interface with TopBar navigation, Contextual Inspector, and Live Pipeline Inspector.
- **Three.js & React Three Fiber Viewport**: Real-time WebGL rendering pipeline supporting 2D photo overlays, displaced 3D surfaces, and parametric human clay meshes.
- **Progressive Visual Reveals**: Asynchronous rendering pipeline displaying intermediate assets (bounding boxes, masks, depth maps, 3D meshes) as soon as they become ready without blocking behind generic loading screens.
- **Camera Navigation Contract**: Intelligent camera controller offering 2D pan/zoom for photographs, bounded parallax for spatial photos, and full 360-degree orbit for 3D multi-person scenes with camera state preservation across view switches.

### 2.3 DRISHTI Scene (Spatial Scene Graph & Representation System)
- **Deterministic Multi-Person Assembly**: Places reconstructed human meshes into a shared metric scene space ($+X$ Right, $+Y$ Up, $+Z$ Forward), preserving horizontal photographic ordering and relative depth layering.
- **Ground Plane Alignment**: Mathematically grounds characters to the horizontal floor ($Y = 0$) using mesh bounding heights.
- **Occlusion Reconciliation**: Detects overlapping 2D subjects and flags visual ambiguity without discarding photographic composition.
- **Unified Representation Registry**: Formal registry defining 9 distinct representations (`original`, `detection`, `segmentation`, `depth`, `spatial_photo`, `wireframe`, `solid3d`, `articulated`, `combined`).

### 2.4 DRISHTI Motion (Skeleton, Rigging & Animation)
- **Canonical 24-Joint Kinematic Skeleton**: Anatomically grounded 24-joint hierarchy aligned with SMPL topology.
- **Multi-Person 2D Pose Estimation**: YOLOv8-pose integration with IoU entity matching.
- **SMPL 3D Joint Regression**: Mathematically regresses joint centers from reconstructed mesh vertices.
- **Linear Blend Skinning (LBS)**: Computes vertex skinning weights normalized to top-4 influences ($L_1 = 1.0$) for hardware shader compatibility.
- **Procedural Animation Engine**: Synthesizes deterministic procedural animation clips (`idle_breathing`, `torso_twist`, `arm_wave`, `gentle_walk`) with transparent provenance labeling.

---

## 3. Ten-Phase Engineering Roadmap Completion

| Phase | Milestone | Outcome in v1.0.0 |
|---|---|---|
| **Phase 01** | Foundation & Architecture | Monorepo layout, FastAPI backend, Next.js frontend, Three.js viewport, versioned scene schemas. |
| **Phase 02** | Image Intelligence & Segmentation | Local YOLOv8 nano detection and instance segmentation producing 8-bit binary PNG masks. |
| **Phase 03** | Depth Estimation & Spatial Photography | Local Depth Anything V2 integration, relative proximity normalization, $128 \times 128$ vertex displacement mesh. |
| **Phase 04** | Live Visualization Engine | SSE push architecture, 500-event FIFO buffer, idempotent event reducer, progressive asset reveals. |
| **Phase 05** | AI Human 3D Reconstruction | SPIN + pure PyTorch SMPL LBS forward layer, 6,890 vertices, 13,776 faces, neutral clay GLB export. |
| **Phase 06** | Multi-Person Scene Assembly | Hierarchical SceneGraph, ground plane contact, horizontal ordering, relative depth spread, partial failure resilience. |
| **Phase 07** | DRISHTI Motion | 24-joint skeleton, YOLOv8-pose, SMPL 3D joint regression, normalized skinning weights, procedural animation clips. |
| **Phase 08** | Unified Hybrid Renderer | 9 representation modes, shared asset cache, single camera contract, hybrid photographic reference plane. |
| **Phase 09** | Interaction, Persistence & Export | Local project management, atomic disk saves, transform manipulation, single/scene GLB export, portable ZIP archives. |
| **Phase 10** | Integration, Security & Release | Production hardening, security HTTP headers, anti-zip-slip, zip bomb limits, 182 passing tests, release documentation. |

---

## 4. Security Hardening Highlights

- **Loopback Privacy**: Default binding restricted strictly to `127.0.0.1:8000`; CORS locked to `http://localhost:3000` and `http://127.0.0.1:3000`.
- **Upload Protection**: File size bounding (25 MB max), structural Pillow verification, animated image rejection, and 25-megapixel decompression bomb defense.
- **Archive Import Security**: Comprehensive Anti-Zip-Slip path component inspection and resource-bounded extraction (`MAX_ARCHIVE_ENTRIES = 250`, `MAX_UNCOMPRESSED_BYTES = 500MB`, compression ratio $\le 100\times$).
- **HTTP Security Headers**: Enforces `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and `X-XSS-Protection: 1; mode=block`.

---

## 5. Verification & Test Metrics

- **Total Automated Tests**: **182 Passed** (100% pass rate).
- **Backend Tests (Pytest)**: 112 passed in 4.61s.
- **Frontend Tests (Vitest)**: 37 passed.
- **Scene Schema Tests (Vitest)**: 18 passed.
- **Shared Types Tests (Vitest)**: 2 passed.
- **Browser E2E Tests (Playwright)**: 13 passed in 4.1s.
- **Total Model Checkpoint Footprint**: ~296 MB.

