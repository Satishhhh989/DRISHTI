# DRISHTI Model & Asset Licensing Compliance Guide

## 1. Overview

DRISHTI integrates state-of-the-art computer vision models, parametric human representations, and graphics engines. This document inventories all models, templates, and libraries, specifying their licenses, redistribution terms, and commercial compliance guidelines.

---

## 2. Licensing Inventory

| Component | Upstream Author / Institution | License | Permitted Use | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8** (`yolov8m-seg`, `yolov8m-pose`) | Ultralytics Inc. | **AGPL-3.0** / Commercial Enterprise | Open Source / Dual License | AGPL-3.0 requires source disclosure if distributed over a network. Enterprise licenses are available directly from Ultralytics. |
| **Depth Anything V2** (`depth-anything-v2-small`) | ByteDance & HKU | **Apache 2.0** | Commercial & Non-Commercial | Highly permissive; allows modification and commercial redistribution with attribution. |
| **SPIN** (Sheng-Yu et al.) | UC Berkeley Computer Vision Group | **MIT License** | Commercial & Non-Commercial | Permissive research implementation for iterative SMPL parameter regression. |
| **SMPL Body Model** (`SMPL_N_model.npz`) | Max Planck Institute for Intelligent Systems (MPI-IS) | **Custom Non-Commercial Research License** | Research / Academic / Non-Commercial | **Requires explicit registration and acceptance of terms** at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de). Commercial deployment requires a license from Meshcapade GmbH. |
| **HMR 2.0 / 4DHumans** (`hmr2`) | Goel et al. / UC Berkeley | **CC-BY-NC 4.0** | Non-Commercial Research | Restricts use to non-commercial academic research. Also depends on the underlying SMPL body model license. |
| **Three.js** | Mr.doob & Three.js Authors | **MIT License** | Commercial & Non-Commercial | Client-side 3D WebGL / WebGPU scene rendering engine. |
| **Trimesh** | Dawson-Haggerty et al. | **MIT License** | Commercial & Non-Commercial | Pure Python triangular mesh loading, raycasting, and GLB serialization. |
| **PyTorch** | Meta Platforms Inc. / Linux Foundation | **Modified BSD License** | Commercial & Non-Commercial | Core tensor computation and autograd differentiation engine. |

---

## 3. Commercial Compliance & Deployment Guidelines

### 3.1 Non-Commercial / Academic / Open-Source Evaluation
Under academic and non-commercial research use:
- All included models (SPIN, SMPL Neutral, HMR 2.0, Depth Anything V2, YOLOv8) can be run locally within DRISHTI without licensing fees.
- Users must accept the SMPL terms of use at MPI-IS upon downloading the neutral body template.

### 3.2 Enterprise / Commercial Production Deployment
For commercial or closed-source enterprise deployments of DRISHTI:
1. **Parametric Body Model**: Obtain a commercial SMPL license from [Meshcapade GmbH](https://meshcapade.com), the commercial licensing entity for MPI-IS body models.
2. **Object Detection & Pose Estimation**: Obtain an Ultralytics Commercial License if proprietary closed-source distribution is required without AGPL-3.0 compliance.
3. **Reconstruction Backend**: Use the MIT-licensed SPIN architecture or proprietary regression heads rather than CC-BY-NC 4.0 HMR 2.0 weights.
4. **Depth Anything V2**: Fully compliant under Apache 2.0.

---

## 4. Local-First Air-Gapped Privacy

DRISHTI does not phone home, upload imagery to third-party cloud APIs, or transmit user photographs across the network. All models run entirely on the user's local machine (Apple Silicon MPS or NVIDIA GPU), guaranteeing data privacy, confidentiality, and compliance with strict data-protection regulations (GDPR, HIPAA).

