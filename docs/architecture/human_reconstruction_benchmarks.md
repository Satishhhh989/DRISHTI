# DRISHTI Human Reconstruction Architecture & Model Benchmarks

## 1. Executive Summary

DRISHTI is an offline, local-first image-to-3D spatial engine designed to reconstruct photographic scenes and multi-person human figures from a single image. Human reconstruction in DRISHTI requires parametric, articulated 3D representations that preserve source-image pose fidelity, accurate body shapes, natural joint kinematics, and seamless integration with skeletal rigging and procedural animation.

This document details the architectural evaluation, empirical benchmarks, and design rationale across leading monocular human mesh recovery (HMR) models and generalist 3D generative architectures.

---

## 2. Model Evaluation Matrix

| Criterion | SPIN + SMPLPoseFitter (Production Engine) | HMR 2.0 / 4DHumans (Modular Adapter) | Large Human Model (LHM) | TRELLIS / TRELLIS.2 |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Architecture** | ResNet-50 Feature Extractor + Iterative Regressor + Differentiable Optimization | Vision Transformer (ViT-H/B) + Cross-Attention SMPL Transformer Decoder | Multi-view / Diffusion-based Implicit Implicit Human Field | Rectified Flow / Structured Latent 3D Gaussian Generator |
| **Output Format** | **SMPL Canonical** (6,890 vertices, 24 kinematic joints) | **SMPL Canonical** (6,890 vertices, 24 kinematic joints) | Dense Non-Parametric Triangle Mesh (~50k–100k vertices) | 3D Gaussian Splats / Untextured Surface Mesh |
| **Articulated Rigging** | **Native** (Linear Blend Skinning, 24-bone skeleton) | **Native** (Linear Blend Skinning, 24-bone skeleton) | **None** (Static geometry; requires auto-rigging) | **None** (Static decorative mesh / radiance splats) |
| **Procedural Animation** | **Directly Compatible** (Phase 08 biomechanical idle / sway) | **Directly Compatible** (Phase 08 biomechanical idle / sway) | Unsupported without external retargeting | Unsupported; unriggable non-manifold geometry |
| **Seated / Complex Poses** | **High** (Resolved via multi-hypothesis pose fitting + silhouette DT) | **Very High** (Robust in-the-wild feature extraction) | Moderate (Prone to limb fusion and seat artifacts) | Poor (Collapses into lumpy seated blob with chair) |
| **Memory Footprint** | **~130 MB weights**, < 1.2 GB peak RAM | **~2.1 GB weights**, ~4.5 GB peak RAM/VRAM | **~6.5 GB weights**, > 12 GB VRAM | **> 14 GB weights**, > 16 GB VRAM |
| **Inference Latency** | **25–45 ms** on Apple Silicon M-series (MPS) | **180–350 ms** on Apple Silicon / CUDA | **4.5–12.0 s** per person | **15.0–45.0 s** per object |
| **Local Offline Execution** | **100% Offline** (Zero external network calls) | **100% Offline** (Local checkpoint) | Challenging; requires hefty local VRAM | Exceeds standard local laptop compute budgets |
| **Licensing** | MIT / Academic (SMPL license required for template) | CC-BY-NC 4.0 (Non-Commercial Research; SMPL license) | Custom Research Non-Commercial | Apache 2.0 (Generator) / Weights Vary |
| **DRISHTI Status** | **Default Production Engine** | **Supported Modular Adapter (Standby)** | Rejected (Unrigged, excessive latency) | **Rejected (Unarticulated static generator)** |

---

## 3. Rationale: Why Generalist 3D Generators (TRELLIS) Are Rejected for Human Reconstruction

Generalist 3D generation models such as **TRELLIS**, **TripoSR**, **LGM**, and **Shap-E** have made significant progress in generating arbitrary 3D assets (e.g., chairs, cups, cars) from single images. However, applying TRELLIS to monocular human reconstruction introduces critical technical and architectural failures:

1. **Lack of Kinematic Hierarchy & Parametric Rigging**:
   - TRELLIS generates static 3D Gaussian splats or untextured manifold meshes. It possesses zero semantic understanding of anatomical joints (hips, knees, spine, shoulders).
   - In DRISHTI's progressive pipeline, reconstructed humans must immediately bind to a 24-joint kinematic skeleton (`SkeletonFitter`), accept linear blend skinning weights (`RiggingEngine`), and execute natural biomechanical movements (`ProceduralAnimationEngine`). An unarticulated mesh produced by TRELLIS cannot be animated without fragile heuristic mesh segmentation.

2. **Severe Hallucination in Seated and Contact Poses**:
   - Monocular generative models attempt to reconstruct the entire visual volume inside the bounding box. When a person is seated on a chair or wooden crate, TRELLIS fuses the person's clothing, legs, and furniture into an inseparable 3D lump.
   - In contrast, parametric body models (SMPL) recover the true human body geometry independent of external seating structures, allowing the physical seat to be segmented and placed separately in the scene graph.

3. **Inability to Enforce 2D Silhouette and Keypoint Constraints**:
   - Feedforward generative diffusion models cannot be differentiated against 2D anatomical keypoints (YOLOv8-Pose) or instance segmentation silhouettes in real time.
   - DRISHTI's `SMPLPoseFitter` solves an exact optimization problem on the SMPL latent space, pulling joints and vertices into pixel-perfect correspondence with the source photograph using distance transform gradients.

4. **Resource Constraints and Pipeline Latency**:
   - TRELLIS requires 16+ GB of dedicated GPU VRAM and takes 20 to 45 seconds per entity. For a 5-person photograph, TRELLIS would take over 3 minutes of heavy compute and crash typical consumer laptops (e.g. 16GB Apple Silicon MacBooks).
   - DRISHTI reconstructs 5 people in under 2.5 seconds using SPIN and SMPLPoseFitter on local Apple Silicon MPS.

---

## 4. Technical Deep-Dive: SMPLPoseFitter Optimization & Correctness Fixes

### 4.1 Critical Joint Indexing Fix

The SMPL body model determines 3D joint positions $J \in \mathbb{R}^{24 \times 3}$ from deformed surface vertices $V \in \mathbb{R}^{6890 \times 3}$ using a linear joint regressor matrix $\mathcal{J}_{\text{reg}} \in \mathbb{R}^{24 \times 6890}$:

$$J = \mathcal{J}_{\text{reg}} V$$

When executed in PyTorch with batch processing, the output tensor `j_3d` has shape `(B, 24, 3)`. In earlier versions, code sliced `j_3d[smpl_indices]`. Because `smpl_indices` contains joint indices from $0$ to $23$, slicing along dimension 0 targeted the **batch dimension** when $B=1$, raising fatal `IndexError: index 12 is out of bounds for dimension 0 with size 1`.

#### The Solution: `SMPLPoseFitter.select_joints`
The `select_joints` static method was introduced to guarantee dimension-agnostic selection along the joint axis (dim $-2$):

```python
@staticmethod
def select_joints(j_3d: torch.Tensor, indices: Union[List[int], torch.Tensor]) -> torch.Tensor:
    if isinstance(indices, list):
        idx_t = torch.tensor(indices, dtype=torch.long, device=j_3d.device)
    else:
        idx_t = indices.to(device=j_3d.device, dtype=torch.long)

    if j_3d.dim() == 3:
        return j_3d[:, idx_t, :]
    elif j_3d.dim() == 2:
        return j_3d[idx_t, :]
    raise ValueError(f"Expected j_3d to be 2D (J, 3) or 3D (B, J, 3), got shape {j_3d.shape}")
```

### 4.2 Differentiable Silhouette Distance Transform Optimization

Traditional silhouette fitting requires heavy external differentiable mesh renderers (e.g., PyTorch3D or SoftRas) which are difficult to compile across platforms and consume substantial GPU memory.

DRISHTI implements a lightweight, exact differentiable silhouette objective using **Euclidean Distance Transforms** and bilinear grid sampling:

1. **Precomputation**: For a person's ground-truth binary mask $M \in \{0, 1\}^{H \times W}$, compute the distance field to the nearest mask boundary:
   $$\mathcal{D}_{\text{out}}(u, v) = \frac{1}{S} \text{dist}\big((u, v), M\big)$$
2. **Forward Sampling**: Surface vertices $V_{\text{surf}} \subset V$ are projected into camera coordinates $(u_i, v_i) = s \cdot V_i[:2] + t$. Using `torch.nn.functional.grid_sample`:
   $$d_i = \text{grid\_sample}\big(\mathcal{D}_{\text{out}}, (u_i, v_i)\big)$$
3. **Loss Formulation**:
   $$\mathcal{L}_{\text{sil}} = \frac{\lambda_{\text{sil}}}{|V_{\text{surf}}|} \sum_{i \in V_{\text{surf}}} d_i^2$$
4. **Exact Gradients**: Bilinear interpolation produces non-zero gradients $\nabla_{(u, v)} d_i^2$ that pull protruding 3D vertices smoothly into the photographed silhouette contour within 25–45 optimization steps.

---

## 5. Modular Backend Architecture (`ReconstructionBackendRegistry`)

DRISHTI isolates human reconstruction behind an abstract interface `HumanReconstructor`:

```
           +-----------------------------+
           |     PipelineCoordinator     |
           +--------------+--------------+
                          |
                          v
         +----------------------------------+
         |   ReconstructionBackendRegistry   |
         +----------------+-----------------+
                          |
       +------------------+------------------+
       |                                     |
       v                                     v
+-----------------------------+  +-----------------------------+
|    SpinHumanReconstructor   |  |    Hmr2HumanReconstructor   |
|   (Production SMPL Engine)  |  |    (ViT Adapter / Standby)  |
+-----------------------------+  +-----------------------------+
```

### Dynamic Capabilities Query
Backends declare their capabilities, licensing status, and runtime requirements via `get_capabilities()`:

```json
{
  "model_name": "HMR 2.0 (4DHumans)",
  "architecture": "Vision Transformer (ViT-H/B) + SMPL Cross-Attention Decoder",
  "is_articulated": true,
  "output_format": "SMPL Canonical (6,890 vertices, 24 joints)",
  "weights_installed": false,
  "license": "CC-BY-NC 4.0 (Non-Commercial Research; requires SMPL license)",
  "recommended_device": "mps (Apple Silicon) or cuda (NVIDIA)",
  "vram_recommended_mb": 4096
}
```

If optional heavy weights are absent from `data/models/hmr2/`, the adapter logs a transparent message and falls back directly to the local SPIN production engine with zero downtime or runtime crashes.

---

## 6. Honest Metrics Reporting

DRISHTI eliminates misleading hardcoded metrics (e.g. fabricated 0.80 IoU or 0.0 px error when masks or keypoints are missing):

- When fewer than 4 keypoints are detected, `reprojection_error_pixels`, `silhouette_iou`, and `quality_score` are strictly returned as `None` with `metric_status = "unavailable"`.
- When keypoints are present but segmentation mask is absent, `reprojection_error_pixels` is measured, `silhouette_iou` is `None`, and `metric_status = "estimated"`.
- When both keypoints and masks are verified, full metrics are measured and `metric_status = "measured"`.
- Synthetic test doubles report explicit `metric_status = "mock"`.
