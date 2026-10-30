/**
 * DRISHTI - Unified Representation Registry
 * Phase 08: Unified Hybrid Renderer
 *
 * Formal registry defining all available representations, their required assets,
 * supported scene types, rendering capabilities, and known limitations.
 */

import {
  VisualizationMode,
  RepresentationRegistryEntry,
} from "./types.js";

export const REPRESENTATION_REGISTRY: Record<
  VisualizationMode,
  RepresentationRegistryEntry
> = {
  original: {
    id: "original",
    displayName: "Original Photograph",
    category: "2d_photo",
    requiredAssets: ["source_image"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "instant",
    renderingBehavior:
      "Native photographic aspect ratio preserved without geometric warping",
    knownLimitations: [
      "2D image view with no depth or 3D rotational perspective",
    ],
    supportsSelection: false,
    supportsInteraction: true,
    supportsAnimation: false,
  },
  detection: {
    id: "detection",
    displayName: "Person Detection",
    category: "inspection",
    requiredAssets: ["source_image", "detection_boxes"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "2D bounding box overlays with person labels and detection confidences",
    knownLimitations: [
      "Bounding boxes approximate spatial extent and do not convey exact silhouettes",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: false,
  },
  segmentation: {
    id: "segmentation",
    displayName: "Instance Segmentation",
    category: "inspection",
    requiredAssets: ["source_image", "instance_masks"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Per-person alpha color mask overlays projected over source image",
    knownLimitations: [
      "Single-view 2D boundary masks; occluded body parts remain unsegmented",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: false,
  },
  depth: {
    id: "depth",
    displayName: "Depth Map",
    category: "inspection",
    requiredAssets: ["source_image", "depth_map"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Normalized grayscale relative depth visualization where brighter indicates closer foreground",
    knownLimitations: [
      "Monocular relative proximity is uncalibrated metric depth",
    ],
    supportsSelection: false,
    supportsInteraction: true,
    supportsAnimation: false,
  },
  spatial_photo: {
    id: "spatial_photo",
    displayName: "Spatial Photograph",
    category: "spatial_photo",
    requiredAssets: ["source_image", "depth_map"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "128x128 vertex-displaced 3D mesh surface with bounded parallax orbit camera",
    knownLimitations: [
      "Disocclusion artifacts and stretching occur at extreme oblique viewing angles",
    ],
    supportsSelection: false,
    supportsInteraction: true,
    supportsAnimation: false,
  },
  wireframe: {
    id: "wireframe",
    displayName: "3D Wireframe",
    category: "3d_mesh",
    requiredAssets: ["mesh_reconstruction"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Full 3D scene assembly rendering genuine reconstructed polygon edge topology",
    knownLimitations: [
      "Reveals mesh triangulation directly without surface shading",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: true,
  },
  solid3d: {
    id: "solid3d",
    displayName: "3D Solid Mesh",
    category: "3d_mesh",
    requiredAssets: ["mesh_reconstruction"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Full 3D scene assembly rendering studio clay PBR material under ambient and directional lighting",
    knownLimitations: [
      "Untextured clay geometry; surface colors derived from neutral shading",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: true,
  },
  articulated: {
    id: "articulated",
    displayName: "Articulated Character",
    category: "3d_mesh",
    requiredAssets: ["mesh_reconstruction", "skeleton_rig"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Rigged human mesh with 24-joint kinematic skeleton overlay and procedural articulation",
    knownLimitations: [
      "Procedural animation clips simulate biomechanical motion; not motion recovered from photo",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: true,
  },
  combined: {
    id: "combined",
    displayName: "Combined Scene",
    category: "hybrid",
    requiredAssets: ["source_image", "mesh_reconstruction"],
    supportedSceneTypes: ["single_person", "multi_person"],
    loadingBehavior: "progressive",
    renderingBehavior:
      "Hybrid spatial composition placing assembled 3D characters in front of an upright photographic reference plane",
    knownLimitations: [
      "Reference photo plane is positioned for spatial correlation, not calibrated projection",
    ],
    supportsSelection: true,
    supportsInteraction: true,
    supportsAnimation: true,
  },
};

export function getAllRepresentations(): RepresentationRegistryEntry[] {
  return Object.values(REPRESENTATION_REGISTRY);
}

export function getRepresentationEntry(
  mode: VisualizationMode
): RepresentationRegistryEntry {
  return REPRESENTATION_REGISTRY[mode] || REPRESENTATION_REGISTRY.original;
}

export function is3DRepresentation(mode: VisualizationMode): boolean {
  return (
    mode === "wireframe" ||
    mode === "solid3d" ||
    mode === "articulated" ||
    mode === "combined"
  );
}

export function is2DRepresentation(mode: VisualizationMode): boolean {
  return (
    mode === "original" ||
    mode === "detection" ||
    mode === "segmentation" ||
    mode === "depth"
  );
}

