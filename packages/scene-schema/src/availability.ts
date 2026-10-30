/**
 * DRISHTI - Scene Representation Availability Model
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 * 
 * Computes availability, dependencies, and selectable status for all
 * scene visualization modes based on authoritative scene assets and active job state.
 */

import {
  RepresentationStatus,
  Scene,
  VisualizationMode,
} from "./types.js";

export interface RepresentationAvailabilityInfo {
  status: RepresentationStatus;
  isSelectable: boolean;
  reason?: string;
}

export type SceneRepresentationAvailability = Record<
  VisualizationMode,
  RepresentationAvailabilityInfo
>;

/**
 * Evaluates the availability of every viewport visualization mode based on real scene data.
 */
export function computeRepresentationAvailability(
  scene: Scene | null,
  activeJobStage?: string | null,
  isAnalyzing: boolean = false
): SceneRepresentationAvailability {
  // 1. Original Photo
  const hasPhoto = Boolean(
    scene?.sourceImage?.assetId || (scene as unknown as Record<string, unknown>)?.localPhotoUrl
  );
  const originalStatus: RepresentationStatus = hasPhoto ? "ready" : "unavailable";

  // 2. Detection Overlay
  const hasDetections = Boolean(scene && scene.entities && scene.entities.length > 0);
  const isDetecting =
    isAnalyzing &&
    (activeJobStage === "person_detection" || activeJobStage === "detection_validation");
  let detectionStatus: RepresentationStatus = "unavailable";
  if (hasDetections) {
    detectionStatus = "ready";
  } else if (isDetecting) {
    detectionStatus = "processing";
  } else if (isAnalyzing) {
    detectionStatus = "pending";
  }

  // 3. Segmentation Masks Overlay
  const hasMasks = Boolean(
    scene?.entities?.some(
      (e) => e.segmentation?.status === "ready" && Boolean(e.segmentation?.maskUri || e.segmentation?.maskAssetId)
    )
  );
  const isSegmenting =
    isAnalyzing &&
    (activeJobStage === "instance_segmentation" || activeJobStage === "asset_persistence");
  let segStatus: RepresentationStatus = "unavailable";
  if (hasMasks) {
    segStatus = "ready";
  } else if (isSegmenting) {
    segStatus = "processing";
  } else if (isAnalyzing) {
    segStatus = "pending";
  }

  // 4. Depth Map
  const hasDepth = Boolean(
    scene?.depth?.status === "ready" &&
      Boolean(scene?.depth?.visualizationUri || scene?.depth?.visualizationAssetId)
  );
  const isDepthEstimating =
    isAnalyzing &&
    (activeJobStage === "depth_estimation" || activeJobStage === "depth_normalization");
  const depthFailed = scene?.depth?.status === "failed";
  let depthStatus: RepresentationStatus = "unavailable";
  if (hasDepth) {
    depthStatus = "ready";
  } else if (depthFailed) {
    depthStatus = "failed";
  } else if (isDepthEstimating) {
    depthStatus = "processing";
  } else if (isAnalyzing) {
    depthStatus = "pending";
  }

  // 5. Spatial Photo
  const spatialReady = Boolean(
    scene?.representations?.spatial_photo?.status === "ready" && hasDepth && hasPhoto
  );
  const spatialFailed = scene?.representations?.spatial_photo?.status === "failed";
  let spatialStatus: RepresentationStatus = "unavailable";
  if (spatialReady) {
    spatialStatus = "ready";
  } else if (spatialFailed) {
    spatialStatus = "failed";
  } else if (
    isDepthEstimating ||
    (isAnalyzing && activeJobStage === "spatial_photo_preparation")
  ) {
    spatialStatus = "processing";
  } else if (isAnalyzing) {
    spatialStatus = "pending";
  }

  return {
    original: {
      status: originalStatus,
      isSelectable: originalStatus === "ready",
      reason: originalStatus === "ready" ? "Original source photo" : "No photograph loaded",
    },
    detection: {
      status: detectionStatus,
      isSelectable: detectionStatus === "ready",
      reason:
        detectionStatus === "ready"
          ? `${scene?.entities?.length || 0} person detections available`
          : detectionStatus === "processing"
          ? "Neural network detection running..."
          : "Requires validated person detections",
    },
    segmentation: {
      status: segStatus,
      isSelectable: segStatus === "ready",
      reason:
        segStatus === "ready"
          ? "Instance segmentation masks available"
          : segStatus === "processing"
          ? "Generating alpha masks..."
          : "Requires instance segmentation masks",
    },
    depth: {
      status: depthStatus,
      isSelectable: depthStatus === "ready",
      reason:
        depthStatus === "ready"
          ? "Depth map visualization available"
          : depthStatus === "processing"
          ? "Estimating relative depth..."
          : depthStatus === "failed"
          ? "Depth estimation failed"
          : "Requires depth map",
    },
    spatial_photo: {
      status: spatialStatus,
      isSelectable: spatialStatus === "ready",
      reason:
        spatialStatus === "ready"
          ? "3D spatial photo with parallax available"
          : spatialStatus === "processing"
          ? "Preparing spatial photo mesh..."
          : spatialStatus === "failed"
          ? "Spatial photo unavailable (depth failed)"
          : "Requires photo and depth map",
    },
    wireframe: {
      status: "unavailable",
      isSelectable: false,
      reason: "Available in Phase 06: Human Mesh Reconstruction",
    },
    solid3d: {
      status: "unavailable",
      isSelectable: false,
      reason: "Available in Phase 06/07: Solid 3D Reconstructed Mesh",
    },
  };
}

