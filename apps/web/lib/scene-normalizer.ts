/**
 * DRISHTI - Scene Schema Normalizer
 * Bridges Python backend (snake_case) and TypeScript frontend (camelCase)
 * guarantees full schema compatibility regardless of ingestion source.
 */

import { Scene, SceneEntity, SceneRepresentations, SourceImageMetadata, RepresentationAsset } from "@photo-to-3d/scene-schema";

function normalizeRepresentation(rep: any): RepresentationAsset {
  if (!rep) {
    return {
      status: "unavailable",
      assetId: null,
      uri: null,
      updatedAt: new Date().toISOString(),
    };
  }
  return {
    status: rep.status || "unavailable",
    assetId: rep.assetId ?? rep.asset_id ?? null,
    uri: rep.uri ?? null,
    error: rep.error ?? null,
    updatedAt: rep.updatedAt ?? rep.updated_at ?? new Date().toISOString(),
    config: rep.config ?? null,
  };
}

export function normalizeEntity(e: any): SceneEntity {
  if (!e) return e;
  const rawBox = e.boundingBox || e.pixel_box || e.bounding_box;
  let boundingBox = undefined;
  if (rawBox) {
    if (Array.isArray(rawBox) && rawBox.length >= 4) {
      boundingBox = {
        xMin: rawBox[0],
        yMin: rawBox[1],
        xMax: rawBox[2],
        yMax: rawBox[3],
      };
    } else if (typeof rawBox === "object") {
      boundingBox = {
        xMin: rawBox.xMin ?? rawBox.x_min ?? 0,
        yMin: rawBox.yMin ?? rawBox.y_min ?? 0,
        xMax: rawBox.xMax ?? rawBox.x_max ?? 1,
        yMax: rawBox.yMax ?? rawBox.y_max ?? 1,
      };
    }
  }

  const rawTransform = e.transform || {};
  const transform = {
    position: rawTransform.position || { x: 0, y: 0, z: -1.5 },
    rotation: rawTransform.rotation || { x: 0, y: 0, z: 0, order: "XYZ" },
    scale: rawTransform.scale || { x: 1, y: 1, z: 1 },
  };

  const rawRecon = e.reconstruction;
  const reconstruction = rawRecon ? {
    ...rawRecon,
    status: rawRecon.status || "none",
    meshAssetId: rawRecon.meshAssetId ?? rawRecon.mesh_asset_id ?? null,
    meshUrl: rawRecon.meshUrl ?? rawRecon.mesh_url ?? rawRecon.meshUri ?? rawRecon.mesh_uri ?? null,
    textureAvailability: rawRecon.textureAvailability ?? rawRecon.texture_availability ?? "none",
  } : undefined;

  const rawPlacement = e.placement;
  const placement = rawPlacement ? {
    method: rawPlacement.method,
    imageSpaceCoords: rawPlacement.imageSpaceCoords ?? rawPlacement.image_space_coords ?? [],
    depthSource: rawPlacement.depthSource ?? rawPlacement.depth_source,
    relativeProximity: rawPlacement.relativeProximity ?? rawPlacement.relative_proximity,
    modelNormalizationApplied: rawPlacement.modelNormalizationApplied ?? rawPlacement.model_normalization_applied,
    orientationCorrection: rawPlacement.orientationCorrection ?? rawPlacement.orientation_correction ?? [0, 0, 0],
    uncertaintyFlags: rawPlacement.uncertaintyFlags ?? rawPlacement.uncertainty_flags ?? [],
    isMetric: rawPlacement.isMetric ?? rawPlacement.is_metric ?? false,
    placedAt: rawPlacement.placedAt ?? rawPlacement.placed_at,
  } : undefined;

  const rawSkeleton = e.skeleton;
  const skeleton = rawSkeleton ? ({
    ...rawSkeleton,
    rootJointId: rawSkeleton.rootJointId ?? rawSkeleton.root_joint_id ?? "joint_00_pelvis",
    joints: rawSkeleton.joints || [],
    bones: rawSkeleton.bones || [],
    validationStatus: rawSkeleton.validationStatus ?? rawSkeleton.validation_status ?? "valid",
    source: rawSkeleton.source ?? "smpl_regressed",
  } as any) : undefined;

  const rawRig = e.rig;
  const rig = rawRig ? ({
    ...rawRig,
    status: rawRig.status || "rigged",
    skinningMethod: rawRig.skinningMethod ?? rawRig.skinning_method ?? "smpl_lbs_native",
    maxInfluencesPerVertex: rawRig.maxInfluencesPerVertex ?? rawRig.max_influences_per_vertex ?? 4,
    weightsNormalized: rawRig.weightsNormalized ?? rawRig.weights_normalized ?? true,
    boneCount: rawRig.boneCount ?? rawRig.bone_count ?? 24,
  } as any) : undefined;

  const rawClips = e.animationClips || e.animation_clips || [];
  const animationClips = Array.isArray(rawClips) ? rawClips.map((c: any) => ({
    ...c,
    name: c.name,
    label: c.label,
    durationSeconds: c.durationSeconds ?? c.duration_seconds,
    fps: c.fps,
    frameCount: c.frameCount ?? c.frame_count,
    provenance: c.provenance,
    loop: c.loop ?? true,
    keyframeTrackCount: c.keyframeTrackCount ?? c.keyframe_track_count,
  })) as any : undefined;

  const rawAnimState = e.animationState || e.animation_state || {};
  const animationState = {
    currentClip: rawAnimState.currentClip ?? rawAnimState.current_clip ?? null,
    isPlaying: rawAnimState.isPlaying ?? rawAnimState.is_playing ?? false,
    playbackSpeed: rawAnimState.playbackSpeed ?? rawAnimState.playback_speed ?? 1.0,
    loop: rawAnimState.loop ?? true,
  };

  const rawDepth = e.depth;
  const depth = rawDepth ? {
    status: rawDepth.status,
    medianProximity: rawDepth.medianProximity ?? rawDepth.median_proximity,
    minProximity: rawDepth.minProximity ?? rawDepth.min_proximity,
    maxProximity: rawDepth.maxProximity ?? rawDepth.max_proximity,
    depthVariation: rawDepth.depthVariation ?? rawDepth.depth_variation,
    validPixelCoverage: rawDepth.validPixelCoverage ?? rawDepth.valid_pixel_coverage,
    method: rawDepth.method,
  } : undefined;

  const rawConf = e.confidence;
  const confidence = rawConf ? {
    detectionConfidence: rawConf.detectionConfidence ?? rawConf.detection_confidence,
    segmentationConfidence: rawConf.segmentationConfidence ?? rawConf.segmentation_confidence,
    reconstructionConfidence: rawConf.reconstructionConfidence ?? rawConf.reconstruction_confidence,
    notes: rawConf.notes,
  } : undefined;

  return {
    id: e.id,
    entityType: e.entityType || e.entity_type || "person",
    sourceImageId: e.sourceImageId || e.source_image_id || "",
    name: e.name,
    boundingBox,
    transform,
    geometryRef: e.geometryRef ?? e.geometry_ref ?? null,
    materialRef: e.materialRef ?? e.material_ref ?? null,
    skeletonRef: e.skeletonRef ?? e.skeleton_ref ?? null,
    animationState,
    visibility: e.visibility !== undefined ? e.visibility : true,
    reconstructionStatus: e.reconstructionStatus || e.reconstruction_status || "unstarted",
    confidence,
    detection: e.detection,
    segmentation: e.segmentation,
    depth,
    reconstruction,
    placement,
    sceneNodeId: e.sceneNodeId ?? e.scene_node_id,
    skeleton,
    rig,
    animationClips,
    showSkeleton: e.showSkeleton ?? e.show_skeleton ?? false,
  };
}

export function normalizeScene(raw: any): Scene {
  if (!raw) return raw;
  const rawSource = raw.sourceImage || raw.source_image || {};
  const width = rawSource.width || 1920;
  const height = rawSource.height || 1080;
  const sourceImage: SourceImageMetadata = {
    assetId: rawSource.assetId || rawSource.asset_id || "photo-source",
    filename: rawSource.filename || "photograph.jpg",
    width,
    height,
    aspectRatio: rawSource.aspectRatio ?? rawSource.aspect_ratio ?? (height ? width / height : 1.777),
    mimeType: rawSource.mimeType || rawSource.mime_type || "image/jpeg",
    sizeBytes: rawSource.sizeBytes ?? rawSource.size_bytes ?? 0,
    checksumSha256: rawSource.checksumSha256 || rawSource.checksum_sha256,
    uploadedAt: rawSource.uploadedAt || rawSource.uploaded_at || new Date().toISOString(),
  };

  const rawEntities = Array.isArray(raw.entities) ? raw.entities : [];
  const entities = rawEntities.map((e: any) => normalizeEntity(e));

  const rawReps = raw.representations || {};
  const representations: SceneRepresentations = {
    photo: normalizeRepresentation(rawReps.photo),
    spatial_photo: normalizeRepresentation(rawReps.spatial_photo),
    wireframe: normalizeRepresentation(rawReps.wireframe),
    solid3d: normalizeRepresentation(rawReps.solid3d),
    articulated: rawReps.articulated ? normalizeRepresentation(rawReps.articulated) : undefined,
    combined: rawReps.combined ? normalizeRepresentation(rawReps.combined) : undefined,
  };

  let depth = undefined;
  const rawDepth = raw.depth;
  if (rawDepth) {
    depth = {
      status: rawDepth.status || "unavailable",
      depthAssetId: rawDepth.depthAssetId ?? rawDepth.depth_asset_id ?? null,
      depthUri: rawDepth.depthUri ?? rawDepth.depth_uri ?? null,
      visualizationAssetId: rawDepth.visualizationAssetId ?? rawDepth.visualization_asset_id ?? null,
      visualizationUri: rawDepth.visualizationUri ?? rawDepth.visualization_uri ?? null,
      model: rawDepth.model || "depth-anything-v2-small",
      modelFamily: rawDepth.modelFamily || rawDepth.model_family || "Depth-Anything-V2",
      depthConvention: rawDepth.depthConvention || rawDepth.depth_convention || "relative_proximity",
      normalization: rawDepth.normalization,
      width: rawDepth.width || width,
      height: rawDepth.height || height,
      device: rawDepth.device || "mps",
      analyzedAt: rawDepth.analyzedAt ?? rawDepth.analyzed_at,
      error: rawDepth.error ?? null,
    };
  }

  const rawCoord = raw.coordinateSystem || raw.coordinate_system || {};
  const coordinateSystem = {
    handedness: rawCoord.handedness || "right-handed",
    upAxis: rawCoord.upAxis || rawCoord.up_axis || "+Y",
    forwardAxis: rawCoord.forwardAxis || rawCoord.forward_axis || "+Z",
    rightAxis: rawCoord.rightAxis || rawCoord.right_axis || "+X",
    unit: rawCoord.unit || "meters",
  };

  return {
    id: raw.id,
    schemaVersion: raw.schemaVersion || raw.schema_version || "1.0.0",
    name: raw.name || `Scene-${raw.id.slice(0, 8)}`,
    createdAt: raw.createdAt || raw.created_at || new Date().toISOString(),
    updatedAt: raw.updatedAt || raw.updated_at || new Date().toISOString(),
    sourceImage,
    coordinateSystem,
    entities,
    representations,
    depth,
    sceneGraph: raw.sceneGraph || raw.scene_graph,
    assemblyReport: raw.assemblyReport || raw.assembly_report,
    processingStatus: raw.processingStatus || raw.processing_status || "unprocessed",
    activeJobId: raw.activeJobId !== undefined ? raw.activeJobId : (raw.active_job_id ?? null),
  };
}
