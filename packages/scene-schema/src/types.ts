/**
 * DRISHTI - Scene Schema Types
 * Version 1.1.0 (Phase 02: Image Intelligence & Person Segmentation)
 * 
 * Formal typed schema representation for scenes, entities, detections,
 * segmentations, representations, and coordinate transforms.
 */

export const CURRENT_SCHEMA_VERSION = "1.0.0";

/**
 * Coordinate system convention:
 * - Handedness: Right-Handed (matching Three.js standard)
 * - X Axis: Points Right (+X = right, -X = left)
 * - Y Axis: Points Up (+Y = up, -Y = down)
 * - Z Axis: Points Towards Viewer / Out of screen (+Z = forward/towards viewer, -Z = into screen)
 * - Units: Metric meters (1.0 = 1 meter in real-world scale reference)
 */
export interface CoordinateSystemInfo {
  readonly handedness: "right-handed";
  readonly upAxis: "+Y";
  readonly forwardAxis: "+Z";
  readonly rightAxis: "+X";
  readonly unit: "meters";
}

export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface EulerRotation {
  x: number; // In radians
  y: number; // In radians
  z: number; // In radians
  order?: "XYZ" | "YXZ" | "ZXY" | "ZYX" | "YZX" | "XZY";
}

export interface Transform {
  position: Vector3D;
  rotation: EulerRotation;
  scale: Vector3D;
}

/**
 * Normalized 2D bounding box within image coordinate space [0, 1].
 * (0,0) is top-left, (1,1) is bottom-right.
 */
export interface BoundingBox2D {
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
}

export type EntityType = "person" | "background" | "environment" | "prop";

export type ReconstructionStatus = 
  | "unstarted" 
  | "detecting" 
  | "reconstructing" 
  | "ready" 
  | "failed";

export interface ConfidenceMetadata {
  detectionConfidence?: number;
  segmentationConfidence?: number;
  reconstructionConfidence?: number;
  notes?: string;
}

export interface AnimationState {
  currentClip: string | null;
  isPlaying: boolean;
  playbackSpeed: number;
  loop: boolean;
}

export interface SkeletonJoint {
  id: string;
  name: string;
  index: number;
  parentId?: string | null;
  position: [number, number, number]; // Local position in meters
  rotation: [number, number, number, number]; // Quaternion [qx, qy, qz, qw]
  confidence?: number;
  isObserved: boolean;
}

export interface BoneDefinition {
  id: string;
  headJointId: string;
  tailJointId: string;
  length: number;
}

export interface SkeletonData {
  id: string;
  entityId: string;
  schemaVersion: "1.0.0";
  rootJointId: string;
  joints: SkeletonJoint[];
  bones: BoneDefinition[];
  coordinateConvention: CoordinateConvention;
  source: "smpl_regressed" | "yolo_pose_projected" | "mock_anatomical";
  isAligned: boolean;
  alignmentTransform?: Transform;
  validationStatus: "valid" | "incomplete" | "invalid";
  validationErrors: string[];
  createdAt: string;
}

export interface RigData {
  id: string;
  entityId: string;
  skeletonId: string;
  meshAssetId?: string | null;
  status: "rigged" | "unrigged" | "failed";
  skinningMethod: "smpl_lbs_native" | "proximity_heat_weights" | "none";
  boneCount: number;
  maxInfluencesPerVertex: number;
  weightsNormalized: boolean;
  errorMessage?: string | null;
  riggedAt?: string;
}

export interface JointTransformTrack {
  jointId: string;
  timestamps: number[];
  rotations: [number, number, number, number][]; // Quaternion per keyframe
  positions?: [number, number, number][];
}

export interface AnimationClipData {
  id: string;
  name: string;
  targetEntityId: string;
  targetSkeletonId: string;
  durationSeconds: number;
  isLooping: boolean;
  fps: number;
  provenance: "procedural";
  tracks: JointTransformTrack[];
  createdAt: string;
}

export interface DetectionInfo {
  confidence: number;
  boundingBox: BoundingBox2D;
  pixelBox: [number, number, number, number]; // [xMin, yMin, xMax, yMax] in original image pixels
  model: string;
  detectedAt: string;
  status: "ready" | "failed";
}

export interface SegmentationInfo {
  status: "unstarted" | "processing" | "ready" | "failed";
  maskAssetId?: string | null;
  maskUri?: string | null;
  confidence?: number;
  model?: string;
  segmentedAt?: string;
  pixelCount?: number;
  error?: string | null;
}

export interface GeometryMetadata {
  vertexCount: number;
  faceCount: number;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  dimensions: [number, number, number];
}

export interface ModelProvenance {
  name: string;
  version: string;
  device: string;
  checkpoint?: string;
}

export interface CoordinateConvention {
  up: "+Y";
  forward: "+Z";
  right: "+X";
  units: "meters";
}

export interface HumanReconstructionData {
  status: "none" | "pending" | "processing" | "ready" | "failed";
  meshAssetId?: string | null;
  meshUrl?: string | null;
  metadataAssetId?: string | null;
  modelProvenance?: ModelProvenance;
  geometryMetadata?: GeometryMetadata;
  coordinateConvention?: CoordinateConvention;
  transform?: {
    position: [number, number, number];
    rotation: [number, number, number];
    scale: [number, number, number];
  };
  textureAvailability: "none" | "partial" | "full";
  qualityScore?: number;
  warnings?: string[];
  errorMessage?: string | null;
  reconstructedAt?: string;
}

export interface PlacementProvenance {
  method: string;
  imageSpaceCoords: [number, number]; // [normalized_x, normalized_y]
  depthSource: string;
  relativeProximity?: number | null;
  modelNormalizationApplied: string;
  orientationCorrection: [number, number, number]; // Euler [x, y, z] in radians
  uncertaintyFlags: string[];
  isMetric: boolean;
  placedAt?: string;
}

export interface SceneGraphNode {
  id: string;
  name: string;
  nodeType: "root" | "camera" | "light" | "reference_plane" | "ground_plane" | "person";
  entityId?: string | null;
  transform: Transform;
  children: string[];
  visible: boolean;
  meshAssetId?: string | null;
  meshUri?: string | null;
  provenance?: PlacementProvenance | null;
}

export interface SceneGraph {
  rootId: string;
  nodes: Record<string, SceneGraphNode>;
}

export interface SceneAssemblyReport {
  status: "completed" | "partial" | "failed";
  assembledAt: string;
  totalEntities: number;
  reconstructedCount: number;
  unreconstructedCount: number;
  occlusionWarnings: string[];
  placementMethod: string;
}

export interface SceneEntity {
  id: string;
  entityType: EntityType;
  sourceImageId: string;
  name?: string;
  boundingBox?: BoundingBox2D;
  transform: Transform;
  geometryRef?: string | null; // Asset ID for reconstructed polygon mesh
  materialRef?: string | null; // Asset ID for texture / PBR material
  skeletonRef?: string | null; // Asset ID for rig / joint hierarchy
  animationState: AnimationState;
  visibility: boolean;
  reconstructionStatus: ReconstructionStatus;
  confidence?: ConfidenceMetadata;
  detection?: DetectionInfo;
  segmentation?: SegmentationInfo;
  depth?: EntityDepthInfo;
  reconstruction?: HumanReconstructionData;
  placement?: PlacementProvenance;
  sceneNodeId?: string;
  skeleton?: SkeletonData;
  rig?: RigData;
  animationClips?: AnimationClipData[];
  showSkeleton?: boolean;
}

export type RepresentationStatus = 
  | "unavailable" 
  | "pending" 
  | "processing" 
  | "ready" 
  | "failed";

export type RepresentationMode =
  | "photo"
  | "wireframe"
  | "solid3d"
  | "articulated"
  | "combined";

export type VisualizationMode =
  | "original"
  | "detection"
  | "segmentation"
  | "depth"
  | "spatial_photo"
  | "wireframe"
  | "solid3d"
  | "articulated"
  | "combined";

export type RepresentationCategory =
  | "2d_photo"
  | "spatial_photo"
  | "inspection"
  | "3d_mesh"
  | "hybrid";

export interface RepresentationRegistryEntry {
  id: VisualizationMode;
  displayName: string;
  category: RepresentationCategory;
  requiredAssets: string[];
  supportedSceneTypes: string[];
  loadingBehavior: "instant" | "progressive";
  renderingBehavior: string;
  knownLimitations: string[];
  supportsSelection: boolean;
  supportsInteraction: boolean;
  supportsAnimation: boolean;
}

export interface DepthNormalizationMetadata {
  method: "percentile_clamped";
  pMin: number;
  pMax: number;
  rawMin: number;
  rawMax: number;
  modelConvention: string;
  canonicalConvention: "relative_proximity";
}

export interface SceneDepthMetadata {
  status: "unavailable" | "pending" | "processing" | "ready" | "failed";
  depthAssetId?: string | null;
  depthUri?: string | null;
  visualizationAssetId?: string | null;
  visualizationUri?: string | null;
  model: string;
  modelFamily: string;
  depthConvention: "relative_proximity";
  normalization?: DepthNormalizationMetadata;
  width: number;
  height: number;
  device: string;
  analyzedAt?: string;
  error?: string | null;
}

export interface SpatialPhotoConfig {
  defaultDepthStrength: number;
  geometrySubdivisions: [number, number];
  disocclusionHandling: "conservative_bounds" | "edge_clamp";
}

export interface EntityDepthInfo {
  status: "unstarted" | "processing" | "ready" | "failed";
  medianProximity: number; // Normalized canonical proximity [0, 1] (higher = nearer)
  minProximity?: number;
  maxProximity?: number;
  depthVariation?: number;
  validPixelCoverage: number;
  method: "segmentation_mask" | "bounding_box";
  depthAssetId?: string | null;
  error?: string | null;
}

export interface RepresentationAsset {
  status: RepresentationStatus;
  assetId?: string | null;
  uri?: string | null;
  error?: string | null;
  updatedAt: string;
  config?: SpatialPhotoConfig;
}

export interface SceneRepresentations {
  photo: RepresentationAsset;
  spatial_photo: RepresentationAsset;
  wireframe: RepresentationAsset;
  solid3d: RepresentationAsset;
  articulated?: RepresentationAsset;
  combined?: RepresentationAsset;
}

export interface SourceImageMetadata {
  assetId: string;
  filename: string;
  width: number;
  height: number;
  aspectRatio: number;
  mimeType: string;
  sizeBytes: number;
  checksumSha256?: string;
  uploadedAt: string;
}

export type SceneProcessingStatus = 
  | "unprocessed" 
  | "processing" 
  | "ready" 
  | "error";

export interface Scene {
  id: string;
  schemaVersion: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  sourceImage: SourceImageMetadata;
  coordinateSystem: CoordinateSystemInfo;
  entities: SceneEntity[];
  representations: SceneRepresentations;
  depth?: SceneDepthMetadata;
  sceneGraph?: SceneGraph;
  assemblyReport?: SceneAssemblyReport;
  processingStatus: SceneProcessingStatus;
  activeJobId?: string | null;
}
