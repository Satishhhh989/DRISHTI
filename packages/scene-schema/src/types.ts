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
}

export type RepresentationStatus = 
  | "unavailable" 
  | "pending" 
  | "processing" 
  | "ready" 
  | "failed";

export type RepresentationMode = "photo" | "wireframe" | "solid3d";

export type VisualizationMode = "original" | "detection" | "segmentation" | "wireframe" | "solid3d";

export interface RepresentationAsset {
  status: RepresentationStatus;
  assetId?: string | null;
  uri?: string | null;
  error?: string | null;
  updatedAt: string;
}

export interface SceneRepresentations {
  photo: RepresentationAsset;
  spatial_photo: RepresentationAsset;
  wireframe: RepresentationAsset;
  solid3d: RepresentationAsset;
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
  processingStatus: SceneProcessingStatus;
  activeJobId?: string | null;
}
