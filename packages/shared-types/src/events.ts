/**
 * DRISHTI - Processing Event Contract
 * Version 1.1.0
 * 
 * Defines the event-driven synchronization contract between
 * backend processing jobs and connected frontends.
 */

export type ProcessingEventType =
  | "job_accepted"
  | "stage_started"
  | "progress_updated"
  | "intermediate_result_available"
  | "asset_ready"
  | "stage_completed"
  | "stage_failed"
  | "job_cancelled"
  | "job_completed";

export type PipelineStageIdentifier =
  | "image_analysis"
  | "image_validation"
  | "image_preparation"
  | "person_detection"
  | "detection_validation"
  | "segmentation"
  | "instance_segmentation"
  | "mask_validation"
  | "asset_persistence"
  | "scene_update"
  | "completed"
  | "depth_estimation"
  | "spatial_photo"
  | "mesh_reconstruction"
  | "skeleton_fitting"
  | "animation_preparation"
  | "scene_assembly";

export type JobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface AssetReference {
  assetId: string;
  assetType: 
    | "original_photo"
    | "segmentation_mask"
    | "depth_map"
    | "mesh"
    | "texture"
    | "skeleton"
    | "export";
  uri: string;
  mimeType: string;
  sizeBytes?: number;
}

export interface StructuredError {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
  stage?: PipelineStageIdentifier;
}

export interface ProcessingEvent {
  eventId: string;
  jobId: string;
  sceneId: string;
  eventType: ProcessingEventType;
  timestamp: string; // ISO 8601
  sequenceNumber: number; // Deterministic strictly-increasing sequence
  stageIdentifier: PipelineStageIdentifier;
  status: JobStatus;
  progress?: number; // 0.0 to 1.0
  message?: string;
  entityId?: string | null;
  assetReference?: AssetReference | null;
  error?: StructuredError | null;
}

/**
 * Event query parameters for catch-up / reconciliation
 */
export interface EventStreamQuery {
  sinceSequence?: number;
  jobId?: string;
  sceneId?: string;
}
