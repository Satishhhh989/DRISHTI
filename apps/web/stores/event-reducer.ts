/**
 * DRISHTI - Processing Event Reducer
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 * 
 * Pure, deterministic reducer function that applies strictly-sequenced processing events
 * to frontend state. Guarantees idempotency, preserves unaffected state, handles partial failures,
 * and tracks stage progression without disrupting camera or active selection.
 */

import { Scene, SceneEntity, VisualizationMode } from "@photo-to-3d/scene-schema";
import { ProcessingEvent } from "@photo-to-3d/shared-types";

export interface StageProgressInfo {
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  message?: string;
  progress?: number;
  error?: string;
}

export interface ReducerState {
  currentScene: Scene | null;
  activeJobId: string | null;
  lastSequenceNumber: number;
  isAnalyzing: boolean;
  analysisProgress: number;
  currentStageMessage: string | null;
  depthVisualizationUrl: string | null;
  visualizationMode: VisualizationMode;
  stages: Record<string, StageProgressInfo>;
  partialFailures: Array<{ stage: string; message: string }>;
}

/**
 * Applies a single typed processing event to the state slice.
 */
export function reduceProcessingEvent<T extends ReducerState>(
  state: T,
  event: ProcessingEvent
): T {
  // 1. Validation: Reject event if addressed to a different scene
  if (
    event.sceneId &&
    state.currentScene &&
    event.sceneId !== state.currentScene.id &&
    !state.currentScene.id.startsWith("local-")
  ) {
    return state;
  }

  // 2. Idempotency: Reject duplicate or out-of-order sequence numbers
  if (event.sequenceNumber <= state.lastSequenceNumber) {
    return state;
  }

  const nextSequenceNumber = event.sequenceNumber;
  const nextStages = { ...state.stages };
  let nextCurrentScene = state.currentScene ? { ...state.currentScene } : null;
  if (nextCurrentScene && nextCurrentScene.id.startsWith("local-") && event.sceneId) {
    nextCurrentScene.id = event.sceneId;
  }
  let nextDepthVisUrl = state.depthVisualizationUrl;
  let nextIsAnalyzing = state.isAnalyzing;
  let nextProgress = state.analysisProgress;
  let nextMessage = event.message || state.currentStageMessage;
  let nextActiveJobId = state.activeJobId;
  const nextPartialFailures = [...state.partialFailures];

  // Update stage status record
  nextStages[event.stageIdentifier] = {
    status:
      event.eventType === "stage_completed"
        ? "completed"
        : event.eventType === "stage_failed"
        ? "failed"
        : event.eventType === "job_cancelled"
        ? "cancelled"
        : "running",
    message: event.message,
    progress: event.progress,
    error: event.error?.message,
  };

  if (event.progress !== undefined && event.progress !== null) {
    nextProgress = event.progress;
  }

  switch (event.eventType) {
    case "job_accepted": {
      nextIsAnalyzing = true;
      nextActiveJobId = event.jobId;
      nextMessage = event.message || "Analysis job accepted";
      break;
    }

    case "stage_started": {
      nextIsAnalyzing = true;
      nextMessage = event.message || `Starting ${event.stageIdentifier}...`;
      break;
    }

    case "progress_updated": {
      nextMessage = event.message || nextMessage;
      break;
    }

    case "intermediate_result_available": {
      nextMessage = event.message || nextMessage;
      break;
    }

    case "asset_ready": {
      nextMessage = event.message || nextMessage;
      const assetRef = event.assetReference;

      if (assetRef && nextCurrentScene) {
        // Individual segmentation mask ready
        if (assetRef.assetType === "segmentation_mask" && event.entityId) {
          const updatedEntities = nextCurrentScene.entities.map((entity) => {
            if (entity.id === event.entityId) {
              return {
                ...entity,
                segmentation: {
                  status: "ready" as const,
                  maskAssetId: assetRef.assetId,
                  maskUri: assetRef.uri,
                  segmentedAt: event.timestamp,
                },
              };
            }
            return entity;
          });
          nextCurrentScene.entities = updatedEntities;
        }

        // Reconstructed 3D Human Mesh ready
        if (assetRef.assetType === "mesh") {
          const targetEntityId = event.entityId || nextCurrentScene.entities.find((e) => assetRef.uri.includes(e.id))?.id;
          if (targetEntityId) {
            const updatedEntities = nextCurrentScene.entities.map((entity) => {
              if (entity.id === targetEntityId) {
                return {
                  ...entity,
                  geometryRef: assetRef.assetId,
                  reconstruction: {
                    status: "ready" as const,
                    meshAssetId: assetRef.assetId,
                    meshUrl: assetRef.uri,
                    textureAvailability: "none" as const,
                    reconstructedAt: event.timestamp,
                  },
                };
              }
              return entity;
            });
            nextCurrentScene.entities = updatedEntities;
            if (nextCurrentScene.representations) {
              nextCurrentScene.representations.wireframe = {
                status: "ready",
                updatedAt: event.timestamp,
              };
              nextCurrentScene.representations.solid3d = {
                status: "ready",
                updatedAt: event.timestamp,
              };
              nextCurrentScene.representations.combined = {
                status: "ready",
                updatedAt: event.timestamp,
              };
            }
          }
        }

        // Skeleton ready
        if (assetRef.assetType === "skeleton") {
          const targetEntityId = event.entityId || nextCurrentScene.entities.find((e) => assetRef.uri.includes(e.id))?.id;
          if (targetEntityId) {
            const updatedEntities: SceneEntity[] = nextCurrentScene.entities.map((entity) => {
              if (entity.id === targetEntityId) {
                return {
                  ...entity,
                  skeletonRef: assetRef.assetId,
                  skeleton: entity.skeleton || {
                    id: assetRef.assetId,
                    entityId: entity.id,
                    schemaVersion: "1.0.0",
                    rootJointId: "joint_00_pelvis",
                    joints: [],
                    bones: [],
                    coordinateConvention: { up: "+Y", forward: "+Z", right: "+X", units: "meters" },
                    source: "smpl_regressed",
                    isAligned: true,
                    validationStatus: "valid",
                    validationErrors: [],
                    createdAt: event.timestamp,
                  },
                };
              }
              return entity;
            });
            nextCurrentScene.entities = updatedEntities;
          }
        }

        // Rig ready
        if (assetRef.assetType === "rig") {
          const targetEntityId = event.entityId || nextCurrentScene.entities.find((e) => assetRef.uri.includes(e.id))?.id;
          if (targetEntityId) {
            const updatedEntities: SceneEntity[] = nextCurrentScene.entities.map((entity) => {
              if (entity.id === targetEntityId) {
                return {
                  ...entity,
                  rig: entity.rig || {
                    id: assetRef.assetId,
                    entityId: entity.id,
                    skeletonId: entity.skeleton?.id || `skel-${entity.id}`,
                    status: "rigged",
                    skinningMethod: "smpl_lbs_native",
                    maxInfluencesPerVertex: 4,
                    weightsNormalized: true,
                    boneCount: 24,
                    riggedAt: event.timestamp,
                  },
                };
              }
              return entity;
            });
            nextCurrentScene.entities = updatedEntities;
            if (nextCurrentScene.representations) {
              nextCurrentScene.representations.articulated = {
                status: "ready",
                updatedAt: event.timestamp,
              };
              nextCurrentScene.representations.combined = {
                status: "ready",
                updatedAt: event.timestamp,
              };
            }
          }
        }

        // Animation ready
        if (assetRef.assetType === "animation") {
          const targetEntityId = event.entityId || nextCurrentScene.entities.find((e) => assetRef.uri.includes(e.id))?.id;
          if (targetEntityId) {
            const updatedEntities = nextCurrentScene.entities.map((entity) => {
              if (entity.id === targetEntityId) {
                return {
                  ...entity,
                  animationState: {
                    ...entity.animationState,
                    currentClip: entity.animationState?.currentClip || "idle_breathing",
                    // Default to false: reconstructed 3D characters preserve the static photograph pose by default. Animation is strictly opt-in.
                    isPlaying: false,
                  },
                };
              }
              return entity;
            });
            nextCurrentScene.entities = updatedEntities;
          }
        }

        // Depth visualization PNG ready
        if (assetRef.assetType === "depth_visualization") {
          nextDepthVisUrl = assetRef.uri;
          if (!nextCurrentScene.depth) {
            nextCurrentScene.depth = {
              status: "ready",
              visualizationAssetId: assetRef.assetId,
              visualizationUri: assetRef.uri,
              model: "depth-anything-v2-small",
              modelFamily: "Depth-Anything-V2",
              depthConvention: "relative_proximity",
              width: nextCurrentScene.sourceImage.width,
              height: nextCurrentScene.sourceImage.height,
              device: "local",
            };
          } else {
            nextCurrentScene.depth = {
              ...nextCurrentScene.depth,
              status: "ready",
              visualizationAssetId: assetRef.assetId,
              visualizationUri: assetRef.uri,
            };
          }
        }
      }
      break;
    }

    case "stage_completed": {
      nextStages[event.stageIdentifier].status = "completed";
      nextMessage = event.message || `${event.stageIdentifier} completed.`;

      // If spatial photo preparation completed, update representation
      if (
        event.stageIdentifier === "spatial_photo_preparation" &&
        nextCurrentScene?.representations
      ) {
        nextCurrentScene.representations.spatial_photo = {
          ...nextCurrentScene.representations.spatial_photo,
          status: "ready",
          uri: nextDepthVisUrl || nextCurrentScene.representations.spatial_photo.uri,
          updatedAt: event.timestamp,
        };
      }

      // If human reconstruction completed, enable wireframe & solid3d
      if (
        (event.stageIdentifier === "human_reconstruction" ||
          event.stageIdentifier === "mesh_reconstruction") &&
        nextCurrentScene?.representations
      ) {
        nextCurrentScene.representations.wireframe = {
          status: "ready",
          updatedAt: event.timestamp,
        };
        nextCurrentScene.representations.solid3d = {
          status: "ready",
          updatedAt: event.timestamp,
        };
      }
      break;
    }

    case "stage_failed": {
      nextStages[event.stageIdentifier].status = "failed";
      if (event.error?.message) {
        nextPartialFailures.push({
          stage: event.stageIdentifier,
          message: event.error.message,
        });
      }

      // If depth failed, update depth representation without destroying detection or mask results
      if (event.stageIdentifier === "depth_estimation" && nextCurrentScene) {
        if (!nextCurrentScene.depth) {
          nextCurrentScene.depth = {
            status: "failed",
            model: "depth-anything-v2-small",
            modelFamily: "Depth-Anything-V2",
            depthConvention: "relative_proximity",
            width: nextCurrentScene.sourceImage.width,
            height: nextCurrentScene.sourceImage.height,
            device: "local",
            error: event.error?.message,
          };
        } else {
          nextCurrentScene.depth = {
            ...nextCurrentScene.depth,
            status: "failed",
            error: event.error?.message,
          };
        }
        if (nextCurrentScene.representations) {
          nextCurrentScene.representations.spatial_photo = {
            ...nextCurrentScene.representations.spatial_photo,
            status: "failed",
            error: event.error?.message,
          };
        }
      }

      // If human reconstruction failed, record representation status
      if (
        (event.stageIdentifier === "human_reconstruction" ||
          event.stageIdentifier === "mesh_reconstruction") &&
        nextCurrentScene?.representations
      ) {
        nextCurrentScene.representations.wireframe = {
          status: "failed",
          error: event.error?.message,
          updatedAt: event.timestamp,
        };
        nextCurrentScene.representations.solid3d = {
          status: "failed",
          error: event.error?.message,
          updatedAt: event.timestamp,
        };
      }

      // If overall pipeline execution failed
      if (event.status === "failed") {
        nextIsAnalyzing = false;
        nextMessage = event.error?.message || "Pipeline execution error";
      }
      break;
    }

    case "job_cancelled": {
      nextIsAnalyzing = false;
      nextActiveJobId = null;
      nextMessage = "Processing cancelled by user.";
      if (nextCurrentScene) {
        nextCurrentScene.processingStatus =
          nextCurrentScene.entities.length > 0 || nextCurrentScene.depth?.status === "ready"
            ? "ready"
            : "unprocessed";
      }
      break;
    }

    case "job_completed": {
      nextIsAnalyzing = false;
      nextActiveJobId = null;
      nextProgress = 1.0;
      nextMessage = event.message || "Analysis complete.";
      if (nextCurrentScene) {
        nextCurrentScene.processingStatus = "ready";
        if (nextCurrentScene.representations) {
          if (nextCurrentScene.entities.some((e) => e.reconstruction?.status === "ready")) {
            nextCurrentScene.representations.wireframe = {
              status: "ready",
              updatedAt: event.timestamp,
            };
            nextCurrentScene.representations.solid3d = {
              status: "ready",
              updatedAt: event.timestamp,
            };
            nextCurrentScene.representations.combined = {
              status: "ready",
              updatedAt: event.timestamp,
            };
          }
          if (nextCurrentScene.entities.some((e) => e.rig?.status === "rigged")) {
            nextCurrentScene.representations.articulated = {
              status: "ready",
              updatedAt: event.timestamp,
            };
          }
        }
      }
      break;
    }
  }

  return {
    ...state,
    currentScene: nextCurrentScene,
    activeJobId: nextActiveJobId,
    lastSequenceNumber: nextSequenceNumber,
    isAnalyzing: nextIsAnalyzing,
    analysisProgress: nextProgress,
    currentStageMessage: nextMessage,
    depthVisualizationUrl: nextDepthVisUrl,
    stages: nextStages,
    partialFailures: nextPartialFailures,
  };
}
