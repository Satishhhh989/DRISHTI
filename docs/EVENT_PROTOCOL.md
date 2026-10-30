# Processing Event Contract & Synchronization Protocol

**Version:** 1.0.0

## 1. Overview

The backend processing pipeline emits typed, strictly-sequenced events as operations progress. This event protocol ensures:
1. **Deterministic Ordering**: Every event includes a strictly monotonically increasing sequence number ($1, 2, 3, \dots$) scoped to the job.
2. **Reconciliation & Catch-Up**: If the client disconnects or reloads, it can query all events since its last received sequence number:
   ```http
   GET /api/v1/jobs/{job_id}/events?since={lastSequenceNumber}
   ```
3. **Decoupled Architecture**: Backend processing state is decoupled from frontend animation state. UI transitions do not serve as proof of inference completion.

---

## 2. Event Types

| Event Type | Trigger Condition |
|---|---|
| `job_accepted` | A new background processing job has been registered and enqueued. |
| `stage_started` | A pipeline stage (e.g. `person_detection`) has begun execution. |
| `progress_updated` | Incremental progress (0.0 to 1.0) reported within an active stage. |
| `intermediate_result_available` | An intermediate asset or metadata item has been produced. |
| `asset_ready` | A pipeline asset (e.g. mask, mesh, depth map) is stored and available. |
| `stage_completed` | The current pipeline stage has completed successfully. |
| `stage_failed` | An error occurred during the stage, aborting the process. |
| `job_cancelled` | The job was cancelled by user intervention or timeout. |
| `job_completed` | All pipeline stages have concluded and the scene is updated. |

---

## 3. Event Payload Structure

```typescript
interface ProcessingEvent {
  eventId: string;                    // Unique event UUID
  jobId: string;                      // Target processing job ID
  sceneId: string;                    // Associated scene ID
  eventType: ProcessingEventType;     // One of the defined event types
  timestamp: string;                  // ISO 8601 UTC timestamp
  sequenceNumber: number;             // Monotonically increasing integer (>= 1)
  stageIdentifier: PipelineStage;     // Pipeline stage name
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress?: number;                  // Progress fraction [0.0 - 1.0]
  message?: string;                   // Human-readable status description
  entityId?: string | null;           // Subject entity affected if applicable
  assetReference?: AssetReference;    // Generated asset metadata reference
  error?: StructuredError;            // Error details if stage failed
}
```

---

## 4. Pipeline Stages

The full engine defines nine successive pipeline stages:

1. `image_analysis`: Photo validation, dimension extraction, color analysis.
2. `person_detection`: Multi-person bounding box detection (Phase 02).
3. `segmentation`: Instance alpha masks and background separation (Phase 03).
4. `depth_estimation`: Monocular metric/relative scene depth (Phase 04).
5. `spatial_photo`: Multiplane / depth-displacement spatial photo (Phase 05).
6. `mesh_reconstruction`: 3D human polygon mesh generation (Phase 06).
7. `skeleton_fitting`: Kinematic bone hierarchy fitting (Phase 08).
8. `animation_preparation`: Skinning weight binding and IK rigging (Phase 08).
9. `scene_assembly`: Final integration into the interactive 3D scene (Phase 09).
