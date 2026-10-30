# DRISHTI Processing Event Contract & Synchronization Protocol

**Version:** 1.0.0

## 1. Overview

The backend processing pipeline emits typed, strictly-sequenced events as operations progress. This event protocol ensures:
1. **Deterministic Ordering**: Every event includes a strictly monotonically increasing sequence number ($1, 2, 3, \dots$) scoped to the job.
2. **Reconciliation & Catch-Up**: If the client disconnects, reloads, or experiences network jitter, it can query all events since its last received sequence number:
   ```http
   GET /api/v1/jobs/{job_id}/events?since={lastSequenceNumber}
   ```
3. **Job Status Polling**: Direct endpoint for retrieving current job status, execution duration, and progress fraction:
   ```http
   GET /api/v1/jobs/{job_id}
   ```
4. **Decoupled Architecture**: Backend processing state is decoupled from frontend animation state. UI transitions do not serve as proof of inference completion.

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

## 3. Phase 02 Pipeline Stages

The Phase 02 Image Intelligence & Person Segmentation pipeline runs 9 sequential stages:

| Stage Identifier | Progress Weight | Description |
|---|---|---|
| `image_validation` | 10% | Validates file format, checks against decompression bombs, ensures integrity. |
| `image_preparation` | 20% | Resolves local path, normalizes EXIF orientation, extracts dimensions. |
| `person_detection` | 40% | Runs local YOLOv8 neural network inference to isolate person bounding boxes. |
| `detection_validation` | 50% | Filters low-confidence detections, validates coordinates, assigns entity IDs. |
| `instance_segmentation` | 70% | Generates individual instance masks and resizes to native photo resolution. |
| `mask_validation` | 80% | Verifies binary mask boundaries, calculates pixel area and coverage statistics. |
| `asset_persistence` | 90% | Encodes and writes 8-bit single-channel PNG masks to `data/storage/masks/`. |
| `scene_update` | 95% | Atomically updates the scene graph with entities, detection, and mask info. |
| `completed` | 100% | Emits final `job_completed` event and notifies clients. |

---

## 4. Event Payload Structure

```typescript
interface ProcessingEvent {
  eventId: string;                    // Unique event UUID
  jobId: string;                      // Target processing job ID
  sceneId: string;                    // Associated scene ID
  eventType: ProcessingEventType;     // One of the defined event types
  timestamp: string;                  // ISO 8601 UTC timestamp
  sequenceNumber: number;             // Monotonically increasing integer (>= 1)
  stageIdentifier: PipelineStage;     // Current pipeline stage name
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress?: number;                  // Progress fraction [0.0 - 1.0]
  message?: string;                   // Human-readable status description
  entityId?: string | null;           // Subject entity affected if applicable
  assetReference?: AssetReference;    // Generated asset metadata reference
  error?: StructuredError;            // Error details if stage failed
}
```

---

## 5. Event Flow Example

1. **Client Triggers Analysis**:
   `POST /api/v1/pipeline/analyze` with `{ "sceneId": "...", "assetId": "..." }`
   Response: `{ "jobId": "...", "status": "accepted" }`
2. **Seq #1 (`job_accepted`)**: Job registered with sequence `1`.
3. **Seq #2 (`stage_started`)**: `image_validation` begins.
4. **Seq #3 (`stage_completed`)**: `image_validation` succeeds.
5. **Seq #4 (`stage_started`)**: `image_preparation` begins.
6. ...
7. **Seq #7 (`stage_started`)**: `person_detection` running on Apple Silicon MPS / CUDA.
8. **Seq #8 (`stage_completed`)**: Found 2 people.
9. **Seq #9 (`stage_started`)**: `instance_segmentation` generating masks.
10. **Seq #10 (`asset_ready`)**: Mask saved for `person-001`.
11. **Seq #11 (`asset_ready`)**: Mask saved for `person-002`.
12. **Seq #12 (`stage_completed`)**: Masks validated.
13. **Seq #13 (`stage_completed`)**: Scene updated.
14. **Seq #14 (`job_completed`)**: Pipeline complete.
