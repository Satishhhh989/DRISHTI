# DRISHTI Processing Event Contract & Synchronization Protocol

**Version:** 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)

## 1. Overview

The backend processing pipeline emits typed, strictly-sequenced events as operations progress. This event protocol ensures:
1. **Deterministic Ordering**: Every event includes a strictly monotonically increasing sequence number ($1, 2, 3, \dots$) scoped to the job.
2. **Real-Time Server-Sent Events (SSE)**: Clients subscribe directly to high-throughput, low-latency live events:
   ```http
   GET /api/v1/jobs/{job_id}/stream?since={lastSequenceNumber}
   ```
   Supports the standard `Last-Event-ID` header and automatically replays historical events upon reconnection.
3. **Reconciliation & Catch-Up Polling**: Polling endpoint available for non-streaming clients or fallback:
   ```http
   GET /api/v1/jobs/{job_id}/events?since={lastSequenceNumber}
   ```
4. **Cancellation Endpoint**: Safely aborts pending or in-flight processing jobs without discarding validated assets:
   ```http
   POST /api/v1/jobs/{job_id}/cancel
   ```
5. **Bounded Retention**: In-memory event buffers are capped at a maximum of 500 events per job with FIFO eviction to prevent memory bloat.
6. **Decoupled Architecture**: Backend processing state is decoupled from frontend animation state. UI transitions do not serve as proof of inference completion.

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

## 3. Phase 03 Pipeline Stages

The Phase 03 Depth Estimation & Spatial Photography pipeline runs 12 sequential stages:

| Stage Identifier | Progress Weight | Description |
|---|---|---|
| `image_validation` | 8% | Validates file format, checks against decompression bombs, ensures integrity. |
| `image_preparation` | 15% | Resolves local path, normalizes EXIF orientation, extracts dimensions. |
| `person_detection` | 30% | Runs local YOLOv8 neural network inference to isolate person bounding boxes. |
| `detection_validation` | 38% | Filters low-confidence detections, validates coordinates, assigns entity IDs. |
| `instance_segmentation` | 50% | Generates individual instance masks and resizes to native photo resolution. |
| `mask_validation` | 58% | Verifies binary mask boundaries, calculates pixel area and coverage statistics. |
| `depth_estimation` | 72% | Executes Depth Anything V2 Small monocular depth estimation across the scene. |
| `depth_normalization` | 80% | Robust 2nd/98th percentile clamping and inversion to canonical proximity $[0, 1]$. |
| `depth_association` | 86% | Computes per-entity median/min/max proximity stats using segmentation masks. |
| `spatial_photo_preparation` | 90% | Prepares 3D spatial mesh displacement properties and parallax camera boundaries. |
| `asset_persistence` | 95% | Encodes and writes PNG masks, `.npy` raw depth arrays, and grayscale PNG visualizations. |
| `scene_update` | 98% | Atomically updates the scene graph with entities, depth metadata, and spatial representations. |
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
   `POST /api/v1/pipeline/analyze` with `{ "sceneId": "...", "assetId": "...", "includeDepth": true }`
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
13. **Seq #13 (`stage_started`)**: `depth_estimation` running Depth Anything V2.
14. **Seq #14 (`stage_completed`)**: Depth estimated.
15. **Seq #15 (`stage_started`)**: `depth_normalization` robust percentile clamping.
16. **Seq #16 (`stage_completed`)**: Depth normalized to $[0.0, 1.0]$.
17. **Seq #17 (`stage_started`)**: `depth_association` associating depths to entities.
18. **Seq #18 (`stage_completed`)**: Depth associated with 2 entities.
19. **Seq #19 (`stage_started`)**: `spatial_photo_preparation` configured mesh and camera bounds.
20. **Seq #20 (`asset_ready`)**: Depth map `.npy` saved.
21. **Seq #22 (`asset_ready`)**: Depth visualization `.png` saved.
22. **Seq #23 (`stage_completed`)**: Scene updated.
23. **Seq #24 (`job_completed`)**: Pipeline complete.
