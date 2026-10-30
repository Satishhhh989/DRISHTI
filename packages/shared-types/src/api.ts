/**
 * PHOTO → 3D ENGINE - API DTO Types
 */

import { Scene } from "@photo-to-3d/scene-schema";
import { ProcessingEvent } from "./events.js";
import { AssetMetadata } from "./assets.js";

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  service: string;
  version: string;
  timestamp: string;
  storageAvailable: boolean;
}

export interface VersionResponse {
  serviceVersion: string;
  apiVersion: string;
  schemaVersion: string;
}

export interface ApiErrorResponse {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown> | null;
  };
}

export interface CreateSceneRequest {
  name?: string;
  sourceAssetId: string;
}

export interface SceneListResponse {
  scenes: Scene[];
  total: number;
}

export interface AssetUploadResponse {
  asset: AssetMetadata;
  scene?: Scene;
}

export interface JobEventsResponse {
  jobId: string;
  events: ProcessingEvent[];
  latestSequenceNumber: number;
}
