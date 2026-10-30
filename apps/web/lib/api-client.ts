/**
 * DRISHTI - Local Backend API Client
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 */

import { HealthResponse, VersionResponse, AssetUploadResponse, ProcessingEvent } from "@photo-to-3d/shared-types";
import { Scene } from "@photo-to-3d/scene-schema";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface AnalyzeJobResponse {
  job_id: string;
  scene_id: string;
  status: string;
  message: string;
}

export interface CancelJobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface HardwareInfo {
  device: string;
  device_name: string;
  is_accelerated: boolean;
  verified: boolean;
  mps_available?: boolean;
  cuda_available?: boolean;
}

export class ApiClient {
  private base: string;

  constructor(base: string = API_BASE) {
    this.base = base;
  }

  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${this.base}/api/v1/health`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Health check failed with status ${res.status}`);
    }
    return res.json();
  }

  async getVersion(): Promise<VersionResponse> {
    const res = await fetch(`${this.base}/api/v1/version`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Version check failed with status ${res.status}`);
    }
    return res.json();
  }

  async uploadPhoto(file: File, sceneName?: string): Promise<AssetUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("auto_create_scene", "true");
    if (sceneName) {
      formData.append("scene_name", sceneName);
    }

    const res = await fetch(`${this.base}/api/v1/assets/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody?.error?.message || `Upload failed with status ${res.status}`);
    }

    return res.json();
  }

  async getScene(sceneId: string): Promise<Scene> {
    const res = await fetch(`${this.base}/api/v1/scenes/${sceneId}`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch scene ${sceneId}`);
    }
    return res.json();
  }

  async triggerAnalysis(
    sceneId: string,
    confidenceThreshold: number = 0.35,
    useMock: boolean = false,
    includeDepth: boolean = true
  ): Promise<AnalyzeJobResponse> {
    const res = await fetch(`${this.base}/api/v1/pipeline/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_id: sceneId,
        confidence_threshold: confidenceThreshold,
        use_mock: useMock,
        include_depth: includeDepth,
      }),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody?.error?.message || `Failed to trigger analysis (status ${res.status})`);
    }
    return res.json();
  }

  async cancelJob(jobId: string): Promise<CancelJobResponse> {
    const res = await fetch(`${this.base}/api/v1/jobs/${jobId}/cancel`, {
      method: "POST",
    });
    if (!res.ok) {
      throw new Error(`Failed to cancel job ${jobId}`);
    }
    return res.json();
  }

  async getJobEvents(jobId: string, since: number = 0) {
    const res = await fetch(`${this.base}/api/v1/jobs/${jobId}/events?since=${since}`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch events for job ${jobId}`);
    }
    return res.json();
  }

  /**
   * Connects to live SSE stream with automatic fallback to polling.
   * Returns an unsubscribe function.
   */
  subscribeToJobEvents(
    jobId: string,
    since: number = 0,
    onEvent: (event: ProcessingEvent) => void,
    onComplete?: () => void,
    onError?: (err: Error) => void
  ): () => void {
    let closed = false;
    let eventSource: EventSource | null = null;
    let pollTimer: NodeJS.Timeout | null = null;
    let lastSeq = since;

    const handleTerminal = () => {
      closed = true;
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
      if (onComplete) onComplete();
    };

    // Try EventSource if available in browser environment
    if (typeof window !== "undefined" && typeof window.EventSource !== "undefined") {
      try {
        const streamUrl = `${this.base}/api/v1/jobs/${jobId}/stream?since=${lastSeq}`;
        eventSource = new EventSource(streamUrl);

        eventSource.onmessage = (e) => {
          if (closed) return;
          try {
            const ev: ProcessingEvent = JSON.parse(e.data);
            lastSeq = Math.max(lastSeq, ev.sequenceNumber);
            onEvent(ev);

            if (
              ev.eventType === "job_completed" ||
              ev.eventType === "job_cancelled" ||
              (ev.eventType === "stage_failed" && ev.status === "failed")
            ) {
              handleTerminal();
            }
          } catch (parseErr) {
            console.warn("Error parsing event stream payload:", parseErr);
          }
        };

        eventSource.onerror = () => {
          if (closed) return;
          // On SSE connection loss, fallback to polling without failing user
          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }
          startPolling();
        };

        return () => {
          closed = true;
          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
        };
      } catch (e) {
        console.warn("EventSource initialization failed, using polling fallback:", e);
      }
    }

    // Polling fallback mechanism
    function startPolling() {
      if (closed) return;

      const poll = async () => {
        if (closed) return;
        try {
          const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/events?since=${lastSeq}`, {
            cache: "no-store",
          });
          if (res.ok) {
            const data = await res.json();
            if (data.events && data.events.length > 0) {
              for (const ev of data.events) {
                lastSeq = Math.max(lastSeq, ev.sequence_number || ev.sequenceNumber);
                onEvent(ev);
                if (
                  ev.event_type === "job_completed" ||
                  ev.event_type === "job_cancelled" ||
                  (ev.event_type === "stage_failed" && ev.status === "failed")
                ) {
                  handleTerminal();
                  return;
                }
              }
            }
          }
        } catch (err) {
          if (onError && !closed) onError(err instanceof Error ? err : new Error(String(err)));
        }

        if (!closed) {
          pollTimer = setTimeout(poll, 600);
        }
      };

      pollTimer = setTimeout(poll, 100);
    }

    startPolling();

    return () => {
      closed = true;
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    };
  }

  async getHardware(): Promise<HardwareInfo> {
    const res = await fetch(`${this.base}/api/v1/pipeline/hardware`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Hardware endpoint failed with status ${res.status}`);
    }
    return res.json();
  }
}

export const apiClient = new ApiClient();
