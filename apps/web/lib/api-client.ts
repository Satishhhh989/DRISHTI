/**
 * DRISHTI - Local Backend API Client
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 */

import { HealthResponse, VersionResponse, AssetUploadResponse, ProcessingEvent } from "@photo-to-3d/shared-types";
import { Scene, Project, ProjectSettings, Vector3D, EulerRotation } from "@photo-to-3d/scene-schema";
import { normalizeScene, normalizeProcessingEvent } from "./scene-normalizer";

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

    const data = await res.json();
    return {
      asset: data.asset,
      scene: data.scene ? normalizeScene(data.scene) : undefined,
    };
  }

  async getScene(sceneId: string): Promise<Scene> {
    const res = await fetch(`${this.base}/api/v1/scenes/${sceneId}`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Failed to fetch scene ${sceneId}`);
    }
    const data = await res.json();
    return normalizeScene(data);
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
    const data = await res.json();
    return {
      ...data,
      events: (data.events || []).map(normalizeProcessingEvent),
    };
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
            const raw = JSON.parse(e.data);
            const ev: ProcessingEvent = normalizeProcessingEvent(raw);
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
              for (const raw of data.events) {
                const ev: ProcessingEvent = normalizeProcessingEvent(raw);
                lastSeq = Math.max(lastSeq, ev.sequenceNumber);
                onEvent(ev);
                if (
                  ev.eventType === "job_completed" ||
                  ev.eventType === "job_cancelled" ||
                  (ev.eventType === "stage_failed" && ev.status === "failed")
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

  // --- Phase 09: Project & Export Methods ---

  async listProjects(): Promise<{ projects: Project[]; total: number }> {
    const res = await fetch(`${this.base}/api/v1/projects`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Failed to list projects (status ${res.status})`);
    }
    return res.json();
  }

  async createProject(sceneId: string, name?: string, settings?: ProjectSettings): Promise<Project> {
    const res = await fetch(`${this.base}/api/v1/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_id: sceneId, name, settings }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to create project (status ${res.status})`);
    }
    return res.json();
  }

  async getProject(projectId: string): Promise<Project> {
    const res = await fetch(`${this.base}/api/v1/projects/${projectId}`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) {
      throw new Error(`Failed to get project ${projectId} (status ${res.status})`);
    }
    return res.json();
  }

  async saveProject(projectId: string, name?: string, settings?: ProjectSettings): Promise<Project> {
    const res = await fetch(`${this.base}/api/v1/projects/${projectId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, settings }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to save project (status ${res.status})`);
    }
    return res.json();
  }

  async updateEntityTransform(
    sceneId: string,
    entityId: string,
    transform: { position?: Vector3D; rotation?: EulerRotation; scale?: Vector3D }
  ): Promise<Scene> {
    const res = await fetch(`${this.base}/api/v1/scenes/${sceneId}/entities/${entityId}/transform`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transform),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to update transform (status ${res.status})`);
    }
    const data = await res.json();
    return normalizeScene(data);
  }

  async resetEntityTransform(sceneId: string, entityId: string): Promise<Scene> {
    const res = await fetch(`${this.base}/api/v1/scenes/${sceneId}/entities/${entityId}/reset-transform`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Failed to reset transform (status ${res.status})`);
    }
    const data = await res.json();
    return normalizeScene(data);
  }

  getMeshExportUrl(sceneId: string, entityId: string, applyTransform: boolean = false): string {
    return `${this.base}/api/v1/export/mesh/${sceneId}/${entityId}?apply_transform=${applyTransform}`;
  }

  getSceneExportUrl(sceneId: string): string {
    return `${this.base}/api/v1/export/scene/${sceneId}`;
  }

  getMetadataExportUrl(sceneId: string): string {
    return `${this.base}/api/v1/export/metadata/${sceneId}`;
  }

  getProjectArchiveUrl(projectId: string): string {
    return `${this.base}/api/v1/export/project/${projectId}/archive`;
  }

  async importProjectArchive(file: File, newName?: string): Promise<Project> {
    const formData = new FormData();
    formData.append("file", file);
    if (newName) {
      formData.append("new_name", newName);
    }
    const res = await fetch(`${this.base}/api/v1/projects/import`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message || `Import failed (status ${res.status})`);
    }
    return res.json();
  }
}

export const apiClient = new ApiClient();
