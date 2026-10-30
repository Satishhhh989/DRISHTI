/**
 * PHOTO → 3D ENGINE - Local Backend API Client
 */

import { HealthResponse, VersionResponse, AssetUploadResponse } from "@photo-to-3d/shared-types";
import { Scene } from "@photo-to-3d/scene-schema";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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
}

export const apiClient = new ApiClient();
