/**
 * DRISHTI - Scene Store (Zustand)
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 * 
 * Manages active scene metadata, loaded representations, local photo assets,
 * visualization modes, live stage progression, and event-driven updates.
 */

import { create } from "zustand";
import { Scene, VisualizationMode, RepresentationMode, Transform, createInitialScene } from "@photo-to-3d/scene-schema";
import { ProcessingEvent } from "@photo-to-3d/shared-types";
import { apiClient } from "@/lib/api-client";
import { assetCache } from "@/lib/asset-cache";
import { normalizeScene, normalizeProcessingEvent } from "@/lib/scene-normalizer";
import {
  reduceProcessingEvent,
  StageProgressInfo,
} from "./event-reducer";

export type QualityPreset = "preview" | "high" | "maximum";

interface SceneState {
  currentScene: Scene | null;
  activeRepresentation: RepresentationMode;
  visualizationMode: VisualizationMode;
  qualityPreset: QualityPreset;
  localPhotoUrl: string | null;
  photoDimensions: { width: number; height: number; aspect: number } | null;
  depthStrength: number;
  depthVisualizationUrl: string | null;
  isUploading: boolean;
  isAnalyzing: boolean;
  analysisProgress: number;
  currentStageMessage: string | null;
  activeJobId: string | null;
  lastSequenceNumber: number;
  stages: Record<string, StageProgressInfo>;
  partialFailures: Array<{ stage: string; message: string }>;

  // Actions
  setQualityPreset: (preset: QualityPreset) => void;
  setScene: (scene: Scene) => void;
  reconcileAuthoritativeScene: (scene: Scene) => void;
  applyEvent: (event: ProcessingEvent) => void;
  setActiveRepresentation: (mode: RepresentationMode) => void;
  setVisualizationMode: (mode: VisualizationMode) => void;
  setDepthStrength: (strength: number) => void;
  setDepthVisualizationUrl: (url: string | null) => void;
  setLocalPhoto: (url: string, width: number, height: number, filename: string) => void;
  setIsUploading: (uploading: boolean) => void;
  setAnalysisState: (
    analyzing: boolean,
    message?: string | null,
    progress?: number,
    jobId?: string | null
  ) => void;
  cancelActiveJob: () => Promise<void>;
  toggleEntityVisibility: (entityId: string) => void;
  updateEntityTransform: (entityId: string, transform: Transform) => void;
  resetEntityTransform: (entityId: string) => void;
  setEntityAnimation: (
    entityId: string,
    clipName: string | null,
    isPlaying: boolean,
    playbackSpeed?: number,
    loop?: boolean
  ) => void;
  toggleEntitySkeleton: (entityId: string) => void;
  resetEntityPose: (entityId: string) => void;
  clearScene: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export const useSceneStore = create<SceneState>((set, get) => ({
  currentScene: null,
  activeRepresentation: "photo",
  visualizationMode: "detection",
  qualityPreset: "high",
  localPhotoUrl: null,
  photoDimensions: null,
  depthStrength: 0.08,
  depthVisualizationUrl: null,
  isUploading: false,
  isAnalyzing: false,
  analysisProgress: 0,
  currentStageMessage: null,
  activeJobId: null,
  lastSequenceNumber: 0,
  stages: {},
  partialFailures: [],

  setQualityPreset: (preset: QualityPreset) => set({ qualityPreset: preset }),

  setScene: (scene) => {
    const norm = normalizeScene(scene);
    let depthVisUrl: string | null = null;
    const uri =
      norm.depth?.visualizationUri ||
      (norm.depth as unknown as Record<string, string>)?.visualization_uri;
    if (uri) {
      depthVisUrl = uri.startsWith("http") ? uri : `${API_BASE}${uri}`;
    }

    const photoUrl = norm.sourceImage?.assetId
      ? `${API_BASE}/api/v1/assets/${norm.sourceImage.assetId}/file`
      : get().localPhotoUrl;

    set({
      currentScene: norm,
      localPhotoUrl: photoUrl,
      photoDimensions: {
        width: norm.sourceImage?.width || 1920,
        height: norm.sourceImage?.height || 1080,
        aspect: norm.sourceImage?.aspectRatio || 1.777,
      },
      depthVisualizationUrl: depthVisUrl,
      visualizationMode:
        norm.representations?.spatial_photo?.status === "ready"
          ? "spatial_photo"
          : norm.entities?.length > 0
          ? "detection"
          : "original",
    });
  },

  reconcileAuthoritativeScene: (scene) => {
    const norm = normalizeScene(scene);
    const state = get();
    let depthVisUrl = state.depthVisualizationUrl;
    const uri =
      norm.depth?.visualizationUri ||
      (norm.depth as unknown as Record<string, string>)?.visualization_uri;
    if (uri) {
      depthVisUrl = uri.startsWith("http") ? uri : `${API_BASE}${uri}`;
    }

    const photoUrl = state.localPhotoUrl || (norm.sourceImage?.assetId
      ? `${API_BASE}/api/v1/assets/${norm.sourceImage.assetId}/file`
      : null);

    // Preserve active visualization mode if it is ready, otherwise fall back to original
    set({
      currentScene: norm,
      localPhotoUrl: photoUrl,
      isAnalyzing: false,
      analysisProgress: 1.0,
      activeJobId: null,
      depthVisualizationUrl: depthVisUrl,
      photoDimensions: {
        width: norm.sourceImage?.width || 1920,
        height: norm.sourceImage?.height || 1080,
        aspect: norm.sourceImage?.aspectRatio || 1.777,
      },
    });
  },

  applyEvent: (event) => {
    const norm = normalizeProcessingEvent(event);
    const state = get();
    const updated = reduceProcessingEvent(state, norm);

    // Normalize depth visualization URL if it was updated
    if (
      updated.depthVisualizationUrl &&
      !updated.depthVisualizationUrl.startsWith("http")
    ) {
      updated.depthVisualizationUrl = `${API_BASE}${updated.depthVisualizationUrl}`;
    }

    set(updated);
  },

  setActiveRepresentation: (mode) =>
    set({
      activeRepresentation: mode,
      visualizationMode:
        mode === "photo" ? "detection" : (mode as VisualizationMode),
    }),

  setVisualizationMode: (mode) =>
    set({
      visualizationMode: mode,
      activeRepresentation:
        mode === "wireframe"
          ? "wireframe"
          : mode === "solid3d"
          ? "solid3d"
          : mode === "articulated"
          ? "articulated"
          : mode === "combined"
          ? "combined"
          : "photo",
    }),

  setDepthStrength: (strength) =>
    set({ depthStrength: Math.max(0.0, Math.min(0.25, strength)) }),

  setDepthVisualizationUrl: (url) => set({ depthVisualizationUrl: url }),

  setLocalPhoto: (url, width, height, filename) => {
    const aspect = roundAspect(width, height);
    const sceneId = "local-" + Math.random().toString(36).substring(2, 10);
    const initialScene = createInitialScene({
      id: sceneId,
      name: filename,
      sourceImage: {
        assetId: "local-asset-" + sceneId,
        filename,
        width,
        height,
        aspectRatio: aspect,
        mimeType: "image/jpeg",
        sizeBytes: 0,
        uploadedAt: new Date().toISOString(),
      },
      photoUri: url,
    });

    set({
      localPhotoUrl: url,
      currentScene: initialScene,
      photoDimensions: { width, height, aspect },
      activeRepresentation: "photo",
      visualizationMode: "detection",
      isAnalyzing: false,
      analysisProgress: 0,
      currentStageMessage: null,
      activeJobId: null,
      lastSequenceNumber: 0,
      stages: {},
      partialFailures: [],
    });
  },

  setIsUploading: (uploading) => set({ isUploading: uploading }),

  setAnalysisState: (analyzing, message = null, progress = 0, jobId = null) =>
    set({
      isAnalyzing: analyzing,
      currentStageMessage: message,
      analysisProgress: progress,
      activeJobId: jobId !== undefined ? jobId : get().activeJobId,
      stages: analyzing ? get().stages : {},
      partialFailures: analyzing ? get().partialFailures : [],
    }),

  cancelActiveJob: async () => {
    const { activeJobId } = get();
    if (!activeJobId) return;

    try {
      await apiClient.cancelJob(activeJobId);
    } catch (err) {
      console.warn("Failed to cancel active job:", err);
    } finally {
      set({
        isAnalyzing: false,
        activeJobId: null,
        currentStageMessage: "Analysis cancelled by user.",
      });
    }
  },

  toggleEntityVisibility: (entityId) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const updatedEntities = currentScene.entities.map((e) =>
      e.id === entityId ? { ...e, visibility: e.visibility === false ? true : false } : e
    );
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });
  },

  updateEntityTransform: (entityId, transform) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const updatedEntities = currentScene.entities.map((e) =>
      e.id === entityId ? { ...e, transform } : e
    );
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });
    // Async server sync if scene is stored on backend
    if (!currentScene.id.startsWith("local-")) {
      apiClient.updateEntityTransform(currentScene.id, entityId, transform).catch(() => {});
    }
  },

  resetEntityTransform: (entityId) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const target = currentScene.entities.find((e) => e.id === entityId);
    if (!target) return;

    let defaultTransform: Transform = {
      position: { x: 0, y: 0, z: -1.5 },
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: 1, z: 1 },
    };

    if (target.placement) {
      const coords = target.placement.imageSpaceCoords || [0.5, 0.5];
      const x = (coords[0] - 0.5) * 2.0;
      const z = target.placement.relativeProximity != null
        ? -0.5 - (1.0 - target.placement.relativeProximity) * 3.0
        : -1.5;
      const rot = target.placement.orientationCorrection || [0, 0, 0];
      defaultTransform = {
        position: { x: Math.round(x * 1000) / 1000, y: 0, z: Math.round(z * 1000) / 1000 },
        rotation: { x: rot[0], y: rot[1], z: rot[2] },
        scale: { x: 1, y: 1, z: 1 },
      };
    }

    const updatedEntities = currentScene.entities.map((e) =>
      e.id === entityId ? { ...e, transform: defaultTransform } : e
    );
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });

    if (!currentScene.id.startsWith("local-")) {
      apiClient.resetEntityTransform(currentScene.id, entityId).catch(() => {});
    }
  },

  setEntityAnimation: (entityId, clipName, isPlaying, playbackSpeed = 1.0, loop = true) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const updatedEntities = currentScene.entities.map((e) => {
      if (e.id !== entityId) return e;
      return {
        ...e,
        animationState: {
          ...e.animationState,
          currentClip: clipName !== undefined ? clipName : e.animationState?.currentClip ?? null,
          isPlaying,
          playbackSpeed: playbackSpeed ?? e.animationState?.playbackSpeed ?? 1.0,
          loop: loop ?? e.animationState?.loop ?? true,
        },
      };
    });
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });
  },

  toggleEntitySkeleton: (entityId) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const updatedEntities = currentScene.entities.map((e) =>
      e.id === entityId ? { ...e, showSkeleton: !e.showSkeleton } : e
    );
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });
  },

  resetEntityPose: (entityId) => {
    const { currentScene } = get();
    if (!currentScene) return;
    const updatedEntities = currentScene.entities.map((e) => {
      if (e.id !== entityId) return e;
      return {
        ...e,
        animationState: {
          ...e.animationState,
          isPlaying: false,
        },
      };
    });
    set({
      currentScene: {
        ...currentScene,
        entities: updatedEntities,
      },
    });
  },

  clearScene: () => {
    if (typeof window !== "undefined") {
      assetCache.disposeAll();
    }
    set({
      currentScene: null,
      localPhotoUrl: null,
      photoDimensions: null,
      activeRepresentation: "photo",
      visualizationMode: "original",
      isAnalyzing: false,
      analysisProgress: 0,
      currentStageMessage: null,
      activeJobId: null,
      lastSequenceNumber: 0,
      stages: {},
      partialFailures: [],
    });
  },
}));

function roundAspect(w: number, h: number): number {
  if (!h) return 1;
  return Math.round((w / h) * 1000) / 1000;
}
