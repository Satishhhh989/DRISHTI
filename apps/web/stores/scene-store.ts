/**
 * DRISHTI - Scene Store (Zustand)
 * Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)
 * 
 * Manages active scene metadata, loaded representations, local photo assets,
 * visualization modes, live stage progression, and event-driven updates.
 */

import { create } from "zustand";
import { Scene, VisualizationMode, RepresentationMode, createInitialScene } from "@photo-to-3d/scene-schema";
import { ProcessingEvent } from "@photo-to-3d/shared-types";
import { apiClient } from "@/lib/api-client";
import {
  reduceProcessingEvent,
  StageProgressInfo,
} from "./event-reducer";

interface SceneState {
  currentScene: Scene | null;
  activeRepresentation: RepresentationMode;
  visualizationMode: VisualizationMode;
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
  clearScene: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export const useSceneStore = create<SceneState>((set, get) => ({
  currentScene: null,
  activeRepresentation: "photo",
  visualizationMode: "detection",
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

  setScene: (scene) => {
    let depthVisUrl: string | null = null;
    const uri =
      scene.depth?.visualizationUri ||
      (scene.depth as unknown as Record<string, string>)?.visualization_uri;
    if (uri) {
      depthVisUrl = uri.startsWith("http") ? uri : `${API_BASE}${uri}`;
    }

    set({
      currentScene: scene,
      photoDimensions: {
        width: scene.sourceImage.width,
        height: scene.sourceImage.height,
        aspect: scene.sourceImage.aspectRatio,
      },
      depthVisualizationUrl: depthVisUrl,
      visualizationMode:
        scene.representations.spatial_photo?.status === "ready"
          ? "spatial_photo"
          : scene.entities.length > 0
          ? "detection"
          : "original",
    });
  },

  reconcileAuthoritativeScene: (scene) => {
    const state = get();
    let depthVisUrl = state.depthVisualizationUrl;
    const uri =
      scene.depth?.visualizationUri ||
      (scene.depth as unknown as Record<string, string>)?.visualization_uri;
    if (uri) {
      depthVisUrl = uri.startsWith("http") ? uri : `${API_BASE}${uri}`;
    }

    // Preserve active visualization mode if it is ready, otherwise fall back to original
    set({
      currentScene: scene,
      depthVisualizationUrl: depthVisUrl,
      photoDimensions: {
        width: scene.sourceImage.width,
        height: scene.sourceImage.height,
        aspect: scene.sourceImage.aspectRatio,
      },
    });
  },

  applyEvent: (event) => {
    const state = get();
    const updated = reduceProcessingEvent(state, event);

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

  clearScene: () =>
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
    }),
}));

function roundAspect(w: number, h: number): number {
  if (!h) return 1;
  return Math.round((w / h) * 1000) / 1000;
}
