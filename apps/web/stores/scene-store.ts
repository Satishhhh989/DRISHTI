/**
 * DRISHTI - Scene Store (Zustand)
 * Manages active scene metadata, loaded representations, local photo assets,
 * visualization modes, and detection/segmentation results.
 */

import { create } from "zustand";
import { Scene, VisualizationMode, RepresentationMode, createInitialScene } from "@photo-to-3d/scene-schema";

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

  // Actions
  setScene: (scene: Scene) => void;
  setActiveRepresentation: (mode: RepresentationMode) => void;
  setVisualizationMode: (mode: VisualizationMode) => void;
  setDepthStrength: (strength: number) => void;
  setDepthVisualizationUrl: (url: string | null) => void;
  setLocalPhoto: (url: string, width: number, height: number, filename: string) => void;
  setIsUploading: (uploading: boolean) => void;
  setAnalysisState: (analyzing: boolean, message?: string | null, progress?: number) => void;
  clearScene: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export const useSceneStore = create<SceneState>((set) => ({
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
      // Default visualization: if spatial photo is ready, offer spatial photo or detection
      visualizationMode:
        scene.representations.spatial_photo?.status === "ready"
          ? "spatial_photo"
          : scene.entities.length > 0
          ? "detection"
          : "original",
    });
  },

  setActiveRepresentation: (mode) =>
    set({
      activeRepresentation: mode,
      // Map representation to visualization mode
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
    });
  },

  setIsUploading: (uploading) => set({ isUploading: uploading }),

  setAnalysisState: (analyzing, message = null, progress = 0) =>
    set({
      isAnalyzing: analyzing,
      currentStageMessage: message,
      analysisProgress: progress,
    }),

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
    }),
}));

function roundAspect(w: number, h: number): number {
  if (!h) return 1;
  return Math.round((w / h) * 1000) / 1000;
}
