/**
 * PHOTO → 3D ENGINE - Scene Store (Zustand)
 * Manages active scene metadata, loaded representations, and local photo assets.
 */

import { create } from "zustand";
import { Scene, RepresentationMode, createInitialScene } from "@photo-to-3d/scene-schema";

interface SceneState {
  currentScene: Scene | null;
  activeRepresentation: RepresentationMode;
  localPhotoUrl: string | null;
  photoDimensions: { width: number; height: number; aspect: number } | null;
  isUploading: boolean;

  // Actions
  setScene: (scene: Scene) => void;
  setActiveRepresentation: (mode: RepresentationMode) => void;
  setLocalPhoto: (url: string, width: number, height: number, filename: string) => void;
  setIsUploading: (uploading: boolean) => void;
  clearScene: () => void;
}

export const useSceneStore = create<SceneState>((set) => ({
  currentScene: null,
  activeRepresentation: "photo",
  localPhotoUrl: null,
  photoDimensions: null,
  isUploading: false,

  setScene: (scene) =>
    set({
      currentScene: scene,
      photoDimensions: {
        width: scene.sourceImage.width,
        height: scene.sourceImage.height,
        aspect: scene.sourceImage.aspectRatio,
      },
    }),

  setActiveRepresentation: (mode) => set({ activeRepresentation: mode }),

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
    });
  },

  setIsUploading: (uploading) => set({ isUploading: uploading }),

  clearScene: () =>
    set({
      currentScene: null,
      localPhotoUrl: null,
      photoDimensions: null,
      activeRepresentation: "photo",
    }),
}));

function roundAspect(w: number, h: number): number {
  if (!h) return 1;
  return Math.round((w / h) * 1000) / 1000;
}
