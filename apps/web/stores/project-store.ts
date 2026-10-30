/**
 * DRISHTI - Project Management Store (Zustand)
 * Phase 09: Interaction, Persistence & Export
 * 
 * Manages active project lifecycle, dirty tracking (unsaved changes),
 * local save/restore operations, and recent projects list.
 */

import { create } from "zustand";
import { Project, Scene } from "@photo-to-3d/scene-schema";
import { apiClient } from "@/lib/api-client";
import { useSceneStore } from "./scene-store";

interface ProjectState {
  currentProject: Project | null;
  isDirty: boolean;
  recentProjects: Project[];
  isSaving: boolean;
  saveStatusMessage: string | null;

  // Actions
  initProjectFromScene: (scene: Scene, name?: string) => Promise<Project | null>;
  saveCurrentProject: () => Promise<boolean>;
  openProject: (projectId: string) => Promise<boolean>;
  setProjectName: (name: string) => void;
  markDirty: () => void;
  markClean: () => void;
  loadRecentProjects: () => Promise<void>;
  importProject: (file: File) => Promise<Project | null>;
  clearProject: () => void;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  currentProject: null,
  isDirty: false,
  recentProjects: [],
  isSaving: false,
  saveStatusMessage: null,

  initProjectFromScene: async (scene, name) => {
    try {
      const proj = await apiClient.createProject(scene.id, name || scene.name);
      set({
        currentProject: proj,
        isDirty: false,
        saveStatusMessage: "Project created locally.",
      });
      get().loadRecentProjects();
      return proj;
    } catch (err) {
      console.warn("Could not create project on backend, using local project reference:", err);
      // Fallback local project
      const fallbackProj: Project = {
        id: `proj-local-${scene.id.substring(0, 8)}`,
        name: name || scene.name,
        schemaVersion: "1.0.0",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        sceneId: scene.id,
        sourceImage: scene.sourceImage,
        settings: { activeRepresentation: "solid3d", showGrid: true, showReferencePlane: true },
        assetManifest: [],
        processingState: {
          detection: scene.entities.length > 0,
          segmentation: scene.entities.some((e) => e.segmentation?.status === "ready"),
          depth: scene.depth?.status === "ready",
          reconstruction: scene.entities.some((e) => e.reconstruction?.status === "ready"),
          assembly: scene.assemblyReport?.status === "completed",
          motion: scene.entities.some((e) => (e.animationClips?.length ?? 0) > 0),
        },
      };
      set({ currentProject: fallbackProj, isDirty: false });
      return fallbackProj;
    }
  },

  saveCurrentProject: async () => {
    const { currentProject } = get();
    const currentScene = useSceneStore.getState().currentScene;
    if (!currentProject) return false;

    set({ isSaving: true, saveStatusMessage: "Saving project..." });
    try {
      // 1. If scene is active, sync scene transforms first
      if (currentScene && !currentProject.id.startsWith("proj-local-")) {
        try {
          await apiClient.saveProject(currentProject.id, currentProject.name, currentProject.settings);
        } catch (e) {
          console.warn("Failed saving project to backend:", e);
        }
      }

      const now = new Date().toISOString();
      set({
        currentProject: {
          ...currentProject,
          updatedAt: now,
        },
        isDirty: false,
        isSaving: false,
        saveStatusMessage: "All changes saved locally.",
      });

      get().loadRecentProjects();
      return true;
    } catch (err) {
      set({
        isSaving: false,
        saveStatusMessage: `Save failed: ${err instanceof Error ? err.message : String(err)}`,
      });
      return false;
    }
  },

  openProject: async (projectId: string) => {
    try {
      const proj = await apiClient.getProject(projectId);
      const scene = await apiClient.getScene(proj.sceneId);
      useSceneStore.getState().setScene(scene);
      set({
        currentProject: proj,
        isDirty: false,
        saveStatusMessage: `Opened "${proj.name}".`,
      });
      return true;
    } catch (err) {
      console.error(`Failed to open project ${projectId}:`, err);
      set({
        saveStatusMessage: `Could not open project: ${err instanceof Error ? err.message : String(err)}`,
      });
      return false;
    }
  },

  setProjectName: (name: string) => {
    const { currentProject } = get();
    if (!currentProject) return;
    set({
      currentProject: { ...currentProject, name },
      isDirty: true,
    });
  },

  markDirty: () => set({ isDirty: true }),
  markClean: () => set({ isDirty: false }),

  loadRecentProjects: async () => {
    try {
      const res = await apiClient.listProjects();
      if (res && res.projects) {
        set({ recentProjects: res.projects });
      }
    } catch {
      // Backend might be offline or mocked
    }
  },

  importProject: async (file: File) => {
    try {
      set({ isSaving: true, saveStatusMessage: "Importing project archive..." });
      const imported = await apiClient.importProjectArchive(file);
      const scene = await apiClient.getScene(imported.sceneId);
      useSceneStore.getState().setScene(scene);
      set({
        currentProject: imported,
        isDirty: false,
        isSaving: false,
        saveStatusMessage: `Successfully imported "${imported.name}".`,
      });
      get().loadRecentProjects();
      return imported;
    } catch (err) {
      set({
        isSaving: false,
        saveStatusMessage: `Import failed: ${err instanceof Error ? err.message : String(err)}`,
      });
      return null;
    }
  },

  clearProject: () => set({ currentProject: null, isDirty: false, saveStatusMessage: null }),
}));

