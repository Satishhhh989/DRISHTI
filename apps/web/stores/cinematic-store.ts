/**
 * DRISHTI - Cinematic Reconstruction Store (Zustand)
 *
 * Manages the 9-stage cinematic reconstruction timeline, per-person presentation focus,
 * camera direction targets, and synchronization with real pipeline events.
 */

import { create } from "zustand";
import { VisualizationMode } from "@photo-to-3d/scene-schema";

export type CinematicStageId =
  | "stage_photo"          // Stage 1: Original Photograph & Introduction
  | "stage_detection"      // Stage 2: Person Detection
  | "stage_segmentation"   // Stage 3: Segmentation
  | "stage_depth"          // Stage 4: Depth Map
  | "stage_spatial_photo"  // Stage 5: Spatial Photograph
  | "stage_wireframe"      // Stage 6: Wireframe Extraction
  | "stage_solid3d"        // Stage 7: 3D Solid Mesh
  | "stage_articulation"   // Stage 8: Articulation and Rigging
  | "stage_combined";      // Stage 9: Final Combined Scene

export type StageStatus = "waiting" | "processing" | "completed" | "failed" | "skipped";

export interface CinematicStageMeta {
  id: CinematicStageId;
  number: number;
  label: string;
  subsystem: string;
  description: string;
  mode: VisualizationMode;
}

export const CINEMATIC_STAGES: CinematicStageMeta[] = [
  {
    id: "stage_photo",
    number: 1,
    label: "Original Photo",
    subsystem: "Ingestion & Validation",
    description: "Source photograph ingested and validated in 3D coordinate space.",
    mode: "original",
  },
  {
    id: "stage_detection",
    number: 2,
    label: "Person Detection",
    subsystem: "YOLOv8m Neural Detector",
    description: "Locating human candidates with confidence scoring.",
    mode: "detection",
  },
  {
    id: "stage_segmentation",
    number: 3,
    label: "Segmentation",
    subsystem: "Instance Mask Engine",
    description: "Extracting precise human silhouette contours from background.",
    mode: "segmentation",
  },
  {
    id: "stage_depth",
    number: 4,
    label: "Depth Map",
    subsystem: "Depth Anything V2",
    description: "Estimating relative scene depth field and spatial distance.",
    mode: "depth",
  },
  {
    id: "stage_spatial_photo",
    number: 5,
    label: "Spatial Photo",
    subsystem: "Depth Displacement Mesh",
    description: "Single-perspective depth relief reconstruction.",
    mode: "spatial_photo",
  },
  {
    id: "stage_wireframe",
    number: 6,
    label: "Wireframe Extraction",
    subsystem: "HMR 2.0 / SMPL Geometry",
    description: "Separating each person into individual 6,890-vertex wireframe topology.",
    mode: "wireframe",
  },
  {
    id: "stage_solid3d",
    number: 7,
    label: "3D Solid Mesh",
    subsystem: "Projective Color Shading",
    description: "Shading volumetric geometry with photograph-projected vertex colors.",
    mode: "solid3d",
  },
  {
    id: "stage_articulation",
    number: 8,
    label: "Articulation & Rigging",
    subsystem: "Kinematic SMPL Rig",
    description: "Fitting 24 canonical joints and LBS skinning rig to reconstructed pose.",
    mode: "articulated",
  },
  {
    id: "stage_combined",
    number: 9,
    label: "Combined Scene",
    subsystem: "Spatial Scene Assembly",
    description: "Assembled multi-person 3D scene with reference photo staging.",
    mode: "combined",
  },
];

export interface StageRuntimeState {
  status: StageStatus;
  progress: number;
  startTime?: number;
  duration?: number;
  itemCount?: number;
  totalCount?: number;
  summary?: string;
  errorMessage?: string;
}

interface CameraTarget {
  position: [number, number, number];
  target: [number, number, number];
}

interface CinematicState {
  isCinematicActive: boolean;
  isPaused: boolean;
  currentStageId: CinematicStageId;
  stages: Record<CinematicStageId, StageRuntimeState>;
  
  // Per-person focal presentation for Stage 2 (Detection), Stage 3 (Segmentation), Stage 6 (Wireframe)
  activePersonIndex: number;
  totalPersons: number;

  // Real-time animation drivers
  cameraTarget: CameraTarget | null;
  depthWipeProgress: number; // 0.0 to 1.0 for Stage 4 depth reveal sweep
  wireframeLiftProgress: number; // 0.0 to 1.0 for Stage 6 overall
  personExtractionProgress: number; // 0.0 to 1.0 for individual person extraction
  surfaceDissolveProgress: number; // 0.0 to 1.0 for Stage 7 wireframe-to-solid dissolve
  assemblyProgress: number; // 0.0 to 1.0 for Stage 9 step-forward

  // Actions
  startCinematic: () => void;
  pauseCinematic: () => void;
  resumeCinematic: () => void;
  skipCinematic: () => void;
  resetCinematic: () => void;
  setCurrentStage: (stageId: CinematicStageId) => void;
  updateStageStatus: (
    stageId: CinematicStageId,
    status: StageStatus,
    updates?: Partial<StageRuntimeState>
  ) => void;
  setActivePersonIndex: (index: number, total?: number) => void;
  setCameraTarget: (target: CameraTarget | null) => void;
  setDepthWipeProgress: (progress: number) => void;
  setWireframeLiftProgress: (progress: number) => void;
  setPersonExtractionProgress: (progress: number) => void;
  setSurfaceDissolveProgress: (progress: number) => void;
  setAssemblyProgress: (progress: number) => void;
}

const initialStages: Record<CinematicStageId, StageRuntimeState> = {
  stage_photo: { status: "waiting", progress: 0 },
  stage_detection: { status: "waiting", progress: 0 },
  stage_segmentation: { status: "waiting", progress: 0 },
  stage_depth: { status: "waiting", progress: 0 },
  stage_spatial_photo: { status: "waiting", progress: 0 },
  stage_wireframe: { status: "waiting", progress: 0 },
  stage_solid3d: { status: "waiting", progress: 0 },
  stage_articulation: { status: "waiting", progress: 0 },
  stage_combined: { status: "waiting", progress: 0 },
};

export const useCinematicStore = create<CinematicState>((set, get) => ({
  isCinematicActive: false,
  isPaused: false,
  currentStageId: "stage_photo",
  stages: { ...initialStages },

  activePersonIndex: -1,
  totalPersons: 0,

  cameraTarget: null,
  depthWipeProgress: 0.0,
  wireframeLiftProgress: 0.0,
  personExtractionProgress: 0.0,
  surfaceDissolveProgress: 0.0,
  assemblyProgress: 0.0,

  startCinematic: () => {
    set({
      isCinematicActive: true,
      isPaused: false,
      currentStageId: "stage_photo",
      stages: {
        ...initialStages,
        stage_photo: { status: "processing", progress: 0.1, startTime: Date.now() },
      },
      activePersonIndex: -1,
      depthWipeProgress: 0.0,
      wireframeLiftProgress: 0.0,
      personExtractionProgress: 0.0,
      surfaceDissolveProgress: 0.0,
      assemblyProgress: 0.0,
    });
  },

  pauseCinematic: () => set({ isPaused: true }),
  resumeCinematic: () => set({ isPaused: false }),

  skipCinematic: () => {
    // Mark completed up to combined and disable cinematic overlay
    const completedStages: Record<CinematicStageId, StageRuntimeState> = { ...get().stages };
    for (const key of Object.keys(completedStages) as CinematicStageId[]) {
      if (completedStages[key].status !== "failed") {
        completedStages[key] = {
          ...completedStages[key],
          status: "completed",
          progress: 1.0,
        };
      }
    }
    set({
      isCinematicActive: false,
      isPaused: false,
      currentStageId: "stage_combined",
      stages: completedStages,
      cameraTarget: null,
      assemblyProgress: 1.0,
      wireframeLiftProgress: 1.0,
      personExtractionProgress: 1.0,
      surfaceDissolveProgress: 1.0,
      depthWipeProgress: 1.0,
      activePersonIndex: -1,
    });
  },

  resetCinematic: () => {
    set({
      isCinematicActive: false,
      isPaused: false,
      currentStageId: "stage_photo",
      stages: { ...initialStages },
      cameraTarget: null,
      activePersonIndex: -1,
      depthWipeProgress: 0.0,
      wireframeLiftProgress: 0.0,
      personExtractionProgress: 0.0,
      surfaceDissolveProgress: 0.0,
      assemblyProgress: 0.0,
    });
  },

  setCurrentStage: (stageId: CinematicStageId) => {
    const prev = get().stages[stageId];
    set({
      currentStageId: stageId,
      stages: {
        ...get().stages,
        [stageId]: {
          ...prev,
          status: prev.status === "completed" ? "completed" : "processing",
          startTime: prev.startTime || Date.now(),
        },
      },
    });
  },

  updateStageStatus: (stageId, status, updates = {}) => {
    const prev = get().stages[stageId];
    const startTime = prev.startTime || Date.now();
    const duration = status === "completed" || status === "failed" ? Date.now() - startTime : prev.duration;
    
    set({
      stages: {
        ...get().stages,
        [stageId]: {
          ...prev,
          ...updates,
          status,
          duration,
        },
      },
    });
  },

  setActivePersonIndex: (index, total) => {
    set({
      activePersonIndex: index,
      totalPersons: total !== undefined ? total : get().totalPersons,
    });
  },

  setCameraTarget: (target) => set({ cameraTarget: target }),
  setDepthWipeProgress: (progress) => set({ depthWipeProgress: Math.max(0, Math.min(1, progress)) }),
  setWireframeLiftProgress: (progress) => set({ wireframeLiftProgress: Math.max(0, Math.min(1, progress)) }),
  setPersonExtractionProgress: (progress) => set({ personExtractionProgress: Math.max(0, Math.min(1, progress)) }),
  setSurfaceDissolveProgress: (progress) => set({ surfaceDissolveProgress: Math.max(0, Math.min(1, progress)) }),
  setAssemblyProgress: (progress) => set({ assemblyProgress: Math.max(0, Math.min(1, progress)) }),
}));

