/**
 * DRISHTI - Reconstruction Controller
 *
 * Coordinates execution of the neural photo-to-3D reconstruction pipeline:
 * - Triggers backend analysis (YOLOv8, Instance Segmentation, Depth Anything V2, HMR 2.0)
 * - Synchronizes with the 9-stage cinematic animation sequence
 * - Manages event streaming and updates authoritative scene state
 * - Defaults to displaying 3D wireframe geometry during and upon completion of reconstruction
 */

import { Scene } from "@photo-to-3d/scene-schema";
import { apiClient } from "@/lib/api-client";
import { useSceneStore } from "@/stores/scene-store";
import { useCinematicStore } from "@/stores/cinematic-store";
import { useUIStore } from "@/stores/ui-store";

export async function executeReconstruction(sceneOverride?: Scene): Promise<void> {
  const sceneState = useSceneStore.getState();
  const uiState = useUIStore.getState();
  const cinematicStore = useCinematicStore.getState();

  const scene = sceneOverride || sceneState.currentScene;
  const localPhotoUrl = sceneState.localPhotoUrl;
  const qualityPreset = sceneState.qualityPreset;
  const hasReconstructedMeshes = Boolean(
    scene?.entities?.some(
      (e) =>
        e.reconstruction?.status === "ready" &&
        Boolean(e.reconstruction?.meshUrl || e.reconstruction?.meshAssetId)
    )
  );

  if (!scene && !localPhotoUrl) {
    uiState.setError("Please upload a photograph first.");
    return;
  }

  // Replay presentation if already reconstructed and not actively analyzing
  if (hasReconstructedMeshes && !sceneState.isAnalyzing && !sceneOverride) {
    cinematicStore.startCinematic();
    return;
  }

  // 1. Immediately launch the cinematic reconstruction animation sequence
  cinematicStore.startCinematic();
  sceneState.setAnalysisState(true, "Initializing neural reconstruction pipeline...", 0.05);
  uiState.setError(null);
  uiState.setStatusNotice("Reconstructing scene (Detection, Segmentation, Depth, HMR 2.0)...");

  // 2. If connected to a real backend scene, trigger pipeline execution
  if (scene && !scene.id.startsWith("local-")) {
    try {
      const job = await apiClient.triggerAnalysis(scene.id, 0.35, false, true, qualityPreset);
      sceneState.setAnalysisState(true, "Job accepted by engine", 0.08, job.job_id);

      apiClient.subscribeToJobEvents(
        job.job_id,
        0,
        (ev) => {
          sceneState.applyEvent(ev);
          if (ev.message) {
            uiState.setStatusNotice(ev.message);
          }
        },
        async () => {
          try {
            const refreshed = await apiClient.getScene(scene.id);
            sceneState.reconcileAuthoritativeScene(refreshed);
            cinematicStore.updateStageStatus("stage_combined", "completed", { progress: 1.0 });

            // Display wireframe upon completion as requested
            sceneState.setVisualizationMode("wireframe");
            uiState.setStatusNotice(
              refreshed.entities.length > 0
                ? `Reconstruction complete: ${refreshed.entities.length} people reconstructed (Wireframe view).`
                : "Reconstruction complete: No people detected."
            );
          } catch {
            // Keep current scene if refresh fails
          }
        },
        (err) => {
          console.warn("Event stream error:", err);
        }
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Reconstruction failed";
      uiState.setError(msg);
      sceneState.setAnalysisState(false, null, 0);
      cinematicStore.skipCinematic();
      sceneState.setVisualizationMode("wireframe");
    }
  } else {
    // Client-local preview mode: animation runs procedurally
    uiState.setStatusNotice("Running reconstruction animation (local preview)...");
  }
}
