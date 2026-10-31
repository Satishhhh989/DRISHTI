"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Canvas } from "@react-three/fiber";
import { CameraController } from "./CameraController";
import { GridFloor } from "./GridFloor";
import { ViewportErrorBoundary } from "./ViewportErrorBoundary";
import { PhotoPlane } from "@/components/scene/PhotoPlane";
import { HumanMeshLayer, Unified3DMode } from "@/components/scene/HumanMeshLayer";
import { LiveReconstructionController } from "@/components/scene/LiveReconstructionController";
import { CinematicSceneDirector } from "@/components/scene/CinematicSceneDirector";
import { ReconstructionTimeline } from "@/components/timeline/ReconstructionTimeline";
import { UploadButton } from "@/components/controls/UploadButton";
import { useSceneStore } from "@/stores/scene-store";
import { useCinematicStore } from "@/stores/cinematic-store";
import { useUIStore } from "@/stores/ui-store";
import { getImageDimensions } from "@/lib/image-utils";
import { apiClient } from "@/lib/api-client";
import styles from "./ThreeViewport.module.css";

export const ThreeViewport: React.FC = () => {
  const [mounted, setMounted] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [assemblyProgress, setAssemblyProgress] = useState(1.0);

  const localPhotoUrl = useSceneStore((state) => state.localPhotoUrl);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const activeRepresentation = useSceneStore((state) => state.activeRepresentation);
  const currentScene = useSceneStore((state) => state.currentScene);
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);
  const currentStageId = useCinematicStore((state) => state.currentStageId);
  const setLocalPhoto = useSceneStore((state) => state.setLocalPhoto);
  const setScene = useSceneStore((state) => state.setScene);
  const setIsUploading = useSceneStore((state) => state.setIsUploading);
  const setError = useUIStore((state) => state.setError);
  const setStatusNotice = useUIStore((state) => state.setStatusNotice);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files?.[0];
      if (!file || !file.type.startsWith("image/")) {
        setError("Please drop a valid image file (JPEG, PNG, or WebP)");
        return;
      }

      try {
        setIsUploading(true);
        setError(null);
        setStatusNotice("Processing dropped photograph...");

        const dims = await getImageDimensions(file);
        const url = URL.createObjectURL(file);
        setLocalPhoto(url, dims.width, dims.height, file.name);
        useCinematicStore.getState().startCinematic();

        try {
          const res = await apiClient.uploadPhoto(file);
          if (res.scene) {
            setScene(res.scene);
          }
          setStatusNotice("Photograph loaded into 3D scene.");
        } catch {
          setStatusNotice("Operating in local offline mode.");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load dropped image";
        setError(msg);
      } finally {
        setIsUploading(false);
      }
    },
    [setLocalPhoto, setScene, setIsUploading, setError, setStatusNotice]
  );

  const hasReconstructedMeshes = Boolean(
    currentScene?.entities?.some(
      (e) =>
        e.reconstruction?.status === "ready" &&
        Boolean(e.reconstruction?.meshUrl || e.reconstruction?.meshAssetId)
    )
  );

  if (!mounted) {
    return <div className={styles.viewportContainer} />;
  }

  const isCinematic3DStage =
    isCinematicActive &&
    (currentStageId === "stage_wireframe" ||
      currentStageId === "stage_solid3d" ||
      currentStageId === "stage_articulation" ||
      currentStageId === "stage_combined");

  const is3DMode =
    activeRepresentation === "wireframe" ||
    activeRepresentation === "solid3d" ||
    activeRepresentation === "articulated" ||
    activeRepresentation === "combined" ||
    (isAnalyzing && hasReconstructedMeshes) ||
    isCinematic3DStage;

  return (
    <div
      className={styles.viewportContainer}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      data-testid="viewport-container"
    >
      {/* Honest Placeholders when mode is not yet available */}
      {activeRepresentation === "wireframe" && !hasReconstructedMeshes && (
        <div className={styles.representationNotice} data-testid="wireframe-placeholder">
          <span className={styles.noticeBadge}>Phase 06</span>
          <span>Polygon wireframe geometry not generated yet. Pipeline activates in Phase 06.</span>
        </div>
      )}

      {activeRepresentation === "solid3d" && !hasReconstructedMeshes && (
        <div className={styles.representationNotice} data-testid="solid3d-placeholder">
          <span className={styles.noticeBadge}>Phase 06 / 07</span>
          <span>Solid volumetric human mesh not generated yet. Reconstruction activates in Phase 06.</span>
        </div>
      )}

      {activeRepresentation === "articulated" && !hasReconstructedMeshes && (
        <div className={styles.representationNotice} data-testid="articulated-placeholder">
          <span className={styles.noticeBadge}>Phase 07 / 08</span>
          <span>Articulated character representation requires reconstructed 3D mesh and fitted kinematic rig.</span>
        </div>
      )}

      {activeRepresentation === "combined" && !hasReconstructedMeshes && (
        <div className={styles.representationNotice} data-testid="combined-placeholder">
          <span className={styles.noticeBadge}>Phase 08</span>
          <span>Combined hybrid scene requires both photograph and reconstructed 3D human meshes.</span>
        </div>
      )}

      {/* Empty State: Inviting user to load a photograph */}
      {!localPhotoUrl ? (
        <div className={styles.emptyState} data-testid="empty-viewport-state">
          <div className={`${styles.emptyCard} ${isDragOver ? styles.dragOver : ""}`}>
            <div className={styles.iconContainer}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </div>
            <h2 className={styles.emptyTitle}>Upload a Photograph</h2>
            <p className={styles.emptyDescription}>
              Select or drop an ordinary photograph with multiple people to begin spatial scene reconstruction.
            </p>
            <UploadButton label="Choose Photograph" />
          </div>
        </div>
      ) : (
        <ViewportErrorBoundary>
          <div className={styles.canvasWrapper} data-testid="r3f-canvas-wrapper">
            <Canvas
              camera={{ position: [0, 1.2, 3.8], fov: 45, near: 0.1, far: 100 }}
              gl={{
                antialias: true,
                alpha: false,
                powerPreference: "high-performance",
              }}
              style={{ background: "#0d0d0f" }}
            >
              <color attach="background" args={["#0d0d0f"]} />
              
              {/* Lighting */}
              <ambientLight intensity={0.7} />
              <directionalLight position={[4, 8, 4]} intensity={0.9} castShadow />

              {/* Spatial Floor Reference */}
              <GridFloor />

              {/* 0. Real-time Live Reconstruction Presentation Controller */}
              <LiveReconstructionController onAssemblyProgress={setAssemblyProgress} />
              <CinematicSceneDirector />

              {/* 1. Photographic plane rendered in 3D coordinate space (Photo / Inspection Modes) */}
              {(activeRepresentation === "photo" || (isAnalyzing && !hasReconstructedMeshes)) && photoDimensions && (
                <PhotoPlane
                  photoUrl={localPhotoUrl}
                  aspectRatio={photoDimensions.aspect}
                  isReferencePlane={false}
                />
              )}

              {/* 2. Hybrid Combined Scene: Photographic Reference Plane + Assembled 3D Characters */}
              {(activeRepresentation === "combined" || (isAnalyzing && hasReconstructedMeshes)) && photoDimensions && (
                <PhotoPlane
                  photoUrl={localPhotoUrl}
                  aspectRatio={photoDimensions.aspect}
                  isReferencePlane={true}
                />
              )}

              {/* 3. Reconstructed 3D Human Meshes (Wireframe, Solid, Articulated, or Combined) */}
              {is3DMode && (
                <HumanMeshLayer
                  mode={
                    (activeRepresentation === "wireframe" ||
                    activeRepresentation === "solid3d" ||
                    activeRepresentation === "articulated" ||
                    activeRepresentation === "combined"
                      ? activeRepresentation
                      : "combined") as Unified3DMode
                  }
                  presentationProgress={assemblyProgress}
                />
              )}

              {/* Camera navigation */}
              <CameraController />
            </Canvas>

            {/* Cinematic Reconstruction Sequence Timeline */}
            <ReconstructionTimeline />
          </div>
        </ViewportErrorBoundary>
      )}
    </div>
  );
};
