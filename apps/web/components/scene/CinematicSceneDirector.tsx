"use client";

import React, { useRef, useEffect, useMemo } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import {
  CinematicStageId,
  CINEMATIC_STAGES,
  useCinematicStore,
} from "@/stores/cinematic-store";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";

const PALETTE = [
  { hex: 0x3b82f6, css: "#3b82f6" }, // Blue
  { hex: 0x10b981, css: "#10b981" }, // Emerald
  { hex: 0xf59e0b, css: "#f59e0b" }, // Amber
  { hex: 0x8b5cf6, css: "#8b5cf6" }, // Purple
  { hex: 0xec4899, css: "#ec4899" }, // Rose
  { hex: 0x06b6d4, css: "#06b6d4" }, // Cyan
];

/**
 * CinematicSceneDirector
 *
 * Drives the 9-stage cinematic reconstruction sequence inside the 3D viewport:
 * - Coordinates camera positioning, smooth pans, and focal zooms per person
 * - Manages stage timing synchronized with real pipeline events
 * - Renders stage-specific visual indicators (framing reticles, detection tags, depth sweeps)
 */
export const CinematicSceneDirector: React.FC = () => {
  const { camera } = useThree();

  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);
  const isPaused = useCinematicStore((state) => state.isPaused);
  const currentStageId = useCinematicStore((state) => state.currentStageId);
  const stages = useCinematicStore((state) => state.stages);
  const activePersonIndex = useCinematicStore((state) => state.activePersonIndex);
  const setCurrentStage = useCinematicStore((state) => state.setCurrentStage);
  const updateStageStatus = useCinematicStore((state) => state.updateStageStatus);
  const setActivePersonIndex = useCinematicStore((state) => state.setActivePersonIndex);
  const setDepthWipeProgress = useCinematicStore((state) => state.setDepthWipeProgress);
  const setWireframeLiftProgress = useCinematicStore((state) => state.setWireframeLiftProgress);
  const setSurfaceDissolveProgress = useCinematicStore((state) => state.setSurfaceDissolveProgress);
  const setAssemblyProgress = useCinematicStore((state) => state.setAssemblyProgress);
  const skipCinematic = useCinematicStore((state) => state.skipCinematic);

  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const backendStages = useSceneStore((state) => state.stages);
  const currentScene = useSceneStore((state) => state.currentScene);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const setVisualizationMode = useSceneStore((state) => state.setVisualizationMode);
  const setActiveRepresentation = useSceneStore((state) => state.setActiveRepresentation);

  const stageTimerRef = useRef(0);
  const personTimerRef = useRef(0);
  const cameraTargetRef = useRef<{ pos: THREE.Vector3; look: THREE.Vector3 }>({
    pos: new THREE.Vector3(0, 1.0, 3.2),
    look: new THREE.Vector3(0, 1.0, 0),
  });

  const height = 2.0;
  const aspect = photoDimensions?.aspect ?? 1.5;
  const width = Math.max(0.2, height * aspect);
  const centerY = height / 2;

  const entities = currentScene?.entities || [];
  const totalEntities = entities.length;

  // Detect user preference for reduced motion
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  // Sync real backend events into cinematic timeline stages
  useEffect(() => {
    if (!isAnalyzing) return;

    // Detect transition points from backend stages
    if (backendStages.person_detection?.status === "completed" && stages.stage_detection.status === "waiting") {
      updateStageStatus("stage_photo", "completed", { progress: 1.0 });
      setCurrentStage("stage_detection");
      updateStageStatus("stage_detection", "processing", {
        itemCount: totalEntities,
        summary: `${totalEntities} people detected`,
      });
    }

    if (backendStages.instance_segmentation?.status === "completed" && stages.stage_segmentation.status === "waiting") {
      updateStageStatus("stage_detection", "completed", { progress: 1.0 });
      setCurrentStage("stage_segmentation");
      updateStageStatus("stage_segmentation", "processing", {
        itemCount: totalEntities,
        totalCount: totalEntities,
        summary: `Segmentation complete · ${totalEntities}/${totalEntities} people`,
      });
    }

    if (backendStages.depth_estimation?.status === "completed" && stages.stage_depth.status === "waiting") {
      updateStageStatus("stage_segmentation", "completed", { progress: 1.0 });
      setCurrentStage("stage_depth");
      updateStageStatus("stage_depth", "processing", {
        summary: "Depth Anything V2 field estimated",
      });
    }

    if (backendStages.spatial_photo_preparation?.status === "completed" && stages.stage_spatial_photo.status === "waiting") {
      updateStageStatus("stage_depth", "completed", { progress: 1.0 });
      setCurrentStage("stage_spatial_photo");
      updateStageStatus("stage_spatial_photo", "processing", {
        summary: "Spatial depth relief active",
      });
    }

    if (
      (backendStages.human_reconstruction?.status === "completed" ||
        backendStages.mesh_reconstruction?.status === "completed") &&
      stages.stage_wireframe.status === "waiting"
    ) {
      updateStageStatus("stage_spatial_photo", "completed", { progress: 1.0 });
      setCurrentStage("stage_wireframe");
      updateStageStatus("stage_wireframe", "processing", {
        itemCount: totalEntities,
        summary: `HMR 2.0 reconstructed ${totalEntities} 3D human meshes`,
      });
    }

    if (backendStages.skeleton_fitting?.status === "completed" && stages.stage_articulation.status === "waiting") {
      updateStageStatus("stage_wireframe", "completed", { progress: 1.0 });
      updateStageStatus("stage_solid3d", "completed", { progress: 1.0 });
      setCurrentStage("stage_articulation");
      updateStageStatus("stage_articulation", "processing", {
        summary: "Kinematic SMPL rigs fitted",
      });
    }

    if (backendStages.scene_assembly?.status === "completed" && stages.stage_combined.status === "waiting") {
      updateStageStatus("stage_articulation", "completed", { progress: 1.0 });
      setCurrentStage("stage_combined");
      updateStageStatus("stage_combined", "processing", {
        summary: "Multi-person scene assembly complete",
      });
      // Finalize
      setTimeout(() => {
        updateStageStatus("stage_combined", "completed", { progress: 1.0 });
        useCinematicStore.getState().skipCinematic();
        setVisualizationMode("combined");
      }, 2000);
    }
  }, [backendStages, isAnalyzing, totalEntities, stages, updateStageStatus, setCurrentStage, setVisualizationMode]);

  // Main Director Frame Loop
  useFrame((_, delta) => {
    if (!isCinematicActive || isPaused) return;

    stageTimerRef.current += delta;
    const stageTime = stageTimerRef.current;

    // -------------------------------------------------------------
    // STAGE 1: Original Photograph & Introduction
    // -------------------------------------------------------------
    if (currentStageId === "stage_photo") {
      if (prefersReducedMotion) {
        cameraTargetRef.current.pos.set(0, 1.0, 3.2);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      } else {
        // Subtle push-in toward photograph: from (0, 1.0, 3.6) to (0, 1.0, 3.1)
        const t = Math.min(1.0, stageTime / 2.5);
        const ease = THREE.MathUtils.smoothstep(t, 0, 1);
        cameraTargetRef.current.pos.set(0, 1.0, 3.6 - 0.5 * ease);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      }

      // Auto-advance if not waiting for real backend
      if (!isAnalyzing && stageTime > 2.5) {
        updateStageStatus("stage_photo", "completed", { progress: 1.0 });
        setCurrentStage("stage_detection");
        setVisualizationMode("detection");
        stageTimerRef.current = 0;
        personTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 2: Person Detection (Individual candidate focus)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_detection") {
      personTimerRef.current += delta;
      const pTime = personTimerRef.current;
      const personDuration = 1.4;

      if (totalEntities > 0) {
        const curIdx = Math.min(totalEntities - 1, Math.floor(stageTime / personDuration));
        if (curIdx !== activePersonIndex && curIdx < totalEntities) {
          setActivePersonIndex(curIdx, totalEntities);
        }

        const entity = entities[curIdx];
        const box = entity?.boundingBox || entity?.detection?.boundingBox;

        if (box && curIdx < totalEntities && !prefersReducedMotion) {
          // Focus camera toward current person's region on the photo plane
          const boxCenterX = ((box.xMin + box.xMax) / 2 - 0.5) * width;
          const boxCenterY = centerY + (0.5 - (box.yMin + box.yMax) / 2) * height;

          cameraTargetRef.current.pos.set(boxCenterX * 0.45, boxCenterY * 0.45 + 0.5, 2.3);
          cameraTargetRef.current.look.set(boxCenterX * 0.4, boxCenterY * 0.4 + 0.5, 0);
        }

        // Pull back after last person has been introduced
        if (stageTime > totalEntities * personDuration) {
          setActivePersonIndex(-1);
          cameraTargetRef.current.pos.set(0, 1.0, 3.2);
          cameraTargetRef.current.look.set(0, 1.0, 0);

          if (!isAnalyzing && stageTime > totalEntities * personDuration + 1.2) {
            updateStageStatus("stage_detection", "completed", {
              progress: 1.0,
              summary: `${totalEntities} people detected`,
            });
            setCurrentStage("stage_segmentation");
            setVisualizationMode("segmentation");
            stageTimerRef.current = 0;
          }
        }
      } else if (!isAnalyzing && stageTime > 2.0) {
        setCurrentStage("stage_depth");
        setVisualizationMode("depth");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 3: Segmentation (Contour silhouette reveal)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_segmentation") {
      cameraTargetRef.current.pos.set(0, 1.0, 3.2);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.8) {
        updateStageStatus("stage_segmentation", "completed", {
          progress: 1.0,
          summary: `Segmentation complete · ${totalEntities}/${totalEntities} people`,
        });
        setCurrentStage("stage_depth");
        setVisualizationMode("depth");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 4: Depth Map (Smooth wipe sweep revealing depth field)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_depth") {
      const wipe = Math.min(1.0, stageTime / 2.4);
      setDepthWipeProgress(wipe);
      cameraTargetRef.current.pos.set(0, 1.0, 3.2);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.8) {
        updateStageStatus("stage_depth", "completed", { progress: 1.0 });
        setCurrentStage("stage_spatial_photo");
        setVisualizationMode("spatial_photo");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 5: Spatial Photograph (Perspective depth relief)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_spatial_photo") {
      if (prefersReducedMotion) {
        cameraTargetRef.current.pos.set(0.4, 1.05, 3.1);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      } else {
        // Orbit to 25-degree perspective to reveal real depth displacement
        const t = Math.min(1.0, stageTime / 2.0);
        const angle = 0.45 * Math.sin(t * Math.PI * 0.5);
        cameraTargetRef.current.pos.set(Math.sin(angle) * 3.2, 1.05, Math.cos(angle) * 3.2);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      }

      if (!isAnalyzing && stageTime > 2.8) {
        updateStageStatus("stage_spatial_photo", "completed", { progress: 1.0 });
        setCurrentStage("stage_wireframe");
        setVisualizationMode("wireframe");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 6: Wireframe Extraction (Per-person lift & topology)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_wireframe") {
      const lift = Math.min(1.0, stageTime / 2.2);
      setWireframeLiftProgress(lift);
      if (prefersReducedMotion) {
        cameraTargetRef.current.pos.set(0, 1.15, 3.4);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      } else {
        // Gentle orbit framing the lifting wireframes
        const angle = 0.25 * Math.sin(stageTime * 0.8);
        cameraTargetRef.current.pos.set(Math.sin(angle) * 3.4, 1.15, Math.cos(angle) * 3.4);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      }

      if (!isAnalyzing && stageTime > 3.0) {
        updateStageStatus("stage_wireframe", "completed", { progress: 1.0 });
        setCurrentStage("stage_solid3d");
        setVisualizationMode("solid3d");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 7: 3D Solid Mesh (Surface shading & vertex colors)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_solid3d") {
      const dissolve = Math.min(1.0, stageTime / 2.0);
      setSurfaceDissolveProgress(dissolve);
      cameraTargetRef.current.pos.set(0.15, 1.15, 3.5);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.6) {
        updateStageStatus("stage_solid3d", "completed", { progress: 1.0 });
        setCurrentStage("stage_articulation");
        setVisualizationMode("articulated");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 8: Articulation & Rigging (Skeleton joints illumination)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_articulation") {
      cameraTargetRef.current.pos.set(0.0, 1.1, 3.4);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.8) {
        updateStageStatus("stage_articulation", "completed", { progress: 1.0 });
        setCurrentStage("stage_combined");
        setVisualizationMode("combined");
        stageTimerRef.current = 0;
      }
    }

    // -------------------------------------------------------------
    // STAGE 9: Final Combined Scene (Step-forward into depth staging)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_combined") {
      const assemble = Math.min(1.0, stageTime / 1.8);
      setAssemblyProgress(assemble);

      // Reframe camera to complete wide composition
      cameraTargetRef.current.pos.set(0.35, 1.35, 3.8);
      cameraTargetRef.current.look.set(0, 1.0, 0.4);

      if (assemble >= 1.0) {
        updateStageStatus("stage_combined", "completed", { progress: 1.0 });
        setActiveRepresentation("combined");
        // Handover smoothly to standard interactive viewport
        setTimeout(() => {
          useCinematicStore.getState().skipCinematic();
        }, 800);
      }
    }

    // Smooth camera interpolation
    camera.position.lerp(cameraTargetRef.current.pos, prefersReducedMotion ? 0.25 : 0.08);
    camera.lookAt(cameraTargetRef.current.look);
  });

  // Reticles, framing outlines, and depth sweeps
  return (
    <group data-testid="cinematic-scene-director">
      {/* Stage 1: Cinematic Corner Framing Reticles */}
      {currentStageId === "stage_photo" && (
        <group position={[0, centerY, 0.015]}>
          {/* Subtle outer crop border */}
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(width + 0.06, height + 0.06)]} />
            <lineBasicMaterial color="#38bdf8" transparent opacity={0.4} />
          </lineSegments>
        </group>
      )}

      {/* Stage 2 & 3: Per-Person Detection Highlights */}
      {(currentStageId === "stage_detection" || currentStageId === "stage_segmentation") &&
        entities.map((entity, idx) => {
          const box = entity.boundingBox || entity.detection?.boundingBox;
          if (!box) return null;

          const boxWidth = Math.max(0.01, (box.xMax - box.xMin) * width);
          const boxHeight = Math.max(0.01, (box.yMax - box.yMin) * height);
          const boxCenterX = ((box.xMin + box.xMax) / 2 - 0.5) * width;
          const boxCenterY = centerY + (0.5 - (box.yMin + box.yMax) / 2) * height;

          const isActive = activePersonIndex === idx || activePersonIndex === -1;
          const color = PALETTE[idx % PALETTE.length];

          return (
            <group key={`cinematic-det-${entity.id}`} position={[boxCenterX, boxCenterY, 0.018]}>
              <lineSegments>
                <edgesGeometry args={[new THREE.PlaneGeometry(boxWidth, boxHeight)]} />
                <lineBasicMaterial
                  color={isActive ? "#ffffff" : color.hex}
                  transparent
                  opacity={isActive ? 0.95 : 0.3}
                  linewidth={isActive ? 2 : 1}
                  depthTest={false}
                />
              </lineSegments>
            </group>
          );
        })}
    </group>
  );
};

