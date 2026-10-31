"use client";

import React, { useRef, useEffect, useMemo } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { useCinematicStore } from "@/stores/cinematic-store";
import { useSceneStore } from "@/stores/scene-store";

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
 * - Fluid, frame-rate independent camera motion (damped interpolation with delta-time)
 * - Zero per-frame React state re-renders (120fps stutter-free performance)
 * - Stage timing synchronized with real backend pipeline events
 * - Clear, elegant framing that shows the whole scene rather than jerky camera snaps
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

  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const backendStages = useSceneStore((state) => state.stages);
  const currentScene = useSceneStore((state) => state.currentScene);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const setVisualizationMode = useSceneStore((state) => state.setVisualizationMode);
  const setActiveRepresentation = useSceneStore((state) => state.setActiveRepresentation);

  const stageTimerRef = useRef(0);
  const personTimerRef = useRef(0);
  const currentLookRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 1.0, 0));
  const cameraTargetRef = useRef<{ pos: THREE.Vector3; look: THREE.Vector3 }>({
    pos: new THREE.Vector3(0, 1.0, 3.25),
    look: new THREE.Vector3(0, 1.0, 0),
  });

  const height = 2.0;
  const aspect = photoDimensions?.aspect ?? 1.5;
  const width = Math.max(0.2, height * aspect);
  const centerY = height / 2;

  const entities = currentScene?.entities || [];
  const totalEntities = entities.length;

  // Reset stage timer whenever active stage transitions
  useEffect(() => {
    stageTimerRef.current = 0;
    personTimerRef.current = 0;
  }, [currentStageId]);

  // Detect user preference for reduced motion
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  // Sync real backend events into cinematic timeline stages
  useEffect(() => {
    if (!isAnalyzing) return;

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
      setTimeout(() => {
        updateStageStatus("stage_combined", "completed", { progress: 1.0 });
        useCinematicStore.getState().skipCinematic();
        setVisualizationMode("wireframe");
      }, 2000);
    }
  }, [backendStages, isAnalyzing, totalEntities, stages, updateStageStatus, setCurrentStage, setVisualizationMode]);

  // Main Director Frame Loop: Frame-Rate Independent, Damped Interpolation
  useFrame((_, delta) => {
    if (!isCinematicActive || isPaused) return;

    stageTimerRef.current += delta;
    const stageTime = stageTimerRef.current;

    // -------------------------------------------------------------
    // STAGE 1: Original Photograph (Dignified framing & soft push-in)
    // -------------------------------------------------------------
    if (currentStageId === "stage_photo") {
      if (!prefersReducedMotion) {
        const t = Math.min(1.0, stageTime / 2.2);
        const ease = THREE.MathUtils.smoothstep(t, 0, 1);
        cameraTargetRef.current.pos.set(0, 1.0, 3.4 - 0.25 * ease);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      } else {
        cameraTargetRef.current.pos.set(0, 1.0, 3.2);
        cameraTargetRef.current.look.set(0, 1.0, 0);
      }

      if (!isAnalyzing && stageTime > 2.0) {
        updateStageStatus("stage_photo", "completed", { progress: 1.0 });
        setCurrentStage("stage_detection");
        setVisualizationMode("detection");
      }
    }

    // -------------------------------------------------------------
    // STAGE 2: Person Detection (Steady wide frame, sequential pulse)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_detection") {
      // Keep camera steady so entire photo and all detections are clearly framed
      cameraTargetRef.current.pos.set(0, 1.05, 3.15);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      // Sequence highlight focus smoothly once per candidate
      const personInterval = 0.8;
      if (totalEntities > 0) {
        const nextPersonIdx = Math.min(totalEntities - 1, Math.floor(stageTime / personInterval));
        if (nextPersonIdx !== activePersonIndex) {
          setActivePersonIndex(nextPersonIdx, totalEntities);
        }
      }

      if (!isAnalyzing && stageTime > Math.max(2.0, totalEntities * personInterval + 0.6)) {
        setActivePersonIndex(-1);
        updateStageStatus("stage_detection", "completed", {
          progress: 1.0,
          summary: `${totalEntities} people detected`,
        });
        setCurrentStage("stage_segmentation");
        setVisualizationMode("segmentation");
      }
    }

    // -------------------------------------------------------------
    // STAGE 3: Segmentation (Silhouette contours illuminate)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_segmentation") {
      cameraTargetRef.current.pos.set(0, 1.05, 3.15);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 1.8) {
        updateStageStatus("stage_segmentation", "completed", {
          progress: 1.0,
          summary: `Segmentation complete · ${totalEntities}/${totalEntities} people`,
        });
        setCurrentStage("stage_depth");
        setVisualizationMode("depth");
      }
    }

    // -------------------------------------------------------------
    // STAGE 4: Depth Map (Laser scanning reveal)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_depth") {
      cameraTargetRef.current.pos.set(0, 1.05, 3.15);
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.0) {
        updateStageStatus("stage_depth", "completed", { progress: 1.0 });
        setCurrentStage("stage_spatial_photo");
        setVisualizationMode("spatial_photo");
      }
    }

    // -------------------------------------------------------------
    // STAGE 5: Spatial Photograph (Gentle 12-degree perspective reveal)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_spatial_photo") {
      if (prefersReducedMotion) {
        cameraTargetRef.current.pos.set(0.3, 1.05, 3.15);
      } else {
        const t = Math.min(1.0, stageTime / 2.0);
        const angle = 0.22 * Math.sin(t * Math.PI * 0.5);
        cameraTargetRef.current.pos.set(Math.sin(angle) * 3.15, 1.05, Math.cos(angle) * 3.15);
      }
      cameraTargetRef.current.look.set(0, 1.0, 0);

      if (!isAnalyzing && stageTime > 2.2) {
        updateStageStatus("stage_spatial_photo", "completed", { progress: 1.0 });
        setCurrentStage("stage_wireframe");
        setVisualizationMode("wireframe");
      }
    }

    // -------------------------------------------------------------
    // STAGE 6: Wireframe Extraction (Graceful forward lift off photo)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_wireframe") {
      // Smoothly elevate camera into three-quarters perspective
      cameraTargetRef.current.pos.set(0.2, 1.25, 3.3);
      cameraTargetRef.current.look.set(0, 0.95, 0);

      if (!isAnalyzing && stageTime > 2.4) {
        updateStageStatus("stage_wireframe", "completed", { progress: 1.0 });
        setCurrentStage("stage_solid3d");
        setVisualizationMode("wireframe");
      }
    }

    // -------------------------------------------------------------
    // STAGE 7: 3D Wireframe Topology (Tracking shot across formation)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_solid3d") {
      // Slow, luxurious tracking shot
      cameraTargetRef.current.pos.set(-0.15, 1.22, 3.25);
      cameraTargetRef.current.look.set(0, 0.95, 0.1);

      if (!isAnalyzing && stageTime > 2.0) {
        updateStageStatus("stage_solid3d", "completed", { progress: 1.0 });
        setCurrentStage("stage_articulation");
        setVisualizationMode("wireframe");
      }
    }

    // -------------------------------------------------------------
    // STAGE 8: Articulation & Rigging (Skeletal joint illumination)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_articulation") {
      cameraTargetRef.current.pos.set(0.0, 1.15, 3.25);
      cameraTargetRef.current.look.set(0, 0.95, 0.15);

      if (!isAnalyzing && stageTime > 2.0) {
        updateStageStatus("stage_articulation", "completed", { progress: 1.0 });
        setCurrentStage("stage_combined");
        setVisualizationMode("wireframe");
      }
    }

    // -------------------------------------------------------------
    // STAGE 9: Combined Scene Assembly (Stepping into depth ground positions)
    // -------------------------------------------------------------
    else if (currentStageId === "stage_combined") {
      // Graceful wide final framing
      cameraTargetRef.current.pos.set(0.25, 1.3, 3.6);
      cameraTargetRef.current.look.set(0, 1.0, 0.2);

      if (!isAnalyzing && stageTime > 2.4) {
        updateStageStatus("stage_combined", "completed", { progress: 1.0 });
        setActiveRepresentation("wireframe");
        setVisualizationMode("wireframe");
        setTimeout(() => {
          useCinematicStore.getState().skipCinematic();
        }, 500);
      }
    }

    // -------------------------------------------------------------
    // Frame-rate independent physics damping (100% fluid at 60/120Hz)
    // -------------------------------------------------------------
    const dampRate = prefersReducedMotion ? 6.0 : 3.8;
    camera.position.x = THREE.MathUtils.damp(camera.position.x, cameraTargetRef.current.pos.x, dampRate, delta);
    camera.position.y = THREE.MathUtils.damp(camera.position.y, cameraTargetRef.current.pos.y, dampRate, delta);
    camera.position.z = THREE.MathUtils.damp(camera.position.z, cameraTargetRef.current.pos.z, dampRate, delta);

    currentLookRef.current.x = THREE.MathUtils.damp(currentLookRef.current.x, cameraTargetRef.current.look.x, dampRate * 1.2, delta);
    currentLookRef.current.y = THREE.MathUtils.damp(currentLookRef.current.y, cameraTargetRef.current.look.y, dampRate * 1.2, delta);
    currentLookRef.current.z = THREE.MathUtils.damp(currentLookRef.current.z, cameraTargetRef.current.look.z, dampRate * 1.2, delta);

    camera.lookAt(currentLookRef.current);
  });

  return (
    <group data-testid="cinematic-scene-director">
      {/* Stage 1: Framing Reticle */}
      {currentStageId === "stage_photo" && (
        <group position={[0, centerY, 0.012]}>
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(width + 0.04, height + 0.04)]} />
            <lineBasicMaterial color="#38bdf8" transparent opacity={0.5} />
          </lineSegments>
        </group>
      )}

      {/* Stage 2 & 3: Person Detection Highlights (Pure WebGL outlines) */}
      {(currentStageId === "stage_detection" || currentStageId === "stage_segmentation") &&
        entities.map((entity, idx) => {
          const box = entity.boundingBox || entity.detection?.boundingBox;
          if (!box) return null;

          const boxWidth = Math.max(0.01, (box.xMax - box.xMin) * width);
          const boxHeight = Math.max(0.01, (box.yMax - box.yMin) * height);
          const boxCenterX = ((box.xMin + box.xMax) / 2 - 0.5) * width;
          const boxCenterY = centerY + (0.5 - (box.yMin + box.yMax) / 2) * height;

          const isFocused = activePersonIndex === idx;
          const color = PALETTE[idx % PALETTE.length];

          return (
            <group key={`cinematic-det-${entity.id}`} position={[boxCenterX, boxCenterY, 0.015]}>
              <lineSegments>
                <edgesGeometry args={[new THREE.PlaneGeometry(boxWidth, boxHeight)]} />
                <lineBasicMaterial
                  color={isFocused ? "#ffffff" : color.hex}
                  transparent
                  opacity={isFocused ? 0.95 : 0.45}
                  linewidth={isFocused ? 2 : 1}
                  depthTest={false}
                />
              </lineSegments>
            </group>
          );
        })}
    </group>
  );
};
