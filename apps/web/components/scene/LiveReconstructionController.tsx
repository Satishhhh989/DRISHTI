"use client";

import React, { useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { useSceneStore } from "@/stores/scene-store";
import { Unified3DMode } from "./HumanMeshLayer";

interface LiveReconstructionControllerProps {
  onAssemblyProgress?: (progress: number) => void;
}

const PALETTE = [
  { hex: 0x3b82f6, css: "#3b82f6" }, // Blue
  { hex: 0x10b981, css: "#10b981" }, // Emerald
  { hex: 0xf59e0b, css: "#f59e0b" }, // Amber
  { hex: 0x8b5cf6, css: "#8b5cf6" }, // Purple
  { hex: 0xec4899, css: "#ec4899" }, // Rose
  { hex: 0x06b6d4, css: "#06b6d4" }, // Cyan
];

/**
 * LiveReconstructionController
 *
 * Orchestrates real-time 3D presentation transitions during pipeline execution:
 * 1. Ingestion/Detection scanline laser sweep across the photograph
 * 2. 3D detection bounding box reveals as persons are located
 * 3. Progressive mesh materialization as individual human reconstructions finish
 * 4. Smooth scene assembly forward translation into estimated depth staging
 * 5. Gentle camera reframing and auto-transition to interactive 3D Combined scene
 */
export const LiveReconstructionController: React.FC<LiveReconstructionControllerProps> = ({
  onAssemblyProgress,
}) => {
  const { camera } = useThree();

  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const stages = useSceneStore((state) => state.stages);
  const currentScene = useSceneStore((state) => state.currentScene);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const activeRepresentation = useSceneStore((state) => state.activeRepresentation);
  const setActiveRepresentation = useSceneStore((state) => state.setActiveRepresentation);

  const scanlineRef = useRef<THREE.Mesh>(null);
  const assemblyProgressRef = useRef(1.0);
  const cameraTransitionRef = useRef(0.0);
  const hasAutoTransitionedRef = useRef(false);

  // Plane dimensions (matches PhotoPlane.tsx canonical sizing)
  const height = 2.0;
  const width = Math.max(0.2, height * (photoDimensions?.aspect ?? 1.5));
  const centerY = height / 2;

  const isScanning =
    isAnalyzing &&
    (!stages.person_detection ||
      stages.person_detection.status === "running" ||
      stages.image_validation?.status === "running");

  const isAssemblyStage =
    stages.scene_assembly?.status === "running" ||
    stages.scene_assembly?.status === "completed" ||
    (isAnalyzing && stages.mesh_rigging?.status === "completed");

  const isPipelineComplete =
    !isAnalyzing &&
    Boolean(
      currentScene?.entities?.some(
        (e) => e.reconstruction?.status === "ready" && Boolean(e.reconstruction?.meshUrl)
      )
    );

  // Trigger assembly step-forward when assembly stage begins
  useEffect(() => {
    if (isAssemblyStage && assemblyProgressRef.current >= 1.0 && !hasAutoTransitionedRef.current) {
      assemblyProgressRef.current = 0.0;
      cameraTransitionRef.current = 0.0;
    }
  }, [isAssemblyStage]);

  // Frame animation loop
  useFrame((_, delta) => {
    // 1. Ingestion Laser Scanline Animation
    if (scanlineRef.current && isScanning) {
      const t = Date.now() * 0.002;
      const scanY = centerY + Math.sin(t) * (height / 2 - 0.05);
      scanlineRef.current.position.y = scanY;
      scanlineRef.current.visible = true;
    } else if (scanlineRef.current) {
      scanlineRef.current.visible = false;
    }

    // 2. Scene Assembly Step-Forward Interpolation
    if (assemblyProgressRef.current < 1.0) {
      assemblyProgressRef.current = Math.min(1.0, assemblyProgressRef.current + delta * 0.85);
      const ease = THREE.MathUtils.smoothstep(assemblyProgressRef.current, 0.0, 1.0);
      if (onAssemblyProgress) {
        onAssemblyProgress(ease);
      }

      // Smooth camera orbit / reframe to showcase depth
      if (cameraTransitionRef.current < 1.0) {
        cameraTransitionRef.current = Math.min(1.0, cameraTransitionRef.current + delta * 0.7);
        const camEase = THREE.MathUtils.smoothstep(cameraTransitionRef.current, 0.0, 1.0);
        // Gently reframe camera to an elevated three-quarters angle to reveal spatial depth
        const targetX = 0.35 * camEase;
        const targetY = 1.0 + 0.3 * camEase;
        const targetZ = 3.2 + 0.6 * camEase;
        camera.position.set(
          THREE.MathUtils.lerp(camera.position.x, targetX, 0.08),
          THREE.MathUtils.lerp(camera.position.y, targetY, 0.08),
          THREE.MathUtils.lerp(camera.position.z, targetZ, 0.08)
        );
      }

      // Auto-transition to Combined representation once step-forward completes
      if (assemblyProgressRef.current >= 1.0 && !hasAutoTransitionedRef.current) {
        hasAutoTransitionedRef.current = true;
        if (activeRepresentation === "photo") {
          setActiveRepresentation("combined");
        }
      }
    }
  });

  const entities = currentScene?.entities || [];

  return (
    <group data-testid="live-reconstruction-controller">
      {/* Laser Scanline Beam across Photo Plane */}
      <mesh ref={scanlineRef} position={[0, centerY, 0.008]} visible={false}>
        <planeGeometry args={[width, 0.02]} />
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={0.75}
          side={THREE.DoubleSide}
          depthTest={false}
        />
      </mesh>

      {/* 3D Detection Frames materialized in Scene Space during live pipeline */}
      {isAnalyzing &&
        stages.person_detection?.status === "completed" &&
        entities.map((entity, idx) => {
          const box = entity.boundingBox || entity.detection?.boundingBox;
          if (!box) return null;

          const boxWidth = Math.max(0.01, (box.xMax - box.xMin) * width);
          const boxHeight = Math.max(0.01, (box.yMax - box.yMin) * height);
          const boxCenterX = ((box.xMin + box.xMax) / 2 - 0.5) * width;
          const boxCenterY = centerY + (0.5 - (box.yMin + box.yMax) / 2) * height;

          const color = PALETTE[idx % PALETTE.length];
          const hasMesh = entity.reconstruction?.status === "ready";

          // If mesh has arrived, fade the 2D bounding frame out smoothly to let the 3D human shine
          const frameOpacity = hasMesh ? 0.2 : 0.85;

          return (
            <group key={`live-det-${entity.id}`} position={[boxCenterX, boxCenterY, 0.012]}>
              <lineSegments>
                <edgesGeometry args={[new THREE.PlaneGeometry(boxWidth, boxHeight)]} />
                <lineBasicMaterial
                  color={color.hex}
                  transparent
                  opacity={frameOpacity}
                  linewidth={2}
                  depthTest={false}
                />
              </lineSegments>
            </group>
          );
        })}
    </group>
  );
};

