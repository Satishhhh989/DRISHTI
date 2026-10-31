"use client";

import React, { useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
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
 * LiveReconstructionController
 *
 * Renders real-time visual telemetry during neural pipeline execution:
 * 1. Ingestion/Detection scanline laser sweep across the photograph
 * 2. 3D detection bounding box reveals as persons are located
 */
export const LiveReconstructionController: React.FC = () => {
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const stages = useSceneStore((state) => state.stages);
  const currentScene = useSceneStore((state) => state.currentScene);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);

  const scanlineRef = useRef<THREE.Mesh>(null);
  const scanPhaseRef = useRef(0.0);

  // Plane dimensions (matches PhotoPlane.tsx canonical sizing)
  const height = 2.0;
  const width = Math.max(0.2, height * (photoDimensions?.aspect ?? 1.5));
  const centerY = height / 2;

  const isScanning =
    isAnalyzing &&
    (!stages.person_detection ||
      stages.person_detection.status === "running" ||
      stages.image_validation?.status === "running");

  // Frame animation loop: pure Three.js updates with zero React state churn
  useFrame((_, delta) => {
    if (scanlineRef.current && isScanning) {
      scanPhaseRef.current += delta * 2.0;
      const scanY = centerY + Math.sin(scanPhaseRef.current) * (height / 2 - 0.05);
      scanlineRef.current.position.y = scanY;
      scanlineRef.current.visible = true;
    } else if (scanlineRef.current) {
      scanlineRef.current.visible = false;
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

