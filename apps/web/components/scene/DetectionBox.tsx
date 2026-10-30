"use client";

import React from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { SceneEntity } from "@photo-to-3d/scene-schema";
import { useUIStore } from "@/stores/ui-store";

interface DetectionBoxProps {
  entity: SceneEntity;
  planeWidth: number;
  planeHeight: number;
  colorHex: number;
  colorCss: string;
}

export const DetectionBox: React.FC<DetectionBoxProps> = ({
  entity,
  planeWidth,
  planeHeight,
  colorHex,
  colorCss,
}) => {
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const selectEntity = useUIStore((state) => state.selectEntity);

  const isSelected = selectedEntityId === entity.id;

  const box = entity.boundingBox;
  if (!box) return null;

  // Transform normalized image coordinates to 3D plane local coordinates
  // (0,0) image top-left -> X = -planeWidth/2, Y = +planeHeight/2
  // (1,1) image bottom-right -> X = +planeWidth/2, Y = -planeHeight/2
  const boxWidth = Math.max(0.01, (box.xMax - box.xMin) * planeWidth);
  const boxHeight = Math.max(0.01, (box.yMax - box.yMin) * planeHeight);

  const centerX = ((box.xMin + box.xMax) / 2 - 0.5) * planeWidth;
  const centerY = (0.5 - (box.yMin + box.yMax) / 2) * planeHeight;

  const confidencePct = entity.detection
    ? Math.round(entity.detection.confidence * 100)
    : entity.confidence?.detectionConfidence
    ? Math.round(entity.confidence.detectionConfidence * 100)
    : null;

  return (
    <group position={[centerX, centerY, 0.005]}>
      {/* Clickable transparent hit area */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          selectEntity(isSelected ? null : entity.id);
        }}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "default";
        }}
      >
        <planeGeometry args={[boxWidth, boxHeight]} />
        <meshBasicMaterial
          transparent
          opacity={isSelected ? 0.25 : 0.05}
          color={colorHex}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Bounding box outline */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(boxWidth, boxHeight)]} />
        <lineBasicMaterial
          color={isSelected ? "#ffffff" : colorHex}
          linewidth={isSelected ? 2 : 1}
        />
      </lineSegments>

      {/* Label Chip at Top-Left */}
      <Html
        position={[-boxWidth / 2, boxHeight / 2, 0]}
        style={{
          transform: "translate(0, -100%)",
          pointerEvents: "none",
          whiteSpace: "nowrap",
          userSelect: "none",
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "4px",
            backgroundColor: isSelected ? "#ffffff" : "rgba(18, 18, 20, 0.88)",
            color: isSelected ? "#09090b" : "#ededed",
            border: `1px solid ${isSelected ? "#ffffff" : colorCss}`,
            borderRadius: "3px",
            padding: "2px 5px",
            fontSize: "10px",
            fontWeight: 600,
            fontFamily: "system-ui, sans-serif",
            boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
            marginBottom: "2px",
          }}
        >
          <span>{entity.name || "Person"}</span>
          {confidencePct !== null && (
            <span style={{ opacity: 0.8, fontSize: "9px" }}>{confidencePct}%</span>
          )}
        </div>
      </Html>
    </group>
  );
};
