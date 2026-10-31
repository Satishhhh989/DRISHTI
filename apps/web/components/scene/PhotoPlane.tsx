"use client";

import React, { useEffect, useState, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useSceneStore } from "@/stores/scene-store";
import { useCinematicStore } from "@/stores/cinematic-store";
import { assetCache } from "@/lib/asset-cache";
import { DetectionBox } from "./DetectionBox";
import { SegmentationMaskLayer } from "./SegmentationMaskLayer";
import { SpatialPhotoSurface } from "./SpatialPhotoSurface";

interface PhotoPlaneProps {
  photoUrl: string;
  aspectRatio: number; // width / height
  isReferencePlane?: boolean;
}

const PALETTE = [
  { hex: 0x3b82f6, css: "#3b82f6" }, // Blue
  { hex: 0x10b981, css: "#10b981" }, // Emerald
  { hex: 0xf59e0b, css: "#f59e0b" }, // Amber
  { hex: 0x8b5cf6, css: "#8b5cf6" }, // Purple
  { hex: 0xec4899, css: "#ec4899" }, // Rose
  { hex: 0x06b6d4, css: "#06b6d4" }, // Cyan
];

export const PhotoPlane: React.FC<PhotoPlaneProps> = ({
  photoUrl,
  aspectRatio,
  isReferencePlane = false,
}) => {
  const [photoTexture, setPhotoTexture] = useState<THREE.Texture | null>(null);
  const [depthVisTexture, setDepthVisTexture] = useState<THREE.Texture | null>(null);

  const currentScene = useSceneStore((state) => state.currentScene);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const depthStrength = useSceneStore((state) => state.depthStrength);
  const depthVisualizationUrl = useSceneStore((state) => state.depthVisualizationUrl);

  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);
  const currentStageId = useCinematicStore((state) => state.currentStageId);

  const wipeMeshRef = useRef<THREE.Mesh>(null);
  const laserRef = useRef<THREE.Mesh>(null);
  const wipeProgressRef = useRef(0.0);

  // Smooth frame-rate independent depth reveal wipe
  useFrame((_, delta) => {
    if (isCinematicActive && currentStageId === "stage_depth" && depthVisTexture) {
      wipeProgressRef.current = Math.min(1.0, wipeProgressRef.current + delta * 0.48);
      const p = THREE.MathUtils.smoothstep(wipeProgressRef.current, 0, 1);
      if (wipeMeshRef.current) {
        wipeMeshRef.current.visible = true;
        wipeMeshRef.current.scale.x = Math.max(0.001, p);
        wipeMeshRef.current.position.x = -width / 2 + (width * p) / 2;
      }
      if (laserRef.current) {
        laserRef.current.visible = p < 1.0;
        laserRef.current.position.x = -width / 2 + width * p;
      }
    } else {
      wipeProgressRef.current = 0.0;
      if (wipeMeshRef.current) wipeMeshRef.current.visible = false;
      if (laserRef.current) laserRef.current.visible = false;
    }
  });

  // 1. Load original photo texture via unified assetCache
  useEffect(() => {
    let active = true;

    assetCache
      .loadTexture(photoUrl, true)
      .then((tex) => {
        if (active) {
          setPhotoTexture(tex);
        }
      })
      .catch((err) => {
        console.error("Failed to load photograph texture into 3D plane:", err);
      });

    return () => {
      active = false;
    };
  }, [photoUrl]);

  // 2. Load depth visualization texture if available
  useEffect(() => {
    if (!depthVisualizationUrl) {
      setDepthVisTexture(null);
      return;
    }

    let active = true;

    assetCache
      .loadTexture(depthVisualizationUrl, false)
      .then((tex) => {
        if (active) {
          setDepthVisTexture(tex);
        }
      })
      .catch((err) => {
        console.error("Failed to load depth visualization texture:", err);
      });

    return () => {
      active = false;
    };
  }, [depthVisualizationUrl]);

  // Plane dimensions: height = 2.0 meters, width = 2.0 * aspectRatio
  const height = 2.0;
  const width = Math.max(0.2, height * (aspectRatio > 0 ? aspectRatio : 1));

  // Center plane so bottom rests at ground plane (Y = height / 2)
  const positionY = height / 2;
  const refPlaneZ = currentScene?.sceneGraph?.nodes?.["plane-reference"]?.transform?.position?.z ?? -1.2;
  const positionZ = isReferencePlane ? refPlaneZ : 0;

  if (!photoTexture) {
    return null;
  }

  const entities = currentScene?.entities || [];

  return (
    <group position={[0, positionY, positionZ]} data-testid={isReferencePlane ? "photo-reference-plane" : "photo-plane-group"}>
      {/* 1. Spatial Photo Mode: Depth-aware displaced 3D mesh (Only in non-reference spatial photo mode) */}
      {!isReferencePlane && visualizationMode === "spatial_photo" && depthVisualizationUrl ? (
        <SpatialPhotoSurface
          photoUrl={photoUrl}
          depthUrl={depthVisualizationUrl}
          planeWidth={width}
          planeHeight={height}
          depthStrength={depthStrength}
        />
      ) : (
        /* 2. Flat 3D Plane Mode: Used for Original, Detection, Segmentation, Depth Map, or Combined Reference */
        <>
          <mesh receiveShadow castShadow>
            <planeGeometry args={[width, height]} />
            <meshBasicMaterial
              map={
                !isReferencePlane && visualizationMode === "depth" && depthVisTexture
                  ? depthVisTexture
                  : photoTexture
              }
              side={THREE.DoubleSide}
              transparent={true}
              opacity={isReferencePlane ? 0.85 : 1.0}
              depthWrite={!isReferencePlane}
            />
          </mesh>

          {/* Depth Wipe Transition Layer (Stage 4) */}
          {!isReferencePlane && isCinematicActive && currentStageId === "stage_depth" && depthVisTexture && (
            <>
              <mesh ref={wipeMeshRef} position={[0, 0, 0.003]} visible={false}>
                <planeGeometry args={[width, height]} />
                <meshBasicMaterial
                  map={depthVisTexture}
                  side={THREE.DoubleSide}
                  transparent={true}
                  opacity={1.0}
                  depthWrite={false}
                />
              </mesh>
              {/* Laser Wipe Leading Edge */}
              <mesh ref={laserRef} position={[0, 0, 0.006]} visible={false}>
                <planeGeometry args={[0.02, height]} />
                <meshBasicMaterial
                  color="#38bdf8"
                  transparent={true}
                  opacity={0.9}
                  side={THREE.DoubleSide}
                  depthTest={false}
                />
              </mesh>
            </>
          )}

          {/* Border outline */}
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(width, height)]} />
            <lineBasicMaterial
              color={
                isReferencePlane
                  ? "#6366f1"
                  : visualizationMode === "depth" || currentStageId === "stage_depth"
                  ? "#38bdf8"
                  : "#3f3f46"
              }
              linewidth={isReferencePlane ? 2 : 1}
            />
          </lineSegments>
        </>
      )}

      {/* Segmentation Mask Overlays (2D segmentation mode or Cinematic Stage 3) */}
      {!isReferencePlane &&
        (visualizationMode === "segmentation" || (isCinematicActive && currentStageId === "stage_segmentation")) &&
        entities.map((entity, idx) => {
          const color = PALETTE[idx % PALETTE.length];
          return (
            <SegmentationMaskLayer
              key={`mask-${entity.id}`}
              entity={entity}
              planeWidth={width}
              planeHeight={height}
              colorHex={color.hex}
            />
          );
        })}

      {/* Detection Bounding Boxes (2D detection/segmentation mode or Cinematic Stage 2/3) */}
      {!isReferencePlane &&
        (visualizationMode === "detection" ||
          visualizationMode === "segmentation" ||
          (isCinematicActive && (currentStageId === "stage_detection" || currentStageId === "stage_segmentation"))) &&
        entities.map((entity, idx) => {
          const color = PALETTE[idx % PALETTE.length];
          return (
            <DetectionBox
              key={`box-${entity.id}`}
              entity={entity}
              planeWidth={width}
              planeHeight={height}
              colorHex={color.hex}
              colorCss={color.css}
            />
          );
        })}
    </group>
  );
};
