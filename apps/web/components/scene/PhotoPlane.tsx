"use client";

import React, { useEffect, useState } from "react";
import * as THREE from "three";
import { useSceneStore } from "@/stores/scene-store";
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
  const positionZ = isReferencePlane ? -0.6 : 0;

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

          {/* Border outline */}
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(width, height)]} />
            <lineBasicMaterial
              color={
                isReferencePlane
                  ? "#6366f1"
                  : visualizationMode === "depth"
                  ? "#a1a1aa"
                  : "#3f3f46"
              }
              linewidth={isReferencePlane ? 2 : 1}
            />
          </lineSegments>
        </>
      )}

      {/* Segmentation Mask Overlays (Only in 2D segmentation inspection mode) */}
      {!isReferencePlane &&
        visualizationMode === "segmentation" &&
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

      {/* Detection Bounding Boxes (In 2D detection or segmentation inspection mode) */}
      {!isReferencePlane &&
        (visualizationMode === "detection" || visualizationMode === "segmentation") &&
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
