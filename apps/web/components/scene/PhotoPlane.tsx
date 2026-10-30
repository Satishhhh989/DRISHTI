"use client";

import React, { useEffect, useState } from "react";
import * as THREE from "three";
import { useSceneStore } from "@/stores/scene-store";
import { DetectionBox } from "./DetectionBox";
import { SegmentationMaskLayer } from "./SegmentationMaskLayer";
import { SpatialPhotoSurface } from "./SpatialPhotoSurface";

interface PhotoPlaneProps {
  photoUrl: string;
  aspectRatio: number; // width / height
}

const PALETTE = [
  { hex: 0x3b82f6, css: "#3b82f6" }, // Blue
  { hex: 0x10b981, css: "#10b981" }, // Emerald
  { hex: 0xf59e0b, css: "#f59e0b" }, // Amber
  { hex: 0x8b5cf6, css: "#8b5cf6" }, // Purple
  { hex: 0xec4899, css: "#ec4899" }, // Rose
  { hex: 0x06b6d4, css: "#06b6d4" }, // Cyan
];

export const PhotoPlane: React.FC<PhotoPlaneProps> = ({ photoUrl, aspectRatio }) => {
  const [photoTexture, setPhotoTexture] = useState<THREE.Texture | null>(null);
  const [depthVisTexture, setDepthVisTexture] = useState<THREE.Texture | null>(null);

  const currentScene = useSceneStore((state) => state.currentScene);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const depthStrength = useSceneStore((state) => state.depthStrength);
  const depthVisualizationUrl = useSceneStore((state) => state.depthVisualizationUrl);

  // 1. Load original photo texture
  useEffect(() => {
    let active = true;
    const loader = new THREE.TextureLoader();

    loader.load(
      photoUrl,
      (loadedTex) => {
        if (!active) {
          loadedTex.dispose();
          return;
        }
        loadedTex.colorSpace = THREE.SRGBColorSpace;
        loadedTex.minFilter = THREE.LinearFilter;
        loadedTex.generateMipmaps = false;
        setPhotoTexture(loadedTex);
      },
      undefined,
      (err) => {
        console.error("Failed to load photograph texture into 3D plane:", err);
      }
    );

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
    const loader = new THREE.TextureLoader();

    loader.load(
      depthVisualizationUrl,
      (loadedTex) => {
        if (!active) {
          loadedTex.dispose();
          return;
        }
        loadedTex.minFilter = THREE.LinearFilter;
        loadedTex.magFilter = THREE.LinearFilter;
        loadedTex.generateMipmaps = false;
        setDepthVisTexture(loadedTex);
      },
      undefined,
      (err) => {
        console.error("Failed to load depth visualization texture:", err);
      }
    );

    return () => {
      active = false;
    };
  }, [depthVisualizationUrl]);

  // Clean up textures on unmount
  useEffect(() => {
    return () => {
      if (photoTexture) photoTexture.dispose();
      if (depthVisTexture) depthVisTexture.dispose();
    };
  }, [photoTexture, depthVisTexture]);

  // Plane dimensions: height = 2.0 meters, width = 2.0 * aspectRatio
  const height = 2.0;
  const width = Math.max(0.2, height * (aspectRatio > 0 ? aspectRatio : 1));

  // Center plane so the bottom rests gently above the ground plane (Y = height / 2)
  const positionY = height / 2;

  if (!photoTexture) {
    return null;
  }

  const entities = currentScene?.entities || [];

  return (
    <group position={[0, positionY, 0]}>
      {/* 1. Spatial Photo Mode: Depth-aware displaced 3D mesh */}
      {visualizationMode === "spatial_photo" && depthVisualizationUrl ? (
        <SpatialPhotoSurface
          photoUrl={photoUrl}
          depthUrl={depthVisualizationUrl}
          planeWidth={width}
          planeHeight={height}
          depthStrength={depthStrength}
        />
      ) : (
        /* 2. Flat 3D Plane Mode: Used for Original, Detection, Segmentation, or Depth Map */
        <>
          <mesh receiveShadow castShadow>
            <planeGeometry args={[width, height]} />
            <meshBasicMaterial
              map={
                visualizationMode === "depth" && depthVisTexture
                  ? depthVisTexture
                  : photoTexture
              }
              side={THREE.DoubleSide}
              transparent={true}
            />
          </mesh>

          {/* Border outline */}
          <lineSegments>
            <edgesGeometry args={[new THREE.PlaneGeometry(width, height)]} />
            <lineBasicMaterial
              color={visualizationMode === "depth" ? "#a1a1aa" : "#3f3f46"}
              linewidth={1}
            />
          </lineSegments>
        </>
      )}

      {/* Segmentation Mask Overlays (Only in segmentation mode) */}
      {visualizationMode === "segmentation" &&
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

      {/* Detection Bounding Boxes (In detection or segmentation mode) */}
      {(visualizationMode === "detection" || visualizationMode === "segmentation") &&
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
