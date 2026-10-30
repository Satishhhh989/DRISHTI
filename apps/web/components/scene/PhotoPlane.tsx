"use client";

import React, { useEffect, useState } from "react";
import * as THREE from "three";
import { useSceneStore } from "@/stores/scene-store";
import { DetectionBox } from "./DetectionBox";
import { SegmentationMaskLayer } from "./SegmentationMaskLayer";

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
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  const currentScene = useSceneStore((state) => state.currentScene);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);

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
        setTexture(loadedTex);
      },
      undefined,
      (err) => {
        console.error("Failed to load photograph texture into 3D plane:", err);
      }
    );

    return () => {
      active = false;
      if (texture) {
        texture.dispose();
      }
    };
  }, [photoUrl]);

  // Plane dimensions: height = 2.0 meters, width = 2.0 * aspectRatio
  const height = 2.0;
  const width = Math.max(0.2, height * (aspectRatio > 0 ? aspectRatio : 1));

  // Center plane so the bottom rests gently above the ground plane (Y = height / 2)
  const positionY = height / 2;

  if (!texture) {
    return null;
  }

  const entities = currentScene?.entities || [];

  return (
    <group position={[0, positionY, 0]}>
      {/* Primary photographic image plane */}
      <mesh receiveShadow castShadow>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial
          map={texture}
          side={THREE.DoubleSide}
          transparent={true}
        />
      </mesh>

      {/* Subtle border outline for the image plane in 3D space */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(width, height)]} />
        <lineBasicMaterial color="#3f3f46" linewidth={1} />
      </lineSegments>

      {/* Segmentation Mask Overlays */}
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

      {/* Detection Bounding Boxes */}
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
