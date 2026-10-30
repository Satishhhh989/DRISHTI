"use client";

import React, { useEffect, useState } from "react";
import * as THREE from "three";
import { SceneEntity } from "@photo-to-3d/scene-schema";
import { useUIStore } from "@/stores/ui-store";

interface SegmentationMaskLayerProps {
  entity: SceneEntity;
  planeWidth: number;
  planeHeight: number;
  colorHex: number;
}

export const SegmentationMaskLayer: React.FC<SegmentationMaskLayerProps> = ({
  entity,
  planeWidth,
  planeHeight,
  colorHex,
}) => {
  const [maskTexture, setMaskTexture] = useState<THREE.Texture | null>(null);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const isSelected = selectedEntityId === entity.id;

  const maskUri = entity.segmentation?.maskUri;

  useEffect(() => {
    if (!maskUri) {
      setMaskTexture(null);
      return;
    }

    let active = true;
    const loader = new THREE.TextureLoader();

    // Mask URI might be relative (/api/v1/assets/...) or absolute
    const fullUrl = maskUri.startsWith("http")
      ? maskUri
      : `${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}${maskUri}`;

    loader.load(
      fullUrl,
      (tex) => {
        if (!active) {
          tex.dispose();
          return;
        }
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        setMaskTexture(tex);
      },
      undefined,
      (err) => {
        console.warn(`Could not load mask texture for entity ${entity.id}:`, err);
      }
    );

    return () => {
      active = false;
      if (maskTexture) {
        maskTexture.dispose();
      }
    };
  }, [maskUri, entity.id]);

  if (!maskTexture) return null;

  return (
    <mesh position={[0, 0, 0.002]}>
      <planeGeometry args={[planeWidth, planeHeight]} />
      <meshBasicMaterial
        alphaMap={maskTexture}
        color={colorHex}
        transparent={true}
        opacity={isSelected ? 0.7 : 0.42}
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
};
