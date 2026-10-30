"use client";

import React, { useEffect, useState } from "react";
import * as THREE from "three";

interface SpatialPhotoSurfaceProps {
  photoUrl: string;
  depthUrl: string;
  planeWidth: number;
  planeHeight: number;
  depthStrength: number;
}

export const SpatialPhotoSurface: React.FC<SpatialPhotoSurfaceProps> = ({
  photoUrl,
  depthUrl,
  planeWidth,
  planeHeight,
  depthStrength,
}) => {
  const [photoTexture, setPhotoTexture] = useState<THREE.Texture | null>(null);
  const [depthTexture, setDepthTexture] = useState<THREE.Texture | null>(null);

  // 1. Load photo texture
  useEffect(() => {
    let active = true;
    const loader = new THREE.TextureLoader();

    loader.load(
      photoUrl,
      (tex) => {
        if (!active) {
          tex.dispose();
          return;
        }
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.minFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        setPhotoTexture(tex);
      },
      undefined,
      (err) => {
        console.error("Failed to load photograph texture for spatial photo:", err);
      }
    );

    return () => {
      active = false;
    };
  }, [photoUrl]);

  // 2. Load depth texture
  useEffect(() => {
    let active = true;
    const loader = new THREE.TextureLoader();

    loader.load(
      depthUrl,
      (tex) => {
        if (!active) {
          tex.dispose();
          return;
        }
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.generateMipmaps = false;
        setDepthTexture(tex);
      },
      undefined,
      (err) => {
        console.error("Failed to load depth texture for spatial photo:", err);
      }
    );

    return () => {
      active = false;
    };
  }, [depthUrl]);

  // Dispose textures when component unmounts
  useEffect(() => {
    return () => {
      if (photoTexture) photoTexture.dispose();
      if (depthTexture) depthTexture.dispose();
    };
  }, [photoTexture, depthTexture]);

  if (!photoTexture || !depthTexture) {
    return null;
  }

  // Constrain depth strength to sensible bounds [0.0, 0.25]
  const clampedScale = Math.max(0.0, Math.min(0.25, depthStrength));

  return (
    <group data-testid="spatial-photo-surface">
      {/* Depth-displaced 3D photo mesh */}
      <mesh receiveShadow castShadow>
        <planeGeometry args={[planeWidth, planeHeight, 128, 128]} />
        <meshStandardMaterial
          map={photoTexture}
          displacementMap={depthTexture}
          displacementScale={clampedScale}
          displacementBias={0.0}
          roughness={0.9}
          metalness={0.0}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Spatial coordinate border outline */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(planeWidth, planeHeight)]} />
        <lineBasicMaterial color="#06b6d4" transparent opacity={0.4} linewidth={1} />
      </lineSegments>
    </group>
  );
};

