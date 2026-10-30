"use client";

import React, { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

interface PhotoPlaneProps {
  photoUrl: string;
  aspectRatio: number; // width / height
}

export const PhotoPlane: React.FC<PhotoPlaneProps> = ({ photoUrl, aspectRatio }) => {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

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
    </group>
  );
};
