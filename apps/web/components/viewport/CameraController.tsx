"use client";

import React, { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useUIStore } from "@/stores/ui-store";
import { useSceneStore } from "@/stores/scene-store";

const DEFAULT_CAMERA_POS: [number, number, number] = [0, 1.0, 3.2];
const DEFAULT_TARGET_POS: [number, number, number] = [0, 1.0, 0];

type OrbitControlsElement = React.ComponentRef<typeof OrbitControls>;

export const CameraController: React.FC = () => {
  const controlsRef = useRef<OrbitControlsElement>(null);
  const { camera } = useThree();
  const cameraResetTrigger = useUIStore((state) => state.cameraResetTrigger);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);

  const isSpatialPhoto = visualizationMode === "spatial_photo";

  // Smoothly reset camera to neutral front position
  useEffect(() => {
    if (controlsRef.current) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
    }
  }, [cameraResetTrigger, camera, visualizationMode]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.06}
      target={DEFAULT_TARGET_POS}
      // In Spatial Photo mode, apply restrained parallax camera bounds to prevent disocclusion holes & distortion
      minAzimuthAngle={isSpatialPhoto ? -Math.PI / 10 : -Infinity}
      maxAzimuthAngle={isSpatialPhoto ? Math.PI / 10 : Infinity}
      minPolarAngle={isSpatialPhoto ? Math.PI / 2 - Math.PI / 14 : 0.01}
      maxPolarAngle={isSpatialPhoto ? Math.PI / 2 + Math.PI / 14 : Math.PI / 2 + 0.05}
      minDistance={isSpatialPhoto ? 1.8 : 0.5}
      maxDistance={isSpatialPhoto ? 4.5 : 20}
    />
  );
};
