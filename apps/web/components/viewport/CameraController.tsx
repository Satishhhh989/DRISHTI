"use client";

import React, { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useUIStore } from "@/stores/ui-store";

const DEFAULT_CAMERA_POS: [number, number, number] = [0, 1.2, 3.8];
const DEFAULT_TARGET_POS: [number, number, number] = [0, 1.0, 0];

type OrbitControlsElement = React.ComponentRef<typeof OrbitControls>;

export const CameraController: React.FC = () => {
  const controlsRef = useRef<OrbitControlsElement>(null);
  const { camera } = useThree();
  const cameraResetTrigger = useUIStore((state) => state.cameraResetTrigger);

  useEffect(() => {
    if (controlsRef.current) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
    }
  }, [cameraResetTrigger, camera]);

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      minDistance={0.5}
      maxDistance={20}
      target={DEFAULT_TARGET_POS}
      maxPolarAngle={Math.PI / 2 + 0.05} // prevent going fully under ground
    />
  );
};
