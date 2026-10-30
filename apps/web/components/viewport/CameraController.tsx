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
  const fitSceneTrigger = useUIStore((state) => state.fitSceneTrigger);
  const focusEntityTrigger = useUIStore((state) => state.focusEntityTrigger);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const currentScene = useSceneStore((state) => state.currentScene);

  const isSpatialPhoto = visualizationMode === "spatial_photo";

  // Smoothly reset camera to neutral front position
  useEffect(() => {
    if (controlsRef.current) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
    }
  }, [cameraResetTrigger, camera, visualizationMode]);

  // Fit camera view to frame all assembled entities
  useEffect(() => {
    if (fitSceneTrigger === 0 || !controlsRef.current) return;

    const visibleEntities = (currentScene?.entities || []).filter((e) => e.visibility !== false);
    if (visibleEntities.length === 0) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
      return;
    }

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const e of visibleEntities) {
      const p = e.transform?.position || { x: 0, y: 1, z: 0 };
      const px = Array.isArray(p) ? p[0] : p.x ?? 0;
      const py = Array.isArray(p) ? p[1] : p.y ?? 1;
      const pz = Array.isArray(p) ? p[2] : p.z ?? 0;

      minX = Math.min(minX, px - 0.5);
      maxX = Math.max(maxX, px + 0.5);
      minY = Math.min(minY, py - 0.8);
      maxY = Math.max(maxY, py + 0.8);
      minZ = Math.min(minZ, pz - 0.5);
      maxZ = Math.max(maxZ, pz + 0.5);
    }

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const spanX = maxX - minX;
    const spanY = maxY - minY;
    const maxDim = Math.max(spanX, spanY, 1.5);
    const camDist = Math.max(3.0, maxDim * 1.6);

    controlsRef.current.target.set(centerX, centerY, centerZ);
    camera.position.set(centerX, centerY + 0.2, centerZ + camDist);
    controlsRef.current.update();
  }, [fitSceneTrigger, currentScene, camera]);

  // Focus on selected person
  useEffect(() => {
    if (focusEntityTrigger === 0 || !controlsRef.current || !selectedEntityId) return;

    const entity = (currentScene?.entities || []).find((e) => e.id === selectedEntityId);
    if (!entity) return;

    const p = entity.transform?.position || { x: 0, y: 1, z: 0 };
    const px = Array.isArray(p) ? p[0] : p.x ?? 0;
    const py = Array.isArray(p) ? p[1] : p.y ?? 1;
    const pz = Array.isArray(p) ? p[2] : p.z ?? 0;

    controlsRef.current.target.set(px, py, pz);
    camera.position.set(px, py + 0.2, pz + 2.2);
    controlsRef.current.update();
  }, [focusEntityTrigger, selectedEntityId, currentScene, camera]);

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
