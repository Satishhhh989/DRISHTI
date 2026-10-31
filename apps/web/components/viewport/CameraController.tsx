"use client";

import React, { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useUIStore } from "@/stores/ui-store";
import { useSceneStore } from "@/stores/scene-store";
import { useCinematicStore } from "@/stores/cinematic-store";
import { is2DRepresentation, is3DRepresentation, VisualizationMode } from "@photo-to-3d/scene-schema";

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
  const photoDimensions = useSceneStore((state) => state.photoDimensions);

  const prevModeRef = useRef<VisualizationMode>(visualizationMode);

  const is2D = is2DRepresentation(visualizationMode);
  const isSpatialPhoto = visualizationMode === "spatial_photo";
  const is3D = is3DRepresentation(visualizationMode);

  // 1. Camera-State Management: Preserve camera state across related 3D representations
  useEffect(() => {
    const prevMode = prevModeRef.current;
    prevModeRef.current = visualizationMode;

    if (!controlsRef.current) return;

    const was3D = is3DRepresentation(prevMode);
    const now3D = is3DRepresentation(visualizationMode);

    // If switching between related 3D representations (solid3d <-> wireframe <-> articulated <-> combined),
    // PRESERVE the user's active camera orbit, pan, and zoom without unexpected jumps!
    if (was3D && now3D) {
      return;
    }

    // When transitioning across the 2D <-> 3D boundary, initialize to appropriate default view
    if (is2D) {
      camera.position.set(0, 1.0, 3.2);
      controlsRef.current.target.set(0, 1.0, 0);
      controlsRef.current.update();
    } else if (isSpatialPhoto) {
      camera.position.set(0, 1.0, 3.0);
      controlsRef.current.target.set(0, 1.0, 0);
      controlsRef.current.update();
    } else if (now3D && !was3D) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
    }
  }, [visualizationMode, is2D, isSpatialPhoto, is3D, camera]);

  // 2. Explicit Camera Reset (User clicked Reset View)
  useEffect(() => {
    if (controlsRef.current) {
      camera.position.set(...DEFAULT_CAMERA_POS);
      controlsRef.current.target.set(...DEFAULT_TARGET_POS);
      controlsRef.current.update();
    }
  }, [cameraResetTrigger, camera]);

  // 3. Reliable Fit-to-View: Computes bounds for 2D photographs or 3D multi-person scenes
  useEffect(() => {
    if (fitSceneTrigger === 0 || !controlsRef.current) return;

    // 2D Photograph Fit-to-View
    if (is2D && photoDimensions) {
      const planeH = 2.0;
      const planeW = Math.max(0.2, planeH * photoDimensions.aspect);
      const camAspect = window.innerWidth / Math.max(1, window.innerHeight);
      const fovRad = (45 * Math.PI) / 180;

      // Distance to fit both width and height with 10% padding
      const distH = (planeH / 2) / Math.tan(fovRad / 2);
      const distW = (planeW / 2) / (Math.tan(fovRad / 2) * camAspect);
      const targetDist = Math.max(distH, distW) * 1.1;

      controlsRef.current.target.set(0, 1.0, 0);
      camera.position.set(0, 1.0, targetDist);
      controlsRef.current.update();
      return;
    }

    // 3D Scene Fit-to-View: Frame all visible assembled human meshes
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

      // Bounds from metadata if available, otherwise anatomical 0.8m x 1.8m box
      const halfW = (e.reconstruction?.geometryMetadata?.dimensions?.[0] ?? 0.6) / 2;
      const halfH = (e.reconstruction?.geometryMetadata?.dimensions?.[1] ?? 1.8) / 2;
      const halfD = (e.reconstruction?.geometryMetadata?.dimensions?.[2] ?? 0.4) / 2;

      minX = Math.min(minX, px - halfW);
      maxX = Math.max(maxX, px + halfW);
      minY = Math.min(minY, py - halfH);
      maxY = Math.max(maxY, py + halfH);
      minZ = Math.min(minZ, pz - halfD);
      maxZ = Math.max(maxZ, pz + halfD);
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
  }, [fitSceneTrigger, is2D, photoDimensions, currentScene, camera]);

  // 4. Focus on selected person (Framing the entity's 3D mesh volume)
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

  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);

  return (
    <OrbitControls
      ref={controlsRef}
      enabled={!isCinematicActive}
      enableDamping
      dampingFactor={0.06}
      target={DEFAULT_TARGET_POS}
      // In 2D photographic mode, lock rotation to face photograph directly, permitting pure pan and zoom
      enableRotate={!is2D}
      minAzimuthAngle={
        is2D
          ? 0
          : isSpatialPhoto
          ? -Math.PI / 10
          : -Infinity
      }
      maxAzimuthAngle={
        is2D
          ? 0
          : isSpatialPhoto
          ? Math.PI / 10
          : Infinity
      }
      minPolarAngle={
        is2D
          ? Math.PI / 2
          : isSpatialPhoto
          ? Math.PI / 2 - Math.PI / 14
          : 0.01
      }
      maxPolarAngle={
        is2D
          ? Math.PI / 2
          : isSpatialPhoto
          ? Math.PI / 2 + Math.PI / 14
          : Math.PI / 2 + 0.05
      }
      minDistance={isSpatialPhoto ? 1.8 : 0.5}
      maxDistance={isSpatialPhoto ? 4.5 : 25}
    />
  );
};
