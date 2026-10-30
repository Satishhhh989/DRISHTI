"use client";

import React, { useEffect, useState, useMemo } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { SceneEntity } from "@photo-to-3d/scene-schema";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";

interface HumanMeshProps {
  entity: SceneEntity;
  isWireframe: boolean;
  isSelected: boolean;
  onSelect: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const SingleHumanMesh: React.FC<HumanMeshProps> = ({
  entity,
  isWireframe,
  isSelected,
  onSelect,
}) => {
  const [modelGroup, setModelGroup] = useState<THREE.Group | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const rawMeshUrl = entity.reconstruction?.meshUrl;
  const meshUrl = useMemo(() => {
    if (!rawMeshUrl) return null;
    return rawMeshUrl.startsWith("http") ? rawMeshUrl : `${API_BASE}${rawMeshUrl}`;
  }, [rawMeshUrl]);

  useEffect(() => {
    if (!meshUrl) return;

    let isCancelled = false;
    const loader = new GLTFLoader();

    loader.load(
      meshUrl,
      (gltf) => {
        if (isCancelled) return;
        const clonedScene = gltf.scene.clone();
        setModelGroup(clonedScene);
        setLoadError(null);
      },
      undefined,
      (err) => {
        if (isCancelled) return;
        console.warn(`Failed to load mesh for ${entity.id}:`, err);
        setLoadError(String(err));
      }
    );

    return () => {
      isCancelled = true;
    };
  }, [meshUrl, entity.id]);

  // Update materials when isWireframe or isSelected changes without re-downloading
  useEffect(() => {
    if (!modelGroup) return;

    modelGroup.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const color = isSelected
          ? (isWireframe ? "#60a5fa" : "#93c5fd")
          : (isWireframe ? "#94a3b8" : "#cbd5e1");

        mesh.material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(color),
          wireframe: isWireframe,
          roughness: 0.55,
          metalness: 0.08,
          side: THREE.DoubleSide,
        });
      }
    });
  }, [modelGroup, isWireframe, isSelected]);

  // Cleanup geometry and materials when unmounting
  useEffect(() => {
    return () => {
      if (modelGroup) {
        modelGroup.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            mesh.geometry?.dispose();
            if (Array.isArray(mesh.material)) {
              mesh.material.forEach((m) => m.dispose());
            } else {
              mesh.material?.dispose();
            }
          }
        });
      }
    };
  }, [modelGroup]);

  if (!modelGroup || loadError) {
    return null;
  }

  const rawPos = entity.reconstruction?.transform?.position || entity.transform?.position;
  const position: [number, number, number] = Array.isArray(rawPos)
    ? [rawPos[0], rawPos[1], rawPos[2]]
    : rawPos
    ? [rawPos.x, rawPos.y, rawPos.z]
    : [0, 0, 0];

  const rawScale = entity.reconstruction?.transform?.scale || entity.transform?.scale;
  const scale: [number, number, number] = Array.isArray(rawScale)
    ? [rawScale[0], rawScale[1], rawScale[2]]
    : rawScale
    ? [rawScale.x, rawScale.y, rawScale.z]
    : [1, 1, 1];

  return (
    <primitive
      object={modelGroup}
      position={position}
      scale={scale}
      onClick={(e: { stopPropagation: () => void }) => {
        e.stopPropagation();
        onSelect();
      }}
    />
  );
};

export const HumanMeshLayer: React.FC<{ mode: "wireframe" | "solid3d" }> = ({ mode }) => {
  const entities = useSceneStore((state) => state.currentScene?.entities || []);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const selectEntity = useUIStore((state) => state.selectEntity);

  const reconstructedEntities = useMemo(() => {
    return entities.filter(
      (e) =>
        e.reconstruction?.status === "ready" &&
        Boolean(e.reconstruction?.meshUrl || e.reconstruction?.meshAssetId)
    );
  }, [entities]);

  const isWireframe = mode === "wireframe";

  return (
    <group data-testid="human-mesh-layer">
      {/* Studio Lighting for Clean Human Geometry Inspection */}
      <ambientLight intensity={0.65} />
      <directionalLight position={[3, 5, 4]} intensity={0.9} />
      <directionalLight position={[-3, 3, -4]} intensity={0.5} />
      <directionalLight position={[0, -4, 2]} intensity={0.3} />

      {reconstructedEntities.map((entity) => (
        <SingleHumanMesh
          key={entity.id}
          entity={entity}
          isWireframe={isWireframe}
          isSelected={selectedEntityId === entity.id}
          onSelect={() => selectEntity(entity.id)}
        />
      ))}
    </group>
  );
};
