"use client";

import React, { useEffect, useState, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { SceneEntity, SkeletonData } from "@photo-to-3d/scene-schema";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";
import { assetCache } from "@/lib/asset-cache";

export type Unified3DMode = "wireframe" | "solid3d" | "articulated" | "combined";

interface HumanMeshProps {
  entity: SceneEntity;
  mode: Unified3DMode;
  isSelected: boolean;
  onSelect: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

function getJointCoordinates(
  pos: [number, number, number] | { x: number; y: number; z: number } | number[]
): [number, number, number] {
  if (Array.isArray(pos)) {
    return [pos[0] ?? 0, pos[1] ?? 0, pos[2] ?? 0];
  }
  if (pos && typeof pos === "object") {
    return [(pos as { x: number }).x ?? 0, (pos as { y: number }).y ?? 0, (pos as { z: number }).z ?? 0];
  }
  return [0, 0, 0];
}

interface SkeletonOverlayProps {
  skeleton: SkeletonData;
  isSelected: boolean;
  entityId: string;
}

export const SkeletonOverlay: React.FC<SkeletonOverlayProps> = ({
  skeleton,
  isSelected,
  entityId,
}) => {
  // Construct lines for canonical bones
  const boneGeometry = useMemo(() => {
    const jointMap = new Map<string, [number, number, number]>();
    for (const j of skeleton.joints) {
      jointMap.set(j.id, getJointCoordinates(j.position));
    }

    const positions: number[] = [];
    for (const bone of skeleton.bones) {
      const p1 = jointMap.get(bone.headJointId);
      const p2 = jointMap.get(bone.tailJointId);
      if (p1 && p2) {
        positions.push(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      }
    }

    const geom = new THREE.BufferGeometry();
    if (positions.length > 0) {
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    }
    return geom;
  }, [skeleton]);

  const boneColor = isSelected ? "#38bdf8" : "#0284c7";

  return (
    <group data-testid={`skeleton-overlay-${entityId}`}>
      {/* 23 Canonical Bone Connectors */}
      {skeleton.bones.length > 0 && (
        <lineSegments geometry={boneGeometry}>
          <lineBasicMaterial
            color={boneColor}
            linewidth={2}
            depthTest={false}
            transparent
            opacity={0.85}
          />
        </lineSegments>
      )}

      {/* 24 Canonical Joint Spheres */}
      {skeleton.joints.map((joint) => {
        const coords = getJointCoordinates(joint.position);
        const isRoot = joint.id === skeleton.rootJointId || joint.name.toLowerCase().includes("pelvis");
        const jointColor = isRoot ? "#f59e0b" : isSelected ? "#34d399" : "#10b981";
        const sphereRadius = isRoot ? 0.035 : 0.022;

        return (
          <mesh key={joint.id} position={coords} data-testid={`joint-node-${joint.id}`}>
            <sphereGeometry args={[sphereRadius, 12, 12]} />
            <meshBasicMaterial
              color={jointColor}
              depthTest={false}
              transparent
              opacity={0.95}
            />
          </mesh>
        );
      })}
    </group>
  );
};

const SingleHumanMesh: React.FC<HumanMeshProps> = ({
  entity,
  mode,
  isSelected,
  onSelect,
}) => {
  const [modelGroup, setModelGroup] = useState<THREE.Group | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const groupRef = useRef<THREE.Group>(null);
  const animTimeRef = useRef(0);

  const isWireframe = mode === "wireframe";
  const isArticulated = mode === "articulated";

  const rawMeshUrl = entity.reconstruction?.meshUrl;
  const meshUrl = useMemo(() => {
    if (!rawMeshUrl) return null;
    return rawMeshUrl.startsWith("http") ? rawMeshUrl : `${API_BASE}${rawMeshUrl}`;
  }, [rawMeshUrl]);

  // Load mesh from unified assetCache (deduplicated & cached across mode switches)
  useEffect(() => {
    if (!meshUrl) return;

    let isCancelled = false;

    assetCache
      .loadModel(meshUrl)
      .then((cloned) => {
        if (!isCancelled) {
          setModelGroup(cloned);
          setLoadError(null);
        }
      })
      .catch((err) => {
        if (!isCancelled) {
          console.warn(`Failed to load mesh for ${entity.id}:`, err);
          setLoadError(String(err));
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [meshUrl, entity.id]);

  // Update materials when wireframe, solid, or selection state changes without re-downloading
  useEffect(() => {
    if (!modelGroup) return;

    modelGroup.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const color = isSelected
          ? (isWireframe ? "#60a5fa" : "#93c5fd")
          : (isWireframe ? "#94a3b8" : "#cbd5e1");

        const hasVertexColors = Boolean(mesh.geometry?.attributes?.color);
        mesh.material = new THREE.MeshStandardMaterial({
          color: isWireframe
            ? new THREE.Color(color)
            : (hasVertexColors && !isSelected ? new THREE.Color(0xffffff) : new THREE.Color(color)),
          vertexColors: !isWireframe && hasVertexColors && !isSelected,
          wireframe: isWireframe,
          roughness: 0.65,
          metalness: 0.05,
          side: THREE.DoubleSide,
        });
      }
    });
  }, [modelGroup, isWireframe, isSelected]);

  const rawPos = entity.transform?.position || entity.reconstruction?.transform?.position;
  const basePosition: [number, number, number] = Array.isArray(rawPos)
    ? [rawPos[0], rawPos[1], rawPos[2]]
    : rawPos
    ? [rawPos.x, rawPos.y, rawPos.z]
    : [0, 0, 0];

  const rawScale = entity.transform?.scale || entity.reconstruction?.transform?.scale;
  const baseScale: [number, number, number] = Array.isArray(rawScale)
    ? [rawScale[0], rawScale[1], rawScale[2]]
    : rawScale
    ? [rawScale.x, rawScale.y, rawScale.z]
    : [1, 1, 1];

  // Procedural articulation loop: driven by selected procedural clip
  useFrame((_, delta) => {
    if (!groupRef.current) return;

    if (entity.animationState?.isPlaying) {
      const speed = entity.animationState.playbackSpeed ?? 1.0;
      animTimeRef.current += delta * speed;
      const t = animTimeRef.current;
      const currentClip = entity.animationState.currentClip || "idle_breathing";

      if (currentClip === "idle_breathing") {
        const breath = Math.sin((t / 3.0) * 2.0 * Math.PI);
        groupRef.current.position.y = basePosition[1] + breath * 0.015;
        groupRef.current.position.x = basePosition[0];
        groupRef.current.position.z = basePosition[2];
        groupRef.current.rotation.set(0, 0, 0);
        groupRef.current.scale.set(
          baseScale[0] * (1.0 + breath * 0.008),
          baseScale[1],
          baseScale[2] * (1.0 + breath * 0.012)
        );
      } else if (currentClip === "torso_twist") {
        const twist = 0.16 * Math.sin((t / 4.0) * 2.0 * Math.PI);
        groupRef.current.position.set(basePosition[0], basePosition[1], basePosition[2]);
        groupRef.current.rotation.set(0, twist, 0);
        groupRef.current.scale.set(baseScale[0], baseScale[1], baseScale[2]);
      } else if (currentClip === "arm_wave") {
        const waveTilt = 0.06 * Math.sin((t / 2.5) * 2.0 * Math.PI);
        groupRef.current.position.set(basePosition[0], basePosition[1], basePosition[2]);
        groupRef.current.rotation.set(0, 0, waveTilt);
        groupRef.current.scale.set(baseScale[0], baseScale[1], baseScale[2]);
      } else if (currentClip === "gentle_walk") {
        const walkPhase = (t / 2.0) * 2.0 * Math.PI;
        const bob = Math.abs(Math.sin(walkPhase)) * 0.025;
        const sway = 0.05 * Math.sin(walkPhase);
        groupRef.current.position.set(basePosition[0], basePosition[1] + bob, basePosition[2]);
        groupRef.current.rotation.set(0, sway, 0);
        groupRef.current.scale.set(baseScale[0], baseScale[1], baseScale[2]);
      }
    } else {
      // Rest pose
      groupRef.current.position.set(basePosition[0], basePosition[1], basePosition[2]);
      groupRef.current.rotation.set(0, 0, 0);
      groupRef.current.scale.set(baseScale[0], baseScale[1], baseScale[2]);
      animTimeRef.current = 0;
    }
  });

  if (!modelGroup || loadError) {
    return null;
  }

  // Show skeleton if toggled, or if in articulated mode and entity has a skeleton
  const shouldRenderSkeleton =
    Boolean(entity.skeleton) &&
    (entity.showSkeleton || (isArticulated && entity.showSkeleton !== false));

  return (
    <group
      ref={groupRef}
      position={basePosition}
      scale={baseScale}
      onClick={(e: { stopPropagation: () => void }) => {
        e.stopPropagation();
        onSelect();
      }}
      data-testid={`human-mesh-entity-${entity.id}`}
    >
      <primitive object={modelGroup} />
      {shouldRenderSkeleton && entity.skeleton && (
        <SkeletonOverlay
          skeleton={entity.skeleton}
          isSelected={isSelected}
          entityId={entity.id}
        />
      )}
    </group>
  );
};

interface UnreconstructedProps {
  entity: SceneEntity;
  isSelected: boolean;
  onSelect: () => void;
}

const UnreconstructedEntityIndicator: React.FC<UnreconstructedProps> = ({
  entity,
  isSelected,
  onSelect,
}) => {
  const rawPos = entity.transform?.position;
  const position: [number, number, number] = Array.isArray(rawPos)
    ? [rawPos[0], rawPos[1], rawPos[2]]
    : rawPos
    ? [rawPos.x, rawPos.y, rawPos.z]
    : [0, 1, 0];

  const color = isSelected ? "#f59e0b" : "#64748b";

  return (
    <group
      position={position}
      onClick={(e: { stopPropagation: () => void }) => {
        e.stopPropagation();
        onSelect();
      }}
      data-testid={`unreconstructed-indicator-${entity.id}`}
    >
      {/* Clean wireframe volume representing the unreconstructed person in scene space */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.5, 1.6, 0.35]} />
        <meshBasicMaterial
          color={color}
          wireframe
          transparent
          opacity={isSelected ? 0.85 : 0.4}
        />
      </mesh>
      {entity.showSkeleton && entity.skeleton && (
        <SkeletonOverlay
          skeleton={entity.skeleton}
          isSelected={isSelected}
          entityId={entity.id}
        />
      )}
    </group>
  );
};

export const HumanMeshLayer: React.FC<{ mode: Unified3DMode }> = ({ mode }) => {
  const entities = useSceneStore((state) => state.currentScene?.entities || []);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const selectEntity = useUIStore((state) => state.selectEntity);

  const visibleEntities = useMemo(() => {
    return entities.filter((e) => e.visibility !== false);
  }, [entities]);

  const { reconstructedEntities, unreconstructedEntities } = useMemo(() => {
    const rec: SceneEntity[] = [];
    const unrec: SceneEntity[] = [];
    for (const e of visibleEntities) {
      if (
        e.reconstruction?.status === "ready" &&
        Boolean(e.reconstruction?.meshUrl || e.reconstruction?.meshAssetId)
      ) {
        rec.push(e);
      } else {
        unrec.push(e);
      }
    }
    return { reconstructedEntities: rec, unreconstructedEntities: unrec };
  }, [visibleEntities]);

  return (
    <group data-testid="human-mesh-layer">
      {/* Studio Lighting for Clean Human Geometry Inspection */}
      <ambientLight intensity={0.65} />
      <directionalLight position={[3, 5, 4]} intensity={0.9} />
      <directionalLight position={[-3, 3, -4]} intensity={0.5} />
      <directionalLight position={[0, -4, 2]} intensity={0.3} />

      {/* Reconstructed Volumetric 3D Meshes */}
      {reconstructedEntities.map((entity) => (
        <SingleHumanMesh
          key={entity.id}
          entity={entity}
          mode={mode}
          isSelected={selectedEntityId === entity.id}
          onSelect={() => selectEntity(entity.id)}
        />
      ))}

      {/* Unreconstructed Entities Displayed Honestly In-Scene */}
      {unreconstructedEntities.map((entity) => (
        <UnreconstructedEntityIndicator
          key={entity.id}
          entity={entity}
          isSelected={selectedEntityId === entity.id}
          onSelect={() => selectEntity(entity.id)}
        />
      ))}
    </group>
  );
};
