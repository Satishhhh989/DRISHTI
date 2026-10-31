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
  presentationProgress?: number;
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
  entity?: SceneEntity;
  animTimeRef?: React.MutableRefObject<number>;
}

export const SkeletonOverlay: React.FC<SkeletonOverlayProps> = ({
  skeleton,
  isSelected,
  entityId,
  entity,
  animTimeRef,
}) => {
  const lineGeometryRef = useRef<THREE.BufferGeometry>(null);
  const jointMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());

  // Base coordinates mapping for all 24 canonical joints
  const baseJointMap = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const j of skeleton.joints) {
      map.set(j.id, getJointCoordinates(j.position));
    }
    return map;
  }, [skeleton]);

  // Construct initial geometry for canonical bones
  const boneGeometry = useMemo(() => {
    const positions: number[] = [];
    for (const bone of skeleton.bones) {
      const p1 = baseJointMap.get(bone.headJointId);
      const p2 = baseJointMap.get(bone.tailJointId);
      if (p1 && p2) {
        positions.push(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      }
    }

    const geom = new THREE.BufferGeometry();
    if (positions.length > 0) {
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    }
    return geom;
  }, [skeleton, baseJointMap]);

  // Real-time skeletal joint articulation in sync with procedural clips
  useFrame(() => {
    if (!entity?.animationState?.isPlaying || !animTimeRef) {
      // Rest pose: restore exact canonical joint coordinates
      baseJointMap.forEach((coords, jId) => {
        const mesh = jointMeshesRef.current.get(jId);
        if (mesh) {
          mesh.position.set(coords[0], coords[1], coords[2]);
        }
      });
      if (lineGeometryRef.current && skeleton.bones.length > 0) {
        const posAttr = lineGeometryRef.current.attributes.position as THREE.BufferAttribute;
        if (posAttr) {
          let idx = 0;
          for (const bone of skeleton.bones) {
            const p1 = baseJointMap.get(bone.headJointId) || [0, 0, 0];
            const p2 = baseJointMap.get(bone.tailJointId) || [0, 0, 0];
            posAttr.setXYZ(idx++, p1[0], p1[1], p1[2]);
            posAttr.setXYZ(idx++, p2[0], p2[1], p2[2]);
          }
          posAttr.needsUpdate = true;
        }
      }
      return;
    }

    const t = animTimeRef.current;
    const stanceDesc = (entity.reconstruction?.stanceDescription || "").toLowerCase();
    const isSeated = stanceDesc.includes("seated") || (entity.animationState.currentClip || "").startsWith("seated");
    const clip = entity.animationState.currentClip || (isSeated ? "seated_idle" : "idle_breathing");

    const animatedCoords = new Map<string, [number, number, number]>();

    baseJointMap.forEach((base, jId) => {
      let dx = 0;
      let dy = 0;
      let dz = 0;

      if (clip === "seated_idle") {
        const phase = (t / 3.5) * 2.0 * Math.PI;
        if (jId === "spine3" || jId.includes("shoulder")) {
          dy = 0.012 * Math.sin(phase);
          dz = 0.008 * Math.sin(phase);
        } else if (jId === "head" || jId === "neck") {
          dy = 0.007 * Math.sin(phase);
          dz = 0.005 * Math.cos(phase * 0.7);
        }
        // Lower body (pelvis, hips, knees, ankles) stays at 0 displacement in seated pose
      } else if (clip === "seated_weight_shift") {
        const phase = (t / 4.5) * 2.0 * Math.PI;
        const roll = 0.025 * Math.sin(phase);
        if (jId.startsWith("spine") || jId.includes("shoulder") || jId.includes("elbow") || jId.includes("wrist")) {
          dx = roll;
          dz = 0.012 * Math.cos(phase * 0.5);
        } else if (jId === "head" || jId === "neck") {
          dx = -roll * 0.5;
        }
      } else if (clip === "idle_breathing" || clip === "standing_idle") {
        const phase = (t / 3.0) * 2.0 * Math.PI;
        if (jId === "spine3" || jId.includes("shoulder")) {
          dy = 0.015 * Math.sin(phase);
          dz = 0.010 * Math.sin(phase);
        }
      } else if (clip === "torso_twist") {
        const phase = (t / 4.0) * 2.0 * Math.PI;
        const twistAngle = 0.16 * Math.sin(phase);
        if (jId === "spine2" || jId === "spine3" || jId.includes("shoulder") || jId.includes("arm") || jId.includes("elbow") || jId.includes("wrist")) {
          const cosA = Math.cos(twistAngle);
          const sinA = Math.sin(twistAngle);
          const x0 = base[0];
          const z0 = base[2];
          animatedCoords.set(jId, [x0 * cosA - z0 * sinA, base[1], x0 * sinA + z0 * cosA]);
          return;
        } else if (jId === "head") {
          const headTwist = -twistAngle * 0.5;
          const cosA = Math.cos(headTwist);
          const sinA = Math.sin(headTwist);
          const x0 = base[0];
          const z0 = base[2];
          animatedCoords.set(jId, [x0 * cosA - z0 * sinA, base[1], x0 * sinA + z0 * cosA]);
          return;
        }
      } else if (clip === "arm_wave") {
        const phase = (t / 2.5) * 2.0 * Math.PI;
        if (jId === "right_elbow") {
          dx = 0.04 * Math.sin(phase * 2.0);
          dy = 0.03 * Math.cos(phase * 2.0);
        } else if (jId === "right_wrist" || jId.includes("right_hand")) {
          dx = 0.08 * Math.sin(phase * 2.0);
          dy = 0.05 * Math.cos(phase * 2.0);
        }
      } else if (clip === "gentle_walk") {
        const phase = (t / 2.0) * 2.0 * Math.PI;
        if (jId === "left_knee" || jId === "left_ankle" || jId === "left_foot") {
          dz = 0.08 * Math.sin(phase);
          dy = Math.max(0, 0.03 * Math.sin(phase));
        } else if (jId === "right_knee" || jId === "right_ankle" || jId === "right_foot") {
          dz = -0.08 * Math.sin(phase);
          dy = Math.max(0, -0.03 * Math.sin(phase));
        } else if (jId === "left_wrist" || jId === "left_elbow") {
          dz = -0.06 * Math.sin(phase);
        } else if (jId === "right_wrist" || jId === "right_elbow") {
          dz = 0.06 * Math.sin(phase);
        }
      }

      animatedCoords.set(jId, [base[0] + dx, base[1] + dy, base[2] + dz]);
    });

    // Update joint spheres positions
    animatedCoords.forEach((coords, jId) => {
      const mesh = jointMeshesRef.current.get(jId);
      if (mesh) {
        mesh.position.set(coords[0], coords[1], coords[2]);
      }
    });

    // Update bone line segments
    if (lineGeometryRef.current && skeleton.bones.length > 0) {
      const posAttr = lineGeometryRef.current.attributes.position as THREE.BufferAttribute;
      if (posAttr) {
        let idx = 0;
        for (const bone of skeleton.bones) {
          const p1 = animatedCoords.get(bone.headJointId) || baseJointMap.get(bone.headJointId) || [0, 0, 0];
          const p2 = animatedCoords.get(bone.tailJointId) || baseJointMap.get(bone.tailJointId) || [0, 0, 0];
          posAttr.setXYZ(idx++, p1[0], p1[1], p1[2]);
          posAttr.setXYZ(idx++, p2[0], p2[1], p2[2]);
        }
        posAttr.needsUpdate = true;
      }
    }
  });

  const boneColor = isSelected ? "#38bdf8" : "#0284c7";

  return (
    <group data-testid={`skeleton-overlay-${entityId}`}>
      {/* 23 Canonical Bone Connectors */}
      {skeleton.bones.length > 0 && (
        <lineSegments geometry={boneGeometry} ref={lineGeometryRef as unknown as React.Ref<THREE.LineSegments>}>
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
        const coords = baseJointMap.get(joint.id) || [0, 0, 0];
        const isRoot = joint.id === skeleton.rootJointId || joint.name.toLowerCase().includes("pelvis");
        const jointColor = isRoot ? "#f59e0b" : isSelected ? "#34d399" : "#10b981";
        const sphereRadius = isRoot ? 0.035 : 0.022;

        return (
          <mesh
            key={joint.id}
            position={coords}
            ref={(node) => {
              if (node) jointMeshesRef.current.set(joint.id, node);
              else jointMeshesRef.current.delete(joint.id);
            }}
            data-testid={`joint-node-${joint.id}`}
          >
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
  presentationProgress,
}) => {
  const [modelGroup, setModelGroup] = useState<THREE.Group | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const groupRef = useRef<THREE.Group>(null);
  const animTimeRef = useRef(0);
  const activeMeshRef = useRef<THREE.Mesh | null>(null);
  const originalPositionsRef = useRef<Float32Array | null>(null);

  const photoDimensions = useSceneStore((state) => state.photoDimensions);
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

  // Update materials and cache original pristine vertex coordinates
  useEffect(() => {
    if (!modelGroup) return;

    modelGroup.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        activeMeshRef.current = mesh;

        // Cache original reconstructed vertex buffer on first load
        if (!originalPositionsRef.current && mesh.geometry?.attributes?.position) {
          originalPositionsRef.current = new Float32Array(mesh.geometry.attributes.position.array);
        }

        const hasVertexColors = Boolean(mesh.geometry?.attributes?.color);
        const wireColor = isSelected ? "#60a5fa" : "#94a3b8";
        const fallbackClayColor = "#cbd5e1"; // Neutral matte studio clay

        mesh.material = new THREE.MeshStandardMaterial({
          color: isWireframe
            ? new THREE.Color(wireColor)
            : hasVertexColors
            ? new THREE.Color(0xffffff)
            : new THREE.Color(fallbackClayColor),
          vertexColors: !isWireframe && hasVertexColors,
          wireframe: isWireframe,
          roughness: 0.65,
          metalness: 0.05,
          side: THREE.DoubleSide,
          emissive: isSelected ? new THREE.Color("#2563eb") : new THREE.Color(0x000000),
          emissiveIntensity: isSelected ? 0.35 : 0.0,
        });
      }
    });
  }, [modelGroup, isWireframe, isSelected]);

  const rawPos = entity.transform?.position || entity.reconstruction?.transform?.position;
  const basePosition: [number, number, number] = useMemo(() => {
    return Array.isArray(rawPos)
      ? [rawPos[0], rawPos[1], rawPos[2]]
      : rawPos
      ? [rawPos.x, rawPos.y, rawPos.z]
      : [0, 0, 0];
  }, [rawPos]);

  const rawScale = entity.transform?.scale || entity.reconstruction?.transform?.scale;
  const baseScale: [number, number, number] = useMemo(() => {
    return Array.isArray(rawScale)
      ? [rawScale[0], rawScale[1], rawScale[2]]
      : rawScale
      ? [rawScale.x, rawScale.y, rawScale.z]
      : [1, 1, 1];
  }, [rawScale]);

  // Visual presentation position: during scene assembly, smoothly steps forward from photo plane
  const visualPosition: [number, number, number] = useMemo(() => {
    const box = entity.boundingBox || entity.detection?.boundingBox;
    if (
      presentationProgress !== undefined &&
      presentationProgress < 1.0 &&
      box &&
      photoDimensions
    ) {
      const planeH = 2.0;
      const planeW = planeH * photoDimensions.aspect;
      const photoX = ((box.xMin + box.xMax) / 2 - 0.5) * planeW;
      const photoY = planeH / 2 + (0.5 - (box.yMin + box.yMax) / 2) * planeH;
      const t = THREE.MathUtils.smoothstep(presentationProgress, 0.0, 1.0);
      return [
        photoX + (basePosition[0] - photoX) * t,
        photoY + (basePosition[1] - photoY) * t,
        0.0 + (basePosition[2] - 0.0) * t,
      ];
    }
    return basePosition;
  }, [basePosition, presentationProgress, entity.boundingBox, entity.detection, photoDimensions]);

  // Procedural articulation & skeletal deformation loop
  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // Anchor the root group position at visualPosition (no artificial whole-person bobbing or wandering)
    groupRef.current.position.set(visualPosition[0], visualPosition[1], visualPosition[2]);
    groupRef.current.rotation.set(0, 0, 0);
    groupRef.current.scale.set(baseScale[0], baseScale[1], baseScale[2]);

    if (entity.animationState?.isPlaying) {
      const speed = entity.animationState.playbackSpeed ?? 1.0;
      animTimeRef.current += delta * speed;
      const t = animTimeRef.current;

      const stanceDesc = (entity.reconstruction?.stanceDescription || "").toLowerCase();
      const isSeated = stanceDesc.includes("seated") || (entity.animationState.currentClip || "").startsWith("seated");
      const currentClip = entity.animationState.currentClip || (isSeated ? "seated_idle" : "idle_breathing");

      // Deform mesh vertices in real-time according to skeletal movement
      if (activeMeshRef.current?.geometry?.attributes?.position && originalPositionsRef.current) {
        const posAttr = activeMeshRef.current.geometry.attributes.position;
        const orig = originalPositionsRef.current;
        const pos = posAttr.array as Float32Array;
        const vertexCount = orig.length / 3;

        if (currentClip === "seated_idle") {
          const phase = (t / 3.5) * 2.0 * Math.PI;
          const breath = Math.sin(phase);
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            // Only upper chest and shoulders breathe; hips and bent knees stay 100% seated
            const wChest = Math.max(0, Math.min(1, (y0 - 0.05) / 0.45)) * Math.max(0, Math.min(1, (0.8 - y0) / 0.3));
            pos[3 * i] = x0 * (1.0 + 0.006 * breath * wChest);
            pos[3 * i + 1] = y0 + 0.010 * breath * wChest;
            pos[3 * i + 2] = z0 + 0.014 * breath * wChest;
          }
        } else if (currentClip === "seated_weight_shift") {
          const phase = (t / 4.5) * 2.0 * Math.PI;
          const roll = 0.03 * Math.sin(phase);
          const yaw = 0.02 * Math.cos(phase * 0.5);
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            const wTorso = Math.max(0, Math.min(1, y0 / 0.6));
            pos[3 * i] = x0 + roll * wTorso;
            pos[3 * i + 1] = y0;
            pos[3 * i + 2] = z0 + yaw * x0 * wTorso;
          }
        } else if (currentClip === "idle_breathing" || currentClip === "standing_idle") {
          const phase = (t / 3.0) * 2.0 * Math.PI;
          const breath = Math.sin(phase);
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            const wChest = Math.max(0, Math.min(1, (y0 - 0.05) / 0.45));
            pos[3 * i] = x0 * (1.0 + 0.008 * breath * wChest);
            pos[3 * i + 1] = y0 + 0.012 * breath * wChest;
            pos[3 * i + 2] = z0 + 0.016 * breath * wChest;
          }
        } else if (currentClip === "torso_twist") {
          const phase = (t / 4.0) * 2.0 * Math.PI;
          const twist = 0.16 * Math.sin(phase);
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            const wTwist = Math.max(0, Math.min(1, y0 / 0.55));
            const angle = twist * wTwist;
            const cosA = Math.cos(angle);
            const sinA = Math.sin(angle);
            pos[3 * i] = x0 * cosA - z0 * sinA;
            pos[3 * i + 1] = y0;
            pos[3 * i + 2] = x0 * sinA + z0 * cosA;
          }
        } else if (currentClip === "arm_wave") {
          const phase = (t / 2.5) * 2.0 * Math.PI;
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            if (x0 > 0.15 && y0 > -0.2) {
              const wArm = Math.max(0, Math.min(1, (x0 - 0.15) / 0.35));
              pos[3 * i] = x0 + 0.07 * Math.sin(phase * 2.0) * wArm;
              pos[3 * i + 1] = y0 + 0.04 * Math.cos(phase * 2.0) * wArm;
              pos[3 * i + 2] = z0;
            } else {
              pos[3 * i] = x0;
              pos[3 * i + 1] = y0;
              pos[3 * i + 2] = z0;
            }
          }
        } else if (currentClip === "gentle_walk") {
          const phase = (t / 2.0) * 2.0 * Math.PI;
          for (let i = 0; i < vertexCount; i++) {
            const x0 = orig[3 * i];
            const y0 = orig[3 * i + 1];
            const z0 = orig[3 * i + 2];
            if (y0 < 0.0) {
              // Legs stride
              const wLeg = Math.max(0, Math.min(1, -y0 / 0.75));
              if (x0 < 0.0) {
                // Left leg
                pos[3 * i] = x0;
                pos[3 * i + 1] = y0 + Math.max(0, 0.03 * Math.sin(phase)) * wLeg;
                pos[3 * i + 2] = z0 + 0.08 * Math.sin(phase) * wLeg;
              } else {
                // Right leg
                pos[3 * i] = x0;
                pos[3 * i + 1] = y0 + Math.max(0, -0.03 * Math.sin(phase)) * wLeg;
                pos[3 * i + 2] = z0 - 0.08 * Math.sin(phase) * wLeg;
              }
            } else {
              // Arms counter-swing
              const wArm = Math.max(0, Math.min(1, Math.abs(x0) / 0.4));
              const armDir = x0 < 0.0 ? -1 : 1;
              pos[3 * i] = x0;
              pos[3 * i + 1] = y0;
              pos[3 * i + 2] = z0 - 0.05 * Math.sin(phase) * armDir * wArm;
            }
          }
        }
        posAttr.needsUpdate = true;
      }
    } else {
      // Stopped / Rest pose / Reset: restore exact canonical photograph reconstructed pose!
      if (originalPositionsRef.current && activeMeshRef.current?.geometry?.attributes?.position) {
        const posAttr = activeMeshRef.current.geometry.attributes.position;
        (posAttr.array as Float32Array).set(originalPositionsRef.current);
        posAttr.needsUpdate = true;
      }
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
      position={visualPosition}
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
          entity={entity}
          animTimeRef={animTimeRef}
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
          entity={entity}
        />
      )}
    </group>
  );
};

export const HumanMeshLayer: React.FC<{
  mode: Unified3DMode;
  presentationProgress?: number;
}> = ({ mode, presentationProgress }) => {
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
          presentationProgress={presentationProgress}
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
