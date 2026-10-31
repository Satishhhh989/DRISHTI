import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ContextualPanel } from "../components/layout/ContextualPanel";
import { useSceneStore } from "../stores/scene-store";
import { useUIStore } from "../stores/ui-store";
import { Scene, SceneEntity } from "@photo-to-3d/scene-schema";

// Mock R3F Three.js Canvas / WebGL context
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="mock-r3f-canvas">{children}</div>,
  useFrame: vi.fn(),
  useThree: () => ({
    camera: { position: { set: vi.fn() } },
  }),
}));

vi.mock("three/examples/jsm/loaders/GLTFLoader.js", () => {
  return {
    GLTFLoader: vi.fn().mockImplementation(() => ({
      load: (_url: string, onLoad: (gltf: { scene: { clone: () => unknown } }) => void) => {
        onLoad({
          scene: {
            clone: () => ({
              traverse: vi.fn(),
            }),
          },
        });
      },
    })),
  };
});

describe("Phase 07 DRISHTI Motion Subsystem View & Controls", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useUIStore.getState().selectEntity(null);
  });

  it("renders pipeline inspector with Pose Estimation, Skeleton Fitting, Mesh Rigging, and Animation Preparation stage rows", () => {
    render(<ContextualPanel />);
    expect(screen.getByTestId("stage-row-pose")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-skeleton")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-rigging")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-animation")).toBeInTheDocument();

    expect(screen.getByText("Pose Estimation")).toBeInTheDocument();
    expect(screen.getByText("Skeleton Fitting")).toBeInTheDocument();
    expect(screen.getByText("Mesh Rigging")).toBeInTheDocument();
    expect(screen.getByText("Procedural Motion")).toBeInTheDocument();
  });

  it("inspects skeleton, rigging, and motion controls for rigged person entity", async () => {
    const mockEntity: SceneEntity = {
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-001",
      name: "Person 1",
      visibility: true,
      reconstructionStatus: "ready",
      transform: {
        position: { x: 0.0, y: 1.0, z: 0.0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1.0, y: 1.0, z: 1.0 },
      },
      animationState: {
        currentClip: "idle_breathing",
        isPlaying: false,
        playbackSpeed: 1.0,
        loop: true,
      },
      showSkeleton: false,
      reconstruction: {
        status: "ready",
        meshAssetId: "mesh-001",
        meshUrl: "/api/v1/assets/meshes/person_001.glb",
        textureAvailability: "none",
        geometryMetadata: {
          vertexCount: 6890,
          faceCount: 13776,
          bounds: { min: [-0.3, -0.9, -0.2], max: [0.3, 0.9, 0.2] },
          dimensions: [0.6, 1.8, 0.4],
        },
      },
      skeleton: {
        id: "skel-person-001",
        entityId: "person-001",
        schemaVersion: "1.0.0",
        rootJointId: "pelvis",
        joints: [
          {
            id: "pelvis",
            name: "Pelvis",
            index: 0,
            parentId: null,
            position: [0.0, 0.0, 0.0],
            rotation: [0, 0, 0, 1],
            isObserved: true,
          },
          {
            id: "spine1",
            name: "Spine1",
            index: 3,
            parentId: "pelvis",
            position: [0.0, 0.12, 0.0],
            rotation: [0, 0, 0, 1],
            isObserved: true,
          },
        ],
        bones: [
          {
            id: "bone-pelvis-spine1",
            headJointId: "pelvis",
            tailJointId: "spine1",
            length: 0.12,
          },
        ],
        coordinateConvention: {
          up: "+Y",
          forward: "+Z",
          right: "+X",
          units: "meters",
        },
        source: "smpl_regressed",
        isAligned: true,
        validationStatus: "valid",
        validationErrors: [],
        createdAt: new Date().toISOString(),
      },
      rig: {
        id: "rig-person-001",
        entityId: "person-001",
        skeletonId: "skel-person-001",
        status: "rigged",
        skinningMethod: "smpl_lbs_native",
        boneCount: 24,
        maxInfluencesPerVertex: 4,
        weightsNormalized: true,
      },
    };

    const mockScene: Scene = {
      id: "scene-motion-001",
      schemaVersion: "1.0.0",
      name: "Motion Test Scene",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      sourceImage: {
        assetId: "photo-001",
        filename: "test.jpg",
        width: 1000,
        height: 800,
        aspectRatio: 1.25,
        mimeType: "image/jpeg",
        sizeBytes: 120000,
        uploadedAt: new Date().toISOString(),
      },
      coordinateSystem: {
        handedness: "right-handed",
        upAxis: "+Y",
        forwardAxis: "+Z",
        rightAxis: "+X",
        unit: "meters",
      },
      entities: [mockEntity],
      representations: {
        photo: { status: "ready", updatedAt: new Date().toISOString() },
        spatial_photo: { status: "ready", updatedAt: new Date().toISOString() },
        wireframe: { status: "ready", updatedAt: new Date().toISOString() },
        solid3d: { status: "ready", updatedAt: new Date().toISOString() },
      },
      processingStatus: "ready",
    };

    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useUIStore.getState().selectEntity("person-001");
    });

    render(<ContextualPanel />);

    // Check Kinematic Skeleton section
    expect(screen.getByTestId("entity-skeleton-section")).toBeInTheDocument();
    expect(screen.getByTestId("skeleton-status")).toHaveTextContent("Ready");
    expect(screen.getByTestId("skeleton-root-joint")).toHaveTextContent("pelvis");
    expect(screen.getByTestId("skeleton-source")).toHaveTextContent("SMPL Regressed (3D)");
    expect(screen.getByTestId("toggle-skeleton-btn")).toHaveTextContent("Show Skeleton Overlay");

    // Check Mesh Rigging section
    expect(screen.getByTestId("entity-rigging-section")).toBeInTheDocument();
    expect(screen.getByTestId("rig-status")).toHaveTextContent("rigged");
    expect(screen.getByTestId("rig-method")).toHaveTextContent("SMPL Native LBS (24 bones)");

    // Check Motion & Animation section
    expect(screen.getByTestId("entity-motion-section")).toBeInTheDocument();
    expect(screen.getByTestId("clip-select")).toHaveValue("idle_breathing");
    expect(screen.getByTestId("anim-play-pause-btn")).toHaveTextContent("Play");
    expect(screen.getByTestId("anim-reset-btn")).toHaveTextContent("Reset Pose");
    expect(screen.getByTestId("anim-disclaimer")).toHaveTextContent(
      "Procedural Test Animation (Not recovered motion from photo)"
    );

    // Toggle Skeleton Overlay
    fireEvent.click(screen.getByTestId("toggle-skeleton-btn"));
    expect(useSceneStore.getState().currentScene?.entities[0].showSkeleton).toBe(true);
    expect(screen.getByTestId("toggle-skeleton-btn")).toHaveTextContent("Hide Skeleton Overlay");

    // Click Play button
    fireEvent.click(screen.getByTestId("anim-play-pause-btn"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.isPlaying).toBe(true);
    expect(screen.getByTestId("anim-play-pause-btn")).toHaveTextContent("Pause");

    // Change animation clip
    fireEvent.change(screen.getByTestId("clip-select"), { target: { value: "torso_twist" } });
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.currentClip).toBe("torso_twist");

    // Change playback speed to 1.5x
    fireEvent.click(screen.getByTestId("speed-1.5x"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.playbackSpeed).toBe(1.5);

    // Click Reset Pose button
    fireEvent.click(screen.getByTestId("anim-reset-btn"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.isPlaying).toBe(false);

    // Test Stop button
    fireEvent.click(screen.getByTestId("anim-play-pause-btn"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.isPlaying).toBe(true);
    fireEvent.click(screen.getByTestId("anim-stop-btn"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.isPlaying).toBe(false);

    // Test 2x speed button
    fireEvent.click(screen.getByTestId("speed-2x"));
    expect(useSceneStore.getState().currentScene?.entities[0].animationState?.playbackSpeed).toBe(2.0);
  });

  it("provides posture-aware clips for seated persons (e.g. Person 3 and 5) avoiding standing walk cycles", () => {
    const mockSeatedEntity: SceneEntity = {
      id: "person-003",
      entityType: "person",
      sourceImageId: "photo-001",
      name: "Person 3 (Seated)",
      visibility: true,
      reconstructionStatus: "ready",
      transform: {
        position: { x: 0.2, y: 0.8, z: 2.8 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1.0, y: 1.0, z: 1.0 },
      },
      animationState: {
        currentClip: "seated_idle",
        isPlaying: false,
        playbackSpeed: 1.0,
        loop: true,
      },
      showSkeleton: false,
      reconstruction: {
        status: "ready",
        meshAssetId: "mesh-003",
        meshUrl: "/api/v1/assets/meshes/person_003.glb",
        textureAvailability: "partial",
        stanceDescription: "Seated Posture (Knee Flexion: 92.4°, Hip Flexion: 114.1°)",
        geometryMetadata: {
          vertexCount: 6890,
          faceCount: 13776,
          bounds: { min: [-0.3, -0.6, -0.4], max: [0.3, 0.6, 0.4] },
          dimensions: [0.6, 1.2, 0.8],
        },
      },
      skeleton: {
        id: "skel-003",
        entityId: "person-003",
        schemaVersion: "1.0.0",
        rootJointId: "pelvis",
        joints: [],
        bones: [],
        coordinateConvention: { up: "+Y", forward: "+Z", right: "+X", units: "meters" },
        source: "smpl_regressed",
        isAligned: true,
        validationStatus: "valid",
        validationErrors: [],
        createdAt: new Date().toISOString(),
      },
      rig: {
        id: "rig-003",
        entityId: "person-003",
        skeletonId: "skel-003",
        status: "rigged",
        skinningMethod: "smpl_lbs_native",
        boneCount: 24,
        maxInfluencesPerVertex: 4,
        weightsNormalized: true,
      },
    };

    const mockScene: Scene = {
      id: "scene-seated",
      schemaVersion: "1.0.0",
      name: "Seated Test Scene",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      sourceImage: {
        assetId: "photo-001",
        filename: "test.jpg",
        width: 1000,
        height: 800,
        aspectRatio: 1.25,
        mimeType: "image/jpeg",
        sizeBytes: 120000,
        uploadedAt: new Date().toISOString(),
      },
      coordinateSystem: {
        handedness: "right-handed",
        upAxis: "+Y",
        forwardAxis: "+Z",
        rightAxis: "+X",
        unit: "meters",
      },
      entities: [mockSeatedEntity],
      representations: {
        photo: { status: "ready", updatedAt: new Date().toISOString() },
        combined: { status: "ready", updatedAt: new Date().toISOString() },
      },
      processingStatus: "ready",
    };

    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useUIStore.getState().selectEntity("person-003");
    });

    render(<ContextualPanel />);

    // Expect seated stance indication badge
    expect(screen.getByText("Seated Stance")).toBeInTheDocument();

    // Expect seated clip selector with seated idle as current value
    const select = screen.getByTestId("clip-select") as HTMLSelectElement;
    expect(select.value).toBe("seated_idle");

    // Options must include seated idle and seated weight shift
    const optionValues = Array.from(select.options).map((opt) => opt.value);
    expect(optionValues).toContain("seated_idle");
    expect(optionValues).toContain("seated_weight_shift");
    // Crucial: Seated persons must NOT have a standing walking cycle
    expect(optionValues).not.toContain("gentle_walk");
  });
});
