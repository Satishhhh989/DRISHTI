import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ContextualPanel } from "../components/layout/ContextualPanel";
import { RepresentationSelector } from "../components/controls/RepresentationSelector";
import { ThreeViewport } from "../components/viewport/ThreeViewport";
import { useSceneStore } from "../stores/scene-store";
import { useUIStore } from "../stores/ui-store";
import { Scene, SceneEntity } from "@photo-to-3d/scene-schema";
import { assetCache } from "../lib/asset-cache";

// Mock R3F Three.js Canvas / WebGL context
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-r3f-canvas">{children}</div>
  ),
  useFrame: vi.fn(),
  useThree: () => ({
    camera: {
      position: { set: vi.fn(), x: 0, y: 1.0, z: 3.2 },
    },
  }),
}));

vi.mock("@react-three/drei", () => ({
  OrbitControls: () => <div data-testid="mock-orbit-controls" />,
}));

vi.mock("three/examples/jsm/loaders/GLTFLoader.js", () => {
  return {
    GLTFLoader: vi.fn().mockImplementation(() => ({
      load: (_url: string, onLoad: (gltf: { scene: { clone: () => unknown; traverse: () => void } }) => void) => {
        onLoad({
          scene: {
            clone: () => ({
              traverse: vi.fn(),
            }),
            traverse: vi.fn(),
          },
        });
      },
    })),
  };
});

describe("Phase 08 Unified Hybrid Renderer & Representation System", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useUIStore.getState().selectEntity(null);
  });

  const createMockSceneWithReconstruction = (): Scene => {
    const mockEntity: SceneEntity = {
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-001",
      name: "Person 1",
      visibility: true,
      reconstructionStatus: "ready",
      transform: {
        position: { x: 0.0, y: 1.0, z: 0.2 },
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
        id: "skel-001",
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
        ],
        bones: [],
        coordinateConvention: { up: "+Y", forward: "+Z", right: "+X", units: "meters" },
        source: "smpl_regressed",
        isAligned: true,
        validationStatus: "valid",
        validationErrors: [],
        createdAt: new Date().toISOString(),
      },
      rig: {
        id: "rig-001",
        entityId: "person-001",
        skeletonId: "skel-001",
        status: "rigged",
        skinningMethod: "smpl_lbs_native",
        boneCount: 24,
        maxInfluencesPerVertex: 4,
        weightsNormalized: true,
      },
    };

    return {
      id: "scene-unified-001",
      schemaVersion: "1.0.0",
      name: "Unified Renderer Test Scene",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      sourceImage: {
        assetId: "photo-001",
        filename: "test.jpg",
        width: 1200,
        height: 800,
        aspectRatio: 1.5,
        mimeType: "image/jpeg",
        sizeBytes: 150000,
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
        articulated: { status: "ready", updatedAt: new Date().toISOString() },
        combined: { status: "ready", updatedAt: new Date().toISOString() },
      },
      depth: {
        status: "ready",
        model: "depth-anything-v2-small",
        modelFamily: "depth-anything",
        depthConvention: "relative_proximity",
        width: 1200,
        height: 800,
        device: "mps",
        visualizationUri: "/api/v1/assets/depth/vis_001.png",
      },
      processingStatus: "ready",
    };
  };

  it("renders all 9 representation tabs from registry in RepresentationSelector", () => {
    render(<RepresentationSelector />);
    expect(screen.getByTestId("mode-photo")).toBeInTheDocument();
    expect(screen.getByTestId("mode-detection")).toBeInTheDocument();
    expect(screen.getByTestId("mode-segmentation")).toBeInTheDocument();
    expect(screen.getByTestId("mode-depth")).toBeInTheDocument();
    expect(screen.getByTestId("mode-spatial-photo")).toBeInTheDocument();
    expect(screen.getByTestId("mode-wireframe")).toBeInTheDocument();
    expect(screen.getByTestId("mode-solid3d")).toBeInTheDocument();
    expect(screen.getByTestId("mode-articulated")).toBeInTheDocument();
    expect(screen.getByTestId("mode-combined")).toBeInTheDocument();
  });

  it("switches representations while preserving active entity selection and transform data", () => {
    const scene = createMockSceneWithReconstruction();

    act(() => {
      useSceneStore.getState().setScene(scene);
      useUIStore.getState().selectEntity("person-001");
    });

    render(
      <div>
        <RepresentationSelector />
        <ContextualPanel />
      </div>
    );

    // Initial mode is detection or original
    expect(useUIStore.getState().selectedEntityId).toBe("person-001");

    // Switch to Solid 3D
    fireEvent.click(screen.getByTestId("mode-solid3d"));
    expect(useSceneStore.getState().visualizationMode).toBe("solid3d");
    expect(useUIStore.getState().selectedEntityId).toBe("person-001");
    expect(screen.getByTestId("active-representation-name")).toHaveTextContent("3D Solid Mesh");

    // Switch to Wireframe 3D
    fireEvent.click(screen.getByTestId("mode-wireframe"));
    expect(useSceneStore.getState().visualizationMode).toBe("wireframe");
    expect(useUIStore.getState().selectedEntityId).toBe("person-001");
    expect(screen.getByTestId("active-representation-name")).toHaveTextContent("3D Wireframe");

    // Switch to Articulated
    fireEvent.click(screen.getByTestId("mode-articulated"));
    expect(useSceneStore.getState().visualizationMode).toBe("articulated");
    expect(useUIStore.getState().selectedEntityId).toBe("person-001");
    expect(screen.getByTestId("active-representation-name")).toHaveTextContent("Articulated Character");

    // Switch to Combined Scene
    fireEvent.click(screen.getByTestId("mode-combined"));
    expect(useSceneStore.getState().visualizationMode).toBe("combined");
    expect(useUIStore.getState().selectedEntityId).toBe("person-001");
    expect(screen.getByTestId("active-representation-name")).toHaveTextContent("Combined Scene");
  });

  it("preserves entity visibility settings across representation switches", () => {
    const scene = createMockSceneWithReconstruction();

    act(() => {
      useSceneStore.getState().setScene(scene);
    });

    render(
      <div>
        <RepresentationSelector />
        <ContextualPanel />
      </div>
    );

    // Hide Person 1
    const toggleBtn = screen.getByTestId("toggle-visibility-person-001");
    fireEvent.click(toggleBtn);
    expect(useSceneStore.getState().currentScene?.entities[0].visibility).toBe(false);

    // Switch from solid3d to wireframe and check visibility is still false
    fireEvent.click(screen.getByTestId("mode-wireframe"));
    expect(useSceneStore.getState().currentScene?.entities[0].visibility).toBe(false);

    // Switch to combined and check visibility is still false
    fireEvent.click(screen.getByTestId("mode-combined"));
    expect(useSceneStore.getState().currentScene?.entities[0].visibility).toBe(false);
  });

  it("displays honest placeholders when representations are not yet produced", () => {
    // Unreconstructed scene
    act(() => {
      useSceneStore.getState().setLocalPhoto("blob:test", 800, 600, "test.jpg");
    });

    render(<ThreeViewport />);

    // Switch to Articulated
    act(() => {
      useSceneStore.getState().setVisualizationMode("articulated");
    });
    expect(screen.getByTestId("articulated-placeholder")).toBeInTheDocument();

    // Switch to Combined
    act(() => {
      useSceneStore.getState().setVisualizationMode("combined");
    });
    expect(screen.getByTestId("combined-placeholder")).toBeInTheDocument();
  });

  it("deduplicates asset loads via unified assetCache", async () => {
    const modelUrl = "/api/v1/assets/meshes/cached_test.glb";

    // Load model once
    const model1 = await assetCache.loadModel(modelUrl);
    expect(model1).toBeDefined();
    expect(assetCache.hasModel(modelUrl)).toBe(true);

    // Load model again: must return cloned cached model without throwing or duplicate network calls
    const model2 = await assetCache.loadModel(modelUrl);
    expect(model2).toBeDefined();
    expect(model1).not.toBe(model2); // Distinct cloned Three.Group instances sharing geometry
  });
});

