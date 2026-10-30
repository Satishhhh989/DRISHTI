import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ContextualPanel } from "../components/layout/ContextualPanel";
import { HumanMeshLayer } from "../components/scene/HumanMeshLayer";
import { useSceneStore } from "../stores/scene-store";
import { useUIStore } from "../stores/ui-store";
import { Scene } from "@photo-to-3d/scene-schema";

// Mock R3F Three.js Canvas / WebGL context
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="mock-r3f-canvas">{children}</div>,
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

describe("Phase 06 Multi-Person Scene Assembly View & Controls", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useUIStore.getState().selectEntity(null);
  });

  it("renders pipeline inspector with Scene Assembly stage row", () => {
    render(<ContextualPanel />);
    expect(screen.getByTestId("stage-row-assembly")).toBeInTheDocument();
    expect(screen.getByText("Scene Assembly")).toBeInTheDocument();
  });

  it("displays multiple people, toggles visibility, and inspects 3D placement", async () => {
    const mockScene: Scene = {
      id: "scene-multi-001",
      schemaVersion: "1.0.0",
      name: "Group Photo Scene",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      sourceImage: {
        assetId: "photo-001",
        filename: "group.jpg",
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
      entities: [
        {
          id: "person-001",
          entityType: "person",
          sourceImageId: "photo-001",
          name: "Person 1",
          visibility: true,
          reconstructionStatus: "ready",
          bounding_box: { xMin: 0.1, yMin: 0.2, xMax: 0.4, yMax: 0.8 },
          transform: {
            position: { x: -0.75, y: 1.0, z: 0.4 },
            rotation: { x: 0, y: 0, z: 0 },
            scale: { x: 1.0, y: 1.0, z: 1.0 },
          },
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
          placement: {
            method: "depth_projected_composition",
            imageSpaceCoords: [0.25, 0.5],
            depthSource: "relative_median_proximity",
            relativeProximity: 0.75,
            modelNormalizationApplied: "canonical_humanoid_scale",
            orientationCorrection: [0, 0, 0],
            uncertaintyFlags: [],
            isMetric: false,
          },
        },
        {
          id: "person-002",
          entityType: "person",
          sourceImageId: "photo-001",
          name: "Person 2",
          visibility: true,
          reconstructionStatus: "unstarted",
          bounding_box: { xMin: 0.6, yMin: 0.2, xMax: 0.9, yMax: 0.8 },
          transform: {
            position: { x: 0.75, y: 1.0, z: -0.2 },
            rotation: { x: 0, y: 0, z: 0 },
            scale: { x: 1.0, y: 1.0, z: 1.0 },
          },
          placement: {
            method: "depth_projected_composition",
            imageSpaceCoords: [0.75, 0.5],
            depthSource: "relative_median_proximity",
            relativeProximity: 0.38,
            modelNormalizationApplied: "canonical_humanoid_scale",
            orientationCorrection: [0, 0, 0],
            uncertaintyFlags: ["mesh_not_reconstructed"],
            isMetric: false,
          },
        },
      ],
      representations: {
        photo: { status: "ready", updatedAt: new Date().toISOString() },
        spatial_photo: { status: "ready", updatedAt: new Date().toISOString() },
        wireframe: { status: "ready", updatedAt: new Date().toISOString() },
        solid3d: { status: "ready", updatedAt: new Date().toISOString() },
      },
      assemblyReport: {
        status: "partial",
        assembledAt: new Date().toISOString(),
        totalEntities: 2,
        reconstructedCount: 1,
        unreconstructedCount: 1,
        occlusionWarnings: [],
        placementMethod: "depth_projected_composition",
      },
      processingStatus: "ready",
    };

    useSceneStore.getState().setScene(mockScene);

    render(<ContextualPanel />);

    // Check entity list
    expect(screen.getByText("Person 1")).toBeInTheDocument();
    expect(screen.getByText("Person 2")).toBeInTheDocument();
    expect(screen.getByText("Mesh")).toBeInTheDocument();
    expect(screen.getByText("In-Scene")).toBeInTheDocument();

    // Toggle visibility of Person 1
    const visBtn = screen.getByTestId("toggle-visibility-person-001");
    fireEvent.click(visBtn);

    const updatedScene = useSceneStore.getState().currentScene;
    expect(updatedScene?.entities[0].visibility).toBe(false);

    // Click Person 1 card to select and view placement
    const person1Card = screen.getByTestId("person-card-person-001");
    fireEvent.click(person1Card);

    // Entity Inspector details
    expect(screen.getByTestId("placement-method")).toHaveTextContent("depth_projected_composition");
    expect(screen.getByTestId("placement-position")).toHaveTextContent("X: -0.75m");
    expect(screen.getByTestId("entity-visibility-status")).toHaveTextContent("Hidden");
    expect(screen.getByTestId("focus-entity-btn")).toBeInTheDocument();
  });
});

