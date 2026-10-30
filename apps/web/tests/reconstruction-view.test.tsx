import React from "react";
import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RepresentationSelector } from "@/components/controls/RepresentationSelector";
import { ContextualPanel } from "@/components/layout/ContextualPanel";
import { useSceneStore } from "@/stores/scene-store";
import { createInitialScene } from "@photo-to-3d/scene-schema";

describe("Phase 05 AI Human 3D Reconstruction Frontend Integration", () => {
  beforeEach(() => {
    useSceneStore.setState({
      currentScene: null,
      selectedEntityId: null,
      visualizationMode: "original",
      activeRepresentation: "photo",
      isAnalyzing: false,
      currentStageMessage: null,
      stages: {},
      partialFailures: [],
    });
  });

  it("enables wireframe and solid3d modes when an entity has a reconstructed 3D mesh", () => {
    const scene = createInitialScene({
      id: "scene-rec-1",
      name: "Reconstructed Human Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 50000,
        uploadedAt: new Date().toISOString(),
      },
    });

    scene.entities.push({
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-1",
      name: "Person 1",
      transform: {
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
      },
      visibility: true,
      animationState: { currentClip: null, isPlaying: false, playbackSpeed: 1, loop: true },
      reconstructionStatus: "ready",
      detection: {
        confidence: 0.95,
        boundingBox: { x_min: 0.2, y_min: 0.1, x_max: 0.5, y_max: 0.9 } as any,
        pixelBox: [160, 60, 400, 540],
        model: "yolov8n-seg",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      reconstruction: {
        status: "ready",
        meshAssetId: "mesh-001",
        meshUrl: "/api/v1/assets/meshes/scene-rec-1_person-001.glb",
        geometryMetadata: {
          vertexCount: 6890,
          faceCount: 13776,
          bounds: { min: [-0.4, -1.0, -0.2], max: [0.4, 0.7, 0.3] },
          dimensions: [0.8, 1.7, 0.5],
        },
        modelProvenance: {
          name: "SPIN",
          version: "1.0.0",
          device: "mps",
        },
        textureAvailability: "none",
        reconstructedAt: new Date().toISOString(),
      },
    });

    useSceneStore.setState({
      currentScene: scene,
      selectedEntityId: "person-001",
    });

    render(
      <div>
        <RepresentationSelector />
        <ContextualPanel />
      </div>
    );

    // 1. Verify representation buttons are enabled
    const wireframeBtn = screen.getByTestId("mode-wireframe");
    const solid3dBtn = screen.getByTestId("mode-solid3d");

    expect(wireframeBtn).toBeInTheDocument();
    expect(solid3dBtn).toBeInTheDocument();

    // 2. Click wireframe and check state
    fireEvent.click(wireframeBtn);
    expect(useSceneStore.getState().visualizationMode).toBe("wireframe");
    expect(useSceneStore.getState().activeRepresentation).toBe("wireframe");

    // 3. Click solid3d and check state
    fireEvent.click(solid3dBtn);
    expect(useSceneStore.getState().visualizationMode).toBe("solid3d");
    expect(useSceneStore.getState().activeRepresentation).toBe("solid3d");

    // 4. Select the person entity to view details in inspector
    const personCard = screen.getByTestId("person-card-person-001");
    fireEvent.click(personCard);

    // 5. Verify reconstruction metrics in ContextualPanel inspector
    expect(screen.getByTestId("reconstruction-status")).toHaveTextContent("ready");
    expect(screen.getByTestId("reconstruction-vertex-count")).toHaveTextContent("6,890");
    expect(screen.getByTestId("reconstruction-face-count")).toHaveTextContent("13,776");
    expect(screen.getByTestId("stage-row-reconstruction")).toHaveTextContent("1 of 1 meshes");
  });
});
