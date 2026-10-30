import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach } from "vitest";
import WorkspacePage from "../app/page";
import { useSceneStore } from "../stores/scene-store";
import { useUIStore } from "../stores/ui-store";
import { createInitialScene } from "@photo-to-3d/scene-schema";

describe("Phase 03 Depth Estimation & Spatial Photography UI", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useUIStore.setState({
      selectedEntityId: null,
      isInspectorOpen: true,
      showGrid: true,
      cameraResetTrigger: 0,
      errorMessage: null,
      statusNotice: null,
    });
  });

  it("renders spatial photo controls and handles depth strength changes", async () => {
    // Populate scene with depth metadata
    const scene = createInitialScene({
      id: "test-spatial-scene",
      sourceImage: {
        assetId: "test-photo-asset",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 10000,
        uploadedAt: new Date().toISOString(),
      },
    });

    scene.depth = {
      status: "ready",
      depthAssetId: "depth-data-1",
      depthUri: "/api/v1/assets/depth-data-1/file",
      visualizationAssetId: "depth-vis-1",
      visualizationUri: "/api/v1/assets/depth-vis-1/file",
      model: "depth-anything-v2-small",
      modelFamily: "Depth-Anything-V2",
      depthConvention: "relative_proximity",
      width: 800,
      height: 600,
      device: "mps",
    };

    scene.representations.spatial_photo = {
      status: "ready",
      assetId: "depth-data-1",
      uri: "/api/v1/assets/depth-vis-1/file",
      updatedAt: new Date().toISOString(),
    };

    await act(async () => {
      useSceneStore.getState().setScene(scene);
      render(<WorkspacePage />);
    });

    // Check depth controls rendered in inspector
    expect(screen.getByTestId("spatial-photo-controls")).toBeInTheDocument();
    expect(screen.getByText("depth-anything-v2-small")).toBeInTheDocument();
    expect(screen.getByText("Relative Proximity")).toBeInTheDocument();
    expect(screen.getByText("MPS")).toBeInTheDocument();

    // Adjust depth strength slider
    const slider = screen.getByTestId("depth-strength-slider");
    expect(slider).toBeInTheDocument();

    await act(async () => {
      fireEvent.change(slider, { target: { value: "0.14" } });
    });
    expect(useSceneStore.getState().depthStrength).toBe(0.14);

    // Reset camera button
    const resetBtn = screen.getByTestId("reset-camera-spatial-btn");
    await act(async () => {
      fireEvent.click(resetBtn);
    });
    expect(useUIStore.getState().cameraResetTrigger).toBe(1);
  });

  it("displays per-person depth statistics in inspector when entity is selected", async () => {
    const scene = createInitialScene({
      id: "test-entity-depth",
      sourceImage: {
        assetId: "test-photo-asset",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 10000,
        uploadedAt: new Date().toISOString(),
      },
    });

    scene.entities = [
      {
        id: "person-001",
        entityType: "person",
        sourceImageId: "test-photo-asset",
        name: "Person 1",
        transform: {
          position: { x: 0, y: 0, z: 0 },
          rotation: { x: 0, y: 0, z: 0, order: "XYZ" },
          scale: { x: 1, y: 1, z: 1 },
        },
        animationState: { currentClip: null, isPlaying: false, playbackSpeed: 1, loop: true },
        visibility: true,
        reconstructionStatus: "unstarted",
        detection: {
          confidence: 0.95,
          boundingBox: { xMin: 0.1, yMin: 0.1, xMax: 0.4, yMax: 0.8 },
          pixelBox: [80, 60, 320, 480],
          model: "yolov8n-seg",
          detectedAt: new Date().toISOString(),
          status: "ready",
        },
        segmentation: {
          status: "ready",
          maskAssetId: "mask-1",
          maskUri: "/api/v1/assets/mask-1/file",
          pixelCount: 25000,
        },
        depth: {
          status: "ready",
          medianProximity: 0.765,
          minProximity: 0.65,
          maxProximity: 0.85,
          depthVariation: 0.05,
          validPixelCoverage: 0.085,
          method: "segmentation_mask",
        },
      },
    ];

    await act(async () => {
      useSceneStore.getState().setScene(scene);
      render(<WorkspacePage />);
    });

    // Select Person 1
    const personCard = screen.getByTestId("person-card-person-001");
    await act(async () => {
      fireEvent.click(personCard);
    });

    // Verify depth stats rendered in inspector
    expect(screen.getByText("76.5%")).toBeInTheDocument(); // Proximity
    expect(screen.getByText("Depth Method")).toBeInTheDocument();
    expect(screen.getAllByText("Mask").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("8.5%")).toBeInTheDocument(); // Pixel Coverage
  });
});
