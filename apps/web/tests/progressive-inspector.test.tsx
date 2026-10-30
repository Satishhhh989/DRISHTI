import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import WorkspacePage from "../app/page";
import { useSceneStore } from "../stores/scene-store";
import { Scene, createInitialScene } from "@photo-to-3d/scene-schema";

describe("Phase 04 Progressive Pipeline Inspector", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok", projects: [] }),
    });
  });

  it("renders the pipeline inspector and reflects idle state", async () => {
    await act(async () => {
      render(<WorkspacePage />);
    });

    expect(screen.getByTestId("pipeline-inspector")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-validation")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-detection")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-segmentation")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-depth")).toBeInTheDocument();
    expect(screen.getByTestId("stage-row-spatial")).toBeInTheDocument();
  });

  it("displays running state and provides cancel analysis button", async () => {
    const scene: Scene = createInitialScene({
      id: "scene-cancel-test",
      name: "Cancel Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "group.jpg",
        width: 1000,
        height: 750,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 8000,
        uploadedAt: new Date().toISOString(),
      },
    });

    await act(async () => {
      useSceneStore.getState().setScene(scene);
      useSceneStore.getState().setAnalysisState(true, "Detecting people in image...", 0.35, "job-cancel-test");
      render(<WorkspacePage />);
    });

    // Inspector should show "Running" badge and Cancel button
    expect(screen.getByText("Running")).toBeInTheDocument();
    const cancelBtn = screen.getByTestId("cancel-job-btn");
    expect(cancelBtn).toBeInTheDocument();
    expect(cancelBtn).toHaveTextContent("Cancel Analysis");

    // Click Cancel
    await act(async () => {
      fireEvent.click(cancelBtn);
    });

    // Verify cancellation updates state
    expect(useSceneStore.getState().isAnalyzing).toBe(false);
  });

  it("displays partial failures without hiding successful results", async () => {
    const scene: Scene = createInitialScene({
      id: "scene-partial-test",
      name: "Partial Test",
      sourceImage: {
        assetId: "photo-1",
        filename: "group.jpg",
        width: 1000,
        height: 750,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 8000,
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
      reconstructionStatus: "unstarted",
      detection: {
        confidence: 0.91,
        boundingBox: { xMin: 0.1, yMin: 0.1, xMax: 0.4, yMax: 0.8 },
        pixelBox: [100, 75, 400, 600],
        model: "mock-yolo",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      segmentation: {
        status: "ready",
        maskAssetId: "mask-1",
        maskUri: "/api/v1/assets/mask-1/file",
      },
    });

    await act(async () => {
      useSceneStore.getState().setScene(scene);
      useSceneStore.setState({
        partialFailures: [{ stage: "depth_estimation", message: "Depth model weights unavailable" }],
      });
      render(<WorkspacePage />);
    });

    // Person 1 should still be present in the inspector
    expect(screen.getByTestId("person-card-person-001")).toBeInTheDocument();
    // Partial failure warning should be visible
    expect(screen.getByText(/Depth model weights unavailable/)).toBeInTheDocument();
  });
});

