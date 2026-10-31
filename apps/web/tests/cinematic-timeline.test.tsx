import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { ReconstructionTimeline } from "../components/timeline/ReconstructionTimeline";
import { useCinematicStore, CINEMATIC_STAGES } from "../stores/cinematic-store";
import { useSceneStore } from "../stores/scene-store";
import { Scene } from "@photo-to-3d/scene-schema";

describe("Cinematic Reconstruction Timeline Experience", () => {
  const mockScene: Scene = {
    id: "scene-cinematic-001",
    schemaVersion: "1.0.0",
    name: "Cinematic Test Scene",
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
    entities: [
      {
        id: "person-001",
        entityType: "person",
        sourceImageId: "photo-001",
        name: "Person 1",
        visibility: true,
        reconstructionStatus: "ready",
        boundingBox: { xMin: 0.1, yMin: 0.2, xMax: 0.4, yMax: 0.8 },
      },
    ],
    representations: {
      photo: { status: "ready", updatedAt: new Date().toISOString() },
    },
    processingStatus: "ready",
  };

  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useCinematicStore.getState().resetCinematic();
  });

  it("renders all 9 canonical cinematic reconstruction stages", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
    });

    render(<ReconstructionTimeline />);

    expect(screen.getByTestId("reconstruction-timeline")).toBeInTheDocument();

    // Verify all 9 stages are represented in the timeline
    expect(screen.getByTestId("timeline-stage-1")).toHaveTextContent("Original Photo");
    expect(screen.getByTestId("timeline-stage-2")).toHaveTextContent("Person Detection");
    expect(screen.getByTestId("timeline-stage-3")).toHaveTextContent("Segmentation");
    expect(screen.getByTestId("timeline-stage-4")).toHaveTextContent("Depth Map");
    expect(screen.getByTestId("timeline-stage-5")).toHaveTextContent("Spatial Photo");
    expect(screen.getByTestId("timeline-stage-6")).toHaveTextContent("Wireframe Extraction");
    expect(screen.getByTestId("timeline-stage-7")).toHaveTextContent("3D Solid Mesh");
    expect(screen.getByTestId("timeline-stage-8")).toHaveTextContent("Articulation & Rigging");
    expect(screen.getByTestId("timeline-stage-9")).toHaveTextContent("Combined Scene");

    // Stage 1 initial active pill
    expect(screen.getByText("Stage 1 / 9")).toBeInTheDocument();
    expect(screen.getByText("Ingestion & Validation")).toBeInTheDocument();
  });

  it("provides pause and resume controls for cinematic playback", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
    });

    render(<ReconstructionTimeline />);

    const pauseBtn = screen.getByTestId("timeline-pause-btn");
    expect(pauseBtn).toHaveTextContent("❚❚ Pause");

    // Pause
    fireEvent.click(pauseBtn);
    expect(useCinematicStore.getState().isPaused).toBe(true);
    expect(screen.getByTestId("timeline-pause-btn")).toHaveTextContent("▶ Resume");

    // Resume
    fireEvent.click(screen.getByTestId("timeline-pause-btn"));
    expect(useCinematicStore.getState().isPaused).toBe(false);
  });

  it("allows skipping to final combined 3D scene directly", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
    });

    render(<ReconstructionTimeline />);

    const skipBtn = screen.getByTestId("timeline-skip-btn");
    fireEvent.click(skipBtn);

    expect(useCinematicStore.getState().isCinematicActive).toBe(false);
    expect(useCinematicStore.getState().currentStageId).toBe("stage_combined");
    expect(useSceneStore.getState().visualizationMode).toBe("combined");
  });

  it("allows clicking completed stages to inspect intermediate results without re-running pipeline", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
      // Mark stages 1, 2, 3 as completed
      useCinematicStore.getState().updateStageStatus("stage_photo", "completed");
      useCinematicStore.getState().updateStageStatus("stage_detection", "completed");
      useCinematicStore.getState().updateStageStatus("stage_segmentation", "completed");
      useCinematicStore.getState().setCurrentStage("stage_depth");
    });

    render(<ReconstructionTimeline />);

    // Click on Stage 2 (Person Detection)
    const stage2Btn = screen.getByTestId("timeline-stage-2");
    fireEvent.click(stage2Btn);

    expect(useSceneStore.getState().visualizationMode).toBe("detection");
    expect(useCinematicStore.getState().currentStageId).toBe("stage_detection");

    // Click on Stage 3 (Segmentation)
    const stage3Btn = screen.getByTestId("timeline-stage-3");
    fireEvent.click(stage3Btn);

    expect(useSceneStore.getState().visualizationMode).toBe("segmentation");
    expect(useCinematicStore.getState().currentStageId).toBe("stage_segmentation");
  });
});

