import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { TopBar } from "../components/layout/TopBar";
import { ReconstructionTimeline } from "../components/timeline/ReconstructionTimeline";
import { useCinematicStore, CINEMATIC_STAGES } from "../stores/cinematic-store";
import { useSceneStore } from "../stores/scene-store";
import { Scene } from "@photo-to-3d/scene-schema";

describe("Cinematic Reconstruction Animation Redesign", () => {
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
        reconstruction: {
          status: "ready",
          meshUrl: "/api/v1/assets/person-001/mesh.glb",
        },
      },
      {
        id: "person-002",
        entityType: "person",
        sourceImageId: "photo-001",
        name: "Person 2",
        visibility: true,
        reconstructionStatus: "ready",
        boundingBox: { xMin: 0.5, yMin: 0.2, xMax: 0.8, yMax: 0.8 },
        reconstruction: {
          status: "ready",
          meshUrl: "/api/v1/assets/person-002/mesh.glb",
        },
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

  it("ensures the bottom timeline is completely absent and renders nothing in the viewport", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
    });

    const { container } = render(<ReconstructionTimeline />);

    // Bottom timeline must not render any DOM elements
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reconstruction-timeline")).not.toBeInTheDocument();
  });

  it("does not start cinematic reconstruction automatically upon loading a scene or image", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useSceneStore.getState().setLocalPhoto("blob:test", 1200, 800, "test.jpg");
    });

    expect(useCinematicStore.getState().isCinematicActive).toBe(false);
    expect(useCinematicStore.getState().currentStageId).toBe("stage_photo");
  });

  it("provides an elegant 'Reconstruct Scene' button in TopBar that starts the cinematic experience", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useSceneStore.getState().setLocalPhoto("blob:test", 1200, 800, "test.jpg");
    });

    render(<TopBar />);

    const reconstructBtn = screen.getByTestId("reconstruct-scene-btn");
    expect(reconstructBtn).toBeInTheDocument();
    expect(reconstructBtn).not.toBeDisabled();

    // User explicitly initiates the reconstruction presentation
    fireEvent.click(reconstructBtn);

    expect(useCinematicStore.getState().isCinematicActive).toBe(true);
    expect(useCinematicStore.getState().currentStageId).toBe("stage_photo");
  });

  it("supports individual person extraction and topology reveal in cinematic store", () => {
    act(() => {
      useSceneStore.getState().setScene(mockScene);
      useCinematicStore.getState().startCinematic();
      useCinematicStore.getState().setCurrentStage("stage_wireframe");
      useCinematicStore.getState().setActivePersonIndex(0, 2);
      useCinematicStore.getState().setPersonExtractionProgress(0.5);
    });

    const state = useCinematicStore.getState();
    expect(state.currentStageId).toBe("stage_wireframe");
    expect(state.activePersonIndex).toBe(0);
    expect(state.personExtractionProgress).toBe(0.5);
    expect(state.totalPersons).toBe(2);

    // Transition to person 1
    act(() => {
      useCinematicStore.getState().setActivePersonIndex(1, 2);
      useCinematicStore.getState().setPersonExtractionProgress(0.8);
    });

    const nextState = useCinematicStore.getState();
    expect(nextState.activePersonIndex).toBe(1);
    expect(nextState.personExtractionProgress).toBe(0.8);
  });
});
