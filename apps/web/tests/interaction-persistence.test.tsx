import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import WorkspacePage from "../app/page";
import { useSceneStore } from "../stores/scene-store";
import { useProjectStore } from "../stores/project-store";
import { useUIStore } from "../stores/ui-store";
import { Scene, createInitialScene, Transform } from "@photo-to-3d/scene-schema";

describe("Phase 09: Interaction, Persistence & Export UI", () => {
  const testScene: Scene = createInitialScene({
    id: "scene-interaction-p9",
    name: "Interaction Test Scene",
    sourceImage: {
      assetId: "photo-p9",
      filename: "family.jpg",
      width: 1200,
      height: 900,
      aspectRatio: 1.333333,
      mimeType: "image/jpeg",
      sizeBytes: 1500000,
      uploadedAt: new Date().toISOString(),
    },
    photoUri: "/test-family.jpg",
  });

  // Add 2 entities with placement and one with reconstructed mesh
  testScene.entities = [
    {
      id: "person-0",
      entityType: "person",
      sourceImageId: "photo-p9",
      name: "Person Zero",
      visibility: true,
      reconstructionStatus: "ready",
      transform: {
        position: { x: -0.4, y: 0.0, z: -1.5 },
        rotation: { x: 0.0, y: 0.2, z: 0.0 },
        scale: { x: 1.0, y: 1.0, z: 1.0 },
      },
      placement: {
        method: "Depth-Projected Composition",
        imageSpaceCoords: [0.3, 0.5],
        relativeProximity: 0.8,
        orientationCorrection: [0.0, 0.2, 0.0],
      },
      reconstruction: {
        status: "ready",
        meshAssetId: "mesh-asset-p0",
        meshUrl: "/api/v1/assets/mesh-asset-p0/file",
        geometryMetadata: {
          vertexCount: 6890,
          faceCount: 13776,
          bounds: { min: [-0.3, -0.9, -0.2], max: [0.3, 0.9, 0.2] },
          dimensions: [0.6, 1.8, 0.4],
        },
      },
    },
    {
      id: "person-1",
      entityType: "person",
      sourceImageId: "photo-p9",
      name: "Person One",
      visibility: true,
      reconstructionStatus: "unstarted",
      transform: {
        position: { x: 0.5, y: 0.0, z: -2.0 },
        rotation: { x: 0.0, y: -0.1, z: 0.0 },
        scale: { x: 1.0, y: 1.0, z: 1.0 },
      },
      placement: {
        method: "Depth-Projected Composition",
        imageSpaceCoords: [0.75, 0.5],
        relativeProximity: 0.5,
        orientationCorrection: [0.0, -0.1, 0.0],
      },
    },
  ];

  beforeEach(() => {
    useSceneStore.getState().clearScene();
    useProjectStore.getState().clearProject();
    useUIStore.getState().selectEntity(null);

    // Mock window.open
    vi.stubGlobal("open", vi.fn());

    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: "ok",
        id: "proj-mock-123",
        name: "Mock Project",
        schema_version: "1.0.0",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        scene_id: "scene-interaction-p9",
        source_image: testScene.sourceImage,
        settings: { active_representation: "solid3d" },
        asset_manifest: [],
        processing_state: {},
        projects: [],
      }),
    });
  });

  it("renders project management controls and tracks dirty state on renaming", async () => {
    await act(async () => {
      useSceneStore.getState().setScene(testScene);
      render(<WorkspacePage />);
    });

    const nameInput = screen.getByTestId("project-name-input");
    expect(nameInput).toBeInTheDocument();

    const saveBtn = screen.getByTestId("save-project-btn");
    expect(saveBtn).toBeInTheDocument();

    // Initially clean
    expect(screen.queryByTestId("unsaved-changes-indicator")).not.toBeInTheDocument();

    // Rename project
    await act(async () => {
      fireEvent.change(nameInput, { target: { value: "My Edited 3D Scene" } });
    });

    // Unsaved changes indicator appears
    expect(screen.getByTestId("unsaved-changes-indicator")).toBeInTheDocument();
    expect(saveBtn).toHaveTextContent("Save *");

    // Save project
    await act(async () => {
      fireEvent.click(saveBtn);
    });

    // Clean state restored
    expect(screen.queryByTestId("unsaved-changes-indicator")).not.toBeInTheDocument();
    expect(saveBtn).toHaveTextContent("Saved");
  });

  it("allows selecting person and adjusting Position X, Y, Z via step buttons", async () => {
    await act(async () => {
      useSceneStore.getState().setScene(testScene);
      render(<WorkspacePage />);
    });

    // Select Person Zero
    const personCard = screen.getByTestId("person-card-person-0");
    await act(async () => {
      fireEvent.click(personCard);
    });

    expect(screen.getByTestId("transform-controls-section")).toBeInTheDocument();

    const inputPosX = screen.getByTestId("input-pos-x");
    expect(inputPosX).toHaveValue(-0.4);

    // Step X +0.1m
    const stepXInc = screen.getByTestId("step-x-inc");
    await act(async () => {
      fireEvent.click(stepXInc);
    });

    // Store updated to -0.3m
    const entityAfter = useSceneStore.getState().currentScene?.entities.find((e) => e.id === "person-0");
    expect(entityAfter?.transform.position.x).toBeCloseTo(-0.3);

    // Project marked dirty
    expect(useProjectStore.getState().isDirty).toBe(true);
  });

  it("resets transform to initial assembled placement when Reset Transform clicked", async () => {
    await act(async () => {
      useSceneStore.getState().setScene(testScene);
      useUIStore.getState().selectEntity("person-0");
      render(<WorkspacePage />);
    });

    // Move person significantly
    await act(async () => {
      const stepXInc = screen.getByTestId("step-x-inc");
      fireEvent.click(stepXInc);
      fireEvent.click(stepXInc);
      fireEvent.click(stepXInc);
    });

    // Click Reset Transform
    const resetBtn = screen.getByTestId("reset-transform-btn");
    await act(async () => {
      fireEvent.click(resetBtn);
    });

    // Restored to initial placement coordinates (-0.4)
    const entityReset = useSceneStore.getState().currentScene?.entities.find((e) => e.id === "person-0");
    expect(entityReset?.transform.position.x).toBeCloseTo(-0.4);
    expect(entityReset?.transform.scale.x).toBe(1.0);
  });

  it("exports single person mesh when reconstructed, and disables export when unreconstructed", async () => {
    await act(async () => {
      useSceneStore.getState().setScene(testScene);
      useUIStore.getState().selectEntity("person-0");
      render(<WorkspacePage />);
    });

    // Person 0 has reconstructed mesh -> Export Mesh button is enabled
    const exportMeshBtn = screen.getByTestId("export-entity-mesh-btn");
    expect(exportMeshBtn).not.toBeDisabled();

    await act(async () => {
      fireEvent.click(exportMeshBtn);
    });

    expect(window.open).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/export/mesh/scene-interaction-p9/person-0"),
      "_blank"
    );

    // Select Person 1 (unreconstructed)
    await act(async () => {
      useUIStore.getState().selectEntity("person-1");
    });

    const exportMeshBtnP1 = screen.getByTestId("export-entity-mesh-btn");
    expect(exportMeshBtnP1).toBeDisabled();
  });

  it("provides export dropdown with Assembled Scene, Metadata, and Portable Archive options", async () => {
    await act(async () => {
      useSceneStore.getState().setScene(testScene);
      render(<WorkspacePage />);
    });

    const exportMenuBtn = screen.getByTestId("export-menu-btn");
    await act(async () => {
      fireEvent.click(exportMenuBtn);
    });

    expect(screen.getByTestId("export-assembled-glb-btn")).toBeInTheDocument();
    expect(screen.getByTestId("export-metadata-json-btn")).toBeInTheDocument();
    expect(screen.getByTestId("export-portable-archive-btn")).toBeInTheDocument();

    // Click Assembled Scene export
    await act(async () => {
      fireEvent.click(screen.getByTestId("export-assembled-glb-btn"));
    });

    expect(window.open).toHaveBeenCalledWith(
      expect.stringContaining("/api/v1/export/scene/scene-interaction-p9"),
      "_blank"
    );
  });
});
