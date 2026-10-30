import { describe, it, expect, beforeEach } from "vitest";
import { useSceneStore } from "../stores/scene-store";
import { useUIStore } from "../stores/ui-store";

describe("Frontend Stores", () => {
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

  it("updates scene store when loading local photograph", () => {
    const store = useSceneStore.getState();
    expect(store.currentScene).toBeNull();
    expect(store.localPhotoUrl).toBeNull();

    store.setLocalPhoto("blob:test-photo-url", 1920, 1080, "test_group.jpg");

    const updated = useSceneStore.getState();
    expect(updated.localPhotoUrl).toBe("blob:test-photo-url");
    expect(updated.photoDimensions?.width).toBe(1920);
    expect(updated.photoDimensions?.height).toBe(1080);
    expect(updated.photoDimensions?.aspect).toBe(1.778);
    expect(updated.currentScene?.sourceImage.filename).toBe("test_group.jpg");
    expect(updated.activeRepresentation).toBe("photo");
  });

  it("switches visualization and representation modes in scene store", () => {
    const store = useSceneStore.getState();

    store.setVisualizationMode("detection");
    expect(useSceneStore.getState().visualizationMode).toBe("detection");

    store.setVisualizationMode("segmentation");
    expect(useSceneStore.getState().visualizationMode).toBe("segmentation");

    store.setVisualizationMode("original");
    expect(useSceneStore.getState().visualizationMode).toBe("original");

    store.setVisualizationMode("wireframe");
    expect(useSceneStore.getState().activeRepresentation).toBe("wireframe");

    store.setVisualizationMode("solid3d");
    expect(useSceneStore.getState().activeRepresentation).toBe("solid3d");
  });

  it("updates analysis progress state", () => {
    const store = useSceneStore.getState();
    store.setAnalysisState(true, "Detecting people...", 0.4);

    const updated = useSceneStore.getState();
    expect(updated.isAnalyzing).toBe(true);
    expect(updated.currentStageMessage).toBe("Detecting people...");
    expect(updated.analysisProgress).toBe(0.4);
  });

  it("manages UI store toggles and entity selection", () => {
    const ui = useUIStore.getState();
    expect(ui.showGrid).toBe(true);
    expect(ui.isInspectorOpen).toBe(true);

    ui.toggleGrid();
    expect(useUIStore.getState().showGrid).toBe(false);

    ui.toggleInspector();
    expect(useUIStore.getState().isInspectorOpen).toBe(false);

    ui.selectEntity("person-123");
    expect(useUIStore.getState().selectedEntityId).toBe("person-123");

    ui.setError("Failed to parse image file");
    expect(useUIStore.getState().errorMessage).toBe("Failed to parse image file");

    ui.resetCamera();
    expect(useUIStore.getState().cameraResetTrigger).toBe(1);
  });
});
