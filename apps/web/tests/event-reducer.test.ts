import { describe, it, expect } from "vitest";
import {
  reduceProcessingEvent,
  ReducerState,
} from "../stores/event-reducer";
import { normalizeProcessingEvent } from "../lib/scene-normalizer";
import { Scene, createInitialScene } from "@photo-to-3d/scene-schema";
import { ProcessingEvent } from "@photo-to-3d/shared-types";

function createMockState(): ReducerState {
  const scene: Scene = createInitialScene({
    id: "scene-test-1",
    name: "Reducer Test",
    sourceImage: {
      assetId: "photo-asset-1",
      filename: "test.jpg",
      width: 800,
      height: 600,
      aspectRatio: 1.333,
      mimeType: "image/jpeg",
      sizeBytes: 5000,
      uploadedAt: new Date().toISOString(),
    },
  });

  // Add 2 mock entities
  scene.entities = [
    {
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-asset-1",
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
        confidence: 0.9,
        boundingBox: { xMin: 0.1, yMin: 0.1, xMax: 0.4, yMax: 0.8 },
        pixelBox: [80, 60, 320, 480],
        model: "mock-yolo",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      segmentation: { status: "processing" },
    },
    {
      id: "person-002",
      entityType: "person",
      sourceImageId: "photo-asset-1",
      name: "Person 2",
      transform: {
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
      },
      visibility: true,
      animationState: { currentClip: null, isPlaying: false, playbackSpeed: 1, loop: true },
      reconstructionStatus: "unstarted",
      detection: {
        confidence: 0.85,
        boundingBox: { xMin: 0.5, yMin: 0.1, xMax: 0.9, yMax: 0.8 },
        pixelBox: [400, 60, 720, 480],
        model: "mock-yolo",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      segmentation: { status: "processing" },
    },
  ];

  return {
    currentScene: scene,
    activeJobId: "job-1",
    lastSequenceNumber: 0,
    isAnalyzing: true,
    analysisProgress: 0.2,
    currentStageMessage: "Analyzing...",
    depthVisualizationUrl: null,
    visualizationMode: "detection",
    stages: {},
    partialFailures: [],
  };
}

describe("reduceProcessingEvent", () => {
  it("rejects events belonging to an unrelated scene", () => {
    const state = createMockState();
    const event: ProcessingEvent = {
      eventId: "ev-1",
      jobId: "job-1",
      sceneId: "different-scene-999",
      eventType: "progress_updated",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "person_detection",
      status: "running",
      progress: 0.5,
      message: "Different scene progress",
    };

    const nextState = reduceProcessingEvent(state, event);
    expect(nextState).toEqual(state);
    expect(nextState.lastSequenceNumber).toBe(0);
  });

  it("handles duplicate sequence numbers idempotently", () => {
    const state = createMockState();
    state.lastSequenceNumber = 5;

    const duplicateEvent: ProcessingEvent = {
      eventId: "ev-dup",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "progress_updated",
      timestamp: new Date().toISOString(),
      sequenceNumber: 5,
      stageIdentifier: "person_detection",
      status: "running",
      progress: 0.8,
      message: "Old duplicate message",
    };

    const nextState = reduceProcessingEvent(state, duplicateEvent);
    expect(nextState).toBe(state);
    expect(nextState.currentStageMessage).toBe("Analyzing...");
  });

  it("attaches individual mask to target entity on asset_ready without disturbing others", () => {
    const state = createMockState();
    const maskEvent: ProcessingEvent = {
      eventId: "ev-mask-1",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "asset_ready",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "asset_persistence",
      status: "running",
      entityId: "person-001",
      assetReference: {
        assetId: "mask-asset-1",
        assetType: "segmentation_mask",
        uri: "/api/v1/assets/mask-asset-1/file",
        mimeType: "image/png",
      },
      message: "Segmentation mask ready for Person 1 (1 of 2).",
    };

    const nextState = reduceProcessingEvent(state, maskEvent);
    expect(nextState.lastSequenceNumber).toBe(1);

    const p1 = nextState.currentScene!.entities.find((e) => e.id === "person-001");
    const p2 = nextState.currentScene!.entities.find((e) => e.id === "person-002");

    // Person 1 mask is ready
    expect(p1!.segmentation?.status).toBe("ready");
    expect(p1!.segmentation?.maskUri).toBe("/api/v1/assets/mask-asset-1/file");

    // Person 2 mask remains processing
    expect(p2!.segmentation?.status).toBe("processing");
    expect(p2!.segmentation?.maskUri).toBeUndefined();
  });

  it("updates depth visualization and spatial photo availability on depth assets", () => {
    const state = createMockState();
    const depthEvent: ProcessingEvent = {
      eventId: "ev-depth-vis",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "asset_ready",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "depth_normalization",
      status: "running",
      assetReference: {
        assetId: "vis-asset-1",
        assetType: "depth_visualization",
        uri: "/api/v1/assets/vis-asset-1/file",
        mimeType: "image/png",
      },
      message: "Depth visualization asset saved.",
    };

    const nextState = reduceProcessingEvent(state, depthEvent);
    expect(nextState.depthVisualizationUrl).toBe("/api/v1/assets/vis-asset-1/file");
    expect(nextState.currentScene?.depth?.visualizationUri).toBe(
      "/api/v1/assets/vis-asset-1/file"
    );
  });

  it("isolates non-fatal stage failures and records partial failures", () => {
    const state = createMockState();
    const failEvent: ProcessingEvent = {
      eventId: "ev-fail-depth",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "stage_failed",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "depth_estimation",
      status: "running", // Job still running (non-fatal)
      error: {
        code: "DEPTH_FAILED",
        message: "Depth model weights missing",
      },
    };

    const nextState = reduceProcessingEvent(state, failEvent);
    expect(nextState.partialFailures).toHaveLength(1);
    expect(nextState.partialFailures[0].stage).toBe("depth_estimation");
    expect(nextState.currentScene?.depth?.status).toBe("failed");
    expect(nextState.currentScene?.representations.spatial_photo.status).toBe("failed");
    // Entities and original photograph remain safe
    expect(nextState.currentScene?.entities).toHaveLength(2);
  });

  it("handles job cancellation cleanly", () => {
    const state = createMockState();
    const cancelEvent: ProcessingEvent = {
      eventId: "ev-cancel",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "job_cancelled",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "instance_segmentation",
      status: "cancelled",
      message: "Job cancelled by user",
    };

    const nextState = reduceProcessingEvent(state, cancelEvent);
    expect(nextState.isAnalyzing).toBe(false);
    expect(nextState.activeJobId).toBeNull();
    expect(nextState.stages.instance_segmentation.status).toBe("cancelled");
    expect(nextState.currentStageMessage).toBe("Processing cancelled by user.");
  });

  it("handles job completion cleanly", () => {
    const state = createMockState();
    const completeEvent: ProcessingEvent = {
      eventId: "ev-complete",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "job_completed",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "completed",
      status: "completed",
      progress: 1.0,
      message: "Analysis complete: 2 people ready.",
    };

    const nextState = reduceProcessingEvent(state, completeEvent);
    expect(nextState.isAnalyzing).toBe(false);
    expect(nextState.analysisProgress).toBe(1.0);
    expect(nextState.currentScene?.processingStatus).toBe("ready");
  });

  it("handles 3D human mesh asset_ready events and transitions representations", () => {
    const state = createMockState();
    const meshEvent: ProcessingEvent = {
      eventId: "ev-mesh-1",
      jobId: "job-1",
      sceneId: "scene-test-1",
      eventType: "asset_ready",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "human_reconstruction",
      status: "running",
      entityId: "person-001",
      assetReference: {
        assetId: "mesh-asset-1",
        assetType: "mesh",
        uri: "/api/v1/assets/meshes/scene-test-1_person-001.glb",
        mimeType: "model/gltf-binary",
        sizeBytes: 331940,
      },
    };

    const nextState = reduceProcessingEvent(state, meshEvent);
    const p1 = nextState.currentScene?.entities.find((e) => e.id === "person-001");
    expect(p1?.reconstruction?.status).toBe("ready");
    expect(p1?.reconstruction?.meshUrl).toBe("/api/v1/assets/meshes/scene-test-1_person-001.glb");
    expect(p1?.geometryRef).toBe("mesh-asset-1");
    expect(nextState.currentScene?.representations.solid3d.status).toBe("ready");
    expect(nextState.currentScene?.representations.wireframe.status).toBe("ready");
  });

  it("normalizes backend snake_case events to camelCase cleanly", () => {
    const rawBackendEvent = {
      event_id: "raw-ev-1",
      job_id: "job-99",
      scene_id: "scene-99",
      event_type: "asset_ready",
      timestamp: "2026-10-30T17:00:00Z",
      sequence_number: 12,
      stage_identifier: "human_reconstruction",
      status: "running",
      progress: 0.95,
      message: "3D human mesh ready",
      entity_id: "person-123",
      asset_reference: {
        asset_id: "asset-mesh-123",
        asset_type: "mesh",
        uri: "/api/v1/assets/meshes/mesh.glb",
        mime_type: "model/gltf-binary",
        size_bytes: 331940,
      },
    };

    const normalized = normalizeProcessingEvent(rawBackendEvent);
    expect(normalized.eventId).toBe("raw-ev-1");
    expect(normalized.eventType).toBe("asset_ready");
    expect(normalized.sequenceNumber).toBe(12);
    expect(normalized.stageIdentifier).toBe("human_reconstruction");
    expect(normalized.entityId).toBe("person-123");
    expect(normalized.assetReference?.assetType).toBe("mesh");
    expect(normalized.assetReference?.uri).toBe("/api/v1/assets/meshes/mesh.glb");
  });
});

