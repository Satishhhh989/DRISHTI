import { describe, it, expect } from "vitest";
import {
  computeRepresentationAvailability,
  createInitialScene,
} from "../src/index.js";

describe("computeRepresentationAvailability", () => {
  it("marks original ready when photo is present, but others unavailable", () => {
    const scene = createInitialScene({
      id: "scene-1",
      name: "Test Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 1000,
        uploadedAt: new Date().toISOString(),
      },
    });

    const availability = computeRepresentationAvailability(scene, null, false);
    expect(availability.original.status).toBe("ready");
    expect(availability.original.isSelectable).toBe(true);

    expect(availability.detection.status).toBe("unavailable");
    expect(availability.detection.isSelectable).toBe(false);

    expect(availability.segmentation.status).toBe("unavailable");
    expect(availability.depth.status).toBe("unavailable");
    expect(availability.spatial_photo.status).toBe("unavailable");
    expect(availability.wireframe.status).toBe("unavailable");
    expect(availability.solid3d.status).toBe("unavailable");
  });

  it("marks detection and segmentation processing during analysis", () => {
    const scene = createInitialScene({
      id: "scene-1",
      name: "Test Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 1000,
        uploadedAt: new Date().toISOString(),
      },
    });

    const detAvailability = computeRepresentationAvailability(scene, "person_detection", true);
    expect(detAvailability.detection.status).toBe("processing");
    expect(detAvailability.segmentation.status).toBe("pending");

    const segAvailability = computeRepresentationAvailability(scene, "instance_segmentation", true);
    expect(segAvailability.segmentation.status).toBe("processing");
  });

  it("marks detection, segmentation, and depth ready when assets exist", () => {
    const scene = createInitialScene({
      id: "scene-1",
      name: "Test Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 1000,
        uploadedAt: new Date().toISOString(),
      },
    });

    scene.entities.push({
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-1",
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
        boundingBox: { xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.8 },
        pixelBox: [80, 120, 400, 480],
        model: "yolo",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      segmentation: {
        status: "ready",
        maskAssetId: "mask-1",
        maskUri: "/api/v1/assets/mask-1/file",
      },
    });

    scene.depth = {
      status: "ready",
      model: "depth-anything",
      modelFamily: "Depth-Anything-V2",
      depthConvention: "relative_proximity",
      width: 800,
      height: 600,
      device: "mps",
      visualizationAssetId: "depth-vis-1",
      visualizationUri: "/api/v1/assets/depth-vis-1/file",
    };

    scene.representations.spatial_photo = {
      status: "ready",
      assetId: "depth-1",
      uri: "/api/v1/assets/depth-vis-1/file",
      updatedAt: new Date().toISOString(),
    };

    const availability = computeRepresentationAvailability(scene, null, false);
    expect(availability.detection.status).toBe("ready");
    expect(availability.detection.isSelectable).toBe(true);

    expect(availability.segmentation.status).toBe("ready");
    expect(availability.segmentation.isSelectable).toBe(true);

    expect(availability.depth.status).toBe("ready");
    expect(availability.depth.isSelectable).toBe(true);

    expect(availability.spatial_photo.status).toBe("ready");
    expect(availability.spatial_photo.isSelectable).toBe(true);

    expect(availability.wireframe.status).toBe("unavailable");
    expect(availability.solid3d.status).toBe("unavailable");
  });

  it("marks wireframe and solid3d ready when a reconstructed 3D mesh is available", () => {
    const scene = createInitialScene({
      id: "scene-1",
      name: "Test Scene",
      sourceImage: {
        assetId: "photo-1",
        filename: "test.jpg",
        width: 800,
        height: 600,
        aspectRatio: 1.333,
        mimeType: "image/jpeg",
        sizeBytes: 1000,
        uploadedAt: new Date().toISOString(),
      },
    });

    scene.entities.push({
      id: "person-001",
      entityType: "person",
      sourceImageId: "photo-1",
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
        boundingBox: { xMin: 0.1, yMin: 0.2, xMax: 0.5, yMax: 0.8 },
        pixelBox: [80, 120, 400, 480],
        model: "yolo",
        detectedAt: new Date().toISOString(),
        status: "ready",
      },
      reconstruction: {
        status: "ready",
        meshAssetId: "mesh-person-001.glb",
        meshUrl: "/api/v1/assets/meshes/mesh-person-001.glb",
        geometryMetadata: {
          vertexCount: 6890,
          faceCount: 13776,
          bounds: { min: [-0.8, -1.0, -0.2], max: [0.8, 0.8, 0.2] },
          dimensions: [1.6, 1.8, 0.4],
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

    const availability = computeRepresentationAvailability(scene, null, false);
    expect(availability.wireframe.status).toBe("ready");
    expect(availability.wireframe.isSelectable).toBe(true);
    expect(availability.solid3d.status).toBe("ready");
    expect(availability.solid3d.isSelectable).toBe(true);
  });
});

