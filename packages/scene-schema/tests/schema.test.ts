import { describe, it, expect } from "vitest";
import {
  createInitialScene,
  validateScene,
  validateTransform,
  validateEntity,
  CURRENT_SCHEMA_VERSION,
  DEFAULT_COORDINATE_SYSTEM
} from "../src/index.js";

describe("Scene Schema Validation", () => {
  it("validates a properly initialized initial scene", () => {
    const scene = createInitialScene({
      id: "scene-test-1234",
      sourceImage: {
        assetId: "asset-img-1",
        filename: "group_photo.jpg",
        width: 1920,
        height: 1080,
        aspectRatio: 1920 / 1080,
        mimeType: "image/jpeg",
        sizeBytes: 2048000,
        uploadedAt: new Date().toISOString()
      },
      photoUri: "blob:http://localhost:3000/test-uuid"
    });

    const result = validateScene(scene);
    expect(result.isValid).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.value?.coordinateSystem).toEqual(DEFAULT_COORDINATE_SYSTEM);
    expect(result.value?.representations.photo.status).toBe("ready");
    expect(result.value?.representations.wireframe.status).toBe("unavailable");
    expect(result.value?.representations.solid3d.status).toBe("unavailable");
  });

  it("rejects an unsupported schema version", () => {
    const scene = createInitialScene({
      id: "scene-bad-ver",
      sourceImage: {
        assetId: "asset-1",
        filename: "test.png",
        width: 800,
        height: 600,
        aspectRatio: 800 / 600,
        mimeType: "image/png",
        sizeBytes: 1000,
        uploadedAt: new Date().toISOString()
      }
    });

    // @ts-expect-error Testing invalid schema version
    scene.schemaVersion = "99.0.0";
    const result = validateScene(scene);
    expect(result.isValid).toBe(false);
    expect(result.errors.some(e => e.code === "UNSUPPORTED_VERSION")).toBe(true);
  });

  it("validates and catches transform scale anomalies", () => {
    const invalidTransform = {
      position: { x: 0, y: 1.5, z: 0 },
      rotation: { x: 0, y: 0, z: 0 },
      scale: { x: 1, y: -0.5, z: 1 } // negative scale
    };

    const errors = validateTransform(invalidTransform);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.some(e => e.code === "INVALID_SCALE")).toBe(true);
  });

  it("validates bounding boxes correctly", () => {
    const invalidEntity = {
      id: "entity-01",
      entityType: "person",
      sourceImageId: "asset-1",
      transform: {
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      },
      boundingBox: {
        xMin: 0.8,
        yMin: 0.2,
        xMax: 0.3, // xMin >= xMax
        yMax: 0.9
      },
      animationState: { currentClip: null, isPlaying: false, playbackSpeed: 1, loop: false },
      visibility: true,
      reconstructionStatus: "unstarted"
    };

    const errors = validateEntity(invalidEntity, 0);
    expect(errors.some(e => e.code === "INVALID_BOUNDS")).toBe(true);
  });

  it("catches duplicate entity IDs in scene", () => {
    const scene = createInitialScene({
      id: "scene-dup",
      sourceImage: {
        assetId: "asset-1",
        filename: "test.jpg",
        width: 1000,
        height: 1000,
        aspectRatio: 1,
        mimeType: "image/jpeg",
        sizeBytes: 5000,
        uploadedAt: new Date().toISOString()
      }
    });

    const ent1 = {
      id: "duplicate-id",
      entityType: "person" as const,
      sourceImageId: "asset-1",
      transform: {
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: { x: 1, y: 1, z: 1 }
      },
      animationState: { currentClip: null, isPlaying: false, playbackSpeed: 1, loop: false },
      visibility: true,
      reconstructionStatus: "unstarted" as const
    };

    scene.entities = [ent1, { ...ent1 }];
    const result = validateScene(scene);
    expect(result.isValid).toBe(false);
    expect(result.errors.some(e => e.code === "DUPLICATE_ID")).toBe(true);
  });
});

