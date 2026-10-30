import { describe, it, expect } from "vitest";
import { validateProject } from "../src/validation.js";
import { Project, CURRENT_SCHEMA_VERSION } from "../src/types.js";

describe("Phase 09: Project Model Schema and Validation", () => {
  const validProject: Project = {
    id: "proj-12345",
    name: "Summer Vacation 3D",
    schemaVersion: CURRENT_SCHEMA_VERSION,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    sceneId: "scene-98765",
    sourceImage: {
      assetId: "asset-img-1",
      filename: "friends.jpg",
      width: 1920,
      height: 1080,
      aspectRatio: 1.777778,
      mimeType: "image/jpeg",
      sizeBytes: 1048576,
      uploadedAt: new Date().toISOString(),
    },
    settings: {
      activeRepresentation: "solid3d",
      showGrid: true,
      showReferencePlane: true,
    },
    assetManifest: [
      {
        assetId: "asset-img-1",
        assetType: "source_photo",
        relativePath: "assets/photos/friends.jpg",
        filename: "friends.jpg",
        sizeBytes: 1048576,
        mimeType: "image/jpeg",
      },
    ],
    processingState: {
      detection: true,
      segmentation: true,
      depth: true,
      reconstruction: true,
      assembly: true,
      motion: false,
    },
  };

  it("validates a compliant project definition", () => {
    const res = validateProject(validProject);
    expect(res.isValid).toBe(true);
    expect(res.errors.length).toBe(0);
    expect(res.value?.id).toBe("proj-12345");
  });

  it("rejects project with missing ID or empty name", () => {
    const invalid = { ...validProject, id: "", name: "" };
    const res = validateProject(invalid);
    expect(res.isValid).toBe(false);
    expect(res.errors.some((e) => e.path === "id")).toBe(true);
    expect(res.errors.some((e) => e.path === "name")).toBe(true);
  });

  it("rejects project with mismatched schema version", () => {
    const invalid = { ...validProject, schemaVersion: "99.0.0" };
    const res = validateProject(invalid);
    expect(res.isValid).toBe(false);
    expect(res.errors.some((e) => e.code === "UNSUPPORTED_VERSION")).toBe(true);
  });

  it("rejects project with missing sceneId or non-array manifest", () => {
    const invalid = { ...validProject, sceneId: "", assetManifest: "not-an-array" };
    const res = validateProject(invalid as unknown);
    expect(res.isValid).toBe(false);
    expect(res.errors.some((e) => e.path === "sceneId")).toBe(true);
    expect(res.errors.some((e) => e.path === "assetManifest")).toBe(true);
  });
});

