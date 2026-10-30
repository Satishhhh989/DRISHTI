import { describe, it, expect } from "vitest";
import {
  REPRESENTATION_REGISTRY,
  getAllRepresentations,
  getRepresentationEntry,
  is3DRepresentation,
  is2DRepresentation,
  VisualizationMode,
} from "../src/index.js";

describe("Representation Registry (Phase 08)", () => {
  const allModes: VisualizationMode[] = [
    "original",
    "detection",
    "segmentation",
    "depth",
    "spatial_photo",
    "wireframe",
    "solid3d",
    "articulated",
    "combined",
  ];

  it("registers all 9 canonical representations with complete specifications", () => {
    for (const mode of allModes) {
      const entry = REPRESENTATION_REGISTRY[mode];
      expect(entry).toBeDefined();
      expect(entry.id).toBe(mode);
      expect(entry.displayName.length).toBeGreaterThan(0);
      expect(["2d_photo", "spatial_photo", "inspection", "3d_mesh", "hybrid"]).toContain(entry.category);
      expect(Array.isArray(entry.requiredAssets)).toBe(true);
      expect(entry.requiredAssets.length).toBeGreaterThan(0);
      expect(Array.isArray(entry.supportedSceneTypes)).toBe(true);
      expect(entry.renderingBehavior.length).toBeGreaterThan(0);
      expect(Array.isArray(entry.knownLimitations)).toBe(true);
      expect(typeof entry.supportsSelection).toBe("boolean");
      expect(typeof entry.supportsInteraction).toBe("boolean");
      expect(typeof entry.supportsAnimation).toBe("boolean");
    }
  });

  it("getAllRepresentations returns all 9 entries", () => {
    const all = getAllRepresentations();
    expect(all.length).toBe(9);
    expect(all.map((r) => r.id)).toEqual(expect.arrayContaining(allModes));
  });

  it("getRepresentationEntry retrieves specific entries or falls back to original", () => {
    const solid = getRepresentationEntry("solid3d");
    expect(solid.displayName).toBe("3D Solid Mesh");
    expect(solid.supportsSelection).toBe(true);
    expect(solid.supportsAnimation).toBe(true);

    const articulated = getRepresentationEntry("articulated");
    expect(articulated.displayName).toBe("Articulated Character");
    expect(articulated.requiredAssets).toContain("skeleton_rig");

    const combined = getRepresentationEntry("combined");
    expect(combined.category).toBe("hybrid");
    expect(combined.requiredAssets).toContain("source_image");
    expect(combined.requiredAssets).toContain("mesh_reconstruction");

    // Invalid fallback
    const fallback = getRepresentationEntry("unknown" as VisualizationMode);
    expect(fallback.id).toBe("original");
  });

  it("correctly categorizes 2D and 3D representations", () => {
    expect(is2DRepresentation("original")).toBe(true);
    expect(is2DRepresentation("detection")).toBe(true);
    expect(is2DRepresentation("segmentation")).toBe(true);
    expect(is2DRepresentation("depth")).toBe(true);
    expect(is2DRepresentation("solid3d")).toBe(false);

    expect(is3DRepresentation("wireframe")).toBe(true);
    expect(is3DRepresentation("solid3d")).toBe(true);
    expect(is3DRepresentation("articulated")).toBe(true);
    expect(is3DRepresentation("combined")).toBe(true);
    expect(is3DRepresentation("original")).toBe(false);
  });
});
