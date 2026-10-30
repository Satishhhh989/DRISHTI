/**
 * PHOTO → 3D ENGINE - Local Asset Types
 */

export type AssetCategory =
  | "original_photo"
  | "segmentation_mask"
  | "depth_map"
  | "mesh"
  | "texture"
  | "skeleton"
  | "export";

export interface AssetMetadata {
  assetId: string;
  category: AssetCategory;
  filename: string;
  relativePath: string;
  mimeType: string;
  sizeBytes: number;
  checksumSha256: string;
  createdAt: string;
  sceneId?: string | null;
  entityId?: string | null;
  dimensions?: {
    width: number;
    height: number;
  } | null;
  formatDetails?: Record<string, unknown>;
}
