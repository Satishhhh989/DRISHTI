import {
  CoordinateSystemInfo,
  CURRENT_SCHEMA_VERSION,
  RepresentationAsset,
  Scene,
  SceneRepresentations,
  SourceImageMetadata,
  Transform
} from "./types.js";

export const DEFAULT_COORDINATE_SYSTEM: CoordinateSystemInfo = {
  handedness: "right-handed",
  upAxis: "+Y",
  forwardAxis: "+Z",
  rightAxis: "+X",
  unit: "meters"
};

export const DEFAULT_IDENTITY_TRANSFORM: Transform = {
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0, order: "XYZ" },
  scale: { x: 1, y: 1, z: 1 }
};

export function createInitialRepresentations(photoUri?: string, photoAssetId?: string): SceneRepresentations {
  const now = new Date().toISOString();
  
  const photoAsset: RepresentationAsset = {
    status: photoUri ? "ready" : "pending",
    assetId: photoAssetId ?? (photoUri ? "photo-source" : null),
    uri: photoUri ?? null,
    updatedAt: now
  };

  const unavailableAsset = (): RepresentationAsset => ({
    status: "unavailable",
    assetId: null,
    uri: null,
    updatedAt: now
  });

  return {
    photo: photoAsset,
    spatial_photo: unavailableAsset(),
    wireframe: unavailableAsset(),
    solid3d: unavailableAsset()
  };
}

export function createInitialScene(params: {
  id: string;
  name?: string;
  sourceImage: SourceImageMetadata;
  photoUri?: string;
}): Scene {
  const now = new Date().toISOString();
  return {
    id: params.id,
    schemaVersion: CURRENT_SCHEMA_VERSION,
    name: params.name ?? `Scene-${params.id.slice(0, 8)}`,
    createdAt: now,
    updatedAt: now,
    sourceImage: params.sourceImage,
    coordinateSystem: DEFAULT_COORDINATE_SYSTEM,
    entities: [],
    representations: createInitialRepresentations(params.photoUri, params.sourceImage.assetId),
    processingStatus: "unprocessed",
    activeJobId: null
  };
}

