import {
  CURRENT_SCHEMA_VERSION,
  RepresentationStatus,
  Scene,
  SceneEntity,
  Transform
} from "./types.js";

export interface ValidationError {
  path: string;
  message: string;
  code: string;
}

export interface ValidationResult<T> {
  isValid: boolean;
  errors: ValidationError[];
  value?: T;
}

const VALID_REPRESENTATION_STATUSES: Set<RepresentationStatus> = new Set([
  "unavailable",
  "pending",
  "processing",
  "ready",
  "failed"
]);

export function validateTransform(t: unknown, pathPrefix = "transform"): ValidationError[] {
  const errors: ValidationError[] = [];
  if (!t || typeof t !== "object") {
    errors.push({ path: pathPrefix, message: "Transform must be an object", code: "INVALID_TYPE" });
    return errors;
  }

  const transform = t as Partial<Transform>;

  // Position
  if (!transform.position || typeof transform.position !== "object") {
    errors.push({ path: `${pathPrefix}.position`, message: "Position is required", code: "REQUIRED_FIELD" });
  } else {
    for (const axis of ["x", "y", "z"] as const) {
      const val = transform.position[axis];
      if (typeof val !== "number" || !Number.isFinite(val)) {
        errors.push({ path: `${pathPrefix}.position.${axis}`, message: `Position ${axis} must be a finite number`, code: "INVALID_NUMBER" });
      }
    }
  }

  // Rotation
  if (!transform.rotation || typeof transform.rotation !== "object") {
    errors.push({ path: `${pathPrefix}.rotation`, message: "Rotation is required", code: "REQUIRED_FIELD" });
  } else {
    for (const axis of ["x", "y", "z"] as const) {
      const val = transform.rotation[axis];
      if (typeof val !== "number" || !Number.isFinite(val)) {
        errors.push({ path: `${pathPrefix}.rotation.${axis}`, message: `Rotation ${axis} must be a finite number in radians`, code: "INVALID_NUMBER" });
      }
    }
  }

  // Scale
  if (!transform.scale || typeof transform.scale !== "object") {
    errors.push({ path: `${pathPrefix}.scale`, message: "Scale is required", code: "REQUIRED_FIELD" });
  } else {
    for (const axis of ["x", "y", "z"] as const) {
      const val = transform.scale[axis];
      if (typeof val !== "number" || !Number.isFinite(val) || val <= 0) {
        errors.push({ path: `${pathPrefix}.scale.${axis}`, message: `Scale ${axis} must be a positive finite number`, code: "INVALID_SCALE" });
      }
    }
  }

  return errors;
}

export function validateEntity(entity: unknown, index: number): ValidationError[] {
  const errors: ValidationError[] = [];
  const prefix = `entities[${index}]`;

  if (!entity || typeof entity !== "object") {
    errors.push({ path: prefix, message: "Entity must be an object", code: "INVALID_TYPE" });
    return errors;
  }

  const e = entity as Partial<SceneEntity>;
  if (!e.id || typeof e.id !== "string" || e.id.trim().length === 0) {
    errors.push({ path: `${prefix}.id`, message: "Entity ID is required and must be non-empty", code: "REQUIRED_FIELD" });
  }

  if (!e.entityType || !["person", "background", "environment", "prop"].includes(e.entityType)) {
    errors.push({ path: `${prefix}.entityType`, message: "Invalid or missing entityType", code: "INVALID_ENUM" });
  }

  if (e.transform) {
    errors.push(...validateTransform(e.transform, `${prefix}.transform`));
  } else {
    errors.push({ path: `${prefix}.transform`, message: "Entity transform is required", code: "REQUIRED_FIELD" });
  }

  if (e.boundingBox) {
    const { xMin, yMin, xMax, yMax } = e.boundingBox;
    if (
      typeof xMin !== "number" || typeof yMin !== "number" ||
      typeof xMax !== "number" || typeof yMax !== "number" ||
      xMin < 0 || yMin < 0 || xMax > 1 || yMax > 1 ||
      xMin >= xMax || yMin >= yMax
    ) {
      errors.push({ path: `${prefix}.boundingBox`, message: "BoundingBox must be valid normalized bounds [0, 1] with min < max", code: "INVALID_BOUNDS" });
    }
  }

  return errors;
}

export function validateScene(raw: unknown): ValidationResult<Scene> {
  const errors: ValidationError[] = [];

  if (!raw || typeof raw !== "object") {
    return {
      isValid: false,
      errors: [{ path: "root", message: "Scene root must be an object", code: "INVALID_ROOT" }]
    };
  }

  const scene = raw as Partial<Scene>;

  // ID
  if (!scene.id || typeof scene.id !== "string" || scene.id.trim().length === 0) {
    errors.push({ path: "id", message: "Scene ID is required", code: "REQUIRED_FIELD" });
  }

  // Schema Version
  if (scene.schemaVersion !== CURRENT_SCHEMA_VERSION) {
    errors.push({
      path: "schemaVersion",
      message: `Unsupported schemaVersion: "${scene.schemaVersion}". Supported version is "${CURRENT_SCHEMA_VERSION}"`,
      code: "UNSUPPORTED_VERSION"
    });
  }

  // Source Image
  if (!scene.sourceImage || typeof scene.sourceImage !== "object") {
    errors.push({ path: "sourceImage", message: "sourceImage metadata is required", code: "REQUIRED_FIELD" });
  } else {
    const img = scene.sourceImage;
    if (typeof img.width !== "number" || img.width <= 0) {
      errors.push({ path: "sourceImage.width", message: "Image width must be a positive number", code: "INVALID_DIMENSION" });
    }
    if (typeof img.height !== "number" || img.height <= 0) {
      errors.push({ path: "sourceImage.height", message: "Image height must be a positive number", code: "INVALID_DIMENSION" });
    }
  }

  // Representations
  if (!scene.representations || typeof scene.representations !== "object") {
    errors.push({ path: "representations", message: "representations object is required", code: "REQUIRED_FIELD" });
  } else {
    for (const key of ["photo", "spatial_photo", "wireframe", "solid3d"] as const) {
      const rep = scene.representations[key];
      if (!rep || typeof rep !== "object" || !VALID_REPRESENTATION_STATUSES.has(rep.status)) {
        errors.push({
          path: `representations.${key}.status`,
          message: `Invalid representation status for ${key}`,
          code: "INVALID_STATUS"
        });
      }
    }
  }

  // Entities
  if (Array.isArray(scene.entities)) {
    const seenIds = new Set<string>();
    scene.entities.forEach((entity, idx) => {
      errors.push(...validateEntity(entity, idx));
      if (entity && typeof entity === "object" && "id" in entity && typeof entity.id === "string") {
        if (seenIds.has(entity.id)) {
          errors.push({ path: `entities[${idx}].id`, message: `Duplicate entity ID "${entity.id}"`, code: "DUPLICATE_ID" });
        }
        seenIds.add(entity.id);
      }
    });
  } else {
    errors.push({ path: "entities", message: "entities must be an array", code: "INVALID_TYPE" });
  }

  return {
    isValid: errors.length === 0,
    errors,
    value: errors.length === 0 ? (scene as Scene) : undefined
  };
}

