# Scene Schema Specification

**Version:** 1.0.0  
**Format:** JSON / Typed Objects

## 1. Coordinate System Convention

The engine uses a **Right-Handed 3D Cartesian Coordinate System**, matching standard 3D computer graphics conventions (Three.js, OpenGL, Blender):

| Axis | Direction | Description |
|---|---|---|
| **+X** | Right | Horizontal lateral axis (+X moves right, -X moves left) |
| **+Y** | Up | Vertical elevation axis (+Y moves up, -Y moves down) |
| **+Z** | Forward | Axis pointing towards viewer / out of screen (+Z towards viewer, -Z into screen) |

- **Base Unit:** Meters ($1.0 = 1\text{ meter}$).
- **Ground Plane:** The horizontal ground plane is defined at $Y = 0$.
- **Angles:** Rotations are expressed in radians, with default Euler order `XYZ`.

---

## 2. Scene Object Model

A complete scene instance conformant to `Scene` schema includes:

```typescript
interface Scene {
  id: string;                         // UUID or unique slug
  schemaVersion: "1.0.0";             // Strict semver string
  name: string;                       // Human-readable project name
  createdAt: string;                  // ISO 8601 timestamp
  updatedAt: string;                  // ISO 8601 timestamp
  sourceImage: SourceImageMetadata;   // Original photo properties
  coordinateSystem: CoordinateSystemInfo; // Coordinate convention
  entities: SceneEntity[];            // List of detected/reconstructed entities
  representations: SceneRepresentations; // Status of scene representations
  processingStatus: "unprocessed" | "processing" | "ready" | "error";
  activeJobId?: string | null;        // Active processing job ID if running
}
```

---

## 3. Entities

Entities represent subjects or objects isolated within the scene:

```typescript
interface SceneEntity {
  id: string;                         // Unique entity ID
  entityType: "person" | "background" | "environment" | "prop";
  sourceImageId: string;              // Association with source photo
  name?: string;                      // Display name (e.g. "Person 1")
  boundingBox?: BoundingBox2D;        // Normalized 2D image coordinates [0, 1]
  transform: Transform;               // 3D position, rotation, scale
  geometryRef?: string | null;        // Asset ID of polygon mesh
  materialRef?: string | null;        // Asset ID of texture/material
  skeletonRef?: string | null;        // Asset ID of humanoid skeleton
  animationState: AnimationState;     // Playback status and clip name
  visibility: boolean;                // Viewport visibility toggle
  reconstructionStatus: "unstarted" | "detecting" | "reconstructing" | "ready" | "failed";
  confidence?: ConfidenceMetadata;    // Model confidence metrics
}
```

---

## 4. Representations

The engine provides three representations of the same scene:

1. **`photo`**: Original photograph appearance and spatial-photo plane.
2. **`wireframe`**: Reconstructed polygon wireframe geometry.
3. **`solid3d`**: Solid reconstructed 3D human meshes with animation support.

Each representation tracks its availability via `RepresentationAsset`:

```typescript
interface RepresentationAsset {
  status: "unavailable" | "pending" | "processing" | "ready" | "failed";
  assetId?: string | null;
  uri?: string | null;
  error?: string | null;
  updatedAt: string;
}
```

In Phase 01:
- `photo`: `ready` (once an image is loaded).
- `wireframe`: `unavailable` (placeholder indicates Phase 06 availability).
- `solid3d`: `unavailable` (placeholder indicates Phase 06/07 availability).

---

## 5. Validation Rules

The schema enforces strict validation in both TypeScript (`packages/scene-schema`) and Python (`services/api/app/schemas/scene.py`):
1. **Schema Version Check**: Must strictly match `"1.0.0"`.
2. **Scale Validation**: Scale along all three axes $(s_x, s_y, s_z)$ must be strictly positive finite numbers ($> 0$).
3. **Normalized Bounding Boxes**: Bounding box coordinates must satisfy $0 \le x_{\min} < x_{\max} \le 1$ and $0 \le y_{\min} < y_{\max} \le 1$.
4. **Unique Entity IDs**: Entity IDs within a scene must be unique.
5. **Image Dimensions**: Source image width and height must be strictly positive integers.
