# DRISHTI Scene Schema Specification

**Version:** 1.0.0  
**Format:** JSON / Typed Objects (TypeScript & Pydantic)

## 1. Coordinate System Convention

DRISHTI uses a **Right-Handed 3D Cartesian Coordinate System**, matching standard 3D computer graphics conventions (Three.js, WebGL, OpenGL, Blender):

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

A complete scene instance conformant to the `Scene` schema includes:

```typescript
interface Scene {
  id: string;                         // UUID or unique slug
  schemaVersion: "1.0.0";             // Strict semver string
  name: string;                       // Human-readable project name
  createdAt: string;                  // ISO 8601 timestamp
  updatedAt: string;                  // ISO 8601 timestamp
  sourceImage: SourceImageMetadata;   // Original photo properties & detection summary
  coordinateSystem: CoordinateSystemInfo; // Coordinate convention
  entities: SceneEntity[];            // List of detected/reconstructed entities
  representations: SceneRepresentations; // Status of scene representations
  processingStatus: "unprocessed" | "processing" | "ready" | "error";
  activeJobId?: string | null;        // Active processing job ID if running
}
```

---

## 3. Source Image Metadata

Tracks the source photograph dimensions, asset identifier, and Phase 02 detection statistics:

```typescript
interface SourceImageMetadata {
  assetId: string;
  width: number;
  height: number;
  aspectRatio: number;                // width / height
  originalFilename?: string;
  format?: string;
  mimeType?: string;
  detectedPeopleCount?: number;       // Total number of validated person detections
  analyzedAt?: string;                // ISO timestamp of pipeline analysis completion
}
```

---

## 4. Scene Entities & Phase 02 Additions

Entities represent subjects isolated within the scene. In Phase 02, each detected person receives a stable entity ID (`person-001`, `person-002`, ...) along with `DetectionInfo` and `SegmentationInfo`:

```typescript
interface SceneEntity {
  id: string;                         // Unique entity ID (e.g. "person-001")
  entityType: "person" | "background" | "environment" | "prop";
  sourceImageId: string;              // Association with source photo
  name?: string;                      // Display name (e.g. "Person 1")
  boundingBox?: BoundingBox2D;        // Normalized 2D image coordinates [0, 1]
  transform: Transform;               // 3D position, rotation, scale
  geometryRef?: string | null;        // Asset ID of polygon mesh (Phase 06)
  materialRef?: string | null;        // Asset ID of texture/material (Phase 07)
  skeletonRef?: string | null;        // Asset ID of humanoid skeleton (Phase 08)
  animationState: AnimationState;     // Playback status and clip name (Phase 09)
  visibility: boolean;                // Viewport visibility toggle
  reconstructionStatus: "unstarted" | "detecting" | "reconstructing" | "ready" | "failed";
  confidence?: ConfidenceMetadata;    // Model confidence metrics
  
  // Phase 02 Extensions
  detectionInfo?: DetectionInfo;      // Detailed detection model results
  segmentationInfo?: SegmentationInfo;// Instance segmentation mask metadata
}
```

### 4.1 Detection Info

```typescript
interface DetectionInfo {
  confidence: number;                 // Detection confidence score [0.0 - 1.0]
  boundingBox: BoundingBox2D;         // Normalized coordinates [xMin, yMin, xMax, yMax]
  pixelBox: [number, number, number, number]; // Pixel coordinates [x1, y1, x2, y2]
  model: string;                      // Model name (e.g. "yolov8n-seg")
  status: "detected" | "validated" | "failed";
}
```

### 4.2 Segmentation Info

```typescript
interface SegmentationInfo {
  status: "segmented" | "failed" | "unstarted";
  maskAssetId?: string;               // Asset ID of 8-bit binary PNG mask
  maskUri?: string;                   // Relative or absolute URI to mask file
  confidence?: number;                // Mean segmentation confidence score
  model?: string;                     // Segmentation model name
  pixelCount?: number;                // Foreground pixel area count
}
```

---

## 5. Viewport Representations & Visualization Modes

DRISHTI provides multiple visualization layers in the viewport:

1. **`original`**: Clean photograph rendered on the 3D plane.
2. **`detection`**: 3D bounding box outlines with name/confidence badge chips.
3. **`segmentation`**: Color-tinted semi-transparent alpha overlays for person silhouettes.
4. **`wireframe`**: Reconstructed 3D polygon geometry (honest placeholder for Phase 06).
5. **`solid3d`**: Solid reconstructed 3D human meshes (honest placeholder for Phase 06/07).

Scene representations tracked in `SceneRepresentations`:
- `photo`: `ready` when an image is loaded.
- `wireframe`: `unavailable` (Phase 06).
- `solid3d`: `unavailable` (Phase 06/07).

---

## 6. Validation Rules

Enforced in both TypeScript (`packages/scene-schema`) and Python (`services/api/app/schemas/scene.py`):

1. **Schema Version Check**: Must strictly match `"1.0.0"`.
2. **Scale Validation**: Scale along all three axes $(s_x, s_y, s_z)$ must be strictly positive finite numbers ($> 0$).
3. **Normalized Bounding Boxes**: Bounding box coordinates must satisfy $0 \le x_{\min} < x_{\max} \le 1$ and $0 \le y_{\min} < y_{\max} \le 1$.
4. **Unique Entity IDs**: Entity IDs within a scene must be unique.
5. **Image Dimensions**: Source image width and height must be strictly positive integers ($32 \le \text{dim} \le 8192$).
6. **Pixel Boxes**: Integer pixel boxes must satisfy $0 \le x_1 < x_2 \le \text{width}$ and $0 \le y_1 < y_2 \le \text{height}$.
