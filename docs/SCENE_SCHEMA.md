# DRISHTI Scene Schema Specification

**Version:** 1.1.0 (Phase 06: Multi-Person Scene Assembly)  
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
- **Model Space vs. Scene Space:**
  - **Canonical Model Space**: The reconstructed human mesh (e.g. SMPL template) is modeled in canonical model space centered at the pelvis origin $(0, 0, 0)$ with canonical scale ($\approx 1.7\text{m}$). The underlying geometry asset is stored intact.
  - **Scene Space**: Each entity is positioned, rotated, and scaled in the shared scene graph:
    $$\mathbf{T}_{\text{scene}} = \mathbf{T}_{\text{placement}} \times \mathbf{R}_{\text{orientation}} \times \mathbf{S}_{\text{normalization}}$$
    This preserves the original mesh topology while placing each person deterministically according to image-space position, relative depth proximity, and scale.

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
  representations: SceneRepresentations; // Status of scene representations (photo, spatialPhoto, wireframe, solid3d)
  depth?: SceneDepthMetadata;         // Phase 03: Scene-wide depth map metadata
  processingStatus: "unprocessed" | "processing" | "ready" | "error";
  activeJobId?: string | null;        // Active processing job ID if running

  // Phase 06 Additions
  sceneGraph?: SceneGraph;            // Hierarchical multi-person scene graph
  assemblyReport?: SceneAssemblyReport; // Multi-person scene assembly summary
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

## 4. Scene Entities & Phase 06 Additions

Entities represent subjects isolated within the scene. In Phase 02, each detected person receives a stable entity ID (`person-001`, `person-002`, ...) along with `DetectionInfo` and `SegmentationInfo`:

```typescript
interface SceneEntity {
  id: string;                         // Unique entity ID (e.g. "person-001")
  entityType: "person" | "background" | "environment" | "prop";
  sourceImageId: string;              // Association with source photo
  name?: string;                      // Display name (e.g. "Person 1")
  boundingBox?: BoundingBox2D;        // Normalized 2D image coordinates [0, 1]
  transform: Transform;               // 3D position, rotation, scale in scene space
  geometryRef?: string | null;        // Asset ID of polygon mesh (Phase 05/06)
  materialRef?: string | null;        // Asset ID of texture/material (Phase 07)
  skeletonRef?: string | null;        // Asset ID of humanoid skeleton (Phase 08)
  animationState: AnimationState;     // Playback status and clip name (Phase 09)
  visibility: boolean;                // Viewport visibility toggle
  reconstructionStatus: "unstarted" | "detecting" | "reconstructing" | "ready" | "failed";
  confidence?: ConfidenceMetadata;    // Model confidence metrics
  
  // Phase 02 Extensions
  detectionInfo?: DetectionInfo;      // Detailed detection model results
  segmentationInfo?: SegmentationInfo;// Instance segmentation mask metadata
  
  // Phase 03 Extensions
  depthInfo?: EntityDepthInfo;        // Per-person depth statistics and proximity
  
  // Phase 05 Extensions
  humanReconstruction?: HumanReconstructionData; // 3D parametric human mesh reconstruction

  // Phase 06 Extensions
  placement?: PlacementProvenance;    // Provenance of 3D scene placement & depth
  sceneNodeId?: string;               // Node ID in the scene graph
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

### 4.3 Entity Depth Info (Phase 03)

```typescript
interface EntityDepthInfo {
  medianProximity: number;            // Median relative proximity [0.0 - 1.0] (1.0 = closest)
  minProximity: number;               // Minimum proximity within mask
  maxProximity: number;               // Maximum proximity within mask
  depthVariation: number;             // Standard deviation across entity pixels
  pixelCoverage: number;              // Number of depth pixels sampled
  method: "mask" | "bounding_box";    // Sampling method utilized
}
```

### 4.4 Human Reconstruction Data (Phase 05)

Tracks genuine 3D parametric human mesh reconstruction, model provenance, and geometric statistics:

```typescript
interface MeshBounds {
  min: [number, number, number];      // [xMin, yMin, zMin] in meters
  max: [number, number, number];      // [xMax, yMax, zMax] in meters
  dimensions: [number, number, number]; // [width, height, depth] in meters
}

interface GeometryMetadata {
  vertexCount: number;                // Standard SMPL = 6,890 vertices
  faceCount: number;                  // Standard SMPL = 13,776 triangular faces
  hasNormals: boolean;                // Surface vertex normals computed
  isWatertight: boolean;              // Watertight manifold status
  bounds: MeshBounds;                 // 3D spatial extents
}

interface ModelProvenance {
  modelName: string;                  // e.g. "SPIN" (Kolotouros et al., ICCV 2019)
  checkpoint: string;                 // Checkpoint filename or version tag
  bodyModel: string;                  // "SMPL-Neutral"
  device: string;                     // "mps", "cuda", or "cpu"
  inferredAt: string;                 // ISO 8601 timestamp
}

interface CoordinateConvention {
  upAxis: "+Y";                       // Vertical elevation axis
  frontAxis: "+Z";                    // Axis pointing towards camera
  rightAxis: "+X";                    // Lateral right axis
  units: "meters";                    // Real-world metric scale
}

interface HumanReconstructionData {
  status: "unstarted" | "reconstructing" | "ready" | "failed";
  meshAssetId?: string;               // Binary GLTF asset ID
  meshUri?: string;                   // Relative or absolute URI to .glb asset
  geometry?: GeometryMetadata;        // Structural mesh statistics
  provenance?: ModelProvenance;       // Model lineage and hardware runtime
  coordinateConvention?: CoordinateConvention; // Coordinate system alignment
  textureStatus: "untextured_clay";   // Studio clay PBR material
  error?: string;                     // Detailed error message if reconstruction failed
}
```

---

## 5. Depth Metadata & Spatial Photo Specification (Phase 03)

### 5.1 Scene Depth Metadata

```typescript
interface SceneDepthMetadata {
  status: "unstarted" | "computed" | "failed";
  modelName: string;                  // e.g. "depth-anything-v2-small"
  depthAssetId?: string;              // Machine-readable float32 .npy asset ID
  depthMapUri?: string;               // URI to .npy raw depth file
  visualizationAssetId?: string;      // Grayscale 8-bit PNG asset ID
  visualizationUri?: string;          // URI to grayscale PNG visualization
  width?: number;                     // Depth map width (matches source photo)
  height?: number;                    // Depth map height (matches source photo)
  normalization?: DepthNormalizationMetadata;
  spatialPhoto?: SpatialPhotoConfig;  // Spatial photo rendering parameters
  error?: string;                     // Non-fatal error message if estimation failed
}
```

### 5.2 Depth Normalization & Canonical Proximity Convention

DRISHTI adopts an explicit **Canonical Relative Proximity** convention:
- $P \in [0.0, 1.0]$: Dimensionless relative scalar value.
- $1.0 = \text{nearest foreground}$ (maximum forward vertex displacement).
- $0.0 = \text{farthest background}$ (zero displacement, anchors to plane).
- Clamped with robust 2nd and 98th percentiles ($p_2, p_{98}$) to eliminate extreme prediction outliers and sensor spikes.

```typescript
interface DepthNormalizationMetadata {
  method: "percentile_clamping";
  p2: number;                         // 2nd percentile raw value
  p98: number;                        // 98th percentile raw value
  inverted: boolean;                  // True (inverted so larger = closer)
  nearPlane: number;                  // 0.0
  farPlane: number;                   // 1.0
  proximityConvention: "1.0_is_nearest_foreground";
}
```

### 5.3 Spatial Photo Configuration

```typescript
interface SpatialPhotoConfig {
  meshResolution: [number, number];   // Subdivisions, e.g. [128, 128]
  displacementScale: number;          // Default vertex displacement multiplier (e.g. 0.08)
  aspectRatio: number;                // Width / Height ratio locked to photo
  cameraParallax: {
    maxAzimuthDeg: number;            // Maximum horizontal orbit (+/- 18 deg)
    maxPolarDeg: number;              // Maximum vertical elevation (75 to 105 deg)
    damping: number;                  // Smooth orbit damping factor (0.06)
  };
}
```

---

## 6. Multi-Person Scene Graph & Assembly Specification (Phase 06)

Phase 06 introduces deterministic multi-person spatial assembly, transforming individually reconstructed people into a unified, coherent 3D scene.

### 6.1 Placement Provenance (`PlacementProvenance`)

Records the complete mathematical derivation and source attribution for an entity's 3D spatial placement:

```typescript
interface PlacementProvenance {
  imageCenterNorm: [number, number];  // Normalized 2D center [cx, cy] in [0, 1]
  imageBboxNorm: [number, number, number, number]; // [x_min, y_min, x_max, y_max]
  depthSource: "entity_median_proximity" | "entity_min_proximity" | "scene_depth_fallback" | "spatial_photo_fallback" | "geometric_prior";
  relativeProximity: number;          // Canonical proximity P in [0.0, 1.0]
  spreadFactor: number;               // Depth spread multiplier (default: 2.5)
  modelNormalizationApplied: boolean; // True if centered and scaled from model origin
  orientationCorrection: [number, number, number]; // Euler angles [rx, ry, rz] in radians
  uncertaintyFlags: string[];         // e.g. ["occlusion_ambiguity", "boundary_clipping"]
  isMetric: boolean;                  // False for monocular reconstruction
}
```

### 6.2 Scene Graph Nodes (`SceneGraphNode`) & Scene Graph (`SceneGraph`)

The scene graph organizes entities in a hierarchical DAG rooted at a dedicated `root` node:

```typescript
interface SceneGraphNode {
  id: string;                         // Node ID, e.g. "node-person-001"
  name: string;                       // Display name, e.g. "Person 1"
  nodeType: "root" | "person" | "camera" | "environment" | "prop";
  entityId?: string | null;           // Associated scene entity ID
  transform: Transform;               // Position [x,y,z], Rotation [rx,ry,rz], Scale [sx,sy,sz]
  children: string[];                 // Child node IDs
  visible: boolean;                   // Viewport visibility toggle
  meshAssetId?: string | null;        // Binary GLTF mesh asset ID if reconstructed
  meshUri?: string | null;            // URI to .glb asset
  provenance?: PlacementProvenance | null; // Spatial calculation lineage
}

interface SceneGraph {
  rootId: string;                     // ID of root node (default: "root")
  nodes: Record<string, SceneGraphNode>; // Map of node ID to node instance
}
```

### 6.3 Assembly Summary Report (`SceneAssemblyReport`)

```typescript
interface SceneAssemblyReport {
  status: "assembled" | "partial" | "empty";
  totalEntities: number;              // Total person entities in scene
  reconstructedCount: number;         // Count with valid 3D polygon meshes
  unreconstructedCount: number;       // Count with in-scene placeholder bounds
  occlusionWarnings: string[];        // Entity IDs involved in occlusion ambiguities
  placementMethod: string;            // e.g. "relative_depth_normalized_plane"
}
```

### 6.4 Spatial Transformation Formulation

The assembly engine maps 2D image coordinates and monocular depth into 3D world space:

1. **Horizontal Alignment ($X$)**:
   Preserves lateral ordering (person on left appears on left):
   $$X = (c_x - 0.5) \times W_{\text{ref}}$$
   where $c_x$ is the horizontal center of the bounding box, and $W_{\text{ref}}$ is the reference plane width derived from the photograph's aspect ratio.

2. **Vertical Grounding ($Y$)**:
   Grounds characters to the $Y = 0$ floor plane:
   $$Y = \frac{H_{\text{entity}}}{2}$$
   where $H_{\text{entity}}$ is the scaled vertical extent of the entity mesh.

3. **Relative Depth Spreading ($Z$)**:
   Translates canonical proximity $P \in [0.0, 1.0]$ into scene depth:
   $$Z = -(P - P_{\text{mid}}) \times S_{\text{depth}}$$
   where $+Z$ points towards the camera/viewer, ensuring subjects with higher proximity (closer to the camera in the photograph) are placed forward in the 3D scene.

4. **Occlusion & Depth Ambiguity**:
   When 2D bounding boxes overlap significantly ($\text{IoU} > 0.12$) while estimated relative proximity is very close ($|\Delta P| < 0.05$), the system flags `"occlusion_ambiguity"` in `uncertaintyFlags` without artificially destroying original spatial positions.

5. **Partial Reconstruction Tolerance**:
   When an entity fails reconstruction or is pending, it is retained in the scene graph as an unreconstructed node (`nodeType: "person"` with `meshAssetId: null`), rendered in the viewport as an in-scene spatial bounding indicator to preserve spatial context.

---

## 7. Viewport Representations & Visualization Modes

DRISHTI provides multiple visualization layers in the viewport:

1. **`original`**: Clean photograph rendered on the upright 3D plane.
2. **`detection`**: 3D bounding box outlines with name/confidence badge chips.
3. **`segmentation`**: Color-tinted semi-transparent alpha overlays for person silhouettes.
4. **`depth`**: Normalized 8-bit grayscale depth visualization ($255 = \text{closest}$, $0 = \text{farthest}$).
5. **`spatial_photo`**: Interactive 3D mesh surface with vertex displacement and bounded parallax camera.
6. **`wireframe`**: Reconstructed 3D polygon geometry showing genuine model topology lines (unlocked in Phase 05).
7. **`solid3d`**: Solid reconstructed 3D human meshes with neutral studio clay PBR material (unlocked in Phase 05).
8. **`articulated`**: Rigged 3D human meshes with 24-joint kinematic skeleton overlay and procedural animation tracks (unlocked in Phase 07).
9. **`combined`**: Hybrid spatial scene composition placing assembled 3D characters in front of an upright photographic reference plane (unlocked in Phase 08).

Scene representations tracked in `SceneRepresentations`:
- `photo`: `ready` when an image is loaded.
- `spatial_photo`: `ready` when depth estimation has successfully computed.
- `wireframe`: `ready` when one or more human entities have reconstructed 3D meshes.
- `solid3d`: `ready` when one or more human entities have reconstructed 3D meshes.
- `articulated`: `ready` when one or more reconstructed entities have valid rigs and skeletons.
- `combined`: `ready` when both source photo and 3D human meshes are available.

### 7.1 Representation Availability Model (`computeRepresentationAvailability`)

Availability of each viewport visualization mode is dynamically calculated using `computeRepresentationAvailability(scene, stage, isAnalyzing)` based on actual asset existence:
- **`original`**: `ready` if source image asset is loaded.
- **`detection`**: `ready` if $\ge 1$ person entity exists; `processing` if detection running; otherwise `pending` or `unavailable`.
- **`segmentation`**: `ready` if $\ge 1$ entity has a persisted mask asset; `processing` if segmentation running; otherwise `pending` or `unavailable`.
- **`depth`**: `ready` if normalized grayscale depth visualization asset is present; `processing` if depth model running; `failed` if depth failed; otherwise `pending` or `unavailable`.
- **`spatial_photo`**: `ready` if both source photo AND depth map are ready; `processing` if depth/spatial preparation running; `failed` if depth failed; otherwise `unavailable`.
- **`wireframe`**: `ready` if $\ge 1$ person entity has a reconstructed 3D mesh asset; `processing` if reconstruction running; `failed` if all reconstructions failed; otherwise `unavailable`.
- **`solid3d`**: `ready` if $\ge 1$ person entity has a reconstructed 3D mesh asset; `processing` if reconstruction running; `failed` if all reconstructions failed; otherwise `unavailable`.
- **`articulated`**: `ready` if $\ge 1$ reconstructed entity has a valid rig and skeleton; `processing` if rigging/fitting running; otherwise `unavailable`.
- **`combined`**: `ready` if source photo is loaded AND $\ge 1$ reconstructed 3D mesh exists; `processing` if scene assembly running; otherwise `unavailable`.

---

## 8. Validation Rules

Enforced in both TypeScript (`packages/scene-schema`) and Python (`services/api/app/schemas/scene.py`):

1. **Schema Version Check**: Must strictly match `"1.0.0"`.
2. **Scale Validation**: Scale along all three axes $(s_x, s_y, s_z)$ must be strictly positive finite numbers ($> 0$).
3. **Normalized Bounding Boxes**: Bounding box coordinates must satisfy $0 \le x_{\min} < x_{\max} \le 1$ and $0 \le y_{\min} < y_{\max} \le 1$.
4. **Unique Entity IDs**: Entity IDs within a scene must be unique.
5. **Image Dimensions**: Source image width and height must be strictly positive integers ($32 \le \text{dim} \le 8192$).
6. **Pixel Boxes**: Integer pixel boxes must satisfy $0 \le x_1 < x_2 \le \text{width}$ and $0 \le y_1 < y_2 \le \text{height}$.
7. **Canonical Depth Proximity**: Normalized depth values and per-entity median proximity must remain strictly bounded within $[0.0, 1.0]$. Extreme values outside this range are rejected or clamped during normalization.
8. **Scene Graph Consistency**: Every node referenced in `children` or `scene_node_id` must exist in `scene_graph.nodes`. The root node must exist and have `id == root_id`. Transform values must be finite numbers and scales strictly positive.
9. **Skeleton Hierarchy Validation**: Kinematic skeletons must have exactly one root joint (`parent_id == None`), exactly 24 canonical joints, strictly positive bone lengths, no duplicate joint names or indices, no cycles, and finite 3D coordinates.
10. **Skinning Weight Normalization**: Skinning weights must be non-negative ($\ge 0$), have at most 4 influences per vertex, with bone indices bounded in $[0, \text{boneCount} - 1]$, and weights summing to $1.0$ ($|1.0 - \sum w_i| < 10^{-4}$).

---

## 9. DRISHTI Motion, Skeleton & Rigging Specification (Phase 07)

### 9.1 Kinematic Skeleton Schema (`SkeletonData`)

```typescript
interface SkeletonJoint {
  id: string;                         // Canonical identifier (e.g. "pelvis", "left_knee")
  name: string;                       // Human-readable title (e.g. "Left Knee")
  index: number;                      // 0 to 23
  parentId?: string | null;           // Parent joint ID or null for root
  position: [number, number, number]; // [x, y, z] in local meters relative to root
  rotation: [number, number, number, number]; // [qx, qy, qz, qw] orientation quaternion
  confidence?: number;                // Estimated confidence [0.0, 1.0]
  isObserved: boolean;                // Whether confirmed by 2D pose keypoint
}

interface BoneDefinition {
  id: string;                         // Bone ID, e.g. "bone-pelvis-left_hip"
  headJointId: string;                // Parent joint
  tailJointId: string;                // Child joint
  length: number;                     // Bone length in meters
}

interface SkeletonData {
  id: string;                         // Unique skeleton identifier (e.g. "skel-person-001")
  entityId: string;                   // Associated person entity
  schemaVersion: "1.0.0";
  rootJointId: string;                // "pelvis"
  joints: SkeletonJoint[];            // Exactly 24 canonical joints
  bones: BoneDefinition[];            // Exactly 23 canonical bones
  coordinateConvention: CoordinateConvention;
  source: "smpl_regressed" | "yolo_pose_projected" | "mock_anatomical";
  isAligned: boolean;
  alignmentTransform?: Transform;
  validationStatus: "valid" | "incomplete" | "invalid";
  validationErrors: string[];
  createdAt: string;
}
```

### 9.2 Mesh Rigging Schema (`RigData`)

```typescript
interface RigData {
  id: string;                         // Unique rig identifier (e.g. "rig-person-001")
  entityId: string;                   // Associated person entity
  skeletonId: string;                 // Associated skeleton
  meshAssetId?: string | null;
  status: "rigged" | "unrigged" | "failed";
  skinningMethod: "smpl_lbs_native" | "proximity_heat_weights" | "none";
  boneCount: number;                  // 24
  maxInfluencesPerVertex: number;     // 4 (hardware shader compatibility)
  weightsNormalized: boolean;         // True (L1 norm = 1.0)
  errorMessage?: string | null;
  riggedAt?: string;
}
```

### 9.3 Procedural Animation Clip Schema (`AnimationClipData`)

```typescript
interface JointTransformTrack {
  jointId: string;                    // Target joint
  timestamps: number[];               // Keyframe times in seconds
  rotations: [number, number, number, number][]; // Quaternion [qx, qy, qz, qw]
  positions?: [number, number, number][];
}

interface AnimationClipData {
  id: string;                         // Unique clip ID (e.g. "clip-idle-person-001")
  name: string;                       // "idle_breathing" | "torso_twist" | "arm_wave" | "gentle_walk"
  targetEntityId: string;
  targetSkeletonId: string;
  durationSeconds: number;
  isLooping: boolean;
  fps: number;
  provenance: "procedural";           // Explicitly notes non-recovered motion
  tracks: JointTransformTrack[];
  createdAt: string;
}
```
