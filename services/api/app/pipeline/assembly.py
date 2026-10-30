"""DRISHTI - Scene Assembly Engine
Version 1.0.0 (Phase 06: Multi-Person Scene Assembly)

Transforms individually detected, segmented, and reconstructed human entities
into a coherent, spatially-organized 3D scene graph while preserving the
original photograph's spatial composition and relative depth relationships.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.logging import logger
from app.schemas.common import EulerRotation, Transform, Vector3D
from app.schemas.scene import (
    PlacementProvenance,
    Scene,
    SceneAssemblyReport,
    SceneEntity,
    SceneGraph,
    SceneGraphNode,
)


@dataclass
class AssemblyConfig:
    """Configurable parameters for spatial composition."""
    reference_plane_height: float = 2.0  # Height of reference photo plane in meters
    depth_range_meters: float = 1.6      # Front-to-back depth spread in scene units
    min_scale_factor: float = 0.75       # Lower bound on humanoid scaling
    max_scale_factor: float = 1.25       # Upper bound on humanoid scaling
    occlusion_iou_threshold: float = 0.12 # IoU overlap triggering occlusion analysis
    occlusion_depth_margin: float = 0.05  # Proximity difference considered ambiguous


@dataclass
class AssemblyResult:
    """Result returned by the scene assembly engine."""
    success: bool
    scene_graph: SceneGraph
    report: SceneAssemblyReport
    entities: List[SceneEntity]
    error_message: Optional[str] = None


def compute_2d_iou(box_a: Any, box_b: Any) -> float:
    """Computes Intersection over Union for two normalized bounding boxes [x1, y1, x2, y2] or BoundingBox2D."""
    if hasattr(box_a, "x_min"):
        a = (box_a.x_min, box_a.y_min, box_a.x_max, box_a.y_max)
    else:
        a = tuple(box_a)
    if hasattr(box_b, "x_min"):
        b = (box_b.x_min, box_b.y_min, box_b.x_max, box_b.y_max)
    else:
        b = tuple(box_b)

    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def validate_transform_values(pos: Tuple[float, float, float], scale: Tuple[float, float, float]) -> bool:
    """Verifies that transform values are finite, non-NaN, and within sane spatial boundaries."""
    for val in pos:
        if not math.isfinite(val) or abs(val) > 100.0:
            return False
    for val in scale:
        if not math.isfinite(val) or val <= 0.0 or val > 50.0:
            return False
    return True


class SceneAssemblyEngine:
    """Deterministic Scene Assembly Engine for DRISHTI."""

    def __init__(self, config: Optional[AssemblyConfig] = None):
        self.config = config or AssemblyConfig()

    def assemble(self, scene: Scene) -> AssemblyResult:
        """Assembles all available scene entities into a unified SceneGraph."""
        logger.info(f"Assembling scene graph for scene '{scene.id}' with {len(scene.entities)} entities")

        aspect_ratio = scene.source_image.aspect_ratio if scene.source_image else 1.333
        plane_h = self.config.reference_plane_height
        plane_w = plane_h * aspect_ratio

        occlusion_warnings: List[str] = []
        updated_entities: List[SceneEntity] = []
        nodes: Dict[str, SceneGraphNode] = {}
        person_node_ids: List[str] = []

        reconstructed_count = 0
        unreconstructed_count = 0

        # -------------------------------------------------------------
        # 1. Analyze pairwise occlusion & horizontal overlap
        # -------------------------------------------------------------
        for i in range(len(scene.entities)):
            for j in range(i + 1, len(scene.entities)):
                e1 = scene.entities[i]
                e2 = scene.entities[j]

                box1 = (
                    e1.bounding_box.x_min if e1.bounding_box else 0.0,
                    e1.bounding_box.y_min if e1.bounding_box else 0.0,
                    e1.bounding_box.x_max if e1.bounding_box else 1.0,
                    e1.bounding_box.y_max if e1.bounding_box else 1.0,
                )
                box2 = (
                    e2.bounding_box.x_min if e2.bounding_box else 0.0,
                    e2.bounding_box.y_min if e2.bounding_box else 0.0,
                    e2.bounding_box.x_max if e2.bounding_box else 1.0,
                    e2.bounding_box.y_max if e2.bounding_box else 1.0,
                )

                iou = compute_2d_iou(box1, box2)
                h_overlap = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))

                if iou >= self.config.occlusion_iou_threshold or h_overlap > 0.15:
                    p1 = e1.depth.median_proximity if e1.depth else 0.5
                    p2 = e2.depth.median_proximity if e2.depth else 0.5
                    delta_p = abs(p1 - p2)

                    if delta_p < self.config.occlusion_depth_margin:
                        warn_msg = (
                            f"Entities '{e1.id}' and '{e2.id}' overlap in 2D image space (IoU: {iou:.2f}) "
                            f"with close relative depth proximity (|ΔP| = {delta_p:.3f}); "
                            f"depth ordering carries single-view ambiguity."
                        )
                        occlusion_warnings.append(warn_msg)
                        logger.info(warn_msg)

        # -------------------------------------------------------------
        # 2. Compute deterministic per-person scene placement
        # -------------------------------------------------------------
        for entity in scene.entities:
            uncertainty_flags: List[str] = []
            has_mesh = bool(
                entity.reconstruction
                and entity.reconstruction.status == "ready"
                and (entity.reconstruction.mesh_url or entity.geometry_ref)
            )

            if has_mesh:
                reconstructed_count += 1
            else:
                unreconstructed_count += 1
                uncertainty_flags.append("mesh_not_reconstructed")

            # Horizontal & vertical coordinates
            if entity.bounding_box:
                cx = (entity.bounding_box.x_min + entity.bounding_box.x_max) / 2.0
                cy = (entity.bounding_box.y_min + entity.bounding_box.y_max) / 2.0
                h_box = (entity.bounding_box.y_max - entity.bounding_box.y_min) * plane_h
                w_box = (entity.bounding_box.x_max - entity.bounding_box.x_min) * plane_w

                # Boundary clipping check
                if (
                    entity.bounding_box.x_min < 0.02
                    or entity.bounding_box.y_min < 0.02
                    or entity.bounding_box.x_max > 0.98
                    or entity.bounding_box.y_max > 0.98
                ):
                    uncertainty_flags.append("boundary_clipped")
            else:
                cx, cy, h_box, w_box = 0.5, 0.5, 1.6, 0.5
                uncertainty_flags.append("default_bounding_box")

            # Horizontal placement (X)
            pos_x = (cx - 0.5) * plane_w

            # Scale and vertical offset estimation
            mesh_center_y = 0.15  # Default SMPL pelvis-to-center offset in canonical space
            mesh_h = 1.70
            if (
                entity.reconstruction
                and entity.reconstruction.geometry_metadata
                and entity.reconstruction.geometry_metadata.bounds
            ):
                bounds = entity.reconstruction.geometry_metadata.bounds
                v_min_y = float(bounds.min[1])
                v_max_y = float(bounds.max[1])
                mesh_h = max(0.2, v_max_y - v_min_y)
                mesh_center_y = (v_max_y + v_min_y) / 2.0

            raw_scale = h_box / mesh_h
            clamped_scale = max(
                self.config.min_scale_factor,
                min(self.config.max_scale_factor, raw_scale)
            )

            # Center height aligned with image centroid, accounting for pelvis offset
            pos_y = (1.0 - cy) * plane_h - (mesh_center_y * clamped_scale)

            # Depth placement (Z) in front of reference plane at Z=-0.6
            if entity.depth and entity.depth.median_proximity is not None:
                prox = max(0.0, min(1.0, entity.depth.median_proximity))
                # Map proximity [0, 1] across depth range centered at 0.0
                pos_z = (prox - 0.5) * self.config.depth_range_meters
                depth_source = "relative_median_proximity"
                relative_proximity_val = prox
            else:
                pos_z = 0.0
                depth_source = "fallback_plane"
                relative_proximity_val = None
                uncertainty_flags.append("missing_depth")

            # Validate numeric sanity
            if not validate_transform_values((pos_x, pos_y, pos_z), (clamped_scale, clamped_scale, clamped_scale)):
                logger.warning(f"Invalid transform computed for {entity.id}, resetting to safe defaults")
                pos_x, pos_y, pos_z = 0.0, 1.0, 0.0
                clamped_scale = 1.0
                uncertainty_flags.append("transform_clamped_invalid")

            # Update entity transform
            transform = Transform(
                position=Vector3D(x=round(pos_x, 4), y=round(pos_y, 4), z=round(pos_z, 4)),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(
                    x=round(clamped_scale, 4),
                    y=round(clamped_scale, 4),
                    z=round(clamped_scale, 4)
                ),
            )
            entity.transform = transform

            provenance = PlacementProvenance(
                method="depth_projected_composition",
                image_space_coords=[round(cx, 4), round(cy, 4)],
                depth_source=depth_source,
                relative_proximity=relative_proximity_val,
                model_normalization_applied="canonical_humanoid_scale",
                orientation_correction=[0.0, 0.0, 0.0],
                uncertainty_flags=uncertainty_flags,
                is_metric=False,
                placed_at=datetime.now(timezone.utc).isoformat(),
            )
            entity.placement = provenance

            node_id = f"node-{entity.id}"
            entity.scene_node_id = node_id
            person_node_ids.append(node_id)

            # Create SceneGraphNode for entity
            mesh_asset_id = entity.geometry_ref or (
                entity.reconstruction.mesh_asset_id if entity.reconstruction else None
            )
            mesh_uri = (
                entity.reconstruction.mesh_url if entity.reconstruction else None
            )

            nodes[node_id] = SceneGraphNode(
                id=node_id,
                name=entity.name or f"Person {entity.id}",
                node_type="person",
                entity_id=entity.id,
                transform=transform,
                children=[],
                visible=entity.visibility,
                mesh_asset_id=mesh_asset_id,
                mesh_uri=mesh_uri,
                provenance=provenance,
            )

            updated_entities.append(entity)

        # -------------------------------------------------------------
        # 3. Assemble Environment Nodes (Root, Camera, Lights, Plane)
        # -------------------------------------------------------------
        root_children = [
            "camera-main",
            "light-ambient",
            "light-directional",
            "plane-reference",
            "plane-ground",
        ] + person_node_ids

        nodes["scene-root"] = SceneGraphNode(
            id="scene-root",
            name="Scene Root",
            node_type="root",
            transform=Transform(
                position=Vector3D(x=0.0, y=0.0, z=0.0),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=1.0, y=1.0, z=1.0),
            ),
            children=root_children,
            visible=True,
        )

        nodes["camera-main"] = SceneGraphNode(
            id="camera-main",
            name="Main Perspective Camera",
            node_type="camera",
            transform=Transform(
                position=Vector3D(x=0.0, y=1.2, z=3.8),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=1.0, y=1.0, z=1.0),
            ),
            children=[],
            visible=True,
        )

        nodes["light-ambient"] = SceneGraphNode(
            id="light-ambient",
            name="Ambient Fill Light",
            node_type="light",
            transform=Transform(
                position=Vector3D(x=0.0, y=0.0, z=0.0),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=1.0, y=1.0, z=1.0),
            ),
            children=[],
            visible=True,
        )

        nodes["light-directional"] = SceneGraphNode(
            id="light-directional",
            name="Directional Key Light",
            node_type="light",
            transform=Transform(
                position=Vector3D(x=3.0, y=5.0, z=4.0),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=1.0, y=1.0, z=1.0),
            ),
            children=[],
            visible=True,
        )

        nodes["plane-reference"] = SceneGraphNode(
            id="plane-reference",
            name="Source Photo Reference Plane",
            node_type="reference_plane",
            transform=Transform(
                position=Vector3D(x=0.0, y=plane_h / 2.0, z=-0.6),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=plane_w, y=plane_h, z=1.0),
            ),
            children=[],
            visible=True,
            mesh_uri=f"/api/v1/assets/{scene.source_image.asset_id}/file" if scene.source_image else None,
        )

        nodes["plane-ground"] = SceneGraphNode(
            id="plane-ground",
            name="Ground Coordinate Grid (Y=0)",
            node_type="ground_plane",
            transform=Transform(
                position=Vector3D(x=0.0, y=0.0, z=0.0),
                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                scale=Vector3D(x=10.0, y=1.0, z=10.0),
            ),
            children=[],
            visible=True,
        )

        scene_graph = SceneGraph(root_id="scene-root", nodes=nodes)

        assembly_status = "completed"
        if unreconstructed_count > 0 and reconstructed_count > 0:
            assembly_status = "partial"
        elif unreconstructed_count > 0 and reconstructed_count == 0:
            assembly_status = "partial" if len(scene.entities) > 0 else "completed"

        report = SceneAssemblyReport(
            status=assembly_status,
            assembled_at=datetime.now(timezone.utc).isoformat(),
            total_entities=len(scene.entities),
            reconstructed_count=reconstructed_count,
            unreconstructed_count=unreconstructed_count,
            occlusion_warnings=occlusion_warnings,
            placement_method="depth_projected_composition",
        )

        return AssemblyResult(
            success=True,
            scene_graph=scene_graph,
            report=report,
            entities=updated_entities,
        )


scene_assembly_engine = SceneAssemblyEngine()

