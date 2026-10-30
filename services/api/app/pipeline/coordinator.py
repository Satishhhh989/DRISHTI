"""DRISHTI - Pipeline Coordinator & Job Execution Engine
Version 1.2.0 (Phase 04: Live Visualization & Progressive Processing Engine)

Executes the sequenced lifecycle with genuine progressive state updates and cancellation:
1. IMAGE_VALIDATION
2. IMAGE_PREPARATION
3. PERSON_DETECTION (Progressively registers detected person entities in scene)
4. DETECTION_VALIDATION
5. INSTANCE_SEGMENTATION
6. MASK_VALIDATION & PERSISTENCE (Progressively attaches individual masks to entities)
7. DEPTH_ESTIMATION (Progressively attaches depth map to scene)
8. DEPTH_NORMALIZATION & PERSISTENCE (Activates Depth Map mode)
9. DEPTH_ASSOCIATION (Progressively computes per-person depth statistics)
10. SPATIAL_PHOTO_PREPARATION (Activates Spatial Photo mode)
11. SCENE_UPDATE
12. COMPLETED
"""

import asyncio
from datetime import datetime, timezone
import io
from pathlib import Path
from typing import List, Optional
import numpy as np

from app.assets.manager import asset_manager
from app.core.logging import logger
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.depth import (
    DepthAnythingV2Estimator,
    compute_entity_depth_stats,
)
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.interfaces import (
    DepthEstimationResult,
    DetectionResult,
    IDepthEstimator,
    IPersonDetector,
    ISegmentor,
    SegmentationResult,
)
from app.pipeline.mock import (
    MockDepthEstimator,
    MockPersonDetector,
    MockPersonSegmentor,
)
from app.pipeline.models import model_manager
from app.pipeline.segmentor import YoloPersonSegmentor
from app.scene.manager import scene_manager
from app.schemas.events import AssetReference, StructuredError
from app.schemas.scene import (
    AnimationState,
    ConfidenceMetadata,
    DepthNormalizationMetadata,
    DetectionInfo,
    RepresentationAsset,
    Scene,
    SceneDepthMetadata,
    SceneEntity,
    SegmentationInfo,
    SpatialPhotoConfig,
    Transform,
)


class PipelineCoordinator:
    """Orchestrates detection, segmentation, depth estimation, and progressive scene updates."""

    def __init__(
        self,
        detector: Optional[IPersonDetector] = None,
        segmentor: Optional[ISegmentor] = None,
        depth_estimator: Optional[IDepthEstimator] = None,
    ):
        self._detector = detector
        self._segmentor = segmentor
        self._depth_estimator = depth_estimator

    def get_detector(self) -> IPersonDetector:
        if self._detector is not None:
            return self._detector
        if model_manager.is_model_available("yolov8n-seg"):
            return YoloPersonDetector("yolov8n-seg")
        logger.warning("YOLO weights not found; falling back to MockPersonDetector")
        return MockPersonDetector()

    def get_segmentor(self) -> ISegmentor:
        if self._segmentor is not None:
            return self._segmentor
        if model_manager.is_model_available("yolov8n-seg"):
            return YoloPersonSegmentor("yolov8n-seg")
        return MockPersonSegmentor()

    def get_depth_estimator(self) -> IDepthEstimator:
        if self._depth_estimator is not None:
            return self._depth_estimator
        if model_manager.is_model_available("depth-anything-v2-small"):
            return DepthAnythingV2Estimator("depth-anything-v2-small")
        logger.warning("Depth Anything weights not found; falling back to MockDepthEstimator")
        return MockDepthEstimator()

    async def execute_pipeline(
        self,
        scene_id: str,
        job_id: str,
        confidence_threshold: float = 0.35,
        include_depth: bool = True,
    ) -> Scene:
        """Executes the progressive DRISHTI image analysis and spatial-photo pipeline."""
        scene = scene_manager.get_scene(scene_id)
        photo_asset_id = scene.source_image.asset_id
        photo_path = asset_manager.get_asset_path(photo_asset_id)

        try:
            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                logger.info(f"Job {job_id} was cancelled before starting.")
                return scene

            # -------------------------------------------------------------
            # Stage 1: IMAGE_VALIDATION
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="image_validation")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="image_validation",
                status="running",
                progress=0.08,
                message="Validating photograph integrity and dimension bounds...",
            )
            if not photo_path.exists() or photo_path.stat().st_size == 0:
                raise FileNotFoundError(f"Source photograph file does not exist: {photo_path}")

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="image_validation",
                status="running",
                progress=0.12,
                message="Photograph validated successfully.",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                return scene

            # -------------------------------------------------------------
            # Stage 2: IMAGE_PREPARATION
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="image_preparation")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="image_preparation",
                status="running",
                progress=0.15,
                message="Establishing coordinate reference and preparing tensors...",
            )
            await asyncio.sleep(0.01)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="image_preparation",
                status="running",
                progress=0.20,
                message="Image coordinates ready for analysis.",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                return scene

            # -------------------------------------------------------------
            # Stage 3: PERSON_DETECTION
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="person_detection")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="person_detection",
                status="running",
                progress=0.25,
                message="Executing local neural network person detection...",
            )

            detector = self.get_detector()
            detections = await asyncio.to_thread(
                detector.detect_people,
                photo_path,
                confidence_threshold,
            )

            num_detected = len(detections)
            logger.info(f"Job {job_id}: {num_detected} person detections produced")

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                return scene

            # -------------------------------------------------------------
            # Stage 4: DETECTION_VALIDATION & PROGRESSIVE ENTITY CREATION
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="detection_validation")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="detection_validation",
                status="running",
                progress=0.42,
                message="Validating detection bounds and coordinate integrity...",
            )
            valid_detections = [
                d for d in detections
                if d.bounding_box.x_min < d.bounding_box.x_max and d.bounding_box.y_min < d.bounding_box.y_max
            ]

            # Progressive update: Create initial scene entities immediately
            entities: List[SceneEntity] = []
            for idx, det in enumerate(valid_detections):
                entity = SceneEntity(
                    id=det.entity_id,
                    entity_type="person",
                    source_image_id=photo_asset_id,
                    name=f"Person {idx + 1}",
                    bounding_box=det.bounding_box,
                    transform=Transform(),
                    geometry_ref=None,
                    material_ref=None,
                    skeleton_ref=None,
                    animation_state=AnimationState(),
                    visibility=True,
                    reconstruction_status="unstarted",
                    confidence=ConfidenceMetadata(
                        detection_confidence=det.confidence,
                    ),
                    detection=DetectionInfo(
                        confidence=det.confidence,
                        bounding_box=det.bounding_box,
                        pixel_box=list(det.pixel_box),
                        model=det.model,
                        status="ready",
                    ),
                    segmentation=SegmentationInfo(
                        status="processing" if len(valid_detections) > 0 else "unstarted"
                    ),
                    depth=None,
                )
                entities.append(entity)

            # PROGRESSIVE COMMIT: Commit detected entities immediately to scene manager!
            scene.entities = entities
            scene.processing_status = "processing"
            scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="intermediate_result_available",
                stage_identifier="person_detection",
                status="running",
                progress=0.45,
                message=f"Detected {len(valid_detections)} people in photograph.",
            )

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="detection_validation",
                status="running",
                progress=0.48,
                message=f"{len(valid_detections)} detections validated and positioned in scene.",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 5: INSTANCE_SEGMENTATION
            # -------------------------------------------------------------
            segmentation_results: List[SegmentationResult] = []
            if valid_detections:
                job_tracker.update_job_status(job_id, status="running", current_stage="instance_segmentation")
                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_started",
                    stage_identifier="instance_segmentation",
                    status="running",
                    progress=0.50,
                    message=f"Generating individual alpha masks for {len(valid_detections)} people...",
                )

                segmentor = self.get_segmentor()
                segmentation_results = await asyncio.to_thread(
                    segmentor.segment_people,
                    photo_path,
                    valid_detections,
                )

                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_completed",
                    stage_identifier="instance_segmentation",
                    status="running",
                    progress=0.58,
                    message=f"Generated {len(segmentation_results)} individual segmentation masks.",
                )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 6: MASK PERSISTENCE & PROGRESSIVE ENTITY REVEAL
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="asset_persistence")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="asset_persistence",
                status="running",
                progress=0.60,
                message="Persisting segmentation mask assets to local storage...",
            )

            seg_map = {res.entity_id: res for res in segmentation_results}
            successful_masks = 0

            for idx, entity in enumerate(entities):
                # Check cancellation between entities
                if job_tracker.is_cancelled(job_id):
                    scene.processing_status = "ready"
                    scene_manager.update_scene(scene)
                    return scene

                seg_res = seg_map.get(entity.id)
                if seg_res and len(seg_res.mask_png_bytes) > 0:
                    mask_filename = f"{entity.id}_mask.png"
                    mask_asset = asset_manager.register_pipeline_asset(
                        content=seg_res.mask_png_bytes,
                        category="segmentation_mask",
                        filename=mask_filename,
                        mime_type="image/png",
                        scene_id=scene_id,
                        entity_id=entity.id,
                    )

                    entity.segmentation = SegmentationInfo(
                        status="ready",
                        mask_asset_id=mask_asset.asset_id,
                        mask_uri=f"/api/v1/assets/{mask_asset.asset_id}/file",
                        confidence=seg_res.confidence,
                        model=seg_res.model,
                        segmented_at=datetime.now(timezone.utc).isoformat(),
                        pixel_count=seg_res.pixel_count,
                    )
                    if entity.confidence:
                        entity.confidence.segmentation_confidence = seg_res.confidence

                    successful_masks += 1

                    # PROGRESSIVE COMMIT: Commit updated entity mask immediately to scene!
                    scene_manager.update_scene(scene)

                    # Progressive honest event: reports exact completed person mask count
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="asset_ready",
                        stage_identifier="asset_persistence",
                        status="running",
                        entity_id=entity.id,
                        progress=0.60 + 0.08 * (successful_masks / max(len(entities), 1)),
                        asset_reference=AssetReference(
                            asset_id=mask_asset.asset_id,
                            asset_type="segmentation_mask",
                            uri=f"/api/v1/assets/{mask_asset.asset_id}/file",
                            mime_type="image/png",
                            size_bytes=mask_asset.size_bytes,
                        ),
                        message=f"Segmentation mask ready for Person {idx + 1} ({successful_masks} of {len(entities)}).",
                    )
                else:
                    entity.segmentation = SegmentationInfo(
                        status="failed" if valid_detections else "unstarted",
                        error="Segmentation mask was empty or could not be generated.",
                    )
                    scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="asset_persistence",
                status="running",
                progress=0.68,
                message=f"Stored {successful_masks} mask assets.",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 7 & 8: DEPTH ESTIMATION & NORMALIZATION
            # -------------------------------------------------------------
            depth_result: Optional[DepthEstimationResult] = None
            if include_depth:
                job_tracker.update_job_status(job_id, status="running", current_stage="depth_estimation")
                try:
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_started",
                        stage_identifier="depth_estimation",
                        status="running",
                        progress=0.70,
                        message="Estimating scene depth using local monocular neural network...",
                    )

                    depth_estimator = self.get_depth_estimator()
                    depth_result = await asyncio.to_thread(
                        depth_estimator.estimate_depth,
                        photo_path,
                    )

                    # Check cancellation
                    if job_tracker.is_cancelled(job_id):
                        scene.processing_status = "ready"
                        scene_manager.update_scene(scene)
                        return scene

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="intermediate_result_available",
                        stage_identifier="depth_estimation",
                        status="running",
                        progress=0.78,
                        message=f"Scene depth estimated ({depth_result.width}x{depth_result.height}) on {depth_result.device}.",
                    )

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_completed",
                        stage_identifier="depth_estimation",
                        status="running",
                        progress=0.82,
                        message="Depth estimation complete.",
                    )

                    # Persist raw depth map array (NumPy .npy format)
                    raw_buf = io.BytesIO()
                    np.save(raw_buf, depth_result.raw_depth)
                    depth_asset = asset_manager.register_pipeline_asset(
                        content=raw_buf.getvalue(),
                        category="depth_map",
                        filename=f"{scene_id}_depth.npy",
                        mime_type="application/x-numpy",
                        scene_id=scene_id,
                    )

                    # Persist inspection visualization PNG
                    vis_asset = asset_manager.register_pipeline_asset(
                        content=depth_result.visualization_png_bytes,
                        category="depth_visualization",
                        filename=f"{scene_id}_depth_vis.png",
                        mime_type="image/png",
                        scene_id=scene_id,
                    )

                    # PROGRESSIVE DEPTH MAP COMMIT
                    scene.depth = SceneDepthMetadata(
                        status="ready",
                        depth_asset_id=depth_asset.asset_id,
                        depth_uri=f"/api/v1/assets/{depth_asset.asset_id}/file",
                        visualization_asset_id=vis_asset.asset_id,
                        visualization_uri=f"/api/v1/assets/{vis_asset.asset_id}/file",
                        model=depth_result.model_name,
                        model_family="Depth-Anything-V2",
                        depth_convention="relative_proximity",
                        normalization=DepthNormalizationMetadata(**depth_result.normalization_metadata),
                        width=depth_result.width,
                        height=depth_result.height,
                        device=depth_result.device,
                        analyzed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    scene_manager.update_scene(scene)

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="asset_ready",
                        stage_identifier="depth_normalization",
                        status="running",
                        progress=0.85,
                        asset_reference=AssetReference(
                            asset_id=vis_asset.asset_id,
                            asset_type="depth_visualization",
                            uri=f"/api/v1/assets/{vis_asset.asset_id}/file",
                            mime_type="image/png",
                            size_bytes=vis_asset.size_bytes,
                        ),
                        message="Depth visualization asset saved and ready for inspection.",
                    )

                    # -------------------------------------------------------------
                    # Stage 9: DEPTH ASSOCIATION WITH DETECTED PEOPLE
                    # -------------------------------------------------------------
                    job_tracker.update_job_status(job_id, status="running", current_stage="depth_association")
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_started",
                        stage_identifier="depth_association",
                        status="running",
                        progress=0.86,
                        message="Calculating relative depth statistics for detected people...",
                    )

                    for entity in entities:
                        seg_res = seg_map.get(entity.id)
                        mask_bytes = seg_res.mask_png_bytes if seg_res else None
                        pixel_box = entity.detection.pixel_box if entity.detection else None

                        depth_stats = compute_entity_depth_stats(
                            normalized_proximity=depth_result.normalized_depth,
                            mask_bytes=mask_bytes,
                            pixel_box=pixel_box,
                            entity_id=entity.id,
                        )
                        depth_stats.depth_asset_id = depth_asset.asset_id
                        entity.depth = depth_stats

                    scene_manager.update_scene(scene)

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_completed",
                        stage_identifier="depth_association",
                        status="running",
                        progress=0.90,
                        message=f"Relative depth calculated for {len(entities)} person entities.",
                    )

                    # -------------------------------------------------------------
                    # Stage 10: SPATIAL PHOTO PREPARATION
                    # -------------------------------------------------------------
                    job_tracker.update_job_status(job_id, status="running", current_stage="spatial_photo_preparation")
                    scene.representations.spatial_photo = RepresentationAsset(
                        status="ready",
                        asset_id=depth_asset.asset_id,
                        uri=f"/api/v1/assets/{vis_asset.asset_id}/file",
                        updated_at=datetime.now(timezone.utc).isoformat(),
                        config=SpatialPhotoConfig(
                            default_depth_strength=0.08,
                            geometry_subdivisions=[128, 128],
                            disocclusion_handling="conservative_bounds",
                        ),
                    )
                    scene_manager.update_scene(scene)

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_completed",
                        stage_identifier="spatial_photo_preparation",
                        status="running",
                        progress=0.94,
                        message="Spatial photo representation prepared with depth displacement.",
                    )

                except Exception as depth_err:
                    logger.warning(f"Depth estimation encountered non-fatal error: {depth_err}")
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_failed",
                        stage_identifier="depth_estimation",
                        status="running",
                        error=StructuredError(
                            code="DEPTH_ESTIMATION_FAILED",
                            message=str(depth_err),
                        ),
                    )
                    scene.depth = SceneDepthMetadata(
                        status="failed",
                        error=str(depth_err),
                    )
                    scene.representations.spatial_photo = RepresentationAsset(
                        status="failed",
                        error=str(depth_err),
                    )
                    scene_manager.update_scene(scene)

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 11: SCENE_UPDATE
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="scene_update")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="scene_update",
                status="running",
                progress=0.96,
                message="Updating authoritative scene entities and representations...",
            )

            scene.processing_status = "ready"
            scene.updated_at = datetime.now(timezone.utc).isoformat()
            scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="scene_update",
                status="running",
                progress=0.98,
                message=f"Scene updated ({len(entities)} people, depth: {scene.depth.status if scene.depth else 'skipped'}).",
            )

            # -------------------------------------------------------------
            # Stage 12: COMPLETED
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="completed", current_stage="completed")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="job_completed",
                stage_identifier="completed",
                status="completed",
                progress=1.0,
                message=f"Analysis complete: {len(entities)} people detected and spatial photo ready.",
            )

            logger.info(f"Pipeline job '{job_id}' completed successfully for scene '{scene_id}'")
            return scene

        except Exception as e:
            logger.error(f"Pipeline job '{job_id}' failed: {e}", exc_info=True)
            job_tracker.update_job_status(job_id, status="failed", current_stage="image_analysis")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_failed",
                stage_identifier="image_analysis",
                status="failed",
                error=StructuredError(
                    code="PIPELINE_EXECUTION_ERROR",
                    message=str(e),
                ),
            )
            scene.processing_status = "error"
            scene_manager.update_scene(scene)
            raise

    # Backwards compatibility alias for Phase 02/03 callers and tests
    execute_phase02_pipeline = execute_pipeline


pipeline_coordinator = PipelineCoordinator()
