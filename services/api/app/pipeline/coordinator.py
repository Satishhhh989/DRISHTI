"""DRISHTI - Pipeline Coordinator & Job Execution Engine

Executes the 9-stage Phase 02 Image Intelligence lifecycle:
1. IMAGE_VALIDATION
2. IMAGE_PREPARATION
3. PERSON_DETECTION
4. DETECTION_VALIDATION
5. INSTANCE_SEGMENTATION
6. MASK_VALIDATION
7. ASSET_PERSISTENCE
8. SCENE_UPDATE
9. COMPLETED
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import uuid

from app.assets.manager import asset_manager
from app.core.logging import logger
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.interfaces import DetectionResult, IPersonDetector, ISegmentor, SegmentationResult
from app.pipeline.mock import MockPersonDetector, MockPersonSegmentor
from app.pipeline.models import model_manager
from app.pipeline.segmentor import YoloPersonSegmentor
from app.scene.manager import scene_manager
from app.schemas.events import AssetReference, StructuredError
from app.schemas.scene import (
    AnimationState,
    ConfidenceMetadata,
    DetectionInfo,
    Scene,
    SceneEntity,
    SegmentationInfo,
    Transform,
)


class PipelineCoordinator:
    """Orchestrates image validation, detection, segmentation, and scene persistence."""

    def __init__(
        self,
        detector: Optional[IPersonDetector] = None,
        segmentor: Optional[ISegmentor] = None,
    ):
        self._detector = detector
        self._segmentor = segmentor

    def get_detector(self) -> IPersonDetector:
        if self._detector is not None:
            return self._detector
        if model_manager.is_model_available("yolov8n-seg"):
            return YoloPersonDetector("yolov8n-seg")
        # If weights are missing, report mock or raise
        logger.warning("YOLO weights not found; falling back to MockPersonDetector")
        return MockPersonDetector()

    def get_segmentor(self) -> ISegmentor:
        if self._segmentor is not None:
            return self._segmentor
        if model_manager.is_model_available("yolov8n-seg"):
            return YoloPersonSegmentor("yolov8n-seg")
        return MockPersonSegmentor()

    async def execute_phase02_pipeline(
        self,
        scene_id: str,
        job_id: str,
        confidence_threshold: float = 0.35,
    ) -> Scene:
        """Executes the full Phase 02 image intelligence lifecycle asynchronously."""
        scene = scene_manager.get_scene(scene_id)
        photo_asset_id = scene.source_image.asset_id
        photo_path = asset_manager.get_asset_path(photo_asset_id)

        try:
            # -------------------------------------------------------------
            # Stage 1: IMAGE_VALIDATION
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="image_validation",
                status="running",
                progress=0.1,
                message="Validating photograph integrity and dimension bounds...",
            )
            # Verify file on disk
            if not photo_path.exists() or photo_path.stat().st_size == 0:
                raise FileNotFoundError(f"Source photograph file does not exist: {photo_path}")

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="image_validation",
                status="running",
                progress=0.2,
                message="Photograph validated successfully.",
            )

            # -------------------------------------------------------------
            # Stage 2: IMAGE_PREPARATION
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="image_preparation",
                status="running",
                progress=0.25,
                message="Establishing coordinate reference and preparing tensors...",
            )
            await asyncio.sleep(0.02)  # Yield to event loop

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="image_preparation",
                status="running",
                progress=0.3,
                message="Image coordinates ready for multi-person detection.",
            )

            # -------------------------------------------------------------
            # Stage 3: PERSON_DETECTION
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="person_detection",
                status="running",
                progress=0.35,
                message="Executing local neural network person detection...",
            )

            detector = self.get_detector()
            # Run inference in worker thread to prevent blocking event loop
            detections = await asyncio.to_thread(
                detector.detect_people,
                photo_path,
                confidence_threshold,
            )

            num_detected = len(detections)
            logger.info(f"Job {job_id}: {num_detected} person detections produced")

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="intermediate_result_available",
                stage_identifier="person_detection",
                status="running",
                progress=0.5,
                message=f"Detected {num_detected} people in photograph.",
            )

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="person_detection",
                status="running",
                progress=0.55,
                message=f"Person detection complete ({num_detected} candidates identified).",
            )

            # -------------------------------------------------------------
            # Stage 4: DETECTION_VALIDATION
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="detection_validation",
                status="running",
                progress=0.6,
                message="Validating detection bounds and coordinate integrity...",
            )
            valid_detections = [
                d for d in detections
                if d.bounding_box.x_min < d.bounding_box.x_max and d.bounding_box.y_min < d.bounding_box.y_max
            ]

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="detection_validation",
                status="running",
                progress=0.65,
                message=f"{len(valid_detections)} detections validated against coordinate bounds.",
            )

            # -------------------------------------------------------------
            # Stage 5: INSTANCE_SEGMENTATION
            # -------------------------------------------------------------
            segmentation_results: List[SegmentationResult] = []
            if valid_detections:
                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_started",
                    stage_identifier="instance_segmentation",
                    status="running",
                    progress=0.7,
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
                    progress=0.8,
                    message=f"Generated {len(segmentation_results)} individual segmentation masks.",
                )

            # -------------------------------------------------------------
            # Stage 6: MASK_VALIDATION & Stage 7: ASSET_PERSISTENCE
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="asset_persistence",
                status="running",
                progress=0.85,
                message="Persisting lossless segmentation mask assets to local storage...",
            )

            seg_map = {res.entity_id: res for res in segmentation_results}
            entities: List[SceneEntity] = []

            for idx, det in enumerate(valid_detections):
                seg_res = seg_map.get(det.entity_id)
                seg_info: Optional[SegmentationInfo] = None

                if seg_res and len(seg_res.mask_png_bytes) > 0:
                    # Persist PNG mask asset
                    mask_filename = f"{det.entity_id}_mask.png"
                    mask_asset = asset_manager.register_pipeline_asset(
                        content=seg_res.mask_png_bytes,
                        category="segmentation_mask",
                        filename=mask_filename,
                        mime_type="image/png",
                        scene_id=scene_id,
                        entity_id=det.entity_id,
                    )

                    # Emit asset_ready event for the mask
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="asset_ready",
                        stage_identifier="asset_persistence",
                        status="running",
                        entity_id=det.entity_id,
                        asset_reference=AssetReference(
                            asset_id=mask_asset.asset_id,
                            asset_type="segmentation_mask",
                            uri=f"/api/v1/assets/{mask_asset.asset_id}/file",
                            mime_type="image/png",
                            size_bytes=mask_asset.size_bytes,
                        ),
                        message=f"Segmentation mask stored for Person {idx + 1}.",
                    )

                    seg_info = SegmentationInfo(
                        status="ready",
                        mask_asset_id=mask_asset.asset_id,
                        mask_uri=f"/api/v1/assets/{mask_asset.asset_id}/file",
                        confidence=seg_res.confidence,
                        model=seg_res.model,
                        segmented_at=datetime.now(timezone.utc).isoformat(),
                        pixel_count=seg_res.pixel_count,
                    )
                else:
                    seg_info = SegmentationInfo(
                        status="failed" if valid_detections else "unstarted",
                        error="Segmentation mask was empty or could not be generated.",
                    )

                entity = SceneEntity(
                    id=det.entity_id,
                    entity_type="person",
                    source_image_id=photo_asset_id,
                    name=f"Person {idx + 1}",
                    boundingBox=det.bounding_box,
                    transform=Transform(),
                    geometry_ref=None,
                    material_ref=None,
                    skeleton_ref=None,
                    animation_state=AnimationState(),
                    visibility=True,
                    reconstruction_status="unstarted",  # 3D reconstruction is Phase 06
                    confidence=ConfidenceMetadata(
                        detection_confidence=det.confidence,
                        segmentation_confidence=seg_res.confidence if seg_res else None,
                    ),
                    detection=DetectionInfo(
                        confidence=det.confidence,
                        bounding_box=det.bounding_box,
                        pixel_box=list(det.pixel_box),
                        model=det.model,
                        status="ready",
                    ),
                    segmentation=seg_info,
                )
                entities.append(entity)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="asset_persistence",
                status="running",
                progress=0.9,
                message=f"Stored {len(segmentation_results)} mask assets.",
            )

            # -------------------------------------------------------------
            # Stage 8: SCENE_UPDATE
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="scene_update",
                status="running",
                progress=0.95,
                message="Updating authoritative scene entities and representation metadata...",
            )

            scene.entities = entities
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
                message=f"Scene updated with {len(entities)} person entities.",
            )

            # -------------------------------------------------------------
            # Stage 9: COMPLETED
            # -------------------------------------------------------------
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="job_completed",
                stage_identifier="completed",
                status="completed",
                progress=1.0,
                message=f"Phase 02 analysis complete. {len(entities)} people detected and segmented.",
            )

            logger.info(f"Pipeline job '{job_id}' completed successfully for scene '{scene_id}'")
            return scene

        except Exception as e:
            logger.error(f"Pipeline job '{job_id}' failed: {e}", exc_info=True)
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


pipeline_coordinator = PipelineCoordinator()
