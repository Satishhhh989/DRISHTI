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
import cv2
from datetime import datetime, timezone
import inspect
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.assets.manager import asset_manager
from app.core.logging import logger
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.assembly import compute_2d_iou, scene_assembly_engine
from app.pipeline.depth import (
    DepthAnythingV2Estimator,
    compute_entity_depth_stats,
)
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.device import device_manager
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
from app.pipeline.object_reconstructor import object_reconstructor
from app.pipeline.motion import (
    IPoseEstimator,
    MockPoseEstimator,
    PoseEstimationResult,
    YoloPoseEstimator,
    procedural_animation_engine,
    rigging_engine,
    skeleton_fitter,
)
from app.pipeline.reconstructor import (
    HumanReconstructor,
    MockHumanReconstructor,
    SpinHumanReconstructor,
)
from app.pipeline.segmentor import YoloPersonSegmentor
from app.scene.manager import scene_manager
from app.schemas.common import EulerRotation, Vector3D
from app.schemas.events import AssetReference, StructuredError
from app.schemas.scene import (
    AnimationState,
    ConfidenceMetadata,
    CoordinateConvention,
    DepthNormalizationMetadata,
    DetectionInfo,
    GeometryMetadata,
    HumanReconstructionData,
    MeshBounds,
    ModelProvenance,
    RepresentationAsset,
    RigData,
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
        reconstructor: Optional[HumanReconstructor] = None,
    ):
        self._detector = detector
        self._segmentor = segmentor
        self._depth_estimator = depth_estimator
        self._reconstructor = reconstructor

    def get_detector(self, quality_preset: str = "high") -> IPersonDetector:
        if self._detector is not None:
            return self._detector
        if quality_preset in ("high", "maximum") and model_manager.is_model_available("yolov8m-seg"):
            return YoloPersonDetector("yolov8m-seg")
        if model_manager.is_model_available("yolov8n-seg"):
            return YoloPersonDetector("yolov8n-seg")
        logger.warning("YOLO weights not found; falling back to MockPersonDetector")
        return MockPersonDetector()

    def get_segmentor(self, quality_preset: str = "high") -> ISegmentor:
        if self._segmentor is not None:
            return self._segmentor
        if quality_preset in ("high", "maximum") and model_manager.is_model_available("yolov8m-seg"):
            return YoloPersonSegmentor("yolov8m-seg")
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

    def get_reconstructor(self, backend: Optional[str] = None) -> HumanReconstructor:
        if self._reconstructor is not None:
            return self._reconstructor
        from app.pipeline.reconstructor import (
            Hmr2HumanReconstructor,
            MockHumanReconstructor,
            SpinHumanReconstructor,
            reconstruction_registry,
        )

        backend_name = (backend or "hmr2").lower()
        reg_backend = reconstruction_registry.get(backend_name)
        if reg_backend is not None:
            return reg_backend

        device = device_manager.device
        spin_reconstructor = None
        if model_manager.is_model_available("spin-human-reconstruction"):
            ckpt_path = model_manager.get_model_path("spin-human-reconstruction")
            smpl_path = ckpt_path.parent / "SMPL_N_model.npz"
            mean_path = ckpt_path.parent / "smpl_mean_params.npz"
            spin_reconstructor = SpinHumanReconstructor(
                checkpoint_path=ckpt_path,
                smpl_npz_path=smpl_path,
                mean_params_path=mean_path,
                device=device,
            )
            reconstruction_registry.register("spin", spin_reconstructor)

        if backend_name == "hmr2":
            hmr2_path = model_manager.models_dir / "hmr2" / "epoch=35-step=1000000.ckpt"
            if hmr2_path.exists():
                logger.info(f"Initializing HMR 2.0 (4DHumans) as primary reconstructor backend from {hmr2_path} on {device}")
                hmr2_rec = Hmr2HumanReconstructor(
                    checkpoint_path=hmr2_path,
                    device=device,
                    fallback_reconstructor=spin_reconstructor or MockHumanReconstructor(device=device),
                )
                reconstruction_registry.register("hmr2", hmr2_rec)
                return hmr2_rec
            elif spin_reconstructor is not None:
                logger.warning(f"HMR 2.0 checkpoint not found at {hmr2_path}; falling back to SPIN")
                return spin_reconstructor

        if spin_reconstructor is not None:
            return spin_reconstructor

        logger.warning("SPIN/SMPL weights not found; falling back to MockHumanReconstructor")
        mock_rec = MockHumanReconstructor(device=device)
        reconstruction_registry.register("mock", mock_rec)
        return mock_rec

    async def execute_pipeline(
        self,
        scene_id: str,
        job_id: str,
        confidence_threshold: float = 0.35,
        include_depth: bool = True,
        include_reconstruction: bool = True,
        quality_preset: str = "high",
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

            detector = self.get_detector(quality_preset=quality_preset)
            if hasattr(detector, "detect_entities"):
                detections = await asyncio.to_thread(
                    detector.detect_entities,
                    photo_path,
                    confidence_threshold,
                    True,
                )
            else:
                detections = await asyncio.to_thread(
                    detector.detect_people,
                    photo_path,
                    confidence_threshold,
                )

            num_detected = len(detections)
            logger.info(f"Job {job_id}: {num_detected} entity detections produced")

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
            p_idx = 1
            o_idx = 1
            for det in valid_detections:
                ent_type = getattr(det, "entity_type", "person")
                obj_cls = getattr(det, "object_class", None)
                if ent_type == "person":
                    name = f"Person {p_idx}"
                    p_idx += 1
                else:
                    cat_name = (obj_cls or "Object").capitalize()
                    name = f"{cat_name} {o_idx}"
                    o_idx += 1

                entity = SceneEntity(
                    id=det.entity_id,
                    entity_type=ent_type,
                    object_class=obj_cls,
                    source_image_id=photo_asset_id,
                    name=name,
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

                segmentor = self.get_segmentor(quality_preset=quality_preset)
                if hasattr(segmentor, "segment_entities"):
                    segmentation_results = await asyncio.to_thread(
                        segmentor.segment_entities,
                        photo_path,
                        valid_detections,
                    )
                else:
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

            # -------------------------------------------------------------
            # Stage 11: POSE_ESTIMATION (Phase 07: 2D Keypoint Extraction for Fitting)
            # -------------------------------------------------------------
            pose_results_by_entity: Dict[str, PoseEstimationResult] = {}
            if entities:
                # Check cancellation
                if job_tracker.is_cancelled(job_id):
                    scene.processing_status = "ready"
                    scene_manager.update_scene(scene)
                    return scene

                job_tracker.update_job_status(job_id, status="running", current_stage="pose_estimation")
                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_started",
                    stage_identifier="pose_estimation",
                    status="running",
                    progress=0.88,
                    message=f"Estimating 2D human pose keypoints for {len(entities)} entities...",
                )

                try:
                    if quality_preset in ("high", "maximum") and model_manager.is_model_available("yolov8m-pose"):
                        pose_model_path = model_manager.get_model_path("yolov8m-pose")
                    else:
                        pose_model_path = model_manager.get_model_path("yolov8n-pose")

                    if pose_model_path.exists() and not isinstance(self.get_detector(quality_preset), MockPersonDetector):
                        pose_estimator: IPoseEstimator = YoloPoseEstimator(model_path=pose_model_path, device=device_manager.device)
                    else:
                        pose_estimator = MockPoseEstimator()

                    img_for_pose = cv2.imread(str(photo_path)) if photo_path.exists() else None
                    if img_for_pose is None:
                        img_for_pose = np.zeros((scene.source_image.height, scene.source_image.width, 3), dtype=np.uint8)

                    entities_with_boxes = [e for e in entities if e.entity_type == "person" and e.detection and e.detection.bounding_box]
                    boxes = [e.detection.bounding_box for e in entities_with_boxes]
                    all_poses = pose_estimator.estimate_poses(img_for_pose, boxes)

                    if len(all_poses) == len(entities_with_boxes):
                        for e, p in zip(entities_with_boxes, all_poses):
                            pose_results_by_entity[e.id] = p
                    else:
                        for entity in entities_with_boxes:
                            best_iou = 0.0
                            best_pose = None
                            for p in all_poses:
                                iou = compute_2d_iou(entity.detection.bounding_box, p.bounding_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_pose = p
                            if best_pose and best_iou > 0.10:
                                pose_results_by_entity[entity.id] = best_pose

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_completed",
                        stage_identifier="pose_estimation",
                        status="running",
                        progress=0.91,
                        message=f"Pose estimation complete ({len(pose_results_by_entity)} of {len(entities)} people matched).",
                    )
                except Exception as pose_err:
                    logger.warning(f"Pose estimation non-fatal error: {pose_err}")
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_failed",
                        stage_identifier="pose_estimation",
                        status="running",
                        error=StructuredError(code="POSE_ESTIMATION_ERROR", message=str(pose_err)),
                    )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 12: HUMAN 3D RECONSTRUCTION ENGINE (With SMPLify Pose Fitting)
            # -------------------------------------------------------------
            reconstructed_vertices: Dict[str, np.ndarray] = {}
            if include_reconstruction and entities:
                job_tracker.update_job_status(job_id, status="running", current_stage="human_reconstruction")
                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_started",
                    stage_identifier="human_reconstruction",
                    status="running",
                    progress=0.91,
                    message=f"Starting image-conditioned 3D human reconstruction for {len(entities)} detected people...",
                )

                try:
                    reconstructor = self.get_reconstructor()
                    total_people = len(entities)
                    successful_meshes = 0

                    img_bgr = cv2.imread(str(photo_path))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = np.zeros((scene.source_image.height, scene.source_image.width, 3), dtype=np.uint8)

                    for idx, entity in enumerate(entities, start=1):
                        # Cooperative cancellation check between people
                        if job_tracker.is_cancelled(job_id):
                            logger.info(f"Cancellation received during human reconstruction at entity {entity.id}")
                            break

                        try:
                            # Retrieve segmentation mask if available
                            seg_res = seg_map.get(entity.id)
                            mask_bytes = seg_res.mask_png_bytes if seg_res else None
                            depth_info = entity.depth

                            # Object Reconstruction Branch
                            if entity.entity_type == "object":
                                mask_np = None
                                if mask_bytes:
                                    with Image.open(io.BytesIO(mask_bytes)) as m_img:
                                        mask_np = np.array(m_img.convert("L"))
                                if mask_np is None:
                                    mask_np = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
                                    box = entity.detection.bounding_box
                                    x1 = int(box.x_min * img_rgb.shape[1])
                                    y1 = int(box.y_min * img_rgb.shape[0])
                                    x2 = int(box.x_max * img_rgb.shape[1])
                                    y2 = int(box.y_max * img_rgb.shape[0])
                                    mask_np[y1:y2, x1:x2] = 255

                                obj_rec_result = object_reconstructor.reconstruct_object(
                                    image_rgb=img_rgb,
                                    mask_uint8=mask_np,
                                    normalized_depth=depth_result.normalized_depth if depth_result else None,
                                    pixel_box=tuple(entity.detection.pixel_box),
                                    category_name=entity.object_class or "object",
                                    entity_id=entity.id,
                                    aspect_ratio=scene.source_image.aspect_ratio if scene.source_image else 1.333,
                                )

                                # Save GLB asset
                                mesh_filename = f"{scene_id}_{entity.id}.glb"
                                mesh_asset = asset_manager.register_mesh(
                                    content=obj_rec_result.glb_bytes,
                                    filename=mesh_filename,
                                    scene_id=scene_id,
                                    entity_id=entity.id,
                                )

                                # Compute spatial transform for the object
                                box = entity.detection.bounding_box
                                img_h, img_w = img_rgb.shape[:2]
                                aspect = img_w / img_h
                                plane_h = 2.0
                                plane_w = plane_h * aspect

                                cx = (box.x_min + box.x_max) / 2.0
                                cy = (box.y_min + box.y_max) / 2.0
                                pos_x = float((cx - 0.5) * plane_w)
                                pos_y = float((1.0 - cy) * plane_h)

                                if depth_info and depth_info.median_proximity is not None:
                                    prox = float(np.clip(depth_info.median_proximity, 0.0, 1.0))
                                    pos_z = float((prox - 0.5) * 0.4)
                                else:
                                    pos_z = 0.0

                                # Scale factor matching 2D bounding box
                                v_min_y = float(obj_rec_result.bounds_min[1])
                                v_max_y = float(obj_rec_result.bounds_max[1])
                                mesh_h = max(0.05, v_max_y - v_min_y)
                                box_h = (box.y_max - box.y_min) * plane_h
                                scale_val = float(np.clip(box_h / mesh_h, 0.2, 3.0))

                                entity_transform = Transform(
                                    position=Vector3D(x=round(pos_x, 4), y=round(pos_y, 4), z=round(pos_z, 4)),
                                    rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                                    scale=Vector3D(x=round(scale_val, 4), y=round(scale_val, 4), z=round(scale_val, 4)),
                                )
                                entity.transform = entity_transform

                                entity.reconstruction = HumanReconstructionData(
                                    status="ready",
                                    mesh_asset_id=mesh_asset.asset_id,
                                    mesh_url=f"/api/v1/assets/meshes/{mesh_asset.filename}",
                                    metadata_asset_id=None,
                                    model_provenance=ModelProvenance(
                                        name=obj_rec_result.model_name,
                                        version=obj_rec_result.model_version,
                                        device=obj_rec_result.device,
                                    ),
                                    geometry_metadata=GeometryMetadata(
                                        vertex_count=obj_rec_result.vertex_count,
                                        face_count=obj_rec_result.face_count,
                                        bounds=MeshBounds(
                                            min=obj_rec_result.bounds_min,
                                            max=obj_rec_result.bounds_max,
                                        ),
                                        dimensions=obj_rec_result.dimensions,
                                    ),
                                    coordinate_convention=CoordinateConvention(
                                        up="+Y", forward="+Z", right="+X", units="meters"
                                    ),
                                    transform=entity_transform,
                                    texture_availability="full",
                                    quality_score=obj_rec_result.quality_score,
                                    quality_preset=quality_preset,
                                    metric_status="measured",
                                    stance_description=f"Object: {entity.object_class or 'Prop'}",
                                    reconstructed_at=datetime.now(timezone.utc).isoformat(),
                                )
                                entity.geometry_ref = mesh_asset.asset_id
                                successful_meshes += 1
                                scene_manager.update_scene(scene)

                                event_emitter.emit(
                                    job_id=job_id,
                                    scene_id=scene_id,
                                    event_type="asset_ready",
                                    stage_identifier="human_reconstruction",
                                    status="running",
                                    progress=0.91 + (0.03 * (idx / total_people)),
                                    entity_id=entity.id,
                                    asset_reference=AssetReference(
                                        asset_id=mesh_asset.asset_id,
                                        asset_type="mesh",
                                        uri=f"/api/v1/assets/meshes/{mesh_asset.filename}",
                                        mime_type="model/gltf-binary",
                                        size_bytes=mesh_asset.size_bytes,
                                    ),
                                    message=f"3D object mesh ready for {entity.name or entity.id} ({idx} of {total_people}).",
                                )
                                continue

                            # Human Reconstruction Branch
                            matched_pose = pose_results_by_entity.get(entity.id)

                            rec_kwargs: Dict[str, Any] = {
                                "image_rgb": img_rgb,
                                "bounding_box": entity.detection.bounding_box,
                                "mask_gray": mask_bytes,
                                "depth_stats": depth_info.model_dump() if depth_info else None,
                            }
                            sig = inspect.signature(reconstructor.reconstruct)
                            if "pose_result" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                                rec_kwargs["pose_result"] = matched_pose
                            if "quality_preset" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                                rec_kwargs["quality_preset"] = quality_preset

                            rec_result = reconstructor.reconstruct(**rec_kwargs)
                            reconstructed_vertices[entity.id] = rec_result.vertices

                            # Save GLB asset
                            mesh_filename = f"{scene_id}_{entity.id}.glb"
                            mesh_asset = asset_manager.register_mesh(
                                content=rec_result.glb_bytes,
                                filename=mesh_filename,
                                scene_id=scene_id,
                                entity_id=entity.id,
                            )

                            if rec_result.overlay_png_bytes:
                                try:
                                    overlay_filename = f"{scene_id}_{entity.id}_reproj.png"
                                    asset_manager.register_pipeline_asset(
                                        content=rec_result.overlay_png_bytes,
                                        category="depth_visualization",
                                        filename=overlay_filename,
                                        mime_type="image/png",
                                        scene_id=scene_id,
                                        entity_id=entity.id,
                                    )
                                except Exception as ov_err:
                                    logger.debug(f"Failed to register reprojection overlay asset: {ov_err}")

                            # Compute spatial placement in scene coordinates
                            # Canonical scene: +X right, +Y up, +Z forward
                            box = entity.detection.bounding_box
                            img_h, img_w = img_rgb.shape[:2]
                            aspect = img_w / img_h
                            plane_h = 2.0
                            plane_w = plane_h * aspect

                            # 1. Metric scale computation:
                            # Weak-perspective camera scale relates the SMPL 3D metric mesh to the cropped box
                            sq_dim = max(
                                int(max((box.x_max - box.x_min) * img_w, (box.y_max - box.y_min) * img_h) * 1.15),
                                128
                            )
                            if rec_result.camera_params and "scale" in rec_result.camera_params:
                                cam_s = float(rec_result.camera_params["scale"])
                                scale_val = float(np.clip(cam_s * (sq_dim / float(img_h)), 0.45, 1.6))
                            else:
                                v_min_y = float(rec_result.bounds_min[1])
                                v_max_y = float(rec_result.bounds_max[1])
                                mesh_h = max(0.2, v_max_y - v_min_y)
                                box_h = (box.y_max - box.y_min) * plane_h
                                scale_val = float(np.clip(box_h / mesh_h, 0.5, 1.5))

                            # 2. Pelvis-anchored 2D-to-3D spatial placement:
                            pelvis_2d_x = (box.x_min + box.x_max) / 2.0
                            pelvis_2d_y = box.y_min + (box.y_max - box.y_min) * 0.52

                            if matched_pose and hasattr(matched_pose, "keypoints"):
                                kps = matched_pose.keypoints
                                l_hip = kps.get("left_hip")
                                r_hip = kps.get("right_hip")
                                if l_hip and r_hip and l_hip.confidence >= 0.2 and r_hip.confidence >= 0.2:
                                    pelvis_2d_x = (l_hip.x_norm + r_hip.x_norm) / 2.0
                                    pelvis_2d_y = (l_hip.y_norm + r_hip.y_norm) / 2.0
                                elif l_hip and l_hip.confidence >= 0.25:
                                    pelvis_2d_x = l_hip.x_norm
                                    pelvis_2d_y = l_hip.y_norm
                                elif r_hip and r_hip.confidence >= 0.25:
                                    pelvis_2d_x = r_hip.x_norm
                                    pelvis_2d_y = r_hip.y_norm

                            target_pelvis_world_x = float((pelvis_2d_x - 0.5) * plane_w)
                            target_pelvis_world_y = float((1.0 - pelvis_2d_y) * plane_h)

                            pelvis_offset_x = rec_result.pelvis_3d[0] if rec_result.pelvis_3d else 0.0
                            pelvis_offset_y = rec_result.pelvis_3d[1] if rec_result.pelvis_3d else 0.0

                            pos_x = float(target_pelvis_world_x - (scale_val * pelvis_offset_x))
                            pos_y = float(target_pelvis_world_y - (scale_val * pelvis_offset_y))

                            # Stagger along Z based on relative depth proximity in front of photo plane (Z=-0.6)
                            if depth_info and depth_info.median_proximity is not None:
                                prox = float(np.clip(depth_info.median_proximity, 0.0, 1.0))
                                pos_z = float((prox - 0.5) * 0.4)
                            else:
                                pos_z = 0.0

                            entity_transform = Transform(
                                position=Vector3D(x=round(pos_x, 4), y=round(pos_y, 4), z=round(pos_z, 4)),
                                rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                                scale=Vector3D(x=round(scale_val, 4), y=round(scale_val, 4), z=round(scale_val, 4)),
                            )
                            entity.transform = entity_transform

                            entity.reconstruction = HumanReconstructionData(
                                status="ready",
                                mesh_asset_id=mesh_asset.asset_id,
                                mesh_url=f"/api/v1/assets/meshes/{mesh_asset.filename}",
                                metadata_asset_id=None,
                                model_provenance=ModelProvenance(
                                    name=rec_result.model_name,
                                    version=rec_result.model_version,
                                    device=rec_result.device,
                                ),
                                geometry_metadata=GeometryMetadata(
                                    vertex_count=len(rec_result.vertices),
                                    face_count=len(rec_result.faces),
                                    bounds=MeshBounds(
                                        min=rec_result.bounds_min,
                                        max=rec_result.bounds_max,
                                    ),
                                    dimensions=rec_result.dimensions,
                                ),
                                coordinate_convention=CoordinateConvention(
                                    up="+Y", forward="+Z", right="+X", units="meters"
                                ),
                                transform=entity_transform,
                                texture_availability=rec_result.texture_availability,
                                quality_score=rec_result.quality_score,
                                quality_preset=rec_result.quality_preset,
                                reprojection_error_pixels=rec_result.reprojection_error_pixels,
                                reprojection_error_normalized=rec_result.reprojection_error_normalized,
                                silhouette_iou=rec_result.silhouette_iou,
                                stance_description=rec_result.stance_description,
                                observed_keypoints_count=rec_result.observed_keypoints_count,
                                pose_convergence_status=rec_result.pose_convergence_status,
                                metric_status=rec_result.metric_status,
                                warnings=rec_result.warnings,
                                reconstructed_at=datetime.now(timezone.utc).isoformat(),
                            )
                            entity.geometry_ref = mesh_asset.asset_id
                            successful_meshes += 1

                            # Progressive entity update commit
                            scene_manager.update_scene(scene)

                            # Emit asset_ready event for the reconstructed mesh
                            reproj_text = (
                                f"{rec_result.reprojection_error_pixels:.1f}px"
                                if rec_result.reprojection_error_pixels is not None
                                else "N/A"
                            )
                            iou_text = (
                                f"{rec_result.silhouette_iou:.2f}"
                                if rec_result.silhouette_iou is not None
                                else "N/A"
                            )
                            event_emitter.emit(
                                job_id=job_id,
                                scene_id=scene_id,
                                event_type="asset_ready",
                                stage_identifier="human_reconstruction",
                                status="running",
                                progress=0.91 + (0.03 * (idx / total_people)),
                                entity_id=entity.id,
                                asset_reference=AssetReference(
                                    asset_id=mesh_asset.asset_id,
                                    asset_type="mesh",
                                    uri=f"/api/v1/assets/meshes/{mesh_asset.filename}",
                                    mime_type="model/gltf-binary",
                                    size_bytes=mesh_asset.size_bytes,
                                ),
                                message=f"3D human mesh ready for {entity.id} ({idx} of {total_people}) [Reproj Error: {reproj_text}, IoU: {iou_text}].",
                            )

                        except Exception as person_rec_err:
                            logger.warning(f"Reconstruction failed for entity {entity.id}: {person_rec_err}")
                            entity.reconstruction = HumanReconstructionData(
                                status="failed",
                                error_message=str(person_rec_err),
                                texture_availability="none",
                                metric_status="unavailable",
                                reconstructed_at=datetime.now(timezone.utc).isoformat(),
                            )
                            scene_manager.update_scene(scene)

                    if successful_meshes > 0:
                        scene.representations.wireframe = RepresentationAsset(
                            status="ready",
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        scene.representations.solid3d = RepresentationAsset(
                            status="ready",
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        scene.representations.combined = RepresentationAsset(
                            status="ready",
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        scene_manager.update_scene(scene)

                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_completed",
                        stage_identifier="human_reconstruction",
                        status="running",
                        progress=0.94,
                        message=f"Human 3D reconstruction complete ({successful_meshes} of {total_people} meshes generated).",
                    )

                except Exception as rec_err:
                    logger.warning(f"Reconstruction subsystem error: {rec_err}")
                    event_emitter.emit(
                        job_id=job_id,
                        scene_id=scene_id,
                        event_type="stage_failed",
                        stage_identifier="human_reconstruction",
                        status="running",
                        error=StructuredError(
                            code="RECONSTRUCTION_FAILED",
                            message=str(rec_err),
                        ),
                    )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene


            # -------------------------------------------------------------
            # Stage 13: SKELETON_FITTING (Phase 07)
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="skeleton_fitting")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="skeleton_fitting",
                status="running",
                progress=0.93,
                message=f"Fitting 3D skeletons for {len(scene.entities)} entities...",
            )

            successful_skeletons = 0
            for entity in scene.entities:
                if entity.entity_type != "person":
                    continue
                try:
                    matched_pose = pose_results_by_entity.get(entity.id)
                    verts = reconstructed_vertices.get(entity.id) if 'reconstructed_vertices' in locals() else None
                    skeleton = skeleton_fitter.build_skeleton(entity, verts, matched_pose)
                    entity.skeleton = skeleton
                    entity.skeleton_ref = skeleton.id
                    if skeleton.validation_status == "valid":
                        successful_skeletons += 1
                        event_emitter.emit(
                            job_id=job_id,
                            scene_id=scene_id,
                            event_type="asset_ready",
                            stage_identifier="skeleton_fitting",
                            status="running",
                            progress=0.93 + 0.01 * (successful_skeletons / max(1, len(scene.entities))),
                            entity_id=entity.id,
                            asset_reference=AssetReference(
                                asset_id=skeleton.id,
                                asset_type="skeleton",
                                uri=f"/api/v1/scenes/{scene.id}/entities/{entity.id}/skeleton",
                                mime_type="application/json",
                            ),
                            message=f"3D skeleton fitted for {entity.id} ({len(skeleton.joints)} joints).",
                        )
                except Exception as skel_err:
                    logger.warning(f"Skeleton fitting error for entity {entity.id}: {skel_err}")

            scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="skeleton_fitting",
                status="running",
                progress=0.94,
                message=f"Skeleton fitting complete ({successful_skeletons} of {len(scene.entities)} skeletons valid).",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 14: MESH_RIGGING (Phase 07)
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="mesh_rigging")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="mesh_rigging",
                status="running",
                progress=0.94,
                message="Binding reconstructed meshes with skeletal skinning weights...",
            )

            successful_rigs = 0
            for entity in scene.entities:
                if entity.entity_type != "person":
                    continue
                try:
                    if entity.skeleton and entity.reconstruction and entity.reconstruction.status == "ready":
                        verts = reconstructed_vertices.get(entity.id) if 'reconstructed_vertices' in locals() else None
                        rig_data = rigging_engine.rig_mesh(entity, entity.skeleton, verts)
                        entity.rig = rig_data
                        if rig_data.status == "rigged":
                            successful_rigs += 1
                            event_emitter.emit(
                                job_id=job_id,
                                scene_id=scene_id,
                                event_type="asset_ready",
                                stage_identifier="mesh_rigging",
                                status="running",
                                progress=0.94 + 0.01 * (successful_rigs / max(1, len(scene.entities))),
                                entity_id=entity.id,
                                asset_reference=AssetReference(
                                    asset_id=rig_data.id,
                                    asset_type="rig",
                                    uri=f"/api/v1/scenes/{scene.id}/entities/{entity.id}/rig",
                                    mime_type="application/json",
                                ),
                                message=f"Mesh rigging complete for {entity.id} ({rig_data.skinning_method}).",
                            )
                    else:
                        entity.rig = RigData(
                            id=f"rig-{entity.id}",
                            entity_id=entity.id,
                            skeleton_id=entity.skeleton.id if entity.skeleton else f"skel-{entity.id}",
                            status="unrigged",
                            error_message="Reconstructed mesh not available for rigging",
                        )
                except Exception as rig_err:
                    logger.warning(f"Rigging error for entity {entity.id}: {rig_err}")
                    entity.rig = RigData(
                        id=f"rig-{entity.id}",
                        entity_id=entity.id,
                        skeleton_id=entity.skeleton.id if entity.skeleton else f"skel-{entity.id}",
                        status="failed",
                        error_message=str(rig_err),
                    )

            if successful_rigs > 0:
                scene.representations.articulated = RepresentationAsset(
                    status="ready",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
            scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="mesh_rigging",
                status="running",
                progress=0.95,
                message=f"Mesh rigging stage complete ({successful_rigs} of {len(scene.entities)} meshes rigged).",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 15: ANIMATION_PREPARATION (Phase 07)
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="animation_preparation")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="animation_preparation",
                status="running",
                progress=0.95,
                message="Generating procedural humanoid animation clips...",
            )

            prepared_animations = 0
            for entity in scene.entities:
                if entity.entity_type != "person":
                    continue
                try:
                    if entity.rig and entity.rig.status == "rigged" and entity.skeleton:
                        is_seated = bool(
                            entity.reconstruction
                            and "seated" in (entity.reconstruction.stance_description or "").lower()
                        )
                        clips = procedural_animation_engine.generate_clips_for_entity(
                            entity.id, entity.skeleton.id, is_seated=is_seated
                        )
                        entity.animation_clips = clips
                        if not entity.animation_state.current_clip:
                            entity.animation_state.current_clip = "seated_idle" if is_seated else "idle_breathing"
                        prepared_animations += 1
                        event_emitter.emit(
                            job_id=job_id,
                            scene_id=scene_id,
                            event_type="asset_ready",
                            stage_identifier="animation_preparation",
                            status="running",
                            progress=0.95 + 0.01 * (prepared_animations / max(1, len(scene.entities))),
                            entity_id=entity.id,
                            asset_reference=AssetReference(
                                asset_id=f"anim-{entity.id}",
                                asset_type="animation",
                                uri=f"/api/v1/scenes/{scene.id}/entities/{entity.id}/animation",
                                mime_type="application/json",
                            ),
                            message=f"Procedural animation clips ready for {entity.id} ({len(clips)} clips).",
                        )
                    else:
                        entity.animation_clips = []
                except Exception as anim_err:
                    logger.warning(f"Animation preparation error for entity {entity.id}: {anim_err}")

            scene_manager.update_scene(scene)

            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_completed",
                stage_identifier="animation_preparation",
                status="running",
                progress=0.96,
                message=f"Animation preparation complete ({prepared_animations} of {len(scene.entities)} characters animated).",
            )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 16: SCENE_ASSEMBLY
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="scene_assembly")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="scene_assembly",
                status="running",
                progress=0.96,
                message=f"Assembling 3D scene graph for {len(scene.entities)} entities...",
            )

            try:
                assembly_result = scene_assembly_engine.assemble(scene)
                scene.scene_graph = assembly_result.scene_graph
                scene.assembly_report = assembly_result.report
                scene.entities = assembly_result.entities
                scene_manager.update_scene(scene)

                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_completed",
                    stage_identifier="scene_assembly",
                    status="running",
                    progress=0.98,
                    message=(
                        f"Scene assembled ({assembly_result.report.reconstructed_count} meshes placed, "
                        f"status: {assembly_result.report.status})."
                    ),
                )
            except Exception as asm_err:
                logger.warning(f"Scene assembly non-fatal error: {asm_err}")
                event_emitter.emit(
                    job_id=job_id,
                    scene_id=scene_id,
                    event_type="stage_failed",
                    stage_identifier="scene_assembly",
                    status="running",
                    error=StructuredError(
                        code="SCENE_ASSEMBLY_ERROR",
                        message=str(asm_err),
                    ),
                )

            # Check cancellation
            if job_tracker.is_cancelled(job_id):
                scene.processing_status = "ready"
                scene_manager.update_scene(scene)
                return scene

            # -------------------------------------------------------------
            # Stage 13: SCENE_UPDATE
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="running", current_stage="scene_update")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="stage_started",
                stage_identifier="scene_update",
                status="running",
                progress=0.98,
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
                progress=0.99,
                message=f"Scene updated ({len(entities)} people, depth: {scene.depth.status if scene.depth else 'skipped'}).",
            )

            # -------------------------------------------------------------
            # Stage 14: COMPLETED
            # -------------------------------------------------------------
            job_tracker.update_job_status(job_id, status="completed", current_stage="completed")
            event_emitter.emit(
                job_id=job_id,
                scene_id=scene_id,
                event_type="job_completed",
                stage_identifier="completed",
                status="completed",
                progress=1.0,
                message=f"Analysis complete: {len(entities)} people detected and assembled in 3D scene.",
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
