"use client";

import React from "react";
import { getRepresentationEntry } from "@photo-to-3d/scene-schema";
import { useSceneStore } from "@/stores/scene-store";
import { useProjectStore } from "@/stores/project-store";
import { useUIStore } from "@/stores/ui-store";
import { apiClient } from "@/lib/api-client";
import styles from "./ContextualPanel.module.css";

const PALETTE_CSS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

export const ContextualPanel: React.FC = () => {
  const currentScene = useSceneStore((state) => state.currentScene);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const currentStageMessage = useSceneStore((state) => state.currentStageMessage);
  const stages = useSceneStore((state) => state.stages);
  const partialFailures = useSceneStore((state) => state.partialFailures);
  const depthStrength = useSceneStore((state) => state.depthStrength);
  const setDepthStrength = useSceneStore((state) => state.setDepthStrength);
  const setAnalysisState = useSceneStore((state) => state.setAnalysisState);
  const applyEvent = useSceneStore((state) => state.applyEvent);
  const cancelActiveJob = useSceneStore((state) => state.cancelActiveJob);
  const reconcileAuthoritativeScene = useSceneStore(
    (state) => state.reconcileAuthoritativeScene
  );
  const toggleEntityVisibility = useSceneStore((state) => state.toggleEntityVisibility);
  const updateEntityTransform = useSceneStore((state) => state.updateEntityTransform);
  const resetEntityTransform = useSceneStore((state) => state.resetEntityTransform);
  const setEntityAnimation = useSceneStore((state) => state.setEntityAnimation);
  const toggleEntitySkeleton = useSceneStore((state) => state.toggleEntitySkeleton);
  const resetEntityPose = useSceneStore((state) => state.resetEntityPose);

  const markDirty = useProjectStore((state) => state.markDirty);

  const isInspectorOpen = useUIStore((state) => state.isInspectorOpen);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const selectEntity = useUIStore((state) => state.selectEntity);
  const resetCamera = useUIStore((state) => state.resetCamera);
  const focusEntity = useUIStore((state) => state.focusEntity);
  const setError = useUIStore((state) => state.setError);
  const setStatusNotice = useUIStore((state) => state.setStatusNotice);

  if (!isInspectorOpen) return null;

  const entities = currentScene?.entities || [];
  const selectedEntity = entities.find((e) => e.id === selectedEntityId);

  const handleRunAnalysis = async () => {
    if (!currentScene) return;

    try {
      setAnalysisState(true, "Queuing progressive analysis pipeline...", 0.05);
      setError(null);

      const job = await apiClient.triggerAnalysis(currentScene.id);
      setAnalysisState(true, "Job accepted by engine", 0.08, job.job_id);
      setStatusNotice("Analyzing photograph (Detection, Segmentation, Depth)...");

      apiClient.subscribeToJobEvents(
        job.job_id,
        0,
        (ev) => {
          applyEvent(ev);
          if (ev.message) {
            setStatusNotice(ev.message);
          }
        },
        async () => {
          try {
            const refreshed = await apiClient.getScene(currentScene.id);
            reconcileAuthoritativeScene(refreshed);
            setStatusNotice(
              `Analysis complete: ${refreshed.entities.length} people identified, spatial depth ready.`
            );
          } catch {
            // Ignore snapshot error
          }
        },
        (err) => {
          console.warn("Event stream error:", err);
        }
      );
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setError(msg);
      setAnalysisState(false, null, 0);
    }
  };

  const handlePositionChange = (axis: "x" | "y" | "z", delta: number) => {
    if (!selectedEntity) return;
    const currentPos = selectedEntity.transform?.position || { x: 0, y: 0, z: -1.5 };
    const currentVal = currentPos[axis] ?? 0;
    const newVal = Math.round((currentVal + delta) * 100) / 100;
    updateEntityTransform(selectedEntity.id, {
      ...selectedEntity.transform,
      position: {
        ...currentPos,
        [axis]: newVal,
      },
    });
    markDirty();
  };

  const handleRotationChange = (axis: "x" | "y" | "z", deltaDeg: number) => {
    if (!selectedEntity) return;
    const currentRot = selectedEntity.transform?.rotation || { x: 0, y: 0, z: 0 };
    const currentRad = currentRot[axis] ?? 0;
    const currentDeg = (currentRad * 180) / Math.PI;
    const newRad = ((currentDeg + deltaDeg) * Math.PI) / 180;
    updateEntityTransform(selectedEntity.id, {
      ...selectedEntity.transform,
      rotation: {
        ...currentRot,
        [axis]: Math.round(newRad * 1000) / 1000,
      },
    });
    markDirty();
  };

  const handleScaleChange = (delta: number) => {
    if (!selectedEntity) return;
    const currentScale = selectedEntity.transform?.scale || { x: 1, y: 1, z: 1 };
    const currentVal = currentScale.x ?? 1;
    const newVal = Math.max(0.1, Math.min(5.0, Math.round((currentVal + delta) * 10) / 10));
    updateEntityTransform(selectedEntity.id, {
      ...selectedEntity.transform,
      scale: { x: newVal, y: newVal, z: newVal },
    });
    markDirty();
  };

  const handleResetTransform = () => {
    if (!selectedEntity) return;
    resetEntityTransform(selectedEntity.id);
    markDirty();
    setStatusNotice(`Reset transform for ${selectedEntity.name || selectedEntity.id}.`);
  };

  const handleExportEntityMesh = () => {
    if (!currentScene || !selectedEntity) return;
    if (!selectedEntity.reconstruction || selectedEntity.reconstruction.status !== "ready") return;
    const url = apiClient.getMeshExportUrl(currentScene.id, selectedEntity.id, true);
    window.open(url, "_blank");
    setStatusNotice(`Exported 3D mesh for ${selectedEntity.name || selectedEntity.id} (GLB).`);
  };

  // Compute honest stage metrics
  const readyMasksCount = entities.filter(
    (e) => e.segmentation?.status === "ready"
  ).length;

  const readyMeshesCount = entities.filter(
    (e) => e.reconstruction?.status === "ready"
  ).length;

  const readySkeletonsCount = entities.filter(
    (e) => e.skeleton?.validationStatus === "valid"
  ).length;

  const readyRigsCount = entities.filter(
    (e) => e.rig?.status === "rigged"
  ).length;

  const readyAnimationsCount = entities.filter(
    (e) => (e.animationClips?.length ?? 0) > 0
  ).length;

  return (
    <aside className={styles.panel} data-testid="contextual-panel">
      {/* Progressive Processing Inspector (Phase 04) */}
      <section className={styles.section} data-testid="pipeline-inspector">
        <div className={styles.sectionHeader}>
          <span>Pipeline Inspector</span>
          <span
            className={`${styles.statusPill} ${
              isAnalyzing ? styles.running : currentScene?.processingStatus || "unstarted"
            }`}
          >
            {isAnalyzing ? "Running" : currentScene?.processingStatus || "Idle"}
          </span>
        </div>

        <div className={styles.stageList}>
          {/* Stage: Validation */}
          <div className={styles.stageRow} data-testid="stage-row-validation">
            <span className={styles.stageName}>Image Validation</span>
            <span
              className={`${styles.stageStatus} ${
                styles[stages.image_validation?.status || (currentScene ? "completed" : "pending")]
              }`}
            >
              {stages.image_validation?.status === "completed" || currentScene
                ? "Completed"
                : stages.image_validation?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Person Detection */}
          <div className={styles.stageRow} data-testid="stage-row-detection">
            <span className={styles.stageName}>Person Detection</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.person_detection?.status ||
                  (entities.length > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${entities.length} found`
                : stages.person_detection?.status === "running"
                ? "Detecting..."
                : stages.person_detection?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Instance Segmentation */}
          <div className={styles.stageRow} data-testid="stage-row-segmentation">
            <span className={styles.stageName}>Segmentation</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.instance_segmentation?.status ||
                  (readyMasksCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${readyMasksCount} of ${entities.length} masks`
                : stages.instance_segmentation?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Depth Estimation */}
          <div className={styles.stageRow} data-testid="stage-row-depth">
            <span className={styles.stageName}>Scene Depth</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.depth_estimation?.status ||
                  (currentScene?.depth?.status === "ready"
                    ? "completed"
                    : currentScene?.depth?.status === "failed"
                    ? "failed"
                    : isAnalyzing
                    ? "running"
                    : "pending")
                ]
              }`}
            >
              {currentScene?.depth?.status === "ready"
                ? "Ready"
                : currentScene?.depth?.status === "failed"
                ? "Failed"
                : stages.depth_estimation?.status === "running"
                ? "Estimating..."
                : "Pending"}
            </span>
          </div>

          {/* Stage: Spatial Photo */}
          <div className={styles.stageRow} data-testid="stage-row-spatial">
            <span className={styles.stageName}>Spatial Photo</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  currentScene?.representations?.spatial_photo?.status === "ready"
                    ? "completed"
                    : "pending"
                ]
              }`}
            >
              {currentScene?.representations?.spatial_photo?.status === "ready"
                ? "Ready"
                : "Unavailable"}
            </span>
          </div>

          {/* Stage: 3D Human Mesh Reconstruction */}
          <div className={styles.stageRow} data-testid="stage-row-reconstruction">
            <span className={styles.stageName}>3D Human Mesh</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.human_reconstruction?.status ||
                  (readyMeshesCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${readyMeshesCount} of ${entities.length} meshes`
                : stages.human_reconstruction?.status || "Pending"}
            </span>
          </div>

          {/* Stage: 2D Pose Estimation (Phase 07) */}
          <div className={styles.stageRow} data-testid="stage-row-pose">
            <span className={styles.stageName}>Pose Estimation</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.pose_estimation?.status ||
                  (readySkeletonsCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {stages.pose_estimation?.status === "completed"
                ? "Keypoints Detected"
                : stages.pose_estimation?.status === "running"
                ? "Running..."
                : readySkeletonsCount > 0
                ? "Completed"
                : stages.pose_estimation?.status || "Pending"}
            </span>
          </div>

          {/* Stage: 3D Skeleton Fitting (Phase 07) */}
          <div className={styles.stageRow} data-testid="stage-row-skeleton">
            <span className={styles.stageName}>Skeleton Fitting</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.skeleton_fitting?.status ||
                  (readySkeletonsCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${readySkeletonsCount} of ${entities.length} skeletons`
                : stages.skeleton_fitting?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Mesh Rigging (Phase 07) */}
          <div className={styles.stageRow} data-testid="stage-row-rigging">
            <span className={styles.stageName}>Mesh Rigging</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.mesh_rigging?.status ||
                  (readyRigsCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${readyRigsCount} of ${entities.length} rigged`
                : stages.mesh_rigging?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Procedural Motion Preparation (Phase 07) */}
          <div className={styles.stageRow} data-testid="stage-row-animation">
            <span className={styles.stageName}>Procedural Motion</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.animation_preparation?.status ||
                  (readyAnimationsCount > 0 ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {entities.length > 0
                ? `${readyAnimationsCount} of ${entities.length} animated`
                : stages.animation_preparation?.status || "Pending"}
            </span>
          </div>

          {/* Stage: Multi-Person Scene Assembly (Phase 06) */}
          <div className={styles.stageRow} data-testid="stage-row-assembly">
            <span className={styles.stageName}>Scene Assembly</span>
            <span
              className={`${styles.stageStatus} ${
                styles[
                  stages.scene_assembly?.status ||
                  (currentScene?.sceneGraph ? "completed" : isAnalyzing ? "running" : "pending")
                ]
              }`}
            >
              {currentScene?.assemblyReport?.status === "completed"
                ? "Completed"
                : currentScene?.assemblyReport?.status === "partial"
                ? "Partial Scene"
                : stages.scene_assembly?.status === "running"
                ? "Assembling..."
                : currentScene?.sceneGraph
                ? "Completed"
                : stages.scene_assembly?.status || "Pending"}
            </span>
          </div>
        </div>

        {/* Cancellation Button */}
        {isAnalyzing && (
          <button
            type="button"
            className={styles.cancelButton}
            onClick={cancelActiveJob}
            data-testid="cancel-job-btn"
          >
            Cancel Analysis
          </button>
        )}

        {/* Partial Failures Warning */}
        {partialFailures.length > 0 && (
          <div className={styles.partialWarning}>
            {partialFailures.map((pf, idx) => (
              <div key={idx}>
                {pf.stage}: {pf.message}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Scene Metadata */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>Scene Summary</div>
        {currentScene ? (
          <>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Scene ID</span>
              <span className={styles.propValue} title={currentScene.id}>
                {currentScene.id.slice(0, 10)}...
              </span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Schema Ver</span>
              <span className={styles.propValue}>{currentScene.schemaVersion}</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Status</span>
              <span className={`${styles.statusPill} ${styles[currentScene.processingStatus]}`}>
                {currentScene.processingStatus}
              </span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Active Mode</span>
              <span className={styles.propValue} data-testid="active-representation-name">
                {getRepresentationEntry(visualizationMode).displayName}
              </span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Mode Category</span>
              <span className={styles.propValue} style={{ textTransform: "capitalize", fontSize: "10px" }}>
                {getRepresentationEntry(visualizationMode).category.replace("_", " ")}
              </span>
            </div>
            {getRepresentationEntry(visualizationMode).knownLimitations.length > 0 && (
              <div
                style={{ marginTop: "4px", fontSize: "9px", color: "#71717a", lineHeight: "1.3" }}
                data-testid="active-representation-limitation"
              >
                ℹ️ {getRepresentationEntry(visualizationMode).knownLimitations[0]}
              </div>
            )}
          </>
        ) : (
          <div className={styles.emptyNotice}>No active scene loaded.</div>
        )}
      </section>

      {/* Spatial Photography & Depth Controls */}
      <section className={styles.section} data-testid="spatial-photo-controls">
        <div className={styles.sectionHeader}>Spatial Photography</div>
        {currentScene?.depth?.status === "ready" ? (
          <>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Depth Model</span>
              <span className={styles.propValue}>{currentScene.depth.model}</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Convention</span>
              <span className={styles.propValue}>Relative Proximity</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Device</span>
              <span className={styles.propValue}>{currentScene.depth.device.toUpperCase()}</span>
            </div>

            {/* Depth Strength Slider */}
            <div className={styles.sliderRow}>
              <div className={styles.sliderHeader}>
                <span className={styles.propLabel}>Depth Strength</span>
                <span className={styles.propValue}>{(depthStrength * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                className={styles.sliderInput}
                min="0.0"
                max="0.25"
                step="0.01"
                value={depthStrength}
                onChange={(e) => setDepthStrength(parseFloat(e.target.value))}
                data-testid="depth-strength-slider"
              />
            </div>

            {/* Camera Reset */}
            <button
              type="button"
              className={styles.actionButton}
              onClick={resetCamera}
              data-testid="reset-camera-spatial-btn"
            >
              Reset Camera to Center
            </button>
          </>
        ) : currentScene?.depth?.status === "failed" ? (
          <div className={styles.emptyNotice}>
            Depth estimation failed: {currentScene.depth.error}
          </div>
        ) : (
          <div className={styles.emptyNotice}>
            {currentScene
              ? "Run analysis to generate depth-aware spatial photograph."
              : "Upload a photograph to enable depth estimation."}
          </div>
        )}
      </section>

      {/* Image Geometry */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>Source Geometry</div>
        {photoDimensions ? (
          <>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Resolution</span>
              <span className={styles.propValue}>
                {photoDimensions.width} × {photoDimensions.height}
              </span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Aspect Ratio</span>
              <span className={styles.propValue}>{photoDimensions.aspect} : 1</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Coordinate Ref</span>
              <span className={styles.propValue}>Right-Handed Y-Up</span>
            </div>
          </>
        ) : (
          <div className={styles.emptyNotice}>Upload a photograph to establish scene coordinates.</div>
        )}
      </section>

      {/* Detected People List */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <span>Detected People</span>
          <span className={styles.propValue}>{entities.length}</span>
        </div>

        {entities.length > 0 ? (
          <div className={styles.personList} data-testid="person-entities-list">
            {entities.map((entity, idx) => {
              const isSelected = entity.id === selectedEntityId;
              const colorCss = PALETTE_CSS[idx % PALETTE_CSS.length];
              const conf = entity.detection
                ? `${Math.round(entity.detection.confidence * 100)}%`
                : "—";

              return (
                <div
                  key={entity.id}
                  className={`${styles.personCard} ${isSelected ? styles.selected : ""}`}
                  onClick={() => selectEntity(isSelected ? null : entity.id)}
                  data-testid={`person-card-${entity.id}`}
                  role="button"
                  tabIndex={0}
                  aria-pressed={isSelected}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      selectEntity(isSelected ? null : entity.id);
                    }
                  }}
                >
                  <div className={styles.personCardLeft}>
                    <button
                      type="button"
                      className={`${styles.visibilityBtn} ${entity.visibility === false ? styles.hidden : ""}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleEntityVisibility(entity.id);
                      }}
                      title={entity.visibility === false ? "Show entity in 3D scene" : "Hide entity in 3D scene"}
                      data-testid={`toggle-visibility-${entity.id}`}
                    >
                      {entity.visibility === false ? "👁️‍🗨️" : "👁️"}
                    </button>
                    <span
                      className={styles.colorDot}
                      style={{ backgroundColor: colorCss }}
                    />
                    <span className={styles.personName}>{entity.name || `Person ${idx + 1}`}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span
                      className={`${styles.tagBadge} ${
                        entity.reconstruction?.status === "ready" ? styles.meshReady : styles.unrec
                      }`}
                    >
                      {entity.reconstruction?.status === "ready" ? "Mesh" : "In-Scene"}
                    </span>
                    <span className={styles.confidenceBadge}>{conf}</span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className={styles.emptyNotice} data-testid="zero-person-notice">
            {currentScene
              ? "No people detected in scene yet."
              : "Upload a photograph to detect people."}
          </div>
        )}

        {/* Manual Trigger Analysis Button */}
        {currentScene && !isAnalyzing && (
          <button
            type="button"
            className={styles.actionButton}
            onClick={handleRunAnalysis}
            data-testid="run-analysis-btn"
          >
            {entities.length > 0 ? "Re-run Spatial Analysis" : "Analyze Photograph"}
          </button>
        )}
      </section>

      {/* Selected Entity Details */}
      <section className={styles.section} data-testid="entity-details-section">
        <div className={styles.sectionHeader}>Entity Inspector</div>
        {selectedEntity ? (
          <div className={styles.entityDetails}>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Entity ID</span>
              <span className={styles.propValue}>{selectedEntity.id}</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Type</span>
              <span className={styles.propValue}>{selectedEntity.entityType}</span>
            </div>
            {selectedEntity.detection && (
              <>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Detection Conf</span>
                  <span className={styles.propValue}>
                    {(selectedEntity.detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {(selectedEntity.detection.pixelBox ||
                  (selectedEntity.detection as unknown as Record<string, number[]>).pixel_box) && (
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Pixel Bounds</span>
                    <span className={styles.propValue} style={{ fontSize: "10px" }}>
                      [
                      {(
                        selectedEntity.detection.pixelBox ||
                        (selectedEntity.detection as unknown as Record<string, number[]>).pixel_box
                      )
                        .map((v) => Math.round(v))
                        .join(", ")}
                      ]
                    </span>
                  </div>
                )}
              </>
            )}

            {/* Depth info if available */}
            {selectedEntity.depth && (
              <>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Relative Proximity</span>
                  <span className={styles.propValue}>
                    {(selectedEntity.depth.medianProximity * 100).toFixed(1)}%
                  </span>
                </div>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Depth Method</span>
                  <span className={styles.propValue} style={{ fontSize: "10px" }}>
                    {selectedEntity.depth.method === "segmentation_mask" ? "Mask" : "Bounding Box"}
                  </span>
                </div>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Pixel Coverage</span>
                  <span className={styles.propValue}>
                    {(selectedEntity.depth.validPixelCoverage * 100).toFixed(1)}%
                  </span>
                </div>
              </>
            )}

            {selectedEntity.segmentation && (
              <>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Mask Status</span>
                  <span className={`${styles.statusPill} ${styles[selectedEntity.segmentation.status]}`}>
                    {selectedEntity.segmentation.status}
                  </span>
                </div>
                {(selectedEntity.segmentation.pixelCount !== undefined ||
                  (selectedEntity.segmentation as unknown as Record<string, number>).pixel_count !== undefined) && (
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Mask Pixels</span>
                    <span className={styles.propValue}>
                      {(
                        (selectedEntity.segmentation.pixelCount ??
                          (selectedEntity.segmentation as unknown as Record<string, number>).pixel_count) ||
                        0
                      ).toLocaleString()}{" "}
                      px
                    </span>
                  </div>
                )}
              </>
            )}
            {/* 3D Human Reconstruction info */}
            {selectedEntity.reconstruction ? (
              <>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>3D Mesh Status</span>
                  <span
                    className={`${styles.statusPill} ${
                      styles[selectedEntity.reconstruction.status] || ""
                    }`}
                    style={{ textTransform: "capitalize" }}
                    data-testid="reconstruction-status"
                  >
                    {selectedEntity.reconstruction.status}
                  </span>
                </div>
                {selectedEntity.reconstruction.geometryMetadata && (
                  <>
                    <div className={styles.propRow}>
                      <span className={styles.propLabel}>Vertices</span>
                      <span className={styles.propValue} data-testid="reconstruction-vertex-count">
                        {selectedEntity.reconstruction.geometryMetadata.vertexCount.toLocaleString()}
                      </span>
                    </div>
                    <div className={styles.propRow}>
                      <span className={styles.propLabel}>Triangles</span>
                      <span className={styles.propValue} data-testid="reconstruction-face-count">
                        {selectedEntity.reconstruction.geometryMetadata.faceCount.toLocaleString()}
                      </span>
                    </div>
                    <div className={styles.propRow}>
                      <span className={styles.propLabel}>Dimensions</span>
                      <span className={styles.propValue} style={{ fontSize: "10px" }}>
                        {selectedEntity.reconstruction.geometryMetadata.dimensions
                          .map((d) => `${d.toFixed(2)}m`)
                          .join(" × ")}
                      </span>
                    </div>
                  </>
                )}
                {selectedEntity.reconstruction.modelProvenance && (
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Model</span>
                    <span className={styles.propValue} style={{ fontSize: "10px" }}>
                      {selectedEntity.reconstruction.modelProvenance.name} (
                      {selectedEntity.reconstruction.modelProvenance.device})
                    </span>
                  </div>
                )}
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Texture</span>
                  <span className={styles.propValue} style={{ fontSize: "10px" }}>
                    Neutral Clay (Geometry only)
                  </span>
                </div>
              </>
            ) : (
              <div className={styles.propRow}>
                <span className={styles.propLabel}>3D Mesh</span>
                <span className={styles.statusPill}>Not Reconstructed</span>
              </div>
            )}

            {/* 3D Scene Placement & Interactive Transforms (Phase 06 & 09) */}
            <div style={{ marginTop: "10px", paddingTop: "8px", borderTop: "1px solid #27272e" }} data-testid="transform-controls-section">
              <div className={styles.sectionHeader} style={{ marginBottom: "6px" }}>
                <span>Transform Controls</span>
                <span className={styles.propValue} style={{ fontSize: "10px" }}>Metric</span>
              </div>

              {selectedEntity.transform && (
                <div className={styles.transformGroup}>
                  {/* Position (m) */}
                  <div style={{ fontSize: "10px", color: "#71717a", fontWeight: 600, marginBottom: "2px" }}>
                    POSITION (M)
                  </div>
                  <div className={styles.transformRow}>
                    <span className={`${styles.axisLabel} ${styles.x}`}>X</span>
                    <div className={styles.axisControls}>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("x", -0.1)}
                        title="Move left 0.1m"
                        data-testid="step-x-dec"
                      >
                        -
                      </button>
                      <input
                        type="number"
                        step="0.05"
                        className={styles.axisInput}
                        value={(selectedEntity.transform.position.x ?? 0).toFixed(2)}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) {
                            updateEntityTransform(selectedEntity.id, {
                              ...selectedEntity.transform,
                              position: { ...selectedEntity.transform.position, x: val },
                            });
                            markDirty();
                          }
                        }}
                        data-testid="input-pos-x"
                      />
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("x", 0.1)}
                        title="Move right 0.1m"
                        data-testid="step-x-inc"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  <div className={styles.transformRow}>
                    <span className={`${styles.axisLabel} ${styles.y}`}>Y</span>
                    <div className={styles.axisControls}>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("y", -0.1)}
                        title="Move down 0.1m"
                        data-testid="step-y-dec"
                      >
                        -
                      </button>
                      <input
                        type="number"
                        step="0.05"
                        className={styles.axisInput}
                        value={(selectedEntity.transform.position.y ?? 0).toFixed(2)}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) {
                            updateEntityTransform(selectedEntity.id, {
                              ...selectedEntity.transform,
                              position: { ...selectedEntity.transform.position, y: val },
                            });
                            markDirty();
                          }
                        }}
                        data-testid="input-pos-y"
                      />
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("y", 0.1)}
                        title="Move up 0.1m"
                        data-testid="step-y-inc"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  <div className={styles.transformRow}>
                    <span className={`${styles.axisLabel} ${styles.z}`}>Z</span>
                    <div className={styles.axisControls}>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("z", -0.1)}
                        title="Move deeper 0.1m"
                        data-testid="step-z-dec"
                      >
                        -
                      </button>
                      <input
                        type="number"
                        step="0.05"
                        className={styles.axisInput}
                        value={(selectedEntity.transform.position.z ?? 0).toFixed(2)}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) {
                            updateEntityTransform(selectedEntity.id, {
                              ...selectedEntity.transform,
                              position: { ...selectedEntity.transform.position, z: val },
                            });
                            markDirty();
                          }
                        }}
                        data-testid="input-pos-z"
                      />
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handlePositionChange("z", 0.1)}
                        title="Move closer 0.1m"
                        data-testid="step-z-inc"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  {/* Rotation Euler Y (yaw) */}
                  <div style={{ fontSize: "10px", color: "#71717a", fontWeight: 600, marginTop: "6px", marginBottom: "2px" }}>
                    ROTATION Y (DEG)
                  </div>
                  <div className={styles.transformRow}>
                    <span className={`${styles.axisLabel} ${styles.y}`}>Y</span>
                    <div className={styles.axisControls}>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handleRotationChange("y", -15)}
                        title="Rotate -15°"
                        data-testid="step-rot-y-dec"
                      >
                        -
                      </button>
                      <span className={styles.axisInput} style={{ textAlign: "center" }} data-testid="value-rot-y">
                        {Math.round(((selectedEntity.transform.rotation?.y ?? 0) * 180) / Math.PI)}°
                      </span>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handleRotationChange("y", 15)}
                        title="Rotate +15°"
                        data-testid="step-rot-y-inc"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  {/* Scale */}
                  <div style={{ fontSize: "10px", color: "#71717a", fontWeight: 600, marginTop: "6px", marginBottom: "2px" }}>
                    UNIFORM SCALE
                  </div>
                  <div className={styles.transformRow}>
                    <span className={styles.axisLabel}>S</span>
                    <div className={styles.axisControls}>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handleScaleChange(-0.1)}
                        title="Scale down 10%"
                        data-testid="step-scale-dec"
                      >
                        -
                      </button>
                      <span className={styles.axisInput} style={{ textAlign: "center" }} data-testid="value-scale">
                        {(selectedEntity.transform.scale?.x ?? 1).toFixed(2)}x
                      </span>
                      <button
                        type="button"
                        className={styles.stepBtn}
                        onClick={() => handleScaleChange(0.1)}
                        title="Scale up 10%"
                        data-testid="step-scale-inc"
                      >
                        +
                      </button>
                    </div>
                  </div>

                  {/* Reset Transform Button */}
                  <button
                    type="button"
                    className={styles.actionButton}
                    onClick={handleResetTransform}
                    title="Reset entity transform to initial assembled placement"
                    data-testid="reset-transform-btn"
                    style={{ marginTop: "6px", padding: "4px 8px", fontSize: "10px" }}
                  >
                    Reset Transform
                  </button>
                </div>
              )}

              <div className={styles.propRow} style={{ marginTop: "6px" }}>
                <span className={styles.propLabel}>Placement Method</span>
                <span className={styles.propValue} style={{ fontSize: "10px" }} data-testid="placement-method">
                  {selectedEntity.placement?.method || "Depth-Projected Composition"}
                </span>
              </div>
              {selectedEntity.transform && (
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>3D Position</span>
                  <span className={styles.propValue} style={{ fontSize: "10px" }} data-testid="placement-position">
                    X: {(selectedEntity.transform.position.x ?? 0).toFixed(2)}m,{" "}
                    Y: {(selectedEntity.transform.position.y ?? 0).toFixed(2)}m,{" "}
                    Z: {(selectedEntity.transform.position.z ?? 0).toFixed(2)}m
                  </span>
                </div>
              )}
              <div className={styles.propRow}>
                <span className={styles.propLabel}>Visibility</span>
                <span
                  className={styles.propValue}
                  style={{ color: selectedEntity.visibility === false ? "#ef4444" : "#22c55e" }}
                  data-testid="entity-visibility-status"
                >
                  {selectedEntity.visibility === false ? "Hidden" : "Visible"}
                </span>
              </div>

              {/* Action Buttons: Focus and Export Mesh */}
              <div style={{ display: "flex", gap: "6px", marginTop: "8px" }}>
                <button
                  type="button"
                  className={styles.actionButton}
                  onClick={focusEntity}
                  data-testid="focus-entity-btn"
                  style={{ flex: 1, padding: "5px 8px", fontSize: "11px" }}
                >
                  Focus Camera
                </button>

                <button
                  type="button"
                  className={styles.actionButton}
                  onClick={handleExportEntityMesh}
                  disabled={selectedEntity.reconstruction?.status !== "ready"}
                  title={
                    selectedEntity.reconstruction?.status === "ready"
                      ? "Export reconstructed 3D mesh (GLB)"
                      : "Mesh not reconstructed yet"
                  }
                  data-testid="export-entity-mesh-btn"
                  style={{
                    flex: 1,
                    padding: "5px 8px",
                    fontSize: "11px",
                    background: selectedEntity.reconstruction?.status === "ready" ? "#312e81" : "#1e1e24",
                    borderColor: selectedEntity.reconstruction?.status === "ready" ? "#6366f1" : "#2e2e38",
                    color: selectedEntity.reconstruction?.status === "ready" ? "#e0e7ff" : "#71717a",
                  }}
                >
                  Export Mesh
                </button>
              </div>
            </div>

            {/* 3D Skeleton (Phase 07) */}
            <div style={{ marginTop: "10px", paddingTop: "8px", borderTop: "1px solid #27272e" }} data-testid="entity-skeleton-section">
              <div className={styles.sectionHeader} style={{ marginBottom: "6px" }}>Kinematic Skeleton</div>
              {selectedEntity.skeleton ? (
                <>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Status</span>
                    <span
                      className={`${styles.statusPill} ${
                        selectedEntity.skeleton.validationStatus === "valid" ? styles.ready : styles.failed
                      }`}
                      data-testid="skeleton-status"
                    >
                      {selectedEntity.skeleton.validationStatus === "valid" ? "Ready" : "Invalid"}
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Topology</span>
                    <span className={styles.propValue} data-testid="skeleton-joint-count">
                      {selectedEntity.skeleton.joints.length} Canonical Joints (23 Bones)
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Root Joint</span>
                    <span className={styles.propValue} data-testid="skeleton-root-joint">
                      {selectedEntity.skeleton.rootJointId}
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Source</span>
                    <span className={styles.propValue} style={{ fontSize: "10px" }} data-testid="skeleton-source">
                      {selectedEntity.skeleton.source === "smpl_regressed"
                        ? "SMPL Regressed (3D)"
                        : selectedEntity.skeleton.source === "yolo_pose_projected"
                        ? "YOLOv8 2D Pose"
                        : "Anatomical Landmarks"}
                    </span>
                  </div>
                  <button
                    type="button"
                    className={styles.actionButton}
                    onClick={() => toggleEntitySkeleton(selectedEntity.id)}
                    data-testid="toggle-skeleton-btn"
                    style={{ marginTop: "6px", padding: "5px 8px", fontSize: "11px" }}
                  >
                    {selectedEntity.showSkeleton ? "Hide Skeleton Overlay" : "Show Skeleton Overlay"}
                  </button>
                </>
              ) : (
                <div className={styles.emptyNotice} style={{ padding: "4px 0", fontSize: "11px" }}>
                  Skeleton not fitted yet.
                </div>
              )}
            </div>

            {/* Mesh Rigging (Phase 07) */}
            <div style={{ marginTop: "10px", paddingTop: "8px", borderTop: "1px solid #27272e" }} data-testid="entity-rigging-section">
              <div className={styles.sectionHeader} style={{ marginBottom: "6px" }}>Mesh Rigging</div>
              {selectedEntity.rig ? (
                <>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Rig Status</span>
                    <span
                      className={`${styles.statusPill} ${
                        selectedEntity.rig.status === "rigged"
                          ? styles.ready
                          : selectedEntity.rig.status === "failed"
                          ? styles.failed
                          : ""
                      }`}
                      style={{ textTransform: "capitalize" }}
                      data-testid="rig-status"
                    >
                      {selectedEntity.rig.status}
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Skinning Method</span>
                    <span className={styles.propValue} style={{ fontSize: "10px" }} data-testid="rig-method">
                      {selectedEntity.rig.skinningMethod === "smpl_lbs_native"
                        ? "SMPL Native LBS (24 bones)"
                        : selectedEntity.rig.skinningMethod === "proximity_heat_weights"
                        ? "Proximity Heat Weights"
                        : "None"}
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Bone Influences</span>
                    <span className={styles.propValue} style={{ fontSize: "10px" }}>
                      Max 4 influences / vertex
                    </span>
                  </div>
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Normalized</span>
                    <span className={styles.propValue} style={{ color: "#22c55e" }}>
                      {selectedEntity.rig.weightsNormalized ? "Yes (L1 = 1.0)" : "No"}
                    </span>
                  </div>
                </>
              ) : (
                <div className={styles.emptyNotice} style={{ padding: "4px 0", fontSize: "11px" }}>
                  Mesh is currently unrigged.
                </div>
              )}
            </div>

            {/* Motion & Procedural Animation (Phase 07) */}
            <div style={{ marginTop: "10px", paddingTop: "8px", borderTop: "1px solid #27272e" }} data-testid="entity-motion-section">
              <div className={styles.sectionHeader} style={{ marginBottom: "6px" }}>Motion & Animation</div>
              {selectedEntity.rig?.status === "rigged" ? (
                <>
                  <label style={{ fontSize: "10px", color: "#a1a1aa" }}>Animation Clip</label>
                  <select
                    className={styles.clipSelect}
                    value={selectedEntity.animationState?.currentClip || "idle_breathing"}
                    onChange={(e) => {
                      setEntityAnimation(
                        selectedEntity.id,
                        e.target.value,
                        selectedEntity.animationState?.isPlaying ?? false,
                        selectedEntity.animationState?.playbackSpeed ?? 1.0,
                        selectedEntity.animationState?.loop ?? true
                      );
                    }}
                    data-testid="clip-select"
                  >
                    <option value="idle_breathing">Idle Breathing (3.0s)</option>
                    <option value="torso_twist">Torso Twist (4.0s)</option>
                    <option value="arm_wave">Arm Wave (2.5s)</option>
                    <option value="gentle_walk">Gentle Walk (2.0s)</option>
                  </select>

                  <div className={styles.animControls}>
                    <button
                      type="button"
                      className={`${styles.animButton} ${
                        selectedEntity.animationState?.isPlaying ? styles.playing : ""
                      }`}
                      onClick={() => {
                        const playing = !selectedEntity.animationState?.isPlaying;
                        setEntityAnimation(
                          selectedEntity.id,
                          selectedEntity.animationState?.currentClip || "idle_breathing",
                          playing
                        );
                      }}
                      data-testid="anim-play-pause-btn"
                    >
                      {selectedEntity.animationState?.isPlaying ? "Pause" : "Play"}
                    </button>
                    <button
                      type="button"
                      className={styles.animButton}
                      onClick={() => resetEntityPose(selectedEntity.id)}
                      data-testid="anim-reset-btn"
                    >
                      Reset Pose
                    </button>
                  </div>

                  <div className={styles.propRow} style={{ marginTop: "6px" }}>
                    <span className={styles.propLabel}>Playback Speed</span>
                    <div style={{ display: "flex", gap: "4px" }}>
                      {[0.5, 1.0, 1.5].map((speed) => (
                        <button
                          key={speed}
                          type="button"
                          className={styles.tagBadge}
                          style={{
                            cursor: "pointer",
                            background:
                              (selectedEntity.animationState?.playbackSpeed ?? 1.0) === speed
                                ? "#3b82f6"
                                : "#27272a",
                            color:
                              (selectedEntity.animationState?.playbackSpeed ?? 1.0) === speed
                                ? "#ffffff"
                                : "#a1a1aa",
                          }}
                          onClick={() => {
                            setEntityAnimation(
                              selectedEntity.id,
                              selectedEntity.animationState?.currentClip || "idle_breathing",
                              selectedEntity.animationState?.isPlaying ?? false,
                              speed
                            );
                          }}
                          data-testid={`speed-${speed}x`}
                        >
                          {speed}x
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className={styles.disclaimerBadge} data-testid="anim-disclaimer">
                    Procedural Test Animation (Not recovered motion from photo)
                  </div>
                </>
              ) : (
                <div className={styles.emptyNotice} style={{ padding: "4px 0", fontSize: "11px" }}>
                  Animation unavailable — character has not been rigged.
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className={styles.emptyNotice}>
            Click a person in the viewport or select them from the list to view spatial and segmentation metrics.
          </div>
        )}
      </section>
    </aside>
  );
};
