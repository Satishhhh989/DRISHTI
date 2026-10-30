"use client";

import React from "react";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";
import { apiClient } from "@/lib/api-client";
import styles from "./ContextualPanel.module.css";

const PALETTE_CSS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

export const ContextualPanel: React.FC = () => {
  const currentScene = useSceneStore((state) => state.currentScene);
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

  const isInspectorOpen = useUIStore((state) => state.isInspectorOpen);
  const selectedEntityId = useUIStore((state) => state.selectedEntityId);
  const selectEntity = useUIStore((state) => state.selectEntity);
  const resetCamera = useUIStore((state) => state.resetCamera);
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

  // Compute honest stage metrics
  const readyMasksCount = entities.filter(
    (e) => e.segmentation?.status === "ready"
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
                    <span
                      className={styles.colorDot}
                      style={{ backgroundColor: colorCss }}
                    />
                    <span className={styles.personName}>{entity.name || `Person ${idx + 1}`}</span>
                  </div>
                  <span className={styles.confidenceBadge}>{conf}</span>
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
            <div className={styles.propRow}>
              <span className={styles.propLabel}>3D Mesh</span>
              <span className={styles.statusPill}>Phase 06</span>
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
