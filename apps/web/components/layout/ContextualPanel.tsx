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
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const depthStrength = useSceneStore((state) => state.depthStrength);
  const setDepthStrength = useSceneStore((state) => state.setDepthStrength);
  const setScene = useSceneStore((state) => state.setScene);
  const setAnalysisState = useSceneStore((state) => state.setAnalysisState);

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
      setAnalysisState(true, "Queuing person detection & spatial depth estimation...", 0.1);
      setError(null);

      const job = await apiClient.triggerAnalysis(currentScene.id);
      setStatusNotice("Analyzing photograph (Detection, Segmentation, Depth)...");

      let completed = false;
      let lastSeq = 0;

      while (!completed) {
        await new Promise((r) => setTimeout(r, 600));
        try {
          const eventData = await apiClient.getJobEvents(job.job_id, lastSeq);
          if (eventData.events && eventData.events.length > 0) {
            for (const ev of eventData.events) {
              lastSeq = Math.max(lastSeq, ev.sequence_number);
              if (ev.message) {
                setAnalysisState(true, ev.message, ev.progress || 0.5);
              }
              if (ev.event_type === "job_completed") {
                completed = true;
              } else if (ev.event_type === "stage_failed" && ev.stage_identifier === "image_analysis") {
                throw new Error(ev.error?.message || "Analysis failed in pipeline stage.");
              }
            }
          }
        } catch (pollErr: unknown) {
          console.warn("Polling error:", pollErr);
        }
      }

      // Fetch refreshed authoritative scene
      const refreshed = await apiClient.getScene(currentScene.id);
      setScene(refreshed);
      setAnalysisState(false, null, 1.0);
      setStatusNotice(`Analysis complete: ${refreshed.entities.length} people identified, spatial depth ready.`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setError(msg);
      setAnalysisState(false, null, 0);
    }
  };

  return (
    <aside className={styles.panel} data-testid="contextual-panel">
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
          <div className={styles.personList} data-testid="detected-person-list">
            {entities.map((entity, idx) => {
              const isSelected = selectedEntityId === entity.id;
              const color = PALETTE_CSS[idx % PALETTE_CSS.length];
              const conf = entity.detection?.confidence
                ? Math.round(entity.detection.confidence * 100)
                : null;

              return (
                <div
                  key={entity.id}
                  className={`${styles.personCard} ${isSelected ? styles.selected : ""}`}
                  onClick={() => selectEntity(isSelected ? null : entity.id)}
                  data-testid={`person-card-${entity.id}`}
                >
                  <div className={styles.personCardLeft}>
                    <span className={styles.colorDot} style={{ backgroundColor: color }} />
                    <span className={styles.personName}>{entity.name || `Person ${idx + 1}`}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    {conf !== null && (
                      <span className={styles.confidenceBadge}>{conf}%</span>
                    )}
                    <span
                      className={`${styles.statusPill} ${
                        styles[entity.segmentation?.status || "unstarted"]
                      }`}
                    >
                      {entity.segmentation?.status === "ready" ? "Mask" : "Box"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className={styles.emptyNotice} data-testid="zero-person-notice">
            {currentScene
              ? "0 people detected in this photograph yet."
              : "Upload a photograph to begin multi-person detection."}
          </div>
        )}

        {/* Action Button: Run Analysis */}
        {currentScene && (
          <button
            type="button"
            className={styles.actionButton}
            onClick={handleRunAnalysis}
            disabled={isAnalyzing}
            data-testid="analyze-people-btn"
          >
            {isAnalyzing ? (
              <span>Analyzing...</span>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <span>{entities.length > 0 ? "Re-Analyze Scene" : "Analyze Scene & Depth"}</span>
              </>
            )}
          </button>
        )}
      </section>

      {/* Selected Person Inspector */}
      <section className={styles.section} data-testid="selected-person-inspector">
        <div className={styles.sectionHeader}>Selected Person</div>
        {selectedEntity ? (
          <div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Name</span>
              <span className={styles.propValue}>{selectedEntity.name}</span>
            </div>
            <div className={styles.propRow}>
              <span className={styles.propLabel}>Entity ID</span>
              <span className={styles.propValue}>{selectedEntity.id}</span>
            </div>
            {selectedEntity.detection && (
              <>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Confidence</span>
                  <span className={styles.propValue}>
                    {(selectedEntity.detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className={styles.propRow}>
                  <span className={styles.propLabel}>Pixel Box</span>
                  <span className={styles.propValue} style={{ fontSize: "10px" }}>
                    [{(selectedEntity.detection.pixelBox || (selectedEntity.detection as unknown as Record<string, number[]>).pixel_box || []).map((n) => Math.round(n)).join(", ")}]
                  </span>
                </div>
              </>
            )}

            {/* Depth Statistics for Selected Person */}
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
                {(selectedEntity.segmentation.pixelCount !== undefined || (selectedEntity.segmentation as unknown as Record<string, number>).pixel_count !== undefined) && (
                  <div className={styles.propRow}>
                    <span className={styles.propLabel}>Mask Pixels</span>
                    <span className={styles.propValue}>
                      {((selectedEntity.segmentation.pixelCount ?? (selectedEntity.segmentation as unknown as Record<string, number>).pixel_count) || 0).toLocaleString()} px
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
