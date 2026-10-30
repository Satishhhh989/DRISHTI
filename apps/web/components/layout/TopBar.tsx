"use client";

import React from "react";
import { RepresentationSelector } from "@/components/controls/RepresentationSelector";
import { UploadButton } from "@/components/controls/UploadButton";
import { useUIStore } from "@/stores/ui-store";
import styles from "./TopBar.module.css";

export const TopBar: React.FC = () => {
  const showGrid = useUIStore((state) => state.showGrid);
  const toggleGrid = useUIStore((state) => state.toggleGrid);
  const isInspectorOpen = useUIStore((state) => state.isInspectorOpen);
  const toggleInspector = useUIStore((state) => state.toggleInspector);
  const resetCamera = useUIStore((state) => state.resetCamera);
  const fitScene = useUIStore((state) => state.fitScene);
  const apiConnected = useUIStore((state) => state.apiConnected);

  return (
    <header className={styles.topBar}>
      <div className={styles.identitySection}>
        <div className={styles.brand}>
          <span className={styles.title}>DRISHTI</span>
          <span className={styles.brandSubtitle}>PHOTO → 3D ENGINE</span>
          <span className={styles.phaseTag}>Phase 07</span>
        </div>
        <div className={styles.runtimeBadge}>
          <span
            className={`${styles.statusDot} ${apiConnected ? styles.connected : ""}`}
            title={apiConnected ? "Backend connected (127.0.0.1:8000)" : "Backend offline / checking"}
          />
          <span>{apiConnected ? "LOCAL RUNTIME" : "STANDBY"}</span>
        </div>
      </div>

      <div className={styles.centerSection}>
        <RepresentationSelector />
      </div>

      <div className={styles.actionsSection}>
        <button
          type="button"
          className={`${styles.toolButton} ${showGrid ? styles.active : ""}`}
          onClick={toggleGrid}
          title="Toggle 3D Coordinate Grid"
          data-testid="toggle-grid-btn"
        >
          Grid
        </button>

        <button
          type="button"
          className={styles.toolButton}
          onClick={fitScene}
          title="Fit Camera to All Assembled People"
          data-testid="fit-scene-btn"
        >
          Fit Scene
        </button>

        <button
          type="button"
          className={styles.toolButton}
          onClick={resetCamera}
          title="Reset Camera View"
          data-testid="reset-camera-btn"
        >
          Reset View
        </button>

        <button
          type="button"
          className={`${styles.toolButton} ${isInspectorOpen ? styles.active : ""}`}
          onClick={toggleInspector}
          title="Toggle Contextual Inspector"
          data-testid="toggle-inspector-btn"
        >
          Inspector
        </button>

        <UploadButton label="Upload Photo" />
      </div>
    </header>
  );
};

