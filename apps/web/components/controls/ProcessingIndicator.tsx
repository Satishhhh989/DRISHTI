"use client";

import React from "react";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";
import styles from "./ProcessingIndicator.module.css";

export const ProcessingIndicator: React.FC = () => {
  const isUploading = useSceneStore((state) => state.isUploading);
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const currentStageMessage = useSceneStore((state) => state.currentStageMessage);
  const errorMessage = useUIStore((state) => state.errorMessage);
  const statusNotice = useUIStore((state) => state.statusNotice);
  const setError = useUIStore((state) => state.setError);

  if (!isUploading && !isAnalyzing && !errorMessage && !statusNotice) {
    return (
      <div className={styles.statusBar} data-testid="processing-indicator">
        <div className={styles.engineState}>
          <span>Engine: Ready</span>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.statusBar} data-testid="processing-indicator">
      {errorMessage ? (
        <div className={styles.errorPill}>
          <span>Error: {errorMessage}</span>
          <button
            type="button"
            className={styles.clearErrorBtn}
            onClick={() => setError(null)}
            title="Dismiss error"
          >
            ✕
          </button>
        </div>
      ) : isAnalyzing ? (
        <div className={styles.engineState}>
          <span className={styles.pulsingDot} />
          <span>{currentStageMessage || "Analyzing multi-person photograph..."}</span>
        </div>
      ) : isUploading ? (
        <div className={styles.engineState}>
          <span className={styles.pulsingDot} />
          <span>Uploading and validating photograph...</span>
        </div>
      ) : (
        <span className={styles.notice}>{statusNotice}</span>
      )}
    </div>
  );
};
