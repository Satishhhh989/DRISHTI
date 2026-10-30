"use client";

import React from "react";
import {
  VisualizationMode,
  computeRepresentationAvailability,
} from "@photo-to-3d/scene-schema";
import { useSceneStore } from "@/stores/scene-store";
import styles from "./RepresentationSelector.module.css";

interface ModeOption {
  id: VisualizationMode;
  label: string;
  badge?: string;
  testId: string;
}

const MODES: ModeOption[] = [
  { id: "original", label: "Original", testId: "mode-photo" },
  { id: "detection", label: "Detection", testId: "mode-detection" },
  { id: "segmentation", label: "Segmentation", testId: "mode-segmentation" },
  { id: "depth", label: "Depth Map", testId: "mode-depth" },
  { id: "spatial_photo", label: "Spatial Photo", testId: "mode-spatial-photo" },
  { id: "wireframe", label: "Wireframe", testId: "mode-wireframe" },
  { id: "solid3d", label: "3D Solid", testId: "mode-solid3d" },
];

export const RepresentationSelector: React.FC = () => {
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const setVisualizationMode = useSceneStore((state) => state.setVisualizationMode);
  const currentScene = useSceneStore((state) => state.currentScene);
  const currentStageMessage = useSceneStore((state) => state.currentStageMessage);
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);

  const availability = computeRepresentationAvailability(
    currentScene,
    currentStageMessage,
    isAnalyzing
  );

  return (
    <nav className={styles.container} aria-label="Visualization and Representation Mode">
      {MODES.map((mode) => {
        const isActive = visualizationMode === mode.id;
        const avail = availability[mode.id];

        let badgeText = mode.badge;
        if (!badgeText) {
          if (avail.status === "processing") {
            badgeText = "LIVE";
          } else if (avail.status === "failed") {
            badgeText = "ERR";
          }
        }

        return (
          <button
            key={mode.id}
            type="button"
            className={`${styles.tab} ${isActive ? styles.active : ""}`}
            onClick={() => setVisualizationMode(mode.id)}
            aria-pressed={isActive}
            data-testid={mode.testId}
            title={avail.reason}
          >
            <span>{mode.label}</span>
            {badgeText && <span className={styles.badge}>{badgeText}</span>}
          </button>
        );
      })}
    </nav>
  );
};
