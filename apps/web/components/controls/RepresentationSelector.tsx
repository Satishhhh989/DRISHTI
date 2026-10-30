"use client";

import React from "react";
import { VisualizationMode } from "@photo-to-3d/scene-schema";
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
  { id: "wireframe", label: "Wireframe", badge: "P06", testId: "mode-wireframe" },
  { id: "solid3d", label: "3D Solid", badge: "P06", testId: "mode-solid3d" },
];

export const RepresentationSelector: React.FC = () => {
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const setVisualizationMode = useSceneStore((state) => state.setVisualizationMode);

  return (
    <nav className={styles.container} aria-label="Visualization and Representation Mode">
      {MODES.map((mode) => {
        const isActive = visualizationMode === mode.id;
        return (
          <button
            key={mode.id}
            type="button"
            className={`${styles.tab} ${isActive ? styles.active : ""}`}
            onClick={() => setVisualizationMode(mode.id)}
            aria-pressed={isActive}
            data-testid={mode.testId}
          >
            <span>{mode.label}</span>
            {mode.badge && <span className={styles.badge}>{mode.badge}</span>}
          </button>
        );
      })}
    </nav>
  );
};
