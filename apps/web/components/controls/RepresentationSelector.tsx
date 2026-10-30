"use client";

import React from "react";
import { RepresentationMode } from "@photo-to-3d/scene-schema";
import { useSceneStore } from "@/stores/scene-store";
import styles from "./RepresentationSelector.module.css";

interface ModeOption {
  id: RepresentationMode;
  label: string;
  badge?: string;
}

const MODES: ModeOption[] = [
  { id: "photo", label: "Photo" },
  { id: "wireframe", label: "Wireframe", badge: "P06" },
  { id: "solid3d", label: "3D Solid", badge: "P06" },
];

export const RepresentationSelector: React.FC = () => {
  const activeRepresentation = useSceneStore((state) => state.activeRepresentation);
  const setActiveRepresentation = useSceneStore((state) => state.setActiveRepresentation);

  return (
    <nav className={styles.container} aria-label="Representation Mode">
      {MODES.map((mode) => {
        const isActive = activeRepresentation === mode.id;
        return (
          <button
            key={mode.id}
            type="button"
            className={`${styles.tab} ${isActive ? styles.active : ""}`}
            onClick={() => setActiveRepresentation(mode.id)}
            aria-pressed={isActive}
            data-testid={`mode-${mode.id}`}
          >
            <span>{mode.label}</span>
            {mode.badge && <span className={styles.badge}>{mode.badge}</span>}
          </button>
        );
      })}
    </nav>
  );
};
