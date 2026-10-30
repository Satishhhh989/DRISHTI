"use client";

import React from "react";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";
import styles from "./ContextualPanel.module.css";

export const ContextualPanel: React.FC = () => {
  const currentScene = useSceneStore((state) => state.currentScene);
  const photoDimensions = useSceneStore((state) => state.photoDimensions);
  const isInspectorOpen = useUIStore((state) => state.isInspectorOpen);

  if (!isInspectorOpen) return null;

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
              <span className={styles.propValue}>Y-Up (Meters)</span>
            </div>
          </>
        ) : (
          <div className={styles.emptyNotice}>Upload a photograph to establish scene coordinates.</div>
        )}
      </section>

      {/* Entities & Person Breakdown */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>Detected People</div>
        {currentScene && currentScene.entities.length > 0 ? (
          <div>
            {currentScene.entities.map((entity) => (
              <div key={entity.id} className={styles.propRow}>
                <span>{entity.name || entity.id}</span>
                <span className={styles.statusPill}>{entity.reconstructionStatus}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.emptyNotice}>
            0 people detected. Multi-person segmentation and detection will be activated in Phase 02.
          </div>
        )}
      </section>

      {/* Reconstructed Person Inspector Placeholder */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>Selected Entity Controls</div>
        <div className={styles.emptyNotice}>
          Individual mesh transforms, skeleton joints, and animation triggers will activate here when an entity is selected.
        </div>
      </section>
    </aside>
  );
};
