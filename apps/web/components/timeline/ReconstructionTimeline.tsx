"use client";

import React, { useMemo } from "react";
import {
  CINEMATIC_STAGES,
  CinematicStageId,
  useCinematicStore,
} from "@/stores/cinematic-store";
import { useSceneStore } from "@/stores/scene-store";
import styles from "./ReconstructionTimeline.module.css";

export const ReconstructionTimeline: React.FC = () => {
  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);
  const isPaused = useCinematicStore((state) => state.isPaused);
  const currentStageId = useCinematicStore((state) => state.currentStageId);
  const stages = useCinematicStore((state) => state.stages);
  const pauseCinematic = useCinematicStore((state) => state.pauseCinematic);
  const resumeCinematic = useCinematicStore((state) => state.resumeCinematic);
  const skipCinematic = useCinematicStore((state) => state.skipCinematic);
  const startCinematic = useCinematicStore((state) => state.startCinematic);

  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const currentScene = useSceneStore((state) => state.currentScene);
  const visualizationMode = useSceneStore((state) => state.visualizationMode);
  const setVisualizationMode = useSceneStore((state) => state.setVisualizationMode);

  const currentStageMeta = useMemo(() => {
    return CINEMATIC_STAGES.find((s) => s.id === currentStageId) || CINEMATIC_STAGES[0];
  }, [currentStageId]);

  const currentStageRuntime = stages[currentStageId] || { status: "waiting", progress: 0 };

  // Calculate timeline track progress percentage
  const completedIndex = useMemo(() => {
    let lastCompleted = -1;
    CINEMATIC_STAGES.forEach((s, idx) => {
      if (stages[s.id]?.status === "completed") {
        lastCompleted = idx;
      } else if (stages[s.id]?.status === "processing" && lastCompleted < idx) {
        lastCompleted = idx - 0.5;
      }
    });
    return Math.max(0, lastCompleted);
  }, [stages]);

  const progressPercentage = Math.round((completedIndex / (CINEMATIC_STAGES.length - 1)) * 100);

  // If there's no photo loaded or scene is empty, do not show timeline
  if (!currentScene?.sourceImage && !isAnalyzing && !isCinematicActive) {
    return null;
  }

  const handleStageClick = (stageId: CinematicStageId) => {
    const stage = CINEMATIC_STAGES.find((s) => s.id === stageId);
    if (!stage) return;

    // Allow jumping to completed stages for inspection
    const runtime = stages[stageId];
    if (runtime?.status === "completed" || !isAnalyzing) {
      setVisualizationMode(stage.mode);
      useCinematicStore.getState().setCurrentStage(stageId);
    }
  };

  const handleSkip = () => {
    skipCinematic();
    setVisualizationMode("combined");
  };

  const isAllComplete = Object.values(stages).every(
    (s) => s.status === "completed" || s.status === "skipped"
  );

  return (
    <div className={styles.timelineContainer} data-testid="reconstruction-timeline">
      <div className={styles.timelineCard}>
        {/* Header: Current Stage Name, Model/Subsystem, Summary & Controls */}
        <div className={styles.timelineHeader}>
          <div className={styles.stageTitleGroup}>
            <span className={styles.currentStagePill}>
              Stage {currentStageMeta.number} / 9
            </span>
            <span className={styles.stageSubsystem}>
              {currentStageMeta.subsystem}
            </span>
            <span className={styles.stageSummary}>
              — {currentStageRuntime.summary || currentStageMeta.description}
            </span>
          </div>

          <div className={styles.controlsGroup}>
            {isCinematicActive && (
              <>
                <button
                  type="button"
                  className={styles.controlBtn}
                  onClick={() => (isPaused ? resumeCinematic() : pauseCinematic())}
                  data-testid="timeline-pause-btn"
                  title={isPaused ? "Resume cinematic playback" : "Pause cinematic playback"}
                >
                  {isPaused ? "▶ Resume" : "❚❚ Pause"}
                </button>
                <button
                  type="button"
                  className={`${styles.controlBtn} ${styles.skipBtn}`}
                  onClick={handleSkip}
                  data-testid="timeline-skip-btn"
                  title="Skip to fully assembled 3D scene"
                >
                  Skip to 3D Scene ⏭
                </button>
              </>
            )}

            {!isCinematicActive && isAllComplete && (
              <button
                type="button"
                className={styles.controlBtn}
                onClick={startCinematic}
                data-testid="timeline-replay-btn"
                title="Replay cinematic reconstruction sequence"
              >
                ↺ Replay Sequence
              </button>
            )}
          </div>
        </div>

        {/* Stages Track */}
        <div className={styles.stagesTrack}>
          <div className={styles.connectorLine}>
            <div
              className={styles.connectorProgress}
              style={{ width: `${progressPercentage}%` }}
            />
          </div>

          {CINEMATIC_STAGES.map((stage) => {
            const runtime = stages[stage.id] || { status: "waiting", progress: 0 };
            const isCurrent = currentStageId === stage.id;
            const isCompleted = runtime.status === "completed";
            const isProcessing = runtime.status === "processing";
            const isFailed = runtime.status === "failed";
            const isClickable = isCompleted || (!isAnalyzing && runtime.status !== "waiting");

            let dotClass = styles.waiting;
            if (isCompleted) dotClass = styles.completed;
            else if (isProcessing) dotClass = styles.processing;
            else if (isFailed) dotClass = styles.failed;

            return (
              <button
                key={stage.id}
                type="button"
                className={`${styles.stageNode} ${isCurrent ? styles.activeNode : ""} ${
                  isCompleted ? styles.completedNode : ""
                }`}
                onClick={() => handleStageClick(stage.id)}
                disabled={!isClickable}
                title={`${stage.label} (${stage.subsystem}): ${runtime.status}`}
                data-testid={`timeline-stage-${stage.number}`}
              >
                <div
                  className={`${styles.nodeDot} ${dotClass} ${
                    isCurrent ? styles.activeStage : ""
                  }`}
                >
                  {isCompleted ? "✓" : stage.number}
                </div>
                <span className={styles.nodeLabel}>{stage.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

