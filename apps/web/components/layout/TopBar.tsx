"use client";

import React, { useState, useEffect, useRef } from "react";
import { RepresentationSelector } from "@/components/controls/RepresentationSelector";
import { UploadButton } from "@/components/controls/UploadButton";
import { useSceneStore } from "@/stores/scene-store";
import { useCinematicStore } from "@/stores/cinematic-store";
import { useProjectStore } from "@/stores/project-store";
import { useUIStore } from "@/stores/ui-store";
import { apiClient } from "@/lib/api-client";
import styles from "./TopBar.module.css";

export const TopBar: React.FC = () => {
  const showGrid = useUIStore((state) => state.showGrid);
  const toggleGrid = useUIStore((state) => state.toggleGrid);
  const isInspectorOpen = useUIStore((state) => state.isInspectorOpen);
  const toggleInspector = useUIStore((state) => state.toggleInspector);
  const resetCamera = useUIStore((state) => state.resetCamera);
  const fitScene = useUIStore((state) => state.fitScene);
  const apiConnected = useUIStore((state) => state.apiConnected);
  const setError = useUIStore((state) => state.setError);
  const setStatusNotice = useUIStore((state) => state.setStatusNotice);

  const currentScene = useSceneStore((state) => state.currentScene);
  const localPhotoUrl = useSceneStore((state) => state.localPhotoUrl);
  const isAnalyzing = useSceneStore((state) => state.isAnalyzing);
  const qualityPreset = useSceneStore((state) => state.qualityPreset);
  const setAnalysisState = useSceneStore((state) => state.setAnalysisState);
  const applyEvent = useSceneStore((state) => state.applyEvent);
  const reconcileAuthoritativeScene = useSceneStore(
    (state) => state.reconcileAuthoritativeScene
  );

  const isCinematicActive = useCinematicStore((state) => state.isCinematicActive);

  const currentProject = useProjectStore((state) => state.currentProject);
  const isDirty = useProjectStore((state) => state.isDirty);
  const isSaving = useProjectStore((state) => state.isSaving);
  const saveCurrentProject = useProjectStore((state) => state.saveCurrentProject);
  const setProjectName = useProjectStore((state) => state.setProjectName);
  const recentProjects = useProjectStore((state) => state.recentProjects);
  const openProject = useProjectStore((state) => state.openProject);
  const loadRecentProjects = useProjectStore((state) => state.loadRecentProjects);
  const importProject = useProjectStore((state) => state.importProject);
  const initProjectFromScene = useProjectStore((state) => state.initProjectFromScene);

  const [isProjectsMenuOpen, setProjectsMenuOpen] = useState(false);
  const [isExportMenuOpen, setExportMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const hasReconstructedMeshes = Boolean(
    currentScene?.entities?.some(
      (e) =>
        e.reconstruction?.status === "ready" &&
        Boolean(e.reconstruction?.meshUrl || e.reconstruction?.meshAssetId)
    )
  );

  const handleReconstructScene = async () => {
    if (!currentScene && !localPhotoUrl) {
      setError("Please upload a photograph first.");
      return;
    }

    // Replay presentation if already reconstructed and not analyzing
    if (hasReconstructedMeshes && !isAnalyzing) {
      useCinematicStore.getState().startCinematic();
      return;
    }

    if (currentScene && !currentScene.id.startsWith("local-")) {
      try {
        useCinematicStore.getState().startCinematic();
        setAnalysisState(true, "Initializing neural reconstruction pipeline...", 0.05);
        setError(null);
        setStatusNotice("Reconstructing scene (Detection, Segmentation, Depth, HMR 2.0)...");

        const job = await apiClient.triggerAnalysis(currentScene.id, 0.35, false, true, qualityPreset);
        setAnalysisState(true, "Job accepted by engine", 0.08, job.job_id);

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
              useCinematicStore.getState().updateStageStatus("stage_combined", "completed", { progress: 1.0 });
              setStatusNotice(
                refreshed.entities.length > 0
                  ? `Reconstruction complete: ${refreshed.entities.length} people identified & reconstructed.`
                  : "Reconstruction complete: No people detected."
              );
            } catch {
              // Ignore refresh error
            }
          },
          (err) => {
            console.warn("Event stream error:", err);
          }
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Reconstruction failed";
        setError(msg);
        setAnalysisState(false, null, 0);
        useCinematicStore.getState().skipCinematic();
      }
    } else {
      // Local preview or mock mode
      useCinematicStore.getState().startCinematic();
    }
  };

  // Initialize project if scene loads and no project active
  useEffect(() => {
    if (currentScene && !currentProject) {
      initProjectFromScene(currentScene);
    }
  }, [currentScene, currentProject, initProjectFromScene]);

  // Load recent projects once on mount
  useEffect(() => {
    loadRecentProjects();
  }, [loadRecentProjects]);

  // Save shortcut (Cmd+S / Ctrl+S)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        saveCurrentProject();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [saveCurrentProject]);

  const handleExportScene = () => {
    if (!currentScene) return;
    const url = apiClient.getSceneExportUrl(currentScene.id);
    window.open(url, "_blank");
    setStatusNotice("Exported assembled 3D scene (GLB).");
    setExportMenuOpen(false);
  };

  const handleExportMetadata = () => {
    if (!currentScene) return;
    const url = apiClient.getMetadataExportUrl(currentScene.id);
    window.open(url, "_blank");
    setStatusNotice("Exported scene metadata (JSON).");
    setExportMenuOpen(false);
  };

  const handleExportArchive = () => {
    if (!currentProject) return;
    const url = apiClient.getProjectArchiveUrl(currentProject.id);
    window.open(url, "_blank");
    setStatusNotice("Exported portable DRISHTI project archive (.zip).");
    setExportMenuOpen(false);
  };

  const handleImportFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const res = await importProject(file);
      if (res) {
        setStatusNotice(`Imported project: ${res.name}`);
      }
    }
    setProjectsMenuOpen(false);
  };

  const projectName = currentProject?.name || currentScene?.name || "Untitled Project";

  return (
    <header className={styles.topBar}>
      <div className={styles.identitySection}>
        <div className={styles.brand}>
          <span className={styles.title}>DRISHTI</span>
          <span className={styles.brandSubtitle}>PHOTO → 3D ENGINE</span>
          <span className={styles.phaseTag}>Phase 10</span>
        </div>

        <div className={styles.runtimeBadge}>
          <span
            className={`${styles.statusDot} ${apiConnected ? styles.connected : ""}`}
            title={apiConnected ? "Backend connected (127.0.0.1:8000)" : "Backend offline / checking"}
          />
          <span>{apiConnected ? "LOCAL RUNTIME" : "STANDBY"}</span>
        </div>

        {/* Project Name and Dirty Indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginLeft: "8px" }}>
          <input
            type="text"
            className={styles.projectNameInput}
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            title="Click to rename project"
            data-testid="project-name-input"
          />
          {isDirty && (
            <span
              className={styles.dirtyDot}
              title="Unsaved changes in workspace"
              data-testid="unsaved-changes-indicator"
            />
          )}
        </div>
      </div>

      <div className={styles.centerSection}>
        <RepresentationSelector />
      </div>

      <div className={styles.actionsSection}>
        {/* Save Button */}
        <button
          type="button"
          className={styles.saveBtn}
          onClick={saveCurrentProject}
          disabled={isSaving}
          title="Save project locally (Cmd+S / Ctrl+S)"
          data-testid="save-project-btn"
        >
          {isSaving ? "Saving..." : isDirty ? "Save *" : "Saved"}
        </button>

        {/* Projects Menu */}
        <div className={styles.menuContainer}>
          <button
            type="button"
            className={styles.toolButton}
            onClick={() => {
              setProjectsMenuOpen(!isProjectsMenuOpen);
              setExportMenuOpen(false);
            }}
            title="Projects and Local Storage"
            data-testid="projects-menu-btn"
          >
            Projects ▾
          </button>

          {isProjectsMenuOpen && (
            <div className={styles.dropdownMenu} data-testid="projects-dropdown">
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={() => fileInputRef.current?.click()}
                data-testid="import-archive-btn"
              >
                <span>Import Project Archive (.zip)</span>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                accept=".zip"
                onChange={handleImportFileChange}
              />

              <div className={styles.dropdownDivider} />
              <div style={{ padding: "4px 10px", fontSize: "10px", color: "#71717a", fontWeight: 600 }}>
                RECENT PROJECTS
              </div>

              {recentProjects.length > 0 ? (
                recentProjects.slice(0, 5).map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={styles.dropdownItem}
                    onClick={() => {
                      openProject(p.id);
                      setProjectsMenuOpen(false);
                    }}
                    data-testid={`recent-project-${p.id}`}
                  >
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {p.name}
                    </span>
                    <span style={{ fontSize: "9px", color: "#71717a" }}>
                      {new Date(p.updatedAt).toLocaleDateString()}
                    </span>
                  </button>
                ))
              ) : (
                <div style={{ padding: "6px 10px", fontSize: "10px", color: "#71717a" }}>
                  No recent projects found.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Export Menu */}
        <div className={styles.menuContainer}>
          <button
            type="button"
            className={styles.toolButton}
            onClick={() => {
              setExportMenuOpen(!isExportMenuOpen);
              setProjectsMenuOpen(false);
            }}
            title="Export Scene, Meshes or Archive"
            data-testid="export-menu-btn"
          >
            Export ▾
          </button>

          {isExportMenuOpen && (
            <div className={styles.dropdownMenu} data-testid="export-dropdown">
              <button
                type="button"
                className={styles.dropdownItem}
                onClick={handleExportScene}
                disabled={!currentScene}
                data-testid="export-assembled-glb-btn"
              >
                <span>Assembled Scene (GLB)</span>
                <span style={{ fontSize: "9px", color: "#71717a" }}>.glb</span>
              </button>

              <button
                type="button"
                className={styles.dropdownItem}
                onClick={handleExportMetadata}
                disabled={!currentScene}
                data-testid="export-metadata-json-btn"
              >
                <span>Scene Metadata</span>
                <span style={{ fontSize: "9px", color: "#71717a" }}>.json</span>
              </button>

              <div className={styles.dropdownDivider} />

              <button
                type="button"
                className={styles.dropdownItem}
                onClick={handleExportArchive}
                disabled={!currentProject}
                data-testid="export-portable-archive-btn"
              >
                <span>Portable Project Archive</span>
                <span style={{ fontSize: "9px", color: "#71717a" }}>.zip</span>
              </button>
            </div>
          )}
        </div>

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

        {/* Primary Reconstruct Scene Button */}
        <button
          type="button"
          className={`${styles.reconstructBtn} ${isAnalyzing || isCinematicActive ? styles.running : ""}`}
          onClick={handleReconstructScene}
          disabled={!currentScene && !localPhotoUrl}
          title={
            !currentScene && !localPhotoUrl
              ? "Upload a photograph first"
              : isAnalyzing || isCinematicActive
              ? "Reconstruction in progress"
              : hasReconstructedMeshes
              ? "Replay 3D reconstruction presentation"
              : "Reconstruct 3D scene from photograph"
          }
          data-testid="reconstruct-scene-btn"
        >
          {(isAnalyzing || isCinematicActive) && <span className={styles.pulsingDot} />}
          <span>
            {isAnalyzing || isCinematicActive
              ? "Reconstructing..."
              : hasReconstructedMeshes
              ? "Replay Reconstruction"
              : "Reconstruct Scene"}
          </span>
        </button>

        <UploadButton label="Upload Photo" />
      </div>
    </header>
  );
};
