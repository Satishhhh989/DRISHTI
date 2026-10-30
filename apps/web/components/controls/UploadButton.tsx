"use client";

import React, { useRef } from "react";
import { apiClient } from "@/lib/api-client";
import { getImageDimensions } from "@/lib/image-utils";
import { useSceneStore } from "@/stores/scene-store";
import { useUIStore } from "@/stores/ui-store";
import styles from "./UploadButton.module.css";

interface UploadButtonProps {
  label?: string;
  className?: string;
}

export const UploadButton: React.FC<UploadButtonProps> = ({
  label = "Upload Photograph",
  className,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isUploading = useSceneStore((state) => state.isUploading);
  const setIsUploading = useSceneStore((state) => state.setIsUploading);
  const setLocalPhoto = useSceneStore((state) => state.setLocalPhoto);
  const setScene = useSceneStore((state) => state.setScene);
  const setAnalysisState = useSceneStore((state) => state.setAnalysisState);
  const applyEvent = useSceneStore((state) => state.applyEvent);
  const reconcileAuthoritativeScene = useSceneStore(
    (state) => state.reconcileAuthoritativeScene
  );
  const setError = useUIStore((state) => state.setError);
  const setStatusNotice = useUIStore((state) => state.setStatusNotice);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      setError(null);
      setStatusNotice("Inspecting photograph dimensions...");

      // 1. Immediately read image dimensions and set local preview URL for zero-latency viewport render
      const dims = await getImageDimensions(file);
      const localUrl = URL.createObjectURL(file);
      setLocalPhoto(localUrl, dims.width, dims.height, file.name);

      // 2. Persist to local backend if available
      try {
        setStatusNotice("Uploading photograph to local storage...");
        const res = await apiClient.uploadPhoto(file);
        if (res.scene) {
          const targetSceneId = res.scene.id;
          setScene(res.scene);

          // 3. Automatically trigger multi-stage progressive analysis pipeline
          try {
            setAnalysisState(
              true,
              "Initializing progressive neural pipeline...",
              0.05
            );
            setStatusNotice("Running local neural networks (detection, segmentation, depth)...");
            const job = await apiClient.triggerAnalysis(targetSceneId);
            setAnalysisState(true, "Job accepted by engine", 0.08, job.job_id);

            // 4. Stream real-time processing events via SSE with fallback
            await new Promise<void>((resolve) => {
              const unsubscribe = apiClient.subscribeToJobEvents(
                job.job_id,
                0,
                (ev) => {
                  applyEvent(ev);
                  if (ev.message) {
                    setStatusNotice(ev.message);
                  }
                },
                async () => {
                  unsubscribe();
                  try {
                    const refreshed = await apiClient.getScene(targetSceneId);
                    reconcileAuthoritativeScene(refreshed);
                    setStatusNotice(
                      refreshed.entities.length > 0
                        ? `Analysis complete: ${refreshed.entities.length} people identified.`
                        : "Analysis complete: No people detected."
                    );
                  } catch {
                    // Ignore snapshot refresh error if offline
                  }
                  resolve();
                },
                (streamErr) => {
                  console.warn("Event stream error:", streamErr);
                }
              );
            });
          } catch (analysisErr: unknown) {
            console.warn("Analysis pipeline error:", analysisErr);
            setAnalysisState(false, null, 0);
          }
        }
      } catch (err: unknown) {
        // Fall back gracefully to client-side session without failing the user
        const msg = err instanceof Error ? err.message : "API offline";
        console.warn("Backend storage not reached, operating in client-local mode:", msg);
        setStatusNotice("Operating in local offline mode.");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load image";
      setError(msg);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className={className}>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: "none" }}
        onChange={handleFileChange}
        data-testid="file-upload-input"
      />
      <button
        type="button"
        className={styles.uploadButton}
        onClick={handleClick}
        disabled={isUploading}
        data-testid="upload-photograph-btn"
      >
        <svg
          className={styles.icon}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <span>{isUploading ? "Uploading..." : label}</span>
      </button>
    </div>
  );
};
