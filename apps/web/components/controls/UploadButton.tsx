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
          setScene(res.scene);
        }
        setStatusNotice("Photograph loaded into 3D scene.");
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

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className={styles.hiddenInput}
        onChange={handleFileChange}
        data-testid="file-upload-input"
      />
      <button
        type="button"
        className={`${styles.uploadBtn} ${className || ""}`}
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        data-testid="upload-photograph-btn"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <span>{isUploading ? "Loading..." : label}</span>
      </button>
    </>
  );
};
