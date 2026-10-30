"use client";

import React from "react";
import { TopBar } from "@/components/layout/TopBar";
import { ContextualPanel } from "@/components/layout/ContextualPanel";
import { ThreeViewport } from "@/components/viewport/ThreeViewport";
import { ProcessingIndicator } from "@/components/controls/ProcessingIndicator";
import { useApiHealth } from "@/hooks/useApiHealth";
import styles from "./page.module.css";

export default function WorkspacePage() {
  // Continuous health monitor for local backend connection status
  useApiHealth();

  return (
    <div className={styles.workspaceLayout} data-testid="workspace-root">
      <TopBar />
      <div className={styles.mainArea}>
        <ThreeViewport />
        <ContextualPanel />
        <ProcessingIndicator />
      </div>
    </div>
  );
}
