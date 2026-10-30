import React from "react";
import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import WorkspacePage from "../app/page";
import { useSceneStore } from "../stores/scene-store";

// Mock fetch for api health check
global.fetch = vi.fn().mockImplementation(() =>
  Promise.resolve({
    ok: true,
    json: () =>
      Promise.resolve({
        status: "ok",
        service: "photo-to-3d-api",
        version: "0.2.0",
        timestamp: new Date().toISOString(),
        storage_available: true,
      }),
  })
);

describe("Workspace Page", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
  });

  it("renders workspace with top bar, empty state, visualization modes, and inspector", async () => {
    await act(async () => {
      render(<WorkspacePage />);
    });

    // Branding in top bar
    expect(screen.getByText("PHOTO → 3D ENGINE")).toBeInTheDocument();
    expect(screen.getByText(/Phase 0[1-7]/)).toBeInTheDocument();

    // Mode options
    expect(screen.getByTestId("mode-photo")).toBeInTheDocument();
    expect(screen.getByTestId("mode-detection")).toBeInTheDocument();
    expect(screen.getByTestId("mode-segmentation")).toBeInTheDocument();
    expect(screen.getByTestId("mode-depth")).toBeInTheDocument();
    expect(screen.getByTestId("mode-spatial-photo")).toBeInTheDocument();
    expect(screen.getByTestId("mode-wireframe")).toBeInTheDocument();
    expect(screen.getByTestId("mode-solid3d")).toBeInTheDocument();

    // Empty state viewport invitation
    expect(screen.getByTestId("empty-viewport-state")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Upload a Photograph" })).toBeInTheDocument();

    // Contextual panel
    expect(screen.getByTestId("contextual-panel")).toBeInTheDocument();
  });
});
