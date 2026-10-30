import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import WorkspacePage from "../app/page";
import { useSceneStore } from "../stores/scene-store";

describe("Representation Mode Switching", () => {
  beforeEach(() => {
    useSceneStore.getState().clearScene();
  });

  it("displays honest placeholder when selecting Wireframe mode", async () => {
    await act(async () => {
      render(<WorkspacePage />);
    });

    const wireframeBtn = screen.getByTestId("mode-wireframe");
    await act(async () => {
      fireEvent.click(wireframeBtn);
    });

    const notice = screen.getByTestId("wireframe-placeholder");
    expect(notice).toBeInTheDocument();
    expect(notice).toHaveTextContent("Polygon wireframe geometry not generated yet");
    expect(notice).toHaveTextContent("Phase 06");
  });

  it("displays honest placeholder when selecting 3D Solid mode", async () => {
    await act(async () => {
      render(<WorkspacePage />);
    });

    const solidBtn = screen.getByTestId("mode-solid3d");
    await act(async () => {
      fireEvent.click(solidBtn);
    });

    const notice = screen.getByTestId("solid3d-placeholder");
    expect(notice).toBeInTheDocument();
    expect(notice).toHaveTextContent("Solid volumetric human mesh not generated yet");
  });
});
