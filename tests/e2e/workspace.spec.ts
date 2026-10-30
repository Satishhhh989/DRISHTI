import { test, expect } from "@playwright/test";

test.describe("Phase 01 - 09: Workspace & Viewport E2E Verification", () => {
  test("1. The application opens and renders workspace", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('[data-testid="workspace-root"]')).toBeVisible();
    await expect(page.getByText("PHOTO → 3D ENGINE")).toBeVisible();
    await expect(page.getByText(/Phase (0[1-9]|10)/)).toBeVisible();
  });

  test("2. The upload control is visible in top bar and empty viewport", async ({ page }) => {
    await page.goto("/");
    const topBarUploadBtn = page.getByRole("button", { name: /Upload Photo/i });
    await expect(topBarUploadBtn).toBeVisible();

    const emptyState = page.locator('[data-testid="empty-viewport-state"]');
    await expect(emptyState).toBeVisible();
    await expect(page.getByRole("heading", { name: "Upload a Photograph" })).toBeVisible();
    await expect(emptyState.getByRole("button", { name: /Choose Photograph/i })).toBeVisible();
  });

  test("3. The viewport container initializes correctly", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('[data-testid="viewport-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="contextual-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-indicator"]')).toBeVisible();
  });

  test("4. Representation controls work and unavailable modes display honest placeholders", async ({ page }) => {
    await page.goto("/");

    // Click Wireframe mode
    const wireframeTab = page.locator('[data-testid="mode-wireframe"]');
    await wireframeTab.click();
    await expect(wireframeTab).toHaveAttribute("aria-pressed", "true");

    // Verify honest placeholder is shown without fake geometry
    const wireframeNotice = page.locator('[data-testid="wireframe-placeholder"]');
    await expect(wireframeNotice).toBeVisible();
    await expect(wireframeNotice).toContainText("Polygon wireframe geometry not generated yet");

    // Click 3D Solid mode
    const solidTab = page.locator('[data-testid="mode-solid3d"]');
    await solidTab.click();
    await expect(solidTab).toHaveAttribute("aria-pressed", "true");

    // Verify honest placeholder is shown
    const solidNotice = page.locator('[data-testid="solid3d-placeholder"]');
    await expect(solidNotice).toBeVisible();
    await expect(solidNotice).toContainText("Solid volumetric human mesh not generated yet");

    // Return to Original photo mode
    const photoTab = page.locator('[data-testid="mode-photo"]');
    await photoTab.click();
    await expect(photoTab).toHaveAttribute("aria-pressed", "true");
    await expect(wireframeNotice).not.toBeVisible();
    await expect(solidNotice).not.toBeVisible();
  });

  test("5. Phase 02 visualization controls (Original, Detection, Segmentation) are functional", async ({ page }) => {
    await page.goto("/");

    const detectionTab = page.locator('[data-testid="mode-detection"]');
    const segmentationTab = page.locator('[data-testid="mode-segmentation"]');
    const originalTab = page.locator('[data-testid="mode-photo"]');

    await expect(detectionTab).toBeVisible();
    await expect(segmentationTab).toBeVisible();
    await expect(originalTab).toBeVisible();

    await detectionTab.click();
    await expect(detectionTab).toHaveAttribute("aria-pressed", "true");

    await segmentationTab.click();
    await expect(segmentationTab).toHaveAttribute("aria-pressed", "true");

    await originalTab.click();
    await expect(originalTab).toHaveAttribute("aria-pressed", "true");
  });

  test("6. Contextual panel reflects empty scene state honestly", async ({ page }) => {
    await page.goto("/");
    const contextualPanel = page.locator('[data-testid="contextual-panel"]');
    await expect(contextualPanel).toBeVisible();
    await expect(contextualPanel).toContainText("Scene Summary");
    await expect(contextualPanel).toContainText("Detected People");
    await expect(contextualPanel).toContainText("Spatial Photography");
    await expect(page.locator('[data-testid="zero-person-notice"]')).toBeVisible();
  });

  test("7. Phase 03 Depth Map and Spatial Photo modes are functional", async ({ page }) => {
    await page.goto("/");

    const depthTab = page.locator('[data-testid="mode-depth"]');
    const spatialTab = page.locator('[data-testid="mode-spatial-photo"]');

    await expect(depthTab).toBeVisible();
    await expect(spatialTab).toBeVisible();

    await depthTab.click();
    await expect(depthTab).toHaveAttribute("aria-pressed", "true");

    await spatialTab.click();
    await expect(spatialTab).toHaveAttribute("aria-pressed", "true");
  });

  test("8. Phase 04 Pipeline Inspector is visible and reflects real-time stage states", async ({ page }) => {
    await page.goto("/");
    const inspector = page.locator('[data-testid="pipeline-inspector"]');
    await expect(inspector).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-validation"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-detection"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-segmentation"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-depth"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-spatial"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-reconstruction"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-pose"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-skeleton"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-rigging"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-animation"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-assembly"]')).toBeVisible();
  });

  test("9. Phase 05 Human 3D Wireframe and Solid modes are functional", async ({ page }) => {
    await page.goto("/");

    const wireframeTab = page.locator('[data-testid="mode-wireframe"]');
    const solidTab = page.locator('[data-testid="mode-solid3d"]');

    await expect(wireframeTab).toBeVisible();
    await expect(solidTab).toBeVisible();

    await wireframeTab.click();
    await expect(wireframeTab).toHaveAttribute("aria-pressed", "true");

    await solidTab.click();
    await expect(solidTab).toHaveAttribute("aria-pressed", "true");
  });

  test("10. Phase 06 Multi-Person Scene Assembly and controls are functional", async ({ page }) => {
    await page.goto("/");

    const fitSceneBtn = page.locator('[data-testid="fit-scene-btn"]');
    await expect(fitSceneBtn).toBeVisible();

    // Verify Scene Assembly row exists in pipeline inspector
    await expect(page.locator('[data-testid="stage-row-assembly"]')).toBeVisible();
    await expect(page.getByText("Scene Assembly")).toBeVisible();
  });

  test("11. Phase 07 DRISHTI Motion stages and animation controls are present", async ({ page }) => {
    await page.goto("/");

    // Verify Motion Pipeline Inspector stage rows
    await expect(page.locator('[data-testid="stage-row-pose"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-skeleton"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-rigging"]')).toBeVisible();
    await expect(page.locator('[data-testid="stage-row-animation"]')).toBeVisible();

    // Verify Entity Inspector container is mounted
    await expect(page.locator('[data-testid="entity-details-section"]')).toBeVisible();
  });

  test("12. Phase 08 Unified Hybrid Renderer representations and honest placeholders are functional", async ({ page }) => {
    await page.goto("/");

    // Verify all 9 representation tabs from the unified registry are mounted
    const photoTab = page.locator('[data-testid="mode-photo"]');
    const detectionTab = page.locator('[data-testid="mode-detection"]');
    const segTab = page.locator('[data-testid="mode-segmentation"]');
    const depthTab = page.locator('[data-testid="mode-depth"]');
    const spatialTab = page.locator('[data-testid="mode-spatial-photo"]');
    const wireframeTab = page.locator('[data-testid="mode-wireframe"]');
    const solidTab = page.locator('[data-testid="mode-solid3d"]');
    const articulatedTab = page.locator('[data-testid="mode-articulated"]');
    const combinedTab = page.locator('[data-testid="mode-combined"]');

    await expect(photoTab).toBeVisible();
    await expect(detectionTab).toBeVisible();
    await expect(segTab).toBeVisible();
    await expect(depthTab).toBeVisible();
    await expect(spatialTab).toBeVisible();
    await expect(wireframeTab).toBeVisible();
    await expect(solidTab).toBeVisible();
    await expect(articulatedTab).toBeVisible();
    await expect(combinedTab).toBeVisible();

    // Click Articulated mode and verify honest placeholder when mesh is not yet reconstructed
    await articulatedTab.click();
    await expect(articulatedTab).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('[data-testid="articulated-placeholder"]')).toBeVisible();

    // Click Combined mode and verify honest placeholder
    await combinedTab.click();
    await expect(combinedTab).toHaveAttribute("aria-pressed", "true");
    await expect(page.locator('[data-testid="combined-placeholder"]')).toBeVisible();

    // Return to Photo mode
    await photoTab.click();
    await expect(photoTab).toHaveAttribute("aria-pressed", "true");
  });

  test("13. Phase 09 Interaction, Persistence, and Export controls are present", async ({ page }) => {
    await page.goto("/");

    // Project management in top bar
    const nameInput = page.locator('[data-testid="project-name-input"]');
    await expect(nameInput).toBeVisible();

    const saveBtn = page.locator('[data-testid="save-project-btn"]');
    await expect(saveBtn).toBeVisible();

    // Projects menu
    const projectsMenuBtn = page.locator('[data-testid="projects-menu-btn"]');
    await expect(projectsMenuBtn).toBeVisible();
    await projectsMenuBtn.click();
    await expect(page.locator('[data-testid="import-archive-btn"]')).toBeVisible();

    // Export menu
    const exportMenuBtn = page.locator('[data-testid="export-menu-btn"]');
    await expect(exportMenuBtn).toBeVisible();
    await exportMenuBtn.click();
    await expect(page.locator('[data-testid="export-assembled-glb-btn"]')).toBeVisible();
    await expect(page.locator('[data-testid="export-metadata-json-btn"]')).toBeVisible();
    await expect(page.locator('[data-testid="export-portable-archive-btn"]')).toBeVisible();
  });
});
