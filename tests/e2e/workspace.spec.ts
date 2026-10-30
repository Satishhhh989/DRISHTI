import { test, expect } from "@playwright/test";

test.describe("Phase 01 & 02: Workspace & Viewport E2E Verification", () => {
  test("1. The application opens and renders workspace", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('[data-testid="workspace-root"]')).toBeVisible();
    await expect(page.getByText("PHOTO → 3D ENGINE")).toBeVisible();
    await expect(page.getByText(/Phase 0[1-3]/)).toBeVisible();
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
    await expect(wireframeNotice).toContainText("Phase 06");

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
});
