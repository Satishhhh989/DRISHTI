import { test, expect } from "@playwright/test";

test.describe("Phase 01: Workspace & Viewport E2E Verification", () => {
  test("1. The application opens and renders workspace", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('[data-testid="workspace-root"]')).toBeVisible();
    await expect(page.getByText("PHOTO → 3D ENGINE")).toBeVisible();
    await expect(page.getByText("Phase 01")).toBeVisible();
  });

  test("2. The upload control is visible in top bar and empty viewport", async ({ page }) => {
    await page.goto("/");
    // Upload button in top bar
    const topBarUploadBtn = page.getByRole("button", { name: /Upload Photo/i });
    await expect(topBarUploadBtn).toBeVisible();

    // Empty state viewport card and button
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

    // Return to Photo mode
    const photoTab = page.locator('[data-testid="mode-photo"]');
    await photoTab.click();
    await expect(photoTab).toHaveAttribute("aria-pressed", "true");
    await expect(wireframeNotice).not.toBeVisible();
    await expect(solidNotice).not.toBeVisible();
  });
});
