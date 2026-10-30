import { describe, it, expect } from "vitest";
import { ProcessingEvent, AssetReference } from "../src/index.js";

describe("Processing Event Types and Contracts", () => {
  it("structures a valid processing event", () => {
    const assetRef: AssetReference = {
      assetId: "photo-001",
      assetType: "original_photo",
      uri: "/api/v1/assets/photo-001",
      mimeType: "image/jpeg",
      sizeBytes: 1048576
    };

    const event: ProcessingEvent = {
      eventId: "evt-12345",
      jobId: "job-67890",
      sceneId: "scene-abcde",
      eventType: "asset_ready",
      timestamp: new Date().toISOString(),
      sequenceNumber: 1,
      stageIdentifier: "image_analysis",
      status: "running",
      progress: 0.1,
      message: "Photograph stored and validated in local storage",
      assetReference: assetRef
    };

    expect(event.eventId).toBe("evt-12345");
    expect(event.sequenceNumber).toBe(1);
    expect(event.stageIdentifier).toBe("image_analysis");
    expect(event.assetReference?.assetType).toBe("original_photo");
  });

  it("handles stage failure event structure", () => {
    const errorEvent: ProcessingEvent = {
      eventId: "evt-err-1",
      jobId: "job-failed-1",
      sceneId: "scene-err",
      eventType: "stage_failed",
      timestamp: new Date().toISOString(),
      sequenceNumber: 2,
      stageIdentifier: "person_detection",
      status: "failed",
      error: {
        code: "NO_PEOPLE_DETECTED",
        message: "No human subjects were detected with sufficient confidence in this image",
        stage: "person_detection"
      }
    };

    expect(errorEvent.status).toBe("failed");
    expect(errorEvent.error?.code).toBe("NO_PEOPLE_DETECTED");
  });
});

