import "@testing-library/jest-dom";
import { vi } from "vitest";

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock URL.createObjectURL and revokeObjectURL
if (typeof window !== "undefined") {
  window.URL.createObjectURL = vi.fn(() => "blob:http://localhost:3000/mock-uuid");
  window.URL.revokeObjectURL = vi.fn();
}

