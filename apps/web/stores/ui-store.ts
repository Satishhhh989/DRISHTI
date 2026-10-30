/**
 * PHOTO → 3D ENGINE - UI State Store (Zustand)
 * Manages viewport overlays, selections, and transient UI notifications.
 */

import { create } from "zustand";

interface UIState {
  selectedEntityId: string | null;
  isInspectorOpen: boolean;
  showGrid: boolean;
  cameraResetTrigger: number;
  apiConnected: boolean;
  errorMessage: string | null;
  statusNotice: string | null;

  // Actions
  selectEntity: (id: string | null) => void;
  toggleInspector: () => void;
  setInspectorOpen: (open: boolean) => void;
  toggleGrid: () => void;
  resetCamera: () => void;
  setApiConnected: (connected: boolean) => void;
  setError: (msg: string | null) => void;
  setStatusNotice: (notice: string | null) => void;
}

export const useUIStore = create<UIState>((set) => ({
  selectedEntityId: null,
  isInspectorOpen: true,
  showGrid: true,
  cameraResetTrigger: 0,
  apiConnected: false,
  errorMessage: null,
  statusNotice: null,

  selectEntity: (id) => set({ selectedEntityId: id }),
  toggleInspector: () => set((state) => ({ isInspectorOpen: !state.isInspectorOpen })),
  setInspectorOpen: (open) => set({ isInspectorOpen: open }),
  toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),
  resetCamera: () => set((state) => ({ cameraResetTrigger: state.cameraResetTrigger + 1 })),
  setApiConnected: (connected) => set({ apiConnected: connected }),
  setError: (msg) => set({ errorMessage: msg }),
  setStatusNotice: (notice) => set({ statusNotice: notice }),
}));
