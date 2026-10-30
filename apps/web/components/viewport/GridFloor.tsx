"use client";

import React from "react";
import { useUIStore } from "@/stores/ui-store";

export const GridFloor: React.FC = () => {
  const showGrid = useUIStore((state) => state.showGrid);

  if (!showGrid) return null;

  return (
    <group position={[0, -0.001, 0]}>
      {/* 10m x 10m ground grid with 10 subdivisions (1m cells) */}
      <gridHelper args={[10, 10, 0x52525b, 0x27272a]} />
    </group>
  );
};

