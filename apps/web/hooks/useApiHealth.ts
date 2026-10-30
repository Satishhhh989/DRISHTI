"use client";

import { useEffect } from "react";
import { apiClient } from "@/lib/api-client";
import { useUIStore } from "@/stores/ui-store";

export function useApiHealth() {
  const setApiConnected = useUIStore((state) => state.setApiConnected);
  const apiConnected = useUIStore((state) => state.apiConnected);

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        const health = await apiClient.getHealth();
        if (mounted) {
          setApiConnected(health.status === "ok");
        }
      } catch {
        if (mounted) {
          setApiConnected(false);
        }
      }
    }

    check();
    const interval = setInterval(check, 8000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [setApiConnected]);

  return { apiConnected };
}

