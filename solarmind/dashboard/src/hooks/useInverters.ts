import { useEffect, useMemo } from "react";
import { useStore } from "@/store/useStore";

export function useInverters() {
  const inverterMap = useStore((state) => state.inverterMap);
  const isLoading = useStore((state) => state.isLoading);
  const error = useStore((state) => state.error);
  const initialize = useStore((state) => state.initialize);
  const selectedInverterId = useStore((state) => state.selectedInverterId);
  const setSelectedInverterId = useStore((state) => state.setSelectedInverterId);

  useEffect(() => {
    initialize();
  }, [initialize]);

  const inverters = useMemo(() => {
    return Object.values(inverterMap).sort((a, b) => a.id.localeCompare(b.id));
  }, [inverterMap]);

  // Ensure a selected inverter once data arrives
  useEffect(() => {
    if (!selectedInverterId && inverters.length > 0) {
      setSelectedInverterId(inverters[0].id);
    }
  }, [selectedInverterId, inverters, setSelectedInverterId]);

  return { inverters, inverterMap, isLoading, error, selectedInverterId, setSelectedInverterId };
}
