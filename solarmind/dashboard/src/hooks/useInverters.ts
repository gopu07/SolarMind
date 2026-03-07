import { useEffect, useMemo } from "react";
import { useStore } from "@/store/useStore";

export function useInverters() {
    const inverterMap = useStore((state) => state.inverterMap);
    const isLoading = useStore((state) => state.isLoading);
    const error = useStore((state) => state.error);
    const initialize = useStore((state) => state.initialize);

    useEffect(() => {
        initialize();
    }, [initialize]);

    const inverters = useMemo(() => {
        return Object.values(inverterMap).sort((a, b) => a.id.localeCompare(b.id));
    }, [inverterMap]);

    return { inverters, inverterMap, isLoading, error };
}
