import { create } from "zustand";
import { getInverters } from "@/services/api";
import { wsService } from "@/services/websocket";
import type { Inverter } from "@/data/mockData";

interface AppState {
    inverterMap: Record<string, Inverter>;
    isLoading: boolean;
    error: string | null;
    isInitialized: boolean;
    initialize: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
    inverterMap: {},
    isLoading: true,
    error: null,
    isInitialized: false,

    initialize: async () => {
        if (get().isInitialized) return;
        set({ isInitialized: true, isLoading: true });

        try {
            const data = await getInverters();
            const initialMap: Record<string, Inverter> = {};
            data.forEach((inv) => (initialMap[inv.id] = inv));
            set({ inverterMap: initialMap, isLoading: false });
        } catch (err) {
            console.warn("[STORE] Init failed, starting with empty map");
            set({ inverterMap: {}, isLoading: false });
        }

        wsService.connect();
        wsService.subscribe((incoming) => {
            set((state) => {
                const next = { ...state.inverterMap };
                incoming.forEach((inv) => {
                    next[inv.id] = inv;
                });
                return { inverterMap: next };
            });
        });
    },
}));
