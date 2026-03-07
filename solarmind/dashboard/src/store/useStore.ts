import { create } from "zustand";
import { getInverters } from "@/services/api";
import { wsService, type WSStatus } from "@/services/websocket";
import type { Inverter } from "@/data/mockData";

interface AppState {
  inverterMap: Record<string, Inverter>;
  isLoading: boolean;
  error: string | null;
  isInitialized: boolean;
  selectedInverterId: string | null;
  wsStatus: WSStatus;
  lastTelemetryTimestamp: string | null;
  initialize: () => Promise<void>;
  setSelectedInverterId: (id: string) => void;
}

export const useStore = create<AppState>((set, get) => ({
  inverterMap: {},
  isLoading: true,
  error: null,
  isInitialized: false,
  selectedInverterId: null,
  wsStatus: "disconnected",
  lastTelemetryTimestamp: null,

  setSelectedInverterId: (id: string) => set({ selectedInverterId: id }),

  initialize: async () => {
    if (get().isInitialized) return;
    set({ isInitialized: true, isLoading: true });

    try {
      const data = await getInverters();
      const initialMap: Record<string, Inverter> = {};
      data.forEach((inv) => (initialMap[inv.id] = inv));
      const firstId = Object.keys(initialMap).sort()[0] ?? null;
      console.log("[STORE] Initialized from /inverters", {
        count: data.length,
        firstId,
      });
      set({ inverterMap: initialMap, isLoading: false, selectedInverterId: firstId });
    } catch (err) {
      console.warn("[STORE] Init failed, starting with empty map", err);
      set({ inverterMap: {}, isLoading: false });
    }

    wsService.connect();

    // Track websocket connection status
    wsService.onStatus((status) => {
      set({ wsStatus: status });
    });

    // Stream inverter updates into the global map
    wsService.subscribe((incoming) => {
      set((state) => {
        const next = { ...state.inverterMap };
        incoming.forEach((inv) => {
          next[inv.id] = inv;
        });
        const lastTs = incoming.length > 0 ? incoming[0].last_updated : state.lastTelemetryTimestamp;
        return { inverterMap: next, lastTelemetryTimestamp: lastTs ?? state.lastTelemetryTimestamp };
      });
    });
  },
}));
