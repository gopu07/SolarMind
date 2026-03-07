import { useEffect, useState } from "react";
import { Activity, CloudLightning, Radio, Timer } from "lucide-react";
import { useStore } from "@/store/useStore";
import { getSystemStatus, type SystemStatus } from "@/services/api";

interface DebugState {
  status: SystemStatus | null;
  latencyMs: number | null;
}

const WS_LABEL: Record<string, string> = {
  connecting: "Connecting",
  connected: "Connected",
  disconnected: "Disconnected",
  error: "Error",
};

export function DebugPanel() {
  const inverterMap = useStore((s) => s.inverterMap);
  const wsStatus = useStore((s) => s.wsStatus);
  const lastTelemetryTimestamp = useStore((s) => s.lastTelemetryTimestamp);

  const [debug, setDebug] = useState<DebugState>({ status: null, latencyMs: null });

  useEffect(() => {
    let cancelled = false;

    const fetchStatus = async () => {
      try {
        const { status, latencyMs } = await getSystemStatus();
        if (!cancelled) {
          setDebug({ status, latencyMs });
        }
      } catch (err) {
        console.error("[DEBUG] Failed to fetch /system/status", err);
      }
    };

    // Initial fetch and then poll every 15s
    fetchStatus();
    const id = setInterval(fetchStatus, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const inverterCount = Object.keys(inverterMap).length;
  const wsLabel = WS_LABEL[wsStatus] ?? wsStatus;

  const lastTs =
    lastTelemetryTimestamp ||
    debug.status?.last_broadcast ||
    null;

  return (
    <div className="flex items-center gap-2 rounded-full border border-border/40 bg-background/80 px-3 py-1 text-[10px] font-mono text-muted-foreground shadow-sm">
      <span className="inline-flex items-center gap-1">
        <Radio className={`h-3 w-3 ${wsStatus === "connected" ? "text-primary" : wsStatus === "error" ? "text-destructive" : "text-warning"}`} />
        <span>WS: {wsLabel}</span>
      </span>
      <span className="inline-flex items-center gap-1 border-l border-border/40 pl-2">
        <CloudLightning className="h-3 w-3 text-secondary" />
        <span>Inv: {inverterCount}</span>
      </span>
      <span className="hidden md:inline-flex items-center gap-1 border-l border-border/40 pl-2">
        <Activity className="h-3 w-3 text-emerald-400" />
        <span>
          API:{" "}
          {debug.latencyMs != null ? `${debug.latencyMs.toFixed(0)}ms` : "…"}
        </span>
      </span>
      <span className="hidden lg:inline-flex items-center gap-1 border-l border-border/40 pl-2">
        <Timer className="h-3 w-3 text-blue-400" />
        <span>
          TS:{" "}
          {lastTs ? new Date(lastTs).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—"}
        </span>
      </span>
    </div>
  );
}

