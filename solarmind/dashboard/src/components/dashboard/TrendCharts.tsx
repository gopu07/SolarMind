import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getTelemetry } from "@/services/api";
import type { TelemetryPoint } from "@/data/mockData";

interface TrendChartsProps {
  inverterId?: string;
}

export function TrendCharts({ inverterId = "INV-14" }: TrendChartsProps) {
  const [data, setData] = useState<TelemetryPoint[]>([]);
  const [activeChart, setActiveChart] = useState<"temperature" | "efficiency" | "string_mismatch">("temperature");

  useEffect(() => {
    if (!inverterId) return;
    getTelemetry(inverterId).then(setData);
  }, [inverterId]);

  const chartConfig = {
    temperature: { label: "Temperature (°C)", color: "hsl(0, 84%, 60%)", key: "temperature" as const },
    efficiency: { label: "Efficiency (%)", color: "hsl(142, 72%, 50%)", key: "efficiency" as const },
    string_mismatch: { label: "String Mismatch", color: "hsl(210, 100%, 56%)", key: "string_mismatch" as const },
  };

  const config = chartConfig[activeChart];
  const formatted = data.map(d => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  }));

  return (
    <div className="glass-card p-6 shadow-2xl relative overflow-hidden group">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h3 className="text-sm font-black text-foreground uppercase tracking-widest flex items-center gap-2">
            <div className="h-4 w-1 bg-secondary rounded-full" />
            Telemetry <span className="text-secondary opacity-70">— {inverterId}</span>
          </h3>
          <p className="text-[10px] text-muted-foreground font-mono mt-1 uppercase">Historical Performance Patterns</p>
        </div>
        <div className="flex bg-muted/30 p-1 rounded-xl border border-border/40">
          {Object.entries(chartConfig).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setActiveChart(key as typeof activeChart)}
              className={`px-4 py-2 text-[10px] rounded-lg font-bold transition-all ${activeChart === key
                ? "bg-secondary text-secondary-foreground shadow-lg"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
            >
              {cfg.label.split(" (")[0].toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formatted} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={config.color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={config.color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.2)" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="hsl(var(--muted-foreground))"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              interval={Math.floor(formatted.length / 6)}
              dy={10}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(val) => `${val}${activeChart === "efficiency" ? "%" : ""}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "12px",
                fontSize: "12px",
                boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
              }}
              itemStyle={{ color: config.color, fontWeight: "bold" }}
            />
            <Line
              type="monotone"
              dataKey={config.key}
              stroke={config.color}
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, stroke: "white", strokeWidth: 2, fill: config.color }}
              animationDuration={1500}
              animationEasing="ease-in-out"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
