import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import type { Inverter } from "@/data/mockData";

interface RiskHeatmapProps {
  inverters: Inverter[];
  onSelect?: (id: string) => void;
}

function getRiskColor(score: number) {
  if (score >= 0.8) return { bg: "bg-destructive/20", border: "border-destructive/50", text: "text-destructive" };
  if (score >= 0.6) return { bg: "bg-warning/20", border: "border-warning/50", text: "text-warning" };
  if (score >= 0.3) return { bg: "bg-yellow-500/20", border: "border-yellow-500/40", text: "text-yellow-400" };
  return { bg: "bg-primary/20", border: "border-primary/40", text: "text-primary" };
}

export function RiskHeatmap({ inverters, onSelect }: RiskHeatmapProps) {
  const navigate = useNavigate();

  return (
    <div className="glass-card p-6 shadow-2xl relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
        <Activity className="h-24 w-24 text-primary" />
      </div>

      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-sm font-black text-foreground uppercase tracking-widest flex items-center gap-2">
            <div className="h-1 w-4 bg-primary rounded-full" />
            Risk Heatmap
          </h3>
          <p className="text-[10px] text-muted-foreground font-mono mt-1">REAL-TIME INVERTER ANOMALY TRACKING</p>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground bg-muted/20 px-3 py-1.5 rounded-full border border-border/40">
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-primary" /> LOW</span>
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-yellow-400" /> MED</span>
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-warning" /> HIGH</span>
          <span className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-destructive" /> CRIT</span>
        </div>
      </div>

      <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
        {inverters.map((inv, i) => {
          const colors = getRiskColor(inv.risk_score);
          return (
            <motion.button
              key={inv.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.02, type: "spring", stiffness: 200 }}
              whileHover={{ scale: 1.05, zIndex: 10 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                if (onSelect) {
                  onSelect(inv.id);
                } else {
                  navigate(`/inverter/${inv.id}`);
                }
              }}
              className={`relative p-3 rounded-xl border-2 ${colors.bg} ${colors.border} cursor-pointer transition-all hover:shadow-[0_0_20px_rgba(var(--primary),0.2)] group/cell overflow-hidden`}
            >
              <div className="absolute inset-0 bg-white/5 opacity-0 group-hover/cell:opacity-100 transition-opacity" />
              <div className={`text-[9px] font-mono font-bold ${colors.text} opacity-70 mb-1`}>{inv.id.replace("INV_", "")}</div>
              <div className={`text-xl font-black font-mono tracking-tighter ${colors.text}`}>
                {(inv.risk_score * 100).toFixed(0)}<span className="text-[10px] font-normal opacity-60">%</span>
              </div>
              {inv.risk_score >= 0.8 && (
                <div className="absolute top-2 right-2 h-2 w-2 rounded-full bg-destructive animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]" />
              )}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
