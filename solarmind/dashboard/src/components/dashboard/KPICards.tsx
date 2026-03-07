import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Server, ShieldCheck, AlertTriangle, AlertOctagon, Clock, Wrench } from "lucide-react";
import { getTimeline, getMaintenanceSchedule } from "@/services/api";
import type { Inverter } from "@/data/mockData";

interface KPICardsProps {
  inverters: Inverter[];
}

export function KPICards({ inverters }: KPICardsProps) {
  const [predictedFailures24h, setPredictedFailures24h] = useState(0);
  const [scheduledTasks, setScheduledTasks] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [timeline, maintenance] = await Promise.all([
          getTimeline(),
          getMaintenanceSchedule()
        ]);
        setPredictedFailures24h(timeline.filter(t => t.predicted_failure_hours <= 24).length);
        setScheduledTasks(maintenance.length);
      } catch (err) {
        console.error("Failed to fetch Phase 2.5 KPIs", err);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const total = inverters.length;
  const healthy = inverters.filter(i => i.status === "healthy").length;
  const highRisk = inverters.filter(i => i.status === "high_risk" || i.status === "warning").length;
  const critical = inverters.filter(i => i.status === "critical").length;

  const cards = [
    { label: "Total Inverters", value: total, icon: Server, color: "text-secondary", bg: "bg-secondary/10", border: "border-secondary/20" },
    { label: "Healthy", value: healthy, icon: ShieldCheck, color: "text-primary", bg: "bg-primary/10", border: "border-primary/20" },
    { label: "High Risk", value: highRisk, icon: AlertTriangle, color: "text-warning", bg: "bg-warning/10", border: "border-warning/20" },
    { label: "Predicted <24h", value: predictedFailures24h, icon: Clock, color: "text-blue-400", bg: "bg-blue-400/10", border: "border-blue-400/20" },
    { label: "Tasks", value: scheduledTasks, icon: Wrench, color: "text-indigo-400", bg: "bg-indigo-400/10", border: "border-indigo-400/20" },
    { label: "Critical", value: critical, icon: AlertOctagon, color: "text-destructive", bg: "bg-destructive/10", border: "border-destructive/20", glow: critical > 0 },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-6 gap-4">
      {cards.map((card, i) => (
        <motion.div
          key={card.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05, type: "spring", stiffness: 100 }}
          whileHover={{ y: -5, transition: { duration: 0.2 } }}
          className={`glass-card p-5 border ${card.border} ${card.glow ? "neon-glow-red animate-pulse-glow" : ""} relative overflow-hidden group`}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="flex items-center justify-between mb-4 relative z-10">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-[0.2em]">{card.label}</span>
            <div className={`p-2 rounded-xl ${card.bg} border border-white/5 shadow-inner`}>
              <card.icon className={`h-4 w-4 ${card.color}`} />
            </div>
          </div>
          <div className="relative z-10 flex items-baseline gap-1">
            <div className={`text-4xl font-black font-mono tracking-tighter ${card.color} neon-text`}>
              {card.value}
            </div>
          </div>
          <div className={`absolute bottom-0 left-0 w-full h-1 ${card.color.replace("text-", "bg-")} opacity-20`} />
        </motion.div>
      ))}
    </div>
  );
}
