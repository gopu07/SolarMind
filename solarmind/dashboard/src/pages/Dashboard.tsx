import { motion } from "framer-motion";
import { KPICards } from "@/components/dashboard/KPICards";
import { RiskHeatmap } from "@/components/dashboard/RiskHeatmap";
import { TrendCharts } from "@/components/dashboard/TrendCharts";
import { DiagnosisCard } from "@/components/dashboard/DiagnosisCard";
import { AlertPanel } from "@/components/dashboard/AlertPanel";
import { DriftPanel } from "@/components/dashboard/DriftPanel";
import { TicketPanel } from "@/components/dashboard/TicketPanel";
import { FailureTimeline } from "@/components/dashboard/FailureTimeline";
import { MaintenanceSchedule } from "@/components/dashboard/MaintenanceSchedule";
import { AppLayout } from "@/components/layout/AppLayout";

import { useInverters } from "@/hooks/useInverters";

export default function Dashboard() {
  const { inverters, isLoading, error, selectedInverterId, setSelectedInverterId } = useInverters();

  if (isLoading) return <div className="flex h-screen items-center justify-center font-mono text-primary">Loading monitoring data...</div>;
  if (error) return <div className="flex h-screen items-center justify-center text-destructive">{error}</div>;


  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="p-4 lg:p-6 space-y-6"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card/30 p-6 rounded-2xl border border-border/20 backdrop-blur-md shadow-2xl">
          <div className="space-y-1">
            <h1 className="text-3xl font-black tracking-tight text-foreground">
              Plant <span className="text-primary neon-text">Overview</span>
            </h1>
            <div className="flex items-center gap-2 text-sm text-muted-foreground font-mono">
              <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
              Solar Farm Alpha · Real-time Monitoring
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-muted-foreground hidden sm:inline">SELECT UNIT:</span>
            <select
              value={selectedInverterId ?? ""}
              onChange={(e) => setSelectedInverterId(e.target.value)}
              className="rounded-xl border border-border/40 bg-muted/30 px-4 py-2 text-sm text-foreground font-mono focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all cursor-pointer hover:bg-muted/50"
            >
              {inverters.map(inv => (
                <option key={inv.id} value={inv.id} className="bg-background">
                  {inv.id} — {inv.risk_level}
                </option>
              ))}
            </select>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <KPICards inverters={inverters} />
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <RiskHeatmap inverters={inverters} onSelect={setSelectedInverterId} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="space-y-6"
          >
            <DriftPanel />
            <div className="h-[280px] overflow-hidden">
              <AlertPanel />
            </div>
          </motion.div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-2 space-y-6"
          >
            <TrendCharts inverterId={selectedInverterId ?? undefined} />
            <DiagnosisCard inverterId={selectedInverterId ?? undefined} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="h-[650px]"
          >
            <TicketPanel />
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="grid lg:grid-cols-3 gap-6"
        >
          <div className="lg:col-span-1">
            <FailureTimeline />
          </div>
          <div className="lg:col-span-2">
            <MaintenanceSchedule />
          </div>
        </motion.div>
      </motion.div>
    </AppLayout>
  );
}
