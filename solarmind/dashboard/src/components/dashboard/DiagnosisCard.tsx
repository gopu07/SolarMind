import { useState, useEffect, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle, ShieldCheck, AlertOctagon, Info } from "lucide-react";
import { getInverterReport, type InverterReport } from "@/services/api";

interface DiagnosisCardProps {
  inverterId?: string;
}

function getRiskIcon(level: string) {
  if (level === "CRITICAL") return <AlertOctagon className="h-5 w-5 text-destructive" />;
  if (level === "HIGH") return <AlertTriangle className="h-5 w-5 text-warning" />;
  return <ShieldCheck className="h-5 w-5 text-primary" />;
}

const confidenceLookup: Record<InverterReport["confidence"], number> = {
  LOW: 0.4,
  MEDIUM: 0.7,
  HIGH: 0.9,
};

export function DiagnosisCard({ inverterId }: DiagnosisCardProps) {
  const [report, setReport] = useState<InverterReport | null>(null);

  useEffect(() => {
    if (!inverterId) return;
    getInverterReport(inverterId).then(setReport);
  }, [inverterId]);

  const confidencePct = useMemo(() => {
    if (!report) return 0;
    return Math.round((confidenceLookup[report.confidence] ?? 0.5) * 100);
  }, [report]);

  if (!report || !inverterId) return null;

  const drivers = (report.causal_drivers ?? []).map((d) => ({
    feature: d.feature,
    contribution: d.delta_shap,
  }));

  return (
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        {getRiskIcon(report.risk_level)}
        <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider">
          AI Diagnosis — {inverterId}
        </h3>
        <span className="ml-auto text-xs font-mono text-muted-foreground">
          Confidence: {confidencePct}%
        </span>
      </div>

      <div className="grid gap-3 mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground w-20">Risk Score</span>
          <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                report.risk_score > 0.85
                  ? "bg-destructive"
                  : report.risk_score > 0.6
                  ? "bg-warning"
                  : "bg-primary"
              }`}
              style={{ width: `${report.risk_score * 100}%` }}
            />
          </div>
          <span className="text-sm font-mono font-bold">
            {(report.risk_score * 100).toFixed(0)}%
          </span>
        </div>

        <div className="rounded-lg bg-muted/30 p-3">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-secondary mt-0.5 shrink-0" />
            <p className="text-sm text-muted-foreground leading-relaxed">
              {report.summary}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="rounded-lg bg-destructive/5 border border-destructive/10 p-3">
            <span className="text-[10px] uppercase tracking-wider text-destructive font-semibold">
              Root Cause
            </span>
            <p className="text-xs text-muted-foreground mt-1">{report.root_cause}</p>
          </div>
          <div className="rounded-lg bg-primary/5 border border-primary/10 p-3">
            <span className="text-[10px] uppercase tracking-wider text-primary font-semibold">
              Recommended Action
            </span>
            <p className="text-xs text-muted-foreground mt-1">{report.action}</p>
          </div>
        </div>
      </div>

      {drivers.length > 0 && (
        <>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Causal Drivers
          </h4>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={drivers} layout="vertical">
                <XAxis type="number" stroke="hsl(215, 20%, 55%)" fontSize={10} />
                <YAxis
                  dataKey="feature"
                  type="category"
                  stroke="hsl(215, 20%, 55%)"
                  fontSize={9}
                  width={130}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(220, 18%, 10%)",
                    border: "1px solid hsl(220, 14%, 18%)",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar
                  dataKey="contribution"
                  fill="hsl(210, 100%, 56%)"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
