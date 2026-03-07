import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle2, Activity } from "lucide-react";
import axios from "axios";

interface DriftSignal {
    baseline_mean: number;
    current_mean: number;
    z_score: number;
    status: "stable" | "drifted";
}

interface DriftData {
    status: string;
    overall_drift_score: number;
    drift_detected: boolean;
    timestamp: string;
    signals: Record<string, DriftSignal>;
}

export function DriftPanel() {
    const [data, setData] = useState<DriftData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchDrift = async () => {
            try {
                const response = await axios.get(`${import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}/model/drift`);
                setData(response.data);
            } catch (error) {
                console.error("Failed to fetch drift data", error);
            } finally {
                setLoading(false);
            }
        };

        fetchDrift();
        const interval = setInterval(fetchDrift, 30000); // Poll every 30s
        return () => clearInterval(interval);
    }, []);

    if (loading) return <Card className="p-4">Loading drift analysis...</Card>;
    if (!data) return null;

    return (
        <Card className="border-border/40 bg-muted/20 backdrop-blur-sm">
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <div className="space-y-1">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                            <Activity className="h-4 w-4 text-primary" />
                            Model Drift Monitor
                        </CardTitle>
                        <CardDescription className="text-xs">
                            Feature Distribution Analysis vs Baseline
                        </CardDescription>
                    </div>
                    <Badge variant={data.drift_detected ? "destructive" : "secondary"} className="font-mono">
                        {data.drift_detected ? "DRIFT DETECTED" : "STABLE"}
                    </Badge>
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-end justify-between">
                    <div className="text-2xl font-bold font-mono">
                        {data.overall_drift_score.toFixed(2)}
                        <span className="text-xs text-muted-foreground ml-1 font-normal">avg Z-score</span>
                    </div>
                    <div className="text-[10px] text-muted-foreground font-mono">
                        Last Checked: {new Date(data.timestamp).toLocaleTimeString()}
                    </div>
                </div>

                <div className="space-y-2">
                    {Object.entries(data.signals).map(([name, signal]) => (
                        <div key={name} className="flex items-center justify-between text-xs p-2 rounded-md bg-background/50 border border-border/20">
                            <span className="font-mono text-muted-foreground capitalize">
                                {name.replace(/_/g, " ")}
                            </span>
                            <div className="flex items-center gap-3">
                                <span className="font-mono tabular-nums">
                                    Z: {signal.z_score.toFixed(2)}
                                </span>
                                {signal.status === "stable" ? (
                                    <CheckCircle2 className="h-3 w-3 text-emerald-500" />
                                ) : (
                                    <AlertTriangle className="h-3 w-3 text-destructive animate-pulse" />
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
