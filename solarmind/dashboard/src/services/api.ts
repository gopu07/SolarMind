import axios from "axios";
import {
  mockInverters,
  generateTelemetry,
  generateSHAPValues,
  generateDeltaSHAP,
  mockChatResponses,
  type Inverter,
  type TelemetryPoint,
  type SHAPValue,
  type DeltaSHAPValue,
} from "@/data/mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 45000,
  headers: { "Content-Type": "application/json" },
});

const useMock = false;

export interface SystemStatus {
  api_status: string;
  model_loaded: boolean;
  telemetry_rows: number;
  inverter_count: number;
  last_broadcast: string | null;
  assistant_status: string;
}

export interface InverterReport {
  inverter_id: string;
  plant_id: string;
  risk_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  summary: string;
  root_cause: string;
  action: string;
  confidence: "LOW" | "MEDIUM" | "HIGH";
  data_quality: string;
  delta_shap_available: boolean;
  causal_drivers?: { feature: string; delta_shap: number; direction: "UP" | "DOWN" }[] | null;
  data_gap_note?: string | null;
}

export interface QueryDiagnosticReport {
  diagnosis: string;
  risk_level: string;
  root_cause_hypothesis: string;
  confidence: string;
  data_quality: string;
  recommended_actions: { action: string; priority: string; justification?: string }[];
}

export interface QueryResponse {
  answer: string;
  diagnostic_report?: QueryDiagnosticReport | null;
  latency_ms: number;
}

export async function getSystemStatus(): Promise<{ status: SystemStatus; latencyMs: number }> {
  const start = performance.now();
  const { data } = await api.get<SystemStatus>("/system/status");
  const latencyMs = performance.now() - start;
  console.log("[API] /system/status", data, `${latencyMs.toFixed(1)}ms`);
  return { status: data, latencyMs };
}

export async function getInverters(): Promise<Inverter[]> {
  if (useMock) return mockInverters;
  try {
    const { data } = await api.get<Inverter[]>("/inverters");
    console.log("[API] /inverters returned", data.length, "inverters");
    return data;
  } catch (err) {
    console.warn("[API] /inverters failed — starting with empty state, WebSocket will populate", err);
    return [];
  }
}

export async function getInverterReport(id: string): Promise<InverterReport | null> {
  if (useMock) {
    console.warn("[API] getInverterReport using mock data");
    // When mocking, we don't have the full backend schema; return null rather than mixing types.
    return null;
  }
  try {
    const { data } = await api.get<InverterReport>(`/inverters/${id}/report`);
    console.log("[API] /inverters/{id}/report", { id, risk_score: data.risk_score, risk_level: data.risk_level });
    return data;
  } catch (err) {
    console.error("[API] getInverterReport failed", err);
    return null;
  }
}

export async function getTelemetry(id: string): Promise<TelemetryPoint[]> {
  if (useMock) return generateTelemetry(id);
  try {
    const { data } = await api.get<TelemetryPoint[]>(`/inverters/${id}/trends`);
    console.log("[API] /inverters/{id}/trends returned", data.length, "points for", id);
    return data;
  } catch (err) {
    console.error("[API] getTelemetry failed", err);
    return [];
  }
}

export async function getSHAPValues(id: string): Promise<SHAPValue[]> {
  if (useMock) return generateSHAPValues(id);
  try {
    const { data } = await api.get<SHAPValue[]>(`/inverters/${id}/shap`);
    console.log("[API] /inverters/{id}/shap returned", data.length, "features for", id);
    return data;
  } catch (err) {
    console.error("[API] getSHAPValues failed", err);
    return [];
  }
}

export async function getDeltaSHAP(id: string): Promise<DeltaSHAPValue[]> {
  if (useMock) return generateDeltaSHAP(id);
  try {
    const { data } = await api.get<DeltaSHAPValue[]>(`/inverters/${id}/delta-shap`);
    console.log("[API] /inverters/{id}/delta-shap returned", data.length, "features for", id);
    return data;
  } catch (err) {
    console.error("[API] getDeltaSHAP failed", err);
    return [];
  }
}

export async function queryAI(
  question: string,
  sessionId: string = "default",
  opts?: { inverterId?: string; plantId?: string }
): Promise<QueryResponse> {
  if (useMock) {
    await new Promise((r) => setTimeout(r, 800 + Math.random() * 1200));
    const q = question.toLowerCase();
    let answer: string;
    if (q.includes("thermal") || q.includes("temperature") || q.includes("heat")) answer = mockChatResponses.thermal;
    else if (q.includes("risk") || q.includes("week") || q.includes("at risk")) answer = mockChatResponses.risk;
    else if (q.includes("inv-14") || q.includes("inverter 14")) answer = mockChatResponses["inv-14"];
    else answer = mockChatResponses.default;

    return { answer, diagnostic_report: undefined, latency_ms: 0 };
  }

  try {
    let payload: any = { question, session_id: sessionId };

    // If we have inverter context but the question doesn't mention it, prepend a hint
    if (opts?.inverterId && !question.toUpperCase().match(/INV[-_]\d+/)) {
      payload.question = `For inverter ${opts.inverterId}, ${question}`;
    }
    if (opts?.plantId) {
      payload.plant_id = opts.plantId;
    }

    const { data } = await api.post<QueryResponse>("/query", payload);
    console.log("[API] /query", { latency_ms: data.latency_ms, hasDiagnostic: !!data.diagnostic_report });
    return data;
  } catch (err) {
    console.error("[API] queryAI failed", err);
    return {
      answer: "Error calling AI assistant. Please check backend logs for details.",
      diagnostic_report: undefined,
      latency_ms: 0,
    };
  }
}

export async function predictInverter(id: string): Promise<{ risk_score: number; risk_level: string }> {
  if (useMock) {
    const inv = mockInverters.find((i) => i.id === id);
    return { risk_score: inv?.risk_score || 0.5, risk_level: inv?.risk_level || "MEDIUM" };
  }
  const { data } = await api.post<{ risk_score: number; risk_level: string }>("/predict", { inverter_id: id });
  return data;
}

export interface Alert {
  id: string;
  inverter_id: string;
  plant_id: string;
  risk_score: number;
  level: "warning" | "critical";
  message: string;
  timestamp: string;
}

export interface Ticket {
  id: string;
  inverter_id: string;
  plant_id: string;
  risk_score: number;
  suspected_issue: string;
  recommended_action: string;
  status: string;
  created_at: string;
}

export async function getAlerts(): Promise<Alert[]> {
  try {
    const { data } = await api.get("/alerts");
    return data;
  } catch {
    return [];
  }
}

export async function getTickets(): Promise<Ticket[]> {
  try {
    const { data } = await api.get("/tickets");
    return data;
  } catch {
    return [];
  }
}

export interface TimelineEvent {
  inverter_id: string;
  predicted_failure_time: string;
  predicted_failure_hours: number;
  risk_score: number;
  failure_type: string;
}

export interface MaintenanceTask {
  maintenance_id: string;
  inverter_id: string;
  recommended_time: string;
  priority: "CRITICAL" | "HIGH" | "MEDIUM";
  recommended_action: string;
}

export async function getTimeline(): Promise<TimelineEvent[]> {
  try {
    const { data } = await api.get<TimelineEvent[]>("/timeline");
    console.log("[API] /timeline returned", data.length, "events");
    return data;
  } catch {
    return [];
  }
}

export async function getMaintenanceSchedule(): Promise<MaintenanceTask[]> {
  try {
    const { data } = await api.get<MaintenanceTask[]>("/maintenance_schedule");
    console.log("[API] /maintenance_schedule returned", data.length, "tasks");
    return data;
  } catch {
    return [];
  }
}
