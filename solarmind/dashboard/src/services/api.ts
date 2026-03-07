import axios from "axios";
import {
  mockInverters,
  generateTelemetry,
  generateReport,
  generateSHAPValues,
  generateDeltaSHAP,
  mockChatResponses,
  type Inverter,
  type TelemetryPoint,
  type DiagnosticReport,
  type SHAPValue,
  type DeltaSHAPValue,
} from "@/data/mockData";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
  headers: { "Content-Type": "application/json" },
});

const useMock = false;

export async function getInverters(): Promise<Inverter[]> {
  if (useMock) return mockInverters;
  try {
    const { data } = await api.get("/inverters");
    console.log("[API] /inverters returned", data.length, "inverters");
    return data;
  } catch {
    // FIX: Do NOT fall back to mock data — return empty array.
    // The WebSocket will populate real inverters within seconds.
    console.warn("[API] /inverters failed — starting with empty state, WebSocket will populate");
    return [];
  }
}

export async function getInverterReport(id: string): Promise<DiagnosticReport | null> {
  if (useMock) return generateReport(id);
  try {
    const { data } = await api.get(`/inverters/${id}/report`);
    return data;
  } catch (err) {
    console.error("[API] getInverterReport failed", err);
    return null;
  }
}

export async function getTelemetry(id: string): Promise<TelemetryPoint[]> {
  if (useMock) return generateTelemetry(id);
  try {
    const { data } = await api.get(`/inverters/${id}/trends`);
    return data;
  } catch (err) {
    console.error("[API] getTelemetry failed", err);
    return [];
  }
}

export async function getSHAPValues(id: string): Promise<SHAPValue[]> {
  if (useMock) return generateSHAPValues(id);
  try {
    const { data } = await api.get(`/inverters/${id}/shap`);
    return data;
  } catch (err) {
    console.error("[API] getSHAPValues failed", err);
    return [];
  }
}

export async function getDeltaSHAP(id: string): Promise<DeltaSHAPValue[]> {
  if (useMock) return generateDeltaSHAP(id);
  try {
    const { data } = await api.get(`/inverters/${id}/delta-shap`);
    return data;
  } catch (err) {
    console.error("[API] getDeltaSHAP failed", err);
    return [];
  }
}

export async function queryAI(question: string, sessionId: string = "default"): Promise<string> {
  if (useMock) {
    await new Promise(r => setTimeout(r, 800 + Math.random() * 1200));
    const q = question.toLowerCase();
    if (q.includes("thermal") || q.includes("temperature") || q.includes("heat")) return mockChatResponses.thermal;
    if (q.includes("risk") || q.includes("week") || q.includes("at risk")) return mockChatResponses.risk;
    if (q.includes("inv-14") || q.includes("inverter 14")) return mockChatResponses["inv-14"];
    return mockChatResponses.default;
  }
  try {
    const { data } = await api.post("/query", { question, session_id: sessionId });
    return data.response || data.answer || JSON.stringify(data);
  } catch (err) {
    console.error("[API] queryAI failed", err);
    return "AI Assistant is currently unavailable.";
  }
}

export async function predictInverter(id: string): Promise<{ risk_score: number; risk_level: string }> {
  if (useMock) {
    const inv = mockInverters.find(i => i.id === id);
    return { risk_score: inv?.risk_score || 0.5, risk_level: inv?.risk_level || "MEDIUM" };
  }
  const { data } = await api.post("/predict", { inverter_id: id });
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
    const { data } = await api.get("/timeline");
    return data;
  } catch {
    return [];
  }
}

export async function getMaintenanceSchedule(): Promise<MaintenanceTask[]> {
  try {
    const { data } = await api.get("/maintenance_schedule");
    return data;
  } catch {
    return [];
  }
}
