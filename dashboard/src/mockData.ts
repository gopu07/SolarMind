export const MOCK_HEATMAP = Array.from({ length: 40 }).map((_, i) => ({
    id: `INV_${(i + 1).toString().padStart(3, '0')}`,
    block: `B${Math.floor(i / 10) + 1}`,
    risk_level: Math.random() > 0.9 ? 'CRITICAL' : Math.random() > 0.7 ? 'HIGH' : Math.random() > 0.4 ? 'MEDIUM' : 'HEALTHY',
    risk_score: Math.random()
}));

export const MOCK_TRENDS = Array.from({ length: 48 }).map((_, i) => ({
    time: `T-${48 - i}h`,
    temperature: 40 + Math.random() * 20 + (i > 40 ? 15 : 0),
    efficiency: 0.95 - Math.random() * 0.1 - (i > 42 ? 0.2 : 0),
    cv: 0.1 + Math.random() * 0.1 + (i > 40 ? 0.2 : 0)
}));

export const MOCK_REPORT = {
    inverter_id: "INV_001",
    plant_id: "PLANT_1",
    risk_score: 0.88,
    risk_level: "HIGH",
    action: "Dispatch technician to inspect cooling system and check string connections.",
    summary: "Inverter INV_001 is exhibiting high risk due to elevated temperatures and power mismatch.",
    root_cause: "Likely cooling fan failure leading to thermal derating.",
    confidence: "HIGH",
    data_quality: "COMPLETE",
    causal_drivers: [
        { feature: "thermal_gradient", delta_shap: 0.15, direction: "UP" },
        { feature: "string_mismatch_cv", delta_shap: 0.12, direction: "UP" },
        { feature: "ambient_temp", delta_shap: 0.05, direction: "UP" },
        { feature: "conversion_efficiency", delta_shap: -0.08, direction: "DOWN" }
    ]
};
