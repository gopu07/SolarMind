# SolarMind Predictive Maintenance: Technical Report

## 1. Executive Summary
SolarMind is an industrial AI platform designed to transform raw solar array telemetry into actionable, predictive maintenance tickets. By unpivoting complex multidimensional datalogger arrays into localized inverter streams, the platform accurately models equipment degradation and calculates component-level risk scores.

## 2. Machine Learning Rationale & Architecture
### 2.1 Model Selection: Heterogeneous Ensemble
We selected a **weighted ensemble ensemble** consisting of XGBoost, LightGBM, and CatBoost as the core classifier to model the `will_fail_24h` target variable.
- **Ensemble Composition:** XGBoost (50%), LightGBM (30%), and CatBoost (20%). This blend leverages the strengths of each algorithm: XGBoost's structured performance, LightGBM's speed and leaf-wise growth, and CatBoost's robust handling of categorical noise and feature interactions.
- **Handling Nonlinearity:** Sensor relationships within solar inverters (e.g., thermal gradients versus specific MPPT string output) are highly non-linear. The ensemble inherently handles these complex feature interactions better than a single model.
- **Explainability:** As an industrial deployment, maintenance engineers require specific component-level drivers for any generated alert. Tree-based models map directly to SHapley Additive exPlanations (SHAP), giving us precise, localized feature importance (e.g., `pv1_power` vs `internal_temp`) to route physical inspections.
- **Imbalance Handling:** Predictive maintenance datasets are heavily skewed towards nominal states. XGBoost's `scale_pos_weight` parameter dynamically adjusts the loss gradient to penalize missed failure events (False Negatives), prioritizing sensitivity over raw accuracy.

### 2.2 Feature Engineering
The raw telemetry lacked the context necessary for causal prediction. The feature pipeline was designed to inject temporal and electrical physics constraints:
- **Temporal Lags & Rolling Aggregates:** We computed 15-minute, 1-hour, and 24-hour moving averages. This exposes trajectory degradation (e.g., slow thermal runaway) rather than instantaneous spikes.
- **Electrical Efficiency:** A hardware efficiency ratio (`meter_active_power / total_dc_power`) was formulated to identify silent hardware failure modes where the inverter operational state claims 'healthy' but actual conversion efficiency degrades over time.

### 2.3 Chronological Validation
A strict time-based split was enforced to prevent data leakage. The model was trained entirely on the first 18-months of historical data. The final 6-months (180 days) were strictly reserved for the Replay Simulation Engine, ensuring our evaluated ROC-AUC and F1-Scores represent true forward-looking predictive power.

## 3. Generative AI Design & Prompt Iteration
The native XGBoost model provides mathematical drivers, but not actionable intelligence. We integrated a Large Language Model (OpenAI/Claude) using a Retrieval-Augmented Generation (RAG) architecture to synthesize human-readable tickets.

### 3.1 Prompt Iteration 1 (Baseline)
*Objective:* Convert SHAP data into a paragraph.
*Outcome:* The LLM provided generic descriptions ("The inverter is too hot"). It failed to cross-reference plant-wide comparisons, leading to false alarms when the entire plant was simply experiencing a cloudy day.

### 3.2 Prompt Iteration 2 (Grounded Diagnostics)
*Objective:* Enforce structural fault isolation logic.
*Outcome:* We modified the RAG system prompt to explicitly include the **FAULT DETECTION LOGIC** matrix:
1. *MPPT Imbalance Rule:* If `|pv1 - pv2| / total_dc` is high, diagnose as string shading or blown fuse.
2. *Thermal Disconnect:* If `temperature > 70C` but power is low, diagnose as active cooling failure.
3. *Cross-Inverter Reference:* Compare the target `INV_001` against the real-time average of the other 31 inverters.

This constrained the LLM output, forcing it to act as an industrial diagnostic constraint solver rather than a creative writer, significantly increasing the precision and technical value of the generated CMMS tickets.

## 4. System Limitations
- **Cold Start Anomaly Drift:** The model detects failure signatures present in the 18-month training data. It will likely misclassify novel, unseen hardware failure modes (e.g., a new type of grid-surge damage).
- **Latency Restrictions:** Running SHAP calculations across a continuous stream of ensemble models (XGBoost + LightGBM + CatBoost) for 32 inverters in real-time is computationally intensive. To solve this, we precomputed the 6-month simulation timeline to disk (`replay_predictions.parquet`) to allow the Dashboard Replay Engine to stream efficiently.

## 5. Future Improvements
1. **Unsupervised Anomaly Detection:** Implement an Isolation Forest or a localized Autoencoder to run in parallel with the XGBoost classifier. This would catch "Unknown-Unknown" failure modes by identifying multivariate deviation from historical healthy states, even if a specific alarm code is not tripped.
2. **Edge Computing:** Shift the lightweight feature pipeline and inference engine directly to the datalogger edge hardware (IoT gateway), sending only high-confidence anomaly payloads to the cloud to reduce network ingress costs.
