# ☀️ SolarMind: Industrial AI for Solar Excellence

**SolarMind** is a production-ready, industrial AI platform designed for predictive maintenance, fault isolation, and operational intelligence in utility-scale solar PV plants. 

By unpivoting complex solar telemetry into actionable insights, SolarMind empowers O&M teams to move from reactive repairs to proactive excellence.

---

## 🚀 Platform Architecture

SolarMind follows a **10-Layer Industrial AI Architecture**, ensuring modularity, scalability, and high reliability.

### 1. Ingestion Layer
Processes raw telemetry from SCADA systems and solar loggers.
- **Components**: `scripts/backfill_replay.py`, `scripts/generate_replay_predictions.py`
- **Output**: Cleaned Parquet streams in `data/processed/`

### 2. Feature Engineering Layer
A stateless pipeline that transforms raw sensors into physics-aware features.
- **Features**: Cyclical time encodings, conversion efficiency, mppt imbalance, string current variance, and plant-relative benchmarking.
- **Module**: `solarmind/features/pipeline.py`

### 3. Analytics & ML Layer
The "Brain" of SolarMind.
- **Ensemble Model**: A high-performance ensemble of **XGBoost (50%)**, **LightGBM (30%)**, and **CatBoost (20%)** for multiclass fault classification (Ground Fault, Array Soiling, MPPT Failure, etc.).
- **Explainability**: Integrated **SHAP** values for every prediction, providing "why" behind the risk score.
- **Module**: `solarmind/models/ensemble.py`

### 4. Integrity (Drift) Layer
Monitors the "Health of the AI."
- **Detection**: Uses Z-score analysis to compare live telemetry against a 180-day training baseline.
- **Function**: Automatically flags feature drift or sensor degradation before they impact prediction quality.
- **Module**: `solarmind/models/drift.py`

### 5. Knowledge (RAG) Layer
Retrieval-Augmented Generation for semantic O&M intelligence.
- **Store**: **ChromaDB** vector store containing digitized solar O&M manuals.
- **Retriever**: Hybrid search (BM25 + Semantic) to find relevant maintenance procedures for specific fault codes.
- **Module**: `solarmind/rag/`

### 6. Intelligence (GenAI) Layer
Autonomous diagnostic reasoning.
- **LLM**: Gemini-1.5-Pro / GPT-4 integration.
- **Diagnostics**: Combines ML risk scores with RAG knowledge to generate natural language diagnostic reports and maintenance tickets.
- **Module**: `solarmind/genai/`

### 7. API Gateway Layer
High-concurrency interface for the frontend.
- **Framework**: **FastAPI** with asynchronous request handling.
- **Security**: OAuth2 with JWT-token based authentication.
- **Status**: Live masked configuration status and health endpoints.
- **Module**: `solarmind/api/`

### 8. Real-time Stream Layer
Dual-channel communication.
- **Engine**: WebSocket broadcaster pushing chronological telemetry slices every 2 seconds.
- **Protocol**: Efficient JSON payloads for live heatmap updates.

### 9. Industrial Dashboard Layer
Premium, high-fidelity UI for mission control.
- **Stack**: React, Vite, TailwindCSS, Framer Motion.
- **Features**: Interactive Risk Heatmaps, TTF (Time-to-Failure) Timelines, and ML Drift Panels.
- **Module**: `solarmind/dashboard/`

### 10. Operations (DevOps) Layer
Deployment and Monitoring infrastructure.
- **Containerization**: Optimized Docker builds for Render/Cloud deployment.
- **CI/CD**: GitHub Actions for automated testing and linting.
- **Monitoring**: Prometheus metrics for request latency and error rates.

---

## 🛠️ Tech Stack & Dependencies

| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.11, FastAPI, Pydantic, Structlog |
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, ShadcnUI |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, CatBoost, SHAP |
| **Database/Storage** | ChromaDB (Vector), Pandas/PyArrow (Parquet) |
| **Monitoring** | Prometheus, Z-score Drift Detection |
| **Deployment** | Docker, Docker Compose, Render |

---

## 📦 Project Structure

```text
SolarMind/
├── solarmind/              # Core Application Source
│   ├── api/                # FastAPI Routers & Logic
│   ├── app_config/         # Secure Environment Configuration
│   ├── dashboard/          # React Frontend (Vite)
│   ├── data/               # Processed & Raw Data (Ignored)
│   ├── features/           # Feature Engineering Pipeline
│   ├── genai/              # LLM Core & Guardrails
│   ├── models/             # ML Models & Drift Monitoring
│   ├── rag/                # Hybrid Vector Retrieval
│   ├── scripts/            # Training & Ingestion Utilities
│   └── tests/              # Comprehensive Pytest Suite
├── Dockerfile              # Backend Container Configuration
├── docker-compose.yml      # Full-stack Orchestration
├── requirements.txt        # Backend Dependencies
└── README.md               # You are here
```

---

## ⚙️ Configuration & Environment

SolarMind uses a secure, environment-based configuration system. Create a `.env` file in the `solarmind/` directory:

```env
# AI API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Environment Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
JWT_SECRET=your-secret-here

# Data Paths
DATABASE_URL=sqlite:///./solarmind.db
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
Launch the entire stack (API + Dashboard) with one command:
```bash
docker-compose up --build
```
- **Dashboard**: http://localhost:8080
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

### Option 2: Local Development

#### Backend
```bash
cd solarmind
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

#### Frontend
```bash
cd solarmind/dashboard
npm install
npm run dev
```

---

## 📈 Monitoring & Reliability

SolarMind includes built-in observability:
- **Health Check**: `GET /health` provides a deep-dive into model status and client counts.
- **Config Status**: `GET /config/status` allows for safe debugging of environment variables.
- **ML Drift**: The dashboard provides a live visualization of Z-scores for critical sensor signals.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

*SolarMind — Empowering the Solar Industrial Revolution.*
