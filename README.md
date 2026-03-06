# ☀️ SolarMind

**Industrial AI for Predictive Maintenance & Intelligence in Solar PV Plants**

SolarMind is a production-grade platform designed to transform raw solar array telemetry into actionable, predictive maintenance insights. By leveraging advanced machine learning and generative AI, it identifies equipment degradation before failure occurs, providing maintenance teams with precise diagnostic narratives and automated ticketing.

---

## 🚀 Key Features

- **Predictive Analytics**: Utilizes **XGBoost** to model failure risk for individual inverters with high sensitivity to equipment degradation.
- **Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** values to provide "why" behind every alert, identifying specific sensor drivers.
- **GenAI Diagnostics**: Automatically generates human-readable diagnostic narratives using **LLMs**, grounded in real-time telemetry.
- **RAG-Powered Intelligence**: A **Retrieval-Augmented Generation** pipeline that connects historical maintenance logs and telemetry for smarter troubleshooting.
- **Real-Time Monitoring**: A high-performance **React Dashboard** featuring a "32-Inverter Command Center" with live WebSocket streaming.
- **Automated Workflows**: End-to-end agentic workflow for automated ticket generation and notifications via **LangGraph**.

---

## 🏗️ Architecture

```mermaid
graph TD
    %% Define Styles
    classDef ui fill:#3b82f6,stroke:#1e3a8a,stroke-width:2px,color:#fff;
    classDef api fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef pipeline fill:#8b5cf6,stroke:#5b21b6,stroke-width:2px,color:#fff;
    classDef data fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef llm fill:#ec4899,stroke:#9d174d,stroke-width:2px,color:#fff;

    %% Data Layer
    subgraph Data Sources
        RawCSV[("Raw Datalogger CSVs")]:::data
        ProcessedPQ[("Processed Parquet")]:::data
        VectorDB[("ChromaDB Vector Store")]:::data
    end

    %% Pipeline Layer
    subgraph ML Pipeline
        Ingest[("Data Ingestion")]:::pipeline
        Features[("Feature Engineering")]:::pipeline
        Train[("XGBoost Trainer")]:::pipeline
    end

    %% API Layer
    subgraph Backend Services (FastAPI)
        PredictAPI["POST /predict"]:::api
        QueryAPI["POST /query (RAG)"]:::api
        WebSocket["WS /ws/stream"]:::api
    end
    
    %% AI/LLM Services
    subgraph GenAI Subsystem
        Agent[("GenAI Agent")]:::llm
        Guardrails[("Output Validator")]:::llm
    end

    %% Frontend Layer
    subgraph Dashboard (React/Vite)
        WebUI["Command Center UI"]:::ui
    end

    %% Connections
    RawCSV --> Ingest --> ProcessedPQ
    ProcessedPQ --> Features --> Train
    Train --> PredictAPI
    
    PredictAPI --> Agent
    QueryAPI --> Agent
    VectorDB --> QueryAPI
    Agent --> Guardrails
    
    WebSocket --> WebUI
    PredictAPI -.-> WebUI
    QueryAPI <--> WebUI
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, XGBoost, SHAP, Pandas, Scikit-learn
- **AI/LLM**: OpenAI/Claude, ChromaDB (Vector Store), LangGraph
- **Frontend**: React, Vite, TailwindCSS, Recharts, Lucide Icons
- **Data**: Parquet (Efficient Storage), CSV (Ingestion)
- **DevOps**: Docker, Docker Compose

---

## 📥 Getting Started

### Prerequisites
- Python 3.10 or higher
- Node.js & npm
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the Project**
   ```bash
   git clone https://github.com/gopu07/SolarMind.git
   cd SolarMind
   ```

2. **Backend Configuration**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate # Windows: venv\\Scripts\\activate

   # Install dependencies
   pip install -r requirements.txt

   # Configure environment variables
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Data & Model Pipeline**
   ```bash
   # Run the full pipeline
   python scripts/ingest_raw.py
   python features/pipeline.py
   python models/train.py
   python scripts/generate_replay_predictions.py
   ```

4. **Start the API Server**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Frontend Setup**
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

---

## 📂 Project Structure

```text
solarmind/
├── agent/            # LangGraph workflows
├── api/              # FastAPI application & routers
├── dashboard/        # React frontend
├── data/             # Raw & processed datasets
├── features/         # Feature engineering pipeline
├── genai/            # LLM prompts & guardrails
├── models/           # Training & prediction logic
├── rag/              # RAG ingestion & retrieval
├── scripts/          # Ingestion & utility scripts
└── tests/            # Pytest suite
```

---

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file.
