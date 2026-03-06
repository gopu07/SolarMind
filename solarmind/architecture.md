# System Architecture

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
        ReplayDB[("Replay Predictions Pq")]:::data
    end

    %% Pipeline Layer
    subgraph ML Pipeline
        Ingest[("Data Ingestion Script")]:::pipeline
        Features[("Feature Engineering")]:::pipeline
        Train[("XGBoost Trainer")]:::pipeline
        PredictGen[("Replay Prediction Generator")]:::pipeline
    end

    %% API Layer
    subgraph Backend Services (FastAPI)
        PredictAPI["POST /predict"]:::api
        NarrativeAPI["POST /narrative"]:::api
        QueryAPI["POST /query (RAG)"]:::api
        WebSocket["WS /ws/stream (Replay Engine)"]:::api
    end
    
    %% AI/LLM Services
    subgraph GenAI Subsystem
        Agent[("OpenAI / Claude Client")]:::llm
        Guardrails[("Output Validator")]:::llm
    end

    %% Frontend Layer
    subgraph Dashboard (React/Vite)
        WebUI["32-Inverter Command Center"]:::ui
    end

    %% Connections
    RawCSV --> Ingest --> ProcessedPQ
    ProcessedPQ --> Features --> Train
    Train --> PredictGen --> ReplayDB
    
    ReplayDB --> WebSocket
    ProcessedPQ --> PredictAPI
    ProcessedPQ --> QueryAPI
    VectorDB --> QueryAPI
    
    PredictAPI --> Agent
    NarrativeAPI --> Agent
    QueryAPI --> Agent
    Agent --> Guardrails
    
    WebSocket --> WebUI
    PredictAPI -.-> WebUI
    NarrativeAPI -.-> WebUI
    QueryAPI <--> WebUI
```
