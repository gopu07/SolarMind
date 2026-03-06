# SolarMind Predictive Maintenance System

## Overview
SolarMind is an AI-powered predictive maintenance and diagnostics platform for utility-scale solar PV plants. It ingests array-level datalogger telemetry from solar inverters, processes it through an advanced feature engineering pipeline, models failure risk using XGBoost, and generates dynamic natural-language diagnostic narratives via LLM.

## Setup Instructions

### Local Development
1. **Clone the repository:**
   ```bash
   git clone https://github.com/amitesh103/Hackamined.git
   cd Hackamined/solarmind
   ```
2. **Setup Backend:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   
   # Add your OpenAI API key
   echo "OPENAI_API_KEY=your-key-here" > .env
   
   # Run the ingest and pipeline (make sure data/raw has the CSVs)
   python scripts/ingest_raw.py
   python features/pipeline.py
   python models/train.py
   python scripts/generate_replay_predictions.py
   
   # Start FastAPI server
   uvicorn api.main:app --reload
   ```
3. **Setup Frontend:**
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

### Docker Deployment
Run everything at once using Docker Compose:
```bash
docker-compose up --build
```
The React dashboard will be available at `http://localhost:5174` and the FastAPI backend at `http://localhost:8000`.

## Architecture & Design Decisions
- **Data Normalization**: Array columns are unpivoted to support granular inverter-level ML modeling.
- **Chronological Restraint**: Prevented data leakage by maintaining strict chronological 18-month train / 6-month simulation holdouts.
- **WebSocket Streaming**: Created an asynchronous Replay Engine to broadcast the 6-month simulated timeline to the frontend continuously, rather than straining the prediction model with 32 synchronous HTTP endpoint calls every second.
- **RAG + GenAI**: Grounded the language model on real-time operational telemetry and vector-stored maintenance logs to produce specific, actionable ticket generation summaries.