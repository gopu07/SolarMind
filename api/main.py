"""
Layer 7 — Main FastAPI Application Entry Point.
"""

import asyncio
import json
import uuid
import time
from typing import Dict

from fastapi import FastAPI, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, make_asgi_app
import structlog
from contextlib import asynccontextmanager

import config
from api.auth import create_access_token
from api.schemas.models import Token
from api.routers import health, predict, query

# Set up structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'App Request Count', ['method', 'endpoint', 'http_status'])
ERROR_COUNT = Counter('error_count', 'App Error Count', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])

# active WebSocket connections per plant
active_websockets: Dict[str, list[WebSocket]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    # Spawn background task to push WS updates every 60s
    task = asyncio.create_task(websocket_push_loop())
    yield
    task.cancel()

app = FastAPI(
    title="SolarMind AI API",
    version="1.0.0",
    lifespan=lifespan
)

# Prometheus endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def instrumentation_middleware(request: Request, call_next):
    """Inject Request ID and track Prometheus metrics."""
    req_id = str(uuid.uuid4())
    # Store request id (simplified here via direct log context simulation if supported)
    with structlog.contextvars.bound_contextvars(request_id=req_id):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            ERROR_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
            raise e
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
            REQUEST_COUNT.labels(
                method=request.method, 
                endpoint=request.url.path, 
                http_status=status_code
            ).inc()
            
        response.headers["X-Request-ID"] = req_id
        return response


# Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(query.router)


@app.post("/auth/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate via OAuth2."""
    if form_data.username == config.DEV_USERNAME and form_data.password == config.DEV_PASSWORD:
        access_token = create_access_token(data={"sub": form_data.username})
        return Token(access_token=access_token, token_type="bearer")
    
    log.warning("failed_login_attempt", username=form_data.username)
    raise HTTPException(status_code=401, detail="Incorrect username or password")


@app.get("/inverters/{inverter_id}/report", tags=["GenAI"])
async def get_cached_report(inverter_id: str):
    """Fetch cached InverterReport.
    
    Placeholder for Redis integration. Currently returns 404 if no recent run made.
    """
    # Simple mock fallback as per spec
    raise HTTPException(status_code=404, detail="No cached report available")


async def websocket_push_loop():
    """Background task to push updates to active WebSockets every 60s."""
    import pandas as pd
    from models.predict import predict_inverter
    while True:
        await asyncio.sleep(60)
        
        # Load all inverters to simulate updates
        master_path = config.PROCESSED_DIR / "master_labelled.parquet"
        if not master_path.exists():
            continue
            
        try:
            df = pd.read_parquet(master_path, columns=["inverter_id", "plant_id"])
        except Exception:
            continue
            
        for plant_id, connections in list(active_websockets.items()):
            if not connections:
                continue
                
            inverters = df[df["plant_id"] == plant_id]["inverter_id"].unique().tolist()
            updates = []
            
            for inv in inverters:
                # Mock update to avoid running heavy model on 60s loop for everyone without batching
                # In prod, we'd read this from a Redis cached state populated by the background pipeline
                updates.append({
                    "inverter_id": inv,
                    "event": "tick",
                    "timestamp": time.time()
                })
                
            dead_conns = []
            for ws in connections:
                try:
                    await ws.send_json({"type": "update", "data": updates})
                except Exception:
                    dead_conns.append(ws)
                    
            for ws in dead_conns:
                connections.remove(ws)


@app.websocket("/ws/stream/{plant_id}")
async def websocket_endpoint(websocket: WebSocket, plant_id: str):
    """WebSocket stream for real-time dashboard updates."""
    await websocket.accept()
    
    if plant_id not in active_websockets:
        active_websockets[plant_id] = []
    active_websockets[plant_id].append(websocket)
    
    try:
        # Immediately send mock initial state
        await websocket.send_json({
            "type": "initial", 
            "plant_id": plant_id,
            "message": "Connected to continuous telemetry stream"
        })
        
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets[plant_id].remove(websocket)
        log.info("websocket_disconnected", plant_id=plant_id)
