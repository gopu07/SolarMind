from typing import Dict, List, Optional
from pydantic import BaseModel

class SessionState(BaseModel):
    session_id: str
    last_inverter: Optional[str] = None
    last_intent: Optional[str] = None
    history: List[Dict[str, str]] = []

# Simple in-memory global state (suitable for single-user dev environments)
# In production, this would be backed by Redis or a database using session IDs
_sessions: Dict[str, SessionState] = {}

def get_session(session_id: str = "default") -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState(session_id=session_id)
    return _sessions[session_id]
