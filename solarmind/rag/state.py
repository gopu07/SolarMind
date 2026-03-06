from typing import Dict, Optional

class SessionState:
    def __init__(self):
        self.last_inverter: Optional[str] = None
        self.last_intent: Optional[str] = None

# Simple in-memory global state (suitable for single-user dev environments)
# In production, this would be backed by Redis or a database using session IDs
_sessions: Dict[str, SessionState] = {}

def get_session(session_id: str = "default") -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]
