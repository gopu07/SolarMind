"""
Layer 7 — Authentication utilities.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import structlog

import config

log = structlog.get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM
    )
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Validate token and return username."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError as e:
        log.warning("jwt_validation_failed", error=str(e))
        raise credentials_exception
        
    return username
