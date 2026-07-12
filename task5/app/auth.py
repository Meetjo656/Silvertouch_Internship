"""
app/auth.py — API key authentication for NLU service.
"""

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from app.config import NLU_API_KEY

# Setup API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key for protected endpoints.

    If NLU_API_KEY is not set (empty), authentication is disabled (development mode).
    """
    if not NLU_API_KEY:
        # No auth configured (development mode) - allow all requests
        return api_key
    if not api_key:
        raise HTTPException(status_code=403, detail="API Key missing")
    if api_key != NLU_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
