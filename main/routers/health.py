"""
routers/health.py
-----------------
GET /health  —  público, sin autenticación.
Usado para verificar que el servidor está vivo y cuántos modelos están cargados.
"""

from fastapi import APIRouter

from ..model_loader import MODELS
from ..schemas import HealthResponse

router = APIRouter(tags=["Sistema"])


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check del servidor.
    No requiere autenticación.
    """
    return HealthResponse(status="ok", models_loaded=len(MODELS))
