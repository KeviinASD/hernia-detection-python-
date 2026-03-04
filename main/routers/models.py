"""
routers/models.py
-----------------
GET /models  —  lista los modelos disponibles.
Requiere header x-api-key.
"""

import torch
from fastapi import APIRouter

from ..dependencies import Auth
from ..model_loader import MODELS
from ..schemas import ModelsResponse

router = APIRouter(tags=["Modelos"])


@router.get("/models", response_model=ModelsResponse)
def list_models(_: Auth):
    """
    Devuelve la lista de modelos cargados y el dispositivo en uso.
    Requiere header `x-api-key`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModelsResponse(available_models=list(MODELS.keys()), device=device)
