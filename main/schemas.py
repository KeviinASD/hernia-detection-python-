"""
schemas.py
----------
Modelos Pydantic que definen la forma de las respuestas JSON del API.
"""

from pydantic import BaseModel


class Detection(BaseModel):
    """Una sola detección dentro de una imagen."""
    label_id:   int         # 1 = disc | 2 = hdisc
    label_name: str         # "disc"   | "hdisc"
    confidence: float       # 0.0 – 1.0
    bbox:       list[float] # [x1, y1, x2, y2] en píxeles


class PredictResponse(BaseModel):
    """Respuesta completa del endpoint POST /predict."""
    model_used:          str
    inference_time_s:    float
    n_total:             int            # total de detecciones
    n_hernias:           int            # detecciones con label hdisc
    avg_confidence:      float
    hernia_detected:     bool
    detections:          list[Detection]
    annotated_image_b64: str            # imagen PNG anotada en base64


class ModelResult(BaseModel):
    """Resultado de un solo modelo dentro de /predict/all."""
    model_used:          str
    inference_time_s:    float
    n_total:             int
    n_hernias:           int
    avg_confidence:      float
    hernia_detected:     bool
    detections:          list[Detection]
    annotated_image_b64: str
    error:               str | None = None   # si ese modelo falló


class PredictAllResponse(BaseModel):
    """Respuesta del endpoint POST /predict/all."""
    total_models_run:       int              # cuántos modelos se ejecutaron
    models_detecting_hernia: int             # cuántos detectaron hernia
    consensus_hernia:       bool             # mayoría detectó hernia
    total_inference_time_s: float            # tiempo total acumulado
    results:                list[ModelResult]


class ModelsResponse(BaseModel):
    """Respuesta del endpoint GET /models."""
    available_models: list[str]
    device:           str               # "cuda" | "cpu"


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""
    status:        str   # "ok"
    models_loaded: int