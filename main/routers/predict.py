"""
routers/predict.py
------------------
POST /predict      — inferencia con un modelo específico.
POST /predict/all  — inferencia con todos los modelos cargados.

Ambos requieren header x-api-key.
"""

import io

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image

from ..dependencies import Auth
from ..inference import HERNIA_LABEL, image_to_b64, is_grayscale, run_inference
from ..model_loader import MODELS
from ..schemas import ModelResult, PredictAllResponse, PredictResponse

router = APIRouter(tags=["Inferencia"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    _:     Auth,
    image: UploadFile = File(...,  description="Imagen de RM (jpg/png)"),
    model: str        = Form(...,  description="Nombre del modelo a usar"),
    conf:  float      = Form(0.25, description="Umbral de confianza (0–1)"),
    iou:   float      = Form(0.45, description="Umbral IoU para NMS (0–1)"),
):
    """
    Corre inferencia sobre la imagen enviada con el modelo indicado.

    Devuelve:
    - Lista de detecciones con label, confianza y bounding box.
    - Imagen anotada en base64 (PNG).
    - Resumen: n_hernias, n_total, confianza media, tiempo de inferencia.

    Requiere header `x-api-key`.
    """

    # ── Validar modelo ──
    if model not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo '{model}' no encontrado. Disponibles: {list(MODELS.keys())}",
        )

    # ── Validar parámetros numéricos ──
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(status_code=422, detail="'conf' debe estar entre 0.0 y 1.0")
    if not 0.0 <= iou <= 1.0:
        raise HTTPException(status_code=422, detail="'iou' debe estar entre 0.0 y 1.0")

    # ── Validar tipo MIME ──
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser una imagen (jpg, png, etc.).",
        )

    # ── Leer y decodificar imagen ──
    raw = await image.read()
    try:
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    # ── Validar que sea escala de grises (RM real) ──
    if not is_grayscale(img_pil):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La imagen contiene color. Solo se aceptan imágenes de RM en escala de grises.",
        )

    # ── Correr inferencia ──
    try:
        annotated, detections, elapsed = run_inference(MODELS[model], img_pil, conf, iou)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {e}",
        )

    # ── Validar que se detectó al menos un disco ──
    if not detections:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se detectó ningún disco lumbar en la imagen. Verifica que la imagen sea una RM de columna lumbar o reduce el umbral de confianza.",
        )

    # ── Calcular métricas resumen ──
    n_hernias = sum(1 for d in detections if d.label_id == HERNIA_LABEL)
    avg_conf  = float(np.mean([d.confidence for d in detections])) if detections else 0.0

    return PredictResponse(
        model_used=model,
        inference_time_s=round(elapsed, 4),
        n_total=len(detections),
        n_hernias=n_hernias,
        avg_confidence=round(avg_conf, 4),
        hernia_detected=n_hernias > 0,
        detections=detections,
        annotated_image_b64=image_to_b64(annotated),
    )


@router.post("/predict/all", response_model=PredictAllResponse)
async def predict_all(
    _:     Auth,
    image: UploadFile = File(...,  description="Imagen de RM (jpg/png)"),
    conf:  float      = Form(0.25, description="Umbral de confianza (0–1)"),
    iou:   float      = Form(0.45, description="Umbral IoU para NMS (0–1)"),
):
    """
    Corre inferencia sobre la imagen con **todos** los modelos cargados.

    Devuelve:
    - Resultado individual de cada modelo (detecciones, imagen anotada, tiempo).
    - Resumen global: cuántos modelos detectaron hernia y consenso por mayoría.

    Requiere header `x-api-key`.
    """

    # ── Validar parámetros numéricos ──
    if not 0.0 <= conf <= 1.0:
        raise HTTPException(status_code=422, detail="'conf' debe estar entre 0.0 y 1.0")
    if not 0.0 <= iou <= 1.0:
        raise HTTPException(status_code=422, detail="'iou' debe estar entre 0.0 y 1.0")

    # ── Validar tipo MIME ──
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser una imagen (jpg, png, etc.).",
        )

    # ── Leer y decodificar imagen ──
    raw = await image.read()
    try:
        img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

    # ── Validar escala de grises ──
    if not is_grayscale(img_pil):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La imagen contiene color. Solo se aceptan imágenes de RM en escala de grises.",
        )

    if not MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No hay modelos cargados en el servidor.",
        )

    # ── Correr inferencia con cada modelo ──
    results: list[ModelResult] = []
    total_time = 0.0

    for model_name, m_info in MODELS.items():
        try:
            annotated, detections, elapsed = run_inference(m_info, img_pil, conf, iou)
            total_time += elapsed
            n_hernias = sum(1 for d in detections if d.label_id == HERNIA_LABEL)
            avg_conf  = float(np.mean([d.confidence for d in detections])) if detections else 0.0

            results.append(ModelResult(
                model_used=model_name,
                inference_time_s=round(elapsed, 4),
                n_total=len(detections),
                n_hernias=n_hernias,
                avg_confidence=round(avg_conf, 4),
                hernia_detected=n_hernias > 0,
                detections=detections,
                annotated_image_b64=image_to_b64(annotated),
                error=None,
            ))
        except Exception as e:
            # Si un modelo falla, se registra el error y se continúa con el resto
            results.append(ModelResult(
                model_used=model_name,
                inference_time_s=0.0,
                n_total=0,
                n_hernias=0,
                avg_confidence=0.0,
                hernia_detected=False,
                detections=[],
                annotated_image_b64="",
                error=str(e),
            ))

    models_with_hernia = sum(1 for r in results if r.hernia_detected)
    successful = [r for r in results if r.error is None]

    # Si ningún modelo detectó discos, informar al cliente
    if successful and all(r.n_total == 0 for r in successful):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ningún modelo detectó discos lumbares. Verifica la imagen o reduce el umbral de confianza.",
        )

    return PredictAllResponse(
        total_models_run=len(results),
        models_detecting_hernia=models_with_hernia,
        consensus_hernia=models_with_hernia > len(results) // 2,
        total_inference_time_s=round(total_time, 4),
        results=results,
    )
