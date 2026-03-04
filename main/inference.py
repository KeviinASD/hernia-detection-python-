"""
inference.py
------------
Lógica de inferencia desacoplada de FastAPI.
  - is_grayscale():  valida que la imagen sea escala de grises (RM real).
  - run_inference(): corre el modelo y devuelve detecciones + imagen anotada.
  - image_to_b64():  convierte PIL Image a string base64 PNG.
"""

import base64
import io
import time

import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from .schemas import Detection

# Mapeo unificado de labels (YOLO cls+1 == torchvision labels)
CLASS_NAMES  = {1: "disc", 2: "hdisc"}
HERNIA_LABEL = 2

# Tolerancia para considerar una imagen como escala de grises.
# Diferencia media máxima entre canal R y canal G (en píxeles 0-255).
_GRAYSCALE_TOLERANCE = 10.0


def is_grayscale(img: Image.Image) -> bool:
    """
    Verifica si la imagen es escala de grises comparando los canales R y G.
    En una RM real los tres canales son iguales (R ≈ G ≈ B).
    Devuelve False si la imagen tiene colores (no es RM válida).
    """
    arr  = np.array(img.convert("RGB"))
    diff = np.abs(arr[:, :, 0].astype(np.int32) - arr[:, :, 1].astype(np.int32)).mean()
    return float(diff) < _GRAYSCALE_TOLERANCE


def run_inference(
    m_info:   dict,
    image:    Image.Image,
    conf_thr: float,
    iou_thr:  float,
) -> tuple[Image.Image, list[Detection], float]:
    """
    Ejecuta inferencia con el modelo indicado.

    Args:
        m_info:   entrada del dict MODELS (contiene "model", "type", "device").
        image:    imagen PIL en modo RGB.
        conf_thr: umbral de confianza mínima.
        iou_thr:  umbral de IoU para NMS.

    Returns:
        (imagen_anotada, lista_de_detecciones, tiempo_en_segundos)
    """
    import torch  # import local para no cargar torch al importar el módulo

    t0 = time.time()

    if m_info["type"] == "yolo":
        res = m_info["model"].predict(
            image, conf=conf_thr, iou=iou_thr, verbose=False
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,),   dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int32)
        else:
            boxes  = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            # YOLO: 0-indexed → +1 para unificar con torchvision (1-indexed)
            labels = res.boxes.cls.cpu().numpy().astype(int) + 1

    else:
        # Torchvision (Faster R-CNN / SSD)
        img_t = F.to_tensor(image).unsqueeze(0).to(m_info["device"])
        with torch.no_grad():
            out = m_info["model"](img_t)[0]

        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()   # ya son 1-indexed

        # Torchvision no filtra por confianza automáticamente
        mask = scores >= conf_thr
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    elapsed = time.time() - t0

    # ── Anotar imagen ──
    canvas = image.copy()
    draw   = ImageDraw.Draw(canvas)
    detections: list[Detection] = []

    for box, score, label in zip(boxes, scores, labels):
        lbl_id    = int(label)
        lbl_name  = CLASS_NAMES.get(lbl_id, f"cls{lbl_id}")
        is_hernia = lbl_id == HERNIA_LABEL
        color     = "red" if is_hernia else "#00CC44"
        text      = f"{lbl_name.upper()} {score:.2f}"

        # Rectángulo de la detección
        draw.rectangle(box.tolist(), outline=color, width=3)

        # Fondo del texto para legibilidad sobre imagen oscura
        tw, th = 95, 14
        draw.rectangle([box[0], box[1] - th - 2, box[0] + tw, box[1]], fill=color)
        draw.text((box[0] + 2, box[1] - th), text, fill="white")

        detections.append(Detection(
            label_id=lbl_id,
            label_name=lbl_name,
            confidence=round(float(score), 4),
            bbox=[round(float(x), 2) for x in box.tolist()],
        ))

    return canvas, detections, elapsed


def image_to_b64(img: Image.Image) -> str:
    """Convierte una imagen PIL a string base64 (formato PNG)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
