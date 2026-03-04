"""
model_loader.py
---------------
Responsabilidades:
  1. Constructores de arquitecturas (Faster R-CNN, SSD).
  2. Reparación de state_dict con shape mismatch (SSD fix).
  3. Carga de todos los modelos disponibles en la carpeta models/.
  4. MODELS: dict global compartido con el resto de la app.
"""

import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Dict global: { nombre_modelo: info_dict }
# info_dict = {"model": ..., "type": "yolo"|"torch", "device": torch.device}
# ─────────────────────────────────────────────
MODELS: dict = {}


# ─────────────────────────────────────────────
# Constructores de arquitecturas
# ─────────────────────────────────────────────

def build_fasterrcnn(num_classes: int):
    """Faster R-CNN con backbone ResNet-50 FPN."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssd(num_classes: int):
    """SSD Lite con backbone MobileNetV3 Large."""
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None,
        num_classes=num_classes,
    )


def clean_state_dict(model, state_dict: dict) -> dict:
    """
    Filtra el state_dict para cargar solo las capas con shape coincidente.
    Resuelve el shape mismatch en las cabezas de SSD (ej. 80 vs 160 canales).
    """
    model_dict = model.state_dict()
    fixed: dict = {}

    for k, v in state_dict.items():
        # Corrige nombres de claves de la cabeza SSD (plano → anidado)
        if (
            ("head.module_list" in k or "head.classification_head" in k or "head.regression_head" in k)
            and len(k.split(".")) == 5
        ):
            p = k.split(".")
            fixed[f"{p[0]}.{p[1]}.{p[2]}.{p[3]}.0.0.{p[4]}"] = v
        else:
            fixed[k] = v

    # Solo conservar capas que existen en el modelo y tienen el mismo shape
    return {
        k: v for k, v in fixed.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }


# ─────────────────────────────────────────────
# Cargador principal
# ─────────────────────────────────────────────

def load_all_models(models_dir: str) -> dict[str, str]:
    """
    Carga todos los modelos encontrados en `models_dir` dentro del dict MODELS.
    Devuelve un dict { nombre: mensaje_de_error } con los modelos que fallaron.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3   # 0=background, 1=disc, 2=hdisc
    errors: dict[str, str] = {}

    if not os.path.isdir(models_dir):
        errors["general"] = f"Carpeta de modelos no encontrada: '{models_dir}'"
        return errors

    # ── YOLOv8 (n / s / m / l) ──
    for variant in ["n", "s", "m", "l"]:
        name = f"yolov8{variant}"
        path = os.path.join(models_dir, f"{name}.pt")
        if not os.path.isfile(path):
            continue
        try:
            MODELS[name] = {"model": YOLO(path), "type": "yolo"}
            print(f"  ✅ {name}")
        except Exception as e:
            errors[name] = str(e)
            print(f"  ❌ {name}: {e}")

    # ── Faster R-CNN ──
    faster_path = os.path.join(models_dir, "faster.pth")
    if os.path.isfile(faster_path):
        try:
            m    = build_fasterrcnn(num_classes)
            ckpt = torch.load(faster_path, map_location=device, weights_only=False)
            sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            m.load_state_dict(sd, strict=False)
            m.to(device).eval()
            MODELS["faster_rcnn"] = {"model": m, "type": "torch", "device": device}
            print("  ✅ faster_rcnn")
        except Exception as e:
            errors["faster_rcnn"] = str(e)
            print(f"  ❌ faster_rcnn: {e}")

    # ── SSD MobileNet ──
    ssd_path = os.path.join(models_dir, "ssd.pth")
    if os.path.isfile(ssd_path):
        try:
            m    = build_ssd(num_classes)
            ckpt = torch.load(ssd_path, map_location=device, weights_only=False)
            sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            m.load_state_dict(clean_state_dict(m, sd), strict=False)
            m.to(device).eval()
            MODELS["ssd_mobilenet"] = {"model": m, "type": "torch", "device": device}
            print("  ✅ ssd_mobilenet")
        except Exception as e:
            errors["ssd_mobilenet"] = str(e)
            print(f"  ❌ ssd_mobilenet: {e}")

    return errors
