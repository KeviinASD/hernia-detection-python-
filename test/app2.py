import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import time
import os

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
st.set_page_config(page_title="Análisis RM Lumbar", layout="wide")
st.title("🏥 Diagnóstico Multimodelo de Hernia Lumbar")

MODELS_DIR = "models"

# Clases del dataset  (0-indexed en YOLO, 1-indexed en torchvision)
# YOLO:  0 = disc   | 1 = hdisc
# Torch: 1 = disc   | 2 = hdisc   (0 = background)
CLASS_NAMES = {1: "DISCO", 2: "HERNIA"}
HERNIA_LABEL = 2   # etiqueta de hernia en espacio 1-indexed

# ---------------------------------------------------------
# CONSTRUCTORES
# ---------------------------------------------------------

def build_fasterrcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssd_standard(num_classes):
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None,
        num_classes=num_classes,
    )


def clean_state_dict(model, state_dict):
    """Filtra pesos para que solo carguen capas con shape coincidente."""
    model_dict = model.state_dict()
    fixed_sd = {}
    for k, v in state_dict.items():
        if (
            ("head.module_list" in k or "head.classification_head" in k or "head.regression_head" in k)
            and len(k.split(".")) == 5
        ):
            p = k.split(".")
            new_key = f"{p[0]}.{p[1]}.{p[2]}.{p[3]}.0.0.{p[4]}"
            fixed_sd[new_key] = v
        else:
            fixed_sd[k] = v
    filtered_sd = {
        k: v for k, v in fixed_sd.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    return filtered_sd


# ---------------------------------------------------------
# CARGA DE MODELOS
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelos…")
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3   # background + disc + hdisc
    models = {}
    errors = {}

    if not os.path.isdir(MODELS_DIR):
        st.error(f"❌ No se encontró la carpeta `{MODELS_DIR}/`")
        return models, errors, device

    # ── YOLOv8 variants ──
    for variant in ["n", "s", "m", "l"]:
        pt_path = os.path.join(MODELS_DIR, f"yolov8{variant}.pt")
        if not os.path.isfile(pt_path):
            continue
        try:
            models[f"YOLOv8-{variant.upper()}"] = {
                "model": YOLO(pt_path),
                "type":  "yolo",
            }
        except Exception as e:
            errors[f"YOLOv8-{variant.upper()}"] = str(e)

    # ── Faster R-CNN ──
    faster_path = os.path.join(MODELS_DIR, "faster.pth")
    if os.path.isfile(faster_path):
        try:
            f_model = build_fasterrcnn(num_classes)
            ckpt = torch.load(faster_path, map_location=device, weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            f_model.load_state_dict(sd, strict=False)
            f_model.to(device).eval()
            models["Faster R-CNN"] = {"model": f_model, "type": "torch", "device": device}
        except Exception as e:
            errors["Faster R-CNN"] = str(e)

    # ── SSD MobileNet ──
    ssd_path = os.path.join(MODELS_DIR, "ssd.pth")
    if os.path.isfile(ssd_path):
        try:
            s_model = build_ssd_standard(num_classes)
            ckpt = torch.load(ssd_path, map_location=device, weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            corrected_sd = clean_state_dict(s_model, sd)
            s_model.load_state_dict(corrected_sd, strict=False)
            s_model.to(device).eval()
            models["SSD MobileNet"] = {"model": s_model, "type": "torch", "device": device}
        except Exception as e:
            errors["SSD MobileNet"] = str(e)

    return models, errors, device


models_dict, load_errors, DEVICE = load_models()

# Mostrar errores de carga en sidebar
if load_errors:
    with st.sidebar:
        st.markdown("### ⚠️ Modelos con error")
        for name, err in load_errors.items():
            st.error(f"**{name}**: {err[:120]}")


# ---------------------------------------------------------
# INFERENCIA
# ---------------------------------------------------------

def run_model(m_info, img: Image.Image, conf_thr: float, iou_thr: float):
    """
    Devuelve (imagen_anotada, n_hernias, n_total, tiempo_s, conf_media).
    """
    t0 = time.time()

    if m_info["type"] == "yolo":
        res = m_info["model"].predict(
            img,
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
        )[0]

        # res.boxes puede ser None si no hay detecciones
        if res.boxes is None or len(res.boxes) == 0:
            boxes, scores, labels = (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,),   dtype=np.float32),
                np.zeros((0,),   dtype=np.int32),
            )
        else:
            boxes  = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            # YOLO cls 0-indexed → llevamos a 1-indexed para unificar con torch
            labels = res.boxes.cls.cpu().numpy().astype(int) + 1

    else:
        # Torchvision detection models
        t_img = F.to_tensor(img).unsqueeze(0).to(m_info["device"])
        with torch.no_grad():
            out = m_info["model"](t_img)[0]
        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        # Filtrar por umbral de confianza (torch no lo aplica automáticamente)
        mask   = scores >= conf_thr
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    # ── Dibujar bboxes ──
    canvas = img.copy()
    draw   = ImageDraw.Draw(canvas)
    h_count = 0

    for box, score, label in zip(boxes, scores, labels):
        is_hernia = (label == HERNIA_LABEL)
        if is_hernia:
            h_count += 1
        color     = "red" if is_hernia else "#00CC44"
        cls_name  = CLASS_NAMES.get(int(label), f"cls{label}")
        text      = f"{cls_name} {score:.2f}"

        draw.rectangle(box.tolist(), outline=color, width=3)

        # Fondo del texto para mejor legibilidad
        tw, th = 90, 14
        draw.rectangle(
            [box[0], box[1] - th - 2, box[0] + tw, box[1]],
            fill=color,
        )
        draw.text((box[0] + 2, box[1] - th), text, fill="white")

    elapsed = time.time() - t0
    conf_media = float(np.mean(scores)) if len(scores) > 0 else 0.0
    return canvas, h_count, len(boxes), elapsed, conf_media


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Ajustes")
    umbral = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.05)
    iou    = st.slider("IoU (NMS)",        0.1, 1.0, 0.45, 0.05)

    st.divider()
    st.markdown("🔴 **HERNIA**  |  🟢 **DISCO**")
    st.divider()

    st.markdown("**Modelos cargados**")
    if models_dict:
        for name in models_dict:
            st.success(f"✅ {name}")
    else:
        st.error("Ningún modelo cargado.")

# ---------------------------------------------------------
# UI PRINCIPAL
# ---------------------------------------------------------

file = st.file_uploader("📂 Subir imagen de RM", type=["jpg", "png", "jpeg", "bmp", "tiff"])

if file:
    img_in = Image.open(file).convert("RGB")

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(img_in, caption="Imagen original", use_container_width=True)
    with col_info:
        st.markdown(f"**Resolución:** {img_in.width} × {img_in.height} px")
        st.markdown(f"**Modelos disponibles:** {len(models_dict)}")

    if not models_dict:
        st.warning("⚠️ No hay modelos cargados. Verifica la carpeta `models/`.")
    else:
        if st.button("🚀 Analizar con todos los modelos", use_container_width=True):
            model_names = list(models_dict.keys())
            tabs        = st.tabs(model_names)
            resumen     = []

            for tab, name in zip(tabs, model_names):
                info = models_dict[name]
                with tab:
                    with st.spinner(f"Procesando con {name}…"):
                        try:
                            res_img, h_n, t_n, dur, conf_med = run_model(
                                info, img_in, umbral, iou
                            )
                        except Exception as e:
                            st.error(f"Error durante la inferencia: {e}")
                            continue

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(res_img, use_container_width=True)
                    with col2:
                        st.metric("Hernias detectadas", h_n)
                        st.metric("Total detecciones",  t_n)
                        st.metric("Confianza media",    f"{conf_med:.1%}")
                        st.metric("Tiempo inferencia",  f"{dur:.3f} s")

                        if h_n > 0:
                            st.error(f"⚠️ Se detectó hernia")
                        else:
                            st.success("✅ Sin hernias detectadas")

                    resumen.append({
                        "Modelo":           name,
                        "Hernias":          h_n,
                        "Total detecciones": t_n,
                        "Confianza media":  round(conf_med, 3),
                        "Tiempo (s)":       round(dur, 4),
                    })

            # ── Tabla comparativa ──
            if resumen:
                st.divider()
                st.subheader("📊 Comparativa de modelos")
                df = pd.DataFrame(resumen)
                # Highlight fila con más hernias
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )
