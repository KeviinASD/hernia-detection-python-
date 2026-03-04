import os
import time

import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuración de Estética y Título
# ---------------------------------------------------------
st.set_page_config(page_title="Análisis RM Lumbar - 6 Modelos", layout="wide")
st.title("🏥 Diagnóstico Multimodelo de Hernia Lumbar")
st.markdown("""
Esta aplicación analiza imágenes de Resonancia Magnética (RM) utilizando 6 arquitecturas de
Inteligencia Artificial entrenadas. Detecta **Discos Normales** y **Discos con Hernia**.
""")

MODELS_DIR = "models"
# Clases unificadas: YOLO cls+1 y Torchvision comparten el mismo espacio
# 0 = background (solo torchvision), 1 = disc, 2 = hdisc
HERNIA_LABEL = 2

# ---------------------------------------------------------
# Constructores de Modelos
# ---------------------------------------------------------

def build_fasterrcnn_r50(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssd_mnv3(num_classes):
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None,
        num_classes=num_classes,
    )


def clean_state_dict(model, state_dict):
    """
    Filtra pesos para cargar solo las capas con shape coincidente.
    Soluciona el error de dimensiones en SSD (80 vs 160, etc.).
    """
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
# Carga de Modelos con Caché
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelos…")
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes_torch = 3   # background + disc + hdisc
    models = {}
    errors = {}

    if not os.path.isdir(MODELS_DIR):
        errors["general"] = f"No se encontró la carpeta `{MODELS_DIR}/`"
        return models, errors, device

    # ── 1. Modelos YOLOv8 ──
    for variant in ["n", "s", "m", "l"]:
        name = f"yolov8{variant}"
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        if not os.path.isfile(path):
            continue
        try:
            models[name] = {"model": YOLO(path), "type": "yolo"}
        except Exception as e:
            errors[name] = str(e)

    # ── 2. Faster R-CNN ──
    faster_path = os.path.join(MODELS_DIR, "faster.pth")
    if os.path.isfile(faster_path):
        try:
            frcnn = build_fasterrcnn_r50(num_classes_torch)
            ckpt  = torch.load(faster_path, map_location=device, weights_only=False)
            sd    = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            frcnn.load_state_dict(sd, strict=False)   # strict=False: tolera keys extra/faltantes
            frcnn.to(device).eval()
            models["Faster R-CNN"] = {"model": frcnn, "type": "torch", "device": device}
        except Exception as e:
            errors["Faster R-CNN"] = str(e)

    # ── 3. SSD MobileNet ──
    ssd_path = os.path.join(MODELS_DIR, "ssd.pth")
    if os.path.isfile(ssd_path):
        try:
            ssd  = build_ssd_mnv3(num_classes_torch)
            ckpt = torch.load(ssd_path, map_location=device, weights_only=False)
            sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            # clean_state_dict resuelve el shape mismatch (80 vs 160, etc.)
            corrected_sd = clean_state_dict(ssd, sd)
            ssd.load_state_dict(corrected_sd, strict=False)
            ssd.to(device).eval()
            models["SSD MobileNet"] = {"model": ssd, "type": "torch", "device": device}
        except Exception as e:
            errors["SSD MobileNet"] = str(e)

    return models, errors, device


models_dict, load_errors, DEVICE = load_all_models()

# Mostrar errores de carga en sidebar
if load_errors:
    with st.sidebar:
        st.markdown("### ⚠️ Errores de carga")
        for name, err in load_errors.items():
            st.error(f"**{name}**: {err[:120]}")

# ---------------------------------------------------------
# Validación de imagen
# ---------------------------------------------------------

def is_valid_mri(img: Image.Image) -> bool:
    """
    Verifica si la imagen es escala de grises analizando la varianza R-G-B.
    Una RM real (gris) tendrá canales R ≈ G ≈ B.
    """
    arr  = np.array(img.convert("RGB"))
    diff = np.abs(arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)).mean()
    return diff < 10.0


# ---------------------------------------------------------
# Inferencia
# ---------------------------------------------------------

def process_image(model_info: dict, image: Image.Image,
                  conf_thr: float, iou_thr: float):
    start = time.time()

    if model_info["type"] == "yolo":
        results = model_info["model"].predict(
            image,
            conf=conf_thr,
            iou=iou_thr,
            verbose=False,
        )[0]

        # res.boxes puede ser None si no hay ninguna detección
        if results.boxes is None or len(results.boxes) == 0:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,),   dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int32)
        else:
            boxes  = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            # YOLO: 0=disc, 1=hdisc → sumamos 1 para unificar con torchvision
            labels = results.boxes.cls.cpu().numpy().astype(int) + 1

    else:
        # Torchvision detection models
        img_t = F.to_tensor(image).unsqueeze(0).to(model_info["device"])
        with torch.no_grad():
            out = model_info["model"](img_t)[0]
        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()   # 1=disc, 2=hdisc

        # Aplicar umbral de confianza (torchvision no lo aplica automáticamente)
        keep = scores >= conf_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    # ── Dibujar bboxes ──
    canvas = image.copy()
    draw   = ImageDraw.Draw(canvas)
    h_count = 0

    for b, s, l in zip(boxes, scores, labels):
        is_hernia = (l == HERNIA_LABEL)
        if is_hernia:
            h_count += 1
        color    = "red" if is_hernia else "green"
        cls_name = "HERNIA" if is_hernia else "DISCO"
        text     = f"{cls_name} {s:.2f}"

        draw.rectangle(b.tolist(), outline=color, width=3)

        # Fondo de texto para mejor legibilidad
        tw, th = 90, 14
        draw.rectangle(
            [b[0], b[1] - th - 2, b[0] + tw, b[1]],
            fill=color,
        )
        draw.text((b[0] + 2, b[1] - th), text, fill="white")

    duration = time.time() - start
    avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0
    return canvas, h_count, len(boxes), duration, avg_conf


# ---------------------------------------------------------
# UI de Streamlit
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuración")
    target_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.25, 0.05)
    iou_threshold    = st.slider("IoU (NMS)",            0.1, 1.0, 0.45, 0.05)
    st.divider()
    st.write("**Clases:**")
    st.write("🟢 Verde: Disco Normal")
    st.write("🔴 Rojo: Hernia de Disco")
    st.divider()
    st.write("**Modelos cargados:**")
    if models_dict:
        for mname in models_dict:
            st.success(f"✅ {mname}")
    else:
        st.error("Ningún modelo cargado.")

uploaded_file = st.file_uploader(
    "Subir imagen de Resonancia Magnética",
    type=["jpg", "png", "jpeg", "bmp", "tiff"],
)

if uploaded_file:
    img_input = Image.open(uploaded_file).convert("RGB")

    if not is_valid_mri(img_input):
        st.error(
            "❌ Imagen Rechazada: Se detectaron colores. "
            "Por favor suba una RM original en escala de grises."
        )
    else:
        st.success("✅ Imagen validada como RM.")
        st.image(img_input, caption="Vista previa de la RM", width=400)

        if not models_dict:
            st.warning(
                "⚠️ No hay modelos cargados. "
                f"Verifica que existan archivos `.pt` / `.pth` en `{MODELS_DIR}/`."
            )
        elif st.button("🔍 ANALIZAR CON LOS 6 MODELOS", use_container_width=True):
            st.divider()

            model_names = list(models_dict.keys())
            tab_list    = st.tabs(model_names)
            summary_data = []

            for tab, m_name in zip(tab_list, model_names):
                m_info = models_dict[m_name]
                with tab:
                    with st.spinner(f"Procesando con {m_name}…"):
                        try:
                            res_img, h_total, total, t_exec, conf = process_image(
                                m_info, img_input, target_threshold, iou_threshold
                            )
                        except Exception as e:
                            st.error(f"Error durante la inferencia: {e}")
                            continue

                    col_a, col_b = st.columns([2, 1])

                    with col_a:
                        st.image(res_img, use_container_width=True,
                                 caption=f"Resultado: {m_name}")

                    with col_b:
                        st.subheader("Estadísticas")
                        st.metric("Hernias Detectadas", h_total)
                        st.metric("Total de Discos",    total)
                        st.metric("Confianza Media",    f"{conf:.2%}")
                        st.metric("Tiempo de Respuesta", f"{t_exec:.4f}s")

                        st.markdown("**Interpretación:**")
                        if h_total > 0:
                            st.error(f"MODELO {m_name.upper()} DETECTA PRESENCIA DE HERNIA.")
                        else:
                            st.success(f"MODELO {m_name.upper()} NO DETECTA HERNIAS.")

                    summary_data.append({
                        "Modelo":              m_name,
                        "Hernia Detectada":    "SÍ" if h_total > 0 else "NO",
                        "N° Hernias":          h_total,
                        "Confianza Promedio":  f"{conf:.2f}",
                        "Inferencia (seg)":    round(t_exec, 4),
                    })

            # ── Tabla Comparativa Final ──
            if summary_data:
                st.divider()
                st.subheader("📊 Tabla Comparativa Global")
                st.table(pd.DataFrame(summary_data))
