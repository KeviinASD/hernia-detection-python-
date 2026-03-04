# Diagnóstico Multimodelo de Hernia Lumbar

Sistema de detección de hernias de disco lumbar en imágenes de Resonancia Magnética (RM) utilizando 6 modelos de inteligencia artificial.

---

## Estructura del proyecto

```
hernia_software/
│
├── main.py                  ← Punto de entrada del backend (ejecutar aquí)
├── app.py                   ← Frontend Streamlit (interfaz principal)
├── app2.py                  ← Frontend Streamlit (interfaz alternativa)
├── app_eval.py              ← Frontend Streamlit para evaluación de modelos
│
├── .env                     ← Variables de entorno (API key, puerto, etc.)
├── .env.example             ← Plantilla del .env
├── data.yaml                ← Configuración del dataset (clases, rutas)
├── requirements_eval.txt    ← Dependencias del proyecto
│
├── models/                  ← Modelos entrenados
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov8m.pt
│   ├── yolov8l.pt
│   ├── faster.pth
│   └── ssd.pth
│
├── main/                    ← Backend FastAPI (paquete)
│   ├── main.py              ← App FastAPI, lifespan, CORS, routers
│   ├── config.py            ← Configuración desde .env
│   ├── dependencies.py      ← Verificación x-api-key
│   ├── schemas.py           ← Modelos Pydantic de respuesta
│   ├── model_loader.py      ← Carga de modelos al iniciar
│   ├── inference.py         ← Lógica de inferencia y anotación
│   └── routers/
│       ├── health.py        ← GET  /health
│       ├── models.py        ← GET  /models
│       └── predict.py       ← POST /predict
│
└── train/                   ← Scripts de entrenamiento
    ├── train_models.py      ← Entrena YOLOv8 + Faster R-CNN + SSD
    └── train_torchvision.py ← Entrena Faster R-CNN y SSD (versión original)
```

---

## Modelos disponibles

| Nombre clave     | Arquitectura          | Framework      |
|------------------|-----------------------|----------------|
| `yolov8n`        | YOLOv8 Nano           | Ultralytics    |
| `yolov8s`        | YOLOv8 Small          | Ultralytics    |
| `yolov8m`        | YOLOv8 Medium         | Ultralytics    |
| `yolov8l`        | YOLOv8 Large          | Ultralytics    |
| `faster_rcnn`    | Faster R-CNN ResNet50 | Torchvision    |
| `ssd_mobilenet`  | SSD MobileNetV3       | Torchvision    |

**Clases detectadas:** `disc` (disco normal) y `hdisc` (disco con hernia)

---

## Instalación

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac
```

### 2. Instalar dependencias
```bash
pip install -r requirements_eval.txt
```

> **Si tienes GPU NVIDIA**, instala PyTorch con CUDA primero:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Reemplaza `cu121` con tu versión de CUDA (`cu118`, `cu124`, etc.).

### 3. Configurar variables de entorno
```bash
cp .env.example .env
```
Abre `.env` y cambia el valor de `API_KEY` por una clave segura:
```
API_KEY=mi-clave-secreta
MODELS_DIR=models
HOST=0.0.0.0
PORT=8000
```

---

## Ejecución

### Backend FastAPI

#### Paso 1 — Verifica que el `.env` esté configurado
```
API_KEY=mi-clave-secreta   ← cámbiala, esta es la llave para usar la API
MODELS_DIR=models          ← carpeta donde están los .pt y .pth
HOST=0.0.0.0               ← 0.0.0.0 = acepta conexiones de cualquier IP
PORT=8000                  ← puerto donde escuchará el servidor
```

#### Paso 2 — Verifica que los modelos existan
```
models/
  yolov8n.pt   yolov8s.pt   yolov8m.pt   yolov8l.pt
  faster.pth   ssd.pth
```

#### Paso 3 — Ejecutar
```bash
# Desde la raíz del proyecto (hernia_software/)
python main.py
```

#### Paso 4 — Verificar que arrancó correctamente
Al iniciar verás esto en consola:
```
[startup] Cargando modelos desde: models
  ✅ yolov8n
  ✅ yolov8s
  ✅ yolov8m
  ✅ yolov8l
  ✅ faster_rcnn
  ✅ ssd_mobilenet

──────────────────────────────────────────────────
  🚀  Servidor corriendo en : http://0.0.0.0:8000
  📄  Documentación (Swagger): http://0.0.0.0:8000/docs
  📘  Documentación (ReDoc)  : http://0.0.0.0:8000/redoc
  ❤️   Health check           : http://0.0.0.0:8000/health
──────────────────────────────────────────────────
```

#### Paso 5 — Comprobar que responde
Abre en el navegador:
- **http://localhost:8000/health** → debe devolver `{"status":"ok","models_loaded":6}`
- **http://localhost:8000/docs** → Swagger UI interactivo para probar los endpoints

> Si ves `models_loaded: 0` significa que ningún modelo cargó.
> Verifica que la carpeta `models/` exista y los archivos estén completos.

#### Alternativa con uvicorn directo
```bash
uvicorn main.main:app --host 0.0.0.0 --port 8000 --reload
```
`--reload` recarga automáticamente si cambias el código (útil en desarrollo).

---

### Frontend Streamlit
```bash
# Interfaz principal (todos los modelos)
streamlit run app.py

# Interfaz alternativa
streamlit run app2.py

# Evaluación de modelos (mAP, métricas, inferencia por carpeta)
streamlit run app_eval.py
```

---

## API REST

### Endpoints

| Método | Ruta       | Auth         | Descripción                         |
|--------|------------|--------------|-------------------------------------|
| GET    | `/health`  | No requerida | Estado del servidor                 |
| GET    | `/models`  | x-api-key    | Lista modelos cargados              |
| POST   | `/predict` | x-api-key    | Inferencia sobre una imagen de RM   |

### POST `/predict`

**Headers:**
```
x-api-key: tu-clave-secreta
```

**Form data:**

| Campo   | Tipo  | Requerido | Default | Descripción                          |
|---------|-------|-----------|---------|--------------------------------------|
| `image` | file  | ✅        | —       | Imagen JPG/PNG en escala de grises   |
| `model` | str   | ✅        | —       | Nombre del modelo a usar             |
| `conf`  | float | ❌        | `0.25`  | Umbral mínimo de confianza (0–1)     |
| `iou`   | float | ❌        | `0.45`  | Umbral IoU para NMS (0–1)            |

**Validaciones:**
- La imagen **debe** ser en escala de grises (RM real)
- Si el modelo **no detecta ningún disco**, devuelve error `422`

**Respuesta exitosa (`200`):**
```json
{
  "model_used": "yolov8s",
  "inference_time_s": 0.087,
  "n_total": 3,
  "n_hernias": 1,
  "avg_confidence": 0.761,
  "hernia_detected": true,
  "detections": [
    {
      "label_id": 2,
      "label_name": "hdisc",
      "confidence": 0.91,
      "bbox": [120.5, 80.2, 340.1, 210.8]
    }
  ],
  "annotated_image_b64": "iVBORw0KGgo..."
}
```

### Ejemplo con curl
```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: mi-clave-secreta" \
  -F "image=@rm_lumbar.jpg" \
  -F "model=yolov8s" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

---

## Entrenamiento

Los modelos fueron entrenados con el dataset de hernias de disco de Roboflow:
- **2 clases:** `disc` (disco normal), `hdisc` (disco con hernia)
- **Framework YOLO:** Ultralytics YOLOv8
- **Framework Torchvision:** Faster R-CNN y SSD con transfer learning desde ImageNet

```bash
# Entrenar YOLOv8 Small
python train/train_models.py --model yolov8s --data data.yaml --epochs 100

# Entrenar Faster R-CNN
python train/train_models.py --model fasterrcnn_r50 --data data.yaml --epochs 60

# Entrenar SSD
python train/train_models.py --model ssd_mnv2 --data data.yaml --epochs 80
```
