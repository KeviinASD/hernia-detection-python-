"""
main.py
-------
Aplicación FastAPI — Hernia Lumbar.

Correr desde la raíz del proyecto con:
    python main.py
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model_loader import MODELS, load_all_models
from .routers import health, models, predict


# ─────────────────────────────────────────────
# Lifespan: carga modelos al iniciar, limpia al cerrar
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[startup] Cargando modelos desde:", settings.MODELS_DIR)
    errors = load_all_models(settings.MODELS_DIR)

    if MODELS:
        print(f"[startup] Modelos cargados: {list(MODELS.keys())}")
    if errors:
        print(f"[startup] Errores de carga: {errors}")
    if not MODELS:
        print("[startup] ⚠️  Ningún modelo se cargó correctamente.")

    base_url = f"http://{settings.HOST}:{settings.PORT}"
    print("\n" + "─" * 50)
    print(f"  🚀  Servidor corriendo en : {base_url}")
    print(f"  📄  Documentación (Swagger): {base_url}/docs")
    print(f"  📘  Documentación (ReDoc)  : {base_url}/redoc")
    print(f"  ❤️   Health check           : {base_url}/health")
    print("─" * 50 + "\n")

    yield  # la app está corriendo aquí

    print("\n[shutdown] Liberando modelos…")
    MODELS.clear()


# ─────────────────────────────────────────────
# Aplicación
# ─────────────────────────────────────────────

app = FastAPI(
    title="Hernia Lumbar — API de Inferencia",
    description=(
        "Backend de detección de hernias de disco lumbar. "
        "Soporta YOLOv8 (n/s/m/l), Faster R-CNN y SSD MobileNet. "
        "Todos los endpoints (excepto /health) requieren el header `x-api-key`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringir en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)


# ─────────────────────────────────────────────
# Punto de entrada directo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
