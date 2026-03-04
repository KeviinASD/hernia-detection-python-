"""
config.py
---------
Carga las variables del archivo .env y las expone como un objeto Settings.
"""

import os
from dotenv import load_dotenv

# Busca el .env en la raíz del proyecto (un nivel arriba de backend/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class Settings:
    API_KEY:    str = os.getenv("API_KEY",    "")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    HOST:       str = os.getenv("HOST",       "0.0.0.0")
    PORT:       int = int(os.getenv("PORT",   "8000"))


settings = Settings()

if not settings.API_KEY:
    raise RuntimeError(
        "API_KEY no está definida. Crea un archivo .env en la raíz del proyecto."
    )
