"""
main.py  —  Punto de entrada del backend FastAPI.
Ejecutar: python main.py
"""

import uvicorn
from main.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
