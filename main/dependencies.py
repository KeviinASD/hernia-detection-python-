"""
dependencies.py
---------------
Dependencias reutilizables de FastAPI.
  - verify_api_key: valida el header x-api-key en cada request protegido.
  - Auth: tipo anotado listo para usar en cualquier endpoint.
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from .config import settings


def verify_api_key(x_api_key: Annotated[str, Header()]) -> None:
    """
    Verifica que el header `x-api-key` coincida con la clave definida en .env.
    Lanza HTTP 401 si la clave es incorrecta o no está presente.
    """
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Tipo conveniente para anotar los parámetros de los endpoints protegidos:
#   async def my_endpoint(_: Auth): ...
Auth = Annotated[None, Depends(verify_api_key)]
