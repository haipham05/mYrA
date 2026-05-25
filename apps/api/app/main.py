import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.config import Settings
from app.logging import configure_logging

settings = Settings.from_environment()
api_v1 = APIRouter(prefix="/api/v1")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    logging.getLogger("myra.api").info("application_started")
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


@api_v1.get("/health", tags=["health"])
@app.get("/health", tags=["health"], include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "service": "myra-api"}


@api_v1.get("/system/status", tags=["system"])
async def system_status() -> dict[str, Any]:
    """Return system configuration status without exposing sensitive credentials."""
    current_settings = Settings.from_environment()
    return {
        "status": "ok",
        "provider": "deepseek",
        "deepseek_configured": bool(current_settings.deepseek_api_key),
        "storage_backend": "gcs" if current_settings.gcs_bucket_name else "local",
        "max_upload_size_bytes": current_settings.max_upload_size_bytes,
        "max_pdf_pages": current_settings.max_pdf_pages,
    }


api_v1.include_router(api_router)
app.include_router(api_v1)
