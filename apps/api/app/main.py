import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_methods=["GET"],
    allow_headers=[],
)


@api_v1.get("/health", tags=["health"])
@app.get("/health", tags=["health"], include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "service": "myra-api"}


app.include_router(api_v1)
