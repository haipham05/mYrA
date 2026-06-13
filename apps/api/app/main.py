import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api.v1 import api_router
from app.config import Settings
from app.db.session import get_db
from app.logging import configure_logging

settings = Settings.from_environment()
api_v1 = APIRouter(prefix="/api/v1")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    current_settings = Settings.from_environment()
    configure_logging(current_settings.log_level)
    if current_settings.check_migration_compatibility:
        from app.db.compatibility import check_schema_compatibility
        from app.db.session import engine

        check_schema_compatibility(engine)
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


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or uuid4().hex
    )
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = correlation_id
    return response


@api_v1.get("/health", tags=["health"])
@app.get("/health", tags=["health"], include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Return the service health liveness status."""
    return {"status": "ok", "service": "myra-api"}


@api_v1.get("/health/ready", tags=["health"])
@app.get("/health/ready", tags=["health"], include_in_schema=False)
async def readiness_check() -> dict[str, Any]:
    """Readiness probe checking database connectivity, schema compatibility, and storage."""
    from fastapi import HTTPException
    from sqlalchemy import text

    from app.db.compatibility import get_schema_revisions
    from app.db.session import engine
    from app.storage.factory import get_storage

    current_settings = Settings.from_environment()
    readiness: dict[str, Any] = {
        "status": "ready",
        "service": "myra-api",
        "database": "unknown",
        "storage": "unknown",
    }

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        readiness["database"] = "connected"
    except Exception as err:
        readiness["status"] = "unready"
        readiness["database"] = "unreachable"
        raise HTTPException(
            status_code=503,
            detail={"status": "unready", "error": "Database connectivity check failed"},
        ) from err

    current_rev = None
    if current_settings.check_migration_compatibility:
        try:
            current_rev, expected_heads = get_schema_revisions(engine)
            if not current_rev or current_rev not in expected_heads:
                readiness["status"] = "unready"
                readiness["schema"] = "incompatible"
                err_msg = f"Schema mismatch (current: {current_rev}, expected: {expected_heads})"
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "unready",
                        "error": err_msg,
                    },
                )
            readiness["schema"] = "compatible"
            readiness["schema_revision"] = current_rev
        except HTTPException:
            raise
        except Exception as err:
            readiness["status"] = "unready"
            raise HTTPException(
                status_code=503,
                detail={"status": "unready", "error": "Schema revision check failed"},
            ) from err

    try:
        storage = get_storage(current_settings)
        await storage.exists("__readiness_probe__")
        readiness["storage"] = "available"
    except Exception as err:
        readiness["status"] = "unready"
        readiness["storage"] = "unavailable"
        raise HTTPException(
            status_code=503,
            detail={"status": "unready", "error": "Storage backend check failed"},
        ) from err

    return readiness


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
        "graphrag_enabled": current_settings.graphrag_enabled,
        "neo4j_configured": bool(current_settings.neo4j_uri),
    }


@api_v1.post("/system/reconcile", tags=["system"])
async def trigger_reconciliation(
    dry_run: bool = True,
    scan_storage: bool = False,
    prefix: str = "papers/",
    min_age_seconds: int = Query(
        900,
        ge=300,
        description="Minimum age of orphaned storage objects in seconds before deletion",
    ),
    confirm_destructive: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Reconcile stranded resources (expired worker leases and unreferenced storage objects)."""
    if min_age_seconds < 300:
        raise HTTPException(
            status_code=400,
            detail=(
                f"min_age_seconds must be at least 300 seconds (got {min_age_seconds}) "
                "to protect in-flight uploads."
            ),
        )

    if scan_storage and not dry_run and not confirm_destructive:
        raise HTTPException(
            status_code=400,
            detail="Destructive storage reconciliation requires confirm_destructive=true.",
        )

    from app.crud.reconciliation import reconcile_stranded_resources_async, validate_scan_prefix
    from app.storage.factory import get_storage

    storage = None
    scan_prefix = None
    if scan_storage:
        try:
            scan_prefix = validate_scan_prefix(prefix)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        storage = get_storage()

    report = await reconcile_stranded_resources_async(
        db=db,
        storage=storage,
        dry_run=dry_run,
        scan_prefix=scan_prefix,
        min_age_seconds=min_age_seconds,
    )
    return {
        "dry_run": report.dry_run,
        "stuck_jobs_expired": report.stuck_jobs_expired,
        "stuck_jobs_recovered": report.stuck_jobs_recovered,
        "orphaned_storage_keys": report.orphaned_storage_keys,
        "skipped_in_flight_keys": report.skipped_in_flight_keys,
        "storage_keys_scanned": report.storage_keys_scanned,
        "deletion_errors": report.deletion_errors,
    }


api_v1.include_router(api_router)
app.include_router(api_v1)
