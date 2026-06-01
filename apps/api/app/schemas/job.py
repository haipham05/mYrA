import re
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class JobStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobStage(StrEnum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobResponse(BaseModel):
    id: UUID
    paper_id: UUID
    status: JobStatus
    stage: JobStage
    progress: float = Field(ge=0.0, le=1.0)
    error_message: str | None = None
    error_code: str | None = None
    is_retryable: bool = False
    retry_count: int = 0
    attempt_count: int = 0
    max_retries: int = 3
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def populate_computed_fields(cls, data: Any) -> Any:
        if hasattr(data, "__table__") or hasattr(data, "status"):  # ORM object
            retry_count = getattr(data, "retry_count", 0) or 0
            status_val = str(getattr(data, "status", ""))
            claimed_at = getattr(data, "claimed_at", None)
            error_msg = getattr(data, "error_message", None)

            attempt_count = retry_count + (
                1 if claimed_at is not None or status_val != "PENDING" or retry_count > 0 else 0
            )

            error_code = None
            if error_msg:
                match = re.match(r"^\[([A-Za-z0-9_]+)\]", str(error_msg))
                if match:
                    error_code = match.group(1)

            return {
                "id": data.id,
                "paper_id": data.paper_id,
                "status": data.status,
                "stage": data.stage,
                "progress": data.progress,
                "error_message": data.error_message,
                "error_code": error_code,
                "is_retryable": data.is_retryable,
                "retry_count": retry_count,
                "attempt_count": attempt_count,
                "max_retries": getattr(data, "max_retries", 3),
                "created_at": data.created_at,
                "updated_at": data.updated_at,
            }
        elif isinstance(data, dict):
            retry_count = data.get("retry_count", 0) or 0
            status_val = str(data.get("status", ""))
            claimed_at = data.get("claimed_at")
            error_msg = data.get("error_message")
            if "attempt_count" not in data:
                data["attempt_count"] = retry_count + (
                    1 if claimed_at is not None or status_val != "PENDING" or retry_count > 0 else 0
                )
            if "error_code" not in data and error_msg:
                match = re.match(r"^\[([A-Za-z0-9_]+)\]", str(error_msg))
                data["error_code"] = match.group(1) if match else None
        return data
