from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import PaginatedResponse


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(PaginatedResponse[ProjectResponse]):
    pass
