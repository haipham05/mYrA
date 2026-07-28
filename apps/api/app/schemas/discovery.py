"""Small, source-aware DTOs for academic catalog search."""

from typing import Literal

from pydantic import BaseModel, Field

CatalogName = Literal["openalex", "arxiv"]


class CatalogCandidate(BaseModel):
    catalog: CatalogName
    catalog_id: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=2_000)
    authors: list[str] = Field(default_factory=list, max_length=100)
    publication_year: int | None = Field(default=None, ge=1000, le=2100)
    doi: str | None = Field(default=None, max_length=255)
    arxiv_id: str | None = Field(default=None, max_length=128)
    abstract: str | None = Field(default=None, max_length=50_000)
    source_url: str = Field(min_length=1, max_length=4_000)
    pdf_url: str | None = Field(default=None, max_length=4_000)
    open_access: bool | None = None
    possible_duplicate: bool = False


class CatalogSearchResult(BaseModel):
    catalog: CatalogName
    query: str
    page: int = Field(ge=1, le=3)
    page_size: int = Field(ge=1, le=25)
    items: list[CatalogCandidate] = Field(max_length=25)
    has_more: bool


class CatalogSearchError(Exception):
    """Safe provider failure without response bodies or credentials."""

    def __init__(self, catalog: CatalogName, reason: str) -> None:
        self.catalog = catalog
        self.reason = reason
        super().__init__(f"{catalog} search unavailable: {reason}")
