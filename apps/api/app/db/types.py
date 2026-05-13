from typing import Any

from sqlalchemy.types import UserDefinedType


class PGVector(UserDefinedType):
    """PostgreSQL pgvector / halfvec type with fallback for SQLite tests."""

    cache_ok = True

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def get_col_spec(self, **kw: Any) -> str:
        return f"halfvec({self.dim})"

    def bind_processor(self, dialect: Any):
        def process(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return "[" + ",".join(str(float(x)) for x in value) + "]"
            return str(value)

        return process

    def result_processor(self, dialect: Any, coltype: Any):
        def process(value: Any) -> list[float] | None:
            if value is None:
                return None
            if isinstance(value, str):
                return [float(x) for x in value.strip("[]").split(",") if x.strip()]
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            return value

        return process


class TSVector(UserDefinedType):
    """PostgreSQL tsvector type with fallback for SQLite tests."""

    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "tsvector"
