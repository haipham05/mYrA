import json
import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    app_name: str = "mYrA API"
    runtime_profile: Literal["auto", "local", "cloud-data"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
    )
    cors_headers: list[str] = Field(default_factory=lambda: ["*"])
    database_url: str | None = None
    budget_database_url: str | None = None
    local_storage_root: str = "data/storage"
    supabase_project_ref: str | None = None
    google_cloud_project: str | None = None
    gcs_bucket_name: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_name: str = "deepseek-flash"
    deepseek_max_output_tokens: int = Field(default=4096, ge=1, le=65_536)
    max_upload_size_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 150
    check_migration_compatibility: bool = True
    graphrag_enabled: bool = False
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    neo4j_timeout_seconds: float = 10.0
    graph_batch_limit: int = Field(default=50, ge=1, le=100)
    cache_enabled: bool = False
    redis_url: str | None = None
    cache_namespace: str = "production"

    @model_validator(mode="after")
    def validate_cloud_data_profile(self) -> "Settings":
        if self.runtime_profile == "cloud-data":
            if not self.database_url:
                raise ValueError("cloud-data profile requires DATABASE_URL")
            if not self.gcs_bucket_name:
                raise ValueError("cloud-data profile requires GCS_BUCKET_NAME")
            if not self.budget_database_url:
                raise ValueError("cloud-data profile requires MYRA_BUDGET_DATABASE_URL")
        return self

    @classmethod
    def from_environment(cls) -> "Settings":
        values: dict[str, object] = {}
        if runtime_profile := os.getenv("MYRA_RUNTIME_PROFILE"):
            values["runtime_profile"] = runtime_profile
        if app_name := os.getenv("MYRA_APP_NAME"):
            values["app_name"] = app_name
        if log_level := os.getenv("MYRA_LOG_LEVEL"):
            values["log_level"] = log_level
        if cors_origins := os.getenv("MYRA_CORS_ORIGINS"):
            values["cors_origins"] = json.loads(cors_origins)
        if cors_methods := os.getenv("MYRA_CORS_METHODS"):
            values["cors_methods"] = json.loads(cors_methods)
        if cors_headers := os.getenv("MYRA_CORS_HEADERS"):
            values["cors_headers"] = json.loads(cors_headers)
        if database_url := os.getenv("DATABASE_URL"):
            values["database_url"] = database_url
        if budget_database_url := os.getenv("MYRA_BUDGET_DATABASE_URL"):
            values["budget_database_url"] = budget_database_url
        if storage_root := os.getenv("MYRA_LOCAL_STORAGE_ROOT"):
            values["local_storage_root"] = storage_root
        if supabase_ref := os.getenv("SUPABASE_PROJECT_REF"):
            values["supabase_project_ref"] = supabase_ref
        if gcp_project := os.getenv("GOOGLE_CLOUD_PROJECT"):
            values["google_cloud_project"] = gcp_project
        if gcs_bucket := os.getenv("GCS_BUCKET_NAME"):
            values["gcs_bucket_name"] = gcs_bucket
        if deepseek_key := os.getenv("DEEPSEEK_API_KEY"):
            values["deepseek_api_key"] = deepseek_key
        if deepseek_base := os.getenv("DEEPSEEK_BASE_URL"):
            values["deepseek_base_url"] = deepseek_base
        if deepseek_model := os.getenv("MYRA_DEEPSEEK_MODEL"):
            values["deepseek_model_name"] = deepseek_model
        if max_output_tokens := os.getenv("MYRA_DEEPSEEK_MAX_OUTPUT_TOKENS"):
            values["deepseek_max_output_tokens"] = int(max_output_tokens)
        if max_size := os.getenv("MYRA_MAX_UPLOAD_SIZE_BYTES"):
            values["max_upload_size_bytes"] = int(max_size)
        if max_pages := os.getenv("MYRA_MAX_PDF_PAGES"):
            values["max_pdf_pages"] = int(max_pages)
        if check_migrations := os.getenv("MYRA_CHECK_MIGRATIONS"):
            values["check_migration_compatibility"] = check_migrations.lower() in (
                "true",
                "1",
                "yes",
            )
        if graphrag_enabled := os.getenv("MYRA_GRAPHRAG_ENABLED"):
            values["graphrag_enabled"] = graphrag_enabled.lower() in (
                "true",
                "1",
                "yes",
            )
        if neo4j_uri := os.getenv("NEO4J_URI"):
            values["neo4j_uri"] = neo4j_uri
        if neo4j_user := os.getenv("NEO4J_USER"):
            values["neo4j_user"] = neo4j_user
        if neo4j_password := os.getenv("NEO4J_PASSWORD"):
            values["neo4j_password"] = neo4j_password
        if neo4j_database := os.getenv("NEO4J_DATABASE"):
            values["neo4j_database"] = neo4j_database
        if neo4j_timeout := os.getenv("NEO4J_TIMEOUT_SECONDS"):
            values["neo4j_timeout_seconds"] = float(neo4j_timeout)
        if graph_batch_limit := os.getenv("MYRA_GRAPH_BATCH_LIMIT"):
            values["graph_batch_limit"] = int(graph_batch_limit)
        if cache_enabled := os.getenv("MYRA_CACHE_ENABLED"):
            values["cache_enabled"] = cache_enabled.lower() in ("true", "1", "yes")
        if redis_url := os.getenv("MYRA_REDIS_URL"):
            values["redis_url"] = redis_url
        if cache_namespace := os.getenv("MYRA_CACHE_NAMESPACE"):
            values["cache_namespace"] = cache_namespace
        return cls.model_validate(values)
