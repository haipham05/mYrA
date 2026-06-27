import json
import logging
from datetime import UTC, datetime

from app.observability import get_operation_context, sanitize_observation, sanitize_text

_SAFE_EXTRA_FIELDS = frozenset(
    {
        "action",
        "attempt",
        "cache_hit",
        "cache_miss",
        "candidate_count",
        "citations_count",
        "conversation_id",
        "current_revision",
        "duration_ms",
        "error_code",
        "event_id",
        "evidence_count",
        "expected_heads",
        "failure_class",
        "job_id",
        "latency_ms",
        "model_revision",
        "outcome",
        "paper_id",
        "provider_model",
        "provider_response_id",
        "provider_usage",
        "requested_model",
        "reported_model",
        "response_id",
        "queue_age_seconds",
        "retry_count",
        "stage",
        "token_count_estimate",
        "worker_id",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = get_operation_context()
        event: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_text(record.getMessage()),
        }
        if context.correlation_id:
            event["correlation_id"] = context.correlation_id
        if context.trace_id and context.sampled:
            event["trace_id"] = context.trace_id

        selected_extra = {
            key: record.__dict__[key] for key in _SAFE_EXTRA_FIELDS if key in record.__dict__
        }
        safe_extra = sanitize_observation(selected_extra)
        event.update(safe_extra)
        return json.dumps(
            sanitize_observation(event),
            ensure_ascii=False,
            separators=(",", ":"),
        )


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("myra")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
