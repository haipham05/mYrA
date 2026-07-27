from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.comparison import (
    BenchmarkContext,
    ComparabilityStatus,
    ComparisonDimension,
    ComparisonRequest,
)
from app.services.comparison_rules import compare_benchmark_contexts


def _benchmark(**overrides: str) -> BenchmarkContext:
    values = {
        "task": "image classification",
        "dataset": "ImageNet",
        "split": "validation",
        "metric": "top-1 accuracy",
        "unit": "%",
        "comparison_condition": "single crop, 224px",
    }
    values.update(overrides)
    return BenchmarkContext(**values)


def test_matching_known_context_is_directly_comparable() -> None:
    result = compare_benchmark_contexts(_benchmark(), _benchmark(dataset=" imagenet "))

    assert result.status is ComparabilityStatus.DIRECTLY_COMPARABLE
    assert result.reasons == []


@pytest.mark.parametrize(
    ("field", "other_value"),
    [("split", "test"), ("unit", "fraction")],
)
def test_split_or_unit_mismatch_is_not_directly_comparable(field: str, other_value: str) -> None:
    result = compare_benchmark_contexts(_benchmark(), _benchmark(**{field: other_value}))

    assert result.status is ComparabilityStatus.NOT_DIRECTLY_COMPARABLE
    assert any(field in reason for reason in result.reasons)


def test_missing_context_is_not_directly_comparable() -> None:
    result = compare_benchmark_contexts(_benchmark(), _benchmark(comparison_condition=""))

    assert result.status is ComparabilityStatus.NOT_DIRECTLY_COMPARABLE
    assert "comparison_condition is missing from one or both papers" in result.reasons


def test_comparison_request_requires_two_to_six_unique_papers_and_defaults_dimensions() -> None:
    request = ComparisonRequest(project_id=uuid4(), paper_ids=[uuid4(), uuid4()])

    assert request.dimensions == list(ComparisonDimension)
    with pytest.raises(ValidationError):
        ComparisonRequest(project_id=uuid4(), paper_ids=[uuid4()])
    duplicated_id = uuid4()
    with pytest.raises(ValidationError):
        ComparisonRequest(project_id=uuid4(), paper_ids=[duplicated_id, duplicated_id])
