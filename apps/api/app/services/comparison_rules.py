from app.schemas.comparison import (
    BenchmarkContext,
    ComparabilityResult,
    ComparabilityStatus,
)

_CONTEXT_FIELDS = ("task", "dataset", "split", "metric", "unit", "comparison_condition")


def compare_benchmark_contexts(
    left: BenchmarkContext, right: BenchmarkContext
) -> ComparabilityResult:
    """Only permit direct comparison when every benchmark dimension is known and equal."""
    reasons: list[str] = []
    for field in _CONTEXT_FIELDS:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            reasons.append(f"{field} is missing from one or both papers")
        elif left_value.casefold() != right_value.casefold():
            reasons.append(f"{field} differs between papers")

    if reasons:
        return ComparabilityResult(
            status=ComparabilityStatus.NOT_DIRECTLY_COMPARABLE,
            reasons=reasons,
        )
    return ComparabilityResult(status=ComparabilityStatus.DIRECTLY_COMPARABLE)
