import json
from uuid import uuid4

import pytest

from app.schemas.comparison import ComparabilityStatus, ComparisonDimension
from app.schemas.comparison_result import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonExcerpt,
    ComparisonMatrix,
)
from app.schemas.evidence import AnchorStatus, Citation, EvidenceItem
from app.services.comparison_synthesis import FindingKind, synthesize_comparison
from app.services.llm import GenerationResult, GenerationUsage, LLMProvider


class _StubProvider:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _MetadataProvider(LLMProvider):
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "test"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("metadata generation path should be used")

    async def generate_result(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return GenerationResult(
            content=self.response,
            requested_model="test-model",
            reported_model="test-model-reported",
            usage=GenerationUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        )


def _matrix(*, paper_count: int = 2, with_evidence: bool = True) -> ComparisonMatrix:
    paper_ids = [uuid4() for _ in range(paper_count)]
    cells: list[ComparisonCell] = []
    for index, paper_id in enumerate(paper_ids, start=1):
        excerpts: list[ComparisonExcerpt] = []
        if with_evidence:
            evidence_id = f"C{index}"
            quote = (
                "The Transformer uses scaled dot-product attention for sequence modeling."
                if index == 1
                else "The model uses recurrent layers to process each input sequence."
            )
            evidence = EvidenceItem(
                id=evidence_id,
                paper_id=paper_id,
                paper_title=f"Paper {index}",
                chunk_id=uuid4(),
                quote=quote,
                page_number=index,
                document_sha256=str(index) * 64,
            )
            citation = Citation(
                citation_index=index,
                evidence_id=evidence_id,
                paper_id=paper_id,
                page_number=index,
                quote=quote,
                document_sha256=str(index) * 64,
                anchor_status=AnchorStatus.UNRESOLVED,
            )
            excerpts.append(ComparisonExcerpt(evidence=evidence, citation=citation))
        cells.append(
            ComparisonCell(
                paper_id=paper_id,
                dimension=ComparisonDimension.METHOD_ARCHITECTURE,
                status=(
                    ComparisonCellStatus.EVIDENCE_AVAILABLE
                    if excerpts
                    else ComparisonCellStatus.NOT_FOUND
                ),
                excerpts=excerpts,
                message=None if excerpts else "Not reported in retrieved evidence",
            )
        )
    return ComparisonMatrix(
        project_id=uuid4(),
        question="Compare their sequence modeling methods",
        paper_ids=paper_ids,
        dimensions=[ComparisonDimension.METHOD_ARCHITECTURE],
        cells=cells,
    )


def _response(
    *findings: dict[str, object], benchmark_comparisons: list[dict[str, object]] | None = None
) -> str:
    return json.dumps(
        {
            "findings": list(findings),
            "benchmark_comparisons": benchmark_comparisons or [],
        }
    )


@pytest.mark.anyio
async def test_supported_direct_finding_is_returned_with_nullable_usage() -> None:
    matrix = _matrix()
    provider = _MetadataProvider(
        _response(
            {
                "text": "The Transformer uses scaled dot-product attention for sequence modeling.",
                "kind": "direct",
                "evidence_ids": ["C1"],
            }
        )
    )

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.outcome == "completed"
    assert result.findings[0].kind is FindingKind.DIRECT
    assert result.findings[0].evidence_ids == ["C1"]
    assert result.usage == GenerationUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28)
    assert result.requested_model == "test-model"
    assert provider.calls == 1


@pytest.mark.anyio
async def test_unsupported_direct_finding_is_filtered_with_gap_warning() -> None:
    provider = _StubProvider(
        _response(
            {
                "text": "The Transformer significantly improves every translation benchmark.",
                "kind": "direct",
                "evidence_ids": ["C1"],
            }
        )
    )

    result = await synthesize_comparison(_matrix(), provider=provider)

    assert result.findings == []
    assert result.outcome == "insufficient_evidence"
    assert any("unsupported direct finding" in warning for warning in result.warnings)
    assert result.usage is None


@pytest.mark.anyio
async def test_unknown_evidence_reference_is_rejected() -> None:
    provider = _StubProvider(
        _response(
            {
                "text": "The Transformer uses scaled dot-product attention for sequence modeling.",
                "kind": "direct",
                "evidence_ids": ["unknown"],
            }
        )
    )

    result = await synthesize_comparison(_matrix(), provider=provider)

    assert result.findings == []
    assert any("outside the comparison matrix" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_interpretation_requires_two_distinct_selected_papers() -> None:
    matrix = _matrix()
    provider = _StubProvider(
        _response(
            {
                "text": "The papers use different sequence modeling approaches.",
                "kind": "interpretation",
                "evidence_ids": ["C1"],
            }
        )
    )

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.findings == []
    assert any("independently supported" in warning for warning in result.warnings)


@pytest.mark.anyio
async def test_unsupported_interpretation_is_rejected_even_with_two_paper_sources() -> None:
    provider = _StubProvider(
        _response(
            {
                "text": "The papers use different sequence modeling approaches.",
                "kind": "interpretation",
                "evidence_ids": ["C1", "C2"],
            }
        )
    )

    result = await synthesize_comparison(_matrix(), provider=provider)

    assert result.findings == []
    assert any("independently supported" in warning for warning in result.warnings)


def _benchmark_proposal(matrix: ComparisonMatrix, *, right_split: str = "validation") -> dict:
    left_id, right_id = matrix.paper_ids
    quotes = [
        "Image classification on ImageNet validation split reports top-1 accuracy in percent, "
        "single crop 224px, result 90 percent.",
        f"Image classification on ImageNet {right_split} split reports top-1 accuracy in percent, "
        "single crop 224px, result 88 percent.",
    ]
    for cell, quote in zip(matrix.cells, quotes, strict=True):
        excerpt = cell.excerpts[0]
        cell.excerpts[0] = excerpt.model_copy(
            update={
                "evidence": excerpt.evidence.model_copy(update={"quote": quote}),
                "citation": excerpt.citation.model_copy(update={"quote": quote}),
            }
        )
    return {
        "left_paper_id": str(left_id),
        "right_paper_id": str(right_id),
        "left_context": {
            "task": "image classification",
            "dataset": "ImageNet",
            "split": "validation",
            "metric": "top-1 accuracy",
            "unit": "percent",
            "comparison_condition": "single crop 224px",
        },
        "right_context": {
            "task": "image classification",
            "dataset": "ImageNet",
            "split": right_split,
            "metric": "top-1 accuracy",
            "unit": "percent",
            "comparison_condition": "single crop 224px",
        },
        "left_result": "90 percent",
        "right_result": "88 percent",
        "left_context_quote": quotes[0],
        "right_context_quote": quotes[1],
        "left_evidence_ids": ["C1"],
        "right_evidence_ids": ["C2"],
    }


@pytest.mark.anyio
async def test_source_backed_matching_benchmarks_are_comparable() -> None:
    matrix = _matrix()
    provider = _StubProvider(_response(benchmark_comparisons=[_benchmark_proposal(matrix)]))

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.outcome == "completed"
    assert len(result.benchmark_comparisons) == 1
    comparison = result.benchmark_comparisons[0]
    assert comparison.comparability.status is ComparabilityStatus.DIRECTLY_COMPARABLE
    assert (comparison.left_result, comparison.right_result) == ("90 percent", "88 percent")


@pytest.mark.anyio
async def test_different_split_is_reported_not_directly_comparable() -> None:
    matrix = _matrix()
    provider = _StubProvider(
        _response(benchmark_comparisons=[_benchmark_proposal(matrix, right_split="test")])
    )

    result = await synthesize_comparison(matrix, provider=provider)

    assert len(result.benchmark_comparisons) == 1
    comparison = result.benchmark_comparisons[0]
    assert comparison.comparability.status is ComparabilityStatus.NOT_DIRECTLY_COMPARABLE
    assert any("split differs" in reason for reason in comparison.comparability.reasons)


@pytest.mark.anyio
async def test_invented_benchmark_value_is_omitted() -> None:
    matrix = _matrix()
    proposal = _benchmark_proposal(matrix)
    proposal["left_result"] = "190 percent"
    provider = _StubProvider(_response(benchmark_comparisons=[proposal]))

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.benchmark_comparisons == []


@pytest.mark.anyio
async def test_result_cannot_be_assigned_to_the_wrong_dataset() -> None:
    matrix = _matrix()
    ambiguous_quote = (
        "Image classification on ImageNet validation, top-1 accuracy percent, single crop "
        "224px: 80 percent; CIFAR validation, top-1 accuracy percent, single crop 224px: "
        "90 percent. The condition was evaluated at 80 and 90 percent."
    )
    excerpt = matrix.cells[0].excerpts[0]
    matrix.cells[0].excerpts[0] = excerpt.model_copy(
        update={
            "evidence": excerpt.evidence.model_copy(update={"quote": ambiguous_quote}),
            "citation": excerpt.citation.model_copy(update={"quote": ambiguous_quote}),
        }
    )

    proposal = _benchmark_proposal(matrix)
    proposal["left_context_quote"] = ambiguous_quote
    proposal["left_context"]["comparison_condition"] = "evaluated at 80 and 90 percent"
    provider = _StubProvider(_response(benchmark_comparisons=[proposal]))

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.benchmark_comparisons == []


@pytest.mark.anyio
async def test_result_unit_must_match_its_reported_metric_context() -> None:
    matrix = _matrix()
    mixed_quote = (
        "Image classification on ImageNet validation reports top-1 accuracy 80 percent "
        "under single crop 224px; CIFAR text retrieval reports BLEU 90 BLEU."
    )
    excerpt = matrix.cells[0].excerpts[0]
    matrix.cells[0].excerpts[0] = excerpt.model_copy(
        update={
            "evidence": excerpt.evidence.model_copy(update={"quote": mixed_quote}),
            "citation": excerpt.citation.model_copy(update={"quote": mixed_quote}),
        }
    )
    proposal = _benchmark_proposal(matrix)
    proposal["left_result"] = "90 BLEU"
    proposal["left_context_quote"] = mixed_quote
    provider = _StubProvider(_response(benchmark_comparisons=[proposal]))

    result = await synthesize_comparison(matrix, provider=provider)

    assert result.benchmark_comparisons == []


@pytest.mark.anyio
async def test_no_evidence_skips_provider_call() -> None:
    provider = _StubProvider("unused")

    result = await synthesize_comparison(_matrix(with_evidence=False), provider=provider)

    assert result.outcome == "insufficient_evidence"
    assert result.findings == []
    assert provider.calls == 0


@pytest.mark.anyio
async def test_provider_failure_returns_safe_empty_result() -> None:
    provider = _StubProvider(RuntimeError("private provider details"))

    result = await synthesize_comparison(_matrix(), provider=provider)

    assert result.outcome == "failed"
    assert result.findings == []
    assert "private provider details" not in " ".join(result.warnings)
    assert provider.calls == 1


@pytest.mark.anyio
async def test_numeric_winner_is_withheld_without_comparability_evidence() -> None:
    provider = _StubProvider(
        _response(
            {
                "text": "The Transformer is better with 28.4 BLEU than the RNN model.",
                "kind": "interpretation",
                "evidence_ids": ["C1", "C2"],
            }
        )
    )

    result = await synthesize_comparison(_matrix(), provider=provider)

    assert result.findings == []
    assert any("independently supported" in warning for warning in result.warnings)
