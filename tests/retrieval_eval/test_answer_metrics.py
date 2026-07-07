from answer_metrics import (
    Citation,
    EvaluationCase,
    ExpectedClaim,
    ExpectedSource,
    GeneratedClaim,
    score_answer,
    score_human_label,
)

PAGE_TEXT = "The transformer uses exact attention for sequence modeling."
EXACT_QUOTE = "exact attention for sequence modeling"


def source() -> ExpectedSource:
    return ExpectedSource(
        source_id="s1",
        paper_id="paper-a",
        page_number=3,
        exact_quote=EXACT_QUOTE,
        key_phrase="exact attention",
        page_text=PAGE_TEXT,
    )


def citation(**overrides: object) -> Citation:
    values: dict[str, object] = {
        "citation_id": "c1",
        "paper_id": "paper-a",
        "page_number": 3,
        "quote": EXACT_QUOTE,
        "anchor_verified": True,
        "source_char_start": PAGE_TEXT.index(EXACT_QUOTE),
        "source_char_end": PAGE_TEXT.index(EXACT_QUOTE) + len(EXACT_QUOTE),
    }
    values.update(overrides)
    return Citation(**values)  # type: ignore[arg-type]


def case(
    citations: tuple[Citation, ...] | None = None,
    *,
    generated_claims: tuple[GeneratedClaim, ...] = (
        GeneratedClaim("claim-1", ("c1",)),
    ),
    expected_abstention: bool = False,
    actual_abstention: bool = False,
) -> EvaluationCase:
    return EvaluationCase(
        expected_sources=(source(),),
        citations=(citation(),) if citations is None else citations,
        expected_claims=(ExpectedClaim("claim-1", ("s1",)),),
        generated_claims=generated_claims,
        expected_abstention=expected_abstention,
        actual_abstention=actual_abstention,
    )


def test_exact_quote_and_verified_offsets_count_as_source_correct() -> None:
    result = score_answer(case())
    assert result.citation_precision == 1.0
    assert result.correct_source_coverage == 1.0
    assert result.unsupported_claim_rate == 0.0
    assert result.abstention_correct is True
    assert result.to_dict()["correct_citations"] == 1


def test_wrong_paper_page_or_quote_does_not_count() -> None:
    for changed in (
        citation(paper_id="paper-b"),
        citation(page_number=4),
        citation(quote="approximate attention for sequence modeling"),
    ):
        result = score_answer(case((changed,)))
        assert result.citation_precision == 0.0
        assert result.correct_source_coverage == 0.0
        assert result.unsupported_claim_rate == 1.0


def test_unverified_or_stale_anchor_does_not_count() -> None:
    stale_offset = citation(source_char_start=0, source_char_end=len(EXACT_QUOTE))
    missing_offset = citation(source_char_start=None, source_char_end=None)
    unverified = citation(anchor_verified=False)
    for item in (stale_offset, missing_offset, unverified):
        assert score_answer(case((item,))).correct_citations == 0


def test_bbox_cannot_count_without_text_anchor_offsets() -> None:
    # The schema intentionally has no bbox-only validity path.
    bbox_like = citation(source_char_start=None, source_char_end=None)
    assert score_answer(case((bbox_like,))).citation_precision == 0.0


def test_unsupported_claim_and_missing_expected_support_are_counted() -> None:
    unsupported = GeneratedClaim("claim-1", ())
    assert (
        score_answer(case(generated_claims=(unsupported,))).unsupported_claim_rate
        == 1.0
    )


def test_correct_abstention_and_no_claim_denominator() -> None:
    result = score_answer(
        EvaluationCase((), (), (), (), expected_abstention=True, actual_abstention=True)
    )
    assert result.abstention_correct is True
    assert result.unsupported_claim_rate is None


def test_incorrect_abstention_is_reported() -> None:
    result = score_answer(case(expected_abstention=True, actual_abstention=False))
    assert result.abstention_correct is False


def test_ambiguous_human_label_is_explicitly_unscored() -> None:
    assert score_human_label("ambiguous").to_dict() == {
        "label": "ambiguous",
        "score": None,
        "status": "unscored",
        "reason": "ambiguous",
    }


def test_human_positive_and_negative_labels_are_machine_readable() -> None:
    assert score_human_label("faithful").score == 1
    assert score_human_label("not_faithful").score == 0
