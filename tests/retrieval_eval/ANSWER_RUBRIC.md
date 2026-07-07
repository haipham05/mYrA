# Answer and Source Evaluation Rubric

This rubric supplements the deterministic checks in `answer_metrics.py`; it does
not replace independent expected-source annotations. Record labels as
`relevant`, `not_relevant`, `faithful`, `not_faithful`, or `ambiguous`.
Ambiguous cases are stored with `score: null` and `status: unscored`; do not
force a binary judgment. Record the annotator and a short rationale in the
versioned evaluation manifest.

## Relevance

- `relevant` (1): the answer directly addresses the question using the expected
  fact(s), without substituting a nearby topic.
- `not_relevant` (0): it misses, contradicts, or materially changes the asked
  fact.
- `ambiguous`: the question, answer, or independent expected fact permits more
  than one defensible interpretation.

## Faithfulness

- `faithful` (1): each material factual claim is supported by the independently
  annotated paper/page/quote; qualifying citations also have verified text
  offsets whose page-text slice equals the cited quote.
- `not_faithful` (0): at least one material claim is unsupported, conflicts with
  its cited passage, or is attributed to the wrong paper/page/quote.
- `ambiguous`: source wording or claim boundaries do not permit a consistent
  support judgment. Do not resolve this with a bounding box or anchor presence.

## Machine-readable scoring

The deterministic report includes `citation_precision` (correct citations / all
citation objects), `correct_source_coverage` (distinct expected sources cited
correctly / expected sources), `unsupported_claim_rate` (unsupported generated
claims / generated claims, or `null` when there are none), and
`abstention_correct`. A correct source requires paper ID, page, exact annotated
quote, key phrase, verified status, and offsets matching the page text. Expected
claims map each claim ID to its independently specified required source IDs.
Malformed, stale, missing, or bbox-only anchors fail source validation. Empty
citation precision/coverage denominators are reported as `0.0`; the no-claim
unsupported rate is `null`.
