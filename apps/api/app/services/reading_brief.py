"""Formatting contracts for a one-call, evidence-backed paper reading brief."""

import re

SECTION_TITLES = (
    ("research_question", "Research question"),
    ("contributions", "Contributions"),
    ("method", "Method"),
    ("assumptions", "Assumptions"),
    ("evaluation_setup", "Evaluation setup"),
    ("results", "Results"),
    ("limitations", "Limitations"),
)

READING_BRIEF_GUIDANCE = """Return a concise reading brief using these headings in this order:
## Research question
## Contributions
## Method
## Assumptions
## Evaluation setup
## Results
## Limitations

Under each heading, include only claims supported by the supplied evidence and cite each factual
sentence with its evidence marker, such as [E1]. If the retrieved evidence does not report that
section, write exactly: Not found in retrieved evidence. Do not infer missing assumptions,
limitations, or evaluation details from general knowledge."""

_HEADING = re.compile(r"^\s{0,3}(?:#{1,3}\s+(.+?)|\*\*(.+?)\*\*:?|(.+?):)\s*$")
_CITATION = re.compile(r"\[(\d+)\]")


def parse_reading_brief_sections(content: str) -> list[dict[str, object]]:
    """Extract known headings without guessing that unstructured prose fills a field."""
    title_to_key = {title.casefold(): key for key, title in SECTION_TITLES}
    collected: dict[str, list[str]] = {key: [] for key, _ in SECTION_TITLES}
    current_key: str | None = None
    seen: set[str] = set()

    for line in content.splitlines():
        heading = _HEADING.match(line)
        candidate = next((part for part in heading.groups() if part), "") if heading else ""
        key = title_to_key.get(candidate.strip().rstrip(":*# ").casefold())
        if key is not None:
            current_key = key
            seen.add(key)
            continue
        if current_key is not None and line.strip():
            collected[current_key].append(line.strip())

    sections: list[dict[str, object]] = []
    for key, title in SECTION_TITLES:
        text = " ".join(collected[key]).strip()
        if not text:
            text = (
                "Not found in retrieved evidence."
                if key in seen
                else "Not returned in the reading-brief format."
            )
        sections.append(
            {
                "key": key,
                "title": title,
                "content": text,
                "citation_indexes": sorted(
                    {int(match.group(1)) for match in _CITATION.finditer(text)}
                ),
            }
        )
    return sections
