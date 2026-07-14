"""Topic-keyword relevance gate for the community signal layer.

The community adapters return whatever their platform ranks highly within the
recency window — frequently unrelated viral threads (sports, novels, memes) when
the actual topic has thin recent chatter. This gate keeps only items whose title
or snippet mentions at least one topic keyword and suppresses the entire layer
when more than 70% of items are off-topic, so a low-signal community layer never
dilutes synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from polysearch.output.schema import SourceResult

# Kept-item ratio below which the whole layer is suppressed. 0.3 means "suppress
# when more than 70% of items are off-topic".
_SUPPRESS_THRESHOLD = 0.3

SUPPRESSION_NOTE = (
    "Community signal suppressed: over 70% of retrieved items were off-topic "
    "for the query."
)

# Generic stopwords + a minimum token length keep filler words ("the", "and")
# and tiny tokens ("of") from becoming match terms that pass everything.
_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "over",
    "about", "what", "when", "where", "why", "how", "which", "who", "does",
    "are", "was", "were",
}
_MIN_TERM_LEN = 3


@dataclass
class FilterOutcome:
    """Result of the relevance gate: the kept items plus the suppression signal.

    ``filter_community`` returns only ``results``; callers that need to log the
    Pipeline Decisions note when the layer is dropped use ``evaluate_community``.
    """

    results: list[SourceResult]
    suppressed: bool
    note: str | None


def _derive_terms(topic: str) -> list[str]:
    """Tokenize a topic into lowercase match terms, dropping stopwords and
    tokens shorter than three characters. Order-preserving and deduplicated."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in topic.split():
        clean = "".join(ch for ch in token if ch.isalnum()).lower()
        if len(clean) >= _MIN_TERM_LEN and clean not in _STOPWORDS and clean not in seen:
            seen.add(clean)
            terms.append(clean)
    return terms


def _matches_any(result: SourceResult, terms: list[str]) -> bool:
    haystack = f"{result.title} {result.snippet}".lower()
    return any(term in haystack for term in terms)


def evaluate_community(
    results: list[SourceResult],
    topic: str,
    must_match_terms: list[str] | None = None,
    *,
    suppress_threshold: float = _SUPPRESS_THRESHOLD,
) -> FilterOutcome:
    """Apply the relevance gate and report the decision.

    ``must_match_terms`` overrides the terms derived from ``topic`` when given.
    Returns an empty, suppressed outcome (with ``SUPPRESSION_NOTE``) when the
    on-topic ratio falls below ``suppress_threshold``; otherwise returns the
    on-topic subset.
    """
    if not results:
        return FilterOutcome(results=[], suppressed=False, note=None)

    terms = [t.lower() for t in must_match_terms] if must_match_terms else _derive_terms(topic)
    if not terms:
        # No usable terms to gate on — pass everything through rather than
        # suppress a layer we can't fairly judge.
        return FilterOutcome(results=list(results), suppressed=False, note=None)

    kept = [r for r in results if _matches_any(r, terms)]
    if len(kept) / len(results) < suppress_threshold:
        return FilterOutcome(results=[], suppressed=True, note=SUPPRESSION_NOTE)
    return FilterOutcome(results=kept, suppressed=False, note=None)


def filter_community(
    results: list[SourceResult],
    topic: str,
    must_match_terms: list[str] | None = None,
    *,
    suppress_threshold: float = _SUPPRESS_THRESHOLD,
) -> list[SourceResult]:
    """Keep on-topic community items; return ``[]`` when the layer is suppressed.

    Thin wrapper over ``evaluate_community`` for callers that only need the
    filtered list.
    """
    return evaluate_community(
        results, topic, must_match_terms, suppress_threshold=suppress_threshold
    ).results


__all__ = ["FilterOutcome", "SUPPRESSION_NOTE", "evaluate_community", "filter_community"]
