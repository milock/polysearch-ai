"""Unit tests for polysearch.extractors.subject.extract_core_subject."""

from __future__ import annotations

from polysearch.extractors.subject import extract_core_subject

# A long, keyword-stuffed query that exceeds the 12-token trim threshold. It
# carries a clear core noun phrase ("project management software") wrapped in a
# question prefix, boolean/site operators, a stray year, and research filler.
STUFFED = (
    "what are the best project management software tools pricing comparison "
    "2026 for teams OR agencies reviews forum site:reddit.com"
)


# ── Short queries pass through untouched ────────────────────────────────────

def test_short_query_unchanged() -> None:
    query = "kubernetes horizontal pod autoscaling configuration"
    assert extract_core_subject(query) == query


def test_query_at_token_cap_unchanged() -> None:
    # Exactly max_tokens words → still returned verbatim.
    query = "one two three four five six seven eight nine ten eleven twelve"
    assert len(query.split()) == 12
    assert extract_core_subject(query) == query


def test_empty_query_unchanged() -> None:
    assert extract_core_subject("") == ""


# ── Long stuffed queries are trimmed to the core subject ────────────────────

def test_stuffed_query_capped_to_max_tokens() -> None:
    result = extract_core_subject(STUFFED)
    assert len(result.split()) <= 12


def test_stuffed_query_preserves_core_noun_phrase() -> None:
    result = extract_core_subject(STUFFED).lower()
    assert "project management software" in result


def test_stuffed_query_strips_site_operator() -> None:
    result = extract_core_subject(STUFFED).lower()
    assert "site:" not in result
    assert "reddit.com" not in result


def test_stuffed_query_strips_boolean_operator() -> None:
    tokens = [t.lower() for t in extract_core_subject(STUFFED).split()]
    assert "or" not in tokens


def test_stuffed_query_strips_year() -> None:
    assert "2026" not in extract_core_subject(STUFFED)


def test_stuffed_query_strips_filler() -> None:
    tokens = [t.lower() for t in extract_core_subject(STUFFED).split()]
    for filler in ("best", "reviews", "comparison"):
        assert filler not in tokens


# ── Parameterization ────────────────────────────────────────────────────────

def test_custom_max_tokens_respected() -> None:
    result = extract_core_subject(STUFFED, max_tokens=3)
    assert len(result.split()) <= 3


def test_never_returns_empty_for_all_noise() -> None:
    # A long query made entirely of noise words should still yield something
    # rather than collapsing to an empty string.
    query = "what are the best top latest new popular trending hot good great tools"
    assert len(query.split()) > 12
    assert extract_core_subject(query).strip() != ""
