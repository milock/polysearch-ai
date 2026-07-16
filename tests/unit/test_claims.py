"""Unit tests for polysearch.extractors.claims.

The number-extraction cases guard the verification-yield fix where
enumerated-list markers and version/section numbers ("1.1", "2.3")
were extracted as factual figures and forced a NUMBER_MISMATCH on nearly every
claim. The ``extract_claims`` cases exercise the public entry point: only
sentences carrying a verifiable quote or number become Claims.
"""

from __future__ import annotations

import pytest

from polysearch.extractors.claims import _extract_numbers, extract_claims

URLS = ["https://example.gov/a", "https://docs.example.com/b"]


# ── Number-noise filter (ported from internal suite) ─────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        # Noise that must be dropped
        ("Sections 1.1, 1.2, 1.3, and 1.4 cover this.", []),
        ("Version 2.1 of the rule, phase 1.2 rollout.", []),
        ("Days in AR fell to 32 from 41.", []),
        ("Across 42 states in 2026.", []),
        ("", []),
        # Real figures that must be kept
        ("Cost fell to $127.14 per unit.", ["$127.14"]),
        ("Errors rose 7.7% year over year.", ["7.7%"]),
        ("A $1.2M annual loss for a mid-size team.", ["$1.2M"]),
        ("About 1,246 accounts signed on.", ["1,246"]),
        ("50 million records processed.", ["50 million"]),
    ],
)
def test_extract_numbers_filters_noise_keeps_figures(text, expected):
    assert _extract_numbers(text) == expected


def test_extract_numbers_respects_limit():
    text = "Rates of 1%, 2%, 3%, 4%, 5%, and 6% were observed."
    assert len(_extract_numbers(text, limit=3)) == 3


def test_extract_numbers_dedupes():
    text = "It rose 5% then fell 5% again, ending at 5%."
    assert _extract_numbers(text) == ["5%"]


# ── extract_claims ───────────────────────────────────────────────────────────


def test_sentence_with_number_becomes_claim():
    text = "The vendor reported that adoption climbed 42% across the quarter."
    claims = extract_claims(text, URLS)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.numbers == ["42%"]
    assert claim.quotes == []
    assert claim.source_urls == URLS
    assert claim.text == text
    assert claim.claim_id  # non-empty hash


def test_sentence_with_quote_becomes_claim():
    text = 'The report described the rollout as "a decisive strategic shift" for the team.'
    claims = extract_claims(text, URLS)
    assert len(claims) == 1
    assert claims[0].quotes == ["a decisive strategic shift"]
    assert claims[0].numbers == []


def test_pure_synthesis_sentence_is_dropped():
    # No quote, no number, and long enough to clear the filler filter.
    text = "The overall direction of the market appears broadly favorable for new entrants."
    assert extract_claims(text, URLS) == []


def test_short_filler_sentence_is_dropped():
    text = "It grew 9%."  # under the 30-char minimum
    assert extract_claims(text, URLS) == []


def test_number_noise_sentence_produces_no_claim():
    text = "Sections 1.1, 1.2, and 1.3 cover the applicable rollout phases in detail."
    assert extract_claims(text, URLS) == []


def test_empty_text_returns_no_claims():
    assert extract_claims("", URLS) == []


def test_no_source_urls_returns_no_claims():
    text = "The vendor reported that adoption climbed 42% across the quarter."
    assert extract_claims(text, []) == []


def test_multiple_sentences_yield_multiple_claims():
    text = (
        "Adoption climbed 42% across the quarter, the strongest result on record. "
        "The market outlook remains broadly positive for the coming year. "
        'One analyst called the shift "a genuine inflection point" for the category.'
    )
    claims = extract_claims(text, URLS)
    # Sentence 1 (number) and sentence 3 (quote) qualify; sentence 2 is dropped.
    assert len(claims) == 2
    assert claims[0].numbers == ["42%"]
    assert claims[1].quotes == ["a genuine inflection point"]


# ── Per-claim source localization (P2: tighten verification pairing) ──────────


def test_localizes_claim_to_topically_related_source():
    """When ``sources`` (url, snippet) are supplied, a claim is attributed only to
    sources whose snippet plausibly relates to the claim sentence — not the whole
    corpus."""
    text = "The federal funds target range held at 3.50% to 3.75% in June 2026."
    sources = [
        (
            "https://federalreserve.gov/fomc",
            "The FOMC kept the federal funds target range at 3.50 to 3.75 percent.",
        ),
        (
            "https://fda.gov/devices",
            "AI-enabled medical devices guidance for premarket submissions.",
        ),
    ]
    claims = extract_claims(text, URLS, sources=sources)
    assert len(claims) == 1
    # Only the topically-overlapping Fed page is attributed; the off-topic FDA
    # page is dropped even though it is in the corpus.
    assert claims[0].source_urls == ["https://federalreserve.gov/fomc"]


def test_number_in_snippet_localizes_source():
    """A source whose snippet contains the claim's figure is attributed even when
    prose token overlap is thin."""
    text = "Effective rates fell to 3.63% by mid-2026 from a year earlier."
    sources = [
        ("https://fred.example/series", "Monthly series value printed 3.63 for the period."),
        ("https://unrelated.example/x", "Completely different subject matter about gardening."),
    ]
    claims = extract_claims(text, URLS, sources=sources)
    assert claims[0].source_urls == ["https://fred.example/series"]


def test_falls_back_to_all_urls_when_no_source_relates():
    """No related source -> the claim keeps the corpus-wide URL list so it is
    never left un-verifiable."""
    text = "Adoption climbed 42% across the quarter, a record for the category."
    sources = [
        ("https://a.example/x", "An article about unrelated maritime shipping tariffs."),
        ("https://b.example/y", "Notes on medieval architecture and cathedrals."),
    ]
    claims = extract_claims(text, URLS, sources=sources)
    assert claims[0].source_urls == URLS


def test_no_sources_param_attaches_all_urls():
    """Backward-compatible: without ``sources``, every claim gets the full URL
    list, exactly as before."""
    text = "The vendor reported that adoption climbed 42% across the quarter."
    claims = extract_claims(text, URLS)
    assert claims[0].source_urls == URLS
