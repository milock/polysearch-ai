"""Unit tests for the community relevance gate.

``filter_community`` keeps items whose title/snippet mentions at least one topic
keyword and suppresses the whole layer (returns ``[]``) when more than 70% of
items are off-topic. ``evaluate_community`` exposes the same decision plus the
Pipeline Decisions note for the orchestrator to surface.
"""

from __future__ import annotations

from polysearch.community.filter import (
    SUPPRESSION_NOTE,
    evaluate_community,
    filter_community,
)
from polysearch.output.schema import SourceResult


def _sr(title: str, snippet: str = "") -> SourceResult:
    return SourceResult(
        url=f"https://ex.test/{abs(hash((title, snippet)))}",
        title=title,
        snippet=snippet,
        tier="COMMUNITY",
        published_date=None,
        layer="reddit",
        engagement=0,
    )


def test_keeps_on_topic_items() -> None:
    results = [
        _sr("widget scheduler tips"),
        _sr("best widget frameworks"),
        _sr("unrelated cooking recipe"),
    ]
    kept = filter_community(results, "widget scheduler", ["widget", "scheduler"])
    titles = [r.title for r in kept]
    assert "widget scheduler tips" in titles
    assert "best widget frameworks" in titles
    assert "unrelated cooking recipe" not in titles


def test_matches_against_snippet_too() -> None:
    results = [_sr("generic title", "deep dive into widgets and gadgets")]
    kept = filter_community(results, "widget", ["widget"])
    assert len(kept) == 1


def test_suppresses_layer_when_over_70_percent_off_topic() -> None:
    results = [
        _sr("widget news"),  # on-topic
        _sr("sports scores"),
        _sr("fantasy novel discussion"),
        _sr("stock market today"),
    ]  # 1/4 = 25% on-topic → >70% off-topic → suppress
    assert filter_community(results, "widget", ["widget"]) == []


def test_does_not_suppress_at_exactly_threshold() -> None:
    results = [
        _sr("widget one"),
        _sr("widget two"),
        _sr("sports"),
        _sr("cooking"),
    ]  # 2/4 = 50% on-topic, well above the 30% keep floor
    kept = filter_community(results, "widget", ["widget"])
    assert len(kept) == 2


def test_derives_terms_from_topic_when_none_given() -> None:
    results = [_sr("widget scheduler guide"), _sr("random noise")]
    kept = filter_community(results, "widget scheduler")
    assert [r.title for r in kept] == ["widget scheduler guide"]


def test_stopwords_and_short_tokens_do_not_leak_as_terms() -> None:
    # "the"/"and" are stopwords and "of" is too short, so none of them become
    # match terms — only "widgets" does. An item built purely from those filler
    # words must therefore read as off-topic.
    results = [_sr("the of and the"), _sr("widgets everywhere")]
    kept = filter_community(results, "the of widgets and")
    assert [r.title for r in kept] == ["widgets everywhere"]


def test_empty_input_returns_empty_and_not_suppressed() -> None:
    outcome = evaluate_community([], "widget", ["widget"])
    assert outcome.results == []
    assert outcome.suppressed is False
    assert outcome.note is None


def test_evaluate_surfaces_note_on_suppression() -> None:
    results = [_sr("widget"), _sr("a"), _sr("b"), _sr("c")]
    outcome = evaluate_community(results, "widget", ["widget"])
    assert outcome.suppressed is True
    assert outcome.results == []
    assert outcome.note == SUPPRESSION_NOTE


def test_evaluate_note_is_none_when_kept() -> None:
    results = [_sr("widget one"), _sr("widget two")]
    outcome = evaluate_community(results, "widget", ["widget"])
    assert outcome.suppressed is False
    assert outcome.note is None
    assert len(outcome.results) == 2
