"""Tests for polysearch.verification.recovery.

Ported from the internal ``extensions.recovery_pass`` suite and adapted to the
public API: thresholds live on :class:`~polysearch.config.Settings` (not module
constants), ``research`` is reached via the ``providers.perplexity`` module, and
``merge_reports`` is complete over the public ``VerificationReport`` schema
(which carries ``blocked_sources``, absent from the internal report).
"""

from __future__ import annotations

from polysearch.config import Settings
from polysearch.output.schema import (
    Claim,
    VerificationReport,
    VerificationResult,
)
from polysearch.providers import perplexity
from polysearch.providers.perplexity import Citation, PerplexityResult
from polysearch.sources.authority import high_tier_domains
from polysearch.verification import recovery


# ── Builders ─────────────────────────────────────────────────────────────────


def _claim(claim_id: str, text: str, numbers: list[str] | None = None) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        numbers=["42"] if numbers is None else numbers,
        source_urls=["https://dead.example.com/a"],
    )


def _report_with_failure(claim_id: str) -> VerificationReport:
    result = VerificationResult(
        url="https://dead.example.com/a", claim_id=claim_id, status="URL_DEAD"
    )
    return VerificationReport(
        total_citations=10,
        verified_ok=1,
        broken=1,
        quote_mismatches=0,
        number_mismatches=0,
        paywalled=0,
        undated=0,
        skipped_budget=0,
        results=[result],
        total_cost_usd=0.0,
        total_duration_ms=0,
    )


def _pr(
    *,
    answer: str = "",
    citations: list[Citation] | None = None,
    error: str | None = None,
) -> PerplexityResult:
    return PerplexityResult(
        question="q",
        answer=answer,
        citations=citations or [],
        model="sonar-pro",
        search_results=[],
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
        duration_ms=0,
        error=error,
    )


def _cited() -> Citation:
    return Citation(url="https://cms.gov/rule", domain="cms.gov")


def _report(**overrides) -> VerificationReport:
    base = dict(
        total_citations=0,
        verified_ok=0,
        broken=0,
        quote_mismatches=0,
        number_mismatches=0,
        paywalled=0,
        undated=0,
        skipped_budget=0,
        results=[],
        total_cost_usd=0.0,
        total_duration_ms=0,
    )
    base.update(overrides)
    return VerificationReport(**base)


SETTINGS = Settings()


# ── should_recover ───────────────────────────────────────────────────────────


def test_should_recover_true_when_rate_low_and_enough_citations():
    report = _report(total_citations=10, verified_ok=1)
    assert recovery.should_recover(report, SETTINGS) is True


def test_should_recover_false_below_min_citations():
    # rate is terrible (0/3) but there aren't enough citations to bother.
    report = _report(total_citations=3, verified_ok=0)
    assert recovery.should_recover(report, SETTINGS) is False


def test_should_recover_false_when_rate_above_threshold():
    report = _report(total_citations=10, verified_ok=9)
    assert recovery.should_recover(report, SETTINGS) is False


def test_should_recover_rate_exactly_at_threshold_no_recovery():
    # 7/20 == 0.35 == recovery_rate_threshold. The guard is a strict ``<`` so a
    # rate sitting exactly on the threshold does NOT trigger recovery.
    report = _report(total_citations=20, verified_ok=7)
    assert SETTINGS.recovery_rate_threshold == 0.35
    assert recovery.should_recover(report, SETTINGS) is False


def test_should_recover_none_report():
    assert recovery.should_recover(None, SETTINGS) is False


# ── recover ──────────────────────────────────────────────────────────────────


async def test_recover_passes_high_tier_domain_filter(monkeypatch):
    calls: list[dict] = []

    async def _fake_research(query, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(perplexity, "research", _fake_research)

    claim = _claim("c1", "Something with a load-bearing number 42 in it")
    report = _report_with_failure("c1")

    await recovery.recover("derm billing topic", report, [claim], settings=SETTINGS)

    assert len(calls) == 1
    expected = high_tier_domains()[:20]
    assert calls[0]["domain_filter"] == expected
    # Perplexity's search_domain_filter API caps at 20 entries.
    assert len(expected) <= 20


async def test_recover_with_no_failed_claims_never_calls_research(monkeypatch):
    calls: list[dict] = []

    async def _fake_research(query, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(perplexity, "research", _fake_research)

    ok_result = VerificationResult(url="https://cms.gov/a", claim_id="c1", status="OK")
    report = _report(total_citations=10, verified_ok=10, results=[ok_result])
    claim = _claim("c1", "Something verified fine")

    out = await recovery.recover("topic", report, [claim], settings=SETTINGS)

    assert out == []
    assert calls == []


async def test_recover_drops_unverified_and_uncited_results(monkeypatch):
    """Honest-empty filtering: keep only results that carry citations and were
    not declared UNVERIFIED. Errored, UNVERIFIED-no-citation, and citation-less
    results are all dropped."""

    async def _fake_research(query, **kwargs):
        return [
            _pr(error="RuntimeError: boom"),  # errored -> drop
            _pr(answer="UNVERIFIED - no primary source found"),  # honest-empty -> drop
            _pr(answer="Some answer but no sources"),  # no citations -> drop
            _pr(answer="Solid answer", citations=[_cited()]),  # keep
        ]

    monkeypatch.setattr(perplexity, "research", _fake_research)

    claim = _claim("c1", "A claim with number 42")
    report = _report_with_failure("c1")

    out = await recovery.recover("topic", report, [claim], settings=SETTINGS)

    assert len(out) == 1
    assert out[0].answer == "Solid answer"
    assert out[0].citations[0].url == "https://cms.gov/rule"


async def test_recover_caps_and_dedupes_failed_claims(monkeypatch):
    """recovery_max_queries caps the fan-out; near-identical claim texts (same
    first 60 chars) collapse to a single query."""
    queried: list[str] = []

    async def _fake_research(query, **kwargs):
        queried.append(query)
        return []

    monkeypatch.setattr(perplexity, "research", _fake_research)

    prefix = "Independent dermatology practices lost revenue last year"  # 56 chars
    claims = [
        _claim("c1", prefix + " to downcoding number one"),
        _claim("c2", prefix + " to downcoding number two"),  # dupe of c1 by 60ch
        _claim("c3", "Medicare cut the conversion factor to 33.29 dollars"),
        _claim("c4", "Prior auth denials rose across the board significantly"),
        _claim("c5", "Staffing shortages hit small practices hardest of all"),
    ]
    results = [
        VerificationResult(
            url="https://dead.example.com/a", claim_id=c.claim_id, status="URL_DEAD"
        )
        for c in claims
    ]
    report = _report(total_citations=10, verified_ok=1, broken=5, results=results)

    await recovery.recover("topic", report, claims, settings=SETTINGS)

    # capped at recovery_max_queries (3)
    assert len(queried) == SETTINGS.recovery_max_queries == 3
    # the c1/c2 dupe collapses -> only one of them queried
    dupe_hits = sum(1 for q in queried if "number one" in q or "number two" in q)
    assert dupe_hits == 1


# ── merge_reports ────────────────────────────────────────────────────────────


def test_merge_reports_sums_total_duration_ms():
    """Known past bug: total_duration_ms was dropped on merge. Pin it summed."""
    first = _report(total_duration_ms=100)
    second = _report(total_duration_ms=50)
    assert recovery.merge_reports(first, second).total_duration_ms == 150


def test_merge_reports_ors_credits_and_sums_counters():
    first = _report(
        total_citations=2,
        verified_ok=1,
        fetch_blocked=1,
        blocked_sources=2,
        total_cost_usd=0.01,
        total_duration_ms=100,
        claims_total=2,
        claims_supported=1,
        credits_exhausted_hit=True,
    )
    second = _report(
        total_citations=3,
        verified_ok=3,
        fetch_blocked=0,
        blocked_sources=1,
        total_cost_usd=0.02,
        total_duration_ms=50,
        claims_total=3,
        claims_supported=3,
        credits_exhausted_hit=False,
    )

    merged = recovery.merge_reports(first, second)

    assert merged.credits_exhausted_hit is True
    assert merged.fetch_blocked == 1
    # blocked_sources is threaded on the PUBLIC merge (internal merge omitted it).
    assert merged.blocked_sources == 3
    assert merged.claims_total == 5
    assert merged.claims_supported == 4
    assert merged.total_citations == 5
    assert merged.total_cost_usd == 0.03
    assert merged.total_duration_ms == 150

    first_clean = first.model_copy(update={"credits_exhausted_hit": False})
    second_clean = second.model_copy(update={"credits_exhausted_hit": False})
    assert recovery.merge_reports(first_clean, second_clean).credits_exhausted_hit is False


def test_merge_reports_concatenates_results():
    r1 = VerificationResult(url="https://a.gov/1", claim_id="c1", status="OK")
    r2 = VerificationResult(url="https://b.gov/2", claim_id="c2", status="URL_DEAD")
    merged = recovery.merge_reports(_report(results=[r1]), _report(results=[r2]))
    assert [r.url for r in merged.results] == ["https://a.gov/1", "https://b.gov/2"]
