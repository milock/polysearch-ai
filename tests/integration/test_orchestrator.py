"""Integration tests for the pipeline orchestrator (``run_research``).

Every provider is mocked or null — no network. Covers the flow contract:
  - tier-0 (all-null) run: report saves and notes every downgraded layer
  - full-mock standard run: md + json written, synthesis + verification present,
    at least one refinement iteration when the evaluator says the goal is unmet
  - one layer raising is isolated into pipeline_errors and the run still completes
  - quick depth runs zero refinement iterations
  - --max-iterations 0 disables the loop even at standard depth
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from polysearch.config import Settings
from polysearch.orchestrator import run_research
from polysearch.output.schema import LayerOutput, SourceResult, VerificationReport
from polysearch.providers.base import (
    NullResearchProvider,
    NullSynthesizer,
    NullVerifier,
    NullWebGrounder,
    Providers,
)

# A neutral snippet that carries a verifiable figure so extract_claims yields a claim.
_CLAIMY = "The metric rose to 42% in 2026 according to the cited source."


# ── builders ────────────────────────────────────────────────────────────────


def _src(url: str, *, layer: str = "research") -> SourceResult:
    return SourceResult(
        url=url,
        title="A report",
        snippet=_CLAIMY,
        tier="UNKNOWN",
        published_date="2026-01-01",
        layer=layer,
    )


def _empty_verification(*, total: int = 2, ok: int = 2) -> VerificationReport:
    return VerificationReport(
        total_citations=total,
        verified_ok=ok,
        broken=0,
        quote_mismatches=0,
        number_mismatches=0,
        paywalled=0,
        undated=0,
        skipped_budget=0,
        results=[],
        total_cost_usd=0.01,
        total_duration_ms=5,
        claims_total=1,
        claims_supported=1,
    )


class _FakeResearch:
    name = "research"

    async def research(self, topic, *, sub_questions, depth):
        return LayerOutput(
            layer="research",
            results=[_src("https://example.test/primary")],
            cost_usd=0.02,
            answers=[f"On {topic}, the figure reached 42% in 2026 per the cited source."],
        )


class _RaisingResearch:
    name = "research"

    async def research(self, topic, *, sub_questions, depth):
        raise ValueError("research backend exploded")


class _FreshResearch:
    """First pass yields one source; each refinement follow-up yields a NEW url so
    the loop's dedupe finds fresh material."""

    name = "research"

    def __init__(self):
        self.calls = 0

    async def research(self, topic, *, sub_questions, depth):
        self.calls += 1
        return LayerOutput(
            layer="research",
            results=[_src(f"https://example.test/{self.calls}-{topic[:8]}")],
            cost_usd=0.01,
            answers=[f"The {topic} figure was 42% in 2026 per the cited source."],
        )


class _FakeGrounder:
    name = "grounding"

    async def ground(self, topic, *, limit, scrape_top_k):
        return LayerOutput(layer="grounding")


class _FakeSynth:
    async def synthesize(self, topic, layers, *, style_constraints):
        return "## Synthesis\n\nThe key figure is 42% as of 2026.", 0.01


class _FakeVerifier:
    async def verify(self, claims, *, budget_usd, max_concurrency):
        return _empty_verification()


class _SpyVerifier:
    """Records verify-call kwargs and the claims it was handed, for asserts."""

    def __init__(self, *, credits_exhausted: bool = False):
        self.calls: list[dict] = []
        self.seen_claims: list = []
        self._credits_exhausted = credits_exhausted

    async def verify(self, claims, *, budget_usd, max_concurrency):
        self.calls.append({"budget_usd": budget_usd, "max_concurrency": max_concurrency})
        self.seen_claims.extend(claims)
        report = _empty_verification()
        report.credits_exhausted_hit = self._credits_exhausted
        return report


_FR_URL = "https://www.federalregister.gov/documents/2026/01/15/2026-00001/rule"
# Markdown that matches the bundled federalregister.gov schema (docket / effective
# date / CFR cite) and carries a verifiable dollar figure in the CFR context so the
# extracted fact becomes a checkable claim.
_FR_MARKDOWN = (
    "Document Number: 2026-00001\n"
    "Effective Date: January 15, 2026\n"
    "The rule adjusts payments by $1,250 under 42 CFR Part 405.\n"
)


class _GrounderWithHighTierPage:
    """Grounder whose last run scraped one HIGH-tier page with full markdown."""

    name = "grounding"

    def __init__(self):
        from polysearch.providers.firecrawl import GroundedItem

        self.last_items = [
            GroundedItem(
                url=_FR_URL,
                title="A Federal Register rule",
                markdown=_FR_MARKDOWN,
                domain="federalregister.gov",
                tier="HIGH",
            )
        ]

    async def ground(self, topic, *, limit, scrape_top_k):
        return LayerOutput(
            layer="grounding",
            results=[
                SourceResult(
                    url=_FR_URL,
                    title="A Federal Register rule",
                    snippet=_FR_MARKDOWN[:200],
                    tier="HIGH",
                    published_date="2026-01-15",
                    layer="grounding",
                )
            ],
        )


def _providers(
    *,
    research=None,
    grounder=None,
    synthesizer=None,
    verifier=None,
    community=None,
    deep_research=None,
    linkedin=None,
) -> Providers:
    return Providers(
        research=research or _FakeResearch(),
        deep_research=deep_research,
        grounder=grounder or _FakeGrounder(),
        synthesizer=synthesizer or _FakeSynth(),
        verifier=verifier or _FakeVerifier(),
        community_sources=community or [],
        linkedin=linkedin,
    )


def _null_providers() -> Providers:
    return Providers(
        research=NullResearchProvider("PERPLEXITY_API_KEY not set"),
        deep_research=None,
        grounder=NullWebGrounder("FIRECRAWL_API_KEY not set"),
        synthesizer=NullSynthesizer("OPENAI_API_KEY or ANTHROPIC_API_KEY not set"),
        verifier=NullVerifier("FIRECRAWL_API_KEY not set"),
        community_sources=[],
        linkedin=None,
    )


# ── coverage-evaluator fake (patches openai.AsyncOpenAI) ────────────────────


_VERDICT_JSON = json.dumps(
    {
        "goal_met": False,
        "coverage_score": 0.5,
        "gaps": ["needs more recent data"],
        "followup_queries": ["battery storage cost trend 2026"],
        "needs_deeper_verification": False,
        "reasoning": "coverage is partial",
    }
)


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_VERDICT_JSON))]
        )


# ── tests ───────────────────────────────────────────────────────────────────


async def test_tier0_run_notes_every_downgraded_layer(tmp_output_dir):
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_null_providers(),
        output_dir=tmp_output_dir,
    )

    # No synthesizer ran; the report still saves (degraded, not a crash).
    assert report.synthesis_md == ""
    files = list(tmp_output_dir.glob("*.md")) + list(tmp_output_dir.glob("*.json"))
    assert len(files) == 2

    joined = " | ".join(report.pipeline_errors)
    assert "PERPLEXITY_API_KEY not set" in joined  # research null reason
    assert "FIRECRAWL_API_KEY not set" in joined  # grounder null reason
    assert "OPENAI_API_KEY or ANTHROPIC_API_KEY not set" in joined  # synth reason


async def test_full_mock_standard_run_writes_report_and_refines(
    tmp_output_dir, monkeypatch
):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)

    report = await run_research(
        "grid battery storage economics",
        depth="standard",
        settings=Settings(openai_api_key="test-key"),
        providers=_providers(research=_FreshResearch()),
        output_dir=tmp_output_dir,
    )

    md_files = list(tmp_output_dir.glob("*.md"))
    json_files = list(tmp_output_dir.glob("*.json"))
    assert len(md_files) == 1 and len(json_files) == 1

    assert "42%" in report.synthesis_md
    assert report.verification is not None
    # The evaluator said "not met" with a fresh follow-up, so at least one
    # refinement round actually ran queries.
    assert any(trace.queries_run for trace in report.refinement_iterations)
    md_text = md_files[0].read_text()
    assert "## Synthesis" in md_text


async def test_one_layer_raising_is_isolated(tmp_output_dir):
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(research=_RaisingResearch()),
        output_dir=tmp_output_dir,
    )

    # The run completed and wrote a report despite the raising layer.
    assert list(tmp_output_dir.glob("*.md"))
    assert any(e.startswith("research:") for e in report.pipeline_errors)


async def test_quick_depth_runs_zero_refinement_iterations(tmp_output_dir):
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(openai_api_key="test-key"),
        providers=_providers(),
        output_dir=tmp_output_dir,
    )
    assert report.refinement_iterations == []


async def test_first_pass_verify_uses_depth_profile_budget_and_concurrency(
    tmp_output_dir,
):
    # Quick depth: profile.verify_budget_usd == 1, verify_concurrency == 5, and no
    # refinement pass to muddy which verify call is the first one.
    spy = _SpyVerifier()
    await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(verifier=spy),
        output_dir=tmp_output_dir,
    )
    assert spy.calls, "verifier was never called"
    assert spy.calls[0]["budget_usd"] == 1
    assert spy.calls[0]["max_concurrency"] == 5


async def test_verify_budget_override_is_passed_through(tmp_output_dir):
    spy = _SpyVerifier()
    await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(verifier=spy),
        verify_budget=0.5,
        output_dir=tmp_output_dir,
    )
    assert spy.calls[0]["budget_usd"] == 0.5


async def test_credits_exhausted_flag_raises_pipeline_alert(tmp_output_dir):
    # The alert reads the report-level boolean flag — never a string scan.
    spy = _SpyVerifier(credits_exhausted=True)
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(verifier=spy),
        output_dir=tmp_output_dir,
    )
    assert any("credits exhausted" in e for e in report.pipeline_errors)


async def test_authoritative_extraction_runs_on_high_tier_pages(tmp_output_dir):
    # Standard depth (authoritative_top_k == 2); max_iterations=0 keeps refinement
    # out so the spy only sees the first-pass claim set.
    spy = _SpyVerifier()
    report = await run_research(
        "medicare payment rule effective date",
        depth="standard",
        settings=Settings(),
        providers=_providers(
            grounder=_GrounderWithHighTierPage(),
            verifier=spy,
        ),
        max_iterations=0,
        output_dir=tmp_output_dir,
    )

    # Counted in Pipeline Decisions (rendered from the classifier's reasons).
    reasons = " | ".join(report.classification.get("reasons", []))
    assert "authoritative extraction:" in reasons
    assert "HIGH-tier page" in reasons

    # The extracted fact reached the claim input and is attributed to its page.
    # The $1,250 figure lives ONLY in the page markdown, which reaches claims
    # solely via authoritative extraction (grounding snippets are not mined), so
    # a claim carrying it proves the fact -> claim path.
    fact_claims = [c for c in spy.seen_claims if "1,250" in c.text]
    assert fact_claims
    assert any(_FR_URL in c.source_urls for c in fact_claims)

    md_text = next(tmp_output_dir.glob("*.md")).read_text()
    assert "authoritative extraction:" in md_text


async def test_authoritative_extraction_skipped_at_quick_depth(tmp_output_dir):
    # Quick depth has authoritative_top_k == 0 -> the step is skipped entirely.
    spy = _SpyVerifier()
    report = await run_research(
        "medicare payment rule effective date",
        depth="quick",
        settings=Settings(),
        providers=_providers(
            grounder=_GrounderWithHighTierPage(),
            verifier=spy,
        ),
        output_dir=tmp_output_dir,
    )
    reasons = " | ".join(report.classification.get("reasons", []))
    assert "authoritative extraction:" not in reasons
    # No fact-derived claim: the page's $1,250 figure never reaches the claim set
    # because the extraction step was skipped at quick depth.
    assert not any("1,250" in c.text for c in spy.seen_claims)


async def test_stale_grounder_last_items_not_mined_when_grounding_skipped(
    tmp_output_dir,
):
    # Library-reuse hazard: a reused Providers bundle keeps the prior run's
    # last_items. If grounding is NOT run this turn, those stale HIGH-tier pages
    # must not be mined into this report (silent cross-topic contamination).
    spy = _SpyVerifier()
    grounder = _GrounderWithHighTierPage()  # last_items already populated (as if prior run)
    report = await run_research(
        "medicare payment rule effective date",
        depth="standard",  # authoritative_top_k == 2
        settings=Settings(),
        providers=_providers(grounder=grounder, verifier=spy),
        enabled_layers={"research"},  # grounding excluded this run
        max_iterations=0,
        output_dir=tmp_output_dir,
    )
    reasons = " | ".join(report.classification.get("reasons", []))
    assert "authoritative extraction:" not in reasons
    assert not any("1,250" in c.text for c in spy.seen_claims)


async def test_max_iterations_zero_disables_loop(tmp_output_dir, monkeypatch):
    # Even at standard depth (profile allows 2), --max-iterations 0 wins.
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    report = await run_research(
        "grid battery storage economics",
        depth="standard",
        settings=Settings(openai_api_key="test-key"),
        providers=_providers(research=_FreshResearch()),
        max_iterations=0,
        output_dir=tmp_output_dir,
    )
    assert report.refinement_iterations == []


# ── time budget (task r3c) ───────────────────────────────────────────────────


class _SlowResearch:
    """Sleeps well past any tight test budget before returning a normal layer."""

    name = "research"

    def __init__(self, sleep_s: float = 0.5):
        self._sleep_s = sleep_s

    async def research(self, topic, *, sub_questions, depth):
        await asyncio.sleep(self._sleep_s)
        return LayerOutput(layer="research", results=[_src("https://example.test/slow")])


class _SlowCommunitySource:
    """A community source stuck well past the per-adapter timeout."""

    name = "slow"
    last_error = None

    def __init__(self, sleep_s: float = 0.5):
        self._sleep_s = sleep_s

    async def search(self, topic, *, window_days, limit):
        await asyncio.sleep(self._sleep_s)
        return [
            SourceResult(
                url="https://example.test/slow-community",
                title="slow",
                snippet="never arrives in time",
                tier="COMMUNITY",
                layer="slow",
            )
        ]


class _FastCommunitySource:
    """Returns immediately — its result must survive a slow sibling's timeout."""

    name = "fast"
    last_error = None

    async def search(self, topic, *, window_days, limit):
        return [
            SourceResult(
                url="https://example.test/fast-community",
                title="grid battery storage economics discussion",
                snippet="arrived on time, on-topic for the relevance gate",
                tier="COMMUNITY",
                layer="fast",
            )
        ]


class _SpyDeepResearch:
    """Records the ``time_budget_s`` kwarg the orchestrator threads through."""

    name = "deep_research"

    def __init__(self):
        self.received_time_budget_s: float | None = "not-called"  # sentinel

    async def research(self, topic, *, sub_questions, depth, time_budget_s=None):
        self.received_time_budget_s = time_budget_s
        return LayerOutput(layer="deep_research")


async def test_layer_cancelled_at_budget_records_error_and_writes_report(tmp_output_dir):
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(research=_SlowResearch(sleep_s=0.5)),
        time_budget_s=0.05,
        output_dir=tmp_output_dir,
    )

    # The run degrades gracefully — a cancelled layer never blocks the report.
    assert list(tmp_output_dir.glob("*.md"))
    assert any(
        e.startswith("research:") and "exceeded remaining time budget" in e
        for e in report.pipeline_errors
    )
    research_layer = next(lyr for lyr in report.layers if lyr.layer == "research")
    assert research_layer.error is not None
    assert research_layer.results == []
    # Cancelled well before the full 500ms sleep would have elapsed.
    assert research_layer.duration_ms < 400


async def test_deep_research_receives_remaining_time_budget(tmp_output_dir):
    spy = _SpyDeepResearch()
    await run_research(
        "grid battery storage economics",
        depth="deep",
        settings=Settings(),
        providers=_providers(deep_research=spy),
        time_budget_s=90.0,
        output_dir=tmp_output_dir,
    )
    assert spy.received_time_budget_s is not None
    assert spy.received_time_budget_s != "not-called"
    # Some time has elapsed since the budget snapshot was taken; allow slack.
    assert 0 < spy.received_time_budget_s <= 90.0


async def test_community_adapter_timeout_lets_siblings_survive(tmp_output_dir):
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(community_adapter_timeout_s=0.05),
        providers=_providers(
            community=[_SlowCommunitySource(sleep_s=0.5), _FastCommunitySource()]
        ),
        output_dir=tmp_output_dir,
    )

    community_layer = next(lyr for lyr in report.layers if lyr.layer == "community")
    urls = {s.url for s in community_layer.results}
    assert "https://example.test/fast-community" in urls
    assert "https://example.test/slow-community" not in urls
    assert any(
        e.startswith("community/slow:") and "timed out after" in e
        for e in report.pipeline_errors
    )
    # Bounded well under the slow source's full 500ms sleep.
    assert community_layer.duration_ms < 400


async def test_no_time_budget_set_behavior_unchanged(tmp_output_dir):
    # A layer that takes some measurable-but-short time must complete normally
    # (no cancellation, no budget-related pipeline error) when time_budget_s is
    # never set — the default, pre-existing behavior.
    report = await run_research(
        "grid battery storage economics",
        depth="quick",
        settings=Settings(),
        providers=_providers(research=_SlowResearch(sleep_s=0.02)),
        output_dir=tmp_output_dir,
    )
    assert not any("time budget" in e for e in report.pipeline_errors)
    research_layer = next(lyr for lyr in report.layers if lyr.layer == "research")
    assert research_layer.error is None
    assert research_layer.results != []


async def test_refinement_skipped_when_budget_exhausted(tmp_output_dir, monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    report = await run_research(
        "grid battery storage economics",
        depth="standard",
        settings=Settings(openai_api_key="test-key"),
        providers=_providers(research=_FreshResearch()),
        time_budget_s=0.0,
        output_dir=tmp_output_dir,
    )
    assert report.refinement_iterations == []
    assert any(
        "refinement" in e and "skipped" in e and "time budget" in e
        for e in report.pipeline_errors
    )
    # Still writes a real report despite the exhausted budget.
    assert list(tmp_output_dir.glob("*.md"))
