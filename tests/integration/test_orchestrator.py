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
    """Records the kwargs of its first verify call for budget/concurrency asserts."""

    def __init__(self, *, credits_exhausted: bool = False):
        self.calls: list[dict] = []
        self._credits_exhausted = credits_exhausted

    async def verify(self, claims, *, budget_usd, max_concurrency):
        self.calls.append({"budget_usd": budget_usd, "max_concurrency": max_concurrency})
        report = _empty_verification()
        report.credits_exhausted_hit = self._credits_exhausted
        return report


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
