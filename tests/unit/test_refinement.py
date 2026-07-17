"""Tests for polysearch.refinement — the goal-driven refinement loop.

All LLM / provider calls are mocked. Covers every loop guard: goal-met exit,
max-iterations stop, cost-ceiling stop (profile + settings override), no-unique-
queries exit, zero-new-sources dry exit, URL-canonical dedupe, evaluator parse-
failure graceful abort, Anthropic-only graceful degrade, quick depth (zero
iterations), remaining-ceiling verify budget, and recovery re-verification.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polysearch.config import DEPTH_PROFILES, Settings
from polysearch.output.schema import (
    Claim,
    CoverageVerdict,
    LayerOutput,
    SourceResult,
    VerificationReport,
)
from polysearch.providers.perplexity import Citation, PerplexityResult
from polysearch.refinement import (
    RefinementState,
    _canonical_url,
    evaluate_coverage,
    run_refinement,
)

# A snippet that carries a verifiable figure so extract_claims yields a claim.
_CLAIMY = "Grid battery storage costs fell to 42% of the 2020 level last year."


# ── builders ──────────────────────────────────────────────────────────────


def _src(url: str, *, snippet: str = _CLAIMY, layer: str = "research") -> SourceResult:
    return SourceResult(
        url=url,
        title="Battery cost report",
        snippet=snippet,
        tier="UNKNOWN",
        published_date="2026-01-01",
        layer=layer,
    )


def _layer(sources: list[SourceResult], *, layer: str = "research", cost: float = 0.0):
    return LayerOutput(layer=layer, results=list(sources), cost_usd=cost)


def _perp(question: str, url: str) -> PerplexityResult:
    return PerplexityResult(
        question=question,
        answer=(
            f'The report put the figure at 42% for "{question}" per the cited '
            "source and its underlying dataset."
        ),
        citations=[Citation(url=url, domain="example.com")],
        model="sonar-pro",
        search_results=[],
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
        duration_ms=0,
    )


def _empty_verification() -> VerificationReport:
    return VerificationReport(
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


def _profile(*, iterations: int = 4, ceiling: float = 8.0):
    return replace(
        DEPTH_PROFILES["deep"],
        max_refinement_iterations=iterations,
        refinement_cost_ceiling_usd=ceiling,
    )


def _settings(**kw) -> Settings:
    kw.setdefault("openai_api_key", "fake")
    return Settings(**kw)


def _state(*, sources: list[SourceResult] | None = None) -> RefinementState:
    srcs = list(sources or [])
    seen = {_canonical_url(s.url) for s in srcs}
    return RefinementState(
        layers=[_layer(srcs)] if srcs else [],
        claims=[],
        verification=_empty_verification(),
        synthesis_md="Existing synthesis body.",
        prior_queries=["grid-scale battery storage costs"],
        seen_urls=seen,
    )


def _providers(*, research=None, ground=None, verify=None, synthesize=None):
    async def _default_research(query, *, sub_questions, depth):
        return _layer([_src(f"https://new.example.com/{query}")])

    async def _default_ground(query, *, limit, scrape_top_k):
        return _layer([], layer="grounding")

    async def _default_verify(claims, *, budget_usd, max_concurrency):
        return _empty_verification()

    async def _default_synth(topic, layers, *, style_constraints):
        return "Re-synthesized body.", 0.0

    return SimpleNamespace(
        research=SimpleNamespace(research=research or _default_research),
        grounder=SimpleNamespace(ground=ground or _default_ground),
        verifier=SimpleNamespace(verify=verify or _default_verify),
        synthesizer=SimpleNamespace(synthesize=synthesize or _default_synth),
    )


async def _noop_recover(topic, verification, claims, **kwargs):
    return []


# ── canonical URL ───────────────────────────────────────────────────────────


def test_canonical_url_normalizes_host_scheme_and_trailing_slash():
    a = _canonical_url("https://WWW.Example.com/Path/")
    b = _canonical_url("http://example.com/Path")
    assert a == b


def test_canonical_url_drops_fragment_keeps_query():
    assert _canonical_url("https://x.com/a?b=1#frag") == _canonical_url(
        "https://x.com/a?b=1"
    )
    assert _canonical_url("https://x.com/a?b=1") != _canonical_url("https://x.com/a?b=2")


# ── loop guards ─────────────────────────────────────────────────────────────


async def test_goal_met_iter1_single_trace_no_queries():
    research_spy = AsyncMock(return_value=_layer([]))
    ground_spy = AsyncMock(return_value=_layer([], layer="grounding"))

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=True, coverage_score=0.95, followup_queries=["unused"]
        )

    providers = _providers()
    providers.research.research = research_spy
    providers.grounder.ground = ground_spy
    state = _state()

    traces = await run_refinement(
        "topic", state, providers, _profile(), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].verdict.goal_met is True
    assert traces[0].stopped_reason == "goal_met"
    assert traces[0].queries_run == []  # queries NOT run when goal met
    research_spy.assert_not_called()
    ground_spy.assert_not_called()


async def test_never_met_stops_at_max_iterations():
    counter = {"n": 0}

    async def _eval(**kwargs):
        counter["n"] += 1
        # Unique follow-ups each iteration so the loop never dry-exits early.
        return CoverageVerdict(
            goal_met=False, coverage_score=0.4,
            followup_queries=[f"followup number {counter['n']}"],
        )

    traces = await run_refinement(
        "topic", _state(), _providers(), _profile(iterations=2, ceiling=100.0),
        settings=_settings(), evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 2
    assert all(t.verdict.goal_met is False for t in traces)


async def test_cost_ceiling_stops_loop():
    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.3, followup_queries=["expensive angle"]
        )

    async def _pricey_research(query, *, sub_questions, depth):
        return _layer([_src(f"https://new.example.com/{query}")], cost=5.0)

    traces = await run_refinement(
        "topic", _state(), _providers(research=_pricey_research),
        _profile(iterations=5, ceiling=1.0),
        settings=_settings(), evaluate=_eval, recover=_noop_recover,
    )

    assert traces[-1].stopped_reason == "cost_ceiling"
    assert len(traces) < 5


async def test_settings_ceiling_overrides_profile():
    # Profile ceiling is generous; the settings override is tight -> stop early.
    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.3, followup_queries=["expensive angle"]
        )

    async def _pricey_research(query, *, sub_questions, depth):
        return _layer([_src(f"https://new.example.com/{query}")], cost=5.0)

    traces = await run_refinement(
        "topic", _state(), _providers(research=_pricey_research),
        _profile(iterations=5, ceiling=100.0),
        settings=_settings(refinement_cost_ceiling_usd=1.0),
        evaluate=_eval, recover=_noop_recover,
    )

    assert traces[-1].stopped_reason == "cost_ceiling"
    assert len(traces) < 5


async def test_zero_new_sources_dry_stop():
    seen = _src("https://seen.example.com/a")
    state = _state(sources=[seen])

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.5, followup_queries=["rehash"]
        )

    async def _research(query, *, sub_questions, depth):
        return _layer([_src("https://seen.example.com/a")])  # already seen

    traces = await run_refinement(
        "topic", state, _providers(research=_research), _profile(),
        settings=_settings(), evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].stopped_reason == "dry"
    assert traces[0].new_sources == 0


async def test_url_dedupe_excludes_seen():
    seen = _src("https://seen.example.com/a")
    state = _state(sources=[seen])

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.5, followup_queries=["mix"]
        )

    async def _research(query, *, sub_questions, depth):
        # One seen URL (deduped out) + one genuinely new URL.
        return _layer([
            _src("https://seen.example.com/a"),
            _src("https://fresh.example.com/b"),
        ])

    traces = await run_refinement(
        "topic", state, _providers(research=_research),
        _profile(iterations=1), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].new_sources == 1  # only the fresh URL counts


async def test_no_new_queries_exit_when_only_rephrasings():
    research_spy = AsyncMock(return_value=_layer([]))

    async def _eval(**kwargs):
        # A case-variant of the seeded prior query -> unique guard strips it.
        return CoverageVerdict(
            goal_met=False, coverage_score=0.5,
            followup_queries=["GRID-SCALE battery storage costs"],
        )

    providers = _providers()
    providers.research.research = research_spy

    traces = await run_refinement(
        "topic", _state(), providers, _profile(), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].stopped_reason == "no_new_queries"
    research_spy.assert_not_called()


async def test_parse_failure_graceful_abort():
    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.0,
            parse_error="Expecting value: line 1 column 1",
        )

    state = _state()
    traces = await run_refinement(
        "topic", state, _providers(), _profile(), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].stopped_reason == "parse_abort"
    # Never crashes; records a pipeline error the orchestrator surfaces.
    assert any("refinement" in e for e in state.pipeline_errors)


async def test_anthropic_only_degrades_to_no_refinement():
    # No OpenAI key -> real evaluate_coverage returns parse_error -> loop aborts
    # gracefully with no provider calls. Uses the REAL evaluator (not mocked).
    research_spy = AsyncMock(return_value=_layer([]))
    providers = _providers()
    providers.research.research = research_spy
    state = _state()

    traces = await run_refinement(
        "topic", state, providers, _profile(),
        settings=Settings(anthropic_api_key="x"),  # no openai key
        recover=_noop_recover,
    )

    assert len(traces) == 1
    assert traces[0].stopped_reason == "parse_abort"
    assert any("OpenAI" in e for e in state.pipeline_errors)
    research_spy.assert_not_called()


async def test_max_iterations_zero_returns_empty_and_never_evaluates():
    eval_spy = AsyncMock()
    traces = await run_refinement(
        "topic", _state(), _providers(), _profile(iterations=0, ceiling=0.0),
        settings=_settings(), evaluate=eval_spy, recover=_noop_recover,
    )
    assert traces == []
    eval_spy.assert_not_called()


async def test_new_claims_verified_with_remaining_ceiling_budget():
    # First iteration, zero prior spend, zero-cost follow-ups -> the remaining
    # budget handed to verify() must equal the full ceiling (not a constant).
    verify_calls: list[dict] = []

    async def _verify(claims, **kwargs):
        verify_calls.append(kwargs)
        return _empty_verification()

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.4,
            followup_queries=["fresh statistics angle"],
            needs_deeper_verification=False,
        )

    async def _research(query, *, sub_questions, depth):
        return _layer([_src("https://fresh.example.com/new")])

    await run_refinement(
        "topic", _state(), _providers(research=_research, verify=_verify),
        _profile(iterations=1, ceiling=8.0), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert len(verify_calls) == 1
    assert verify_calls[0]["budget_usd"] == 8.0  # remaining ceiling, not a constant


async def test_needs_deeper_verification_runs_recovery():
    recover_spy = AsyncMock(return_value=[])

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.5,
            followup_queries=["deep angle"], needs_deeper_verification=True,
        )

    await run_refinement(
        "topic", _state(), _providers(), _profile(iterations=1),
        settings=_settings(), evaluate=_eval, recover=recover_spy,
    )

    recover_spy.assert_awaited()


async def test_recovered_claims_are_reverified_within_remaining_budget():
    # needs_deeper_verification -> recovery returns claim-bearing results, which
    # must themselves be verified (second verify call) with the remaining budget.
    verify_calls: list[dict] = []

    async def _verify(claims, **kwargs):
        verify_calls.append(kwargs)
        return _empty_verification()

    async def _recover(topic, verification, claims, **kwargs):
        return [_perp("recovered", "https://recovered.example.com/x")]

    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.4,
            followup_queries=["fresh angle"], needs_deeper_verification=True,
        )

    async def _research(query, *, sub_questions, depth):
        return _layer([_src("https://fresh.example.com/new")])

    await run_refinement(
        "topic", _state(), _providers(research=_research, verify=_verify),
        _profile(iterations=1, ceiling=8.0), settings=_settings(),
        evaluate=_eval, recover=_recover,
    )

    # Two verify calls: new follow-up claims, then recovered claims.
    assert len(verify_calls) == 2
    assert all(c["budget_usd"] > 0 for c in verify_calls)


async def test_resynthesis_and_corpus_growth():
    async def _eval(**kwargs):
        return CoverageVerdict(
            goal_met=False, coverage_score=0.4, followup_queries=["fresh angle"]
        )

    async def _research(query, *, sub_questions, depth):
        return _layer([_src("https://fresh.example.com/new")])

    state = _state()
    layers_before = len(state.layers)

    await run_refinement(
        "topic", state, _providers(research=_research),
        _profile(iterations=1, ceiling=8.0), settings=_settings(),
        evaluate=_eval, recover=_noop_recover,
    )

    assert state.synthesis_md == "Re-synthesized body."
    assert len(state.layers) > layers_before  # refinement layer appended
    assert "grid-scale battery storage costs" in state.prior_queries
    assert "fresh angle" in state.prior_queries
    assert _canonical_url("https://fresh.example.com/new") in state.seen_urls


def test_provider_and_recovery_call_shapes_match_real_signatures():
    # The loop mocks providers with permissive signatures; a real-signature drift
    # would silently no-op the feature. Bind the EXACT call shapes run_refinement
    # uses against the REAL functions — no network, fails loudly on drift.
    import inspect

    from polysearch.providers.base import (
        NullResearchProvider,
        NullSynthesizer,
        NullVerifier,
        NullWebGrounder,
    )
    from polysearch.verification.recovery import recover as real_recover

    inspect.signature(NullResearchProvider("x").research).bind(
        "q", sub_questions=1, depth="standard"
    )
    inspect.signature(NullWebGrounder("x").ground).bind("q", limit=5, scrape_top_k=3)
    inspect.signature(NullVerifier("x").verify).bind(
        [], budget_usd=1.0, max_concurrency=8
    )
    inspect.signature(NullSynthesizer("x").synthesize).bind(
        "t", [], style_constraints=None
    )
    inspect.signature(real_recover).bind("topic", None, [], settings=Settings())


# ── evaluate_coverage structured-output call ────────────────────────────────


async def test_evaluate_coverage_parses_valid_json():
    mock_client = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"goal_met": false, "coverage_score": 0.6, "gaps": ["pricing"], '
                    '"followup_queries": ["q1", "q2"], "needs_deeper_verification": true, '
                    '"reasoning": "missing pricing"}'
                )
            )
        )
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_msg)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        verdict = await evaluate_coverage(
            topic="t", synthesis_md="summary",
            layers=[_layer([_src("https://cms.example.gov/a")])],
            verification=None, prior_queries=["t"], settings=_settings(),
        )

    assert verdict.parse_error is None
    assert verdict.goal_met is False
    assert verdict.coverage_score == 0.6
    assert verdict.needs_deeper_verification is True
    assert len(verdict.followup_queries) == 2


async def test_evaluate_coverage_bad_json_sets_parse_error():
    mock_client = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock(message=MagicMock(content="not json at all"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_msg)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        verdict = await evaluate_coverage(
            topic="t", synthesis_md="summary", layers=[],
            verification=None, prior_queries=["t"], settings=_settings(),
        )

    assert verdict.parse_error is not None
    assert verdict.goal_met is False


async def test_evaluate_coverage_empty_choices_sets_parse_error():
    mock_client = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.choices = []  # no choices -> must not raise
    mock_client.chat.completions.create = AsyncMock(return_value=mock_msg)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        verdict = await evaluate_coverage(
            topic="t", synthesis_md="s", layers=[],
            verification=None, prior_queries=[], settings=_settings(),
        )

    assert verdict.parse_error is not None
    assert verdict.goal_met is False


async def test_evaluate_coverage_clamps_followups_to_three():
    mock_client = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"goal_met": false, "coverage_score": 0.5, "gaps": [], '
                    '"followup_queries": ["a", "b", "c", "d", "e"], '
                    '"needs_deeper_verification": false, "reasoning": "r"}'
                )
            )
        )
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_msg)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        verdict = await evaluate_coverage(
            topic="t", synthesis_md="s", layers=[],
            verification=None, prior_queries=[], settings=_settings(),
        )

    assert len(verdict.followup_queries) <= 3


async def test_evaluate_coverage_no_openai_key_returns_parse_error():
    verdict = await evaluate_coverage(
        topic="t", synthesis_md="s", layers=[],
        verification=None, prior_queries=[],
        settings=Settings(anthropic_api_key="x"),  # no openai key
    )
    assert verdict.parse_error is not None
    assert "OpenAI" in verdict.parse_error
    assert verdict.goal_met is False
