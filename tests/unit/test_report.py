"""Unit tests for polysearch.output.report — markdown + JSON report writer.

Covers the markdown + JSON report writer over the ``PipelineReport`` schema:
placeholder guard, dead-URL exclusion, depth reconciliation, blocked-source
exclusion separate from dead links, fetch-blocked notation, the refinement-trace
section, and a full-render snapshot over a neutral topic.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

from tests._leakage_terms import BANNED_TOKENS

from polysearch.config import Settings
from polysearch.output.report import (
    render_markdown,
    write_report,
    _render_pipeline_decisions,
    _render_refinement_trace,
    _render_sources_by_tier,
)
from polysearch.output.schema import (
    CoverageVerdict,
    LayerOutput,
    PipelineReport,
    RefinementTrace,
    SourceResult,
    VerificationReport,
    VerificationResult,
)

# ── Neutral fixtures (generic topics — no internal/domain vocabulary) ────────

DEAD_URL = "https://example.com/hallucinated-source"
LIVE_URL = "https://research.example.org/rooftop-yields"
BLOCKED_URL = "https://members.example.net/gated-study"
FETCH_BLOCKED_URL = "https://slow.example.io/timed-out"


def _source(url: str, tier: str, layer: str = "research", title: str = "") -> SourceResult:
    return SourceResult(
        url=url,
        title=title or f"Source at {url}",
        snippet="A neutral snippet about rooftop beekeeping yields.",
        tier=tier,
        layer=layer,
        published_date="2026-05-01",
    )


def _verification(results: list[VerificationResult], **counts: int) -> VerificationReport:
    base = dict(
        total_citations=len(results),
        verified_ok=0,
        broken=0,
        quote_mismatches=0,
        number_mismatches=0,
        paywalled=0,
        undated=0,
        skipped_budget=0,
        results=results,
        total_cost_usd=0.0,
        total_duration_ms=0,
    )
    base.update(counts)
    return VerificationReport(**base)


def _sources_report() -> PipelineReport:
    """One layer with live/dead/blocked/fetch-blocked sources + verification."""
    layer = LayerOutput(
        layer="research",
        results=[
            _source(LIVE_URL, "HIGH", title="Peer-reviewed yield study"),
            _source(DEAD_URL, "UNKNOWN", title="Hallucinated Source"),
            _source(BLOCKED_URL, "MEDIUM", title="Gated Trade Study"),
            _source(FETCH_BLOCKED_URL, "LOW", title="Timed-out Blog"),
        ],
        cost_usd=0.10,
        duration_ms=1200,
    )
    verification = _verification(
        [
            VerificationResult(claim_id="c1", url=LIVE_URL, status="OK"),
            VerificationResult(claim_id="c2", url=DEAD_URL, status="URL_DEAD"),
            VerificationResult(claim_id="c3", url=BLOCKED_URL, status="BLOCKED_SOURCE"),
            VerificationResult(claim_id="c4", url=FETCH_BLOCKED_URL, status="FETCH_BLOCKED"),
        ],
        verified_ok=1,
        broken=1,
        fetch_blocked=1,
        blocked_sources=1,
    )
    return PipelineReport(
        topic="urban rooftop beekeeping yields",
        depth="standard",
        classification={"suggested_depth": "standard"},
        layers=[layer],
        synthesis_md="A neutral synthesis paragraph.",
        verification=verification,
    )


def _classification(suggested_depth: str = "deep") -> dict:
    return {
        "topic": "municipal composting adoption 2026",
        "query_type": "THEMATIC",
        "time_sensitive": True,
        "domain_related": False,
        "competitive_mode": False,
        "suggested_depth": suggested_depth,
        "flags": [],
        "reasons": ["recency keyword"],
    }


def _trace(**kw) -> RefinementTrace:
    defaults = dict(
        iteration=1,
        verdict=CoverageVerdict(
            goal_met=False, coverage_score=0.6, reasoning="Two gaps remain on pricing."
        ),
        queries_run=["composting cost per ton 2026", "municipal composting subsidies"],
        new_sources=3,
        new_claims=2,
        cost_usd=0.0421,
        stopped_reason=None,
    )
    defaults.update(kw)
    return RefinementTrace(**defaults)


def _full_report() -> PipelineReport:
    return PipelineReport(
        topic="urban rooftop beekeeping yields",
        depth="deep",
        classification=_classification("deep"),
        layers=[
            LayerOutput(
                layer="research",
                results=[_source(LIVE_URL, "HIGH")],
                cost_usd=0.30,
                duration_ms=2000,
            ),
            LayerOutput(layer="grounding", results=[], cost_usd=0.05, duration_ms=800),
        ],
        synthesis_md="## Findings\n\nRooftop hives yield more in dense districts.",
        verification=_verification(
            [VerificationResult(claim_id="c1", url=LIVE_URL, status="OK")],
            verified_ok=1,
            claims_total=1,
            claims_supported=1,
        ),
        recovery_ran=True,
        refinement_iterations=[_trace(), _trace(iteration=2, stopped_reason="dry", queries_run=[])],
        pipeline_errors=["community: reddit adapter timed out"],
        totals={"cost_usd": 1.23, "duration_sec": 42.0},
    )


# ── Placeholder guard (ported from test_placeholder_guard.py) ────────────────

def test_write_blocks_unresolved_placeholders(tmp_path) -> None:
    report = PipelineReport(topic="t", depth="quick", synthesis_md="")  # empty -> placeholder
    with pytest.raises(RuntimeError, match="unresolved placeholders"):
        write_report(report, output_dir=tmp_path)


def test_settings_opt_in_allows_placeholders(tmp_path) -> None:
    report = PipelineReport(topic="t", depth="quick", synthesis_md="")
    md_path, _ = write_report(
        report, output_dir=tmp_path, settings=Settings(allow_placeholders=True)
    )
    assert md_path.exists()
    assert "{{SYNTHESIS" in md_path.read_text()


# ── Sources by tier: dead-URL exclusion (ported) ─────────────────────────────

def test_dead_url_omitted_from_main_tier_buckets() -> None:
    out = _render_sources_by_tier(_sources_report())
    excluded_idx = out.index("### Excluded (dead links)")
    assert DEAD_URL not in out[:excluded_idx]


def test_dead_url_appears_in_excluded_dead_links_subsection() -> None:
    out = _render_sources_by_tier(_sources_report())
    assert "### Excluded (dead links)" in out
    assert "(1 URL_DEAD)" in out
    excluded_idx = out.index("### Excluded (dead links)")
    assert DEAD_URL in out[excluded_idx:]


def test_live_url_still_appears_in_main_tier_bucket() -> None:
    out = _render_sources_by_tier(_sources_report())
    assert "High (primary, peer-reviewed, official)" in out
    assert LIVE_URL in out


def test_no_excluded_dead_links_when_none() -> None:
    report = _sources_report()
    assert report.verification is not None
    report.verification.results = [
        r for r in report.verification.results if r.status != "URL_DEAD"
    ]
    out = _render_sources_by_tier(report)
    assert "### Excluded (dead links)" not in out


# ── Sources by tier: blocked-source + fetch-blocked (public-only) ────────────

def test_blocked_source_in_separate_excluded_section() -> None:
    out = _render_sources_by_tier(_sources_report())
    assert "### Excluded (blocked sources)" in out
    dead_idx = out.index("### Excluded (dead links)")
    blocked_idx = out.index("### Excluded (blocked sources)")
    # Blocked URL surfaces only under its own section, never the main buckets.
    assert BLOCKED_URL not in out[: min(dead_idx, blocked_idx)]
    assert BLOCKED_URL in out[blocked_idx:]
    # And it is NOT counted among the dead links.
    assert BLOCKED_URL not in out[dead_idx:blocked_idx] if dead_idx < blocked_idx else True


def test_fetch_blocked_url_kept_in_buckets_and_noted() -> None:
    out = _render_sources_by_tier(_sources_report())
    # Kept as a live source in its tier bucket (not excluded).
    low_idx = out.index("Low (opinion")
    excluded_idx = out.index("### Excluded (dead links)")
    assert FETCH_BLOCKED_URL in out[low_idx:excluded_idx]
    # Noted distinctly as blocked-but-alive.
    assert "fetch-blocked" in out.lower()


# ── Pipeline decisions: depth reconciliation (ported to dict shape) ──────────

def test_pipeline_decisions_shows_depth_ran() -> None:
    out = _render_pipeline_decisions(_classification("deep"), depth_ran="quick")
    assert "**Depth ran:** quick" in out


def test_pipeline_decisions_shows_depth_suggested() -> None:
    out = _render_pipeline_decisions(_classification("deep"), depth_ran="quick")
    assert "**Depth suggested by classifier:** deep" in out


def test_pipeline_decisions_divergence_note_when_different() -> None:
    out = _render_pipeline_decisions(_classification("deep"), depth_ran="quick")
    assert "_Note: ran at quick, classifier suggested deep._" in out


def test_pipeline_decisions_no_divergence_when_match() -> None:
    out = _render_pipeline_decisions(_classification("standard"), depth_ran="standard")
    assert "Note: ran at" not in out
    assert "**Depth ran:** standard" in out
    assert "**Depth suggested by classifier:** standard" in out


def test_pipeline_decisions_no_legacy_suggested_depth_label() -> None:
    out = _render_pipeline_decisions(_classification("deep"), depth_ran="quick")
    assert "**Suggested depth:**" not in out


# ── Refinement trace (public-only) ───────────────────────────────────────────

def test_refinement_trace_renders_iteration_details() -> None:
    out = _render_refinement_trace([_trace()])
    assert "## Refinement Trace" in out
    assert "Iteration 1" in out
    assert "coverage 0.60" in out
    assert "Two gaps remain on pricing." in out
    assert "composting cost per ton 2026" in out
    assert "**New sources:** 3" in out
    assert "**New claims:** 2" in out
    assert "$0.0421" in out


def test_refinement_trace_omitted_when_empty() -> None:
    out = render_markdown(
        PipelineReport(topic="t", depth="quick", synthesis_md="x", refinement_iterations=[])
    )
    assert "## Refinement Trace" not in out


def test_refinement_trace_stopped_reason_note() -> None:
    out = _render_refinement_trace([_trace(stopped_reason="dry", queries_run=[])])
    assert "converged" in out.lower()


def test_refinement_trace_goal_met_marker() -> None:
    verdict = CoverageVerdict(goal_met=True, coverage_score=0.95, reasoning="Goal satisfied.")
    out = _render_refinement_trace([_trace(verdict=verdict, stopped_reason="goal_met")])
    assert "goal met" in out.lower()


# ── Pipeline errors + stats ──────────────────────────────────────────────────

def test_pipeline_errors_rendered() -> None:
    out = render_markdown(_full_report())
    assert "## Pipeline Errors" in out
    assert "community: reddit adapter timed out" in out


def test_stats_table_present() -> None:
    out = render_markdown(_full_report())
    assert "## Pipeline Stats" in out
    assert "$1.23" in out  # total cost from totals


# ── Style audit (only when constraints set) ──────────────────────────────────

def test_style_audit_present_when_constraints_set() -> None:
    out = render_markdown(_full_report(), settings=Settings(style_constraints="No hype words."))
    assert "## Style Audit" in out
    assert "No hype words." in out


def test_style_audit_absent_when_no_constraints() -> None:
    out = render_markdown(_full_report(), settings=Settings(style_constraints=None))
    assert "## Style Audit" not in out


# ── write_report: files, naming, JSON round-trip ─────────────────────────────

def test_write_report_paths_and_naming(tmp_path) -> None:
    report = _full_report()
    md_path, json_path = write_report(
        report, output_dir=tmp_path, generated_at=datetime(2026, 7, 14, 9, 0, 0)
    )
    assert md_path.exists() and json_path.exists()
    assert md_path.name == "2026-07-14-urban-rooftop-beekeeping-yields.md"
    assert json_path.name == "2026-07-14-urban-rooftop-beekeeping-yields.json"
    assert md_path.parent == tmp_path


def test_write_report_json_roundtrips(tmp_path) -> None:
    report = _full_report()
    _, json_path = write_report(
        report, output_dir=tmp_path, generated_at=datetime(2026, 7, 14)
    )
    loaded = json.loads(json_path.read_text())
    assert loaded == report.model_dump(mode="json")


def test_generated_at_override_sets_date_prefix(tmp_path) -> None:
    report = _full_report()
    md_path, _ = write_report(
        report, output_dir=tmp_path, generated_at=datetime(2025, 1, 2)
    )
    assert md_path.name.startswith("2025-01-02-")


# ── Full-render snapshot (generic) ───────────────────────────────────────────

def test_full_render_snapshot() -> None:
    out = render_markdown(_full_report(), generated_at=datetime(2026, 7, 14, 9, 0, 0))
    for heading in (
        "# Research: urban rooftop beekeeping yields",
        "## Pipeline Decisions",
        "## Sources by Quality Tier",
        "## Refinement Trace",
        "## Pipeline Errors",
        "## Pipeline Stats",
    ):
        assert heading in out
    # Header carries generated date + depth.
    assert "2026-07-14" in out
    assert "Depth: deep" in out


def test_no_internal_vocabulary_in_output() -> None:
    """The public writer must carry zero trace of private-project vocabulary."""
    out = render_markdown(_full_report(), settings=Settings(style_constraints="x"))
    lowered = out.lower()
    extra_terms = ("battlecard", "customer relationship snapshot", "warm intro")
    for banned in tuple(BANNED_TOKENS) + extra_terms:
        assert not re.search(rf"\b{re.escape(banned)}\b", lowered), banned
