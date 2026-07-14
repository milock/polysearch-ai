"""Unit tests for polysearch.output.schema — pydantic model round-trips."""

from __future__ import annotations

import pytest

from polysearch.output.schema import (
    Claim,
    CoverageVerdict,
    LayerOutput,
    PipelineReport,
    RefinementTrace,
    SourceResult,
    VerificationReport,
    VerificationResult,
)


def _roundtrip(model):
    """Serialize to a dict then rebuild — the parsed copy must equal the original."""
    return type(model).model_validate(model.model_dump())


def test_claim_roundtrip() -> None:
    c = Claim(
        claim_id="abc123",
        text="Half of practices report X.",
        numbers=["50%"],
        quotes=["report X"],
        source_urls=["https://example.com/a"],
    )
    assert _roundtrip(c) == c


def test_claim_list_defaults() -> None:
    c = Claim(claim_id="id1", text="bare claim")
    assert c.numbers == []
    assert c.quotes == []
    assert c.source_urls == []


def test_verification_result_status_literal() -> None:
    r = VerificationResult(claim_id="id1", url="https://example.com", status="OK")
    assert r.status == "OK"
    assert r.detail is None
    assert _roundtrip(r) == r


@pytest.mark.parametrize(
    "status",
    [
        "OK",
        "URL_DEAD",
        "QUOTE_NOT_FOUND",
        "NUMBER_MISMATCH",
        "PAYWALLED",
        "UNDATED",
        "SKIPPED_BUDGET",
        "FETCH_BLOCKED",
        "BLOCKED_SOURCE",
    ],
)
def test_verification_result_accepts_all_statuses(status: str) -> None:
    r = VerificationResult(claim_id="id1", url="https://example.com", status=status)
    assert r.status == status


def test_verification_result_rejects_bad_status() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VerificationResult(claim_id="id1", url="https://x.com", status="NONSENSE")


def test_verification_report_roundtrip_and_phase_a_fields() -> None:
    report = VerificationReport(
        total_citations=3,
        verified_ok=2,
        broken=1,
        quote_mismatches=0,
        number_mismatches=0,
        paywalled=0,
        undated=0,
        skipped_budget=0,
        results=[
            VerificationResult(claim_id="id1", url="https://example.com", status="OK"),
        ],
        total_cost_usd=0.01,
        total_duration_ms=1234,
    )
    # Phase A additions default correctly.
    assert report.fetch_blocked == 0
    assert report.credits_exhausted_hit is False
    assert _roundtrip(report) == report


def test_source_result_roundtrip() -> None:
    s = SourceResult(
        url="https://example.com",
        title="A title",
        snippet="a snippet",
        tier="HIGH",
        published_date="2026-07-01",
        layer="web",
    )
    assert s.engagement is None
    assert _roundtrip(s) == s


def test_layer_output_roundtrip() -> None:
    lo = LayerOutput(
        layer="web",
        results=[
            SourceResult(url="https://a.com", title="t", snippet="s", tier="HIGH", layer="web"),
        ],
        cost_usd=0.5,
        duration_ms=900,
    )
    assert lo.error is None
    assert _roundtrip(lo) == lo


def test_coverage_verdict_roundtrip() -> None:
    v = CoverageVerdict(
        goal_met=False,
        coverage_score=0.4,
        gaps=["missing pricing"],
        followup_queries=["what is the price"],
        needs_deeper_verification=True,
        reasoning="thin coverage",
    )
    assert _roundtrip(v) == v


def test_refinement_trace_roundtrip() -> None:
    verdict = CoverageVerdict(goal_met=True, coverage_score=0.9)
    t = RefinementTrace(
        iteration=1,
        verdict=verdict,
        queries_run=["q1"],
        new_sources=3,
        new_claims=2,
        cost_usd=0.25,
    )
    assert _roundtrip(t) == t


def test_pipeline_report_roundtrip() -> None:
    report = PipelineReport(
        topic="test topic",
        depth="standard",
        synthesis_md="# Summary",
    )
    assert report.layers == []
    assert report.refinement_iterations == []
    assert report.pipeline_errors == []
    assert report.recovery_ran is False
    assert report.verification is None
    assert _roundtrip(report) == report


def test_pipeline_report_nested_roundtrip() -> None:
    report = PipelineReport(
        topic="nested",
        depth="deep",
        layers=[
            LayerOutput(
                layer="web",
                results=[SourceResult(url="https://a.com", title="t", snippet="s", tier="HIGH", layer="web")],
                cost_usd=0.1,
                duration_ms=10,
            )
        ],
        synthesis_md="body",
        verification=VerificationReport(
            total_citations=1,
            verified_ok=1,
            broken=0,
            quote_mismatches=0,
            number_mismatches=0,
            paywalled=0,
            undated=0,
            skipped_budget=0,
            results=[],
            total_cost_usd=0.0,
            total_duration_ms=0,
        ),
        refinement_iterations=[
            RefinementTrace(iteration=1, verdict=CoverageVerdict(goal_met=True)),
        ],
    )
    assert _roundtrip(report) == report
