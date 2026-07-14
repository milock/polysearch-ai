"""Core pydantic models shared across the pipeline.

Field names mirror the internal research pipeline so ported verification,
synthesis, and refinement logic transplants without renaming. Every model is a
plain ``pydantic.BaseModel`` and round-trips through ``model_dump`` /
``model_validate`` losslessly — the pipeline serializes reports to JSON on disk.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# Verification outcome for a single (claim, cited-URL) pair. FETCH_BLOCKED and
# BLOCKED_SOURCE were added alongside the resilient-fetch work: the former marks
# a URL that could not be retrieved by any fetch method, the latter a source
# host that is categorically un-fetchable (login-walled, robots-blocked).
VerificationStatus = Literal[
    "OK",
    "URL_DEAD",
    "QUOTE_NOT_FOUND",
    "NUMBER_MISMATCH",
    "PAYWALLED",
    "UNDATED",
    "SKIPPED_BUDGET",
    "FETCH_BLOCKED",
    "BLOCKED_SOURCE",
]


class Claim(BaseModel):
    """An atomic factual assertion extracted from synthesis, with its citations."""

    claim_id: str
    text: str
    numbers: list[str] = []
    quotes: list[str] = []
    source_urls: list[str] = []


class VerificationResult(BaseModel):
    """The verdict for one claim checked against one cited URL."""

    claim_id: str
    url: str
    status: VerificationStatus
    detail: str | None = None


class VerificationReport(BaseModel):
    """Aggregate citation-integrity rollup for a whole pipeline run."""

    total_citations: int
    verified_ok: int
    broken: int
    quote_mismatches: int
    number_mismatches: int
    paywalled: int
    undated: int
    skipped_budget: int
    fetch_blocked: int = 0
    blocked_sources: int = 0
    results: list[VerificationResult]
    total_cost_usd: float
    total_duration_ms: int
    # Claim-level rollup (a claim is supported when each of its numbers/quotes
    # matches in at least one cited source). recovery_pass.merge_reports sums
    # these across passes.
    claims_total: int = 0
    claims_supported: int = 0
    # True when Firecrawl reported 402/credits-exhausted at any point in the run.
    credits_exhausted_hit: bool = False


class SourceResult(BaseModel):
    """A single source surfaced by one search/discovery layer."""

    url: str
    title: str
    snippet: str
    tier: str
    published_date: str | None = None
    layer: str
    engagement: int | None = None


class LayerOutput(BaseModel):
    """Everything one layer produced, plus its cost/latency and any error."""

    layer: str
    results: list[SourceResult] = []
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


class CoverageVerdict(BaseModel):
    """One evaluator judgement of the current corpus against the research goal."""

    goal_met: bool
    coverage_score: float = 0.0
    gaps: list[str] = []
    followup_queries: list[str] = []
    needs_deeper_verification: bool = False
    reasoning: str = ""
    # Set (non-None) when the evaluator response could not be parsed — the
    # refinement loop reads this as a graceful-abort signal.
    parse_error: str | None = None


class RefinementTrace(BaseModel):
    """Per-iteration audit record of the goal-driven refinement loop."""

    iteration: int
    verdict: CoverageVerdict
    queries_run: list[str] = []
    new_sources: int = 0
    new_claims: int = 0
    cost_usd: float = 0.0
    # Why the loop stopped this iteration: goal_met|cost_ceiling|dry|parse_abort|
    # no_new_queries. Rendered into the report's stop note.
    stopped_reason: str | None = None


class PipelineReport(BaseModel):
    """The full result of a pipeline run — the top-level serialized artifact."""

    topic: str
    depth: str
    classification: dict[str, Any] = {}
    layers: list[LayerOutput] = []
    synthesis_md: str = ""
    verification: VerificationReport | None = None
    recovery_ran: bool = False
    refinement_iterations: list[RefinementTrace] = []
    pipeline_errors: list[str] = []
    totals: dict[str, Any] = {}
