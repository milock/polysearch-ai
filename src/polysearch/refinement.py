"""Goal-driven refinement loop.

After the first full pass (layers -> claims -> verification -> recovery) a
rubric-based evaluator judges coverage against the research goal. When the goal
is unmet it emits angle-diverse follow-up queries; the loop runs them through the
research + web-grounding providers, verifies the new material, optionally re-runs
recovery, re-synthesizes with the full corpus, and repeats — bounded by the depth
profile's iteration cap and cost ceiling.

Loop guards (all mandatory — a runaway loop is the worst failure here):
  1. goal_met exit
  2. max iterations (from the depth profile)
  3. cumulative cost ceiling (profile, overridden by ``settings`` when set)
  4. no-unique-follow-up-queries exit (the evaluator only rephrased prior queries)
  5. zero-new-unique-sources dry exit (URL-canonical dedupe vs everything seen)
  6. evaluator parse failure / no evaluator -> graceful abort with a
     ``pipeline_errors`` entry (never crashes the run)

Interfaces:
    evaluate_coverage(...) -> CoverageVerdict     (one structured-output LLM call)
    run_refinement(topic, state, providers, profile, *, settings) -> list[RefinementTrace]

The coverage evaluator is OpenAI-based (``settings.synthesis_model``, JSON-schema
response_format). When only an Anthropic key is configured there is no evaluator,
so refinement degrades gracefully to a no-op. NEVER request URLs inside the JSON
schema — models fabricate them; real URLs come from the layers.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from polysearch.config import DepthProfile, Settings
from polysearch.extractors.claims import extract_claims
from polysearch.output.schema import (
    Claim,
    CoverageVerdict,
    LayerOutput,
    RefinementTrace,
    SourceResult,
    VerificationReport,
)
from polysearch.providers.perplexity import PerplexityResult
from polysearch.sources.authority import classify_url
from polysearch.verification.recovery import merge_reports
from polysearch.verification.recovery import recover as _module_recover

_MAX_FOLLOWUPS = 3

# Follow-up sourcing is intentionally shallow: one sub-question per query and a
# small web-grounding fetch (Task 17: WebGrounder limit=5).
_FOLLOWUP_DEPTH = "standard"
_FOLLOWUP_WEB_LIMIT = 5
_FOLLOWUP_SCRAPE_TOP_K = 3


# ── contracts ─────────────────────────────────────────────────────────────────


@dataclass
class RefinementState:
    """The mutable research corpus the loop grows in place.

    The orchestrator (Task 18) builds this from the first pass and reads it back
    after ``run_refinement`` returns — every field is updated in place:

    - ``layers``: all search-layer outputs; the loop appends a ``refinement-<i>``
      (and any ``recovery-<i>``) layer per iteration.
    - ``claims``: verifiable claims extracted so far; extended with new material.
    - ``verification``: the running citation-integrity rollup (merged each pass).
    - ``synthesis_md``: the current synthesized report body; replaced each pass.
    - ``prior_queries``: every query already run — the evaluator must not repeat
      or rephrase these, and follow-ups are appended as they run.
    - ``seen_urls``: canonical URLs of every source in the corpus, for dedupe.
    - ``pipeline_errors``: run-level error strings; the loop appends one on a
      graceful abort so the orchestrator can surface it in the report.
    """

    layers: list[LayerOutput]
    claims: list[Claim]
    verification: VerificationReport | None
    synthesis_md: str
    prior_queries: list[str]
    seen_urls: set[str]
    pipeline_errors: list[str] = field(default_factory=list)


# ── URL canonicalization + query dedupe ─────────────────────────────────────


def _canonical_url(url: str) -> str:
    """Canonical form for dedupe: lowercase scheme+host (www-stripped), path
    without a trailing slash, query kept, fragment dropped."""
    try:
        parts = urlsplit(url.strip())
    except Exception:  # noqa: BLE001 — malformed URL falls back to a plain key
        return url.strip().lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if parts.port:
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/")
    return urlunsplit(("", host, path, parts.query, ""))


def _domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return ""
    return host[4:] if host.startswith("www.") else host


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _unique_queries(candidates: list[str], prior: list[str]) -> list[str]:
    """Keep follow-ups that are not a rephrasing of any already-run query, capped
    at three. Comparison is whitespace/case-normalized."""
    prior_norm = {_norm_query(p) for p in prior}
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        n = _norm_query(c)
        if not n or n in prior_norm or n in seen:
            continue
        seen.add(n)
        out.append(c)
        if len(out) >= _MAX_FOLLOWUPS:
            break
    return out


def _material_text(sources: list[SourceResult]) -> str:
    """Join source titles + snippets into a body the claim extractor can mine for
    verifiable figures and quotes."""
    parts: list[str] = []
    for s in sources:
        text = f"{s.title}. {s.snippet}".strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _retier(s: SourceResult) -> SourceResult:
    """Re-classify a source's tier from its URL (providers emit UNKNOWN)."""
    return s.model_copy(
        update={"tier": classify_url(s.url, published_date=s.published_date)}
    )


def _recovered_material(
    recovered: list[PerplexityResult],
) -> tuple[list[SourceResult], list[Claim]]:
    """Turn recovered Perplexity answers into (source rows, verifiable claims)."""
    sources: list[SourceResult] = []
    claims: list[Claim] = []
    for r in recovered:
        urls = [c.url for c in r.citations if c.url]
        claims.extend(extract_claims(r.answer, urls))
        for c in r.citations:
            if not c.url:
                continue
            sources.append(
                SourceResult(
                    url=c.url,
                    title=c.title or "",
                    snippet=(r.answer or "")[:600],
                    tier=classify_url(c.url, published_date=c.published_date),
                    published_date=c.published_date,
                    layer="recovery",
                )
            )
    return sources, claims


# ── coverage evaluator (one structured-output LLM call) ─────────────────────


_COVERAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal_met": {"type": "boolean"},
        "coverage_score": {"type": "number"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "followup_queries": {"type": "array", "items": {"type": "string"}},
        "needs_deeper_verification": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "goal_met",
        "coverage_score",
        "gaps",
        "followup_queries",
        "needs_deeper_verification",
        "reasoning",
    ],
    "additionalProperties": False,
}


def _build_coverage_prompt(
    *,
    topic: str,
    synthesis_md: str,
    sources: list[dict[str, Any]],
    verification: VerificationReport | None,
    prior_queries: list[str],
    strictness: str,
) -> str:
    tier_hist: dict[str, int] = {}
    for s in sources:
        t = str(s.get("tier") or "UNKNOWN")
        tier_hist[t] = tier_hist.get(t, 0) + 1
    dated = sum(1 for s in sources if s.get("date"))
    domains = sorted({str(s.get("domain") or "") for s in sources if s.get("domain")})

    if verification is not None and verification.total_citations:
        verify_line = (
            f"{verification.verified_ok}/{verification.total_citations} citation-source "
            f"pairs verified; {verification.claims_supported}/{verification.claims_total} "
            f"claims supported by >=1 source; broken={verification.broken}, "
            f"quote_mismatch={verification.quote_mismatches}, "
            f"number_mismatch={verification.number_mismatches}"
        )
    else:
        verify_line = "no citation verification data available"

    strict_line = (
        "Grade STRICTLY. Mark goal_met only when every sub-question is answered, "
        "sources are diverse and mostly HIGH/MEDIUM tier, and load-bearing numbers "
        "are verified."
        if strictness == "deep"
        else "Grade with a reasonable bar. Mark goal_met when the main sub-questions "
        "are answered and the key numbers are supported, even if minor gaps remain."
    )

    return "\n".join(
        [
            f"You are the coverage judge for a research pipeline. Research goal: {topic}",
            "",
            "Judge whether the current corpus fully answers the goal. Score on four "
            "rubric dimensions and weigh them together:",
            "1. Factual coverage of the goal's sub-questions.",
            "2. Source diversity and tier quality (HIGH/MEDIUM vs LOW/UNKNOWN).",
            "3. Citation verification rate (are load-bearing numbers actually sourced?).",
            "4. Recency, where the topic demands current data.",
            "",
            strict_line,
            "",
            "If the goal is NOT met, propose up to THREE follow-up queries. Rules:",
            "- Each MUST be genuinely new — never a rephrasing of an already-run query.",
            "- Each MUST be self-contained (a standalone search string).",
            "- Make them angle-diverse: statistics vs expert opinion vs case study vs "
            "recent news — not three variants of one angle.",
            "- Target the specific gaps you identified, not the topic in general.",
            "Do NOT include any URLs in your response.",
            "",
            "--- CURRENT SYNTHESIS ---",
            synthesis_md.strip() or "(none)",
            "",
            "--- SOURCE INVENTORY ---",
            f"Tier histogram: {tier_hist or '{}'}",
            f"Dated sources: {dated}/{len(sources)}",
            f"Domains: {', '.join(domains[:30]) or '(none)'}",
            "",
            "--- CITATION INTEGRITY ---",
            verify_line,
            "",
            "--- QUERIES ALREADY RUN (do not repeat or rephrase these) ---",
            "; ".join(prior_queries) or "(none)",
        ]
    )


async def evaluate_coverage(
    *,
    topic: str,
    synthesis_md: str,
    layers: list[LayerOutput],
    verification: VerificationReport | None,
    prior_queries: list[str],
    settings: Settings,
    strictness: str = "standard",
) -> CoverageVerdict:
    """One cheap structured-output LLM call judging coverage vs the goal.

    The evaluator is OpenAI-based (``settings.synthesis_model``). When no OpenAI
    key is configured — including an Anthropic-only install — there is no
    evaluator, so this returns a verdict with ``parse_error`` set and
    ``goal_met=False``; the loop reads that as a graceful abort (no refinement).
    Any client / parse failure returns the same shape so the loop never crashes.
    """
    api_key = settings.openai_api_key
    if not api_key:
        return CoverageVerdict(
            goal_met=False,
            parse_error="OPENAI_API_KEY not set; coverage evaluator requires OpenAI",
        )

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return CoverageVerdict(
            goal_met=False, parse_error="openai package not installed"
        )

    sources = [
        {"tier": s.tier, "domain": _domain(s.url), "date": s.published_date}
        for layer in layers
        for s in layer.results
    ]

    client = AsyncOpenAI(api_key=api_key)
    prompt = _build_coverage_prompt(
        topic=topic,
        synthesis_md=synthesis_md,
        sources=sources,
        verification=verification,
        prior_queries=prior_queries,
        strictness=strictness,
    )

    try:
        msg = await client.chat.completions.create(
            model=settings.synthesis_model,
            max_completion_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "coverage_verdict",
                    "strict": True,
                    "schema": _COVERAGE_SCHEMA,
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 — any client error aborts gracefully
        return CoverageVerdict(
            goal_met=False, parse_error=f"evaluator call failed: {exc}"
        )

    choices = getattr(msg, "choices", None) or []
    if not choices:
        return CoverageVerdict(
            goal_met=False, parse_error="evaluator returned no choices"
        )

    text = (choices[0].message.content or "").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return CoverageVerdict(
            goal_met=False, parse_error=f"evaluator JSON parse failed: {exc}"
        )

    if not isinstance(data, dict) or "goal_met" not in data:
        return CoverageVerdict(
            goal_met=False, parse_error="evaluator JSON missing required keys"
        )

    try:
        score = float(data.get("coverage_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    followups = [
        str(q).strip() for q in (data.get("followup_queries") or []) if str(q).strip()
    ][:_MAX_FOLLOWUPS]

    return CoverageVerdict(
        goal_met=bool(data.get("goal_met")),
        coverage_score=score,
        gaps=[str(g) for g in (data.get("gaps") or [])],
        followup_queries=followups,
        needs_deeper_verification=bool(data.get("needs_deeper_verification")),
        reasoning=str(data.get("reasoning") or ""),
        parse_error=None,
    )


# ── the loop ────────────────────────────────────────────────────────────────


# Injectable seams so the loop is testable without network. ``EvaluateFn`` matches
# ``evaluate_coverage``; ``RecoverFn`` matches ``recovery.recover``.
EvaluateFn = Callable[..., Awaitable[CoverageVerdict]]
RecoverFn = Callable[..., Awaitable[list[PerplexityResult]]]


def _strictness_for(profile: DepthProfile) -> str:
    """Deep runs grade strictly; standard runs grade leniently (Task 17 rubric)."""
    return "deep" if profile.max_refinement_iterations >= 4 else "standard"


async def _run_followups(
    queries: list[str], providers: Any
) -> tuple[list[SourceResult], float]:
    """Run each follow-up through the research provider (one sub-question) and the
    web grounder (limit=5) in parallel, failure-isolated. Returns the pooled
    source rows and the summed layer cost."""

    async def _one(query: str) -> tuple[list[SourceResult], float]:
        research_task = providers.research.research(
            query, sub_questions=1, depth=_FOLLOWUP_DEPTH
        )
        ground_task = providers.grounder.ground(
            query, limit=_FOLLOWUP_WEB_LIMIT, scrape_top_k=_FOLLOWUP_SCRAPE_TOP_K
        )
        outputs = await asyncio.gather(
            research_task, ground_task, return_exceptions=True
        )
        rows: list[SourceResult] = []
        cost = 0.0
        for out in outputs:
            if isinstance(out, LayerOutput):
                rows.extend(out.results)
                cost += out.cost_usd
        return rows, cost

    batches = await asyncio.gather(*(_one(q) for q in queries), return_exceptions=True)

    all_rows: list[SourceResult] = []
    layer_cost = 0.0
    for b in batches:
        if isinstance(b, BaseException):
            continue
        rows, cost = b
        all_rows.extend(rows)
        layer_cost += cost
    return all_rows, layer_cost


async def run_refinement(
    topic: str,
    state: RefinementState,
    providers: Any,
    profile: DepthProfile,
    *,
    settings: Settings,
    evaluate: EvaluateFn = evaluate_coverage,
    recover: RecoverFn = _module_recover,
) -> list[RefinementTrace]:
    """Iterate coverage evaluation -> follow-up sourcing -> verify -> re-synthesize
    until the goal is met or a guard trips. Mutates ``state`` in place and returns
    a per-iteration trace (empty when the profile disables refinement).

    ``providers`` is the run's :class:`~polysearch.providers.base.Providers`
    bundle; the loop uses its ``research``, ``grounder``, ``synthesizer``, and
    ``verifier`` fields. ``evaluate`` / ``recover`` are injectable so the loop is
    testable without network; they default to the real functions.
    """
    traces: list[RefinementTrace] = []
    max_iterations = profile.max_refinement_iterations
    if max_iterations <= 0:
        return traces

    ceiling = (
        settings.refinement_cost_ceiling_usd
        if settings.refinement_cost_ceiling_usd is not None
        else profile.refinement_cost_ceiling_usd
    )
    strictness = _strictness_for(profile)
    cumulative_cost = 0.0

    for i in range(1, max_iterations + 1):
        verdict = await evaluate(
            topic=topic,
            synthesis_md=state.synthesis_md,
            layers=state.layers,
            verification=state.verification,
            prior_queries=state.prior_queries,
            settings=settings,
            strictness=strictness,
        )

        # Guard 6: no usable judgement (parse failure or no evaluator) — abort clean.
        if verdict.parse_error is not None:
            state.pipeline_errors.append(
                f"refinement: coverage evaluator failed on iteration {i} "
                f"({verdict.parse_error}); refinement aborted"
            )
            traces.append(
                RefinementTrace(iteration=i, verdict=verdict, stopped_reason="parse_abort")
            )
            break

        # Guard 1: goal met — done, no queries run.
        if verdict.goal_met:
            traces.append(
                RefinementTrace(iteration=i, verdict=verdict, stopped_reason="goal_met")
            )
            break

        # Guard 3: cumulative cost ceiling reached before spending more.
        if cumulative_cost >= ceiling:
            traces.append(
                RefinementTrace(
                    iteration=i, verdict=verdict, stopped_reason="cost_ceiling"
                )
            )
            break

        # Guard 4: the evaluator only rephrased already-run queries.
        queries = _unique_queries(verdict.followup_queries, state.prior_queries)
        if not queries:
            traces.append(
                RefinementTrace(
                    iteration=i, verdict=verdict, stopped_reason="no_new_queries"
                )
            )
            break

        # Run the follow-ups through the layers, failure-isolated.
        pooled, layer_cost = await _run_followups(queries, providers)
        state.prior_queries.extend(queries)

        # Guard 5: convergence — dedupe new sources against everything seen.
        fresh: list[SourceResult] = []
        new_canon: set[str] = set()
        for s in pooled:
            if not s.url:
                continue
            cu = _canonical_url(s.url)
            if cu in state.seen_urls or cu in new_canon:
                continue
            new_canon.add(cu)
            fresh.append(s)

        if not new_canon:
            cumulative_cost += layer_cost
            traces.append(
                RefinementTrace(
                    iteration=i,
                    verdict=verdict,
                    queries_run=queries,
                    new_sources=0,
                    cost_usd=layer_cost,
                    stopped_reason="dry",
                )
            )
            break

        state.seen_urls |= new_canon
        fresh = [_retier(s) for s in fresh]

        # Extract + verify ONLY the new claims, budgeted by the remaining ceiling.
        new_claims = extract_claims(_material_text(fresh), [s.url for s in fresh])
        verify_cost = 0.0
        if new_claims:
            remaining = max(0.0, ceiling - cumulative_cost - layer_cost)
            if remaining > 0:
                new_ver = await providers.verifier.verify(
                    new_claims,
                    budget_usd=remaining,
                    max_concurrency=profile.verify_concurrency,
                )
                verify_cost += new_ver.total_cost_usd
                state.verification = (
                    merge_reports(state.verification, new_ver)
                    if state.verification is not None
                    else new_ver
                )

        # Merge the fresh material into the corpus.
        state.layers.append(
            LayerOutput(layer=f"refinement-{i}", results=fresh, cost_usd=layer_cost)
        )
        state.claims.extend(new_claims)

        # needs_deeper_verification -> re-run recovery on the merged report, then
        # RE-VERIFY the recovered material so recovered citations never reach
        # synthesis unverified. This runs on the final iteration too (within the
        # remaining ceiling), mirroring the first-pass orchestrator recovery.
        if verdict.needs_deeper_verification and state.verification is not None:
            recovered = await recover(
                topic, state.verification, list(state.claims), settings=settings
            )
            if recovered:
                rec_sources, rec_claims = _recovered_material(recovered)
                if rec_claims:
                    rec_remaining = max(
                        0.0, ceiling - cumulative_cost - layer_cost - verify_cost
                    )
                    if rec_remaining > 0:
                        rec_ver = await providers.verifier.verify(
                            rec_claims,
                            budget_usd=rec_remaining,
                            max_concurrency=profile.verify_concurrency,
                        )
                        verify_cost += rec_ver.total_cost_usd
                        state.verification = merge_reports(state.verification, rec_ver)
                if rec_sources:
                    state.layers.append(
                        LayerOutput(
                            layer=f"recovery-{i}", results=rec_sources, cost_usd=0.0
                        )
                    )
                    state.claims.extend(rec_claims)
                    for s in rec_sources:
                        if s.url:
                            state.seen_urls.add(_canonical_url(s.url))

        # Re-synthesize with the full corpus.
        synthesis_md, synth_cost = await providers.synthesizer.synthesize(
            topic, state.layers, style_constraints=settings.style_constraints
        )
        state.synthesis_md = synthesis_md

        iter_cost = layer_cost + verify_cost + synth_cost
        cumulative_cost += iter_cost
        traces.append(
            RefinementTrace(
                iteration=i,
                verdict=verdict,
                queries_run=queries,
                new_sources=len(new_canon),
                new_claims=len(new_claims),
                cost_usd=iter_cost,
                stopped_reason=None,
            )
        )

    return traces


__all__ = [
    "RefinementState",
    "evaluate_coverage",
    "run_refinement",
]
