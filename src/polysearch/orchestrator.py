"""Pipeline orchestrator — wires every layer into one research run.

``run_research`` is the public entry point (re-exported from ``polysearch``). It
runs the full flow (classify → parallel first pass → claim extraction →
verification → recovery → refinement → report), talking to the outside world
only through the :class:`~polysearch.providers.base.Providers` bundle so every
layer is credential-gated and failure-isolated.

Flow:
  1. classify the topic (rules only, no network)
  2. fire the first pass in parallel: research (Perplexity), web grounding
     (Firecrawl), the fused community layer, and — when active — deep research
  3. for PERSON topics, run the registered person-context hook and, when a
     LinkedIn profile URL is present and enrichment is active, the LinkedIn layer
  4. synthesize the first pass into a cited markdown body
  5. extract verifiable claims from BOTH the synthesis and each research layer's
     narrative answers, then verify them within the depth budget
  6. run the recovery pass when verification came back weak
  7. run the goal-driven refinement loop (Task 17), which grows the corpus and
     re-synthesizes until the goal is met or a guard trips
  8. assemble + write the :class:`~polysearch.output.schema.PipelineReport`

Every layer is isolated: a raising layer becomes a ``pipeline_errors`` entry and
the run continues; a null provider short-circuits with its ``reason`` surfaced in
the report. No layer failure aborts the run.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from polysearch.classifier import classify
from polysearch.community.filter import evaluate_community
from polysearch.community.fusion import fuse
from polysearch.config import DEPTH_PROFILES, Settings
from polysearch.extractors.claims import extract_claims
from polysearch.output.report import slugify, write_report
from polysearch.run_status import RunManifest, clear_sentinel, write_sentinel
from polysearch.output.schema import (
    Claim,
    LayerOutput,
    PipelineReport,
    SourceResult,
)
from polysearch.providers.base import Providers, build_providers
from polysearch.refinement import RefinementState, _canonical_url, run_refinement
from polysearch.sources.authority import classify_url
from polysearch.verification.recovery import (
    merge_reports,
    recover as recovery_recover,
    should_recover,
)

# Native community adapters window unrelated viral chatter out on their own; 30
# days is the default community recency window.
_COMMUNITY_WINDOW_DAYS = 30

_LINKEDIN_URL_RE = re.compile(r"https?://(?:[\w.-]*\.)?linkedin\.com/[^\s\"'<>]+", re.IGNORECASE)

# The first pass gets only a FRACTION of whatever's remaining when a global
# time_budget_s is set, reserving a tail slice for synthesis, verification, and
# the recovery pass — those phases run strictly AFTER the first pass and were
# previously handed only the harness's outer slack (~120s) when the first pass
# legitimately consumed the whole remaining budget, risking a SIGKILL before
# write_report ever ran. No time_budget_s set -> this fraction never applies
# (first_pass_budget stays None, unbounded, exactly as before).
_FIRST_PASS_BUDGET_FRACTION = 0.8

# Below this many remaining seconds, the refinement loop is skipped outright
# rather than started — a loop needs at least one coverage-evaluator call plus
# a research/grounding fan-out plus verify plus re-synthesis, so a few seconds
# of remaining budget is not enough to do anything useful before being cut off
# mid-iteration. Deliberately NOT an exact `== 0.0` check: a small positive
# remainder (e.g. 3s) must not be treated as "plenty of time" and start a
# multi-minute loop anyway.
_MIN_REFINEMENT_BUDGET_S = 30.0


def _provider_reason(provider: Any) -> str | None:
    """The ``reason`` a null provider carries (why its layer is inactive), or None."""
    return getattr(provider, "reason", None)


def _linkedin_url_in(topic: str) -> str | None:
    m = _LINKEDIN_URL_RE.search(topic)
    return m.group(0) if m else None


async def _isolate(
    name: str, awaitable: Any, *, budget: float | None = None
) -> tuple[LayerOutput, str | None]:
    """Await a layer coroutine, converting a raise into an empty errored layer.

    ``budget`` (seconds, ``None`` = unbounded) bounds how long the layer is
    allowed to run; a layer that would exceed it is cancelled rather than left to
    run unbounded — this is the wall-clock-budget enforcement point every
    first-pass layer goes through. ``asyncio.wait_for`` with ``timeout=None``
    behaves exactly like a plain ``await``, so passing ``budget=None`` (the
    default, "no time budget configured") is a no-op and preserves prior
    behavior exactly.

    Returns ``(layer, pipeline_error)``: the layer output (empty on failure) plus
    a pipeline-error string when the layer raised, timed out, OR came back
    carrying an ``error`` (a null provider's reason, a structured provider
    failure). ``duration_ms`` is populated on every path, including timeout and
    generic-exception failures, so a layer's wall-clock is always visible.
    """
    start = time.perf_counter()
    try:
        out = await asyncio.wait_for(awaitable, timeout=budget)
    except asyncio.TimeoutError:
        duration_ms = int((time.perf_counter() - start) * 1000)
        msg = f"cancelled: exceeded remaining time budget ({budget:.0f}s)"
        return (
            LayerOutput(layer=name, error=msg, duration_ms=duration_ms),
            f"{name}: {msg}",
        )
    except Exception as exc:  # noqa: BLE001 — one layer must never sink the run
        duration_ms = int((time.perf_counter() - start) * 1000)
        return (
            LayerOutput(layer=name, error=f"{type(exc).__name__}: {exc}", duration_ms=duration_ms),
            f"{name}: {exc}",
        )
    err = f"{name}: {out.error}" if out.error else None
    return out, err


async def _bounded_await(coro: Any, *, remaining: float | None) -> Any:
    """Await ``coro`` bounded by ``remaining`` seconds (``None`` = unbounded).

    Raises ``asyncio.TimeoutError`` (never anything else on the budget path) when
    the budget is either already exhausted (``remaining <= 0`` — the coroutine is
    closed WITHOUT ever being awaited, so no half-started background work and no
    "coroutine was never awaited" warning) or is exceeded mid-flight. Callers
    fold this into their existing ``except Exception`` isolation by catching
    ``asyncio.TimeoutError`` first with their own pipeline-errors message — this
    is the shared enforcement point for every post-first-pass phase (synthesis,
    verification, recovery, refinement) that previously ran unbounded once the
    first pass had spent most or all of a configured ``time_budget_s``.
    """
    if remaining is not None and remaining <= 0:
        coro.close()
        raise asyncio.TimeoutError("no remaining time budget")
    return await asyncio.wait_for(coro, timeout=remaining)


async def _run_community(
    sources: list[Any],
    topic: str,
    *,
    window_days: int,
    limit: int,
    adapter_timeout: float | None = None,
) -> tuple[LayerOutput, list[str]]:
    """Search every community source in parallel, fuse, and relevance-gate.

    Each adapter is failure-isolated (they set ``last_error`` and return ``[]``
    rather than raise); its ``last_error`` is surfaced as a pipeline note.
    ``adapter_timeout`` (seconds, ``None`` = unbounded) additionally bounds each
    adapter's ``search()`` call directly — a source stuck in retries/backoff
    (Reddit's OAuth->public->ScrapeCreators fallback chain, a rate-limiter 429
    backoff of up to 300s) is cancelled rather than stalling every sibling and the
    whole fused layer; fusion proceeds with whatever the other sources returned.
    The fused, tier-COMMUNITY results pass through the relevance gate, which
    suppresses the whole layer with a note when more than 70% of items are
    off-topic.
    """
    start = time.perf_counter()

    async def _bounded(src: Any) -> list[SourceResult]:
        coro = src.search(topic, window_days=window_days, limit=limit)
        return await asyncio.wait_for(coro, timeout=adapter_timeout)

    tasks = [_bounded(src) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors: list[str] = []
    per_source: list[list[SourceResult]] = []
    for src, res in zip(sources, results, strict=True):
        name = getattr(src, "name", "?")
        if isinstance(res, asyncio.TimeoutError):
            errors.append(f"community/{name}: timed out after {adapter_timeout:.0f}s")
            continue
        if isinstance(res, BaseException):
            errors.append(f"community/{name}: {type(res).__name__}: {res}")
            continue
        per_source.append(res)
        last_error = getattr(src, "last_error", None)
        if last_error:
            errors.append(f"community/{name}: {last_error}")

    fused = fuse(per_source, limit=limit, window_days=window_days) if per_source else []
    outcome = evaluate_community(fused, topic)
    if outcome.suppressed and outcome.note:
        errors.append(f"community: {outcome.note}")
    duration_ms = int((time.perf_counter() - start) * 1000)
    return LayerOutput(layer="community", results=outcome.results, duration_ms=duration_ms), errors


def _recovered_material(recovered: list[Any]) -> tuple[list[SourceResult], list[Claim]]:
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


async def run_research(
    topic: str,
    *,
    depth: str = "standard",
    settings: Settings | None = None,
    providers: Providers | None = None,
    person_hook: Any | None = None,
    enabled_layers: set[str] | None = None,
    verify: bool = True,
    recovery: bool = True,
    verify_budget: float | None = None,
    deep_research: bool = False,
    max_iterations: int | None = None,
    output_dir: Path | str | None = None,
    write: bool = True,
    time_budget_s: float | None = None,
) -> PipelineReport:
    """Run the full research pipeline for ``topic`` and return its report.

    ``providers`` defaults to :func:`build_providers` over ``settings`` (env);
    inject a bundle to run with mocks/nulls. ``enabled_layers`` restricts which
    first-pass layers run (names: ``research``, ``grounding``, ``community``,
    ``deep_research``, ``linkedin``); ``None`` runs them all. ``verify`` /
    ``recovery`` toggle those passes; ``verify_budget`` overrides the depth
    profile's first-pass verification budget; ``deep_research`` forces the deep
    layer at any depth (still requires an active deep-research provider);
    ``max_iterations`` overrides the profile's refinement cap (0 disables it).
    ``person_hook`` is the optional PERSON-context seam. When ``write`` is set the
    report is written under ``output_dir`` (defaults to ``settings.output_dir``).

    ``time_budget_s`` (falls back to ``settings.time_budget_s``, default ``None``
    = unbounded) is a global wall-clock budget for the whole run. When set, the
    first pass (research, grounding, deep-research, community) gets only
    ``_FIRST_PASS_BUDGET_FRACTION`` (80%) of whatever's remaining — a layer that
    would exceed its slice is cancelled, recorded as a ``pipeline_errors`` entry
    with its ``duration_ms`` populated, and the run degrades gracefully. The
    reserved 20% tail, plus every phase that follows (synthesis, verification,
    recovery), is itself bounded by the actual remaining budget at that point,
    so none of those phases can be squeezed to nothing (or SIGKILLed with no
    report) just because the first pass used most of the budget. The refinement
    loop is skipped outright below a small floor of remaining time, and bounded
    for its own duration once started, rather than running unbounded to
    completion.

    **Completion signal.** Every run maintains a heartbeat-bearing manifest at
    ``~/.cache/polysearch/runs/<run-id>.json`` (phase + ``heartbeat_at``, so
    orchestrators can distinguish queued from dead), and — when ``write`` is
    set — leaves a ``<report>.done.json`` sentinel next to the report on exit:
    ``status: "complete"`` after the atomic save, ``status: "failed"`` when the
    run raises. Gate on the sentinel, never on the ``.md`` alone (see
    :mod:`polysearch.run_status`). The written paths are exposed on the report
    as ``output_md_path`` / ``output_json_path``.
    """
    settings = settings or Settings.from_env()

    # Expected report path — used for the failure sentinel when the run dies
    # before (or during) the write. The success sentinel uses the actual paths.
    expected_md: Path | None = None
    if write:
        out_dir = Path(output_dir) if output_dir is not None else settings.output_dir
        expected_md = (
            Path(out_dir).expanduser().resolve()
            / f"{time.strftime('%Y-%m-%d')}-{slugify(topic)}.md"
        )
        clear_sentinel(expected_md)

    manifest = RunManifest.create(topic=topic, depth=depth, output_path=expected_md)
    heartbeat = asyncio.create_task(manifest.heartbeat_loop())
    try:
        report = await _run_research_inner(
            topic,
            depth=depth,
            settings=settings,
            providers=providers,
            person_hook=person_hook,
            enabled_layers=enabled_layers,
            verify=verify,
            recovery=recovery,
            verify_budget=verify_budget,
            deep_research=deep_research,
            max_iterations=max_iterations,
            output_dir=output_dir,
            write=write,
            time_budget_s=time_budget_s,
            manifest=manifest,
        )
    except (Exception, KeyboardInterrupt) as exc:
        reason = (
            "interrupted (SIGINT)"
            if isinstance(exc, KeyboardInterrupt)
            else f"{type(exc).__name__}: {exc}"
        )
        if expected_md is not None:
            write_sentinel(
                expected_md,
                status="failed",
                started_at=manifest.started_at,
                exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1,
                phases_completed=manifest.phases_completed,
                error=reason,
            )
        manifest.finish("failed", error=reason)
        raise
    finally:
        heartbeat.cancel()

    if write and report.output_md_path:
        write_sentinel(
            report.output_md_path,
            status="complete",
            started_at=manifest.started_at,
            exit_code=0,
            phases_completed=manifest.phases_completed + ["saving"],
            json_path=report.output_json_path,
            total_cost_usd=report.totals.get("cost_usd"),
            pipeline_error_count=len(report.pipeline_errors),
        )
    manifest.finish("complete")
    return report


async def _run_research_inner(
    topic: str,
    *,
    depth: str,
    settings: Settings,
    providers: Providers | None,
    person_hook: Any | None,
    enabled_layers: set[str] | None,
    verify: bool,
    recovery: bool,
    verify_budget: float | None,
    deep_research: bool,
    max_iterations: int | None,
    output_dir: Path | str | None,
    write: bool,
    time_budget_s: float | None,
    manifest: RunManifest,
) -> PipelineReport:
    """The actual pipeline body — see :func:`run_research` for the contract."""
    started = time.perf_counter()
    if providers is None:
        providers = build_providers(settings)

    budget_s = time_budget_s if time_budget_s is not None else settings.time_budget_s
    deadline = (started + budget_s) if budget_s is not None else None

    def _remaining() -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.perf_counter())

    classification = classify(topic)
    if depth not in DEPTH_PROFILES:
        depth = "standard"
    profile = DEPTH_PROFILES[depth]
    manifest.update_phase("first-pass")

    pipeline_errors: list[str] = []
    layers: list[LayerOutput] = []

    def _enabled(name: str) -> bool:
        return enabled_layers is None or name in enabled_layers

    # ── Parallel first pass ──────────────────────────────────────────────────
    # Every first-pass layer shares the SAME remaining-budget snapshot, taken
    # once here (not re-read per layer) — they run concurrently, so this bounds
    # the whole first-pass phase to one slice rather than compounding waits.
    # Only _FIRST_PASS_BUDGET_FRACTION of it is handed out; the rest is a
    # reserved tail for synthesis/verification/recovery/refinement-shutdown,
    # which run strictly after and must not be squeezed to nothing.
    _remaining_at_first_pass = _remaining()
    first_pass_budget = (
        _remaining_at_first_pass * _FIRST_PASS_BUDGET_FRACTION
        if _remaining_at_first_pass is not None
        else None
    )

    layer_coros: dict[str, Any] = {}
    if _enabled("research"):
        layer_coros["research"] = _isolate(
            "research",
            providers.research.research(
                topic, sub_questions=profile.perplexity_sub_questions, depth=depth
            ),
            budget=first_pass_budget,
        )
    if _enabled("grounding"):
        # The grounder's ``ground`` protocol carries no recency parameter and the
        # constructor takes none either, so the classifier's time_sensitive signal
        # cannot be threaded through without changing that seam. Grounding uses its
        # documented default recency window ("month"); noted rather than forced.
        layer_coros["grounding"] = _isolate(
            "grounding",
            providers.grounder.ground(
                topic,
                limit=profile.firecrawl_limit,
                scrape_top_k=profile.firecrawl_scrape_top_k,
            ),
            budget=first_pass_budget,
        )
    deep_should_run = (
        providers.deep_research is not None
        and _enabled("deep_research")
        and (deep_research or profile.deep_research_eligible)
    )
    if deep_should_run:
        layer_coros["deep_research"] = _isolate(
            "deep_research",
            providers.deep_research.research(
                topic,
                sub_questions=profile.perplexity_sub_questions,
                depth=depth,
                time_budget_s=first_pass_budget,
            ),
            budget=first_pass_budget,
        )

    community_task: asyncio.Task | None = None
    if _enabled("community") and providers.community_sources:
        # Per-adapter timeout is always applied (config default ~60s) regardless
        # of whether a global time_budget_s is set — a blocked adapter (Reddit's
        # fallback chain, a 429 backoff) must never stall the fused layer. When a
        # global budget IS set and leaves less time than that default, the
        # smaller of the two wins.
        adapter_timeout = settings.community_adapter_timeout_s
        if first_pass_budget is not None:
            adapter_timeout = max(0.0, min(adapter_timeout, first_pass_budget))
        community_task = asyncio.create_task(
            _run_community(
                providers.community_sources,
                topic,
                window_days=_COMMUNITY_WINDOW_DAYS,
                limit=profile.community_limit,
                adapter_timeout=adapter_timeout,
            )
        )

    gathered = await asyncio.gather(*layer_coros.values()) if layer_coros else []
    layer_by_name: dict[str, LayerOutput] = {}
    for name, (out, err) in zip(layer_coros.keys(), gathered, strict=True):
        layer_by_name[name] = out
        layers.append(out)
        if err:
            pipeline_errors.append(err)

    if community_task is not None:
        # A second, outer budget check: the per-adapter timeout already bounds
        # the community task closely, but this is the same safety net every
        # other first-pass layer gets, using the budget remaining right now.
        community_budget = _remaining()
        community_start = time.perf_counter()
        try:
            community_layer, community_errors = await asyncio.wait_for(
                community_task, timeout=community_budget
            )
        except asyncio.TimeoutError:
            duration_ms = int((time.perf_counter() - community_start) * 1000)
            msg = f"cancelled: exceeded remaining time budget ({community_budget:.0f}s)"
            community_layer = LayerOutput(layer="community", error=msg, duration_ms=duration_ms)
            community_errors = [f"community: {msg}"]
        layers.append(community_layer)
        pipeline_errors.extend(community_errors)

    research_layer = layer_by_name.get("research")
    deep_layer = layer_by_name.get("deep_research")

    # ── PERSON enrichment (person-context hook + LinkedIn) ───────────────────
    if classification.query_type == "PERSON":
        if person_hook is not None:
            hook_start = time.perf_counter()
            try:
                pr = await person_hook.lookup(topic)
            except Exception as exc:  # noqa: BLE001
                pr = None
                pipeline_errors.append(f"person_hook: {exc}")
            if pr is not None:
                hook_ms = int((time.perf_counter() - hook_start) * 1000)
                layers.append(LayerOutput(layer="person-context", results=[pr], duration_ms=hook_ms))

        linkedin = getattr(providers, "linkedin", None)
        if linkedin is not None and getattr(linkedin, "active", False) and _enabled("linkedin"):
            profile_url = _linkedin_url_in(topic)
            if profile_url:
                li_start = time.perf_counter()
                try:
                    sr = await linkedin.enrich(profile_url)
                except Exception as exc:  # noqa: BLE001
                    sr = None
                    pipeline_errors.append(f"linkedin: {exc}")
                if sr is not None:
                    li_ms = int((time.perf_counter() - li_start) * 1000)
                    layers.append(LayerOutput(layer="linkedin", results=[sr], duration_ms=li_ms))
                elif getattr(linkedin, "reason", None):
                    pipeline_errors.append(f"linkedin: {linkedin.reason}")
            else:
                pipeline_errors.append(
                    "linkedin: PERSON topic carries no LinkedIn profile URL to enrich (skipped)"
                )

    # ── First-pass synthesis ─────────────────────────────────────────────────
    manifest.update_phase("synthesis")
    synthesis_md = ""
    synth_cost = 0.0
    try:
        synthesis_md, synth_cost = await _bounded_await(
            providers.synthesizer.synthesize(
                topic, layers, style_constraints=settings.style_constraints
            ),
            remaining=_remaining(),
        )
    except asyncio.TimeoutError:
        pipeline_errors.append("synthesis: cancelled — exceeded remaining time budget")
    except Exception as exc:  # noqa: BLE001
        pipeline_errors.append(f"synthesis: {exc}")
    if not synthesis_md.strip():
        reason = _provider_reason(providers.synthesizer)
        if reason:
            pipeline_errors.append(f"synthesizer: {reason}")

    # ── Authoritative-source extraction on HIGH-tier scraped pages ───────────
    # The grounder stashes its enriched GroundedItems (full markdown + tier) on
    # ``last_items``; the projected SourceResults carry only a snippet, so
    # structured extraction runs against the markdown here. profile.authoritative_top_k
    # caps how many HIGH-tier pages are mined (quick 0 / standard 2 / deep 4).
    # Gated on grounding having run THIS turn: a reused Providers bundle keeps the
    # prior run's last_items, so mining them when grounding was skipped this run
    # would splice a previous topic's pages into this report.
    authoritative_facts: list[Any] = []
    authoritative_pages = 0
    if profile.authoritative_top_k > 0 and "grounding" in layer_by_name:
        last_items = getattr(providers.grounder, "last_items", None) or []
        high_pages = [
            it
            for it in last_items
            if getattr(it, "tier", None) == "HIGH" and getattr(it, "markdown", "")
        ][: profile.authoritative_top_k]
        authoritative_pages = len(high_pages)
        if high_pages:
            from polysearch.extractors.authoritative import extract_auto, load_schemas

            try:
                schemas = load_schemas()
                for item in high_pages:
                    authoritative_facts.extend(extract_auto(item.url, item.markdown, schemas))
            except Exception as exc:  # noqa: BLE001 — extraction must never sink the run
                pipeline_errors.append(f"authoritative: {exc}")

    # ── Claim extraction: synthesis + research answers + authoritative facts ──
    all_urls = [s.url for lyr in layers for s in lyr.results if s.url]
    # (url, snippet) pairs so extract_claims can attribute each claim to the
    # sources whose snippet actually relates to it, instead of the whole corpus —
    # tightens verification pairing and drops off-topic pages from a claim's cites.
    all_sources = [
        (s.url, s.snippet or "") for lyr in layers for s in lyr.results if s.url
    ]
    claims: list[Claim] = []
    seen_claim_ids: set[str] = set()

    def _add_claims(new_claims: list[Claim]) -> list[Claim]:
        added: list[Claim] = []
        for c in new_claims:
            if c.claim_id not in seen_claim_ids:
                seen_claim_ids.add(c.claim_id)
                claims.append(c)
                added.append(c)
        return added

    if synthesis_md.strip():
        _add_claims(extract_claims(synthesis_md, all_urls, sources=all_sources))
    for lyr in (research_layer, deep_layer):
        if lyr is None:
            continue
        lyr_sources = [(s.url, s.snippet or "") for s in lyr.results if s.url]
        lyr_urls = [u for (u, _s) in lyr_sources] or all_urls
        # A layer's own sources localize its answer claims; fall back to the
        # corpus when the layer surfaced no sources of its own.
        answer_sources = lyr_sources or all_sources
        for answer in lyr.answers:
            _add_claims(extract_claims(answer, lyr_urls, sources=answer_sources))
    # Authoritative facts become claim-bearing text so they get verified like
    # everything else — value + surrounding context carry the figure to check.
    for fact in authoritative_facts:
        _add_claims(extract_claims(f"{fact.value}. {fact.context}", [fact.source_url]))

    # ── Verification ─────────────────────────────────────────────────────────
    manifest.update_phase("verification")
    verification = None
    vbudget = verify_budget if verify_budget is not None else profile.verify_budget_usd
    if verify and claims:
        try:
            verification = await _bounded_await(
                providers.verifier.verify(
                    claims,
                    budget_usd=vbudget,
                    max_concurrency=profile.verify_concurrency,
                ),
                remaining=_remaining(),
            )
        except asyncio.TimeoutError:
            pipeline_errors.append("verification: cancelled — exceeded remaining time budget")
        except Exception as exc:  # noqa: BLE001
            pipeline_errors.append(f"verification: {exc}")
    if verification is not None and verification.total_citations == 0:
        reason = _provider_reason(providers.verifier)
        if reason:
            pipeline_errors.append(f"verifier: {reason}")

    # ── Recovery pass (automated sonar-pro rescue) ───────────────────────────
    manifest.update_phase("recovery")
    recovery_ran = False
    if recovery and verification is not None and should_recover(verification, settings):
        recovery_start = time.perf_counter()
        try:
            recovered = await _bounded_await(
                recovery_recover(topic, verification, claims, settings=settings),
                remaining=_remaining(),
            )
        except asyncio.TimeoutError:
            recovered = []
            pipeline_errors.append("recovery_pass: cancelled — exceeded remaining time budget")
        except Exception as exc:  # noqa: BLE001
            recovered = []
            pipeline_errors.append(f"recovery_pass: {exc}")
        if recovered:
            recovery_ran = True
            ok_before = verification.verified_ok
            rec_sources, rec_claims = _recovered_material(recovered)
            if rec_sources:
                recovery_ms = int((time.perf_counter() - recovery_start) * 1000)
                layers.append(
                    LayerOutput(layer="recovery", results=rec_sources, duration_ms=recovery_ms)
                )
            fresh_rec_claims = _add_claims(rec_claims)
            if fresh_rec_claims and verify:
                try:
                    rec_ver = await _bounded_await(
                        providers.verifier.verify(
                            fresh_rec_claims,
                            budget_usd=min(1.0, vbudget),
                            max_concurrency=max(4, profile.verify_concurrency // 2),
                        ),
                        remaining=_remaining(),
                    )
                    verification = merge_reports(verification, rec_ver)
                except asyncio.TimeoutError:
                    pipeline_errors.append(
                        "recovery_verify: cancelled — exceeded remaining time budget"
                    )
                except Exception as exc:  # noqa: BLE001
                    pipeline_errors.append(f"recovery_verify: {exc}")
            pipeline_errors.append(
                f"recovery_pass: initial verification was low; ran {len(recovered)} "
                f"scoped re-source queries (verified {ok_before} -> "
                f"{verification.verified_ok} of {verification.total_citations})"
            )

    # ── Firecrawl credits-exhausted alert (read the flag, never string-scan) ──
    if verification is not None and verification.credits_exhausted_hit:
        pipeline_errors.append(
            "firecrawl: credits exhausted during verification; remaining citations "
            f"verified via httpx fallback ({verification.verified_ok}/"
            f"{verification.total_citations} citation-source pairs verified)"
        )

    # ── Goal-driven refinement loop (Task 17) ────────────────────────────────
    manifest.update_phase("refinement")
    refine_profile = profile
    if max_iterations is not None:
        refine_profile = replace(profile, max_refinement_iterations=max(0, max_iterations))
    traces: list[Any] = []
    refinement_budget = _remaining()
    if refine_profile.max_refinement_iterations > 0 and (
        refinement_budget is not None and refinement_budget < _MIN_REFINEMENT_BUDGET_S
    ):
        # The budget is already spent (or nearly so) by first pass +
        # verification + recovery. Skip the loop rather than starting
        # iterations doomed to be cut off mid-flight — the report still gets
        # whatever the first pass produced instead of a hard timeout with
        # nothing. This is a FLOOR, not an exact-zero check: a small positive
        # remainder is not "plenty of time" for a coverage-evaluator call plus
        # a research/grounding fan-out plus verify plus re-synthesis.
        pipeline_errors.append(
            "refinement: skipped — insufficient remaining time budget "
            f"({refinement_budget:.0f}s < {_MIN_REFINEMENT_BUDGET_S:.0f}s floor)"
        )
    elif refine_profile.max_refinement_iterations > 0:
        seen_urls = {_canonical_url(s.url) for lyr in layers for s in lyr.results if s.url}
        state = RefinementState(
            layers=layers,
            claims=claims,
            verification=verification,
            synthesis_md=synthesis_md,
            prior_queries=[topic],
            seen_urls=seen_urls,
            pipeline_errors=[],
        )
        try:
            # Bounded so a loop that's already running can't outlive the
            # budget either — the loop's own guards (max iterations, cost
            # ceiling) don't know about wall-clock time, so this is the
            # enforcement point. ``state`` reflects whatever iterations fully
            # completed before cancellation; a mid-iteration cancel leaves it
            # internally consistent (no half-applied assignment), just short
            # of one more round.
            traces = await _bounded_await(
                run_refinement(topic, state, providers, refine_profile, settings=settings),
                remaining=refinement_budget,
            )
        except asyncio.TimeoutError:
            pipeline_errors.append(
                "refinement: cancelled mid-loop — exceeded remaining time budget"
            )
        except Exception as exc:  # noqa: BLE001
            pipeline_errors.append(f"refinement: {exc}")
        pipeline_errors.extend(state.pipeline_errors)
        # The loop mutates ``state`` in place; read the grown corpus back out.
        layers = state.layers
        claims = state.claims
        verification = state.verification
        synthesis_md = state.synthesis_md

    # ── Totals + report ──────────────────────────────────────────────────────
    total_cost = sum(lyr.cost_usd for lyr in layers) + synth_cost
    if verification is not None:
        total_cost += verification.total_cost_usd
    duration = time.perf_counter() - started

    # Surface the authoritative-extraction outcome in Pipeline Decisions (the
    # report writer renders the classifier's ``reasons`` under that section).
    classification_dict = classification.model_dump()
    if authoritative_pages:
        classification_dict.setdefault("reasons", []).append(
            f"authoritative extraction: {len(authoritative_facts)} fact(s) from "
            f"{authoritative_pages} HIGH-tier page(s)"
        )

    report = PipelineReport(
        topic=topic,
        depth=depth,
        classification=classification_dict,
        layers=layers,
        synthesis_md=synthesis_md,
        verification=verification,
        recovery_ran=recovery_ran,
        refinement_iterations=traces,
        pipeline_errors=pipeline_errors,
        totals={"cost_usd": round(total_cost, 6), "duration_sec": round(duration, 3)},
    )

    if write:
        manifest.update_phase("saving")
        out_dir = Path(output_dir) if output_dir is not None else settings.output_dir
        # A null synthesizer (missing credential) is an expected degraded mode, not
        # a bug — let the placeholder-bearing report save rather than refusing it.
        write_settings = settings if synthesis_md.strip() else replace(settings, allow_placeholders=True)
        md_path, json_path = write_report(report, output_dir=out_dir, settings=write_settings)
        report.output_md_path = str(md_path)
        report.output_json_path = str(json_path)

    return report


# ── Cross-report parallel synthesis (`--synthesize-parallel`) ────────────────


_CROSS_REPORT_PROMPT = (
    "You are synthesizing ACROSS {n} independent research reports on related "
    "topics. Read them all and produce exactly three markdown sections, using "
    "these headers verbatim:\n\n"
    "## TL;DR\n2-4 sentences on the single most important cross-cutting takeaway.\n\n"
    "## Tradeoffs\n3-6 bullets naming the tensions, disagreements, or "
    "decision tradeoffs that only appear when the reports are read together.\n\n"
    "## Gaps and Open Questions\n3-6 bullets on what none of the reports "
    "answer, or where they conflict and neither resolves it.\n\n"
    "Ground every point in the reports below. Do not introduce facts not present "
    "in them.\n\n---\n"
)


async def _cross_report_synthesize(
    bodies: list[tuple[str, str]], *, settings: Settings
) -> tuple[str, float]:
    """LLM cross-report synthesis. ``bodies`` is ``[(name, content), ...]``.

    Uses the OpenAI synthesis model, falling back to Anthropic; returns
    ``("", 0.0)`` when neither key is set so the caller can still emit a
    templated, index-only file.
    """
    prompt = _CROSS_REPORT_PROMPT.format(n=len(bodies))
    prompt += "\n\n".join(f"### Report: {name}\n\n{content}" for name, content in bodies)

    if settings.openai_api_key:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        msg = await client.chat.completions.create(
            model=settings.synthesis_model,
            max_completion_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.choices[0].message.content or ""
        usage = getattr(msg, "usage", None)
        cost = 0.0
        if usage:
            cost = (
                usage.prompt_tokens * settings.synthesis_price_in
                + usage.completion_tokens * settings.synthesis_price_out
            ) / 1_000_000
        return text.strip(), round(cost, 6)

    if settings.anthropic_api_key:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        msg = await client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in msg.content)
        return text.strip(), 0.0

    return "", 0.0


async def run_parallel_synthesis(
    glob_pattern: str, *, output_dir: Path | str, settings: Settings | None = None
) -> int:
    """Glob N existing report files, cross-synthesize them, write one report.

    Cross-report mode: matches ``glob_pattern`` (a filesystem glob), EXCLUDES
    any prior ``*-synthesis.md`` output to avoid feedback loops,
    reads the rest, and writes ``<output_dir>/<stem>-synthesis.md``. Returns 0 on
    success, 1 when no sub-reports match or all fail to read.
    """
    import sys
    from datetime import datetime

    settings = settings or Settings.from_env()
    out_dir = Path(output_dir).expanduser().resolve()
    started_at = datetime.now().isoformat(timespec="seconds")

    matches = sorted(_glob.glob(glob_pattern))
    sub_paths = [Path(p) for p in matches if not p.endswith("-synthesis.md")]
    if not sub_paths:
        sys.stderr.write(
            f"polysearch: parallel synthesis matched no reports for {glob_pattern!r} "
            "(excluding *-synthesis.md)\n"
        )
        print(f"FAILED: no reports matched {glob_pattern!r}")
        return 1

    bodies: list[tuple[str, str]] = []
    for path in sub_paths:
        try:
            bodies.append((path.name, path.read_text(encoding="utf-8")))
        except OSError as exc:
            sys.stderr.write(f"polysearch: skipping {path.name}: {exc}\n")
    if not bodies:
        sys.stderr.write("polysearch: parallel synthesis — all candidates failed to read\n")
        print("FAILED: all sub-report candidates failed to read")
        return 1

    body, cost = await _cross_report_synthesize(bodies, settings=settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y-%m-%d") + "-cross-report"
    out_path = out_dir / f"{stem}-synthesis.md"
    clear_sentinel(out_path)

    header = (
        f"# Cross-report synthesis\n\nSub-reports: {len(bodies)} | "
        f"Cost: ${cost:.4f}\n\n"
    )
    if not body:
        body = (
            "_No synthesis model configured (set OPENAI_API_KEY or "
            "ANTHROPIC_API_KEY); index only._"
        )
    index = "\n\n## Sub-report index\n\n" + "\n".join(
        f"- {name}" for name, _ in bodies
    )
    out_path.write_text(header + body + index + "\n", encoding="utf-8")
    write_sentinel(
        out_path,
        status="complete",
        started_at=started_at,
        exit_code=0,
        phases_completed=["parallel-synthesis"],
        total_cost_usd=cost,
    )
    sys.stderr.write(f"polysearch: cross-report synthesis written: {out_path}\n")
    print(f"RESULT: {out_path}")
    return 0


__all__ = ["run_research", "run_parallel_synthesis"]
