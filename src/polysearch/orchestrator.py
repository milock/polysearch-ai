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
from polysearch.output.report import write_report
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


def _provider_reason(provider: Any) -> str | None:
    """The ``reason`` a null provider carries (why its layer is inactive), or None."""
    return getattr(provider, "reason", None)


def _linkedin_url_in(topic: str) -> str | None:
    m = _LINKEDIN_URL_RE.search(topic)
    return m.group(0) if m else None


async def _isolate(name: str, awaitable: Any) -> tuple[LayerOutput, str | None]:
    """Await a layer coroutine, converting a raise into an empty errored layer.

    Returns ``(layer, pipeline_error)``: the layer output (empty on failure) plus
    a pipeline-error string when the layer raised OR came back carrying an
    ``error`` (a null provider's reason, a structured provider failure).
    """
    try:
        out = await awaitable
    except Exception as exc:  # noqa: BLE001 — one layer must never sink the run
        return LayerOutput(layer=name, error=f"{type(exc).__name__}: {exc}"), f"{name}: {exc}"
    err = f"{name}: {out.error}" if out.error else None
    return out, err


async def _run_community(
    sources: list[Any], topic: str, *, window_days: int, limit: int
) -> tuple[LayerOutput, list[str]]:
    """Search every community source in parallel, fuse, and relevance-gate.

    Each adapter is failure-isolated (they set ``last_error`` and return ``[]``
    rather than raise); its ``last_error`` is surfaced as a pipeline note. The
    fused, tier-COMMUNITY results pass through the relevance gate, which suppresses
    the whole layer with a note when more than 70% of items are off-topic.
    """
    tasks = [src.search(topic, window_days=window_days, limit=limit) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors: list[str] = []
    per_source: list[list[SourceResult]] = []
    for src, res in zip(sources, results, strict=True):
        name = getattr(src, "name", "?")
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
    return LayerOutput(layer="community", results=outcome.results), errors


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
    """
    started = time.perf_counter()
    settings = settings or Settings.from_env()
    if providers is None:
        providers = build_providers(settings)

    classification = classify(topic)
    if depth not in DEPTH_PROFILES:
        depth = "standard"
    profile = DEPTH_PROFILES[depth]

    pipeline_errors: list[str] = []
    layers: list[LayerOutput] = []

    def _enabled(name: str) -> bool:
        return enabled_layers is None or name in enabled_layers

    # ── Parallel first pass ──────────────────────────────────────────────────
    layer_coros: dict[str, Any] = {}
    if _enabled("research"):
        layer_coros["research"] = _isolate(
            "research",
            providers.research.research(
                topic, sub_questions=profile.perplexity_sub_questions, depth=depth
            ),
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
                topic, sub_questions=profile.perplexity_sub_questions, depth=depth
            ),
        )

    community_task: asyncio.Task | None = None
    if _enabled("community") and providers.community_sources:
        community_task = asyncio.create_task(
            _run_community(
                providers.community_sources,
                topic,
                window_days=_COMMUNITY_WINDOW_DAYS,
                limit=profile.community_limit,
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
        community_layer, community_errors = await community_task
        layers.append(community_layer)
        pipeline_errors.extend(community_errors)

    research_layer = layer_by_name.get("research")
    deep_layer = layer_by_name.get("deep_research")

    # ── PERSON enrichment (person-context hook + LinkedIn) ───────────────────
    if classification.query_type == "PERSON":
        if person_hook is not None:
            try:
                pr = await person_hook.lookup(topic)
            except Exception as exc:  # noqa: BLE001
                pr = None
                pipeline_errors.append(f"person_hook: {exc}")
            if pr is not None:
                layers.append(LayerOutput(layer="person-context", results=[pr]))

        linkedin = getattr(providers, "linkedin", None)
        if linkedin is not None and getattr(linkedin, "active", False) and _enabled("linkedin"):
            profile_url = _linkedin_url_in(topic)
            if profile_url:
                try:
                    sr = await linkedin.enrich(profile_url)
                except Exception as exc:  # noqa: BLE001
                    sr = None
                    pipeline_errors.append(f"linkedin: {exc}")
                if sr is not None:
                    layers.append(LayerOutput(layer="linkedin", results=[sr]))
                elif getattr(linkedin, "reason", None):
                    pipeline_errors.append(f"linkedin: {linkedin.reason}")
            else:
                pipeline_errors.append(
                    "linkedin: PERSON topic carries no LinkedIn profile URL to enrich (skipped)"
                )

    # ── First-pass synthesis ─────────────────────────────────────────────────
    synthesis_md = ""
    synth_cost = 0.0
    try:
        synthesis_md, synth_cost = await providers.synthesizer.synthesize(
            topic, layers, style_constraints=settings.style_constraints
        )
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
    authoritative_facts: list[Any] = []
    authoritative_pages = 0
    if profile.authoritative_top_k > 0:
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
        _add_claims(extract_claims(synthesis_md, all_urls))
    for lyr in (research_layer, deep_layer):
        if lyr is None:
            continue
        lyr_urls = [s.url for s in lyr.results if s.url] or all_urls
        for answer in lyr.answers:
            _add_claims(extract_claims(answer, lyr_urls))
    # Authoritative facts become claim-bearing text so they get verified like
    # everything else — value + surrounding context carry the figure to check.
    for fact in authoritative_facts:
        _add_claims(extract_claims(f"{fact.value}. {fact.context}", [fact.source_url]))

    # ── Verification ─────────────────────────────────────────────────────────
    verification = None
    vbudget = verify_budget if verify_budget is not None else profile.verify_budget_usd
    if verify and claims:
        try:
            verification = await providers.verifier.verify(
                claims,
                budget_usd=vbudget,
                max_concurrency=profile.verify_concurrency,
            )
        except Exception as exc:  # noqa: BLE001
            pipeline_errors.append(f"verification: {exc}")
    if verification is not None and verification.total_citations == 0:
        reason = _provider_reason(providers.verifier)
        if reason:
            pipeline_errors.append(f"verifier: {reason}")

    # ── Recovery pass (automated sonar-pro rescue) ───────────────────────────
    recovery_ran = False
    if recovery and verification is not None and should_recover(verification, settings):
        try:
            recovered = await recovery_recover(topic, verification, claims, settings=settings)
        except Exception as exc:  # noqa: BLE001
            recovered = []
            pipeline_errors.append(f"recovery_pass: {exc}")
        if recovered:
            recovery_ran = True
            ok_before = verification.verified_ok
            rec_sources, rec_claims = _recovered_material(recovered)
            if rec_sources:
                layers.append(LayerOutput(layer="recovery", results=rec_sources))
            fresh_rec_claims = _add_claims(rec_claims)
            if fresh_rec_claims and verify:
                try:
                    rec_ver = await providers.verifier.verify(
                        fresh_rec_claims,
                        budget_usd=min(1.0, vbudget),
                        max_concurrency=max(4, profile.verify_concurrency // 2),
                    )
                    verification = merge_reports(verification, rec_ver)
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
    refine_profile = profile
    if max_iterations is not None:
        refine_profile = replace(profile, max_refinement_iterations=max(0, max_iterations))
    traces: list[Any] = []
    if refine_profile.max_refinement_iterations > 0:
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
            traces = await run_refinement(
                topic, state, providers, refine_profile, settings=settings
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
        out_dir = Path(output_dir) if output_dir is not None else settings.output_dir
        # A null synthesizer (missing credential) is an expected degraded mode, not
        # a bug — let the placeholder-bearing report save rather than refusing it.
        write_settings = settings if synthesis_md.strip() else replace(settings, allow_placeholders=True)
        write_report(report, output_dir=out_dir, settings=write_settings)

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

    settings = settings or Settings.from_env()
    out_dir = Path(output_dir).expanduser().resolve()

    matches = sorted(_glob.glob(glob_pattern))
    sub_paths = [Path(p) for p in matches if not p.endswith("-synthesis.md")]
    if not sub_paths:
        sys.stderr.write(
            f"polysearch: parallel synthesis matched no reports for {glob_pattern!r} "
            "(excluding *-synthesis.md)\n"
        )
        return 1

    bodies: list[tuple[str, str]] = []
    for path in sub_paths:
        try:
            bodies.append((path.name, path.read_text(encoding="utf-8")))
        except OSError as exc:
            sys.stderr.write(f"polysearch: skipping {path.name}: {exc}\n")
    if not bodies:
        sys.stderr.write("polysearch: parallel synthesis — all candidates failed to read\n")
        return 1

    body, cost = await _cross_report_synthesize(bodies, settings=settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y-%m-%d") + "-cross-report"
    out_path = out_dir / f"{stem}-synthesis.md"

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
    sys.stderr.write(f"polysearch: cross-report synthesis written: {out_path}\n")
    return 0


__all__ = ["run_research", "run_parallel_synthesis"]
