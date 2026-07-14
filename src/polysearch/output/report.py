"""Markdown + JSON report writer for a completed pipeline run.

``write_report`` renders a :class:`PipelineReport` to a human-readable markdown
file and a sibling JSON dump, written atomically (temp file + ``os.replace``)
under ``YYYY-MM-DD-<slug>`` names.

Ported behaviours (from the internal research-pipeline report writer):
- tier-bucketed "Sources by Quality Tier", with ``URL_DEAD`` citations excluded
  from the buckets and listed under "Excluded (dead links)";
- a placeholder guard that refuses to save a report still holding unresolved
  ``{{...}}`` tokens unless ``settings.allow_placeholders`` is set;
- depth reconciliation in "Pipeline Decisions" (what ran vs. what the classifier
  suggested), plus a Pipeline Errors section and a cost/stats block.

Public-only behaviour:
- ``BLOCKED_SOURCE`` citations are excluded into their own "Excluded (blocked
  sources)" section, kept separate from dead links; ``FETCH_BLOCKED`` sources
  are blocked-but-alive, so they stay in the buckets and are only noted;
- a "Refinement Trace" section auditing the goal-driven refinement loop;
- a "Style Audit" section, rendered only when ``settings.style_constraints`` set.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from polysearch import __version__
from polysearch.config import Settings
from polysearch.output.schema import PipelineReport, RefinementTrace

# Matches an unresolved template token like ``{{EXEC_SUMMARY — agent to fill}}``.
_UNRESOLVED_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\s*[—-].*?\}\}")

_TIER_ORDER = ["HIGH", "MEDIUM", "LOW", "COMMUNITY", "SME", "UNKNOWN"]
_TIER_LABELS = {
    "HIGH": "High (primary, peer-reviewed, official)",
    "MEDIUM": "Medium (reputable secondary, trade)",
    "LOW": "Low (opinion, marketing, unverified)",
    "COMMUNITY": "Community (engagement signal — sentiment, not proof)",
    "SME": "SME (internal knowledge base)",
    "UNKNOWN": "Unknown (not in whitelist)",
}

_STOP_LABELS = {
    "goal_met": "goal met",
    "cost_ceiling": "cost ceiling reached",
    "dry": "no new unique sources (converged)",
    "no_new_queries": "no unique follow-up queries",
    "parse_abort": "evaluator parse failure (aborted)",
}


def slugify(value: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "research"


def write_report(
    pipeline_report: PipelineReport,
    *,
    output_dir: Path | str,
    settings: Settings | None = None,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Render + atomically write the markdown report and its JSON sibling.

    ``settings`` defaults to a bare :class:`Settings` (no env read), so the
    writer is hermetic; the orchestrator passes its resolved settings so
    ``allow_placeholders`` and ``style_constraints`` take effect. Returns
    ``(md_path, json_path)``.
    """
    settings = settings or Settings()
    when = generated_at or datetime.now()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{when.strftime('%Y-%m-%d')}-{slugify(pipeline_report.topic)}"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    md = render_markdown(pipeline_report, settings=settings, generated_at=when)
    unresolved = _UNRESOLVED_PLACEHOLDER_RE.findall(md)
    if unresolved and not settings.allow_placeholders:
        raise RuntimeError(
            f"Refusing to save report: {len(unresolved)} unresolved placeholders. "
            f"First: {unresolved[0]!r}. Run with a synthesis model, or set "
            f"allow_placeholders / POLYSEARCH_ALLOW_PLACEHOLDERS=1 for hand-fill."
        )

    _atomic_write(md_path, md)
    data = json.dumps(
        pipeline_report.model_dump(mode="json"), indent=2, sort_keys=True, default=str
    )
    _atomic_write(json_path, data)
    return md_path, json_path


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, prefix=".tmp-", suffix=path.suffix, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def render_markdown(
    report: PipelineReport,
    *,
    settings: Settings | None = None,
    generated_at: datetime | None = None,
) -> str:
    settings = settings or Settings()
    when = generated_at or datetime.now()
    cost, duration = _cost_duration(report)

    parts: list[str] = []
    parts.append(f"# Research: {report.topic}\n")
    parts.append(
        f"Generated: {when.isoformat(timespec='seconds')} | "
        f"Depth: {report.depth} | polysearch v{__version__} | "
        f"Cost: ${cost:.2f} | Duration: {_fmt_duration(duration)}\n"
    )
    parts.append(_render_synthesis(report.synthesis_md))
    if settings.style_constraints:
        parts.append(_render_style_audit(settings.style_constraints))
    if report.classification:
        parts.append(_render_pipeline_decisions(report.classification, report.depth))
    parts.append(_render_sources_by_tier(report))
    if report.refinement_iterations:
        parts.append(_render_refinement_trace(report.refinement_iterations))
    if report.pipeline_errors:
        parts.append(_render_pipeline_errors(report.pipeline_errors))
    parts.append(_render_stats(report))

    return "\n".join(p for p in parts if p)


def _render_synthesis(synthesis_md: str) -> str:
    body = (synthesis_md or "").strip() or (
        "{{SYNTHESIS — synthesizer did not run; rerun with a synthesis model "
        "or set allow_placeholders}}"
    )
    return f"\n## Synthesis\n\n{body}\n"


def _render_style_audit(constraints: str) -> str:
    """Surface that style constraints were in effect for this run.

    Ports the mechanism only — the constraint text is whatever the caller
    passed; this writer does not evaluate or enforce it.
    """
    return (
        "\n## Style Audit\n\n"
        "_Synthesis was generated under these style constraints:_\n\n"
        f"{constraints.strip()}\n"
    )


def _render_pipeline_decisions(classification: dict, depth_ran: str) -> str:
    lines = ["\n## Pipeline Decisions"]
    if classification.get("query_type"):
        lines.append(f"- **Topic type:** {classification['query_type']}")
    if "time_sensitive" in classification:
        lines.append(f"- **Time-sensitive:** {classification['time_sensitive']}")
    if "domain_related" in classification:
        lines.append(f"- **Domain-related:** {classification['domain_related']}")
    lines.append(f"- **Depth ran:** {depth_ran}")
    suggested = classification.get("suggested_depth")
    if suggested:
        lines.append(f"- **Depth suggested by classifier:** {suggested}")
        if suggested != depth_ran:
            lines.append(
                f"- _Note: ran at {depth_ran}, classifier suggested {suggested}._"
            )
    if classification.get("reasons"):
        lines.append(f"- **Reasons:** {'; '.join(classification['reasons'])}")
    return "\n".join(lines) + "\n"


def _render_sources_by_tier(report: PipelineReport) -> str:
    dead_urls: set[str] = set()
    blocked_urls: set[str] = set()
    fetch_blocked = 0
    if report.verification:
        for r in report.verification.results:
            if r.status == "URL_DEAD":
                dead_urls.add(r.url)
            elif r.status == "BLOCKED_SOURCE":
                blocked_urls.add(r.url)
            elif r.status == "FETCH_BLOCKED":
                fetch_blocked += 1

    buckets: dict[str, list[tuple[str, str, str]]] = {t: [] for t in _TIER_ORDER}
    for layer in report.layers:
        for s in layer.results:
            if s.url in dead_urls or s.url in blocked_urls:
                continue
            buckets.setdefault(s.tier, []).append((s.url, s.title, s.published_date or ""))

    lines = ["\n## Sources by Quality Tier"]
    ordered = _TIER_ORDER + [t for t in buckets if t not in _TIER_ORDER]
    for tier in ordered:
        items = buckets.get(tier, [])
        if not items:
            continue
        seen: set[str] = set()
        unique = [it for it in items if not (it[0] in seen or seen.add(it[0]))]
        label = _TIER_LABELS.get(tier, f"{tier} (uncategorized)")
        lines.append(f"\n### {label} ({len(unique)})")
        for url, title, date in unique[:15]:
            d = f" _[{date}]_" if date else ""
            lines.append(f"- [{title[:100]}]({url}){d}")

    if dead_urls:
        lines.append(f"\n### Excluded (dead links) ({len(dead_urls)} URL_DEAD)")
        for url in sorted(dead_urls)[:15]:
            lines.append(f"- {url}")
    if blocked_urls:
        lines.append(
            f"\n### Excluded (blocked sources) ({len(blocked_urls)} BLOCKED_SOURCE)"
        )
        for url in sorted(blocked_urls)[:15]:
            lines.append(f"- {url}")
    if fetch_blocked:
        lines.append(
            f"\n_{fetch_blocked} source(s) fetch-blocked (403/429/5xx) — kept in "
            "buckets as blocked-but-alive, not counted as dead._"
        )
    return "\n".join(lines) + "\n"


def _render_refinement_trace(traces: list[RefinementTrace]) -> str:
    """Per-iteration audit of the goal-driven refinement loop: verdict, coverage
    score, follow-up queries run, new sources/claims found, cost, and stop note."""
    ran = [t for t in traces if t.queries_run and t.stopped_reason != "goal_met"]
    lines = ["\n## Refinement Trace"]
    lines.append(
        f"Iterations: {len(traces)} | Follow-up rounds run: {len(ran)} | "
        f"Refinement cost: ${sum(t.cost_usd for t in traces):.4f}"
    )
    for t in traces:
        header = f"\n### Iteration {t.iteration} — coverage {t.verdict.coverage_score:.2f}"
        if t.verdict.goal_met:
            header += " · goal met"
        lines.append(header)
        if t.verdict.reasoning:
            lines.append(f"- **Verdict:** {t.verdict.reasoning}")
        if t.queries_run:
            lines.append("- **Follow-up queries run:**")
            for q in t.queries_run:
                lines.append(f"  - {q}")
        lines.append(
            f"- **New sources:** {t.new_sources} · "
            f"**New claims:** {t.new_claims} · **Cost:** ${t.cost_usd:.4f}"
        )
        stop = _STOP_LABELS.get(t.stopped_reason or "")
        if stop:
            lines.append(f"- **Stopped:** {stop}")
    return "\n".join(lines) + "\n"


def _render_pipeline_errors(errors: list[str]) -> str:
    lines = ["\n## Pipeline Errors"]
    for e in errors:
        lines.append(f"- {e}")
    return "\n".join(lines) + "\n"


def _render_stats(report: PipelineReport) -> str:
    cost, duration = _cost_duration(report)
    lines = ["\n## Pipeline Stats"]
    for layer in report.layers:
        n = len(layer.results)
        err = f" · error: {layer.error}" if layer.error else ""
        lines.append(
            f"- {layer.layer}: {n} sources · ${layer.cost_usd:.3f} · "
            f"{layer.duration_ms}ms{err}"
        )
    if report.verification:
        v = report.verification
        lines.append(
            f"- Verification: ${v.total_cost_usd:.3f} · "
            f"{v.claims_supported}/{v.claims_total} claims supported "
            f"({v.verified_ok}/{v.total_citations} pairs OK)"
        )
    if report.recovery_ran:
        lines.append("- Recovery pass: ran")
    lines.append(f"- **Total:** ${cost:.2f} · {_fmt_duration(duration)}")
    return "\n".join(lines) + "\n"


def _cost_duration(report: PipelineReport) -> tuple[float, float]:
    """Resolve total cost + duration, preferring ``totals`` then layer sums."""
    totals = report.totals or {}
    cost = totals.get("cost_usd")
    if cost is None:
        cost = sum(layer.cost_usd for layer in report.layers)
        if report.verification:
            cost += report.verification.total_cost_usd
    duration = totals.get("duration_sec")
    if duration is None:
        duration = sum(layer.duration_ms for layer in report.layers) / 1000.0
    return float(cost), float(duration)


def _fmt_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s"
