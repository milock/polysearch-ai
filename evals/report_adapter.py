"""Normalize a pipeline-report json into one shape the metrics can read.

Two report shapes reach the harness:

- **public** — the ``polysearch`` package schema: ``totals.cost_usd`` /
  ``totals.duration_sec``, sources under ``layers[].results[]``, refinement under
  ``refinement_iterations[]`` (``queries_run``), synthesis in ``synthesis_md``.
- **internal** — the upstream pipeline: ``total_cost_usd`` / ``duration_sec`` at
  the top level, sources spread across several top-level lists (grounded web
  results, knowledge-base hits), refinement under ``refinement_traces[]``
  (``followup_queries``), synthesis in a ``synthesis`` object. ``verification``
  carries the *same* field names in both.

Rather than hardcode the internal list names, the source collector scans every
top-level list whose elements carry a ``tier`` field — robust to naming and
free of any internal vocabulary.

The normalizer resolves each logical field from whichever shape carries it and
**never silently defaults a missing field to zero** — an absent field becomes
``None`` (or an empty container plus a warning). Silent zeros poisoned a whole
eval round; a ``None`` plus a per-row warning is honest and diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedReport:
    """A report reduced to the fields the metrics need, plus provenance."""

    shape: str  # "public" | "internal" | "unknown"
    cost_usd: float | None
    duration_sec: float | None
    sources: list[dict[str, Any]]  # {"url": str|None, "tier": str, "title": str}
    sources_found: bool  # whether any known source container existed at all
    verification: dict[str, Any] | None
    refinement_rounds: int | None
    synthesis_text: str
    warnings: list[str] = field(default_factory=list)


def detect_shape(report: dict) -> str:
    """Best-effort label of which pipeline produced ``report`` (for warnings)."""
    if any(k in report for k in ("total_cost_usd", "web_items", "refinement_traces")):
        return "internal"
    if any(k in report for k in ("totals", "layers", "synthesis_md")):
        return "public"
    return "unknown"


def _resolve_cost(report: dict) -> float | None:
    totals = report.get("totals")
    if isinstance(totals, dict) and totals.get("cost_usd") is not None:
        return float(totals["cost_usd"])
    if report.get("total_cost_usd") is not None:
        return float(report["total_cost_usd"])
    return None


def _resolve_duration(report: dict) -> float | None:
    totals = report.get("totals")
    if isinstance(totals, dict) and totals.get("duration_sec") is not None:
        return float(totals["duration_sec"])
    if report.get("duration_sec") is not None:
        return float(report["duration_sec"])
    return None


def _as_source(s: dict) -> dict[str, Any]:
    return {
        "url": s.get("url"),
        "tier": str(s.get("tier", "UNKNOWN")).upper(),
        "title": s.get("title", ""),
    }


def _collect_sources(report: dict) -> tuple[list[dict[str, Any]], bool]:
    """Return (sources, found). ``found`` is True when a source container was
    present — distinguishing "no sources" from "wrong shape".

    Public reports nest sources under ``layers[].results[]``. Internal reports
    spread tiered sources across several top-level lists; rather than name them,
    collect from every top-level list whose dict elements carry a ``tier`` field.
    Untiered lists (raw search results, errors, refinement traces) are skipped, so
    they never inflate the tier-mix denominator.
    """
    sources: list[dict[str, Any]] = []
    found = False

    if "layers" in report:
        found = True
        for layer in report.get("layers") or []:
            for s in layer.get("results", []) or []:
                sources.append(_as_source(s))
        return sources, found

    for value in report.values():
        if not isinstance(value, list) or not value:
            continue
        if any(isinstance(s, dict) and "tier" in s for s in value):
            found = True
            for s in value:
                if isinstance(s, dict) and "tier" in s:
                    sources.append(_as_source(s))

    return sources, found


def _resolve_refinement_rounds(report: dict) -> int | None:
    """Count refinement iterations that ran follow-up queries. ``None`` when
    neither shape's refinement container is present (can't tell); ``0`` when the
    container exists but no iteration ran queries."""
    if "refinement_iterations" in report:
        return sum(
            1 for it in report["refinement_iterations"] or [] if it.get("queries_run")
        )
    if "refinement_traces" in report:
        return sum(
            1 for t in report["refinement_traces"] or [] if t.get("followup_queries")
        )
    return None


def _resolve_synthesis_text(report: dict) -> str:
    """Fallback text for placeholder / coverage scans when no report md is on hand.
    The collected report markdown is always preferred; this only backstops it."""
    md = report.get("synthesis_md")
    if isinstance(md, str) and md.strip():
        return md
    syn = report.get("synthesis")
    if isinstance(syn, dict):
        parts: list[str] = []
        for key in ("executive_summary", "key_findings", "quality_notes"):
            val = syn.get(key)
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, list):
                parts.extend(str(x) for x in val)
        return "\n".join(parts)
    return ""


def normalize_report(report: dict) -> NormalizedReport:
    """Reduce either report shape to a :class:`NormalizedReport`, accruing a
    warning for every field that could not be found in either shape."""
    shape = detect_shape(report)
    warnings: list[str] = []

    cost = _resolve_cost(report)
    if cost is None:
        warnings.append("cost not found (neither totals.cost_usd nor total_cost_usd)")

    duration = _resolve_duration(report)
    if duration is None:
        warnings.append("duration not found (neither totals.duration_sec nor duration_sec)")

    sources, sources_found = _collect_sources(report)
    if not sources_found:
        warnings.append("no source container found (no layers, no tiered top-level list)")

    verification = report.get("verification")
    if not isinstance(verification, dict):
        verification = None
        warnings.append("no verification block found")

    refinement_rounds = _resolve_refinement_rounds(report)
    if refinement_rounds is None:
        warnings.append(
            "no refinement container found (refinement_iterations / refinement_traces)"
        )

    return NormalizedReport(
        shape=shape,
        cost_usd=cost,
        duration_sec=duration,
        sources=sources,
        sources_found=sources_found,
        verification=verification,
        refinement_rounds=refinement_rounds,
        synthesis_text=_resolve_synthesis_text(report),
        warnings=warnings,
    )
