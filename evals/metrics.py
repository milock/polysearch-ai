"""Programmatic metrics computed from a pipeline report json + its task.

These are the objective, model-free measurements that sit beside the LLM judge:
citation verification rate, source-tier mix, dead-link count, key-fact coverage
(fuzzy-matched against the task's checklist), refinement rounds run vs. expected,
placeholder leaks, cost, and duration. Everything is derived from the serialized
report — no network, no model — so the metrics are cheap, deterministic, and
identical for the public and internal targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from polysearch.config import DEPTH_PROFILES

# A key fact counts as covered when its best fuzzy alignment against any window
# of the report text clears this partial-ratio threshold (0–100).
KEY_FACT_MATCH_THRESHOLD = 80.0

# Fallback refinement ceiling for an unknown depth: the most permissive profile,
# so an unrecognized depth never false-flags a ceiling violation.
_FALLBACK_CEILING = max(p.max_refinement_iterations for p in DEPTH_PROFILES.values())


def refinement_ceiling_for_depth(depth: str | None) -> int:
    """The pipeline's ``max_refinement_iterations`` cap for ``depth``.

    Read from ``polysearch.config.DEPTH_PROFILES`` (quick 0 / standard 2 / deep 4)
    so the ceiling is per-depth, not a flat constant — a standard-depth run that
    somehow ran 3 iterations is a real violation the depth's cap of 2 catches.
    ``--depth-override`` is honored because it mutates the task's depth before the
    sweep, so the metric sees the depth the run actually used.
    """
    profile = DEPTH_PROFILES.get(depth or "")
    return profile.max_refinement_iterations if profile else _FALLBACK_CEILING

# Same unresolved-template shape the report writer guards against.
_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\s*[—-].*?\}\}")

# Tiers that count as trustworthy for the HIGH+MEDIUM mix metric.
_TRUSTED_TIERS = frozenset({"HIGH", "MEDIUM"})


@dataclass
class RunMetrics:
    """Objective metrics for a single task run."""

    verification_rate: float
    tier_mix_high_medium: float
    dead_links: int
    key_fact_coverage: float
    key_facts_total: int
    key_facts_covered: int
    refinement_rounds: int
    expects_refinement: bool
    refinement_ok: bool
    refinement_ceiling: int
    refinement_within_ceiling: bool
    placeholder_leaks: int
    cost_usd: float
    duration_sec: float
    total_citations: int
    total_sources: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_rate": round(self.verification_rate, 4),
            "tier_mix_high_medium": round(self.tier_mix_high_medium, 4),
            "dead_links": self.dead_links,
            "key_fact_coverage": round(self.key_fact_coverage, 4),
            "key_facts_total": self.key_facts_total,
            "key_facts_covered": self.key_facts_covered,
            "refinement_rounds": self.refinement_rounds,
            "expects_refinement": self.expects_refinement,
            "refinement_ok": self.refinement_ok,
            "refinement_ceiling": self.refinement_ceiling,
            "refinement_within_ceiling": self.refinement_within_ceiling,
            "placeholder_leaks": self.placeholder_leaks,
            "cost_usd": round(self.cost_usd, 6),
            "duration_sec": round(self.duration_sec, 3),
            "total_citations": self.total_citations,
            "total_sources": self.total_sources,
        }


def _report_text(report: dict, report_md: str | None) -> str:
    """The text key facts are matched against: prefer the rendered markdown, fall
    back to the synthesis body from the json."""
    if report_md:
        return report_md
    return str(report.get("synthesis_md", ""))


def _verification_rate(verification: dict | None) -> tuple[float, int, int]:
    """(rate, verified_ok, total_citations). Rate is 1.0 when nothing needed
    checking (no citations = no failures)."""
    if not verification:
        return 1.0, 0, 0
    total = int(verification.get("total_citations", 0) or 0)
    ok = int(verification.get("verified_ok", 0) or 0)
    if total == 0:
        return 1.0, ok, 0
    return ok / total, ok, total


def _dead_links(verification: dict | None) -> int:
    if not verification:
        return 0
    return sum(
        1 for r in verification.get("results", []) if r.get("status") == "URL_DEAD"
    )


def _tier_mix(report: dict) -> tuple[float, int]:
    """(fraction HIGH+MEDIUM, total unique sources). Deduplicated by URL so a
    source surfaced by several layers is not double-counted."""
    tier_by_url: dict[str, str] = {}
    for layer in report.get("layers", []):
        for src in layer.get("results", []):
            url = src.get("url")
            if url and url not in tier_by_url:
                tier_by_url[url] = str(src.get("tier", "UNKNOWN")).upper()
    total = len(tier_by_url)
    if total == 0:
        return 0.0, 0
    trusted = sum(1 for t in tier_by_url.values() if t in _TRUSTED_TIERS)
    return trusted / total, total


def _key_fact_coverage(key_facts: list[str], text: str) -> tuple[float, int, int]:
    """(coverage fraction, covered, total). Each fact is fuzzy-matched against the
    report text with rapidfuzz partial_ratio."""
    facts = [f for f in (key_facts or []) if f and f.strip()]
    if not facts:
        return 1.0, 0, 0
    haystack = text.lower()
    covered = 0
    for fact in facts:
        if fuzz.partial_ratio(fact.lower(), haystack) >= KEY_FACT_MATCH_THRESHOLD:
            covered += 1
    return covered / len(facts), covered, len(facts)


def _refinement_rounds(report: dict) -> int:
    """Count refinement iterations that actually ran follow-up queries. An
    iteration that only recorded a goal-met verdict without running queries is
    not a round."""
    rounds = 0
    for it in report.get("refinement_iterations", []):
        if it.get("queries_run"):
            rounds += 1
    return rounds


def _placeholder_leaks(report: dict, text: str) -> int:
    return len(_PLACEHOLDER_RE.findall(text))


def compute_metrics(
    report: dict, task: dict, *, report_md: str | None = None
) -> RunMetrics:
    """Compute every programmatic metric for one run.

    ``report`` is the parsed report json; ``task`` is the task definition (for
    ``key_facts`` and ``expects_refinement``); ``report_md`` is the rendered
    markdown when available (key-fact matching and placeholder detection run
    against it, else against the json's synthesis body).
    """
    text = _report_text(report, report_md)
    verification = report.get("verification")

    rate, _ok, total_citations = _verification_rate(verification)
    dead = _dead_links(verification)
    tier_frac, total_sources = _tier_mix(report)
    coverage, covered, facts_total = _key_fact_coverage(task.get("key_facts", []), text)
    rounds = _refinement_rounds(report)
    expects = bool(task.get("expects_refinement", False))
    # A run is refinement-OK when it ran at least one round if expected, and did
    # not run rounds it should not have (a task not expecting refinement running
    # a bounded round or two is fine — the ceiling is enforced at the gate).
    refinement_ok = (rounds >= 1) if expects else True
    ceiling = refinement_ceiling_for_depth(task.get("depth"))
    within_ceiling = rounds <= ceiling
    leaks = _placeholder_leaks(report, text)

    totals = report.get("totals") or {}
    cost = float(totals.get("cost_usd", 0.0) or 0.0)
    duration = float(totals.get("duration_sec", 0.0) or 0.0)

    return RunMetrics(
        verification_rate=rate,
        tier_mix_high_medium=tier_frac,
        dead_links=dead,
        key_fact_coverage=coverage,
        key_facts_total=facts_total,
        key_facts_covered=covered,
        refinement_rounds=rounds,
        expects_refinement=expects,
        refinement_ok=refinement_ok,
        refinement_ceiling=ceiling,
        refinement_within_ceiling=within_ceiling,
        placeholder_leaks=leaks,
        cost_usd=cost,
        duration_sec=duration,
        total_citations=total_citations,
        total_sources=total_sources,
    )
