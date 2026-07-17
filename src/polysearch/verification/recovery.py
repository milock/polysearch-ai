"""Recovery pass — re-source weakly-verified claims before the run gives up.

When a first verification pass comes back weak, this module re-sources the
failed claims with a few scoped Perplexity queries so a better citation can
replace a dead or mismatched one, then hands the results back for
re-verification.

Trigger (checked by the caller via :func:`should_recover`): verification ran,
produced at least ``settings.recovery_min_citations`` claims, and fewer than
``settings.recovery_rate_threshold`` of them verified.

Mechanism: take the failed claims (dead URLs, quote/number mismatches),
re-source each with a hardened single-shot Perplexity query that prefers primary
domains and is told to answer "UNVERIFIED" rather than invent a source, then
return the recovered results for tier-tagging + re-verification.

The "prefer primary domains" instruction is prompt-only (see
``_QUERY_TEMPLATE``). It used to also be API-enforced via
``domain_filter=high_tier_domains()[:20]`` — the curated HIGH-tier core of
``domain_tiers.yaml`` — but that forced every recovery query to the same
generic allowlist regardless of topic, so an off-topic report (e.g.
PostgreSQL internals) got NIH/BLS/SEC citations injected into its HIGH bucket.
Recovered results are now relevance-gated against the claim they were
re-sourced for instead (see ``_relevant_to_claim``).
"""

from __future__ import annotations

import asyncio

from polysearch.config import Settings
from polysearch.extractors.claims import _content_tokens, _number_core, _MIN_OVERLAP
from polysearch.output.schema import Claim, VerificationReport
from polysearch.providers import perplexity
from polysearch.providers.perplexity import PerplexityResult

_FAILED_STATUSES = {"URL_DEAD", "QUOTE_NOT_FOUND", "NUMBER_MISMATCH"}

_QUERY_TEMPLATE = (
    'Find the primary, citable source for this claim: "{claim}"\n'
    "Topic context: {topic}\n"
    "Rules: cite ONLY URLs you are certain exist. Prefer official and primary "
    "domains (government agencies, standards bodies, vendor newsrooms, "
    "peer-reviewed journals, major trade press). State the exact figure or "
    "fact and give the direct URL to the page that contains it. If you cannot "
    "find a verifiable primary source, reply exactly: UNVERIFIED - no primary "
    "source found. Never invent a URL, a number, or a quote."
)


def should_recover(report: VerificationReport | None, settings: Settings) -> bool:
    if report is None or report.total_citations < settings.recovery_min_citations:
        return False
    return (report.verified_ok / report.total_citations) < settings.recovery_rate_threshold


def _failed_claims(
    report: VerificationReport, claims: list[Claim], max_queries: int
) -> list[Claim]:
    by_id = {c.claim_id: c for c in claims}
    failed_ids = {
        r.claim_id for r in report.results if r.status in _FAILED_STATUSES
    }
    failed = [by_id[i] for i in failed_ids if i in by_id]
    # Number-bearing claims are the load-bearing ones; re-source those first.
    failed.sort(key=lambda c: (len(c.numbers), len(c.text)), reverse=True)
    # De-dupe near-identical claim texts (same first 60 chars)
    seen: set[str] = set()
    unique: list[Claim] = []
    for c in failed:
        key = c.text[:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:max_queries]


def _relevant_to_claim(claim: Claim, result: PerplexityResult) -> bool:
    """Gate a recovered result against the claim it was re-sourced for.

    Mirrors ``extractors.claims._localize_source_urls``'s scoring: content-word
    overlap of >= ``_MIN_OVERLAP`` between the claim text and the recovered
    answer, or a literal hit on one of the claim's numeric figures. A result
    that never mentions the claim's subject terms is noise regardless of how
    authoritative its domain is.
    """
    claim_tokens = _content_tokens(claim.text)
    number_cores = [c for c in (_number_core(n) for n in claim.numbers) if c]
    answer_low = (result.answer or "").lower()
    if any(core in answer_low for core in number_cores):
        return True
    overlap = (
        len(claim_tokens & _content_tokens(result.answer or "")) / len(claim_tokens)
        if claim_tokens
        else 0.0
    )
    return overlap >= _MIN_OVERLAP


async def recover(
    topic: str,
    report: VerificationReport,
    claims: list[Claim],
    *,
    settings: Settings,
) -> list[PerplexityResult]:
    """Re-source failed claims. Returns recovered PerplexityResults (may be empty)."""
    targets = _failed_claims(report, claims, settings.recovery_max_queries)
    if not targets:
        return []

    async def _one(claim: Claim) -> tuple[Claim, list[PerplexityResult]]:
        query = _QUERY_TEMPLATE.format(claim=claim.text[:400], topic=topic[:200])
        results = await perplexity.research(
            query,
            depth="standard",
            sub_questions=1,  # no decomposition — the query IS the sub-question
            recency="month",
            api_key=settings.perplexity_api_key,
        )
        return claim, results

    batches = await asyncio.gather(*(_one(c) for c in targets), return_exceptions=True)

    recovered: list[PerplexityResult] = []
    for batch in batches:
        if isinstance(batch, BaseException):
            continue
        claim, results = batch
        for r in results:
            if r.error:
                continue
            # Honest-empty: the model said it couldn't source it. Keep only
            # answers that actually carry citations and aren't declared dead.
            if "UNVERIFIED" in (r.answer or "")[:200] and not r.citations:
                continue
            if not r.citations:
                continue
            if not _relevant_to_claim(claim, r):
                continue
            recovered.append(r)
    return recovered


def merge_reports(
    first: VerificationReport, second: VerificationReport
) -> VerificationReport:
    return VerificationReport(
        total_citations=first.total_citations + second.total_citations,
        verified_ok=first.verified_ok + second.verified_ok,
        broken=first.broken + second.broken,
        quote_mismatches=first.quote_mismatches + second.quote_mismatches,
        number_mismatches=first.number_mismatches + second.number_mismatches,
        paywalled=first.paywalled + second.paywalled,
        undated=first.undated + second.undated,
        skipped_budget=first.skipped_budget + second.skipped_budget,
        fetch_blocked=first.fetch_blocked + second.fetch_blocked,
        blocked_sources=first.blocked_sources + second.blocked_sources,
        results=first.results + second.results,
        total_cost_usd=first.total_cost_usd + second.total_cost_usd,
        total_duration_ms=first.total_duration_ms + second.total_duration_ms,
        claims_total=first.claims_total + second.claims_total,
        claims_supported=first.claims_supported + second.claims_supported,
        credits_exhausted_hit=first.credits_exhausted_hit or second.credits_exhausted_hit,
    )
