"""Citation verifier — fetches each cited URL, checks status, date, paywall
state, and that claimed quotes / numbers actually appear in the scraped content.

Design notes:
- Tier-prioritized ordering (``priority="tier"``): classify every URL first,
  then scrape HIGH -> MEDIUM -> COMMUNITY -> LOW -> UNKNOWN. When the budget cap
  bites, the important sources are already verified.
- Scraping routes through :mod:`polysearch.sources.scrape` — the resilient
  Firecrawl -> httpx -> Playwright chain. A page that is merely blocked
  (403/429/5xx, bot challenge, Firecrawl credit exhaustion) comes back
  ``FETCH_BLOCKED`` and is skipped-with-note, never scored ``URL_DEAD``; only an
  origin-confirmed dead page (404/410, DNS failure) is ``URL_DEAD``.
- Quote matching: ``rapidfuzz.fuzz.partial_ratio`` for fast fuzzy match, with an
  optional OpenAI-embedding fallback for borderline (0.70-0.85) scores to rescue
  paraphrases.
- Number matching: normalized regex tolerating trailing-zero, comma, and
  magnitude-suffix variants ("$1.2M" ~ "$1.2 million").
- Firecrawl 402 short-circuit: an :class:`asyncio.Event` latch so peers skip
  straight to the free httpx tier the moment credits run out; surfaced on the
  report via ``credits_exhausted_hit``.
- Status precedence: URL_DEAD > FETCH_BLOCKED > PAYWALLED > QUOTE_NOT_FOUND >
  NUMBER_MISMATCH > UNDATED > OK. Claim-level rollup marks a claim supported when
  each of its numbers and quotes matches in AT LEAST ONE cited source.

The public :class:`~polysearch.output.schema.VerificationResult` is a 4-field
model (claim_id/url/status/detail); the rich per-source scoring lives on the
module-private :class:`_ScoredResult` and is folded into ``detail`` when the
report is assembled.

Public:
    ``FirecrawlVerifier(settings).verify(claims, *, budget_usd, max_concurrency)``
    ``await verify(claims, *, budget_usd, max_concurrency, settings, ...)``
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from rapidfuzz import fuzz

from polysearch import ratelimit
from polysearch.config import Settings
from polysearch.output.schema import (
    Claim,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from polysearch.sources import authority, scrape
from polysearch.sources.authority import Tier
from polysearch.sources.scrape import FirecrawlCreditsExhausted

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Internal rich per-source result (mapped DOWN to the 4-field public model)
# -----------------------------------------------------------------------------


@dataclass
class _ScoredResult:
    """Rich verdict for one (claim, cited-URL) pair. Folded into the public
    ``VerificationResult`` (claim_id/url/status/detail) at report time; kept rich
    here so ``_aggregate`` can compute the claim-level rollup and provenance."""

    claim_id: str
    url: str
    status: VerificationStatus
    http_status: int | None = None
    tier_before: Tier = "UNKNOWN"
    tier_after: Tier = "UNKNOWN"
    published_date: str | None = None
    paywall_detected: bool = False
    quote_matches: list[dict] = field(default_factory=list)
    number_matches: list[dict] = field(default_factory=list)
    error: str | None = None
    scrape_method: str | None = None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_PAYWALL_MARKERS = (
    "subscribe to continue",
    "sign in to read",
    "paywall",
    "become a member",
    "unlock this article",
    "register to read",
    "subscribe to read",
    "subscribers only",
    "log in to continue reading",
)

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_BODY_MONTH_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+(20\d\d)\b",
    re.IGNORECASE,
)
_MONTH_NUM = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}

_TIER_RANK = {"HIGH": 0, "MEDIUM": 1, "COMMUNITY": 2, "SME": 2, "LOW": 3, "UNKNOWN": 4}

# Embedding fallback thresholds.
_EMBED_LOWER = 0.70
_EMBED_UPPER = 0.85
_EMBED_COSINE_THRESHOLD = 0.87


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _extract_date(metadata: dict[str, Any] | None, markdown: str | None = None) -> str | None:
    """Pull a YYYY-MM-DD publication date. Prefer HTML metadata; fall back to a
    dateline in the page head. Many government / newsletter / press-release pages
    put the date only in the body, so a metadata-only check wrongly marks them
    UNDATED."""
    if metadata:
        for key in (
            "published_date",
            "publishedDate",
            "article:published_time",
            "og:article:published_time",
            "datePublished",
            "published_time",
        ):
            val = metadata.get(key)
            if not val:
                continue
            m = _DATE_RE.search(str(val))
            if m:
                return m.group(0)
    # Body fallback: scan the head of the page for a dateline.
    if markdown:
        head = markdown[:2000]
        iso = _DATE_RE.search(head)
        if iso:
            return iso.group(0)
        mon = _BODY_MONTH_DATE_RE.search(head)
        if mon:
            month = _MONTH_NUM.get(mon.group(1).lower())
            if month:
                try:
                    return f"{int(mon.group(3)):04d}-{month:02d}-{int(mon.group(2)):02d}"
                except ValueError:
                    return None
    return None


def _looks_paywalled(markdown: str) -> bool:
    if not markdown:
        return False
    lowered = markdown.lower()
    return any(marker in lowered for marker in _PAYWALL_MARKERS)


def _normalize_number_to_regex(number: str) -> re.Pattern[str] | None:
    """Turn a number-string like '45%' or '$1.2M' into a regex that matches the
    same numeric value with flexible spacing / magnitude suffix.

    Tolerates trailing-zero variants ('50' ~ '50.0' ~ '50.00'), comma formatting
    ('1500' ~ '1,500'), and magnitude abbreviation vs word ('$50M' ~ '$50
    million'). Returns None if the string has no digits.
    """
    s = number.strip()
    if not s or not re.search(r"\d", s):
        return None

    # Detect magnitude suffix (M/B/million/billion/K/thousand).
    mag_match = re.search(r"(?i)(million|billion|thousand|[MBK])\b", s)
    mag = mag_match.group(1).lower() if mag_match else ""

    is_percent = "%" in s
    is_dollar = "$" in s

    # Extract the numeric portion (digits, commas, decimal).
    num_match = re.search(r"[\d][\d,]*(?:\.\d+)?", s)
    if not num_match:
        return None
    raw_num = num_match.group(0).replace(",", "")

    # Build a list of equivalent surface forms for the numeric value.
    variants: list[str] = [raw_num]
    if "." in raw_num:
        int_part, frac_part = raw_num.split(".", 1)
        frac_stripped = frac_part.rstrip("0")
        if frac_stripped != frac_part:
            variants.append(f"{int_part}.{frac_stripped}" if frac_stripped else int_part)
    else:
        variants.extend([f"{raw_num}.0", f"{raw_num}.00"])
        # Comma-formatted form for integers >= 4 digits.
        if len(raw_num) >= 4:
            rev = raw_num[::-1]
            with_commas = ",".join(rev[i : i + 3] for i in range(0, len(rev), 3))[::-1]
            variants.append(with_commas)

    # Dedupe while preserving order.
    seen: set[str] = set()
    unique_variants: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    num_pattern = "(?:" + "|".join(re.escape(v) for v in unique_variants) + ")"

    parts: list[str] = []
    if is_dollar:
        parts.append(r"\$\s*")
    parts.append(num_pattern)

    if is_percent:
        parts.append(r"\s*%")
    elif mag in {"m", "million"}:
        parts.append(r"\s*(?:M\b|million)")
    elif mag in {"b", "billion"}:
        parts.append(r"\s*(?:B\b|billion)")
    elif mag in {"k", "thousand"}:
        parts.append(r"\s*(?:K\b|thousand)")

    try:
        return re.compile("".join(parts), re.IGNORECASE)
    except re.error:
        return None


def _find_number(pattern: re.Pattern[str], markdown: str) -> tuple[bool, str]:
    """Search for pattern; return (matched, 80-char surrounding context)."""
    m = pattern.search(markdown)
    if not m:
        return False, ""
    start = max(0, m.start() - 40)
    end = min(len(markdown), m.end() + 40)
    ctx = markdown[start:end].replace("\n", " ").strip()
    return True, ctx


def _match_quote(quote: str, markdown: str, threshold: float) -> tuple[bool, float, str]:
    """Fuzzy-match ``quote`` against ``markdown``. Returns (matched, score_0_1,
    best_excerpt). Uses rapidfuzz partial_ratio_alignment for locating."""
    if not quote or not markdown:
        return False, 0.0, ""

    q_low = quote.lower()
    m_low = markdown.lower()

    score_raw: float
    try:
        alignment = fuzz.partial_ratio_alignment(q_low, m_low)
        score_raw = float(alignment.score)
        dst_start = int(getattr(alignment, "dest_start", 0))
        dst_end = int(getattr(alignment, "dest_end", 0))
    except Exception:
        score_raw = float(fuzz.partial_ratio(q_low, m_low))
        idx = m_low.find(q_low[:20]) if len(q_low) >= 20 else m_low.find(q_low)
        dst_start = idx if idx >= 0 else 0
        dst_end = dst_start + len(quote)

    score = score_raw / 100.0
    # Build a 200-char excerpt centered on the matched region in the ORIGINAL text.
    pad = max(0, (200 - (dst_end - dst_start)) // 2)
    start = max(0, dst_start - pad)
    end = min(len(markdown), dst_end + pad)
    excerpt = markdown[start:end].replace("\n", " ").strip()

    return score >= threshold, score, excerpt


async def _embed_cosine_max(
    client: Any, quote: str, markdown: str, model: str
) -> float | None:
    """Split markdown into paragraphs, embed them + quote, return max cosine.
    Returns None on any failure."""
    try:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", markdown) if p.strip()]
        if not paragraphs:
            return None
        # Cap paragraphs to keep cost bounded.
        paragraphs = paragraphs[:40]
        inputs = [quote] + paragraphs
        async with ratelimit.acquire("openai"):
            resp = await client.embeddings.create(model=model, input=inputs)
        vectors = [d.embedding for d in resp.data]
        q_vec = vectors[0]
        best = 0.0
        for vec in vectors[1:]:
            dot = sum(a * b for a, b in zip(q_vec, vec, strict=True))
            na = sum(a * a for a in q_vec) ** 0.5
            nb = sum(b * b for b in vec) ** 0.5
            if na == 0 or nb == 0:
                continue
            cos = dot / (na * nb)
            if cos > best:
                best = cos
        return best
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", None)
        if status == 429:
            resp_headers = getattr(getattr(e, "response", None), "headers", None)
            retry_after = resp_headers.get("retry-after") if resp_headers else None
            ratelimit.record_429("openai", float(retry_after) if retry_after else None)
        log.debug("embedding fallback failed: %s", e)
        return None


# -----------------------------------------------------------------------------
# Scrape + verify one URL
# -----------------------------------------------------------------------------


async def _verify_one(
    claim: Claim,
    url: str,
    tier_before: Tier,
    *,
    app: Any,
    openai_client: Any | None,
    fuzzy_threshold: float,
    embedding_model: str,
    sem: asyncio.Semaphore,
    credits_exhausted: asyncio.Event,
) -> _ScoredResult:
    async with sem:
        # Credit exhaustion is a Firecrawl-account-level failure, not a per-URL
        # signal — once the account is out of credits, there's no point burning
        # another Firecrawl call. But the citation itself may still be fetchable,
        # so every citation still gets the free httpx (+ Playwright) tier attempt
        # rather than being discarded wholesale.
        if credits_exhausted.is_set():
            result = await scrape.fetch_page(url, skip_firecrawl=True)
        else:
            try:
                result = await scrape.fetch_page(
                    url, firecrawl_app=app, on_credits_exhausted="raise"
                )
            except FirecrawlCreditsExhausted:
                # Latch the shared flag so peers skip straight to httpx the moment
                # they acquire the semaphore, then retry this same citation.
                credits_exhausted.set()
                result = await scrape.fetch_page(url, skip_firecrawl=True)

    if result.status == "URL_DEAD":
        return _ScoredResult(
            url=url,
            claim_id=claim.claim_id,
            status="URL_DEAD",
            http_status=result.http_status,
            tier_before=tier_before,
            tier_after=authority.downgrade_if_undated(tier_before, False),
            error=result.detail,
            scrape_method=result.method,
        )

    if result.status == "FETCH_BLOCKED":
        # Blocked-but-alive (403/429/5xx, bot-challenge, credit exhaustion) — NOT
        # excluded from Sources as dead, tier untouched since we have no evidence
        # the page itself is bad.
        return _ScoredResult(
            url=url,
            claim_id=claim.claim_id,
            status="FETCH_BLOCKED",
            http_status=result.http_status,
            tier_before=tier_before,
            tier_after=tier_before,
            error=result.detail,
            scrape_method=result.method,
        )

    markdown = result.markdown or ""

    # Date + paywall. Structured metadata (og:article:published_time etc.) is only
    # available when the winning tier was Firecrawl; httpx/Playwright fall back to
    # ``_extract_date``'s body-text dateline scan.
    published_date = _extract_date(result.metadata, markdown)
    paywalled = _looks_paywalled(markdown)

    # Quote matching.
    quote_matches: list[dict] = []
    any_quote_missing = False
    for q in claim.quotes:
        matched, score, excerpt = _match_quote(q, markdown, fuzzy_threshold)
        # Embedding fallback for borderline scores.
        if (
            not matched
            and openai_client is not None
            and _EMBED_LOWER <= score < _EMBED_UPPER
        ):
            cos = await _embed_cosine_max(openai_client, q, markdown, embedding_model)
            if cos is not None and cos >= _EMBED_COSINE_THRESHOLD:
                matched = True
                score = cos
                excerpt = excerpt or "(embedding match)"
        quote_matches.append(
            {
                "quote": q,
                "matched": matched,
                "best_score": round(score, 4),
                "best_match_excerpt": excerpt,
            }
        )
        if not matched:
            any_quote_missing = True

    # Number matching.
    number_matches: list[dict] = []
    any_number_missing = False
    for n in claim.numbers:
        pattern = _normalize_number_to_regex(n)
        if pattern is None:
            number_matches.append(
                {"number": n, "matched": False, "context": "(unparseable)"}
            )
            any_number_missing = True
            continue
        matched, ctx = _find_number(pattern, markdown)
        number_matches.append({"number": n, "matched": matched, "context": ctx})
        if not matched:
            any_number_missing = True

    # Status precedence: URL_DEAD / FETCH_BLOCKED (handled above) > PAYWALLED >
    # QUOTE_NOT_FOUND > NUMBER_MISMATCH > UNDATED > OK.
    if paywalled:
        status: VerificationStatus = "PAYWALLED"
    elif any_quote_missing:
        status = "QUOTE_NOT_FOUND"
    elif any_number_missing:
        status = "NUMBER_MISMATCH"
    elif published_date is None:
        status = "UNDATED"
    else:
        status = "OK"

    tier_after = authority.downgrade_if_undated(tier_before, published_date is not None)

    return _ScoredResult(
        url=url,
        claim_id=claim.claim_id,
        status=status,
        http_status=result.http_status,
        tier_before=tier_before,
        tier_after=tier_after,
        published_date=published_date,
        paywall_detected=paywalled,
        quote_matches=quote_matches,
        number_matches=number_matches,
        scrape_method=result.method,
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def _flatten_targets(claims: list[Claim]) -> list[tuple[Claim, str, Tier]]:
    """Expand claims into (claim, url, tier) triples."""
    out: list[tuple[Claim, str, Tier]] = []
    for c in claims:
        for u in c.source_urls:
            tier, _ = authority.classify_domain(u)
            out.append((c, u, tier))
    return out


def _sort_by_tier(
    targets: list[tuple[Claim, str, Tier]]
) -> list[tuple[Claim, str, Tier]]:
    return sorted(targets, key=lambda t: _TIER_RANK.get(t[2], 99))


def _render_detail(r: _ScoredResult) -> str | None:
    """Fold the rich per-source scoring into the public 4-field model's
    ``detail`` string."""
    if r.status in ("URL_DEAD", "FETCH_BLOCKED"):
        if r.http_status is not None and r.error:
            return f"HTTP {r.http_status}: {r.error}"
        if r.http_status is not None:
            return f"HTTP {r.http_status}"
        return r.error
    if r.status in ("SKIPPED_BUDGET", "BLOCKED_SOURCE"):
        return r.error

    # Content statuses: compact provenance summary.
    parts: list[str] = [f"tier {r.tier_before}->{r.tier_after}"]
    parts.append(f"dated {r.published_date}" if r.published_date else "undated")
    missing_q = [m["quote"] for m in r.quote_matches if not m.get("matched")]
    missing_n = [m["number"] for m in r.number_matches if not m.get("matched")]
    if missing_q:
        parts.append("missing quote(s): " + " | ".join(missing_q))
    if missing_n:
        parts.append("missing number(s): " + ", ".join(missing_n))
    if r.paywall_detected:
        parts.append("paywall detected")
    if r.scrape_method:
        parts.append(f"via {r.scrape_method}")
    return "; ".join(parts)


def _aggregate(
    scored: list[_ScoredResult],
    total_cost: float,
    total_duration_ms: int,
) -> VerificationReport:
    tally = {
        "OK": 0,
        "URL_DEAD": 0,
        "QUOTE_NOT_FOUND": 0,
        "NUMBER_MISMATCH": 0,
        "PAYWALLED": 0,
        "UNDATED": 0,
        "SKIPPED_BUDGET": 0,
        "FETCH_BLOCKED": 0,
        "BLOCKED_SOURCE": 0,
    }
    for r in scored:
        tally[r.status] = tally.get(r.status, 0) + 1

    # Claim-level rollup: group per-source results by claim, then mark a claim
    # supported when each of its numbers and quotes matched in at least one live
    # source. A claim with no figures counts as supported if any source is alive.
    # URL_DEAD / FETCH_BLOCKED / BLOCKED_SOURCE never yielded content, so they
    # don't count toward "alive" on their own.
    by_claim: dict[str, list[_ScoredResult]] = {}
    for r in scored:
        by_claim.setdefault(r.claim_id, []).append(r)
    claims_supported = 0
    for rs in by_claim.values():
        if not any(
            r.status not in ("URL_DEAD", "FETCH_BLOCKED", "BLOCKED_SOURCE") for r in rs
        ):
            continue
        num_ok: dict[str, bool] = {}
        quote_ok: dict[str, bool] = {}
        for r in rs:
            for m in r.number_matches:
                k = m.get("number")
                if k is not None:
                    num_ok[k] = num_ok.get(k, False) or bool(m.get("matched"))
            for q in r.quote_matches:
                k = q.get("quote")
                if k is not None:
                    quote_ok[k] = quote_ok.get(k, False) or bool(q.get("matched"))
        if all(num_ok.values()) and all(quote_ok.values()):
            claims_supported += 1

    results = [
        VerificationResult(
            claim_id=r.claim_id,
            url=r.url,
            status=r.status,
            detail=_render_detail(r),
        )
        for r in scored
    ]

    return VerificationReport(
        total_citations=len(scored),
        verified_ok=tally["OK"],
        broken=tally["URL_DEAD"],
        quote_mismatches=tally["QUOTE_NOT_FOUND"],
        number_mismatches=tally["NUMBER_MISMATCH"],
        paywalled=tally["PAYWALLED"],
        undated=tally["UNDATED"],
        skipped_budget=tally["SKIPPED_BUDGET"],
        fetch_blocked=tally["FETCH_BLOCKED"],
        blocked_sources=tally["BLOCKED_SOURCE"],
        results=results,
        total_cost_usd=round(total_cost, 6),
        total_duration_ms=total_duration_ms,
        claims_total=len(by_claim),
        claims_supported=claims_supported,
    )


async def verify(
    claims: list[Claim],
    *,
    budget_usd: float = 3.00,
    max_concurrency: int = 8,
    settings: Settings | None = None,
    firecrawl_cost_per_scrape: float | None = None,
    fuzzy_threshold: float | None = None,
    priority: Literal["tier", "order"] = "tier",
    firecrawl_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> VerificationReport:
    """Verify URL + quote + number claims in parallel.

    Thresholds and pricing come from ``settings`` (``fuzzy_threshold``,
    ``firecrawl_usd_per_credit``, ``embedding_model``); explicit keyword overrides
    take precedence when supplied. ``settings`` defaults to ``Settings.from_env()``.
    """
    started = time.perf_counter()
    settings = settings or Settings.from_env()
    fuzzy = settings.fuzzy_threshold if fuzzy_threshold is None else fuzzy_threshold
    cost_per_scrape = (
        settings.firecrawl_usd_per_credit
        if firecrawl_cost_per_scrape is None
        else firecrawl_cost_per_scrape
    )

    key = firecrawl_api_key or settings.firecrawl_api_key or os.environ.get(
        "FIRECRAWL_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY is not set. Populate the environment / .env or pass "
            "firecrawl_api_key= to verify()."
        )

    # Late import so tests can monkeypatch ``firecrawl``.
    from firecrawl import AsyncFirecrawl

    app = AsyncFirecrawl(api_key=key)

    openai_client = None
    oai_key = openai_api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if oai_key:
        try:
            from openai import AsyncOpenAI

            openai_client = AsyncOpenAI(api_key=oai_key)
        except Exception as e:  # noqa: BLE001
            log.debug("OpenAI client init failed: %s", e)
            openai_client = None

    all_targets = _flatten_targets(claims)
    # BLOCKED domains are hard-excluded — never scraped, never budgeted. Split
    # them out before tier-sort/budget so they can't consume Firecrawl spend or
    # displace a real scrape in the budget cutoff.
    blocked_targets = [t for t in all_targets if t[2] == "BLOCKED"]
    targets = [t for t in all_targets if t[2] != "BLOCKED"]
    if priority == "tier":
        targets = _sort_by_tier(targets)

    # Budget enforcement: determine how many scrapes we can afford.
    max_scrapes = (
        int(budget_usd // cost_per_scrape) if cost_per_scrape > 0 else len(targets)
    )
    to_verify = targets[:max_scrapes]
    to_skip = targets[max_scrapes:]

    sem = asyncio.Semaphore(max_concurrency)
    credits_exhausted = asyncio.Event()
    coros = [
        _verify_one(
            claim,
            url,
            tier,
            app=app,
            openai_client=openai_client,
            fuzzy_threshold=fuzzy,
            embedding_model=settings.embedding_model,
            sem=sem,
            credits_exhausted=credits_exhausted,
        )
        for (claim, url, tier) in to_verify
    ]
    verified_results: list[_ScoredResult] = []
    if coros:
        raw = await asyncio.gather(*coros, return_exceptions=True)
        for (claim, url, tier), item in zip(to_verify, raw, strict=True):
            # ``_verify_one`` no longer raises FirecrawlCreditsExhausted out to
            # here — a 402 is caught internally and retried via the httpx tier.
            # This branch is a defensive catch-all for unexpected exceptions.
            if isinstance(item, BaseException):
                verified_results.append(
                    _ScoredResult(
                        url=url,
                        claim_id=claim.claim_id,
                        status="URL_DEAD",
                        tier_before=tier,
                        tier_after=authority.downgrade_if_undated(tier, False),
                        error=f"{type(item).__name__}: {item}",
                    )
                )
            else:
                verified_results.append(item)

    # If credits ran out mid-batch, surface a clearer reason on the already-
    # deferred ``to_skip`` set so callers can distinguish budget-skip vs
    # credit-skip.
    skip_reason = (
        "firecrawl credits exhausted"
        if credits_exhausted.is_set()
        else "budget exhausted before scrape"
    )
    skipped_results = [
        _ScoredResult(
            url=url,
            claim_id=claim.claim_id,
            status="SKIPPED_BUDGET",
            tier_before=tier,
            tier_after=tier,
            error=skip_reason,
        )
        for (claim, url, tier) in to_skip
    ]

    blocked_results = [
        _ScoredResult(
            url=url,
            claim_id=claim.claim_id,
            status="BLOCKED_SOURCE",
            tier_before=tier,
            tier_after=tier,
            error="domain is hard-excluded (BLOCKED tier) — never scraped",
        )
        for (claim, url, tier) in blocked_targets
    ]

    all_results = verified_results + skipped_results + blocked_results
    total_cost = len(verified_results) * cost_per_scrape
    total_duration_ms = int((time.perf_counter() - started) * 1000)
    report = _aggregate(all_results, total_cost, total_duration_ms)
    report.credits_exhausted_hit = credits_exhausted.is_set()
    return report


class FirecrawlVerifier:
    """``CitationVerifier`` implementation backed by the resilient scrape chain.

    Holds a resolved ``Settings`` and delegates to the module-level
    :func:`verify`, which reads thresholds and pricing from it.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def verify(
        self, claims: list[Claim], *, budget_usd: float, max_concurrency: int
    ) -> VerificationReport:
        return await verify(
            claims,
            budget_usd=budget_usd,
            max_concurrency=max_concurrency,
            settings=self._settings,
        )


__all__ = [
    "Claim",
    "FirecrawlVerifier",
    "VerificationReport",
    "VerificationResult",
    "VerificationStatus",
    "verify",
]
