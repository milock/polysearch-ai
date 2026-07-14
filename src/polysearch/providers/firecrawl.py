"""Web grounding: pluggable discovery + Firecrawl scraping + authority tagging.

Discovery finds candidate URLs; the top-K are then fetched/scraped through the
resilient fallback chain (``sources.scrape``) for full content. Each result is
classified by ``sources.authority.classify_domain``, downgraded when no
publication date is detected or when the scraped content looks paywalled, and
BLOCKED-tier hosts are dropped before the scrape step (never fetched, never
returned).

Discovery is pluggable per ``Settings.discovery_backend`` (dispatched through the
single ``_discover`` seam):

- ``"perplexity"`` (default): the flat-rate Perplexity Search API
  (``providers.perplexity.search``) — no extra key beyond the Perplexity one.
- ``"brave"``: the Brave Search API (``X-Subscription-Token`` header), when a
  ``brave_api_key`` is configured.
- ``"firecrawl"``: Firecrawl ``/v2/search`` (~2 credits / 10 results).

``FirecrawlGrounder(settings)`` implements the ``WebGrounder`` protocol; its
``ground`` returns a ``LayerOutput`` of ``SourceResult``. The internal
``_search`` returns the richer ``GroundedItem`` (with markdown / tier reason /
scrape provenance) that the ported unit tests assert against.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from polysearch import ratelimit
from polysearch.config import Settings
from polysearch.extractors.subject import extract_core_subject
from polysearch.output.schema import LayerOutput, SourceResult
from polysearch.providers import perplexity
from polysearch.sources import authority, scrape
from polysearch.sources.authority import Tier

log = logging.getLogger(__name__)

Recency = Literal["hour", "day", "week", "month", "year", "all"]

# Brave Search freshness parameter mapping (discovery time filter).
_FRESHNESS_BY_RECENCY: dict[str, str | None] = {
    "hour": "pd",  # Brave has no sub-day window; past-day is the closest.
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
    "all": None,
}

# Rough paywall markers. Case-insensitive substring match on markdown.
_PAYWALL_MARKERS = (
    "subscribe to continue",
    "sign in to read",
    "subscribe to read",
    "log in to continue reading",
    "become a subscriber",
    "subscribers only",
    "to continue reading, subscribe",
)

# Downgrade chain for paywall hits (one step).
_DOWNGRADE_ONE: dict[str, Tier] = {
    "HIGH": "MEDIUM",
    "MEDIUM": "LOW",
    "LOW": "COMMUNITY",
}

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# Trim a verbose, keyword-stuffed query to a core subject before it hits a
# search API (Perplexity gets the verbose form upstream; this only affects the
# discovery leg).
_MAX_SEARCH_TOKENS = 12

_SCRAPE_CONCURRENCY = 5


class GroundedItem(BaseModel):
    """Enriched grounding result — the internal contract the scrape/tier pipeline
    produces and the unit tests assert against. ``ground`` projects it onto the
    public ``SourceResult``."""

    url: str
    title: str
    markdown: str
    published_date: str | None = None
    domain: str
    tier: Tier = "UNKNOWN"
    tier_reason: str = ""
    snippet: str | None = None
    scraped: bool = False
    scraped_via: str | None = None  # "firecrawl" | "httpx" | "playwright" | None


class DiscoveryRetryable(Exception):
    """Wraps 429 / 5xx discovery responses so tenacity can retry them."""


def _extract_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _extract_date(metadata: dict[str, Any] | None) -> str | None:
    """Pull a YYYY-MM-DD publication date from scrape metadata.

    Preference order: explicit published_date -> modern firecrawl v2
    published_time -> og:article:published_time -> JSON-LD datePublished ->
    modified_time -> None."""
    if not metadata:
        return None
    candidates: list[Any] = [
        metadata.get("published_date"),
        metadata.get("publishedDate"),
        metadata.get("published_time"),  # modern firecrawl v2 metadata key
        metadata.get("article:published_time"),
        metadata.get("og:article:published_time"),
        metadata.get("datePublished"),
        metadata.get("modified_time"),
    ]
    for cand in candidates:
        if not cand:
            continue
        m = _DATE_RE.search(str(cand))
        if m:
            return m.group(0)
    return None


def _looks_paywalled(markdown: str) -> bool:
    if not markdown:
        return False
    lowered = markdown.lower()
    return any(marker in lowered for marker in _PAYWALL_MARKERS)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort convert a Pydantic-model-or-dict discovery result."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _build_initial_item(raw: Any) -> GroundedItem:
    """Turn a raw discovery document (url/title/description[/page_age]) into a
    snippet-only ``GroundedItem``. A ``page_age`` timestamp, when present (Brave
    supplies one on most results), seeds the published date so snippet-only items
    still carry a date; a later scrape overrides it only when it finds a better
    one."""
    d = _as_dict(raw)
    url = d.get("url") or ""
    title = d.get("title") or (d.get("metadata") or {}).get("title") or url
    snippet = d.get("description") or d.get("snippet") or d.get("markdown") or ""
    domain = _extract_domain(url)
    published = None
    page_age = d.get("page_age") or d.get("age") or d.get("published_date")
    if page_age:
        m = _DATE_RE.search(str(page_age))
        if m:
            published = m.group(0)
    return GroundedItem(
        url=url,
        title=title,
        markdown=snippet or "",
        snippet=snippet or None,
        domain=domain,
        published_date=published,
        scraped=False,
    )


def _apply_tier(item: GroundedItem, *, paywalled: bool = False) -> GroundedItem:
    """Tag tier + reason and apply undated/paywall downgrades."""
    tier, reason = authority.classify_domain(item.url)
    tier = authority.downgrade_if_undated(tier, item.published_date is not None)
    if item.published_date is None and tier != "UNKNOWN":
        reason = f"{reason}; undated"
    if paywalled:
        new_tier = _DOWNGRADE_ONE.get(tier, tier)
        if new_tier != tier:
            reason = f"{reason}; paywalled (downgraded {tier}->{new_tier})"
            tier = new_tier
        else:
            reason = f"{reason}; paywalled"
    item.tier = tier
    item.tier_reason = reason
    return item


class FirecrawlGrounder:
    """Live-web grounder: pluggable discovery + Firecrawl-backed scraping.

    Constructed only when ``FIRECRAWL_API_KEY`` is present (see
    ``providers.base._build_grounder``), so the scrape tier always has a key.
    """

    name = "grounding"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- public protocol ---------------------------------------------------

    async def ground(
        self, topic: str, *, limit: int, scrape_top_k: int
    ) -> LayerOutput:
        start = time.perf_counter()
        try:
            items = await self._search(topic, limit=limit, scrape_top_k=scrape_top_k)
        except Exception as e:  # noqa: BLE001 — surface as a layer error, never crash the run
            duration_ms = int((time.perf_counter() - start) * 1000)
            return LayerOutput(
                layer="grounding", duration_ms=duration_ms, error=f"{type(e).__name__}: {e}"
            )

        results = [
            SourceResult(
                url=it.url,
                title=it.title,
                snippet=(it.snippet or it.markdown or "")[:500],
                tier=it.tier,
                published_date=it.published_date,
                layer="grounding",
                engagement=None,
            )
            for it in items
        ]
        duration_ms = int((time.perf_counter() - start) * 1000)
        return LayerOutput(layer="grounding", results=results, duration_ms=duration_ms)

    # -- discovery dispatch ------------------------------------------------

    async def _discover(
        self,
        query: str,
        *,
        limit: int,
        recency: Recency,
        domain_filter: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Dispatch discovery to the configured backend, returning raw result
        dicts shaped for ``_build_initial_item`` (url/title/description
        [/page_age])."""
        backend = self.settings.discovery_backend
        if backend == "brave":
            freshness = _FRESHNESS_BY_RECENCY.get(recency)
            return await self._brave_search(query, count=limit, freshness=freshness)
        if backend == "firecrawl":
            return await self._firecrawl_search(query, limit=limit)
        return await self._perplexity_search(query, limit=limit)

    async def _perplexity_search(
        self, query: str, *, limit: int
    ) -> list[dict[str, Any]]:
        """Default backend: the flat-rate Perplexity Search API. Maps each
        ``SourceResult`` onto a raw discovery dict."""
        results = await perplexity.search(
            query, limit=limit, api_key=self.settings.perplexity_api_key
        )
        return [
            {
                "url": r.url,
                "title": r.title or r.url,
                "description": r.snippet or "",
                "page_age": r.published_date,
            }
            for r in results
            if r.url
        ]

    @retry(
        retry=retry_if_exception_type(DiscoveryRetryable),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=8),
        reraise=True,
    )
    async def _brave_search(
        self, query: str, *, count: int, freshness: str | None
    ) -> list[dict[str, Any]]:
        """Discover fresh results via the Brave Search API (search-only; scraping
        stays on Firecrawl). ``DiscoveryRetryable`` is the tenacity backoff signal
        for 429/5xx responses."""
        import httpx

        key = self.settings.brave_api_key
        if not key:
            raise RuntimeError(
                "BRAVE_API_KEY is not set but discovery_backend='brave'. "
                "Set brave_api_key or choose another discovery backend."
            )
        params: dict[str, Any] = {"q": query, "count": min(max(count, 1), 20)}
        if freshness:
            params["freshness"] = freshness
        headers = {"Accept": "application/json", "X-Subscription-Token": key}
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with ratelimit.acquire("brave"):
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params=params,
                    headers=headers,
                )
        if resp.status_code == 429 or resp.status_code >= 500:
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                ratelimit.record_429(
                    "brave", ratelimit.parse_retry_after(retry_after)
                )
            raise DiscoveryRetryable(f"brave http {resp.status_code}")
        resp.raise_for_status()
        results = ((resp.json().get("web") or {}).get("results")) or []
        out: list[dict[str, Any]] = []
        for r in results:
            url = r.get("url")
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "title": r.get("title") or url,
                    "description": r.get("description") or "",
                    "page_age": r.get("page_age") or r.get("age"),
                }
            )
        return out

    async def _firecrawl_search(
        self, query: str, *, limit: int
    ) -> list[dict[str, Any]]:
        """Firecrawl ``/v2/search`` discovery (~2 credits / 10 results)."""
        app = self._firecrawl_app()
        async with ratelimit.acquire("firecrawl"):
            data = await app.search(query, limit=limit)
        web = _as_dict(data).get("web") or []
        out: list[dict[str, Any]] = []
        for r in web:
            d = _as_dict(r)
            url = d.get("url")
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "title": d.get("title") or url,
                    "description": d.get("description") or "",
                    "page_age": None,
                }
            )
        return out

    # -- firecrawl app -----------------------------------------------------

    def _firecrawl_app(self) -> Any:
        """Construct the modern async Firecrawl client (lazy import keeps the
        module import-light and test stubs simple)."""
        from firecrawl import AsyncFirecrawl

        return AsyncFirecrawl(api_key=self.settings.firecrawl_api_key)

    # -- core grounding ----------------------------------------------------

    def _trim_query(self, query: str, domain_filter: list[str] | None) -> str:
        q = query.strip()
        # Keyword-stuffed research strings return 0 results from search APIs; trim
        # to a core subject, then hard-cap at ~12 tokens.
        if len(q.split()) > _MAX_SEARCH_TOKENS:
            try:
                q_short = (extract_core_subject(q) or "").strip()
                if q_short and len(q_short) >= 3:
                    q = q_short
            except Exception as exc:  # noqa: BLE001
                log.debug("query trim failed (%s), using raw query", exc)
            tokens = q.split()
            if len(tokens) > _MAX_SEARCH_TOKENS:
                q = " ".join(tokens[:_MAX_SEARCH_TOKENS])
        if domain_filter:
            sites = " OR ".join(f"site:{d}" for d in domain_filter)
            q = f"{q} ({sites})"
        return q

    async def _scrape_and_merge(
        self, app: Any, item: GroundedItem, sem: asyncio.Semaphore
    ) -> GroundedItem:
        async with sem:
            try:
                result = await scrape.fetch_page(
                    item.url, firecrawl_app=app, on_credits_exhausted="continue"
                )
            except Exception as e:  # noqa: BLE001
                log.warning("scrape failed for %s: %s", item.url, e)
                return _apply_tier(item)

        if result.status != "OK" or not result.markdown:
            log.warning("scrape failed for %s: %s", item.url, result.detail)
            return _apply_tier(item)

        markdown = result.markdown
        metadata = result.metadata or {}
        paywalled = _looks_paywalled(markdown)
        item.markdown = markdown
        item.scraped = True
        item.scraped_via = result.method
        item.published_date = _extract_date(metadata) or item.published_date
        title = metadata.get("title")
        if title:
            item.title = title
        return _apply_tier(item, paywalled=paywalled)

    async def _search(
        self,
        query: str,
        *,
        limit: int = 10,
        recency: Recency = "month",
        domain_filter: list[str] | None = None,
        scrape_top_k: int = 5,
        stats: dict[str, int] | None = None,
    ) -> list[GroundedItem]:
        """Discover, drop BLOCKED hosts, scrape top-K, tier-tag every result.

        ``stats`` (optional output dict) is populated with ``blocked_dropped`` —
        the count of BLOCKED-tier results dropped before scraping — without
        changing the return type, so callers can surface it.
        """
        q = self._trim_query(query, domain_filter)
        try:
            raw_data = await self._discover(
                q, limit=limit, recency=recency, domain_filter=domain_filter
            )
        except Exception as e:  # noqa: BLE001
            log.warning("discovery failed: %s", e)
            if stats is not None:
                stats["blocked_dropped"] = 0
            return []

        items = [_build_initial_item(r) for r in raw_data if _as_dict(r).get("url")]

        # BLOCKED domains are hard-excluded — drop them before the scrape step so
        # we never spend a Firecrawl call on them and they never surface downstream.
        blocked_dropped = 0
        kept: list[GroundedItem] = []
        for it in items:
            if authority.classify_domain(it.url)[0] == "BLOCKED":
                blocked_dropped += 1
            else:
                kept.append(it)
        if blocked_dropped:
            log.info("dropped %d blocked-source results before scraping", blocked_dropped)
        items = kept
        if stats is not None:
            stats["blocked_dropped"] = blocked_dropped

        if not items:
            return []

        scrape_targets = items[:scrape_top_k]
        keep_as_snippet = items[scrape_top_k:]

        scraped: list[GroundedItem] = []
        if scrape_targets:
            app = self._firecrawl_app()
            sem = asyncio.Semaphore(_SCRAPE_CONCURRENCY)
            scraped = list(
                await asyncio.gather(
                    *(self._scrape_and_merge(app, it, sem) for it in scrape_targets),
                    return_exceptions=False,
                )
            )
        for it in keep_as_snippet:
            _apply_tier(it)

        return scraped + keep_as_snippet


__all__ = ["FirecrawlGrounder", "GroundedItem"]
