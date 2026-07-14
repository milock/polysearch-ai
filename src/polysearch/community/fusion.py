"""Cross-source fusion for the native community adapters.

URL-canonical dedupe + recency-window drop + engagement-z-score / recency-decay
ranking over the ``SourceResult`` items produced by ``community/adapters.py``.
Per-source normalization keys off each result's ``layer`` (the source slug —
``reddit``, ``hackernews``, …), so a naturally high-volume platform can't
dominate the ranking purely on scale.

Adapted from the last30days community-signal library
(``dedupe`` / ``fusion`` / ``dates``, MIT) — simplified to the six native
adapters' flat ``SourceResult`` shape. See ATTRIBUTION.md for full credit.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from polysearch.output.schema import SourceResult

# utm_* is handled by prefix; these are additional common tracking params.
_TRACKING_PARAMS = {"ref", "ref_source", "ref_src", "share_id", "context", "igshid", "fbclid"}
# old./np.reddit.com are read-only/no-participation mirrors of www.reddit.com;
# m. is the generic mobile subdomain. Unifying these is what lets a post reached
# via a different mirror dedupe against itself.
_STRIP_SUBDOMAINS = ("www.", "old.", "np.", "m.")

_RECENCY_WEIGHT = 0.4
_ENGAGEMENT_WEIGHT = 0.6


def _canonical_url(url: str) -> str:
    """Normalize a URL for cross-source dedupe: lowercase host, strip a leading
    www/old/np/m subdomain, drop tracking query params, and strip a trailing
    slash from the path."""
    parsed = urlparse(url.strip().lower())
    netloc = parsed.netloc
    for prefix in _STRIP_SUBDOMAINS:
        if netloc.startswith(prefix):
            netloc = netloc[len(prefix):]
            break
    params = parse_qs(parsed.query)
    clean_params = {
        k: v
        for k, v in sorted(params.items())
        if k not in _TRACKING_PARAMS and not k.startswith("utm_")
    }
    query = urlencode(clean_params, doseq=True)
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, netloc, path, "", query, ""))


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _drop_stale(items: list[SourceResult], window_days: int) -> list[SourceResult]:
    """Drop items definitively older than the window. Undated items are kept —
    an unknown date isn't evidence of staleness."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    kept: list[SourceResult] = []
    for item in items:
        dt = _parse_date(item.published_date)
        if dt is not None and dt < cutoff:
            continue
        kept.append(item)
    return kept


def _dedupe(items: list[SourceResult]) -> list[SourceResult]:
    """Keep one item per canonical URL, preferring the higher-engagement copy."""
    best: dict[str, SourceResult] = {}
    for item in items:
        if not item.url:
            continue
        key = _canonical_url(item.url)
        existing = best.get(key)
        if existing is None:
            best[key] = item
            continue
        if (item.engagement or 0) > (existing.engagement or 0):
            best[key] = item
    return list(best.values())


def _recency_score(item: SourceResult, window_days: int) -> float:
    dt = _parse_date(item.published_date)
    if dt is None:
        return 0.0
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
    return max(0.0, 1.0 - (age_days / max(window_days, 1)))


def _engagement_zscores(items: list[SourceResult]) -> dict[int, float]:
    """z-score engagement within each source (keyed by ``id(item)``).

    Per-source normalization keeps a naturally high-engagement platform (Reddit
    upvotes vs. a quiet GitHub repo's star count) from dominating on scale alone.
    """
    by_source: dict[str, list[SourceResult]] = {}
    for item in items:
        by_source.setdefault(item.layer or "unknown", []).append(item)

    z_by_id: dict[int, float] = {}
    for source_items in by_source.values():
        values = [float(it.engagement or 0) for it in source_items]
        mean = statistics.fmean(values) if values else 0.0
        stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
        for it, val in zip(source_items, values):
            z_by_id[id(it)] = (val - mean) / stdev if stdev > 0 else 0.0
    return z_by_id


def fuse(
    results: list[list[SourceResult]],
    *,
    limit: int,
    window_days: int = 30,
) -> list[SourceResult]:
    """Fuse per-source result lists into one ranked, deduped list.

    1. Flatten, then drop items definitively older than ``window_days`` (undated
       items are kept).
    2. Dedupe cross-source by canonical URL, keeping the higher-engagement copy.
    3. Rank by engagement z-score (computed per source) blended with recency
       decay, so no single source dominates purely on posting volume.
    4. Truncate to ``limit``.
    """
    flattened = [item for sub in results for item in sub]
    flattened = _drop_stale(flattened, window_days)
    deduped = _dedupe(flattened)

    z_scores = _engagement_zscores(deduped)
    scored = [
        (
            _ENGAGEMENT_WEIGHT * z_scores.get(id(it), 0.0)
            + _RECENCY_WEIGHT * _recency_score(it, window_days),
            it,
        )
        for it in deduped
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [it for _, it in scored[:limit]]


__all__ = ["fuse"]
