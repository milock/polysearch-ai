"""Unit tests for cross-source community fusion.

``fuse`` flattens per-source ``SourceResult`` lists, drops items older than the
window, dedupes by canonical URL (keeping the higher-engagement copy), and ranks
by per-source engagement z-score blended with recency decay.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polysearch.community.fusion import fuse
from polysearch.output.schema import SourceResult


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _sr(
    url: str,
    *,
    layer: str = "reddit",
    engagement: int | None = 0,
    days_ago: float = 1.0,
    title: str = "t",
) -> SourceResult:
    return SourceResult(
        url=url,
        title=title,
        snippet="s",
        tier="COMMUNITY",
        published_date=_iso(days_ago),
        layer=layer,
        engagement=engagement,
    )


def test_fuse_dedupes_cross_posted_url_keeping_higher_engagement() -> None:
    a = _sr("https://reddit.com/r/x/comments/1/post/", layer="reddit", engagement=10)
    # Same post via the old.reddit mirror + a tracking param — must canonicalize
    # to the same key and dedupe.
    b = _sr(
        "https://old.reddit.com/r/x/comments/1/post/?utm_source=share",
        layer="reddit",
        engagement=99,
    )
    fused = fuse([[a], [b]], limit=10)
    assert len(fused) == 1
    assert fused[0].engagement == 99


def test_fuse_ranks_higher_engagement_first_within_source() -> None:
    low = _sr("https://ex.test/low", layer="hackernews", engagement=1, days_ago=1)
    high = _sr("https://ex.test/high", layer="hackernews", engagement=500, days_ago=1)
    fused = fuse([[low, high]], limit=10)
    assert [r.url for r in fused] == ["https://ex.test/high", "https://ex.test/low"]


def test_fuse_drops_items_older_than_window() -> None:
    fresh = _sr("https://ex.test/fresh", days_ago=2)
    stale = _sr("https://ex.test/stale", days_ago=400)
    fused = fuse([[fresh, stale]], limit=10, window_days=30)
    assert [r.url for r in fused] == ["https://ex.test/fresh"]


def test_fuse_keeps_undated_items() -> None:
    dated = _sr("https://ex.test/dated", days_ago=2)
    undated = SourceResult(
        url="https://ex.test/undated",
        title="t",
        snippet="s",
        tier="COMMUNITY",
        published_date=None,
        layer="github",
        engagement=5,
    )
    fused = fuse([[dated], [undated]], limit=10, window_days=30)
    assert {r.url for r in fused} == {
        "https://ex.test/dated",
        "https://ex.test/undated",
    }


def test_fuse_normalizes_engagement_per_source() -> None:
    # A quiet GitHub repo (few stars) should still rank against a naturally
    # high-volume Reddit post because z-scores are computed within each source.
    gh = _sr("https://ex.test/gh", layer="github", engagement=40, days_ago=0.5)
    reddit_hi = _sr("https://ex.test/rh", layer="reddit", engagement=5000, days_ago=8)
    reddit_lo = _sr("https://ex.test/rl", layer="reddit", engagement=10, days_ago=8)
    fused = fuse([[gh], [reddit_hi, reddit_lo]], limit=10)
    # gh is the sole item in its source (z=0) but very recent; it should not be
    # buried beneath the low-engagement, older reddit item.
    urls = [r.url for r in fused]
    assert urls.index("https://ex.test/gh") < urls.index("https://ex.test/rl")


def test_fuse_truncates_to_limit() -> None:
    items = [_sr(f"https://ex.test/{i}", engagement=i) for i in range(10)]
    fused = fuse([items], limit=3)
    assert len(fused) == 3


def test_fuse_empty_input_returns_empty() -> None:
    assert fuse([], limit=10) == []
    assert fuse([[], []], limit=10) == []
