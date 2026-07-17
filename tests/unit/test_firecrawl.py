"""Unit tests for polysearch.providers.firecrawl — the web grounder.

Covers the scrape+tier pipeline (WebItem/GroundedItem shape, authority tagging,
undated + paywall downgrades, snippet-only-beyond-top-k, BLOCKED-source drop)
and the pluggable discovery dispatch (Perplexity default; Brave when selected +
keyed).
"""

from __future__ import annotations

import types
from typing import Any

import httpx
import pytest
import respx

from polysearch.config import Settings
from polysearch.output.schema import SourceResult
from polysearch.providers import firecrawl as fc
from polysearch.providers import perplexity
from polysearch.sources import authority


# ---------- helpers -------------------------------------------------------


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "firecrawl_api_key": "fc-test-key",
        "perplexity_api_key": "pplx-test-key",
    }
    base.update(over)
    return Settings(**base)


class _FakeFirecrawlApp:
    """Modern ``AsyncFirecrawl`` stand-in. ``.scrape`` returns a canned response
    dict (normalized by the scrape chain); ``.search`` returns canned web
    results."""

    def __init__(
        self,
        api_key: str,
        scrape_data_by_url: dict[str, dict[str, Any]] | None = None,
        scrape_fail_urls: set[str] | None = None,
        search_web: list[dict[str, Any]] | None = None,
    ) -> None:
        self.api_key = api_key
        self._scrapes = scrape_data_by_url or {}
        self._fail = scrape_fail_urls or set()
        self._search_web = search_web or []
        self.scrape_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.scrape_calls.append(url)
        if url in self._fail:
            raise RuntimeError(f"boom: {url}")
        return self._scrapes.get(url, {"markdown": "", "metadata": {}})

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append({"query": query, **kwargs})
        return {"web": list(self._search_web)}


def _install_fake_firecrawl(monkeypatch: pytest.MonkeyPatch, app: _FakeFirecrawlApp) -> None:
    """Inject a fake ``firecrawl`` module so ``_firecrawl_app`` returns ``app``."""
    fake_mod = types.ModuleType("firecrawl")

    def _factory(api_key: str) -> _FakeFirecrawlApp:
        app.api_key = api_key
        return app

    fake_mod.AsyncFirecrawl = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "firecrawl", fake_mod)


def _patch_perplexity_discovery(
    monkeypatch: pytest.MonkeyPatch, raw: list[dict[str, Any]], *, calls: list[str] | None = None
) -> None:
    """Patch the default (Perplexity) discovery backend to return canned results
    shaped as ``SourceResult`` objects."""

    async def _fake_search(
        query: str, *, limit: int = 10, recency: str | None = None, api_key: str | None = None
    ):
        if calls is not None:
            calls.append(query)
        return [
            SourceResult(
                url=r["url"],
                title=r.get("title") or r["url"],
                snippet=r.get("description") or "",
                tier="UNKNOWN",
                published_date=r.get("page_age"),
                layer="grounding",
                engagement=None,
            )
            for r in raw
        ]

    monkeypatch.setattr(perplexity, "search", _fake_search)


# ---------- scrape + tier pipeline ----------------------------------------


async def test_grounded_item_shape_and_tier_tagging(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch,
        [
            {"url": "https://www.cms.gov/medicare/fee-schedule", "title": "MPFS 2026", "description": "Snippet text"},
            {"url": "https://randomblog.example.org/post", "title": "Random", "description": "blah"},
        ],
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/medicare/fee-schedule": {
                "markdown": "# MPFS\n\nReal content.",
                "metadata": {"published_date": "2026-01-15", "title": "MPFS 2026 final"},
            },
            "https://randomblog.example.org/post": {"markdown": "body", "metadata": {}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("mpfs 2026", limit=2, scrape_top_k=2)

    assert len(items) == 2
    cms = next(i for i in items if "cms.gov" in i.url)
    rnd = next(i for i in items if "randomblog" in i.url)

    assert cms.scraped is True
    assert cms.tier == "HIGH"
    assert cms.published_date == "2026-01-15"
    assert cms.domain == "cms.gov"
    assert cms.markdown.startswith("# MPFS")
    assert cms.title == "MPFS 2026 final"

    assert rnd.tier == "UNKNOWN"
    assert rnd.scraped is True
    assert rnd.domain == "randomblog.example.org"


async def test_modern_published_time_metadata_dates_the_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scrape metadata using the modern firecrawl v2 ``published_time`` key dates
    the item (proving the modernized date mapping), keeping HIGH at HIGH."""
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://cms.gov/x", "title": "t", "description": "s"}]
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://cms.gov/x": {
                "markdown": "body",
                "metadata": {"published_time": "2026-03-01T12:00:00Z", "status_code": 200},
            },
        },
    )
    _install_fake_firecrawl(monkeypatch, app)
    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=1, scrape_top_k=1)
    assert items[0].published_date == "2026-03-01"
    assert items[0].tier == "HIGH"


async def test_undated_high_downgrades_to_medium(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://cms.gov/x", "title": "t", "description": "s"}]
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={"https://cms.gov/x": {"markdown": "body", "metadata": {}}},
    )
    _install_fake_firecrawl(monkeypatch, app)
    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=1, scrape_top_k=1)
    assert items[0].tier == "MEDIUM"
    assert "undated" in items[0].tier_reason


async def test_paywall_detection_downgrades(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://wsj.com/story", "title": "t", "description": "s"}]
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://wsj.com/story": {
                "markdown": "Please subscribe to continue reading this article.",
                "metadata": {"published_date": "2026-01-10"},
            },
        },
    )
    _install_fake_firecrawl(monkeypatch, app)
    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=1, scrape_top_k=1)
    # wsj is MEDIUM; paywall drops one step -> LOW
    assert items[0].tier == "LOW"
    assert "paywall" in items[0].tier_reason.lower()


async def test_scrape_failure_keeps_snippet(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://cms.gov/a", "title": "T", "description": "snippet here"}]
    )
    app = _FakeFirecrawlApp(api_key="x", scrape_fail_urls={"https://cms.gov/a"})
    _install_fake_firecrawl(monkeypatch, app)

    # Firecrawl fails here, so the scrape chain would otherwise try a real httpx
    # request. This test only cares about "nothing worked, keep the snippet", so
    # stub the httpx tier to fail deterministically.
    async def _httpx_fails(url: str, timeout_s: float) -> tuple[None, None, str]:
        return None, None, "mocked: no network in test"

    from polysearch.sources import scrape

    monkeypatch.setattr(scrape, "_fetch_via_httpx", _httpx_fails)

    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=1, scrape_top_k=1)
    assert len(items) == 1
    assert items[0].scraped is False
    assert items[0].snippet == "snippet here"
    assert items[0].tier in {"HIGH", "MEDIUM"}


async def test_snippet_only_beyond_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch,
        [{"url": f"https://cms.gov/{i}", "title": f"t{i}", "description": f"s{i}"} for i in range(4)],
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            f"https://cms.gov/{i}": {"markdown": f"body {i}", "metadata": {"published_date": "2026-02-01"}}
            for i in range(4)
        },
    )
    _install_fake_firecrawl(monkeypatch, app)
    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=4, scrape_top_k=2)
    assert len(items) == 4
    scraped = [i for i in items if i.scraped]
    snippet_only = [i for i in items if not i.scraped]
    assert len(scraped) == 2
    assert len(snippet_only) == 2
    assert all(i.tier == "HIGH" for i in scraped)
    # Only the top-K were actually fetched.
    assert len(app.scrape_calls) == 2


async def test_domain_filter_injects_site_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _patch_perplexity_discovery(monkeypatch, [], calls=calls)
    grounder = fc.FirecrawlGrounder(_settings())
    await grounder._search("downcoding", limit=5, domain_filter=["cms.gov", "oig.hhs.gov"])
    assert calls
    q = calls[0]
    assert "site:cms.gov" in q
    assert "site:oig.hhs.gov" in q


async def test_long_query_trimmed_to_core_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    """A verbose >12-token query is trimmed to its core subject before discovery."""
    calls: list[str] = []
    _patch_perplexity_discovery(monkeypatch, [], calls=calls)
    grounder = fc.FirecrawlGrounder(_settings())
    verbose = "what are the best latest news and updates about renewable energy storage battery technology for residential homes in 2026"
    await grounder._search(verbose, limit=5)
    assert calls
    assert len(calls[0].split()) <= 12


# ---------- BLOCKED-source drop -------------------------------------------


async def test_blocked_domain_dropped_before_scraping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BLOCKED-tier result must be dropped before the scrape step — never
    scraped, never returned — while a normal result alongside it comes through."""
    monkeypatch.setitem(
        authority._DOMAIN_TO_TIER, "blocked-grounding.test", ("BLOCKED", "test block")
    )
    _patch_perplexity_discovery(
        monkeypatch,
        [
            {"url": "https://blocked-grounding.test/a", "title": "t", "description": "s"},
            {"url": "https://www.cms.gov/x", "title": "t2", "description": "s2"},
        ],
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/x": {"markdown": "body", "metadata": {"published_date": "2026-01-01"}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    items = await grounder._search("q", limit=2, scrape_top_k=2)

    assert len(items) == 1
    assert items[0].domain == "cms.gov"
    assert app.scrape_calls == ["https://www.cms.gov/x"]


async def test_blocked_domain_drop_count_surfaced_via_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        authority._DOMAIN_TO_TIER, "blocked-grounding2.test", ("BLOCKED", "test block")
    )
    _patch_perplexity_discovery(
        monkeypatch,
        [
            {"url": "https://blocked-grounding2.test/a", "title": "t", "description": "s"},
            {"url": "https://www.cms.gov/y", "title": "t2", "description": "s2"},
        ],
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/y": {"markdown": "body", "metadata": {"published_date": "2026-01-01"}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    stats: dict[str, int] = {}
    await grounder._search("q", limit=2, scrape_top_k=2, stats=stats)

    assert stats == {"blocked_dropped": 1}


async def test_blocked_stats_zero_when_no_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://www.cms.gov/z", "title": "t", "description": "s"}]
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/z": {"markdown": "body", "metadata": {"published_date": "2026-01-01"}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    stats: dict[str, int] = {}
    await grounder._search("q", limit=1, scrape_top_k=1, stats=stats)

    assert stats == {"blocked_dropped": 0}


# ---------- ground() -> LayerOutput projection ----------------------------


async def test_ground_returns_layer_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://www.cms.gov/x", "title": "MPFS", "description": "s"}]
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/x": {"markdown": "# body", "metadata": {"published_date": "2026-01-01"}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    layer = await grounder.ground("mpfs", limit=1, scrape_top_k=1)

    assert layer.layer == "grounding"
    assert layer.error is None
    assert len(layer.results) == 1
    r = layer.results[0]
    assert isinstance(r, SourceResult)
    assert r.tier == "HIGH"
    assert r.layer == "grounding"
    assert r.published_date == "2026-01-01"


async def test_ground_backfills_snippet_from_markdown_when_discovery_snippet_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scraped result with no discovery snippet gets its SourceResult.snippet
    backfilled from the scraped markdown (whitespace collapsed); a result that
    already has a discovery snippet keeps it."""
    _patch_perplexity_discovery(
        monkeypatch,
        [
            {"url": "https://www.cms.gov/nosnip", "title": "No snippet", "description": ""},
            {"url": "https://www.cms.gov/hassnip", "title": "Has snippet", "description": "discovery snippet here"},
        ],
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/nosnip": {
                "markdown": "# Heading\n\nScraped body content   with   runs.",
                "metadata": {"published_date": "2026-01-01"},
            },
            "https://www.cms.gov/hassnip": {
                "markdown": "# Other\n\nDifferent scraped body.",
                "metadata": {"published_date": "2026-01-01"},
            },
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings())
    layer = await grounder.ground("cms", limit=2, scrape_top_k=2)

    by_url = {r.url: r for r in layer.results}
    backfilled = by_url["https://www.cms.gov/nosnip"]
    preserved = by_url["https://www.cms.gov/hassnip"]

    # Empty discovery snippet -> backfilled from markdown, whitespace collapsed.
    assert backfilled.snippet == "# Heading Scraped body content with runs."
    # Existing discovery snippet is preserved, not overwritten by markdown.
    assert preserved.snippet == "discovery snippet here"


async def test_ground_widens_backfill_for_authoritative_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HIGH/MEDIUM scraped pages with no discovery snippet backfill up to the
    configured wide budget (facts live beyond the 600-char base); lower tiers stay
    capped at the base."""
    _patch_perplexity_discovery(
        monkeypatch,
        [
            {"url": "https://www.federalreserve.gov/fomc", "title": "FOMC", "description": ""},
            {"url": "https://randomblog.example.org/post", "title": "Blog", "description": ""},
        ],
    )
    long_md = "word " * 400  # ~2000 chars
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.federalreserve.gov/fomc": {
                "markdown": long_md,
                "metadata": {"published_date": "2026-06-17"},
            },
            "https://randomblog.example.org/post": {
                "markdown": long_md,
                "metadata": {"published_date": "2026-06-17"},
            },
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    grounder = fc.FirecrawlGrounder(_settings(synthesis_excerpt_chars=1200))
    layer = await grounder.ground("rates", limit=2, scrape_top_k=2)
    by_url = {r.url: r for r in layer.results}

    # HIGH tier (federalreserve.gov) -> widened to 1200; UNKNOWN blog -> base 600.
    assert len(by_url["https://www.federalreserve.gov/fomc"].snippet) == 1200
    assert len(by_url["https://randomblog.example.org/post"].snippet) == 600


async def test_ground_surfaces_discovery_error_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(query: str, *, limit: int = 10, recency: str | None = None, api_key: str | None = None):
        raise RuntimeError("discovery exploded")

    # Route the failure through the search path but let _search swallow it to [].
    monkeypatch.setattr(perplexity, "search", _boom)
    grounder = fc.FirecrawlGrounder(_settings())
    layer = await grounder.ground("q", limit=2, scrape_top_k=1)
    # Discovery failure is caught inside _search -> empty results, no error.
    assert layer.layer == "grounding"
    assert layer.results == []


async def test_grounder_threads_recency_into_perplexity_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grounder's recency reaches ``perplexity.search`` as its ``recency`` arg
    (default backend time-filters discovery)."""
    seen: dict[str, Any] = {}

    async def _fake_search(
        query: str, *, limit: int = 10, recency: str | None = None, api_key: str | None = None
    ):
        seen["recency"] = recency
        return []

    monkeypatch.setattr(perplexity, "search", _fake_search)
    grounder = fc.FirecrawlGrounder(_settings())
    await grounder._search("q", limit=3, recency="week", scrape_top_k=0)
    assert seen["recency"] == "week"


# ---------- discovery backend switch --------------------------------------


async def test_discovery_defaults_to_perplexity(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, the default backend is Perplexity Search — its
    ``search`` is what gets called."""
    calls: list[str] = []
    _patch_perplexity_discovery(
        monkeypatch, [{"url": "https://www.cms.gov/x", "title": "t", "description": "s"}], calls=calls
    )
    app = _FakeFirecrawlApp(
        api_key="x",
        scrape_data_by_url={
            "https://www.cms.gov/x": {"markdown": "body", "metadata": {"published_date": "2026-01-01"}},
        },
    )
    _install_fake_firecrawl(monkeypatch, app)

    settings = _settings()
    assert settings.discovery_backend == "perplexity"
    grounder = fc.FirecrawlGrounder(settings)
    items = await grounder._search("medicare fee schedule", limit=1, scrape_top_k=1)

    assert calls == ["medicare fee schedule"]
    assert len(items) == 1


@respx.mock
async def test_discovery_brave_backend_hits_brave_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``discovery_backend='brave'`` and a key is set, discovery hits the
    Brave Search endpoint (not Perplexity)."""
    # Perplexity would raise if it were (wrongly) called.
    async def _pplx_should_not_run(query: str, *, limit: int = 10, recency: str | None = None, api_key: str | None = None):
        raise AssertionError("perplexity discovery must not run when backend='brave'")

    monkeypatch.setattr(perplexity, "search", _pplx_should_not_run)

    route = respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://www.cms.gov/mpfs",
                            "title": "MPFS",
                            "description": "fee schedule",
                            "page_age": "2026-01-15T00:00:00",
                        }
                    ]
                }
            },
        )
    )

    settings = _settings(discovery_backend="brave", brave_api_key="brave-test-key")
    grounder = fc.FirecrawlGrounder(settings)
    # scrape_top_k=0 keeps this test on the discovery leg only (no Firecrawl app).
    items = await grounder._search("mpfs 2026", limit=3, scrape_top_k=0)

    assert route.called
    # The Brave subscription-token header carried the configured key.
    sent = route.calls.last.request
    assert sent.headers["X-Subscription-Token"] == "brave-test-key"
    assert len(items) == 1
    assert items[0].domain == "cms.gov"
    assert items[0].published_date == "2026-01-15"


async def test_discovery_firecrawl_backend_uses_v2_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``discovery_backend='firecrawl'`` routes discovery through the Firecrawl
    app's ``.search`` (v2 /search)."""
    async def _pplx_should_not_run(query: str, *, limit: int = 10, recency: str | None = None, api_key: str | None = None):
        raise AssertionError("perplexity discovery must not run when backend='firecrawl'")

    monkeypatch.setattr(perplexity, "search", _pplx_should_not_run)

    app = _FakeFirecrawlApp(
        api_key="x",
        search_web=[{"url": "https://www.cms.gov/s", "title": "t", "description": "d"}],
    )
    _install_fake_firecrawl(monkeypatch, app)

    settings = _settings(discovery_backend="firecrawl")
    grounder = fc.FirecrawlGrounder(settings)
    items = await grounder._search("mpfs", limit=5, scrape_top_k=0)

    assert app.search_calls and app.search_calls[0]["query"] == "mpfs"
    assert len(items) == 1
    assert items[0].domain == "cms.gov"
