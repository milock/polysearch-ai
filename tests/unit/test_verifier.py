"""Unit tests for polysearch.verification.verifier.

Covers the citation verifier. Two notes on the test doubles:

1. Scraping routes through ``polysearch.sources.scrape`` (Task 7's modern
   Firecrawl → httpx → Playwright chain), so the fake Firecrawl app implements
   the modern ``AsyncFirecrawl.scrape(url, formats=[...])`` method (not the legacy
   ``scrape_url``). A 404 raised/returned by Firecrawl is origin-confirmed dead
   and short-circuits before httpx, keeping these tests hermetic without a
   network stub.
2. The public ``VerificationResult`` is the 4-field schema model
   (claim_id/url/status/detail); the rich per-source scoring lives on the
   module-private ``_ScoredResult`` and is folded into ``detail``. Assertions
   that inspected rich fields now check ``status`` and ``detail`` substrings.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest

from polysearch.verification.verifier import Claim, verify


# -----------------------------------------------------------------------------
# Fake Firecrawl app (modern .scrape shape)
# -----------------------------------------------------------------------------


class _FakeApp:
    """Minimal modern ``AsyncFirecrawl`` stand-in. Drives responses from a dict
    keyed by URL. A value shaped ``{"__fail__": "msg"}`` raises. An unconfigured
    URL comes back as an origin-confirmed 404."""

    def __init__(
        self,
        api_key: str,
        responses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.api_key = api_key
        self._responses = responses or {}
        self.scrape_calls: list[str] = []

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.scrape_calls.append(url)
        resp = self._responses.get(url)
        if resp is None:
            return {
                "success": False,
                "markdown": "",
                "metadata": {"statusCode": 404},
                "error": "not found",
            }
        if "__fail__" in resp:
            raise RuntimeError(resp["__fail__"])
        return resp


def _install_fake_app(monkeypatch: pytest.MonkeyPatch, app: _FakeApp) -> None:
    """Patch ``firecrawl.AsyncFirecrawl`` inside the verifier's late import."""
    fake_mod = types.ModuleType("firecrawl")

    def _factory(api_key: str) -> _FakeApp:
        app.api_key = api_key
        return app

    fake_mod.AsyncFirecrawl = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "firecrawl", fake_mod)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
    # Absence of OPENAI_API_KEY keeps the embedding fallback off by default.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


async def test_url_dead_404(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/missing": {
                "success": False,
                "markdown": "",
                "metadata": {"statusCode": 404},
                "error": "Not Found",
            }
        },
    )
    _install_fake_app(monkeypatch, app)

    claim = Claim(
        text="A claim", source_urls=["https://cms.gov/missing"], claim_id="c1"
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.total_citations == 1
    assert report.broken == 1
    r = report.results[0]
    assert r.status == "URL_DEAD"
    assert "404" in (r.detail or "")


async def test_quote_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/a": {
                "success": True,
                "markdown": "The document discusses budget policy in detail.",
                "metadata": {"published_date": "2026-01-15"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(
        text="...",
        source_urls=["https://cms.gov/a"],
        quotes=["nothing like this appears on the page whatsoever"],
        claim_id="c2",
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.quote_mismatches == 1
    r = report.results[0]
    assert r.status == "QUOTE_NOT_FOUND"


async def test_quote_fuzzy_match_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    content = (
        "The annual report establishes new budget figures for 2026. "
        "Regional offices will see modest changes to their allocations."
    )
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/b": {
                "success": True,
                "markdown": content,
                "metadata": {"published_date": "2026-02-01"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)

    # exact
    claim_exact = Claim(
        text="...",
        source_urls=["https://cms.gov/b"],
        quotes=["new budget figures for 2026"],
        claim_id="c-exact",
    )
    report = await verify([claim_exact], budget_usd=1.0)
    assert report.results[0].status == "OK"

    # slight paraphrase (small edit distance: singular vs plural)
    claim_paraphrase = Claim(
        text="...",
        source_urls=["https://cms.gov/b"],
        quotes=["new budget figure for 2026"],
        claim_id="c-para",
    )
    report2 = await verify([claim_paraphrase], budget_usd=1.0)
    assert report2.results[0].status == "OK"

    # partial substring match
    claim_partial = Claim(
        text="...",
        source_urls=["https://cms.gov/b"],
        quotes=["modest changes to their allocations"],
        claim_id="c-partial",
    )
    report3 = await verify([claim_partial], budget_usd=1.0)
    assert report3.results[0].status == "OK"


async def test_number_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/nums": {
                "success": True,
                "markdown": "The rate decreased by 42% compared to last year.",
                "metadata": {"published_date": "2026-03-01"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(
        text="...",
        source_urls=["https://cms.gov/nums"],
        numbers=["45%"],
        claim_id="cnum",
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.number_mismatches == 1
    assert report.results[0].status == "NUMBER_MISMATCH"


async def test_number_match_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/nums2": {
                "success": True,
                "markdown": "Revenue hit $1.2M in Q1, up 45% YoY.",
                "metadata": {"published_date": "2026-03-05"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(
        text="...",
        source_urls=["https://cms.gov/nums2"],
        numbers=["45%", "$1.2M"],
        claim_id="cnum2",
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.results[0].status == "OK"


async def test_paywalled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://wsj.com/x": {
                "success": True,
                "markdown": "Please subscribe to continue reading this story.",
                "metadata": {"published_date": "2026-01-10"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(text="...", source_urls=["https://wsj.com/x"], claim_id="cpw")
    report = await verify([claim], budget_usd=1.0)
    assert report.results[0].status == "PAYWALLED"


async def test_undated_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FakeApp(
        api_key="x",
        responses={
            "https://cms.gov/undated": {
                "success": True,
                "markdown": "Some published content with no publication date.",
                "metadata": {},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(
        text="...", source_urls=["https://cms.gov/undated"], claim_id="cu"
    )
    report = await verify([claim], budget_usd=1.0)
    r = report.results[0]
    assert r.status == "UNDATED"
    # cms.gov is HIGH -> undated -> MEDIUM; the downgrade is folded into detail.
    assert "HIGH" in (r.detail or "")
    assert "MEDIUM" in (r.detail or "")


async def test_budget_exhaustion_marks_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls = [f"https://cms.gov/page-{i}" for i in range(5)]
    app = _FakeApp(
        api_key="x",
        responses={
            u: {
                "success": True,
                "markdown": "content",
                "metadata": {"published_date": "2026-01-01"},
            }
            for u in urls
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(text="...", source_urls=urls, claim_id="cbudget")
    # budget for 2 scrapes only (0.01 / 0.005 = 2)
    report = await verify(
        [claim],
        budget_usd=0.01,
        firecrawl_cost_per_scrape=0.005,
    )
    assert report.total_citations == 5
    assert report.skipped_budget == 3
    ok_count = sum(1 for r in report.results if r.status == "OK")
    skipped_count = sum(1 for r in report.results if r.status == "SKIPPED_BUDGET")
    assert ok_count == 2
    assert skipped_count == 3
    # Only 2 actual scrape calls happened.
    assert len(app.scrape_calls) == 2


async def test_priority_tier_verifies_high_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high_url = "https://cms.gov/important"
    community_url = "https://reddit.com/r/test/post"
    unknown_url = "https://randomblog.example.org/post"
    app = _FakeApp(
        api_key="x",
        responses={
            high_url: {
                "success": True,
                "markdown": "HIGH content",
                "metadata": {"published_date": "2026-01-01"},
            },
            community_url: {
                "success": True,
                "markdown": "community content",
                "metadata": {"published_date": "2026-01-01"},
            },
            unknown_url: {
                "success": True,
                "markdown": "blog content",
                "metadata": {"published_date": "2026-01-01"},
            },
        },
    )
    _install_fake_app(monkeypatch, app)

    # Claim lists UNKNOWN first, then COMMUNITY, then HIGH. "tier" priority must
    # flip to HIGH, COMMUNITY, UNKNOWN so a 1-scrape budget lands on HIGH.
    claim = Claim(
        text="...",
        source_urls=[unknown_url, community_url, high_url],
        claim_id="corder",
    )
    report = await verify(
        [claim],
        priority="tier",
        budget_usd=0.005,
        firecrawl_cost_per_scrape=0.005,
    )
    verified = [r for r in report.results if r.status != "SKIPPED_BUDGET"]
    assert len(verified) == 1
    assert verified[0].url == high_url


async def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    claim = Claim(text="...", source_urls=["https://cms.gov/a"], claim_id="c")
    with pytest.raises(RuntimeError, match="FIRECRAWL_API_KEY"):
        await verify([claim])


async def test_status_precedence_paywalled_wins_over_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Page is paywalled AND quote is absent -> status should be PAYWALLED.
    app = _FakeApp(
        api_key="x",
        responses={
            "https://wsj.com/a": {
                "success": True,
                "markdown": "Subscribe to continue reading this article.",
                "metadata": {"published_date": "2026-02-01"},
            }
        },
    )
    _install_fake_app(monkeypatch, app)
    claim = Claim(
        text="...",
        source_urls=["https://wsj.com/a"],
        quotes=["some missing quote"],
        claim_id="cprec",
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.results[0].status == "PAYWALLED"


def test_citation_verifier_defaults_raised() -> None:
    """Module-level ``verify`` defaults must match the standard-depth budget so
    direct callers (tests, ad-hoc scripts) get the right behavior."""
    sig = inspect.signature(verify)
    assert sig.parameters["budget_usd"].default == 3.00
    assert sig.parameters["max_concurrency"].default == 8


# -----------------------------------------------------------------------------
# BLOCKED tier (hard-exclude)
# -----------------------------------------------------------------------------


async def test_blocked_domain_never_scraped_gets_blocked_source_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A citation to a BLOCKED-tier domain must get a distinct BLOCKED_SOURCE
    status without ever being scraped — never URL_DEAD, never verified OK."""
    from polysearch.sources import authority

    monkeypatch.setitem(
        authority._DOMAIN_TO_TIER, "blocked-example.test", ("BLOCKED", "test block")
    )
    app = _FakeApp(api_key="x", responses={})
    _install_fake_app(monkeypatch, app)

    claim = Claim(
        text="...", source_urls=["https://blocked-example.test/a"], claim_id="cblocked"
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.total_citations == 1
    assert report.blocked_sources == 1
    r = report.results[0]
    assert r.status == "BLOCKED_SOURCE"
    # Never actually scraped — no Firecrawl call spent on a hard-excluded domain.
    assert app.scrape_calls == []


async def test_blocked_source_never_counts_as_claim_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polysearch.sources import authority

    monkeypatch.setitem(
        authority._DOMAIN_TO_TIER, "blocked-example2.test", ("BLOCKED", "test block")
    )
    app = _FakeApp(api_key="x", responses={})
    _install_fake_app(monkeypatch, app)

    claim = Claim(
        text="...",
        source_urls=["https://blocked-example2.test/a"],
        claim_id="cblocked2",
    )
    report = await verify([claim], budget_usd=1.0)
    assert report.claims_total == 1
    assert report.claims_supported == 0


def test_aggregate_claim_level_support() -> None:
    """A claim is 'supported' when each number/quote matches in >=1 cited source,
    even if other cited sources (sharing the same multi-URL citation list) don't
    contain it. Guards against the per-pair undercount that made the headline
    metric near-zero."""
    from polysearch.verification.verifier import _ScoredResult, _aggregate

    results = [
        # Claim A: "5%" found on url1, missing on url2 -> claim supported.
        _ScoredResult(
            url="https://a.com/1", claim_id="A", status="OK",
            number_matches=[{"number": "5%", "matched": True, "context": "x"}],
        ),
        _ScoredResult(
            url="https://a.com/2", claim_id="A", status="NUMBER_MISMATCH",
            number_matches=[{"number": "5%", "matched": False, "context": ""}],
        ),
        # Claim B: "9%" never matched -> unsupported.
        _ScoredResult(
            url="https://b.com/1", claim_id="B", status="NUMBER_MISMATCH",
            number_matches=[{"number": "9%", "matched": False, "context": ""}],
        ),
        # Claim C: no figures, source alive but undated -> supported.
        _ScoredResult(url="https://c.com/1", claim_id="C", status="UNDATED"),
        # Claim D: only source is dead -> unsupported.
        _ScoredResult(url="https://d.com/1", claim_id="D", status="URL_DEAD"),
    ]
    report = _aggregate(results, total_cost=0.0, total_duration_ms=0)
    assert report.claims_total == 4
    assert report.claims_supported == 2  # A and C
    assert report.total_citations == 5
    assert report.verified_ok == 1  # only the single OK pair


# -----------------------------------------------------------------------------
# Firecrawl 402 / credits-exhausted short-circuit
# -----------------------------------------------------------------------------


class _ExhaustedApp:
    """Modern ``AsyncFirecrawl`` stand-in that always raises 402 / credits
    exhausted."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.scrape_calls: list[str] = []

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.scrape_calls.append(url)
        raise RuntimeError("HTTP 402: Insufficient credits to perform this request")


def _install_exhausted_app(
    monkeypatch: pytest.MonkeyPatch, app: _ExhaustedApp
) -> None:
    fake_mod = types.ModuleType("firecrawl")

    def _factory(api_key: str) -> _ExhaustedApp:
        app.api_key = api_key
        return app

    fake_mod.AsyncFirecrawl = _factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "firecrawl", fake_mod)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _install_httpx_stub(
    monkeypatch: pytest.MonkeyPatch, *, succeed_urls: set[str]
) -> list[str]:
    """Stub the httpx tier so the post-latch fallback is hermetic and
    deterministic: URLs in ``succeed_urls`` resolve, everything else comes back
    blocked (500). Returns the list of URLs httpx was actually called for."""
    from polysearch.sources import scrape

    calls: list[str] = []

    async def _fake_fetch_via_httpx(
        url: str, timeout_s: float
    ) -> tuple[str | None, int | None, str | None]:
        calls.append(url)
        if url in succeed_urls:
            return f"Real content dated 2026-01-01 for {url}", 200, None
        return None, 500, "http 500"

    monkeypatch.setattr(scrape, "_fetch_via_httpx", _fake_fetch_via_httpx)
    return calls


async def test_402_short_circuits_firecrawl_but_falls_through_to_httpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 402 from Firecrawl should stop further Firecrawl dispatch quickly (not 5
    claims x 3 retries), but must NOT discard the remaining citations wholesale —
    each still gets the free httpx tier attempt."""
    app = _ExhaustedApp(api_key="x")
    _install_exhausted_app(monkeypatch, app)

    urls = [f"https://example.com/{i}" for i in range(5)]
    succeed_urls = {urls[0], urls[2], urls[4]}
    httpx_calls = _install_httpx_stub(monkeypatch, succeed_urls=succeed_urls)

    claims = [
        Claim(
            text=f"claim {i}",
            source_urls=[urls[i]],
            quotes=[],
            numbers=[],
            claim_id=f"id{i}",
        )
        for i in range(5)
    ]

    report = await verify(claims, budget_usd=10.0, max_concurrency=1)

    # Exactly one real Firecrawl attempt after the 402 latch.
    assert len(app.scrape_calls) == 1

    # Every citation still got an httpx attempt — none skipped wholesale.
    assert set(httpx_calls) == set(urls)
    assert report.skipped_budget == 0

    by_url = {r.url: r for r in report.results}
    for url in succeed_urls:
        assert by_url[url].status == "OK", f"{url} should have resolved via httpx"
        assert "httpx" in (by_url[url].detail or "")
    for url in set(urls) - succeed_urls:
        assert by_url[url].status == "FETCH_BLOCKED"

    # The report-level flag is the source of truth for the outage.
    assert report.credits_exhausted_hit is True


async def test_credits_exhausted_hit_false_on_clean_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: a run with no 402 never sets the flag."""

    class _CleanApp:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "success": True,
                "markdown": "Content dated 2026-01-01.",
                "metadata": {"published_date": "2026-01-01"},
            }

    fake_mod = types.ModuleType("firecrawl")
    fake_mod.AsyncFirecrawl = lambda api_key: _CleanApp(api_key)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "firecrawl", fake_mod)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    claim = Claim(text="claim", source_urls=["https://cms.gov/a"], claim_id="c1")
    report = await verify([claim], budget_usd=1.0)

    assert report.credits_exhausted_hit is False


# -----------------------------------------------------------------------------
# Gated live integration test
# -----------------------------------------------------------------------------


@pytest.mark.live
async def test_live_verify_homepage() -> None:
    claim = Claim(
        text="CMS homepage exists",
        source_urls=["https://www.cms.gov/"],
        claim_id="live-1",
    )
    report = await verify([claim], budget_usd=0.05)
    assert report.total_citations == 1
    # The homepage should resolve — OK or UNDATED, never URL_DEAD.
    assert report.results[0].status != "URL_DEAD"
