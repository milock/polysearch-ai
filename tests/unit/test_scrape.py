"""Unit tests for polysearch.sources.scrape — the Firecrawl -> httpx ->
Playwright fallback chain.

Motivating bug (see module docstring): Firecrawl 429s were causing citations to
be falsely marked URL_DEAD. These tests pin down (1) that each tier failing in
turn correctly advances the chain, (2) that Playwright is a strictly optional
tier that's skipped cleanly when not importable, (3) the FETCH_BLOCKED vs
URL_DEAD classification for each failure class, and (4) that the modern-SDK
port drives its error taxonomy off the raised exception's ``status_code`` (a
404 raised by ``.scrape`` still short-circuits as URL_DEAD).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest
import respx

from polysearch.sources import scrape
from polysearch.sources.scrape import FirecrawlCreditsExhausted, fetch_page

URL = "https://example.com/article"


class _FakeFirecrawlError(Exception):
    """Stand-in for a modern firecrawl-py typed error carrying ``.status_code``."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FirecrawlApp:
    """Minimal modern ``AsyncFirecrawl`` stand-in. ``.scrape`` returns a canned
    response dict (which the chain normalizes) or raises a canned exception."""

    def __init__(
        self, response: dict[str, Any] | None = None, raises: Exception | None = None
    ) -> None:
        self._response = response
        self._raises = raises
        self.calls: list[str] = []

    async def scrape(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(url)
        if self._raises is not None:
            raise self._raises
        return self._response or {}


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch, *, markdown: str, status: int = 200
) -> None:
    """Inject a fake ``playwright.async_api`` so ``_playwright_importable()``
    reports True and ``_fetch_via_playwright`` returns the canned page."""

    class _FakeResponse:
        def __init__(self, status: int) -> None:
            self.status = status

    class _FakePage:
        async def goto(self, url: str, timeout: float) -> _FakeResponse:
            return _FakeResponse(status)

        async def content(self) -> str:
            return f"<html><body>{markdown}</body></html>"

    class _FakeBrowser:
        async def new_page(self, user_agent: str) -> _FakePage:
            return _FakePage()

        async def close(self) -> None:
            return None

    class _FakeChromium:
        async def launch(self) -> _FakeBrowser:
            return _FakeBrowser()

    class _FakePlaywrightCtx:
        chromium = _FakeChromium()

        async def __aenter__(self) -> "_FakePlaywrightCtx":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    def _async_playwright() -> _FakePlaywrightCtx:
        return _FakePlaywrightCtx()

    fake_async_api = types.ModuleType("playwright.async_api")
    fake_async_api.async_playwright = _async_playwright  # type: ignore[attr-defined]
    fake_playwright = types.ModuleType("playwright")
    fake_playwright.async_api = fake_async_api  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "playwright", fake_playwright)
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_async_api)


def _ensure_playwright_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against test-pollution from another test's fake module injection."""
    monkeypatch.delitem(sys.modules, "playwright", raising=False)
    monkeypatch.delitem(sys.modules, "playwright.async_api", raising=False)


# -----------------------------------------------------------------------------
# Tier 1 success — no fallback needed
# -----------------------------------------------------------------------------


async def test_firecrawl_success_returns_ok_no_httpx_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FirecrawlApp(response={"success": True, "markdown": "# Real content", "metadata": {}})
    with respx.mock:
        # No route registered — any httpx call would raise inside the mock.
        result = await fetch_page(URL, firecrawl_app=app)
    assert result.status == "OK"
    assert result.method == "firecrawl"
    assert result.markdown == "# Real content"
    assert len(app.calls) == 1


async def test_firecrawl_document_metadata_maps_modern_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A modern v2 Document's snake_case metadata (``status_code``,
    ``published_time``) is normalized and surfaced through ``result.metadata``."""
    app = _FirecrawlApp(
        response={
            "markdown": "# Body",
            "metadata": {"status_code": 200, "published_time": "2026-01-15T09:00:00Z", "title": "T"},
        }
    )
    with respx.mock:
        result = await fetch_page(URL, firecrawl_app=app)
    assert result.status == "OK"
    assert result.method == "firecrawl"
    assert result.http_status == 200
    assert result.metadata is not None
    assert result.metadata.get("published_time") == "2026-01-15T09:00:00Z"


# -----------------------------------------------------------------------------
# Each tier failing in turn advances the chain
# -----------------------------------------------------------------------------


@respx.mock
async def test_firecrawl_fails_falls_to_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FirecrawlApp(raises=RuntimeError("boom"))
    respx.get(URL).mock(
        return_value=httpx.Response(200, text="<html><body>Fresh via httpx</body></html>")
    )
    _ensure_playwright_absent(monkeypatch)

    result = await fetch_page(URL, firecrawl_app=app)

    assert result.status == "OK"
    assert result.method == "httpx"
    assert "Fresh via httpx" in (result.markdown or "")
    assert len(app.calls) == 1


@respx.mock
async def test_firecrawl_and_httpx_fail_falls_to_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FirecrawlApp(raises=RuntimeError("boom"))
    respx.get(URL).mock(return_value=httpx.Response(503, text=""))
    _install_fake_playwright(monkeypatch, markdown="Rendered via headless Chromium")

    result = await fetch_page(URL, firecrawl_app=app)

    assert result.status == "OK"
    assert result.method == "playwright"
    assert "Rendered via headless Chromium" in (result.markdown or "")


@respx.mock
async def test_all_tiers_exhausted_no_firecrawl_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Firecrawl app supplied at all -> chain starts at httpx."""
    respx.get(URL).mock(return_value=httpx.Response(500, text=""))
    _ensure_playwright_absent(monkeypatch)

    result = await fetch_page(URL, firecrawl_app=None)

    assert result.markdown is None
    assert result.method is None
    assert result.status == "FETCH_BLOCKED"


# -----------------------------------------------------------------------------
# Playwright-absent path skips tier 3 cleanly
# -----------------------------------------------------------------------------


@respx.mock
async def test_playwright_absent_skips_tier3_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FirecrawlApp(raises=RuntimeError("boom"))
    respx.get(URL).mock(return_value=httpx.Response(500, text=""))
    _ensure_playwright_absent(monkeypatch)
    assert scrape._playwright_importable() is False

    result = await fetch_page(URL, firecrawl_app=app)

    assert result.status in ("FETCH_BLOCKED", "URL_DEAD")
    assert result.method is None
    assert result.markdown is None


# -----------------------------------------------------------------------------
# FETCH_BLOCKED vs URL_DEAD mapping per failure class
# -----------------------------------------------------------------------------


@respx.mock
async def test_403_maps_to_fetch_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(URL).mock(return_value=httpx.Response(403, text="Access Denied"))
    _ensure_playwright_absent(monkeypatch)
    result = await fetch_page(URL, firecrawl_app=None)
    assert result.status == "FETCH_BLOCKED"


@respx.mock
async def test_429_maps_to_fetch_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(URL).mock(return_value=httpx.Response(429, text="Too Many Requests"))
    _ensure_playwright_absent(monkeypatch)
    result = await fetch_page(URL, firecrawl_app=None)
    assert result.status == "FETCH_BLOCKED"


@respx.mock
async def test_200_status_challenge_page_advances_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cloudflare/Akamai-style challenge page served with HTTP 200 must not be
    accepted as real content — the chain should advance past it."""
    respx.get(URL).mock(
        return_value=httpx.Response(
            200,
            text="<html><body>Checking your browser before accessing this site. CAPTCHA required.</body></html>",
        )
    )
    _ensure_playwright_absent(monkeypatch)
    result = await fetch_page(URL, firecrawl_app=None)
    assert result.status == "FETCH_BLOCKED"
    assert result.method is None


@respx.mock
async def test_long_article_mentioning_captcha_is_not_a_false_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long, legitimate article that merely mentions 'captcha' deep in the body
    (not a real challenge page) must still be accepted as OK."""
    filler = "This is a real paragraph about web security practices. " * 100
    body = (
        f"<html><body>{filler} Somewhere deep in this article we discuss captcha "
        f"systems in passing. {filler}</body></html>"
    )
    assert len(body) > 4096
    respx.get(URL).mock(return_value=httpx.Response(200, text=body))
    _ensure_playwright_absent(monkeypatch)
    result = await fetch_page(URL, firecrawl_app=None)
    assert result.status == "OK"
    assert result.method == "httpx"


@respx.mock
async def test_dns_failure_maps_to_url_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    respx.get(URL).mock(
        side_effect=httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
    )
    _ensure_playwright_absent(monkeypatch)
    result = await fetch_page(URL, firecrawl_app=None)
    assert result.status == "URL_DEAD"


async def test_firecrawl_404_maps_to_url_dead_and_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A modern ``.scrape`` raises a typed 404 error (``.status_code == 404``);
    that is origin-confirmed dead, so the chain must NOT waste an httpx/Playwright
    call chasing it further."""
    app = _FirecrawlApp(raises=_FakeFirecrawlError("Not Found", status_code=404))
    with respx.mock:
        # No route registered — an httpx call here would raise inside the mock.
        result = await fetch_page(URL, firecrawl_app=app)
    assert result.status == "URL_DEAD"
    assert result.http_status == 404
    assert len(app.calls) == 1


@respx.mock
async def test_firecrawl_402_maps_to_fetch_blocked_when_continuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grounding-style usage: 402 falls through the chain
    (on_credits_exhausted='continue')."""
    app = _FirecrawlApp(
        raises=_FakeFirecrawlError("Payment Required: Failed to scrape.", status_code=402)
    )
    respx.get(URL).mock(return_value=httpx.Response(500, text=""))
    _ensure_playwright_absent(monkeypatch)

    result = await fetch_page(URL, firecrawl_app=app, on_credits_exhausted="continue")

    assert result.status == "FETCH_BLOCKED"


async def test_firecrawl_402_raises_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifier-style usage: 402 raises immediately, no httpx/Playwright attempted
    — lets a batch caller short-circuit remaining work without wasted calls."""
    app = _FirecrawlApp(
        raises=_FakeFirecrawlError("Payment Required: Failed to scrape.", status_code=402)
    )
    with respx.mock:
        # No route registered — would raise inside the mock if httpx were called.
        with pytest.raises(FirecrawlCreditsExhausted):
            await fetch_page(URL, firecrawl_app=app, on_credits_exhausted="raise")
    assert len(app.calls) == 1


async def test_firecrawl_402_message_only_still_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain exception (no ``.status_code``) whose message names a 402 is still
    classified as credits-exhausted via the message-substring fallback."""
    app = _FirecrawlApp(raises=RuntimeError("HTTP 402: Insufficient credits to perform this request"))
    with respx.mock:
        with pytest.raises(FirecrawlCreditsExhausted):
            await fetch_page(URL, firecrawl_app=app, on_credits_exhausted="raise")
    assert len(app.calls) == 1


# -----------------------------------------------------------------------------
# skip_firecrawl bypasses tier 1 even when an app is supplied
# -----------------------------------------------------------------------------


@respx.mock
async def test_skip_firecrawl_bypasses_tier1(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _FirecrawlApp(response={"success": True, "markdown": "should not be used", "metadata": {}})
    respx.get(URL).mock(return_value=httpx.Response(200, text="<html><body>via httpx only</body></html>"))

    result = await fetch_page(URL, firecrawl_app=app, skip_firecrawl=True)

    assert result.status == "OK"
    assert result.method == "httpx"
    assert app.calls == []
