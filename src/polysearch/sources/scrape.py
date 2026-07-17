"""Scrape fallback chain: Firecrawl -> httpx -> Playwright (optional).

Motivating bug: Firecrawl 429s (and other transient/blocking failures) were
causing citations to be falsely marked URL_DEAD, silently dropping live sources
from a report. ``fetch_page()`` tries progressively-more-robust methods before
giving up, and — the point of this module — distinguishes a page that is
genuinely unreachable (``URL_DEAD``: DNS failure, connection refused, 404/410)
from one that is merely blocked but still alive (``FETCH_BLOCKED``: 403/429/5xx,
bot-challenge pages, Firecrawl credit exhaustion) so blocked citations aren't
wrongly excluded as dead.

Chain, in order:
    1. Firecrawl scrape (only if a Firecrawl app is supplied and not skipped).
       An explicit, origin-confirmed dead signal (404/410) short-circuits here —
       Firecrawl already reached the origin and got a definitive answer, so
       there's nothing for httpx/Playwright to add.
    2. Plain ``httpx`` GET with a realistic browser User-Agent.
    3. Playwright headless Chromium — OPTIONAL dependency (``polysearch-ai[browser]``),
       only attempted if ``playwright`` is importable. Never required, never
       installed by this module.

Modern-SDK note: the firecrawl-py v3+ client exposes a flat
``AsyncFirecrawl.scrape(url, formats=[...])`` that returns a v2 ``Document`` on
success and *raises* a typed ``FirecrawlError`` (carrying ``.status_code``) on an
HTTP error — where the legacy client returned a ``{"success": False,
"metadata": {"statusCode": ...}}`` dict. The error taxonomy (402 credits, 429/5xx
retry, 404/410 dead) is therefore driven off the raised exception's
``status_code`` (with a message-substring fallback so plain-``Exception`` test
stubs and older shapes still classify correctly).

Public:
    await fetch_page(url, ...) -> ScrapeResult
    firecrawl_scrape_with_retry(app, url) -> dict   # shared by grounding + verifier
    FirecrawlCreditsExhausted                        # shared exception
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from polysearch import ratelimit

log = logging.getLogger(__name__)

Status = Literal["OK", "FETCH_BLOCKED", "URL_DEAD"]
Method = Literal["firecrawl", "httpx", "playwright"]

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Evidence that the page itself is gone, not merely blocking us.
_DEAD_HTTP_STATUSES = {404, 410}
_DEAD_ERROR_MARKERS = (
    "nodename nor servname",  # macOS DNS resolution failure
    "name or service not known",  # Linux DNS resolution failure
    "getaddrinfo failed",
    "temporary failure in name resolution",
    "connection refused",
    "no address associated with hostname",
)

# Evidence the page is blocked-but-alive — never marked dead on this alone.
_BLOCKED_HTTP_STATUSES = {401, 403, 407, 408, 429}
_BOT_CHALLENGE_MARKERS = (
    "checking your browser",
    "captcha",
    "cloudflare ray id",
    "access denied",
    "are you a human",
    "verify you are human",
)
# A challenge/interstitial page is often served with HTTP 200, so marker text
# alone isn't enough evidence — a long, legitimate article can mention "captcha"
# in passing. Only treat it as a real challenge when the whole page is short (a
# real challenge page has little else on it) or the marker appears near the very
# top (real articles bury an incidental mention deep in the body, not in the
# first few hundred characters).
_CHALLENGE_SNIFF_MAX_LEN = 4096
_CHALLENGE_MARKER_HEAD_WINDOW = 500


def _is_challenge_page(markdown: str | None) -> bool:
    if not markdown:
        return False
    low = markdown.lower()
    haystack = (
        low if len(markdown) <= _CHALLENGE_SNIFF_MAX_LEN else low[:_CHALLENGE_MARKER_HEAD_WINDOW]
    )
    return any(marker in haystack for marker in _BOT_CHALLENGE_MARKERS)


@dataclass
class ScrapeResult:
    markdown: str | None
    status: Status
    method: Method | None
    detail: str | None = None
    http_status: int | None = None
    # Structured metadata (e.g. published_time) — only available when
    # ``method == "firecrawl"``; None for httpx/Playwright tiers, which only give
    # back extracted body text.
    metadata: dict[str, Any] | None = None


class FirecrawlCreditsExhausted(Exception):
    """Raised when Firecrawl reports 402 / credits exhausted. Account-level
    failure — unrelated to whether the target page itself is reachable."""


def is_credits_exhausted_message(msg: str) -> bool:
    m = msg.lower()
    return (
        "402" in m
        or "payment required" in m
        or "insufficient credits" in m
        or ("credit" in m and "exhaust" in m)
    )


def _status_code_of(exc: BaseException) -> int | None:
    """Pull a ``.status_code`` off a modern FirecrawlError (or its wrapped
    ``.response``). Returns None for plain exceptions / test stubs."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status if isinstance(status, int) else None


def _as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort convert a Pydantic-model-or-dict Firecrawl response."""
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


def _document_to_response_dict(doc: Any) -> dict[str, Any]:
    """Normalize a modern firecrawl v2 ``Document`` into the internal
    response-dict shape the chain consumes (``markdown`` / ``metadata`` /
    ``success``). A plain dict (test stub or an already-normalized response)
    passes through with ``success`` defaulted, so both call shapes work."""
    if isinstance(doc, dict):
        doc.setdefault("success", True)
        return doc
    data = _as_dict(doc)
    return {
        "markdown": data.get("markdown") or "",
        "metadata": data.get("metadata") or {},
        "success": True,
    }


def _http_status_from_response(resp: dict[str, Any]) -> int | None:
    """Best-effort HTTP status code extraction from a Firecrawl response.

    Reads both the legacy camelCase (``statusCode``) and modern snake_case
    (``status_code``) metadata keys."""
    if not resp:
        return None
    meta = resp.get("metadata") or {}
    for key in ("statusCode", "status_code", "httpStatus", "http_status"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
    v = resp.get("statusCode") or resp.get("status_code")
    return v if isinstance(v, int) else None


def _is_definitive_dead(http_status: int | None, error: str | None) -> bool:
    """Evidence-based DEAD check: DNS failure, connection refused, 404/410.
    Deliberately narrow — anything else falls to FETCH_BLOCKED so a citation is
    never wrongly excluded as dead on ambiguous evidence."""
    if http_status in _DEAD_HTTP_STATUSES:
        return True
    if error:
        low = error.lower()
        if any(marker in low for marker in _DEAD_ERROR_MARKERS):
            return True
    return False


def _is_blocked_evidence(
    http_status: int | None, error: str | None, markdown: str | None = None
) -> bool:
    """Evidence the page is blocked-but-alive (403/429/5xx, bot-challenge page,
    Firecrawl credit exhaustion)."""
    if http_status is not None and (
        http_status in _BLOCKED_HTTP_STATUSES or 500 <= http_status < 600
    ):
        return True
    if error:
        low = error.lower()
        if is_credits_exhausted_message(low):
            return True
        if "429" in low or "rate limit" in low or "too many requests" in low:
            return True
    if _is_challenge_page(markdown):
        return True
    return False


def _tier_ok(markdown: str | None, http_status: int | None) -> bool:
    """A tier only counts as a real success when it has content, the status isn't
    an error, and the content isn't a bot-challenge page served with a 200 — that
    last case is exactly what motivates continuing the chain instead of accepting
    the challenge page as real content."""
    return (
        bool(markdown)
        and (http_status is None or http_status < 400)
        and not _is_challenge_page(markdown)
    )


_TAG_BLOCK_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ALL_TAGS_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    """Minimal HTML -> text fallback: drop script/style blocks, strip remaining
    tags, unescape entities, collapse whitespace. This is a plain tag-strip, NOT
    readability-style main-content extraction (no boilerplate removal) — good
    enough for a last-resort tier without a new dependency."""
    stripped = _TAG_BLOCK_RE.sub(" ", html)
    stripped = _ALL_TAGS_RE.sub(" ", stripped)
    stripped = html_lib.unescape(stripped)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


class _FirecrawlRetryable(Exception):
    """Internal signal for tenacity to retry a Firecrawl 429/5xx-like error."""


async def firecrawl_scrape_with_retry(app: Any, url: str) -> dict[str, Any]:
    """Retrying Firecrawl scrape shared by grounding + citation verification.

    Calls the modern ``app.scrape(url, formats=["markdown"])`` and normalizes the
    returned ``Document`` into a response dict. Retries 429/5xx-like errors up to
    3x. 402 / credit-exhaustion is a hard signal — no point retrying an
    out-of-credits account — so it raises ``FirecrawlCreditsExhausted`` immediately
    instead. Error class is read from the exception's ``.status_code`` when present
    (modern typed errors), falling back to message-substring matching.
    """

    async def _do() -> dict[str, Any]:
        try:
            async with ratelimit.acquire("firecrawl"):
                resp = await app.scrape(url, formats=["markdown"])
        except Exception as e:  # noqa: BLE001
            status = _status_code_of(e)
            msg = str(e).lower()
            # Check 402 before 429/5xx so a "402" substring (or a payment-required
            # status) can't be mistaken for a retryable rate-limit / server error.
            if status == 402 or is_credits_exhausted_message(msg):
                raise FirecrawlCreditsExhausted(str(e)) from e
            retryable_status = status == 429 or (status is not None and 500 <= status < 600)
            retryable_msg = (
                "429" in msg or "rate" in msg or "503" in msg or "502" in msg or "500" in msg
            )
            if retryable_status or retryable_msg:
                # Bare "rate" also matches "moderate"/"accelerate" — narrow the
                # shared-backoff signal to a genuine 429 / rate-limit so those
                # don't trigger a spurious 429 broadcast (the broader
                # classification above still retries them).
                if status == 429 or "429" in msg or "rate limit" in msg:
                    ratelimit.record_429("firecrawl")
                raise _FirecrawlRetryable(str(e)) from e
            raise
        return _document_to_response_dict(resp)

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(_FirecrawlRetryable),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=10),
        reraise=True,
    ):
        with attempt:
            return await _do()
    return {}  # pragma: no cover — retry loop always returns or raises


async def _fetch_via_firecrawl(
    app: Any, url: str
) -> tuple[str | None, int | None, str | None, dict[str, Any] | None]:
    """One Firecrawl attempt (with its own 429/5xx retry). Returns (markdown,
    http_status, error, metadata). Raises ``FirecrawlCreditsExhausted`` on 402 so
    callers can decide whether to stop or fall through.

    A non-402 error is caught here (not propagated) and its ``.status_code`` — the
    modern SDK carries it on the raised ``FirecrawlError`` — is surfaced as
    ``http_status`` so the chain can still distinguish an origin-confirmed dead
    (404/410) from a blocked page without a returned response body."""
    try:
        resp = await firecrawl_scrape_with_retry(app, url)
    except FirecrawlCreditsExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        return None, _status_code_of(e), f"{type(e).__name__}: {e}", None
    http_status = _http_status_from_response(resp)
    markdown = resp.get("markdown") or ""
    metadata = resp.get("metadata") or None
    error = None
    if resp.get("success") is False:
        error = resp.get("error") or f"http_status={http_status}"
    return (markdown or None), http_status, error, metadata


async def _fetch_via_httpx(
    url: str, timeout_s: float
) -> tuple[str | None, int | None, str | None]:
    """Plain GET with a realistic browser UA. Returns (markdown, http_status,
    error). Never raises — connection/DNS failures are reported as errors so the
    chain can classify them as dead vs blocked."""
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": BROWSER_UA},
        ) as client:
            resp = await client.get(url)
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"

    text = _html_to_text(resp.text) if resp.text else ""
    error = None if resp.status_code < 400 else f"http {resp.status_code}"
    return (text or None), resp.status_code, error


def _playwright_importable() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


async def _fetch_via_playwright(
    url: str, timeout_s: float
) -> tuple[str | None, int | None, str | None]:
    """Headless Chromium fetch. Only called when ``playwright`` is importable — an
    optional dependency this module never requires or installs."""
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page(user_agent=BROWSER_UA)
                resp = await page.goto(url, timeout=timeout_s * 1000)
                body_html = await page.content()
                http_status = resp.status if resp is not None else None
                text = _html_to_text(body_html) if body_html else ""
                error = (
                    None
                    if (http_status is None or http_status < 400)
                    else f"http {http_status}"
                )
                return (text or None), http_status, error
            finally:
                await browser.close()
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


async def fetch_page(
    url: str,
    *,
    timeout_s: float = 30,
    firecrawl_app: Any | None = None,
    skip_firecrawl: bool = False,
    on_credits_exhausted: Literal["raise", "continue"] = "continue",
) -> ScrapeResult:
    """Fetch a single page, falling back Firecrawl -> httpx -> Playwright.

    Args:
        url: Page to fetch.
        timeout_s: Per-tier timeout budget (seconds).
        firecrawl_app: An already-constructed Firecrawl app/client. Tier 1 is
            skipped entirely when this is None.
        skip_firecrawl: Bypass tier 1 even if ``firecrawl_app`` is supplied — used
            by callers that already know Firecrawl is unavailable (e.g. a
            verifier's credits-exhausted latch).
        on_credits_exhausted: "raise" propagates ``FirecrawlCreditsExhausted``
            immediately on a 402 (no httpx/Playwright attempted) so a batch caller
            can short-circuit remaining work. "continue" (default) treats 402 like
            any other Firecrawl failure and falls through the chain — appropriate
            for one-off grounding scrapes.
    """
    # (method, http_status, error, markdown) for tiers that didn't return OK.
    attempts: list[tuple[Method, int | None, str | None, str | None]] = []

    if firecrawl_app is not None and not skip_firecrawl:
        metadata: dict[str, Any] | None = None
        try:
            markdown, http_status, error, metadata = await _fetch_via_firecrawl(firecrawl_app, url)
        except FirecrawlCreditsExhausted:
            if on_credits_exhausted == "raise":
                raise
            markdown, http_status, error = None, None, "firecrawl credits exhausted (402)"

        if _tier_ok(markdown, http_status):
            return ScrapeResult(
                markdown=markdown,
                status="OK",
                method="firecrawl",
                http_status=http_status,
                metadata=metadata,
            )
        # Firecrawl reached the origin and got a definitive dead answer — nothing
        # for httpx/Playwright to add, so stop here.
        if _is_definitive_dead(http_status, error):
            return ScrapeResult(
                markdown=None, status="URL_DEAD", method=None, detail=error, http_status=http_status
            )
        if error is None and _is_challenge_page(markdown):
            error = "bot challenge detected"
        attempts.append(("firecrawl", http_status, error, markdown))

    markdown, http_status, error = await _fetch_via_httpx(url, timeout_s)
    if _tier_ok(markdown, http_status):
        return ScrapeResult(markdown=markdown, status="OK", method="httpx", http_status=http_status)
    if error is None and _is_challenge_page(markdown):
        error = "bot challenge detected"
    attempts.append(("httpx", http_status, error, markdown))

    if _playwright_importable():
        markdown, http_status, error = await _fetch_via_playwright(url, timeout_s)
        if _tier_ok(markdown, http_status):
            return ScrapeResult(
                markdown=markdown, status="OK", method="playwright", http_status=http_status
            )
        if error is None and _is_challenge_page(markdown):
            error = "bot challenge detected"
        attempts.append(("playwright", http_status, error, markdown))

    any_dead = any(_is_definitive_dead(hs, err) for (_, hs, err, _md) in attempts)
    any_blocked = any(_is_blocked_evidence(hs, err, md) for (_, hs, err, md) in attempts)
    status: Status = "URL_DEAD" if (any_dead and not any_blocked) else "FETCH_BLOCKED"
    detail = "; ".join(f"{m}={err or hs}" for (m, hs, err, _md) in attempts if err or hs) or None
    last_http_status = next(
        (hs for (_, hs, _e, _md) in reversed(attempts) if hs is not None), None
    )
    return ScrapeResult(
        markdown=None, status=status, method=None, detail=detail, http_status=last_http_status
    )


__all__ = [
    "ScrapeResult",
    "Status",
    "Method",
    "FirecrawlCreditsExhausted",
    "is_credits_exhausted_message",
    "firecrawl_scrape_with_retry",
    "fetch_page",
]
