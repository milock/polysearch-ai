"""Unit tests for the opt-in LinkedIn enricher and the PersonContextHook seam.

The enricher is credential-gated: without ``scrapecreators_api_key`` it is
inactive and ``enrich`` short-circuits to ``None`` (no network) with a reason.
With a key, it fetches a profile via ScrapeCreators (mocked with respx) and
normalizes it into a single ``SourceResult`` — surfacing only unmasked fields.
The ``PersonContextHook`` tests cover the protocol shape (a stub with the right
``lookup`` signature is an instance; one without it is not).
"""

from __future__ import annotations

import httpx
import respx

from polysearch.config import Settings
from polysearch.output.schema import SourceResult
from polysearch.providers.base import PersonContextHook
from polysearch.providers.linkedin import LinkedInEnricher

_PROFILE_URL = "https://www.linkedin.com/in/jane-doe/"
_API = "https://api.scrapecreators.com/v1/linkedin/profile"

# A ScrapeCreators standard-tier payload: `about`, `followers`, one recent post,
# and an unmasked experience *location* come back clean, while employer names
# and education are asterisk-masked.
_SUCCESS_PAYLOAD = {
    "success": True,
    "name": "Jane Doe",
    "location": "San Francisco Bay Area",
    "followers": 1234,
    "about": "Product leader building developer tools.",
    "recentPosts": [
        {
            "title": "Shipping something new today.",
            "datePublished": "2026-01-05T00:00:00",
            "link": "https://example.test/post/1",
        }
    ],
    "experience": [{"name": "*****", "location": "Remote"}],
    "education": [{"name": "*****"}],
}


def _settings(key: str | None = "sc-test") -> Settings:
    return Settings(scrapecreators_api_key=key)


# ── Inactive path: no key → None, a reason, and no network call ──────────────

async def test_enricher_inactive_without_key() -> None:
    enricher = LinkedInEnricher(Settings())
    assert enricher.active is False
    assert enricher.reason is not None
    assert "SCRAPECREATORS_API_KEY" in enricher.reason


@respx.mock
async def test_enrich_inactive_returns_none_without_calling_api() -> None:
    route = respx.get(_API).mock(return_value=httpx.Response(200, json=_SUCCESS_PAYLOAD))
    enricher = LinkedInEnricher(Settings())
    assert await enricher.enrich(_PROFILE_URL) is None
    assert not route.called


# ── Happy path: profile → SourceResult, unmasked fields only ─────────────────

@respx.mock
async def test_enrich_happy_path_returns_source_result() -> None:
    route = respx.get(_API).mock(
        return_value=httpx.Response(200, json=_SUCCESS_PAYLOAD)
    )
    enricher = LinkedInEnricher(_settings())
    result = await enricher.enrich(_PROFILE_URL)

    assert isinstance(result, SourceResult)
    assert result.url == _PROFILE_URL
    assert result.title == "Jane Doe"
    assert result.layer == "linkedin"
    assert result.engagement == 1234
    # snippet carries the self-authored signal (about + recent post).
    assert "Product leader" in result.snippet
    assert "Shipping something new" in result.snippet


@respx.mock
async def test_enrich_sends_scrapecreators_auth_header() -> None:
    route = respx.get(_API).mock(
        return_value=httpx.Response(200, json=_SUCCESS_PAYLOAD)
    )
    await LinkedInEnricher(_settings("secret-key")).enrich(_PROFILE_URL)
    assert route.called
    assert route.calls.last.request.headers["x-api-key"] == "secret-key"
    assert route.calls.last.request.url.params["url"] == _PROFILE_URL


@respx.mock
async def test_enrich_flags_masked_sections_instead_of_asterisks() -> None:
    respx.get(_API).mock(return_value=httpx.Response(200, json=_SUCCESS_PAYLOAD))
    result = await LinkedInEnricher(_settings()).enrich(_PROFILE_URL)
    assert result is not None
    # Masked employer names / education are flagged, never emitted as "*****".
    assert "*****" not in result.snippet
    assert "Masked" in result.snippet


# ── Failure paths: API error and empty/unsuccessful response → None + reason ─

@respx.mock
async def test_enrich_api_error_returns_none_and_surfaces_error() -> None:
    respx.get(_API).mock(return_value=httpx.Response(500))
    enricher = LinkedInEnricher(_settings())
    result = await enricher.enrich(_PROFILE_URL)
    assert result is None
    assert enricher.reason is not None
    assert "500" in enricher.reason or "fail" in enricher.reason.lower()


@respx.mock
async def test_enrich_unsuccessful_payload_returns_none() -> None:
    respx.get(_API).mock(return_value=httpx.Response(200, json={"success": False}))
    enricher = LinkedInEnricher(_settings())
    assert await enricher.enrich(_PROFILE_URL) is None


# ── PersonContextHook: runtime-checkable protocol shape ──────────────────────

def test_person_context_hook_accepts_stub_with_lookup() -> None:
    class StubHook:
        async def lookup(self, name: str) -> SourceResult | None:
            return None

    assert isinstance(StubHook(), PersonContextHook)


def test_person_context_hook_rejects_object_without_lookup() -> None:
    class NotAHook:
        async def fetch(self, name: str) -> SourceResult | None:
            return None

    assert not isinstance(NotAHook(), PersonContextHook)
