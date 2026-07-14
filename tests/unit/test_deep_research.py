"""Unit tests for polysearch.providers.deep_research.

Covers the deep-research provider (sonar-deep-research async Sonar API): a
module-level ``research`` returns a ``PerplexityResult``, and
``DeepResearchProvider`` adapts it to the ``ResearchProvider`` protocol (returns
a ``LayerOutput``).

All network I/O is mocked with respx — the suite runs with zero env vars.
Submission is ``POST /v1/async/sonar`` with a wrapped ``{"request": {...}}``
chat-completions body; polling hits ``GET /v1/async/sonar/{id}`` until a
terminal (UPPERCASE) status. The finished chat completion arrives under the
``response`` field. ``_SLEEP_FN`` is a module-level, monkeypatch-overridable
hook so tests never actually sleep. Fixtures are generic (no real topics,
no private data).
"""

from __future__ import annotations

import json as _json

import httpx
import pytest
import respx

from polysearch.config import Settings
from polysearch.output.schema import LayerOutput
from polysearch.providers import deep_research as dr
from polysearch.providers.base import ResearchProvider

SUBMIT_URL = "https://api.perplexity.ai/v1/async/sonar"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-test-not-real")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Never actually sleep between polls."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(dr, "_SLEEP_FN", _instant)


def _completed_job(
    *,
    text: str = "Deep research findings.",
    citations: list[str] | None = None,
    search_results: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    """A terminal async job whose ``response`` is a standard chat completion."""
    return {
        "id": "req_abc123",
        "model": "sonar-deep-research",
        "created_at": 0,
        "status": "COMPLETED",
        "response": {
            "id": "chatcmpl-1",
            "model": "sonar-deep-research",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            "citations": citations or [],
            "search_results": search_results or [],
            "usage": usage or {},
        },
    }


# -----------------------------------------------------------------------------
# Submit request shape
# -----------------------------------------------------------------------------


@respx.mock
async def test_submit_wraps_chat_payload_under_request():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(200, json=_completed_job())
    )

    await dr.research("solid-state battery commercialization")

    submit_calls = [c for c in respx.calls if c.request.method == "POST"]
    assert len(submit_calls) == 1
    payload = _json.loads(submit_calls[0].request.content)
    assert payload["request"]["model"] == "sonar-deep-research"
    assert payload["request"]["messages"] == [
        {"role": "user", "content": "solid-state battery commercialization"}
    ]


@respx.mock
async def test_submit_uses_bearer_auth_header():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(200, json=_completed_job())
    )

    await dr.research("topic", api_key="sk-explicit-key")

    submit_call = [c for c in respx.calls if c.request.method == "POST"][0]
    assert submit_call.request.headers["authorization"] == "Bearer sk-explicit-key"


@respx.mock
async def test_submit_uses_settings_model_when_overridden():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(200, json=_completed_job())
    )

    await dr.research("topic", model="sonar-deep-research-pro")

    payload = _json.loads(
        [c for c in respx.calls if c.request.method == "POST"][0].request.content
    )
    assert payload["request"]["model"] == "sonar-deep-research-pro"


# -----------------------------------------------------------------------------
# Polling: CREATED -> IN_PROGRESS -> COMPLETED (uppercase statuses)
# -----------------------------------------------------------------------------


@respx.mock
async def test_polls_through_created_in_progress_completed():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    poll_route = respx.get(f"{SUBMIT_URL}/req_abc123")
    poll_route.side_effect = [
        httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"}),
        httpx.Response(200, json={"id": "req_abc123", "status": "IN_PROGRESS"}),
        httpx.Response(200, json=_completed_job(text="Final synthesized answer.")),
    ]

    result = await dr.research("topic")

    assert poll_route.call_count == 3
    assert result.error is None
    assert result.answer == "Final synthesized answer."
    assert result.model == "sonar-deep-research"


# -----------------------------------------------------------------------------
# Citation parsing (search_results[] + citations[])
# -----------------------------------------------------------------------------


@respx.mock
async def test_citations_parsed_from_search_results_and_citations_array():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json=_completed_job(
                citations=["https://example.org/deep-dive"],
                search_results=[
                    {
                        "url": "https://www.energy.gov/battery-report",
                        "title": "Battery Report",
                        "snippet": "Final rule.",
                        "date": "2026-01-05",
                    }
                ],
            ),
        )
    )

    result = await dr.research("topic")

    by_url = {c.url: c for c in result.citations}
    assert set(by_url) == {
        "https://www.energy.gov/battery-report",
        "https://example.org/deep-dive",
    }
    assert by_url["https://www.energy.gov/battery-report"].title == "Battery Report"
    assert by_url["https://www.energy.gov/battery-report"].domain == "energy.gov"
    assert by_url["https://example.org/deep-dive"].domain == "example.org"


# -----------------------------------------------------------------------------
# Cost accounting (four-meter billing)
# -----------------------------------------------------------------------------


@respx.mock
async def test_cost_prefers_api_reported_total():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json=_completed_job(
                usage={
                    "prompt_tokens": 33,
                    "completion_tokens": 9925,
                    "citation_tokens": 4133,
                    "reasoning_tokens": 55940,
                    "num_search_queries": 45,
                    "cost": {"total_cost": 0.48055},
                }
            ),
        )
    )

    result = await dr.research("topic")

    assert result.cost_usd == pytest.approx(0.48055)
    assert result.tokens_input == 33
    assert result.tokens_output == 9925


@respx.mock
async def test_cost_falls_back_to_four_meters_on_chat_usage_keys():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json=_completed_job(
                usage={
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "citation_tokens": 1_000_000,
                    "reasoning_tokens": 1_000_000,
                    "num_search_queries": 1_000,
                }
            ),
        )
    )

    result = await dr.research("topic")

    # $2 (in) + $8 (out) + $2 (citation) + $3 (reasoning) + $5 (search) = $20
    assert result.cost_usd == pytest.approx(20.0)


# -----------------------------------------------------------------------------
# Timeout / failure isolation — never raises, always returns an error result
# -----------------------------------------------------------------------------


@respx.mock
async def test_timeout_returns_error_result_without_raising():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "IN_PROGRESS"})
    )

    result = await dr.research("topic", poll_interval=15.0, timeout=30.0)

    assert result.error is not None
    assert "timeout" in result.error.lower() or "timed out" in result.error.lower()
    assert result.answer == ""


@respx.mock
async def test_failed_status_returns_error_result_with_message():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json={"id": "req_abc123", "status": "FAILED", "error_message": "model overloaded"},
        )
    )

    result = await dr.research("topic")

    assert result.error is not None
    assert "model overloaded" in result.error


@respx.mock
async def test_http_error_on_submit_returns_error_result():
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))

    result = await dr.research("topic")

    assert result.error is not None
    assert result.answer == ""
    assert result.citations == []


@respx.mock
async def test_400_unsupported_model_surfaces_as_error_not_crash():
    """Regression: the old /v1/agent endpoint 400'd with 'model not supported'.
    Any 400 on submit must isolate as a structured error, never raise."""
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": 'model "sonar-deep-research" is not supported', "code": 400}},
        )
    )

    result = await dr.research("topic")

    assert result.error is not None
    assert result.answer == ""


@respx.mock
async def test_poll_429_records_backoff_but_continues():
    """A 429 on a poll GET broadcasts a 429 to the shared ledger, then the run
    continues to completion (polls are exempt from the acquire bucket)."""
    calls: list[tuple] = []
    import polysearch.ratelimit as rl

    def _record(provider, retry_after=None):
        calls.append((provider, retry_after))

    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    poll_route = respx.get(f"{SUBMIT_URL}/req_abc123")
    poll_route.side_effect = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"}),
        httpx.Response(200, json=_completed_job()),
    ]

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(rl, "record_429", _record)
        result = await dr.research("topic")

    assert result.error is None
    assert ("perplexity", 7.0) in calls


# -----------------------------------------------------------------------------
# Missing API key
# -----------------------------------------------------------------------------


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        await dr.research("topic", api_key=None)


# -----------------------------------------------------------------------------
# DeepResearchProvider — adapts research() to the ResearchProvider protocol
# -----------------------------------------------------------------------------


def _settings(**over) -> Settings:
    base = dict(
        perplexity_api_key="sk-test-not-real",
        enable_deep_research=True,
        deep_research_model="sonar-deep-research",
        deep_research_timeout_s=3600,
    )
    base.update(over)
    return Settings(**base)


def test_provider_conforms_to_protocol():
    provider = dr.DeepResearchProvider(_settings())
    assert isinstance(provider, ResearchProvider)
    assert provider.name == "deep_research"


def test_provider_inactive_when_flag_disabled():
    provider = dr.DeepResearchProvider(_settings(enable_deep_research=False))
    assert provider.active is False
    assert provider.reason is not None
    assert "deep research" in provider.reason.lower()


def test_provider_inactive_when_no_api_key():
    provider = dr.DeepResearchProvider(_settings(perplexity_api_key=None))
    assert provider.active is False
    assert "PERPLEXITY_API_KEY" in (provider.reason or "")


async def test_provider_inactive_research_returns_error_layer_without_network():
    provider = dr.DeepResearchProvider(_settings(enable_deep_research=False))
    out = await provider.research("topic", sub_questions=1, depth="deep")
    assert isinstance(out, LayerOutput)
    assert out.layer == "deep_research"
    assert out.error == provider.reason
    assert out.results == []


@respx.mock
async def test_provider_active_maps_citations_to_layer_output():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json=_completed_job(
                text="A cited synthesis.",
                citations=["https://example.org/a"],
                search_results=[
                    {
                        "url": "https://www.energy.gov/b",
                        "title": "Gov B",
                        "snippet": "snip",
                        "date": "2026-02-02",
                    }
                ],
                usage={"cost": {"total_cost": 0.25}},
            ),
        )
    )

    provider = dr.DeepResearchProvider(_settings())
    out = await provider.research("topic", sub_questions=1, depth="deep")

    assert isinstance(out, LayerOutput)
    assert out.layer == "deep_research"
    assert out.error is None
    assert out.cost_usd == pytest.approx(0.25)
    urls = {s.url for s in out.results}
    assert urls == {"https://example.org/a", "https://www.energy.gov/b"}
    assert all(s.layer == "deep_research" for s in out.results)


@respx.mock
async def test_provider_surfaces_failed_status_as_layer_error():
    respx.post(SUBMIT_URL).mock(
        return_value=httpx.Response(200, json={"id": "req_abc123", "status": "CREATED"})
    )
    respx.get(f"{SUBMIT_URL}/req_abc123").mock(
        return_value=httpx.Response(
            200,
            json={"id": "req_abc123", "status": "FAILED", "error_message": "overloaded"},
        )
    )

    provider = dr.DeepResearchProvider(_settings())
    out = await provider.research("topic", sub_questions=1, depth="deep")

    assert out.error is not None
    assert "overloaded" in out.error
    assert out.results == []
