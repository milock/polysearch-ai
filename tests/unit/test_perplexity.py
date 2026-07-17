"""Unit tests for polysearch.providers.perplexity.

Covers the Perplexity provider (chat-completions Sonar path, ``<think>``
stripping, and the Search API): the provider returns a ``LayerOutput``,
``search`` returns ``SourceResult`` objects, and ``strip_think_blocks`` is
public. All network I/O is mocked (AsyncOpenAI monkeypatch / respx) — the suite
runs with zero env vars.
"""

from __future__ import annotations

import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from polysearch.config import Settings
from polysearch.output.schema import LayerOutput, SourceResult
from polysearch.providers import perplexity as pp
from polysearch.providers.base import ResearchProvider


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def test_domain_strips_www_and_lowercases():
    assert pp._domain_of("https://WWW.EXAMPLE.gov/path") == "example.gov"
    assert pp._domain_of("https://www.Example.COM") == "example.com"
    assert pp._domain_of("https://sub.example.com/x") == "sub.example.com"


def test_recency_mapping():
    assert pp._map_recency("quarter") == "month"
    assert pp._map_recency("all") is None
    assert pp._map_recency("month") == "month"
    assert pp._map_recency("week") == "week"


def test_model_selection_by_depth():
    assert pp._model_for_depth("quick") == "sonar-pro"
    assert pp._model_for_depth("standard") == "sonar-pro"
    assert pp._model_for_depth("deep") == "sonar-reasoning-pro"


def test_parse_citations_merges_urls_and_search_results():
    urls = ["https://www.example.gov/a", "https://docs.python.org/b"]
    results = [
        {
            "url": "https://www.example.gov/a",
            "title": "Gov doc",
            "snippet": "snip",
            "date": "2026-01-05",
        },
        {"url": "https://other.com/c", "title": "Other"},
    ]
    cits = pp._parse_citations(urls, results)
    by_url = {c.url: c for c in cits}
    assert set(by_url) == {
        "https://www.example.gov/a",
        "https://docs.python.org/b",
        "https://other.com/c",
    }
    # search_results metadata wins
    assert by_url["https://www.example.gov/a"].title == "Gov doc"
    assert by_url["https://www.example.gov/a"].snippet == "snip"
    assert by_url["https://www.example.gov/a"].published_date == "2026-01-05"
    assert by_url["https://www.example.gov/a"].domain == "example.gov"
    # URL-only citation gets domain but no metadata
    assert by_url["https://docs.python.org/b"].title is None
    assert by_url["https://docs.python.org/b"].domain == "docs.python.org"


def test_extract_json_array_tolerates_noise():
    assert pp._extract_json_array('["a", "b", "c"]') == ["a", "b", "c"]
    assert pp._extract_json_array('Here you go: ["x","y"] thanks!') == ["x", "y"]
    assert pp._extract_json_array("no array here") is None
    assert pp._extract_json_array('["ok", 123, ""]') == ["ok", "123"]


def test_build_response_format_wraps_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    assert pp._build_response_format(schema) == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


# -----------------------------------------------------------------------------
# <think> block stripping
# -----------------------------------------------------------------------------


def test_strips_single_think_block():
    raw = "<think>Let me reason about this...</think>\nFinal answer here."
    assert pp.strip_think_blocks(raw) == "Final answer here."


def test_strips_multiple_think_blocks():
    raw = "<think>step 1</think>\n<think>step 2</think>\nThe conclusion."
    assert pp.strip_think_blocks(raw) == "The conclusion."


def test_handles_multiline_think():
    raw = "<think>\nlong\nmultiline\nreasoning\n</think>\nAnswer."
    assert pp.strip_think_blocks(raw) == "Answer."


def test_passthrough_when_no_think_block():
    raw = "Just a plain answer with no chain of thought."
    assert pp.strip_think_blocks(raw) == raw


def test_strips_case_insensitive():
    raw = "<THINK>thinking</THINK>\nA."
    assert pp.strip_think_blocks(raw) == "A."


# -----------------------------------------------------------------------------
# Mock-based async tests (chat-completions Sonar path)
# -----------------------------------------------------------------------------


def _mock_chat_response(
    *,
    content: str = "",
    citations: list[str] | None = None,
    search_results: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build an object that looks like an OpenAI SDK ChatCompletion."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "choices": [{"message": {"content": content}}],
        "citations": citations or [],
        "search_results": search_results or [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    mock.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    return mock


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, create_side_effect):
    """Patch AsyncOpenAI to return a client whose create() matches side effect."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)

    def _factory(*_args, **_kwargs):
        return mock_client

    monkeypatch.setattr(pp, "AsyncOpenAI", _factory)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-test-not-real")
    return mock_client


async def test_decompose_fallback_returns_topic_on_failure(monkeypatch):
    async def _boom(**_kwargs):
        raise RuntimeError("decompose broke")

    _install_mock_client(monkeypatch, _boom)
    client = pp._make_client("sk-test-not-real")
    out = await pp._decompose(client, "project management software pricing", 3)
    assert out == ["project management software pricing"]


async def test_research_uses_single_query_when_sub_questions_is_one(monkeypatch):
    calls: list[dict] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(
            content="Answer.",
            citations=["https://www.example.gov/x"],
            search_results=[{"url": "https://www.example.gov/x", "title": "t"}],
        )

    _install_mock_client(monkeypatch, _create)
    results = await pp.research(
        "topic X", depth="quick", sub_questions=1, recency="all"
    )
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert r.answer == "Answer."
    assert r.question == "topic X"
    assert r.model == "sonar-pro"
    assert len(r.citations) == 1
    assert r.citations[0].domain == "example.gov"
    # Only one call: the research call itself (no decomposition for n=1).
    assert len(calls) == 1
    # `recency=all` should omit the recency filter.
    assert "search_recency_filter" not in calls[0]["extra_body"]


async def test_research_quarter_maps_to_month(monkeypatch):
    calls: list[dict] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(content="ok")

    _install_mock_client(monkeypatch, _create)
    client = pp._make_client("sk-test-not-real")
    await pp._run_one(
        client,
        "q",
        model="sonar-pro",
        recency=pp._map_recency("quarter"),
        domain_filter=None,
        search_mode="web",
        context_size="low",
    )
    assert calls[0]["extra_body"]["search_recency_filter"] == "month"


async def test_research_deep_uses_reasoning_model(monkeypatch):
    seen_models: list[str] = []

    async def _create(**kwargs):
        seen_models.append(kwargs["model"])
        if "Decompose" in kwargs["messages"][0]["content"]:
            return _mock_chat_response(content='["sub1", "sub2"]')
        return _mock_chat_response(content="deep answer")

    _install_mock_client(monkeypatch, _create)
    results = await pp.research("topic", depth="deep", sub_questions=2)
    assert len(results) == 2
    research_models = [m for m in seen_models if m == "sonar-reasoning-pro"]
    assert len(research_models) == 2
    for r in results:
        assert r.model == "sonar-reasoning-pro"


async def test_one_failure_does_not_abort_batch(monkeypatch):
    async def _create(**kwargs):
        content = kwargs["messages"][0]["content"]
        if "Decompose" in content:
            return _mock_chat_response(content='["sub-a", "sub-b"]')
        if "sub-a" in content:
            return _mock_chat_response(content="A ok", citations=["https://a.com/x"])
        raise ValueError("boom on sub-b")

    _install_mock_client(monkeypatch, _create)
    results = await pp.research("topic", depth="standard", sub_questions=2)
    assert len(results) == 2
    by_q = {r.question: r for r in results}
    assert by_q["sub-a"].error is None
    assert by_q["sub-a"].answer == "A ok"
    assert by_q["sub-b"].error is not None
    assert "boom on sub-b" in by_q["sub-b"].error
    assert by_q["sub-b"].answer == ""


async def test_research_domain_filter_appears_in_payload(monkeypatch):
    calls: list[dict] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(content="ok")

    _install_mock_client(monkeypatch, _create)
    await pp.research(
        "topic",
        depth="quick",
        sub_questions=1,
        domain_filter=["example.gov", "-spam.com"],
    )
    assert calls[0]["extra_body"]["search_domain_filter"] == ["example.gov", "-spam.com"]


async def test_research_without_domain_filter_omits_it_from_payload(monkeypatch):
    calls: list[dict] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(content="ok")

    _install_mock_client(monkeypatch, _create)
    await pp.research("topic", depth="quick", sub_questions=1)
    assert "search_domain_filter" not in calls[0]["extra_body"]


async def test_research_response_schema_appears_in_payload(monkeypatch):
    calls: list[dict] = []
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(content='{"summary": "ok"}')

    _install_mock_client(monkeypatch, _create)
    await pp.research("topic", depth="quick", sub_questions=1, response_schema=schema)
    assert calls[0]["extra_body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


async def test_research_without_response_schema_omits_response_format(monkeypatch):
    calls: list[dict] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        return _mock_chat_response(content="ok")

    _install_mock_client(monkeypatch, _create)
    await pp.research("topic", depth="quick", sub_questions=1)
    assert "response_format" not in calls[0]["extra_body"]


async def test_citation_domain_extraction_in_research(monkeypatch):
    async def _create(**kwargs):
        if "Decompose" in kwargs["messages"][0]["content"]:
            return _mock_chat_response(content='["only"]')
        return _mock_chat_response(
            content="answer",
            citations=["https://WWW.EXAMPLE.gov/foo"],
            search_results=[{"url": "https://www.Other.COM/a", "title": "ex"}],
        )

    _install_mock_client(monkeypatch, _create)
    results = await pp.research("t", depth="quick", sub_questions=2)
    domains = {c.domain for r in results for c in r.citations}
    assert "example.gov" in domains
    assert "other.com" in domains
    for d in domains:
        assert d == d.lower()
        assert not d.startswith("www.")


async def test_research_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        await pp.research("topic", api_key=None)


def test_estimate_cost_pricing_override_beats_module_table():
    # Module table for sonar-pro is $3/$15 per 1M tokens.
    base = pp._estimate_cost("sonar-pro", 1_000_000, 1_000_000)
    assert base == pytest.approx(3.0 + 15.0)
    # An explicit pricing override wins; models absent from it fall back.
    override = {"sonar-pro": {"input": 1.0, "output": 2.0, "search": 0.0}}
    assert pp._estimate_cost("sonar-pro", 1_000_000, 1_000_000, override) == pytest.approx(3.0)
    # not in override → module table ($2 input + $5/1M per-query search cost).
    assert pp._estimate_cost(
        "sonar-reasoning-pro", 1_000_000, 0, override
    ) == pytest.approx(2.0 + 5.0 / 1_000_000)


async def test_provider_threads_settings_pricing_into_cost(monkeypatch):
    captured: dict = {}

    async def _fake_research(topic, **kwargs):
        captured.update(kwargs)
        return [
            pp.PerplexityResult(
                question="q",
                answer="a",
                citations=[pp.Citation(url="https://ex.gov/a", domain="ex.gov")],
                model="sonar-pro",
                search_results=[],
                tokens_input=0,
                tokens_output=0,
                cost_usd=0.0,
                duration_ms=1,
            )
        ]

    monkeypatch.setattr(pp, "research", _fake_research)
    settings = Settings(
        perplexity_api_key="sk-x", perplexity_price_in=1.5, perplexity_price_out=7.0
    )
    await pp.PerplexityProvider(settings).research("topic", sub_questions=1, depth="quick")
    assert captured["pricing"] == {
        "sonar-pro": {"input": 1.5, "output": 7.0, "search": 0.0}
    }


async def test_research_uses_settings_models_via_provider_override(monkeypatch):
    seen_models: list[str] = []

    async def _create(**kwargs):
        seen_models.append(kwargs["model"])
        return _mock_chat_response(content="ok")

    _install_mock_client(monkeypatch, _create)
    await pp.research(
        "topic", depth="quick", sub_questions=1, model="custom-sonar"
    )
    assert seen_models == ["custom-sonar"]


# -----------------------------------------------------------------------------
# Search API (respx-mocked) — returns SourceResult objects
# -----------------------------------------------------------------------------


@pytest.fixture
def _search_api_key(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-test-not-real")


@respx.mock
async def test_search_returns_parsed_source_results(_search_api_key):
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "req-1",
                "results": [
                    {
                        "title": "Software Pricing Guide",
                        "url": "https://www.example.gov/pricing",
                        "snippet": "Pricing details.",
                        "date": "2026-01-05",
                        "last_updated": "2026-01-06",
                    },
                    {
                        "title": "Licensing Update",
                        "url": "https://docs.python.org/licensing",
                        "snippet": "Licensing changes.",
                        "date": "2026-02-01",
                        "last_updated": "2026-02-02",
                    },
                ],
                "server_time": 0.12,
            },
        )
    )

    results = await pp.search("project management software pricing")

    assert route.called
    assert all(isinstance(r, SourceResult) for r in results)
    assert [r.url for r in results] == [
        "https://www.example.gov/pricing",
        "https://docs.python.org/licensing",
    ]
    first = results[0]
    assert first.title == "Software Pricing Guide"
    assert first.snippet == "Pricing details."
    assert first.published_date == "2026-01-05"
    assert first.tier == "UNKNOWN"
    assert first.layer == "grounding"


@respx.mock
async def test_search_sends_query_and_default_limit(_search_api_key):
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-2", "results": []})
    )

    await pp.search("open source database licensing")

    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer sk-test-not-real"
    payload = _json.loads(sent.content)
    assert payload["query"] == "open source database licensing"
    assert payload["max_results"] == 10


@respx.mock
async def test_search_respects_custom_limit(_search_api_key):
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-3", "results": []})
    )

    await pp.search("topic", limit=3)

    payload = _json.loads(route.calls.last.request.content)
    assert payload["max_results"] == 3


@respx.mock
async def test_search_recency_sends_search_recency_filter(_search_api_key):
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-r", "results": []})
    )

    await pp.search("topic", recency="week")

    payload = _json.loads(route.calls.last.request.content)
    assert payload["search_recency_filter"] == "week"


@respx.mock
async def test_search_recency_all_omits_filter(_search_api_key):
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-r2", "results": []})
    )

    await pp.search("topic", recency="all")

    payload = _json.loads(route.calls.last.request.content)
    assert "search_recency_filter" not in payload


@respx.mock
async def test_search_empty_results_returns_empty_list(_search_api_key):
    respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-4", "results": []})
    )

    results = await pp.search("topic with no hits")
    assert results == []


async def test_search_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PERPLEXITY_API_KEY"):
        await pp.search("topic", api_key=None)


@respx.mock
async def test_search_uses_explicit_api_key_over_env(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "sk-env-key")
    route = respx.post("https://api.perplexity.ai/search").mock(
        return_value=httpx.Response(200, json={"id": "req-5", "results": []})
    )

    await pp.search("topic", api_key="sk-explicit-key")

    assert route.calls.last.request.headers["authorization"] == "Bearer sk-explicit-key"


# -----------------------------------------------------------------------------
# PerplexityProvider (ResearchProvider protocol) → LayerOutput
# -----------------------------------------------------------------------------


def test_provider_name_and_protocol():
    provider = pp.PerplexityProvider(Settings(perplexity_api_key="sk-x"))
    assert provider.name == "perplexity"
    assert isinstance(provider, ResearchProvider)


async def test_provider_research_maps_results_to_layer_output(monkeypatch):
    async def _fake_research(topic, **kwargs):
        return [
            pp.PerplexityResult(
                question="q1",
                answer="a1",
                citations=[
                    pp.Citation(url="https://example.gov/a", domain="example.gov"),
                    pp.Citation(url="https://other.com/b", domain="other.com"),
                ],
                model="sonar-pro",
                search_results=[],
                tokens_input=10,
                tokens_output=20,
                cost_usd=0.01,
                duration_ms=5,
            ),
            pp.PerplexityResult(
                question="q2",
                answer="a2",
                citations=[
                    # duplicate URL should be de-duped across sub-questions
                    pp.Citation(url="https://example.gov/a", domain="example.gov"),
                ],
                model="sonar-pro",
                search_results=[],
                tokens_input=5,
                tokens_output=5,
                cost_usd=0.02,
                duration_ms=7,
            ),
        ]

    monkeypatch.setattr(pp, "research", _fake_research)
    provider = pp.PerplexityProvider(Settings(perplexity_api_key="sk-x"))
    layer = await provider.research("topic", sub_questions=2, depth="standard")

    assert isinstance(layer, LayerOutput)
    assert layer.layer == "research"
    assert layer.error is None
    assert [r.url for r in layer.results] == [
        "https://example.gov/a",
        "https://other.com/b",
    ]
    assert all(r.layer == "research" for r in layer.results)
    assert layer.cost_usd == pytest.approx(0.03)


async def test_provider_research_all_errors_surfaces_error(monkeypatch):
    async def _fake_research(topic, **kwargs):
        return [
            pp.PerplexityResult(
                question="q1",
                answer="",
                citations=[],
                model="sonar-pro",
                search_results=[],
                tokens_input=0,
                tokens_output=0,
                cost_usd=0.0,
                duration_ms=0,
                error="ValueError: boom",
            )
        ]

    monkeypatch.setattr(pp, "research", _fake_research)
    provider = pp.PerplexityProvider(Settings(perplexity_api_key="sk-x"))
    layer = await provider.research("topic", sub_questions=1, depth="quick")
    assert layer.results == []
    assert layer.error == "ValueError: boom"


# -----------------------------------------------------------------------------
# build_providers wiring
# -----------------------------------------------------------------------------


def test_build_providers_wires_perplexity_when_key_present():
    from polysearch.providers.base import build_providers

    providers = build_providers(Settings(perplexity_api_key="sk-x"))
    assert isinstance(providers.research, pp.PerplexityProvider)
    assert providers.research.name == "perplexity"
