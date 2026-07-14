"""Perplexity Sonar research provider.

Two entry points into the Perplexity API:

- ``research`` — the chat-completions Sonar path. Decomposes a topic into
  sub-questions, fires them in parallel via the OpenAI SDK (pointed at
  ``api.perplexity.ai`` with a custom ``base_url``), parses citations, tracks
  cost, and strips ``sonar-reasoning-pro`` ``<think>`` chain-of-thought blocks.
  Supports current-API features: ``search_domain_filter`` (allow/deny lists),
  ``search_recency_filter``, and structured outputs (``response_format``).
- ``search`` — the flat-rate Search API. Raw ranked results, no LLM synthesis;
  the default discovery backend for web grounding.

``PerplexityProvider`` adapts ``research`` to the ``ResearchProvider`` protocol,
returning a ``LayerOutput``. The module-level ``research``/``search`` functions
keep a signature compatible with the internal pipeline so ported callers (the
recovery pass, coverage evaluator) transplant without edits — the new params
(``domain_filter``, ``response_schema``, ``model``/``deep_model``) are optional.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from polysearch import ratelimit
from polysearch.config import Settings
from polysearch.output.schema import LayerOutput, SourceResult

# ── Rate-limit seam ──────────────────────────────────────────────────────────
# Every outbound Perplexity call funnels through this single seam so one
# integration point governs cross-process pacing (see ``polysearch.ratelimit``).
# Referencing ``ratelimit.acquire`` by attribute at call time keeps it patchable
# in tests.


def _acquire(provider: str = "perplexity"):
    return ratelimit.acquire(provider)


# ── <think> block stripping ──────────────────────────────────────────────────

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove sonar-reasoning-pro chain-of-thought blocks from response text."""
    return _THINK_BLOCK_RE.sub("", text).strip()


# ── Types / contracts ────────────────────────────────────────────────────────

Tier = Literal["HIGH", "MEDIUM", "LOW", "COMMUNITY", "SME", "UNKNOWN"]
Depth = Literal["quick", "standard", "deep"]
Recency = Literal["hour", "day", "week", "month", "year", "all"]
SearchMode = Literal["web", "academic"]


class Citation(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None
    published_date: str | None = None  # YYYY-MM-DD
    domain: str
    tier: Tier = "UNKNOWN"


class PerplexityResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model: str
    search_results: list[dict]
    tokens_input: int
    tokens_output: int
    cost_usd: float
    duration_ms: int
    error: str | None = None
    reasoning_trace: str | None = None


# ── Constants ────────────────────────────────────────────────────────────────

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_SEARCH_URL = f"{PERPLEXITY_BASE_URL}/search"

# Search API is flat-rate, not token-metered — for callers tallying spend
# alongside ``PerplexityResult.cost_usd``.
SEARCH_API_COST_USD = 0.005  # $5 / 1,000 requests

_MODEL_SONAR_PRO = "sonar-pro"
_MODEL_SONAR_REASONING_PRO = "sonar-reasoning-pro"

# Published Perplexity pricing ($ per 1M tokens), used as the fallback table.
# ``Settings`` now carries the primary model's per-token pricing
# (``perplexity_price_in``/``perplexity_price_out``, defaults matching sonar-pro
# below), which ``PerplexityProvider`` threads in per run via ``_estimate_cost``'s
# ``pricing`` override. The deep/reasoning model retains this table's rates.
_PRICING: dict[str, dict[str, float]] = {
    _MODEL_SONAR_PRO: {"input": 3.0, "output": 15.0, "search": 0.0},
    _MODEL_SONAR_REASONING_PRO: {"input": 2.0, "output": 8.0, "search": 5.0},
}

_DEPTH_SUBQ_DEFAULT: dict[Depth, int] = {"quick": 2, "standard": 4, "deep": 6}
_DEPTH_CONTEXT_SIZE: dict[Depth, str] = {
    "quick": "low",
    "standard": "medium",
    "deep": "high",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _domain_of(url: str) -> str:
    """Return lowercased netloc with leading 'www.' stripped."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _model_for_depth(depth: Depth) -> str:
    return _MODEL_SONAR_REASONING_PRO if depth == "deep" else _MODEL_SONAR_PRO


def _map_recency(recency: str) -> str | None:
    """Map extended recency values to Perplexity-supported ones.

    "quarter" -> "month"; "all" -> None (omit filter).
    """
    if recency == "all":
        return None
    if recency == "quarter":
        return "month"
    return recency


def _estimate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    rates = (pricing or {}).get(model) or _PRICING.get(
        model, {"input": 0.0, "output": 0.0, "search": 0.0}
    )
    search_cost = rates.get("search", 0.0) / 1_000_000  # per-query approximation
    return (
        (tokens_in / 1_000_000) * rates["input"]
        + (tokens_out / 1_000_000) * rates["output"]
        + search_cost
    )


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_of(exc: BaseException) -> float | None:
    """Pull a Retry-After (seconds) off an HTTP error's response headers."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    raw = headers.get("retry-after") if headers else None
    return ratelimit.parse_retry_after(raw)


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 + 5xx. Inspect status if present, else fall back on message."""
    status = _status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    msg = str(exc)
    return "429" in msg or "rate limit" in msg.lower() or "5" in msg[:4]


def _parse_citations(
    urls: list[str] | None,
    search_results: list[dict] | None,
) -> list[Citation]:
    """Merge top-level ``citations[]`` URLs with ``search_results[]`` dicts."""
    by_url: dict[str, Citation] = {}
    for sr in search_results or []:
        if not isinstance(sr, dict):
            continue
        url = sr.get("url")
        if not url:
            continue
        by_url[url] = Citation(
            url=url,
            title=sr.get("title"),
            snippet=sr.get("snippet"),
            published_date=sr.get("date"),
            domain=_domain_of(url),
        )
    for url in urls or []:
        if not url or url in by_url:
            continue
        by_url[url] = Citation(url=url, domain=_domain_of(url))
    return list(by_url.values())


def _build_response_format(schema: dict) -> dict:
    """Wrap a JSON Schema into Perplexity's structured-output ``response_format``.

    Never put URL fields in ``schema`` — models fabricate URLs when asked to
    produce them directly inside structured output. Real URLs must come from the
    response's ``citations``/``search_results`` fields instead.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema},
    }


def _get_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY not set (checked arg and environment)")
    return key


def _make_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)


# ── Sub-question decomposition ───────────────────────────────────────────────

_DECOMPOSE_PROMPT = (
    "Decompose this topic into {n} specific, searchable sub-questions. "
    "Return ONLY a JSON array of strings, no prose, no markdown fences. "
    "Topic: {topic}"
)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _extract_json_array(text: str) -> list[str] | None:
    """Pull the first JSON array out of possibly-noisy text."""
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out = [str(item).strip() for item in data if str(item).strip()]
    return out or None


async def _decompose(client: AsyncOpenAI, topic: str, n: int) -> list[str]:
    """Break the topic into n searchable sub-questions; fall back to [topic]."""
    try:
        async with _acquire():
            resp = await client.chat.completions.create(
                model=_MODEL_SONAR_PRO,
                messages=[
                    {"role": "user", "content": _DECOMPOSE_PROMPT.format(n=n, topic=topic)}
                ],
                extra_body={"search_context_size": "low"},
            )
        content = resp.choices[0].message.content or ""
        parsed = _extract_json_array(content)
        if parsed:
            return parsed[:n]
    except Exception:  # noqa: BLE001 — decomposition is best-effort
        pass
    return [topic]


# ── Single sub-question call ─────────────────────────────────────────────────


async def _run_one(
    client: AsyncOpenAI,
    question: str,
    *,
    model: str,
    recency: str | None,
    domain_filter: list[str] | None,
    search_mode: SearchMode,
    context_size: str,
    response_schema: dict | None = None,
    pricing: dict[str, dict[str, float]] | None = None,
) -> PerplexityResult:
    extra: dict[str, Any] = {
        "search_context_size": context_size,
        "search_mode": search_mode,
    }
    if recency is not None:
        extra["search_recency_filter"] = recency
    if domain_filter:
        extra["search_domain_filter"] = domain_filter
    if response_schema is not None:
        extra["response_format"] = _build_response_format(response_schema)

    start = time.perf_counter()
    retryer = AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )

    try:
        async for attempt in retryer:
            with attempt:
                async with _acquire():
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": question}],
                            extra_body=extra,
                        )
                    except Exception as exc:  # noqa: BLE001 — record then re-raise
                        # Broadcast a 429 to sibling processes via the shared
                        # ledger before tenacity handles the retry/backoff.
                        if _status_code(exc) == 429:
                            ratelimit.record_429("perplexity", _retry_after_of(exc))
                        raise
    except Exception as exc:  # noqa: BLE001 — surface as structured error
        duration_ms = int((time.perf_counter() - start) * 1000)
        return PerplexityResult(
            question=question,
            answer="",
            citations=[],
            model=model,
            search_results=[],
            tokens_input=0,
            tokens_output=0,
            cost_usd=0.0,
            duration_ms=duration_ms,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration_ms = int((time.perf_counter() - start) * 1000)

    # Tolerate both SDK objects and dicts.
    raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    choices = raw.get("choices") or []
    raw_answer = ""
    if choices:
        msg = choices[0].get("message") or {}
        raw_answer = msg.get("content") or ""
    answer = strip_think_blocks(raw_answer)

    citations_urls = raw.get("citations") or []
    search_results = raw.get("search_results") or []
    citations = _parse_citations(citations_urls, search_results)

    usage = raw.get("usage") or {}
    tokens_in = int(usage.get("prompt_tokens") or 0)
    tokens_out = int(usage.get("completion_tokens") or 0)
    cost = _estimate_cost(model, tokens_in, tokens_out, pricing)

    return PerplexityResult(
        question=question,
        answer=answer,
        citations=citations,
        model=model,
        search_results=search_results if isinstance(search_results, list) else [],
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        cost_usd=cost,
        duration_ms=duration_ms,
        error=None,
        reasoning_trace=raw_answer if raw_answer != answer else None,
    )


# ── Public functions ─────────────────────────────────────────────────────────


async def research(
    topic: str,
    *,
    depth: Depth = "standard",
    sub_questions: int | None = None,
    recency: Recency = "month",
    domain_filter: list[str] | None = None,
    search_mode: SearchMode = "web",
    response_schema: dict | None = None,
    api_key: str | None = None,
    model: str | None = None,
    deep_model: str | None = None,
    pricing: dict[str, dict[str, float]] | None = None,
) -> list[PerplexityResult]:
    """Decompose topic into sub-questions, fire them in parallel, return results.

    ``model``/``deep_model`` override the built-in Sonar model names (the
    ``PerplexityProvider`` passes them from ``Settings``); when omitted, the
    depth-appropriate default is used. ``pricing`` (model → per-1M-token rates)
    overrides the module ``_PRICING`` table for cost estimation, letting the
    provider source rates from ``Settings``; models absent from it fall back to
    the module table.
    """
    key = _get_api_key(api_key)
    client = _make_client(key)

    n = sub_questions if sub_questions is not None else _DEPTH_SUBQ_DEFAULT[depth]
    n = max(1, n)

    default_model = _model_for_depth(depth)
    chosen_model = (deep_model if depth == "deep" else model) or default_model

    if n == 1:
        sub_qs = [topic]
    else:
        sub_qs = await _decompose(client, topic, n)

    context_size = _DEPTH_CONTEXT_SIZE[depth]
    mapped_recency = _map_recency(recency)

    coros = [
        _run_one(
            client,
            sq,
            model=chosen_model,
            recency=mapped_recency,
            domain_filter=domain_filter,
            search_mode=search_mode,
            context_size=context_size,
            response_schema=response_schema,
            pricing=pricing,
        )
        for sq in sub_qs
    ]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    results: list[PerplexityResult] = []
    for sq, item in zip(sub_qs, raw_results, strict=True):
        if isinstance(item, BaseException):
            results.append(
                PerplexityResult(
                    question=sq,
                    answer="",
                    citations=[],
                    model=chosen_model,
                    search_results=[],
                    tokens_input=0,
                    tokens_output=0,
                    cost_usd=0.0,
                    duration_ms=0,
                    error=f"{type(item).__name__}: {item}",
                )
            )
        else:
            results.append(item)
    return results


async def search(
    query: str,
    *,
    limit: int = 10,
    recency: str | None = None,
    api_key: str | None = None,
) -> list[SourceResult]:
    """Query the Perplexity Search API for raw ranked results (no LLM synthesis).

    A flat-rate endpoint ($5 / 1,000 requests — see ``SEARCH_API_COST_USD``),
    distinct from the chat-completions Sonar path used by ``research``. Results
    map into ``SourceResult`` objects tagged ``layer="grounding"`` (tier
    ``UNKNOWN`` — classification happens downstream).

    ``recency`` (hour/day/week/month/year; "all"/"quarter" normalized via
    ``_map_recency``) is passed through as the Search API's
    ``search_recency_filter`` so grounding discovery can time-window results;
    omit or pass "all" to disable it.
    """
    key = _get_api_key(api_key)
    body: dict[str, Any] = {"query": query, "max_results": limit}
    mapped_recency = _map_recency(recency) if recency is not None else None
    if mapped_recency is not None:
        body["search_recency_filter"] = mapped_recency
    async with httpx.AsyncClient() as client:
        async with _acquire():
            resp = await client.post(
                PERPLEXITY_SEARCH_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
        resp.raise_for_status()
        data = resp.json()

    out: list[SourceResult] = []
    for r in data.get("results") or []:
        if not isinstance(r, dict):
            continue
        url = r.get("url")
        if not url:
            continue
        out.append(
            SourceResult(
                url=url,
                title=r.get("title") or "",
                snippet=r.get("snippet") or "",
                tier="UNKNOWN",
                published_date=r.get("date"),
                layer="grounding",
                engagement=None,
            )
        )
    return out


# ── Provider ─────────────────────────────────────────────────────────────────


class PerplexityProvider:
    """Adapts the Sonar ``research`` path to the ``ResearchProvider`` protocol."""

    name = "perplexity"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def research(
        self, topic: str, *, sub_questions: int, depth: str
    ) -> LayerOutput:
        start = time.perf_counter()
        # Source the primary model's per-token pricing from Settings; the deep
        # model keeps the module ``_PRICING`` table (single price pair in
        # Settings is matched to ``perplexity_model``).
        pricing = {
            self.settings.perplexity_model: {
                "input": self.settings.perplexity_price_in,
                "output": self.settings.perplexity_price_out,
                "search": 0.0,
            }
        }
        results = await research(
            topic,
            depth=depth,  # type: ignore[arg-type]
            sub_questions=sub_questions,
            api_key=self.settings.perplexity_api_key,
            model=self.settings.perplexity_model,
            deep_model=self.settings.perplexity_deep_model,
            pricing=pricing,
        )

        sources: list[SourceResult] = []
        seen: set[str] = set()
        for r in results:
            for c in r.citations:
                if c.url in seen:
                    continue
                seen.add(c.url)
                sources.append(
                    SourceResult(
                        url=c.url,
                        title=c.title or "",
                        snippet=c.snippet or "",
                        tier=c.tier,
                        published_date=c.published_date,
                        layer="research",
                        engagement=None,
                    )
                )

        cost = sum(r.cost_usd for r in results)
        duration_ms = int((time.perf_counter() - start) * 1000)
        errors = [r.error for r in results if r.error]
        error = errors[0] if (not sources and errors) else None

        return LayerOutput(
            layer="research",
            results=sources,
            cost_usd=cost,
            duration_ms=duration_ms,
            error=error,
        )


__all__ = [
    "Citation",
    "PerplexityProvider",
    "PerplexityResult",
    "research",
    "search",
    "strip_think_blocks",
]
