"""Opt-in Deep Research layer: sonar-deep-research via the async Sonar API.

Verified against docs.perplexity.ai + live API (2026-07-13): sonar-deep-research
is a chat-completions model, not an Agent model. The ``/v1/agent`` endpoint
rejects it with ``400 model "sonar-deep-research" is not supported``. Async
submission is ``POST /v1/async/sonar`` with a wrapped ``{"request": {...}}`` body
containing a standard chat-completions payload; it returns immediately with an
``id`` + ``status`` ("CREATED"). Poll ``GET /v1/async/sonar/{id}`` until a
terminal status; the finished chat completion arrives under the ``response``
field.

The module-level ``research`` returns a ``PerplexityResult`` (reusing the
Perplexity types) so deep-research output feeds the same tier-tagging,
claim-extraction, citation-verification, and cost-accounting path as every other
Perplexity layer. ``DeepResearchProvider`` adapts it to the ``ResearchProvider``
protocol, returning a ``LayerOutput`` of ``SourceResult`` (one per cited source),
so a synthesizer/verifier covers deep-research citations like any other layer.

Opt-in and paid. Activates only when ``enable_deep_research`` and
``perplexity_api_key`` are both set. sonar-deep-research bills on four meters:
  - input/output tokens: $2 / $8 per 1M
  - citation tokens:      $2 per 1M
  - reasoning tokens:     $3 per 1M
  - search queries:       $5 per 1K
The live API also returns its own ``usage.cost.total_cost``, which is preferred
when present (the four-meter formula is the fallback).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from polysearch import ratelimit
from polysearch.config import Settings
from polysearch.output.schema import LayerOutput, SourceResult
from polysearch.providers.perplexity import Citation, PerplexityResult, _domain_of

SUBMIT_URL = "https://api.perplexity.ai/v1/async/sonar"
MODEL = "sonar-deep-research"

DEFAULT_POLL_INTERVAL = 15.0
DEFAULT_TIMEOUT = 3600.0

# Async job statuses are UPPERCASE on the live API; match case-insensitively.
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "INCOMPLETE"}

# $ per 1M tokens, except "search" which is $ per 1K queries.
_PRICING = {
    "input": 2.0,
    "output": 8.0,
    "citation": 2.0,
    "reasoning": 3.0,
    "search": 5.0,
}

# Test-overridable hook (monkeypatched to avoid real sleeping between polls).
_SLEEP_FN = asyncio.sleep


def _get_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY not set (checked arg and environment)")
    return key


def _estimate_cost(usage: dict[str, Any]) -> float:
    """Prefer the API-reported total; fall back to the four-meter formula.

    Chat-completions usage uses ``prompt_tokens``/``completion_tokens`` (not the
    Agent API's ``input_tokens``/``output_tokens``).
    """
    cost = usage.get("cost")
    if isinstance(cost, dict) and cost.get("total_cost") is not None:
        return float(cost["total_cost"])

    tokens_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    tokens_out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    citation_tokens = usage.get("citation_tokens") or 0
    reasoning_tokens = usage.get("reasoning_tokens") or 0
    search_queries = usage.get("num_search_queries") or 0
    return (
        (tokens_in / 1_000_000) * _PRICING["input"]
        + (tokens_out / 1_000_000) * _PRICING["output"]
        + (citation_tokens / 1_000_000) * _PRICING["citation"]
        + (reasoning_tokens / 1_000_000) * _PRICING["reasoning"]
        + (search_queries / 1_000) * _PRICING["search"]
    )


def _extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant message text from the chat-completion ``choices`` array."""
    for choice in response.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        text = message.get("content")
        if text:
            return text.strip()
    return ""


def _extract_citations(response: dict[str, Any]) -> list[Citation]:
    """Parse citations from the completion's ``search_results[]`` (richest
    metadata) merged with the bare-URL ``citations[]`` array."""
    by_url: dict[str, Citation] = {}

    def _add(
        url: str | None,
        title: str | None = None,
        snippet: str | None = None,
        date: str | None = None,
    ) -> None:
        if not url:
            return
        existing = by_url.get(url)
        if existing is None:
            by_url[url] = Citation(
                url=url,
                title=title,
                snippet=snippet,
                published_date=date,
                domain=_domain_of(url),
            )
        elif title and not existing.title:
            existing.title = title

    for sr in response.get("search_results") or []:
        if isinstance(sr, dict):
            _add(sr.get("url"), sr.get("title"), sr.get("snippet"), sr.get("date"))

    for url in response.get("citations") or []:
        if isinstance(url, str):
            _add(url)

    return list(by_url.values())


def _error_result(topic: str, model: str, duration_ms: int, error: str) -> PerplexityResult:
    return PerplexityResult(
        question=topic,
        answer="",
        citations=[],
        model=model,
        search_results=[],
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
        duration_ms=duration_ms,
        error=error,
    )


async def research(
    topic: str,
    *,
    api_key: str | None = None,
    model: str = MODEL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: float = DEFAULT_TIMEOUT,
) -> PerplexityResult:
    """Submit ``topic`` to sonar-deep-research and poll until terminal.

    Never raises on submission/poll/timeout failure — always returns a
    ``PerplexityResult`` with ``.error`` set, so callers can isolate this layer
    without aborting the run (matches ``perplexity._run_one``).
    """
    key = _get_api_key(api_key)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"request": {"model": model, "messages": [{"role": "user", "content": topic}]}}
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Submit is the only call charged against the shared 50rpm bucket.
            # Polls are cheap GETs issued every ~15s; running them through
            # ``acquire`` would starve sibling submitters, so they are exempt.
            async with ratelimit.acquire("perplexity"):
                resp = await client.post(SUBMIT_URL, headers=headers, json=body)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                ratelimit.record_429("perplexity", ratelimit.parse_retry_after(retry_after))
            resp.raise_for_status()
            data = resp.json()
            job_id = data["id"]

            elapsed = 0.0
            while (data.get("status") or "").upper() not in _TERMINAL_STATUSES:
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"deep research poll budget ({timeout:.0f}s) exceeded for "
                        f"job {job_id} (last status={data.get('status')!r})"
                    )
                await _SLEEP_FN(poll_interval)
                elapsed += poll_interval
                poll_resp = await client.get(f"{SUBMIT_URL}/{job_id}", headers=headers)
                if poll_resp.status_code == 429:
                    retry_after = poll_resp.headers.get("retry-after")
                    ratelimit.record_429("perplexity", ratelimit.parse_retry_after(retry_after))
                    continue
                poll_resp.raise_for_status()
                data = poll_resp.json()
        except Exception as exc:  # noqa: BLE001 — surface as structured error, never raise
            duration_ms = int((time.perf_counter() - start) * 1000)
            return _error_result(topic, model, duration_ms, f"{type(exc).__name__}: {exc}")

    duration_ms = int((time.perf_counter() - start) * 1000)
    status = (data.get("status") or "").upper()
    if status != "COMPLETED":
        detail = data.get("error_message") or f"status={data.get('status')!r}"
        return _error_result(topic, model, duration_ms, f"deep research ended: {detail}")

    response = data.get("response") or {}
    usage = response.get("usage") or {}
    return PerplexityResult(
        question=topic,
        answer=_extract_text(response),
        citations=_extract_citations(response),
        model=model,
        search_results=response.get("search_results") or [],
        tokens_input=int(usage.get("prompt_tokens") or 0),
        tokens_output=int(usage.get("completion_tokens") or 0),
        cost_usd=_estimate_cost(usage),
        duration_ms=duration_ms,
        error=None,
    )


# ── Provider ─────────────────────────────────────────────────────────────────


class DeepResearchProvider:
    """Adapts the sonar-deep-research async path to the ``ResearchProvider`` protocol.

    Active only when ``enable_deep_research`` and ``perplexity_api_key`` are both
    set; ``reason`` names what is missing when inactive (mirroring the
    reason-carrying null providers in ``base``). Depth-eligibility gating (the
    ``deep`` profile's ``deep_research_eligible`` flag, or a ``--deep-research``
    force) is the orchestrator's job — this provider only owns credential/flag
    activation. Backend-agnostic: an OpenAI deep-research backend could slot in
    behind the same class later.
    """

    name = "deep_research"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.enable_deep_research:
            self.active = False
            self.reason: str | None = "deep research disabled (enable_deep_research is false)"
        elif not settings.perplexity_api_key:
            self.active = False
            self.reason = "PERPLEXITY_API_KEY not set"
        else:
            self.active = True
            self.reason = None

    async def research(
        self, topic: str, *, sub_questions: int, depth: str
    ) -> LayerOutput:
        # sonar-deep-research runs the whole topic as one long-form job — it does
        # its own internal decomposition, so ``sub_questions``/``depth`` are not
        # forwarded (they gate whether this layer runs, upstream).
        if not self.active:
            return LayerOutput(layer=self.name, error=self.reason)

        start = time.perf_counter()
        result = await research(
            topic,
            api_key=self.settings.perplexity_api_key,
            model=self.settings.deep_research_model,
            timeout=float(self.settings.deep_research_timeout_s),
        )

        sources: list[SourceResult] = []
        seen: set[str] = set()
        for c in result.citations:
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
                    layer=self.name,
                    engagement=None,
                )
            )

        duration_ms = int((time.perf_counter() - start) * 1000)
        error = result.error if (not sources and result.error) else None
        return LayerOutput(
            layer=self.name,
            results=sources,
            cost_usd=result.cost_usd,
            duration_ms=duration_ms,
            error=error,
            # The deep-research narrative is where its figures live; carry it so
            # claim extraction + verification see the long-form answer, not just
            # the bare citations.
            answers=[result.answer] if result.answer else [],
        )


__all__ = [
    "DeepResearchProvider",
    "research",
    "MODEL",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_TIMEOUT",
]
