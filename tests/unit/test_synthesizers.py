"""Unit tests for polysearch.providers.synthesizers.

Covers the shared prompt builder and both concrete synthesizers. All SDK I/O is
mocked (AsyncOpenAI / AsyncAnthropic monkeypatched) so the suite runs with zero
env vars. The humanizer-absence test is mandatory: the public prompt must NOT
carry the internal FORBIDDEN-STYLE constraints, and MUST carry
``style_constraints`` when the deployment sets one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polysearch.config import Settings
from polysearch.output.schema import LayerOutput, SourceResult
from polysearch.providers import synthesizers as syn
from polysearch.providers.base import Synthesizer
from polysearch.providers.synthesizers import (
    AnthropicSynthesizer,
    OpenAISynthesizer,
    build_synthesis_prompt,
)


def _layers() -> list[LayerOutput]:
    return [
        LayerOutput(
            layer="research",
            results=[
                SourceResult(
                    url="https://example.gov/report",
                    title="Annual Sector Report",
                    snippet="Adoption climbed 42% across the quarter.",
                    tier="HIGH",
                    published_date="2026-05-01",
                    layer="research",
                ),
            ],
        ),
        LayerOutput(
            layer="grounding",
            results=[
                SourceResult(
                    url="https://blog.example.com/take",
                    title="One Team's Take",
                    snippet="A smaller shop saw slower gains.",
                    tier="COMMUNITY",
                    layer="grounding",
                ),
            ],
        ),
    ]


# ── Prompt builder ───────────────────────────────────────────────────────────


def test_prompt_has_three_section_headers():
    prompt = build_synthesis_prompt("topic", _layers(), style_constraints=None)
    assert "## Executive Summary" in prompt
    assert "## Key Findings" in prompt
    assert "## Source Quality Notes" in prompt


def test_prompt_consumes_layer_material():
    prompt = build_synthesis_prompt("topic", _layers(), style_constraints=None)
    # snippet + tier + published_date must reach the model.
    assert "Adoption climbed 42% across the quarter." in prompt
    assert "HIGH" in prompt
    assert "2026-05-01" in prompt
    assert "COMMUNITY" in prompt


def test_prompt_has_no_humanizer_phrases():
    """Mandatory: the private FORBIDDEN-STYLE block must never surface."""
    prompt = build_synthesis_prompt(
        "topic", _layers(), style_constraints="Prefer active voice."
    ).lower()
    for banned in (
        "kill shot",
        "em-dash",
        "em-dashes",
        "punchline",
        "the conversation",
        "the moment",
        "at the end of the day",
        "staccato",
        "forbidden style",
    ):
        assert banned not in prompt, f"leaked humanizer phrase: {banned}"
    assert "write in plain, direct prose." in prompt


def test_prompt_appends_style_constraints_when_set():
    prompt = build_synthesis_prompt(
        "topic", _layers(), style_constraints="Prefer active voice."
    )
    assert "Prefer active voice." in prompt


def test_prompt_omits_style_constraints_when_unset():
    prompt = build_synthesis_prompt("topic", _layers(), style_constraints=None)
    assert "Prefer active voice." not in prompt


# ── OpenAISynthesizer ────────────────────────────────────────────────────────


def _openai_msg(content: str, prompt_tokens: int, completion_tokens: int):
    msg = MagicMock()
    msg.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    msg.usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return msg


def _install_openai(monkeypatch, msg):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=msg)
    monkeypatch.setattr(syn, "AsyncOpenAI", lambda *a, **k: client)
    return client


def test_openai_conforms_to_protocol():
    s = OpenAISynthesizer(Settings(openai_api_key="sk-test"))
    assert isinstance(s, Synthesizer)


@pytest.mark.asyncio
async def test_openai_returns_markdown_and_cost(monkeypatch):
    settings = Settings(
        openai_api_key="sk-test",
        synthesis_model="gpt-5.4-mini",
        synthesis_price_in=0.75,
        synthesis_price_out=4.50,
    )
    md = "## Executive Summary\nAll good.\n\n## Key Findings\n- One [HIGH: example.gov, 2026-05-01]\n\n## Source Quality Notes\n- Thin community coverage."
    client = _install_openai(
        monkeypatch, _openai_msg(md, prompt_tokens=1000, completion_tokens=2000)
    )

    markdown, cost = await OpenAISynthesizer(settings).synthesize(
        "topic", _layers(), style_constraints=None
    )

    assert markdown == md
    # (1000 * 0.75 + 2000 * 4.50) / 1e6 = 0.00975
    assert cost == pytest.approx(0.00975)
    call = client.chat.completions.create.call_args.kwargs
    assert call["model"] == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_openai_style_constraints_reach_prompt(monkeypatch):
    settings = Settings(openai_api_key="sk-test")
    client = _install_openai(
        monkeypatch, _openai_msg("## Executive Summary\nx", 10, 10)
    )
    await OpenAISynthesizer(settings).synthesize(
        "topic", _layers(), style_constraints="Prefer active voice."
    )
    sent_prompt = client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert "Prefer active voice." in sent_prompt


@pytest.mark.asyncio
async def test_openai_zero_cost_when_usage_missing(monkeypatch):
    settings = Settings(openai_api_key="sk-test")
    msg = MagicMock()
    msg.choices = [SimpleNamespace(message=SimpleNamespace(content="## Executive Summary\nx"))]
    msg.usage = None
    _install_openai(monkeypatch, msg)
    _, cost = await OpenAISynthesizer(settings).synthesize(
        "topic", _layers(), style_constraints=None
    )
    assert cost == 0.0


# ── AnthropicSynthesizer ─────────────────────────────────────────────────────


def _anthropic_msg(text: str, input_tokens: int, output_tokens: int):
    msg = MagicMock()
    msg.content = [SimpleNamespace(text=text)]
    msg.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


def _install_anthropic(monkeypatch, msg):
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=msg)
    monkeypatch.setattr(syn, "AsyncAnthropic", lambda *a, **k: client)
    return client


def test_anthropic_conforms_to_protocol():
    s = AnthropicSynthesizer(Settings(anthropic_api_key="sk-ant"))
    assert isinstance(s, Synthesizer)


@pytest.mark.asyncio
async def test_anthropic_returns_markdown_and_cost(monkeypatch):
    settings = Settings(anthropic_api_key="sk-ant")
    md = "## Executive Summary\nAll good.\n\n## Key Findings\n- One [HIGH]\n\n## Source Quality Notes\n- Note."
    client = _install_anthropic(
        monkeypatch, _anthropic_msg(md, input_tokens=1000, output_tokens=2000)
    )

    markdown, cost = await AnthropicSynthesizer(settings).synthesize(
        "topic", _layers(), style_constraints=None
    )

    assert markdown == md
    # (1000 * 3.0 + 2000 * 15.0) / 1e6 = 0.033
    assert cost == pytest.approx(0.033)
    call = client.messages.create.call_args.kwargs
    assert call["model"] == "claude-sonnet-5"


@pytest.mark.asyncio
async def test_anthropic_joins_multiple_text_blocks(monkeypatch):
    settings = Settings(anthropic_api_key="sk-ant")
    msg = MagicMock()
    msg.content = [SimpleNamespace(text="## Executive Summary\n"), SimpleNamespace(text="Body.")]
    msg.usage = SimpleNamespace(input_tokens=10, output_tokens=10)
    _install_anthropic(monkeypatch, msg)
    markdown, _ = await AnthropicSynthesizer(settings).synthesize(
        "topic", _layers(), style_constraints=None
    )
    assert markdown == "## Executive Summary\nBody."
