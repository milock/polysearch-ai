"""OpenAI + Anthropic synthesizers.

Fuse the pipeline's layer outputs into a single cited markdown report, adapting
the internal cross-layer synthesis to the public :class:`Synthesizer` protocol
(``synthesize(topic, layers, *, style_constraints) -> (markdown, cost_usd)``).
Both providers share one prompt builder and one output contract — Executive
Summary / Key Findings / Source Quality Notes — and differ only in SDK and
pricing.

Two intentional departures from the internal port:

- **No humanizer constraints.** The internal FORBIDDEN-STYLE block is private
  and is NOT ported. The public prompt asks for plain, direct prose and appends
  the deployment's ``style_constraints`` when one is set.
- **No rate-limit seam.** ``polysearch.ratelimit`` carries no bucket for the
  OpenAI/Anthropic synthesis models, and synthesis is only 1-2 calls per run, so
  these calls are not routed through ``ratelimit.acquire``. Add a bucket there
  first if that ever changes.

The SDK classes are imported at module top (both are hard dependencies) and
referenced as module attributes so tests can monkeypatch them.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from polysearch.config import Settings
from polysearch.output.schema import LayerOutput

__all__ = ["OpenAISynthesizer", "AnthropicSynthesizer", "build_synthesis_prompt"]

# Anthropic default model + pricing ($ per 1M tokens). Kept as module constants
# because Settings only carries the OpenAI synthesis price pair
# (``synthesis_price_in``/``synthesis_price_out``); update here if Anthropic
# pricing or the default Sonnet model changes.
_ANTHROPIC_MODEL = "claude-sonnet-5"
_ANTHROPIC_PRICE_IN = 3.0
_ANTHROPIC_PRICE_OUT = 15.0

_MAX_TOKENS = 4000


def build_synthesis_prompt(
    topic: str,
    layers: list[LayerOutput],
    *,
    style_constraints: str | None = None,
) -> str:
    """Build the synthesis user prompt from ``topic`` + layer material.

    Each source is rendered tier-labeled with its published date and snippet
    (mirroring the internal ``snippet or markdown[:600]`` consumption), grouped
    under its layer. ``style_constraints`` is appended verbatim when set.
    """
    parts = [
        f"Research topic: {topic}",
        "",
        "You are synthesizing a research report from multiple search layers.",
        "Output exactly three markdown sections, using these headers verbatim:",
        "",
        "## Executive Summary",
        "2-4 sentences on the single most important thing someone preparing for a "
        "decision should know.",
        "",
        "## Key Findings",
        "3-7 bullet points. End each with a source tag like [HIGH: domain, date], "
        "[COMMUNITY], or [SME].",
        "",
        "## Source Quality Notes",
        "2-5 bullets on conflicts, gaps, or source-tier concentration risks. Be "
        "brief and specific.",
        "",
        "Ground every claim in the material below. Do not introduce facts not "
        "present in the input.",
        "Write in plain, direct prose.",
    ]
    if style_constraints:
        parts.append(style_constraints)
    parts.append("\n---")

    for layer in layers:
        if not layer.results:
            continue
        parts.append(f"\n## Layer: {layer.layer} ({len(layer.results)} sources)")
        for src in layer.results:
            date = src.published_date or "undated"
            snippet = (src.snippet or "")[:600]
            parts.append(
                f"- [{src.tier} | {date}] {src.title} ({src.url}) — {snippet}"
            )

    return "\n".join(parts)


class OpenAISynthesizer:
    """Synthesize via the OpenAI chat-completions API (default backend)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.synthesis_model
        self.price_in = settings.synthesis_price_in
        self.price_out = settings.synthesis_price_out

    async def synthesize(
        self,
        topic: str,
        layers: list[LayerOutput],
        *,
        style_constraints: str | None,
    ) -> tuple[str, float]:
        prompt = build_synthesis_prompt(
            topic, layers, style_constraints=style_constraints
        )
        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        msg = await client.chat.completions.create(
            model=self.model,
            max_completion_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.choices[0].message.content or ""
        usage = getattr(msg, "usage", None)
        cost = 0.0
        if usage:
            cost = (
                usage.prompt_tokens * self.price_in
                + usage.completion_tokens * self.price_out
            ) / 1_000_000
        return text.strip(), round(cost, 6)


class AnthropicSynthesizer:
    """Synthesize via the Anthropic Messages API (fallback backend)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = _ANTHROPIC_MODEL
        self.price_in = _ANTHROPIC_PRICE_IN
        self.price_out = _ANTHROPIC_PRICE_OUT

    async def synthesize(
        self,
        topic: str,
        layers: list[LayerOutput],
        *,
        style_constraints: str | None,
    ) -> tuple[str, float]:
        prompt = build_synthesis_prompt(
            topic, layers, style_constraints=style_constraints
        )
        client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        msg = await client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(block, "text", "") for block in msg.content)
        usage = getattr(msg, "usage", None)
        cost = 0.0
        if usage:
            cost = (
                usage.input_tokens * self.price_in
                + usage.output_tokens * self.price_out
            ) / 1_000_000
        return text.strip(), round(cost, 6)
