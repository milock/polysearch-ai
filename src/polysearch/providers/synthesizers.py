"""OpenAI + Anthropic synthesizers.

Fuse the pipeline's layer outputs into a single cited markdown report behind the
:class:`Synthesizer` protocol
(``synthesize(topic, layers, *, style_constraints) -> (markdown, cost_usd)``).
Both providers share one prompt builder and one output contract — Executive
Summary / Key Findings / Source Quality Notes — and differ only in SDK and
pricing.

Two design notes:

- **Style is caller-supplied.** The prompt asks for plain, direct prose and
  appends the deployment's ``style_constraints`` verbatim when one is set,
  rather than baking in a fixed house style.
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

# Per-source excerpt budgets (chars). Authoritative sources (HIGH/MEDIUM) carry
# the figures a good report must state, so they get the wider excerpt; lower tiers
# stay at the base. The HIGH/MEDIUM budget is caller-overridable
# (``Settings.synthesis_excerpt_chars``); the base is fixed.
_BASE_EXCERPT_CHARS = 600
_DEFAULT_HI_EXCERPT_CHARS = 1200
_WIDE_EXCERPT_TIERS = frozenset({"HIGH", "MEDIUM"})


def build_synthesis_prompt(
    topic: str,
    layers: list[LayerOutput],
    *,
    style_constraints: str | None = None,
    excerpt_chars: int | None = None,
) -> str:
    """Build the synthesis user prompt from ``topic`` + layer material.

    Each source is rendered tier-labeled with its published date and snippet,
    grouped under its layer. HIGH/MEDIUM-tier sources get a wider excerpt
    (``excerpt_chars``, default 1200) than lower tiers (600), since the
    authoritative pages carry the concrete figures Key Findings must state.
    ``style_constraints`` is appended verbatim when set.
    """
    hi_chars = excerpt_chars if excerpt_chars is not None else _DEFAULT_HI_EXCERPT_CHARS
    parts = [
        f"Research topic: {topic}",
        "",
        "You are synthesizing a research report from multiple search layers.",
        "Output exactly three markdown sections, using these headers verbatim:",
        "",
        "## Executive Summary",
        "2-4 sentences on the single most important thing someone preparing for a "
        "decision should know. Lead with the concrete answer — the specific "
        "figure, date, name, or range the topic asks for — not a general "
        "characterization of it.",
        "",
        "## Key Findings",
        "3-7 bullet points. Each bullet must state a concrete fact: the actual "
        "number, date, name, range, or measured value, with its unit — not an "
        "allusion to it. State the specific figure the finding rests on (prefer "
        "\"held at 3.50%-3.75% as of June 2026\" over \"rates held steady\"). When "
        "the topic asks for a specific quantity, entity, count, or date, state it "
        "explicitly with its value and cite the source whose figure you used. End "
        "each bullet with a source tag like [HIGH: domain, date], [COMMUNITY], or "
        "[SME].",
        "",
        "## Source Quality Notes",
        "2-5 bullets on conflicts, gaps, or source-tier concentration risks. Be "
        "brief and specific.",
        "",
        "Ground every claim in the material below. Do not introduce facts not "
        "present in the input.",
        "Keep the report internally consistent: any figure, count, or date you "
        "state in prose must match the figures and counts in your own bullets and "
        "lists (if you say \"three cuts,\" list three).",
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
            budget = hi_chars if src.tier in _WIDE_EXCERPT_TIERS else _BASE_EXCERPT_CHARS
            snippet = (src.snippet or "")[:budget]
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
            topic,
            layers,
            style_constraints=style_constraints,
            excerpt_chars=self.settings.synthesis_excerpt_chars,
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
            topic,
            layers,
            style_constraints=style_constraints,
            excerpt_chars=self.settings.synthesis_excerpt_chars,
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
