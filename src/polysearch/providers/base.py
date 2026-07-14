"""Provider protocols, null implementations, and credential-gated wiring.

The pipeline talks to the outside world through a small set of structural
protocols — a research provider, a web grounder, a synthesizer, a citation
verifier, and community sources. Every protocol is ``runtime_checkable`` so the
orchestrator can ``isinstance``-probe a slot when it needs to.

``build_providers`` resolves a ``Settings`` into a concrete ``Providers`` bundle.
Real provider classes (Perplexity, Firecrawl, OpenAI/Anthropic, …) are imported
lazily *inside* the credential-gated branches so this module imports cleanly
before those classes exist, and so a partial install degrades gracefully: a
missing credential — or a not-yet-shipped provider module — falls back to a null
provider whose ``reason`` names exactly what is missing. Those reason strings are
surfaced later in the pipeline's decision log, so they are part of the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from polysearch.config import Settings
from polysearch.output.schema import (
    Claim,
    LayerOutput,
    SourceResult,
    VerificationReport,
)

# ── Protocols ───────────────────────────────────────────────────────────────


@runtime_checkable
class ResearchProvider(Protocol):
    """Answers a topic by fanning out into sub-questions (e.g. Perplexity)."""

    name: str

    async def research(
        self, topic: str, *, sub_questions: int, depth: str
    ) -> LayerOutput: ...


@runtime_checkable
class WebGrounder(Protocol):
    """Grounds a topic in live web results, scraping the top hits (e.g. Firecrawl)."""

    async def ground(
        self, topic: str, *, limit: int, scrape_top_k: int
    ) -> LayerOutput: ...


@runtime_checkable
class Synthesizer(Protocol):
    """Fuses layer outputs into a cited markdown report; returns ``(markdown, cost_usd)``."""

    async def synthesize(
        self, topic: str, layers: list[LayerOutput], *, style_constraints: str | None
    ) -> tuple[str, float]: ...


@runtime_checkable
class CitationVerifier(Protocol):
    """Checks each claim's citations against their sources within a budget."""

    async def verify(
        self, claims: list[Claim], *, budget_usd: float, max_concurrency: int
    ) -> VerificationReport: ...


@runtime_checkable
class CommunitySource(Protocol):
    """A recency-windowed community/social signal source (Reddit, GitHub, X, …)."""

    name: str

    async def search(
        self, topic: str, *, window_days: int, limit: int
    ) -> list[SourceResult]: ...


@runtime_checkable
class PersonContextHook(Protocol):
    """Optional, pluggable person-context lookup for PERSON-classified runs.

    Given a person's name, return supplemental context as a ``SourceResult`` (or
    ``None`` when nothing is found). This is the public seam where a deployment
    can wire its own private source — a CRM, a directory, an internal graph —
    without that source living in the package. Register one per run via
    ``run_research(..., person_hook=...)``; the default is ``None`` (no-op).
    """

    async def lookup(self, name: str) -> SourceResult | None: ...


# ── Null implementations ─────────────────────────────────────────────────────
#
# Each null returns an empty result and carries a ``reason`` naming the missing
# credential. Layer-producing nulls also stamp that reason into
# ``LayerOutput.error`` so it travels with the (empty) layer.


class NullResearchProvider:
    """Research provider stand-in used when no research credential is present."""

    name = "research"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def research(
        self, topic: str, *, sub_questions: int, depth: str
    ) -> LayerOutput:
        return LayerOutput(layer="research", error=self.reason)


class NullWebGrounder:
    """Web grounder stand-in used when ``FIRECRAWL_API_KEY`` is absent."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def ground(
        self, topic: str, *, limit: int, scrape_top_k: int
    ) -> LayerOutput:
        return LayerOutput(layer="grounding", error=self.reason)


class NullSynthesizer:
    """Synthesizer stand-in used when no synthesis model credential is present."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def synthesize(
        self, topic: str, layers: list[LayerOutput], *, style_constraints: str | None
    ) -> tuple[str, float]:
        return "", 0.0


class NullVerifier:
    """Verifier stand-in used when the verification backend is unavailable."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def verify(
        self, claims: list[Claim], *, budget_usd: float, max_concurrency: int
    ) -> VerificationReport:
        return VerificationReport(
            total_citations=0,
            verified_ok=0,
            broken=0,
            quote_mismatches=0,
            number_mismatches=0,
            paywalled=0,
            undated=0,
            skipped_budget=0,
            results=[],
            total_cost_usd=0.0,
            total_duration_ms=0,
        )


# ── Provider bundle + wiring ─────────────────────────────────────────────────


@dataclass
class Providers:
    """The concrete provider set a single pipeline run talks through."""

    research: ResearchProvider
    deep_research: ResearchProvider | None
    grounder: WebGrounder
    synthesizer: Synthesizer
    verifier: CitationVerifier
    community_sources: list[CommunitySource] = field(default_factory=list)
    linkedin: CommunitySource | None = None


def build_providers(settings: Settings) -> Providers:
    """Resolve ``settings`` into a ``Providers`` bundle.

    Real providers are instantiated only when their credentials exist, and their
    classes are imported lazily inside each branch so this function works today
    (all-nulls) and keeps working as real providers land in later tasks. If a
    real provider's module is missing, the slot degrades to its null with a
    reason rather than raising.
    """
    return Providers(
        research=_build_research(settings),
        deep_research=_build_deep_research(settings),
        grounder=_build_grounder(settings),
        synthesizer=_build_synthesizer(settings),
        verifier=_build_verifier(settings),
        community_sources=_build_community_sources(settings),
        linkedin=_build_linkedin(settings),
    )


def _build_research(settings: Settings) -> ResearchProvider:
    if not settings.perplexity_api_key:
        return NullResearchProvider("PERPLEXITY_API_KEY not set")
    try:
        from polysearch.providers.perplexity import PerplexityProvider
    except ImportError as exc:
        return NullResearchProvider(f"perplexity provider unavailable: {exc}")
    return PerplexityProvider(settings)


def _build_deep_research(settings: Settings) -> ResearchProvider | None:
    if not (settings.enable_deep_research and settings.perplexity_api_key):
        return None
    try:
        from polysearch.providers.deep_research import DeepResearchProvider
    except ImportError:
        return None
    return DeepResearchProvider(settings)


def _build_grounder(settings: Settings) -> WebGrounder:
    if not settings.firecrawl_api_key:
        return NullWebGrounder("FIRECRAWL_API_KEY not set")
    try:
        from polysearch.providers.firecrawl import FirecrawlGrounder
    except ImportError as exc:
        return NullWebGrounder(f"firecrawl grounder unavailable: {exc}")
    return FirecrawlGrounder(settings)


def _build_synthesizer(settings: Settings) -> Synthesizer:
    if settings.openai_api_key:
        try:
            from polysearch.providers.synthesizers import OpenAISynthesizer
        except ImportError as exc:
            return NullSynthesizer(f"openai synthesizer unavailable: {exc}")
        return OpenAISynthesizer(settings)
    if settings.anthropic_api_key:
        try:
            from polysearch.providers.synthesizers import AnthropicSynthesizer
        except ImportError as exc:
            return NullSynthesizer(f"anthropic synthesizer unavailable: {exc}")
        return AnthropicSynthesizer(settings)
    return NullSynthesizer("OPENAI_API_KEY or ANTHROPIC_API_KEY not set")


def _build_verifier(settings: Settings) -> CitationVerifier:
    if not settings.firecrawl_api_key:
        return NullVerifier("FIRECRAWL_API_KEY not set")
    try:
        from polysearch.verification.verifier import FirecrawlVerifier
    except ImportError as exc:
        return NullVerifier(f"verifier unavailable: {exc}")
    return FirecrawlVerifier(settings)


def _build_community_sources(settings: Settings) -> list[CommunitySource]:
    """Assemble the community-signal layer from ``community/adapters.py``.

    Every adapter lives in the single canonical ``community.adapters`` module.
    Two classes of source:

    - **Keyless (any-tier):** ``RedditSource`` (unauthenticated fallback),
      ``HackerNewsSource``, ``BlueskySource``, and ``GitHubSource`` (unauth)
      activate at ANY install tier — they are attempted unconditionally and use
      credentials only to lift rate limits when present.
    - **Key-gated:** ``YouTubeSource`` (needs ``youtube_api_key``) and
      ``XSource`` (needs ``scrapecreators_api_key``) are appended only when their
      credential is set.

    The whole layer degrades to *skip* (not a null append) when a source — or the
    ``adapters`` module itself — is not yet present, so today's no-keys install
    returns ``[]`` and the keyless sources light up automatically once the module
    ships, with no changes here.
    """
    sources: list[CommunitySource] = []
    try:
        from polysearch.community import adapters
    except ImportError:
        return sources

    def _add(class_name: str) -> None:
        cls = getattr(adapters, class_name, None)
        if cls is not None:
            sources.append(cls(settings))

    # Keyless adapters — active at any tier.
    for name in ("RedditSource", "HackerNewsSource", "BlueskySource", "GitHubSource"):
        _add(name)
    # Key-gated adapters.
    if settings.youtube_api_key:
        _add("YouTubeSource")
    if settings.scrapecreators_api_key:
        _add("XSource")
    return sources


def _build_linkedin(settings: Settings) -> CommunitySource | None:
    if not settings.scrapecreators_api_key:
        return None
    try:
        from polysearch.providers.linkedin import LinkedInEnricher
    except ImportError:
        return None
    return LinkedInEnricher(settings)
