"""Runtime configuration: Settings, per-depth profiles, and install tiers.

Everything downstream imports from here. ``Settings`` is a plain dataclass built
from the environment (``dotenv.load_dotenv()`` + ``os.environ``) — no
pydantic-settings dependency. API keys use conventional names
(``OPENAI_API_KEY``); all other settings read a ``POLYSEARCH_`` prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

DiscoveryBackend = Literal["perplexity", "brave", "firecrawl"]


def _get(name: str) -> str | None:
    """Read an env var, treating empty/whitespace-only as unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _get_str(name: str, default: str) -> str:
    return _get(name) or default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    return float(raw) if raw is not None else default


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    return int(raw) if raw is not None else default


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_csv(name: str) -> list[str]:
    raw = _get(name)
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    """Resolved runtime configuration for a pipeline run."""

    # ── API keys (conventional names, all optional) ──────────────────────
    perplexity_api_key: str | None = None
    firecrawl_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    brave_api_key: str | None = None
    scrapecreators_api_key: str | None = None
    youtube_api_key: str | None = None
    github_token: str | None = None
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None

    # ── Source config ────────────────────────────────────────────────────
    x_handles: list[str] = field(default_factory=list)
    output_dir: Path = Path("./reports")

    # ── Synthesis ────────────────────────────────────────────────────────
    synthesis_model: str = "gpt-5.4-mini"
    synthesis_price_in: float = 0.75
    synthesis_price_out: float = 4.50
    # Per-source excerpt budget (chars) fed to synthesis for HIGH/MEDIUM-tier
    # sources. Authoritative pages carry the figures a good report must state, so
    # they get a wider excerpt than lower tiers (which stay at the 600-char base).
    synthesis_excerpt_chars: int = 1200

    # ── Discovery / search ───────────────────────────────────────────────
    perplexity_model: str = "sonar-pro"
    perplexity_deep_model: str = "sonar-reasoning-pro"
    # Per-1M-token pricing for the primary Perplexity model (sonar-pro rates).
    perplexity_price_in: float = 3.0
    perplexity_price_out: float = 15.0
    discovery_backend: DiscoveryBackend = "perplexity"
    embedding_model: str = "text-embedding-3-small"

    # ── Verification / recovery ──────────────────────────────────────────
    fuzzy_threshold: float = 0.85
    recovery_rate_threshold: float = 0.35
    recovery_min_citations: int = 4
    recovery_max_queries: int = 3
    firecrawl_usd_per_credit: float = 0.0032

    # ── Deep research ────────────────────────────────────────────────────
    enable_deep_research: bool = False
    deep_research_model: str = "sonar-deep-research"
    deep_research_timeout_s: int = 3600

    # ── Refinement / output ──────────────────────────────────────────────
    refinement_cost_ceiling_usd: float | None = None
    style_constraints: str | None = None
    allow_placeholders: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Build Settings from ``.env`` + process environment."""
        load_dotenv()

        output_dir_raw = _get("POLYSEARCH_OUTPUT_DIR")
        ceiling_raw = _get("POLYSEARCH_REFINEMENT_COST_CEILING_USD")
        style_raw = _get("POLYSEARCH_STYLE_CONSTRAINTS")

        return cls(
            perplexity_api_key=_get("PERPLEXITY_API_KEY"),
            firecrawl_api_key=_get("FIRECRAWL_API_KEY"),
            openai_api_key=_get("OPENAI_API_KEY"),
            anthropic_api_key=_get("ANTHROPIC_API_KEY"),
            brave_api_key=_get("BRAVE_API_KEY"),
            scrapecreators_api_key=_get("SCRAPECREATORS_API_KEY"),
            youtube_api_key=_get("YOUTUBE_API_KEY"),
            github_token=_get("GITHUB_TOKEN"),
            reddit_client_id=_get("REDDIT_CLIENT_ID"),
            reddit_client_secret=_get("REDDIT_CLIENT_SECRET"),
            x_handles=_get_csv("POLYSEARCH_X_HANDLES"),
            output_dir=Path(output_dir_raw) if output_dir_raw else Path("./reports"),
            synthesis_model=_get_str("POLYSEARCH_SYNTHESIS_MODEL", "gpt-5.4-mini"),
            synthesis_price_in=_get_float("POLYSEARCH_SYNTHESIS_PRICE_IN", 0.75),
            synthesis_price_out=_get_float("POLYSEARCH_SYNTHESIS_PRICE_OUT", 4.50),
            synthesis_excerpt_chars=_get_int("POLYSEARCH_SYNTHESIS_EXCERPT_CHARS", 1200),
            perplexity_model=_get_str("POLYSEARCH_PERPLEXITY_MODEL", "sonar-pro"),
            perplexity_deep_model=_get_str(
                "POLYSEARCH_PERPLEXITY_DEEP_MODEL", "sonar-reasoning-pro"
            ),
            perplexity_price_in=_get_float("POLYSEARCH_PERPLEXITY_PRICE_IN", 3.0),
            perplexity_price_out=_get_float("POLYSEARCH_PERPLEXITY_PRICE_OUT", 15.0),
            discovery_backend=_resolve_discovery_backend(),
            embedding_model=_get_str(
                "POLYSEARCH_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            fuzzy_threshold=_get_float("POLYSEARCH_FUZZY_THRESHOLD", 0.85),
            recovery_rate_threshold=_get_float(
                "POLYSEARCH_RECOVERY_RATE_THRESHOLD", 0.35
            ),
            recovery_min_citations=_get_int("POLYSEARCH_RECOVERY_MIN_CITATIONS", 4),
            recovery_max_queries=_get_int("POLYSEARCH_RECOVERY_MAX_QUERIES", 3),
            firecrawl_usd_per_credit=_get_float(
                "POLYSEARCH_FIRECRAWL_USD_PER_CREDIT", 0.0032
            ),
            enable_deep_research=_get_bool("POLYSEARCH_DEEP_RESEARCH", False),
            deep_research_model=_get_str(
                "POLYSEARCH_DEEP_RESEARCH_MODEL", "sonar-deep-research"
            ),
            deep_research_timeout_s=_get_int("POLYSEARCH_DEEP_RESEARCH_TIMEOUT_S", 3600),
            refinement_cost_ceiling_usd=(
                float(ceiling_raw) if ceiling_raw is not None else None
            ),
            style_constraints=style_raw,
            allow_placeholders=_get_bool("POLYSEARCH_ALLOW_PLACEHOLDERS", False),
        )


def _resolve_discovery_backend() -> DiscoveryBackend:
    raw = _get("POLYSEARCH_DISCOVERY_BACKEND")
    if raw in ("perplexity", "brave", "firecrawl"):
        return raw  # type: ignore[return-value]
    return "perplexity"


@dataclass(frozen=True)
class DepthProfile:
    """Per-depth knobs for search breadth, verification budget, and refinement."""

    perplexity_sub_questions: int
    firecrawl_limit: int
    firecrawl_scrape_top_k: int
    community_limit: int
    authoritative_top_k: int
    verify_budget_usd: float
    verify_concurrency: int
    max_refinement_iterations: int
    refinement_cost_ceiling_usd: float
    deep_research_eligible: bool


DEPTH_PROFILES: dict[str, DepthProfile] = {
    "quick": DepthProfile(
        perplexity_sub_questions=2,
        firecrawl_limit=5,
        firecrawl_scrape_top_k=3,
        community_limit=10,
        authoritative_top_k=0,
        verify_budget_usd=1,
        verify_concurrency=5,
        max_refinement_iterations=0,
        refinement_cost_ceiling_usd=0,
        deep_research_eligible=False,
    ),
    "standard": DepthProfile(
        perplexity_sub_questions=4,
        firecrawl_limit=10,
        firecrawl_scrape_top_k=5,
        community_limit=20,
        authoritative_top_k=2,
        verify_budget_usd=3,
        verify_concurrency=8,
        max_refinement_iterations=2,
        refinement_cost_ceiling_usd=2.00,
        deep_research_eligible=False,
    ),
    "deep": DepthProfile(
        perplexity_sub_questions=6,
        firecrawl_limit=20,
        firecrawl_scrape_top_k=10,
        community_limit=30,
        authoritative_top_k=4,
        verify_budget_usd=10,
        verify_concurrency=10,
        max_refinement_iterations=4,
        refinement_cost_ceiling_usd=8.00,
        deep_research_eligible=True,
    ),
}


def resolve_install_tier(settings: Settings) -> int:
    """Return the highest install tier the credentials satisfy (0, 1, or 2).

    Tiers are cumulative, per the re-tiered ``.env.example``:
      - Tier 0: baseline (Perplexity-only, or nothing).
      - Tier 1: Firecrawl web grounding + a synthesis model (OpenAI or Anthropic).
      - Tier 2: Tier 1 plus at least one extra source connector.
    """
    has_synth = bool(settings.openai_api_key or settings.anthropic_api_key)
    has_tier1 = bool(settings.firecrawl_api_key) and has_synth
    has_connector = any(
        [
            settings.scrapecreators_api_key,
            settings.youtube_api_key,
            settings.github_token,
            settings.reddit_client_id,
            settings.reddit_client_secret,
        ]
    )
    if has_tier1 and has_connector:
        return 2
    if has_tier1:
        return 1
    return 0
