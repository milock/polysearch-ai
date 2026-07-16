"""Unit tests for polysearch.config — Settings, DepthProfile, install tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from polysearch import config
from polysearch.config import (
    DEPTH_PROFILES,
    DepthProfile,
    Settings,
    resolve_install_tier,
)

# Every credential / config env var the config layer reads. Cleared before each
# test so the developer's real shell environment can't leak into assertions.
_ENV_VARS = [
    "PERPLEXITY_API_KEY",
    "FIRECRAWL_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "BRAVE_API_KEY",
    "SCRAPECREATORS_API_KEY",
    "YOUTUBE_API_KEY",
    "GITHUB_TOKEN",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "POLYSEARCH_X_HANDLES",
    "POLYSEARCH_OUTPUT_DIR",
    "POLYSEARCH_SYNTHESIS_MODEL",
    "POLYSEARCH_PERPLEXITY_PRICE_IN",
    "POLYSEARCH_PERPLEXITY_PRICE_OUT",
    "POLYSEARCH_DISCOVERY_BACKEND",
    "POLYSEARCH_DEEP_RESEARCH",
    "POLYSEARCH_FUZZY_THRESHOLD",
    "POLYSEARCH_ENABLE_DEEP_RESEARCH",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic env: drop every known var and neutralize .env autoloading."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Never let a developer's real .env file bleed into the test run.
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)


# ── Settings.from_env ──────────────────────────────────────────────────────

def test_empty_env_all_keys_none() -> None:
    s = Settings.from_env()
    assert s.perplexity_api_key is None
    assert s.firecrawl_api_key is None
    assert s.openai_api_key is None
    assert s.anthropic_api_key is None
    assert s.brave_api_key is None
    assert s.scrapecreators_api_key is None
    assert s.youtube_api_key is None
    assert s.github_token is None
    assert s.reddit_client_id is None
    assert s.reddit_client_secret is None


def test_defaults() -> None:
    s = Settings.from_env()
    assert s.x_handles == []
    assert s.output_dir == Path("./reports")
    assert s.synthesis_model == "gpt-5.4-mini"
    assert s.synthesis_price_in == 0.75
    assert s.synthesis_price_out == 4.50
    assert s.synthesis_excerpt_chars == 1200
    assert s.perplexity_model == "sonar-pro"
    assert s.perplexity_deep_model == "sonar-reasoning-pro"
    # Default sonar-pro per-1M-token pricing.
    assert s.perplexity_price_in == 3.0
    assert s.perplexity_price_out == 15.0
    assert s.discovery_backend == "perplexity"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.fuzzy_threshold == 0.85
    assert s.recovery_rate_threshold == 0.35
    assert s.recovery_min_citations == 4
    assert s.recovery_max_queries == 3
    assert s.firecrawl_usd_per_credit == 0.0032
    assert s.enable_deep_research is False
    assert s.deep_research_model == "sonar-deep-research"
    assert s.deep_research_timeout_s == 3600
    assert s.refinement_cost_ceiling_usd is None
    assert s.style_constraints is None
    assert s.allow_placeholders is False


def test_api_keys_read_conventional_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-1")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-1")
    s = Settings.from_env()
    assert s.perplexity_api_key == "pk-1"
    assert s.firecrawl_api_key == "fc-1"
    assert s.openai_api_key == "oa-1"


def test_synthesis_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_SYNTHESIS_MODEL", "gpt-6-nano")
    s = Settings.from_env()
    assert s.synthesis_model == "gpt-6-nano"


def test_synthesis_excerpt_chars_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_SYNTHESIS_EXCERPT_CHARS", "800")
    s = Settings.from_env()
    assert s.synthesis_excerpt_chars == 800


def test_perplexity_pricing_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_PERPLEXITY_PRICE_IN", "1.25")
    monkeypatch.setenv("POLYSEARCH_PERPLEXITY_PRICE_OUT", "6.5")
    s = Settings.from_env()
    assert s.perplexity_price_in == 1.25
    assert s.perplexity_price_out == 6.5


def test_output_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_OUTPUT_DIR", "/tmp/ps-out")
    s = Settings.from_env()
    assert s.output_dir == Path("/tmp/ps-out")


def test_x_handles_csv_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_X_HANDLES", "one, two ,three")
    s = Settings.from_env()
    assert s.x_handles == ["one", "two", "three"]


def test_x_handles_empty_is_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_X_HANDLES", "")
    s = Settings.from_env()
    assert s.x_handles == []


def test_fuzzy_threshold_env_override_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_FUZZY_THRESHOLD", "0.9")
    s = Settings.from_env()
    assert s.fuzzy_threshold == 0.9


def test_enable_deep_research_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_DEEP_RESEARCH", "1")
    s = Settings.from_env()
    assert s.enable_deep_research is True


# ── resolve_install_tier ────────────────────────────────────────────────────

def test_tier_zero_empty_env() -> None:
    assert resolve_install_tier(Settings.from_env()) == 0


def test_tier_zero_perplexity_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    assert resolve_install_tier(Settings.from_env()) == 0


def test_tier_one_firecrawl_plus_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-1")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-1")
    assert resolve_install_tier(Settings.from_env()) == 1


def test_tier_one_firecrawl_plus_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "an-1")
    assert resolve_install_tier(Settings.from_env()) == 1


def test_firecrawl_without_synth_is_tier_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-1")
    assert resolve_install_tier(Settings.from_env()) == 0


def test_tier_two_requires_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-1")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-1")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-1")
    assert resolve_install_tier(Settings.from_env()) == 2


def test_connector_without_tier_one_is_not_tier_two(monkeypatch: pytest.MonkeyPatch) -> None:
    # Connector key present but no Tier-1 base → tier stays 0 (tiers are cumulative).
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pk-1")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-1")
    assert resolve_install_tier(Settings.from_env()) == 0


# ── DepthProfile / DEPTH_PROFILES ───────────────────────────────────────────

def test_depth_profiles_keys() -> None:
    assert set(DEPTH_PROFILES) == {"quick", "standard", "deep"}


def test_depth_profile_is_dataclass_type() -> None:
    assert all(isinstance(p, DepthProfile) for p in DEPTH_PROFILES.values())


def test_refinement_iterations_by_depth() -> None:
    assert DEPTH_PROFILES["quick"].max_refinement_iterations == 0
    assert DEPTH_PROFILES["standard"].max_refinement_iterations == 2
    assert DEPTH_PROFILES["deep"].max_refinement_iterations == 4


def test_refinement_cost_ceiling_by_depth() -> None:
    assert DEPTH_PROFILES["quick"].refinement_cost_ceiling_usd == 0
    assert DEPTH_PROFILES["standard"].refinement_cost_ceiling_usd == 2.00
    assert DEPTH_PROFILES["deep"].refinement_cost_ceiling_usd == 8.00


def test_depth_profile_table_values() -> None:
    quick = DEPTH_PROFILES["quick"]
    assert quick.perplexity_sub_questions == 2
    assert quick.firecrawl_limit == 5
    assert quick.firecrawl_scrape_top_k == 3
    assert quick.community_limit == 10
    assert quick.authoritative_top_k == 0
    assert quick.verify_budget_usd == 1
    assert quick.verify_concurrency == 5
    assert quick.deep_research_eligible is False

    deep = DEPTH_PROFILES["deep"]
    assert deep.perplexity_sub_questions == 6
    assert deep.firecrawl_limit == 20
    assert deep.firecrawl_scrape_top_k == 10
    assert deep.community_limit == 30
    assert deep.authoritative_top_k == 4
    assert deep.verify_budget_usd == 10
    assert deep.verify_concurrency == 10
    assert deep.deep_research_eligible is True


def test_only_deep_is_deep_research_eligible() -> None:
    assert DEPTH_PROFILES["standard"].deep_research_eligible is False
    assert DEPTH_PROFILES["deep"].deep_research_eligible is True
