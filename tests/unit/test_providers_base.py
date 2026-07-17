"""Unit tests for polysearch.providers.base — protocols, null impls, wiring.

The no-credentials path is the primary contract here: with an empty ``Settings``
every slot must resolve to a null provider (or ``None``/empty list) whose
``reason`` names the missing credential. Real providers arrive in later tasks and
are lazy-imported inside credential-gated branches, so this module imports and
these tests pass with zero environment variables.
"""

from __future__ import annotations

from polysearch.config import Settings
from polysearch.output.schema import (
    Claim,
    LayerOutput,
    VerificationReport,
)
from polysearch.providers.base import (
    CitationVerifier,
    CommunitySource,
    NullResearchProvider,
    NullSynthesizer,
    NullVerifier,
    NullWebGrounder,
    Providers,
    ResearchProvider,
    Synthesizer,
    WebGrounder,
    build_providers,
)


def _no_keys_settings() -> Settings:
    """A Settings with every credential unset (the fresh-install baseline)."""
    return Settings()


# ── Null providers: empty output + a reason naming the missing credential ────

async def test_null_research_returns_empty_layer_with_reason() -> None:
    provider = NullResearchProvider("PERPLEXITY_API_KEY not set")
    out = await provider.research("anything", sub_questions=4, depth="standard")
    assert isinstance(out, LayerOutput)
    assert out.results == []
    assert out.cost_usd == 0.0
    assert out.error == "PERPLEXITY_API_KEY not set"
    assert provider.reason == "PERPLEXITY_API_KEY not set"


async def test_null_grounder_returns_empty_layer_with_reason() -> None:
    grounder = NullWebGrounder("FIRECRAWL_API_KEY not set")
    out = await grounder.ground("anything", limit=10, scrape_top_k=5)
    assert isinstance(out, LayerOutput)
    assert out.results == []
    assert out.error == "FIRECRAWL_API_KEY not set"
    assert grounder.reason == "FIRECRAWL_API_KEY not set"


async def test_null_synthesizer_returns_empty_markdown_zero_cost() -> None:
    synth = NullSynthesizer("OPENAI_API_KEY or ANTHROPIC_API_KEY not set")
    markdown, cost = await synth.synthesize("topic", [], style_constraints=None)
    assert markdown == ""
    assert cost == 0.0
    assert synth.reason == "OPENAI_API_KEY or ANTHROPIC_API_KEY not set"


async def test_null_verifier_returns_empty_report_with_reason() -> None:
    verifier = NullVerifier("FIRECRAWL_API_KEY not set")
    claims = [Claim(claim_id="c1", text="x", numbers=["1"], source_urls=["http://a"])]
    report = await verifier.verify(claims, budget_usd=3.0, max_concurrency=8)
    assert isinstance(report, VerificationReport)
    assert report.total_citations == 0
    assert report.verified_ok == 0
    assert report.results == []
    assert report.total_cost_usd == 0.0
    assert verifier.reason == "FIRECRAWL_API_KEY not set"


# ── runtime_checkable isinstance behaviour ──────────────────────────────────

def test_null_impls_satisfy_their_protocols() -> None:
    assert isinstance(NullResearchProvider("r"), ResearchProvider)
    assert isinstance(NullWebGrounder("g"), WebGrounder)
    assert isinstance(NullSynthesizer("s"), Synthesizer)
    assert isinstance(NullVerifier("v"), CitationVerifier)


def test_protocols_are_distinct_not_cross_satisfied() -> None:
    # A grounder is not a research provider (no `research`, no `name`).
    assert not isinstance(NullWebGrounder("g"), ResearchProvider)
    # A research provider is not a verifier (no `verify`).
    assert not isinstance(NullResearchProvider("r"), CitationVerifier)


def test_research_provider_exposes_name_attribute() -> None:
    # `name` is part of the ResearchProvider data protocol.
    assert isinstance(NullResearchProvider("r").name, str)


# ── build_providers: no-credentials path → all nulls ────────────────────────

def test_build_providers_no_keys_returns_providers_dataclass() -> None:
    providers = build_providers(_no_keys_settings())
    assert isinstance(providers, Providers)


def test_build_providers_no_keys_all_null() -> None:
    providers = build_providers(_no_keys_settings())
    assert isinstance(providers.research, NullResearchProvider)
    assert isinstance(providers.grounder, NullWebGrounder)
    assert isinstance(providers.synthesizer, NullSynthesizer)
    assert isinstance(providers.verifier, NullVerifier)


def test_build_providers_no_keys_deep_research_is_none() -> None:
    providers = build_providers(_no_keys_settings())
    assert providers.deep_research is None


def test_build_providers_no_keys_activates_keyless_community_and_no_linkedin() -> None:
    # With `community.adapters` now shipping, the keyless sources (Reddit's
    # unauthenticated fallback, Hacker News, Bluesky, GitHub unauth) light up at
    # ANY tier — a zero-credential install still gets a live community layer.
    # LinkedIn stays gated on the ScrapeCreators key.
    providers = build_providers(_no_keys_settings())
    names = {s.name for s in providers.community_sources}
    assert names == {"reddit", "hackernews", "bluesky", "github"}
    assert providers.linkedin is None


def test_build_providers_adds_youtube_only_when_keyed() -> None:
    base = {s.name for s in build_providers(_no_keys_settings()).community_sources}
    assert "youtube" not in base
    keyed = build_providers(Settings(youtube_api_key="yt-1")).community_sources
    assert "youtube" in {s.name for s in keyed}


def test_build_providers_adds_x_only_when_scrapecreators_keyed() -> None:
    base = {s.name for s in build_providers(_no_keys_settings()).community_sources}
    assert "x" not in base
    keyed = build_providers(Settings(scrapecreators_api_key="sc-1")).community_sources
    assert "x" in {s.name for s in keyed}


def test_build_providers_all_keys_activates_full_community_set() -> None:
    providers = build_providers(
        Settings(youtube_api_key="yt-1", scrapecreators_api_key="sc-1")
    )
    names = {s.name for s in providers.community_sources}
    assert names == {"reddit", "hackernews", "bluesky", "github", "youtube", "x"}


def test_build_providers_reasons_name_missing_credentials() -> None:
    providers = build_providers(_no_keys_settings())
    assert providers.research.reason == "PERPLEXITY_API_KEY not set"
    assert providers.grounder.reason == "FIRECRAWL_API_KEY not set"
    assert providers.synthesizer.reason == "OPENAI_API_KEY or ANTHROPIC_API_KEY not set"
    assert providers.verifier.reason == "FIRECRAWL_API_KEY not set"


def test_build_providers_community_is_a_list() -> None:
    providers = build_providers(_no_keys_settings())
    assert isinstance(providers.community_sources, list)


def test_community_sources_degrade_to_skip_when_adapters_missing(
    monkeypatch: object,
) -> None:
    """A missing ``community.adapters`` module must skip the layer, not raise.

    Forces the ``from polysearch.community import adapters`` import to fail even
    once the module later ships, and sets both key-gated credentials — the
    result must still be an empty list (degrade-to-skip), never an exception or a
    null append.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        fromlist = args[2] if len(args) >= 3 else kwargs.get("fromlist")
        if name == "polysearch.community" and fromlist and "adapters" in fromlist:
            raise ImportError("adapters not present yet")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)  # type: ignore[attr-defined]

    settings = Settings(youtube_api_key="yt-1", scrapecreators_api_key="sc-1")
    providers = build_providers(settings)
    assert providers.community_sources == []


def test_community_sources_satisfy_community_source_protocol() -> None:
    # CommunitySource is a runtime-checkable protocol; every wired source (the
    # keyless set at no-keys, plus the key-gated ones) must satisfy it.
    sources = build_providers(
        Settings(youtube_api_key="yt-1", scrapecreators_api_key="sc-1")
    ).community_sources
    assert sources  # non-empty
    assert all(isinstance(s, CommunitySource) for s in sources)
