"""Unit tests for polysearch.sources.authority (source-tier classification).

Fixtures are generic (gov / news / dev / social domains) — no vertical-specific
domains. Every domain referenced here resolves against the shipped public
`polysearch/data/domain_tiers.yaml`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import get_args

import pytest

from polysearch.sources import authority
from polysearch.sources.authority import (
    Tier,
    classify_bulk,
    classify_domain,
    classify_url,
    downgrade_if_undated,
    high_tier_domains,
    reload,
    set_output_dir,
)


def setup_module(_module: object) -> None:
    reload()


# --- classify_domain basic tiers ------------------------------------------


def test_gov_root_high() -> None:
    tier, reason = classify_domain("https://www.nih.gov/some/path")
    assert tier == "HIGH"
    assert "nih.gov" in reason


def test_subdomain_matches_root() -> None:
    tier, reason = classify_domain("https://docs.python.org/3/library")
    assert tier == "HIGH"
    assert "subdomain" in reason.lower() or "python.org" in reason


def test_news_domain_medium() -> None:
    tier, _ = classify_domain("https://www.reuters.com/world/article")
    assert tier == "MEDIUM"


def test_reddit_is_community() -> None:
    tier, _ = classify_domain("https://www.reddit.com/r/python/comments/x")
    assert tier == "COMMUNITY"


def test_generic_blog_platform_is_low() -> None:
    tier, _ = classify_domain("https://medium.com/@someone/a-post")
    assert tier == "LOW"


def test_unknown_domain() -> None:
    tier, reason = classify_domain("https://unknown-domain.xyz/post")
    assert tier == "UNKNOWN"
    assert "whitelist" in reason


def test_unparseable_url() -> None:
    tier, _ = classify_domain("")
    assert tier == "UNKNOWN"


# --- path downgrades -------------------------------------------------------


def test_marketing_path_downgrades_known_domain_to_low() -> None:
    tier, reason = classify_domain("https://www.reuters.com/resources/whitepaper")
    assert tier == "LOW"
    assert "/resources/" in reason


def test_non_marketing_path_keeps_base_tier() -> None:
    tier, _ = classify_domain("https://www.reuters.com/world/article-123")
    assert tier == "MEDIUM"


def test_unknown_domain_with_marketing_path_stays_unknown() -> None:
    """An unknown domain is never promoted into LOW just by a marketing path."""
    tier, _ = classify_domain("https://someblog.io/blog/post")
    assert tier == "UNKNOWN"


# --- path upgrades (Appendix A: github/huggingface primary artifacts) ------


def test_github_root_is_community() -> None:
    tier, _ = classify_domain("https://github.com/someuser")
    assert tier == "COMMUNITY"


def test_github_release_artifact_upgrades_to_medium() -> None:
    tier, reason = classify_domain("https://github.com/org/repo/releases/tag/v1.0.0")
    assert tier == "MEDIUM"
    assert "github.com" in reason


def test_huggingface_dataset_artifact_upgrades_to_medium() -> None:
    tier, _ = classify_domain("https://huggingface.co/datasets/some/set")
    assert tier == "MEDIUM"


# --- domain-scoped downgrades (Appendix A rule 6: contributor / native ads) -


def test_forbes_contributor_path_downgrades_to_low() -> None:
    tier, reason = classify_domain("https://www.forbes.com/sites/someone/2026/a-post")
    assert tier == "LOW"
    assert "/sites/" in reason


def test_forbes_editorial_path_stays_medium() -> None:
    tier, _ = classify_domain("https://www.forbes.com/regular-article")
    assert tier == "MEDIUM"


def test_scoped_pattern_does_not_downgrade_other_domains() -> None:
    """/sites/ is a Forbes-only rule — a gov asset path must not be demoted."""
    tier, _ = classify_domain("https://www.nih.gov/sites/default/files/report.pdf")
    assert tier == "HIGH"


def test_linkedin_pulse_is_low() -> None:
    tier, _ = classify_domain("https://www.linkedin.com/pulse/some-article")
    assert tier == "LOW"


# --- BLOCKED hard-exclude --------------------------------------------------


def test_blocked_domain_classifies_blocked() -> None:
    tier, _ = classify_domain("https://4chan.org/b/")
    assert tier == "BLOCKED"


def test_blocked_subdomain_matches_root() -> None:
    tier, reason = classify_domain("https://boards.4chan.org/b/")
    assert tier == "BLOCKED"
    assert "subdomain" in reason.lower()


def test_blocked_ignores_path_downgrade() -> None:
    """BLOCKED is a hard exclude — a /blog/ path must never promote it to LOW."""
    tier, _ = classify_domain("https://4chan.org/blog/post")
    assert tier == "BLOCKED"


# --- classify_url (public entry, folds in the undated downgrade) -----------


def test_classify_url_returns_bare_tier() -> None:
    assert classify_url("https://nih.gov/x", published_date="2026-01-01") == "HIGH"


def test_classify_url_undated_high_to_medium() -> None:
    assert classify_url("https://nih.gov/x") == "MEDIUM"


def test_classify_url_undated_medium_to_low() -> None:
    assert classify_url("https://reuters.com/world/x") == "LOW"


def test_classify_url_dated_medium_stays_medium() -> None:
    assert classify_url("https://reuters.com/world/x", published_date="2026-01-01") == "MEDIUM"


def test_classify_url_blocked_stays_blocked_even_undated() -> None:
    assert classify_url("https://4chan.org/b/") == "BLOCKED"


# --- downgrade_if_undated --------------------------------------------------


def test_downgrade_undated_high_to_medium() -> None:
    assert downgrade_if_undated("HIGH", False) == "MEDIUM"


def test_downgrade_undated_medium_to_low() -> None:
    assert downgrade_if_undated("MEDIUM", False) == "LOW"


def test_dated_high_stays_high() -> None:
    assert downgrade_if_undated("HIGH", True) == "HIGH"


@pytest.mark.parametrize("tier", ["LOW", "COMMUNITY", "BLOCKED", "UNKNOWN"])
def test_downgrade_lower_tiers_noop(tier: str) -> None:
    assert downgrade_if_undated(tier, False) == tier  # type: ignore[arg-type]


# --- classify_bulk ---------------------------------------------------------


def test_classify_bulk_preserves_keys() -> None:
    urls = ["https://nih.gov/a", "https://reddit.com/r/x", "https://unknown.example.org"]
    out = classify_bulk(urls)
    assert set(out.keys()) == set(urls)
    assert out["https://nih.gov/a"][0] == "HIGH"
    assert out["https://reddit.com/r/x"][0] == "COMMUNITY"
    assert out["https://unknown.example.org"][0] == "UNKNOWN"


# --- high_tier_domains -----------------------------------------------------


def test_high_tier_domains_only_high() -> None:
    domains = high_tier_domains()
    assert "nih.gov" in domains
    assert "reddit.com" not in domains  # COMMUNITY
    assert "medium.com" not in domains  # LOW
    assert "reuters.com" not in domains  # MEDIUM
    assert "4chan.org" not in domains  # BLOCKED


def test_high_tier_domains_stable_order() -> None:
    assert high_tier_domains() == high_tier_domains()


def test_high_tier_domains_excludes_pattern_style(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(authority._DOMAIN_TO_TIER, "*.example.gov", ("HIGH", "pattern"))
    monkeypatch.setitem(authority._DOMAIN_TO_TIER, "<placeholder>.org", ("HIGH", "pattern"))
    domains = high_tier_domains()
    assert "*.example.gov" not in domains
    assert "<placeholder>.org" not in domains


# Curated-core recovery allowlist: the first 20 HIGH domains must be the
# hand-curated core, not alphabetical imports.
_CURATED_CORE_SAMPLE = {
    "cms.gov", "hhs.gov", "oig.hhs.gov", "federalregister.gov", "cdc.gov",
    "nih.gov", "fda.gov", "medicare.gov", "medicaid.gov", "bls.gov",
    "sec.gov", "census.gov", "usa.gov", "mgma.com", "hfma.org",
    "ama-assn.org", "aad.org", "aga.org", "asds.net", "nejm.org",
    "jamanetwork.com", "bmj.com", "thelancet.com", "sciencedirect.com",
}


def test_high_tier_domains_first_20_are_curated_core() -> None:
    top20 = high_tier_domains()[:20]
    assert len(top20) == 20
    assert set(top20) <= _CURATED_CORE_SAMPLE, f"non-curated domain in top 20: {top20}"


def test_high_tier_domains_never_includes_blocked() -> None:
    domains = high_tier_domains()
    assert "4chan.org" not in domains
    assert "infowars.com" not in domains


# --- unknown-domain logging (injected output_dir, never repo-relative) -----


def test_unknown_domain_logged_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "unknown_domains.log"
    monkeypatch.setattr(authority, "_UNKNOWN_LOG", log_path)
    authority._UNKNOWN_SEEN.clear()

    classify_domain("https://brand-new-domain-abc.example/a")
    classify_domain("https://brand-new-domain-abc.example/b")  # dup — no re-append

    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert lines.count("brand-new-domain-abc.example") == 1


def test_set_output_dir_directs_unknown_log(tmp_path: Path) -> None:
    set_output_dir(tmp_path)
    authority._UNKNOWN_SEEN.clear()
    classify_domain("https://yet-another-unknown-qqq.example/x")
    assert (tmp_path / "unknown_domains.log").exists()


# --- POLYSEARCH_DOMAIN_TIERS env override ----------------------------------


def test_env_override_custom_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom_tiers.yaml"
    custom.write_text("high:\n  curated_core:\n    - onlyme.test\n", encoding="utf-8")
    monkeypatch.setenv("POLYSEARCH_DOMAIN_TIERS", str(custom))
    try:
        reload()
        assert classify_domain("https://onlyme.test/x")[0] == "HIGH"
        assert classify_domain("https://nih.gov/x")[0] == "UNKNOWN"  # not in custom map
    finally:
        monkeypatch.delenv("POLYSEARCH_DOMAIN_TIERS", raising=False)
        reload()


# --- SME tier dropped ------------------------------------------------------


def test_sme_tier_dropped_from_type() -> None:
    assert "SME" not in get_args(Tier)


# --- performance guard (dict lookup, not linear scan) ----------------------


def test_classify_stays_dict_lookup_at_scale() -> None:
    assert len(authority._DOMAIN_TO_TIER) > 10_000
    urls = [f"https://nih.gov/page-{i}" for i in range(5_000)] + [
        f"https://4chan.org/page-{i}" for i in range(5_000)
    ]
    start = time.perf_counter()
    for url in urls:
        classify_domain(url)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"10k classifications took {elapsed:.3f}s — expected a dict lookup"


# --- leakage guard: shipped YAML carries ZERO banned strings ---------------
# The banned list lives once in tests/_leakage_terms.py (the leakage gate owns
# it); import it here so the raw token literals live in a single file.
from tests._leakage_terms import BANNED_TOKENS as _BANNED


def test_shipped_yaml_has_no_banned_strings() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (
        root / "src" / "polysearch" / "data" / "domain_tiers.yaml"
    ).read_text(encoding="utf-8").lower()
    hits = [tok for tok in _BANNED if tok in text]
    assert not hits, f"banned strings leaked into shipped YAML: {hits}"
