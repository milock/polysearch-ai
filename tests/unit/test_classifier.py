"""Unit tests for polysearch.classifier — rule-based query classifier.

Ported from the internal research-pipeline trigger tests, minus the
vertical-specific cases (those live in a user ``DomainProfile`` now, not the
package). Adds profile-driven coverage: a custom profile turns generic tokens
into recognised entities/competitors.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from polysearch.classifier import (
    Classification,
    DomainProfile,
    classify,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── core classification (no domain profile) ────────────────────────────────

@pytest.mark.parametrize(
    "topic,expected",
    [
        (
            "What is a webhook?",
            {
                "query_type": "FACTUAL",
                "suggested_depth": "skip",
                "time_sensitive": False,
            },
        ),
        ("Peter Steinberger", {"query_type": "PERSON"}),
        ("Postgres vs MySQL", {"query_type": "COMPARISON"}),
        ("Notion API pricing", {"query_type": "PRODUCT"}),
        ("deep dive open source telemetry", {"suggested_depth": "deep"}),
        (
            "latest Kubernetes releases this week",
            {"time_sensitive": True, "suggested_depth": "standard"},
        ),
        ("what's happening with Rust", {"time_sensitive": True}),
        (
            "everything about distributed consensus algorithms",
            {"suggested_depth": "deep"},
        ),
    ],
)
def test_classify_core(topic, expected):
    result = classify(topic)
    for k, v in expected.items():
        actual = getattr(result, k)
        assert actual == v, f"{k} for {topic!r}: expected {v}, got {actual}"


def test_returns_classification_instance():
    assert isinstance(classify("Postgres vs MySQL"), Classification)


def test_no_profile_means_not_domain_related():
    result = classify("AcmeCo pricing")
    assert result.domain_related is False
    assert result.competitive_mode is False
    assert "--domain-context" not in result.flags


def test_depth_flag_always_present():
    for topic in [
        "What is a webhook?",
        "Peter Steinberger",
        "everything about distributed consensus algorithms",
    ]:
        result = classify(topic)
        assert any(f.startswith("--depth=") for f in result.flags), (
            f"no --depth= flag for {topic!r}: {result.flags}"
        )


def test_runtime_and_cost_per_depth():
    cases = {
        "quick": (45, 0.04),
        "standard": (150, 0.12),
        "deep": (300, 0.35),
        "skip": (5, 0.01),
    }
    topics_by_depth = {
        "skip": "What is a webhook?",
        "standard": "Peter Steinberger",
        "deep": "deep dive open source telemetry",
        "quick": "puppy training tips",
    }
    for depth, topic in topics_by_depth.items():
        result = classify(topic)
        assert result.suggested_depth == depth, (
            f"{topic!r} expected depth={depth}, got {result.suggested_depth}"
        )
        exp_runtime, exp_cost = cases[depth]
        assert result.cost_estimate.runtime_sec == exp_runtime
        assert result.cost_estimate.cost_usd == exp_cost


def test_x_handle_flag_and_person_type():
    result = classify("@steipete latest launches")
    assert result.query_type == "PERSON"
    assert "--x-handle=steipete" in result.flags


def test_github_flags():
    result = classify("github.com/rails/rails internals")
    assert "--github-user=rails" in result.flags
    assert "--github-repo=rails" in result.flags


def test_recency_week_when_time_sensitive():
    result = classify("latest Kubernetes releases this week")
    assert "--recency=week" in result.flags


def test_recency_month_by_default():
    result = classify("Peter Steinberger")
    assert "--recency=month" in result.flags


def test_reasons_non_empty():
    assert len(classify("Postgres vs MySQL").reasons) >= 1


# ── domain-profile-driven classification ───────────────────────────────────

@pytest.fixture
def acme_profile() -> DomainProfile:
    return DomainProfile(
        name="acme",
        entities=["AcmeCo", "BetaCorp"],
        competitors=["AcmeCo", "BetaCorp"],
        keywords=["widget"],
    )


def test_profile_comparison_is_competitive(acme_profile):
    result = classify("AcmeCo vs BetaCorp", profile=acme_profile)
    assert result.query_type == "COMPARISON"
    assert result.competitive_mode is True
    assert result.domain_related is True
    assert "--competitive" in result.flags


def test_profile_entity_classifies_as_product(acme_profile):
    # "AcmeCo" reads like a Title-case name, but a profile entity preempts PERSON.
    result = classify("AcmeCo pricing", profile=acme_profile)
    assert result.query_type == "PRODUCT"
    assert result.domain_related is True
    assert "--domain-context" in result.flags


def test_profile_keyword_marks_domain_related(acme_profile):
    result = classify("widget throughput tuning", profile=acme_profile)
    assert result.domain_related is True
    assert "--domain-context" in result.flags


# ── DomainProfile loading ──────────────────────────────────────────────────

def test_domain_profile_from_yaml(tmp_path: Path):
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            name: acme
            entities:
              - AcmeCo
              - BetaCorp
            competitors:
              - BetaCorp
            keywords:
              - widget
            """
        )
    )
    profile = DomainProfile.from_yaml(yaml_path)
    assert profile.name == "acme"
    assert profile.entities == ["AcmeCo", "BetaCorp"]
    assert profile.competitors == ["BetaCorp"]
    assert profile.keywords == ["widget"]


def test_domain_profile_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text("name: acme\ncompetitors:\n  - AcmeCo\n")
    monkeypatch.setenv("POLYSEARCH_DOMAIN_PROFILE", str(yaml_path))
    profile = DomainProfile.from_env()
    assert profile is not None
    assert profile.name == "acme"
    assert profile.competitors == ["AcmeCo"]


def test_domain_profile_from_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POLYSEARCH_DOMAIN_PROFILE", raising=False)
    assert DomainProfile.from_env() is None


def test_classify_loads_env_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text("name: acme\ncompetitors:\n  - AcmeCo\n  - BetaCorp\n")
    monkeypatch.setenv("POLYSEARCH_DOMAIN_PROFILE", str(yaml_path))
    result = classify("AcmeCo vs BetaCorp")
    assert result.competitive_mode is True
    assert "--competitive" in result.flags


# ── performance + CLI ──────────────────────────────────────────────────────

def test_classify_is_fast():
    import time

    start = time.perf_counter()
    for _ in range(50):
        classify("Postgres vs MySQL performance benchmarks")
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    assert elapsed_ms < 10, f"classify averaged {elapsed_ms:.2f}ms (>10ms budget)"


def test_cli_mode_returns_json():
    proc = subprocess.run(
        [sys.executable, "-m", "polysearch.classifier", "Postgres vs MySQL"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    for key in (
        "topic",
        "time_sensitive",
        "query_type",
        "domain_related",
        "competitive_mode",
        "suggested_depth",
        "flags",
        "cost_estimate",
        "reasons",
    ):
        assert key in data, f"missing key {key} in CLI output"
    assert data["query_type"] == "COMPARISON"
    assert data["cost_estimate"]["runtime_sec"] > 0
