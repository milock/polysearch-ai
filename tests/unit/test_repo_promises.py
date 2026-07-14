"""Repo-promise regression tests.

The README, the installer, and the plugin manifest all promise that certain
files exist and stay in sync. An earlier version of the project advertised a
skill, an installer, and config files that were not actually in the tree; these
tests exist so that gap cannot reappear silently.

Every path the README's "What's in the box" table links to must resolve, the
plugin and pyproject versions must stay locked together, and the shipped
manifests must be valid JSON with the fields the tooling requires.
"""

from __future__ import annotations

import importlib.resources
import json
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


# ── README-referenced paths ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "relpath",
    [
        "install.sh",
        "config/domain_tiers.yaml",
        "skills/research/SKILL.md",
        "agents/researcher.md",
        "examples/quickstart.py",
        "examples/tiers.md",
        "docs/agent-integration.md",
        ".claude-plugin/plugin.json",
        ".claude-plugin/marketplace.json",
        ".env.example",
        "LICENSE",
    ],
)
def test_promised_file_exists(relpath: str) -> None:
    assert _p(relpath).is_file(), f"README/installer promises {relpath}, but it is missing"


@pytest.mark.parametrize("reldir", ["agents", "examples", "examples/sample-output", "docs", "skills/research"])
def test_promised_dir_exists(reldir: str) -> None:
    assert _p(reldir).is_dir(), f"promised directory {reldir} is missing"


def test_sample_output_has_a_report() -> None:
    reports = list(_p("examples", "sample-output").glob("*.md"))
    assert reports, "examples/sample-output/ must ship at least one sample report"


def test_sample_output_is_marked_synthetic() -> None:
    for report in _p("examples", "sample-output").glob("*.md"):
        assert "SYNTHETIC" in report.read_text(), (
            f"{report.name} must be clearly marked SYNTHETIC so it is never mistaken for real research"
        )


# ── install.sh contract ──────────────────────────────────────────────────────


def test_install_sh_is_executable_and_posix() -> None:
    install = _p("install.sh")
    text = install.read_text()
    assert text.startswith("#!/bin/sh"), "install.sh must be POSIX sh"
    import os

    assert os.access(install, os.X_OK), "install.sh must be executable"


# ── manifests are valid + version-locked ─────────────────────────────────────


def _pyproject_version() -> str:
    with _p("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def test_plugin_manifest_is_valid_json_with_required_fields() -> None:
    data = json.loads(_p(".claude-plugin", "plugin.json").read_text())
    for field in ("name", "description", "version", "author"):
        assert field in data, f"plugin.json missing required field: {field}"
    assert data["name"] == "polysearch"


def test_marketplace_manifest_lists_the_plugin() -> None:
    data = json.loads(_p(".claude-plugin", "marketplace.json").read_text())
    assert data.get("name")
    names = {p.get("name") for p in data.get("plugins", [])}
    assert "polysearch" in names, "marketplace.json must list the polysearch plugin"


def test_plugin_version_tracks_pyproject() -> None:
    """The plugin version must move in lockstep with the package version so a
    plugin update always maps to a released package version."""
    plugin = json.loads(_p(".claude-plugin", "plugin.json").read_text())
    assert plugin["version"] == _pyproject_version(), (
        "plugin.json version must equal pyproject version (bump both together at release)"
    )


def test_marketplace_plugin_version_tracks_pyproject() -> None:
    data = json.loads(_p(".claude-plugin", "marketplace.json").read_text())
    entry = next(p for p in data["plugins"] if p.get("name") == "polysearch")
    if "version" in entry:
        assert entry["version"] == _pyproject_version()


def test_package_version_matches_pyproject() -> None:
    from polysearch import __version__

    assert __version__ == _pyproject_version()


# ── skill frontmatter ────────────────────────────────────────────────────────


def test_skill_has_name_and_description_frontmatter() -> None:
    text = _p("skills", "research", "SKILL.md").read_text()
    assert text.startswith("---"), "SKILL.md must open with YAML frontmatter"
    fm = text.split("---", 2)[1]
    assert "name:" in fm and "description:" in fm


# ── ledgered packaging gap (TODO(23)) ────────────────────────────────────────


@pytest.mark.xfail(
    reason="TODO(23): config/domain_tiers.yaml and config/authoritative_schemas/*.yaml "
    "are located via Path(__file__).parents[3]/config (repo root) and are NOT bundled "
    "into the wheel — an installed package cannot find them and every source downgrades "
    "to UNKNOWN. Fix at release: move config/ under src/polysearch/data/, switch "
    "authority.py and extractors/authoritative.py to importlib.resources, and add the "
    "package-data glob. This xfail flips to pass once the resource is importable.",
    strict=True,
)
def test_domain_tiers_yaml_ships_as_package_resource() -> None:
    resource = importlib.resources.files("polysearch") / "data" / "domain_tiers.yaml"
    assert resource.is_file()
