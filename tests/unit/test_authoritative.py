"""Unit tests for polysearch.extractors.authoritative — schema-driven extraction.

The schema YAML *drives* extraction here (patterns are compiled and run), so
these tests exercise both the loader (config dir + POLYSEARCH_SCHEMA_DIR) and
the generic pattern engine against the Federal Register fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polysearch.extractors import authoritative
from polysearch.extractors.authoritative import (
    AuthoritativeSchema,
    ExtractedFact,
    Pattern,
    extract,
    extract_auto,
    load_schema,
    load_schemas,
    schema_for_url,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "authoritative"
    / "federal_register_sample.md"
)
_FR_URL = "https://www.federalregister.gov/documents/2025/11/15/2025-12345/medicare"


@pytest.fixture
def fr_content() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def fr_schema() -> AuthoritativeSchema:
    schemas = load_schemas()
    return schemas["federalregister.gov"]


# ── loader ──────────────────────────────────────────────────────────────────

def test_default_schemas_include_federalregister() -> None:
    schemas = load_schemas()
    assert "federalregister.gov" in schemas
    assert isinstance(schemas["federalregister.gov"], AuthoritativeSchema)


def test_load_schema_parses_patterns() -> None:
    path = (
        Path(authoritative.__file__).resolve().parents[3]
        / "config"
        / "authoritative_schemas"
        / "federalregister.yaml"
    )
    schema = load_schema(path)
    assert schema.domain == "federalregister.gov"
    names = {p.name for p in schema.patterns}
    assert {"docket_number", "effective_date", "cfr_cite"} <= names
    assert all(isinstance(p, Pattern) for p in schema.patterns)


# ── extraction against the fixture (adapted from internal cases) ─────────────

def test_extract_docket_number(fr_content: str, fr_schema: AuthoritativeSchema) -> None:
    facts = extract(_FR_URL, fr_content, fr_schema)
    dockets = [f.value for f in facts if f.name == "docket_number"]
    assert dockets == ["2025-12345"]


def test_extract_effective_date(fr_content: str, fr_schema: AuthoritativeSchema) -> None:
    facts = extract(_FR_URL, fr_content, fr_schema)
    eff = [f.value for f in facts if f.name == "effective_date"]
    assert eff == ["January 1, 2026"]


def test_extract_cfr_cites(fr_content: str, fr_schema: AuthoritativeSchema) -> None:
    facts = extract(_FR_URL, fr_content, fr_schema)
    cites = [f.value for f in facts if f.name == "cfr_cite"]
    assert "42 CFR Part 410" in cites
    assert "42 CFR Part 414" in cites
    assert "45 CFR Part 170" in cites


def test_extracted_facts_carry_source_and_context(
    fr_content: str, fr_schema: AuthoritativeSchema
) -> None:
    facts = extract(_FR_URL, fr_content, fr_schema)
    assert facts, "expected at least one fact"
    for f in facts:
        assert isinstance(f, ExtractedFact)
        assert f.source_url == _FR_URL
        assert f.domain == "federalregister.gov"
        assert f.value in f.context


def test_duplicate_matches_deduped(fr_schema: AuthoritativeSchema) -> None:
    content = "Document Number: 2025-99999\nDocument Number: 2025-99999"
    facts = extract(_FR_URL, content, fr_schema)
    dockets = [f.value for f in facts if f.name == "docket_number"]
    assert dockets == ["2025-99999"]


# ── unknown domain → empty ──────────────────────────────────────────────────

def test_unknown_domain_returns_empty(
    fr_content: str, fr_schema: AuthoritativeSchema
) -> None:
    facts = extract("https://example.com/whatever", fr_content, fr_schema)
    assert facts == []


def test_extract_auto_unknown_domain_returns_empty(fr_content: str) -> None:
    facts = extract_auto("https://example.com/whatever", fr_content)
    assert facts == []


def test_schema_for_url_none_when_unmatched() -> None:
    schemas = load_schemas()
    assert schema_for_url("https://example.com/x", schemas) is None


def test_extract_auto_matches_federalregister(fr_content: str) -> None:
    facts = extract_auto(_FR_URL, fr_content)
    assert any(f.name == "docket_number" for f in facts)


# ── user schema dir extends the built-in set ────────────────────────────────

def test_user_schema_dir_extends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_schema = tmp_path / "example.yaml"
    user_schema.write_text(
        "domain: example.gov\n"
        "patterns:\n"
        "  - name: order_id\n"
        "    regex: 'Order No\\.?\\s*([0-9]+)'\n"
        "    context_window: 20\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYSEARCH_SCHEMA_DIR", str(tmp_path))
    schemas = load_schemas()
    # Built-in still present…
    assert "federalregister.gov" in schemas
    # …and the user domain was added.
    assert "example.gov" in schemas

    facts = extract(
        "https://data.example.gov/records", "Order No. 8841 shipped", schemas["example.gov"]
    )
    assert [f.value for f in facts if f.name == "order_id"] == ["8841"]


def test_user_schema_dir_overrides_same_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "federalregister.yaml"
    override.write_text(
        "domain: federalregister.gov\n"
        "patterns:\n"
        "  - name: only_pattern\n"
        "    regex: 'RIN\\s+([0-9A-Za-z-]+)'\n"
        "    context_window: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYSEARCH_SCHEMA_DIR", str(tmp_path))
    schemas = load_schemas()
    names = {p.name for p in schemas["federalregister.gov"].patterns}
    assert names == {"only_pattern"}


# ── robustness ──────────────────────────────────────────────────────────────

def test_empty_content_yields_no_facts(fr_schema: AuthoritativeSchema) -> None:
    assert extract(_FR_URL, "", fr_schema) == []


def test_subdomain_matches(fr_schema: AuthoritativeSchema) -> None:
    facts = extract(
        "https://sub.federalregister.gov/x", "Document Number: 2025-777", fr_schema
    )
    assert [f.value for f in facts if f.name == "docket_number"] == ["2025-777"]
