"""Schema-driven authoritative-fact extraction.

Given the markdown/text of a primary-source page, this module extracts
structured facts using regex patterns declared in a YAML schema. The schema
is the single source of truth: it names a ``domain`` and a list of
``patterns`` — each ``{name, regex, context_window}`` — and the engine runs
those patterns directly. There is no per-domain hard-coded logic; adding a
new authoritative source is a matter of dropping a sibling YAML into the
bundled ``polysearch/data/authoritative_schemas/`` or a directory named by
``$POLYSEARCH_SCHEMA_DIR``.

Public API::

    load_schemas() -> dict[str, AuthoritativeSchema]   # keyed by domain
    load_schema(path) -> AuthoritativeSchema
    schema_for_url(url, schemas) -> AuthoritativeSchema | None
    extract(url, content, schema) -> list[ExtractedFact]
    extract_auto(url, content, schemas=None) -> list[ExtractedFact]

``extract`` returns ``[]`` when ``url``'s domain does not match the schema's
domain; ``extract_auto`` returns ``[]`` when no loaded schema matches the URL.
"""

from __future__ import annotations

import importlib.resources as resources
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel


class SchemaError(ValueError):
    """Raised when a schema file is malformed — a missing required key or a
    pattern whose regex does not compile. The message names the offending
    file, domain, and pattern so a typo in a user-dropped YAML is actionable
    rather than an opaque traceback."""


class Pattern(BaseModel):
    """One named extraction rule. If ``regex`` has a capturing group, group 1
    is the fact value; otherwise the whole match is used."""

    name: str
    regex: str
    context_window: int = 0


class AuthoritativeSchema(BaseModel):
    """A domain's extraction schema: which host it applies to and its patterns."""

    domain: str
    patterns: list[Pattern] = []


class ExtractedFact(BaseModel):
    """A single fact pulled from source content by one pattern match."""

    name: str
    value: str
    context: str
    source_url: str
    domain: str


# ---------- Domain helpers ------------------------------------------------


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _domain_matches(host: str, domain: str) -> bool:
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)


# ---------- Schema loading ------------------------------------------------


def _default_schema_dir() -> Path:
    """The bundled schema directory (``polysearch/data/authoritative_schemas``).

    Ships inside the package, so it resolves the same in a checkout and in an
    installed wheel.
    """
    return Path(resources.files("polysearch") / "data" / "authoritative_schemas")


def load_schema(path: str | Path) -> AuthoritativeSchema:
    """Parse a single schema YAML into an ``AuthoritativeSchema``.

    Validates eagerly: a missing ``domain`` key or a pattern whose ``regex``
    does not compile raises :class:`SchemaError` naming the file (and, for a
    bad regex, the domain and pattern) rather than failing later mid-extract.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "domain" not in data:
        raise SchemaError(f"schema file {path} is missing required key 'domain'")
    domain = data["domain"]
    patterns = [Pattern(**p) for p in data.get("patterns", [])]
    for pattern in patterns:
        try:
            _compile(pattern.regex)  # eager: populates cache, surfaces bad regex now
        except re.error as e:
            raise SchemaError(
                f"invalid regex for pattern '{pattern.name}' in schema "
                f"'{domain}' ({path}): {e}"
            ) from e
    return AuthoritativeSchema(domain=domain, patterns=patterns)


def load_schemas(
    schema_dirs: list[str | Path] | None = None,
) -> dict[str, AuthoritativeSchema]:
    """Load every ``*.yaml`` schema, keyed by domain.

    When ``schema_dirs`` is omitted, the bundled directory is read first, then
    ``$POLYSEARCH_SCHEMA_DIR`` (if set) — so a user directory *extends* the
    built-in set and *overrides* any schema for the same domain.
    """
    if schema_dirs is None:
        dirs: list[Path] = [_default_schema_dir()]
        user_dir = os.environ.get("POLYSEARCH_SCHEMA_DIR")
        if user_dir and user_dir.strip():
            dirs.append(Path(user_dir.strip()))
    else:
        dirs = [Path(d) for d in schema_dirs]

    schemas: dict[str, AuthoritativeSchema] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            schema = load_schema(path)
            schemas[schema.domain] = schema
    return schemas


def schema_for_url(
    url: str, schemas: dict[str, AuthoritativeSchema]
) -> AuthoritativeSchema | None:
    """Return the schema whose domain matches ``url``, or ``None``."""
    host = _domain(url)
    for schema in schemas.values():
        if _domain_matches(host, schema.domain):
            return schema
    return None


# ---------- Extraction ----------------------------------------------------


@lru_cache(maxsize=256)
def _compile(regex: str) -> re.Pattern[str]:
    return re.compile(regex)


def extract(
    url: str, content: str, schema: AuthoritativeSchema
) -> list[ExtractedFact]:
    """Run ``schema``'s patterns over ``content`` and return the facts found.

    Returns ``[]`` if ``url``'s domain does not match the schema's domain.
    Each unique ``(pattern name, value)`` yields one fact; duplicates within
    the same content are collapsed.
    """
    if not _domain_matches(_domain(url), schema.domain):
        return []

    facts: list[ExtractedFact] = []
    seen: set[tuple[str, str]] = set()
    for pattern in schema.patterns:
        rx = _compile(pattern.regex)
        for match in rx.finditer(content):
            value = (match.group(1) if match.groups() else match.group(0)).strip()
            if not value:
                continue
            key = (pattern.name, value)
            if key in seen:
                continue
            seen.add(key)
            window = pattern.context_window
            start = max(0, match.start() - window)
            end = min(len(content), match.end() + window)
            facts.append(
                ExtractedFact(
                    name=pattern.name,
                    value=value,
                    context=content[start:end].strip(),
                    source_url=url,
                    domain=schema.domain,
                )
            )
    return facts


def extract_auto(
    url: str,
    content: str,
    schemas: dict[str, AuthoritativeSchema] | None = None,
) -> list[ExtractedFact]:
    """Pick the schema matching ``url`` and extract; ``[]`` if none matches."""
    if schemas is None:
        schemas = load_schemas()
    schema = schema_for_url(url, schemas)
    if schema is None:
        return []
    return extract(url, content, schema)
