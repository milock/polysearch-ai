"""Source-authority classifier.

Maps URLs to research-quality tiers (HIGH / MEDIUM / LOW / COMMUNITY / BLOCKED /
UNKNOWN), backed by the domain map in ``polysearch/data/domain_tiers.yaml``. Downstream
grounding, community, and citation-verification layers use the tier to prefer
authoritative sources, drop BLOCKED ones, and flag LOW/UNKNOWN ones.

Public entry point is :func:`classify_url`, which returns a bare :data:`Tier`
and folds in the undated downgrade (an undated source loses one tier of trust).
:func:`classify_domain` exposes the underlying ``(tier, reason)`` pair.

Configuration:
- ``POLYSEARCH_DOMAIN_TIERS`` overrides the YAML path.
- Unknown domains are appended (de-duplicated) to ``unknown_domains.log`` under
  the injected output directory (``Settings.output_dir``) — never a path
  relative to the installed package.
"""

from __future__ import annotations

import importlib.resources as resources
import logging
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml

log = logging.getLogger(__name__)

Tier = Literal["HIGH", "MEDIUM", "LOW", "COMMUNITY", "BLOCKED", "UNKNOWN"]

# The domain map ships inside the package at polysearch/data/domain_tiers.yaml,
# so it resolves the same in a checkout and in an installed wheel. Overridable
# via POLYSEARCH_DOMAIN_TIERS.

# Undated downgrade: an undated source drops one tier of trust.
_DOWNGRADE_STEP: dict[str, Tier] = {"HIGH": "MEDIUM", "MEDIUM": "LOW"}

# Tier ranking used by the path-upgrade rule (higher = more authoritative).
_RANK: dict[str, int] = {"UNKNOWN": 0, "COMMUNITY": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4}

# ── Module caches (populated at import and by reload()) ──────────────────────
_DOMAIN_TO_TIER: dict[str, tuple[Tier, str]] = {}
_PATH_DOWNGRADES: list[str] = []
_PATH_DOWNGRADES_BY_DOMAIN: dict[str, list[str]] = {}
_PATH_UPGRADES: dict[str, tuple[Tier, list[str]]] = {}
_UNKNOWN_SEEN: set[str] = set()

# Resolved lazily (or via set_output_dir); never a package-relative path.
_UNKNOWN_LOG: Path | None = None


def _config_path() -> Path:
    override = os.environ.get("POLYSEARCH_DOMAIN_TIERS")
    if override:
        return Path(override)
    return Path(resources.files("polysearch") / "data" / "domain_tiers.yaml")


def _flatten(section: object) -> list[str]:
    """Flatten a nested dict-of-lists tier section into a flat domain list."""
    out: list[str] = []
    if isinstance(section, dict):
        for val in section.values():
            out.extend(_flatten(val))
    elif isinstance(section, list):
        out.extend(str(x).strip().lower() for x in section if x)
    return out


def _load() -> None:
    """(Re)load tier mappings and path rules from YAML into module caches."""
    global _DOMAIN_TO_TIER, _PATH_DOWNGRADES, _PATH_DOWNGRADES_BY_DOMAIN, _PATH_UPGRADES
    _DOMAIN_TO_TIER = {}
    _PATH_DOWNGRADES = []
    _PATH_DOWNGRADES_BY_DOMAIN = {}
    _PATH_UPGRADES = {}

    path = _config_path()
    if not path.exists():
        log.warning("domain_tiers.yaml not found at %s", path)
        return

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    tier_reasons: list[tuple[str, Tier, str]] = [
        ("high", "HIGH", "whitelisted HIGH"),
        ("medium", "MEDIUM", "whitelisted MEDIUM"),
        ("low", "LOW", "whitelisted LOW"),
        ("community", "COMMUNITY", "whitelisted COMMUNITY"),
        ("blocked", "BLOCKED", "hard-excluded BLOCKED"),
    ]
    for key, tier, label in tier_reasons:
        for domain in _flatten(data.get(key)):
            _DOMAIN_TO_TIER.setdefault(domain, (tier, f"{label} ({domain})"))

    rules = data.get("path_rules") or {}
    _PATH_DOWNGRADES = [str(p).strip().lower() for p in (rules.get("downgrades") or []) if p]
    for domain, patterns in (rules.get("downgrades_by_domain") or {}).items():
        cleaned = [str(p).strip().lower() for p in (patterns or []) if p]
        if cleaned:
            _PATH_DOWNGRADES_BY_DOMAIN[domain.strip().lower()] = cleaned
    for domain, spec in (rules.get("upgrades") or {}).items():
        target = str(spec.get("to", "")).upper()
        patterns = [str(p).strip().lower() for p in (spec.get("patterns") or []) if p]
        if target in _RANK and patterns:
            _PATH_UPGRADES[domain.strip().lower()] = (target, patterns)  # type: ignore[assignment]


def reload() -> None:
    """Re-read the YAML file (honours ``POLYSEARCH_DOMAIN_TIERS``)."""
    _load()


def set_output_dir(output_dir: str | os.PathLike[str]) -> None:
    """Inject the directory the unknown-domain log is written under."""
    global _UNKNOWN_LOG
    _UNKNOWN_LOG = Path(output_dir) / "unknown_domains.log"


def _normalize(url: str) -> tuple[str, str]:
    """Return ``(host, path)``: host lowercased with a leading ``www.`` stripped."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except ValueError:
        return "", ""
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host, (parsed.path or "").lower()


def _root_domain(host: str) -> str:
    """Naive last-two-label root extraction (docs.cms.gov -> cms.gov)."""
    parts = host.split(".")
    return host if len(parts) <= 2 else ".".join(parts[-2:])


def _resolve_log_path() -> Path:
    if _UNKNOWN_LOG is not None:
        return _UNKNOWN_LOG
    # Lazy default from injected settings — the configured output dir, never a
    # path relative to the installed package.
    from polysearch.config import Settings

    return Path(Settings.from_env().output_dir) / "unknown_domains.log"


def _log_unknown(host: str) -> None:
    if not host or host in _UNKNOWN_SEEN:
        return
    _UNKNOWN_SEEN.add(host)
    path = _resolve_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(host + "\n")
    except OSError as exc:
        log.debug("unknown_domains log write failed: %s", exc)


def classify_domain(url: str) -> tuple[Tier, str]:
    """Classify ``url`` into a ``(tier, reason)`` pair.

    Order: exact/subdomain host match → BLOCKED short-circuit → global path
    downgrade → domain-scoped path downgrade (both force LOW) → path upgrade
    (raise a known primary artifact) → base tier. An unknown domain is never
    promoted or demoted by its path.
    """
    host, path = _normalize(url)
    if not host:
        return "UNKNOWN", "unparseable url"

    base: Tier | None = None
    reason = ""
    if host in _DOMAIN_TO_TIER:
        base, reason = _DOMAIN_TO_TIER[host]
    else:
        root = _root_domain(host)
        if root != host and root in _DOMAIN_TO_TIER:
            base, reason = _DOMAIN_TO_TIER[root]
            reason = f"{reason} [subdomain {host}]"

    if base is None:
        _log_unknown(host)
        return "UNKNOWN", "not in whitelist"

    # BLOCKED is a hard exclude — never softened by any path rule.
    if base == "BLOCKED":
        return base, reason

    for pattern in _PATH_DOWNGRADES:
        if pattern in path:
            return "LOW", f"vendor marketing path '{pattern}' on {host}"

    # Domain-scoped downgrades: contributor networks / native advertising on
    # specific hosts (e.g. forbes.com/sites/, linkedin.com/pulse/) — scoped so a
    # generic path like /sites/ never demotes an unrelated domain's own URLs
    # (e.g. a gov site's /sites/default/files/ asset path).
    scoped = _PATH_DOWNGRADES_BY_DOMAIN.get(host) or _PATH_DOWNGRADES_BY_DOMAIN.get(_root_domain(host))
    if scoped:
        for pattern in scoped:
            if pattern in path:
                return "LOW", f"contributor/native path '{pattern}' on {host}"

    upgrade = _PATH_UPGRADES.get(host) or _PATH_UPGRADES.get(_root_domain(host))
    if upgrade:
        target, patterns = upgrade
        if _RANK[target] > _RANK[base] and any(p in path for p in patterns):
            return target, f"primary-artifact path on {host} -> {target}"

    return base, reason


def downgrade_if_undated(tier: Tier, has_date: bool) -> Tier:
    """Demote HIGH->MEDIUM and MEDIUM->LOW when the source is undated."""
    if has_date:
        return tier
    return _DOWNGRADE_STEP.get(tier, tier)


def classify_url(url: str, *, published_date: object | None = None) -> Tier:
    """Classify ``url`` to a bare tier, applying the undated downgrade.

    ``published_date`` is treated as present when truthy; an absent date demotes
    the tier one step (see :func:`downgrade_if_undated`).
    """
    tier, _reason = classify_domain(url)
    return downgrade_if_undated(tier, bool(published_date))


def classify_bulk(urls: list[str]) -> dict[str, tuple[Tier, str]]:
    """Classify a batch of URLs, preserving the input URLs as keys."""
    return {u: classify_domain(u) for u in urls}


def high_tier_domains() -> list[str]:
    """HIGH-tier domains in stable load order (curated core first).

    Pattern-style entries (containing ``*`` or ``<``) are excluded so this can
    feed a literal domain filter. BLOCKED domains never appear.
    """
    return [
        domain
        for domain, (tier, _reason) in _DOMAIN_TO_TIER.items()
        if tier == "HIGH" and "*" not in domain and "<" not in domain
    ]


# Populate caches at import.
_load()
