"""Rule-based query classifier with pluggable domain profiles.

Tells the pipeline what kind of research task a topic string represents and what
flags to pass downstream. Pure rules — fast (<10ms), no network, no LLM.

The classifier ships with zero vertical knowledge. To teach it about your own
entities, competitors, and topic vocabulary, supply a ``DomainProfile`` — either
passed explicitly or loaded from a YAML file named by the
``POLYSEARCH_DOMAIN_PROFILE`` environment variable.

Usage:
    from polysearch.classifier import classify
    result = classify("Postgres vs MySQL")

CLI:
    python3 -m polysearch.classifier "Postgres vs MySQL"
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

QueryType = Literal[
    "PERSON",
    "PRODUCT",
    "COMPARISON",
    "THEMATIC",
    "FACTUAL",
]

Depth = Literal["quick", "standard", "deep"]


class CostEstimate(BaseModel):
    """Rough runtime and dollar estimate for a run at the suggested depth."""

    runtime_sec: int
    cost_usd: float


class Classification(BaseModel):
    """The classifier's verdict for one topic string."""

    topic: str
    query_type: QueryType
    time_sensitive: bool
    domain_related: bool
    competitive_mode: bool = False
    suggested_depth: Depth
    # ``flags`` only ever holds flags the public CLI actually accepts (today just
    # ``--depth=``). Other extracted signals live as typed fields below, not as
    # invented flag strings.
    flags: list[str]
    x_handle: str | None = None
    github_user: str | None = None
    github_repo: str | None = None
    cost_estimate: CostEstimate
    reasons: list[str]


# ---------- domain profile ----------


class DomainProfile(BaseModel):
    """User-supplied vocabulary that makes the classifier domain-aware.

    ``entities`` are known products/vendors (matching one preempts a PERSON
    guess and yields PRODUCT). ``competitors`` is the subset whose mention turns
    on competitive mode. ``keywords`` are topic terms that mark a query as
    domain-related without changing its type. All matching is word-boundary,
    case-insensitive, and literal (list items are regex-escaped).
    """

    name: str = "default"
    entities: list[str] = []
    competitors: list[str] = []
    keywords: list[str] = []

    @classmethod
    def from_yaml(cls, path: str | Path) -> DomainProfile:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> DomainProfile | None:
        """Load the profile named by ``POLYSEARCH_DOMAIN_PROFILE``, or None."""
        raw = os.environ.get("POLYSEARCH_DOMAIN_PROFILE")
        if raw is None or not raw.strip():
            return None
        return cls.from_yaml(raw.strip())

    def _match(self, terms: list[str], topic: str) -> str | None:
        if not terms:
            return None
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE
        )
        m = pattern.search(topic)
        return m.group(0) if m else None

    def match_entity(self, topic: str) -> str | None:
        return self._match(self.entities, topic)

    def match_competitor(self, topic: str) -> str | None:
        return self._match(self.competitors, topic)

    def match_keyword(self, topic: str) -> str | None:
        return self._match(self.keywords, topic)


# ---------- regex patterns ----------

_TIME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(latest|recent|recently|this\s+week|this\s+month|last\s+30\s+days|"
            r"current|now|today|happening|just\s+(?:launched|released|announced)|"
            r"breaking|update)\b",
            re.IGNORECASE,
        ),
        "recency keyword",
    ),
    (
        re.compile(
            r"\b(?:before|prep(?:aring)?\s+for|ahead\s+of)\s+(?:my\s+)?"
            r"(?:meeting|call|interview|presentation)\b",
            re.IGNORECASE,
        ),
        "prepping for meeting",
    ),
    (
        re.compile(r"\b(?:what'?s\s+happening|what'?s\s+new)\b", re.IGNORECASE),
        "what's happening / what's new",
    ),
]

_COMPARISON_RE = re.compile(
    r"\b(vs\.?|versus|compared\s+to|better\s+than)\b", re.IGNORECASE
)
# "A or B" style comparison needs two Title-Case tokens on each side.
_OR_COMPARISON_RE = re.compile(
    r"\b([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)*)\s+or\s+([A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+)*)\b"
)

_PRODUCT_TERMS_RE = re.compile(
    r"\b(platform|app|tool|API|SaaS|software|service|library|framework|SDK)\b",
    re.IGNORECASE,
)

_HANDLE_RE = re.compile(r"@(\w+)")
_GITHUB_RE = re.compile(r"github\.com/([\w-]+)(?:/([\w.-]+))?", re.IGNORECASE)
_TITLE_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")

_DEEP_RE = re.compile(
    r"\b(deep\s+dive|comprehensive|thorough|exhaustive|"
    r"detailed\s+analysis|everything\s+about)\b",
    re.IGNORECASE,
)

_SIMPLE_Q_RE = re.compile(
    r"^\s*(what\s+is|when\s+is|where\s+is|who\s+is)\b", re.IGNORECASE
)
_NUMERIC_CODE_RE = re.compile(r"\b\d{3,5}\b")


# ---------- cost table ----------

_DEPTH_COSTS: dict[Depth, tuple[int, float]] = {
    "quick": (45, 0.04),
    "standard": (150, 0.12),
    "deep": (300, 0.35),
}


def _detect_time_sensitive(topic: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for pat, label in _TIME_PATTERNS:
        m = pat.search(topic)
        if m:
            reasons.append(f"matched {m.group(0)!r} -> time_sensitive ({label})")
            return True, reasons
    return False, reasons


def _detect_domain(
    topic: str, profile: DomainProfile | None
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if profile is None:
        return False, reasons
    for label, match in (
        ("entity", profile.match_entity(topic)),
        ("competitor", profile.match_competitor(topic)),
        ("keyword", profile.match_keyword(topic)),
    ):
        if match:
            reasons.append(f"profile {label} {match!r} -> domain_related")
            return True, reasons
    return False, reasons


def _is_simple_definitional(topic: str) -> bool:
    """'What is X?' / 'When is Y?' style — no numeric code, no handle, no comparison."""
    if not _SIMPLE_Q_RE.match(topic):
        return False
    if _NUMERIC_CODE_RE.search(topic):
        return False
    if _HANDLE_RE.search(topic):
        return False
    if _COMPARISON_RE.search(topic) or _OR_COMPARISON_RE.search(topic):
        return False
    return True


def _detect_query_type(
    topic: str, profile: DomainProfile | None
) -> tuple[QueryType, list[str]]:
    reasons: list[str] = []

    # Short-circuit: definitional "what is X?" questions -> FACTUAL
    if _is_simple_definitional(topic):
        reasons.append("definitional 'what is X' question -> FACTUAL")
        return "FACTUAL", reasons

    # 1. COMPARISON
    m = _COMPARISON_RE.search(topic)
    if m:
        reasons.append(f"matched {m.group(0)!r} -> COMPARISON")
        return "COMPARISON", reasons
    m = _OR_COMPARISON_RE.search(topic)
    if m:
        reasons.append(f"'{m.group(1)} or {m.group(2)}' -> COMPARISON")
        return "COMPARISON", reasons

    # 2. PERSON via @handle
    if _HANDLE_RE.search(topic):
        reasons.append("contains @handle -> PERSON")
        return "PERSON", reasons

    # 3. Profile entity/competitor -> PRODUCT (preempts PERSON false positives
    # like "AcmeCo" which reads as a Title-Case name but is a known vendor).
    entity_match = profile.match_entity(topic) if profile else None
    competitor_match = profile.match_competitor(topic) if profile else None
    if entity_match or competitor_match:
        hit = entity_match or competitor_match
        reasons.append(f"profile entity {hit!r} -> PRODUCT")
        return "PRODUCT", reasons

    # 4. PERSON: Title-Case two/three-word name, no product term
    name_match = _TITLE_NAME_RE.search(topic)
    if name_match and _PRODUCT_TERMS_RE.search(topic) is None:
        tokens = name_match.group(1).split()
        if len(tokens) >= 2:
            reasons.append(
                f"'{name_match.group(1)}' looks like a person name -> PERSON"
            )
            return "PERSON", reasons

    # 5. PRODUCT: generic product vocabulary
    m = _PRODUCT_TERMS_RE.search(topic)
    if m:
        reasons.append(f"product term {m.group(0)!r} -> PRODUCT")
        return "PRODUCT", reasons

    # 6. Fallback: any broad topic -> THEMATIC
    reasons.append("no specific entity trigger -> THEMATIC")
    return "THEMATIC", reasons


def _suggest_depth(
    topic: str,
    query_type: QueryType,
    time_sensitive: bool,
    domain_related: bool,
) -> tuple[Depth, list[str]]:
    reasons: list[str] = []
    word_count = len(topic.split())

    # "deep" wins over everything
    m = _DEEP_RE.search(topic)
    if m:
        reasons.append(f"matched {m.group(0)!r} -> deep")
        return "deep", reasons
    if word_count > 12:
        reasons.append(f"word_count={word_count} > 12 -> deep")
        return "deep", reasons

    # trivial definitional questions need only a light lookup
    if query_type == "FACTUAL" and not time_sensitive:
        reasons.append("definitional question, no strong triggers -> quick")
        return "quick", reasons

    # time_sensitive forces at least standard
    if time_sensitive:
        reasons.append("time_sensitive -> standard")
        return "standard", reasons

    # "quick" if short and thematic and not domain-related
    if word_count <= 4 and query_type == "THEMATIC" and not domain_related:
        reasons.append(
            f"short ({word_count} words), THEMATIC, not domain -> quick"
        )
        return "quick", reasons

    reasons.append("default -> standard")
    return "standard", reasons


def _extract_references(
    topic: str,
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Pull an @handle and any github user/repo out of the topic as typed data."""
    reasons: list[str] = []
    x_handle: str | None = None
    github_user: str | None = None
    github_repo: str | None = None

    handle_match = _HANDLE_RE.search(topic)
    if handle_match:
        x_handle = handle_match.group(1)
        reasons.append(f"@handle detected -> x_handle={x_handle}")

    gh_match = _GITHUB_RE.search(topic)
    if gh_match:
        github_user = gh_match.group(1)
        reasons.append(f"github reference -> github_user={github_user}")
        if gh_match.group(2):
            github_repo = gh_match.group(2)
            reasons.append(f"github repo -> github_repo={github_repo}")

    return x_handle, github_user, github_repo, reasons


def classify(topic: str, profile: DomainProfile | None = None) -> Classification:
    """Classify a research topic string. <10ms, no network.

    ``profile`` supplies domain vocabulary. When omitted, a profile is loaded
    from the ``POLYSEARCH_DOMAIN_PROFILE`` env var if set; otherwise the
    classifier stays fully generic.
    """
    topic = topic.strip()
    if profile is None:
        profile = DomainProfile.from_env()

    reasons: list[str] = []

    time_sensitive, ts_reasons = _detect_time_sensitive(topic)
    reasons.extend(ts_reasons)

    domain_related, dm_reasons = _detect_domain(topic, profile)
    reasons.extend(dm_reasons)

    query_type, qt_reasons = _detect_query_type(topic, profile)
    reasons.extend(qt_reasons)

    depth, d_reasons = _suggest_depth(
        topic, query_type, time_sensitive, domain_related
    )
    reasons.extend(d_reasons)

    competitive_mode = False
    if profile is not None:
        competitor_hit = profile.match_competitor(topic)
        if competitor_hit or (domain_related and query_type == "COMPARISON"):
            competitive_mode = True
            reasons.append(
                "profile competitor or domain comparison -> competitive_mode"
            )

    # Only the depth flag maps to a real public CLI option; everything else the
    # classifier learned is exposed as typed fields, not invented flag strings.
    flags = [f"--depth={depth}"]

    x_handle, github_user, github_repo, ref_reasons = _extract_references(topic)
    reasons.extend(ref_reasons)

    runtime, cost = _DEPTH_COSTS[depth]

    return Classification(
        topic=topic,
        query_type=query_type,
        time_sensitive=time_sensitive,
        domain_related=domain_related,
        competitive_mode=competitive_mode,
        suggested_depth=depth,
        flags=flags,
        x_handle=x_handle,
        github_user=github_user,
        github_repo=github_repo,
        cost_estimate=CostEstimate(runtime_sec=runtime, cost_usd=cost),
        reasons=reasons,
    )


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(
            "usage: python3 -m polysearch.classifier 'topic string'",
            file=sys.stderr,
        )
        return 2
    topic = " ".join(sys.argv[1:])
    result = classify(topic)
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
