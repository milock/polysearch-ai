"""Programmatic metrics computed from a pipeline report + its task.

Objective, model-free measurements that sit beside the LLM judge. Every metric
is derived through :mod:`evals.report_adapter`, so the public and internal report
shapes are measured identically.

Two rules learned the hard way:

- **No silent zeros.** A field that cannot be found in either report shape yields
  ``None`` and a per-run warning, not ``0.0``. A metric of ``0`` must mean the
  pipeline genuinely produced zero, never that the harness looked in the wrong
  place.
- **Coverage matches the report markdown.** Key-fact coverage always fuzzy-matches
  the collected report ``.md`` text (falling back to the synthesis body only if no
  md was collected), windowed over 1-3 consecutive sentences so paraphrase (even
  split across two sentences) is caught but words merely scattered across a long
  document are not. Facts are matched after markdown-noise stripping, suffix
  normalization, and fact-side stopword dropping (see ``KEY_FACT_MATCH_THRESHOLD``).
  Trailing pipeline scaffolding (Sources-by-Tier, Refinement Trace, Pipeline
  Decisions/Errors/Stats, Citation Integrity) is excluded first — coverage
  measures the synthesis/findings body, not an appendix or a raw search-query
  echo. A fact carrying a number additionally requires its covering window to
  state that number; matching vocabulary alone is not sufficient.

The gated citation metric is **claim-level** (``claims_supported/claims_total``);
the raw pair-level rate is kept as a secondary, ungated column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from polysearch.config import DEPTH_PROFILES

from evals.report_adapter import NormalizedReport, normalize_report

# A key fact counts as covered when its best token_set_ratio over any 1-3
# sentence window clears this threshold (0-100). Calibrated down from the
# original 70 (task r3b): hand-checked morphological-variant and paraphrased
# facts in real reports land in the 50-75 range once tokens are normalized,
# while genuinely-absent facts land in the low 20s-40s (see the calibration
# tests in tests/unit/test_eval_harness.py and evals/README.md). 45 sits with
# clear margin on both sides of that split.
KEY_FACT_MATCH_THRESHOLD = 45.0

# Small suffix set stripped from every token before matching, so a fact and a
# report sentence that use different inflections of the same word ("replicas"
# vs "replica", "cutting" vs "cut") still line up. Deliberately crude (no
# nltk/spacy) — order matters: longest/most specific suffix first.
def _strip_word_suffix(word: str) -> str:
    if word.endswith("tions") and len(word) > 6:
        return word[:-5] + "t"
    if word.endswith("tion") and len(word) > 5:
        return word[:-4] + "t"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        # Only strip both letters when the singular needs the sibilant vowel
        # back ("boxes" -> "box", "watches" -> "watch"); otherwise the "e"
        # belongs to the base word and only the "-s" is the plural marker
        # ("codes" -> "code", "types" -> "type", not "cod"/"typ").
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
        return word[:-1]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


# Dropped from the FACT side only (never the report text) before matching, so
# short facts that are mostly connective tissue ("options for each", "typical
# ... suited to") aren't dominated by words carrying no content of their own.
_FACT_STOPWORDS = frozenset({"and", "the", "of", "for", "each", "to"})

# Markdown emphasis/code markers are formatting noise, not content — reports
# routinely bold the key nouns a fact needs to match, and the literal "**"
# characters otherwise count against the fuzzy ratio.
_MARKDOWN_NOISE_RE = re.compile(r"\*\*|__|`")

# How many consecutive sentences a fact is allowed to match across (best
# window wins) — a fact's evidence sometimes spans a claim sentence plus its
# immediate follow-on, not just one sentence in isolation.
_MAX_FACT_WINDOW = 3


def _normalize_for_match(text: str, *, drop_stopwords: bool) -> str:
    """Lowercase, strip markdown noise, tokenize, and suffix-normalize every
    word. ``drop_stopwords`` is only ever True for the fact side of a match."""
    words = re.findall(r"[a-z0-9]+", _MARKDOWN_NOISE_RE.sub("", text.lower()))
    stemmed = [_strip_word_suffix(w) for w in words]
    if drop_stopwords:
        stemmed = [w for w in stemmed if w not in _FACT_STOPWORDS]
    return " ".join(stemmed)


def _fact_windows(units: list[str]) -> list[tuple[str, str]]:
    """Every run of 1 to ``_MAX_FACT_WINDOW`` consecutive sentence units, as
    ``(raw, normalized)`` pairs computed once so a coverage check over many
    facts doesn't re-tokenize the same windows repeatedly. ``raw`` (just
    joined, not stemmed/stopword-processed) is what the numeric guard checks —
    stemming tokenizes on ``[a-z0-9]+`` and would split "18.8%" into "18"/"8"."""
    n = len(units)
    windows = []
    for size in range(1, _MAX_FACT_WINDOW + 1):
        for i in range(n - size + 1):
            raw = " ".join(units[i : i + size])
            windows.append((raw, _normalize_for_match(raw, drop_stopwords=False)))
    return windows


# Matches a number in either the fact or the report text: optional leading
# "$", digits with optional comma grouping and decimal point, optional
# trailing "%". Units are deliberately not required to co-occur with the
# digits — the guard only needs the numeric value itself.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _number_match_pattern(number_text: str) -> re.Pattern[str] | None:
    """Build a regex matching ``number_text``'s numeric value with flexible
    formatting (trailing zeros, comma grouping) in report text. Adapted from
    the citation-number-matching approach in
    ``polysearch.verification.verifier._normalize_number_to_regex`` — kept
    self-contained here rather than imported, since that module pulls in the
    verifier's network/scrape stack (httpx/Playwright/Firecrawl), which the
    eval harness has no other reason to depend on."""
    m = _NUMBER_RE.search(number_text)
    if not m:
        return None
    raw_num = m.group(0).replace(",", "")
    variants = [raw_num]
    if "." in raw_num:
        int_part, frac_part = raw_num.split(".", 1)
        frac_stripped = frac_part.rstrip("0")
        if frac_stripped != frac_part:
            variants.append(f"{int_part}.{frac_stripped}" if frac_stripped else int_part)
    else:
        variants.extend([f"{raw_num}.0", f"{raw_num}.00"])
        if len(raw_num) >= 4:
            rev = raw_num[::-1]
            with_commas = ",".join(rev[i : i + 3] for i in range(0, len(rev), 3))[::-1]
            variants.append(with_commas)
    seen: set[str] = set()
    unique = [v for v in variants if not (v in seen or seen.add(v))]
    return re.compile("(?:" + "|".join(re.escape(v) for v in unique) + ")")


def _fact_required_numbers(fact: str) -> list[re.Pattern[str]]:
    """Every number in ``fact``, each as a flexible-format regex a covering
    window's raw text must contain at least one of. Deliberately unit-
    agnostic (18.8 (fact) matches "18.8%", "18.8 percent", or bare "18.8" in
    the text) — the guard exists only to stop a textually-similar-but-
    numberless window from riding on the fact's non-numeric vocabulary, not to
    enforce that a % or $ sign specifically co-occurs."""
    return [
        p
        for p in (_number_match_pattern(n) for n in _NUMBER_RE.findall(fact))
        if p is not None
    ]


# Same unresolved-template shape the report writer guards against.
_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_]+\s*[—-].*?\}\}")

# Sentence/line splitter that does NOT break decimals ("3.75%" stays whole): a
# terminator must be followed by whitespace, or it is a newline run.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Tiers that count as trustworthy for the HIGH+MEDIUM mix metric.
_TRUSTED_TIERS = frozenset({"HIGH", "MEDIUM"})

_FALLBACK_CEILING = max(p.max_refinement_iterations for p in DEPTH_PROFILES.values())


def refinement_ceiling_for_depth(depth: str | None) -> int:
    """The pipeline's ``max_refinement_iterations`` cap for ``depth`` (quick 0 /
    standard 2 / deep 4), read from ``polysearch.config.DEPTH_PROFILES``.
    ``--depth-override`` is honored because it mutates the task depth before the
    sweep. Unknown depth falls back to the most permissive cap."""
    profile = DEPTH_PROFILES.get(depth or "")
    return profile.max_refinement_iterations if profile else _FALLBACK_CEILING


@dataclass
class RunMetrics:
    """Objective metrics for a single task run. ``None`` marks a metric the report
    did not carry (never a silent zero)."""

    verification_rate: float | None  # GATED: claim-level claims_supported/claims_total
    citation_pair_rate: float | None  # secondary, ungated: verified_ok/total_citations
    tier_mix_high_medium: float | None
    dead_links: int | None
    key_fact_coverage: float | None
    key_facts_total: int
    key_facts_covered: int
    refinement_rounds: int | None
    expects_refinement: bool
    refinement_ok: bool
    refinement_ceiling: int
    refinement_within_ceiling: bool
    placeholder_leaks: int
    cost_usd: float | None
    duration_sec: float | None
    total_citations: int | None
    total_sources: int
    claims_total: int | None
    claims_supported: int | None
    report_shape: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_rate": _r(self.verification_rate, 4),
            "citation_pair_rate": _r(self.citation_pair_rate, 4),
            "tier_mix_high_medium": _r(self.tier_mix_high_medium, 4),
            "dead_links": self.dead_links,
            "key_fact_coverage": _r(self.key_fact_coverage, 4),
            "key_facts_total": self.key_facts_total,
            "key_facts_covered": self.key_facts_covered,
            "refinement_rounds": self.refinement_rounds,
            "expects_refinement": self.expects_refinement,
            "refinement_ok": self.refinement_ok,
            "refinement_ceiling": self.refinement_ceiling,
            "refinement_within_ceiling": self.refinement_within_ceiling,
            "placeholder_leaks": self.placeholder_leaks,
            "cost_usd": _r(self.cost_usd, 6),
            "duration_sec": _r(self.duration_sec, 3),
            "total_citations": self.total_citations,
            "total_sources": self.total_sources,
            "claims_total": self.claims_total,
            "claims_supported": self.claims_supported,
            "report_shape": self.report_shape,
            "warnings": self.warnings,
        }


def _r(value: float | None, ndigits: int) -> float | None:
    return round(value, ndigits) if value is not None else None


def _claim_level_rate(verification: dict | None) -> tuple[float | None, int | None, int | None]:
    """(gated claim-level rate, claims_supported, claims_total). ``None`` when no
    verification block. A block with zero claims scores 1.0 (nothing to support)."""
    if not verification:
        return None, None, None
    total = int(verification.get("claims_total", 0) or 0)
    supported = int(verification.get("claims_supported", 0) or 0)
    if total == 0:
        return 1.0, supported, 0
    return supported / total, supported, total


def _pair_level_rate(verification: dict | None) -> tuple[float | None, int | None]:
    """(pair-level rate, total_citations). ``None`` when no verification block."""
    if not verification:
        return None, None
    total = int(verification.get("total_citations", 0) or 0)
    ok = int(verification.get("verified_ok", 0) or 0)
    if total == 0:
        return None, 0
    return ok / total, total


def _dead_links(verification: dict | None) -> int | None:
    if not verification:
        return None
    return sum(
        1 for r in verification.get("results", []) or [] if r.get("status") == "URL_DEAD"
    )


def _tier_mix(norm: NormalizedReport) -> tuple[float | None, int]:
    """(fraction HIGH+MEDIUM, unique source count). ``None`` when no source
    container was found or the container held zero sources — a fraction of nothing
    is not zero, it is unknown."""
    if not norm.sources_found:
        return None, 0
    seen: dict[str, str] = {}
    for s in norm.sources:
        key = s.get("url") or f"{s.get('tier')}:{s.get('title')}"
        if key not in seen:
            seen[key] = s.get("tier", "UNKNOWN")
    total = len(seen)
    if total == 0:
        return None, 0
    trusted = sum(1 for t in seen.values() if t in _TRUSTED_TIERS)
    return trusted / total, total


# A markdown heading ("### High (primary, peer-reviewed, official) (22)") is
# document structure, not stated content — a source-tier bucket label like
# "High" or "Low" can share vocabulary with an unrelated fact ("high
# availability") purely by coincidence. Excluded from the fact-matching pool.
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s")

# H2 sections that are the pipeline's own scaffolding, never a stated finding:
# a Refinement-Trace follow-up query is maximally keyword-dense by
# construction (it exists to search well, not to report a fact), and a
# Sources-by-Tier bibliography entry can share vocabulary with a fact purely
# through an article title. Broader than evals/judge.py's
# ``strip_process_sections`` — the judge still needs the source list for
# citation-accuracy scoring; coverage has no use for it. Verified against
# every real report artifact on hand: this exact heading text (case-
# insensitive substring) appears consistently across both the public and
# internal report shapes.
_NON_SYNTHESIS_SECTION_KEYWORDS: tuple[str, ...] = (
    "pipeline decisions",
    "pipeline stats",
    "pipeline errors",
    "refinement trace",
    "citation integrity",
    "sources by",
)

_H2_RE = re.compile(r"^## +(.*)$")


def _strip_non_synthesis_sections(md: str) -> str:
    """Remove every H2 section (and everything under it, including its own H3
    subsections) whose heading names pipeline scaffolding rather than report
    content, so key-fact matching only sees the synthesis/findings body."""
    out: list[str] = []
    skip = False
    for line in md.splitlines():
        h2 = _H2_RE.match(line)
        if h2:
            skip = any(k in h2.group(1).lower() for k in _NON_SYNTHESIS_SECTION_KEYWORDS)
            if skip:
                continue
        if skip:
            continue
        out.append(line)
    return "\n".join(out)


def _sentence_units(text: str) -> list[str]:
    return [
        u.strip().lower()
        for u in _SENTENCE_SPLIT_RE.split(text)
        if u.strip() and not _MARKDOWN_HEADING_RE.match(u.strip())
    ]


def _key_fact_coverage(
    key_facts: list[str], text: str
) -> tuple[float | None, int, int]:
    """(coverage fraction, covered, total). Each fact is matched against every
    1-3 sentence window of the synthesis/findings body (trailing pipeline
    scaffolding — Sources-by-Tier, Refinement Trace, Pipeline Decisions/Errors/
    Stats, Citation Integrity — excluded first) after markdown-noise stripping,
    suffix normalization, and (fact-side only) stopword dropping. A fact that
    contains a number additionally requires its covering window to contain
    that number (format-flexible); text similarity alone is not enough.
    ``None`` when there is no text to match against."""
    facts = [f for f in (key_facts or []) if f and f.strip()]
    total = len(facts)
    if not facts:
        return 1.0, 0, 0
    units = _sentence_units(_strip_non_synthesis_sections(text))
    if not units:
        return None, 0, total
    windows = _fact_windows(units)
    covered = 0
    for fact in facts:
        fact_norm = _normalize_for_match(fact, drop_stopwords=True)
        required_numbers = _fact_required_numbers(fact)
        for raw_window, norm_window in windows:
            if fuzz.token_set_ratio(fact_norm, norm_window) < KEY_FACT_MATCH_THRESHOLD:
                continue
            if required_numbers and not any(p.search(raw_window) for p in required_numbers):
                continue
            covered += 1
            break
    return covered / total, covered, total


def compute_metrics(
    report: dict, task: dict, *, report_md: str | None = None
) -> RunMetrics:
    """Compute every programmatic metric for one run, across both report shapes.

    ``report_md`` (the collected report markdown) is the preferred text for
    key-fact coverage and placeholder detection; the synthesis body is a fallback
    only. Missing fields become ``None`` with a warning, never a silent zero.
    """
    norm = normalize_report(report)
    warnings = list(norm.warnings)

    # Text for coverage + placeholder scan: prefer the collected md.
    text = report_md if (report_md and report_md.strip()) else norm.synthesis_text
    if not (text and text.strip()):
        warnings.append("no report text (md or synthesis) available for coverage")
        text = ""

    verification = norm.verification
    claim_rate, claims_supported, claims_total = _claim_level_rate(verification)
    pair_rate, total_citations = _pair_level_rate(verification)
    dead = _dead_links(verification)

    tier_frac, total_sources = _tier_mix(norm)

    coverage, covered, facts_total = _key_fact_coverage(task.get("key_facts", []), text)
    if coverage is None:
        warnings.append("key-fact coverage could not be computed (no text)")

    rounds = norm.refinement_rounds
    expects = bool(task.get("expects_refinement", False))
    # Refinement-OK: ran at least one round if expected. Unknown rounds (None)
    # cannot confirm the expectation, so it is not OK when refinement was expected.
    if expects:
        refinement_ok = rounds is not None and rounds >= 1
    else:
        refinement_ok = True

    ceiling = refinement_ceiling_for_depth(task.get("depth"))
    within_ceiling = True if rounds is None else rounds <= ceiling

    leaks = len(_PLACEHOLDER_RE.findall(text))

    return RunMetrics(
        verification_rate=claim_rate,
        citation_pair_rate=pair_rate,
        tier_mix_high_medium=tier_frac,
        dead_links=dead,
        key_fact_coverage=coverage,
        key_facts_total=facts_total,
        key_facts_covered=covered,
        refinement_rounds=rounds,
        expects_refinement=expects,
        refinement_ok=refinement_ok,
        refinement_ceiling=ceiling,
        refinement_within_ceiling=within_ceiling,
        placeholder_leaks=leaks,
        cost_usd=norm.cost_usd,
        duration_sec=norm.duration_sec,
        total_citations=total_citations,
        total_sources=total_sources,
        claims_total=claims_total,
        claims_supported=claims_supported,
        report_shape=norm.shape,
        warnings=warnings,
    )
