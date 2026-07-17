"""Claim extraction from a synthesized report body.

Sentence, quote, and number extraction that feeds the citation verifier. Given
the synthesis markdown and the URLs cited in the run, split the text into
sentences and emit a :class:`Claim` for every sentence carrying a verifiable
figure or direct quote. Sentences that are pure synthesis (no quote, no number)
are unverifiable and dropped, so the verifier is never asked to check an
assertion it cannot ground.

The number-noise filter is the load-bearing piece: enumerated-list markers and
version / section numbers ("1.1", "2.3") must not be mistaken for factual
figures, or they flood the verifier with unmatchable numbers.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from polysearch.output.schema import Claim

__all__ = ["extract_claims"]

_QUOTE_RE = re.compile(r'[“”]([^“”\n]{8,240})[“”]|"([^"\n]{8,240})"')
_NUMBER_RE = re.compile(
    r"""(?x)
    (?<![a-zA-Z])
    (
        \$?\d+(?:,\d{3})*(?:\.\d+)?
        \s*(?:%|percent|pts|[KMB]|million|billion|thousand|bps|basis\s+points)?
    )
    (?![a-zA-Z])
    """
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Sentences shorter than this are filler (headers, list stubs) — skip them.
_MIN_SENTENCE_LEN = 30

# ── Per-claim source localization ────────────────────────────────────────────
# Content-word tokens (>=3 chars) used to score how well a source snippet relates
# to a claim sentence. A tiny stopword list keeps generic connective words from
# inflating overlap; the goal is topical, not linguistic, matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """the and for that with from this these those over past year years into than
    was were has have had are its their they them then such about which while when
    also been being more most some many much other another across around between
    per via not but out only just very both each any all one two per""".split()
)
# A claim is attributed to a source when their content-word overlap (as a fraction
# of the claim's tokens) clears this bar, or when the source snippet literally
# contains one of the claim's figures. Tuned so a clearly on-topic snippet passes
# and an off-topic one does not; below it, the claim falls back to corpus-wide.
_MIN_OVERLAP = 0.30


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    }


def _number_core(number: str) -> str:
    """The bare numeric core of a figure ('$3.63%' -> '3.63'), for substring
    presence checks against a snippet."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", number)
    return m.group(0).replace(",", "") if m else ""


def _localize_source_urls(
    sentence: str,
    numbers: list[str],
    sources: Sequence[tuple[str, str]],
    fallback: list[str],
) -> list[str]:
    """Attribute a claim to the sources whose snippet plausibly supports it.

    Scores each ``(url, snippet)`` by content-word overlap with the claim
    sentence and by literal presence of the claim's figures; returns the URLs
    that clear the bar (order-preserving, deduped). Falls back to ``fallback``
    (the corpus-wide URL list) when nothing relates, so a claim is never left
    without a source to verify against.
    """
    claim_tokens = _content_tokens(sentence)
    number_cores = [c for c in (_number_core(n) for n in numbers) if c]
    related: list[str] = []
    seen: set[str] = set()
    for url, snippet in sources:
        if not url or url in seen:
            continue
        snip = (snippet or "").strip()
        if not snip:
            continue
        snip_low = snip.lower()
        num_hit = any(core in snip_low for core in number_cores)
        if claim_tokens:
            overlap = len(claim_tokens & _content_tokens(snip)) / len(claim_tokens)
        else:
            overlap = 0.0
        if num_hit or overlap >= _MIN_OVERLAP:
            related.append(url)
            seen.add(url)
    return related or fallback


def _claim_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _extract_quotes(text: str, limit: int = 3) -> list[str]:
    """Pull short direct quotes from the text for verification."""
    seen: set[str] = set()
    quotes: list[str] = []
    for match in _QUOTE_RE.finditer(text):
        q = (match.group(1) or match.group(2) or "").strip()
        if 8 <= len(q) <= 240 and q.lower() not in seen:
            quotes.append(q)
            seen.add(q.lower())
        if len(quotes) >= limit:
            break
    return quotes


def _extract_numbers(text: str, limit: int = 5) -> list[str]:
    """Pull numeric assertions (percentages, dollar amounts, benchmarks) for verification.

    Skips noise that isn't a verifiable factual figure: bare integers <= 4 digits
    and small unitless numbers like list indices, section numbers, or version
    markers ("1.1", "2.3"). A unitless number is kept only when it is "large"
    (comma-grouped or >= 5 digits), which tends to mark a real count; anything
    carrying a %, $, or magnitude unit is always kept. Without this, enumerated
    lists in the synthesis ("1.1 ... 1.2 ...") flood the verifier with
    unmatchable numbers and force a mismatch on nearly every claim.
    """
    seen: set[str] = set()
    numbers: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        num = match.group(1).strip()
        has_unit = bool(
            re.search(
                r"(?i)[%$]|percent|pts|bps|\b(?:[KMB]|million|billion|thousand|basis\s+points)\b",
                num,
            )
        )
        if not has_unit:
            digits = re.sub(r"\D", "", num)
            if "," not in num and len(digits) <= 4:
                continue  # bare integer/decimal noise (indices, versions, years)
        if num in seen:
            continue
        numbers.append(num)
        seen.add(num)
        if len(numbers) >= limit:
            break
    return numbers


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def extract_claims(
    text: str,
    source_urls: list[str],
    *,
    sources: Sequence[tuple[str, str]] | None = None,
) -> list[Claim]:
    """Extract verifiable Claims from ``text``, attributing each to its sources.

    Every sentence long enough to carry substance and containing a direct quote
    or a numeric assertion becomes one Claim. Attribution:

    - Without ``sources``, the run's whole ``source_urls`` list is attached to
      every claim (legacy behavior) — the verifier checks each claim against the
      full corpus.
    - With ``sources`` (``(url, snippet)`` pairs), each claim is attributed only
      to the sources whose snippet plausibly relates to that sentence (content
      overlap or a literal figure match), falling back to the corpus-wide
      ``source_urls`` when nothing relates. This tightens verification pairing so
      a claim is scored against the pages that could actually support it, not
      every page in the run.

    Returns an empty list when the text is empty or no source URLs are available.
    """
    urls = [u for u in source_urls if u]
    if not text or not urls:
        return []

    localizable = [(u, s) for (u, s) in (sources or []) if u]

    claims: list[Claim] = []
    for sentence in _split_sentences(text):
        if len(sentence) < _MIN_SENTENCE_LEN:
            continue
        quotes = _extract_quotes(sentence)
        numbers = _extract_numbers(sentence)
        # Only verify sentences with a specific quote or number; pure synthesis
        # is unverifiable.
        if not quotes and not numbers:
            continue
        claim_urls = (
            _localize_source_urls(sentence, numbers, localizable, urls)
            if localizable
            else urls
        )
        claims.append(
            Claim(
                claim_id=_claim_id(sentence),
                text=sentence,
                numbers=numbers,
                quotes=quotes,
                source_urls=claim_urls,
            )
        )
    return claims
