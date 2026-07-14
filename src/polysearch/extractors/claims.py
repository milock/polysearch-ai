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


def extract_claims(text: str, source_urls: list[str]) -> list[Claim]:
    """Extract verifiable Claims from ``text``, attributing each to ``source_urls``.

    Every sentence long enough to carry substance and containing a direct quote
    or a numeric assertion becomes one Claim; the run's ``source_urls`` are
    attached so the verifier knows which cited pages to check. Returns an empty
    list when the text is empty or no source URLs are available.
    """
    urls = [u for u in source_urls if u]
    if not text or not urls:
        return []

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
        claims.append(
            Claim(
                claim_id=_claim_id(sentence),
                text=sentence,
                numbers=numbers,
                quotes=quotes,
                source_urls=urls,
            )
        )
    return claims
