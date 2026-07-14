"""Core-subject extraction: trim a verbose, keyword-stuffed search query down
to its core noun phrase.

Adapted from the last30days research skill's ``lib/query.py`` (MIT). See
ATTRIBUTION.md.
"""

from __future__ import annotations

import re

__all__ = ["extract_core_subject"]

# Multi-word question/meta prefixes stripped from the head of a query (longest
# first so "what are the best" wins over "what are").
_PREFIXES = (
    "what are the best",
    "what is the best",
    "what are the latest",
    "what are people saying about",
    "what do people think about",
    "how do i use",
    "how to use",
    "how to",
    "what are",
    "what is",
    "tips for",
    "best practices for",
)

# Single-token filler: articles, question words, and research/meta descriptors
# that carry no subject signal.
_NOISE_WORDS = frozenset({
    # Articles / prepositions / conjunctions
    "a", "an", "the", "is", "are", "was", "were", "and", "or",
    "of", "in", "on", "for", "with", "about", "to",
    # Question words
    "how", "what", "which", "who", "why", "when", "where",
    "does", "should", "could", "would",
    # Research / meta descriptors
    "best", "top", "good", "great", "awesome", "latest", "new",
    "news", "update", "updates", "trending", "hottest", "hot",
    "popular", "viral", "practices", "guide", "tutorial",
    "recommendations", "advice", "review", "reviews", "comparison",
    "versus", "vs", "forum",
    # Action / filler
    "using", "uses", "use", "people", "saying", "think", "lately",
})

_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_URL = re.compile(r"^(?:https?://|[\w-]+\.(?:com|org|net|io|gov|edu|co))", re.I)


def _is_operator(token: str) -> bool:
    """True for search operators (site:, filetype:), bare URLs, and domains."""
    return ":" in token or bool(_URL.match(token))


def extract_core_subject(query: str, max_tokens: int = 12) -> str:
    """Trim a verbose query to its core subject.

    Queries at or under ``max_tokens`` words are returned verbatim — the caller
    only reaches for this on longer, keyword-stuffed queries. Longer queries get
    their question prefix, search operators, stray years, and filler stripped,
    then the remainder is capped at ``max_tokens`` words.

    Args:
        query: Raw search query.
        max_tokens: Word ceiling below which the query is left untouched and to
            which a trimmed query is capped.

    Returns:
        The original query (if short) or its trimmed core subject.
    """
    tokens = query.split()
    if len(tokens) <= max_tokens:
        return query

    # Strip a leading multi-word prefix (compare on a lowercased copy so casing
    # in the kept tokens is preserved).
    lowered = query.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix + " "):
            tokens = tokens[len(prefix.split()):]
            break

    kept = [
        t for t in tokens
        if t.lower() not in _NOISE_WORDS
        and not _YEAR.match(t)
        and not _is_operator(t)
    ]

    # If filtering removed everything, fall back to the pre-filter tokens so we
    # never hand back an empty subject.
    chosen = (kept or tokens)[:max_tokens]
    return " ".join(chosen).rstrip("?!.")
