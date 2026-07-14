"""Canonical banned-token list for the repo leakage gate.

This is the ONE module allowed to hold the raw banned tokens as literals. The
leakage gate (``tests/unit/test_no_internal_leakage.py``) and the two narrower
guard tests (``test_authority``, ``test_report``) all import ``BANNED_TOKENS``
from here, so the tokens live in exactly one place and the gate excludes exactly
one file from its own scan. Not a test module (no ``test_`` prefix), so pytest
does not collect it.

The tokens are the names, products, and vocabulary of the private project this
package was extracted from. None of them belong anywhere in the public repo.
Matching rules (see the gate) are case-insensitive and token-aware: short or
common tokens match on a word boundary so ordinary words never false-positive,
and ``derm`` matches as a prefix so it also catches ``dermatology`` and
``dermatologist``.
"""

from __future__ import annotations

# Lowercase. Order is irrelevant.
BANNED_TOKENS: tuple[str, ...] = (
    "michaellock",
    "clarityrcm",
    "clarity",
    "ashwin",
    "tami",
    "lindsay",
    "danielle",
    "jessica",
    "ellie",
    "modmed",
    "ezderm",
    "experity",
    "personal_context",
    "qdrant",
    "personal corpus",
    "fathom",
    "dermatology",
    "derm",
    "rcm",
    "cpt",
    "icd-10",
    "icd10",
)

# Tokens matched as a prefix (``\bderm`` catches derm, dermatology,
# dermatologist, dermal) rather than as a whole word. Everything else matches on
# both-side word boundaries.
PREFIX_TOKENS: frozenset[str] = frozenset({"derm"})
