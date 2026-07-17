"""Leakage gate — the repo tree must carry zero private-project strings or secrets.

This test is CI-enforced forever. It runs two scans over every tracked file in
the repo (the ``git ls-files`` set, which already excludes ``.git``, build
output, virtualenvs, and anything git-ignored):

1. **Banned internal strings.** The names, products, and vocabulary of the
   private project this package was extracted from, defined once in
   ``tests/_leakage_terms.py``.
2. **Secret patterns.** API-key shapes (``sk-``, ``fc-``, ``pplx-``, ``pat-``,
   GitHub tokens) and hardcoded ``api_key = "..."`` literals.

Both must return zero hits.

Matching rule (documented so it stays intentional):

- Banned tokens match case-insensitively. Most match on both-side word
  boundaries, so an ordinary word never false-positives on a short token buried
  inside it. The medical-vertical token matches as a prefix (word boundary then
  the stem plus trailing letters) so it also catches its longer clinical forms,
  while a boundary before the stem keeps it from firing inside unrelated words.
- Secret patterns require a realistic key length (16+ contiguous alphanumerics
  after the prefix), so short test placeholders do not match while a real leaked
  key would.

The one file allowed to hold the raw banned tokens is ``tests/_leakage_terms.py``
(it defines them), so it is excluded from the banned-string scan only. Every
file, including that one, is scanned for secrets. This module keeps no banned
token as a literal, so it needs no exclusion from its own scan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests._leakage_terms import BANNED_TOKENS, PREFIX_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Relative POSIX paths excluded from the banned-string scan (not the secret scan).
_BANNED_SCAN_EXCLUSIONS = frozenset({"tests/_leakage_terms.py"})


def _tracked_files() -> list[Path]:
    """Every file tracked by git, as absolute paths. Falls back to a filtered
    os.walk if git is unavailable (keeps the gate working outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
        rels = [p for p in out.decode("utf-8", "replace").split("\0") if p]
        return [REPO_ROOT / p for p in rels]
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - fallback
        skip_dirs = {
            ".git", "build", "dist", ".venv", "venv", "__pycache__",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", ".superpowers", "reports",
        }
        found: list[Path] = []
        import os

        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
            for name in files:
                found.append(Path(root) / name)
        return found


def _banned_matchers() -> list[tuple[str, re.Pattern[str]]]:
    matchers: list[tuple[str, re.Pattern[str]]] = []
    for tok in BANNED_TOKENS:
        if tok in PREFIX_TOKENS:
            pattern = r"\b" + re.escape(tok) + r"[a-z]*"
        else:
            pattern = r"\b" + re.escape(tok) + r"\b"
        matchers.append((tok, re.compile(pattern, re.IGNORECASE)))
    return matchers


# Secret shapes. Each requires enough contiguous key material that short test
# placeholders (sk-test-not-real, fc-test-key) cannot match.
_SECRET_MATCHERS: list[tuple[str, re.Pattern[str]]] = [
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("firecrawl-key", re.compile(r"fc-[A-Za-z0-9]{16,}")),
    ("perplexity-key", re.compile(r"pplx-[A-Za-z0-9]{16,}")),
    ("scrapecreators-pat", re.compile(r"pat-[A-Za-z0-9]{16,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("github-fine-grained-pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    (
        # A quoted value assigned to api_key that is long enough to be a real key
        # (20+ chars of key material). Short placeholders like "sk-test-not-real"
        # (16 chars) stay under the floor and do not match.
        "hardcoded-api-key-literal",
        re.compile(
            r"""api_key\s*=\s*["'][A-Za-z0-9][A-Za-z0-9_-]{19,}["']""", re.IGNORECASE
        ),
    ),
]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:  # pragma: no cover
        return ""


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _scan(text: str, matchers: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    """Return the labels of every matcher that hits ``text``."""
    return [label for label, rx in matchers if rx.search(text)]


def test_tracked_file_list_is_populated() -> None:
    """Guard against a broken file walk making the scans vacuously pass."""
    files = _tracked_files()
    assert len(files) > 20, f"expected the tracked-file scan to see the repo, got {len(files)}"


def test_no_banned_internal_strings() -> None:
    matchers = _banned_matchers()
    hits: list[str] = []
    for path in _tracked_files():
        rel = _rel(path)
        if rel in _BANNED_SCAN_EXCLUSIONS:
            continue
        for label in _scan(_read(path), matchers):
            hits.append(f"{rel}: banned token '{label}'")
    assert not hits, "internal strings leaked into the public repo:\n" + "\n".join(sorted(hits))


def test_no_secret_patterns() -> None:
    hits: list[str] = []
    for path in _tracked_files():
        for label in _scan(_read(path), _SECRET_MATCHERS):
            hits.append(f"{_rel(path)}: secret pattern '{label}'")
    assert not hits, "secret-shaped strings found in the repo:\n" + "\n".join(sorted(hits))


def test_scanners_actually_detect_hits() -> None:
    """Positive control: the matchers must fire on planted samples. The samples
    are assembled at runtime so this file carries no literal token or key."""
    banned_sample = "contact " + "michael" + "lock about the " + "der" + "m" + "atology data"
    assert _scan(banned_sample, _banned_matchers()), "banned-token matcher failed to fire"

    secret_sample = 'client = X(api_key="' + "sk-" + "A" * 24 + '")'
    labels = _scan(secret_sample, _SECRET_MATCHERS)
    assert "openai-key" in labels and "hardcoded-api-key-literal" in labels, labels
