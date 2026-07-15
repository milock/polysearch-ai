"""The unit suite must be hermetic against a developer's ``.env`` file.

``polysearch.config.Settings.from_env`` calls ``load_dotenv()``. Without the
autouse ``_neutralize_dotenv`` fixture (in ``tests/conftest.py``), a real ``.env``
on disk — a documented, encouraged developer setup — would be loaded mid-test and
repopulate keys that a test cleared with ``monkeypatch.delenv``, making the suite
pass or fail depending on the machine. These tests lock the fixture in place.
"""

from __future__ import annotations

from pathlib import Path

import polysearch.config as config
from polysearch.config import Settings


def test_load_dotenv_is_neutralized_even_with_env_file_present(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A ``.env`` sitting in the working directory must not leak into the process
    environment when the code calls ``load_dotenv`` — the fixture no-ops it."""
    env_file = tmp_path / ".env"
    env_file.write_text("POLYSEARCH_HERMETIC_SENTINEL=leaked\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    # The call site the code uses. Under the fixture this is a no-op.
    config.load_dotenv()

    import os

    assert os.environ.get("POLYSEARCH_HERMETIC_SENTINEL") is None


def test_from_env_does_not_repopulate_a_cleared_key(monkeypatch: object) -> None:
    """Clearing a key then building Settings.from_env must keep it cleared — the
    .env reload inside from_env is neutralized, so the key stays absent."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "temp")  # type: ignore[attr-defined]
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)  # type: ignore[attr-defined]

    settings = Settings.from_env()

    assert not settings.firecrawl_api_key
