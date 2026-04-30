"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.live (real API calls).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live tests skipped (use --run-live to enable)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def project_root() -> Path:
    """Path to the polysearch project root (the parent of `tests/`)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to `tests/fixtures/` for recorded API responses."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """A temporary directory for pipeline output (`./reports/` analog)."""
    out = tmp_path / "reports"
    out.mkdir()
    return out
