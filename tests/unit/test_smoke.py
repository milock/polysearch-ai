"""Smoke tests that verify the package imports and basic CLI surface."""

from __future__ import annotations

import polysearch
from polysearch import cli


def test_package_version_matches_pyproject() -> None:
    assert polysearch.__version__ == "0.1.0"


def test_cli_parser_builds() -> None:
    parser = cli.build_parser()
    # Help should not raise; --version should be registered.
    assert parser.prog == "polysearch"


def test_cli_no_topic_prints_help_and_returns_zero(capsys: object) -> None:
    exit_code = cli.main([])
    assert exit_code == 0


def test_cli_with_topic_returns_not_implemented_exit_code() -> None:
    exit_code = cli.main(["--topic", "anything", "--depth", "quick"])
    # Phase 0 placeholder uses exit code 2 to signal "not implemented yet"
    assert exit_code == 2


def test_cli_version_flag_exits_zero() -> None:
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
