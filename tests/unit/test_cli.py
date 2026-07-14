"""Unit tests for the polysearch CLI surface.

All network-touching modes are patched out; only the argument routing, utility
modes (--classify, --diagnose), and exit-code contract are exercised here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polysearch import cli
from polysearch.output.schema import PipelineReport


def test_parser_help_has_no_bare_percent() -> None:
    # argparse treats a bare '%' in help as a format token and crashes at render.
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "%" not in help_text


def test_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_no_topic_returns_usage_code() -> None:
    assert cli.main([]) == 2


def test_classify_prints_json_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    rc = cli.main(["--classify", "--topic", "Postgres vs MySQL"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_type"] == "COMPARISON"


def test_classify_requires_topic() -> None:
    assert cli.main(["--classify"]) == 2


def test_diagnose_no_network(tmp_path, capsys: pytest.CaptureFixture) -> None:
    rc = cli.main(["--diagnose", "--output-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "install tier" in out
    assert "layers:" in out


def test_successful_run_returns_zero(monkeypatch, tmp_path) -> None:
    fake = AsyncMock(return_value=_stub_report())
    monkeypatch.setattr(cli, "run_research", fake)
    rc = cli.main(["--topic", "distributed consensus", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert fake.await_count == 1


def test_pipeline_error_returns_one(monkeypatch, tmp_path) -> None:
    async def _boom(*args, **kwargs):
        raise RuntimeError("layer meltdown")

    monkeypatch.setattr(cli, "run_research", _boom)
    rc = cli.main(["--topic", "distributed consensus", "--output-dir", str(tmp_path)])
    assert rc == 1


def test_parse_layers_allowlist() -> None:
    assert cli._parse_layers(None) is None
    assert cli._parse_layers("research, grounding") == {"research", "grounding"}


def _stub_report() -> PipelineReport:
    return PipelineReport(topic="t", depth="quick")
