"""Unit tests for the eval harness's time-budget pass-through (task r3c).

``run_evals.py`` exports ``POLYSEARCH_TIME_BUDGET_S`` to the target subprocess,
kept safely under the harness's own kill timeout, so a run that would otherwise
blow the eval ceiling self-limits and reports partial results instead of being
killed with nothing collected. All subprocess calls are mocked — no network, no
real pipeline invocation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from evals import run_evals


def test_time_budget_sec_defaults_under_the_default_timeout(monkeypatch) -> None:
    monkeypatch.delenv("POLYSEARCH_EVAL_TIMEOUT_SEC", raising=False)
    assert run_evals.timeout_sec() == run_evals.DEFAULT_TIMEOUT_SEC
    assert run_evals.time_budget_sec() == run_evals.DEFAULT_TIMEOUT_SEC - 120


def test_time_budget_sec_honors_the_ceiling_override(monkeypatch) -> None:
    monkeypatch.setenv("POLYSEARCH_EVAL_TIMEOUT_SEC", "600")
    assert run_evals.time_budget_sec() == 480


def test_time_budget_sec_never_drops_below_the_floor(monkeypatch) -> None:
    # A tiny configured ceiling must not produce a budget of zero or negative.
    monkeypatch.setenv("POLYSEARCH_EVAL_TIMEOUT_SEC", "90")
    assert run_evals.time_budget_sec() == run_evals._MIN_TIME_BUDGET_SEC


def _fake_completed_process(returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = ""
    return proc


def test_run_public_target_exports_time_budget_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("POLYSEARCH_EVAL_TIMEOUT_SEC", raising=False)
    captured = {}

    def _fake_run(cmd, *, capture_output, text, timeout, env):
        captured["env"] = env
        return _fake_completed_process()

    with patch.object(run_evals.subprocess, "run", side_effect=_fake_run):
        with patch.object(
            run_evals, "_collect_report", return_value=({}, "", tmp_path / "r.md")
        ):
            run_evals.run_public_target({"topic": "t", "depth": "quick"}, tmp_path)

    assert captured["env"]["POLYSEARCH_TIME_BUDGET_S"] == str(
        run_evals.DEFAULT_TIMEOUT_SEC - 120
    )


def test_run_internal_target_exports_time_budget_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("POLYSEARCH_EVAL_INTERNAL_CMD", "echo hi")
    monkeypatch.delenv("POLYSEARCH_EVAL_TIMEOUT_SEC", raising=False)
    captured = {}

    def _fake_run(cmd, *, capture_output, text, timeout, env):
        captured["env"] = env
        return _fake_completed_process()

    with patch.object(run_evals.subprocess, "run", side_effect=_fake_run):
        with patch.object(
            run_evals, "_collect_report", return_value=({}, "", tmp_path / "r.md")
        ):
            run_evals.run_internal_target({"topic": "t", "depth": "quick"}, tmp_path)

    assert captured["env"]["POLYSEARCH_TIME_BUDGET_S"] == str(
        run_evals.DEFAULT_TIMEOUT_SEC - 120
    )
    assert captured["env"]["POLYSEARCH_OUTPUT_DIR"] == str(tmp_path)
