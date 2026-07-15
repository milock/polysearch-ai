"""Unit tests for the eval harness (``evals/``).

Fully mocked — no live API calls, no subprocess to a real CLI. The harness is
exercised through its pure seams:

- ``metrics.compute_metrics`` computes programmatic metrics from a fixture
  report json + task definition;
- ``run_evals.render_scoreboard`` renders a scoreboard from run rows;
- ``run_evals.run_sweep`` drives tasks through injected target/judge callables,
  isolating a crashing task as ERROR while the sweep continues;
- ``run_evals`` honours ``--tasks-file`` and ``--tasks`` selection;
- ``judge.build_judge_prompt`` carries the rubric + the task's key_facts;
- ``judge.parse_judge_response`` maps a malformed judge payload to a parse error
  (which the caller records as task ERROR, never a crash);
- ``judge.JUDGE_SCHEMA`` never requests a URL field (models fabricate them).
- ``run_evals.evaluate_quality_bar`` enforces the documented release gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import judge, metrics, run_evals


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _fixture_report() -> dict:
    """A representative pipeline-report json with two supported claims, one dead
    link, a HIGH/MEDIUM/COMMUNITY tier mix, and one refinement round run."""
    return {
        "topic": "example generic topic",
        "depth": "standard",
        "synthesis_md": (
            "## Executive Summary\n\nThe federal funds rate sits near 4.25 percent "
            "and has fallen over the past year.\n\n## Key Findings\n\n"
            "- Inflation cooled to roughly 3 percent [HIGH: bls.gov, 2026].\n"
        ),
        "layers": [
            {
                "layer": "research",
                "results": [
                    {"url": "https://bls.gov/a", "title": "A", "snippet": "", "tier": "HIGH"},
                    {"url": "https://reuters.com/b", "title": "B", "snippet": "", "tier": "MEDIUM"},
                    {"url": "https://news.ycombinator.com/x", "title": "C", "snippet": "", "tier": "COMMUNITY"},
                    {"url": "https://dead.example/z", "title": "Z", "snippet": "", "tier": "LOW"},
                ],
            }
        ],
        "verification": {
            "total_citations": 5,
            "verified_ok": 4,
            "broken": 1,
            "quote_mismatches": 0,
            "number_mismatches": 0,
            "paywalled": 0,
            "undated": 0,
            "skipped_budget": 0,
            "fetch_blocked": 0,
            "blocked_sources": 0,
            "claims_total": 2,
            "claims_supported": 2,
            "credits_exhausted_hit": False,
            "total_cost_usd": 0.03,
            "total_duration_ms": 1200,
            "results": [
                {"claim_id": "c1", "url": "https://bls.gov/a", "status": "OK"},
                {"claim_id": "c1", "url": "https://reuters.com/b", "status": "OK"},
                {"claim_id": "c2", "url": "https://news.ycombinator.com/x", "status": "OK"},
                {"claim_id": "c2", "url": "https://bls.gov/a", "status": "OK"},
                {"claim_id": "c3", "url": "https://dead.example/z", "status": "URL_DEAD"},
            ],
        },
        "refinement_iterations": [
            {
                "iteration": 1,
                "verdict": {"goal_met": False, "coverage_score": 0.5},
                "queries_run": ["a follow-up query"],
                "new_sources": 2,
                "new_claims": 1,
                "cost_usd": 0.01,
                "stopped_reason": "goal_met",
            }
        ],
        "pipeline_errors": [],
        "totals": {"cost_usd": 0.08, "duration_sec": 42.0},
    }


def _task() -> dict:
    return {
        "id": "factual-rates",
        "topic": "example generic topic",
        "depth": "standard",
        "category": "FACTUAL",
        "key_facts": [
            "federal funds rate near 4.25 percent",
            "inflation cooled to about 3 percent",
            "rate fell over the past year",
        ],
        "expects_refinement": True,
    }


# --------------------------------------------------------------------------- #
# metrics.compute_metrics
# --------------------------------------------------------------------------- #
def test_verification_rate_from_fixture() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.verification_rate == pytest.approx(4 / 5)


def test_tier_mix_high_medium_fraction() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    # 2 of 4 unique sources are HIGH/MEDIUM.
    assert m.tier_mix_high_medium == pytest.approx(0.5)


def test_dead_link_count_from_verification() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.dead_links == 1


def test_key_fact_coverage_fuzzy_match() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    # All three key facts appear (fuzzily) in the synthesis text.
    assert m.key_fact_coverage == pytest.approx(1.0)
    assert m.key_facts_total == 3


def test_key_fact_coverage_partial() -> None:
    task = _task()
    task["key_facts"] = task["key_facts"] + ["gold reserves doubled in Antarctica"]
    m = metrics.compute_metrics(_fixture_report(), task)
    assert m.key_facts_total == 4
    assert m.key_fact_coverage == pytest.approx(3 / 4)


def test_refinement_rounds_and_expectation() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.refinement_rounds == 1
    assert m.expects_refinement is True
    assert m.refinement_ok is True


def test_cost_and_duration_from_totals() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.cost_usd == pytest.approx(0.08)
    assert m.duration_sec == pytest.approx(42.0)


def test_placeholder_leak_detected() -> None:
    report = _fixture_report()
    report["synthesis_md"] += "\n{{EXEC_SUMMARY — agent to fill}}\n"
    m = metrics.compute_metrics(report, _task())
    assert m.placeholder_leaks >= 1


def test_no_placeholder_leak_on_clean_report() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.placeholder_leaks == 0


# --------------------------------------------------------------------------- #
# run_evals.render_scoreboard
# --------------------------------------------------------------------------- #
def test_scoreboard_renders_md_and_json() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    row = run_evals.RunRow(
        task_id="factual-rates",
        category="FACTUAL",
        status="OK",
        metrics=m,
        judge=judge.JudgeResult.from_scores(
            {
                "factual_accuracy": {"score": 0.9, "justification": "solid"},
                "citation_accuracy": {"score": 0.85, "justification": "ok"},
                "completeness": {"score": 0.88, "justification": "ok"},
                "source_quality": {"score": 0.8, "justification": "ok"},
                "coherence": {"score": 0.9, "justification": "ok"},
                "overall": {"score": 0.87, "pass": True, "justification": "pass"},
            }
        ),
    )
    md = run_evals.render_scoreboard_md([row], target="public", label="r1")
    assert "factual-rates" in md
    assert "FACTUAL" in md
    payload = run_evals.build_scoreboard_json([row], target="public", label="r1")
    assert payload["label"] == "r1"
    assert payload["target"] == "public"
    assert payload["rows"][0]["task_id"] == "factual-rates"
    # Round-trips as json.
    assert json.loads(json.dumps(payload))["rows"][0]["status"] == "OK"


def test_scoreboard_includes_error_rows() -> None:
    row = run_evals.RunRow(
        task_id="boom", category="TREND", status="ERROR", metrics=None, judge=None,
        error="target crashed",
    )
    md = run_evals.render_scoreboard_md([row], target="public", label="r1")
    assert "boom" in md
    assert "ERROR" in md


# --------------------------------------------------------------------------- #
# run_evals.run_sweep — failure isolation
# --------------------------------------------------------------------------- #
def test_sweep_isolates_crashing_task(tmp_path: Path) -> None:
    tasks = [
        {"id": "good", "topic": "t1", "depth": "quick", "category": "FACTUAL",
         "key_facts": ["federal funds rate near 4.25 percent"], "expects_refinement": False},
        {"id": "bad", "topic": "t2", "depth": "quick", "category": "TREND",
         "key_facts": ["x"], "expects_refinement": False},
    ]

    def fake_target(task: dict, out_dir: Path) -> tuple[dict, str]:
        if task["id"] == "bad":
            raise RuntimeError("simulated pipeline crash")
        return _fixture_report(), _fixture_report()["synthesis_md"]

    def fake_judge(topic: str, key_facts: list[str], report_md: str) -> judge.JudgeResult:
        return judge.JudgeResult.from_scores(
            {
                "factual_accuracy": {"score": 0.9, "justification": "j"},
                "citation_accuracy": {"score": 0.9, "justification": "j"},
                "completeness": {"score": 0.9, "justification": "j"},
                "source_quality": {"score": 0.9, "justification": "j"},
                "coherence": {"score": 0.9, "justification": "j"},
                "overall": {"score": 0.9, "pass": True, "justification": "j"},
            }
        )

    rows = run_evals.run_sweep(
        tasks, target="public", out_root=tmp_path, run_target=fake_target, judge_fn=fake_judge
    )
    by_id = {r.task_id: r for r in rows}
    assert by_id["good"].status == "OK"
    assert by_id["bad"].status == "ERROR"
    assert "simulated pipeline crash" in (by_id["bad"].error or "")


def test_sweep_marks_error_when_judge_returns_parse_error(tmp_path: Path) -> None:
    tasks = [
        {"id": "good", "topic": "t1", "depth": "quick", "category": "FACTUAL",
         "key_facts": ["federal funds rate near 4.25 percent"], "expects_refinement": False},
    ]

    def fake_target(task: dict, out_dir: Path) -> tuple[dict, str]:
        return _fixture_report(), _fixture_report()["synthesis_md"]

    def fake_judge(topic: str, key_facts: list[str], report_md: str) -> judge.JudgeResult:
        return judge.JudgeResult(error="judge parse failure")

    rows = run_evals.run_sweep(
        tasks, target="public", out_root=tmp_path, run_target=fake_target, judge_fn=fake_judge
    )
    assert rows[0].status == "ERROR"
    assert "judge parse failure" in (rows[0].error or "")


# --------------------------------------------------------------------------- #
# Task loading: --tasks-file override + --tasks selection
# --------------------------------------------------------------------------- #
def test_load_tasks_from_default(tmp_path: Path) -> None:
    tasks = run_evals.load_tasks(tasks_file=None)
    assert len(tasks) == 12
    for t in tasks:
        for field in ("id", "topic", "depth", "category", "key_facts", "expects_refinement"):
            assert field in t, f"task {t.get('id')!r} missing {field}"
        assert 3 <= len(t["key_facts"]) <= 5


def test_tasks_file_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "tasks:\n"
        "  - id: only-one\n"
        "    topic: a generic topic\n"
        "    depth: quick\n"
        "    category: FACTUAL\n"
        "    key_facts: [alpha, beta, gamma]\n"
        "    expects_refinement: false\n",
        encoding="utf-8",
    )
    tasks = run_evals.load_tasks(tasks_file=custom)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "only-one"


def test_tasks_id_selection() -> None:
    tasks = run_evals.load_tasks(tasks_file=None)
    picked = run_evals.select_tasks(tasks, ids="factual-fed-rate")
    assert len(picked) == 1
    assert picked[0]["id"] == "factual-fed-rate"


def test_depth_override_applies() -> None:
    tasks = run_evals.load_tasks(tasks_file=None)
    overridden = run_evals.apply_depth_override(tasks, "deep")
    assert all(t["depth"] == "deep" for t in overridden)


# --------------------------------------------------------------------------- #
# judge.build_judge_prompt + schema + parsing
# --------------------------------------------------------------------------- #
def test_judge_prompt_includes_rubric_and_key_facts() -> None:
    prompt = judge.build_judge_prompt(
        topic="a generic topic",
        key_facts=["alpha fact one", "beta fact two"],
        report_md="## Synthesis\n\nsome body text",
    )
    text = prompt if isinstance(prompt, str) else json.dumps(prompt)
    # Rubric dimensions present.
    for dim in ("factual", "citation", "completeness", "source quality", "coherence"):
        assert dim.lower() in text.lower()
    # Key facts carried verbatim.
    assert "alpha fact one" in text
    assert "beta fact two" in text
    # Report md carried.
    assert "some body text" in text


def test_judge_schema_never_requests_urls() -> None:
    schema_blob = json.dumps(judge.JUDGE_SCHEMA).lower()
    assert "url" not in schema_blob
    assert "http" not in schema_blob
    assert "source_urls" not in schema_blob


def test_judge_parse_success() -> None:
    payload = json.dumps(
        {
            "factual_accuracy": {"score": 0.9, "justification": "accurate"},
            "citation_accuracy": {"score": 0.8, "justification": "cited"},
            "completeness": {"score": 0.85, "justification": "covers facts"},
            "source_quality": {"score": 0.75, "justification": "mixed"},
            "coherence": {"score": 0.95, "justification": "clear"},
            "overall": {"score": 0.85, "pass": True, "justification": "good"},
        }
    )
    result = judge.parse_judge_response(payload)
    assert result.error is None
    assert result.overall == pytest.approx(0.85)
    assert result.passed is True
    assert result.scores["factual_accuracy"] == pytest.approx(0.9)


def test_judge_parse_failure_returns_error_not_crash() -> None:
    result = judge.parse_judge_response("this is not json {{{")
    assert result.error is not None
    assert result.overall is None
    assert result.passed is None


def test_judge_parse_failure_on_missing_dimension() -> None:
    payload = json.dumps({"factual_accuracy": {"score": 0.9, "justification": "x"}})
    result = judge.parse_judge_response(payload)
    assert result.error is not None


def test_judge_report_uses_injected_client_and_records_cost() -> None:
    """judge_report drives the model via an injected client, never a live call."""

    class _FakeMessage:
        content = json.dumps(
            {
                "factual_accuracy": {"score": 0.9, "justification": "x"},
                "citation_accuracy": {"score": 0.9, "justification": "x"},
                "completeness": {"score": 0.9, "justification": "x"},
                "source_quality": {"score": 0.9, "justification": "x"},
                "coherence": {"score": 0.9, "justification": "x"},
                "overall": {"score": 0.9, "pass": True, "justification": "x"},
            }
        )

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeUsage:
        prompt_tokens = 1000
        completion_tokens = 200

    class _FakeCompletion:
        choices = [_FakeChoice()]
        usage = _FakeUsage()

    class _FakeCompletions:
        def create(self, **kwargs):
            # The rubric + key facts must reach the model.
            blob = json.dumps(kwargs)
            assert "key_fact_x" in blob or "key_fact_x" in json.dumps(kwargs.get("messages"))
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    result = judge.judge_report(
        topic="t",
        key_facts=["key_fact_x"],
        report_md="## Synthesis\n\nbody",
        client=_FakeClient(),
    )
    assert result.error is None
    assert result.passed is True
    assert result.cost_usd > 0


# --------------------------------------------------------------------------- #
# evaluate_quality_bar
# --------------------------------------------------------------------------- #
def _passing_row(task_id: str, *, expects_refinement: bool = False, rounds: int = 0) -> run_evals.RunRow:
    report = _fixture_report()
    task = {
        "id": task_id, "topic": "t", "depth": "standard", "category": "FACTUAL",
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
        "expects_refinement": expects_refinement,
    }
    if not expects_refinement:
        report["refinement_iterations"] = []
    m = metrics.compute_metrics(report, task)
    jr = judge.JudgeResult.from_scores(
        {
            "factual_accuracy": {"score": 0.9, "justification": "j"},
            "citation_accuracy": {"score": 0.9, "justification": "j"},
            "completeness": {"score": 0.9, "justification": "j"},
            "source_quality": {"score": 0.9, "justification": "j"},
            "coherence": {"score": 0.9, "justification": "j"},
            "overall": {"score": 0.9, "pass": True, "justification": "j"},
        }
    )
    return run_evals.RunRow(
        task_id=task_id, category="FACTUAL", status="OK", metrics=m, judge=jr
    )


def test_quality_bar_passes_on_healthy_sweep() -> None:
    rows = [_passing_row("a"), _passing_row("b", expects_refinement=True)]
    passed, failures = run_evals.evaluate_quality_bar(rows)
    assert passed, failures
    assert failures == []


def test_quality_bar_fails_on_crash() -> None:
    rows = [
        _passing_row("a"),
        run_evals.RunRow(task_id="boom", category="TREND", status="ERROR",
                         metrics=None, judge=None, error="crash"),
    ]
    passed, failures = run_evals.evaluate_quality_bar(rows)
    assert not passed
    assert any("crash" in f.lower() or "error" in f.lower() for f in failures)


def test_quality_bar_fails_on_low_judge_mean() -> None:
    row = _passing_row("a")
    row.judge = judge.JudgeResult.from_scores(
        {
            "factual_accuracy": {"score": 0.5, "justification": "j"},
            "citation_accuracy": {"score": 0.5, "justification": "j"},
            "completeness": {"score": 0.5, "justification": "j"},
            "source_quality": {"score": 0.5, "justification": "j"},
            "coherence": {"score": 0.5, "justification": "j"},
            "overall": {"score": 0.5, "pass": False, "justification": "j"},
        }
    )
    passed, failures = run_evals.evaluate_quality_bar([row])
    assert not passed
    assert any("judge" in f.lower() for f in failures)


def _report_with_n_refinement_rounds(n: int) -> dict:
    report = _fixture_report()
    report["refinement_iterations"] = [
        {
            "iteration": i + 1,
            "verdict": {"goal_met": False, "coverage_score": 0.5},
            "queries_run": [f"follow-up query {i + 1}"],
            "new_sources": 1,
            "new_claims": 1,
            "cost_usd": 0.01,
            "stopped_reason": "cost_ceiling",
        }
        for i in range(n)
    ]
    return report


def test_refinement_ceiling_is_per_depth() -> None:
    # quick 0 / standard 2 / deep 4, read from polysearch.config.DEPTH_PROFILES.
    assert metrics.refinement_ceiling_for_depth("quick") == 0
    assert metrics.refinement_ceiling_for_depth("standard") == 2
    assert metrics.refinement_ceiling_for_depth("deep") == 4


def test_standard_task_three_iterations_flags_ceiling_violation() -> None:
    report = _report_with_n_refinement_rounds(3)
    task = {
        "id": "std-over", "topic": "t", "depth": "standard", "category": "COMPARISON",
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
        "expects_refinement": True,
    }
    m = metrics.compute_metrics(report, task)
    assert m.refinement_rounds == 3
    assert m.refinement_ceiling == 2
    assert m.refinement_within_ceiling is False

    jr = judge.JudgeResult.from_scores(
        {
            "factual_accuracy": {"score": 0.9, "justification": "j"},
            "citation_accuracy": {"score": 0.9, "justification": "j"},
            "completeness": {"score": 0.9, "justification": "j"},
            "source_quality": {"score": 0.9, "justification": "j"},
            "coherence": {"score": 0.9, "justification": "j"},
            "overall": {"score": 0.9, "pass": True, "justification": "j"},
        }
    )
    row = run_evals.RunRow(task_id="std-over", category="COMPARISON", status="OK", metrics=m, judge=jr)
    passed, failures = run_evals.evaluate_quality_bar([row])
    assert not passed
    assert any("ceiling" in f.lower() for f in failures)


def test_deep_task_four_iterations_is_within_ceiling() -> None:
    report = _report_with_n_refinement_rounds(4)
    task = {
        "id": "deep-ok", "topic": "t", "depth": "deep", "category": "TECHNICAL",
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
        "expects_refinement": True,
    }
    m = metrics.compute_metrics(report, task)
    assert m.refinement_rounds == 4
    assert m.refinement_ceiling == 4
    assert m.refinement_within_ceiling is True

    jr = judge.JudgeResult.from_scores(
        {
            "factual_accuracy": {"score": 0.9, "justification": "j"},
            "citation_accuracy": {"score": 0.9, "justification": "j"},
            "completeness": {"score": 0.9, "justification": "j"},
            "source_quality": {"score": 0.9, "justification": "j"},
            "coherence": {"score": 0.9, "justification": "j"},
            "overall": {"score": 0.9, "pass": True, "justification": "j"},
        }
    )
    row = run_evals.RunRow(task_id="deep-ok", category="TECHNICAL", status="OK", metrics=m, judge=jr)
    passed, failures = run_evals.evaluate_quality_bar([row])
    assert passed, failures


def test_depth_override_changes_ceiling() -> None:
    # A standard task overridden to deep gets the deep cap of 4.
    tasks = run_evals.load_tasks(tasks_file=None)
    standard = next(t for t in tasks if t["depth"] == "standard")
    overridden = run_evals.apply_depth_override([standard], "deep")[0]
    report = _report_with_n_refinement_rounds(3)
    m_std = metrics.compute_metrics(report, standard)
    m_deep = metrics.compute_metrics(report, overridden)
    assert m_std.refinement_within_ceiling is False  # 3 > standard cap 2
    assert m_deep.refinement_within_ceiling is True   # 3 <= deep cap 4


def test_quality_bar_fails_when_refinement_never_triggers() -> None:
    report = _fixture_report()
    report["refinement_iterations"] = []  # expected but none ran
    task = {
        "id": "r", "topic": "t", "depth": "standard", "category": "TREND",
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
        "expects_refinement": True,
    }
    m = metrics.compute_metrics(report, task)
    jr = judge.JudgeResult.from_scores(
        {
            "factual_accuracy": {"score": 0.9, "justification": "j"},
            "citation_accuracy": {"score": 0.9, "justification": "j"},
            "completeness": {"score": 0.9, "justification": "j"},
            "source_quality": {"score": 0.9, "justification": "j"},
            "coherence": {"score": 0.9, "justification": "j"},
            "overall": {"score": 0.9, "pass": True, "justification": "j"},
        }
    )
    row = run_evals.RunRow(task_id="r", category="TREND", status="OK", metrics=m, judge=jr)
    passed, failures = run_evals.evaluate_quality_bar([row])
    assert not passed
    assert any("refinement" in f.lower() for f in failures)
