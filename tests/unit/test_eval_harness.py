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

from evals import judge, metrics, report_adapter, run_evals


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
def test_verification_rate_is_claim_level() -> None:
    # Gated metric is claim-level claims_supported/claims_total (2/2), NOT the
    # pair-level 4/5 that poisoned round 1.
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.verification_rate == pytest.approx(2 / 2)
    assert m.claims_supported == 2
    assert m.claims_total == 2


def test_citation_pair_rate_is_secondary() -> None:
    m = metrics.compute_metrics(_fixture_report(), _task())
    assert m.citation_pair_rate == pytest.approx(4 / 5)
    assert m.total_citations == 5


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

    def fake_target(task: dict, out_dir: Path) -> tuple[dict, str, Path]:
        if task["id"] == "bad":
            raise RuntimeError("simulated pipeline crash")
        return _fixture_report(), _fixture_report()["synthesis_md"], out_dir / "r.md"

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

    def fake_target(task: dict, out_dir: Path) -> tuple[dict, str, Path]:
        return _fixture_report(), _fixture_report()["synthesis_md"], out_dir / "r.md"

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


# =========================================================================== #
# Round-1 harness fixes (F1–F5)
# =========================================================================== #

def _internal_report() -> dict:
    """A report in the INTERNAL pipeline shape: different top-level field names
    than the public schema (total_cost_usd, duration_sec, web_items,
    refinement_traces, synthesis object). verification is identical to public."""
    return {
        "topic": "example generic topic",
        "depth": "deep",
        "total_cost_usd": 3.33,
        "duration_sec": 1049.2,
        "web_items": [
            {"url": "https://a.gov/1", "title": "A", "tier": "HIGH"},
            {"url": "https://b.com/2", "title": "B", "tier": "MEDIUM"},
            {"url": "https://c.io/3", "title": "C", "tier": "UNKNOWN"},
            {"url": "https://d.dead/4", "title": "D", "tier": "LOW"},
        ],
        "kb_hits": [{"title": "kb", "tier": "SME"}],  # a second tiered top-level list
        "refinement_traces": [
            {"iteration": 1, "followup_queries": ["q1", "q2"], "stopped_reason": None},
            {"iteration": 2, "followup_queries": ["q3"], "stopped_reason": None},
            {"iteration": 3, "followup_queries": [], "stopped_reason": "no_new_queries"},
        ],
        "synthesis": {
            "executive_summary": "The federal funds rate is near 4.25 percent.",
            "key_findings": ["Inflation cooled to about 3 percent."],
            "quality_notes": "Sources are mostly primary.",
        },
        "verification": {
            "total_citations": 200,
            "verified_ok": 25,
            "claims_total": 16,
            "claims_supported": 16,
            "results": [{"claim_id": "c1", "url": "https://d.dead/4", "status": "URL_DEAD"}],
            "total_cost_usd": 0.5,
            "total_duration_ms": 5000,
        },
    }


# ---- F1: internal-report adapter ------------------------------------------ #
def test_shape_detection() -> None:
    assert report_adapter.detect_shape(_fixture_report()) == "public"
    assert report_adapter.detect_shape(_internal_report()) == "internal"
    assert report_adapter.detect_shape({"topic": "x"}) == "unknown"


def test_internal_report_cost_and_duration_extract() -> None:
    task = {"id": "i", "depth": "deep", "expects_refinement": True, "key_facts": ["x"]}
    m = metrics.compute_metrics(_internal_report(), task)
    assert m.cost_usd == pytest.approx(3.33)  # from total_cost_usd, NOT a silent 0
    assert m.duration_sec == pytest.approx(1049.2)  # from duration_sec
    assert m.report_shape == "internal"


def test_internal_report_tier_mix_extract() -> None:
    task = {"id": "i", "depth": "deep", "expects_refinement": True, "key_facts": ["x"]}
    m = metrics.compute_metrics(_internal_report(), task)
    # web_items (4) + kb_hits (1) = 5 tiered sources; HIGH+MEDIUM = 2.
    assert m.total_sources == 5
    assert m.tier_mix_high_medium == pytest.approx(2 / 5)


def test_internal_report_refinement_extract() -> None:
    task = {"id": "i", "depth": "deep", "expects_refinement": True, "key_facts": ["x"]}
    m = metrics.compute_metrics(_internal_report(), task)
    # Two traces ran follow-up queries; the third (empty) did not.
    assert m.refinement_rounds == 2
    assert m.refinement_ok is True


def test_internal_report_verification_claim_level() -> None:
    task = {"id": "i", "depth": "deep", "expects_refinement": True, "key_facts": ["x"]}
    m = metrics.compute_metrics(_internal_report(), task)
    assert m.verification_rate == pytest.approx(16 / 16)  # claim-level
    assert m.citation_pair_rate == pytest.approx(25 / 200)  # pair-level secondary
    assert m.dead_links == 1


def test_missing_field_yields_none_and_warning_not_zero() -> None:
    """A report missing cost/duration/verification/sources must produce None +
    warnings, never a silent zero that poisons the aggregate."""
    report = {"topic": "x", "depth": "standard", "synthesis_md": "body text here"}
    task = {"id": "x", "depth": "standard", "expects_refinement": False, "key_facts": ["x"]}
    m = metrics.compute_metrics(report, task)
    assert m.cost_usd is None
    assert m.duration_sec is None
    assert m.verification_rate is None
    assert m.tier_mix_high_medium is None
    assert m.refinement_rounds is None
    assert m.warnings  # non-empty
    assert any("cost" in w for w in m.warnings)
    assert any("verification" in w for w in m.warnings)


def test_gate_fails_when_verification_unavailable_everywhere() -> None:
    """A poisoned round where no run has verification must FAIL, not pass by
    having nothing to measure."""
    report = {"topic": "x", "depth": "standard", "synthesis_md": "body"}
    task = {"id": "x", "depth": "standard", "expects_refinement": False, "key_facts": ["x"]}
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
    row = run_evals.RunRow(task_id="x", category="FACTUAL", status="OK", metrics=m, judge=jr)
    passed, failures = run_evals.evaluate_quality_bar([row])
    assert not passed
    assert any("verification" in f.lower() for f in failures)


# ---- F2: coverage extraction against report md ---------------------------- #
def test_coverage_matches_paraphrased_facts_in_md() -> None:
    """A real-ish report md where 2 of 3 facts appear phrased differently → ~0.67.
    The third fact is genuinely absent."""
    md = (
        "# Research: monetary policy\n\n"
        "## Executive Summary\n\n"
        "The Fed has held the federal funds target range at 3.50% to 3.75% since "
        "December.\n\n"
        "## Key Findings\n\n"
        "- Over the past year, rates moved downward as policymakers cut "
        "repeatedly before pausing.\n"
        "- Labor market indicators stayed resilient.\n"
    )
    facts = [
        "federal funds target range set by the Fed",   # present, reordered/filler
        "rates moved downward over the past year",      # present, reordered/filler
        "the current unemployment rate percentage",     # absent
    ]
    cov, covered, total = metrics._key_fact_coverage(facts, md)
    assert total == 3
    assert covered == 2
    assert cov == pytest.approx(2 / 3)


def test_coverage_prefers_collected_md_over_json_synthesis() -> None:
    """Coverage must read the collected md (which internal reports carry) rather
    than a public-only synthesis_md json field."""
    internal = _internal_report()  # has no synthesis_md field
    md = (
        "The federal funds rate is near 4.25 percent. Inflation cooled to about "
        "3 percent. Rates fell over the past year."
    )
    task = {
        "id": "i", "depth": "deep", "expects_refinement": True,
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
    }
    m = metrics.compute_metrics(internal, task, report_md=md)
    assert m.key_fact_coverage == pytest.approx(1.0)


def test_coverage_none_when_no_text() -> None:
    report = {"topic": "x", "depth": "standard"}  # no md, no synthesis
    task = {"id": "x", "depth": "standard", "expects_refinement": False,
            "key_facts": ["alpha", "beta"]}
    m = metrics.compute_metrics(report, task)
    assert m.key_fact_coverage is None
    assert any("coverage" in w.lower() for w in m.warnings)


def test_coverage_splitter_preserves_decimals() -> None:
    # "3.75%" must not be split on its period into "3" / "75%".
    units = metrics._sentence_units("The range is 3.50% to 3.75% today.")
    assert any("3.75%" in u for u in units)


# ---- F3b: coverage-metric vocabulary calibration (task r3b) --------------- #
# Hand-checked defect: task comparison-postgres-mysql-oltp, fact "concurrency
# and MVCC handling differences" against the report's actual MVCC sentence
# scored 45 pre-fix (exact-token rapidfuzz match: "concurrent" != "concurrency",
# "differences" absent, and meta-words like "and"/"handling" dilute a short
# fact). These tests calibrate the fix directly against that hand-checked case
# and the real r2 artifact it came from.
_PG_MYSQL_REPORT = (
    Path(__file__).resolve().parents[2]
    / "evals/results/r2/public/comparison-postgres-mysql-oltp"
    / "2026-07-16-postgresql-versus-mysql-for-oltp-workloads-at-mid.md"
)

_MVCC_SENTENCE = (
    "The material explicitly ties **PostgreSQL** to better handling of "
    "concurrent writes through **MVCC** and says readers do not block "
    "writers, while **MySQL/InnoDB** is described as especially defensible "
    "for update-heavy OLTP and simple read-heavy workloads."
)


def test_morphological_variant_clears_threshold() -> None:
    """'concurrency' (fact) vs 'concurrent' (report text) is the same concept
    in a different part of speech; pre-fix this scored 45 and was marked
    uncovered. Normalization must clear the (lowered, calibrated) threshold."""
    cov, covered, total = metrics._key_fact_coverage(
        ["concurrency and mvcc handling differences"], _MVCC_SENTENCE
    )
    assert covered == 1
    assert cov == pytest.approx(1.0)


def test_unrelated_fact_against_same_text_still_fails() -> None:
    """A fact with no real support in the text must not ride along on the
    lowered threshold — same text as the morphological-variant case above."""
    cov, covered, total = metrics._key_fact_coverage(
        ["replication and high-availability options for each"], _MVCC_SENTENCE
    )
    assert covered == 0
    assert cov == pytest.approx(0.0)


def test_stopword_heavy_fact_does_not_false_positive_on_generic_text() -> None:
    """A short fact made mostly of dropped stopwords ('for', 'each', 'of',
    'the') must not match generic text that happens to reuse those same
    stopwords. Pre-fix (no stopword drop) this scores 48 — a real false
    positive once the threshold moves down to admit morphological variants;
    dropping stopwords from the fact keeps it clear of the gate."""
    fact = "options for each of the two approaches"
    generic_text = (
        "The report covers each region separately, with details for every "
        "department across the organization for the entire fiscal year."
    )
    cov, covered, total = metrics._key_fact_coverage([fact], generic_text)
    assert covered == 0


def test_sliding_window_matches_fact_split_across_sentences() -> None:
    """A fact whose evidence spans two consecutive sentences must be found by
    the 2-3 sentence window, not just the single best sentence in isolation."""
    text = (
        "PostgreSQL uses multi-version concurrency control for its writes. "
        "MySQL's InnoDB engine relies on row-level locking instead."
    )
    single_best = max(
        metrics._key_fact_coverage(["MVCC versus row-level locking approach"], u)[0]
        for u in metrics._sentence_units(text)
    )
    windowed, covered, _ = metrics._key_fact_coverage(
        ["MVCC versus row-level locking approach"], text
    )
    assert covered == 1
    assert windowed >= single_best


def test_real_artifact_postgres_mysql_facts_1_and_4_pass() -> None:
    """Calibration against the actual r2 artifact named in the task brief:
    facts 1 and 4 are genuinely covered by the report (just inflected /
    paraphrased differently than the task's key_facts wording)."""
    text = _PG_MYSQL_REPORT.read_text()
    fact_1_cov, fact_1_covered, _ = metrics._key_fact_coverage(
        ["concurrency and mvcc handling differences"], text
    )
    fact_4_cov, fact_4_covered, _ = metrics._key_fact_coverage(
        ["typical workloads each is better suited to"], text
    )
    assert fact_1_covered == 1
    assert fact_4_covered == 1


def test_source_tier_heading_does_not_false_positive_on_availability_fact() -> None:
    """Regression: the report's own '### High (primary, peer-reviewed, official)
    (22)' bibliography-tier heading shares the word 'high' with a fact about
    'high availability' and scored a false-positive 45.28 (>= the calibrated
    threshold) before markdown headings were excluded from the fact-matching
    pool. Headings are structure, not stated content, and must never carry a
    fact regardless of the tier-name vocabulary they happen to contain."""
    text = _PG_MYSQL_REPORT.read_text()
    fact = (
        "replication and high availability: streaming replication, "
        "group replication, failover"
    )
    cov, covered, _ = metrics._key_fact_coverage([fact], text)
    assert covered == 0


def test_real_artifact_postgres_mysql_genuinely_missing_fact_still_fails() -> None:
    """Fact 2 (replication / HA) is not substantively answered in this
    report's synthesis — it only appears as a follow-up *query* the pipeline
    ran, never as a finding. The fix must not let it ride along on the
    lowered threshold: this is the 'genuinely missing fact stays FAIL' case
    (no raw r1 per-task artifacts survive in this repo to check instead — only
    r1's aggregate scoreboards and diagnosis.md remain, see evals/README.md)."""
    text = _PG_MYSQL_REPORT.read_text()
    cov, covered, _ = metrics._key_fact_coverage(
        ["replication and high-availability options for each"], text
    )
    assert covered == 0


# ---- F4: timeout env override --------------------------------------------- #
def test_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYSEARCH_EVAL_TIMEOUT_SEC", raising=False)
    assert run_evals.timeout_sec() == 2700


def test_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_EVAL_TIMEOUT_SEC", "3600")
    assert run_evals.timeout_sec() == 3600


def test_timeout_env_garbage_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYSEARCH_EVAL_TIMEOUT_SEC", "not-a-number")
    assert run_evals.timeout_sec() == 2700


# ---- F5: artifact persistence --------------------------------------------- #
def test_sweep_persists_artifacts_and_records_path(tmp_path: Path) -> None:
    tasks = [
        {"id": "good", "topic": "example generic topic", "depth": "standard",
         "category": "FACTUAL", "key_facts": ["federal funds rate near 4.25 percent"],
         "expects_refinement": False},
    ]

    def fake_target(task: dict, out_dir: Path) -> tuple[dict, str, Path]:
        # Simulate a target that writes its md+json into out_dir.
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = run_evals.slugify(task["topic"])
        md_path = out_dir / f"2026-07-15-{stem}.md"
        md_path.write_text("# report\n\nbody", encoding="utf-8")
        (out_dir / f"2026-07-15-{stem}.json").write_text("{}", encoding="utf-8")
        return _fixture_report(), "# report\n\nbody", md_path

    rows = run_evals.run_sweep(
        tasks, target="public", out_root=tmp_path, run_target=fake_target, judge_fn=None
    )
    row = rows[0]
    assert row.status == "OK"
    # Report path is recorded relative to the scoreboard dir (out_root/target).
    assert row.report_path is not None
    assert row.report_path.startswith("good/")
    persisted = tmp_path / "public" / row.report_path
    assert persisted.exists()
    # And it appears in the scoreboard json.
    payload = run_evals.build_scoreboard_json(rows, target="public", label="r1")
    assert payload["rows"][0]["report_path"] == row.report_path


def test_collect_report_prefers_slug_match(tmp_path: Path) -> None:
    # Two reports in one dir; collection must return the one matching the topic.
    (tmp_path / "2026-07-15-other-topic.json").write_text('{"topic":"other"}', encoding="utf-8")
    (tmp_path / "2026-07-15-other-topic.md").write_text("other", encoding="utf-8")
    slug = run_evals.slugify("my special topic")
    (tmp_path / f"2026-07-15-{slug}.json").write_text('{"topic":"my special topic"}', encoding="utf-8")
    (tmp_path / f"2026-07-15-{slug}.md").write_text("mine", encoding="utf-8")
    report, md, md_path = run_evals._collect_report(tmp_path, "my special topic")
    assert report["topic"] == "my special topic"
    assert md == "mine"


# =========================================================================== #
# Score-only rescore mode (evals/rescore.py)
# =========================================================================== #
from evals import rescore as rescore_mod  # noqa: E402


def _passing_judge(topic: str, key_facts: list[str], report_md: str) -> judge.JudgeResult:
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


def _write_pair(dir_: Path, topic: str, report: dict, md: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    stem = f"2026-07-15-{run_evals.slugify(topic)}"
    (dir_ / f"{stem}.json").write_text(json.dumps(report), encoding="utf-8")
    (dir_ / f"{stem}.md").write_text(md, encoding="utf-8")


def _rescore_task(topic: str, task_id: str = "t") -> dict:
    return {
        "id": task_id, "topic": topic, "depth": "standard", "category": "FACTUAL",
        "key_facts": ["federal funds rate near 4.25 percent",
                      "inflation cooled to about 3 percent",
                      "rate fell over the past year"],
        "expects_refinement": False,
    }


def test_rescore_flat_pairs(tmp_path: Path) -> None:
    topic = "example generic topic"
    md = _fixture_report()["synthesis_md"]
    _write_pair(tmp_path, topic, _fixture_report(), md)
    rows = rescore_mod.rescore(
        [_rescore_task(topic, "factual-rates")], artifacts_dir=tmp_path, judge_fn=_passing_judge
    )
    assert len(rows) == 1
    r = rows[0]
    assert r.status == "OK"
    assert r.metrics is not None
    assert r.judge is not None and r.judge.passed is True
    assert r.report_path and r.report_path.endswith(".md")


def test_rescore_per_task_subdir(tmp_path: Path) -> None:
    topic = "example generic topic"
    subdir = tmp_path / "factual-rates"
    _write_pair(subdir, topic, _fixture_report(), _fixture_report()["synthesis_md"])
    rows = rescore_mod.rescore(
        [_rescore_task(topic, "factual-rates")], artifacts_dir=tmp_path, judge_fn=_passing_judge
    )
    assert rows[0].status == "OK"
    assert rows[0].metrics is not None


def test_rescore_missing_artifact_is_skipped_not_error(tmp_path: Path) -> None:
    # No artifact written for this topic.
    rows = rescore_mod.rescore(
        [_rescore_task("a topic with no report", "ghost")],
        artifacts_dir=tmp_path,
        judge_fn=_passing_judge,
    )
    assert rows[0].status == "SKIPPED"
    assert rows[0].metrics is None
    assert "no artifact" in (rows[0].error or "")


def test_rescore_no_judge_skips_judging(tmp_path: Path) -> None:
    topic = "example generic topic"
    _write_pair(tmp_path, topic, _fixture_report(), _fixture_report()["synthesis_md"])
    rows = rescore_mod.rescore(
        [_rescore_task(topic)], artifacts_dir=tmp_path, judge_fn=None
    )
    assert rows[0].status == "OK"
    assert rows[0].judge is None
    assert rows[0].metrics is not None


def test_rescore_slug_match_picks_right_report(tmp_path: Path) -> None:
    # Two reports in one flat dir; rescore must pick the one for the task's topic.
    _write_pair(tmp_path, "the first distinct topic",
                {**_fixture_report(), "topic": "the first distinct topic"},
                "first topic body")
    _write_pair(tmp_path, "a second unrelated subject",
                {**_fixture_report(), "topic": "a second unrelated subject"},
                "second topic body")
    rows = rescore_mod.rescore(
        [_rescore_task("a second unrelated subject", "s2")],
        artifacts_dir=tmp_path, judge_fn=None,
    )
    assert rows[0].status == "OK"
    # It matched the right file (its md is what coverage/judge would see).
    report, md, _ = run_evals._load_report(
        rescore_mod._match_report_json(rescore_mod.find_artifact_jsons(tmp_path),
                                       "a second unrelated subject")
    )
    assert report["topic"] == "a second unrelated subject"
    assert md == "second topic body"


def test_find_artifact_jsons_both_layouts(tmp_path: Path) -> None:
    (tmp_path / "2026-07-15-flat.json").write_text("{}", encoding="utf-8")
    sub = tmp_path / "task-a"
    sub.mkdir()
    (sub / "2026-07-15-nested.json").write_text("{}", encoding="utf-8")
    found = {p.name for p in rescore_mod.find_artifact_jsons(tmp_path)}
    assert found == {"2026-07-15-flat.json", "2026-07-15-nested.json"}


def test_rescore_writes_scoreboard_with_skipped(tmp_path: Path) -> None:
    topic = "example generic topic"
    _write_pair(tmp_path, topic, _fixture_report(), _fixture_report()["synthesis_md"])
    tasks = [_rescore_task(topic, "have-it"), _rescore_task("missing topic", "missing")]
    rows = rescore_mod.rescore(tasks, artifacts_dir=tmp_path, judge_fn=_passing_judge)
    # Scoreboard renders without crashing and counts the skip.
    md = run_evals.render_scoreboard_md(rows, target="internal", label="r1-rescored")
    assert "have-it" in md and "missing" in md
    assert "SKIPPED" in md
    payload = run_evals.build_scoreboard_json(rows, target="internal", label="r1-rescored")
    assert payload["summary"]["skipped"] == 1
    assert payload["summary"]["ok"] == 1


def test_rescore_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topic = "example generic topic"
    _write_pair(tmp_path, topic, _fixture_report(), _fixture_report()["synthesis_md"])
    tasks_file = tmp_path / "tasks.yaml"
    tasks_file.write_text(
        "tasks:\n"
        f"  - id: one\n    topic: {topic}\n    depth: standard\n    category: FACTUAL\n"
        "    key_facts: [alpha, beta, gamma]\n    expects_refinement: false\n",
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    monkeypatch.setattr(run_evals, "RESULTS_ROOT", results_root)
    rc = rescore_mod.main(
        [
            "--target", "internal", "--artifacts", str(tmp_path),
            "--label", "rX", "--tasks-file", str(tasks_file), "--no-judge",
        ]
    )
    assert rc in (0, 1)  # exit code reflects the gate, either is a clean run
    board = results_root / "rX" / "internal" / "scoreboard.json"
    assert board.exists()
    data = json.loads(board.read_text())
    assert data["rows"][0]["task_id"] == "one"
    assert data["rows"][0]["status"] == "OK"


# =========================================================================== #
# Judge rate-limit (429) retry/backoff
# =========================================================================== #
class RateLimitError(Exception):
    """Stands in for openai.RateLimitError (matched by class name)."""


def _ok_completion():
    payload = {
        "factual_accuracy": {"score": 0.9, "justification": "x"},
        "citation_accuracy": {"score": 0.9, "justification": "x"},
        "completeness": {"score": 0.9, "justification": "x"},
        "source_quality": {"score": 0.9, "justification": "x"},
        "coherence": {"score": 0.9, "justification": "x"},
        "overall": {"score": 0.9, "pass": True, "justification": "x"},
    }

    class _Msg:
        content = json.dumps(payload)

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 1000
        completion_tokens = 100

    class _Completion:
        choices = [_Choice()]
        usage = _Usage()

    return _Completion()


def _client_raising(seq: list) -> object:
    """A client whose create() pops from ``seq``: an Exception is raised, anything
    else is returned as the completion."""

    class _Completions:
        def create(self, **kwargs):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


class InternalServerError(Exception):
    """Stands in for openai.InternalServerError (matched by class name)."""


def test_judge_retries_on_rate_limit_then_succeeds() -> None:
    slept: list[float] = []
    client = _client_raising([
        RateLimitError("Rate limit reached... try again in 2.5s"),
        _ok_completion(),
    ])
    result = judge.judge_report(
        "t", ["k"], "md", client=client, sleep_fn=slept.append
    )
    assert result.error is None
    assert result.passed is True
    assert len(slept) == 1 and slept[0] >= 4.0  # one exponential backoff wait


def test_judge_retries_on_server_error() -> None:
    slept: list[float] = []
    client = _client_raising([InternalServerError("500 oops"), _ok_completion()])
    result = judge.judge_report("t", ["k"], "md", client=client, sleep_fn=slept.append)
    assert result.error is None
    assert len(slept) == 1


def test_judge_gives_up_after_max_retries() -> None:
    slept: list[float] = []
    client = _client_raising([RateLimitError("429 rate limit")] * 10)
    result = judge.judge_report(
        "t", ["k"], "md", client=client, max_retries=3, sleep_fn=slept.append
    )
    assert result.error is not None
    assert "429" in result.error or "rate limit" in result.error.lower()
    assert len(slept) == 3  # retried exactly max_retries times before giving up


def test_judge_non_rate_limit_error_is_not_retried() -> None:
    slept: list[float] = []
    client = _client_raising([ValueError("bad request")])
    result = judge.judge_report(
        "t", ["k"], "md", client=client, max_retries=3, sleep_fn=slept.append
    )
    assert result.error is not None
    assert slept == []  # never retried a non-rate-limit error


def test_is_rate_limit_and_server_error_detection() -> None:
    assert judge._is_rate_limit(RateLimitError("x")) is True
    assert judge._is_rate_limit(Exception("Error code: 429")) is True
    assert judge._is_rate_limit(Exception("Rate Limit exceeded")) is True
    assert judge._is_rate_limit(ValueError("bad json")) is False
    assert judge._is_server_error(InternalServerError("x")) is True
    assert judge._is_server_error(ValueError("x")) is False


def test_judge_spacing_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYSEARCH_EVAL_JUDGE_RPS", raising=False)
    assert judge.judge_spacing_sec() == pytest.approx(0.5)
    monkeypatch.setenv("POLYSEARCH_EVAL_JUDGE_RPS", "4")
    assert judge.judge_spacing_sec() == pytest.approx(0.25)
    monkeypatch.setenv("POLYSEARCH_EVAL_JUDGE_RPS", "garbage")
    assert judge.judge_spacing_sec() == pytest.approx(0.5)


# ---- Process-section stripping (end-state judging) ------------------------ #
def _report_md_with_audit() -> str:
    return (
        "# Research: monetary policy\n\n"
        "Generated: 2026-07-15 | Depth: deep | Cost: $1.06\n\n"
        "## Executive Summary\n\n"
        "The federal funds rate is near 4.25 percent and fell over the year.\n\n"
        "## Key Findings\n\n"
        "- Inflation cooled to about 3 percent.\n\n"
        "## Pipeline Decisions\n\n"
        "- Topic type: THEMATIC\n\n"
        "## Sources by Quality Tier\n\n"
        "### High (primary) (2)\n"
        "- [BLS](https://bls.gov/a)\n"
        "### Excluded (dead links) (1 URL_DEAD)\n"
        "- https://dead.example/z\n\n"
        "## Citation Integrity\n\n"
        "0/84 fully match. Pervasive NUMBER_MISMATCH detected across citations.\n"
        "### Failed citations (details)\n"
        "- claim c1: NUMBER_MISMATCH\n\n"
        "## Refinement Trace\n\n"
        "Iteration 1 — coverage 0.34\n\n"
        "## Pipeline Stats\n\n"
        "- Total: $1.06 · 7m 49s\n"
    )


def test_strip_process_sections_removes_audit_keeps_content() -> None:
    stripped = judge.strip_process_sections(_report_md_with_audit())
    # Kept: synthesis body + sources-by-tier bucket.
    assert "Executive Summary" in stripped
    assert "federal funds rate is near 4.25 percent" in stripped
    assert "Sources by Quality Tier" in stripped
    assert "bls.gov/a" in stripped
    # Dropped: every process/audit section and the poisoning audit strings.
    assert "Pipeline Decisions" not in stripped
    assert "Pipeline Stats" not in stripped
    assert "Refinement Trace" not in stripped
    assert "Citation Integrity" not in stripped
    assert "NUMBER_MISMATCH" not in stripped
    assert "0/84 fully match" not in stripped
    assert "Failed citations" not in stripped
    # The Excluded subsection under a kept H2 is dropped too.
    assert "Excluded (dead links)" not in stripped
    assert "dead.example" not in stripped


def test_build_judge_prompt_strips_audit_by_default() -> None:
    prompt = judge.build_judge_prompt("t", ["k"], _report_md_with_audit())
    assert "Executive Summary" in prompt
    assert "NUMBER_MISMATCH" not in prompt
    assert "Citation Integrity" not in prompt


def test_build_judge_prompt_full_md_keeps_everything() -> None:
    prompt = judge.build_judge_prompt("t", ["k"], _report_md_with_audit(), full_md=True)
    assert "NUMBER_MISMATCH" in prompt  # escape hatch: nothing stripped
    assert "Pipeline Stats" in prompt


def test_judge_report_full_md_passthrough() -> None:
    seen: dict = {}

    class _Completions:
        def create(self, **kwargs):
            seen["prompt"] = kwargs["messages"][0]["content"]
            return _ok_completion()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    judge.judge_report("t", ["k"], _report_md_with_audit(), client=_Client(), full_md=True)
    assert "NUMBER_MISMATCH" in seen["prompt"]

    judge.judge_report("t", ["k"], _report_md_with_audit(), client=_Client(), full_md=False)
    assert "NUMBER_MISMATCH" not in seen["prompt"]
