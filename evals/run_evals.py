"""Eval sweep runner.

Drives the 12 simulated deep-research tasks through a target CLI, computes
programmatic metrics (``evals.metrics``), scores each report with the LLM judge
(``evals.judge``), and writes a scoreboard per round to
``evals/results/<label>/<target>/scoreboard.{md,json}``.

Targets:

- ``public`` invokes the installed ``polysearch`` CLI.
- ``internal`` invokes the command in ``POLYSEARCH_EVAL_INTERNAL_CMD`` (the topic
  is appended as the final argument). Nothing about the internal target is
  hardcoded here — no path, no module name — so the public repo stays clean.
- ``both`` runs each in turn.

Failure isolation: a task whose target crashes, returns non-zero, produces no
report, or trips the judge parser is recorded as ``ERROR`` in the scoreboard and
the sweep continues.

Usage::

    python -m evals.run_evals --target public --rounds-label r1
    python -m evals.run_evals --target both --rounds-label r2 --tasks factual-fed-rate,trend-agentic-coding
    python -m evals.run_evals --target internal --rounds-label r3 --tasks-file my_private_tasks.yaml
    python -m evals.run_evals --target public --rounds-label r4 --depth-override deep
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

import yaml

from evals import judge as judge_mod
from evals import metrics as metrics_mod
from evals.judge import JudgeResult
from evals.metrics import RunMetrics

# --------------------------------------------------------------------------- #
# Release gate constants (documented in evals/README.md)
# --------------------------------------------------------------------------- #
MIN_JUDGE_OVERALL = 0.80
MIN_VERIFICATION_RATE = 0.70
MIN_KEY_FACT_COVERAGE = 0.85
MAX_PLACEHOLDER_LEAKS = 0
MAX_CRASHES = 0
MIN_REFINEMENT_TRIGGER_RATE = 0.80
# The refinement ceiling is not a flat constant — each run is checked against its
# own depth's max_refinement_iterations cap (quick 0 / standard 2 / deep 4), read
# from polysearch.config.DEPTH_PROFILES by evals.metrics. A run exceeding its cap
# signals a broken stop condition.

DEFAULT_TASKS_FILE = Path(__file__).resolve().parent / "tasks.yaml"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Per-task subprocess timeout (seconds). A deep run can be slow; well past that
# is a hang, so we cut it and mark the task ERROR.
TARGET_TIMEOUT_SEC = 1800


# --------------------------------------------------------------------------- #
# Row model
# --------------------------------------------------------------------------- #
@dataclass
class RunRow:
    """One task's result in a sweep."""

    task_id: str
    category: str
    status: str  # "OK" | "ERROR"
    metrics: Optional[RunMetrics] = None
    judge: Optional[JudgeResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "status": self.status,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "judge": self.judge.to_dict() if self.judge else None,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Task loading + selection
# --------------------------------------------------------------------------- #
def load_tasks(*, tasks_file: Path | str | None) -> list[dict]:
    """Load tasks from ``tasks_file`` (or the bundled ``tasks.yaml``).

    An external ``--tasks-file`` overrides the repo's tasks entirely — this is
    how a private task suite runs through the same harness without ever living
    in the public repo.
    """
    path = Path(tasks_file) if tasks_file else DEFAULT_TASKS_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def select_tasks(tasks: list[dict], *, ids: str | None) -> list[dict]:
    """Filter to a comma-separated ``ids`` allowlist, preserving file order."""
    if not ids:
        return list(tasks)
    wanted = {tok.strip() for tok in ids.split(",") if tok.strip()}
    picked = [t for t in tasks if t.get("id") in wanted]
    missing = wanted - {t.get("id") for t in picked}
    if missing:
        raise ValueError(f"unknown task id(s): {', '.join(sorted(missing))}")
    return picked


def apply_depth_override(tasks: list[dict], depth: str | None) -> list[dict]:
    """Return copies of ``tasks`` with ``depth`` forced, if given."""
    if not depth:
        return tasks
    return [{**t, "depth": depth} for t in tasks]


# --------------------------------------------------------------------------- #
# Target runners
# --------------------------------------------------------------------------- #
def _collect_report(out_dir: Path) -> tuple[dict, str]:
    """Load the newest ``*.json`` report in ``out_dir`` and its sibling ``.md``."""
    jsons = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise RuntimeError(f"target produced no report json in {out_dir}")
    json_path = jsons[-1]
    report = json.loads(json_path.read_text(encoding="utf-8"))
    md_path = json_path.with_suffix(".md")
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else report.get(
        "synthesis_md", ""
    )
    return report, md


def run_public_target(task: dict, out_dir: Path) -> tuple[dict, str]:
    """Run the installed ``polysearch`` CLI for one task into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "polysearch",
        "--topic",
        task["topic"],
        "--depth",
        task.get("depth", "standard"),
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=TARGET_TIMEOUT_SEC
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"polysearch exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return _collect_report(out_dir)


def run_internal_target(task: dict, out_dir: Path) -> tuple[dict, str]:
    """Run the ``POLYSEARCH_EVAL_INTERNAL_CMD`` command for one task.

    The base command is read from the environment and shell-split; the topic is
    appended as the final positional argument. ``POLYSEARCH_OUTPUT_DIR`` is
    exported so a cooperating internal pipeline writes into ``out_dir``; the
    report is then collected the same way as for the public target.
    """
    base = os.environ.get("POLYSEARCH_EVAL_INTERNAL_CMD")
    if not base:
        raise RuntimeError(
            "internal target requires POLYSEARCH_EVAL_INTERNAL_CMD (base command; "
            "the topic is appended as the final argument)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = shlex.split(base) + [task["topic"]]
    env = {**os.environ, "POLYSEARCH_OUTPUT_DIR": str(out_dir)}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=TARGET_TIMEOUT_SEC, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"internal target exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return _collect_report(out_dir)


def _default_judge(topic: str, key_facts: list[str], report_md: str) -> JudgeResult:
    return judge_mod.judge_report(topic, key_facts, report_md)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    tasks: list[dict],
    *,
    target: str,
    out_root: Path,
    run_target: Callable[[dict, Path], tuple[dict, str]],
    judge_fn: Optional[Callable[[str, list[str], str], Optional[JudgeResult]]] = None,
) -> list[RunRow]:
    """Run every task, isolating failures as ERROR rows.

    ``run_target(task, out_dir) -> (report_json, report_md)`` produces the report;
    ``judge_fn(topic, key_facts, report_md) -> JudgeResult | None`` scores it
    (``None`` skips judging). A crash in either, a judge parse error, or a metrics
    failure marks the task ERROR and the sweep moves on.
    """
    rows: list[RunRow] = []
    for task in tasks:
        task_id = task.get("id", "<unknown>")
        category = task.get("category", "")
        task_dir = out_root / target / task_id
        try:
            report, report_md = run_target(task, task_dir)
            run_metrics = metrics_mod.compute_metrics(report, task, report_md=report_md)
            judge_result: Optional[JudgeResult] = None
            if judge_fn is not None:
                judge_result = judge_fn(
                    task["topic"], task.get("key_facts", []), report_md
                )
            if judge_result is not None and judge_result.error:
                rows.append(
                    RunRow(
                        task_id=task_id,
                        category=category,
                        status="ERROR",
                        metrics=run_metrics,
                        judge=judge_result,
                        error=judge_result.error,
                    )
                )
                continue
            rows.append(
                RunRow(
                    task_id=task_id,
                    category=category,
                    status="OK",
                    metrics=run_metrics,
                    judge=judge_result,
                )
            )
        except Exception as exc:  # noqa: BLE001 — failure isolation: one task must not sink the sweep
            rows.append(
                RunRow(
                    task_id=task_id,
                    category=category,
                    status="ERROR",
                    error=str(exc),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Quality bar
# --------------------------------------------------------------------------- #
def evaluate_quality_bar(rows: list[RunRow]) -> tuple[bool, list[str]]:
    """Check a sweep against the documented release gate.

    Returns ``(passed, failures)``; ``failures`` is a human-readable list of every
    threshold missed (empty when the gate passes).
    """
    failures: list[str] = []

    errors = [r for r in rows if r.status == "ERROR"]
    if len(errors) > MAX_CRASHES:
        ids = ", ".join(r.task_id for r in errors)
        failures.append(f"{len(errors)} task(s) ended in ERROR (> {MAX_CRASHES}): {ids}")

    ok = [r for r in rows if r.status == "OK" and r.metrics is not None]
    if not ok:
        failures.append("no successful task runs to evaluate")
        return False, failures

    judged = [r for r in ok if r.judge is not None and r.judge.overall is not None]
    if judged:
        judge_mean = mean(r.judge.overall for r in judged)  # type: ignore[misc]
        if judge_mean < MIN_JUDGE_OVERALL:
            failures.append(
                f"mean judge overall {judge_mean:.3f} < {MIN_JUDGE_OVERALL}"
            )

    verif_mean = mean(r.metrics.verification_rate for r in ok)
    if verif_mean < MIN_VERIFICATION_RATE:
        failures.append(
            f"mean verification rate {verif_mean:.3f} < {MIN_VERIFICATION_RATE}"
        )

    coverage_mean = mean(r.metrics.key_fact_coverage for r in ok)
    if coverage_mean < MIN_KEY_FACT_COVERAGE:
        failures.append(
            f"mean key-fact coverage {coverage_mean:.3f} < {MIN_KEY_FACT_COVERAGE}"
        )

    leaks = sum(r.metrics.placeholder_leaks for r in ok)
    if leaks > MAX_PLACEHOLDER_LEAKS:
        failures.append(f"{leaks} placeholder leak(s) (> {MAX_PLACEHOLDER_LEAKS})")

    expecting = [r for r in ok if r.metrics.expects_refinement]
    if expecting:
        triggered = sum(1 for r in expecting if r.metrics.refinement_rounds >= 1)
        rate = triggered / len(expecting)
        if rate < MIN_REFINEMENT_TRIGGER_RATE:
            failures.append(
                f"refinement triggered on {rate:.0%} of expecting tasks "
                f"(< {MIN_REFINEMENT_TRIGGER_RATE:.0%})"
            )

    over_ceiling = [r for r in ok if not r.metrics.refinement_within_ceiling]
    if over_ceiling:
        ids = ", ".join(
            f"{r.task_id} ({r.metrics.refinement_rounds} > {r.metrics.refinement_ceiling})"
            for r in over_ceiling
        )
        failures.append(f"refinement exceeded per-depth ceiling on: {ids}")

    return (not failures), failures


# --------------------------------------------------------------------------- #
# Scoreboard rendering
# --------------------------------------------------------------------------- #
def _summary(rows: list[RunRow]) -> dict[str, Any]:
    ok = [r for r in rows if r.status == "OK" and r.metrics is not None]
    judged = [r for r in ok if r.judge is not None and r.judge.overall is not None]
    passed, failures = evaluate_quality_bar(rows)
    return {
        "tasks": len(rows),
        "ok": len(ok),
        "errors": sum(1 for r in rows if r.status == "ERROR"),
        "mean_judge_overall": round(mean(r.judge.overall for r in judged), 4) if judged else None,  # type: ignore[misc]
        "mean_verification_rate": round(mean(r.metrics.verification_rate for r in ok), 4) if ok else None,
        "mean_key_fact_coverage": round(mean(r.metrics.key_fact_coverage for r in ok), 4) if ok else None,
        "placeholder_leaks": sum(r.metrics.placeholder_leaks for r in ok),
        "total_cost_usd": round(sum(r.metrics.cost_usd for r in ok), 4),
        "gate_passed": passed,
        "gate_failures": failures,
    }


def build_scoreboard_json(rows: list[RunRow], *, target: str, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "target": target,
        "summary": _summary(rows),
        "rows": [r.to_dict() for r in rows],
    }


def render_scoreboard_md(rows: list[RunRow], *, target: str, label: str) -> str:
    summary = _summary(rows)
    out: list[str] = [
        f"# Eval Scoreboard — {label} · {target}",
        "",
        f"- Tasks: {summary['tasks']} · OK: {summary['ok']} · Errors: {summary['errors']}",
        f"- Mean judge overall: {_fmt(summary['mean_judge_overall'])} "
        f"(gate ≥ {MIN_JUDGE_OVERALL})",
        f"- Mean verification rate: {_fmt(summary['mean_verification_rate'])} "
        f"(gate ≥ {MIN_VERIFICATION_RATE})",
        f"- Mean key-fact coverage: {_fmt(summary['mean_key_fact_coverage'])} "
        f"(gate ≥ {MIN_KEY_FACT_COVERAGE})",
        f"- Placeholder leaks: {summary['placeholder_leaks']} (gate = {MAX_PLACEHOLDER_LEAKS})",
        f"- Total cost: ${summary['total_cost_usd']:.4f}",
        "",
        f"**Release gate: {'PASS' if summary['gate_passed'] else 'FAIL'}**",
    ]
    if summary["gate_failures"]:
        out.append("")
        for f in summary["gate_failures"]:
            out.append(f"- ❌ {f}")
    out.append("")
    out.append(
        "| Task | Category | Status | Judge | Verif | Tier H+M | Coverage | "
        "Dead | Refine | Leaks | Cost | Dur |"
    )
    out.append(
        "|------|----------|--------|-------|-------|----------|----------|"
        "------|--------|-------|------|-----|"
    )
    for r in rows:
        if r.status == "ERROR" or r.metrics is None:
            out.append(
                f"| {r.task_id} | {r.category} | ERROR | - | - | - | - | - | - | - | - | - |"
            )
            continue
        m = r.metrics
        judge_cell = _fmt(r.judge.overall) if r.judge and r.judge.overall is not None else "-"
        refine_cell = f"{m.refinement_rounds}"
        if m.expects_refinement:
            refine_cell += "/exp"
        out.append(
            f"| {r.task_id} | {r.category} | OK | {judge_cell} | "
            f"{m.verification_rate:.2f} | {m.tier_mix_high_medium:.2f} | "
            f"{m.key_fact_coverage:.2f} | {m.dead_links} | {refine_cell} | "
            f"{m.placeholder_leaks} | ${m.cost_usd:.3f} | {m.duration_sec:.0f}s |"
        )
    out.append("")
    return "\n".join(out)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def write_scoreboard(
    rows: list[RunRow], *, target: str, label: str, results_root: Path | None = None
) -> tuple[Path, Path]:
    root = (results_root or RESULTS_ROOT) / label / target
    root.mkdir(parents=True, exist_ok=True)
    md_path = root / "scoreboard.md"
    json_path = root / "scoreboard.json"
    md_path.write_text(render_scoreboard_md(rows, target=target, label=label), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_scoreboard_json(rows, target=target, label=label), indent=2),
        encoding="utf-8",
    )
    return md_path, json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_evals",
        description="Run the polysearch eval sweep and write a scoreboard.",
    )
    parser.add_argument(
        "--target",
        choices=["public", "internal", "both"],
        required=True,
        help="Which pipeline to evaluate.",
    )
    parser.add_argument(
        "--rounds-label",
        required=True,
        help="Round label, e.g. r1 (names the results/<label>/ dir).",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task ids to run (default: all).",
    )
    parser.add_argument(
        "--tasks-file",
        help="External tasks YAML overriding evals/tasks.yaml (for private suites).",
    )
    parser.add_argument(
        "--depth-override",
        choices=["quick", "standard", "deep"],
        help="Force every task to this depth.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judging (programmatic metrics only).",
    )
    parser.add_argument(
        "--output-root",
        help="Where per-task reports are written (default: a temp dir).",
    )
    return parser


def _targets(target: str) -> list[str]:
    return ["public", "internal"] if target == "both" else [target]


_RUNNERS: dict[str, Callable[[dict, Path], tuple[dict, str]]] = {
    "public": run_public_target,
    "internal": run_internal_target,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tasks = load_tasks(tasks_file=args.tasks_file)
    tasks = select_tasks(tasks, ids=args.tasks)
    tasks = apply_depth_override(tasks, args.depth_override)

    judge_fn = None if args.no_judge else _default_judge

    if args.output_root:
        out_root = Path(args.output_root)
    else:
        import tempfile

        out_root = Path(tempfile.mkdtemp(prefix="polysearch-evals-"))

    overall_pass = True
    for target in _targets(args.target):
        runner = _RUNNERS[target]
        rows = run_sweep(
            tasks,
            target=target,
            out_root=out_root,
            run_target=runner,
            judge_fn=judge_fn,
        )
        md_path, json_path = write_scoreboard(
            rows, target=target, label=args.rounds_label
        )
        passed, failures = evaluate_quality_bar(rows)
        overall_pass = overall_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{target}] gate: {status} → {md_path}")
        for f in failures:
            print(f"  - {f}", file=sys.stderr)

    return 0 if overall_pass else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
