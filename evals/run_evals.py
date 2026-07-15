"""Eval sweep runner.

Drives the simulated deep-research tasks through a target CLI, computes
programmatic metrics (``evals.metrics``), scores each report with the LLM judge
(``evals.judge``), and writes a scoreboard per round to
``evals/results/<label>/<target>/scoreboard.{md,json}``. The collected md+json
for each task are persisted alongside, under
``evals/results/<label>/<target>/<task_id>/``, and linked from the scoreboard row.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from polysearch.output.report import slugify

from evals import judge as judge_mod
from evals import metrics as metrics_mod
from evals.judge import JudgeResult
from evals.metrics import RunMetrics

# --------------------------------------------------------------------------- #
# Release gate constants (documented in evals/README.md)
# --------------------------------------------------------------------------- #
MIN_JUDGE_OVERALL = 0.80
MIN_VERIFICATION_RATE = 0.70  # applied to the CLAIM-level rate (claims_supported/claims_total)
MIN_KEY_FACT_COVERAGE = 0.85
MAX_PLACEHOLDER_LEAKS = 0
MAX_CRASHES = 0
MIN_REFINEMENT_TRIGGER_RATE = 0.80
# The refinement ceiling is not a flat constant — each run is checked against its
# own depth's max_refinement_iterations cap (quick 0 / standard 2 / deep 4), read
# from polysearch.config.DEPTH_PROFILES by evals.metrics.

DEFAULT_TASKS_FILE = Path(__file__).resolve().parent / "tasks.yaml"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# Per-task subprocess timeout (seconds). A deep run under N-way rate-sharing is
# slow; the default is generous and overridable via POLYSEARCH_EVAL_TIMEOUT_SEC.
DEFAULT_TIMEOUT_SEC = 2700


def timeout_sec() -> int:
    """Resolve the per-task subprocess timeout, honoring the env override."""
    raw = os.environ.get("POLYSEARCH_EVAL_TIMEOUT_SEC")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_SEC


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
    report_path: Optional[str] = None  # persisted md, relative to the scoreboard dir

    @property
    def warnings(self) -> list[str]:
        return list(self.metrics.warnings) if self.metrics else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "status": self.status,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "judge": self.judge.to_dict() if self.judge else None,
            "error": self.error,
            "report_path": self.report_path,
            "warnings": self.warnings,
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
def _match_report_json(json_paths: list[Path], topic: str) -> Path | None:
    """The newest json whose stem ends with the topic's slug, or ``None``.

    Slug-matching means a directory holding many reports never hands back the
    wrong task's file. Shared by the live collector (which falls back to newest)
    and the rescore finder (which treats no match as SKIPPED).
    """
    slug = slugify(topic)
    matched = sorted(
        (p for p in json_paths if p.stem.endswith(slug)),
        key=lambda p: p.stat().st_mtime,
    )
    return matched[-1] if matched else None


def _load_report(json_path: Path) -> tuple[dict, str, Path]:
    """Load a report json and its sibling ``.md`` (empty md when none). Returns
    ``(report, md, md_path)``."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    md_path = json_path.with_suffix(".md")
    md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    return report, md, md_path


def _collect_report(out_dir: Path, topic: str) -> tuple[dict, str, Path]:
    """Load the report json for ``topic`` from ``out_dir`` and its sibling md.

    Prefers the file whose stem ends with the topic's slug (so a shared output
    dir never hands back a different task's report); falls back to the newest
    json. Returns ``(report, md, md_path)``.
    """
    jsons = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise RuntimeError(f"target produced no report json in {out_dir}")
    json_path = _match_report_json(jsons, topic) or jsons[-1]
    return _load_report(json_path)


def run_public_target(task: dict, out_dir: Path) -> tuple[dict, str, Path]:
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
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec())
    if proc.returncode != 0:
        raise RuntimeError(
            f"polysearch exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return _collect_report(out_dir, task["topic"])


def run_internal_target(task: dict, out_dir: Path) -> tuple[dict, str, Path]:
    """Run the ``POLYSEARCH_EVAL_INTERNAL_CMD`` command for one task.

    The base command is read from the environment and shell-split; the topic is
    appended as the final positional argument. ``POLYSEARCH_OUTPUT_DIR`` is
    exported so a cooperating internal pipeline (or its wrapper) writes into
    ``out_dir``; the report is then collected the same way as for the public
    target.
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
        cmd, capture_output=True, text=True, timeout=timeout_sec(), env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"internal target exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    return _collect_report(out_dir, task["topic"])


# Monotonic timestamp of the last real judge call, for inter-call pacing. Lives
# here (not in judge_report) so only production sweeps pace — tests inject their
# own judge_fn and never sleep.
_last_judge_call_ts = 0.0


def make_default_judge(
    full_md: bool = False,
) -> Callable[[str, list[str], str], JudgeResult]:
    """Build the production judge callable: paces calls by the configured spacing
    (POLYSEARCH_EVAL_JUDGE_RPS or 0.5s) so a multi-task sweep doesn't burst the org
    rate limit, then delegates to the retrying ``judge_report``."""
    import time

    def _judge(topic: str, key_facts: list[str], report_md: str) -> JudgeResult:
        global _last_judge_call_ts
        interval = judge_mod.judge_spacing_sec()
        wait = _last_judge_call_ts + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_judge_call_ts = time.monotonic()
        return judge_mod.judge_report(topic, key_facts, report_md, full_md=full_md)

    return _judge


def _persist_artifacts(md_path: Path, task_dir: Path) -> Optional[str]:
    """Ensure the collected md+json live under ``task_dir`` and return the md path
    relative to the target dir (the scoreboard's dir), for the scoreboard row.

    When the target already wrote into ``task_dir`` (the normal case), this is a
    no-op copy; when it wrote elsewhere, the artifacts are copied in so a round's
    reports survive for diagnosis.
    """
    import shutil

    task_dir.mkdir(parents=True, exist_ok=True)
    json_src = md_path.with_suffix(".json")
    for src in (md_path, json_src):
        if src.exists() and src.parent.resolve() != task_dir.resolve():
            shutil.copy2(src, task_dir / src.name)
    persisted_md = task_dir / md_path.name
    if persisted_md.exists():
        # target dir is task_dir.parent; return path relative to it.
        return persisted_md.relative_to(task_dir.parent).as_posix()
    return None


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    tasks: list[dict],
    *,
    target: str,
    out_root: Path,
    run_target: Callable[[dict, Path], tuple[dict, str, Path]],
    judge_fn: Optional[Callable[[str, list[str], str], Optional[JudgeResult]]] = None,
) -> list[RunRow]:
    """Run every task, isolating failures as ERROR rows.

    ``run_target(task, out_dir) -> (report_json, report_md, md_path)`` produces the
    report; ``judge_fn(topic, key_facts, report_md) -> JudgeResult | None`` scores
    it (``None`` skips judging). A crash in either, a judge parse error, or a
    metrics failure marks the task ERROR and the sweep moves on.

    Per-task reports persist under ``out_root/target/task_id/`` and the row records
    the md path relative to ``out_root/target`` (the scoreboard's dir).
    """
    rows: list[RunRow] = []
    for task in tasks:
        task_id = task.get("id", "<unknown>")
        category = task.get("category", "")
        task_dir = out_root / target / task_id
        try:
            report, report_md, md_path = run_target(task, task_dir)
            report_path = _persist_artifacts(md_path, task_dir)
            run_metrics = metrics_mod.compute_metrics(report, task, report_md=report_md)
            judge_result: Optional[JudgeResult] = None
            if judge_fn is not None:
                judge_result = judge_fn(
                    task["topic"], task.get("key_facts", []), report_md
                )
            status = "ERROR" if (judge_result is not None and judge_result.error) else "OK"
            rows.append(
                RunRow(
                    task_id=task_id,
                    category=category,
                    status=status,
                    metrics=run_metrics,
                    judge=judge_result,
                    error=judge_result.error if judge_result else None,
                    report_path=report_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 — failure isolation: one task must not sink the sweep
            rows.append(
                RunRow(
                    task_id=task_id, category=category, status="ERROR", error=str(exc)
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Aggregation helpers (None-safe — a missing metric never counts as zero)
# --------------------------------------------------------------------------- #
def _mean_opt(values: list[float | None]) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------- #
# Quality bar
# --------------------------------------------------------------------------- #
def evaluate_quality_bar(rows: list[RunRow]) -> tuple[bool, list[str]]:
    """Check a sweep against the documented release gate.

    Returns ``(passed, failures)``; ``failures`` lists every threshold missed
    (empty when the gate passes). A metric that is ``None`` for *every* OK run
    (e.g. no verification block anywhere) is itself a failure — a poisoned round
    must not pass by having nothing to measure.
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
        judge_mean = _mean_opt([r.judge.overall for r in judged])
        if judge_mean is not None and judge_mean < MIN_JUDGE_OVERALL:
            failures.append(f"mean judge overall {judge_mean:.3f} < {MIN_JUDGE_OVERALL}")

    verif_mean = _mean_opt([r.metrics.verification_rate for r in ok])
    if verif_mean is None:
        failures.append("claim-level verification rate unavailable for every run")
    elif verif_mean < MIN_VERIFICATION_RATE:
        failures.append(
            f"mean claim-level verification {verif_mean:.3f} < {MIN_VERIFICATION_RATE}"
        )

    coverage_mean = _mean_opt([r.metrics.key_fact_coverage for r in ok])
    if coverage_mean is None:
        failures.append("key-fact coverage unavailable for every run")
    elif coverage_mean < MIN_KEY_FACT_COVERAGE:
        failures.append(
            f"mean key-fact coverage {coverage_mean:.3f} < {MIN_KEY_FACT_COVERAGE}"
        )

    leaks = sum(r.metrics.placeholder_leaks for r in ok)
    if leaks > MAX_PLACEHOLDER_LEAKS:
        failures.append(f"{leaks} placeholder leak(s) (> {MAX_PLACEHOLDER_LEAKS})")

    expecting = [r for r in ok if r.metrics.expects_refinement]
    if expecting:
        triggered = sum(1 for r in expecting if (r.metrics.refinement_rounds or 0) >= 1)
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
    rows_with_warnings = sum(1 for r in rows if r.warnings)
    return {
        "tasks": len(rows),
        "ok": len(ok),
        "errors": sum(1 for r in rows if r.status == "ERROR"),
        "skipped": sum(1 for r in rows if r.status == "SKIPPED"),
        "mean_judge_overall": _round_opt(_mean_opt([r.judge.overall for r in judged])),
        "mean_verification_rate": _round_opt(
            _mean_opt([r.metrics.verification_rate for r in ok])
        ),
        "mean_citation_pair_rate": _round_opt(
            _mean_opt([r.metrics.citation_pair_rate for r in ok])
        ),
        "mean_key_fact_coverage": _round_opt(
            _mean_opt([r.metrics.key_fact_coverage for r in ok])
        ),
        "placeholder_leaks": sum(r.metrics.placeholder_leaks for r in ok),
        "total_cost_usd": _round_opt(
            sum(r.metrics.cost_usd for r in ok if r.metrics.cost_usd is not None), 4
        ),
        "rows_with_warnings": rows_with_warnings,
        "gate_passed": passed,
        "gate_failures": failures,
    }


def _round_opt(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


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
        f"- Tasks: {summary['tasks']} · OK: {summary['ok']} · "
        f"Errors: {summary['errors']} · Skipped: {summary['skipped']}",
        f"- Mean judge overall: {_fmt(summary['mean_judge_overall'])} "
        f"(gate ≥ {MIN_JUDGE_OVERALL})",
        f"- Mean claim-level verification: {_fmt(summary['mean_verification_rate'])} "
        f"(gate ≥ {MIN_VERIFICATION_RATE}) · pair-level (ungated): "
        f"{_fmt(summary['mean_citation_pair_rate'])}",
        f"- Mean key-fact coverage: {_fmt(summary['mean_key_fact_coverage'])} "
        f"(gate ≥ {MIN_KEY_FACT_COVERAGE})",
        f"- Placeholder leaks: {summary['placeholder_leaks']} (gate = {MAX_PLACEHOLDER_LEAKS})",
        f"- Total cost: {_fmt_cost(summary['total_cost_usd'])} · "
        f"rows with warnings: {summary['rows_with_warnings']}",
        "",
        f"**Release gate: {'PASS' if summary['gate_passed'] else 'FAIL'}**",
    ]
    if summary["gate_failures"]:
        out.append("")
        for f in summary["gate_failures"]:
            out.append(f"- ❌ {f}")
    out.append("")
    out.append(
        "| Task | Category | Status | Shape | Judge | Verif(claim) | Pair | "
        "Tier H+M | Coverage | Dead | Refine | Leaks | Cost | Dur | Report |"
    )
    out.append(
        "|------|----------|--------|-------|-------|--------------|------|"
        "----------|----------|------|--------|-------|------|-----|--------|"
    )
    for r in rows:
        if r.metrics is None:
            # ERROR (crash) or SKIPPED (no artifact) — no metrics to render.
            out.append(
                f"| {r.task_id} | {r.category} | {r.status} | - | - | - | - | - | - | "
                "- | - | - | - | - | - |"
            )
            continue
        m = r.metrics
        judge_cell = _fmt(r.judge.overall) if r.judge and r.judge.overall is not None else "-"
        refine_cell = "-" if m.refinement_rounds is None else str(m.refinement_rounds)
        if m.expects_refinement:
            refine_cell += "/exp"
        report_cell = r.report_path or "-"
        out.append(
            f"| {r.task_id} | {r.category} | {r.status} | {m.report_shape} | "
            f"{judge_cell} | {_fmt(m.verification_rate, 2)} | "
            f"{_fmt(m.citation_pair_rate, 2)} | {_fmt(m.tier_mix_high_medium, 2)} | "
            f"{_fmt(m.key_fact_coverage, 2)} | {_fmt_int(m.dead_links)} | "
            f"{refine_cell} | {m.placeholder_leaks} | {_fmt_cost(m.cost_usd)} | "
            f"{_fmt_dur(m.duration_sec)} | {report_cell} |"
        )
    out.append("")

    warned = [r for r in rows if r.warnings]
    if warned:
        out.append("## Warnings")
        out.append("")
        for r in warned:
            out.append(f"- **{r.task_id}**: {'; '.join(r.warnings)}")
        out.append("")
    return "\n".join(out)


def _fmt(value: float | None, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}" if value is not None else "-"


def _fmt_int(value: int | None) -> str:
    return str(value) if value is not None else "-"


def _fmt_cost(value: float | None) -> str:
    return f"${value:.3f}" if value is not None else "n/a"


def _fmt_dur(value: float | None) -> str:
    return f"{value:.0f}s" if value is not None else "n/a"


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
    parser.add_argument("--tasks", help="Comma-separated task ids to run (default: all).")
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
        "--judge-full-md",
        action="store_true",
        help="Feed the whole report md to the judge (default strips process/audit sections).",
    )
    parser.add_argument(
        "--output-root",
        help="Where per-task reports persist (default: evals/results/<label>).",
    )
    return parser


def _targets(target: str) -> list[str]:
    return ["public", "internal"] if target == "both" else [target]


_RUNNERS: dict[str, Callable[[dict, Path], tuple[dict, str, Path]]] = {
    "public": run_public_target,
    "internal": run_internal_target,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tasks = load_tasks(tasks_file=args.tasks_file)
    tasks = select_tasks(tasks, ids=args.tasks)
    tasks = apply_depth_override(tasks, args.depth_override)

    judge_fn = None if args.no_judge else make_default_judge(args.judge_full_md)

    out_root = Path(args.output_root) if args.output_root else RESULTS_ROOT / args.rounds_label

    overall_pass = True
    for target in _targets(args.target):
        runner = _RUNNERS[target]
        rows = run_sweep(
            tasks, target=target, out_root=out_root, run_target=runner, judge_fn=judge_fn
        )
        md_path, _ = write_scoreboard(rows, target=target, label=args.rounds_label)
        passed, failures = evaluate_quality_bar(rows)
        overall_pass = overall_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"[{target}] gate: {status} → {md_path}")
        for f in failures:
            print(f"  - {f}", file=sys.stderr)

    return 0 if overall_pass else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
