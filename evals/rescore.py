"""Score-only rescore mode.

Re-scores *existing* report artifacts without re-running any target. Point it at
a directory of reports (flat ``md``+``json`` pairs, or per-task subdirectories),
and for each task it locates the matching report by topic slug, recomputes the
programmatic metrics through the same adapter the live sweep uses, optionally
re-judges the markdown, and writes a scoreboard — reusing the exact builders,
gate, and row model from :mod:`evals.run_evals` so a rescored round is directly
comparable to a live one.

A task with no matching artifact is recorded ``SKIPPED`` (not ERROR) — it simply
was not run in the round being rescored.

Usage::

    python -m evals.rescore --target internal --artifacts DIR --label r1-rescored
    python -m evals.rescore --target internal --artifacts DIR --label r1-lm \\
        --tasks-file /path/to/internal-tasks.yaml
    python -m evals.rescore --target public --artifacts DIR --label r1-rescored --no-judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from evals import metrics as metrics_mod
from evals.judge import JudgeResult
from evals.run_evals import (
    RunRow,
    _default_judge,
    _load_report,
    _match_report_json,
    apply_depth_override,
    evaluate_quality_bar,
    load_tasks,
    select_tasks,
    write_scoreboard,
)


def find_artifact_jsons(artifacts_dir: Path) -> list[Path]:
    """Every report json under ``artifacts_dir``, oldest-first.

    Supports both layouts: flat ``md``+``json`` pairs directly in the dir, and
    per-task subdirectories one level down (as the live sweep persists them).
    """
    flat = set(artifacts_dir.glob("*.json"))
    nested = set(artifacts_dir.glob("*/*.json"))
    return sorted(flat | nested, key=lambda p: p.stat().st_mtime)


def _report_rel(json_path: Path, artifacts_dir: Path) -> str:
    md_path = json_path.with_suffix(".md")
    try:
        return md_path.relative_to(artifacts_dir).as_posix()
    except ValueError:
        return md_path.name


def rescore(
    tasks: list[dict],
    *,
    artifacts_dir: Path,
    judge_fn: Optional[Callable[[str, list[str], str], Optional[JudgeResult]]] = None,
) -> list[RunRow]:
    """Rescore each task against the artifacts in ``artifacts_dir``.

    ``judge_fn`` is injected in tests; in production it is the OpenAI judge. A
    missing artifact yields a ``SKIPPED`` row; a metrics/judge failure on a found
    artifact yields ``ERROR``; otherwise ``OK`` (or ``ERROR`` if the judge could
    not be parsed).
    """
    all_jsons = find_artifact_jsons(artifacts_dir)
    rows: list[RunRow] = []
    for task in tasks:
        task_id = task.get("id", "<unknown>")
        category = task.get("category", "")
        json_path = _match_report_json(all_jsons, task["topic"])
        if json_path is None:
            rows.append(
                RunRow(
                    task_id=task_id,
                    category=category,
                    status="SKIPPED",
                    error="no artifact found matching topic slug",
                )
            )
            continue
        try:
            report, md, _ = _load_report(json_path)
            run_metrics = metrics_mod.compute_metrics(report, task, report_md=md)
            judge_result: Optional[JudgeResult] = None
            if judge_fn is not None:
                judge_result = judge_fn(task["topic"], task.get("key_facts", []), md)
            status = "ERROR" if (judge_result is not None and judge_result.error) else "OK"
            rows.append(
                RunRow(
                    task_id=task_id,
                    category=category,
                    status=status,
                    metrics=run_metrics,
                    judge=judge_result,
                    error=judge_result.error if judge_result else None,
                    report_path=_report_rel(json_path, artifacts_dir),
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad artifact must not sink the rescore
            rows.append(
                RunRow(task_id=task_id, category=category, status="ERROR", error=str(exc))
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rescore",
        description="Re-score existing report artifacts without re-running targets.",
    )
    parser.add_argument(
        "--target",
        choices=["public", "internal"],
        required=True,
        help="Names the results/<label>/<target>/ dir the scoreboard is written to.",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        help="Directory of report artifacts (flat md+json pairs or per-task subdirs).",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Round label, e.g. r1-rescored (names the results/<label>/ dir).",
    )
    parser.add_argument(
        "--tasks-file",
        help="External tasks YAML overriding evals/tasks.yaml (for private suites).",
    )
    parser.add_argument("--tasks", help="Comma-separated task ids to rescore (default: all).")
    parser.add_argument(
        "--depth-override",
        choices=["quick", "standard", "deep"],
        help="Force every task to this depth (shifts the refinement ceiling).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judging (programmatic metrics only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    tasks = load_tasks(tasks_file=args.tasks_file)
    tasks = select_tasks(tasks, ids=args.tasks)
    tasks = apply_depth_override(tasks, args.depth_override)

    judge_fn = None if args.no_judge else _default_judge

    rows = rescore(tasks, artifacts_dir=Path(args.artifacts), judge_fn=judge_fn)
    md_path, _ = write_scoreboard(rows, target=args.target, label=args.label)
    passed, failures = evaluate_quality_bar(rows)
    status = "PASS" if passed else "FAIL"
    skipped = sum(1 for r in rows if r.status == "SKIPPED")
    print(f"[{args.target}] rescored {len(rows)} tasks ({skipped} skipped) → {md_path}")
    print(f"[{args.target}] gate: {status}")
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
