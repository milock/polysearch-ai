"""Machine-readable run status: per-run manifest + completion sentinel.

Orchestrators that fan polysearch out across agents or processes need a
dependable "done" signal — inferring completion from prose or from the report
``.md`` existing is unreliable (a report can exist half-written). Two artifacts
make "done" a declared fact:

1. **Completion sentinel** — ``<report>.done.json`` written next to the report
   markdown on exit, success or failure. Poll THIS, never the ``.md``: the
   sentinel is written only after the atomic report save (or with
   ``status: "failed"`` when the run dies).

2. **Run manifest** — ``~/.cache/polysearch/runs/<run-id>.json``, updated at
   every phase boundary and by a ~30s heartbeat. Turns "queued vs dead vs
   finished" into a file read: stale heartbeat + dead PID + no sentinel = dead;
   fresh heartbeat = alive (possibly queued at the cross-process rate limiter).

Never crashes the caller: all writes are best-effort — a status-file failure
must not take down a research run that otherwise succeeded.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

HEARTBEAT_INTERVAL_SEC = 30
MANIFEST_MAX_AGE_DAYS = 7


def runs_dir() -> Path:
    return Path.home() / ".cache" / "polysearch" / "runs"


def sentinel_path(report_md_path: Path | str) -> Path:
    """``reports/2026-08-05-topic.md`` -> ``reports/2026-08-05-topic.done.json``.

    Distinct from the sibling structured payload ``2026-08-05-topic.json``.
    """
    return Path(report_md_path).with_suffix(".done.json")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, prefix=".tmp-", suffix=".json", encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def clear_sentinel(report_md_path: Path | str) -> None:
    """Remove any sentinel from a prior run of the same output path, so a
    relaunch can't be mistaken for already-done."""
    try:
        sentinel_path(report_md_path).unlink(missing_ok=True)
    except OSError:
        pass


def write_sentinel(
    report_md_path: Path | str,
    *,
    status: str,  # "complete" | "failed"
    started_at: str,
    exit_code: int,
    phases_completed: list[str],
    error: str | None = None,
    json_path: Path | str | None = None,
    total_cost_usd: float | None = None,
    pipeline_error_count: int | None = None,
) -> Path | None:
    """Write ``<report>.done.json``. Best-effort — returns the path or ``None``."""
    md_path = Path(report_md_path)
    report_bytes = md_path.stat().st_size if md_path.is_file() else 0
    data = {
        "status": status,
        "output_path": str(md_path),
        "json_path": str(json_path) if json_path else None,
        "report_bytes": report_bytes,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "phases_completed": phases_completed,
        "pipeline_error_count": pipeline_error_count,
        "total_cost_usd": total_cost_usd,
        "error": error,
        "pid": os.getpid(),
    }
    try:
        path = sentinel_path(md_path)
        _atomic_write_json(path, data)
        return path
    except OSError:
        return None


class RunManifest:
    """Heartbeat-bearing manifest for one pipeline run."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def create(
        cls,
        *,
        topic: str,
        depth: str,
        output_path: Path | str | None,
    ) -> "RunManifest":
        started = datetime.now()
        run_id = f"{started.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"
        path = runs_dir() / f"{run_id}.json"
        data = {
            "run_id": run_id,
            "pid": os.getpid(),
            "topic": topic,
            "depth": depth,
            "output_path": str(output_path) if output_path else None,
            "status": "running",
            "phase": "starting",
            "phases_completed": [],
            "started_at": started.isoformat(timespec="seconds"),
            "heartbeat_at": started.isoformat(timespec="seconds"),
            "finished_at": None,
            "error": None,
        }
        manifest = cls(path, data)
        manifest._save()
        _prune_old_manifests()
        return manifest

    @property
    def started_at(self) -> str:
        return self.data["started_at"]

    @property
    def phases_completed(self) -> list[str]:
        return list(self.data["phases_completed"])

    def _save(self) -> None:
        try:
            _atomic_write_json(self.path, self.data)
        except OSError:
            pass

    def touch(self) -> None:
        self.data["heartbeat_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()

    def update_phase(self, phase: str) -> None:
        prev = self.data.get("phase")
        if prev and prev != "starting" and prev not in self.data["phases_completed"]:
            self.data["phases_completed"].append(prev)
        self.data["phase"] = phase
        self.touch()

    def finish(self, status: str, *, error: str | None = None) -> None:
        self.update_phase("done")
        self.data["status"] = status
        self.data["error"] = error
        self.data["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()

    async def heartbeat_loop(self) -> None:
        """Run as a background asyncio task; cancelled by the orchestrator."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
                self.touch()
        except asyncio.CancelledError:
            pass


def _prune_old_manifests(max_age_days: int = MANIFEST_MAX_AGE_DAYS) -> None:
    cutoff = time.time() - timedelta(days=max_age_days).total_seconds()
    try:
        for f in runs_dir().glob("*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        pass
