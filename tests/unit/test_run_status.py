"""Completion sentinel + run manifest (``polysearch.run_status``), and the
``run_research`` wrapper's guarantee that every run leaves a sentinel."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from polysearch import orchestrator, run_status
from polysearch.config import Settings
from polysearch.output.schema import PipelineReport

# ── Sentinel ──────────────────────────────────────────────────────────────


def test_sentinel_path_is_distinct_from_payload_json(tmp_path: Path) -> None:
    md = tmp_path / "2026-08-05-topic.md"
    assert run_status.sentinel_path(md) == tmp_path / "2026-08-05-topic.done.json"


def test_write_sentinel_roundtrip(tmp_path: Path) -> None:
    md = tmp_path / "2026-08-05-topic.md"
    md.write_text("# Report\n\nbody\n", encoding="utf-8")
    path = run_status.write_sentinel(
        md,
        status="complete",
        started_at="2026-08-05T10:00:00",
        exit_code=0,
        phases_completed=["first-pass", "verification", "saving"],
        json_path=md.with_suffix(".json"),
        total_cost_usd=1.23,
        pipeline_error_count=0,
    )
    assert path is not None and path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["output_path"] == str(md)
    assert data["report_bytes"] == md.stat().st_size > 0
    assert data["exit_code"] == 0
    assert data["phases_completed"] == ["first-pass", "verification", "saving"]
    assert data["error"] is None
    assert data["pid"] == os.getpid()


def test_write_sentinel_failed_without_report_file(tmp_path: Path) -> None:
    md = tmp_path / "never-written.md"
    path = run_status.write_sentinel(
        md,
        status="failed",
        started_at="2026-08-05T10:00:00",
        exit_code=1,
        phases_completed=["first-pass"],
        error="RuntimeError: boom",
    )
    assert path is not None
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert data["report_bytes"] == 0
    assert data["error"] == "RuntimeError: boom"


def test_clear_sentinel(tmp_path: Path) -> None:
    md = tmp_path / "r.md"
    run_status.write_sentinel(
        md, status="complete", started_at="x", exit_code=0, phases_completed=[]
    )
    assert run_status.sentinel_path(md).is_file()
    run_status.clear_sentinel(md)
    assert not run_status.sentinel_path(md).exists()
    run_status.clear_sentinel(md)  # idempotent on missing


# ── Manifest ──────────────────────────────────────────────────────────────


def test_manifest_lifecycle() -> None:
    m = run_status.RunManifest.create(
        topic="test topic", depth="standard", output_path="/tmp/r.md"
    )
    assert m.path.is_file()
    data = json.loads(m.path.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert data["phase"] == "starting"
    assert data["pid"] == os.getpid()

    m.update_phase("first-pass")
    m.update_phase("verification")
    data = json.loads(m.path.read_text(encoding="utf-8"))
    assert data["phase"] == "verification"
    assert data["phases_completed"] == ["first-pass"]

    m.finish("complete")
    data = json.loads(m.path.read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["finished_at"] is not None
    assert "verification" in data["phases_completed"]


def test_manifest_prune_removes_old() -> None:
    runs = run_status.runs_dir()
    runs.mkdir(parents=True, exist_ok=True)
    old = runs / "ancient.json"
    old.write_text("{}", encoding="utf-8")
    stale = time.time() - 8 * 86400
    os.utime(old, (stale, stale))
    run_status.RunManifest.create(topic="t", depth="quick", output_path=None)
    assert not old.exists()


# ── run_research wrapper: sentinel on failure and success ────────────────


def test_run_research_writes_failed_sentinel_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(topic: str, **kwargs: object) -> PipelineReport:
        raise RuntimeError("layer exploded")

    monkeypatch.setattr(orchestrator, "_run_research_inner", boom)
    with pytest.raises(RuntimeError, match="layer exploded"):
        asyncio.run(
            orchestrator.run_research(
                "crash topic",
                settings=Settings(),
                output_dir=tmp_path,
                write=True,
            )
        )

    expected_md = tmp_path / f"{time.strftime('%Y-%m-%d')}-crash-topic.md"
    sentinel = run_status.sentinel_path(expected_md)
    assert sentinel.is_file()
    data = json.loads(sentinel.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "layer exploded" in data["error"]

    manifests = list(run_status.runs_dir().glob("*.json"))
    assert len(manifests) == 1
    mdata = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert mdata["status"] == "failed"


def test_run_research_writes_complete_sentinel_and_clears_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_md = tmp_path / f"{time.strftime('%Y-%m-%d')}-ok-topic.md"
    # A stale sentinel from a prior run must not survive into the new run.
    run_status.write_sentinel(
        expected_md, status="complete", started_at="stale", exit_code=0, phases_completed=[]
    )
    seen: dict[str, bool] = {}

    async def fake_inner(topic: str, **kwargs: object) -> PipelineReport:
        seen["stale_cleared"] = not run_status.sentinel_path(expected_md).exists()
        expected_md.write_text("# Report\n", encoding="utf-8")
        return PipelineReport(
            topic=topic,
            depth="standard",
            totals={"cost_usd": 0.5},
            output_md_path=str(expected_md),
            output_json_path=str(expected_md.with_suffix(".json")),
        )

    monkeypatch.setattr(orchestrator, "_run_research_inner", fake_inner)
    report = asyncio.run(
        orchestrator.run_research(
            "ok topic", settings=Settings(), output_dir=tmp_path, write=True
        )
    )
    assert seen["stale_cleared"] is True
    assert report.output_md_path == str(expected_md)

    sentinel = run_status.sentinel_path(expected_md)
    data = json.loads(sentinel.read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["report_bytes"] == expected_md.stat().st_size
    assert data["total_cost_usd"] == 0.5
    assert data["started_at"] != "stale"
