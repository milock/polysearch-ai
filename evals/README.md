# polysearch evals

An offline eval harness for the polysearch pipeline. It runs a fixed suite of
simulated deep-research tasks through a target CLI, computes objective metrics
from each report, scores each report with an LLM judge, and writes a scoreboard
per round. Use it to catch regressions and to measure improvement across rounds.

This directory ships in the repo but is **excluded from the built package** — the
distribution contains only `src/polysearch`. Nothing here is imported at runtime
by the library or CLI.

## What it measures

For every task the harness records two kinds of signal.

**Programmatic metrics** (`metrics.py`, model-free and deterministic):

| Metric | Meaning |
|--------|---------|
| Citation verification rate | `verified_ok / total_citations` from the run's verification report |
| Source-tier mix | fraction of unique sources that are HIGH or MEDIUM tier |
| Dead-link count | citations marked `URL_DEAD` |
| Key-fact coverage | fraction of the task's `key_facts` found in the report (rapidfuzz partial-ratio ≥ 80) |
| Refinement rounds | goal-driven refinement iterations that ran follow-up queries, vs. `expects_refinement` |
| Placeholder leaks | unresolved `{{…}}` template tokens left in the report |
| Cost / duration | from the report's `totals` |

**LLM judge** (`judge.py`, `gpt-5.4-nano`, structured output): factual accuracy,
citation accuracy, completeness against the task's key facts, source quality, and
coherence/insight — each scored 0.0–1.0 with a one-line justification, plus an
overall score and pass/fail. The judge sees the report **markdown only** (it
evaluates the end state, not the process). The judge schema never requests a URL
field — models fabricate URLs when a schema invites one, so citation accuracy is
judged from the source tags shown, and URL integrity is measured programmatically
upstream.

## Running

The harness targets a CLI, not the library directly:

```bash
# Score the installed public CLI, round r1
python -m evals.run_evals --target public --rounds-label r1

# A subset of tasks
python -m evals.run_evals --target public --rounds-label r1 \
    --tasks factual-fed-rate,trend-agentic-coding

# Force a depth for the whole suite
python -m evals.run_evals --target public --rounds-label r2 --depth-override deep

# Metrics only, skip the judge (no OpenAI key needed)
python -m evals.run_evals --target public --rounds-label r1 --no-judge
```

Scoreboards are written to `evals/results/<rounds-label>/<target>/scoreboard.{md,json}`.
The raw per-task reports under `results/` are gitignored; commit the scoreboards
per round.

### Targets

- **`public`** invokes the installed `polysearch` CLI (`polysearch --topic … --depth …`).
- **`internal`** invokes the command in the `POLYSEARCH_EVAL_INTERNAL_CMD`
  environment variable, with the topic appended as the final argument. No
  internal path or module name lives in this repo — the internal command is
  supplied entirely from the environment. The internal pipeline should honour
  `POLYSEARCH_OUTPUT_DIR` (exported by the harness) or otherwise write its
  `md`+`json` report into the per-task directory.
- **`both`** runs each in turn and writes a scoreboard for each.

### Private task suites

`--tasks-file PATH` replaces the bundled `tasks.yaml` with an external YAML of the
same shape. This lets private or vertical task suites run through the same harness
without ever living in the public repo.

## Cost expectations

Each task is one pipeline run plus one judge call. Pipeline cost dominates and
depends on depth (roughly a few cents for `quick`/`standard`, more for `deep`).
The judge uses `gpt-5.4-nano` ($0.20 / $1.25 per 1M in/out) — a fraction of a
cent per report — so four or more full rounds stay inexpensive. Use `--no-judge`
to run the free, deterministic metrics alone.

## Release gate

`evaluate_quality_bar()` (asserted at the end of each sweep, and reflected in the
scoreboard) enforces, across both targets:

| Gate | Threshold |
|------|-----------|
| Mean judge overall | ≥ 0.80 |
| Mean citation verification rate | ≥ 0.70 |
| Mean key-fact coverage | ≥ 0.85 |
| Placeholder leaks | 0 |
| Unhandled task crashes | 0 |
| Refinement triggers on `expects_refinement` tasks | ≥ 80% |
| Refinement rounds on any task | ≤ 4 (never exceeds the loop ceiling) |

The runner exits non-zero when the gate fails, so it can gate a release in CI or a
script.

## Files

- `tasks.yaml` — the 12-task suite (2 each across FACTUAL, COMPARISON, TECHNICAL,
  TREND, RECENT, CONTESTED).
- `run_evals.py` — sweep runner, scoreboard, and quality-bar gate.
- `metrics.py` — programmatic metrics from a report json.
- `judge.py` — LLM-as-judge (rubric, schema, parsing).
- `results/` — per-round scoreboards (raw reports gitignored).
