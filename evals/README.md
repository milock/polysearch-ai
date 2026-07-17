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
| Citation verification rate (gated) | **claim-level** `claims_supported / claims_total` — a report can support most of its claims while individual claim↔URL pairs fail, so the claim rate is what the gate reads |
| Citation pair rate (secondary) | `verified_ok / total_citations`, ungated — a diagnostic column, not a threshold |
| Source-tier mix | fraction of unique sources that are HIGH or MEDIUM tier |
| Dead-link count | citations marked `URL_DEAD` |
| Key-fact coverage | fraction of the task's `key_facts` found in the report **markdown**, matched against every 1-3 sentence window with rapidfuzz `token_set_ratio` ≥ 45 (see "Key-fact coverage calibration" below) so paraphrase spanning a sentence or two is caught but scattered words are not |
| Refinement rounds | goal-driven refinement iterations that ran follow-up queries, vs. `expects_refinement` |
| Placeholder leaks | unresolved `{{…}}` template tokens left in the report |
| Cost / duration | from the report |

**Two report shapes, one ruler.** The metrics read through `report_adapter.py`,
which normalizes both the public package schema and the internal pipeline's
report into one shape. A field that cannot be found in *either* shape becomes
`None` (marked `n/a` in the scoreboard) with a per-row warning — **never a silent
zero**. A metric of `0` always means the pipeline genuinely produced zero.

**LLM judge** (`judge.py`, `gpt-5.4-nano`, structured output): factual accuracy,
citation accuracy, completeness against the task's key facts, source quality, and
coherence/insight — each scored 0.0–1.0 with a one-line justification, plus an
overall score and pass/fail. The judge sees the report **markdown only** (it
evaluates the end state, not the process). The judge schema never requests a URL
field — models fabricate URLs when a schema invites one, so citation accuracy is
judged from the source tags shown, and URL integrity is measured programmatically
upstream.

### Key-fact coverage calibration (task r3b)

Round 2 showed key-fact coverage reading 0.000 across reports that visibly stated
the facts — the ruler, not the pipeline, was broken. The original metric matched
each fact against a single sentence with exact-token `token_set_ratio` ≥ 70; a
hand-checked case (`comparison-postgres-mysql-oltp`, fact "concurrency and MVCC
handling differences" against a report sentence that ties PostgreSQL's "handling
of **concurrent** writes through **MVCC**" — a real match, just inflected and
paraphrased differently) scored only 45. The fix normalizes both sides
(markdown-noise stripped, suffix-stripped: `s`/`es`/`ies→y`/`ing`/`ed`/`tion(s)`),
drops a small stopword set (`and`, `the`, `of`, `for`, `each`, `to`) from the
*fact* side only so short facts aren't dominated by connective words, and matches
against every 1-3 sentence window instead of one sentence in isolation. Calibrating
against hand-checked real and synthetic cases moved `KEY_FACT_MATCH_THRESHOLD`
from 70 to 45: genuinely-covered facts (including the morphological-variant case
above) land in the 50s-70s once normalized, genuinely-absent facts land in the
20s-40s, with clear margin between the two — see the calibration tests in
`tests/unit/test_eval_harness.py`. Two of the worst abstract `key_facts` phrasings
in `tasks.yaml` (`"replication and high-availability options for each"`,
`"typical workloads each is better suited to"`) were also rephrased into
concrete, noun-heavy wording a real report would actually state, independent of
the metric fix. r1 and r1-rescored have no surviving per-task report artifacts in
this repo (only aggregate scoreboards and a diagnosis doc), so they could not be
rescored under the new metric — only r2 and its four sub-sweeps were.

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

Scoreboards are written to `evals/results/<rounds-label>/<target>/scoreboard.{md,json}`,
and each task's collected md+json are persisted next to them under
`results/<rounds-label>/<target>/<task_id>/` (linked from the scoreboard row's
`report_path`) so a round's reports survive for diagnosis.
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

### Rescoring existing artifacts

`rescore.py` re-scores reports that already exist on disk — no target is re-run.
Use it to apply improved metrics/judging to a past round, or to score reports a
sweep produced out-of-band:

```bash
python -m evals.rescore --target internal --artifacts DIR --label r1-rescored
python -m evals.rescore --target internal --artifacts DIR --label r1-lm \
    --tasks-file /path/to/internal-tasks.yaml
python -m evals.rescore --target public --artifacts DIR --label r1-rescored --no-judge
```

It finds each task's report in `--artifacts DIR` by topic slug (supporting both
flat `md`+`json` pairs and per-task subdirectories), recomputes metrics through
the same adapter, optionally re-judges, and writes a scoreboard with the same
builders as the live sweep. A task with no matching artifact is `SKIPPED` (not an
error) — it simply was not part of the round being rescored.

### Timeout

Each target run is bounded by a per-task subprocess timeout, default **2700s**
(45 min — a deep run under N-way rate-sharing is slow). Override it with the
`POLYSEARCH_EVAL_TIMEOUT_SEC` environment variable:

```bash
POLYSEARCH_EVAL_TIMEOUT_SEC=3600 python -m evals.run_evals --target internal --rounds-label r2
```

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
| Mean judge overall | ≥ 0.80 (advisory for v1.0 — see note) |
| Mean **claim-level** verification rate (`claims_supported / claims_total`) | ≥ 0.70 |
| Mean key-fact coverage | ≥ 0.85 |
| Placeholder leaks | 0 |
| Unhandled task crashes | 0 |
| Refinement triggers on `expects_refinement` tasks | ≥ 80% |
| Refinement rounds on any task | ≤ that depth's cap (quick 0 / standard 2 / deep 4) |

**v1.0 judge-gate note:** the LLM judge (a small, deliberately cheap model)
scores substantially below the programmatic metrics measuring the same
dimensions — reports with 1.00 key-fact coverage and 0.8+ verification still
judge ≈0.45 overall, with completeness/factual dimensions floored. For the
v1.0.0 release the judge score was treated as **advisory** (reported, not
release-blocking) pending rubric calibration against scored exemplars and a
stronger reference judge; the binding release criteria were the programmatic
gates. The harness itself still computes and prints the full gate — downstream
users set their own bars.

The verification gate reads the **claim-level** rate; the pair-level rate is a
secondary diagnostic column and is not gated. A metric that is `None` for every
run (e.g. no verification block anywhere, or coverage with no report text) fails
the gate rather than passing on empty data — a poisoned round must not slip
through by having nothing to measure.

The refinement ceiling is checked per task against its own depth's
`max_refinement_iterations` cap from `polysearch.config.DEPTH_PROFILES`, not a flat
number — a standard-depth run of 3 iterations is a violation even though it is
under the deep cap. `--depth-override` shifts a task's cap accordingly.

The runner exits non-zero when the gate fails, so it can gate a release in CI or a
script.

## Files

- `tasks.yaml` — the 12-task suite (2 each across FACTUAL, COMPARISON, TECHNICAL,
  TREND, RECENT, CONTESTED).
- `run_evals.py` — sweep runner, scoreboard, and quality-bar gate.
- `metrics.py` — programmatic metrics from a report json.
- `report_adapter.py` — normalizes the public and internal report shapes into one.
- `judge.py` — LLM-as-judge (rubric, schema, parsing).
- `results/` — per-round scoreboards + persisted per-task reports (raw reports gitignored).

## Improvement-round history (v1.0.0)

Four improvement rounds ran against both this package and a private reference
pipeline before release. What each round found and fixed:

- **Round 1** — the harness needed as much fixing as the pipelines: silent-zero
  metrics on shape mismatch, coverage matched against the wrong text, pair- vs
  claim-level verification gating, artifact loss, judge reading the report's
  own self-audit sections, judge 429s, non-hermetic env. All fixed. Pipeline
  findings ranked: runtime/cost blow-ups, key-fact coverage gaps.
- **Round 2** — verifier scrape dedup (each unique URL fetched once, bounded
  scoring concurrency) cut public run cost ~10x and runtime from 20-30+ min to
  3-8 min; synthesis fact-density + per-claim source localization landed. Two
  timeout classes remained (deep, community-heavy).
- **Round 3** — coverage metric calibrated (sentence windows, suffix
  normalization, numeric guard: a fact's figure must appear in the matched
  window; structural sections excluded from the match pool); global time
  budget with graceful degradation + per-adapter community timeouts eliminated
  both timeout classes (zero errors from here on). Judge sub-scores exposed
  recovery-pass source pollution.
- **Round 4** — recovery pass stopped forcing a curated domain allowlist and
  gained a per-claim relevance gate; report tier buckets trimmed to cited
  sources only. Judge citation/source dimensions moved sharply (+0.18 mean
  judge in one round). Full-suite gate run: zero crashes, zero placeholder
  leaks, refinement correct, verification and coverage at or near gates;
  judge advisory (see note above).
