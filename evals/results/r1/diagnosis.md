# Round 1 Diagnosis (2026-07-15)

Round 1's headline: the harness needed as much fixing as the pipelines. All harness
defects are FIXED and committed; the pipeline findings below are ranked for Round 2.

## Harness defects found and fixed this round

1. Silent-zero metrics on internal reports (shape mismatch) → report_adapter.py, None + warning, None-safe gate (35c67f3).
2. Key-fact coverage matched the wrong text source → collected-md matching, sentence-localized token_set_ratio ≥70 (35c67f3).
3. Gated verification was pair-level (noisy) → claim-level gated, pair-level demoted to ungated column (35c67f3).
4. 1800s task timeout killed legitimate deep runs → 2700s default + POLYSEARCH_EVAL_TIMEOUT_SEC (35c67f3).
5. Per-task artifacts lived in a throwaway tmpdir → persisted under results/<label>/<target>/<task_id>/ (35c67f3).
6. Judge read the pipeline's own audit sections and punished honest self-reporting → end-state-only judge input (de8ea09).
7. Judge 429s errored tasks → tenacity retry + pacing (1eda405, de8ea09).
8. Suite not hermetic against a developer .env → conftest dotenv neutralization (6a1c46b).
9. No score-only mode → evals/rescore.py (5407812).
10. Console-script PATH assumption in the sweep runner (operational, documented in README usage).

## Real pipeline findings (rescored data), ranked

| Metric | Public (8/12) | Internal generic (12/12) | Internal landmine (5/5) | Gate |
|---|---|---|---|---|
| Judge overall | 0.275 | 0.415 | 0.412 | ≥0.80 |
| Claim-level verification | 0.797 | 0.877 | 0.804 | ≥0.70 |
| Key-fact coverage | 0.292 | 0.528 | 0.370 | ≥0.85 |
| Cost | $39.10 | $32.83 | $17.42 | — |

1. **P1 — Public runtime/timeout (blocker-class):** 4 of 12 public tasks exceeded the
   (old) 30-min timeout and produced nothing; completed public runs took 20-30+ min and
   $3.05-7.61 each — roughly 2x internal cost and far slower. Needs profiling: suspects
   are verification budget consumption at standard depth, refinement re-verify loops,
   scrape-chain retry stacking, and rate-limiter over-pacing. Internal runs the same
   logical flow in 2-14 min.
2. **P2 — Key-fact coverage is the main quality gap (all targets):** reports verify what
   they cite (0.80-0.88 claim-level) but omit facts a good report must state — public
   0.29, internal 0.53, landmines 0.37 (the trap-avoiding facts are exactly what's
   missing). Levers: synthesis prompt (concrete facts/numbers per section), refinement
   evaluator rubric weighting factual completeness, snippet budget into synthesis.
3. **P3 — Judge overall far below gate (0.28-0.42):** dominated by completeness and
   factual-accuracy dimensions (coverage-linked), plus one confirmed synthesis
   self-inconsistency ("cut twice" vs three listed cuts, internal fed-rate). Fixing P2
   moves this; synthesis coherence checks would help the remainder.
4. **P4 — RECENT category weakest everywhere (0.26-0.28):** community-layer yield is
   thin. Known contributor: Reddit runs keyless (no REDDIT_CLIENT_ID in env) and public
   Reddit/Bluesky 403 intermittently (A8 finding). OAuth creds + fusion of what does
   arrive are the levers.
5. **P5 — Cost per run:** internal avg $2.9/run, public avg $4.9/run — both above the
   original $0.5-1.5 estimate. Round budget math must use measured numbers; caps
   (--verify-budget, refinement ceilings) are the tuning knobs.

## Round 1 spend (actuals)

Internal generic $32.83 + internal landmines $17.42 + public $39.10 + judge/rescores <$1
= **~$90 total** (estimate was $15-30; the miss is P5 + running the full 12x2+5 matrix).

## Round 2 proposal (pending approval)

Subset matrix: 4 generic tasks (one per weak category: comparison, technical, recent,
contested) x both targets + 2 landmine tasks internal-only, with --verify-budget 1.5 and
tightened max-cost caps ≈ **$25-30 estimated**, measured against these Round 1 baselines.
Run after P1/P2 fixes land.
