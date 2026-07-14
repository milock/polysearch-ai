<!--
  ⚠️  SYNTHETIC SAMPLE — NOT REAL RESEARCH  ⚠️
  Every value below is fabricated by mock providers, not a live pipeline run.
  It exists to show the shape of a polysearch report: the section order, the
  source-tier buckets, the Refinement Trace, and the Pipeline Stats.

  Tells that this is synthetic, not a real report:
    - all sources resolve to `example.test` (a reserved, non-routable domain)
    - the recurring "42%" figure is a fixed placeholder from the mock
    - costs and durations are near-zero because nothing hit the network

  Generated offline from the mocked orchestrator path in
  tests/integration/test_orchestrator.py. A real run writes a file like this to
  ./reports/ with live sources, real figures, and verified citations.
-->

# Research: grid-scale battery storage economics in 2026

Generated: 2026-07-14T08:30:49 | Depth: standard | polysearch v0.1.0 | Cost: $0.05 | Duration: 0.0s

## Synthesis

The key figure is 42% as of 2026.

## Pipeline Decisions
- **Topic type:** THEMATIC
- **Time-sensitive:** False
- **Domain-related:** False
- **Depth ran:** standard
- **Depth suggested by classifier:** standard
- **Reasons:** no specific entity trigger -> THEMATIC; default -> standard

## Sources by Quality Tier

### Unknown (not in whitelist) (2)
- [A report](https://example.test/1-grid-sca) _[2026-01-01]_
- [A report](https://example.test/2-battery ) _[2026-01-01]_

## Refinement Trace
Iterations: 2 | Follow-up rounds run: 1 | Refinement cost: $0.0300

### Iteration 1 — coverage 0.50
- **Verdict:** coverage is partial
- **Follow-up queries run:**
  - battery storage cost trend 2026
- **New sources:** 1 · **New claims:** 1 · **Cost:** $0.0300

### Iteration 2 — coverage 0.50
- **Verdict:** coverage is partial
- **New sources:** 0 · **New claims:** 0 · **Cost:** $0.0000
- **Stopped:** no unique follow-up queries

## Pipeline Stats
- research: 1 sources · $0.010 · 0ms
- grounding: 0 sources · $0.000 · 0ms
- refinement-1: 1 sources · $0.010 · 0ms
- Verification: $0.020 · 2/2 claims supported (4/4 pairs OK)
- **Total:** $0.05 · 0.0s
