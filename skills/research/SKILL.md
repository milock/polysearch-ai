---
name: research
description: 'Run a research topic through the polysearch pipeline: parallel Perplexity sub-question research, Firecrawl web grounding, and community signal, with citation verification and source-tier classification. Produces a saved, auditable markdown report. Use when asked to research, investigate, fact-check, compare options, do a deep dive, or gather cited evidence on any topic.'
argument-hint: "[topic]"
---

# Research with polysearch

Research runs through one command: the `polysearch` CLI. It fans a topic out
across parallel layers, synthesizes the results, verifies the cited sources
against the pages they came from, classifies every source by authority tier,
and writes a markdown report (plus a sibling JSON) to `./reports/`.

The pipeline does the work. Your job is to run it, read the report, and relay
the findings with the tier caveats intact.

## Run it

```bash
polysearch --topic "<the question>"
```

That runs the default `standard` depth. The report path is printed at the end;
open it and read the **Synthesis** section first.

Before a real run, you can preview how the pipeline will treat a topic for free,
with no network calls:

```bash
polysearch --classify --topic "<the question>"
```

This prints a JSON verdict — query type, whether the topic is time-sensitive,
a suggested depth, and an estimated runtime and cost. Use it to decide depth
when you are unsure.

If a run comes back thin or empty, check credentials and layer availability:

```bash
polysearch --diagnose
```

It prints which API keys are present, which layers are active, and where reports
will be written. That is the first stop for "0 sources," a missing synthesis, or
a Firecrawl error.

## Depth

`--depth` trades cost and time for coverage. Pick it to match the stakes.

| Depth | Runtime | Sub-questions | Web scrapes | Refinement cap | Use for |
|---|---|---|---|---|---|
| `quick` | 45–60s | 2 | 3 | 0 iterations | A single factual question, a definition, a fast sanity check |
| `standard` (default) | 90–180s | 4 | 5 | 2 iterations | Most research — a topic with a few angles |
| `deep` | 3–5m | 6 | 10 | 4 iterations | High-stakes, multi-faceted questions where you need breadth and want the deep-research layer on |

```bash
polysearch --topic "<question>" --depth deep
```

Cost scales with depth and with how many layers you have keys for. A `quick`
run on one key is cents; a `deep` run with the full stack and heavy citation
verification can reach a few dollars. Cap verification spend with
`--verify-budget`.

## The layers, and turning them off

By default the pipeline runs every layer it has credentials for. Restrict the
first pass with `--providers`, a comma-separated allowlist:

- `research` — decomposed sub-question research (Perplexity)
- `grounding` — web grounding and scraping (Firecrawl)
- `community` — community signal from the last 30 days (Reddit, Hacker News, and similar)
- `deep_research` — the native deep-research layer (see below)
- `linkedin` — profile enrichment for person-focused topics

```bash
polysearch --topic "<question>" --providers research,grounding
```

Any layer whose credentials are missing is skipped with a null provider — the
run continues with whatever is available. A one-key install still produces a
report; it just has fewer layers.

## Deep research layer and its cost

`--deep-research` forces the native deep-research layer at any depth (it is on
automatically at `--depth deep`). It runs a long, agentic research call and is
the single most expensive part of a run. It bills on four separate meters —
input/output tokens, citation tokens, reasoning tokens, and search queries — so
one deep-research call can cost several times a normal sub-question. Check your
budget before forcing it on a cost-sensitive run; leave it off with
`--depth quick` or `--depth standard` when you do not need it.

## Synthesis backend

`--synthesizer` forces which model writes the synthesis: `openai` or
`anthropic`. Left unset, the pipeline auto-picks based on which key is present
(if both are set, it defaults to OpenAI). With neither key, synthesis is skipped
and the report carries the raw layer outputs with a note. The specific model is
configurable through environment variables; you rarely need to touch it.

## Verification and recovery

Every claim's citation is checked against the page it came from — dead links,
quote mismatches, and number mismatches are all flagged. Two knobs:

- `--verify-budget <usd>` caps how much the verification pass spends on scrapes.
- `--no-verify` skips verification entirely (faster, cheaper, unaudited — avoid for anything you will act on).
- `--no-recovery` skips the follow-up pass that tries to re-source claims when the first verification round comes back weak.

## The refinement loop

After the first pass, a coverage evaluator judges the report against the topic.
If coverage is short, it writes follow-up questions, re-runs the research and
grounding layers, re-verifies, and re-synthesizes — bounded by the depth's
iteration cap and cost ceiling. Override the cap with `--max-iterations <n>`;
`--max-iterations 0` disables refinement.

When refinement runs, the report gets a **Refinement Trace** section. Read it to
judge whether a thin report converged cleanly or got cut short. Each iteration
lists its coverage score, the evaluator's verdict, the follow-up queries it ran,
how many new sources it found, and a stop reason. A stop reason of "cost ceiling
reached" on an important report is a signal to re-run with `--max-iterations`
raised — not to treat the report as final. A stop reason of "goal met" or "no
new unique sources" means it converged on its own.

## Reading the report

The report opens with a header (topic, depth, cost, duration) and then these
sections, in order:

1. **Synthesis** — the written answer: summary and key findings. Read this first.
2. **Pipeline Decisions** — how the topic was classified and which depth ran.
3. **Sources by Quality Tier** — every source, bucketed by authority (see below), with dead and blocked links pulled into their own "Excluded" lists.
4. **Refinement Trace** — present only when the refinement loop ran.
5. **Pipeline Errors** — present only when a layer raised; a run can still succeed with one layer down.
6. **Pipeline Stats** — cost, tokens, and duration, layer by layer.

## Source authority tiers

Every source is classified against a domain map so you can weight it correctly.
Do not treat all citations as equal.

- **HIGH** — primary, peer-reviewed, or official (government, standards bodies, major journals).
- **MEDIUM** — reputable secondary and trade press.
- **LOW** — opinion, marketing, and unverified pages. Vendor blogs land here even on otherwise-reputable domains.
- **COMMUNITY** — forum and social posts. This is an engagement signal — attention, not proof. A claim resting only on COMMUNITY sources is directional, not established.
- **SME** — internal sources, when you supply your own source provider through the Python API (`providers=`). Not wired to any key in the default install.
- **UNKNOWN** — not in the domain map. Treat with caution and verify independently.

Sources on the hard-blocked list (spam, content farms) are dropped before
scraping and never appear as valid citations; if one still resolves it is
diverted into the "Excluded (blocked sources)" list rather than counted.

When you relay findings, lead with what HIGH and MEDIUM sources support, and
label anything that rests on COMMUNITY or UNKNOWN sources as unconfirmed.

## Cross-synthesizing several reports

If you have run several related reports on one topic and want a single rollup,
point `--synthesize-parallel` at a glob of the report files:

```bash
polysearch --synthesize-parallel "./reports/2026-07-13-vendor-*.md"
```

It reads each matching report and writes one synthesis that surfaces
cross-cutting tradeoffs, contradictions between sources, and gaps still open
across the set.

## Full flag reference

| Flag | Default | What it does |
|---|---|---|
| `--topic` | required for a real run | The research question |
| `--depth {quick,standard,deep}` | `standard` | Coverage vs. cost/time profile |
| `--output-dir` | `./reports/` | Where the report and JSON are written (or `POLYSEARCH_OUTPUT_DIR`) |
| `--providers` | all available | Comma-list of first-pass layers: `research,grounding,community,deep_research,linkedin` |
| `--synthesizer {openai,anthropic}` | auto | Force the synthesis backend |
| `--verify-budget <usd>` | per-depth default | Cap the citation-verification spend |
| `--no-verify` | off | Skip citation verification |
| `--no-recovery` | off | Skip the weak-verification recovery pass |
| `--max-iterations <n>` | per-depth default | Override the refinement cap (`0` disables it) |
| `--deep-research` | auto (on at `deep`) | Force the native deep-research layer |
| `--classify` | — | Print the classifier verdict as JSON and exit (no network) |
| `--diagnose` | — | Print credential, tier, and layer status and exit (no network) |
| `--synthesize-parallel <glob>` | — | Cross-synthesize existing reports matching the glob |
| `--version` | — | Print the version and exit |

## Rules

- **Every research task produces a saved report.** Point people at the file, not just a paraphrase.
- **Never invent a source or a URL.** The verifier catches fabricated citations; do not add your own on top.
- **Weight by tier.** HIGH and MEDIUM carry a claim; COMMUNITY and UNKNOWN do not on their own.
- **A missing layer is not a failure.** The pipeline runs with whatever keys are present and notes what it skipped.
- **Read the Refinement Trace before trusting a thin report.** A cost-ceiling stop means it was cut short, not that the topic is covered.
