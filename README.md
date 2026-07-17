# polysearch

> Multi-source research pipeline with citation tier classification and self-verifying output. A pip-installable Python CLI plus a drop-in Claude Code plugin.

[![PyPI version](https://img.shields.io/pypi/v/polysearch-ai.svg?cacheSeconds=3600)](https://pypi.org/project/polysearch-ai/)
[![Python](https://img.shields.io/pypi/pyversions/polysearch-ai.svg?cacheSeconds=3600)](https://pypi.org/project/polysearch-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/milock/polysearch-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/milock/polysearch-ai/actions/workflows/tests.yml)

polysearch runs one research topic through several web-backed layers in parallel, synthesizes the results into a cited answer, then checks every citation against the page it came from. What you get back is a markdown report where each source carries an authority tier and each claim has been fetched and matched, not just asserted.

The package installs as `polysearch-ai`; the import name and CLI are both `polysearch`.

```bash
pip install polysearch-ai
```

It is modular by credential. A user with only a Perplexity key gets sub-question research and a cited report. Add a Firecrawl key and a synthesis model and the pipeline grounds against live pages and verifies citations. Add source connectors and it folds in community signal. Every layer is gated on its own credential: a missing key swaps in a null provider and the run continues with what it has, rather than failing.

---

## Quickstart

One key, about a minute:

```bash
pip install polysearch-ai
echo "PERPLEXITY_API_KEY=your_key_here" > .env
polysearch --topic "What is the current US federal funds rate?" --depth quick
```

You get a markdown report (and a sibling JSON) in `./reports/`, with citations from Perplexity. A quick run costs roughly $0.10 to $0.50.

The same capability as a Claude Code plugin:

```
/plugin marketplace add milock/polysearch-ai
/plugin install polysearch
```

Then research from any prompt, or call `/polysearch:research` directly. See [`docs/agent-integration.md`](docs/agent-integration.md) for the plugin, skill-copy, generic-harness, and Python-API paths.

---

## How it works

polysearch runs a first pass across up to four research layers in parallel, synthesizes it, verifies it, and then refines it until the answer covers the question or a guard stops the loop.

```
                       ┌──────────────────────┐
                       │    polysearch CLI    │
                       └──────────┬───────────┘
                                  │
                       ┌──────────┴───────────┐
                       │     Orchestrator     │
                       └──────────┬───────────┘
                                  │
      ┌───────────────┬───────────┼───────────────┬──────────────┐
      ▼               ▼           ▼               ▼              ▼
┌───────────┐  ┌────────────┐ ┌──────────┐  ┌────────────┐ ┌──────────┐
│  Research  │  │    Web     │ │Community │  │   Deep     │ │  Source  │
│  sub-Qs    │  │ grounding  │ │ signal   │  │  research  │ │  tiering │
│(Perplexity)│  │(Firecrawl) │ │(native)  │  │ (opt-in)   │ │(YAML map)│
└─────┬──────┘  └─────┬──────┘ └────┬─────┘  └─────┬──────┘ └────┬─────┘
      └───────────────┴────────────┴──────────────┘              │
                              │                                  │
                     ┌────────┴─────────┐                        │
                     │    Synthesis     │◄───────────────────────┘
                     │ (OpenAI/Claude)  │
                     └────────┬─────────┘
                              │
                     ┌────────┴─────────┐
                     │ Claim extraction │
                     │  + Verification  │   fetch each cited page,
                     │                  │   match quotes & numbers
                     └────────┬─────────┘
                              │
                     ┌────────┴─────────┐
                     │  Recovery pass   │   re-source weak citations
                     └────────┬─────────┘
                              │
                     ┌────────┴─────────┐
                     │ Refinement loop  │──┐  goal not met? emit new
                     └────────┬─────────┘  │  queries, grow the corpus,
                              │◄───────────┘  re-synthesize, re-verify
                     ┌────────┴─────────┐
                     │  Report writer   │
                     │   (md + json)    │
                     └──────────────────┘
```

**The four first-pass research layers:**

1. **Research (Perplexity).** Decomposes the topic into sub-questions and answers each with citation-aware Sonar results.
2. **Web grounding (Firecrawl).** Searches the live web, scrapes the top hits, and mines structured facts from HIGH-tier pages.
3. **Community signal (native).** Reddit, Hacker News, Bluesky, GitHub, X, and YouTube adapters, fused and relevance-gated. Classified COMMUNITY tier: sentiment, not proof.
4. **Deep research (opt-in).** Perplexity `sonar-deep-research` for high-stakes topics. On automatically at `--depth deep`, or forced with `--deep-research`.

Everything downstream is credential-gated and failure-isolated. One layer raising becomes a note in the report; it never sinks the run.

For the full flow, the refinement-loop design, and the verification-status vocabulary, see [`docs/architecture.md`](docs/architecture.md).

---

## What makes it different

- **Refinement loop.** After the first pass, a rubric-based evaluator judges coverage against the topic. If the answer falls short, it emits angle-diverse follow-up queries, runs them, verifies the new material, and re-synthesizes. The loop is bounded by an iteration cap, a cost ceiling, and dry-exit guards, so it grows the answer without running away.
- **Citations verified even for deep research.** Deep-research and Perplexity narrative answers are mined for claims and checked against their sources like everything else. A figure in a deep-research paragraph still has to survive a fetch-and-match.
- **Recovery pass.** When the first verification comes back weak, polysearch runs scoped re-sourcing queries to find better citations before it gives up on a claim.
- **Cross-process rate limiting.** A shared on-disk ledger coordinates API rate limits across concurrent runs, so a 429 in one process backs off its siblings instead of hammering the same endpoint.
- **Blocked-source exclusion.** Hosts on the hard-blocked list (spam, content farms) are dropped before scraping. If one still resolves, it lands in an "Excluded (blocked sources)" section rather than counting as a citation.
- **Authority tiers on every source.** HIGH / MEDIUM / LOW / COMMUNITY / SME / UNKNOWN, from a bundled domain map you can override. The report weights sources so you do not treat a forum post like a government filing.

---

## Install tiers

Each key turns on another layer. A run uses whatever is present and skips the rest. Start with one key and grow. These tiers match `resolve_install_tier` and [`.env.example`](.env.example); [`examples/tiers.md`](examples/tiers.md) walks through each.

| Tier | What you add | What you get | Cost per query |
|---|---|---|---|
| **0** | `PERPLEXITY_API_KEY` | Decomposed sub-question research, returned as a cited report | ~$0.10–$0.50 |
| **1** | Firecrawl key + a synthesis model (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) | Web grounding, cross-source synthesis, and citation verification | ~$0.50–$2.00 |
| **2** | One or more source connectors (`SCRAPECREATORS_API_KEY`, `YOUTUBE_API_KEY`, `GITHUB_TOKEN`, `REDDIT_CLIENT_ID`/`_SECRET`) | Higher community coverage and rate limits, plus X and YouTube | Tier 1 + each connector's own cost |

The community layer's keyless sources (Reddit, Hacker News, Bluesky, GitHub) run at any tier without a key; connectors raise their rate limits and add the key-gated sources. The X adapter needs `SCRAPECREATORS_API_KEY` plus a watch list of handles set via `POLYSEARCH_X_HANDLES` (that is configuration, not a credential). Run `polysearch --diagnose` to see which layers are active for your current keys.

---

## CLI reference

```
polysearch --topic "..." [options]
```

| Flag | Default | Notes |
|---|---|---|
| `--topic` | required | Research subject (also required by `--classify`) |
| `--depth` | `standard` | `quick` / `standard` / `deep` |
| `--output-dir` | `./reports/` | Where the markdown + JSON are written |
| `--providers` | all | Comma-list of first-pass layers to run: `research,grounding,community,deep_research,linkedin` |
| `--synthesizer` | auto | Force `openai` or `anthropic` (default: auto by available key) |
| `--verify-budget` | per depth | Override the first-pass verification budget, in USD |
| `--no-verify` | off | Skip citation verification |
| `--no-recovery` | off | Skip the recovery pass after weak verification |
| `--max-iterations` | per depth | Override the refinement iteration cap (`0` disables the loop) |
| `--deep-research` | off | Force the deep-research layer at any depth (needs a Perplexity key) |
| `--classify` | off | Print the classifier verdict as JSON and exit (no network) |
| `--diagnose` | off | Print credential / tier / layer / output-dir status and exit (no network) |
| `--synthesize-parallel GLOB` | — | Cross-synthesize existing report files matching GLOB into one rollup |

Synthesizer auto-resolution: if both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are set, OpenAI is used (cheaper). If only one is set, that one is used. If neither, synthesis is skipped and the report carries the raw layer outputs with a note.

---

## Cost by depth

Rough per-query cost. Actual spend depends on how much verification scrapes; cap it with `--verify-budget`.

| Tier | Quick | Standard | Deep |
|---|---|---|---|
| 0 (Perplexity only) | ~$0.10 | ~$0.30 | ~$1.00 |
| 1 (+Firecrawl, synthesis) | ~$0.30 | ~$0.80 | ~$2.50 |
| 2 (+community connectors) | ~$0.40 | ~$1.10 | ~$3.50 |

The deep-research layer is the largest single cost. It bills on input, output, citation, and reasoning tokens plus search queries, so a deep run costs several times a standard one. Leave it off for routine work.

---

## Python API

The one public entry point is `run_research`, an async coroutine that returns a structured report and (by default) writes the markdown and JSON.

```python
import asyncio
from polysearch import run_research

async def main():
    report = await run_research(
        "What is the current US federal funds rate?",
        depth="quick",
    )
    print(report.synthesis_md)
    print(f"cost: ${report.totals.get('cost_usd', 0.0):.4f}")

asyncio.run(main())
```

Every layer sits behind a Protocol, so you can inject a bundle of mock or null providers to run fully offline. See [`docs/providers.md`](docs/providers.md) for the protocols and how to add your own, and [`docs/agent-integration.md`](docs/agent-integration.md) for the full keyword-argument list.

---

## What's in the box

| Path | What it is |
|---|---|
| [`src/polysearch/`](src/polysearch/) | The Python package: provider protocols, orchestrator, CLI. |
| [`src/polysearch/providers/`](src/polysearch/providers/) | Concrete providers behind each protocol (Perplexity, Firecrawl, OpenAI/Anthropic, deep research, LinkedIn). |
| [`src/polysearch/community/`](src/polysearch/community/) | Native community adapters, fusion, and the relevance gate. |
| [`src/polysearch/data/`](src/polysearch/data/) | Bundled `domain_tiers.yaml` and authoritative-source schemas, shipped inside the wheel. Override with `POLYSEARCH_DOMAIN_TIERS` / `POLYSEARCH_SCHEMA_DIR`. |
| [`skills/research/`](skills/research/) | The Claude Code skill. |
| [`agents/`](agents/) | A thin research-agent template. |
| [`examples/`](examples/) | Quickstart script, tier walkthrough, sample output. |
| [`tests/`](tests/) | Unit, integration (mocked providers), and opt-in live tests. |
| [`docs/`](docs/) | Architecture, providers, configuration, and agent integration. |
| [`install.sh`](install.sh) | One-line installer for the Claude Code skill. |

---

## A note on LinkedIn and X

The LinkedIn enricher and the X adapter reach those platforms through ScrapeCreators. Automated collection can run against LinkedIn's and X's Terms of Service. Both layers are opt-in and off unless you set `SCRAPECREATORS_API_KEY` (and, for X, `POLYSEARCH_X_HANDLES`). You are responsible for using them within the platforms' terms and applicable law.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md): the layer flow, the refinement loop, and verification statuses.
- [`docs/providers.md`](docs/providers.md): the provider protocols and how to add a provider or a community adapter.
- [`docs/configuration.md`](docs/configuration.md): every setting and environment variable.
- [`docs/agent-integration.md`](docs/agent-integration.md): wiring polysearch into Claude Code or any agent harness.

---

## Contributing

The provider protocols in [`src/polysearch/providers/base.py`](src/polysearch/providers/base.py) are the easiest place to contribute a new implementation. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License and credits

MIT, see [`LICENSE`](LICENSE). polysearch composes several upstream services and one adapted open-source library; full credits are in [`ATTRIBUTION.md`](ATTRIBUTION.md).
