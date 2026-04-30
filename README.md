# polysearch

> Modular, multi-source research pipeline with citation tier classification. A drop-in Claude Code skill plus a pip-installable Python CLI.

[![PyPI version](https://img.shields.io/pypi/v/polysearch.svg)](https://pypi.org/project/polysearch/)
[![Python](https://img.shields.io/pypi/pyversions/polysearch.svg)](https://pypi.org/project/polysearch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/milock/polysearch/actions/workflows/tests.yml/badge.svg)](https://github.com/milock/polysearch/actions/workflows/tests.yml)

> **Status: v0.1.0 — alpha. The CLI flags and Python API may change before v1.0.** Pin `polysearch==0.1.0` in production.

polysearch runs a single research topic through up to four parallel layers — vector search over a personal corpus, decomposed sub-question research (Perplexity), web grounding (Firecrawl), and community signal (last30days). It synthesizes the results with claims-level extraction, verifies the cited sources against scraped pages, and writes a tiered, auditable report.

It's modular by design: pick which providers you want, pay only for those, and swap implementations behind the same protocol. A user with only a Perplexity key gets a thin pipeline; a user with the full stack gets all four layers blended.

---

## Quickstart (Tier 0 — one API key, one minute)

```bash
pip install polysearch
echo "PERPLEXITY_API_KEY=your_key_here" > .env
polysearch --topic "What is the current US federal funds rate?" --depth quick
```

Result: a markdown report in `./reports/` with citations from Perplexity. Cost: ~$0.10–$0.50.

To get the same thing as a Claude Code skill:

```bash
git clone https://github.com/milock/polysearch.git
cd polysearch && ./install.sh
# Then in Claude Code: "research the current federal funds rate"
```

---

## Architecture

```
                  ┌─────────────────────┐
                  │    polysearch CLI   │
                  └──────────┬──────────┘
                             │
                  ┌──────────┴──────────┐
                  │     Orchestrator    │
                  └──────────┬──────────┘
                             │
        ┌─────────┬──────────┼──────────┬─────────┐
        ▼         ▼          ▼          ▼         ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
   │ Vector │ │  Web   │ │ Decomp │ │  Comm. │ │  Source  │
   │ Search │ │ Ground │ │  Sub-Q │ │ Signal │ │  Tiering │
   │(Qdrant)│ │(Firecr)│ │(Perplx)│ │(L30D)  │ │(YAML map)│
   └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬─────┘
        └──────────┴──────────┴──────────┘          │
                          │                         │
                  ┌───────┴────────┐                │
                  │   Synthesis    │◄───────────────┘
                  │ (OpenAI/Claude)│
                  └────────┬───────┘
                           │
                  ┌────────┴───────┐
                  │   Citation     │
                  │  Verification  │
                  └────────┬───────┘
                           │
                  ┌────────┴───────┐
                  │ Report Writer  │
                  │  (md + json)   │
                  └────────────────┘
```

Each layer has a Protocol (`ResearchProvider`, `VectorStore`, `WebGrounder`, `Synthesizer`, `CitationVerifier`). Missing credentials substitute null implementations rather than failing — tier downgrade is non-fatal.

---

## Install tiers

| Tier | What you need | What you get | Cost per query |
|---|---|---|---|
| **0** | `PERPLEXITY_API_KEY` | Decomposed sub-question research, no web grounding, no verification | ~$0.10–$0.50 |
| **1** | Tier 0 + `FIRECRAWL_API_KEY` + (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) | 3 layers, synthesis, citation verification | ~$0.50–$2.00 |
| **2** | Tier 1 + `QDRANT_URL` + `QDRANT_API_KEY` + a personal corpus | Personal context blended into research | ~$0.50–$5.00 + Qdrant |
| **3** | Tier 2 + `last30days-skill` installed | Community signal layer (Reddit/X/HN/Polymarket) | + last30days API costs |

If a tier's credentials aren't present, that layer gets a null provider and the pipeline runs with what it has. You don't have to wire everything to start.

---

## What's in the box

| Path | What it is |
|---|---|
| [`src/polysearch/`](src/polysearch/) | The Python package. Provider abstractions, orchestrator, CLI. |
| [`src/polysearch/providers/`](src/polysearch/providers/) | Concrete implementations of each provider protocol. |
| [`config/`](config/) | Default `domain_tiers.yaml` (gov, academic, major news) plus annotated example. |
| [`skills/research/`](skills/research/) | Drop-in for Claude Code at `~/.claude/skills/research/`. |
| [`agents/`](agents/) | Thin agent template — drop in if you want a slash-command-driven research agent. |
| [`examples/`](examples/) | Quickstart scripts, tier setup walkthroughs, sample outputs. |
| [`tests/`](tests/) | Unit, integration (mocked providers), and opt-in live tests. |
| [`docs/`](docs/) | Architecture, providers, modes, citation tiers, cost modeling, migration. |
| [`install.sh`](install.sh) | One-line installer for the Claude Code skill. |

---

## CLI reference

```
polysearch --topic "..." [options]
```

| Flag | Default | Notes |
|---|---|---|
| `--topic` | required | Research subject |
| `--depth` | `standard` | `quick` (45–60s) / `standard` (90–180s) / `deep` (3–5m) |
| `--output-dir` | `./reports/` | Markdown + JSON written here |
| `--providers` | auto | Comma-list to override: `perplexity,firecrawl,qdrant,community` |
| `--synthesizer` | auto | `openai` / `anthropic` / `none` |
| `--verify-budget` | `5.00` | Max USD for citation verification scrapes |
| `--no-verify` | off | Skip citation verification entirely |

Auto-resolution: if both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are set, OpenAI wins (cheaper). If only one is set, that one is used. If neither, synthesis is skipped and the report contains raw layer outputs with a note.

---

## Cost estimates

Per query, by tier:

| Tier | Quick | Standard | Deep |
|---|---|---|---|
| 0 (Perplexity only) | ~$0.10 | ~$0.30 | ~$1.00 |
| 1 (+Firecrawl, synthesis) | ~$0.30 | ~$0.80 | ~$2.50 |
| 2 (+Qdrant) | ~$0.30 | ~$0.85 | ~$2.60 |
| 3 (+community signal) | ~$0.40 | ~$1.10 | ~$3.50 |

Citation verification adds ~$0.50–$5.00 depending on scrape volume. Use `--verify-budget` to cap.

---

## How polysearch differs from existing tools

This is not OpenAI's deep research, Perplexity-as-a-service, or gpt-researcher. The closest analogies and how polysearch differs:

- **vs. gpt-researcher / autoresearch frameworks:** polysearch ships a fixed, opinionated 4-layer architecture rather than an agentic loop. No tool-calling chain to debug; the orchestrator runs all layers in parallel and synthesis is one pass with a self-audit. More predictable, less flexible.
- **vs. Perplexity directly:** Perplexity is one of polysearch's layers. Polysearch adds web grounding, citation verification against scraped pages, optional vector context, and a tiered report format. Use Perplexity directly if you want the conversational interface; use polysearch if you want a self-hosted, modular, batch-friendly pipeline.
- **vs. OpenAI's deep research API:** when that API is publicly available, polysearch will likely add it as an alternate `ResearchProvider`. Until then, Perplexity Sonar is the equivalent layer.
- **vs. AI detector-evasion / paraphrase tools:** unrelated. polysearch is for original research with verifiable citations, not for laundering AI-generated text.

---

## Roadmap

- More provider implementations: Tavily, Brave Search, Weaviate, Pinecone, Chroma
- OpenAI deep research API as an alternate `ResearchProvider` (when GA)
- Configurable depth profiles (cost caps per phase)
- Native Slack / Discord bot wrappers
- Long-form report mode (multi-section briefings)

PRs welcome. The provider protocols (`src/polysearch/providers/base.py`) are the easiest place to contribute a new implementation.

---

## License & credits

MIT — see [`LICENSE`](LICENSE).

This pipeline composes work from several upstreams. Full credits in [`ATTRIBUTION.md`](ATTRIBUTION.md).
