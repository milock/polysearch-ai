# Install tiers — what each key unlocks, and what it costs

polysearch is modular. You do not wire everything up front. Each API key you add
turns on another layer; a run uses whatever is present and skips the rest with a
null provider (and a note in the report). Start with one key and grow.

## The tiers

### Tier 0 — one key

```bash
echo "PERPLEXITY_API_KEY=your_key_here" > .env
polysearch --topic "What is the current US federal funds rate?" --depth quick
```

- **Unlocks:** decomposed sub-question research via Perplexity.
- **Skipped:** web grounding, citation verification, vector context, community signal.
- **You get:** a report with Perplexity's cited answers, tier-classified but not verified against the source pages.
- **Cost:** ~$0.10–$0.50 per query.

### Tier 1 — add web grounding and a synthesizer

```
PERPLEXITY_API_KEY=...
FIRECRAWL_API_KEY=...
OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY
```

- **Unlocks:** Firecrawl web grounding, synthesis, and citation verification — every cited claim is checked against the page it came from.
- **You get:** the full three-layer report: a synthesized answer plus a verification pass that flags dead links, quote mismatches, and number mismatches.
- **Cost:** ~$0.50–$2.00 per query, plus verification scrapes (cap with `--verify-budget`).

### Tier 2 — add vector search over your own corpus

```
QDRANT_URL=...
QDRANT_API_KEY=...
```

- **Unlocks:** vector search over a corpus you have indexed, blended into the research as SME-tier context.
- **You get:** your own documents weighted alongside public sources, so internal knowledge shows up in the report.
- **Cost:** ~$0.50–$5.00 per query, plus your Qdrant hosting.

### Tier 3 — add community signal

- **Unlocks:** the community layer — Reddit, Hacker News, and similar, from the last 30 days.
- **You get:** a read on what people are actually saying, classified as COMMUNITY tier. Treat it as an engagement signal, not proof.
- **Cost:** Tier 2 plus the community layer's own API costs.

## Cost by depth

Rough per-query cost, by tier and depth. Actual spend depends on how much
citation verification scrapes; `--verify-budget` caps it.

| Tier | Quick | Standard | Deep |
|---|---|---|---|
| 0 (Perplexity only) | ~$0.10 | ~$0.30 | ~$1.00 |
| 1 (+Firecrawl, synthesis) | ~$0.30 | ~$0.80 | ~$2.50 |
| 2 (+vector search) | ~$0.30 | ~$0.85 | ~$2.60 |
| 3 (+community signal) | ~$0.40 | ~$1.10 | ~$3.50 |

The deep-research layer (on automatically at `--depth deep`, or forced with
`--deep-research`) is the biggest single cost. It bills on four meters —
input/output tokens, citation tokens, reasoning tokens, and search queries — so
a deep run costs several times a standard one. Leave it off for routine work.

## Checking what is active

```bash
polysearch --diagnose
```

prints which keys are present, which layers will run, and where reports are
written. Run it any time a report looks thinner than you expected — a missing
key is the usual reason a layer did not appear.
