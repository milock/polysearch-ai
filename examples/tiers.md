# Install tiers — what each key unlocks, and what it costs

polysearch is modular. You do not wire everything up front. Each API key you add
turns on another layer; a run uses whatever is present and skips the rest with a
null provider (and a note in the report). Start with one key and grow.

There are three tiers, matching `.env.example` and the package's own
`resolve_install_tier`.

## The tiers

### Tier 0 — Perplexity only (the minimum to run)

```bash
echo "PERPLEXITY_API_KEY=your_key_here" > .env
polysearch --topic "What is the current US federal funds rate?" --depth quick
```

- **Unlocks:** decomposed sub-question research via Perplexity, returned as a cited report.
- **Skipped:** web grounding, citation verification, and the community connectors.
- **Cost:** ~$0.10–$0.50 per query.

### Tier 1 — add web grounding and a synthesizer

```
PERPLEXITY_API_KEY=...
FIRECRAWL_API_KEY=...
OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY (if both, OpenAI is preferred)
```

- **Unlocks:** Firecrawl web grounding, a dedicated cross-source synthesizer, and citation verification — every cited claim is checked against the page it came from.
- **You get:** the full three-layer report: a synthesized answer plus a verification pass that flags dead links, quote mismatches, and number mismatches. Recommended for most users.
- **Cost:** ~$0.50–$2.00 per query, plus verification scrapes (cap with `--verify-budget`).

### Tier 2 — add extra source connectors (all optional)

Tier 1 plus at least one of the community connectors. Each key lights up one more
source; set only the ones you want, and leave the rest blank.

```
SCRAPECREATORS_API_KEY=...     # social / creator content
YOUTUBE_API_KEY=...            # video + transcript search
GITHUB_TOKEN=...               # repo / code / issue search at a higher rate limit
REDDIT_CLIENT_ID=...           # community discussion search (with the secret below)
REDDIT_CLIENT_SECRET=...
POLYSEARCH_X_HANDLES=handle_one,handle_two   # recent posts from specific X handles
```

- **Unlocks:** the community layer — the sources above, classified as COMMUNITY tier.
- **You get:** a read on what people are actually saying. Treat it as an engagement signal, not proof.
- **Cost:** Tier 1 plus each connector's own API costs.

## Cost by depth

Rough per-query cost, by tier and depth. Actual spend depends on how much
citation verification scrapes; `--verify-budget` caps it.

| Tier | Quick | Standard | Deep |
|---|---|---|---|
| 0 (Perplexity only) | ~$0.10 | ~$0.30 | ~$1.00 |
| 1 (+Firecrawl, synthesis) | ~$0.30 | ~$0.80 | ~$2.50 |
| 2 (+community connectors) | ~$0.40 | ~$1.10 | ~$3.50 |

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
