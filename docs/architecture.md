# Architecture

polysearch is a pipeline, not an agent loop. One topic goes in; the orchestrator runs a fixed sequence of layers, and a cited, verified report comes out. The sequence is deterministic, which makes cost and behavior predictable and makes each stage easy to test in isolation.

This document covers the run flow, the goal-driven refinement loop, and the verification-status vocabulary. For the provider protocols and how to extend them, see [`providers.md`](providers.md); for settings, [`configuration.md`](configuration.md).

## The run

`run_research(topic, ...)` in [`src/polysearch/orchestrator.py`](../src/polysearch/orchestrator.py) is the single entry point. It runs these stages in order:

1. **Classify.** A rules-only pass (no network) tags the topic with a query type and time-sensitivity. PERSON topics get an extra enrichment stage later.
2. **First pass, in parallel.** Up to four research layers fire at once:
   - **research**: Perplexity decomposes the topic into sub-questions and answers each with citation-aware Sonar results.
   - **grounding**: Firecrawl searches the live web and scrapes the top hits.
   - **community**: the native adapters (Reddit, Hacker News, Bluesky, GitHub, X, YouTube) search in parallel, fuse into one ranked list, and pass through a relevance gate that suppresses the layer when most items are off-topic.
   - **deep_research**: Perplexity `sonar-deep-research`, only when the depth profile marks it eligible (`deep`) or `--deep-research` forces it.
3. **PERSON enrichment.** For PERSON topics, an optional person-context hook runs, and, when the topic carries a LinkedIn URL and the enricher is active, the LinkedIn layer fetches the profile.
4. **Synthesis.** The synthesizer (OpenAI or Anthropic) fuses the layer outputs into a cited markdown answer.
5. **Authoritative extraction.** On the top HIGH-tier scraped pages, structured extractors pull named facts (dates, figures) from the full page markdown. How many pages are mined is set per depth (quick 0, standard 2, deep 4).
6. **Claim extraction.** Verifiable claims are pulled from the synthesis, from each research layer's narrative answers, and from the authoritative facts. Extracting from the narrative answers is what lets a figure buried in a deep-research paragraph reach verification.
7. **Verification.** Each claim's cited URLs are fetched and checked (see below). The pass is bounded by a per-depth USD budget and a concurrency cap; sources are checked HIGH tier first, so the budget protects the citations that matter most.
8. **Recovery pass.** When verification comes back weak, scoped re-sourcing queries look for better citations, and the new material is verified and merged in. See [`../src/polysearch/verification/recovery.py`](../src/polysearch/verification/recovery.py).
9. **Refinement loop.** A coverage evaluator judges the corpus against the topic and, while the goal is unmet and the guards allow, grows the corpus and re-synthesizes (below).
10. **Report.** The orchestrator totals cost and duration and writes a markdown report plus a sibling JSON.

Every layer is failure-isolated. A layer that raises becomes a `pipeline_errors` entry and the run continues; a null provider (missing credential) short-circuits with its `reason` surfaced in the report's decision log. No single layer aborts the run.

### Parallelism and rate limits

The first-pass layers run concurrently, and the community adapters run concurrently within their layer. Because several layers (and several concurrent `polysearch` processes) can hit the same API, rate limiting is coordinated through a shared on-disk ledger in [`src/polysearch/ratelimit.py`](../src/polysearch/ratelimit.py). When one caller gets a 429, it records the `Retry-After` so siblings back off the same endpoint instead of retrying into the same wall.

## The refinement loop

The first pass answers the question once. The refinement loop asks whether that answer actually covers the topic, and keeps working if it does not. It lives in [`src/polysearch/refinement.py`](../src/polysearch/refinement.py).

Each iteration:

1. A coverage evaluator (an OpenAI structured-output call) scores the current corpus against the topic and returns a verdict: whether the goal is met, a coverage score, the gaps it sees, and angle-diverse follow-up queries.
2. If the goal is met, the loop stops.
3. Otherwise the follow-up queries run through the research and web-grounding providers. The new sources are verified, recovery optionally re-runs, and the synthesizer re-runs over the grown corpus.
4. The loop repeats.

Five guards bound it, because a runaway loop is the worst failure mode here:

- **goal met**: the evaluator is satisfied.
- **iteration cap**: from the depth profile (`quick` 0, `standard` 2, `deep` 4), overridable with `--max-iterations`.
- **cost ceiling**: a cumulative USD cap from the profile, overridable in settings.
- **no new queries**: the evaluator only rephrased queries already run.
- **no new sources**: a URL-canonical dedupe found nothing the corpus had not already seen.

The evaluator is OpenAI-based. With only an Anthropic key configured there is no evaluator, so refinement degrades to a no-op rather than failing. Each iteration writes a `RefinementTrace` (verdict, queries run, new sources, new claims, cost, stop reason) that renders into the report's "Refinement Trace" section.

## Verification statuses

Verification fetches each cited URL and scores the (claim, URL) pair. The scoring routes through the resilient fetch chain in [`src/polysearch/sources/scrape.py`](../src/polysearch/sources/scrape.py) (Firecrawl, then httpx, then Playwright), so a page that is merely blocked is not confused with a page that is gone.

| Status | Meaning |
|---|---|
| `OK` | The page was fetched and the claim's quotes and numbers matched. |
| `URL_DEAD` | Origin-confirmed dead: 404/410 or DNS failure. Excluded from the source buckets and listed under "Excluded (dead links)". |
| `FETCH_BLOCKED` | The page could not be retrieved by any fetch method (403/429/5xx, bot challenge, Firecrawl credit exhaustion). Blocked but likely alive, so it stays in the buckets with a note; it is never scored `URL_DEAD`. |
| `BLOCKED_SOURCE` | The host is categorically un-fetchable (login-walled, robots-blocked) or on the hard-blocked list. Diverted into "Excluded (blocked sources)", kept separate from dead links. |
| `PAYWALLED` | A paywall marker was detected in the page body. |
| `QUOTE_NOT_FOUND` | The page loaded but a claimed quote did not match, even with the fuzzy and embedding fallbacks. |
| `NUMBER_MISMATCH` | The page loaded but a claimed number did not match after magnitude and formatting normalization. |
| `UNDATED` | No publication date could be found in metadata or the page head. |
| `SKIPPED_BUDGET` | The verification budget was spent before this citation was reached. |

Status precedence, worst first, is `URL_DEAD > FETCH_BLOCKED > PAYWALLED > QUOTE_NOT_FOUND > NUMBER_MISMATCH > UNDATED > OK`. A claim rolls up to supported when each of its numbers and quotes matches in at least one cited source, so a single dead mirror does not sink a claim that other sources back.

Quote matching uses `rapidfuzz` partial-ratio, with an OpenAI-embedding fallback for borderline scores to rescue paraphrases. Number matching normalizes for trailing zeros, commas, and magnitude suffixes, so `$1.2M` matches `$1.2 million`.

## Source tiers

Every source is classified against a bundled domain map ([`src/polysearch/data/domain_tiers.yaml`](../src/polysearch/data/domain_tiers.yaml)) into one of six tiers:

- **HIGH**: primary, peer-reviewed, or official (government, standards bodies, major journals).
- **MEDIUM**: reputable secondary and trade press.
- **LOW**: opinion, marketing, unverified. Vendor blogs land here even on otherwise-reputable domains.
- **COMMUNITY**: forum and social posts. An engagement signal, not proof.
- **SME**: sources you supply yourself through the Python API (a private provider bundle). Not wired to any key in the default install.
- **UNKNOWN**: not in the map. Treat with caution.

The report buckets sources by tier so the reader can weight them. The synthesis and any downstream summary should lead with what HIGH and MEDIUM sources support and label anything resting on COMMUNITY or UNKNOWN as unconfirmed.

## Output

The report writer ([`src/polysearch/output/report.py`](../src/polysearch/output/report.py)) renders a `PipelineReport` to markdown plus a sibling JSON, written atomically under `YYYY-MM-DD-<slug>` names. The markdown carries the synthesis, the pipeline decisions, sources bucketed by tier (with dead and blocked links pulled into their own excluded lists), the refinement trace when the loop ran, any pipeline errors, and a cost/stats block. A placeholder guard refuses to save a report still holding unresolved template tokens unless placeholders are explicitly allowed, which is the expected degraded mode when no synthesis model is configured.
