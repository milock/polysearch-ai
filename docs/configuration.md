# Configuration

All runtime configuration is a single `Settings` dataclass built from the environment. There is no config file to manage and no `pydantic-settings` dependency: `Settings.from_env()` reads `.env` (via `python-dotenv`) plus the process environment, and everything downstream imports the resolved object. The source of truth is [`src/polysearch/config.py`](../src/polysearch/config.py).

Two naming conventions:

- **API keys use their conventional names** (`OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, and so on), so the keys you already have in your environment work unchanged.
- **Everything else uses a `POLYSEARCH_` prefix.**

An empty or whitespace-only value is treated as unset, so a blank line in `.env` falls back to the default. Booleans are true when set to `1`, `true`, `yes`, or `on` (case-insensitive); anything else is false. List values are comma-separated and trimmed.

## API keys

All optional. A missing key swaps the layer that needs it for a null provider; the run continues. See [`.env.example`](../.env.example) for signup links.

| Environment variable | Setting | Used for |
|---|---|---|
| `PERPLEXITY_API_KEY` | `perplexity_api_key` | Sub-question research and the deep-research layer |
| `FIRECRAWL_API_KEY` | `firecrawl_api_key` | Web grounding and citation-verification scrapes |
| `OPENAI_API_KEY` | `openai_api_key` | Synthesis, the coverage evaluator, and query embeddings |
| `ANTHROPIC_API_KEY` | `anthropic_api_key` | Synthesis (alternate to OpenAI) |
| `BRAVE_API_KEY` | `brave_api_key` | Optional supplementary web-search backend |
| `SCRAPECREATORS_API_KEY` | `scrapecreators_api_key` | The X adapter and the LinkedIn enricher |
| `YOUTUBE_API_KEY` | `youtube_api_key` | The YouTube community adapter |
| `GITHUB_TOKEN` | `github_token` | Raises the GitHub adapter's rate limit |
| `REDDIT_CLIENT_ID` | `reddit_client_id` | Reddit OAuth (with the secret below); otherwise the adapter uses its keyless fallback |
| `REDDIT_CLIENT_SECRET` | `reddit_client_secret` | Reddit OAuth secret |

## Sources and output

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_X_HANDLES` | `x_handles` | (empty) | Comma-separated X handles to pull recent posts from. ScrapeCreators has no keyword-search endpoint for X, so collection is handle-based. |
| `POLYSEARCH_OUTPUT_DIR` | `output_dir` | `./reports` | Where reports are written. `--output-dir` overrides it per run. |

## Synthesis

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_SYNTHESIS_MODEL` | `synthesis_model` | `gpt-5.4-mini` | The OpenAI model for synthesis and the coverage evaluator. |
| `POLYSEARCH_SYNTHESIS_PRICE_IN` | `synthesis_price_in` | `0.75` | Input price per 1M tokens, for cost accounting. |
| `POLYSEARCH_SYNTHESIS_PRICE_OUT` | `synthesis_price_out` | `4.50` | Output price per 1M tokens. |

## Discovery and search

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_PERPLEXITY_MODEL` | `perplexity_model` | `sonar-pro` | The standard research model. |
| `POLYSEARCH_PERPLEXITY_DEEP_MODEL` | `perplexity_deep_model` | `sonar-reasoning-pro` | Used for deeper reasoning passes. |
| `POLYSEARCH_PERPLEXITY_PRICE_IN` | `perplexity_price_in` | `3.0` | Input price per 1M tokens (sonar-pro rates). |
| `POLYSEARCH_PERPLEXITY_PRICE_OUT` | `perplexity_price_out` | `15.0` | Output price per 1M tokens. |
| `POLYSEARCH_DISCOVERY_BACKEND` | `discovery_backend` | `perplexity` | One of `perplexity`, `brave`, `firecrawl`. An unrecognized value falls back to `perplexity`. |
| `POLYSEARCH_EMBEDDING_MODEL` | `embedding_model` | `text-embedding-3-small` | The OpenAI embedding model for the verifier's paraphrase-rescue fallback. |

## Verification and recovery

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_FUZZY_THRESHOLD` | `fuzzy_threshold` | `0.85` | Partial-ratio score at or above which a quote counts as matched. |
| `POLYSEARCH_RECOVERY_RATE_THRESHOLD` | `recovery_rate_threshold` | `0.35` | The recovery pass triggers when the verified-OK rate falls below this. |
| `POLYSEARCH_RECOVERY_MIN_CITATIONS` | `recovery_min_citations` | `4` | Recovery only triggers once at least this many citations were checked. |
| `POLYSEARCH_RECOVERY_MAX_QUERIES` | `recovery_max_queries` | `3` | Cap on scoped re-sourcing queries per recovery pass. |
| `POLYSEARCH_FIRECRAWL_USD_PER_CREDIT` | `firecrawl_usd_per_credit` | `0.0032` | Price per Firecrawl credit, for cost accounting. |

## Deep research

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_DEEP_RESEARCH` | `enable_deep_research` | `false` | Enables the deep-research layer. `--depth deep` and `--deep-research` also turn it on for a run. |
| `POLYSEARCH_DEEP_RESEARCH_MODEL` | `deep_research_model` | `sonar-deep-research` | The Perplexity deep-research model. |
| `POLYSEARCH_DEEP_RESEARCH_TIMEOUT_S` | `deep_research_timeout_s` | `3600` | Timeout in seconds for a deep-research call. |

## Refinement and output

| Environment variable | Setting | Default | Notes |
|---|---|---|---|
| `POLYSEARCH_REFINEMENT_COST_CEILING_USD` | `refinement_cost_ceiling_usd` | (unset) | When set, overrides the depth profile's cumulative cost ceiling for the refinement loop. |
| `POLYSEARCH_STYLE_CONSTRAINTS` | `style_constraints` | (unset) | Free-text style constraints passed to the synthesizer; when set, the report adds a "Style Audit" section. |
| `POLYSEARCH_ALLOW_PLACEHOLDERS` | `allow_placeholders` | `false` | Lets a report save even when it still holds unresolved `{{...}}` template tokens. Set automatically when no synthesizer is configured. |

## Data overrides

These two are read at load time by the tiering and schema loaders, not through `Settings`. They point the bundled data files at your own copies.

| Environment variable | Overrides |
|---|---|
| `POLYSEARCH_DOMAIN_TIERS` | Path to a `domain_tiers.yaml` that replaces the bundled domain map. |
| `POLYSEARCH_SCHEMA_DIR` | Directory of authoritative-source extraction schemas that replaces the bundled `data/authoritative_schemas/`. |

## Depth profiles

Depth is chosen per run (`--depth quick|standard|deep`), not through the environment. Each profile sets the search breadth, verification budget, and refinement bounds for the run. The values live in `DEPTH_PROFILES` in `config.py`:

| Knob | quick | standard | deep |
|---|---|---|---|
| Perplexity sub-questions | 2 | 4 | 6 |
| Firecrawl search limit | 5 | 10 | 20 |
| Firecrawl pages scraped | 3 | 5 | 10 |
| Community results | 10 | 20 | 30 |
| Authoritative pages mined | 0 | 2 | 4 |
| Verification budget (USD) | 1 | 3 | 10 |
| Verification concurrency | 5 | 8 | 10 |
| Refinement iterations | 0 | 2 | 4 |
| Refinement cost ceiling (USD) | 0 | 2.00 | 8.00 |
| Deep research eligible | no | no | yes |

## Install tier resolution

`resolve_install_tier(settings)` reports the highest tier the current keys satisfy, and `polysearch --diagnose` prints it. Tiers are cumulative:

- **Tier 0**: baseline (Perplexity only, or nothing).
- **Tier 1**: a Firecrawl key plus a synthesis model (OpenAI or Anthropic).
- **Tier 2**: Tier 1 plus at least one source connector (`scrapecreators_api_key`, `youtube_api_key`, `github_token`, `reddit_client_id`, or `reddit_client_secret`).

See [`examples/tiers.md`](../examples/tiers.md) for what each tier gets you in practice.
