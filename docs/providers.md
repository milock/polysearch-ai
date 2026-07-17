# Providers

Every layer talks to the outside world through a small structural protocol. The orchestrator never imports a concrete provider directly; it holds a `Providers` bundle of things that satisfy the protocols, so you can swap any implementation, add a new one, or inject nulls and mocks for a fully offline run. The protocols live in [`src/polysearch/providers/base.py`](../src/polysearch/providers/base.py).

## The protocols

Each is `runtime_checkable`, so the orchestrator can probe a slot with `isinstance` when it needs to.

| Protocol | Method | Returns | Default implementation |
|---|---|---|---|
| `ResearchProvider` | `async research(topic, *, sub_questions, depth)` | `LayerOutput` | `PerplexityProvider`; `DeepResearchProvider` for the opt-in deep layer |
| `WebGrounder` | `async ground(topic, *, limit, scrape_top_k)` | `LayerOutput` | `FirecrawlGrounder` |
| `Synthesizer` | `async synthesize(topic, layers, *, style_constraints)` | `(markdown, cost_usd)` | `OpenAISynthesizer`, `AnthropicSynthesizer` |
| `CitationVerifier` | `async verify(claims, *, budget_usd, max_concurrency)` | `VerificationReport` | `FirecrawlVerifier` |
| `CommunitySource` | `async search(topic, *, window_days, limit)` | `list[SourceResult]` | `RedditSource`, `HackerNewsSource`, `BlueskySource`, `GitHubSource`, `XSource`, `YouTubeSource` |
| `PersonContextHook` | `async lookup(name)` | `SourceResult \| None` | none (a seam for your own private source) |

`LayerOutput`, `SourceResult`, `Claim`, and `VerificationReport` are the pydantic models in [`src/polysearch/output/schema.py`](../src/polysearch/output/schema.py). A provider's job is to fill those shapes; nothing else in the pipeline needs to know how.

## How wiring works

`build_providers(settings)` in `base.py` resolves a `Settings` into a concrete `Providers` bundle. Each slot is built by a small `_build_*` function that checks for the credential it needs and imports the real class lazily inside the credential-gated branch. Two things follow from that:

- **Missing credential means a null provider, not a crash.** Each null returns an empty result and carries a `reason` string naming exactly what is missing. That reason travels into the report's decision log, so a thin run explains itself.
- **A layer never half-initializes.** If a real provider's module is unavailable, the slot degrades to its null with a reason rather than raising during startup.

You rarely call `build_providers` yourself. Pass keys through the environment (or a `.env`) and the CLI builds the bundle. Call it directly only when you want to inspect or override a slot before a run.

## Adding a provider

Say you want to add a second web grounder. Implement `WebGrounder`:

```python
# src/polysearch/providers/tavily.py
from polysearch.config import Settings
from polysearch.output.schema import LayerOutput, SourceResult
from polysearch.sources.authority import classify_url


class TavilyGrounder:
    """Web grounding via Tavily. Satisfies the WebGrounder protocol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def ground(self, topic: str, *, limit: int, scrape_top_k: int) -> LayerOutput:
        results: list[SourceResult] = []
        # ... call Tavily, build SourceResult rows ...
        for hit in await self._search(topic, limit=limit):
            results.append(
                SourceResult(
                    url=hit["url"],
                    title=hit["title"],
                    snippet=hit["content"][:600],
                    tier=classify_url(hit["url"]),
                    layer="grounding",
                )
            )
        return LayerOutput(layer="grounding", results=results, cost_usd=0.0)
```

Then wire it in. Either edit `_build_grounder` in `base.py` to prefer Tavily when its key is set, or construct the bundle yourself and pass it to `run_research(..., providers=...)`.

Guidelines that keep a provider well-behaved in the pipeline:

- **Classify every URL** with `classify_url` from [`src/polysearch/sources/authority.py`](../src/polysearch/sources/authority.py) so the source lands in the right tier.
- **Isolate failures.** A raise inside a layer is caught by the orchestrator, but prefer to catch your own I/O errors and return an empty `LayerOutput` (or set `error`) so the reason is specific.
- **Report cost.** Fill `cost_usd` on the `LayerOutput` from real usage numbers when the API returns them; the orchestrator sums these into the run total.
- **Stay async.** Every protocol method is a coroutine. Do network I/O with an async client (the package uses `httpx`) so the layer runs concurrently with its siblings.

When you contribute one back, add a `[<provider>]` extra in `pyproject.toml` if it pulls a heavy SDK, a line in `.env.example` for any new keys, and a subsection here. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Adding a community adapter

Community sources are a lighter contract than the research layers. An adapter exposes a `name` and an `async search(topic, *, window_days, limit)` that returns tier-`COMMUNITY` `SourceResult` rows, and it never raises. They live in [`src/polysearch/community/adapters.py`](../src/polysearch/community/adapters.py).

```python
class LobstersSource:
    """Community signal from Lobsters. Satisfies CommunitySource."""

    name = "lobsters"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.last_error: str | None = None

    async def search(self, topic: str, *, window_days: int, limit: int) -> list[SourceResult]:
        try:
            rows = await self._fetch(topic, limit=limit)
        except Exception as exc:  # never raise out of an adapter
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        return [
            SourceResult(
                url=r["url"],
                title=r["title"],
                snippet=r["excerpt"],
                tier="COMMUNITY",
                published_date=r.get("date"),
                layer=self.name,
                engagement=r.get("score"),
            )
            for r in rows
        ]
```

Four conventions make an adapter fit the layer:

- **Never raise.** Catch everything, set `self.last_error`, and return `[]`. The orchestrator surfaces `last_error` as a pipeline note and keeps going.
- **Set `engagement`.** Fusion ranks partly on it, and the report shows it as the "attention, not proof" signal.
- **Set `layer` to your source slug.** Fusion normalizes scores per source, so a slug that is unique to your adapter keeps its items from being averaged against another source's.
- **Read credentials from `Settings`, not the environment.** Use a key only to lift rate limits or unlock an authenticated endpoint; a keyless adapter should still return results at any tier.

Register it in `_build_community_sources` in `base.py`. Keyless adapters are added unconditionally (they run at any tier); key-gated ones are appended only when their credential is set. Rate limiting flows through the shared `ratelimit` ledger, so record a 429 with `ratelimit.record_429` to back off sibling processes rather than retrying blindly.
