# Attribution

polysearch composes work from several upstream services and one adapted open-source library. This document credits them.

---

## Adapted open-source code

### last30days community-signal library

- **Project:** [last30days-skill](https://github.com/mvanhorn/last30days-skill)
- **Author:** Matthew Van Horn ([@mvanhorn](https://github.com/mvanhorn))
- **License:** MIT

polysearch's community layer is adapted from last30days: the design of the multi-source community search, and portions of the per-source adapter and fusion logic in [`src/polysearch/community/`](src/polysearch/community/). The layer is native to polysearch (the adapters call each source's API directly), not a subprocess wrapper around the upstream tool. Under the MIT license, the copyright and permission notice is retained below.

```
MIT License

Copyright (c) 2026 Matt Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Services and APIs

### Perplexity Sonar

- **Project:** [Perplexity AI](https://www.perplexity.ai)
- **License:** commercial API
- **How polysearch uses it:** the default `ResearchProvider`. Decomposed sub-question research with citation-aware results, using `sonar-pro` (standard) and `sonar-reasoning-pro`. The opt-in deep-research layer uses `sonar-deep-research`, whose narrative answers are mined for claims and verified like any other source.

### Firecrawl

- **Project:** [Firecrawl](https://www.firecrawl.dev)
- **License:** commercial API
- **How polysearch uses it:** the default `WebGrounder` and `CitationVerifier`. Web search plus scrape with paywall detection, and the first step of the resilient fetch chain during verification.

### ScrapeCreators

- **Project:** [ScrapeCreators](https://scrapecreators.com)
- **License:** commercial API
- **How polysearch uses it:** the X community adapter (handle-based post collection) and the LinkedIn profile enricher. Both are opt-in and off unless `SCRAPECREATORS_API_KEY` is set.

### OpenAI

- **Project:** [OpenAI API](https://platform.openai.com)
- **License:** commercial API
- **How polysearch uses it:** the default `Synthesizer` (`gpt-5.4-mini`), the coverage evaluator that drives the refinement loop, and embeddings (`text-embedding-3-small`) for the verifier's paraphrase-rescue fallback.

### Anthropic

- **Project:** [Anthropic API](https://docs.anthropic.com)
- **License:** commercial API
- **How polysearch uses it:** the alternate `Synthesizer`, used when only `ANTHROPIC_API_KEY` is set or when requested with `--synthesizer anthropic`.

### Community source APIs

The native community adapters read from these public and freemium endpoints:

- **Hacker News via Algolia** ([hn.algolia.com](https://hn.algolia.com/api)): unauthenticated search over HN stories and comments.
- **Bluesky AppView** ([public.api.bsky.app](https://docs.bsky.app)): the public AppView needs no auth for reads.
- **GitHub Search API** ([docs.github.com](https://docs.github.com/en/rest/search)): repository, code, and issue search; a `GITHUB_TOKEN` raises the rate limit.
- **Reddit** ([reddit.com/dev/api](https://www.reddit.com/dev/api)): the OAuth API when Reddit credentials are set, with an unauthenticated fallback.
- **YouTube Data API** ([developers.google.com/youtube](https://developers.google.com/youtube/v3)): video and transcript search, key-gated.

---

## Methodology references

The pipeline architecture, tiered citation model, and synthesis discipline draw on public research-tool patterns:

- **OpenAI Deep Research**: polysearch's fixed-pipeline approach is a deliberate alternative to an agentic loop, but it shares the emphasis on citation quality.
- **gpt-researcher** ([open-source agentic researcher](https://github.com/assafelovic/gpt-researcher)): the quick/standard/deep depth framing takes inspiration from this project's research modes.
- **Wikipedia "Reliable sources" guidance**: informed the tier-by-source-type structure of the default `domain_tiers.yaml`.

The Claude Code skill under [`skills/research/`](skills/research/) follows the conventions in the [Anthropic Agent Skills](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills) documentation: progressive disclosure, frontmatter triggers, third-person descriptions.

---

## How to credit polysearch

A link in your README is plenty:

```markdown
Powered by [polysearch](https://github.com/milock/polysearch).
```

For academic or research citations:

```bibtex
@software{polysearch,
  author = {polysearch contributors},
  title = {polysearch: multi-source research pipeline with citation verification},
  year = {2026},
  url = {https://github.com/milock/polysearch}
}
```
