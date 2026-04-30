# Attribution

polysearch composes work from several upstreams. This document credits them.

---

## Direct dependencies

### Last30days community signal layer
- **Project:** [last30days-skill](https://github.com/mvanhorn/last30days-skill)
- **Author:** mvanhorn
- **License:** MIT
- **How polysearch uses it:** as the optional Tier-3 community signal layer. polysearch's `Last30DaysCommunitySignal` provider invokes the upstream CLI via subprocess if installed; otherwise falls back to a null implementation. polysearch does not vendor or modify the last30days source.

### Perplexity Sonar API
- **Project:** [Perplexity AI](https://www.perplexity.ai)
- **License:** Commercial API (per their terms)
- **How polysearch uses it:** the default `ResearchProvider` (Tier 0). Decomposed sub-question research with citation-aware results. Models: `sonar-pro` (standard) and `sonar-reasoning-pro` (deep).

### Firecrawl
- **Project:** [Firecrawl](https://www.firecrawl.dev)
- **License:** Commercial API
- **How polysearch uses it:** the default `WebGrounder` and `CitationVerifier` (Tier 1). Web search + scrape with paywall detection and tier-aware downgrades.

### Qdrant
- **Project:** [Qdrant](https://qdrant.tech)
- **License:** Apache 2.0 (open source) and Cloud (commercial)
- **How polysearch uses it:** the default `VectorStore` (Tier 2 — optional). Personal/organizational corpus retrieval blended with research outputs.

### OpenAI
- **Project:** [OpenAI API](https://platform.openai.com)
- **License:** Commercial API
- **How polysearch uses it:** the default `Synthesizer` for cross-layer synthesis (`gpt-5-mini`) and embeddings for vector queries (`text-embedding-3-small`).

### Anthropic
- **Project:** [Anthropic API](https://docs.anthropic.com)
- **License:** Commercial API
- **How polysearch uses it:** the alternate `Synthesizer` (`claude-sonnet-*`). Used when only `ANTHROPIC_API_KEY` is set, or when explicitly requested via `--synthesizer anthropic`.

---

## Methodology references

The 4-layer pipeline architecture, citation tier system, and synthesis prompting style draw on patterns from public research-tool projects:

- **OpenAI Deep Research** (the agentic-research methodology) — polysearch's structured-pipeline approach is a deliberate alternative to agentic loops, but the citation-quality discipline is shared.
- **gpt-researcher** ([open-source agentic researcher](https://github.com/assafelovic/gpt-researcher)) — multi-source synthesis, depth profiles. polysearch's depth tiers (quick/standard/deep) take inspiration from this project's research-mode framing.
- **Perplexity Sonar API documentation** — sub-question decomposition patterns.
- **Wikipedia "Reliable sources" guidelines** — informed the default `domain_tiers.yaml` structure (tier-by-source-type rather than tier-by-domain-list).

---

## Embedded references in the pipeline

When polysearch ships content for the Claude Code skill (under `skills/research/`), it borrows formatting conventions from the [Anthropic Agent Skills](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills) documentation — progressive disclosure, frontmatter triggers, third-person descriptions.

---

## How to credit polysearch

If you build on polysearch in a public project, a link in your README is plenty:

```markdown
Powered by [polysearch](https://github.com/milock/polysearch).
```

For academic or research citations:

```bibtex
@software{polysearch,
  author = {polysearch contributors},
  title = {polysearch: modular multi-source research pipeline},
  year = {2026},
  url = {https://github.com/milock/polysearch}
}
```
