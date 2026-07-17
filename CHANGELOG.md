# Changelog

All notable changes to this project will be documented here. Format loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/), with the caveat that the API is **unstable until v1.0** — minor bumps may break compatibility.

## [Unreleased]

## [1.0.1] — 2026-07-17

### Fixed
- `.env` discovery for installed packages: `Settings.from_env()` now searches
  for `.env` from the current working directory upward (`find_dotenv(usecwd=True)`).
  The previous no-arg `load_dotenv()` searched from the package's own file
  location, which worked in a source checkout but silently found nothing when
  installed via pip/pipx — credentials in a perfectly good `.env` were ignored
  and `--diagnose` reported tier 0.

## [1.0.0] — 2026-07-17

First stable release: the full pipeline, packaged for PyPI.

### Packaging
- **Published to PyPI as `polysearch-ai`.** Install with `pip install polysearch-ai`; the import name and CLI stay `polysearch`.
- Ships as a Claude Code plugin (`/plugin install polysearch`) and as a copyable skill via `install.sh`.
- Config data (`domain_tiers.yaml`, authoritative-source schemas) ships inside the wheel and resolves from installed packages; override with `POLYSEARCH_DOMAIN_TIERS` / `POLYSEARCH_SCHEMA_DIR`.

### Pipeline
- **Orchestrator** running the first-pass layers in parallel, then synthesis, verification, recovery, and refinement, with every layer credential-gated and failure-isolated.
- **Four research layers:** Perplexity sub-question research, Firecrawl web grounding with authoritative-fact extraction on HIGH-tier pages, a native community layer, and an opt-in Perplexity deep-research layer (`--depth deep` or `--deep-research`).
- **Community layer is now native.** The adapters (Reddit, Hacker News, Bluesky, GitHub, X, YouTube) call each source's API directly, fuse into one ranked list, and pass a relevance gate. This was previously planned as a subprocess wrapper around an external tool; it is now first-party code adapted from last30days (MIT); see ATTRIBUTION.md.
- **Goal-driven refinement loop:** a coverage evaluator judges the corpus against the topic and emits follow-up queries, bounded by iteration, cost, and dry-exit guards.
- **Citation verification** that fetches every cited page and matches quotes and numbers, including citations mined from deep-research and Perplexity narrative answers.
- **Recovery pass** that re-sources weak citations before giving up on a claim.
- **Cross-process rate limiting** through a shared on-disk ledger, so a 429 in one run backs off concurrent runs.
- **Blocked-source exclusion:** hard-blocked hosts are dropped before scraping; a resolved one is diverted into an "Excluded (blocked sources)" report section.

### Providers and synthesis
- `PerplexityProvider`, `DeepResearchProvider`, `FirecrawlGrounder`, `FirecrawlVerifier`, `OpenAISynthesizer`, `AnthropicSynthesizer`, and the LinkedIn enricher, all behind `runtime_checkable` protocols in `providers/base.py`.
- Synthesizer auto-resolution (OpenAI preferred, Anthropic fallback), overridable with `--synthesizer`.

### CLI
- `--topic`, `--depth`, `--output-dir`, `--providers`, `--synthesizer`, `--verify-budget`, `--no-verify`, `--no-recovery`, `--max-iterations`, `--deep-research`.
- No-network utility modes: `--classify`, `--diagnose`, and `--synthesize-parallel` for cross-report rollups.

### Docs and tests
- README, architecture, providers, configuration, and agent-integration docs.
- A repo-tree leakage gate (`tests/unit/test_no_internal_leakage.py`) enforced in CI.

## [0.1.0] — 2026-04-29

Initial public release.

### Added
- Repo bootstrap: package skeleton, `pyproject.toml`, MIT license, README, contributing guide, attribution.
- CLI entry point (`polysearch --topic ...`) — Phase 0 placeholder; orchestrator lands in Phase 1.
- Test scaffolding: pytest configuration with `live`, `integration`, `slow` markers; smoke test asserting package import + version.
- GitHub Actions CI: tests on push/PR (Python 3.12 + 3.13), PyPI release on tag push.
- Issue templates and CONTRIBUTING guide.

### Pending (subsequent phases — tracked in the implementation plan)
- Provider protocol definitions and null implementations.
- Concrete providers behind each protocol (research, web grounding, synthesis, verification, community).
- Orchestrator with parallel layer execution, depth profiles, citation verification, hollow-flag handling.
- Claude Code skill (`skills/research/SKILL.md`) and thin agent template (`agents/researcher.md`).
- Documentation: architecture, providers, modes, citation tiers, cost modeling, migration.
- PyPI publish on v0.1.0 tag.
