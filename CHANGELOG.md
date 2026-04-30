# Changelog

All notable changes to this project will be documented here. Format loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/), with the caveat that the API is **unstable until v1.0** — minor bumps may break compatibility.

## [Unreleased]

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
- `PerplexityProvider`, `FirecrawlGrounder`, `OpenAISynthesizer`, `AnthropicSynthesizer`, `QdrantVectorStore`, `Last30DaysCommunitySignal`.
- Orchestrator with parallel layer execution, depth profiles, citation verification, hollow-flag handling.
- Claude Code skill (`skills/research/SKILL.md`) and thin agent template (`agents/researcher.md`).
- Documentation: architecture, providers, modes, citation tiers, cost modeling, migration.
- PyPI publish on v0.1.0 tag.
