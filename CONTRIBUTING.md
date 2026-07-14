# Contributing to polysearch

Thanks for considering it. polysearch is a small, focused project — the most valuable contributions are new provider implementations, sharper prompts, and better test fixtures. Feature creep is the enemy.

---

## What contributions are most useful

### High value

- **New provider implementations.** The protocols in `src/polysearch/providers/base.py` (`ResearchProvider`, `WebGrounder`, `Synthesizer`, `CitationVerifier`, `CommunitySource`, `PersonContextHook`) are the spine. Implementations of any of them are the highest-leverage contribution.
  - `ResearchProvider`: OpenAI deep research (when GA), Tavily Research, You.com Research API
  - `WebGrounder`: Tavily, Brave Search, Serper, Bing Web Search
  - `Synthesizer`: any frontier model not yet covered
  - `CitationVerifier`: alternate scrape backends
  - `CommunitySource`: a new forum or social source (see `docs/providers.md`)
- **Worked examples** for new use cases — e.g., `examples/finance.md`, `examples/legal-research.md`. The `domain_tiers.yaml` ships with public-friendly defaults; domain-specific overrides as examples are gold.
- **Bug reports** with a reproducible input. Including the depth, providers, env state, and expected vs. actual output helps reproduce.

### Medium value

- **Refining the default `domain_tiers.yaml`.** If a domain belongs in a different tier than the default places it, open an issue with the rationale (path-pattern downgrade evidence, regulatory authority docs, etc.).
- **Improving prompts.** The synthesizer and verifier prompts live in their respective provider files. A/B evidence helps land changes.
- **Fixture additions.** Recorded API responses for new test scenarios.
- **Documentation gaps.** If a new contributor hits friction during setup, the fix usually goes in `docs/` or the README quickstart.

### Low value

- Adding more banned words, generic style nits, or "make it strict" preference changes without evidence.
- Wholesale reformatting / rename PRs.
- Adding new optional dependencies for niche use cases.

---

## How to propose a new provider

Open a PR that adds:

1. A new file under `src/polysearch/providers/<name>.py` implementing one or more of the protocols in `base.py`.
2. Tests in `tests/unit/test_providers_<name>.py` with recorded fixtures under `tests/fixtures/<name>/`.
3. A brief subsection in [`docs/providers.md`](docs/providers.md) explaining when to choose this provider over alternatives, and what env vars / SDK setup it needs.
4. A line in `.env.example` for any new env vars (with comment + signup link).
5. A `[<provider>]` extra in `pyproject.toml` if the SDK is large or rarely used (so the default install stays slim).

The protocol contracts are stable; implementations behind them aren't required to support every feature. Document what your implementation skips with `NotImplementedError` raises and a comment.

---

## Coding standards

- **Python 3.12+.** We use `type Foo = ...` syntax, `match` statements, and modern union (`X | Y`) annotations.
- **Async by default** for any provider method that does I/O. The orchestrator runs layers in parallel.
- **Type hints required** on public functions and class methods. Run `pyright` or `mypy` if you have one configured locally.
- **No `print()` for logging.** Use the `logging` module; the orchestrator wires up a logger.
- **No silent failures.** A provider failing should log a warning and substitute null behavior, not return wrong data.
- **House style.** Black-compatible formatting. 100-char line limit. Imports sorted (`isort` or `ruff` defaults).

---

## Testing

There's a real test pyramid. Match it:

```
tests/
├── unit/         Fast, no network, mocks everything. Default CI run.
├── integration/  End-to-end orchestrator with mocked providers.
└── live/         Real APIs, opt-in via `pytest --run-live -m live`.
```

For each PR:
1. Add unit tests for any new code paths.
2. Update integration tests if the orchestrator behavior changes.
3. Live tests are optional unless you're testing real-API behavior.
4. Run `pytest tests/unit tests/integration -v` before opening the PR.

Coverage target: 85% for `src/polysearch/`, 100% for protocol contracts, 70% for individual provider implementations.

Mocking pattern: use `respx` for httpx-based providers, `pytest-mock` for SDK mocks, JSON fixtures under `tests/fixtures/<provider>/` keyed by query hash.

---

## Filing issues

A useful bug report:

- The CLI command (or Python call) that reproduces the issue, with `--topic` redacted if sensitive
- Tier (which providers were active)
- Env state: which keys were set (don't paste the keys themselves)
- Expected vs. actual output — include the full Humanizer Report block from the markdown output
- polysearch version (`polysearch --version`)
- Python version

A useful feature request:

- The specific scenario where polysearch falls short
- Why an existing mechanism (provider extension, config flag, custom domain tiers) doesn't already cover it
- A proposed change in concrete terms: which file, which protocol, which behavior

---

## License

By contributing, you agree your contributions are released under the MIT license (see [`LICENSE`](LICENSE)).
