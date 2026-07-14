# Integrating polysearch into an agent

polysearch is a command-line tool that writes a report file. That makes it easy
to give any agent a real research capability: have the agent run the CLI, then
read the file it produces. This guide covers four ways to wire it in, from a
one-line Claude Code install to a direct Python call.

The contract is the same everywhere:

> **Run `polysearch --topic "..."`. It writes a markdown report (and a sibling
> JSON) to `./reports/`. Read that file. Summarize it, keeping the source-tier
> caveats intact.**

---

## 1. Claude Code — as a plugin

The whole repository is a Claude Code plugin. Installing it gives you the
`/polysearch:research` skill and a `researcher` agent.

Add the marketplace and install, straight from the repo:

```
/plugin marketplace add milock/polysearch
/plugin install polysearch
```

Then research from any prompt — "research the current federal funds rate" — or
call the skill directly with `/polysearch:research`.

To test a local checkout without publishing, point Claude Code at the directory:

```bash
claude --plugin-dir ~/dev/polysearch
```

and use `/polysearch:research`.

---

## 2. Claude Code — as a skill copy

If you do not want the full plugin, copy just the skill. The installer does this
for you:

```bash
git clone https://github.com/milock/polysearch.git
cd polysearch && ./install.sh
```

`install.sh` installs the Python package and copies `skills/research` to
`~/.claude/skills/research`. After that, any research request in Claude Code
picks up the skill and runs the CLI.

---

## 3. Any other agent harness

polysearch does not need Claude Code. Any harness that can run a shell command
and read a file can use it. Two things to give your agent:

**A. Install and credentials.** `pip install polysearch-ai` (or run `install.sh`
from a checkout), then set at least one API key in the environment or a `.env`
file. `polysearch --diagnose` reports what is active.

**B. An instruction block.** Drop something like this into your agent's system
prompt or an `AGENTS.md` file:

```markdown
## Research

For any research, fact-check, comparison, or deep-dive request, use the
`polysearch` CLI instead of answering from memory.

1. Preview scope (free, no network): `polysearch --classify --topic "<topic>"`
2. Run it: `polysearch --topic "<topic>"` (add `--depth deep` for high-stakes
   topics, `--depth quick` for a single fact).
3. The command prints a report path. Read that markdown file.
4. Summarize the Synthesis section. Weight sources by their tier: HIGH and
   MEDIUM carry a claim; COMMUNITY and UNKNOWN are unconfirmed on their own.
   Always cite the report file path.

Never fabricate a source or URL. If a run comes back empty, run
`polysearch --diagnose` to see which layers and credentials are active.
```

That is the entire integration surface: call the CLI, read the report, respect
the tiers.

---

## 4. Python API — for SDK-based agents

If your agent runs in Python, skip the subprocess and call the pipeline
directly. The one public entry point is `run_research`, an async coroutine that
returns a structured report object and (by default) also writes the markdown and
JSON files.

```python
import asyncio
from polysearch import run_research

async def main():
    report = await run_research(
        "What is the current US federal funds rate?",
        depth="quick",
    )
    print(report.synthesis_md)                    # the written answer
    print(f"cost: ${report.totals.get('cost_usd', 0.0):.4f}")

asyncio.run(main())
```

Useful keyword arguments (all optional):

| Argument | Type | Effect |
|---|---|---|
| `depth` | `str` | `"quick"`, `"standard"` (default), or `"deep"` |
| `enabled_layers` | `set[str]` | Restrict first-pass layers (`research`, `grounding`, `community`, `deep_research`, `linkedin`); `None` runs all |
| `verify` | `bool` | Toggle citation verification (default `True`) |
| `recovery` | `bool` | Toggle the weak-verification recovery pass (default `True`) |
| `verify_budget` | `float` | Override the depth profile's verification budget, in USD |
| `deep_research` | `bool` | Force the deep-research layer at any depth |
| `max_iterations` | `int` | Override the refinement cap (`0` disables it) |
| `output_dir` | `str \| Path` | Where the report is written |
| `write` | `bool` | Set `False` to get the report object back without writing files |

For tests or fully offline runs, inject a `providers` bundle of null or mock
implementations — every layer is behind a Protocol, so nothing hits the network.
See `tests/integration/test_orchestrator.py` for the pattern.

---

## Credentials, briefly

polysearch runs with whatever keys are present and skips the rest. There is no
all-or-nothing requirement.

| Key | Unlocks |
|---|---|
| `PERPLEXITY_API_KEY` | Sub-question research and the deep-research layer |
| `FIRECRAWL_API_KEY` | Web grounding and citation-verification scrapes |
| `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | Synthesis |
| `QDRANT_URL` + `QDRANT_API_KEY` | Vector search over your own corpus |

See `.env.example` for the full list, and `examples/tiers.md` for what each
combination gets you and what it costs.

---

## CI usage

`polysearch --classify` and `polysearch --diagnose` make no network calls and
exit non-zero on a hard error, so they are safe smoke checks in CI. A full
`--topic` run costs money and hits live APIs; gate it behind a manual trigger or
a scheduled job with the keys provided as secrets, not on every push.
