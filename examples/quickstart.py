"""Minimal polysearch quickstart.

Runs one research topic through the pipeline and prints the synthesized answer.

Prerequisites:
  pip install polysearch-ai
  echo "PERPLEXITY_API_KEY=your_key_here" > .env   # at least one key

Run:
  python examples/quickstart.py
"""

import asyncio

from polysearch import run_research


async def main() -> None:
    report = await run_research(
        "What is the current US federal funds rate?",
        depth="quick",
    )
    print(report.synthesis_md)
    cost = report.totals.get("cost_usd", 0.0)
    print(f"\ncost: ${cost:.4f}  |  report written under ./reports/")


if __name__ == "__main__":
    asyncio.run(main())
