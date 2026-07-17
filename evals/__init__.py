"""polysearch eval harness.

This package ships in the repo but is excluded from the built distribution (the
package build discovers only ``src/polysearch``). It drives simulated
deep-research tasks through a target CLI, computes programmatic metrics, scores
each report with an LLM judge, and writes a scoreboard per round.

Nothing here imports the private project it was extracted from: the public
target is the installed ``polysearch`` CLI, and the internal target — when used
— comes solely from the ``POLYSEARCH_EVAL_INTERNAL_CMD`` environment variable,
so no internal path or string lives in this repo.
"""
