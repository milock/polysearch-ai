"""polysearch — modular multi-source research pipeline."""

__version__ = "1.0.0"

# Imported after ``__version__`` is set: the orchestrator's dependency chain
# reaches ``output.report``, which does ``from polysearch import __version__``.
from polysearch.orchestrator import run_research  # noqa: E402

__all__ = ["__version__", "run_research"]
