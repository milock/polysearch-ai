"""LLM-as-judge for eval reports.

Scores a finished research report (markdown, end-state only — the judge never
sees the pipeline's process) across five dimensions from the Anthropic
research-system rubric: factual accuracy, citation accuracy, completeness
against the task's key facts, source quality, and coherence/insight. Each
dimension is a 0.0–1.0 score plus a one-line justification; the model also
returns an overall score and a pass/fail verdict.

Two hard rules baked in:

- **The judge JSON schema never asks for a URL.** Both the Perplexity and OpenAI
  structured-output guidance warn that models fabricate URLs when a schema field
  invites one. Citations are checked programmatically upstream; the judge only
  scores the prose it is shown.
- **A parse failure is not a crash.** ``parse_judge_response`` maps any malformed
  or incomplete payload to a :class:`JudgeResult` carrying ``error``; the sweep
  records that run as ERROR and moves on.

Model: ``gpt-5.4-nano`` ($0.20 / $1.25 per 1M in/out) — cheap enough to keep four
or more improvement rounds affordable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# gpt-5.4-nano pricing, $ per 1M tokens. Update here if the judge model changes.
JUDGE_MODEL = "gpt-5.4-nano"
JUDGE_PRICE_IN = 0.20
JUDGE_PRICE_OUT = 1.25

_MAX_TOKENS = 1200

# The five rubric dimensions, in scoreboard order. ``overall`` is separate.
DIMENSIONS: tuple[str, ...] = (
    "factual_accuracy",
    "citation_accuracy",
    "completeness",
    "source_quality",
    "coherence",
)

_DIMENSION_GUIDANCE: dict[str, str] = {
    "factual_accuracy": "Are the report's factual claims correct and internally consistent?",
    "citation_accuracy": (
        "Do claims carry source tags, and do the tags match the claim's strength "
        "(no marketing/opinion source dressed up as authoritative)?"
    ),
    "completeness": "Does the report cover each of the task's key facts listed below?",
    "source_quality": (
        "Is the tier mix appropriate — primary/official for factual claims, "
        "community treated as sentiment not proof?"
    ),
    "coherence": "Is the report clear, well-organized, and does it surface real insight?",
}


def _dimension_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "justification": {"type": "string"},
        },
        "required": ["score", "justification"],
    }


# json_schema for OpenAI structured output. No field invites a URL — see module
# docstring. ``strict`` + closed objects force the model to return exactly this.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{dim: _dimension_schema() for dim in DIMENSIONS},
        "overall": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "pass": {"type": "boolean"},
                "justification": {"type": "string"},
            },
            "required": ["score", "pass", "justification"],
        },
    },
    "required": [*DIMENSIONS, "overall"],
}


@dataclass
class JudgeResult:
    """Parsed judge verdict, or a parse error.

    ``scores`` maps each dimension to its 0.0–1.0 score; ``justifications`` maps
    each dimension (plus ``overall``) to its one-liner. ``overall`` / ``passed``
    are ``None`` when ``error`` is set.
    """

    scores: dict[str, float] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    overall: float | None = None
    passed: bool | None = None
    cost_usd: float = 0.0
    error: str | None = None

    @classmethod
    def from_scores(cls, payload: dict[str, Any], *, cost_usd: float = 0.0) -> "JudgeResult":
        """Build from a validated payload dict (raises nothing on valid input)."""
        scores = {dim: float(payload[dim]["score"]) for dim in DIMENSIONS}
        justifications = {dim: str(payload[dim]["justification"]) for dim in DIMENSIONS}
        overall_obj = payload["overall"]
        justifications["overall"] = str(overall_obj["justification"])
        return cls(
            scores=scores,
            justifications=justifications,
            overall=float(overall_obj["score"]),
            passed=bool(overall_obj["pass"]),
            cost_usd=cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "justifications": self.justifications,
            "overall": self.overall,
            "passed": self.passed,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error,
        }


# H2 sections that are the pipeline's own process/audit, not report content. The
# judge evaluates the end state, not the machinery — and must never dock citation
# accuracy on the pipeline's own citation-integrity tally (which lists its
# NUMBER_MISMATCH/URL_DEAD findings). Matched case-insensitively as substrings of
# the H2 heading text; every keyword here is generic (no internal vocabulary).
_PROCESS_SECTION_KEYWORDS: tuple[str, ...] = (
    "pipeline decisions",
    "pipeline stats",
    "pipeline errors",
    "refinement trace",
    "citation integrity",
    "style audit",
)
# H3 subsections inside kept sections that are audit residue (e.g. the "Excluded
# (dead links)" lists under Sources by Tier, or "Failed citations" details).
_PROCESS_SUBSECTION_KEYWORDS: tuple[str, ...] = ("excluded", "failed citations")

_H2_RE = re.compile(r"^## +(.*)$")
_H3_RE = re.compile(r"^### +(.*)$")


def strip_process_sections(md: str) -> str:
    """Remove the pipeline's process/audit sections from a report md.

    Excises whole H2 sections whose heading names a process/audit section, and
    audit H3 subsections (Excluded lists, Failed-citations details) inside kept
    sections. Leaves the synthesis body and the Sources-by-Tier listing so the
    judge sees the report's end state, not the machinery that produced it.
    Works for both report shapes by matching heading text, not position.
    """
    lines = md.splitlines()
    out: list[str] = []
    skip_h2 = False
    skip_h3 = False
    for line in lines:
        h2 = _H2_RE.match(line)
        if h2:
            heading = h2.group(1).lower()
            skip_h2 = any(k in heading for k in _PROCESS_SECTION_KEYWORDS)
            skip_h3 = False  # a new H2 clears any H3-skip
            if skip_h2:
                continue
            out.append(line)
            continue
        h3 = _H3_RE.match(line)
        if h3 and not skip_h2:
            heading = h3.group(1).lower()
            skip_h3 = any(k in heading for k in _PROCESS_SUBSECTION_KEYWORDS)
            if skip_h3:
                continue
        if skip_h2 or skip_h3:
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def build_judge_prompt(
    topic: str, key_facts: list[str], report_md: str, *, full_md: bool = False
) -> str:
    """Assemble the judge user prompt: rubric + the task's key facts + the report.

    The report markdown is the *only* view of the run the judge gets — this is
    end-state evaluation, not process evaluation. By default the pipeline's own
    process/audit sections are stripped (see :func:`strip_process_sections`);
    ``full_md=True`` is an escape hatch that feeds the whole report unmodified.
    """
    if not full_md:
        report_md = strip_process_sections(report_md)
    lines = [
        "You are a strict evaluator of a research report. Score the report below "
        "on each dimension from 0.0 (worst) to 1.0 (best) and give a one-line "
        "justification for each. Then give an overall score and a pass/fail "
        "verdict (pass means the report is decision-ready).",
        "",
        f"Research topic: {topic}",
        "",
        "Rubric dimensions:",
    ]
    for dim in DIMENSIONS:
        lines.append(f"- {dim.replace('_', ' ')}: {_DIMENSION_GUIDANCE[dim]}")
    lines.append("")
    lines.append("Key facts a complete report on this topic MUST cover:")
    for fact in key_facts:
        lines.append(f"- {fact}")
    lines.append("")
    lines.append(
        "Judge only what is written in the report. Do not reward or invent "
        "sources, links, or facts that are not present. Score citation accuracy "
        "from the source tags shown, not from any URL you might expect."
    )
    lines.append("")
    lines.append("--- REPORT START ---")
    lines.append(report_md)
    lines.append("--- REPORT END ---")
    return "\n".join(lines)


def parse_judge_response(raw: str) -> JudgeResult:
    """Parse a judge JSON payload into a :class:`JudgeResult`.

    Any failure (invalid JSON, missing dimension, non-numeric score, out-of-range
    value) returns a result carrying ``error`` rather than raising — the caller
    records the run as ERROR.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return JudgeResult(error=f"judge response was not valid JSON: {exc}")

    if not isinstance(payload, dict):
        return JudgeResult(error="judge response was not a JSON object")

    try:
        for dim in DIMENSIONS:
            block = payload[dim]
            score = float(block["score"])
            if not 0.0 <= score <= 1.0:
                return JudgeResult(error=f"{dim} score out of range: {score}")
            _ = str(block["justification"])
        overall = payload["overall"]
        overall_score = float(overall["score"])
        if not 0.0 <= overall_score <= 1.0:
            return JudgeResult(error=f"overall score out of range: {overall_score}")
        _ = bool(overall["pass"])
        _ = str(overall["justification"])
    except (KeyError, TypeError, ValueError) as exc:
        return JudgeResult(error=f"judge response missing/invalid field: {exc}")

    return JudgeResult.from_scores(payload)


def _judge_cost(usage: Any) -> float:
    if not usage:
        return 0.0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return round(
        (prompt_tokens * JUDGE_PRICE_IN + completion_tokens * JUDGE_PRICE_OUT)
        / 1_000_000,
        6,
    )


# Retry budget for a transient judge failure. Large reports (tens of thousands of
# tokens) routinely bump the model's tokens-per-minute ceiling during a sweep, and
# that is transient — a plain failure there wastes the whole task's run.
JUDGE_MAX_RETRIES = 5
_BACKOFF_MULTIPLIER_SEC = 4.0
_BACKOFF_MIN_SEC = 4.0
_BACKOFF_CAP_SEC = 60.0
# Default inter-call spacing (seconds) so a multi-task sweep doesn't burst the org
# rate limit. Overridable via POLYSEARCH_EVAL_JUDGE_RPS (requests per second).
_DEFAULT_JUDGE_SPACING_SEC = 0.5


def _is_rate_limit(exc: BaseException) -> bool:
    """Whether ``exc`` is an OpenAI rate-limit (429), by type, status, or message."""
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


def _is_server_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a retryable 5xx / connection / timeout server error."""
    if type(exc).__name__ in {
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
    }:
        return True
    code = getattr(exc, "status_code", None)
    return isinstance(code, int) and 500 <= code < 600


def _should_retry_judge(exc: BaseException) -> bool:
    return _is_rate_limit(exc) or _is_server_error(exc)


def judge_spacing_sec() -> float:
    """Inter-call spacing for the judge, from ``POLYSEARCH_EVAL_JUDGE_RPS`` (as
    ``1/rps``) or the 0.5s default. Used by the sweep to pace judge calls."""
    import os

    rps = os.environ.get("POLYSEARCH_EVAL_JUDGE_RPS")
    if rps:
        try:
            value = float(rps)
            if value > 0:
                return 1.0 / value
        except ValueError:
            pass
    return _DEFAULT_JUDGE_SPACING_SEC


def judge_report(
    topic: str,
    key_facts: list[str],
    report_md: str,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    full_md: bool = False,
    max_retries: int = JUDGE_MAX_RETRIES,
    sleep_fn: Any = None,
) -> JudgeResult:
    """Score one report with the judge model.

    ``client`` (an OpenAI-shaped client exposing ``chat.completions.create``) is
    injected in tests; in production it is built from ``api_key`` /
    ``OPENAI_API_KEY``. A rate-limit (429) or 5xx is retried up to ``max_retries``
    times with exponential backoff (via tenacity); any other error is surfaced as
    an ERROR result — the sweep never crashes on a single bad call. ``full_md``
    feeds the whole report unstripped. ``sleep_fn`` is injected in tests so retry
    backoff doesn't actually wait.
    """
    from tenacity import (
        Retrying,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential,
    )

    if client is None:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    if sleep_fn is None:
        import time

        sleep_fn = time.sleep

    prompt = build_judge_prompt(topic, key_facts, report_md, full_md=full_md)

    def _call() -> Any:
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            max_completion_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "report_evaluation",
                    "strict": True,
                    "schema": JUDGE_SCHEMA,
                },
            },
        )

    try:
        completion = Retrying(
            retry=retry_if_exception(_should_retry_judge),
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(
                multiplier=_BACKOFF_MULTIPLIER_SEC, min=_BACKOFF_MIN_SEC, max=_BACKOFF_CAP_SEC
            ),
            sleep=sleep_fn,
            reraise=True,
        )(_call)
    except Exception as exc:  # noqa: BLE001 — one bad judge call must not crash the sweep
        return JudgeResult(error=f"judge call failed: {exc}")

    content = completion.choices[0].message.content or ""
    result = parse_judge_response(content)
    result.cost_usd = _judge_cost(getattr(completion, "usage", None))
    return result
