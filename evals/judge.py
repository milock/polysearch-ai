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


def build_judge_prompt(topic: str, key_facts: list[str], report_md: str) -> str:
    """Assemble the judge user prompt: rubric + the task's key facts + the report.

    The report markdown is the *only* view of the run the judge gets — this is
    end-state evaluation, not process evaluation.
    """
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


# How many times to retry a rate-limited (429) judge call before giving up, and
# the backoff floor. Large reports (tens of thousands of tokens) routinely bump
# the model's tokens-per-minute ceiling during a sweep, and that is transient —
# a plain failure there wastes the whole task's run, so we wait and retry.
JUDGE_MAX_RETRIES = 5
_BACKOFF_BASE_SEC = 8.0
_BACKOFF_CAP_SEC = 60.0
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)s")


def _is_rate_limit(exc: Exception) -> bool:
    """Whether ``exc`` is an OpenAI rate-limit (429), by type, status, or message."""
    if type(exc).__name__ == "RateLimitError":
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt: the server's suggested wait (plus
    a small buffer) when it gives one, else capped exponential backoff."""
    hint = _RETRY_AFTER_RE.search(str(exc))
    if hint:
        return float(hint.group(1)) + 1.0
    return min(_BACKOFF_BASE_SEC * (2**attempt), _BACKOFF_CAP_SEC)


def judge_report(
    topic: str,
    key_facts: list[str],
    report_md: str,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    max_retries: int = JUDGE_MAX_RETRIES,
    sleep_fn: Any = None,
) -> JudgeResult:
    """Score one report with the judge model.

    ``client`` (an OpenAI-shaped client exposing ``chat.completions.create``) is
    injected in tests; in production it is built from ``api_key`` /
    ``OPENAI_API_KEY``. A rate-limit (429) is retried up to ``max_retries`` times
    with backoff (honoring the server's suggested wait); any other transport error
    is surfaced as an ERROR result — the sweep never crashes on a single bad call.
    ``sleep_fn`` is injected in tests so retries don't actually wait.
    """
    if client is None:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    if sleep_fn is None:
        import time

        sleep_fn = time.sleep

    prompt = build_judge_prompt(topic, key_facts, report_md)
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
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
        except Exception as exc:  # noqa: BLE001 — one bad judge call must not crash the sweep
            if attempt < max_retries and _is_rate_limit(exc):
                sleep_fn(_retry_delay(exc, attempt))
                continue
            return JudgeResult(error=f"judge call failed: {exc}")

        content = completion.choices[0].message.content or ""
        result = parse_judge_response(content)
        result.cost_usd = _judge_cost(getattr(completion, "usage", None))
        return result

    return JudgeResult(error="judge call failed: retries exhausted")  # pragma: no cover
