"""polysearch command-line interface.

``polysearch --topic "..."`` runs the full pipeline (see
:func:`polysearch.orchestrator.run_research`) and writes a report. Utility modes
short-circuit before any network:

- ``--classify`` prints the classifier's verdict as JSON and exits 0.
- ``--diagnose`` prints which credentials are present, the resolved install tier,
  each layer's active/null status, and output-dir writability, then exits 0.
- ``--synthesize-parallel GLOB`` cross-synthesizes existing report files.

Exit codes: 0 success, 1 pipeline error (an unhandled failure during the run),
2 usage error (missing ``--topic`` for a real run, or argparse rejecting input).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from polysearch import __version__
from polysearch.classifier import classify
from polysearch.config import Settings, resolve_install_tier
from polysearch.orchestrator import run_parallel_synthesis, run_research
from polysearch.providers.base import (
    NullSynthesizer,
    Providers,
    build_providers,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polysearch",
        description="Modular, multi-source research pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"polysearch {__version__}")
    parser.add_argument("--topic", help="Research topic (required for a real run).")
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Research depth profile (default: standard).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the report (default: ./reports/ or POLYSEARCH_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--providers",
        help=(
            "Comma-separated allowlist of first-pass layers to run: research, "
            "grounding, community, deep_research, linkedin. Default runs all."
        ),
    )
    parser.add_argument(
        "--synthesizer",
        choices=["openai", "anthropic"],
        help="Force the synthesis backend (default: auto by available key).",
    )
    parser.add_argument(
        "--verify-budget",
        type=float,
        help="Override the depth profile's first-pass verification budget (USD).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip citation verification.",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Skip the recovery pass after weak verification.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Override the profile's refinement iteration cap (0 disables the loop).",
    )
    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Force the deep-research layer at any depth (needs a Perplexity key).",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        metavar="SECONDS",
        help=(
            "Global wall-clock budget in seconds for the whole run (default: "
            "unbounded, or POLYSEARCH_TIME_BUDGET_S). A layer that would exceed "
            "the remaining budget is cancelled and reported, not left to run "
            "unbounded."
        ),
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Print the classifier verdict as JSON and exit (no network).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print credential / tier / layer / output-dir status and exit (no network).",
    )
    parser.add_argument(
        "--synthesize-parallel",
        metavar="GLOB",
        help="Cross-synthesize existing report files matching GLOB into one report.",
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    updates: dict = {}
    if args.output_dir:
        updates["output_dir"] = Path(args.output_dir)
    if getattr(args, "deep_research", False):
        # --deep-research forces the layer at any depth; the provider still needs a
        # Perplexity key, but enabling it here is what lets the provider be built.
        updates["enable_deep_research"] = True
    return replace(settings, **updates) if updates else settings


def _parse_layers(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {token.strip().lower() for token in raw.split(",") if token.strip()}


def _override_synthesizer(
    providers: Providers, settings: Settings, choice: str
) -> Providers:
    """Force the synthesis backend, degrading to a reason-carrying null when the
    forced backend's key is absent."""
    from polysearch.providers.synthesizers import (
        AnthropicSynthesizer,
        OpenAISynthesizer,
    )

    if choice == "openai":
        providers.synthesizer = (
            OpenAISynthesizer(settings)
            if settings.openai_api_key
            else NullSynthesizer("OPENAI_API_KEY not set (forced --synthesizer openai)")
        )
    else:
        providers.synthesizer = (
            AnthropicSynthesizer(settings)
            if settings.anthropic_api_key
            else NullSynthesizer(
                "ANTHROPIC_API_KEY not set (forced --synthesizer anthropic)"
            )
        )
    return providers


def _writable(path: Path) -> bool:
    """Whether ``path`` (or its nearest existing ancestor) is writable. Creates
    nothing — safe to call from --diagnose."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return os.access(probe, os.W_OK)


def _layer_status(provider: object | None, *, none_label: str = "inactive") -> str:
    if provider is None:
        return none_label
    reason = getattr(provider, "reason", None)
    if reason:
        return f"null ({reason})"
    if getattr(provider, "active", True) is False:
        return "inactive"
    return "active"


def _diagnose(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    providers = build_providers(settings)
    tier = resolve_install_tier(settings)

    lines = ["polysearch diagnose", f"  install tier: {tier}", "  credentials:"]
    creds = [
        ("PERPLEXITY_API_KEY", settings.perplexity_api_key),
        ("FIRECRAWL_API_KEY", settings.firecrawl_api_key),
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        ("BRAVE_API_KEY", settings.brave_api_key),
        ("SCRAPECREATORS_API_KEY", settings.scrapecreators_api_key),
        ("YOUTUBE_API_KEY", settings.youtube_api_key),
        ("GITHUB_TOKEN", settings.github_token),
        ("REDDIT_CLIENT_ID", settings.reddit_client_id),
    ]
    for label, value in creds:
        lines.append(f"    {label}: {'present' if value else 'missing'}")

    lines.append("  layers:")
    lines.append(f"    research: {_layer_status(providers.research)}")
    lines.append(f"    grounding: {_layer_status(providers.grounder)}")
    lines.append(
        f"    deep_research: {_layer_status(providers.deep_research, none_label='inactive (not enabled)')}"
    )
    lines.append(f"    synthesizer: {_layer_status(providers.synthesizer)}")
    lines.append(f"    verifier: {_layer_status(providers.verifier)}")
    lines.append(f"    community sources: {len(providers.community_sources)} active")
    lines.append(
        f"    linkedin: {_layer_status(providers.linkedin, none_label='inactive (no key)')}"
    )

    out_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
    lines.append(
        f"  output dir {out_dir}: {'writable' if _writable(out_dir) else 'NOT writable'}"
    )
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --classify: verdict JSON, no network.
    if args.classify:
        if not args.topic:
            print("polysearch: --classify requires --topic", file=sys.stderr)
            return 2
        print(json.dumps(classify(args.topic).model_dump(), indent=2))
        return 0

    # --diagnose: status only, no network.
    if args.diagnose:
        return _diagnose(args)

    # --synthesize-parallel: cross-report synthesis over existing files.
    if args.synthesize_parallel:
        settings = _settings_from_args(args)
        out_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
        return asyncio.run(
            run_parallel_synthesis(
                args.synthesize_parallel, output_dir=out_dir, settings=settings
            )
        )

    # A real run needs a topic.
    if not args.topic:
        parser.print_help()
        return 2

    settings = _settings_from_args(args)
    try:
        providers = build_providers(settings)
        if args.synthesizer:
            providers = _override_synthesizer(providers, settings, args.synthesizer)
        report = asyncio.run(
            run_research(
                args.topic,
                depth=args.depth,
                settings=settings,
                providers=providers,
                enabled_layers=_parse_layers(args.providers),
                verify=not args.no_verify,
                recovery=not args.no_recovery,
                verify_budget=args.verify_budget,
                deep_research=args.deep_research,
                max_iterations=args.max_iterations,
                time_budget_s=args.time_budget,
            )
        )
    except Exception as exc:  # noqa: BLE001 — top-level guard: map to exit code 1
        print(f"polysearch: pipeline error: {exc}", file=sys.stderr)
        return 1
    _log_layer_durations(report)
    return 0


def _log_layer_durations(report: object) -> None:
    """Cheap stderr instrumentation for future timeout diagnosis: one line per
    layer naming how long it ran, in the order layers were produced. Never
    raises — a malformed/stubbed report in a test just logs nothing."""
    layers = getattr(report, "layers", None) or []
    for layer in layers:
        name = getattr(layer, "layer", "?")
        duration_ms = getattr(layer, "duration_ms", 0) or 0
        print(f"polysearch: layer {name} finished in {duration_ms / 1000:.1f}s", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
