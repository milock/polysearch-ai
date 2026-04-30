"""polysearch CLI entry point.

This is a skeleton for v0.1.0 Phase 0. The orchestrator and providers
land in subsequent phases. For now `polysearch --version` and a help
message work; `--topic` returns a not-implemented exit code so
downstream scripts can detect the placeholder reliably.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


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
        default="./reports",
        help="Where to write the research report (default: ./reports/).",
    )
    parser.add_argument(
        "--providers",
        help="Comma-separated provider override (advanced; defaults resolve from env).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.topic:
        parser.print_help()
        return 0

    print(
        f"polysearch {__version__}: pipeline orchestrator not implemented in this build.\n"
        "This is the Phase 0 skeleton. Provider implementations land in Phase 1+.\n"
        f"You asked: --topic {args.topic!r} --depth {args.depth} --output-dir {args.output_dir}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
