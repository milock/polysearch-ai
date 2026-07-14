#!/bin/sh
# polysearch installer — installs the Python package and drops the Claude Code
# research skill into ~/.claude/skills/research.
#
# Idempotent: safe to run repeatedly. Re-running upgrades the package and
# refreshes the skill copy.
#
# Usage:
#   ./install.sh

set -eu

# Resolve the directory this script lives in, so it works from any cwd.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "polysearch installer"
echo "===================="

# 1. Install the Python package (editable, from this checkout).
if command -v pip >/dev/null 2>&1; then
  PIP="pip"
elif command -v pip3 >/dev/null 2>&1; then
  PIP="pip3"
else
  echo "error: pip not found. Install Python 3.12+ and pip, then re-run." >&2
  exit 1
fi

echo "Installing the polysearch package with $PIP ..."
"$PIP" install -e "$SCRIPT_DIR"

# 2. Copy the Claude Code skill into ~/.claude/skills/research.
SKILL_SRC="$SCRIPT_DIR/skills/research"
SKILL_DEST="${HOME}/.claude/skills/research"

if [ -d "$SKILL_SRC" ]; then
  echo "Installing the research skill to $SKILL_DEST ..."
  mkdir -p "$SKILL_DEST"
  cp -R "$SKILL_SRC/." "$SKILL_DEST/"
else
  echo "warning: $SKILL_SRC not found; skipping skill install." >&2
fi

# 3. Print next steps for credentials.
echo ""
echo "Done."
echo ""
echo "Next: add at least one API key so the pipeline has a layer to run."
echo "Create a .env file in the directory you run polysearch from:"
echo ""
echo "  echo 'PERPLEXITY_API_KEY=your_key_here' > .env"
echo ""
echo "See .env.example for every supported key (Firecrawl, OpenAI/Anthropic,"
echo "vector search, and tuning knobs). Then try:"
echo ""
echo "  polysearch --diagnose"
echo "  polysearch --topic \"What is the current US federal funds rate?\" --depth quick"
echo ""
