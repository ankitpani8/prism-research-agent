#!/usr/bin/env bash
# Package the source as a clean zip for submission. Uses `git archive`, so ONLY
# committed/tracked files are included — .env, CLAUDE.md, .venv, __pycache__, and
# data/local are gitignored and therefore excluded automatically. Commit first.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT="prism-research-agent.zip"
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: you have uncommitted changes — they will NOT be in the zip."
  echo "         commit first so the package matches the repo. Continuing in 3s..."
  sleep 3
fi

rm -f "$OUT"
git archive --format=zip --prefix=prism-research-agent/ -o "$OUT" HEAD

echo ">> wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo ">> sanity — key files present, secrets absent:"
unzip -l "$OUT" | grep -E 'README.md|core/graph.py|api/app.py|requirements.txt|sample_dashboard.png' || true
if unzip -l "$OUT" | grep -qE '\.env$|CLAUDE.md'; then
  echo "!! ERROR: secret/ignored file leaked into the zip"; exit 1
fi
echo ">> clean. No .env / CLAUDE.md in the archive."
