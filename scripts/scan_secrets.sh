#!/usr/bin/env bash
# Pre-submission secret scan. Verifies no keys are tracked OR in git history
# (this repo had a key-exposure incident before — be strict). Exit non-zero on
# any hit so it can gate `package.sh`.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

# Google AIza... | Anthropic sk-ant-... | OpenAI sk-...(40+) | Tavily tvly-...
PAT='AIza[0-9A-Za-z_-]{30,}|sk-ant-[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{40,}|tvly-[0-9A-Za-z]{20,}'
rc=0

echo "== 1. .env / CLAUDE.md must NOT be tracked =="
if git ls-files | grep -nE '(^|/)\.env$|(^|/)CLAUDE\.md$'; then
  echo "  !! ERROR: an ignored/secret file is TRACKED"; rc=1
else
  echo "  ok — .env and CLAUDE.md are not tracked"
fi

echo "== 2. tracked content (excluding .env.example) =="
if git grep -nIE "$PAT" -- . ':!.env.example' >/tmp/_sec1 2>/dev/null; then
  echo "  !! possible key in tracked files:"; cat /tmp/_sec1; rc=1
else
  echo "  ok — no key-shaped strings in tracked files"
fi

echo "== 3. full git history =="
if git log -p --all 2>/dev/null | grep -nIE "$PAT" >/tmp/_sec2 2>/dev/null; then
  echo "  !! possible key in HISTORY (review; may need git filter-repo / BFG):"
  head -20 /tmp/_sec2; rc=1
else
  echo "  ok — no key-shaped strings in history"
fi

rm -f /tmp/_sec1 /tmp/_sec2
[ "$rc" -eq 0 ] && echo ">> secret scan CLEAN." || echo ">> secret scan FAILED — fix before submitting."
exit "$rc"
