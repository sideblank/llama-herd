#!/usr/bin/env bash
# Commit-policy check. Single source of truth for CI and local use.
#
#   scripts/check-commits.sh [<range>]
#
# With no argument, checks every commit reachable from HEAD.
# Exits non-zero and prints every violation found (it does not stop at the first).
set -uo pipefail

REQUIRED_EMAIL="benjamin.goldman@gmail.com"
range="${1:-}"

if [ -n "$range" ]; then
  commits="$(git rev-list "$range")" || { echo "check-commits: bad range '$range'" >&2; exit 2; }
else
  commits="$(git rev-list HEAD)"
fi

if [ -z "$commits" ]; then
  echo "check-commits: no commits in range — nothing to check."
  exit 0
fi

fail=0
n=0

while read -r sha; do
  [ -n "$sha" ] || continue
  n=$((n + 1))
  subject="$(git log -1 --format=%s "$sha")"
  msg="$(git log -1 --format=%B "$sha")"
  ae="$(git log -1 --format=%ae "$sha")"
  ce="$(git log -1 --format=%ce "$sha")"

  bad=""

  if printf '%s\n' "$msg" | grep -qiE '^Co-Authored-By:'; then
    bad="${bad}    - Co-Authored-By trailer (commits here have a single author)\n"
  fi
  if printf '%s\n' "$msg" | grep -qiE '^(Claude-Session:|Generated with)'; then
    bad="${bad}    - tool attribution line (session link / \"Generated with ...\")\n"
  fi
  if printf '%s\n' "$msg" | grep -qiE 'noreply@anthropic\.com|claude\.(ai|com)'; then
    bad="${bad}    - assistant address or URL in the message\n"
  fi
  if [ "$ae" != "$REQUIRED_EMAIL" ]; then
    bad="${bad}    - author email is '$ae' (must be $REQUIRED_EMAIL)\n"
  fi
  if [ "$ce" != "$REQUIRED_EMAIL" ]; then
    bad="${bad}    - committer email is '$ce' (must be $REQUIRED_EMAIL)\n"
  fi

  if [ -n "$bad" ]; then
    fail=1
    echo "FAIL ${sha:0:8}  $subject"
    printf "$bad"
  fi
done <<< "$commits"

# Agent-tooling files must never be tracked, at any commit in the range.
forbidden_re='^(CLAUDE\.md|CLAUDE\.local\.md|\.claude/|\.claude\.json|\.claude\.local\.json|\.mcp\.json|\.cursorrules|\.cursor/)'
while read -r sha; do
  [ -n "$sha" ] || continue
  hits="$(git ls-tree -r --name-only "$sha" | grep -E "$forbidden_re" || true)"
  if [ -n "$hits" ]; then
    fail=1
    echo "FAIL ${sha:0:8}  tracks agent-tooling files that must be gitignored:"
    printf '    - %s\n' $hits
  fi
done <<< "$commits"

if [ "$fail" -ne 0 ]; then
  echo
  echo "Commit policy violated. See CONTRIBUTING.md."
  echo "History already pushed? Rewrite the offending commits and force-push."
  exit 1
fi

echo "check-commits: OK — $n commit(s) pass policy."
