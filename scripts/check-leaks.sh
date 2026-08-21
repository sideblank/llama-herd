#!/usr/bin/env bash
# Blocks internal identifiers and credentials from entering the tree.
#
#   scripts/check-leaks.sh [<path>...]      default: all tracked files
#
# Two pattern sources:
#   1. STRUCTURAL patterns below — shapes, not names. Safe to publish.
#   2. NAME patterns supplied out-of-tree, because a list of your internal service names
#      must never be committed to a repo that will be public:
#        - local:  a gitignored .internal-patterns file (one regex per line, # for comments)
#        - CI:     the INTERNAL_PATTERNS env var, fed from a repository secret
#
# Mark a deliberate match with a trailing:  leak-check: ignore
set -uo pipefail

STRUCTURAL=(
  'private IPv4:\b(10\.[0-9]{1,3}|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b'
  'internal hostname:[A-Za-z0-9_-]+\.(internal|intranet|corp|lan)\b'
  'k8s cluster DNS:\.svc\.cluster\.local'
  'AWS access key:\b(AKIA|ASIA)[0-9A-Z]{16}\b'
  'GitHub token:\b(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'
  'Slack token:\bxox[baprs]-[A-Za-z0-9-]{10,}'
  'private key block:-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'JSON web token:\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}'
  'bearer literal:[Bb]earer[[:space:]]+[A-Za-z0-9._-]{24,}'
  'inline secret:(secret|token|password|passwd|api[_-]?key)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{16,}["'"'"']'
)

SELF='^scripts/check-leaks\.sh$'
files=()
if [ "$#" -gt 0 ]; then files=("$@"); else mapfile -t files < <(git ls-files); fi

fail=0
scan() {                       # scan <label> <regex>
  local label="$1" re="$2" f hit
  for f in "${files[@]}"; do
    [[ "$f" =~ $SELF ]] && continue
    [ -f "$f" ] || continue
    grep -Iq . "$f" 2>/dev/null || continue          # skip binaries
    while IFS= read -r hit; do
      [ -n "$hit" ] || continue
      case "$hit" in *"leak-check: ignore"*) continue;; esac
      fail=1
      echo "LEAK  $f:${hit%%:*}  [$label]"
      printf '      %s\n' "$(printf '%s' "${hit#*:}" | cut -c1-120)"
    done < <(grep -nEIi "$re" "$f" 2>/dev/null)
  done
}

for entry in "${STRUCTURAL[@]}"; do
  scan "${entry%%:*}" "${entry#*:}"
done

extra=""
[ -f .internal-patterns ] && extra="$(grep -vE '^[[:space:]]*(#|$)' .internal-patterns || true)"
[ -n "${INTERNAL_PATTERNS:-}" ] && extra="$extra"$'\n'"$(printf '%s' "$INTERNAL_PATTERNS" | grep -vE '^[[:space:]]*(#|$)' || true)"

if [ -n "${extra// /}" ]; then
  n=0
  while IFS= read -r re; do
    [ -n "${re// /}" ] || continue
    n=$((n+1)); scan "internal name" "$re"
  done <<< "$extra"
  echo "check-leaks: applied $n out-of-tree name pattern(s)."
else
  echo "check-leaks: no out-of-tree name patterns configured (structural checks only)."
  echo "            add .internal-patterns locally, or set the INTERNAL_PATTERNS secret in CI."
fi

if [ "$fail" -ne 0 ]; then
  echo; echo "Internal identifiers or credentials found. Remove them before committing."
  echo "If a match is deliberate, append:  leak-check: ignore"
  exit 1
fi
echo "check-leaks: OK — ${#files[@]} file(s) clean."
