#!/usr/bin/env bash
# Commit-policy check. Single source of truth for the hooks and CI.
#
#   scripts/check-commits.sh [<range>]      validate commits (default: all of HEAD)
#   scripts/check-commits.sh --message <f>  validate one prepared commit message
#
# Policy is expressed as ALLOW-lists: anything not explicitly permitted is rejected.
# Nothing here names a vendor, so no new tool needs a new rule.
set -uo pipefail

# ---- policy ----------------------------------------------------------------
ALLOWED_EMAIL="benjamin.goldman@gmail.com"          # the only identity that may author or commit
ALLOWED_TRAILERS="Signed-off-by Refs Fixes Closes"
ALLOWED_URL_HOSTS="github.com huggingface.co"
# ----------------------------------------------------------------------------

in_list() { local n="$1"; shift; local i; for i in $*; do [ "${n,,}" = "${i,,}" ] && return 0; done; return 1; }

# Message checks. $1 = message text, $2 = label for output. Echoes findings, returns 1 if any.
check_message() {
  local msg="$1" bad="" key host email

  # 1. Trailers: only the allow-listed keys may appear.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    key="${line%%:*}"
    in_list "$key" $ALLOWED_TRAILERS || \
      bad="${bad}    - trailer '${key}:' is not permitted (allowed: ${ALLOWED_TRAILERS// /, })\n"
  done < <(printf '%s\n' "$msg" | git interpret-trailers --parse 2>/dev/null)

  # 2. Email addresses: only the project identity may appear anywhere in the message.
  while IFS= read -r email; do
    [ -n "$email" ] || continue
    [ "${email,,}" = "${ALLOWED_EMAIL,,}" ] || \
      bad="${bad}    - address '${email}' is not permitted (allowed: ${ALLOWED_EMAIL})\n"
  done < <(printf '%s\n' "$msg" | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' | sort -u)

  # 3. URLs: only the allow-listed hosts may be linked.
  while IFS= read -r host; do
    [ -n "$host" ] || continue
    in_list "$host" $ALLOWED_URL_HOSTS || \
      bad="${bad}    - link to '${host}' is not permitted (allowed: ${ALLOWED_URL_HOSTS// /, })\n"
  done < <(printf '%s\n' "$msg" | grep -oiE 'https?://[^][ )>,"'"'"']+' | sed -E 's#^[a-zA-Z]+://##; s#[/:?].*$##' | sort -u)

  [ -z "$bad" ] && return 0
  printf "$bad"
  return 1
}

# ---- single-message mode (used by the commit-msg hook) ----------------------
if [ "${1:-}" = "--message" ]; then
  [ -n "${2:-}" ] || { echo "usage: $0 --message <file>" >&2; exit 2; }
  out="$(check_message "$(cat "$2")")" && exit 0
  echo "commit-msg: rejected — this repo allows only the listed trailers, addresses, and link hosts." >&2
  printf "%b\n" "$out" >&2
  echo "See CONTRIBUTING.md." >&2
  exit 1
fi

# ---- range mode ------------------------------------------------------------
range="${1:-}"
if [ -n "$range" ]; then
  commits="$(git rev-list "$range")" || { echo "check-commits: bad range '$range'" >&2; exit 2; }
else
  commits="$(git rev-list HEAD)"
fi
[ -n "$commits" ] && [ "$commits" != "" ] || { echo "check-commits: no commits in range."; exit 0; }

fail=0; n=0
while read -r sha; do
  [ -n "$sha" ] || continue
  n=$((n + 1))
  bad=""
  bad="${bad}$(check_message "$(git log -1 --format=%B "$sha")")"

  ae="$(git log -1 --format=%ae "$sha")"; ce="$(git log -1 --format=%ce "$sha")"
  [ "$ae" = "$ALLOWED_EMAIL" ] || bad="${bad}    - author email '${ae}' is not the project identity (${ALLOWED_EMAIL})\n"
  [ "$ce" = "$ALLOWED_EMAIL" ] || bad="${bad}    - committer email '${ce}' is not the project identity (${ALLOWED_EMAIL})\n"

  # Tracked files must obey .gitignore. The ignore file is the only place names live.
  hits="$(git ls-tree -r --name-only "$sha" | git check-ignore --no-index --stdin 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    while IFS= read -r h; do
      [ -n "$h" ] && bad="${bad}    - tracks '${h}', which .gitignore excludes\n"
    done <<< "$hits"
  fi

  if [ -n "$bad" ]; then
    fail=1
    echo "FAIL ${sha:0:8}  $(git log -1 --format=%s "$sha")"
    printf "%b" "$bad"
  fi
done <<< "$commits"

if [ "$fail" -ne 0 ]; then
  echo; echo "Commit policy violated. See CONTRIBUTING.md."; exit 1
fi
echo "check-commits: OK — $n commit(s) pass policy."
