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
PROJECT_EMAIL="benjamin.goldman@gmail.com"          # the maintainer identity
ALLOWED_TRAILERS="Signed-off-by Refs Fixes Closes"
ALLOWED_URL_HOSTS="github.com huggingface.co"
REQUIRE_SIGNOFF=1                              # DCO: every commit needs a matching sign-off

# Set to 0 BEFORE the repo goes public and starts accepting outside pull requests.
# While 1, every commit must be authored by PROJECT_EMAIL — correct for a solo private
# repo, but it would reject every external contributor once the project is open.
SOLO_IDENTITY=1
# ----------------------------------------------------------------------------

in_list() { local n="$1"; shift; local i; for i in $*; do [ "${n,,}" = "${i,,}" ] && return 0; done; return 1; }

# check_message <msg> [extra_allowed_email]
check_message() {
  local msg="$1" extra="${2:-}" bad="" key host email

  while IFS= read -r line; do
    [ -n "$line" ] || continue
    key="${line%%:*}"
    in_list "$key" $ALLOWED_TRAILERS || \
      bad="${bad}    - trailer '${key}:' is not permitted (allowed: ${ALLOWED_TRAILERS// /, })\n"
  done < <(printf '%s\n' "$msg" | git interpret-trailers --parse 2>/dev/null)

  while IFS= read -r email; do
    [ -n "$email" ] || continue
    [ "${email,,}" = "${PROJECT_EMAIL,,}" ] && continue
    [ -n "$extra" ] && [ "${email,,}" = "${extra,,}" ] && continue   # the author's own sign-off
    bad="${bad}    - address '${email}' is not permitted here\n"
  done < <(printf '%s\n' "$msg" | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' | sort -u)

  while IFS= read -r host; do
    [ -n "$host" ] || continue
    in_list "$host" $ALLOWED_URL_HOSTS || \
      bad="${bad}    - link to '${host}' is not permitted (allowed: ${ALLOWED_URL_HOSTS// /, })\n"
  done < <(printf '%s\n' "$msg" | grep -oiE 'https?://[^][ )>,"'"'"']+' | sed -E 's#^[a-zA-Z]+://##; s#[/:?].*$##' | sort -u)

  [ -z "$bad" ] && return 0
  printf "$bad"
  return 1
}

# signoff_ok <msg> <author_email>
signoff_ok() {
  [ "$REQUIRE_SIGNOFF" -eq 1 ] || return 0
  printf '%s\n' "$1" | git interpret-trailers --parse 2>/dev/null \
    | grep -iE '^Signed-off-by:' \
    | grep -oE '<[^>]+>' | tr -d '<>' \
    | grep -qxiF "$2"
}

# ---- single-message mode (commit-msg hook) ---------------------------------
if [ "${1:-}" = "--message" ]; then
  [ -n "${2:-}" ] || { echo "usage: $0 --message <file>" >&2; exit 2; }
  msg="$(cat "$2")"; me="$(git config user.email || true)"
  out="$(check_message "$msg" "$me")"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "commit-msg: rejected — only the listed trailers, addresses, and link hosts are allowed." >&2
    printf "%b\n" "$out" >&2; echo "See CONTRIBUTING.md." >&2; exit 1
  fi
  if ! signoff_ok "$msg" "$me"; then
    echo "commit-msg: rejected — missing DCO sign-off for <$me>." >&2
    echo "Commit with -s, or add:  Signed-off-by: $(git config user.name) <$me>" >&2
    echo "See .github/DCO and CONTRIBUTING.md." >&2; exit 1
  fi
  exit 0
fi

# ---- range mode ------------------------------------------------------------
range="${1:-}"
if [ -n "$range" ]; then
  commits="$(git rev-list "$range")" || { echo "check-commits: bad range '$range'" >&2; exit 2; }
else
  commits="$(git rev-list HEAD)"
fi
[ -n "$commits" ] || { echo "check-commits: no commits in range."; exit 0; }

fail=0; n=0
while read -r sha; do
  [ -n "$sha" ] || continue
  n=$((n + 1))
  msg="$(git log -1 --format=%B "$sha")"
  ae="$(git log -1 --format=%ae "$sha")"; ce="$(git log -1 --format=%ce "$sha")"
  bad="$(check_message "$msg" "$ae")"

  if [ "$SOLO_IDENTITY" -eq 1 ]; then
    [ "$ae" = "$PROJECT_EMAIL" ] || bad="${bad}    - author email '${ae}' is not the project identity (${PROJECT_EMAIL})\n"
    [ "$ce" = "$PROJECT_EMAIL" ] || bad="${bad}    - committer email '${ce}' is not the project identity (${PROJECT_EMAIL})\n"
  fi

  signoff_ok "$msg" "$ae" || \
    bad="${bad}    - missing DCO sign-off matching the author <${ae}> (commit with -s)\n"

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
