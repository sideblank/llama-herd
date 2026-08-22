#!/usr/bin/env bash
# Measure what the KV pool layout is worth on this machine, with this model.
#
# The library decides how to split a batch on `kv_unified`: one pool runs every stream's tokens
# in a single forward pass, a pool per stream runs one pass PER SEQUENCE. The second costs the
# herd its whole reason for existing, and nothing in the serving metrics reports it — tokens per
# pass counts the batch handed over, not what was done with it, and reads the same either way.
#
# So measure it. Two manifests differing in one field, run back to back on one machine, several
# times. Back to back matters: rented hardware varies by more between nodes than this is worth.
#
#   ./scripts/kv-pool-ab.sh /path/to/model.gguf [streams] [reps]
set -euo pipefail

MODEL="${1:?usage: kv-pool-ab.sh <model.gguf> [streams] [reps]}"
STREAMS="${2:-4}"
REPS="${3:-3}"
CTX=$((STREAMS * 4096))
ADMIT=$((CTX / STREAMS - 512))   # a little under the share, so the pool cannot be oversubscribed
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for cfg in split unified; do
  extra=""
  [ "$cfg" = unified ] && extra=", \"kv_unified\": true, \"admit_context\": $ADMIT"
  cat > "$TMP/$cfg.json" <<JSON
{ "listen": "127.0.0.1:0",
  "models": [ { "name": "ab", "path": "$MODEL", "context": $CTX, "batch": 512,
                "streams": $STREAMS $extra } ] }
JSON
done

echo "model   : $MODEL"
echo "streams : $STREAMS   pool: $CTX   admit (unified): $ADMIT"
echo
for rep in $(seq 1 "$REPS"); do
  for cfg in split unified; do
    out="$TMP/out-$cfg-$rep.txt"
    # The selftest runs before the listener, so the line appears whether or not serving starts.
    timeout 300 llama-herd serve --manifest "$TMP/$cfg.json" > "$out" 2>&1 &
    pid=$!
    for _ in $(seq 1 150); do
      grep -qa 'selftest:' "$out" && break
      sleep 2
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    printf '%-8s rep%s  %s\n' "$cfg" "$rep" \
      "$(grep -a 'selftest:' "$out" | head -1 | sed 's/.*selftest: //')"
  done
done
echo
echo "Compare the aggregate figures, not tokens per pass — that reads the same in both."
