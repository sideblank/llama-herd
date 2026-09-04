#!/usr/bin/env bash
# Container entrypoint.
#
# The image ships no model, so a container needs one before it can serve. Two ways in:
# mount a manifest at /etc/llama-herd/manifest.json, or set the environment variables below
# and let this script fetch a model and write the manifest for you.
#
#   LLAMA_HERD_MODEL_URL     GGUF to download (required unless a manifest is mounted)
#   LLAMA_HERD_MODEL_NAME    name requests address it by          (default: chat)
#   LLAMA_HERD_CONTEXT       total context across streams         (default: 8192)
#   LLAMA_HERD_ADMIT_CONTEXT what one request may occupy, below the per-stream share.
#                            The gap is slack no caller can consume: it keeps a stream from
#                            reaching the end of its window, and is spendable on the
#                            deployment's own tokens.
#   LLAMA_HERD_STREAMS       concurrent generations               (default: 4)
#   LLAMA_HERD_BATCH         tokens per decode pass               (default: 2048)
#   LLAMA_HERD_GPU_LAYERS    -1 offloads everything               (default: -1)
#   LLAMA_HERD_LOAD_MTP      load multi-token-prediction layers   (default: false)
#   LLAMA_HERD_KV_TYPE_K     key cache precision                  (default: f16)
#   LLAMA_HERD_KV_TYPE_V     value cache precision                (default: f16)
#   LLAMA_HERD_FLASH_ATTN    flash attention, required by any quantized KV (default: false)
#   LLAMA_HERD_MMPROJ_URL    multimodal projector, for vision models
#   LLAMA_HERD_KV_UNIFIED    share one KV pool across streams          (default: false)
#   LLAMA_HERD_LIBBENCH      set to "1" to run llama-bench on the model before serving and
#                            log the result. That measures the LIBRARY — prompt processing and
#                            token generation, one sequence — where the startup selftest
#                            measures THIS ENGINE across its configured streams. Having both
#                            from one boot, on one card, is what separates a library that is
#                            slow from a herd that is not amortising.
#   LLAMA_HERD_LIBBENCH_TIMEOUT  seconds before the library bench is killed (default: 900).
#                            It runs before the server binds and holds the card exclusively, so
#                            it is capped: the deployment serves with no library figure rather
#                            than not at all.
#   LLAMA_HERD_LIBBENCH_REPS repetitions per test for the library bench (default: 2)
#   LLAMA_HERD_STANDBY       set to "0" to skip holding the port during preparation
#                            (default: 1). While preparing, /health answers 200 and every
#                            other path answers 503 with the current phase, so a long boot is
#                            not mistaken for an unhealthy instance and can be watched.
#   LLAMA_HERD_SWEEP         sweep arguments, e.g. "--streams 4,6,8 --reps 2". When set, a
#                            configuration matrix is measured against one resident copy of the
#                            weights before serving, and published on /v1/info. Empty (the
#                            default) skips it.
#   LLAMA_HERD_SWEEP_TIMEOUT seconds before the sweep is killed (default: 2400)
#   LLAMA_HERD_SELFTEST      set to "off" to skip the startup measurement (default: on).
#                            It runs the engine at its configured stream count for a few
#                            seconds and publishes the result on /v1/info, which is the only
#                            way to tell a slow card or a changed library from a slow engine.
#   LLAMA_HERD_SPEC_TYPE     speculative draft source: none | lookup | mtp  (default: none)
#                            mtp needs LLAMA_HERD_LOAD_MTP=true and a quant that kept the head
#   LLAMA_HERD_SPEC_MAX      tokens proposed per step                  (default: 4)
#   LLAMA_HERD_SPEC_PATTERN  lookup match length                       (default: 3)
#
# Quantized KV is what makes long context fit — f16 costs twice q8 per token — but it does
# not work without flash attention, so the two are set together or not at all.
set -euo pipefail

MANIFEST=${LLAMA_HERD_MANIFEST:-/etc/llama-herd/manifest.json}
BOOT_STATUS=/tmp/boot-status
STANDBY_PID=""

# Report what the boot is doing, to stdout and to the endpoint standby serves.
#
# The second half is what matters on a host with no log access: without it a boot that takes
# most of an hour is indistinguishable from one that has hung, which cost several wasted waits
# before it was worth fixing.
boot_phase() {
  echo "entrypoint: $1"
  printf '%s' "$1" > "$BOOT_STATUS" 2>/dev/null || true
}

# Hold the port so the platform's health check succeeds while the model is being fetched.
#
# Without this the instance is declared unhealthy minutes into a download that takes far
# longer, and is replaced — discarding the download and starting again. Observed twice in one
# night on boots that were otherwise progressing normally.
start_standby() {
  [ "${LLAMA_HERD_STANDBY:-1}" = "1" ] || return 0
  llama-herd standby --addr "${LLAMA_HERD_STANDBY_ADDR:-:8080}" --status "$BOOT_STATUS" &
  STANDBY_PID=$!
}

# Release the port and wait for it to actually be free, since the real server binds next and a
# half-released listener would fail the bind rather than the health check.
stop_standby() {
  [ -n "$STANDBY_PID" ] || return 0
  kill "$STANDBY_PID" 2>/dev/null || true
  wait "$STANDBY_PID" 2>/dev/null || true
  STANDBY_PID=""
  sleep 1
}
trap 'stop_standby' EXIT

# Only "serve" needs a model. version and doctor must run without one — doctor especially,
# since diagnosing a container that cannot start is precisely when it is wanted.
if [ "${1:-}" != "serve" ]; then
  exec llama-herd "$@"
fi

if [ ! -f "$MANIFEST" ]; then
  if [ -z "${LLAMA_HERD_MODEL_URL:-}" ]; then
    echo "entrypoint: no manifest at $MANIFEST and LLAMA_HERD_MODEL_URL is unset." >&2
    echo "            Mount a manifest or set LLAMA_HERD_MODEL_URL to a GGUF." >&2
    exit 2
  fi

  start_standby
  boot_phase "fetching the model"

  mkdir -p /models "$(dirname "$MANIFEST")"
  model_file="/models/$(basename "${LLAMA_HERD_MODEL_URL%%\?*}")"

  if [ ! -f "$model_file" ]; then
    echo "entrypoint: fetching $LLAMA_HERD_MODEL_URL"
    # Download to a temporary name and move into place only on success, so an
    # interrupted pull cannot leave a truncated file that later looks cached.
    tmp="${model_file}.partial"
    # -C - resumes a partial rather than restarting it. These files run to tens of
    # gigabytes, so a connection dropped at 90% otherwise costs the whole download again,
    # and --retry alone restarts from zero. The partial is named after the target file, so
    # a resume can only ever continue the same model.
    # --speed-limit/--speed-time abort a transfer that is technically alive but moving at
    # nothing. Without them --retry never fires on a stalled connection: curl only retries
    # on failure, and a socket delivering a trickle never fails. On a rented node that is
    # the difference between failing over in a minute and billing by the hour for a
    # download that will not finish.
    curl -fL -C - --retry 5 --retry-delay 5 --retry-all-errors \
         --speed-limit 1048576 --speed-time 60 \
         -o "$tmp" "$LLAMA_HERD_MODEL_URL"
    mv "$tmp" "$model_file"
    echo "entrypoint: fetched $(du -h "$model_file" | cut -f1)"
  else
    echo "entrypoint: reusing cached $model_file"
  fi

  # A vision model needs its projector fetched too; without it the weights are text-only.
  MMPROJ_JSON=""
  if [ -n "${LLAMA_HERD_MMPROJ_URL:-}" ]; then
    mmproj_file="/models/$(basename "${LLAMA_HERD_MMPROJ_URL%%\?*}")"
    if [ ! -f "$mmproj_file" ]; then
      echo "entrypoint: fetching projector $LLAMA_HERD_MMPROJ_URL"
      curl -fL --retry 3 --retry-delay 5 -o "${mmproj_file}.partial" "$LLAMA_HERD_MMPROJ_URL"
      mv "${mmproj_file}.partial" "$mmproj_file"
    fi
    MMPROJ_JSON=",
      \"mmproj_path\": \"$mmproj_file\",
      \"vision_gpu\": true"
  fi

  # Speculation is off unless asked for. Every proposal occupies a batch entry, so it
  # trades batch capacity for tokens that may be rejected — worth it where output repeats
  # context, wasteful where it does not.
  SPEC_JSON=""
  if [ -n "${LLAMA_HERD_SPEC_TYPE:-}" ] && [ "${LLAMA_HERD_SPEC_TYPE}" != "none" ]; then
    # pattern is the n-gram width lookup matches on and means nothing to a trained head.
    # Emitting it anyway would put a knob in the generated config that changes nothing,
    # which is worse than omitting it.
    SPEC_PATTERN=""
    if [ "${LLAMA_HERD_SPEC_TYPE}" = "lookup" ]; then
      SPEC_PATTERN=",
        \"pattern\": ${LLAMA_HERD_SPEC_PATTERN:-3}"
    fi
    SPEC_JSON=",
      \"speculation\": {
        \"type\": \"${LLAMA_HERD_SPEC_TYPE}\",
        \"max_draft\": ${LLAMA_HERD_SPEC_MAX:-4}${SPEC_PATTERN}
      }"
  fi

  ADMIT_JSON=""
  if [ -n "${LLAMA_HERD_ADMIT_CONTEXT:-}" ]; then
    ADMIT_JSON="
      \"admit_context\": ${LLAMA_HERD_ADMIT_CONTEXT},"
  fi

  # The library's own measurement, for comparison against ours. Reported in llama-bench's
  # format so it can also be checked against published figures for this model and quant.
  boot_phase "measuring the library with llama-bench"
  if [ "${LLAMA_HERD_LIBBENCH:-0}" = "1" ] && [ -x /opt/llama-herd/bin/llama-bench ]; then
    echo "entrypoint: measuring the library with llama-bench (this is not the engine)"
    # Hard-capped, because this runs BEFORE the server binds and holds the card exclusively.
    # A diagnostic must never be able to stop the service from starting: if the bench does not
    # finish in the budget it is killed and the deployment serves without a library figure.
    # Observed once at over 80 minutes on a rented node with no way to read stdout, which is
    # indistinguishable from a hang by any check available from outside.
    timeout -k 30 "${LLAMA_HERD_LIBBENCH_TIMEOUT:-900}" \
      /opt/llama-herd/bin/llama-bench \
      -m "$model_file" \
      -p 512 -n 128 -r "${LLAMA_HERD_LIBBENCH_REPS:-2}" \
      -ngl "${LLAMA_HERD_GPU_LAYERS:--1}" \
      -ctk "${LLAMA_HERD_KV_TYPE_K:-f16}" -ctv "${LLAMA_HERD_KV_TYPE_V:-f16}" \
      -fa "${LLAMA_HERD_FLASH_ATTN:-auto}" > /tmp/libbench.txt 2>&1
    rc=$?
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      echo "llama-bench exceeded ${LLAMA_HERD_LIBBENCH_TIMEOUT:-900}s and was killed; serving without it" \
        >> /tmp/libbench.txt
    elif [ $rc -ne 0 ]; then
      echo "llama-bench failed (exit $rc)" >> /tmp/libbench.txt
    fi
    sed 's/^/  libbench: /' /tmp/libbench.txt
    # Also kept on disk and pointed at by an environment variable, because a hosted runtime
    # may give no way to read a container's stdout — which makes a measurement that only logs
    # unreachable in exactly the deployment that needed it.
    export LLAMA_HERD_LIBBENCH_FILE=/tmp/libbench.txt
  fi

  cat > "$MANIFEST" <<JSON
{
  "listen": ":8080",
  "models": [
    {
      "name": "${LLAMA_HERD_MODEL_NAME:-chat}",
      "path": "$model_file",
      "gpu_layers": ${LLAMA_HERD_GPU_LAYERS:--1},
      "context": ${LLAMA_HERD_CONTEXT:-8192},${ADMIT_JSON}
      "batch": ${LLAMA_HERD_BATCH:-2048},
      "streams": ${LLAMA_HERD_STREAMS:-4},
      "load_mtp": ${LLAMA_HERD_LOAD_MTP:-false},
      "kv_type_k": "${LLAMA_HERD_KV_TYPE_K:-f16}",
      "kv_type_v": "${LLAMA_HERD_KV_TYPE_V:-f16}",
      "kv_unified": ${LLAMA_HERD_KV_UNIFIED:-false},
      "flash_attention": ${LLAMA_HERD_FLASH_ATTN:-false}${MMPROJ_JSON}${SPEC_JSON}
    }
  ]
}
JSON
  echo "entrypoint: wrote $MANIFEST"
  cat "$MANIFEST"

  # A configuration sweep, if one was asked for. It runs before serving because it needs the
  # card to itself, and it reuses one resident copy of the weights across every configuration —
  # which is the point, since getting the model onto a rented machine costs 50 to 85 minutes
  # against seconds of measuring.
  #
  # Results are published on /v1/info rather than only logged, because the hosts this runs on
  # offer no way to read a container's stdout, and a sweep that can only be logged is
  # unavailable exactly where it was needed.
  if [ -n "${LLAMA_HERD_SWEEP:-}" ]; then
    boot_phase "sweeping configurations ($LLAMA_HERD_SWEEP)"
    echo "entrypoint: sweeping configurations ($LLAMA_HERD_SWEEP)"
    # shellcheck disable=SC2086
    timeout -k 30 "${LLAMA_HERD_SWEEP_TIMEOUT:-2400}" \
      llama-herd sweep --manifest "$MANIFEST" --json /tmp/sweep.json $LLAMA_HERD_SWEEP \
      2>&1 | sed 's/^/  sweep: /'
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "sweep exceeded ${LLAMA_HERD_SWEEP_TIMEOUT:-2400}s and was killed" > /tmp/sweep-note.txt
    elif [ "$rc" -ne 0 ]; then
      echo "sweep failed (exit $rc)" > /tmp/sweep-note.txt
    fi
    [ -f /tmp/sweep.json ] && export LLAMA_HERD_SWEEP_FILE=/tmp/sweep.json
  fi

  boot_phase "starting the server"
  stop_standby
fi

exec llama-herd "$@"
