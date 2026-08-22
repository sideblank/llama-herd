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
  if [ "${LLAMA_HERD_LIBBENCH:-0}" = "1" ] && [ -x /opt/llama-herd/bin/llama-bench ]; then
    echo "entrypoint: measuring the library with llama-bench (this is not the engine)"
    /opt/llama-herd/bin/llama-bench \
      -m "$model_file" \
      -p 512 -n 128 -r 3 \
      -ngl "${LLAMA_HERD_GPU_LAYERS:--1}" \
      -ctk "${LLAMA_HERD_KV_TYPE_K:-f16}" -ctv "${LLAMA_HERD_KV_TYPE_V:-f16}" \
      -fa "${LLAMA_HERD_FLASH_ATTN:-auto}" 2>&1 | sed 's/^/  libbench: /' || \
      echo "  libbench: failed — continuing to serve"
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
fi

exec llama-herd "$@"
