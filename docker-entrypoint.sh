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
#   LLAMA_HERD_STREAMS       concurrent generations               (default: 4)
#   LLAMA_HERD_BATCH         tokens per decode pass               (default: 2048)
#   LLAMA_HERD_GPU_LAYERS    -1 offloads everything               (default: -1)
#   LLAMA_HERD_LOAD_MTP      load multi-token-prediction layers   (default: false)
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
    curl -fL --retry 3 --retry-delay 5 -o "$tmp" "$LLAMA_HERD_MODEL_URL"
    mv "$tmp" "$model_file"
    echo "entrypoint: fetched $(du -h "$model_file" | cut -f1)"
  else
    echo "entrypoint: reusing cached $model_file"
  fi

  cat > "$MANIFEST" <<JSON
{
  "listen": ":8080",
  "models": [
    {
      "name": "${LLAMA_HERD_MODEL_NAME:-chat}",
      "path": "$model_file",
      "gpu_layers": ${LLAMA_HERD_GPU_LAYERS:--1},
      "context": ${LLAMA_HERD_CONTEXT:-8192},
      "batch": ${LLAMA_HERD_BATCH:-2048},
      "streams": ${LLAMA_HERD_STREAMS:-4},
      "load_mtp": ${LLAMA_HERD_LOAD_MTP:-false}
    }
  ]
}
JSON
  echo "entrypoint: wrote $MANIFEST"
  cat "$MANIFEST"
fi

exec llama-herd "$@"
