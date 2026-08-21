#!/usr/bin/env bash
# Build one model from its spec. The same steps every time, so a new model is a spec file.
#
#   scripts/build-model.sh models/<name>.json [--stage acquire|convert|imatrix|quantize|all]
#
# Every stage has a gate, and a failed gate stops the build rather than producing something
# that looks finished. The gates exist because each failure they catch is silent: a source
# that never had MTP tensors, a conversion that dropped them, a quantization that dropped
# them, or an importance matrix computed against different weights than it will be applied to.
set -euo pipefail

SPEC="${1:?usage: build-model.sh <spec.json> [--stage <stage>]}"
STAGE="all"
[ "${2:-}" = "--stage" ] && STAGE="${3:?--stage needs a value}"

WORK="${MODEL_BUILD_DIR:-$PWD/testing/build}"
HERD="${LLAMA_HERD:-$PWD/testing/llama-herd}"

# Read the spec once into shell variables. Extracting per-field invited quoting mistakes that
# produced empty values and made gates fire for the wrong reason.
eval "$(python3 - "$SPEC" <<'SPECPY'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
def q(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return shlex.quote("" if v is None else str(v))
out = {
    "NAME":     d["name"],
    "REPO":     d["source"]["repo"],
    "REV":      d["source"].get("revision", ""),
    "LICENSE":  d["source"].get("license", ""),
    "KEEP_MTP": d.get("convert", {}).get("keep_mtp", True),
    "MMPROJ":   d.get("convert", {}).get("mmproj", False),
    "OUTTYPE":  d.get("convert", {}).get("outtype", "bf16"),
    "CORPUS":   d.get("imatrix", {}).get("corpus", ""),
}
for k, v in out.items():
    print(f"{k}={q(v)}")
SPECPY
)"

SRC="$WORK/$NAME/src"
GGUF="$WORK/$NAME/$NAME-$OUTTYPE.gguf"
IMATRIX="$WORK/$NAME/imatrix.gguf"
mkdir -p "$WORK/$NAME"

die() { echo "build-model: $*" >&2; exit 1; }
note() { echo; echo "== $* =="; }

# --- gate: licence, before any GPU time is spent -----------------------------------------
[ -n "$LICENSE" ] || die \
  "no licence recorded for $NAME. Resolve redistribution terms before converting: converting
   is cheap, discovering afterwards that a model cannot be redistributed is not."

# --- gate: a pinned revision, or the build is not reproducible ----------------------------
[ -n "$REV" ] && [ "$REV" != "PIN-ME" ] || die \
  "source.revision is not pinned for $NAME. Model repositories are updated in place, so a
   branch name does not identify a build."

stage_acquire() {
  note "acquire $REPO @ $REV"
  [ -d "$SRC" ] || python3 - "$REPO" "$REV" "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(repo_id=repo, revision=rev, local_dir=dest,
                  allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
PY
  echo "  source at $SRC"
}

stage_convert() {
  note "convert to $OUTTYPE"
  local args=(--outfile "$GGUF" --outtype "$OUTTYPE")
  # MTP tensors are included by default; a publisher must opt out. We never opt out.
  [ "$KEEP_MTP" = "true" ] || args+=(--no-nextn)
  python3 "${LLAMA_CPP_SRC:?set LLAMA_CPP_SRC}/convert_hf_to_gguf.py" "$SRC" "${args[@]}"

  if [ "$MMPROJ" = "true" ]; then
    python3 "$LLAMA_CPP_SRC/convert_hf_to_gguf.py" "$SRC" --mmproj \
      --outfile "$WORK/$NAME/mmproj-$NAME.gguf"
  fi

  # Gate: the tensors must actually be there. A declaration in metadata is not evidence.
  "$HERD" inspect "$GGUF" | grep -q "MTP: metadata declares" || {
    [ "$KEEP_MTP" = "true" ] && die \
      "conversion produced no MTP declaration for a model built to keep them"
  }
  echo "  converted: $GGUF"
}

stage_imatrix() {
  note "importance matrix"
  [ -n "$CORPUS" ] && [ "$CORPUS" != "PIN-ME" ] || die \
    "imatrix.corpus is not set. A matrix is only as good as what it saw, and the corpus is
     recorded in the model card so a reader can judge whether it matches their workload."
  # Computed on the converted weights. Any weight-level modification must already have
  # happened, or this describes weights that will not be the ones quantized.
  llama-imatrix -m "$GGUF" -f "$CORPUS" -o "$IMATRIX" -ngl 999
  echo "  imatrix: $IMATRIX"
}

stage_quantize() {
  note "quantize"
  python3 - "$SPEC" <<'PY' | while read -r qtype cards; do
import json,sys
for q in json.load(open(sys.argv[1]))["quants"]:
    print(q["type"], ",".join(q.get("target_cards", [])))
PY
    out="$WORK/$NAME/$NAME-$qtype.gguf"
    echo "  -> $qtype for $cards"
    llama-quantize --imatrix "$IMATRIX" "$GGUF" "$out" "$qtype"

    # Gate: MTP must survive quantization. This fails for different reasons than the
    # conversion gate, which is why both exist.
    if [ "$KEEP_MTP" = "true" ]; then
      "$HERD" inspect "$out" | grep -q "MTP: metadata declares" \
        || die "$qtype lost its MTP declaration"
    fi

    # Report what this quant can actually hold on each target card, so the serve settings
    # in the spec are checked against the file rather than assumed.
    for card in ${cards//,/ }; do
      "$HERD" fit --card "$card" --streams 4 --context 128k "$out" | sed -n '1,12p' | sed 's/^/    /'
    done
  done
}

case "$STAGE" in
  acquire)  stage_acquire ;;
  convert)  stage_convert ;;
  imatrix)  stage_imatrix ;;
  quantize) stage_quantize ;;
  all)      stage_acquire; stage_convert; stage_imatrix; stage_quantize ;;
  *)        die "unknown stage $STAGE" ;;
esac

note "done: $NAME"
