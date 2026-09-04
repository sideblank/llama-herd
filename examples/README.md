# Example manifests

`manifest.json` is a working two-model configuration. Two settings in it are not obvious and
are worth understanding before copying it.

## `kv_unified` decides whether the herd amortises at all

This is the single largest throughput setting here, and it is not about capacity.

The inference library chooses how to split a batch on this flag. With **one pool** it runs every
stream's tokens in a single forward pass. With **a pool per stream** it runs one pass *per
sequence* — so six streams cost six passes, and sharing one resident copy of the weights buys
nothing, which is the entire premise of this engine.

Measured on a 3090 with a 35B-A3B at four streams: **182 tok/s with one pool against 55 with
four.** Nothing in the serving metrics reports it. Tokens-per-pass counts the batch handed to
the library, not what the library did with it, so it reads a healthy 3.88-of-4 either way.

The reading that exposes it is aggregate throughput against the same model's **single-stream**
rate. A herd no faster than one of its members is not amortising. The startup selftest reports
exactly that ratio:

```bash
curl -s localhost:8080/v1/info | jq '.models[0].selftest'
```

## `admit_context` is what makes one pool safe

With a shared pool a per-stream ceiling stops reserving anything — any one request can claim
the whole cache, and several admitted on that basis evict their way out of it, which a caller
sees as an answer truncating for no stated reason.

`admit_context` caps what a single request may occupy, below the per-stream share. Cache memory
is reserved up front from `context` whether or not requests can fill it, so admitting less costs
nothing and buys a guarantee: a stream that cannot reach the end of its window cannot be evicted
from it. The refusal happens at submit time, with a number, instead of mid-answer.

In `manifest.json` the `chat` model allocates a 384k pool across 6 streams and admits 60k each —
360k of 384k, leaving slack no request can consume. The manifest refuses `kv_unified` with
several streams unless such a cap is set.

## Speculation is off here deliberately

`load_mtp` is false and no `speculation` block is set. On a model whose prediction head predicts
one token ahead, drafting is **slower** than not drafting: each drafted token costs a full
decode call, so a step issues four calls to produce roughly 1.7 tokens. Measured at 57%
acceptance it was 2.1x slower than plain decoding.

`docs/INVARIANTS.md` carries the arithmetic. Measure before enabling it, and judge it on decode
calls per token rather than on acceptance rate.

## `3090-throughput.json` — the measured high-throughput profile

The configuration that measured **728.71 tok/s aggregate** on one RTX 3090 with
`Qwen3.6-35B-A3B-UD-IQ3_S`. Numbers, method and caveats: `docs/results/3090.md`.

Build the image with `--build-arg GGML_CUDA_FORCE_MMQ=ON`; that was worth 7-9% and is not the
upstream default.

Three things this profile assumes, which decide whether it is the right one:

- **Shallow requests.** It was measured with near-empty caches — agent swarms, short independent
  requests, high-concurrency chat. At 16k per stream the deepest herd that fits is 24 streams, at
  around 52 tok/s, where **8 streams beats 24 by more than 2x**. Serve deep contexts and this is the wrong profile.
- **`admit_context` below the per-stream share.** 48 streams over 425,984 gives each 8,874, so
  8,192 admitted leaves 682 tokens to generate into. Admit the whole share and a request that
  fills its stream has nowhere to put its answer.
- **`load_mtp: false`.** The head costs VRAM and speculation measured net-negative on this model,
  so this profile does not carry it. Set it true only alongside a `speculation` block, and only
  after measuring — see the roadmap.
