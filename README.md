# llama-herd

A weight-shared, multi-stream inference runtime for llama.cpp — **one GPU, many models, many
concurrent streams**, with a single resident copy of each model's weights.

Paired with a set of quantized model builds tuned for the cards people actually own:
**RTX 3090, 4090, and 5090**.

---

## Why

`llama-server --parallel N` gives you one batched context. That shares weights in *memory*, but
every additional model you want on the card costs you another full weight copy, and your
concurrency story is one model deep.

llama-herd runs a **registry of engines** on one card. Each model is loaded once and driven by a
single decode loop with per-sequence KV isolation, so N streams against a model ride **one**
`llama_decode` per tick — one kernel pass over the resident weights, not N. Several models can be
co-resident and all decoding simultaneously.

The practical result: a 24 GB card hosts a working ensemble — a chat model, a classifier, a guard
model, a coder — all live at once, instead of one model and a lot of idle VRAM.

## How it works

```
one decode-loop goroutine  ── each tick: gather the next token from every active stream
                              into ONE llama_batch (tagged by seq_id), llama_decode once,
                              sample each stream's logits, route tokens back to their streams
N stream goroutines        ── submit {prompt, params}, receive tokens over a channel
Registry                   ── one Engine per model in the manifest, each with its own
                              resident weights + decode loop, sharing the GPU
```

Only the decode loop touches llama.cpp, so concurrent-`llama_decode` thread safety is a non-issue
by construction. Per-sequence KV isolation is llama.cpp-native (`seq_id` masking; a finished stream
frees its cells). Continuous batching means new work is admitted *while* decoding — mixed
prefill+decode in the same batch, with slot recycling.

The engine core is backend-agnostic; the llama.cpp binding sits behind an interface, and an ONNX
lane handles encoder-only models.

## Hardware targets

| Card | VRAM | Arch |
|------|------|------|
| RTX 3090 | 24 GB | Ampere |
| RTX 4090 | 24 GB | Ada |
| RTX 5090 | 32 GB | Blackwell |

These are the build and tuning targets: quantization choice, batch and context sizing, and stream
counts get picked per card rather than assuming a datacenter part.

## Models

A companion Hugging Face org will host quantized builds matched to the table above — each with the
manifest settings (stream count, context, KV budget) that fit the card, so a build drops in without
you re-deriving the numbers.

Not published yet. See Status.

## Status

**Early — the code is being written now.** llama-herd is a clean rewrite, not a port: no code is
carried over from anywhere, down to the llama.cpp binding.

It is also **not a stripped-down edition of a private product.** llama-herd has features its
closed-source predecessor does not, and leaves out that system's platform-specific plumbing
entirely. The two diverge in both directions.

The architecture above is not speculative. The author built a runtime of this design privately and
measured it on a single 3090:

- 4 models co-resident on one card — 22 streams total, 11.2 GB of 24 GB, all four decoding
  simultaneously with zero cross-stream contamination
- ~318 tok/s aggregate decode driving 6 streams of an 8B-class model at Q4 through one context
- 128k chunked prefill; input ceiling is the per-sequence context, not the batch size

Those numbers came from a **different implementation** of this architecture. They are recorded as
the bar this rewrite aims to reproduce — not as a claim about this repo's code, which does not yet
exist. Treat them as a design target, not a benchmark result, until CI can regenerate them here.

See [PROVENANCE.md](PROVENANCE.md) for what does and does not carry over.

## Building

Requires a CUDA build of llama.cpp and a Go toolchain. Build instructions land with the first
code drop.

## License

[Apache License 2.0](LICENSE). You may use, modify, and redistribute this code, including in
commercial and closed-source products, subject to the licence terms.

Origin and chain of title for the imported runtime are recorded in
[PROVENANCE.md](PROVENANCE.md). Contributions are accepted under the same licence, certified by a
[DCO](.github/DCO) sign-off — see [CONTRIBUTING.md](CONTRIBUTING.md). Contributors keep the
copyright in their own work; there is no CLA and no copyright assignment.

Model weights published alongside this project carry the licence of their upstream base model,
which is **not** Apache-2.0 and may restrict redistribution or commercial use. Check the model
card for each build.
