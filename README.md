# llama-herd

A high-performance inference server for GGUF models, and the models to go with it.

One GPU hosts a **herd** — several models resident at once, each driven by many concurrent
streams over a single copy of its weights. Built for the cards people actually own, and for
throughput above all else.

---

## What this is

Two halves, co-designed:

**The engine.** A weight-shared, multi-stream runtime. Each model is loaded once and served by a
single decode loop with per-sequence KV isolation, so N streams ride **one** forward pass rather
than N. Multiple models share the card. Multi-token prediction is a first-class path, not an
afterthought.

**The models.** Quantized builds published alongside this engine, built specifically
for this engine — Qwen, GLM, and likely Nemotron lines. They are not repackaged third-party quants,
for a concrete reason given below.

The comparison points are llama.cpp's `llama-server` and Ollama. The difference we are chasing is
throughput per card and models tuned to the runtime that serves them.

## Why we build our own quants

This is the part that is easy to mistake for branding. It is not.

Models with native multi-token-prediction layers — Qwen, Granite, Nemotron — carry MTP weights
that let the model propose several tokens per step and have them verified in one pass. **Widely
distributed GGUF quants drop those tensors.** With the MTP head stripped, an `--mtp` flag is a
no-op: there is nothing to run. This was measured, not assumed.

Speculative decoding is the single largest throughput lever available on a consumer card, and you
cannot pull it on a borrowed quant. Building the quants is what makes the engine's fastest path
reachable at all. Quantization choices, KV precision, and MTP retention get made for this runtime
rather than inherited.

## Hardware

Primary targets are consumer NVIDIA cards:

| Card | VRAM | Arch | Throughput goal |
|------|------|------|-----------------|
| RTX 3090 | 24 GB | Ampere | 500 tok/s |
| RTX 4090 | 24 GB | Ada | 750 tok/s |
| RTX 5090 | 32 GB | Blackwell | 1000 tok/s |

Aggregate decode across concurrent streams on a single card, with a long-context target per herd
member. See [docs/ROADMAP.md](docs/ROADMAP.md) for how these numbers were arrived at and what
stands between here and them.

**It is not limited to those cards.** Anything llama.cpp runs on, this runs on. A stack of 3060s is
a legitimate deployment, and heterogeneous multi-GPU — different cards, different capacities, in
one host — is a design goal rather than an accident.

## How the engine works

```
one decode-loop goroutine  ── each tick: gather the next token from every active stream
                              into ONE llama_batch (tagged by seq_id), llama_decode once,
                              sample each stream's logits, route tokens back to their streams
N stream goroutines        ── submit {prompt, params}, receive tokens over a channel
Registry                   ── one Engine per model, each with its own resident weights and
                              decode loop, sharing the GPU
```

Only the decode loop touches llama.cpp, so concurrent-`llama_decode` thread safety is a non-issue
by construction. Per-sequence KV isolation is llama.cpp-native — a finished stream frees its cells
without disturbing its neighbours. Continuous batching admits new work *while* decoding: mixed
prefill and decode in one pass, with slot recycling.

The engine core is backend-agnostic; the llama.cpp binding sits behind an interface.

## Status

**Early — the code is being written now.** llama-herd is a clean rewrite, not a port: no code is
carried over from anywhere, down to the llama.cpp binding.

It is also **not a stripped-down edition of a private product.** llama-herd has features its
closed-source predecessor does not, and leaves out that system's platform-specific plumbing
entirely.

The architecture is not speculative. A runtime of this design was built privately and measured on a
single RTX 3090:

- **4 models co-resident on one card** — 22 streams total, 11.2 GB of 24 GB, all four decoding
  simultaneously with zero cross-stream contamination
- **407 tok/s aggregate at 10 streams** for a 35B MoE (A3B) at 4-bit, fully offloaded — 140 tok/s
  single-stream, so weight sharing bought ~2.9× on one card
- **~318 tok/s aggregate** driving 6 streams of an 8B-class model at Q4 through one context
- 128k chunked prefill; the input ceiling is per-sequence context, not batch size

Those came from a **different implementation** of this architecture and are recorded as the bar
this rewrite aims to clear — a design target, not a benchmark of this code, which does not yet
exist. See [PROVENANCE.md](PROVENANCE.md).

## Running

Describe the herd in a manifest — one entry per model, sized to the card it will sit on:

```json
{
  "listen": ":8080",
  "models": [
    { "name": "chat", "path": "/models/model.gguf",
      "gpu_layers": -1, "context": 32768, "streams": 6, "load_mtp": true }
  ]
}
```

```bash
llama-herd serve --manifest manifest.json
```

That exposes an OpenAI-compatible API, so any agent with a configurable base URL works
against it unmodified:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "chat",
  "messages": [{"role":"user","content":"hello"}],
  "stream": true
}'
```

**Vision** works through the standard content format, for models that ship a projector:

```json
{"model": "vl", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "what is this?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}]}
```

Images must be inlined as data URLs. Remote URLs are refused: fetching a caller-supplied URL
server-side would let anyone reach whatever the server can, including cloud metadata endpoints
and private networks.

Per-request sampling is honoured — `temperature`, `top_p`, `top_k`, `min_p`, the penalties and
`seed` layer over the model's manifest defaults, and an explicit `"temperature": 0` means greedy
rather than "unset".

`GET /v1/info` reports the build and the hardware actually in use, and says so explicitly when
no accelerator was found. Worth checking after any deploy: a server that silently fell back to
CPU answers every request correctly and passes its health check, so nothing else reveals it.

`GET /v1/models` lists what is loaded. `GET /health` reports per-model liveness and returns
503 if a model's decode loop has died — so a load balancer stops sending traffic to a server
that is still listening but can no longer answer.

Sizing is per model rather than global on purpose: a herd may span cards of different
capacities, and one stream count applied to all of them either wastes the large card or
fails on the small one. See [examples/manifest.json](examples/manifest.json).

## Install

Pre-built archives for Linux, macOS and Windows are attached to each
[release](https://github.com/sideblank/llama-herd/releases). Each contains the binary with its
llama.cpp libraries beside it, so it runs without a system-wide llama.cpp install.

```bash
tar xzf llama-herd_<version>_<platform>.tar.gz
cd llama-herd
./llama-herd doctor      # verifies linkage and reports the llama.cpp build
```

Releases are CPU-only today; GPU builds are tracked in [LAUNCH.md](LAUNCH.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Targets, the reasoning behind them, and the constraints in the way |
| [docs/INVARIANTS.md](docs/INVARIANTS.md) | Things that are true and were expensive to learn — mostly silent failures |
| [docs/MODEL-BUILD.md](docs/MODEL-BUILD.md) | How model builds are produced and what each must carry |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | What each measurement means and how to reproduce it |
| [PROVENANCE.md](PROVENANCE.md) | Origin and chain of title |
| [LAUNCH.md](LAUNCH.md) | What remains before this goes public |

## Building

Requires a Go toolchain and llama.cpp. Header and library locations are not hard-coded — supply
them through the standard cgo environment variables:

```bash
export CGO_CFLAGS="-I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include"
export CGO_LDFLAGS="-L/path/to/llama.cpp/build/bin"

go build ./...
go vet ./...
```

The binding compiles and vets against llama.cpp's headers alone, with no GPU and no built library
present, which keeps signature and struct-layout errors out of the loop early. Linking and running
need a full build of llama.cpp and a card.

## License

[Apache License 2.0](LICENSE). Use, modify, and redistribute freely, including in commercial and
closed-source products.

Contributions are accepted under the same licence, certified by a [DCO](.github/DCO) sign-off —
see [CONTRIBUTING.md](CONTRIBUTING.md). Contributors keep the copyright in their own work; there is
no CLA and no copyright assignment. Origin and chain of title are recorded in
[PROVENANCE.md](PROVENANCE.md).

Published model weights carry the licence of their upstream base model, which is **not**
Apache-2.0 and may restrict redistribution or commercial use. Check the model card for each build.
