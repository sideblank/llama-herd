# llama-herd

A high-throughput inference server for GGUF models, built for the GPUs people actually own.

One card hosts a **herd**: one or more models resident at once, each driven by many concurrent
streams over a single copy of its weights. Every stream rides the same forward pass, so the card
retires several times the work it would serving one sequence at a time. Bring your own GGUF; this
project ships no model weights and builds no quantizations.

---

## What it does

- **Weight-shared, multi-stream decoding.** A model is loaded once and served by a single decode
  loop with per-sequence KV isolation. N streams cost one forward pass, not N.
- **Several models on one card**, each with its own weights, decode loop and stream budget, sized
  per model so a herd can span cards of different capacities.
- **An OpenAI-compatible API** with streaming, per-request sampling, vision input for models that
  ship a projector, and grammar-constrained output.
- **Speculative decoding** from the sequence's own context, or from a model's multi-token
  prediction head when the GGUF kept one, with a startup check that refuses it on caches that
  cannot be rewound.
- **Measurement built in.** Startup self-test against the library's own figure, a configuration
  sweep that loads the weights once, and a `/v1/info` endpoint that reports where the time went.

The comparison points are llama.cpp's `llama-server` and Ollama. The difference being chased is
aggregate throughput per card.

## Measured

| Card | VRAM | Goal | Measured |
|------|------|------|----------|
| RTX 3090 | 24 GB | 500 tok/s | **728.71 tok/s** aggregate at 48 streams |
| RTX 4090 | 24 GB | 750 tok/s | not yet measured |
| RTX 5090 | 32 GB | 1000 tok/s | not yet measured |

The 3090 figure is one node, one model (`Qwen3.6-35B-A3B-UD-IQ3_S`), short prompts, median of
three, and 5.27x what the library's own `llama-bench` retires on a single sequence on the same
boot. It varies by more than 20% between rented cards of the same model, and it is a different
number at depth: at 16k of context per stream the same card sustains roughly 110 tok/s however
the streams are arranged. Method, caveats and the full surface are in
[docs/results/3090.md](docs/results/3090.md); the manifest that produced it is
[examples/3090-throughput.json](examples/3090-throughput.json).

Anything llama.cpp runs on, this runs on. A stack of 3060s is a legitimate deployment, and
heterogeneous multi-GPU is a design goal rather than an accident.

## How it works

```
one decode-loop goroutine  ── each tick: gather the next token from every active stream
                              into ONE llama_batch (tagged by seq_id), llama_decode once,
                              sample each stream's logits, route tokens back to their streams
N stream goroutines        ── submit {prompt, params}, receive tokens over a channel
Registry                   ── one Engine per model, each with its own resident weights and
                              decode loop, sharing the GPU
```

Only the decode loop touches llama.cpp, so concurrent-`llama_decode` thread safety is a non-issue
by construction. Per-sequence KV isolation is llama.cpp-native: a finished stream frees its cells
without disturbing its neighbours. Continuous batching admits new work while decoding, with mixed
prefill and decode in one pass and slot recycling. Active streams consume the batch budget before
prefill, so a busy herd never starves the streams already talking.

The engine core is backend-agnostic and tested without a GPU; the llama.cpp binding sits behind
an interface. llama.cpp itself is stock upstream at a pinned tag, built with non-default flags.

## Running

Describe the herd in a manifest, one entry per model, sized to the card it will sit on:

```json
{
  "listen": ":8080",
  "models": [
    { "name": "chat", "path": "/models/model.gguf",
      "gpu_layers": -1, "context": 425984, "streams": 48,
      "kv_unified": true, "admit_context": 8192 }
  ]
}
```

```bash
llama-herd serve --manifest manifest.json
```

That exposes an OpenAI-compatible API, so any agent with a configurable base URL works against
it unmodified:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "chat",
  "messages": [{"role":"user","content":"hello"}],
  "stream": true
}'
```

Or run the container, which downloads the model, measures the library, checks itself, and then
serves:

```bash
docker build -t llama-herd .
docker run --gpus all -p 8080:8080 -v llama-herd-models:/models \
  -e LLAMA_HERD_MODEL_URL=https://huggingface.co/<org>/<repo>/resolve/main/<model>.gguf \
  -e LLAMA_HERD_CONTEXT=425984 -e LLAMA_HERD_STREAMS=48 -e LLAMA_HERD_ADMIT_CONTEXT=8192 \
  -e LLAMA_HERD_KV_UNIFIED=true -e LLAMA_HERD_FLASH_ATTN=true \
  -e LLAMA_HERD_KV_TYPE_K=q8_0 -e LLAMA_HERD_KV_TYPE_V=q8_0 \
  llama-herd
```

Every manifest field has an environment variable; `docker-entrypoint.sh` lists them. While the
model downloads, the port answers `/health` and reports the boot phase on every other path, so a
long first boot is not mistaken for a dead instance.

**The KV pool.** `kv_unified` puts every stream in one pool instead of giving each a fixed share.
It is off by default and it is what the headline figure runs under: on a 3090 with a 35B-A3B it
was worth 182 tok/s against 55 from this flag alone. One pool means a per-stream ceiling reserves
nothing, so the manifest requires `admit_context` alongside it and refuses a cap that, multiplied
by the stream count, exceeds the pool. That is what stops concurrent long requests being evicted
mid-answer.

**Vision** works through the standard content format for models that ship a projector. Images
must be inlined as data URLs; remote URLs are refused, because fetching a caller-supplied URL
server-side would let anyone reach whatever the server can reach.

**Sampling** is per request: `temperature`, `top_p`, `top_k`, `min_p`, the penalties and `seed`
layer over the model's manifest defaults, and an explicit `"temperature": 0` means greedy. A
request may also carry a GBNF `grammar`, so its output is valid by construction and many streams'
answers can be merged by code.

## Speculative decoding

Two draft sources, both verified by the target in the same forward pass, and neither changes
what the model would have said: a rejected draft costs batch space, never a token.

*Lookup* drafts from the sequence's own context, the prompt included, at no memory cost:

```json
"speculation": { "type": "lookup", "max_draft": 4, "pattern": 3 }
```

It pays where output repeats context, which is most of what an agent does: editing a file that is
in the prompt, filling a schema, continuing a transcript. On free-form prose it proposes nothing.

*MTP* drafts from the model's own trained prediction head, so it works on output that repeats
nothing, at the cost of the memory the head occupies. It needs a GGUF that kept the head, and
most published quantizations strip it silently; `llama-herd inspect` reports whether one did. It
is off by default because it has to be measured per model: on a 35B-A3B with a one-layer head it
drafted at 57% acceptance and was still slower than not speculating, because each drafted token
costs a decode call.

```json
"load_mtp": true,
"speculation": { "type": "mtp", "max_draft": 4 }
```

Not every model can speculate at all. Drafts are written into the cache and taken back on
rejection; an attention cache rewinds by position, and the recurrent or sliding-window state that
hybrid architectures carry does not. At load the runtime decodes two tokens and tries to remove
one. A rewindable cache speculates directly; a non-rewindable one is snapshotted before each step
and restored on rejection, at a forward pass per rejection; a cache that can do neither is served
without speculation, with the reason logged. `GET /metrics` reports the draft acceptance rate,
which is the number that says whether a loaded head is earning its VRAM.

## Observability

`GET /v1/info` reports the build, every device with its live free VRAM, host load, per-model
utilisation, the startup self-test against the library's own figure, and where engine time went:
staging a batch, the library's forward pass, or harvesting tokens. It says so explicitly when no
accelerator was found, which is worth checking after any deploy, since a server that fell back to
CPU answers correctly and passes its health check.

`GET /metrics` serves the same figures in Prometheus format with no client library behind it.
The two numbers worth alerting on are queue depth persistently above zero, which means the model
is saturated, and evictions rising, which means the context budget is over-committed.

`GET /v1/models` lists what is loaded. `GET /health` returns 503 if a model's decode loop has
died, so a load balancer stops sending traffic to a server that is listening but cannot answer.

## Commands

| Command | What it does |
|---|---|
| `serve` | Host the manifest's models over the API |
| `bench` | Measure throughput and emit a reproducible report |
| `sweep` | Measure a matrix of stream counts, KV precisions and depths against one resident copy of the weights |
| `fit` | Report what streams and context fit on a card, beside the configuration that was actually measured |
| `inspect` | Report what a GGUF declares, including whether it kept its MTP layers |
| `models` | List which supported models this machine can install, and why the rest cannot |
| `vcontext`, `tasks` | Run an input of any size, or a dependency-ordered task graph, across the herd |
| `canon` | Measure what each canonicalisation pass of a payload costs in real tokens |
| `standby` | Hold the port and report boot progress while a long preparation runs |
| `doctor` | Verify linkage and list the devices it can see |

## Install

Pre-built archives are attached to each
[release](https://github.com/sideblank/llama-herd/releases), with the llama.cpp libraries beside
the binary so no system-wide install is needed. Releases are CPU-only today; the CUDA build for
the 3090, 4090 and 5090 is the container image, built from the `Dockerfile` for all three compute
capabilities. Verify what you are running after any deploy:

```bash
./llama-herd doctor      # verifies linkage and reports the llama.cpp build
curl -s localhost:8080/v1/info | jq .build
```

## Building

Requires Go 1.24 and a build of llama.cpp with the multimodal library on:

```bash
cmake -S llama.cpp -B build -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_MTMD=ON \
      -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DCMAKE_INSTALL_PREFIX=/opt/llama
cmake --build build -j "$(nproc)" && cmake --install build

# The speculative shim. This is the stub, which reports no speculation; the Dockerfile
# builds the real one against llama.cpp's common library.
g++ -O2 -fPIC -shared -std=c++17 -I/opt/llama/include -Ishim shim/lhspec_stub.cpp \
    -o /opt/llama/lib/liblhspec.so && cp shim/lhspec.h /opt/llama/include/

export CGO_CFLAGS="-I/opt/llama/include"
export CGO_LDFLAGS="-L/opt/llama/lib"
go build ./... && go vet ./... && go test ./... -race
```

The scheduler is backend-agnostic on purpose, so its tests run anywhere; that is where the real
failure modes are. The binding's tests need the library present to link, not a GPU.

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Start here.** The shape of the system, the facts that constrain it, and the principles that recur |
| [docs/results/](docs/results/) | The measurements themselves |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | What each measurement means and how to reproduce it |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | What this ships as, the capacity model, and how a configuration is chosen for a machine |
| [docs/INVARIANTS.md](docs/INVARIANTS.md) | Things that are true and were expensive to learn, mostly silent failures |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Targets, the reasoning behind them, and the constraints in the way |
| [docs/VIRTUAL-CONTEXT.md](docs/VIRTUAL-CONTEXT.md) | The layer that accepts an input of any size |
| [docs/DAG-SCHEDULING.md](docs/DAG-SCHEDULING.md) | Ordering within a request |
| [docs/CODE-GRAPH.md](docs/CODE-GRAPH.md) | Code generation, code judging, and the fan-out boundary |
| [docs/PREFIX-REUSE.md](docs/PREFIX-REUSE.md) | Sharing a prompt prefix's KV cells across the herd |
| [docs/HLSR.md](docs/HLSR.md) | Hierarchical latent speculative reduction |
| [docs/MODELS.md](docs/MODELS.md) | Auxiliary models, and the rule that decides whether one earns its place |
| [PROVENANCE.md](PROVENANCE.md) | Origin and chain of title |
| [LAUNCH.md](LAUNCH.md) | What remains before this goes public |

## Status

Measured, not yet released. The engine, the API, speculation, the KV pool, the sweep and
self-test tooling, and the layers above the engine are built and tested; the 3090 result above
came from this code. The 4090 and 5090 targets are projections until a card is measured.

llama-herd is a clean rewrite, not a port: no code is carried over from anywhere, down to the
llama.cpp binding. A runtime of this design was built privately first and measured on one 3090
at 407 tok/s aggregate across 10 streams; that was the bar this rewrite had to clear, and it did.
It is not a stripped-down edition of that system. It has capabilities the private one does not,
and leaves that system's platform-specific plumbing out entirely. See
[PROVENANCE.md](PROVENANCE.md).

## License

[Apache License 2.0](LICENSE). Use, modify, and redistribute freely, including in commercial and
closed-source products.

Contributions are accepted under the same licence, certified by a [DCO](.github/DCO) sign-off;
see [CONTRIBUTING.md](CONTRIBUTING.md). Contributors keep the copyright in their own work; there
is no CLA and no copyright assignment.

Model weights are not part of this project. Whatever GGUF you point it at carries its own licence.
