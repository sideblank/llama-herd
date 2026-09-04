# Roadmap

What llama-herd is aiming at, how the numbers were arrived at, and which parts are hard.

This document is deliberately honest about constraints. A target with a known obstacle in front of
it is more useful than one without.

---

## 0. Where this sits

llama-herd is the **open engine**. It exposes a standard chat-completions API, so any agent with a
configurable base URL — editor plugins, terminal agents, custom tooling — points at it and works.
No per-agent integration, no plugins to write, no fork of anything.

Orchestration that makes a modest local model behave like an ensemble is a **separate product**
and is not part of this repository. The boundary is the API: the engine serves models, the layer
above decides what to ask them.

That split is deliberate. The engine is useful standalone — run a model, get an endpoint, point
your agent at it — and nothing in it depends on the layer existing.

**One consequence worth stating plainly, because it constrains what can be built on top:** this
engine is open source, so anything sent to it is visible to whoever runs it. A user can build the
engine with a log line and see every request verbatim. That is not a defect to be worked around —
it is what open source means. Anything above this API should be designed on the assumption that
individual requests are observable, and should keep its value in policy and models rather than in
the text of any single prompt.

## 1. Throughput targets

| Card | VRAM | Goal (aggregate decode) | Status |
|------|------|-------------------------|--------|
| RTX 3090 | 24 GB | 500 tok/s | **met: 728.71 measured** (§7b, `results/3090.md`) |
| RTX 4090 | 24 GB | 750 tok/s | projection, not measured |
| RTX 5090 | 32 GB | 1000 tok/s | projection, not measured |

These were not invented. On a single 3090, a prior implementation of this architecture measured:

| Model | Streams | Aggregate | Per stream |
|-------|---------|-----------|------------|
| 35B MoE (A3B), 4-bit | 1 | 140.7 tok/s | 140.7 |
| 35B MoE (A3B), 4-bit | 6 | 344.7 tok/s | 57.4 |
| 35B MoE (A3B), 4-bit | 10 | **407.7 tok/s** | 40.8 |
| 8B-class, Q4 | 6 | ~318 tok/s | ~53 |

That put ~500 tok/s on a 3090 within reach of the *existing* mechanic, and this implementation has
since measured 728.71 there (§7b). The 4090 and 5090 goals follow from memory bandwidth plus the
levers in §2 and remain unmeasured. The decode loop is
**memory-bandwidth bound**, not compute bound: one pass over resident weights serves every active
stream, which is why aggregate throughput climbs with stream count while per-stream throughput
falls.

## 2. Multi-token prediction — the main lever

Models with native MTP layers (Qwen, Granite, Nemotron) can propose several tokens per step and
have them verified in a single forward pass. On a bandwidth-bound decode loop this is the largest
single win available in principle.

**Measured verdict so far: net-negative.** On `Qwen3.6-35B-A3B` the head drafts and rollback is
correct, acceptance is 57% at `max_draft=2`, and throughput still fell 2.1x at one stream and 2.6x
at four. The 728.71 tok/s result was taken with speculation off. The caveats on that measurement,
and what would overturn it, are in §7b; until it is re-run, the rest of this section describes the
mechanism, not a win.

**The obstacle, and why we build our own quants:** widely distributed GGUF quants **drop the MTP
tensors**. With no `nextn` weights present, an MTP flag is a no-op — there is nothing to execute.
This was measured on a real third-party quant, not assumed.

Consequently the model builds are not a branding exercise. Retaining MTP heads, choosing KV
precision, and picking quantization for this runtime is what makes the fastest path reachable at
all.

**Upstream support is already there.** At the pinned llama.cpp revision, MTP is first-class in the
public API: `llama_model_params.load_mtp` loads the MTP layers, and a context can be created as
`LLAMA_CONTEXT_TYPE_MTP`. Both are exposed by the binding. So the missing piece is not runtime
support — it is a quantization that still contains the tensors to load.

**Loading the layers is not using them.** Measured on a real 35B-A3B with MTP tensors
confirmed loaded: **exactly 1.00 tokens per forward pass**, meaning no speculation at all. The
head was resident, occupying memory, and returning nothing.

The cause is architectural rather than a missing flag. Driving an MTP head needs a **second
context** created with the MTP context type and linked to the target context, plus a
draft-then-verify loop: the draft proposes several tokens, the target verifies them in one
pass, and the longest matching prefix is accepted. Setting the model flag alone loads the
weights and nothing drives them.

**The API needed to drive an MTP head is not public.** Feeding a draft context requires the
target's hidden states, and the function that returns them lives in a staging header that
upstream describes as work in progress, permits breaking changes and C++ in, is not installed
by the build, and asks callers not to include it. The public header exposes only the layer
count.

**Route 1 was taken and is built.** `shim/lhspec.{h,cpp}` wraps upstream's
`common_speculative` behind a C ABI, and `internal/llama/speculative.go` drives it as an
ordinary `engine.Drafter`. Set `"speculation": {"type": "mtp"}` alongside `"load_mtp": true`
and the model's own head supplies the drafts. The three routes considered were:

1. **Link llama.cpp's common library and wrap its speculative implementation.** This is what
   upstream's own server does, and it covers all three MTP architectures rather than one. It
   is C++ with C++ types in its interface, so it needs a small C shim to be callable — the
   same shape of seam a previous implementation used for the same reason. Cost: a C++
   dependency and a shim to maintain.
2. **Vendor the staging header and bind it directly.** Least code today, and it breaks on any
   upstream change, silently, in a feature whose failure mode is already "loaded and doing
   nothing".
3. **Implement classic draft-model speculation instead**, which needs only public API: a small
   model drafts, the target verifies, the longest matching prefix is accepted. It works today
   and on any model, but the draft weights cost VRAM that an MTP head does not, which is
   exactly the advantage MTP was chosen for.

That order held. A zero-VRAM lookup drafter made the loop real and measurable first, and
route 1 then swapped the draft source into a loop that was already tested — which is why the
MTP work was a binding rather than a feature.

Route 2 was rejected for the reason it deserves naming: the failure mode of this feature is
already "loaded and doing nothing", and an unstable header would add a second way to arrive
there silently.

**The draft-verify loop is built.** The engine stages a stream's next token together with
whatever a drafter proposes, verifies every position in one forward pass, accepts the longest
prefix the target agrees with, and rewinds the cache past the divergence. A stream that
accepts drafts produces several tokens from a single pass, which is exactly what
tokens-per-pass measures.

Three properties it holds, each tested:

- **A wrong draft costs nothing but batch space.** The token at a divergence is the target's
  own choice, so a rejected draft still yields one real token. Speculation never costs a
  step; it only sometimes saves several.
- **Drafts consume batch budget like any other entry.** Counting only the real token would let
  a speculating stream overrun the batch, which the backend rejects outright rather than
  degrading.
- **A failing drafter degrades to ordinary decoding.** Speculation is an optimisation, so a
  broken draft source must not fail the request.

The draft source is abstract on purpose: a companion model, a trained head, or an n-gram cache
all produce candidate tokens, and the loop does not care which.

Two sources ship today, and **MTP has been measured and does not pay on a one-layer head.**

On a 3090 with Qwen3.6-35B-A3B IQ3_S, correct KV pool layout: 57% acceptance, 1.72 tokens per
pass against 1.00, forward passes cut by a third — and **2.1x slower in wall-clock** than not
speculating at one stream, 2.6x slower at four. The head predicts one token ahead, so drafting k tokens is k
sequential decodes on the draft context, plus one to resynchronise it, plus the target's own
pass. That context touches one layer of forty-one, so it ought to be nearly free, and it is not:
per-call cost dominates. Four decode calls bought 1.72 tokens.

Acceptance is therefore the wrong figure to optimise. 57% is close to the ceiling for one layer,
and it was not enough. What changes the arithmetic is a head that predicts SEVERAL tokens per
pass — one decode buying three or four drafts instead of one — which is a property of the
weights, not of this runtime. That is the case worth building for in the model pipeline, and
until such weights exist speculation should stay off for this model class.

**Lookup** predicts from the sequence's own context and costs no memory or extra decode, so its
arithmetic is different: it adds batch entries to an existing pass rather than adding passes. It
remains worth measuring where output repeats input.

An earlier reading here — 32% acceptance, a 2.3x penalty — was taken while the KV pool was split
per stream, which made the library run a forward pass per sequence and inflated the cost of
every draft. Both figures above are from the corrected configuration.

Work items:

- Measure accept rate and end-to-end speedup per model and per card. Done once for
  Qwen3.6-35B-A3B on a 3090 (net-negative, above). What is open is whether the sampler rework
  since then changes the verdict, and what the other models and cards show.
- Decide how a draft context shares the KV budget. It is a second context over the same
  weights, so its cache competes with the target's for the same VRAM.
- Confirm `load_mtp` finds the tensors in each candidate quant, which `llama-herd inspect`
  already reports.
- **A draft-model source** using public API, for models with no native head at all. Lookup
  covers that case for repetitive traffic; a companion model would cover the rest, at the
  VRAM cost that MTP exists to avoid.

## 3. Long context, and the constraint that bites

The goal is **128k context per herd member**. The engine mechanic for this exists — chunked
prefill means the input ceiling is per-sequence context rather than batch size.

The constraint is VRAM, and it is worth stating in numbers before anyone is surprised by it. KV
cache size is:

```
bytes = 2 (K and V) × n_layers × n_kv_heads × head_dim × n_tokens × bytes_per_element
```

Crucially, `n_layers` here means **layers that actually hold a cache**. Hybrid architectures
cache only every Nth layer and use linear attention elsewhere, whose state is constant-size
and does not grow with context — so their KV cost is a fraction of a dense model's, and
assuming otherwise overstates it by that factor.

For a dense 8B-class model with grouped-query attention (32 layers, 8 KV heads, 128 head dim)
that is **64 KiB per token** at 1 byte per element. Which gives, for one sequence:

| Context | KV @ 4-bit | KV @ 8-bit | KV @ fp16 |
|---------|-----------|-----------|-----------|
| 8k      | 0.27 GB   | 0.54 GB   | 1.07 GB   |
| 32k     | 1.07 GB   | 2.15 GB   | 4.29 GB   |
| 128k    | **4.29 GB** | **8.59 GB** | 17.2 GB |

On a 24 GB card holding ~5 GB of weights, roughly 19 GB remains. That is **two** concurrent 128k
streams at 8-bit KV, or four at 4-bit — not twenty. Larger models with more layers are worse.

So "128k per member" and "many concurrent members" are in direct tension, and the honest design is
a **mixed workload**: most streams short, a few long, with capacity admitted against a real KV
budget rather than a fixed slot count. Levers to pursue, in order of expected value:

1. **KV quantization** (`kv_type_k` / `kv_type_v`) — built and measured for throughput
   (`results/3090.md`). Quality impact needs measuring, not guessing.
2. **Admission control against a KV budget** rather than a slot count, so a long-context request is
   admitted only when its cells actually fit. Half built: `admit_context` caps every stream at a
   static budget; admitting against live pool occupancy is not done.
3. **Sliding-window / partial attention** where the model supports it.
4. **KV offload and paging** to host memory for idle streams — costs PCIe bandwidth, buys slots.
5. **Prefix sharing** across streams with a common prompt, which is close to free when it applies.
   Built (`internal/engine/prefixshare.go`), not yet measured on a card.

## 3b. Many cards: capacity is not throughput

Rented hosts with eight consumer cards are available cheaply, and 8 × 24 GB is 192 GB of
VRAM — more than a single datacenter card, for less money. That opens models a 24 GB card
cannot hold. But it buys two different things, and confusing them leads to disappointment.

**Capacity mode — one model split across cards.** Layer splitting puts different layers on
different cards. Only the hidden state crosses the bus at each boundary, which is tiny next
to the weights, so this works acceptably over plain PCIe without the fast interconnects
training requires.

What it does **not** do is multiply throughput. The layers run in sequence, so a token passes
through card 1, then card 2, and so on: the cards take turns rather than working at once.
Aggregate throughput lands near a single card's, minus transfer overhead. Eight cards let you
*run* a model you otherwise could not. They do not make it eight times faster.

**Throughput mode — one model per card.** Eight independent herds, each with its own resident
weights and decode loop, is genuinely eight times the aggregate throughput, because nothing
is shared and nothing waits. This is the mode that matches the per-card throughput targets.

The two are composable — four cards holding a large model, four serving a small one — and the
right split is a property of the workload. What matters is that the manifest expresses it, and
that a benchmark states which mode produced a number. A capacity-mode figure and a
throughput-mode figure are not comparable.

## 3c. Fixed shares or one pool

A context is divided between streams in one of two ways, and the choice decides whether a
single request can ever exceed its share.

**Split** — the default — gives each stream `context / streams`, fixed. A herd configured for
four 128k streams gives every request exactly 128k, and a request cannot borrow from idle
slots. Three quarters of the cache sits unused while one long request is refused at 128k.

**Unified** puts every stream in one pool, and any stream may use all of it. The same herd
then serves one 512k request, or four 128k ones, or any mix that fits — allocation follows
demand instead of a partition fixed at startup.

The cost is that nothing is reserved. One long request can consume the pool and leave the rest
of the herd nothing, where a split guarantees each slot its share. So unified suits a workload
where request sizes vary and the herd is not adversarial with itself — which describes calls
fanned out from one plan, and does not describe unrelated tenants.

**Status: off by default, and the configuration every measured result in §7b ran under.** The
setting reaches the context and the ceiling becomes the whole pool. On this card it was worth
182 tok/s against 55 at four streams from the flag alone, and the 728.71 headline is 48 streams
in one pool.

**Admission control had to change with it.** Under a split the per-stream ceiling is a real
reservation. Under one pool it is the whole cache, so requests could each be admitted believing
they may use everything, and the herd would evict its way out of the overcommitment. The manifest
now refuses `kv_unified` with more than one stream unless `admit_context` is set, and refuses an
`admit_context` that, multiplied by the stream count, exceeds the pool; admission checks every
request against that static budget. Admitting against measured free capacity instead, which would
let streams be sized unevenly, is not done (§3, lever 2).

## 4. Multi-GPU and heterogeneous fleets

Not every deployment is one 5090. A stack of 3060s is a legitimate target, as is a mixed host —
different cards, different capacities.

**Upstream already provides the mechanics**, and the binding exposes them:

| Mode | What it does |
|------|--------------|
| `SplitNone` | whole model on one device |
| `SplitLayer` | layers and KV divided across devices |
| `SplitRow` | layers and KV divided, with tensor parallelism where supported |
| `SplitTensor` | full tensor parallelism |

`TensorSplit` sets the proportion each device receives. That field is what makes a *mixed* host
work rather than merely a multi-card one: a 24 GB card and a 12 GB card must not receive equal
shares, and an even split silently fails on the smaller one.

The binding also enumerates devices — name, description, type, and free/total memory — so
placement can be decided from what is actually present. Integrated GPUs are reported separately
from dedicated ones on purpose: an iGPU's memory is the host's, so its capacity number means
something entirely different.

Plan:

- **Model-per-GPU placement** first: put each model wholly on the card that fits it. Simple, and
  it avoids interconnect entirely.
- **Layer splitting** for models too large for one card, accepting the interconnect cost.
- **Proportional splitting on mixed hosts**, derived from measured free memory rather than
  assumed to be uniform.
- **Capacity-aware routing**: a request goes to the card that can host it, accounting for both
  resident model and free KV.
- Heterogeneous fleets mean **per-card manifests** — stream counts and context budgets sized to
  each card, not one global setting.

## 5. Splitting work across models

A stated goal is to split work across backend models and merge the results so the user perceives
maximum throughput. This works, but only in specific forms, and it is worth being precise because
one intuitive version of it cannot work.

**What cannot work:** interleaving tokens from two independently decoding models into a single
completion. Autoregressive generation is sequential — token *n+1* is conditioned on token *n* — so
two models generating alternate tokens of one response produce incoherent text, not a faster
response.

**What does work, and gets the same outcome:**

- **Speculative decoding / MTP** (§2) — a draft proposes, the target verifies in one pass. This is
  genuinely "several models producing one stream", it preserves the target model's output
  distribution exactly, and it speeds up a *single* request. This is the real version of the idea.
- **Continuous batching** — many independent requests share forward passes. Raises aggregate
  throughput, which is what a multi-user endpoint actually needs.
- **Task decomposition** — where a request genuinely decomposes into independent parts (sections,
  parallel tool calls, map-then-reduce), run them concurrently on different models and assemble.
  Applies to structured work, not to one prose completion.
- **Tensor and pipeline parallelism** — split the *model* across cards, not the tokens.

The endpoint should therefore be built to exploit the first three, and should not promise the
impossible fourth.

## 5b. Vision and KV sharing

**Vision needs no shim.** llama.cpp's multimodal library builds standalone against ggml and
llama alone, and the flow is narrow: encode the image, prefill the result into the
sequence's KV at a given position, then continue through the ordinary decode loop from the
position it returns. There is no separate vision decode path — only a different way of
filling the prompt, so a vision request occupies a slot exactly like a text one.

The prompt must carry the media marker. Without it the tokenizer produces a chunk list with
no media, the image is dropped, and the model answers fluently about nothing — a silent
failure, not an error.

**KV sharing is a real choice, not a default.** A context can give each sequence its own
attention buffer or share one across them. Independent requests that have nothing in common
are better served per-sequence. But requests fanned out from one plan — the same system
message and context issued several ways — share a large prefix, and a unified buffer
exploits it. Upstream also warns that disabling sharing across several sequences can hurt in
some cases.

That makes it a per-deployment measurement rather than a setting to pick once, and it is
exposed in the manifest for exactly that reason. It is also a good first use of the
benchmark harness: the same model, the same sweep, the flag both ways.

## 6. Shared state between models

Longer term: give co-resident models a way to consult one another's data — a shared store the
engine owns.

Legitimate uses are real: a shared prefix/KV cache across models, a retrieval index consulted by
several models, and a scratchpad for a multi-model pipeline. The caution is that this is an
**orchestration feature, not a throughput one**, and embedding a database in a latency-critical
runtime is a serious commitment — durability, concurrency, crash recovery, and a schema that
outlives its first use.

The sensible first step is a small in-process store behind a narrow interface, sized to the actual
first use case, with persistence added only if it is needed. Choosing a database before choosing
the access pattern is the failure mode here.

## 6b. The model build process

The weights are half the product, and the reason is narrow: MTP tensors are frequently
missing from redistributed quantizations, nothing on a model page says so, and the loss
cannot be recovered afterwards.

Upstream conversion includes them by default — a publisher must opt out — so this is a
publisher-choice gap rather than a tooling gap, and building our own is tractable with the
standard tools. Export support covers Qwen, GLM, Nemotron, DeepSeek and others, which is the
portfolio.

See [MODEL-BUILD.md](MODEL-BUILD.md) for the pipeline and its gates. The two that matter:
verify MTP tensors after conversion **and** after quantization, since they fail at different
stages; and measure the accept rate rather than mere presence, since tensors that load but
propose poorly give back little.

## 7. Model lines and distribution

**Deferred.** The first release ships the engine only, with no model builds; it serves any GGUF
the user points it at. The lines below are the plan for a later version, kept because §2 explains
why an MTP-retaining build would have to be made rather than borrowed.

Planned to be published alongside the engine, later:

- **Qwen line** — native MTP, strong across sizes.
- **GLM line** — likely 4.7.
- **Nemotron line** — candidate; native MTP via its own head.

Each build ships a model card recording base model and its licence, quantization, KV precision,
whether MTP tensors are retained, and the manifest settings — stream count, context, KV budget —
for its target card. A build without those numbers makes the user re-derive them.

Distribution is **Hugging Face** for general availability and **Cloudflare** for application
delivery. Note that base-model licences travel with quantized builds and are not Apache-2.0; each
line needs its licence resolved before publication.

## 7b. Where the 3090 campaign landed (2026-08-23)

The §1 target for the 3090 was 500 tok/s aggregate. **Measured 728.71** at 48 streams; the curve
had not turned over in that run, and the ladder above 48 measured afterwards puts the plateau at
48-64 (below). Full numbers, method and the exact reproducible configuration:
`docs/results/3090.md` — the settings are recorded there rather than here so there is one place
to change when they are re-measured.

The short form: `Qwen3.6-35B-A3B-UD-IQ3_S`, llama.cpp `b10545` built with
`GGML_CUDA_FORCE_MMQ=ON`, **48 streams over a 425,984-token unified KV pool** (8,874 each),
`q8_0`/`q8_0` cache, flash attention on, batch 2,048, speculation off. Depth 0.

What the campaign established, in the order it matters:

- **The engine is no longer the cost.** Staging, sampling, detokenization and delivery together
  are **2.39%** of engine time; the remaining 97.61% is the library's forward pass. Further
  scheduler tuning cannot recover more than that.
- **Stream count is the throughput lever, and it is depth-dependent.** 24 streams beat 8 by 1.8x
  on an empty cache and lost to it by 2.2x at 16k. There is no single best stream count, only a
  best one for the depth being served.
- **~110 tok/s is the ceiling at 16k depth**, flat across 2 to 8 streams and unmoved by KV
  precision. This is a hybrid-attention model, so its cache is already small and attention at
  depth is compute-bound; shrinking a cache that is not the bottleneck cannot help.
- **Subdividing the same content across more streams is faster**, up to 1.9x at constant resident
  tokens. Depth within a sequence is what costs, not occupancy across the herd.
- **Prefill does not chunk.** Splitting one input lost ~19% of ingest throughput while roughly
  doubling decode, so ingest-heavy and generate-heavy workloads want opposite configurations.
- **`GGML_CUDA_FORCE_MMQ=ON` is worth about +7-9% shallow**, neutral at depth. Worth keeping.

Settled since the headline:

- **The stream-count peak is 48-64 and there is nothing above it.** Measured on one image and one
  node with 48 as control: 564.16 / 560.93 / 582.04 at 48/56/64, then collapse to 11.50 at 72. Going
  past 48 buys 3%. The 728.71 figure is the same behaviour on a faster node (library 138.40 vs
  119.31); normalised they agree. An earlier ladder on a third node ran flat through 72 and failed
  at 80, so the cliff moves with the node. **Ship 48.** Nothing measured supports a figure near
  1200. The full history of how this was established, including the run that 128 streams killed,
  is in `results/3090.md`.

Unresolved, and honestly so:

- **MTP — measured, and the open part is narrow.** It works: the shim drafts, checkpoint-and-replay
  makes rollback correct on a hybrid cache, and acceptance is **57%** at `max_draft=2`. It was also
  **net-negative on throughput** — 2.1x slower at one stream, 2.6x at four — which matches
  independent reports of MTP being net-negative on this same MoE. That verdict stands as the
  current best evidence.

  What is open is only whether it still holds. It was taken when sampling cost roughly nine times
  what it now does, and when the phase clock was charging GPU wait to the sampler — so the measured
  cost of a drafted token was both larger and mis-attributed. The re-measurement is built:
  speculation is a sweep axis, so `none` and `mtp` run in one boot rather than across two nodes.
  It has not been run to completion. Until it is, quote 57% acceptance and net-negative, and note
  the caveat.
- **KV prefix reuse.** Implemented (`internal/prefix`, `internal/engine/prefixshare.go`, #37) and
  not yet measured on a card. The most likely remaining win for chunked workloads that share a
  system prompt: prefill it once and share it, rather than paying for it per chunk. The figures in
  `PREFIX-REUSE.md` are arithmetic from the measured prefill rate, not a measurement.
- **4090 and 5090.** Never measured. The §1 targets for them remain projections.

## 7c. Virtual context — the layer above the engine

Built: `internal/vcontext/` and the `llama-herd vcontext` command, designed in `VIRTUAL-CONTEXT.md`.
A layer that accepts an input of any size, splits it across streams, keeps cross-chunk state
outside the model, and reassembles the answer.

The measurements that constrain it are already in hand and two of them are counter-intuitive:
**subdividing the same content across more streams is up to 1.9x faster** (depth within a sequence
is what costs, not tokens resident across the herd), while **prefill does not chunk at all** and
loses ~19% when split. So chunk count must follow whether the job is read-heavy or generate-heavy,
not how large its input is.

The largest unmeasured win is **KV prefix reuse** across chunks sharing a system prompt — at 48
chunks a 2k shared prefix is ~96k tokens of duplicated ingest. It is built; the saving is
arithmetic until it is measured on a card.

## 7d. Auxiliary models

`MODELS.md` — which small models do which jobs around the engine, and the rule that decides
whether one earns its place: **it must beat a cheap floor first.**

They serve two purposes, and the second is larger. Internally, virtual context needs retrieval,
entity extraction and classification. Externally they are **endpoints an agent calls**, so an agent
running against a local card does that work locally instead of round-tripping to a datacenter.

That second purpose is why model choice is not a benchmark question: local embeddings and remote
ones must land in the **same vector space**, or the two indexes cannot be queried together.

## 7f. DAG scheduling — ordering inside a request

Built: `DAG-SCHEDULING.md`, `internal/vcontext/{dag,schedule}.go`.

A long request is not a uniform pile of work: dependent sequences are interleaved with background
work. The fan-out layer treats every chunk as independent, which is right for retrieval and wrong
for instructions.

**Ordering is not asked of the model.** A model told to respect "step four needs step two" across a
long context can reorder or invent a step, and the output still reads as a coherent plan — nothing
errors. So the graph is extracted once under a grammar, then enforced by code. A cycle means no
valid ordering exists and is reported before anything runs.

**Tiers describe the graph; they do not execute it.** Running tier-by-tier makes every task wait for
the slowest member of the previous tier, including tasks it has no relationship with — the straggler
problem reintroduced as architecture. Dispatch is on dependency-satisfaction instead: a task starts
when *its own* prerequisites finish. Same ordering guarantee, and on a graph with one long pole it
saves most of the wall clock.

**Context between tasks is text, not `h_last`** — latent forwarding is the intended end state but
depends on the unresolved projection (#21), which #19 measured as lossy. Building on it would make
every dependent step quietly wrong in exactly the way this design exists to prevent.

`CriticalPath` bounds what any width could achieve, so a request too serial to fill 48 streams is
known *before* dispatch rather than inferred from a disappointing wall clock.

## 7e. Distributed intelligence — routing at the edge

Not now. Recorded because it is cheap to write down and expensive to reconstruct.

Expose the platform's MCP tools at `/v2/mcp`, embed the turn-router classifier in the engine, and
tool selection happens **on the local card** instead of in a datacenter — every edge node routing
its own turns.

**The sharpest argument is context budget, not latency.** Most agent stacks put the tool list in
the model's context and ask it to choose. At ~176 tools that is tens of thousands of tokens of
definitions re-read *every turn* — at ~3,300 tok/s prefill, a 30k manifest is ~9 seconds per turn
spent re-reading a list that did not change. And it competes directly with the thing virtual
context is fighting for: a stream holds 8,874 tokens, and tool definitions would be eating them.

A classifier decides in ~22 ms with no definitions in context at all. Only the selected tool's
schema reaches the large model. The freed budget goes to the user's content.

**It is span retrieval, applied to tools.** Do not put everything in context and hope attention
sorts it out — select first, load only what is needed. Tools and spans are the same problem, and
the same discipline applies to both.

Three properties worth carrying over:

- **Fail open.** The router escalates rather than guesses when it has no verdict. A wrong tool is a
  failed task; three schemas instead of one costs nothing. Widen the candidate set on low
  confidence.
- **Spare streams can verify the pick.** §2c establishes idle capacity riding the same forward
  pass, so a selection can be checked concurrently rather than serially.
- **The hard part is MCP, not the routing.** It is a stateful JSON-RPC protocol; serving `/v2/mcp`
  makes the engine an MCP server with sessions and lifecycle. There is also a federation question —
  host the tools or proxy them. Proxying keeps a network hop for *execution* but not for
  *selection*, and selection is where the per-turn cost lives.

## 7g. Code: symbol graphs, judging, and the fan-out boundary

Built: `CODE-GRAPH.md`, `internal/{codegraph,dag,prefill}/`.

**Generation.** Ordering comes from symbols, not from a types/impl/tests prior — the prior only
breaks ties the request left open, and never pulls a unit earlier than a real edge allows. Three
resolution outcomes where a task graph has two: a requirement nothing provides is **external**
(`context`, `std::vector`), not a broken extraction. A dependency **cycle is consolidated, not
rejected** — mutually recursive types have no per-file ordering and always have a joint one, so
Tarjan's SCC condenses them into one generation pass.

**Contracts are exact text.** Forwarding `h_last` from the type stream to the implementation stream
is foreclosed by the design's own mitigation — signature drift turns on precise bytes, and asking a
stream to reconstruct them from a projected hidden state reintroduces the drift the tiering removes.
Contracts render declarations **without bodies**; at 8,874 tokens per stream a dependent given
function bodies spends the scarce budget on code it must not reimplement. Injection reduces drift,
`CheckDrift` catches it.

**Judging.** Local passes emit grammar-constrained assertions; Go cross-references them. ⛔ **"Zero
conflicts" is not correctness** — the check contradicts claims that were made, and is blind to a
symbol nobody asserted about or a chunk that returned nothing. There is no `Correct()`, only
`NoConflictsFound()`, and coverage travels with every verdict.

**The header makes prefix reuse a prerequisite.** A shared symbol table across 48 streams is ~72k
duplicated tokens — 28% of a 256k payload, ~22 s at measured prefill — which is why the ~35-40 s
estimate lands nearer ~104 s.

**Fan-out is wide above the engine and single-threaded at it.** libllama's context is not safe for
concurrent use, and 48 concurrent callers would also produce 48 single-sequence passes instead of
one 48-sequence pass — losing exactly what `kv_unified` buys.

## 7h. What this ships as, and planning capacity for it

`DEPLOYMENT.md`. A **curated** runtime, not a general one: a closed model library at quantisations we
build, against known hardware — 3090/4090/5090 single and multi-GPU, plus Apple Silicon — with
unsupported combinations refused rather than attempted. One workstation, usually one agent, owning
the whole herd.

**The pool is the resource and it is not partitioned.** Under `kv_unified` a single sequence may take
the whole pool, so uneven stream sizing is an unused capability rather than a new one. Per-pass cost
is Σ(depth). The shipped layout comes from the observed distribution — **40×8k + 3×32k**, packing the
3090 pool exactly — with **no 128k class**, because a 128k stream competes with decomposition for the
same requests and is the worse answer to them. The cap is sequenced last: keep the existing ceiling as
a net until the >32k decomposition path is verified.

**Routing and admission take different inputs.** `max_tokens` is a ceiling, not a prediction — routing
on it sends `say "hello"` to the deep class on a client default nobody chose. Admission uses worst
case; class assignment uses input tokens alone.

**Single-tenancy deletes the hard part.** No fairness floors, no per-class queues — with one agent
there is nobody to be unfair to. Σ depth becomes a number the agent plans with rather than a rule the
engine enforces, and bounded input turns admission into a static check.

**Install gating is authored from measurement, never from a fit calculation.** 128 streams fit the
arithmetic on the 3090 and killed the container; a computed gate would have shipped that as supported.
`fit` generates candidates, `reference.go` is the authority.

**Two things carry the design.** Whether throughput scales with stream count at fixed Σ depth (#43)
is now measured: at a constant 393,216 resident tokens, 48 shallow streams retired 89.77 tok/s
against 50.98 for 8 deep ones, up to 1.9x (`results/3090.md`). Where the model's focus actually
degrades with depth (#44) is still folklore, and is the load-bearing input to the whole class
layout.

## 8. Sequencing

1. Engine core: decode loop, slot table, continuous batching, admission control.
2. **Chat-completions API with streaming.** Promoted: it is what makes the engine usable by
   anything at all, and every consumer above it depends on the contract. Getting it wrong is
   expensive to change once agents point at it.
3. Benchmark harness — reproducing the §1 numbers on real cards is the gate for everything after.
4. KV quantization and a real KV budget (§3).
5. MTP: re-run the speculation sweep axis — one boot, both arms (§7b). The standing verdict is
   57% acceptance and net-negative throughput; the re-run only tests whether the sampler rework
   changed it. One sweep answers it.
5a. ~~Find the stream-count turnover: sweep `56` through `96` at depth 0, never 128.~~ Done (§7b):
   plateau 48-64, cliff at 72 or 80 depending on the node, ship 48.
5a-i. Make sweep results durable before running another long sweep — persist outside the
   container, and have standby serve the partial file so rows are readable mid-run rather than
   only after the sweep completes (§7b).
5b. KV prefix reuse across streams sharing a prompt — built, likely the largest remaining win for
   chunked workloads, and unmeasured on a card (§7b).
6. Multi-GPU placement and capacity-aware routing (§4).
7. Model lines and published builds (§7).
8. Shared state, if a concrete use case demands it (§6).
