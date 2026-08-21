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

| Card | VRAM | Goal (aggregate decode) |
|------|------|-------------------------|
| RTX 3090 | 24 GB | 500 tok/s |
| RTX 4090 | 24 GB | 750 tok/s |
| RTX 5090 | 32 GB | 1000 tok/s |

These are not invented. On a single 3090, a prior implementation of this architecture measured:

| Model | Streams | Aggregate | Per stream |
|-------|---------|-----------|------------|
| 35B MoE (A3B), 4-bit | 1 | 140.7 tok/s | 140.7 |
| 35B MoE (A3B), 4-bit | 6 | 344.7 tok/s | 57.4 |
| 35B MoE (A3B), 4-bit | 10 | **407.7 tok/s** | 40.8 |
| 8B-class, Q4 | 6 | ~318 tok/s | ~53 |

So ~500 tok/s on a 3090 is within reach of the *existing* mechanic for a smaller model, and the
4090 and 5090 goals follow from memory bandwidth plus the levers in §2. The decode loop is
**memory-bandwidth bound**, not compute bound: one pass over resident weights serves every active
stream, which is why aggregate throughput climbs with stream count while per-stream throughput
falls.

## 2. Multi-token prediction — the main lever

Models with native MTP layers (Qwen, Granite, Nemotron) can propose several tokens per step and
have them verified in a single forward pass. On a bandwidth-bound decode loop this is the largest
single win available.

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

That leaves three routes, and the choice matters more than the code:

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

Route 3 is implementable now and is the sane first step: it makes the draft-verify loop real
and measurable against any model, and the loop is the same regardless of where drafts come
from. Route 1 then swaps the draft source once the loop exists.

Work items:

- **Implement the draft-verify loop** with a draft model, using public API only. The loop is
  the reusable part; the draft source is swappable.
- Decide how a draft context shares the KV budget. It is a second context over the same
  weights, so its cache competes with the target's for the same VRAM.
- Confirm `load_mtp` finds the tensors in each candidate quant, which is already measurable.
- Measure accept rate and the real end-to-end speedup, per model and per card.
- Fall back to draft-model speculative decoding where a model has no native MTP head.

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

1. **KV quantization** (`type_k` / `type_v`) — the largest and cheapest win. Quality impact needs
   measuring, not guessing.
2. **Admission control against a KV budget** rather than a slot count, so a long-context request is
   admitted only when its cells actually fit.
3. **Sliding-window / partial attention** where the model supports it.
4. **KV offload and paging** to host memory for idle streams — costs PCIe bandwidth, buys slots.
5. **Prefix sharing** across streams with a common prompt, which is close to free when it applies.

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

Published alongside the engine:

- **Qwen line** — native MTP, strong across sizes.
- **GLM line** — likely 4.7.
- **Nemotron line** — candidate; native MTP via its own head.

Each build ships a model card recording base model and its licence, quantization, KV precision,
whether MTP tensors are retained, and the manifest settings — stream count, context, KV budget —
for its target card. A build without those numbers makes the user re-derive them.

Distribution is **Hugging Face** for general availability and **Cloudflare** for application
delivery. Note that base-model licences travel with quantized builds and are not Apache-2.0; each
line needs its licence resolved before publication.

## 8. Sequencing

1. Engine core: decode loop, slot table, continuous batching, admission control.
2. **Chat-completions API with streaming.** Promoted: it is what makes the engine usable by
   anything at all, and every consumer above it depends on the contract. Getting it wrong is
   expensive to change once agents point at it.
3. Benchmark harness — reproducing the §1 numbers on real cards is the gate for everything after.
4. KV quantization and a real KV budget (§3).
5. MTP: verified support, then a quant that retains the head (§2).
6. Multi-GPU placement and capacity-aware routing (§4).
7. Model lines and published builds (§7).
8. Shared state, if a concrete use case demands it (§6).
