# Deployment: what this ships as, and how capacity is planned for it

Companion to `ARCHITECTURE.md`, which covers the engine's shape. This one covers the product it is
inside: what hardware it targets, how a configuration is chosen for that hardware, and which of
those decisions are measured rather than reasoned.

---

## 1. Curated, not general

llama-herd is not a runtime that tries to run anything anywhere. The model library is **closed and
known**: a fixed set of models, at quantisations we build (`MODEL-BUILD.md`), against a fixed set of
target hardware. A combination that is not on the list is not offered.

That is the opposite of the general-purpose posture, and it is deliberate. Every number in
`results/` is specific to a model, a quant, a build and a card. A runtime that accepts arbitrary
combinations cannot make any of the promises those numbers support, because it has measured none of
them.

**Target hardware:** NVIDIA RTX 3090 / 4090 / 5090, single and multi-GPU, plus Apple Silicon.

**Deployment shape:** one workstation, usually **one agent**, owning the whole herd. Not a
multi-tenant service. §5 covers how much that assumption removes.

---

## 2. The pool is the resource, and it is not partitioned

A configuration is a KV pool of `n_ctx` tokens shared by `n_seq_max` streams. The measured 3090
configuration is 425,984 tokens across 48 streams.

⛔ **"8,874 tokens per stream" is arithmetic, not a ceiling the library enforces.** The manifest
enforces it instead: `admit_context` must not exceed `context / streams`, and under `kv_unified`
with more than one stream it is required. From `llama-context.cpp`:

```cpp
if (cparams.kv_unified) {
    cparams.n_ctx_seq = cparams.n_ctx;                        // any sequence may use the WHOLE pool
} else {
    cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;    // fixed equal share, hard ceiling
}
```

With `kv_unified` — which both example manifests use — the library lets a single sequence occupy the
entire pool. Boots have run with one sequence holding all 425,984 tokens and with four holding
106,496 each, so uneven sizing is not a new capability in the library. It is one the manifest now
refuses: `kv_unified` with more than one stream requires `admit_context`, and that cap is one value
per model, so every stream is admitted against the same ceiling. Uneven classes need the routing
below before they can exist.

**Allocation and depth are different resources.** Allocation is a memory reservation that, under a
unified pool, is not actually reserved; **depth** is how many tokens a sequence currently holds, and
it is what costs compute. A stream allowed 32k while sitting at 2k costs 2k.

---

## 3. The cost model: sum of depths

One decode pass emits one token per active stream, and a stream's attention work scales with its
current depth. So per-pass cost is proportional to **Σ(depth)** across active streams, and

```
aggregate throughput  ≈  active streams / pass time
```

Measured, at 16k depth per stream (`results/3090.md`):

| streams | Σ depth | aggregate | pass time | per stream |
|---|---|---|---|---|
| 8 | 131,072 | 112.09 | 71.4 ms | 14.01 |
| 16 | 262,144 | 59.35 | 269.6 ms | 3.71 |
| 24 | 393,216 | 51.62 | 464.9 ms | 2.15 |

Two consequences:

- **A deep stream should be counted as several streams.** A 128k stream carries about sixteen times
  an 8k stream's attention work on every pass.
- **A deep stream does not pay for its own depth — the herd does.** It receives one token per pass
  like everyone else, so its own tok/s looks unremarkable while it makes every pass more expensive
  for every other stream.

⚠️ **The decisive question is unmeasured.** Every row above varies stream count and Σ depth
*together*, so they cannot separate the two. The open question is whether, **at fixed Σ depth**,
aggregate throughput scales with the number of streams sharing it. If it does, splitting a pool into
many shallow streams plus a few deep ones wins large. If Σ depth is all that matters, the split buys
nothing and the right answer is a uniform shallow herd plus decomposition. One boot settles it: hold
Σ near 384k and run 24×16k, 48×8k, 96×4k (#43).

---

## 4. Workload classes

**Proposed layout, from the observed request distribution** — the vast majority of requests fit in
8k, outliers reach 32k, and effectively none exceed it. What ships today is a uniform herd: 48
streams admitting 8,192 each (`examples/3090-throughput.json`), because `admit_context` is one
value per model and a 32k class would exceed the 9,906-token share that 43 streams leaves.

| class | streams | tokens each | pool | role |
|---|---|---|---|---|
| shallow | 40 | 8,192 | 327,680 | the vast majority of traffic |
| deep | 3 | 32,768 | 98,304 | outliers |
| | **43** | | **425,984** | packs exactly |

⚠️ **This replaces an earlier 8×32k + 40×4k proposal**, which was derived from packing arithmetic
before the real distribution was known. Operational observation beats a tidy split; the layout above
is what the traffic actually looks like.

Note that `3 × 32,768 = 98,304` still exceeds any single request seen in practice, and the deep class
is sized for concurrency of outliers rather than for one enormous request.

### Why there is no 128k class

A 128k ceiling is **free while idle** — under a unified pool a stream occupies its depth, not its
class, so an unused ceiling reserves nothing. It is expensive in two other ways: in use it is 31% of
the pool in one stream, and its worst-case accounting drops the herd from 43 producers to 31 to make
room for it.

But the deciding argument is not arithmetic. ⛔ **A 128k class competes with the decomposition layer
for exactly the same requests, and decomposition is the better answer to them.** It is an escape
hatch from the mechanism that produces the quality: given the choice, a large request takes the
single degraded-focus context instead of 8×32k with cross-validation, and takes it slower. It would
be a fast path to the worse outcome — and at four times the depth where focus is already suspected to
degrade (§4, #44).

So: **cap at 32k and route anything larger to decomposition.**

### ⏸️ Sequencing: lower the ceiling last

The ceiling today is `admit_context`, an admission constant: the shipped 48-stream profile sets it
to 8,192 and refuses anything larger at submit, and a profile that admits 40k serves a 40k request
— slowly, at degraded focus, but it works. After a 32k cap it needs the decomposition path to be
reliable, and that path is not yet measured end to end (#32, #38).

So the cap is not the first move. **Keep a generous ceiling as a safety net while it costs
nothing, build the >32k decomposition path, verify it against the real outliers, then lower the
ceiling.** Under a unified pool this is cheap to defer and cheap to reverse — adding or removing a
ceiling changes no structure, only an admission constant. The ceiling is a transition aid, not a
design commitment.

### The 32k question

⭐ **Why 32k and not more, and this is the strongest argument in the design:** the model's focus is
reported to degrade around 32k. If that holds, 32k is not a performance tuning choice — it is the
point past which a longer stream is *worse*, not merely slower. That kills the single-128k-stream
case on quality grounds independently of throughput, and it is the sharpest justification the whole
decomposition layer has: **chunking is not a workaround for a small pool, it is what keeps every
stream inside the range where the model is still good.**

⚠️ **Unmeasured, and it is currently the load-bearing input to this whole design.** No
quality-versus-depth measurement exists in this repository. A retrieval probe at 4k/8k/16k/32k/64k on
the shipped model and quant would set the deep-class ceiling on evidence instead of folklore (#44).

For reference, a virtual context of 256Ki is **8 × 32,768 = 262,144** — reachable by the
decomposition layer across deep-class streams without any single stream exceeding 32k. That is the
shape the cap is meant to force.

## 5. What a single agent removes

The natural worry about mixed classes is that one deep request starves the shallow ones — head-of-line
blocking, needing reserved floors, per-class queues and fairness accounting.

**With one agent owning every stream, that whole mechanism is unnecessary.** There is no second party
to be unfair to, and the agent already knows what it dispatched.

What survives the simplification, and it is worth separating:

- **Σ depth is physics, not policy.** Eight streams at 32k still slow the forty shallow ones. The
  difference is that this becomes a number the agent needs in order to plan, rather than a rule the
  engine must enforce. **Report it; do not police it.**
- **The hard refusal at admission stays**, and matters more here rather than less. If a single agent
  oversubscribes the pool, that is a bug in its own decomposition, and a refusal at submit time is a
  far better failure than a stream evicted mid-answer — which surfaces as a bad answer with nothing
  in the logs.
- **Bounded input turns admission into a static check.** Worst-case depth is
  `len(tokenize(prompt)) + max_tokens`, both known at submit time — no prediction of output length is
  needed, because `max_tokens` *is* the bound. If class sizes are fixed, worst-case Σ depth is known
  at boot. No overcommit accounting, no eviction path to reason about.

### ⛔ Admission and routing are different decisions with different inputs

Worst-case depth is the right input for *admission* — can this fit at all. It is the wrong input for
*class assignment*, and using it there produces an obvious absurdity: **`say "hello"` routes to the
32k class.**

`max_tokens` is a **ceiling, not a prediction**. A client that leaves it at a 4096 default sends a
one-token greeting to the deep tier on the strength of a number nobody chose, and every trivial turn
lands in the expensive class.

The resolution is that under a unified pool the two decisions do not need the same input, because
**a stream costs its DEPTH, not its class.** Nothing is reserved; `hello` occupies two cells whichever
class it nominally belongs to. A class is therefore a *permission ceiling on growth*, not an
allocation.

So, as designed (today's admission checks `input + max_tokens` against a static per-stream budget,
`min(context / streams, admit_context)`, and does not read pool occupancy — the live checks below
are not built):

| decision | input | why |
|---|---|---|
| may this be admitted | `input + max_tokens` against remaining pool | worst case, because eviction is silent |
| which class | **input tokens alone** | exact, free, and the thing that actually separates a greeting from a document |
| may it keep growing | live pool occupancy | checked as it grows, not reserved up front |

⛔ **Do not reach for a classifier for this.** A cheap turn-router is the right instinct for tool
selection (`ROADMAP.md` §7e), but `len(tokens)` is exact and costs nothing. A model that predicts
what arithmetic already knows is strictly worse: it can be wrong, and it cannot be more right.

⚠️ **The 32k cap is on STREAMS, not on user input.** A user still hands the system 256k; the
decomposition layer is what turns that into 8×32k. If the cap were on input, virtual context would
have no purpose. This distinction is the hinge the entire layer rests on.

---

## 6. KV quantisation is the lever on pool size

Measured on the 3090 at 16k of context per stream, median of two, one node (`results/3090.md`),
aggregate tok/s:

| Streams | q8_0 / q8_0 | q8_0 / q4_0 |
|---|---|---|
| 2 | 119.12 | 108.90 |
| 4 | 105.77 | 118.61 |
| 8 | 109.35 | 106.70 |
| 16 | 70.20 | 68.24 |

**Asymmetric KV buys nothing measurable in throughput.** An apparent 1.23x for q8_0/q4_0 at 8
streams came from a single-repetition run against a baseline that had run slow; repeated, the
ratios were 0.91, 1.12 and 0.98, noise with a sign. `q4_0/q4_0` measured below `q8_0/q4_0` at
every 16k row. What V at q4_0 does buy is memory: it cuts KV bytes about 25%, close to the 23.1%
that 16×32k needs, and q4_0/q4_0 roughly halves them.

⚠️ **These are throughput measurements only.** Nothing here measures what the quantisation costs in
answer quality, and it would be the *third* quality expenditure in the same place: IQ3_S weights, a
q4_0 V-cache, and 32k depth where focus is already suspect. They compound. **Do not spend pool on
V-quant before the depth probe (#44) exists** — that is spending quality budget blind, in exactly the
place it is thinnest.

---

## 7. Multi-GPU

`SplitMode`, `TensorSplit` and `MainGPU` are already plumbed (`internal/llama/`). From `llama.h`,
both relevant modes **split layers *and KV*** across devices, so the pool genuinely scales with card
count:

```
LLAMA_SPLIT_MODE_LAYER = 1,  // split layers and KV across GPUs
LLAMA_SPLIT_MODE_ROW   = 2,  // split layers and KV across GPUs, tensor parallelism if supported
```

⛔ **Three 3090s are not one 72 GB card.** A supported-configuration entry keys on more than total
VRAM:

- **`ROW` depends on the interconnect** and can lose to `LAYER` outright on PCIe without NVLink.
- **`TensorSplit` proportions** decide whether the pool is balanced or one card becomes the ceiling.
- **Mixed cards split by proportion, not capability** — a 3090 beside a 4090 is paced by the slower.
- **Per-device minimum matters independently of the total.** A model needing 40 GB across two devices
  does not necessarily run on six 12 GB cards.

So the key is `(model, quant, [cards], split_mode)`.

⭐ **The best use of a second and third card is probably quality, not capacity.** IQ3_S weights, 32k
depth and a quantised V-cache all push the same direction. Extra VRAM spent on Q6 or Q8 weights
instead of on more streams attacks the focus problem from the other side, and the depth question may
look different there. Worth measuring once the tier exists rather than assuming.

---

## 8. Hardware detection and the model map

Detection is **already built and platform-agnostic**. `llama.Devices()` enumerates through ggml's
backend registry, so CUDA, Metal, Vulkan and ROCm all arrive through one path with per-device
`TotalBytes` and `FreeBytes`; `GPUs()` filters to devices with memory worth placing weights in, and
heterogeneous hosts are handled by construction.

**Built** (#45): `internal/catalog/` — `models.json` embedded with `//go:embed`, matched against the
detected devices, surfaced by `llama-herd models [--all]`.

An entry states a total memory requirement, a **per-device** requirement, a device count, and the
configuration to run. The per-device floor is not derivable from the total and is the constraint
people are surprised by: six 12 GiB cards beat 2×24 GiB on paper and cannot hold a layer that needs
20.

`LLAMA_HERD_CATALOG` names a replacement file — an explicit escape hatch, so someone with hardware
nobody has measured can try a combination at their own risk rather than be unable to run. ⛔ A
missing or malformed override is an **error**, never a silent fall back to the builtin: falling back
would run a configuration the operator did not choose while reporting success.

The refusal **says why**, because a model silently absent from a list tells nobody anything and reads
like a bug. Real output:

```
the device needs 40.0 GiB usable; found 22.8 GiB on "RTX 3090" after margin
each of 2 devices needs 20.0 GiB usable; found 11.4 GiB on "RTX 3060" after margin
needs 2 devices, found 1
the device needs 40.0 GiB usable; found 38.4 GiB on "Apple M3 Max" after margin
```

The last line is the unified-memory margin doing its job: 48 GiB reported, 38.4 usable after a 20%
reserve, where the same figure on a dedicated card would clear at 45.6.

### ⛔ The authoring rule: measured, not computed

The map is populated from **configurations that were run**, never from a fit calculation.

This project has the counterexample. **128 streams fit the arithmetic on the 3090 and took the
container with it.** 72 streams fit, and collapsed on one node while running on another. A gate keyed
to `KVBytesPerToken × pool ≤ VRAM − weights` would have accepted both.

The error runs the other way too: a conservative computed gate refuses configurations that work, and
for a product that is worse than a crash, because the user never learns it would have been fine.

`fit` remains useful as a **candidate generator** — cheap and exhaustive across combinations. It is
not the authority. `internal/bench/reference.go` is the authority, and the published
supported-configuration table should be generated from the same structure the install gate reads, so
documentation cannot drift from enforcement.

---

## 9. Apple Silicon

Metal registers as `GGML_BACKEND_DEVICE_TYPE_GPU` unconditionally, so `IsGPU()` includes it correctly.
(CUDA distinguishes `IGPU` from `GPU`, which is the intended exclusion for integrated NVIDIA parts.)

**The reported numbers are already the right ones.** From `ggml-metal-device.m`:

```objc
*total = dev->mtl_device.recommendedMaxWorkingSetSize;
*free  = *total - dev->mtl_device.currentAllocatedSize;
```

`recommendedMaxWorkingSetSize` is Apple's safe allocation ceiling — roughly 70-75% of unified memory,
not the raw RAM figure — so a 64 GB machine reports about 48 GB, which is the number a fit check
wants. No special-casing: `TotalBytes` means "usable for weights and KV" on both platforms.

⚠️ **One real asymmetry to encode in the map.** On a discrete card the VRAM is yours. On Apple
Silicon that ceiling is shared with everything else the machine is doing, and `FreeBytes` moves
underneath you — `Device` already documents itself as "a snapshot, a budget input rather than a
guarantee." On a 3090 that is pedantic. On a Mac where a browser may take 8 GB mid-run it is the
difference between an install that works and one that OOMs on a bad afternoon. **Unified memory wants
a larger headroom margin than dedicated VRAM.**

---

## 10. Open

| | |
|---|---|
| #43 | Answered: at fixed Σ depth, more and shallower streams are up to 1.9x faster (`results/3090.md`), so classes are worth having |
| #44 | Quality versus depth on the shipped model — sets the deep-class ceiling, currently folklore |
| — | Multi-GPU measurement across split modes and card counts; nothing here is measured yet |
| — | Less-quantised weights on multi-card, measured against the depth probe |
