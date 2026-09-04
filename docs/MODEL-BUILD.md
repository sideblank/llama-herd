# Building models

The engine is half the product; the weights are the other half. This describes how llama-herd
model builds are produced, why they exist at all, and what every published build must carry.

---

## Why build our own

Redistributed GGUF quantizations are unreliable for our purposes in a specific, narrow way:
**multi-token-prediction tensors are frequently absent**, and nothing on the model page tells
you. MTP is the largest throughput lever available on a consumer card, and it cannot be
recovered after the fact — a quantization without the tensors simply has nothing to run.

The gap is not a limitation of the tooling. Upstream conversion **includes MTP tensors by
default**; a publisher must pass an explicit flag to drop them. What varies is publisher
intent:

- Some ship the model with MTP included.
- Some run `--no-nextn` on the model and `--mtp` on a second pass, publishing the target and
  the draft head as two separate files. That is a legitimate speculative-decoding setup, but
  a different one, and it is easy to download half of it without noticing.
- Some publish thirty quantizations and offer MTP for exactly one of them, at a quant level
  you may not want.

Any of those can also change on a re-upload. So the tensors we depend on most are the ones we
control least. That is the whole argument.

Vision has the same shape: the projector is a separate export (`--mmproj`), so a vision model
published without one is text-only regardless of what its weights can do.

## The candidate pool is small

llama.cpp supports **88 model families**. **Seven** of them implement MTP export. Since the
fastest decode path runs through an MTP head, and that cannot be added to a model that has
none, those seven are effectively the candidate pool for a published line.

The other 81 can be served perfectly well. They just cannot be one of this project's lines, because
the reason a line exists is the decode speed MTP buys.

## What upstream supports

MTP export is implemented per architecture. At the pinned revision it covers:

| Family | MTP export |
|---|---|
| Qwen | yes |
| GLM | yes |
| Nemotron | yes |
| DeepSeek | yes (plus a separate DSpark draft export) |
| Hunyuan, BailingMoE3, Step3 | yes |

Four conversion modes matter:

| Flag | Produces |
|---|---|
| *(none)* | full model **including** MTP tensors |
| `--no-nextn` | model **excluding** MTP tensors |
| `--mtp` | **only** the MTP head, as a standalone draft |
| `--mmproj` | the multimodal projector, as a separate file |

Our default is the first: one file, MTP included. The split form is worth producing only when
a target/draft pair is measurably faster than the combined head for that model.

## Sizing depends on the attention architecture, not the parameter count

The capacity question turns on KV cost per token, and that is set by how many layers actually
hold a cache — which is not always all of them.

**Hybrid architectures cache only every Nth layer.** The rest use linear attention, whose
state is constant-size and does not grow with context at all. A 48-layer model with a
full-attention interval of 4 holds KV on 12 layers, so its cache costs a **quarter** of what
a dense model of the same depth would.

The difference is large enough to invert a conclusion. A dense 64-layer, 8-KV-head model
needs 32 GiB for 512k tokens of KV — more than a 24 GB card holds, at any quantization. A
hybrid 48-layer model with 4 KV heads needs **6 GiB at q8** for the same 512k, which leaves
room for the weights on the same card.

So the selection criterion is the attention geometry, not the parameter count. Two models of
similar size can differ by 4× in what they can hold. `llama-herd fit` reads the geometry from
the file, including the full-attention interval, and reports the real number.

**Quantized KV requires flash attention.** Setting a quantized KV type without it does not
work, and that pairing has to be carried in the manifest together.

## The pipeline

Each stage has a gate. A build that fails a gate is not published. `scripts/build-model.sh`
automates `acquire`, `convert`, `imatrix` and `quantize` with the verify gates between them;
`modify`, `measure` and `publish` are done by hand today.

```
1. acquire    pin the source repo AND its revision — not just the repo name
2. modify     any weight-level transform (abliteration etc.) — on safetensors, before GGUF
3. verify     assert MTP tensors survived the transform
4. convert    convert to BF16 GGUF, MTP included; --mmproj separately for vision models
5. verify     assert the MTP tensors are actually present
6. imatrix    compute the importance matrix on a recorded corpus — on the MODIFIED weights
7. quantize   produce the target quant levels using that imatrix
8. verify     assert MTP survived quantization, at every level
9. measure    quality against the modified BF16, MTP accept rate, throughput per card
10. publish   weights plus a card recording every input above
```

### Ordering that is not negotiable

**Weight-level modification comes before conversion.** Tools that alter weights directly —
abliteration being the obvious one — operate on safetensors, not GGUF. Running one after
quantization is not possible, so it belongs early.

**An imatrix computed before modification is invalid after it.** The importance matrix
describes which weights matter; changing the weights changes the answer. Reusing a
pre-modification imatrix silently produces a worse quantization, and nothing about the output
reveals it.

**Verify MTP after modification too.** A transform that rewrites attention or MLP projections
may or may not preserve the MTP head, and it is cheaper to check than to discover after
quantizing. This is a third distinct place the tensors can be lost, alongside conversion and
quantization.

**Quality is measured against the modified BF16, not the original.** Comparing a modified
quantization against unmodified weights conflates two effects and tells you nothing about the
quantization.

### Gates that matter

**After conversion and again after quantization, verify MTP presence.** Not the metadata
declaration — the tensors. `llama-herd inspect` reads the declaration cheaply, which is
enough to catch a source model that never had MTP; confirming the tensors survived requires
loading with the layers enabled and checking the loader read them. Both checks belong in the
pipeline, because they fail at different stages for different reasons.

**Measure quality, do not assume it.** An imatrix-guided quantization can still be worse than
a naive one if the corpus is unrepresentative. Perplexity against the BF16 source, and KL
divergence where available, are cheap next to the cost of publishing a bad build.

**Measure the MTP accept rate, not just its presence.** Tensors that load but propose poorly
give back little. The accept rate is the number that decides whether MTP earned its VRAM, and
it varies by model and by workload.

## Reproducibility

A build is defined by the whole set, and all of it goes in the card:

- source repository **and revision hash** — model repos are updated in place
- llama.cpp revision used for conversion and for quantization, which may differ
- imatrix corpus, its size, and how it was sampled
- quantization type per tensor class, where it deviates from the preset
- whether MTP tensors are present, and the measured accept rate
- whether a projector is included
- the manifest settings each target card was measured with

Two builds with the same name and different source revisions are different models. Naming
carries the source revision for that reason.

## Licensing, per family

Base-model licences travel with quantizations and are **not** uniform:

- **Qwen** — generally Apache-2.0, but confirm per model; some releases differ.
- **GLM** — varies by release; several carry use restrictions.
- **Nemotron** — NVIDIA's open model licence, which is **not** Apache-2.0 and carries terms
  worth reading before redistribution.

**Weight modification interacts with licence terms.** Some model licences carry acceptable-use
terms that speak to safety behaviour; whether a given transform is permitted is a licence
question per model, not a general one, and it belongs in the same check as redistribution.

Resolve the licence per model **before** conversion, not before publication. Converting is
cheap; discovering after a GPU-week of imatrix and quantization work that a model cannot be
redistributed is not.

## One pipeline, many models

The build is driven by a spec file per model and a single script. Bringing up a new model is a
spec plus whatever localised tweaks that architecture genuinely needs — a projector, a
different quant per card — not a new pipeline.

```bash
scripts/build-model.sh models/<name>.json              # everything
scripts/build-model.sh models/<name>.json --stage convert
```

The spec format is in [../models/SPEC.md](../models/SPEC.md). The build environment is a
separate image ([../build/Dockerfile.modelbuild](../build/Dockerfile.modelbuild)) because
building needs Python, torch and the quantization tools, none of which belong on a node whose
only job is answering requests.

### The gates, and why each one exists

Every gate catches a failure that is otherwise silent:

- **Licence recorded** before anything runs. Converting is cheap; discovering after a GPU-week
  that a model cannot be redistributed is not.
- **Revision pinned** to a commit. Repositories are updated in place, so a branch name does
  not identify a build, and two builds with the same name could differ.
- **MTP present after conversion**, checked against the file rather than the model card.
- **MTP present after quantization**, checked separately at every level. It fails here for
  different reasons than it fails at conversion, which is why one check is not enough.
- **Corpus recorded** before the importance matrix runs. A matrix is only as good as what it
  saw, and a reader needs to know whether that matched their workload.
- **Capacity reported per target card** from the finished file, so the serve settings in the
  spec are checked against reality rather than assumed.

## Where builds run

Conversion is CPU and disk bound. Imatrix computation and all measurement need a GPU, and
measurement needs *the target card* — a quantization tuned for 24 GB is a different decision
from one tuned for 32 GB. That makes the build process card-aware rather than a single
pipeline with one output.

## What a published build must state

Every model card carries, at minimum:

- source model and revision
- quantization type and imatrix provenance
- **whether MTP tensors are present**, and the measured accept rate
- whether a vision projector is included
- context length, and the KV cost per token at each supported KV precision
- a manifest fragment sized for each target card: streams, context, batch
- measured throughput on that card, with the method that produced it

The point of the last two is that a user should not have to re-derive the settings. A build
that ships without them makes every adopter repeat work we already did.
