# Hierarchical Latent Speculative Reduction

Synthesising 48 parallel streams into one answer **without a text round-trip** — the streams hand
back latent vectors rather than generated text, and the answer stream is seeded from those vectors
instead of re-reading them as tokens.

Design owner: Ben. This document records the specification and the measurements that bound it.

---

## 1. What it replaces, and what that is actually worth

The naive synthesis path decodes text from every chunk, selects among it, and **re-prefills the
selection** into the answer context. On the measured 3090 profile that last step is the expensive
one: ~96k of selected text at ~3,300 tok/s is **29 seconds**.

HLSR removes it. The answer stream is seeded with a vector, so nothing is re-read.

| Phase | Text re-prefill | HLSR | Saving |
|---|---|---|---|
| 48-stream prefill (256k) | 78 s | 78 s | — |
| Parallel decode (~30 tok/stream) | 9 s | 0 s | −9 s |
| Synthesis context prefill (96k) | 29 s | 0 s | **−29 s** |
| Final answer generation | 8 s | 8 s | — |
| **Total** | **124 s** | **86 s** | **~38 s (31%)** |

⚠️ **Prefill dominates and no routing scheme changes it.** Reading 256k costs ~78 seconds on this
card — the library's `pp512` measured 2,843–3,136 tok/s across five boots and the engine's
single-stream prefill 3,528, and it is *worse* when split (splitting one input
48 ways lost 19% of ingest throughput). Any latency budget that puts the parallel phase in
milliseconds is wrong by three orders of magnitude, and it misdirects effort: eliminating the
chunk-decode step saves 9 s of 124, while eliminating the synthesis re-prefill saves 29.

## 2. Architecture

```
[48 chunks (8k each) + trailing synthesis instruction]
                    │
                    ▼
  48 parallel prefill streams          ~78 s, prefill-bound
  run to the instruction token
                    │
   extract last-layer hidden states    H ∈ R^(48 × d_model), stays in VRAM
                    │
                    ▼
  projection  W_proj : h_out → h_in    applied batch-wise, H·W ∈ R^(48 × d_model)
                    │
                    ▼
  pass 49: inject all 48 via llama_batch.embd at positions 0..47,
           user prompt at 48.., then autoregressive generation     ~8 s
           streams to the caller
```

## 3. Three mechanisms, and why each is needed

### 3a. Trailing synthesis instruction — makes the vector mean something

A causal model's final hidden state predicts **the next token**. It is not a summary of what came
before, and it is dominated by the last few tokens of local context. This is why embedding models
are trained separately, with pooling or contrastive objectives: a document representation is not a
free by-product of language modelling.

So each chunk ends with an instruction that makes the final position a *summary-predicting* state:

```
<chunk content, 8k>
---
Synthesize the primary key entities, facts, and assertions above into a dense representation:
```

The attention heads at that position must reach backward over the whole window to predict what
follows, which is what turns `h_last` from a local tail vector into an aggregate one.

Cost: ~12 tokens × 48 streams ≈ 576 tokens ≈ **175 ms** of prefill across the batch. Negligible
against 78 s, and it is the difference between a usable representation and a useless one.

### 3b. Injection at layer 0, not into the KV cache

**A KV-cache prefix cannot be built from a final-layer hidden state.** KV entries are per-layer —
2 × n_layers × d_head vectors per position — and a single last-layer vector does not contain them.
llama.cpp exposes no API to write K/V directly either.

The implementable path is `llama_batch.embd`, which is present in the pinned build (`b10545`,
`llama.h`): a batch may supply input embeddings instead of token ids, consumed at layer 0.

### 3c. Projection, because the spaces differ

`llama_get_embeddings_ith` (`llama.h:1048`) returns the **final layer output**. `llama_batch.embd`
is consumed as an **input embedding**. Both are `d_model`, so injecting one as the other compiles
and runs — and lands a vector in a space the model does not expect. The likely failure is drift and
degraded output rather than a clean error, which is the worst kind.

A learned `W_proj ∈ R^(d×d)` maps between them. Small, and it can be calibrated on paired states
from the model itself.

## 4. Decided: inject 48 vectors as a sequence, not one pooled vector

`llama_batch.embd` takes a **sequence** — `n_tokens × d_model` — so the 48 states are injected as
48 positions. Pooling to one is rejected.

| | Pooled (1×d) | Sequence (48×d) |
|---|---|---|
| Input feature matrix | 8 KB | 393 KB |
| Chunk-level detail | superposition blur | preserved |
| Decoder attention | one pseudo-token | 48 distinct positions |
| Layer-0 cost | <0.1 ms | ~0.4 ms |

**The deciding argument is superposition, not capacity.** Two chunks holding conflicting specifics
— "System A uses port 8080", "System B uses port 9090" — do not average into something vague. They
average into something *wrong*: a linear superposition in which the features scramble each other,
and the answer stream cannot recover which value belonged to which system. Positional separation is
what lets attention retrieve the right one.

The cost of avoiding that is **47 extra token positions**, ~0.3 ms, against a 78-second baseline.
Free by any measure that matters.

It also simplifies `W_proj` (#21): mapping `h_out → h_in` token-by-token is a linear calibration,
where a pooled design would require learning an aggregation kernel as well.

### Two mechanical consequences

**Ordering is preserved for free, and that is wanted.** Soft-tokens at positions 0..47 carry the
same positional encoding a real sequence would, so the model sees chunk 1 as preceding chunk 48.
Document order survives injection without anything extra.

**Prompt placement matters.** The user's prompt goes *after* the soft-tokens, so causal attention
lets it see every chunk. Placing it first would leave generation attending backward across 48
positions to reach the question. Sandwiching — prompt, chunks, prompt again — is the standard
mitigation for attention decay over distance and costs a few dozen tokens; worth testing once the
basic path works, not before.

## 4b. Superseded: pooling to a single vector

`llama_batch.embd` takes a **sequence** — `n_tokens × d_model`, not a single vector. So the 48
states can be injected as 48 positions rather than pooled into one.

The trade:

- **Pooled (1 vector):** one position, and a fixed ~100,000:1 compression of 256k. Cross-attention
  weighting by the query makes it as good as a single vector can be, but the collapse happens once,
  before the answer stream's own attention ever sees the parts.
- **Sequence (48 vectors):** 48 positions — negligible prefill — and each chunk keeps its own
  representation. The answer stream's attention does the weighting at every layer, rather than a
  kernel doing it once. It also removes the need for a separately trained pooler.

Worth testing both in the same experiment; the cost difference is 47 token positions.

## 5. Gating experiment — before any CUDA or 3090 time

The premise to falsify first is 3a: **does a last hidden state carry chunk content at all?** If it
does not, no pooling kernel or projection rescues it.

On a 0.5B, locally:

1. **Control.** Take a factual 1k chunk `C₁` with no trailing instruction. Extract `h_last`, inject
   via `llama_batch.embd`, generate 50 tokens.
   *Expected:* generic or hallucinated continuation driven by the tail.
2. **Test.** Same chunk with the synthesis instruction appended. Extract, project, inject, generate.
   *Pass condition:* the output references core facts from `C₁` that appear nowhere in the prompt.
3. **Separation.** The test that matters, because it targets superposition directly:

   - `C₁` = "System A uses port 8080", `C₂` = "System B uses port 9090"
   - append the synthesis instruction to both, extract `h_last` for each
   - project both, inject as a 2-vector sequence at positions 0 and 1
   - follow with the prompt: *"What ports do System A and System B use?"*

   **Pass condition: both port numbers recovered and correctly attributed.** Mixing them up is the
   failure a pooled design would produce by construction, and it is the sharpest available signal
   that positional separation is doing real work.

A control that fails and a test that passes is the signal. **Both failing means the representation
is the problem**, and the build stops there rather than after a pooling kernel is written.

## 5b. Result of the gating experiment (2026-08-23)

Run on Qwen2.5-0.5B-Instruct-Q4_K_M via `llama-herd latent-probe`. Two chunks with conflicting
specifics, synthesis instruction appended, injected as a two-vector sequence, followed by
*"What ports do System A and System B use?"*

**Measured:**

```
chunk 0: 40 tokens -> h_last[896]  L2 = 295.1
chunk 1: 39 tokens -> h_last[896]  L2 = 298.7

scale 1.00   degenerate  ("—" repeated)
scale 0.25   degenerate
scale 0.10   degenerate
scale 0.05   degenerate
scale 0.03   coherent, generic
scale 0.02   coherent: "系统 A 和系统 B"   ← entities recovered
scale 0.01   degenerate ("To" repeated)

8080 recovered: never.   9090 recovered: never.
```

### Correction (same day): the second finding below was wrong

A follow-up decomposition — generating **directly** from each chunk's own final position, before any
injection — separates two failures that look identical downstream: *the detail was never in the
state* versus *it was there and did not survive injection*.

With a value-targeting instruction (`"The port number mentioned above is:"`):

```
chunk 0  h_last L2=297.0   model continues directly with: "8080"
chunk 1  h_last L2=295.8   model continues directly with: "9090"

injected at scale 0.02  →  8080: never   9090: never
```

**The detail is in the state.** The model predicts the exact value from that position. It is lost
in the h_out → h_in mapping, not absent from the representation.

That makes `W_proj` (#21) the critical component rather than a polish step, and it gives it a
**demonstrable target**: injection should reproduce what direct continuation already produces. It
also supplies a supervised objective with unlimited data — train `W_proj` so the injected path
matches the direct path, on any text.

**And the instruction is a design lever, not a fixed prefix.** "Synthesize the key entities"
produced a state encoding entities; "the port number is" produced one encoding the port. The final
position encodes *whatever that instruction asks it to predict*. Since the user's query arrives with
the request, the chunk instruction can be **query-conditioned** — "given this question, what here is
relevant" — so the state carries what the answer will actually need.

The original two findings, corrected:

### Two findings, the second superseded above

**1. The space mismatch is real and largely a magnitude problem.** `h_last` has an L2 norm around
296 — far outside the input-embedding distribution — and injecting it unscaled produces garbage.
Scaled to ~0.02 (L2 ≈ 6) the model generates coherent text. That is what `W_proj` (#21) is for, and
it means the projection is at minimum a scaling, possibly not much more.

**2. The representation carries entities, not values.** At the coherent scale the model produced
"System A and System B" — it knew what the chunks were *about*. It never produced 8080 or 9090 at
any scale.

⚠️ **Superseded.** This was read as "a hidden state is a topic vector, not a content vector". The
decomposition above shows that is false: the state encoded the port, and the synthesis instruction
was simply asking it to encode something else. The conclusion drawn from it — that soft-tokens can
never carry detail — does not follow.

What survives: **injection currently loses detail that the state demonstrably holds**, so until
`W_proj` closes that gap, verbatim spans remain necessary and #14 remains load-bearing. The
difference is that this is now a fixable engineering gap with a training signal, rather than a
property of the representation.

### Confounds, stated

This was a 0.5B at Q4_K_M with **scalar scaling, not a learned projection**, over two chunks. A
larger model may carry more; a trained `W_proj` may recover more than a scalar can. What the result
establishes is the **burden of proof**: detail transfer is not free, and anyone proposing that
soft-tokens replace verbatim text now has to demonstrate it rather than assume it.

**What it cost to learn: one afternoon on a laptop-class model.** What it avoided: a pooling
kernel, a projection training run, and 3090 time spent on a pipeline whose premise was untested.

## 5c. The finding that questions the premise (2026-08-23)

HLSR exists to bypass text on the grounds that vectors are denser. That was tested directly.

Same two chunks, same value-targeting instruction. Instead of injecting each chunk's hidden state,
take each chunk's **top predicted token** and concatenate as ordinary text:

```
context given:
  Chunk 0: 8080
  Chunk 1: 9090

generated:
  "System A uses port 8080 and System B uses port 9090."
  8080: true   9090: true
```

**One token per chunk, as text, recovers everything. The vector recovers nothing.**

### Why this is not just an implementation gap

For a model with **tied embeddings** — as small Qwen models have — `logits = E · h_out`, so the
natural `h_out → h_in` map is `Eᵀ · softmax(logits)`: the expected input embedding under the
predicted next-token distribution.

Which means **an injected vector carries approximately one token of information**, plus the shape of
the distribution around it. Not a compression of 8k. One token, softened.

So 48 soft-tokens are worth roughly a 48-token summary — and 48 tokens of text cost nothing to
prefill. The ceiling on HLSR's advantage over "emit a short digest per chunk and concatenate" is the
difference between a token and a token distribution, which is small.

### What this promotes

⚠️ **Qualified by the tied/untied finding below** — this comparison was run on a tied model, where
a hidden state reduces to about one token. **Strategy 3 remains the stronger design *today*** —
available with no projection, kernel or training, and it
is available now: no projection, no pooling kernel, no training, no new API. Grammar-constrained
digests concatenated as text, which is classic map-reduce and works today.

The latent path must **beat** that, not merely work.

### Two limits on this comparison, stated

**The text path was given the ideal instruction.** "The port number mentioned above is" made the top
token the answer. In general the right instruction is not known at index time — except that it
partly is, because the query arrives with the request (#27). Query-conditioned digests are exactly
map-reduce.

**The vector path is currently broken.** No trained `W_proj`, only scalar scaling. A fair comparison
needs #21 finished. But the tied-embedding argument bounds how much that can help: a working
projection recovers *one token's worth*, which is what the text path already has.

### Resolved: the 35B is untied, and the premise survives

Read from the GGUF tensor directory — a 32 MB range request, no download, no GPU:

```
Qwen2.5-0.5B    token_embd.weight [896 x 151936]     output.weight ABSENT   → TIED
Qwen3.6-35B     token_embd.weight [2048 x 248320]    output.weight [2048 x 248320]   → UNTIED
```

**So the 0.5B result does not transfer.** The one-token reduction — `Eᵀ·softmax(logits)` — depends
on the embedding being reused as the output projection. On the 35B they are separate matrices, so
`h_out` lives in a genuinely different space with no such collapse, and may carry considerably more
than one token.

Which means:

- The negative result in §5c is **a property of the 0.5B, not of the architecture.** It was the
  right experiment on the wrong model, and it would have been a wrong conclusion to generalise.
- `W_proj` (#21) is back to being the critical component, and its supervised objective stands.
- The gating experiment must be **re-run on an untied model** before HLSR is judged. The 0.5B can
  no longer answer this question.

⚠️ **The cheap local model is no longer a valid stand-in.** Any future latent-space experiment needs
an untied model, and tied-vs-untied should be checked *first* — it is a metadata read that decides
whether the experiment means anything.

## 6. What is verified, and what is not

**Verified against the pinned build:** `llama_batch.embd` exists; `llama_get_embeddings_ith` and
`llama_get_embeddings_seq` exist; `Embeddings` is already a context parameter in our binding.

`d_model` for Qwen3.6-35B-A3B is **2048**, read from the model file (`token_embd.weight` is
`[2048 x 248320]`, above). The KB figures in the sizing use it: 48 x 2048 x 4 bytes is 393 KB.

**Now measured (§5b):** injection at layer 0 *does* produce coherent generation, but only once the
vector is scaled into the input distribution; a summary-instructed final hidden state carries the
chunk's **entities but not its values**.

**Now measured (correction):** the state **does** encode specific values — direct continuation from
the chunk's own final position produced the exact port numbers. The loss is in the h_out → h_in
mapping.

**Still not verified:** whether a learned `W_proj` closes that gap; whether query-conditioned chunk
instructions carry the right content; whether 48 vectors behave like 2.

**Built and run:** the embeddings binding and the `embd` batch path (`internal/llama/latent.go`),
and the `llama-herd latent-probe` command whose results are in §4. **Not built:** the pooling
kernel, the projection, and dispatch. Nothing has run on the 35B.
