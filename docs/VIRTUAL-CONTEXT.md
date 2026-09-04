# Virtual context

A layer above the engine that accepts an input of any size, splits it across streams, keeps
cross-chunk state outside the model, and reassembles the answer — so callers see a large context
window on a card that cannot hold one at speed.

This document exists because the measurements that constrain the design are already in hand, and
several of them point the opposite way from intuition. Numbers: `results/3090.md`.

---

## 1. The shape

- A request under one stream's context share is served by **one stream, unchunked**. No splitting,
  no reassembly, no overhead. This is the common case and it must stay cheap.
- Above that, the input is divided into chunks sized to the per-stream share, dispatched across
  streams, and the outputs recombined.
- Chunk size is **dynamic** — derived from what arrived, not a fixed constant. A 64k input and a
  256k input produce different splits.
- Cross-chunk state lives in an embedded store outside the model, because the chunks cannot see
  each other.

The engine is unchanged by this. It serves streams; the layer decides what to put in them.

## 2. What the measurements say, including where they contradict intuition

**Splitting the same content across more streams is faster.** Holding total tokens resident
constant and varying only the split measured up to **1.9x** in favour of more, shallower chunks
(131k resident: 8x16k = 91.65 tok/s, 48x2.7k = 171.36). What costs throughput is depth within a
sequence, not occupancy across the herd. This is the finding that makes the architecture
attractive rather than merely convenient, and it is the opposite of what "the same tokens cost the
same wherever they sit" predicts.

**Prefill does not chunk.** Splitting one input across more streams *lost* about 19% of ingest
throughput (3528 -> 2853 tok/s from 1 to 48 chunks). One long prefill already saturates the card;
dividing it adds overhead rather than parallelism. Reading 131k tokens costs roughly 40 seconds
whatever the split.

**So the two halves of a job pull in opposite directions**, and the chunk count must follow which
half dominates:

| Workload | Bound by | Chunking |
|---|---|---|
| Read a lot, emit a little (summarise, extract, classify) | prefill | costs ~19%, do it only to fit |
| Generate substantially, or serve many callers at once | decode | gains ~2x |
| Many small independent requests (sub-agents) | decode | already the native shape — no chunking needed |

**Sub-agent traffic is the case this engine is already best at.** Many short, independent,
shallow requests is exactly the regime where the herd measured 564-728 tok/s. That workload needs
no virtual-context layer at all; it needs the layer to get out of the way.

**Depth is expensive and roughly fixed.** At 16k per stream the card sustains ~110 tok/s in
aggregate regardless of stream count between 2 and 8, and KV precision does not move it — this is
a hybrid-attention model whose cache is already small, so attention at depth is compute-bound.
A design that keeps chunks shallow is working with the hardware; one that fills them is not.

## 2b. The mechanism

**The bar: virtual the way virtual memory is virtual.** 256k goes to `/v1/chat/completions` and a
normal completion comes back — one answer, one voice, streaming. No new parameters, no chunk
metadata, no caveat. If the caller has to know about chunks, it is not virtual.

### Why this is achievable rather than a compromise

A real 256k window is not valuable because every token attends to every other token. It is
valuable because **the tokens that matter for the question are all in one place**. Attention mass
concentrates on a small fraction of a long input; the rest is carried but barely consulted.

So the job is **selection, not summarisation**. If the content relevant to a question fits one
stream, and we put it there *verbatim*, the answer is not an approximation of a long-context
answer — the model is reading the same source text it would have read. Indistinguishable is the
correct expectation, not an aspiration.

This is also why summarising is the wrong primitive. A digest loses exactly the detail a question
might turn on, and no amount of care in reassembly puts it back.

### Two query classes, because they need different machinery

**Selective** — "what does the contract say about termination", "why did the deploy fail", almost
everything. The answer depends on a fraction of the input.

**Aggregate** — "count every occurrence", "summarise the whole thing", "list all parties". The
answer depends on all of it, but the operation is *associative*, which is what makes it tractable.

A cheap classification of the **query alone** (no document) picks the path. Getting it wrong is
recoverable: an aggregate query sent down the selective path returns a visibly partial answer,
which the verifier below catches.

### Path 1 — selective

1. **Index (parallel, 48 streams).** Each chunk emits spans with **exact source offsets** — claims,
   entities, section boundaries — not prose summaries. The output is a pointer table, not a
   digest.
2. **Select.** Rank spans against the query and take source text until one stream's admitted
   context is nearly full — on the 3090 profile, ~96k of 104k. Selection is over pointers, so what
   gets loaded is the original bytes.
3. **Answer (one stream).** Skeleton + selected verbatim source + query. Generated **once**, and
   this is what streams to the caller.

The final generation never sees a summary. That is the whole point.

### Path 2 — aggregate

1. **Map (parallel).** Each chunk performs the operation exactly — count, extract, enumerate — and
   returns structured data, not prose.
2. **Reduce.** Combine associatively. Counts sum; lists concatenate and dedupe.
3. **Answer (one stream).** The combined structure plus the query, generated once.

Exactness comes from the operation being structured. Prose summaries cannot be combined without
loss; counts can.

### The skeleton, and why every path carries it

Before either path, a cheap global outline is built from a downsampled read — headings, first and
last lines of each chunk. It states what the document is, its structure, and the entities running
through it.

It rides in every chunk's context and in the final answer. It is what lets a chunk resolve "the
plaintiff" and what lets the answer pass know the shape of the parts it did not load. It is
identical across chunks, which makes it exactly the shared prefix **KV prefix reuse (#7)** exists
to serve: 2k skeleton across 17 chunks is 34k of duplicated ingest without reuse, 2k with it.

### Verification, because a confident wrong answer is the failure that matters

After the answer pass, a cheap check: does the answer rest on spans that were actually loaded, or
does it reference something selection did not provide?

If it reaches for what it was not given, **re-select and run again** — bounded to one extra round.
If it still cannot be grounded, the request **fails** rather than returning a plausible fabrication.
Slow or refused are acceptable outcomes; subtly incoherent is not.

### What it costs

Reading 256k is unavoidable: at the measured ~3,300 tok/s prefill, ~78 seconds for a cold
document. A real long-context model pays comparably. Everything after is bounded by one generation.

**A document read twice should be indexed once.** That is the argument that would move the store
from ephemeral to persistent — not chunk state, but the index. It is the first thing that would
change the decision in §5.

## 2c. Spare stream capacity, and what it is for

The deployment allocates more context than a caller may send. 48 streams admitting 8,192 each is
**393,216** against a **256k** ceiling — **16 streams already allocated, already paid for, and
idle** unless given work. Anything they do concurrently with the chunk pass adds no wall-clock time,
because they ride the same forward pass.

**Spend it on the one thing chunking actually loses.** Selection, ordering and fidelity are all
solvable with pointers and budget. What splitting destroys is a fact in one chunk relating to a
fact in another: no stream saw both, so no index entry records the relationship and no retrieval
can find it. Spare streams are the ones that see both.

### Bridges

**Adjacent** — a window straddling a cut: the tail of chunk *N* and the head of *N+1*. Same depth
as a chunk, so no throughput penalty.

**Long-range** — a pairing of distant chunks that the index says share material. Their halves are
not contiguous, so the joined text carries an explicit marker; without it the model reads two
distant regions as continuous prose and reasons about a document that does not exist.

### Which boundaries get repaired

**The splitter already knows.** `cutAt` prefers a paragraph break, then a sentence end, then
whitespace, then a hard break — and that preference is recorded per chunk as `CutQuality`. A chunk
ending at a paragraph probably severed nothing; one ending mid-word certainly did.

So bridges go to the worst cuts first. It is a free signal and a far better ranking than position.

Long-range pairs follow, ordered by shared material, because a relationship between distant chunks
only deserves a stream when there is reason to suspect one. Every-pair-with-every-other is 496
pairings at 32 chunks; the index reduces it to a handful.

### The constraint that makes it free

Bridges cost nothing **only if they ride the same batch as the chunks**. One that spills into a
second wave costs a full forward pass, which is the opposite of the point. So the allocator fits
bridges inside the stream budget rather than adding to it, and drops the lowest-priority ones when
the input is large enough to leave little spare.

At the 256k ceiling: 32 chunks, 16 spare, 31 boundaries — half the cuts are repairable in one
wave. At 128k: 16 chunks, 32 spare, and every boundary is covered with room for long-range pairs.

### Naming the blind spots

When spare runs out, some severed boundaries go unrepaired. Those are reported.

This is the difference between *"the document does not relate those things"* and *"we did not have
a stream to look"*. A system that cannot name its blind spots cannot be trusted when it says a
document is silent on something — and for a virtual context claiming to be indistinguishable from
a real one, that distinction is the whole of its honesty.

## 2d. Synthesis without a text round-trip

The synthesis step in §2b re-prefills the selected text — ~96k at ~3,300 tok/s, **29 seconds**, and
the single most expensive avoidable operation in the pipeline.

**HLSR** replaces it: the parallel streams hand back latent vectors, and the answer stream is seeded
from those instead of re-reading them as tokens. ~38 s off a 124 s pipeline, 31%.

Full specification, including the three mechanisms it needs and the experiment that gates it:
**`HLSR.md`**.

## 2e. The aggregate path, built

Grammar-constrained digests merged by code — no second model pass, no parse that can fail.

**Built:** `internal/vcontext/merge.go`, plus grammar support through the sampler chain and the
`"grammar"` field on chat completions.

### Order independence is a correctness requirement

Streams finish in whatever order the scheduler produces. A merge sensitive to arrival order would
answer the same question differently on different runs, with nothing to indicate which answer was
which — a nondeterminism the caller cannot see, cannot reproduce, and cannot report.

So every rule is commutative and digests are sorted by chunk before combining, which also means any
ordering that survives into the output reflects the **document** rather than the race. Asserted by
shuffling the inputs forty times and requiring an identical result.

### Disagreement is surfaced, never resolved

Two sections of a document saying different things is a fact **about the document**, and often the
most important one. A merge that quietly picks one produces a confident answer that may be the
wrong half.

`RuleCollect` is the default for that reason: distinct values are gathered, a conflict is recorded
naming both sides and their chunks, and the rendered output says the sections disagree. Resolving
it is a decision for the answer pass or the caller, with the evidence in front of them.

### Rules

| Rule | For | Behaviour |
|---|---|---|
| `RuleCollect` | default | distinct values; conflict if chunks disagree |
| `RuleSum` | counts | exact addition — the associative case aggregation exists for |
| `RuleUnion` | lists | concatenate, dedupe, document order |
| `RuleAll` | per-chunk observations | keep everything, never conflict |

`RuleSum` refuses a non-numeric field rather than coercing: a grammar should have made that
impossible, so it indicates a stream that was not constrained.

### Provenance

Every field records which chunks contributed. An answer that cannot say where a fact came from
cannot be checked, and a conflict that cannot name its sides cannot be resolved by anything
downstream.

### Why an unconstrained response is an error

`ParseDigest` refuses anything that is not a bare object. A grammar makes the shape valid by
construction, so a parse failure means that stream was not constrained — and silently dropping it
would answer the question from a document with a hole in it, with no indication a section is
missing.

## 2f. Indexing: located deterministically, labelled by a model

**Never ask a model for byte offsets.** A language model cannot count characters, and an offset that
is plausibly wrong is the worst kind of wrong: text is assembled from the neighbouring region and
the answer comes back confident, coherent and about something else. Nothing downstream detects it.

So the two halves are separated by what each is actually good at:

- **Locating is arithmetic.** `Segment` divides a chunk at paragraph boundaries — sentences where a
  paragraph is too large to select usefully — and computes absolute offsets into the source. Exact
  by construction.
- **Labelling is judgement.** A model, or an entity tagger, says what a located span is about.
  It never says where it is.

Sentence splitting keeps the terminator with its sentence, and does not treat a decimal point as a
boundary — a span ending at `3.` reads as truncated to whatever consumes it.

### Relatedness, and the floor before the model

Long-range bridges need to know which distant chunks might discuss the same thing. `RareTermRelated`
scores shared terms weighted by inverse document frequency: two chunks sharing "Acme" and "8080"
have a concrete relationship, two sharing "system" and "the" do not.

Numbers count as significant terms. A port, an amount or a case number is exactly the shared detail
that makes two distant passages worth bridging.

This is the **floor an entity model must beat** (`MODELS.md` §0), built first for that reason. It
will not match "the plaintiff" to "Acme Corp" — that is where GLiNER earns its place, and the
comparison against this is what decides whether it does.

## 3. What chunking is not

**It is not a long context window.** Splitting destroys attention across chunk boundaries, which
is the thing a real 256k window provides. An embedded store plus reassembly is **RAG-shaped**,
with RAG's quality profile: good at retrieval and aggregation, weak where the answer depends on a
relationship between two facts that landed in different chunks.

That is a legitimate product with a legitimate boundary, and the boundary should be stated to
callers rather than implied away. A request that needs genuine cross-context reasoning is not
served by this layer, and the layer should be able to say so.

## 4. The largest unmeasured win

**KV prefix reuse.** When chunks share a prefix — a system prompt, a shared document header,
tool definitions — that prefix was prefilled once per chunk. At 48 chunks a 2k shared
prefix is ~96k tokens of duplicated ingest, against ~3,300 tok/s. Prefilling it once and sharing
the KV across streams removes that entirely.

Given prefill is the bound for exactly the workloads that chunk most (read-heavy map-reduce), this
is likely worth more than any further parallelism. It is built (`internal/engine/prefixshare.go`,
`llama.PrefixCache`, `PREFIX-REUSE.md`) and not yet measured on a card; the saving above is
arithmetic from the measured prefill rate.

## 5. Open questions, in the order they should be answered

1. ~~**Does prefix reuse work on this engine?**~~ Answered: the unified pool permits it and the
   binding gates on it (`Context.SeqCp`, `ErrSeqCpNeedsUnified`). What remains is measuring it.
2. ~~**What is the reassembly contract?**~~ Answered and built: rule-driven merge in
   `internal/vcontext/merge.go`, §3 above.
3. **How does the layer choose chunk count?** From input size alone it will choose wrongly — the
   table in §2 says it must also know whether the job is read-heavy or generate-heavy. That may
   have to be declared by the caller.
4. **Where does the store live?** Answered provisionally: **in process, behind an interface**.

   The state is per-request, at most a few dozen entries, alive for seconds. Nothing outlives the
   request, nothing crosses processes, and a linear scan over 48 entries beats any index. A
   database here would be a dependency bought for nothing — and for an embedded engine it is a
   second native toolchain (SurrealDB'"'"'s Go embedding is FFI to its Rust core) paid by everyone
   who builds the project, on top of the cgo it already carries for llama.cpp.

   `Store` is an interface so that stays a decision rather than a commitment. A real backend earns
   its place the moment either of these becomes true, and not before:

   - cross-chunk state must **outlive the request** — a document session a caller returns to
   - it must be **shared across replicas**

   Both are product decisions, not technical ones.

   Cleanup has two paths deliberately. `Close` runs when a request ends and is the normal one;
   a TTL expires anything that never got there. The normal path is the one that gets skipped —
   cancelled requests, panics, callers that disappear mid-stream — so state that only cleans up on
   the happy path is a leak with extra steps.

## 5b. Two failures the implementation has to refuse, found by building it

Both produce a **confident, coherent, wrong answer**, which is the worst outcome this design can
have — worse than being slow and worse than refusing.

**Answering with no source at all.** If no indexed span fits the answer budget, an empty selection
plus a skeleton is still a valid-looking context. The model answers from its own knowledge and the
result reads as grounded. Refused: a budget that fits nothing is an error.

**Swapping the evidence for filler.** If the best-ranked passage does not fit but a smaller,
less relevant one does, a naive budget loop takes the smaller one — and answers a question about
termination from a passage about shipping. Caught by a test where exactly that happened. Filling
leftover budget with lower-ranked spans is fine; substituting for the primary evidence is not, so
the top-ranked span not fitting is now fatal.

The general rule both come from: **the answer context must contain the evidence the answer rests
on, or there must be no answer.**

## 6. Build plan

All five stages below are built (`plan.go`, `split.go`, `dispatch.go`, `merge.go`, `store.go`); the
plan is kept as the record of why they were ordered this way. Each stage is useful alone and
testable without a GPU where possible.

**Stage 1 — the planner** (`internal/vcontext`). Given a request, decide whether to chunk and into
how many pieces. Pure logic, no engine, no network: it takes a token-count function and a policy
and returns a plan. This is where the §2 measurements are encoded, and it is the piece most likely
to be got wrong by intuition, so it is built first and tested hardest.

**Stage 2 — splitting.** Turn a decision into actual chunks, cutting at boundaries that do not
destroy meaning (paragraph, then sentence, then hard cut). Still pure logic.

**Stage 3 — dispatch.** Run the chunks concurrently against the engine's chat-completions API and
collect per-chunk results. The first stage that needs a running engine.

**Stage 4 — reassembly.** Combine per-chunk outputs per the contract chosen in #9, and carry the
refusal path for requests that need cross-chunk reasoning.

**Stage 5 — cross-chunk store.** Only once 3 and 4 exist and the contract says what must persist.

Stages 1 and 2 are worth building even if the rest is deferred: they answer "how would this be
split, and why" for any input, which is the question the design keeps turning on.

## 6a. Structured input

A JSON payload needs no heuristic chunking — element boundaries are explicit, and partitioning at
them is the structured equivalent of an AST cut. `FlattenJSONArray` streams via `encoding/json`'s
Decoder rather than unmarshalling into maps, so a 256k payload does not allocate an interface box
per value.

⛔ **Flattening JSON syntax to save prefill tokens does not work — measured 0.0%.** The apparent
saving is the pretty-printer's whitespace, and `json.Compact` takes all of it losslessly. Only key
abbreviation saves tokens (21.0%), and it costs a schema header on every stream.
`results/json-canonicalisation.md` has the numbers.

Two failure modes the flat form has and JSON does not, both handled: a delimiter appearing in a
value silently splits a record into the wrong number of fields, and flattening a nested value drops
the nesting with nothing to distinguish it from a string containing delimiters. The first is
escaped, the second refused.

## 6b. Ordering within a request

Chunking assumes chunks are independent. When a request contains dependent steps, that assumption
is wrong and the failure is invisible — the answer is well-formed and in the wrong order.
`DAG-SCHEDULING.md` covers the graph extraction and the scheduler that enforces ordering in code
rather than asking the model for it. It composes with this layer: `Dispatch` still fans out a single
task that is itself wide.

## 7. Scope boundary

This layer is **not** part of the engine and should not be built into it. The engine's contract is
a chat-completions API over streams; the layer is a client of it. Keeping the split means the
engine stays useful standalone and the layer can be replaced without touching inference.

See `ROADMAP.md` §0 for why that boundary is deliberate.
