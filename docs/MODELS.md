# Auxiliary models

Which models do which jobs, why those, and the rule that decides whether a model earns its place
at all.

The engine serves one large model. These are the small ones around it — some internal to virtual
context, some exposed as endpoints in their own right.

---

## 0. The rule, and where it does not apply

**A NEW model must beat a cheap floor before it earns a dependency.** Every integration costs build
time, latency, memory, a failure mode, and something that can be misconfigured. A model that ties
with word-matching has bought none of that back.

`LexicalRetriever` in `internal/vcontext` exists for that reason — a baseline a semantic retriever
has to outscore on this workload, not a placeholder to be swapped out on faith.

**But two of the models below are not new dependencies.** The embedding model and GLiNER are
already relied on heavily in the wider platform, with the artifacts, quantization and serving paths
already characterised in production. The integration cost is paid. For those, the question is not
"does it earn a dependency" but the narrower "is this the right job for it" — and for the embedding
model, §1 settles it before any measurement.

The floor still has a use for them: it says whether they help *this* workload. It is no longer a
gate on adopting them.

## 1. Two purposes, and the second is the larger one

**Internal.** Virtual context needs retrieval, entity extraction and classification to do its job.

**Exposed.** The same models are **endpoints an agent can call**. An agent running against a local
llama-herd does entity extraction and embedding *locally* rather than round-tripping to a
datacenter — which is the point of running inference on a card in the first place.

That second purpose changes what "best model" means. Benchmark position matters less than **being
the same model the rest of the platform uses**, because an agent that embeds locally and a backend
that indexed remotely must land in the same vector space or the two cannot be queried together.

## 2. The table

| Job | Model | New? | Floor / decision |
|---|---|---|---|
| Chunk hidden states (HLSR) | **the 35B itself** | no | no alternative exists |
| Span retrieval (#14) | **Qwen3-Embedding-0.6B → cross-encoder rerank** | in production | space consistency decides it (§1) |
| Bridge scoring (#18) | **GLiNER** | in production | compare against rare-term intersection |
| Skeleton entities | **GLiNER** | in production | adopt — see §3 |
| Query classify + shape (#8) | **ModernBERT int8** | in production | `max_tokens` as a proxy |
| Groundedness (#16) | **small NLI cross-encoder** | **yes** | must beat the 35B grading itself |

Only the NLI reranker is a genuinely new dependency, and it is the one the floor rule is really
aimed at.

## 3. Why each

### Chunk hidden states — the 35B, necessarily

HLSR injects vectors into the 35B's own input-embedding space. They must come from that model. A
1024-d vector from an embedding model is a *different model's* space — injecting it is meaningless
rather than merely worse. Settled by construction, not by preference.

### Span retrieval — two stages, not one

A bi-encoder over ~500 spans is cheap and recall-oriented. A **cross-encoder reranker** over the
surviving ~50 is materially more precise, because it sees query and passage together rather than
comparing two independently-computed vectors.

Precision is what matters here: surviving spans go **verbatim** into the answer context, so a wrong
one costs budget *and* misleads the generation. Fifty pairs is affordable against a 78-second
prefill.

**Qwen3-Embedding-0.6B** for the bi-encoder, and the choice is not a benchmark question. The
platform already indexes with it at 1024-d native, so spans embedded here land in the **same vector
space** as everything indexed elsewhere. A local agent's embeddings and a remote backend's must
match or the two indexes cannot be queried together — and a model that scored higher in isolation
while landing in a different space would be worse, not better.

ONNX int8 on CPU, which matters operationally: it runs **concurrently with the GPU-bound prefill**,
so retrieval is free in wall-clock.

### Bridge scoring — an entity question, not a topic question

Bridging asks "does the same thing appear in two places". Two chunks sharing "Acme Corp" and "8080"
have a concrete relationship; two chunks with high cosine similarity may merely be about similar
subjects. A contract's termination and payment clauses have low semantic similarity and the same
parties — exactly the relationship a bridge repairs.

GLiNER is already the platform's tool for this class of extraction, so it is the natural fit rather
than a new bet. The comparison against **rare-term intersection** is still worth running, not to
decide whether to adopt GLiNER — that is settled — but to learn where surface matching is already
sufficient and the model can be skipped for cost.

### Skeleton entities — GLiNER, and the reason is negative

A typed entity list, deterministically, with no generation pass. The property that matters is
negative: it **cannot invent an entity that is not in the document**, where an LLM asked for the
same list can. For a system claiming to be indistinguishable from a real context window, a skeleton
containing a hallucinated party is worse than no skeleton.

### Query classification — ModernBERT int8

Runs on the query alone, no document, 1–2 ms. Already in production elsewhere, so the artifact,
quantization and serving path are known quantities. It decides the selective/aggregate route and
can carry the read-heavy/generate-heavy shape (#8) on the same pass.

### Groundedness — an NLI cross-encoder, not the 35B

"Does this claim follow from these spans" is natural-language inference, and a small cross-encoder
does it better and far cheaper than a generative pass.

It also removes a structural problem: **the model that wrote the answer should not be the one
certifying it.** Self-assessment is exactly where a fluent, ungrounded answer gets waved through,
and that is the failure this pipeline exists to prevent.

## 4. Exposure

Serving these as endpoints is what makes them worth their weight twice. Present surface:

```
GET  /health
GET  /v1/info
GET  /metrics
GET  /v1/models
POST /v1/chat/completions
```

An agent using the engine as its local compute wants at least:

- `POST /v1/embeddings` — OpenAI-compatible, backed by the bi-encoder
- `POST /v1/rerank` — Jina-compatible, backed by the cross-encoder
- an entity endpoint for GLiNER, zero-shot with caller-supplied labels

**The work is offloaded, not duplicated.** An agent that embeds and extracts entities against a
local card is work the datacenter never sees — which is the argument for the whole arrangement, and
it only holds if the local vectors match the remote ones.

## 5. Sizing that has not been measured

⚠️ **GLiNER over 256k.** It works in windows of a few hundred tokens, so a full document is several
hundred passes. If that fits inside the ~78-second GPU prefill window it is free; if not it is on
the critical path. Cheap to measure, and unmeasured.

⚠️ **Zero-shot needs a type set.** GLiNER requires the labels to look for. Fine for a known domain,
an open question for arbitrary input — either a generic set, or a cheap first pass to infer which
types matter.

⚠️ **CPU contention.** These all run on the host CPU while the GPU prefills. That is what makes them
free, and it stops being true if several run at once on a machine with 8 cores. Their combined
budget needs measuring together, not one at a time.
