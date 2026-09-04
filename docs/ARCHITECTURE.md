# Architecture

The other documents are organised by campaign — what was built, when, and what it measured. This one
is organised by **concept**: the shape of the system, the facts that constrain every decision in it,
and the principles that turned out to apply in more than one place.

Read this first if you are new. Read the campaign docs when you need detail on one layer.

---

## 1. The shape, in one paragraph

llama-herd runs many concurrent streams against **one resident copy of a model's weights** on one
consumer GPU. A single goroutine owns the llama.cpp context and runs a decode loop: each tick it
gathers the next token from every active stream into one `llama_batch`, calls `llama_decode` once,
and routes the sampled tokens back. Everything else — chunking a 256k input, ordering dependent
work, extracting symbol graphs, judging a codebase — is a **layer above that boundary** that turns a
large problem into many stream-sized ones and reassembles the answer in Go.

The engine makes streams cheap. The layers above decide what to spend them on.

---

## 2. Layers, and where a new feature goes

```
  request (any size)
        │
  ┌─────▼──────────────────────────────────────────────────────────┐
  │ codegraph   symbol graphs, contracts, drift, cross-chunk judging│  code-specific
  ├────────────────────────────────────────────────────────────────┤
  │ vcontext    canonicalise · chunk · index · select · dispatch ·  │  general
  │             merge · DAG scheduling                              │
  ├────────────────────────────────────────────────────────────────┤
  │ prefill     fan-out orchestration, single-owner engine boundary │  ◄── the boundary
  ├────────────────────────────────────────────────────────────────┤
  │ engine      slot table, continuous batching, admission control  │
  │ llama       cgo binding; ONE goroutine may enter a context      │
  └────────────────────────────────────────────────────────────────┘
```

**Where does new work go?**

- Does it change how tokens are decoded, batched, or sampled? → `engine` / `llama`.
- Does it decide *what* to send to a stream, or *how to put answers back together*? → `vcontext`.
- Is it true only of source code — symbols, imports, signatures? → `codegraph`.
- Is it about how many things run at once and what happens when one fails? → `prefill`.

The pressure is always to push work **up**. The engine is 97.61% library forward pass
(§3.5), so there is almost nothing to win down there and a great deal to lose.

---

## 3. Five measured facts that constrain everything above the engine

Every number here is reproducible; see `BENCHMARKING.md` and `results/`. **Derive, do not quote** —
if you are about to make a decision on one of these, re-measure it on your hardware.

### 3.1 Depth costs. Residency does not.

The stream count that wins on an empty cache is the one that loses on a full one.

| | depth 0 | depth 16k |
|---|---|---|
| 24 streams | 514.41 tok/s | 51.62 tok/s |
| 8 streams | 291.74 tok/s | **112.09 tok/s — beats 24 by >2×** |

(One node, one boot; 48 streams cannot hold 16k each in a 425,984-token pool, so the deep column
tops out at 24.)

What costs is **how deep a single sequence is**, not how many tokens the herd holds in total. This
is the counter-intuitive core of the system: subdividing the same content across more streams can be
*faster*, because each stream stays shallow.

**Consequence:** there is no single best stream count. Stream count follows the depth being served.

The generalisation: per-pass cost is proportional to **Σ(depth)** across active streams, and
aggregate throughput is `active streams / pass time`. So a deep stream should be counted as several
streams — a 128k stream carries about sixteen times an 8k stream's attention work — and, awkwardly,
**a deep stream does not pay for its own depth; the herd does.** It receives one token per pass like
everyone else while making every pass more expensive for every other stream.

⚠️ Whether throughput scales with stream count at *fixed* Σ depth is unmeasured, and it decides
whether mixed workload classes are worth building (#43). `DEPLOYMENT.md` §3-4.

### 3.2 Prefill does not chunk. Decode does.

Splitting one input across more streams **loses ~19% of ingest throughput**. Splitting generation
across more streams **gains 2.2×**, peaking near 32 chunks.

**Consequence:** chunk count follows whether the job is read-heavy or generate-heavy — never how
large the input is. This is why `vcontext.Shape` exists and why it is a required input, not a hint.

### 3.3 One unified KV pool is what makes the herd a herd — twice over

A fixed per-stream share leaves idle slots holding capacity nobody can use, and the library ends up
running a forward pass per sequence — concurrency then *costs* passes instead of sharing them.

With one pool, 48 streams retire **728.71 tok/s** where the same library retires **138.40** on a
single sequence on that node: **5.27×** from weight and pass sharing. On a slower node the same
configuration gave 564.16 against 119.31, 4.73×. At 24 streams the measured amortisation is 4.23×,
still rising.

A unified pool is *also* the only configuration in which a shared prompt prefix can be reused across
sequences — upstream shares the cells as metadata when both sequences sit in one stream, and asserts
rather than errors on a partial copy when they do not. Two independent reasons, one setting.

### 3.4 The stream budget is the scarce resource: 8,874 tokens

At 48 streams over a 425,984-token unified pool, each stream holds 8,874 tokens. Everything competes
for that: the content, a skeleton of the whole document, a global symbol header, a contract from an
upstream tier, tool definitions, an overlap window.

On the node that measured 564.16 at 48, the curve plateaus 48–64 (564.16 / 582.04) and **collapses
at 72** (11.50). On an earlier, slower node the plateau ran 56–72 and 80 failed to run at all. The
cliff moves with the node; the plateau does not. More streams means less context each, and past the
plateau you are paying capacity for throughput you do not get.

### 3.5 The engine is not the bottleneck — 2.39%

Total engine overhead is **2.39%** of engine time; the remaining 97.61% is the library's forward
pass. Optimisation effort belongs above the boundary, in what gets sent and how little of it there
is.

---

## 4. Principles that showed up in more than one place

These are not style preferences. Each was arrived at independently in at least two layers, which is
why they are recorded here rather than in one campaign doc.

### 4.1 ×48 — anything on every stream is multiplied

A 1.5k-token global symbol header across 48 streams is 72,000 tokens: ~22 s at measured prefill, and
28% of a 256k payload spent re-reading identical bytes. A 300-token overlap window is another 14k.

Every proposal of the form *"just prepend X to each stream"* has to be costed at 48×, against a
budget of 8,874 each. `SkeletonCap = 120` exists because an earlier version of exactly this idea ate
300 of 900 available tokens.

**Corollary:** shared prefixes are the highest-leverage optimisation in the system, because they are
the one case where the ×48 can be paid once — and with a unified cache the sharing is metadata-only,
so it is a memory saving as well. See `PREFIX-REUSE.md` and #37.

### 4.1b Character count is not token count

Reasoning about token savings from character savings is wrong in the direction that flatters the
change. Flattening JSON cuts 17.8% of characters and **0.0% of tokens**; abbreviating keys cuts
58.7% of characters and 21.0% of tokens. A BPE vocabulary already encodes repeated structure
cheaply, so the bloat that is obvious to a reader is frequently invisible to the tokenizer.

Every transformation proposed to save prefill has to be counted with a tokenizer before it is
believed. `MeasureCanon` and `MeasureJSONCanon` exist for exactly this, and both have already
overturned a plausible estimate — see `results/json-canonicalisation.md`.

### 4.1c A label that can be trusted is worth more than a better guess

Structural cutting does not beat paragraph cutting on *position* for Go — blank lines already
separate declarations, measured 4/5 against 4/5. What it adds is a boundary the splitter can label
truthfully, and that label is what lets a downstream pass skip work: a cut known to be clean needs no
overlap window, where a cut that merely got lucky needs one everywhere.

The pattern generalises. Much of what this system spends tokens on is insurance against not knowing
— overlap windows, skeletons, re-sent context. Cheap certainty upstream removes the need for
expensive hedging downstream, and is usually the better trade.

### 4.2 Select, then load — never load and hope attention sorts it out

The same discipline, arrived at three times:

- **spans** — retrieve the relevant passages, do not send the document;
- **contracts** — send declarations without bodies, not the generated file;
- **tools** — classify to one tool's schema, do not put 176 definitions in context.

In each case the naive version fits *technically* and wastes the budget in §3.4.

### 4.3 Let the model find; let Go compose

The model does local, fuzzy work: judge this chunk, extract these symbols, summarise this passage.
Code does global, exact work: topological ordering, cross-referencing assertions, merging results,
computing coverage.

Asking the model to do the second kind is where silent failure comes from — it produces a
well-formed answer with no error to catch. Asking code to do the first kind is impossible. Every
layer in §2 is an instance of this split.

### 4.4 Exact bytes travel as text; latent vectors are for soft context

Forwarding a hidden state is the intended end state for aggregate context and is blocked on an
unresolved projection (#21) that #19 measured as lossy.

But there is a permanent rule underneath the temporary blocker: **anything whose value depends on
precise bytes cannot be forwarded as a latent.** A function signature reconstructed from a projected
state is drift by construction. Latents are for *"what was this passage about"*, never for
*"what exactly did it declare"*.

### 4.5 Barriers convert an ordering constraint into a straggler problem

Executing tiers as barriers makes every task wait for the slowest member of the previous tier,
including tasks it has no relationship with. Dispatching on dependency-satisfaction gives the same
ordering guarantee with none of that cost. `StragglerRatio` exists to make the effect visible.

### 4.6 Coverage travels with every result

A result must carry what it examined. "No conflicts found" and "nothing was checked" print
identically unless coverage is attached — so `Verdict` carries `Coverage`, `Batch` carries `Failed`,
`Run` carries `Blocked` and `Cancelled`, and assembled output marks its gaps.

Naming is part of the enforcement: there is no `Correct()` method, because the check cannot answer
that question.

### 4.7 Determinism is a measurement prerequisite

Unstable ordering makes two runs of the same request assign work to different streams, and any A/B
comparing them measures noise. SCC output is sorted, contracts render in stable order, chunk ids are
dense and explicit.

This is also why the control arm matters: a baseline in a git worktree and a HEAD in the real
checkout differ in ways that have nothing to do with the code.

### 4.8 Fail-fast is a mode, not a default

A failed tier-0 unit poisons everything downstream — stop. One bad chunk out of 48 in a judging run
is a gap to report, not a reason to discard 47 good results. The same orchestrator has to do both,
so the policy is a parameter (`prefill.FailFast` / `prefill.Continue`).

---

## 5. The dominant failure class: well-formed but wrong

Almost every expensive defect in this system produced **output that looked correct**. Nothing threw,
nothing logged, and the failure was visible only to something downstream that nobody had run yet.

| layer | what it looks like |
|---|---|
| task ordering | a reordered plan that reads as a coherent plan |
| code generation | `GetUser(id string)` declared, `GetUser(id int)` implemented — often still parses |
| chunk assembly | an answer built from the chunks that happened to succeed |
| judging | "CORRECT" over regions that returned no assertions |
| symbol resolution | an ambiguous symbol silently bound to the wrong provider |
| measurement | a stale container answering four A/B runs identically |

The defences are structural, not vigilant:

1. **Constrain the shape** — grammars, so malformed output cannot reach the aggregator.
2. **Verify the output, do not trust the prompt** — nothing makes a model honour text it was given;
   parsing what it produced costs microseconds against seconds of GPU time.
3. **Distinguish absence from emptiness** — "examined, found none" and "never examined" must not be
   the same value. A rule that asserts absence has to first prove it looked.
4. **Name what you did not do** — §4.6.
5. **Negative-control it** — run the case that must fail and watch it fail. A check that cannot fail
   is worse than no check, because it reads as evidence.

---

## 6. Method notes

**Stale is a prompt to go looking, not a verdict.** A dependency that has not moved in two years may
be fine, superseded, or forked into something better — the staleness only tells you to check. When
this came up for tree-sitter bindings, the survey found four live options with a 180× size spread,
and the best fit was not the most popular one.

**Popularity measures accumulated age, not current fitness.** In that same survey the 54.7 MB
two-year-stale option had roughly double the stars of the 0.3 MB actively-maintained official one,
purely because it existed first.

**A number without its method is a claim.** State what was measured, on what hardware, and how many
repeats. Medians, not bests.

**Do not cite a number you have not re-read.** A `3.3×` for the unified KV pool was written into
`CODE-GRAPH.md` during the session that produced this document; the only 3.3× anywhere in the
measurements is the cost of an MTP speculative pass, an unrelated figure. It was caught by grepping for the
figure before promoting it here.

**When the arithmetic says a running system is impossible, the arithmetic is wrong.** Go read the
running configuration before arguing with it.

---

## 7. Where to read next

| Document | Contents |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Targets, the reasoning behind them, and the constraints in the way |
| [INVARIANTS.md](INVARIANTS.md) | Things that are true and were expensive to learn — mostly silent failures |
| [VIRTUAL-CONTEXT.md](VIRTUAL-CONTEXT.md) | The layer that accepts an input of any size |
| [DAG-SCHEDULING.md](DAG-SCHEDULING.md) | Ordering within a request, for general task graphs |
| [CODE-GRAPH.md](CODE-GRAPH.md) | Code generation, code judging, and the fan-out boundary |
| [HLSR.md](HLSR.md) | Hierarchical latent speculative reduction |
| [MODELS.md](MODELS.md) | Auxiliary models, and the rule that decides whether one earns its place |
| [BENCHMARKING.md](BENCHMARKING.md) | What each measurement means and how to reproduce it |
| [results/](results/) | The measurements themselves |
| [DEPLOYMENT.md](DEPLOYMENT.md) | What this ships as, the capacity model, and how a configuration is chosen for a machine |
| [PREFIX-REUSE.md](PREFIX-REUSE.md) | Computing a shared prompt prefix once and sharing its KV cells |
| [results/json-canonicalisation.md](results/json-canonicalisation.md) | Why flattening JSON saves no tokens, measured |
