# Code generation and code judging on a 48-stream herd

Built: `internal/codegraph/` (tested, race-clean) and `internal/dag/` (no tests of its own).
Tracked as #34. Companion to
`DAG-SCHEDULING.md`, which covers ordering for general task graphs; this covers the two code cases,
which are stricter.

## 1. Why code needs its own graph

A summary generated out of order reads oddly. An implementation generated before its interface has
imports that do not resolve, signatures that do not match, and types that were never declared —
and none of that is visible in the generated text. It is visible only to a compiler nobody has run.

So the ordering comes from symbols, not from the model's sense of what should come first.

---

# Part A — Generation

## A1. The graph is built from symbols, not from tiers

The natural description is three tiers: types, then implementations, then tests. That is a useful
*prior* and a bad *plan*, because it is a statement about what usually depends on what rather than
about this request. `Unit.Provides` and `Unit.Requires` are resolved against each other to produce
real edges; `Kind` only breaks ties where the request stated no dependency — a test that declared no
requirements would otherwise land in level 0 beside the types it exists to exercise.

The prior can push a unit later. It never pulls one earlier: a real edge always outranks a heuristic
about what tests usually need.

## A2. Three resolution outcomes, and only one is an error

This is where a code graph diverges hardest from a task graph.

| requirement | task graph | code graph |
|---|---|---|
| provided by another unit | edge | edge |
| provided by nothing | **error** — broken extraction | **external** — it is `context`, `fmt`, `std::vector` |
| provided by two units | — | **warning** — genuinely ambiguous |

Treating an unresolved requirement as a missing node would reject nearly every real request, since
most requirements are the standard library. Treating an ambiguous one as resolved is worse: the
graph picks a provider, the dependent imports the wrong unit, and the code looks correct.

## A3. A dependency cycle in code is not an error

Kahn's algorithm detects a cycle. For a task list that ends the request — no valid ordering exists.
For code it does not, because circular references are ordinary: mutually recursive types, two files
in one Go package referring to each other, a C++ header pair.

Those symbols have no valid *per-file* order and always have a valid *joint* one. So cycles are
consolidated rather than rejected: Tarjan's algorithm finds strongly connected components, each
component collapses to one node, and the condensation of any directed graph is acyclic by
construction — the topological sort always succeeds.

A consolidated component generates in **one pass on one stream**, which is reported, because it
changes the dispatch shape and its combined output has to fit a single context.

The SCC pass is iterative rather than recursive. Dependency depth is derived from user text, and a
deep chain should not be a stack limit.

## A4. Context between tiers is exact text. Not latent vectors.

The proposal to forward `h_last` from the type stream into the implementation stream is the one
piece of the design that does not survive, and the argument against it is already in the design's
own mitigations section: *"inject the exact JSON or header AST into Tier 1 streams to guarantee
signature matching."*

Those are two different mechanisms and only one of them can be in place. Signature drift —
`GetUser(id string)` declared, `GetUser(id int)` implemented — turns on the precise bytes of a
declaration. Asking the implementation stream to reconstruct those bytes from a projected hidden
state reintroduces exactly the drift the tiering exists to remove, in a form no reader can see. The
mitigation is right and it forecloses the latent path for this edge.

Independently: `W_proj` is unresolved (#21) and #19 measured injection losing detail the state
provably holds.

What is forwarded is `SignatureSet.Contract()` — the declarations, rendered from the parsed AST
**without bodies**. Not the generated file: at 8,874 tokens per stream, an implementation given its
dependency's function bodies spends the scarce budget on code it must not reimplement.

## A5. Injection reduces drift; only verification catches it

Nothing constrains a model to honour text in its prompt, so the contract is checked after the fact.
`CheckDrift` compares what a tier promised against what the next tier produced, reporting three
shapes: `missing` (declared, never implemented), `changed` (the dangerous one — it usually still
parses), and `redeclared`.

Parsing a generated file costs microseconds against the seconds of GPU time that produced it. There
is no reason not to check.

**Go is implemented natively** (`go/parser`, standard library, no dependency), which settles the
`DeclReader` interface against real code. Partial parses return what they got: a stream that hit its
token limit mid-function still produced a usable contract in the declarations above the break, and
discarding it would turn a recoverable generation into a failed one.

---

# Part B — Judging a 256k codebase

## B1. Cut on declarations, and the overlap becomes conditional

**Built** (#36): `CutDeclaration` in the quality ladder, `SplitAt` taking parser-supplied boundary
offsets, `codegraph.GoBoundaries` producing them from `go/parser`, and `AddOverlap` spending a window
only where a cut was actually ragged.

Fixed-offset cutting slices a function in half and blinds the local pass. Cutting at top-level
declaration boundaries is the fix, and it slots into the existing ladder
(`CutEnd > CutDeclaration > Paragraph > Sentence > Word > Hard`) rather than being a new mechanism.

Boundaries are **preferences, not constraints**. A real file has long stretches with no boundary at
all — a 900-line function has exactly two — so treating them as constraints would produce chunks
that do not fit a stream. A boundary is used when one falls inside the existing backup window;
otherwise the prose ladder is the fallback, and supplying no boundaries is byte-identical to prose
splitting.

### ⚠️ What structural cutting actually buys, measured

It does **not** reliably improve cut *positions* for Go. Idiomatic Go separates declarations with
blank lines, so paragraph cutting already lands on the same offsets — measured **4 of 5 interior
cuts for both strategies** on this project's own `graph.go`, and 5 of 5 for both on synthetic
functions carrying internal blank lines. Crediting structural cutting with better positions here
would be crediting it with a win it does not deliver.

What it buys is **knowing**. A prose cut that happens to land on a declaration cannot tell that from
luck, so every cut must be treated as ragged. A structural cut is labelled `CutDeclaration`
truthfully, and a cut known to be clean needs no repair.

That is what makes the overlap conditional, and the overlap is where the saving is — **against the
unconditional window, not against prose cutting**. Measured on 6 chunks of Go source with a
300-token window: **conditional 0 tokens, unconditional 1,500**. At 48 chunks that is the whole
~14,100-token tax, spent entirely on cuts that severed nothing.

So the two halves of this are one thing: the boundaries exist to make the conditional overlap
possible, and the conditional overlap is the payoff.

A chunk borrows based on how **its own leading edge** was cut, which is the trailing cut of the
chunk before it. Reading its own `Cut` would repair the far edge from the damage.

## B2. The global symbol header makes prefix reuse load-bearing

Prepending an identical symbol table to all 48 streams is the right way to stop a chunk falsely
reporting "undefined type" for something defined three chunks away.

It is also the same bytes at position 0 of 48 streams, which is precisely the KV-prefix-reuse case
already flagged as the largest untested win. Against measured prefill (~3,300 tok/s):

| | tokens | time |
|---|---|---|
| content | 256,000 | 78 s |
| header × 48 (@1.5k) | 72,000 | +22 s |
| overlap × 48 (@0.3k) | 14,400 | +4 s |
| **total** | **342,400** | **~104 s** |

So the ~35–40 s estimate is out by about 2.6×, and **re-reading identical bytes is 21% of the work**
(25% with the overlap window), or 28% of the content itself. Without prefix reuse the header is the second-largest cost in the pipeline; with it the
header is nearly free and the design is straightforwardly right. That promotes prefix reuse from an
optimisation to a prerequisite (#37).

Related and already measured: prefill does not chunk — splitting it costs ~19%. The header is
working against that grain, which is another reason its size is worth minimising. `SkeletonCap`
exists because an earlier version of this same idea ate 300 of 900 available tokens.

## B3. Local judgment, deterministic aggregation

Each stream answers two narrow questions about its own slice — internal faults, and the contracts it
fulfils or assumes — and emits assertions under a grammar so aggregation never parses prose.

`CrossCheck` then does in a map lookup what no stream could do at all: chunk 3 calling `DB.Query`
with three arguments against chunk 18 defining it with two. Four contradiction shapes: `arity`,
`signature`, `undefined`, `redefined`.

Two things it deliberately does not treat as conflicts: reformatting (a model rewraps freely) and an
absent arity (0 means "not reported", not "takes no arguments" — treating an unreported field as a
claim manufactures a conflict on every partial assertion). Identical redefinitions are also fine,
because an overlap window legitimately reports the same declaration twice.

## B4. ⛔ "Zero conflicts" is not a correctness verdict

The design's step 4 says: *if all 48 local assertions cross-reference perfectly with zero graph
errors, the 256k codebase is judged CORRECT.*

It cannot be. The cross-check contradicts claims that were **made**. It is blind to:

- a logic error wholly inside one chunk that the local pass missed;
- a wrong-but-consistent contract that both sides agree on;
- any symbol no chunk asserted about — nothing was checked against it;
- any chunk that returned nothing, whose content was never judged at all.

A rule that asserts absence has to first prove it looked. So there is no `Correct()` method and no
boolean by that name; the method is `NoConflictsFound()`, which reads literally. `Coverage` travels
with every verdict and `Summary()` always renders both, so the two cannot be quoted apart:

```
no cross-chunk conflicts found — checked 412 symbols (180 definitions, 232 uses)
across 46/48 chunks; 2 chunk(s) returned nothing and were NOT judged: [17 31];
9 declared symbol(s) unmentioned by any chunk (Backoff, Codec, …). This is a partial result.
```

Without those numbers, "no conflicts" and "nothing was checked" print identically.

## B5. Explaining a mismatch

The isolate-and-explain pass is right. Send the involved chunks' **text**, not their `h_last` —
same reason as A4, and here the text is already in hand and cheap.

---

## C. Tooling: tree-sitter, and where it belongs

Tree-sitter is the right multi-language parser and the recommendation to use it stands, with two
corrections.

**Which bindings.** `smacker/go-tree-sitter` is **54.7 MB** (it vendors every grammar) and was last
pushed **August 2024**. The official `tree-sitter/go-tree-sitter` is **0.3 MB**, pushed **November
2025**, with grammars as separate per-language modules — so the cost is paid per language actually
supported. `go.mod` currently has zero dependencies, so this is the first one and worth choosing
deliberately.

**Where it runs.** The proposed pipeline puts an AST parse first, over the incoming request. For
**judging**, that is correct — the request contains a real codebase.

For **generation** it cannot: at that point the code does not exist. The request is prose — *"build
a user service with a repository interface"* — and there is no syntax tree to build. Tree-sitter's
place in the generation path is one step later, at the tier boundary, parsing code that now exists
in order to produce the contract the next tier receives (A4) and to verify what it produced (A5).
That is a stronger use than the proposed one: the contract is then derived from what was *actually
generated* rather than from what was *asked for*.

A refactor request is the mixed case — it contains existing source, so the input parse applies to
the fenced code and the prose path applies to the rest.

**Queries, not tree walks.** The design's own note about CGO overhead argues against its example
code, which recurses over every node and calls `Content()` per match. S-expression queries execute
inside C and cross the boundary once per match. Parse per file, query per file.

---

## D. Open

- Wire generation planning to the scheduler and the herd (#34).
- Tree-sitter behind `DeclReader`, official bindings, one language at a time (#35).
- ~~AST-aware cut quality + conditional overlap in the splitter (#36).~~ — built, §B above.
- ~~KV prefix reuse for the shared header (#37).~~ — built (`internal/engine/prefixshare.go`,
  `PREFIX-REUSE.md`); not yet measured on a card, and a prerequisite rather than an optimisation.
- Measure a real judging run: wall clock against the ~104 s projection, and how often local passes
  emit assertions at all (#38).

---

# Part E — Orchestrating the fan-out

Built: `internal/prefill/`, 12 tests, race-clean under `-count=2`. Tracked as #39.

## E1. ⛔ The parallelism is inside the batch, not across goroutines

The proposed worker pool launches 48 goroutines, each calling `ExecutePrefill` into the engine. That
is the one part of the design that cannot ship, and the reason is written in the code it would call:

> *libllama's context is not safe for concurrent use. Exactly one goroutine — the engine's decode
> loop — may call Decode, LogitsIth, or the memory methods on a given Context.*
> — `internal/llama/binding.go:13`, repeated at `internal/llama/runner.go:17`

Forty-eight goroutines entering one context is a data race inside llama.cpp and CUDA — the exact
class of failure the design set out to prevent.

It would also cost the thing that makes this engine fast. The measured 728.71 tok/s comes from 48
sequences riding **one** decode pass — **5.27×** what the same library retires on a single sequence
on that node (138.40). Forty-eight concurrent callers produce 48 passes of one
sequence each: the batch never fills, and the unified KV pool — the thing that makes amortisation
possible at all — has nothing to unify.

**The corrected shape:** the worker pool does host-side work — assembling buffers, parsing results —
and every call into the context is serialised through a single owner goroutine. Fan-out is wide
above the boundary and single-threaded at it. `TestEngineIsNeverEnteredConcurrently` counts peak
occupancy across 200 chunks and fails on anything above 1.

## E2. Fail-fast is right for generation and wrong for judging

`errgroup.WithContext` cancels every worker on the first error and `g.Wait()` returns only that
error, so the proposed code discards all successful results.

For **generation** that is correct: a failed tier-0 unit poisons everything below it, and continuing
spends GPU time on work that cannot be right.

For **judging** it destroys the result. One bad chunk in 48 must not discard the other 47 — they are
still evidence, and B4's coverage machinery exists precisely to report the gap. A run that returns
`nil, err` cannot say "46 of 48 chunks judged, these two were not," which is the honest output.

Hence `Policy`: `FailFast` for generation, `Continue` for judging. `Continue` returns a partial run
with the failures named; `Cancelled` marks a short result set so it is never mistaken for a complete
one.

## E3. Index assignment is right, and needs a guard

`results[chunk.ID] = …` is genuinely lock-free — distinct slice elements, with `Wait` supplying the
happens-before edge for the reader — and is better than a mutex around `append`.

It is safe only while IDs are exactly `0..n-1`. A subset of a larger document carrying its original
ids (12, 13, 30) writes out of range, or silently into another chunk's slot, which is a
misattribution nothing downstream can detect. Dense ids are now checked, along with duplicates.

## E4. Pre-tokenization: correct, and it survives unchanged

Tokenizing before the fan-out rather than inside the workers is right — CGO allocation and GC
pressure in the worker loop lands exactly when the GPU is waiting to be fed.

## E5. `LockOSThread` is right for a different reason, and needs its unlock

The stated reason — stopping the scheduler bouncing cgo calls across OS threads — does not apply
within a call: a goroutine is already pinned to its thread for the duration of any cgo call.

The real reason is that **CUDA context state is thread-local**, so consecutive calls landing on
different OS threads re-enter a different CUDA context. One long-lived locked thread keeps that
state stable across calls.

Two corrections follow. It belongs on the **single engine owner**, not on 48 workers — under the
corrected shape there is only one goroutine that needs it. And it needs `defer
runtime.UnlockOSThread()`: a goroutine that exits while locked destroys its OS thread rather than
returning it to the pool, leaking one per run.

## E6. Two dependency notes

`golang.org/x/sync/errgroup` would be this project's first dependency, and `go.mod` currently has
none. The needed behaviour is a `WaitGroup`, a `CancelFunc` and a first-error field — and `Policy`
means the fail-fast semantics errgroup provides are wanted in only one of the two modes. Tree-sitter
is the dependency worth spending the first slot on.

`streamID := streamID` is dead in Go 1.22+; loop variables are already per-iteration and this module
is `go 1.24`.
