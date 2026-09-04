# KV prefix reuse

Built: `internal/prefix/` (15 tests, no GPU needed) + `Context.SeqCp` / `Context.SeqPosMax` /
`llama.PrefixCache`. Tracked as #37. **Not yet measured on a card** — the throughput claim below is
arithmetic, not a result.

## 1. Why this is a prerequisite and not an optimisation

Every wide fan-out in this system puts the same tokens at position 0 of every stream: a global
symbol table when judging a codebase, a schema header for structured input, a document skeleton, a
system prompt.

At 48 streams a 1,500-token header is **72,000 tokens of prefill spent re-reading identical bytes** —
~22 s at measured prefill rate, and 28% of a 256k payload. That is the second-largest cost in the
judging pipeline and it buys nothing.

Without prefix reuse, the correct advice would be *"keep every shared header as small as possible"*,
which fights against the reason the header exists — a stream that cannot see the global symbol table
falsely reports "undefined type" for something defined three chunks away. **With it, the header is
nearly free and the design is straightforwardly right.** That is why it is filed as a prerequisite.

## 2. The mechanism, and why it is close to free

With a unified KV cache, upstream does not copy any data. From `llama_kv_cache::seq_cp`:

```cpp
if (s0 == s1) {
    // since both sequences are in the same stream, no data copy is necessary
    // we just have to update the cells meta data
    ...
    if (cells.seq_has(i, seq_id_src)) {
        cells.seq_add(i, seq_id_dst);
    }
```

The cell is *marked* as belonging to the destination as well. The prefix is computed once and every
stream attends to the same cells, so this is a **memory** saving as much as a compute one: the
shared cells are held once instead of 48 times.

Cost is one metadata pass over the cells per destination sequence — which is why `Plan.Worth` takes
a minimum shared length rather than assuming sharing always pays. Where that line falls is
unmeasured.

## 3. ⛔ One void function, three behaviours

`llama_memory_seq_cp` returns `void`. There is no failure signal at all, and it does three different
things:

| condition | behaviour |
|---|---|
| `if (other) { return; }` | **silent no-op** — nothing copied, nothing reported |
| same stream (unified cache) | metadata-only share — the intended path |
| cross-stream, partial range | `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")` — **aborts the process** |

A prefix copy is by definition a partial range `[0, n)`. So **without a unified cache this does not
degrade, it takes the process down**, and the precondition has to be enforced on the Go side before
the call. `Context.SeqCp` refuses with `ErrSeqCpNeedsUnified` rather than reaching the assert.

And because the silent-no-op path exists, **the effect is verified rather than assumed**: after each
copy, `SeqPosMax(dst)` must equal `n-1`. An unverified copy leaves a sequence generating fluently
from an empty history — a bad answer, not an error, which is the dominant failure class in this
system.

This gives the unified KV cache a **second independent justification**. It was adopted because it is
what lets the herd amortise a decode pass across sequences (`ARCHITECTURE.md` §3.3). It is also the
only configuration in which prefix sharing is possible at all.

## 4. The prefix is computed on tokens, never on text

⛔ A common *string* prefix does not imply a common *token* prefix. BPE merges across the boundary,
so two prompts sharing the characters `user_` can tokenise to different tokens there depending on
what follows.

Copying cells for a prefix derived from text would hand a sequence KV entries for tokens it does not
actually have. The model would attend to the wrong history, and nothing would signal it.

`Analyse` takes `[][]Token`.

## 5. A sequence is never fully absorbed

If one prompt is a strict prefix of the others, the naive shared length consumes it entirely — and a
sequence with nothing left to evaluate has no logits to sample from, because llama produces logits
from the tokens it is given this pass.

`Analyse` caps the shared run at one token short of the shortest prompt. Identical prompts share
`n-1` and each keeps one.

## 6. Wired into the engine: opportunistic, not batched

`engine.PrefixSharer` is an **optional backend capability**; a backend that cannot share simply
prefills normally. `Runner` implements it, and `Engine.sharePrefixes` runs immediately before the
prefill pass.

**Opportunistic rather than batched, deliberately.** Requests arrive one at a time through `Submit`,
so there is no point at which the whole set is known — matching a fresh slot against whatever is
already resident works regardless of arrival order, and needs no new API.

⭐ **Safe across donor lifetime, which is what makes this work at all.** Cells are reference-counted
by sequence membership upstream:

```cpp
seq[i].reset(seq_id);        // drop this sequence's tag
if (seq[i].none()) {         // only when NO sequence still references it
    pos[i] = -1;             //   ...is the cell actually freed
    return true;
}
return false;                // cell survives, still referenced
```

So a donor finishing drops only its own tag. A borrower never needs the donor to stay alive, and no
lifetime coordination between streams is required.

Four constraints the implementation has to respect, each with a test:

- **The donor must already have prefilled past the shared run.** A slot admitted in the same tick
  holds no cells, so copying from it shares nothing while reporting success. Sharing is capped at
  the donor's cached position.
- **The borrower keeps at least one token.** A prompt entirely absorbed leaves nothing to sample
  from.
- **Only fresh slots.** Sharing into a partly-filled sequence would duplicate positions it holds.
- **Nothing is half-applied.** `pos` and `pending` are untouched until the share is verified, so
  every failure path falls back to ordinary prefill — which computes exactly what the share would
  have copied.

**Donor choice is deterministic.** Go randomises map iteration, so two donors offering an equally
long prefix would otherwise be picked arbitrarily and identical inputs would not produce identical
plans.

### It has to be observable, because failure is invisible

A failed share degrades to ordinary prefill. That is correct behaviour and **indistinguishable from
never having tried** — so a sharing path that silently stopped working looks exactly like one
running on prompts with nothing in common.

`Stats` therefore carries `prefix_shared_total`, `prefix_tokens_saved` and `prefix_share_failed`.
The failure count is the one that matters.

## 7. What is not done

- **No measurement.** The 28%/~22 s figures are arithmetic from the measured prefill rate. The
  actual win depends on the metadata-pass cost, which needs a card (#37).
- **`minSharedPrefix` is 256 and is a guess**, deliberately conservative: a wrong guess here wastes a
  little work, where the opposite error makes short requests slower for no reason. It wants
  measuring alongside the throughput number.
- **`internal/prefix`'s batch planner is now the second path.** `Analyse`/`Share` remain the right
  shape for a caller that genuinely knows its whole set up front (the fan-out orchestrator); the
  engine path covers everything else.
