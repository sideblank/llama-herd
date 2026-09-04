# DAG scheduling: ordering that survives a long context

Tracked as #30. Status: built, tested and race-clean (`internal/vcontext/dag.go`, `schedule.go`),
wired to the HTTP runner (`httprunner.go`, `taskexec.go`) and driven end to end by
`llama-herd tasks`.

## 1. The problem this solves

A 256k request is not a uniform pile of work. It contains dependent sequences — *do B with the
output of A* — interleaved with background work that depends on nothing. The fan-out layer as built
treats every chunk as independent, which is correct for retrieval and wrong for instructions.

The tempting fix is to tell the model about the ordering: put the dependencies in the prompt and ask
it to respect them. That fails in a way which is hard to detect. A model asked to honour "step four
needs step two's result" across a long context can reorder, or invent an intermediate step, and the
output still reads like a coherent plan. Nothing errors. The wrong answer is well-formed.

So ordering is not asked of the model. It is extracted once, then enforced by code.

## 2. Two phases

**Extract the graph.** One pass over the request produces tasks and their prerequisites. The output
is grammar-constrained (`TaskGrammar`) so it cannot come back as prose or omit an id — a
malformed extraction is a scheduling failure discovered after GPU time has been spent, and the
grammar makes the shape valid by construction.

**Resolve and execute.** `Graph.Sort` runs Kahn's algorithm to produce tiers. A cycle means no valid
ordering exists and the request cannot be scheduled at all — reported before anything runs.

## 3. Tiers are for reasoning; they are not the execution unit

The obvious implementation runs tier 0 to completion, then tier 1, and so on. It is wrong, and the
reason is a number this project already measured.

A tier barrier makes every task in tier *n+1* wait for the **slowest** task in tier *n*, including
tasks it has no relationship with. A 40-second summarisation in tier 0 holds back a tier-1 task
whose actual prerequisite finished in two seconds. That is the straggler problem — the one
`Batch.StragglerRatio` exists to expose — reintroduced as architecture rather than as a measurement
artifact.

The ordering guarantee is identical without the barrier. A task must not start before **its own**
dependencies complete; it need not wait for unrelated siblings. So `Schedule` dispatches on
dependency-satisfaction: each completion decrements its dependents' pending count, and any that
reach zero launch immediately, bounded by the stream budget.

On a graph with one long pole and many short branches, the difference is most of the wall clock.
`TestScheduleDoesNotWaitForASlowSiblingInTheSameTier` pins it.

Tiers are still computed and reported on the `Run`, because they are the right way to *describe*
the graph. They are just not how it executes.

## 4. Context between tasks is text, not latent state

A dependent task receives its prerequisites' outputs as text, keyed by task id.

Forwarding `h_last` instead is the faster path in principle and is where this should end up. It is
not the default today because the projection it needs is unresolved (#21) and the gating measurement
(#19) showed injection losing detail the hidden state provably holds. Building the scheduler on that
would make every dependent step quietly lossy — the same well-formed-but-wrong failure this design
exists to eliminate. Text is the honest default until the projection is measured working.

Two related properties, both tested:

- A task receives **only** what it declared a dependency on. Context it did not ask for is context
  it cannot be held to, and passing everything to everything reintroduces the whole-context problem
  one level up.
- Outputs are keyed by task id, never concatenated positionally.

## 5. Failure has two shapes and they need opposite responses

A task that **ran and errored** may be worth retrying. A task that **never ran** because a
prerequisite failed cannot be retried until the prerequisite is fixed. Reporting both as "failed"
hides which is which, so `TaskResult` carries `Blocked` and `BlockedBy`, and `BlockedBy` names the
*originating* failure rather than the immediate parent — the nearest broken thing is what the caller
has to fix.

A failure in one branch does not cancel the graph. Unrelated work still runs, and
`Run.Text()` omits failed and blocked tasks entirely: an answer assembled from whichever tasks
happened to succeed is an answer to a different question, and nothing downstream can tell unless the
gap is named.

## 6. What the graph tells you before you run it

`CriticalPath` is the longest dependency chain, which bounds completion time however wide the herd
is. `Independent` finds tasks with neither prerequisites nor dependents — background work that can
run throughout. `Run.Span` records the widest concurrency actually reached; well under the stream
budget on a slow run means the **graph** was too serial to fill the herd, which is a property of the
request rather than of the deployment.

This matters for the 48-stream target. A request whose critical path is most of its tasks will not
go faster on more streams, and knowing that before dispatch is better than inferring it from a
disappointing wall clock.

## 7. Where it sits

Canonicalisation runs first, once, before chunking. Graph extraction runs on the canonical text.
Then `Schedule` executes, with each task's work going through the existing fan-out machinery.
`Dispatch` remains the right tool for a single task that is itself wide — the two compose, they do
not compete.

## 7b. The path, end to end

`llama-herd tasks` runs it against a real engine: extract, sort, execute, assemble.

```
$ llama-herd tasks --request "Collect logs, audit deps, find errors, summarise"

extracted 4 tasks in 1ms

  tier 0  fetch_logs, scan_deps
  tier 1  parse_errors
  tier 2  summarise

  critical path : fetch_logs -> parse_errors -> summarise (3 of 4 tasks)
  widest tier   : 2 (streams available: 8)
  the graph, not the deployment, is the limit here
...
wall 5ms (extract 1ms, run 4ms) - peak concurrency 2/8
```

Three things it reports that a wall clock alone cannot:

- **The critical path**, printed *before* execution. A request whose critical path is most of its
  tasks will not go faster on more streams, and that is worth knowing in advance rather than
  inferring from a disappointing result.
- **Peak concurrency against the stream budget.** 2 of 8 says the herd was underused because the
  graph was serial, not because the deployment was small — two very different problems that produce
  the same slow run.
- **`--dry-run`** stops after the plan, so a decomposition can be inspected without paying for the
  execution.

**Extraction asks for dependencies, never for an order.** A model asked to list steps in order
produces an order with no way to check it; asked what each step *needs*, it produces a graph that
either sorts or does not. That is the difference between ordering as a claim and ordering as a
computation. The response is grammar-constrained and validated with `Sort` before it is returned, so
a cycle or an invented dependency is caught before any stream runs.

**Assembly names what is missing.** Failed and blocked tasks contribute no text and are listed under
an explicit "This answer is incomplete" — an answer silently built from the tasks that happened to
succeed is an answer to a different request.

**The assembled prompt is budget-checked and refused, never truncated.** A task's prompt is its
description plus its dependencies' outputs, and a task with several long prerequisites can exceed
one stream. Truncating would silently drop part of a prerequisite; `ErrPromptOverBudget` names the
task and the dependencies instead. The shared instruction at the head of every prompt is identical
across streams, which makes it a prefix-reuse candidate — `SharedPrefix()` exposes it.

⚠️ The CLI supplies `CharEstimate` rather than a real tokenizer, so the budget check catches an
order-of-magnitude overflow and not a marginal one. Character count is not a proxy for token count
(#42).

## 8. Open

- ~~Wire `Schedule` to `httprunner`~~ — done (#31): `TaskExecutor`, `ExtractGraph`,
  `Run.Assemble`, `llama-herd tasks`.
- Give the CLI's budget check a real tokenizer instead of `CharEstimate` (#42).
- Measure a real interleaved request end to end, **with a tier-barrier control arm** — §3 is an
  argument until that number exists (#32).
- Extraction quality: a cycle is caught, a missing edge is not. A graph that sorts cleanly can still
  be the wrong graph, which moves the well-formed-but-wrong failure one level up rather than
  removing it (#33).
