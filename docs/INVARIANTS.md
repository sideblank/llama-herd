# Invariants

Things that are true about this system and were expensive to learn. Each one caused, or would
have caused, a failure that looked like something else.

Most are silent failures — the system reports success and produces wrong numbers, wrong
output, or a bill. That is why they are written down rather than left to be rediscovered.

---

## Capacity and memory

**KV cost is set by the layers that actually cache, not by the layer count.** Hybrid
architectures cache on only every Nth layer and use linear attention elsewhere, whose state is
constant-size and does not grow with context. Assuming every layer caches overstates KV by
exactly that factor — enough to conclude that a working production configuration is
impossible. `llama-herd fit` reads the interval from the file.

**The weight budget and the KV budget are the same pool.** Compressing weights frees VRAM
that becomes KV capacity, so quantization does buy context — bounded by the size of the
weights. When a target's KV cost alone exceeds the card, no quantization reaches it.

**Capacity follows attention geometry, not parameter count.** Two models of similar size can
differ fourfold in what they hold. `layers-that-cache × kv_heads × head_dim` is the number
that predicts it.

**A split context cannot be borrowed from.** With the default even split, the per-stream
ceiling is fixed at startup and idle slots hold capacity nobody can use. Only a unified pool
lets one request exceed its share — and then nothing is reserved, so admission must track
real free capacity rather than a nominal per-stream limit.

**Quantized KV requires flash attention.** Setting one without the other does not work. The
manifest refuses the combination.

**Keys tolerate quantization less well than values.** Keys at `q8_0` with values at `q4_0`
often beats both at `q5_1`. It needs a build with all flash-attention quant kernels compiled;
the default build carries only matched pairs, and an unmatched pair silently leaves the fast
path rather than failing.

## Multi-token prediction

**Declared is not present.** A file can carry MTP metadata while carrying none of the tensors,
and the declaration survives quantization even when the weights do not. Only a real load with
the layers enabled proves it.

**Tensors can be lost at three separate stages** — weight-level modification, conversion, and
quantization — for different reasons. Check after each.

**Do not quantize the MTP head at body precision.** A crushed draft head proposes badly, the
accept rate collapses, and the speculative path gives back nothing while still occupying
VRAM. The cost is paid twice.

**Loading the head is not using it.** A model can report its MTP layers loaded and still
produce exactly one token per forward pass. Driving the head requires a second context of the
MTP type, linked to the target, and a draft-then-verify loop; the model flag alone only makes
the weights resident. Measured on a real model: tensors loaded, 1.00 tokens per pass.

**The API for driving an MTP head is not public.** The public header exposes the layer count
and nothing else; feeding a draft context needs the target's hidden states, which live in a
staging header that is not installed, permits breaking changes, and asks not to be included.
Any MTP implementation either wraps upstream's common library or vendors an unstable header.

**Presence is not benefit.** The accept rate is the number that decides whether MTP earned its
space, and it varies by model and by workload.

## The engine

**Active decode tokens must consume the batch budget before prefill is admitted.** Filling
with prefill first lets one long prompt build a batch larger than the backend accepts, which
kills the engine under mixed load rather than erroring.

**Admit against the per-sequence context, not the total.** The total is shared across slots,
so validating against it accepts prompts that run out of cells mid-generation and surface as
an eviction instead of a clean rejection.

**Sampler state is per-sequence.** A chain carries the penalty window, so sharing one across
streams lets their histories penalise each other — visible as quality degradation, not as an
error. Releasing a slot must reset it, or a reused slot inherits the previous request's
history.

**A full KV cache is not a failure.** `llama_decode` returning 1 means evict a sequence and
retry; conflating it with a real error kills a healthy engine.

**A dead decode loop must be refused, not queued against.** Otherwise one engine failure
becomes a pile of requests waiting on a loop that will never tick again.

## Measurement

**The inference library's decode counter is wrong for a batching engine.** It increments only
when a batch holds exactly one token; a continuous-batching engine submits one token per
active stream, so decode work is attributed to prefill and the decode counter reads near
zero. Upstream carries a FIXME acknowledging it. Count passes in the engine instead.

**Tokens per pass has the stream count as its baseline, not one.** One forward pass serves
every active stream, so four streams yielding 3.9 tokens per pass is ordinary batching, not
speculation. Reading 1.0 as the baseline reports speculation on every multi-stream run.

**A model carrying MTP layers that reads at the baseline is the failure worth catching.** The
head is loaded, occupying VRAM, and contributing nothing.

**The first token of an answer comes from the prompt's own logits.** Prefill produces logits at
the last prompt position, and that position must actually be sampled. Recording where they
landed and never reading it leaves the slot with no token to continue from, and the next pass
stages whatever the "next token" field holds — the zero value — inserting a token the model
never produced ahead of the real first one.

**Its symptom depends entirely on the tokenizer, which is why it can survive for a long time.**
Where token 0 is ordinary text the model absorbs it and returns a plausible answer with a stray
prefix, and everything reads as working. Where token 0 ends a turn the request returns nothing
and reports a clean stop, which reads as a broken model or a broken chat template. Both were
observed on the same build, on qwen2 and qwen35 respectively.

**The library defers the GPU wait to the first read of an output, so time it explicitly.**
`llama_decode` queues a graph and returns; `llama_get_logits_ith` calls `llama_synchronize`
underneath. In an engine that samples right after decoding, that bills the GPU's entire compute
time to the sampler. It read as 52.66% of engine time in sampling and 47% in decode; calling
`llama_synchronize` after the decode moved it to 97.61% decode and 2.06% sampling, with no
change in throughput. Any phase timing around a lazily-synchronised API measures where the wait
was collected, not where the work happened.

**With the wait attributed correctly, this engine costs 2.4% of the total.** Staging, sampling,
detokenization and delivery together. That bounds what tuning the scheduler can ever recover and
says the remaining headroom is in the forward pass — stream count, quantization, kernels,
speculation — not here.

**The library's one-call sampling helper rebuilds the whole vocabulary per token.**
`llama_sampler_sample` materialises one candidate record per vocabulary entry on every call —
about 1.8 MB at a 150k vocabulary, above the threshold where the allocator takes a fresh mapping
from the kernel, so each token pays a map, hundreds of page faults and an unmap. Allocate the
candidate buffer once per chain and call `llama_sampler_apply` instead. Whatever replaces the
helper must also call `llama_sampler_accept`: the helper does, and omitting it leaves the penalty
samplers with an empty window, so repetition penalties silently stop applying.

**Narrowing the candidate set before the chain is sound, and it is where the time is.** Every
sampler that runs before the truncation can only lower a logit — repeat penalty divides a
positive and multiplies a negative, frequency and presence subtract — so a token outside the
kept set can never climb into the final top-k. Keeping `TopK + RepeatLastN` candidates
guarantees at least `TopK` untouched ones survive. Measured on a 3090: the library's chain fell
to 0.013 ms per token, from a sampler that had been costing more than the forward pass.

**Then the cost moves into whatever does the narrowing.** Selecting candidates by histogram over
every logit cost 0.98 ms per token — 98.6% of the sampler — so the fix had only relocated the
work. The cutoff estimate does not need every logit: reading one in eight and cutting a margin
lower brings selection to 0.23 ms, because a cutoff that errs low merely keeps more candidates
and the chain no longer cares. Only the pass that collects them has to read everything.

**Sampler changes must be proven byte-identical before they are believed.** Fix the seed, run
one stream, several prompts, a few hundred tokens each, and diff. Every step above was verified
that way; a truncation that quietly changes which token wins would show up as nothing but prose
that reads slightly worse.

**A throughput figure without its context depth means nothing beyond depth zero.** Decode reads
the KV cache for everything already in the sequence, so a herd measured with a short prompt
reports a ceiling it cannot hold at the context the deployment promises. Measured here: 547 tok/s
at 24 streams with a nearly empty cache is a different quantity from the same configuration
serving real conversations, and only the second one ships.

**The aggregate decode window includes other streams' prefill.** It opens at the first token any
stream produces and closes at the last, so while the remaining streams are still prefilling,
their prefill is inside it. Harmless with a short prompt, dominant with a deep one — it read as
throughput collapsing by a factor of thirty when nothing of the sort had happened. Generate
enough tokens at depth that decode outweighs the prefill sharing the window, or measure each
stream's own decode span.

**A benchmark must refuse a configuration that cannot exist.** A prompt has to fit inside one
sequence's share of the context along with what it will generate. A sweep asked 24 streams
sharing 425,984 tokens to hold a 32,768-token prompt — 17,749 each — and reported a throughput
figure for it. A number for an impossible configuration is worse than an error, because it
looks like data.

**Prefill and decode move in opposite directions with stream count, so they cannot share a
default.** Splitting one input across more streams cost about 19% of ingest throughput while
roughly doubling decode. One long prefill already saturates the card; dividing it adds overhead
rather than parallelism. A deployment tuned on decode alone will be tuned wrongly for
ingest-heavy work, and the other way round.

**What costs throughput is depth within a sequence, not tokens resident across the herd.**
Holding the same total resident and varying only the split, more and shallower sequences measured
up to 1.9x faster than fewer and deeper ones. So subdividing a large input across streams is the
faster arrangement, not a compromise — an intuition that the same tokens cost the same wherever
they sit predicts the opposite, and was wrong.

**More streams buy throughput; allocated context does not cost it.** Decode reads KV in
proportion to how full a sequence currently is, not to what was reserved, so shrinking the
allocation frees memory rather than time. What raised aggregate throughput 4x was tokens per
forward pass — one weight read serving more sequences.

**Write results where they outlive the process that produced them.** A sweep that persists after
every configuration is still worthless if it persists into the container's `/tmp`: one
configuration crashing destroyed three that had already succeeded. An hour of GPU time and a day
of downstream work were lost to results that existed and then did not. Durable means outside the
thing that can die — the repository, a mounted volume, an endpoint — and the check is to ask what
happens when the process is killed rather than when it exits.

**Record a finding when it happens, not when the campaign ends.** Results held in a session's
context are lost to a restart, a crash, or a compaction, and what survives is a summary written
from memory — which is how a six-hour effort gets described as forty minutes. The cost is not
symmetric: writing a number down takes seconds, recovering it takes a rented GPU and an hour.

**The stream ceiling is a cliff, and it moves with the node.** On a 3090 with
`Qwen3.6-35B-A3B-UD-IQ3_S` and a 425,984-token unified pool, 128 sequences took the process down.
The ladder above 48 was then measured: throughput plateaus from 48 to 64 (a 3% gain) and collapses
at 72 on one node, while an earlier node ran flat through 72 and failed at 80. Ship 48; it sits
furthest from a cliff whose position nothing measured can predict. See `results/3090.md`.

**A benchmark must prove it is talking to the process it started.** A container left listening
from an earlier run makes every later container fail to bind while the port keeps answering, so
the harness measures the old build and reports it under the new build's name. Observed here as
four consecutive A/B readings agreeing to three decimal places — identical numbers from
different container ids, which is the tell. Compare the listening pid against the container's,
refuse to start when the port is already held, and give each run its own port.

**Identical results are evidence of a broken harness, not a stable measurement.** Real
throughput on the same configuration varies run to run; two runs that agree exactly are almost
always the same run counted twice.

**Never ask a model for byte offsets.** It cannot count characters, and an offset that is
plausibly wrong assembles text from the neighbouring region — the answer is confident, coherent and
about something else, and nothing downstream can detect it. Locate spans arithmetically and let the
model label what was located. Locating is arithmetic; only labelling is judgement.

**A merge over parallel streams must be order-independent, or results depend on a race.** Streams
finish in whatever order the scheduler produces. An order-sensitive combine answers the same
question differently across runs, with nothing in the output to show which answer was which — the
caller cannot see it, reproduce it, or report it. Make every rule commutative, sort by source before
combining, and assert it by shuffling the inputs and demanding an identical result.

**Disagreement between sources is information, not noise to resolve.** Two parts of a document
saying different things is a fact about the document and frequently the important one. A merge that
silently picks a side produces a confident answer that may be the wrong half. Collect distinct
values, record the conflict with both sides and their provenance, and leave resolution to something
that can show its reasoning.

**A grammar belongs first in the sampler chain, and it must disable every fast path.** It masks
tokens that cannot appear next, so samplers after it rank only valid continuations. Placed after
truncation it filters an already-narrowed set, and if top-k has dropped every grammatically valid
token there is nothing left to choose — which presents as a model fault. The greedy fast path must
also be skipped: scanning raw logits picks the highest scorer regardless of the mask, producing
exactly the output the grammar existed to prevent.

**Load a grammar from a file; never interpolate one through a shell.** GBNF is dense with quotes
and backslashes, and shell-then-JSON quoting corrupts it. The failure surfaces as "grammar failed
to parse", which points at the grammar rather than at the quoting that broke it.

**Check tied-vs-untied embeddings before any latent-space experiment.** With tied embeddings the
output projection IS the input embedding matrix, so `logits = E·h` and a hidden state collapses to
roughly one token of information — a whole class of architecture is bounded before it starts. With
untied embeddings no such reduction exists. It is a metadata read (`output.weight` present or
absent) and it decides whether the experiment means anything. Measured here: Qwen2.5-0.5B is tied,
Qwen3.6-35B is not — so a result from the cheap local model did not transfer, and nearly became a
wrong conclusion about the architecture.

**A hidden state encodes whatever its trailing instruction asks it to predict.** The final position
of a causal model is a next-token state, so what it carries is set by what follows the content.
Asked to "synthesize the key entities" it encoded entities and lost a port number; asked for "the
port number mentioned above" it predicted the port exactly. Neither is the model's representation
of the chunk — there is no such thing without an instruction, and choosing one badly looks
identical to the representation being incapable.

**Separate "not in the state" from "lost in transit" before concluding either.** Injecting a hidden
state and getting no detail back has two causes with opposite fixes: the state never held it, or
the injection destroyed it. Generating directly from the same position settles it in one run —
here it showed the value was present and the mapping was at fault, reversing a conclusion already
written down.

**A throughput shortfall against the library is only a lead once it is attributed.** The engine
can be slower than `llama-bench` in exactly three places — staging a batch, the library's
forward pass, harvesting — and the fix differs completely by which. The engine times all three
and publishes the split on the selftest, because four separate theories about a throughput gap
were argued from arithmetic before anyone measured where the time went, and all four were wrong.

**Harvest time is not sampler time.** Harvest covers sampling, detokenization, and handing the
token to the caller, and the last of those blocks when the caller is not reading. A large
harvest share can therefore mean the consumer is the bottleneck while the sampler is innocent.
Time the three apart before optimising any of them.

**A phase share is meaningless without the absolute cost behind it.** The same sampler is ~1.6%
of engine time on a CPU and a large fraction on a GPU, because the denominator moved: the
library's forward pass got 20x faster and the sampler did not. Convert to time per token before
comparing two machines, or a constant cost reads as a regression.

**`llama-bench` does not sample.** Its `tg128` figure is decode with no sampler chain, no
detokenization and no delivery, so an engine that produces usable tokens cannot match it and
should not be expected to. The gap is the price of doing the work, and only the part above that
price is worth chasing.

**A stand-in backend must honour the batch index it is given.** Logits exist only where output
was requested, so sampling a position that produced none reads whatever is in that memory. A
fake that ignores the index and returns the next scripted token cannot distinguish a correct
caller from one reading the wrong row — which is exactly how a defect affecting every response
passed a green test suite.

## Speculation

**Measure acceptance directly; do not infer it from tokens per pass.** A forward pass can
carry prefill for one stream and decode for another, and a prefill pass produces no tokens at
all, so the ratio moves with prompt length and stream mix rather than with speculation. It
read *below one* on a long prompt with a short answer. Count proposals and acceptances
instead: their ratio means exactly one thing.

**A rejected draft costs batch space, never a token.** The token at a divergence is the
target's own choice, so a wrong proposal still yields one real token. Speculation cannot slow
generation down; it can only waste batch capacity.

**Drafts consume batch budget like any other entry.** Counting only the real token lets a
speculating stream overrun the batch, which the backend rejects outright.

**A drafter holds per-sequence state and must be released** when a slot finishes, for the same
reason a sampler is reset.

**A drafted token costs a whole decode call, not a fraction of one.** A model with one
prediction layer predicts one token ahead, so drafting k tokens means k sequential decodes on
the draft context, plus one to resynchronise it, plus the target's own pass. The draft context
touches one layer against the target's forty-one, so the arithmetic looks free — and is not.
Measured on a 3090 with a 35B-A3B at 57% acceptance and max_draft 2: a speculative pass cost
14.2ms against 51.9ms, buying 1.72 tokens instead of 1.00. Four decode calls for 1.7 tokens.
Fixed per-call cost dominates, and no acceptance rate recovers it.

**So judge speculation on decode calls per token, not on acceptance.** Acceptance was 57% here,
near the ceiling for a one-layer head, and speculation was still 2.1x slower than not
speculating. The figure that decides it is (k + 2) calls divided by the tokens a step yields;
below one it pays, above one it cannot. A head predicting several tokens per pass changes that
arithmetic. A higher acceptance rate does not.

**Speculation must not change the output.** It is an optimisation, and the token at a
divergence is the target's own choice, so a speculative run and a plain one must produce the
same text from the same prompt. If they differ, the state is corrupt — no matter how healthy
acceptance, tokens-per-pass and the error count look. All three were green on a run whose prose
had degraded into repetition.

**Not every cache can be rewound by position.** `llama_memory_seq_rm` returns false when a
partial removal is impossible, which is the normal answer for recurrent and hybrid
architectures — the ones long-context models use. Discarding that result leaves the engine
believing it rewound while the cache still holds rejected drafts, and the next batch is refused
for inconsistent positions, killing the model for every stream. Measure the capability at load,
the way upstream does: decode two tokens and try to remove one.

**Where position cannot rewind, checkpoint the rest — and replay what was accepted.** A
snapshot taken while a batch is being built precedes that batch, so restoring it discards the
accepted tokens as well as the rejected ones. Trimming the attention cache to the accepted
position then leaves it holding tokens the recurrent state has never seen. The accepted prefix
must be walked through again. It costs a pass per rejection, and that cost is the reason to
measure whether speculation pays on such a model rather than assuming it does.

**One KV pool or one per stream decides whether the herd amortises at all.** The library splits
a batch on that setting: with a single pool it runs every stream's tokens in one forward pass,
with a pool per stream it runs one pass per sequence. Four streams then cost four passes and
sharing the weights buys nothing — the premise the whole design rests on. Measured on a 3090
with a 35B-A3B: 182 tok/s against 55, from that flag alone.

**And no counter shows it.** Tokens-per-pass reads the batch handed to the library, not what
the library did with it, so it sits at a healthy 3.88-of-4 while four passes run underneath.
The reading that exposes it is aggregate throughput against the same model's single-sequence
rate: a herd below its own one-stream number is not amortising, whatever else looks right.

**Rented machines vary by more than most effects being measured.** The same image, manifest and
model measured 42 and 118 tok/s on two nodes — 2.8x apart. Any conclusion drawn from figures
taken on different machines is drawn from that spread, not from the change under test. Measure
within a node, or repeat until the spread is visible.

**An HTTP measurement is not comparable to a published one.** Published figures for a GGUF come
from in-process loops, so a served number held up against one charges the serving stack to the
engine. Measured within a single node the serving path costs about 7% here — an earlier
cross-node reading put it above 2x, and that number was node variance wearing a methodology
costume.

**Byte-identical output is only a valid test at one stream.** Continuous batching admits
concurrent requests as they arrive and gives each whatever batch room is left, so two runs of
the same prompts interleave differently. Different batch composition means different
floating-point rounding, and one flipped choice between near-tied logits diverges everything
after it. Measured on a 27B with speculation switched off entirely: two concurrent streams
disagreed with each other and with themselves across runs, while the same request run singly
reproduced exactly.

**So verify speculation single-stream, and do not read a multi-stream difference as a defect.**
The comparison that means something is one request at a time, sequentially, with and without
speculation: batch composition is then identical and any difference is real. Above one stream
the control does not reproduce, so it cannot establish anything about the thing being compared
to it. Correctness for the multi-stream case comes from tests that are deterministic by
construction — isolation, rollback and replay — not from comparing text.

**Speculation changes batch shape, so byte-exact equality is a strong signal and not a
guarantee.** Staging several tokens where one would go changes floating-point rounding, and a
near-tie settles differently as a result. Corrupt state diverges early and on nearly every
prompt; a near-tie diverges late and rarely. One failing prompt is not a verdict — this was
read as a defect twice before the pattern was checked.

**Keep the binding buildable against older library revisions.** Referencing an enumerator or a
function that a newer llama.cpp introduced makes it impossible to compile against an older one,
and therefore impossible to tell a change in the library from a change here. That comparison is
the only thing that settles a throughput difference between this engine and one built against a
different revision — a day went into four wrong explanations partly because it could not be run.
Detect the capability rather than assume it: a header grep for the symbol, a stub carrying the
same ABI, constants written as literals.

**Constants written as literals must be checked against the header.** Trading a compile error
for a silent mismatch is only safe if something catches the mismatch. Transcribing
`llama_load_mode` by eye put every mode one place out — the enum starts at -1 — so Auto mapped
to None and Mmap to Mlock, changing how weights load with no symptom but speed. Guard the check
with a build tag so the older header, which is the reason for the literals, still compiles.

**When a working implementation exists, diff against it before theorising.** A theory that
explains the measurements can still be wrong, and a good one is harder to abandon than a bad
one. Decode throughput here sat six times below what the same model, quant and card reach in
the runtime this project was extracted from, and three mechanisms were proposed and tested to
explain it — exhausted VRAM, batching that cannot amortise, expert divergence in a mixture
model. Each fitted the evidence. All three were wrong.

The cause was six lines of context setup: `n_ubatch` forced to the logical batch size instead
of left at the library default, so every compute graph was built four times larger than the
work submitted to it. Reading the two setups side by side would have found it in minutes,
which is less time than any one of the theories took to disprove.

**A stand-in that does not model the contract cannot test it.** A fake recording state when a
token is *staged* rather than when it is *decoded* hides an off-by-one in exactly the step a
checkpoint occupies, and a fake returning scripted output regardless of state cannot detect
state corruption at all. Assert on the sequence the model was walked through, against a run
with no speculation.

**Loading a prediction head is not driving one.** `load_mtp` makes the head resident; it does
not make anything propose from it. The runtime ran for weeks reporting `mtp_loaded=true` with
zero drafts — the head was occupying VRAM and contributing nothing, and every signal except the
draft counters looked healthy. Treat `mtp_loaded` as a statement about memory, never about
speculation.

**A quantization that dropped the head must degrade, not refuse.** Many published quantizations
strip it silently, so a manifest asking for `mtp` against such a file is a configuration mistake
worth naming — not a reason to deny service. Log the cause and serve without drafting.

**A draft context has its own KV cache and must be budgeted for.** Speculating from a model's
own head opens a second context over the same weights. The weights are not duplicated, but the
cache is, and it is charged at the target's context length. Nothing shares it automatically:
`ctx_other` is wired for only a few architectures, and for the rest the draft context allocates
independently.

**Build the draft context from the target's geometry, never from defaults.** An unset context
size means *the length the model was trained with*, and unset cache types mean f16 whatever the
target quantized to. On a long-context model those compound into an allocation that fails on a
card the target fits comfortably. Worse, the failure returns a null context — the same signal a
model with no prediction head gives — so the runtime blames the quantization for a
misconfiguration of its own. Measured: a 3090 left with 0.8 GiB free, loading cleanly and dying
on the first full batch.

**Wrap the maintained speculation surface rather than the internal one.** Driving a head
directly needs `llama_get_embeddings_nextn` from a staging header that upstream does not install
and asks callers not to include. `common_speculative` covers all three MTP architectures and
moves with them; pinning to the internal surface buys a working prototype and a breakage on the
next update.

## Vision

**The prompt must contain the media marker.** Without it the tokenizer produces chunks with no
media, the image is dropped, and the model answers fluently about nothing. Refuse at submit.

**Vision is a different way of filling the prompt, not a different decode path.** Media is
encoded into the sequence's KV, and generation continues through the ordinary loop from the
position returned.

## Builds and packaging

**Backends built as plugins must be registered explicitly.** Without it the device list is
empty, no GPU is found, and the server runs on CPU while reporting itself healthy.

**Those plugins install to `bin/`, not `lib/`.** Copying only `lib/` drops them and produces
exactly the failure above.

**Link the ggml libraries explicitly.** Modern `ld` does not resolve indirect shared-library
dependencies transitively.

**The link must tolerate undefined symbols in shared libraries.** The CUDA backend references
driver symbols supplied by the host at run time and absent from every build image.

**`libgomp1` is required at run time.** The CUDA runtime base does not ship it, and the
failure appears only when the image is run.

**Compile the CPU backend for a generic instruction set.** Building for the build machine's
CPU faults on any older machine the container is scheduled onto.

**Ship the base CUDA image, not the runtime one.** The runtime image carries about 2.8 GB of
libraries; static inspection shows the CUDA backend names only the runtime, cuBLAS and NCCL,
plus the host driver. Copying just those halves the image.

This is a reliability property, not only a speed one: a scheduled worker cold-pulls the whole
image before the container starts, so a smaller image is a shorter window in which a spot node
can vanish mid-pull.

**Never use `ldd` in a build.** It executes the target's ELF interpreter, which silently
no-ops under emulation — printing nothing, copying no dependencies, and making the gate pass
vacuously. Use static inspection.

## Deployment

**A GPU runtime that finds no GPU still works.** It answers every request correctly, passes
its health check, and runs an order of magnitude slower. Nothing in a response reveals it,
and throughput only reveals it if you already know what to expect.

`GET /v1/info` reports the devices actually found and says plainly when none is accelerated.
Check it after every deploy. Judging from response latency alone is unreliable — network
round trip hides a lot at small model sizes.

## Operations

**Some settings are creation-time-immutable.** Container-group priority is accepted at create
and ignored on update; image caching likewise. Getting them wrong means recreating, not
patching.

**A queued change blocks the next one.** Patching while a change is pending is refused with a
bad-request error that says nothing about why. Wait for the queue to drain before every patch,
not only before reading a result back.

**An image change is queued, not applied.** Reading the image back straight after a patch
returns the old value, which is indistinguishable from a silently dropped patch. Wait for the
provider's pending-change flag to clear before judging, then read back — both steps, because
either alone gives a wrong answer.

**An image patch on a running group is silently dropped.** The call returns success and the
live image is unchanged. Stop first, and wait until the group has actually reached stopped
rather than trusting that the stop call returned.

**Registry credentials travel with the image.** Patching the image alone is accepted and then
cannot pull.

**Deploy immutable tags, never floating ones.** A provider that caches images at the edge can
serve a stale copy of a mutable tag: pushing a new `:dev` and restarting brought up the
*previous* build, with its endpoints still missing and nothing reporting a mismatch. The
symptom was a health check passing while a new endpoint returned 404. Pin to a commit or a
digest so a deploy means exactly one image.

**Verify the deployed build, not the pushed one.** `/v1/info` reports the commit actually
running. Checking it is the difference between deploying and believing you deployed.

**Judge whether something is billing by instance counts, not status strings.** A group can sit
in a non-stopped state while pre-caching an image, with nothing allocated and nothing charged.

**macOS CI runners bill at ten times the minute rate.** A full platform matrix on every push
exhausts a month of included minutes in hours, and the failure appears as an unrelated
workflow refusing to start.

## Method

**When the arithmetic says a running system is impossible, the arithmetic is wrong.** Go and
read the running configuration before arguing with it.

**Model cards are not evidence.** Verify against the file.

**A number without its method is a claim.** State what was measured, on what, and how.

## Ordering and dependencies

**A model asked to respect an ordering can violate it without erroring.** Reordered or invented
steps still read as a coherent plan. Extract the dependency graph once, then enforce it in code.

**Tier barriers convert an ordering constraint into a straggler problem.** A task need only wait for
its own dependencies, never for unrelated siblings in the same tier. Dispatch on
dependency-satisfaction; keep tiers for reporting.

**"Failed" and "never ran" need opposite responses.** A failed task may be retryable; one blocked by
a failed prerequisite is not. Collapsing them into one status hides which is which, and name the
originating failure rather than the immediate parent.

**An answer assembled from whichever tasks succeeded is an answer to a different question.** Omit
failed and blocked work from the assembled output and name the gap.

**A cycle is caught; a missing edge is not.** A graph that sorts cleanly can still be the wrong
graph, and nothing in the scheduler can detect it.

## Code graphs and the engine boundary

**A requirement nothing provides is external, not missing.** Most requirements are the standard
library; treating an unresolved one as a broken extraction rejects nearly every real request.

**A dependency cycle in code is a consolidation, not an error.** Mutually recursive symbols have no
valid per-file ordering and always have a valid joint one.

**Signatures cross tier boundaries as exact text.** Drift turns on precise bytes, so a reconstructed
signature reintroduces the failure the tiering exists to remove. Contracts carry declarations
without bodies.

**Injection reduces drift; only parsing the output catches it.** Nothing constrains a model to
honour text in its prompt, and a parse costs microseconds against seconds of GPU time.

**"No conflicts found" is not "correct".** A cross-check can only contradict claims that were made.
Report coverage beside every verdict, and never name a method for the question the check cannot
answer.

**Exactly one goroutine may enter a llama context.** Wide fan-out belongs above the engine boundary;
concurrent callers race inside llama.cpp and also collapse one 48-sequence pass into 48
single-sequence ones.

**`LockOSThread` is about thread-local CUDA state, not about scheduler migration** — a goroutine is
already pinned for the duration of a cgo call. It belongs on the single engine owner, and without a
deferred unlock it leaks an OS thread per run.

## Tokens

**Character count is not a proxy for token count.** Flattening JSON syntax cuts 17.8% of characters
and 0.0% of tokens: BPE already carries merged tokens for `","` and `":"`. Count with a tokenizer
before believing a prefill saving.

**`json.Compact` is the whole win on a pretty-printed payload** — 35.6% of tokens, lossless,
reversible, one stdlib call. Anything more elaborate has to beat that baseline, not the pretty one.

**A saving applied per stream is multiplied by the stream count, and so is its enabling header.** Key
abbreviation saves ~32k tokens across 48 streams and the schema header that decodes it costs ~9.6k of
them back.

## KV prefix reuse

**`llama_memory_seq_cp` returns void and has three behaviours**: a silent no-op when `other` is set,
a metadata-only share when both sequences are in one stream, and a `GGML_ASSERT` that aborts the
process on a partial cross-stream copy. A prefix copy is always a partial range, so the unified-cache
precondition must be checked before the call, and the effect verified with `seq_pos_max` after it.

**A shared prefix is computed on tokens, never on text.** BPE merges across the boundary, so a common
string prefix does not imply a common token prefix — and cells copied for a text-derived prefix give a
sequence history it does not have, silently.

**A sequence must keep at least one token.** llama produces logits from the tokens given this pass, so
a prompt fully absorbed into a shared prefix has nothing to sample from.

## Cutting

**For Go, paragraph cutting already lands on declaration boundaries** — measured 4 of 5 interior cuts
for both strategies. Structural cutting's value is a boundary label that can be *trusted*, not a
better position.

**The overlap saving is against an unconditional window, not against prose cutting.** Measured on Go
source: conditional 0 tokens, unconditional 1,500 over 6 chunks. A window spent where the cut severed
nothing is pure duplicated prefill, multiplied by the stream count.

**A chunk's overlap need is decided by the PRECEDING chunk's cut** — the boundary they share. Reading
its own trailing cut repairs the far edge from the damage.

**Parser boundaries are preferences, not constraints.** A 900-line function has two; treating them as
constraints produces chunks that do not fit a stream.

## Capacity

**Per-stream context is a budgeting convention, not a limit.** Under `kv_unified`, llama.cpp sets
`n_ctx_seq = n_ctx` and a single sequence may occupy the whole pool. Streams need not be equal sized.

**Per-pass cost is Σ(depth) across active streams, and a deep stream does not pay for its own depth
— the herd does.** It receives one token per pass like every other stream while making every pass
more expensive for all of them, so its own tok/s looks unremarkable while throughput falls.

**Allocation and depth are different resources.** Allocation is a memory reservation that a unified
pool does not actually reserve; depth is what costs compute. A stream allowed 32k while sitting at 2k
costs 2k.

**Worst-case depth is known at admission: `len(tokenize(prompt)) + max_tokens`.** No prediction of
output length is required, because `max_tokens` is the bound. Fixed class sizes therefore make
worst-case Σ depth a boot-time constant.

**An install gate must be authored from measurement, not from a fit calculation.** 128 streams fit
the arithmetic on the 3090 and took the container with it; 72 fit and collapsed on one node while
running on another. A conservative computed gate fails the other way, refusing configurations that
work — and the user never learns they would have been fine.

**Apple Silicon reports `recommendedMaxWorkingSetSize`, not raw RAM** — roughly 70-75% of unified
memory, which is the right number for a fit check. But that ceiling is shared with everything else on
the machine, so unified memory wants a larger headroom margin than dedicated VRAM.

**`max_tokens` is a ceiling, not a prediction — never route on it.** Admission must use worst case
(`input + max_tokens`) because eviction is silent, but class assignment must use input alone.
Routing on the worst case sends `say "hello"` to the 32k class on the strength of a client default
nobody chose.

**A stream costs its depth, not its class.** Under a unified pool nothing is reserved, so a class is
a permission ceiling on growth rather than an allocation — which is what lets admission and routing
take different inputs.

**Do not predict with a model what arithmetic already knows exactly.** `len(tokens)` is free and
correct; a classifier over the same question can only be wrong.
