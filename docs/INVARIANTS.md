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
