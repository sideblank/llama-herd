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
