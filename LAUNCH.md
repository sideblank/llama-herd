# Launch checklist

Everything that had to be true before `sideblank/llama-herd` went public, which it did on
2026-09-04, and what remains after. Items are grouped by their real deadline, which is not always
"at launch".

---

## Before the first code commit

llama-herd is a **rewrite**, not a port — no code is copied in from anywhere. These items are
cheap now and expensive once history accumulates.

- [x] **Chain of title recorded** in [PROVENANCE.md](PROVENANCE.md). Sole authorship verified
      against the source history (54 commits, one author). The assignment and licence to Liminal
      Intelligence are both drafted but **unexecuted**, so the copyright sits personally with
      Benjamin Goldman and no counterparty consent is needed to release under Apache-2.0.
- [x] **Sign and date PROVENANCE.md** when the first code commit lands. Do not backdate. Signed
      2026-09-04.
- [ ] **If the drafted IP assignment is later executed**, carve this project out or record the
      Apache-2.0 licence as an existing encumbrance in its schedule, or the schedule will be wrong.
- [x] **Set the `INTERNAL_PATTERNS` repository secret in GitHub.** Done 2026-09-04 from the
      local `.internal-patterns` file, so the name-based half of the leak scan now runs in CI as
      well as locally. The two must be kept in step: a pattern added to one and not the other
      is a guard that is live in only one place.
- [ ] Keep provider-specific test objects in the gitignored `testing/` directory. Two layers back
      this up: `.gitignore` keeps them out of the tree, and `check-leaks.sh` scans tracked files
      for infrastructure names.
- [ ] **Keep platform coupling out by construction.** Job-queue pull workers, storage and dispatch
      integration, and anything else specific to the private platform must never be written here —
      not in code and not in history. This is the reason for the rewrite.
- [ ] **Decide the assignment sequencing.** The drafted IP assignment auto-assigns future works
      that relate to or improve upon the assigned property. Either write this project before that
      assignment is executed, or carve it out explicitly. See
      [PROVENANCE.md](PROVENANCE.md#sequencing-the-future-works-clause).
- [ ] **Add `NOTICE` entries** for anything vendored.

## Before accepting outside contributions

- [ ] **Set `SOLO_IDENTITY=0`** in `scripts/check-commits.sh`. While it is `1`, every commit must
      be authored by `benjamin.goldman@gmail.com` — correct for a solo private repo, but it rejects
      every external contributor.
- [ ] Decide whether `Co-authored-by` should stay banned. It is currently rejected by the trailer
      allow-list, which also blocks GitHub's pair-programming attribution.

## CI cost

Actions minutes are the constraint while the repo is private; they are free and unlimited
once it is public.

- [x] macOS removed from the per-change build. It bills at **10x** — 180 macOS minutes
      consumed 1,800 of 2,000 included minutes and blocked every workflow, including the
      image build.
- [ ] Decide whether darwin release binaries are worth ~200 included minutes per tag. They
      require macOS runners because cgo cannot cross-compile. The alternative is to ship
      Linux and Windows and let Mac users build from source.
- [ ] Raise the org spending limit above its $0 default, or go public. At $0 a blocked job
      reports "recent account payments have failed or your spending limit needs to be
      increased", which reads like a payment problem and is not one.

## At launch

- [x] **Final history scrub.** Public git history is permanent. Run `./scripts/check-leaks.sh`
      across every commit, not just the tip, and squash or rewrite if anything is found. Done 2026-09-04: every commit's tree, every message and every author scanned; history rewritten where it failed.
- [x] **Enable branch protection** on `main`, requiring the `commit-policy` status check. Done 2026-09-04. This is
      free on public repos and is currently impossible (HTTP 403 on the free plan while private).
      It is what finally makes the policy blocking rather than advisory.
- [x] Turn on **private vulnerability reporting** and **Dependabot alerts**. Both on 2026-09-04.
- [x] Set the repository **description and topics**; enable **Discussions** (the issue template
      links to it). Done 2026-09-04.
- [x] Restrict merges to **squash only**, so the commit policy applies to a single tidy commit. Done 2026-09-04.
- [x] Confirm `SECURITY.md`'s advisory link resolves once the repo is public. Confirmed 2026-09-04.

## Releases

- [ ] Stay on `v0.x` until the engine API settles — SemVer allows breaking changes pre-1.0, and
      `1.0` is a stability promise.
- [x] Do not cut a first release until the runtime can actually serve a token. It serves, benches,
      sweeps and holds a standby port; `version` and `doctor` are the two subcommands that need
      no model.
- [ ] Use prereleases (`v0.2.0-rc.1`) for anything touching the decode path — people run this on
      expensive hardware.
- [ ] Add a `CHANGELOG.md`. Conventional commit prefixes would let it be generated; `perf:` earns
      its place here, since throughput regressions are the bug class that hurts most.

## Engine milestones

See [docs/ROADMAP.md](docs/ROADMAP.md) for the reasoning behind each.

- [x] Engine core: decode loop, slot table, continuous batching, admission control
- [x] Chat-completions API with streaming — the integration surface every agent uses
- [x] Benchmark harness — `llama-herd bench`, reproducible and self-describing
- [x] **Run it on a real card** and publish results to `docs/results/`. Done on an RTX 3090:
      `docs/results/3090.md`, 728.71 tok/s aggregate at 48 streams. The 4090 and 5090 remain
      unmeasured.
- [x] KV quantization — `kv_type_k` / `kv_type_v`, a sweep axis, reported on `/v1/info`,
      measured at depth in `docs/results/3090.md`
- [ ] A real KV budget, not a fixed slot count — today `admit_context` restores the reservation a
      unified pool removes; admission does not yet read live pool occupancy
- [x] Verify llama.cpp MTP/speculative support against a quant that retains `nextn` tensors —
      it drafts (57% acceptance at `max_draft=2`) and was net-negative on throughput; see
      `docs/ROADMAP.md` §7b for the caveat
- [ ] Multi-GPU placement and capacity-aware routing

## Model selection

- [ ] **Verify MTP tensors, do not trust the model card.** A file can declare next-token
      prediction layers in metadata while carrying none of the tensors — the declaration
      survives quantization even when the weights do not. `llama-herd inspect` reads the
      declaration cheaply; only a real load with the layers enabled proves they are there.
- [ ] Note that publishers ship MTP unevenly: the most widely used GGUF repo for one model
      offers thirty quantizations and MTP for exactly one of them, as a separate module.
      That unevenness is the argument for building our own.

## Before publishing models

Not part of the first release, which ships the engine only. Kept for the version that adds
model builds.

- [ ] **Check every base model's licence.** Quantized builds inherit the upstream terms, which are
      not Apache-2.0 and may restrict redistribution or commercial use. Some permit both freely;
      others do not. Resolve per model before the Hugging Face org exists.
- [ ] Write a model card per build recording base model, licence, quantization, KV precision,
      **whether MTP tensors are retained**, and the manifest settings (stream count, context, KV
      budget) for its target card.
- [ ] Resolve licences per line — Qwen, GLM, Nemotron each carry their own terms.
- [ ] Decide the Hugging Face org name and the Cloudflare delivery path.

## Ongoing

- [x] Go CI — `build.yml` builds and vets on Linux and Windows. macOS was removed from the
      per-change build for cost (above).
- [x] `-race` unit tests run in CI on both platforms.
- [x] Per-request sampling: temperature and friends are honoured per request, layered over
      the model's configured defaults.
- [x] **`LLAMA_CPP_REF` pinned** to a llama.cpp release tag in both workflows. Bump deliberately
      and note the new ref in the release notes.
- [ ] **Confirm build provenance attestation works on this repo.** `release.yml` attests the
      artifacts; attestations are free for public repos but may need a paid plan while private.
- [ ] Confirm ARM runners are available for the release matrix. `ubuntu-24.04-arm` is free for
      public repos; on a private repo it may need a paid plan.
- [ ] Decide GPU backend coverage for releases. The matrix ships **CPU-only** builds today; CUDA
      and Metal variants multiply the matrix and CUDA needs a toolkit install step.
- [ ] Consider macOS codesigning and notarization — unsigned binaries are blocked on first run.
- [ ] `CODEOWNERS` once there is more than one maintainer.
- [ ] A release and versioning scheme before the first tag.
