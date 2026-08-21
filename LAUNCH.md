# Launch checklist

Everything that must be true before `sideblank/llama-herd` is flipped public. Items are
grouped by their real deadline, which is not always "at launch".

---

## Before the first code commit

llama-herd is a **rewrite**, not a port — no code is copied in from anywhere. These items are
cheap now and expensive once history accumulates.

- [x] **Chain of title recorded** in [PROVENANCE.md](PROVENANCE.md). Sole authorship verified
      against the source history (54 commits, one author). The assignment and licence to Liminal
      Intelligence are both drafted but **unexecuted**, so the copyright sits personally with
      Benjamin Goldman and no counterparty consent is needed to release under Apache-2.0.
- [ ] **Sign and date PROVENANCE.md** when the first code commit lands. Do not backdate.
- [ ] **If the drafted IP assignment is later executed**, carve this project out or record the
      Apache-2.0 licence as an existing encumbrance in its schedule, or the schedule will be wrong.
- [ ] **Configure `.internal-patterns`** locally (gitignored) and set the `INTERNAL_PATTERNS`
      repository secret in GitHub, listing internal service names, hostnames, and infrastructure
      identifiers. Without it `scripts/check-leaks.sh` runs structural checks only. The list must
      never be committed — this repo becomes public.
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

## At launch

- [ ] **Final history scrub.** Public git history is permanent. Run `./scripts/check-leaks.sh`
      across every commit, not just the tip, and squash or rewrite if anything is found.
- [ ] **Enable branch protection** on `main`, requiring the `commit-policy` status check. This is
      free on public repos and is currently impossible (HTTP 403 on the free plan while private).
      It is what finally makes the policy blocking rather than advisory.
- [ ] Turn on **private vulnerability reporting** and **Dependabot alerts**.
- [ ] Set the repository **description and topics**; enable **Discussions** (the issue template
      links to it).
- [ ] Restrict merges to **squash only**, so the commit policy applies to a single tidy commit.
- [ ] Confirm `SECURITY.md`'s advisory link resolves once the repo is public.

## Releases

- [ ] Stay on `v0.x` until the engine API settles — SemVer allows breaking changes pre-1.0, and
      `1.0` is a stability promise.
- [ ] Do not cut a first release until the runtime can actually serve a token. The binary today
      does `version` and `doctor` only.
- [ ] Use prereleases (`v0.2.0-rc.1`) for anything touching the decode path — people run this on
      expensive hardware.
- [ ] Add a `CHANGELOG.md`. Conventional commit prefixes would let it be generated; `perf:` earns
      its place here, since throughput regressions are the bug class that hurts most.

## Before publishing models

- [ ] **Check every base model's licence.** Quantized builds inherit the upstream terms, which are
      not Apache-2.0 and may restrict redistribution or commercial use. Some permit both freely;
      others do not. Resolve per model before the Hugging Face org exists.
- [ ] Write a model card per build recording base model, licence, quantization, and the manifest
      settings (stream count, context, KV budget) for its target card.

## Ongoing

- [x] Go CI — `build.yml` builds and vets on Linux, macOS and Windows.
- [ ] Add `-race` unit tests to CI once the engine core exists.
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
