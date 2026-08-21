# Launch checklist

Everything that must be true before `sideblank/llama-herd` is flipped public. Items are
grouped by their real deadline, which is not always "at launch".

---

## Before the code import

The moment engine code is copied in, decisions become facts recorded in history. These are cheap
now and expensive later.

- [x] **Chain of title recorded** in [PROVENANCE.md](PROVENANCE.md). Sole authorship verified
      against the source history (54 commits, one author). The assignment and licence to Liminal
      Intelligence are both drafted but **unexecuted**, so the copyright sits personally with
      Benjamin Goldman and no counterparty consent is needed to release under Apache-2.0.
- [ ] **Sign and date PROVENANCE.md** when the first import commit lands. Do not backdate.
- [ ] **If the drafted IP assignment is later executed**, record the Apache-2.0 release as an
      existing encumbrance in its schedule of assigned property, or the schedule will be wrong.
- [ ] **Configure `.internal-patterns`** locally (gitignored) and set the `INTERNAL_PATTERNS`
      repository secret in GitHub, listing internal service names, hostnames, and infrastructure
      identifiers. Without it `scripts/check-leaks.sh` runs structural checks only. The list must
      never be committed — this repo becomes public.
- [ ] **Strip the private-side coupling.** `internal/fleet` is entirely queue/storage integration
      with the private platform; leave it behind. Roughly 50 references to internal services span
      nearly every file of the source engine, including `engine.go` and `registry.go`.
- [ ] **Add `NOTICE` entries** for anything vendored during the import.

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

## Before publishing models

- [ ] **Check every base model's licence.** Quantized builds inherit the upstream terms, which are
      not Apache-2.0 and may restrict redistribution or commercial use. Some permit both freely;
      others do not. Resolve per model before the Hugging Face org exists.
- [ ] Write a model card per build recording base model, licence, quantization, and the manifest
      settings (stream count, context, KV budget) for its target card.

## Ongoing

- [ ] Go CI — build, vet, test, `-race` — added with the first code import.
- [ ] `CODEOWNERS` once there is more than one maintainer.
- [ ] A release and versioning scheme before the first tag.
