# Contributing

## Setup

```bash
git clone https://github.com/sideblank/llama-herd.git
cd llama-herd
./scripts/setup-repo.sh     # required — wires hooks + commit identity
```

`core.hooksPath` is local git config and is not carried by `git clone`, so this step is required
once per clone. Without it the local hooks do not run.

## Local scratch

Three directories are gitignored and never committed:

| Directory | For |
|---|---|
| `testing/` | provider-specific deployment and test objects — container group definitions, GPU provider configs, node bootstrap, deploy manifests |
| `scripts/local/` | one-off drivers and helpers. The rest of `scripts/` is tracked and runs in CI |
| `dev/` | anything else local |

`testing/` matters most: this repository is going public and its history is permanent, so
objects naming infrastructure must never enter the tree. Benchmark *methodology* and
*results* are provider-neutral and belong in `docs/`; the machinery that produced them does
not.

## Before changing the engine

Read [docs/INVARIANTS.md](docs/INVARIANTS.md). It records the behaviours that fail silently —
where the system reports success and produces wrong numbers or wrong output. Several are not
discoverable from the code alone.

## Commit rules

The policy is an **allow-list**: anything not explicitly permitted is rejected. Nothing in the
checker names a vendor, so a new tool needs no new rule.

| Aspect | Allowed |
|--------|---------|
| Author + committer | `Benjamin Goldman <benjamin.goldman@gmail.com>` — nothing else |
| Trailers | `Signed-off-by`, `Refs`, `Fixes`, `Closes` — every other key is rejected |
| Addresses in the message | `benjamin.goldman@gmail.com` only |
| Link hosts in the message | `github.com`, `huggingface.co` only |

`Co-Authored-By:` is simply not on the trailer list, so it is rejected — as is any assistant
session trailer, generated-by line, or third-party address or link, whatever tool produced it.

Tracked files must obey `.gitignore`. That file is the one place tool names appear; the checker
just enforces that nothing it excludes is ever committed.

To widen a list, edit the policy block at the top of `scripts/check-commits.sh`.

## Enforcement

Three layers, all running the same check — `scripts/check-commits.sh`:

| Layer | Hook / job | Catches |
|-------|-----------|---------|
| Commit time | `.githooks/pre-commit`, `.githooks/commit-msg` | wrong identity, banned trailers |
| Push time | `.githooks/pre-push` | every commit in the push range |
| CI | `.github/workflows/commit-policy.yml` | the same, on push and PR |

Run it yourself at any time:

```bash
./scripts/check-commits.sh              # whole history
./scripts/check-commits.sh main..HEAD   # a range
```

Do not bypass the hooks with `--no-verify`.

> **Note:** GitHub branch protection / rulesets are not available on this repo's current plan
> (free org, private repo), so CI reports violations but cannot yet *block* a push. When the repo
> goes public — or the org moves to a paid plan — mark `commit-policy` a required status check on
> `main` to make enforcement absolute.

## Licensing and contributions

llama-herd is licensed under the **Apache License 2.0** ([LICENSE](LICENSE)). New source files
should carry the standard header:

```
// Copyright <year> the llama-herd authors
// SPDX-License-Identifier: Apache-2.0
```

### What contributing means

**You keep the copyright in your contribution.** There is no CLA and no copyright assignment. You
remain free to use your own code anywhere else, for any purpose.

By submitting a contribution you license it to the project under Apache-2.0. This follows from
Apache-2.0 §5: a contribution intentionally submitted for inclusion is under the terms of the
licence, without any additional terms. Three consequences worth stating plainly:

- **The grant is irrevocable.** Once merged, a contribution cannot be un-licensed, withdrawn, or
  removed on demand.
- **Contributing conveys no ownership of the project.** Each contributor holds copyright in their
  own contribution only. The project is a collective work; no contributor acquires a stake in the
  codebase as a whole, or any right to control it.
- **Anyone may take this code and use it elsewhere**, including commercially and in closed-source
  products, because Apache-2.0 permits it. That applies to everyone equally.

### Sign your work (DCO)

Every commit must carry a `Signed-off-by:` line matching its author. This certifies you have the
right to submit the code — see [`.github/DCO`](.github/DCO) for the text you are certifying.

```bash
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your@email>
```

Forgot it? `git commit --amend -s`, or for several commits
`git rebase --signoff <base>`. CI checks every commit in the pull request.

Third-party code carries its own licence. Record anything new in [NOTICE](NOTICE), and do not
introduce GPL or AGPL code into first-party sources — it is incompatible with Apache-2.0
distribution here.
