# Contributing

## Setup

```bash
git clone https://github.com/sideblank/llama-herd.git
cd llama-herd
./scripts/setup-repo.sh     # required — wires hooks + commit identity
```

`core.hooksPath` is local git config and is not carried by `git clone`, so this step is required
once per clone. Without it the local hooks do not run.

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

## Licensing

llama-herd is licensed under the **Apache License 2.0** ([LICENSE](LICENSE)). New source files
should carry the standard header:

```
// Copyright <year> the llama-herd authors
// SPDX-License-Identifier: Apache-2.0
```

Contributions are covered by the [Contributor Licence Agreement](.github/CLA.md). **You keep the
copyright in your contribution** — the agreement grants the project a licence, not ownership, and
you stay free to use your own code anywhere else. Sign by adding a row to
[CLA-SIGNATURES.md](CLA-SIGNATURES.md) with your first pull request.

> The CLA is currently a **draft and not in force**; the signature check reports but does not block
> until the receiving entity is formed and counsel has reviewed it.

Third-party code carries its own licence. Record anything new in [NOTICE](NOTICE), and do not
introduce GPL or AGPL code into first-party sources — it is incompatible with Apache-2.0
distribution here.
