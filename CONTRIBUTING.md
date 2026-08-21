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

- Commits carry the project identity only: `Benjamin Goldman <benjamin.goldman@gmail.com>`.
- **One author per commit. No `Co-Authored-By:` trailers of any kind.**
- No tool or assistant attribution: no session links, no "Generated with ..." lines, no
  third-party addresses or URLs. The message describes the change, nothing else.
- Agent-tooling files (`CLAUDE.md`, `.claude/`, `.mcp.json`, `.cursorrules`, ...) are gitignored
  and must never be tracked.

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
