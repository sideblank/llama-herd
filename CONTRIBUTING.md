# Contributing

## Setup

```bash
git clone https://github.com/sideblank/llama-herd.git
cd llama-herd
./scripts/setup-repo.sh     # required — wires hooks + commit identity
```

`core.hooksPath` is local git config and is not carried by `git clone`, so this step is required
once per clone. Without it the commit hooks do not run.

## Commit rules

- Commits carry the project identity only: `Benjamin Goldman <benjamin.goldman@gmail.com>`.
- **One author per commit. No `Co-Authored-By:` trailers of any kind.**
- No tool or assistant attribution: no session links, no "Generated with ..." lines, no
  third-party addresses or URLs. The message describes the change, nothing else.

Enforced by `.githooks/{pre-commit,commit-msg}`. Do not bypass with `--no-verify`.
