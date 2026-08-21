#!/usr/bin/env bash
# One-time local setup. Wires the tracked hooks and the project commit identity.
set -euo pipefail
cd "$(dirname "$0")/.."

git config core.hooksPath .githooks
git config user.name  "Benjamin Goldman"
git config user.email "benjamin.goldman@gmail.com"

echo "hooksPath : $(git config core.hooksPath)"
echo "identity  : $(git config user.name) <$(git config user.email)>"
echo "ok"
