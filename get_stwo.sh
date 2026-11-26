#!/usr/bin/env bash
# Put the verifier's one non-crates.io dependency in ./stwo: the stwo library at commit 69e6f9a0, which is the
# upstream commit f0d79b0f plus 23 commits of ours. Those 23 commits travel in third_party/stwo-69e6f9a0.bundle
# (a git bundle, 58 KB); the upstream commit comes from the upstream repository.
#
# Usage: ./get_stwo.sh [<upstream stwo repository URL or path>]
#   default: https://github.com/starkware-libs/stwo.git
# Needs git. Fetches one upstream commit (shallow), applies the bundle, and checks the resulting commit and tree.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${1:-https://github.com/starkware-libs/stwo.git}"
base=f0d79b0fad440dcb0aaf1e20470fdbb37993ea2a
head=69e6f9a03199d0a17add4855e0da5f3c1a48c785
tree=4f3b119deb0bebd94f32a8b7079069a6f7fc2e03
bundle="$root/third_party/stwo-69e6f9a0.bundle"
bundle_sha256=1701a425f0872dbb9d15a38447dc459d315f3df93d4194478c9c541688d8a573
dest="$root/stwo"

[[ ! -e "$dest" ]] || { echo "$dest already exists; remove it first" >&2; exit 2; }
if command -v sha256sum >/dev/null 2>&1; then got="$(sha256sum "$bundle")"; else got="$(shasum -a 256 "$bundle")"; fi
[[ "${got%% *}" == "$bundle_sha256" ]] || { echo "bundle SHA-256 mismatch: ${got%% *}" >&2; exit 1; }

git init -q "$dest"
git -C "$dest" fetch -q --depth 1 "$upstream" "$base"
git -C "$dest" bundle verify -q "$bundle"
git -C "$dest" fetch -q "$bundle" HEAD
[[ "$(git -C "$dest" rev-parse FETCH_HEAD)" == "$head" ]] || { echo "bundle head is not $head" >&2; exit 1; }
[[ "$(git -C "$dest" rev-parse "$head^{tree}")" == "$tree" ]] || { echo "tree of $head is not $tree" >&2; exit 1; }
git -C "$dest" -c advice.detachedHead=false checkout -q --detach "$head"
echo "stwo at $head (tree $tree): upstream $base plus $(git -C "$dest" rev-list --count "$base..$head") commits:"
git -C "$dest" log --reverse --format='  %h %s' "$base..$head"
