#!/usr/bin/env bash
# Check the four proofs of the paper: first the SHA-256 of every proof file, then the verifier on each proof
# against its statement in statements/, then the verifier's report against statements/expected.txt.
#
# Usage: ./VERIFY.sh <proofs_dir> [<ummon-verify binary>]
#   <proofs_dir> holds proof1_prose/, proof2_python/, proof3_dialogue/ and proof4_french/, each with
#   model_manifest.mdl2 and receipt_00.rec2, receipt_01.rec2, receipt_02.rec2.
#   The binary defaults to verifier/target/release/ummon-verify (README.md, "Build the verifier").
# Exit status 0 only if every hash matches and every proof is accepted with the expected result.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 <proofs_dir> [<ummon-verify binary>]" >&2; exit 2; }
proofs="$(cd "$1" && pwd)"
cli="${2:-$root/verifier/target/release/ummon-verify}"
[[ -x "$cli" ]] || { echo "verifier binary not found: $cli (build it first, see README.md)" >&2; exit 2; }

echo "== 1. SHA-256 of the proof files (SHA256SUMS)"
if command -v sha256sum >/dev/null 2>&1; then sum=(sha256sum); else sum=(shasum -a 256); fi
if ! (cd "$proofs" && "${sum[@]}" -c "$root/SHA256SUMS"); then
  echo "== RESULT: FAIL: a proof file is missing or its SHA-256 differs from SHA256SUMS"
  exit 1
fi

field() { awk -v k="$1" '$1 == k { print $2 }' "$2"; }
json_str() { sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p"; }
json_num() { sed -n "s/.*\"$1\":\([0-9][0-9]*\).*/\1/p"; }

echo "== 2. The verifier on each proof, against the statement in statements/"
failed=0
while read -r name receipts layers bytes final_chain; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  st="$root/statements/$name.txt"
  # A clean environment: the verifier also reads STWO_ML_* variables, which must not leak in from the caller.
  if out="$(env -i PATH="$PATH" HOME="${HOME:-/}" "$cli" model "$proofs/$name/model_manifest.mdl2" \
      --model-anchor "$(field model_anchor "$st")" \
      --initial-chain "$(field initial_chain "$st")" \
      --tokens "$(field tokens "$st")" \
      --gen-tokens "$(field gen_tokens "$st")" \
      --profile "$(field profile "$st")" 2>&1)"; then
    line="$(printf '%s\n' "$out" | grep '"accepted":true' | tail -1 || true)"
    got="$(printf '%s' "$line" | json_num num_receipts) $(printf '%s' "$line" | json_num total_layers)"
    got="$got $(printf '%s' "$line" | json_num total_receipt_bytes) $(printf '%s' "$line" | json_str final_chain)"
    if [[ "$got" == "$receipts $layers $bytes $final_chain" ]]; then
      echo "$name: accepted; $receipts receipts, $layers layers, $bytes receipt bytes, final chain $final_chain"
    else
      echo "$name: FAIL: accepted, but the report differs from statements/expected.txt: got '$got'"
      failed=1
    fi
  else
    rc=$?
    last="$(printf '%s\n' "$out" | tail -1)"
    reason="$(printf '%s' "$last" | json_str reason)"
    echo "$name: FAIL: rejected (exit $rc): ${reason:-$last}"
    failed=1
  fi
done < "$root/statements/expected.txt"

if [[ $failed -ne 0 ]]; then
  echo "== RESULT: FAIL"
  exit 1
fi
echo "== RESULT: all four proofs accepted"
