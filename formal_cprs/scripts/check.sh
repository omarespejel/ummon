#!/usr/bin/env bash
# Build the CPRS formalization and gate on its axioms.
#
#   scripts/check.sh            # build + axiom gate
#
# Exit status 0 only if (1) the sources contain none of: sorry, admit, native_decide, unsafe, axiom;
# (2) `lake build` succeeds; (3) every theorem in namespace `Cprs` (and the ArkLib base theorem) depends
# on nothing beyond propext, Classical.choice, Quot.sound; (4) the sweep visited at least MIN_SWEEP theorems.
set -euo pipefail
cd "$(dirname "$0")/.."
MIN_SWEEP="${MIN_SWEEP:-624}"

command -v lake >/dev/null 2>&1 || {
  echo "lake not found. Install elan (https://github.com/leanprover/elan); the toolchain is pinned in lean-toolchain." >&2
  exit 2
}

echo "== 1/4 forbidden tokens in CprsFormal/*.lean"
if grep -nE '(^|[^A-Za-z_])(sorry|admit|native_decide|unsafe|axiom)([^A-Za-z_]|$)' CprsFormal/*.lean CprsFormal.lean; then
  echo "FAIL: forbidden token in the CPRS sources" >&2
  exit 1
fi
echo "ok"

echo "== 2/4 lake build"
lake build

echo "== 3/4 axiom gate"
out="$(lake env lean scripts/AxiomCheck.lean 2>&1)" || { echo "$out"; echo "FAIL: AxiomCheck did not elaborate" >&2; exit 1; }
echo "$out"
if echo "$out" | grep -q "sorryAx"; then
  echo "FAIL: sorryAx found" >&2
  exit 1
fi
bad="$(echo "$out" | grep "depends on axioms" \
  | grep -v -E "^'.+' depends on axioms: \[(propext|Classical\.choice|Quot\.sound)(, (propext|Classical\.choice|Quot\.sound))*\]$" || true)"
if [ -n "$bad" ]; then
  echo "FAIL: non-standard axioms:" >&2
  echo "$bad" >&2
  exit 1
fi
n="$(echo "$out" | sed -n 's/^CPRS axiom sweep: \([0-9][0-9]*\) theorems.*/\1/p' | head -1)"
if [ -z "$n" ] || [ "$n" -lt "$MIN_SWEEP" ]; then
  echo "FAIL: axiom sweep visited '${n:-0}' theorems (< $MIN_SWEEP)" >&2
  exit 1
fi
echo "ok: $n theorems swept"

echo "== 4/4 summary"
echo "PASS: CPRS Lean development builds; no sorry/axiom beyond [propext, Classical.choice, Quot.sound]."
