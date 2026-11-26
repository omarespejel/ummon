import CprsFormal

/-!
# Axiom gate for the CPRS formalization

`#cprs_axiom_sweep` visits every theorem whose name lies in the `Cprs` namespace (including the
helper lemmas) and fails unless each one depends only on `propext`, `Classical.choice` and
`Quot.sound`.  The base lemma imported from ArkLib is checked explicitly as well.
-/

open Lean Elab Command

elab "#cprs_axiom_sweep" : command => do
  let env ← getEnv
  let std : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  -- `fold` visits both the imported constants and those declared in the current file.
  let cands : Array Name := env.constants.fold (fun acc n ci =>
    if (`Cprs).isPrefixOf n && ci.isTheorem then acc.push n else acc) #[]
  let mut bad : Array Name := #[]
  for n in cands do
    let axs ← collectAxioms n
    if axs.any (fun a => !std.contains a) then bad := bad.push n
  logInfo m!"CPRS axiom sweep: {cands.size} theorems under `Cprs`; with non-standard axioms: {bad}"
  if !bad.isEmpty then throwError "non-standard axioms found: {bad}"

#cprs_axiom_sweep

#print axioms Cprs.Chain.chain_soundness
#print axioms Cprs.Chain.chain_soundness_moore

#print axioms ProximityGap.RS_jointAgreement_of_goodCoeffs_card_gt
#print axioms Cprs.corrAgree_of_arklib
#print axioms Cprs.lemmaW'
#print axioms Cprs.lemmaC
#print axioms Cprs.lemmaF'
#print axioms Cprs.rbr_prob
#print axioms Cprs.PhaseBInst.good_step
#print axioms Cprs.PhaseBInst.soundness
#print axioms Cprs.PhaseBInst.soundness_unconditional
#print axioms Cprs.Packing.conjugate_identity
#print axioms Cprs.gamma_roots
#print axioms Cprs.sumcheck_roots
#print axioms Cprs.mu_step
#print axioms Cprs.PhaseBInst.ofRoot_wf
