-- The axioms of the nine main theorems of formal_cprs. Run from formal_cprs/ after the build:
--   lake env lean <this file>
import CprsFormal

#print axioms Cprs.Chain.chain_soundness
#print axioms Cprs.Chain.chain_soundness_moore
#print axioms Cprs.PhaseBInst.soundness_unconditional
#print axioms Cprs.corrAgree_of_arklib
#print axioms ProximityGap.RS_jointAgreement_of_goodCoeffs_card_gt
#print axioms Cprs.Packing.conjugate_identity_explicit
#print axioms Cprs.sumcheck_sound
#print axioms Cprs.Toy.nonvacuity_false
#print axioms Cprs.Toy.nonvacuity_true
