import CprsFormal.Basic

/-!
# Lemma C (commit step: the fixed pre-commitment weight dominates the post-commitment weight)
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- **Lemma C.** If the post-commitment state `(O, ν·1[O = V])` satisfies the relation, then so
does the pre-commitment state `(V, ν)`, with the same witness. -/
theorem lemmaC (D : Finset K) (k : ℕ) (β : ℚ) (ρ : List K) (ν : K → ℚ)
    (hν : ∀ x ∈ D, 0 ≤ ν x) (V O : K → K) (c E : K)
    (h : Rel D k β ρ (fun x => if O x = V x then ν x else 0) O c E) :
    Rel D k β ρ ν V c E := by
  obtain ⟨P, hP, hβ, hc⟩ := h
  refine ⟨P, hP, ?_, hc⟩
  refine le_trans hβ ?_
  unfold wagree
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  -- the post-commitment sum is a sum of `ν` over the points where `O = P` and `O = V`
  have key : ∑ x ∈ D.filter (fun x => O x = P.eval x), (if O x = V x then ν x else 0)
      = ∑ x ∈ (D.filter (fun x => O x = P.eval x)).filter (fun x => O x = V x), ν x :=
    (Finset.sum_filter (fun x => O x = V x) ν).symm
  rw [key]
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro x hx
    simp only [Finset.mem_filter] at hx ⊢
    obtain ⟨⟨hxD, hxP⟩, hxV⟩ := hx
    exact ⟨hxD, hxV.symm.trans hxP⟩
  · intro x hxD _
    exact hν x (Finset.mem_filter.mp hxD).1

#print axioms lemmaC

end Cprs
