import CprsFormal.Basic

/-!
# Lemma W' (weighted upgrade of correlated agreement; any weight, threshold `|S| > |B|`)

A strengthening of Lemma W, the weighted form of correlated agreement that mirrors BCIKS20
Lemmas 7.5/7.6.
-/

open Polynomial Finset

namespace Cprs

variable {α K : Type*} [Field K] [DecidableEq K] [DecidableEq α]

/-- **Lemma W'.** If `u₀ + a u₁` and `v₀ + a v₁` have weighted agreement `≥ β` for every `a ∈ S`
and `|S|` exceeds the number of points where the *pair* `(u₀,u₁)` differs from `(v₀,v₁)`, then the
pair agrees on weighted mass `≥ β`. -/
theorem lemmaW' (D : Finset α) (ν : α → ℚ) (u0 u1 v0 v1 : α → K) (S : Finset K) (β : ℚ)
    (hS : ∀ a ∈ S, β ≤ wagree D ν (fun y => u0 y + a * u1 y) (fun y => v0 y + a * v1 y))
    (hcard : (D.filter (fun y => ¬ (u0 y = v0 y ∧ u1 y = v1 y))).card < S.card) :
    β ≤ (∑ y ∈ D.filter (fun y => u0 y = v0 y ∧ u1 y = v1 y), ν y) / D.card := by
  -- the "bad" points `B` where the pair differs, and the bad challenges `H` they can block
  set B : Finset α := D.filter (fun y => ¬ (u0 y = v0 y ∧ u1 y = v1 y)) with hB
  set H : Finset K := B.image (fun y => (v0 y - u0 y) / (u1 y - v1 y)) with hH
  have hHcard : H.card < S.card := lt_of_le_of_lt Finset.card_image_le hcard
  obtain ⟨a, haS, haH⟩ := Finset.exists_mem_notMem_of_card_lt_card hHcard
  have hsub := hS a haS
  -- for this challenge the agreement set of the two combined words is exactly the pair-agreement set
  have heq : D.filter (fun y => u0 y + a * u1 y = v0 y + a * v1 y)
      = D.filter (fun y => u0 y = v0 y ∧ u1 y = v1 y) := by
    ext y
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨hyD, hy⟩
      refine ⟨hyD, ?_⟩
      by_contra hne
      have hyB : y ∈ B := by
        rw [hB]
        exact Finset.mem_filter.mpr ⟨hyD, hne⟩
      by_cases h1 : u1 y = v1 y
      · apply hne
        refine ⟨?_, h1⟩
        rw [h1] at hy
        exact add_right_cancel hy
      · apply haH
        rw [hH]
        refine Finset.mem_image.mpr ⟨y, hyB, ?_⟩
        have hne' : u1 y - v1 y ≠ 0 := sub_ne_zero.mpr h1
        rw [div_eq_iff hne']
        linear_combination (-1 : K) * hy
    · rintro ⟨hyD, h0, h1⟩
      exact ⟨hyD, by rw [h0, h1]⟩
  have hw : wagree D ν (fun y => u0 y + a * u1 y) (fun y => v0 y + a * v1 y)
      = (∑ y ∈ D.filter (fun y => u0 y = v0 y ∧ u1 y = v1 y), ν y) / D.card := by
    unfold wagree
    rw [heq]
  rw [hw] at hsub
  exact hsub

#print axioms lemmaW'

end Cprs
