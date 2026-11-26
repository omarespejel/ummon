import CprsFormal.Basic

/-!
# Algebra of the even/odd split and of fibre sums for a squaring layer
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- A word is recovered from its even and odd parts: `u x = u_e(x²) + x u_o(x²)`. -/
theorem word_eq_even_add_odd (h2 : (2 : K) ≠ 0) (u : K → K) {x : K} (hx : x ≠ 0) :
    u x = evenPart u (x ^ 2) + x * oddPart u (x ^ 2) := by
  have hr : sqrtK (x ^ 2) ^ 2 = x ^ 2 := sqrtK_sq ⟨x, rfl⟩
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hr with h | h
  · unfold evenPart oddPart
    rw [h]
    field_simp
    ring
  · unfold evenPart oddPart
    rw [h, neg_neg]
    field_simp
    ring

/-- ... and `u (-x) = u_e(x²) - x u_o(x²)`. -/
theorem word_neg_eq_even_sub_odd (h2 : (2 : K) ≠ 0) (u : K → K) {x : K} (hx : x ≠ 0) :
    u (-x) = evenPart u (x ^ 2) - x * oddPart u (x ^ 2) := by
  have hr : sqrtK (x ^ 2) ^ 2 = x ^ 2 := sqrtK_sq ⟨x, rfl⟩
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hr with h | h
  · unfold evenPart oddPart
    rw [h]
    field_simp
    ring
  · unfold evenPart oddPart
    rw [h, neg_neg]
    field_simp
    ring

/-! ### Helper lemmas on the fibres of the squaring map for a `SqLayer` -/

/-- For `y ∈ D'` the chosen square root lies in `D`. -/
theorem aux_fold_sqrtK_mem {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ∈ D := by
  obtain ⟨x0, hx0, hx0y⟩ := h.sq_surj y hy
  have hr : sqrtK y ^ 2 = y := sqrtK_sq ⟨x0, hx0y⟩
  have hsq : sqrtK y ^ 2 = x0 ^ 2 := by rw [hr, hx0y]
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp hsq with h' | h'
  · rw [h']
    exact hx0
  · rw [h']
    exact h.neg_mem x0 hx0

/-- For `y ∈ D'` the chosen square root really is a square root. -/
theorem aux_fold_sqrtK_sq {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ^ 2 = y := by
  obtain ⟨x0, _, hx0y⟩ := h.sq_surj y hy
  exact sqrtK_sq ⟨x0, hx0y⟩

/-- For `y ∈ D'` the two square roots `√y` and `-√y` are distinct. -/
theorem aux_fold_sqrtK_ne_neg {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ≠ -sqrtK y := by
  intro heq
  have hmem : sqrtK y ∈ D := aux_fold_sqrtK_mem h hy
  have h2 : (2 : K) * sqrtK y = 0 := by linear_combination heq
  have h0 : sqrtK y = 0 := by
    rcases mul_eq_zero.mp h2 with h2' | h2'
    · exact absurd h2' h.two_ne_zero
    · exact h2'
  rw [h0] at hmem
  exact h.zero_notMem hmem

/-- The fibre of the squaring map over `y ∈ D'` is `{√y, -√y}`. -/
theorem aux_fold_fibre {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    D.filter (fun x => x ^ 2 = y) = {sqrtK y, -sqrtK y} := by
  ext x
  simp only [Finset.mem_filter, Finset.mem_insert, Finset.mem_singleton]
  constructor
  · rintro ⟨_, hxy⟩
    have hsq : x ^ 2 = sqrtK y ^ 2 := by rw [hxy, aux_fold_sqrtK_sq h hy]
    exact sq_eq_sq_iff_eq_or_eq_neg.mp hsq
  · rintro (hx | hx)
    · refine ⟨?_, ?_⟩
      · rw [hx]
        exact aux_fold_sqrtK_mem h hy
      · rw [hx]
        exact aux_fold_sqrtK_sq h hy
    · refine ⟨?_, ?_⟩
      · rw [hx]
        exact h.neg_mem _ (aux_fold_sqrtK_mem h hy)
      · rw [hx, neg_sq]
        exact aux_fold_sqrtK_sq h hy

/-- Every fibre of the squaring map over `D'` has exactly two points. -/
theorem aux_fold_fibre_card {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    (D.filter (fun x => x ^ 2 = y)).card = 2 := by
  rw [aux_fold_fibre h hy]
  exact Finset.card_pair (aux_fold_sqrtK_ne_neg h hy)

/-- A squaring layer has exactly two points over every point of `D'`: `|D| = 2 |D'|`. -/
theorem SqLayer.card_eq {D D' : Finset K} (h : SqLayer D D') : D.card = 2 * D'.card := by
  have hmaps : (D : Set K).MapsTo (fun x : K => x ^ 2) (D' : Set K) :=
    fun x hx => h.sq_mem x hx
  have key := Finset.card_eq_sum_card_fiberwise hmaps
  rw [key]
  calc ∑ b ∈ D', (D.filter (fun a => a ^ 2 = b)).card = ∑ b ∈ D', 2 :=
        Finset.sum_congr rfl (fun y hy => aux_fold_fibre_card h hy)
    _ = 2 * D'.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- Fibre sum: for `y ∈ D'` the fibre of the squaring map in `D` is `{√y, -√y}`, hence
`Σ_{x ∈ D, x² ∈ A} f x = Σ_{y ∈ A} (f (√y) + f (-√y))` for every `A ⊆ D'`. -/
theorem SqLayer.sum_fibre {D D' : Finset K} (h : SqLayer D D') (A : Finset K) (hA : A ⊆ D')
    (f : K → ℚ) :
    ∑ x ∈ D.filter (fun x => x ^ 2 ∈ A), f x = ∑ y ∈ A, (f (sqrtK y) + f (-sqrtK y)) := by
  have hmaps : ∀ x ∈ D.filter (fun x => x ^ 2 ∈ A), x ^ 2 ∈ A := by
    intro x hx
    exact (Finset.mem_filter.mp hx).2
  rw [← Finset.sum_fiberwise_of_maps_to hmaps f]
  apply Finset.sum_congr rfl
  intro y hy
  have hyD' : y ∈ D' := hA hy
  have hfib : (D.filter (fun x => x ^ 2 ∈ A)).filter (fun x => x ^ 2 = y)
      = D.filter (fun x => x ^ 2 = y) := by
    ext x
    simp only [Finset.mem_filter]
    constructor
    · rintro ⟨⟨hxD, _⟩, hxy⟩
      exact ⟨hxD, hxy⟩
    · rintro ⟨hxD, hxy⟩
      exact ⟨⟨hxD, by rw [hxy]; exact hy⟩, hxy⟩
  rw [hfib, aux_fold_fibre h hyD', Finset.sum_pair (aux_fold_sqrtK_ne_neg h hyD')]

#print axioms word_eq_even_add_odd
#print axioms word_neg_eq_even_sub_odd
#print axioms SqLayer.card_eq
#print axioms SqLayer.sum_fibre

end Cprs
