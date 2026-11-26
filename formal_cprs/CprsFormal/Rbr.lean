import Mathlib

/-!
# Round-by-round soundness ⇒ interactive soundness (counting form)

A *good* predicate on challenge prefixes is false at the empty prefix; along every doomed prefix
at most a fraction `ε t` of the next challenges make it true.  Then the fraction of length-`t`
challenge vectors for which it is true is at most `Σ_{u<t} ε u`.  This is the union bound
behind "round-by-round soundness error `ε` per round ⇒ interactive soundness error `Σ ε`".
-/

open Finset

namespace Cprs

namespace Aux

/-- Splitting off the last coordinate: `(Fin t → Ω) × Ω ≃ (Fin (t+1) → Ω)` via `Fin.snoc`. -/
def aux_rbr_snocEquiv (Ω : Type*) (t : ℕ) : (Fin t → Ω) × Ω ≃ (Fin (t + 1) → Ω) where
  toFun p := Fin.snoc (α := fun _ => Ω) p.1 p.2
  invFun f := (Fin.init f, f (Fin.last t))
  left_inv p := by simp
  right_inv f := by simp

/-- Counting the filter of a predicate on `(t+1)`-tuples by first fixing the length-`t` prefix. -/
theorem aux_rbr_card_filter_succ {Ω : Type*} [Fintype Ω] (t : ℕ)
    (P : (Fin (t + 1) → Ω) → Prop) [DecidablePred P] :
    (Finset.univ.filter P).card =
      ∑ pre : Fin t → Ω,
        (Finset.univ.filter (fun x : Ω => P (Fin.snoc (α := fun _ => Ω) pre x))).card := by
  simp only [Finset.card_filter]
  rw [← Fintype.sum_prod_type' (fun (pre : Fin t → Ω) (x : Ω) =>
    if P (Fin.snoc (α := fun _ => Ω) pre x) then 1 else 0)]
  exact (Fintype.sum_equiv (aux_rbr_snocEquiv Ω t) _ _ (fun p => rfl)).symm

end Aux

theorem rbr_prob {Ω : Type*} [Fintype Ω] [Nonempty Ω] (m : ℕ)
    (Good : (t : ℕ) → (Fin t → Ω) → Prop) [∀ t pre, Decidable (Good t pre)] (ε : ℕ → ℚ)
    (hε : ∀ t, 0 ≤ ε t)
    (h0 : ∀ pre : Fin 0 → Ω, ¬ Good 0 pre)
    (hstep : ∀ t, t < m → ∀ pre : Fin t → Ω, ¬ Good t pre →
        ((Finset.univ.filter
            (fun x : Ω => Good (t + 1) (Fin.snoc (α := fun _ => Ω) pre x))).card : ℚ)
          ≤ ε t * Fintype.card Ω) :
    ∀ t, t ≤ m → ((Finset.univ.filter (fun pre : Fin t → Ω => Good t pre)).card : ℚ)
        ≤ (∑ u ∈ Finset.range t, ε u) * (Fintype.card Ω : ℚ) ^ t := by
  -- one-step recursion for the number of good length-`(t+1)` vectors
  have hrec : ∀ t, t < m →
      ((Finset.univ.filter (fun pre : Fin (t + 1) → Ω => Good (t + 1) pre)).card : ℚ)
        ≤ ((Finset.univ.filter (fun pre : Fin t → Ω => Good t pre)).card : ℚ)
            * (Fintype.card Ω : ℚ) + ε t * (Fintype.card Ω : ℚ) ^ (t + 1) := by
    intro t ht
    rw [Aux.aux_rbr_card_filter_succ t (Good (t + 1))]
    push_cast
    have hpt : ∀ pre : Fin t → Ω,
        ((Finset.univ.filter
            (fun x : Ω => Good (t + 1) (Fin.snoc (α := fun _ => Ω) pre x))).card : ℚ)
          ≤ (if Good t pre then (Fintype.card Ω : ℚ) else 0) + ε t * (Fintype.card Ω : ℚ) := by
      intro pre
      by_cases h : Good t pre
      · rw [ite_eq_left h]
        have h1 : ((Finset.univ.filter
            (fun x : Ω => Good (t + 1) (Fin.snoc (α := fun _ => Ω) pre x))).card : ℚ)
              ≤ (Fintype.card Ω : ℚ) := by
          have := Finset.card_filter_le (Finset.univ : Finset Ω)
            (fun x : Ω => Good (t + 1) (Fin.snoc (α := fun _ => Ω) pre x))
          rw [Finset.card_univ] at this
          exact_mod_cast this
        have h2 : 0 ≤ ε t * (Fintype.card Ω : ℚ) := mul_nonneg (hε t) (Nat.cast_nonneg _)
        linarith
      · rw [ite_eq_right h]
        have := hstep t ht pre h
        linarith
    have e1 : ∑ pre : Fin t → Ω, (if Good t pre then (Fintype.card Ω : ℚ) else 0)
        = ((Finset.univ.filter (fun pre : Fin t → Ω => Good t pre)).card : ℚ)
            * (Fintype.card Ω : ℚ) := by
      rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    have e2 : ∑ _pre : Fin t → Ω, ε t * (Fintype.card Ω : ℚ)
        = (Fintype.card Ω : ℚ) ^ t * (ε t * (Fintype.card Ω : ℚ)) := by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, nsmul_eq_mul]
      push_cast
      ring
    calc _ ≤ ∑ pre : Fin t → Ω,
          ((if Good t pre then (Fintype.card Ω : ℚ) else 0) + ε t * (Fintype.card Ω : ℚ)) :=
          Finset.sum_le_sum (fun pre _ => hpt pre)
      _ = _ := by
          rw [Finset.sum_add_distrib, e1, e2]
          ring
  intro t
  induction t with
  | zero =>
    intro _
    have hempty : (Finset.univ.filter (fun pre : Fin 0 → Ω => Good 0 pre)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro pre _
      exact h0 pre
    rw [hempty]
    simp
  | succ t ih =>
    intro ht
    have ht' : t < m := Nat.lt_of_succ_le ht
    have ih' := ih (Nat.le_of_lt ht')
    have hs := hrec t ht'
    have hc : (0 : ℚ) ≤ (Fintype.card Ω : ℚ) := Nat.cast_nonneg _
    have h3 := mul_le_mul_of_nonneg_right ih' hc
    rw [Finset.sum_range_succ]
    have h4 : ((∑ u ∈ Finset.range t, ε u) + ε t) * (Fintype.card Ω : ℚ) ^ (t + 1)
        = (∑ u ∈ Finset.range t, ε u) * (Fintype.card Ω : ℚ) ^ t * (Fintype.card Ω : ℚ)
          + ε t * (Fintype.card Ω : ℚ) ^ (t + 1) := by ring
    linarith

#print axioms rbr_prob

end Cprs
