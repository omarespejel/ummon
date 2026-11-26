import CprsFormal.Basic

/-!
# Multilinear twin `P~`: linearity and the even/odd lift
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

theorem mleAux_add (ρ : List K) (c d : ℕ → K) :
    mleAux ρ (fun i => c i + d i) = mleAux ρ c + mleAux ρ d := by
  induction ρ generalizing c d with
  | nil => simp [mleAux]
  | cons r ρ ih =>
    simp only [mleAux]
    rw [ih (fun i => c (2 * i)) (fun i => d (2 * i)),
      ih (fun i => c (2 * i + 1)) (fun i => d (2 * i + 1))]
    ring

theorem mleAux_const_mul (ρ : List K) (a : K) (c : ℕ → K) :
    mleAux ρ (fun i => a * c i) = a * mleAux ρ c := by
  induction ρ generalizing c with
  | nil => simp [mleAux]
  | cons r ρ ih =>
    simp only [mleAux]
    rw [ih (fun i => c (2 * i)), ih (fun i => c (2 * i + 1))]
    ring

theorem mle_add (P Q : K[X]) (ρ : List K) : mle (P + Q) ρ = mle P ρ + mle Q ρ := by
  unfold mle
  simp only [coeff_add]
  exact mleAux_add ρ _ _

theorem mle_C_mul (a : K) (P : K[X]) (ρ : List K) : mle (C a * P) ρ = a * mle P ρ := by
  unfold mle
  simp only [coeff_C_mul]
  exact mleAux_const_mul ρ a _

omit [DecidableEq K] in
/-- Even-index coefficients of the even/odd lift `P₀(X²) + X P₁(X²)`. -/
theorem aux_mle_coeff_even (P0 P1 : K[X]) (i : ℕ) :
    (expand K 2 P0 + X * expand K 2 P1).coeff (2 * i) = P0.coeff i := by
  rw [coeff_add, coeff_expand_mul' (by norm_num : (0 : ℕ) < 2)]
  cases i with
  | zero => simp
  | succ j =>
    have hidx : 2 * (j + 1) = (2 * j + 1) + 1 := by ring
    have hnd : ¬ (2 ∣ 2 * j + 1) := by omega
    rw [hidx, coeff_X_mul, coeff_expand (by norm_num : (0 : ℕ) < 2), ite_eq_right hnd, add_zero]

omit [DecidableEq K] in
/-- Odd-index coefficients of the even/odd lift `P₀(X²) + X P₁(X²)`. -/
theorem aux_mle_coeff_odd (P0 P1 : K[X]) (i : ℕ) :
    (expand K 2 P0 + X * expand K 2 P1).coeff (2 * i + 1) = P1.coeff i := by
  have hnd : ¬ (2 ∣ 2 * i + 1) := by omega
  rw [coeff_add, coeff_X_mul, coeff_expand (by norm_num : (0 : ℕ) < 2), ite_eq_right hnd, zero_add,
    coeff_expand_mul' (by norm_num : (0 : ℕ) < 2)]

/-- The even/odd lift `P(X) = P₀(X²) + X P₁(X²)` has the multilinear twin
`P~(r, ρ) = P₀~(ρ) + r P₁~(ρ)` (CPRS indexing). -/
theorem mle_lift (P0 P1 : K[X]) (r : K) (ρ : List K) :
    mle (expand K 2 P0 + X * expand K 2 P1) (r :: ρ) = mle P0 ρ + r * mle P1 ρ := by
  have h0 : (fun i => (expand K 2 P0 + X * expand K 2 P1).coeff (2 * i))
      = fun i => P0.coeff i := by
    funext i
    exact aux_mle_coeff_even P0 P1 i
  have h1 : (fun i => (expand K 2 P0 + X * expand K 2 P1).coeff (2 * i + 1))
      = fun i => P1.coeff i := by
    funext i
    exact aux_mle_coeff_odd P0 P1 i
  unfold mle
  simp only [mleAux]
  rw [h0, h1]

#print axioms mle_lift

end Cprs
