import CprsFormal.PhaseB

/-!
# Completeness of the Phase-B model: polynomial algebra of the honest prover

The honest prover of the Phase-B state machine (`CprsFormal.PhaseB`) holds a polynomial `P_t`
with `< 2^(m-t)` coefficients, folds it with the verifier's challenges
(`P_(t+1) = P_t,even + a_t * P_t,odd`, where `P_t(X) = P_t,even(X²) + X * P_t,odd(X²)`), and sends
the round polynomial `g_t(X) = E_t * eq₁(ρ_t, X) * (m₀ + m₁ X)` with
`m₀ = mle P_t,even (ρ.drop (t+1))`, `m₁ = mle P_t,odd (ρ.drop (t+1))`.
This file contains the purely algebraic ingredients.
-/

open Polynomial Finset

namespace Cprs
namespace Complete

variable {K : Type*} [Field K] [DecidableEq K]

/-! ### Even and odd parts of a polynomial -/

/-- Even part `Q_e` of `Q`: `Q_e.coeff i = Q.coeff (2 i)`. -/
noncomputable def evenP (Q : K[X]) : K[X] := contract 2 Q

/-- Odd part `Q_o` of `Q`: `Q_o.coeff i = Q.coeff (2 i + 1)`. -/
noncomputable def oddP (Q : K[X]) : K[X] := contract 2 (divX Q)

theorem evenP_coeff (Q : K[X]) (i : ℕ) : (evenP Q).coeff i = Q.coeff (2 * i) := by
  unfold evenP
  rw [coeff_contract (by norm_num : (2 : ℕ) ≠ 0), mul_comm i 2]

theorem oddP_coeff (Q : K[X]) (i : ℕ) : (oddP Q).coeff i = Q.coeff (2 * i + 1) := by
  unfold oddP
  rw [coeff_contract (by norm_num : (2 : ℕ) ≠ 0), coeff_divX, mul_comm i 2]

/-- `Q(X) = Q_e(X²) + X Q_o(X²)`. -/
theorem eq_expand_even_odd (Q : K[X]) :
    Q = expand K 2 (evenP Q) + Polynomial.X * expand K 2 (oddP Q) := by
  ext n
  obtain ⟨i, hi | hi⟩ := Nat.even_or_odd' n
  · subst hi
    rw [aux_mle_coeff_even, evenP_coeff]
  · subst hi
    rw [aux_mle_coeff_odd, oddP_coeff]

theorem eval_eq_even_odd (Q : K[X]) (z : K) :
    Q.eval z = (evenP Q).eval (z ^ 2) + z * (oddP Q).eval (z ^ 2) := by
  have h := congrArg (Polynomial.eval z) (eq_expand_even_odd Q)
  first
    | (rw [eval_add, eval_mul, eval_X, expand_eval, expand_eval] at h; exact h)
    | simpa [eval_add, eval_mul, expand_eval] using h

theorem evenP_mem {k : ℕ} {Q : K[X]} (hQ : Q ∈ degreeLT K (2 * k)) :
    evenP Q ∈ degreeLT K k := by
  rw [mem_degreeLT, degree_lt_iff_coeff_zero] at hQ ⊢
  intro n hn
  rw [evenP_coeff]
  exact hQ _ (by omega)

theorem oddP_mem {k : ℕ} {Q : K[X]} (hQ : Q ∈ degreeLT K (2 * k)) :
    oddP Q ∈ degreeLT K k := by
  rw [mem_degreeLT, degree_lt_iff_coeff_zero] at hQ ⊢
  intro n hn
  rw [oddP_coeff]
  exact hQ _ (by omega)

/-- One fold of a polynomial with the challenge `a`: `Q_e + a Q_o`. -/
noncomputable def foldPoly (a : K) (Q : K[X]) : K[X] := evenP Q + C a * oddP Q

theorem foldPoly_mem {k : ℕ} {Q : K[X]} (a : K) (hQ : Q ∈ degreeLT K (2 * k)) :
    foldPoly a Q ∈ degreeLT K k := by
  have h0 := evenP_mem hQ
  have h1 := oddP_mem hQ
  unfold foldPoly
  rw [C_mul']
  exact Submodule.add_mem _ h0 (Submodule.smul_mem _ a h1)

theorem eval_foldPoly (a : K) (Q : K[X]) (y : K) :
    (foldPoly a Q).eval y = (evenP Q).eval y + a * (oddP Q).eval y := by
  unfold foldPoly
  simp [eval_add, eval_mul, eval_C]

/-! ### The multilinear twin -/

theorem mle_foldPoly (a : K) (Q : K[X]) (ρ : List K) :
    mle (foldPoly a Q) ρ = mle (evenP Q) ρ + a * mle (oddP Q) ρ := by
  unfold foldPoly
  rw [mle_add, mle_C_mul]

theorem mle_cons_even_odd (Q : K[X]) (r : K) (ρ : List K) :
    mle Q (r :: ρ) = mle (evenP Q) ρ + r * mle (oddP Q) ρ := by
  have h := mle_lift (evenP Q) (oddP Q) r ρ
  rw [← eq_expand_even_odd Q] at h
  exact h

theorem mle_nil (Q : K[X]) : mle Q [] = Q.coeff 0 := by
  first
    | rfl
    | simp [mle, mleAux]

theorem mleAux_zero (ρ : List K) : mleAux ρ (fun _ => (0 : K)) = 0 := by
  induction ρ with
  | nil => simp [mleAux]
  | cons r ρ ih => simp [mleAux, ih]

theorem mle_zero' (ρ : List K) : mle (0 : K[X]) ρ = 0 := by
  unfold mle
  simpa [coeff_zero] using mleAux_zero ρ

/-! ### Folding a word that agrees with a polynomial -/

/-- If the word `W` agrees with the polynomial `Q` on `D`, then `fold a W` agrees with the folded
polynomial `Q_e + a Q_o` on the next layer `D'`. -/
theorem fold_eval {D D' : Finset K} (hL : SqLayer D D') (a : K) (Q : K[X]) (W : K → K)
    (hW : ∀ z ∈ D, W z = Q.eval z) {y : K} (hy : y ∈ D') :
    fold a W y = (foldPoly a Q).eval y := by
  obtain ⟨hs0, hs1, hsq, hs3, -⟩ := Aux.LemmaQ.sqrt_mem hL hy
  have h2 : (2 : K) ≠ 0 := hL.two_ne_zero
  have e1 : W (sqrtK y) = (evenP Q).eval y + sqrtK y * (oddP Q).eval y := by
    rw [hW _ hs0, eval_eq_even_odd Q (sqrtK y), hsq]
  have e2 : W (-sqrtK y) = (evenP Q).eval y - sqrtK y * (oddP Q).eval y := by
    rw [hW _ hs1, eval_eq_even_odd Q (-sqrtK y), neg_sq, hsq]
    ring
  have hev : evenPart W y = (evenP Q).eval y := by
    unfold evenPart
    rw [e1, e2, div_eq_iff h2]
    ring
  have hod : oddPart W y = (oddP Q).eval y := by
    unfold oddPart
    rw [e1, e2, div_eq_iff (mul_ne_zero h2 hs3)]
    ring
  show evenPart W y + a * oddPart W y = _
  rw [hev, hod, eval_foldPoly]

/-! ### The honest polynomial, the factor `E_t`, and the round polynomial -/

/-- The honest prover's polynomial after the challenge prefix `pre`: `P_0 = P`,
`P_(t+1) = (P_t)_e + a_t (P_t)_o`. -/
noncomputable def Pt (P : K[X]) : (t : ℕ) → (Fin t → K) → K[X]
  | 0, _ => P
  | t + 1, pre => foldPoly (pre (Fin.last t)) (Pt P t (Fin.init pre))

/-- The factor `E_t = ∏_(u<t) eq₁(ρ_u, a_u)`. -/
noncomputable def Et (ρ : List K) : (t : ℕ) → (Fin t → K) → K
  | 0, _ => 1
  | t + 1, pre => Et ρ t (Fin.init pre) * eq1 (ρ.getD t 0) (pre (Fin.last t))

theorem Pt_snoc (P : K[X]) (t : ℕ) (pre : Fin t → K) (x : K) :
    Pt P (t + 1) (Fin.snoc (α := fun _ => K) pre x) = foldPoly x (Pt P t pre) := by
  first
    | (simp only [Pt, Fin.init_snoc, Fin.snoc_last]; done)
    | (rw [Pt, Fin.init_snoc, Fin.snoc_last])

theorem Et_snoc (ρ : List K) (t : ℕ) (pre : Fin t → K) (x : K) :
    Et ρ (t + 1) (Fin.snoc (α := fun _ => K) pre x) = Et ρ t pre * eq1 (ρ.getD t 0) x := by
  first
    | (simp only [Et, Fin.init_snoc, Fin.snoc_last]; done)
    | (rw [Et, Fin.init_snoc, Fin.snoc_last])

/-- The degrees of the honest polynomials: `P_t` has `< 2^(m-t)` coefficients. -/
theorem Pt_mem (P : K[X]) (m : ℕ) (hP : P ∈ degreeLT K (2 ^ m)) :
    ∀ t, t ≤ m → ∀ pre : Fin t → K, Pt P t pre ∈ degreeLT K (2 ^ (m - t)) := by
  intro t
  induction t with
  | zero =>
    intro _ pre
    have e0 : Pt P 0 pre = P := by
      first
        | rfl
        | simp only [Pt]
    rw [e0, Nat.sub_zero]
    exact hP
  | succ t ih =>
    intro ht pre
    have h := ih (by omega) (Fin.init pre)
    have hk : 2 ^ (m - t) = 2 * 2 ^ (m - (t + 1)) := by
      have e : m - t = (m - (t + 1)) + 1 := by omega
      rw [e, pow_succ]
      ring
    rw [hk] at h
    have e1 : Pt P (t + 1) pre = foldPoly (pre (Fin.last t)) (Pt P t (Fin.init pre)) := by
      first
        | rfl
        | simp only [Pt]
    rw [e1]
    exact foldPoly_mem (pre (Fin.last t)) h

/-- A polynomial with `< 1` coefficient is the constant `coeff 0`. -/
theorem eval_const_of_mem (Q : K[X]) (hQ : Q ∈ degreeLT K 1) (z : K) :
    Q.eval z = Q.coeff 0 := by
  have h : Q = C (Q.coeff 0) := by
    rw [mem_degreeLT, degree_lt_iff_coeff_zero] at hQ
    ext n
    cases n with
    | zero => simp
    | succ j =>
      rw [coeff_C_succ]
      exact hQ (j + 1) (by omega)
  have h2 : Q.eval z = (C (Q.coeff 0)).eval z := congrArg (Polynomial.eval z) h
  rw [h2, eval_C]

/-- The honest round polynomial `g(X) = E * eq₁(ρ_t, X) * (m₀ + m₁ X)`. -/
noncomputable def gPoly (E ρt m0 m1 : K) : K[X] :=
  C E * (C ρt * Polynomial.X + C (1 - ρt) * (1 - Polynomial.X)) * (C m0 + C m1 * Polynomial.X)

theorem gPoly_eval (E ρt m0 m1 a : K) :
    (gPoly E ρt m0 m1).eval a = E * eq1 ρt a * (m0 + a * m1) := by
  simp only [gPoly, eval_mul, eval_add, eval_sub, eval_C, eval_X, eval_one, eq1]
  ring

theorem gPoly_natDegree (E ρt m0 m1 : K) : (gPoly E ρt m0 m1).natDegree ≤ 2 := by
  unfold gPoly
  compute_degree

/-- The sum-check identity `g(0) + g(1) = E * (m₀ + ρ_t m₁)`. -/
theorem gPoly_sum (E ρt m0 m1 : K) :
    (gPoly E ρt m0 m1).eval 0 + (gPoly E ρt m0 m1).eval 1 = E * (m0 + ρt * m1) := by
  simp only [gPoly_eval, eq1]
  ring

/-! ### Weighted agreement for constant weight `1` -/

theorem wagree_eq_one {D : Finset K} (hD : D.card ≠ 0) (ν : K → ℚ) (u v : K → K)
    (hν : ∀ x ∈ D, ν x = 1) (huv : ∀ x ∈ D, u x = v x) : wagree D ν u v = 1 := by
  have hf : D.filter (fun x => u x = v x) = D := Finset.filter_true_of_mem huv
  have hsum : ∑ x ∈ D, ν x = (D.card : ℚ) := by
    calc ∑ x ∈ D, ν x = ∑ x ∈ D, (1 : ℚ) := Finset.sum_congr rfl hν
      _ = (D.card : ℚ) := by simp
  unfold wagree
  rw [hf, hsum]
  exact div_self (by exact_mod_cast hD)

end Complete
end Cprs
