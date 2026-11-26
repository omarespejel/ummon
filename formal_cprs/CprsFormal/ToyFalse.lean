import CprsFormal.Layers

/-!
# Non-vacuity, part 1: a false claim on a concrete well-formed instance over `ZMod 17`

`K = F_17`, `m = s = 1`, `ζ = 4` (`4² = -1`, order `4 = 2^(m+s)`), so `D_0 = {1, 4, 16, 13}` and
`D_1 = {1, 16}`; `ρ = (7)`, no interior block end, `t₁ = 1 = m`, `β = 3/4 = (2^s + 1)/2^(s+1)`.
With `π ≡ 0` and `c₀ = 1` the initial relation `R_0(π, 1, c₀, 1)` is false, so the hypotheses of
`PhaseBInst.soundness_unconditional` are satisfiable together.
-/

open Polynomial Finset

namespace Cprs
namespace Toy

instance factPrime17 : Fact (Nat.Prime 17) := ⟨by norm_num⟩

theorem orderOf_four : orderOf (4 : ZMod 17) = 2 ^ (1 + 1) := by
  first
    | exact orderOf_eq_prime_pow (by decide) (by decide)
    | exact orderOf_eq_prime_pow (p := 2) (n := 1) (by decide) (by decide)

/-- The toy instance: `m = s = 1`, `ζ = 4`, `ρ = (7)`, `isEnd ≡ false`, `t₁ = 1`. -/
def toyInst (π : ZMod 17 → ZMod 17) (c0 : ZMod 17) : PhaseBInst (ZMod 17) :=
  PhaseBInst.ofRoot (4 : ZMod 17) 1 1 [7] (fun _ => false) 1 π c0

theorem toyInst_wf (π : ZMod 17 → ZMod 17) (c0 : ZMod 17) : (toyInst π c0).WF :=
  PhaseBInst.ofRoot_wf (4 : ZMod 17) 1 1 orderOf_four le_rfl [7] rfl (fun _ => false) 1
    Nat.one_pos le_rfl (fun t ht => absurd ht (by simp)) (fun h => absurd h (lt_irrefl 1))
    (fun t _ _ => rfl) π c0

section General

variable {K : Type*} [Field K] [DecidableEq K]

theorem wagree_one (D : Finset K) (u v : K → K) :
    wagree D (fun _ => (1 : ℚ)) u v =
      ((D.filter (fun x => u x = v x)).card : ℚ) / (D.card : ℚ) := by
  unfold wagree
  simp

theorem mleAux_zero (ρ : List K) : mleAux ρ (fun _ => (0 : K)) = 0 := by
  induction ρ with
  | nil => simp [mleAux]
  | cons r ρ ih => simp [mleAux, ih]

theorem mle_zero' (ρ : List K) : mle (0 : K[X]) ρ = 0 := by
  unfold mle
  simpa [coeff_zero] using mleAux_zero ρ

/-- A polynomial with `< k` coefficients that vanishes on `≥ k` points is zero. -/
theorem poly_zero_of_many_roots (A : Finset K) (P : K[X]) (k : ℕ) (hP : P ∈ degreeLT K k)
    (hk : k ≤ A.card) (hroot : ∀ x ∈ A, P.eval x = 0) : P = 0 := by
  have hkW : ((k : ℕ) : WithBot ℕ) ≤ ((A.card : ℕ) : WithBot ℕ) := by exact_mod_cast hk
  have hdegP : P.degree < ((A.card : ℕ) : WithBot ℕ) := lt_of_lt_of_le (mem_degreeLT.1 hP) hkW
  have hdeg0 : (0 : K[X]).degree < ((A.card : ℕ) : WithBot ℕ) :=
    mem_degreeLT.1 (Submodule.zero_mem _)
  exact Polynomial.eq_of_degrees_lt_of_eval_finset_eq A hdegP hdeg0
    (fun x hx => by rw [hroot x hx, eval_zero])

/-- With `m = s = 1`, `π ≡ 0` and `c₀ = 1` the initial relation is false at `β = 3/4`: a polynomial
with `< 2` coefficients that agrees with `0` on `≥ 3` of the `4` points of `D_0` is `0`, whose
twin is `0 ≠ 1`. -/
theorem not_rel_of_zero (I : PhaseBInst K) (hWF : I.WF) (hm : I.m = 1) (hs : I.s = 1)
    (hπ : ∀ x, I.π x = 0) (hc0 : I.c0 = 1) :
    ¬ Rel (I.D 0) (2 ^ I.m) (3 / 4) I.ρ (fun _ => 1) I.π I.c0 1 := by
  rintro ⟨P, hP, hag, hc⟩
  have hcard : (I.D 0).card = 4 := by
    have h := hWF.card 0 (Nat.zero_le _)
    rw [hm, hs] at h
    simpa using h
  have hcardQ : ((I.D 0).card : ℚ) = 4 := by exact_mod_cast hcard
  have hag' : (3 : ℚ) / 4 ≤
      (((I.D 0).filter (fun x => I.π x = P.eval x)).card : ℚ) / ((I.D 0).card : ℚ) :=
    hag.trans_eq (wagree_one (I.D 0) I.π (fun x => P.eval x))
  rw [hcardQ] at hag'
  have h3Q : (3 : ℚ) ≤ (((I.D 0).filter (fun x => I.π x = P.eval x)).card : ℚ) := by
    linarith
  have h3 : 3 ≤ ((I.D 0).filter (fun x => I.π x = P.eval x)).card := by exact_mod_cast h3Q
  have hk : 2 ^ I.m ≤ ((I.D 0).filter (fun x => I.π x = P.eval x)).card := by
    have h2 : 2 ^ I.m = 2 := by simp [hm]
    rw [h2]
    omega
  have hP0 : P = 0 := by
    refine poly_zero_of_many_roots ((I.D 0).filter (fun x => I.π x = P.eval x)) P (2 ^ I.m) hP hk ?_
    intro x hx
    have h := (Finset.mem_filter.1 hx).2
    rw [hπ] at h
    exact h.symm
  rw [hP0, mle_zero', hc0] at hc
  simp at hc

end General

/-- The false toy instance: `π ≡ 0`, `c₀ = 1`. -/
def toyFalse : PhaseBInst (ZMod 17) := toyInst (fun _ => 0) 1

theorem toyFalse_wf : toyFalse.WF := toyInst_wf _ _

theorem toyFalse_beta : (3 / 4 : ℚ) = (2 ^ toyFalse.s + 1) / 2 ^ (toyFalse.s + 1) := by
  show (3 / 4 : ℚ) = (2 ^ 1 + 1) / 2 ^ (1 + 1)
  norm_num

theorem toyFalse_not_rel :
    ¬ Rel (toyFalse.D 0) (2 ^ toyFalse.m) (3 / 4) toyFalse.ρ (fun _ => 1) toyFalse.π
      toyFalse.c0 1 :=
  not_rel_of_zero toyFalse toyFalse_wf rfl rfl (fun _ => rfl) rfl

/-- **Non-vacuity (false claim).** -/
theorem nonvacuity_false : ∃ I : PhaseBInst (ZMod 17), I.WF ∧
    ¬ Rel (I.D 0) (2 ^ I.m) (3 / 4) I.ρ (fun _ => 1) I.π I.c0 1 :=
  ⟨toyFalse, toyFalse_wf, toyFalse_not_rel⟩

/-- All hypotheses of `PhaseBInst.soundness_unconditional` hold together on a single instance:
well-formedness, `β = (2^s + 1)/2^(s+1)` (with `s = 1`), and a false initial relation. -/
theorem soundness_hypotheses_satisfiable : ∃ I : PhaseBInst (ZMod 17), I.WF ∧
    (3 / 4 : ℚ) = (2 ^ I.s + 1) / 2 ^ (I.s + 1) ∧
    ¬ Rel (I.D 0) (2 ^ I.m) (3 / 4) I.ρ (fun _ => 1) I.π I.c0 1 :=
  ⟨toyFalse, toyFalse_wf, toyFalse_beta, toyFalse_not_rel⟩

/-- `PhaseBInst.soundness_unconditional` applies to the toy instance (any strategy, any `Q`). -/
example (S : Strategy (ZMod 17) toyFalse.m) (Q : ℕ) :=
  PhaseBInst.soundness_unconditional toyFalse toyFalse_wf (3 / 4) toyFalse_beta
    toyFalse_not_rel S Q

#print axioms nonvacuity_false
#print axioms soundness_hypotheses_satisfiable

end Toy
end Cprs
