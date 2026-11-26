import CprsFormal.PhaseA
import CprsFormal.Packing
import CprsFormal.Layers
import CprsFormal.PhaseB

/-!
# Definitions for the unified Model-H chain theorem (no proofs here; shared by `Bridge` and `SumCheck`)
-/

open Polynomial Finset

namespace Cprs

section Defs
variable {K : Type*} [Field K]

/-- The Boolean point `x : Fin m → Bool` as a point of `K^m` (`true ↦ 1`, `false ↦ 0`). -/
def boolPt {m : ℕ} (x : Fin m → Bool) : Fin m → K := fun i => if x i then 1 else 0

/-- `F : K^m → K` has degree `≤ e` in every single coordinate (the others held fixed). -/
def CoordDeg {m : ℕ} (e : ℕ) (F : (Fin m → K) → K) : Prop :=
  ∀ (t : Fin m) (ρ : Fin m → K), ∃ q : K[X], q.natDegree ≤ e ∧
    ∀ x : K, F (Function.update ρ t x) = q.eval x

/-- The Phase-A sum-check polynomial `F(ρ) = Σ_j c_j · eq(p_j, ρ) · T_j~(ρ)`, where `T_j~` is the
Lagrange-form multilinear extension of the table `T_j` and `eq(p, ρ) = ∏_t eq₁(p_t, ρ_t)`. -/
noncomputable def chainF {m : ℕ} {J : Type*} [Fintype J] (c : J → K) (p : J → Fin m → K)
    (T : J → (Fin m → Bool) → K) : (Fin m → K) → K :=
  fun ρ => ∑ j, c j * ((∏ t, eq1 (p j t) (ρ t)) * Packing.mleTable (T j) ρ)

/-- A sum-check prover strategy: the round polynomial `g_t` as a function of the challenge prefix. -/
abbrev SCStrat (K : Type*) [Field K] := (t : ℕ) → (Fin t → K) → K[X]

/-- The running claim after the challenge prefix `pre` (`claim_0 = c0`, `claim_{t+1} = g_t(ρ_t)`). -/
noncomputable def scClaim (g : SCStrat K) (c0 : K) : (t : ℕ) → (Fin t → K) → K
  | 0, _ => c0
  | t + 1, pre => (g t (Fin.init pre)).eval (pre (Fin.last t))

/-- All round checks `g_t(0) + g_t(1) = claim_t` passed along the prefix. -/
def scOk (g : SCStrat K) (c0 : K) : (t : ℕ) → (Fin t → K) → Prop
  | 0, _ => True
  | t + 1, pre => scOk g c0 t (Fin.init pre) ∧
      (g t (Fin.init pre)).eval 0 + (g t (Fin.init pre)).eval 1 = scClaim g c0 t (Fin.init pre)

end Defs

end Cprs
