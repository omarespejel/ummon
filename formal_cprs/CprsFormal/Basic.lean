import Mathlib

/-!
# CPRS formal development — basic objects

Definitions used by the machine-checked part of the CPRS soundness argument
(the setting of Phase B and its state relation).

* words are functions `K → K` (only their values on the relevant domain matter);
* `fold a u = u_e + a • u_o` for the squaring map `x ↦ x²`;
* `fibAvg ν` is the fibre average of a weight;
* `wagree D ν u v = (1/|D|) Σ_{x ∈ D, u x = v x} ν x` is the weighted agreement;
* `mle P ρ` is the multilinear twin `P~(ρ)` with the CPRS indexing
  `P~(x₀, x') = P~_e(x') + x₀ P~_o(x')`;
* `Rel D k β ρ ν W c E` is the state relation `R_t(W, ν, c, E)`.
-/

open Polynomial Finset

namespace Cprs

section Basic

variable {K : Type*} [Field K] [DecidableEq K]

/-- A chosen square root: some `x` with `x ^ 2 = y` if one exists (and an arbitrary element
otherwise). -/
noncomputable def sqrtK (y : K) : K := Classical.epsilon (fun x : K => x ^ 2 = y)

theorem sqrtK_sq {y : K} (h : ∃ x : K, x ^ 2 = y) : sqrtK y ^ 2 = y :=
  Classical.epsilon_spec (p := fun x : K => x ^ 2 = y) h

/-- Even part `u_e` of a word for the squaring map: `u_e (x²) = (u x + u (-x)) / 2`. -/
noncomputable def evenPart (u : K → K) (y : K) : K := (u (sqrtK y) + u (-sqrtK y)) / 2

/-- Odd part `u_o`: `u_o (x²) = (u x - u (-x)) / (2 x)`. -/
noncomputable def oddPart (u : K → K) (y : K) : K :=
  (u (sqrtK y) - u (-sqrtK y)) / (2 * sqrtK y)

/-- The fold of a word with challenge `a`: `fold_a u = u_e + a • u_o`. -/
noncomputable def fold (a : K) (u : K → K) : K → K := fun y => evenPart u y + a * oddPart u y

/-- Fibre average of a weight for the squaring map: `ν^v (x²) = (ν x + ν (-x)) / 2`. -/
noncomputable def fibAvg (ν : K → ℚ) : K → ℚ := fun y => (ν (sqrtK y) + ν (-sqrtK y)) / 2

/-- `eq₁(x, y) = x y + (1 - x)(1 - y)`. -/
def eq1 (x y : K) : K := x * y + (1 - x) * (1 - y)

/-- Weighted agreement `agree_ν(u, v) = (1 / |D|) Σ_{x ∈ D, u x = v x} ν x`. -/
noncomputable def wagree {α : Type*} (D : Finset α) (ν : α → ℚ) (u v : α → K) : ℚ :=
  (∑ x ∈ D.filter (fun x => u x = v x), ν x) / D.card

/-- Number of points of `D` at which the word `u` agrees with the polynomial `P`. -/
def agreeCount (D : Finset K) (u : K → K) (P : K[X]) : ℕ :=
  (D.filter (fun x => u x = P.eval x)).card

/-- Multilinear evaluation of a coefficient function, CPRS indexing: the least significant bit of
the monomial index pairs with the first variable, so that
`P~(x₀, x') = P~_e(x') + x₀ P~_o(x')` where `P_e, P_o` are the even/odd coefficient subsequences. -/
noncomputable def mleAux : List K → (ℕ → K) → K
  | [], c => c 0
  | r :: ρ, c => mleAux ρ (fun i => c (2 * i)) + r * mleAux ρ (fun i => c (2 * i + 1))

/-- The multilinear twin `P~(ρ)` of a polynomial `P` (same monomial coefficients). -/
noncomputable def mle (P : K[X]) (ρ : List K) : K := mleAux ρ (fun i => P.coeff i)

/-- The state relation `R_t(W, ν, c, E)` of Theorem B: there is a polynomial `P` with `< k`
coefficients, weighted agreement `≥ β` with `W` on `D`, and `c = E · P~(ρ)`. -/
def Rel (D : Finset K) (k : ℕ) (β : ℚ) (ρ : List K) (ν : K → ℚ) (W : K → K) (c E : K) : Prop :=
  ∃ P : K[X], P ∈ degreeLT K k ∧ β ≤ wagree D ν W (fun x => P.eval x) ∧ c = E * mle P ρ

/-- Two consecutive layers `D → D'` of the folding: `D'` is the image of `D` under squaring,
`D` is closed under negation, does not contain `0`, and the characteristic is not `2`.
(For the subgroups `D_t = ⟨ζ^(2^t)⟩ ⊂ K*` of even order this holds; see `Cprs.Layers`.) -/
structure SqLayer (D D' : Finset K) : Prop where
  two_ne_zero : (2 : K) ≠ 0
  zero_notMem : (0 : K) ∉ D
  neg_mem : ∀ x ∈ D, -x ∈ D
  sq_mem : ∀ x ∈ D, x ^ 2 ∈ D'
  sq_surj : ∀ y ∈ D', ∃ x ∈ D, x ^ 2 = y

end Basic

end Cprs
