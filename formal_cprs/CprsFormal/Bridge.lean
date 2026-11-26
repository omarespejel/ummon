import CprsFormal.ChainDefs

/-!
# Bridge: Lagrange form versus coefficient form of the multilinear twin, and coordinate degrees

This file proves the five "Bridge" statements used by the chain theorem (`Chain.lean`)
(`mle_eq_mleTable`, `exists_poly_of_table`, `coordDeg_mleTable`, `coordDeg_chainF`,
`sum_bool_chainF`), plus the extra fact `mleTable_boolPt` (the Lagrange extension agrees with its
table on Boolean points).

It imports `CprsFormal.ChainDefs` only.  All auxiliary declarations are named `aux_bridge_*`.

Proof outline.
* `mle_eq_mleTable`: `mleAux` of a list of length `m` reads only the coefficients `c 0 … c (2^m - 1)`,
  and the identity `mleAux (List.ofFn r) c = Σ_x mleAux (List.ofFn (boolPt x)) c · eqPoly x r` holds
  for *every* coefficient function `c` (induction on `m`, splitting off the first coordinate with
  `Fin.cons`; the first variable is the least significant bit of the monomial index).  The degree
  hypothesis `hP` is therefore not needed; the binder is kept so that the statement reads as the
  chain theorem uses it.
* `exists_poly_of_table`: induction on `m` with `P = P₀(X²) + X · P₁(X²)` (`mle_lift`), where
  `P₀` interpolates `T(false, ·)` and `P₁` interpolates `T(true, ·) - T(false, ·)`.
* `coordDeg_mleTable`, `coordDeg_chainF`: along a coordinate line `x ↦ F (update ρ t x)` every
  factor is a polynomial of degree `≤ 1` (the factor `eq₁ (p t) x` or `if b t then x else 1 - x`)
  times a constant; the degrees add under products and are preserved under finite sums.
* `sum_bool_chainF`: on Boolean points `∏ t, eq₁ (p t) (boolPt b t) = eqPoly b p` and
  `mleTable T (boolPt b) = T b`, then exchange the two finite sums.

NB: ArkLib makes the bare token `X` bivariate, so `Polynomial.X` is always written in full.
-/

open Polynomial Finset

namespace Cprs

section AuxNoDec

variable {K : Type*} [Field K]

/-! ### Splitting off the first coordinate of a Boolean point -/

theorem aux_bridge_boolPt_cons {m : ℕ} (b : Bool) (y : Fin m → Bool) :
    (boolPt (Fin.cons b y : Fin (m + 1) → Bool) : Fin (m + 1) → K) =
      Fin.cons (if b then (1 : K) else 0) (boolPt y : Fin m → K) := by
  funext i
  refine Fin.cases ?_ (fun j => ?_) i
  · simp [boolPt]
  · simp [boolPt]

theorem aux_bridge_ofFn_cons {α : Type*} {m : ℕ} (a : α) (f : Fin m → α) :
    List.ofFn (Fin.cons a f : Fin (m + 1) → α) = a :: List.ofFn f := by
  rw [List.ofFn_succ]
  simp

theorem aux_bridge_eqPoly_cons {m : ℕ} (b : Bool) (y : Fin m → Bool) (r : Fin (m + 1) → K) :
    Packing.eqPoly (Fin.cons b y : Fin (m + 1) → Bool) r =
      (if b then r 0 else 1 - r 0) * Packing.eqPoly y (fun i => r i.succ) := by
  unfold Packing.eqPoly
  rw [Fin.prod_univ_succ]
  simp

theorem aux_bridge_sum_cons {M : Type*} [AddCommMonoid M] {m : ℕ}
    (F : (Fin (m + 1) → Bool) → M) :
    ∑ x : Fin (m + 1) → Bool, F x =
      ∑ y : Fin m → Bool, F (Fin.cons true y) + ∑ y : Fin m → Bool, F (Fin.cons false y) := by
  have h1 := Fintype.sum_equiv (Fin.consEquiv (fun _ => Bool))
    (fun p => F (Fin.cons p.1 p.2)) F (fun p => rfl)
  rw [Fintype.sum_prod_type, Fintype.sum_bool] at h1
  exact h1.symm

/-! ### The coefficient-function version of the Lagrange identity -/

theorem aux_bridge_mleAux_cons (r : K) (ρ : List K) (c : ℕ → K) :
    mleAux (r :: ρ) c = mleAux ρ (fun i => c (2 * i)) + r * mleAux ρ (fun i => c (2 * i + 1)) := by
  simp only [mleAux]

/-- `mleAux` of a list of length `m` only reads the first `2^m` coefficients; it equals the
Lagrange form of its own Boolean table.  No support hypothesis on `c` is needed. -/
theorem aux_bridge_mleAux_ofFn {m : ℕ} :
    ∀ (c : ℕ → K) (r : Fin m → K), mleAux (List.ofFn r) c =
      ∑ x : Fin m → Bool, mleAux (List.ofFn (boolPt x : Fin m → K)) c * Packing.eqPoly x r := by
  induction m with
  | zero =>
    intro c r
    simp [mleAux, Packing.eqPoly]
  | succ m ih =>
    intro c r
    have hT : ∀ y : Fin m → Bool,
        mleAux (List.ofFn (boolPt (Fin.cons true y : Fin (m + 1) → Bool) : Fin (m + 1) → K)) c *
            Packing.eqPoly (Fin.cons true y : Fin (m + 1) → Bool) r =
          (mleAux (List.ofFn (boolPt y : Fin m → K)) (fun i => c (2 * i)) +
              mleAux (List.ofFn (boolPt y : Fin m → K)) (fun i => c (2 * i + 1))) *
            (r 0 * Packing.eqPoly y (fun i => r i.succ)) := by
      intro y
      rw [aux_bridge_boolPt_cons, aux_bridge_ofFn_cons, aux_bridge_mleAux_cons,
        aux_bridge_eqPoly_cons]
      simp
    have hF : ∀ y : Fin m → Bool,
        mleAux (List.ofFn (boolPt (Fin.cons false y : Fin (m + 1) → Bool) : Fin (m + 1) → K)) c *
            Packing.eqPoly (Fin.cons false y : Fin (m + 1) → Bool) r =
          mleAux (List.ofFn (boolPt y : Fin m → K)) (fun i => c (2 * i)) *
            ((1 - r 0) * Packing.eqPoly y (fun i => r i.succ)) := by
      intro y
      rw [aux_bridge_boolPt_cons, aux_bridge_ofFn_cons, aux_bridge_mleAux_cons,
        aux_bridge_eqPoly_cons]
      simp
    rw [aux_bridge_sum_cons]
    simp only [hT, hF]
    rw [List.ofFn_succ, aux_bridge_mleAux_cons, ih (fun i => c (2 * i)) (fun i => r i.succ),
      ih (fun i => c (2 * i + 1)) (fun i => r i.succ), Finset.mul_sum,
      ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun y _ => ?_)
    ring

/-! ### Degree bookkeeping for the existence statement -/

theorem aux_bridge_mem_degreeLT_lift {m : ℕ} {P0 P1 : K[X]} (h0 : P0 ∈ degreeLT K (2 ^ m))
    (h1 : P1 ∈ degreeLT K (2 ^ m)) :
    expand K 2 P0 + Polynomial.X * expand K 2 P1 ∈ degreeLT K (2 ^ (m + 1)) := by
  rw [mem_degreeLT, Polynomial.degree_lt_iff_coeff_zero] at h0 h1 ⊢
  intro j hj
  have h2 : 2 ^ (m + 1) = 2 * 2 ^ m := by ring
  obtain ⟨i, rfl | rfl⟩ : ∃ i, j = 2 * i ∨ j = 2 * i + 1 := ⟨j / 2, by omega⟩
  · rw [aux_mle_coeff_even]
    exact h0 i (by omega)
  · rw [aux_mle_coeff_odd]
    exact h1 i (by omega)

theorem aux_bridge_C_mem_degreeLT (a : K) : Polynomial.C a ∈ degreeLT K (2 ^ 0) := by
  rw [mem_degreeLT, Polynomial.degree_lt_iff_coeff_zero]
  intro j hj
  have hj' : 1 ≤ j := by rwa [pow_zero] at hj
  rw [Polynomial.coeff_C, ite_eq_right (by omega)]

/-! ### One-variable polynomial degree bookkeeping -/

/-- `g : K → K` is the evaluation function of a polynomial of degree `≤ e`. -/
def aux_bridge_PolyDeg (e : ℕ) (g : K → K) : Prop :=
  ∃ q : K[X], q.natDegree ≤ e ∧ ∀ x : K, g x = q.eval x

theorem aux_bridge_polyDeg_const (e : ℕ) (a : K) : aux_bridge_PolyDeg e (fun _ => a) :=
  ⟨Polynomial.C a, by simp, fun x => by simp⟩

theorem aux_bridge_polyDeg_id : aux_bridge_PolyDeg 1 (fun x : K => x) :=
  ⟨Polynomial.X, by simp, fun x => by simp⟩

theorem aux_bridge_polyDeg_mono {e e' : ℕ} {g : K → K} (h : aux_bridge_PolyDeg e g)
    (hee : e ≤ e') : aux_bridge_PolyDeg e' g := by
  obtain ⟨q, hq, hg⟩ := h
  exact ⟨q, hq.trans hee, hg⟩

theorem aux_bridge_polyDeg_sub {e : ℕ} {f g : K → K} (hf : aux_bridge_PolyDeg e f)
    (hg : aux_bridge_PolyDeg e g) : aux_bridge_PolyDeg e (fun x => f x - g x) := by
  obtain ⟨p, hp, hpf⟩ := hf
  obtain ⟨q, hq, hqg⟩ := hg
  refine ⟨p - q, (Polynomial.natDegree_sub_le p q).trans (max_le hp hq), fun x => ?_⟩
  show f x - g x = Polynomial.eval x (p - q)
  rw [hpf, hqg, Polynomial.eval_sub]

theorem aux_bridge_polyDeg_mul {e e' : ℕ} {f g : K → K} (hf : aux_bridge_PolyDeg e f)
    (hg : aux_bridge_PolyDeg e' g) : aux_bridge_PolyDeg (e + e') (fun x => f x * g x) := by
  obtain ⟨p, hp, hpf⟩ := hf
  obtain ⟨q, hq, hqg⟩ := hg
  refine ⟨p * q, Polynomial.natDegree_mul_le.trans (add_le_add hp hq), fun x => ?_⟩
  show f x * g x = Polynomial.eval x (p * q)
  rw [hpf, hqg, Polynomial.eval_mul]

theorem aux_bridge_polyDeg_sum {ι : Type*} [Fintype ι] {e : ℕ} {f : ι → K → K}
    (hf : ∀ i, aux_bridge_PolyDeg e (f i)) : aux_bridge_PolyDeg e (fun x => ∑ i, f i x) := by
  have hf' : ∀ i, ∃ q : K[X], q.natDegree ≤ e ∧ ∀ x : K, f i x = q.eval x := hf
  choose q hq hqf using hf'
  refine ⟨∑ i, q i, Polynomial.natDegree_sum_le_of_forall_le _ _ (fun i _ => hq i), fun x => ?_⟩
  rw [Polynomial.eval_finsetSum]
  exact Finset.sum_congr rfl (fun i _ => hqf i x)

theorem aux_bridge_polyDeg_ite (bt : Bool) :
    aux_bridge_PolyDeg 1 (fun x : K => if bt then x else 1 - x) := by
  cases bt
  · have h := aux_bridge_polyDeg_sub (aux_bridge_polyDeg_const 1 (1 : K)) aux_bridge_polyDeg_id
    simpa using h
  · simpa using (aux_bridge_polyDeg_id : aux_bridge_PolyDeg 1 (fun x : K => x))

theorem aux_bridge_eq1_polyDeg (a : K) : aux_bridge_PolyDeg 1 (fun x : K => eq1 a x) := by
  refine ⟨Polynomial.C (2 * a - 1) * Polynomial.X + Polynomial.C (1 - a),
    Polynomial.natDegree_linear_le, fun x => ?_⟩
  simp only [eq1, Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C, Polynomial.eval_X]
  ring

/-! ### Coordinate-wise structure of `eqPoly`, `mleTable` and the `eq`-product -/

theorem aux_bridge_eqPoly_update {m : ℕ} (b : Fin m → Bool) (ρ : Fin m → K) (t : Fin m) (x : K) :
    Packing.eqPoly b (Function.update ρ t x) =
      (if b t then x else 1 - x) *
        ∏ i ∈ (Finset.univ : Finset (Fin m)).erase t, (if b i then ρ i else 1 - ρ i) := by
  unfold Packing.eqPoly
  rw [← Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ t)]
  congr 1
  · rw [Function.update_self]
  · refine Finset.prod_congr rfl (fun i hi => ?_)
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hi)]

theorem aux_bridge_mleTable_update_polyDeg {m : ℕ} (T : (Fin m → Bool) → K) (t : Fin m)
    (ρ : Fin m → K) :
    aux_bridge_PolyDeg 1 (fun x => Packing.mleTable T (Function.update ρ t x)) := by
  have hfun : (fun x => Packing.mleTable T (Function.update ρ t x)) =
      fun x => ∑ b : Fin m → Bool, T b * ((if b t then x else 1 - x) *
        ∏ i ∈ (Finset.univ : Finset (Fin m)).erase t, (if b i then ρ i else 1 - ρ i)) := by
    funext x
    unfold Packing.mleTable
    refine Finset.sum_congr rfl (fun b _ => ?_)
    rw [aux_bridge_eqPoly_update]
  rw [hfun]
  refine aux_bridge_polyDeg_sum (fun b => ?_)
  have h2 := aux_bridge_polyDeg_mul (aux_bridge_polyDeg_ite (K := K) (b t))
    (aux_bridge_polyDeg_const 0
      (∏ i ∈ (Finset.univ : Finset (Fin m)).erase t, (if b i then ρ i else 1 - ρ i)))
  have h3 := aux_bridge_polyDeg_mul (aux_bridge_polyDeg_const 0 (T b)) h2
  exact aux_bridge_polyDeg_mono h3 (by norm_num)

theorem aux_bridge_prod_eq1_update {m : ℕ} (a ρ : Fin m → K) (t : Fin m) (x : K) :
    ∏ s, eq1 (a s) (Function.update ρ t x s) =
      eq1 (a t) x * ∏ s ∈ (Finset.univ : Finset (Fin m)).erase t, eq1 (a s) (ρ s) := by
  rw [← Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ t)]
  congr 1
  · rw [Function.update_self]
  · refine Finset.prod_congr rfl (fun s hs => ?_)
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hs)]

theorem aux_bridge_prod_eq1_polyDeg {m : ℕ} (a ρ : Fin m → K) (t : Fin m) :
    aux_bridge_PolyDeg 1 (fun x : K => ∏ s, eq1 (a s) (Function.update ρ t x s)) := by
  have hfun : (fun x : K => ∏ s, eq1 (a s) (Function.update ρ t x s)) =
      fun x => eq1 (a t) x *
        ∏ s ∈ (Finset.univ : Finset (Fin m)).erase t, eq1 (a s) (ρ s) := by
    funext x
    exact aux_bridge_prod_eq1_update a ρ t x
  rw [hfun]
  have h := aux_bridge_polyDeg_mul (aux_bridge_eq1_polyDeg (a t))
    (aux_bridge_polyDeg_const 0
      (∏ s ∈ (Finset.univ : Finset (Fin m)).erase t, eq1 (a s) (ρ s)))
  exact aux_bridge_polyDeg_mono h (by norm_num)

/-! ### Evaluation on Boolean points -/

theorem aux_bridge_eqPoly_boolPt {m : ℕ} (x b : Fin m → Bool) :
    Packing.eqPoly x (boolPt b : Fin m → K) = if x = b then 1 else 0 := by
  unfold Packing.eqPoly
  by_cases h : x = b
  · rw [ite_eq_left h]
    refine Finset.prod_eq_one (fun i _ => ?_)
    have hi : x i = b i := congrFun h i
    cases hb : b i <;> simp [boolPt, hi, hb]
  · rw [ite_eq_right h]
    obtain ⟨i, hi⟩ := Function.ne_iff.mp h
    refine Finset.prod_eq_zero (Finset.mem_univ i) ?_
    cases hx : x i <;> cases hb : b i <;> simp_all [boolPt]

/-- (Extra) The Lagrange-form multilinear extension agrees with its table on Boolean points. -/
theorem mleTable_boolPt {m : ℕ} (T : (Fin m → Bool) → K) (b : Fin m → Bool) :
    Packing.mleTable T (boolPt b) = T b := by
  unfold Packing.mleTable
  rw [Finset.sum_eq_single b]
  · rw [aux_bridge_eqPoly_boolPt, ite_eq_left rfl, mul_one]
  · intro x _ hx
    rw [aux_bridge_eqPoly_boolPt, ite_eq_right hx, mul_zero]
  · intro hb
    exact absurd (Finset.mem_univ b) hb

theorem aux_bridge_prod_eq1_boolPt {m : ℕ} (a : Fin m → K) (b : Fin m → Bool) :
    ∏ t, eq1 (a t) (boolPt b t) = Packing.eqPoly b a := by
  unfold Packing.eqPoly
  refine Finset.prod_congr rfl (fun t _ => ?_)
  cases h : b t <;> simp [boolPt, eq1, h]

end AuxNoDec

section Lemmas
variable {K : Type*} [Field K] [DecidableEq K]

set_option linter.unusedSectionVars false

set_option linter.unusedVariables false in
/-- (Bridge) Lagrange form = coefficient form of the multilinear twin, for polynomials of degree `< 2^m`. -/
theorem mle_eq_mleTable {m : ℕ} (P : K[X]) (hP : P ∈ degreeLT K (2 ^ m)) (r : Fin m → K) :
    mle P (List.ofFn r) =
      Packing.mleTable (fun x : Fin m → Bool => mle P (List.ofFn (boolPt x))) r := by
  unfold mle Packing.mleTable
  exact aux_bridge_mleAux_ofFn (fun i => P.coeff i) r

/-- (Bridge) Every table on the Boolean cube is the Boolean table of a polynomial of degree `< 2^m`. -/
theorem exists_poly_of_table {m : ℕ} (T : (Fin m → Bool) → K) :
    ∃ P ∈ degreeLT K (2 ^ m), ∀ x : Fin m → Bool, mle P (List.ofFn (boolPt x)) = T x := by
  induction m with
  | zero =>
    refine ⟨Polynomial.C (T default), aux_bridge_C_mem_degreeLT _, ?_⟩
    intro x
    have hx : x = default := Subsingleton.elim _ _
    rw [hx]
    simp [mle, mleAux]
  | succ m ih =>
    obtain ⟨P0, hP0, h0⟩ := ih (fun y => T (Fin.cons false y))
    obtain ⟨P1, hP1, h1⟩ := ih (fun y => T (Fin.cons true y) - T (Fin.cons false y))
    refine ⟨expand K 2 P0 + Polynomial.X * expand K 2 P1,
      aux_bridge_mem_degreeLT_lift hP0 hP1, ?_⟩
    intro x
    obtain ⟨b, y, rfl⟩ : ∃ (b : Bool) (y : Fin m → Bool), x = Fin.cons b y :=
      ⟨x 0, Fin.tail x, (Fin.cons_self_tail x).symm⟩
    rw [aux_bridge_boolPt_cons, aux_bridge_ofFn_cons, mle_lift, h0, h1]
    cases b <;> simp

/-- (Bridge) The Lagrange-form multilinear extension has degree `≤ 1` in each coordinate. -/
theorem coordDeg_mleTable {m : ℕ} (T : (Fin m → Bool) → K) : CoordDeg 1 (Packing.mleTable T) := by
  intro t ρ
  exact aux_bridge_mleTable_update_polyDeg T t ρ

/-- (Bridge) The Phase-A polynomial has degree `≤ 2` in each coordinate. -/
theorem coordDeg_chainF {m : ℕ} {J : Type*} [Fintype J] (c : J → K) (p : J → Fin m → K)
    (T : J → (Fin m → Bool) → K) : CoordDeg 2 (chainF c p T) := by
  intro t ρ
  have h : aux_bridge_PolyDeg 2 (fun x => chainF c p T (Function.update ρ t x)) := by
    show aux_bridge_PolyDeg 2 (fun x => ∑ j, c j * ((∏ s, eq1 (p j s) (Function.update ρ t x s)) *
      Packing.mleTable (T j) (Function.update ρ t x)))
    refine aux_bridge_polyDeg_sum (fun j => ?_)
    have h1 := aux_bridge_polyDeg_mul (aux_bridge_prod_eq1_polyDeg (p j) ρ t)
      (aux_bridge_mleTable_update_polyDeg (T j) t ρ)
    have h2 := aux_bridge_polyDeg_mul (aux_bridge_polyDeg_const 0 (c j)) h1
    exact aux_bridge_polyDeg_mono h2 (by norm_num)
  exact h

/-- (Bridge) Its Boolean-cube sum is `Σ_j c_j T_j~(p_j)`. -/
theorem sum_bool_chainF {m : ℕ} {J : Type*} [Fintype J] (c : J → K) (p : J → Fin m → K)
    (T : J → (Fin m → Bool) → K) :
    ∑ b : Fin m → Bool, chainF c p T (boolPt b) = ∑ j, c j * Packing.mleTable (T j) (p j) := by
  have h : ∀ j, ∑ b : Fin m → Bool, c j * ((∏ t, eq1 (p j t) (boolPt b t)) *
      Packing.mleTable (T j) (boolPt b)) = c j * Packing.mleTable (T j) (p j) := by
    intro j
    have h2 : ∀ b : Fin m → Bool, c j * ((∏ t, eq1 (p j t) (boolPt b t)) *
        Packing.mleTable (T j) (boolPt b)) = c j * (T j b * Packing.eqPoly b (p j)) := by
      intro b
      rw [aux_bridge_prod_eq1_boolPt, mleTable_boolPt, mul_comm (Packing.eqPoly b (p j))]
    rw [Finset.sum_congr rfl (fun b _ => h2 b)]
    rw [Packing.mleTable, Finset.mul_sum]
  show ∑ b : Fin m → Bool, ∑ j, c j * ((∏ t, eq1 (p j t) (boolPt b t)) *
    Packing.mleTable (T j) (boolPt b)) = _
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl (fun j _ => h j)

end Lemmas

end Cprs

#print axioms Cprs.mle_eq_mleTable
#print axioms Cprs.exists_poly_of_table
#print axioms Cprs.coordDeg_mleTable
#print axioms Cprs.coordDeg_chainF
#print axioms Cprs.sum_bool_chainF
#print axioms Cprs.mleTable_boolPt
