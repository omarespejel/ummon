import Mathlib

/-!
# The conjugate-packing identity

CPRS packs `d = 4` consecutive base-field (`F = F_p`, `p = 2^31 - 1`) weights into one element of
`K = F_{p^4}` using an `F`-basis `b_0..b_3` of `K`.  If `ω(x) = Σ_i b_i · W_i(x)` with `W_i`
`F`-valued tables, the multilinear extension `ω~` over `K` satisfies, for every `r ∈ K^m`,

  `W_i~(r) = Σ_k (M⁻¹)_{i k} · σ^k( ω~( σ^(-k) r ) )`,  `M_{k i} = σ^k(b_i)`  (Moore matrix),

where `σ` generates `Gal(K/F)` (for CPRS: `σ(x) = x^p`).  Hence the verifier can recompute the
base-field evaluation `W~(r_lo, r_hi) = Σ_i eq(r_lo, i) W_i~(r_hi)` from the `d` evaluations
`ω~(σ^(-k) r_hi)`, `k < d`.
-/

open Finset

namespace Cprs.Packing

variable {F K : Type*} [Field F] [Field K] [Algebra F K]

/-- `eq(x, r) = ∏_i (if x_i then r_i else 1 - r_i)`. -/
def eqPoly {m : ℕ} (x : Fin m → Bool) (r : Fin m → K) : K :=
  ∏ i, if x i then r i else 1 - r i

/-- Multilinear extension of a table on the Boolean hypercube, evaluated at `r ∈ K^m`. -/
def mleTable {m : ℕ} (T : (Fin m → Bool) → K) (r : Fin m → K) : K :=
  ∑ x, T x * eqPoly x r

/-! ### Auxiliary lemmas for `conjugate_identity` -/

/-- Ring homomorphisms (in particular algebra automorphisms) commute with `eqPoly`. -/
theorem aux_map_eqPoly {H : Type*} [FunLike H K K] [RingHomClass H K K] (τ : H) {m : ℕ}
    (x : Fin m → Bool) (r : Fin m → K) :
    τ (eqPoly x r) = eqPoly x (fun l => τ (r l)) := by
  unfold eqPoly
  rw [map_prod]
  refine Finset.prod_congr rfl (fun i _ => ?_)
  by_cases h : x i = true
  · simp [h]
  · simp [h]

/-- Ring homomorphisms commute with `mleTable`, acting on both the table and the point. -/
theorem aux_map_mleTable {H : Type*} [FunLike H K K] [RingHomClass H K K] (τ : H) {m : ℕ}
    (T : (Fin m → Bool) → K) (r : Fin m → K) :
    τ (mleTable T r) = mleTable (fun x => τ (T x)) (fun l => τ (r l)) := by
  unfold mleTable
  rw [map_sum]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  rw [map_mul, aux_map_eqPoly τ]

/-- `mleTable` is `K`-linear in the table. -/
theorem aux_mleTable_pack {m d : ℕ} (c : Fin d → K) (T : Fin d → (Fin m → Bool) → K)
    (r : Fin m → K) :
    mleTable (fun x => ∑ j, c j * T j x) r = ∑ j, c j * mleTable (T j) r := by
  unfold mleTable
  simp_rw [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun j _ => Finset.sum_congr rfl (fun x _ => ?_))
  ring

/-- The `k`-th conjugate evaluation of the packed table, in terms of the `d` base-field
evaluations `W_j~(r)`: `σ^k ( ω~ (σ^(-k) r) ) = Σ_j σ^k(b_j) · W_j~(r)`. -/
theorem aux_conj_coord (d : ℕ) (b : Module.Basis (Fin d) F K) (τ : K ≃ₐ[F] K) {m : ℕ}
    (W : Fin d → (Fin m → Bool) → F) (r : Fin m → K) :
    τ (mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W j x)) (fun l => τ.symm (r l))) =
      ∑ j : Fin d, τ (b j) * mleTable (fun x => algebraMap F K (W j x)) r := by
  rw [aux_mleTable_pack, map_sum]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [map_mul, aux_map_mleTable τ]
  have h1 : (fun x => τ (algebraMap F K (W j x))) = fun x => algebraMap F K (W j x) := by
    funext x
    exact τ.commutes (W j x)
  have h2 : (fun l => τ (τ.symm (r l))) = r := by
    funext l
    exact τ.apply_symm_apply (r l)
  rw [h1, h2]

/-- **Conjugate identity (explicit form).**  Let `K/F` be a finite-dimensional extension of degree
`d` and `σ ∈ Aut(K/F)` of order `d`.  For every `F`-basis `b` of `K`, every `m`, every family of
`F`-valued tables `W_i` and every `r ∈ K^m`, the `i`-th base-field evaluation is recovered from the
`d` conjugate evaluations of the packed table `Σ_j b_j W_j` by the inverse of the Moore matrix
`(σ^k b_i)_{k,i}`:
`W_i~(r) = Σ_k (M⁻¹)_{i k} · σ^k(ω~(σ^(-k) r))`. -/
theorem conjugate_identity_explicit [FiniteDimensional F K] (d : ℕ) (hd : Module.finrank F K = d)
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K) (hσ : orderOf σ = d) :
    ∀ (m : ℕ) (W : Fin d → (Fin m → Bool) → F) (r : Fin m → K) (i : Fin d),
      mleTable (fun x => algebraMap F K (W i x)) r =
        ∑ k : Fin d,
          (Matrix.of fun (a c : Fin d) => (σ ^ (a : ℕ)) (b c) : Matrix (Fin d) (Fin d) K)⁻¹ i k *
          (σ ^ (k : ℕ)) (mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W j x))
            (fun l => ((σ ^ (k : ℕ)).symm (r l)))) := by
  classical
  -- the Moore matrix `M k i = σ^k (b i)`
  let M : Matrix (Fin d) (Fin d) K := Matrix.of fun k i => (σ ^ (k : ℕ)) (b i)
  -- the powers `σ^0, …, σ^(d-1)` are pairwise distinct algebra homomorphisms
  have hinj : Function.Injective
      (fun k : Fin d => ((σ ^ (k : ℕ) : K ≃ₐ[F] K) : K →ₐ[F] K)) := by
    intro k l hkl
    have h1 : σ ^ (k : ℕ) = σ ^ (l : ℕ) := AlgEquiv.coe_toAlgHom_injective hkl
    have hk : (k : ℕ) ∈ Set.Iio (orderOf σ) := by
      rw [Set.mem_Iio, hσ]; exact k.isLt
    have hl : (l : ℕ) ∈ Set.Iio (orderOf σ) := by
      rw [Set.mem_Iio, hσ]; exact l.isLt
    exact Fin.ext (pow_injOn_Iio_orderOf hk hl h1)
  -- Dedekind / Artin: they are `K`-linearly independent as `F`-linear maps `K → K`
  have hli : LinearIndependent K
      (fun k : Fin d => (((σ ^ (k : ℕ) : K ≃ₐ[F] K) : K →ₐ[F] K).toLinearMap)) :=
    (linearIndependent_algHom_toLinearMap F K K).comp _ hinj
  -- hence the rows of `M` are independent, and `M` is invertible
  have hunit : IsUnit M := by
    rw [← Matrix.linearIndependent_rows_iff_isUnit, Fintype.linearIndependent_iff]
    intro g hg
    have hlin : (∑ k, g k • (((σ ^ (k : ℕ) : K ≃ₐ[F] K) : K →ₐ[F] K).toLinearMap)) = 0 := by
      apply b.ext
      intro i
      have h := congrFun hg i
      simp only [Finset.sum_apply, Pi.smul_apply, Pi.zero_apply, smul_eq_mul] at h
      simpa [M, Matrix.row] using h
    exact Fintype.linearIndependent_iff.mp hli g hlin
  have hMinv : M⁻¹ * M = 1 := Matrix.nonsing_inv_mul M ((Matrix.isUnit_iff_isUnit_det M).mp hunit)
  intro m W r i
  -- the `d` base-field evaluations and the `d` conjugate evaluations of the packed table
  set w : Fin d → K := fun j => mleTable (fun x => algebraMap F K (W j x)) r with hw
  set y : Fin d → K := fun k =>
    (σ ^ (k : ℕ)) (mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W j x))
      (fun l => ((σ ^ (k : ℕ)).symm (r l)))) with hy
  -- `y = M w`
  have hyM : y = M.mulVec w := by
    funext k
    simp only [hy, hw, Matrix.mulVec, dotProduct, M, Matrix.of_apply]
    exact aux_conj_coord d b (σ ^ (k : ℕ)) W r
  -- hence `w = M⁻¹ y`
  have key : M⁻¹.mulVec y = w := by
    rw [hyM, Matrix.mulVec_mulVec, hMinv, Matrix.one_mulVec]
  have hi := congrFun key i
  simp only [Matrix.mulVec, dotProduct] at hi
  exact hi.symm

/-- **Conjugate identity.**  Existential form of `conjugate_identity_explicit`: there is a matrix
`M' ∈ K^{d×d}` (the inverse of the Moore matrix `(σ^k b_i)`) such that for every `m`, every family of
`F`-valued tables `W_i` and every `r ∈ K^m`, the `i`-th base-field evaluation is recovered from the
`d` conjugate evaluations of the packed table. -/
theorem conjugate_identity [FiniteDimensional F K] (d : ℕ) (hd : Module.finrank F K = d)
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K) (hσ : orderOf σ = d) :
    ∃ M' : Matrix (Fin d) (Fin d) K, ∀ (m : ℕ) (W : Fin d → (Fin m → Bool) → F) (r : Fin m → K)
      (i : Fin d),
      mleTable (fun x => algebraMap F K (W i x)) r =
        ∑ k : Fin d, M' i k *
          (σ ^ (k : ℕ)) (mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W j x))
            (fun l => ((σ ^ (k : ℕ)).symm (r l)))) :=
  ⟨(Matrix.of fun (a c : Fin d) => (σ ^ (a : ℕ)) (b c) : Matrix (Fin d) (Fin d) K)⁻¹,
    conjugate_identity_explicit d hd b σ hσ⟩

end Cprs.Packing

#print axioms Cprs.Packing.conjugate_identity_explicit
#print axioms Cprs.Packing.conjugate_identity
