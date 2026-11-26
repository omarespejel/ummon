import CprsFormal.Chain

/-!
# `chain_soundness` with the real packing matrix

`Chain.chain_soundness` takes the matrix `M'` that recovers the base-field evaluations from the
Frobenius-conjugate evaluations as a hypothesis (`hM`).  Here `M'` is the inverse of the Moore
matrix `(σ^k b_i)` of the packing basis, which is what the verifier uses, and `hM` is
`Packing.conjugate_identity_explicit`; the theorem then has no hypothesis about `M'`.
-/

namespace Cprs.Chain

/-- The matrix used by the verifier to recover the base-field evaluations: the inverse of the Moore
matrix `(σ^k b_i)_{k,i}` of the packing basis. -/
noncomputable abbrev mooreInv {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {d : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K) : Matrix (Fin d) (Fin d) K :=
  (Matrix.of fun (a c : Fin d) => (σ ^ (a : ℕ)) (b c) : Matrix (Fin d) (Fin d) K)⁻¹

open scoped Classical in
/-- **`chain_soundness` for the real packing matrix.**  Same statement as `chain_soundness` with the
matrix `M'` taken to be the inverse of the Moore matrix; the hypothesis `hM` is discharged by
`Packing.conjugate_identity_explicit` (which uses `hd` and `hσ`). -/
theorem chain_soundness_moore
    {F : Type*} [Field F] {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra F K]
    [FiniteDimensional F K]
    (d : ℕ) (hd : Module.finrank F K = d) (b : Module.Basis (Fin d) F K)
    (σ : K ≃ₐ[F] K) (hσ : orderOf σ = d)
    (m s : ℕ) (hs : 1 ≤ s) (ζ : K) (hζ : orderOf ζ = 2 ^ (m + s))
    (isEnd : ℕ → Bool) (t1 : ℕ) (ht1pos : 0 < t1) (ht1m : t1 ≤ m)
    (hend : ∀ t, isEnd t = true → t1 ≤ t ∧ t < m)
    (hend1 : t1 < m → isEnd t1 = true) (hno : ∀ t, 0 < t → t < t1 → isEnd t = false)
    (L' : ℕ) (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F)
    (lam : Fin (L' + 1) → Fin d → K) (rhi : Fin (L' + 1) → Fin m → K) (v : Fin (L' + 1) → K)
    (P : Fin (L' + 1) → Polynomial K) (hPdeg : ∀ a, P a ∈ Polynomial.degreeLT K (2 ^ m))
    (hP : ∀ (a : Fin (L' + 1)) (r : Fin m → K), mle (P a) (List.ofFn r) =
      Packing.mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W a j x)) r)
    (hfalse : ∃ a, v a ≠ ∑ j, lam a j *
      Packing.mleTable (fun x => algebraMap F K (W a j x)) (rhi a))
    (y : Fin (L' + 1) → Fin d → K) (gA : K → SCStrat K)
    (hgA : ∀ γ t pre, (gA γ t pre).natDegree ≤ 2)
    (z : K → (Fin m → K) → Fin (L' + 1) → K)
    (SB : K → (Fin m → K) → (Fin L' → K) → Strategy K m) (Q : ℕ) :
    ((Finset.univ.filter (fun p : K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1)) =>
        accepts σ (mooreInv b σ) ζ m s isEnd t1 lam rhi v P y gA z SB Q
          p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2)).card : ℚ) /
      (Fintype.card (K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1))) : ℚ)
    ≤ (((d * (L' + 1) - 1 : ℕ) : ℚ) + 2 * (m : ℚ) + (if 2 ≤ L' + 1 then (1 : ℚ) else 0) +
        ∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ)) / Fintype.card K
      + ((2 ^ s + 1) / 2 ^ (s + 1) : ℚ) ^ Q :=
  chain_soundness d hd b σ hσ (mooreInv b σ)
    (fun m' W' r i => Packing.conjugate_identity_explicit d hd b σ hσ m' W' r i)
    m s hs ζ hζ isEnd t1 ht1pos ht1m hend hend1 hno L' W lam rhi v P hPdeg hP hfalse y gA hgA z SB Q

#print axioms Cprs.Chain.chain_soundness_moore

end Cprs.Chain
