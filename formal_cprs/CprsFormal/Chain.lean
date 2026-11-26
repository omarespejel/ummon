import CprsFormal.Bridge
import CprsFormal.SumCheck
import CprsFormal.PhaseA
import CprsFormal.Packing
import CprsFormal.Layers
import CprsFormal.PhaseB

/-!
# The unified Model-H chain theorem `Cprs.Chain.chain_soundness`

Interactive soundness of the whole CPRS opening protocol (Phase 0 conjugate check, `γ`-batching,
Phase-A sum-check, `μ`-combination, Phase B with queries) for honest registration (Model H).
`chain_soundness` below states it; the proof composes the round-by-round lemmas of the other files.

Organisation of this file:
1. generic counting lemmas (`card_filter_prod_le`, `nested_count`);
2. field arithmetic (`mu_arith`, `final_arith_eq`);
3. definitions (`claim0`, `Estar`, `ystar`, `phaseAF`, `accepts`);
4. algebraic lemmas for Steps 0-3 of the proof;
5. the `μ`-step (`badMu_card`) and the Phase-B step (`phaseB_fibre_le`);
6. the theorem `chain_soundness`.
-/

namespace Cprs.Chain

/-! ### 1. Generic counting lemmas -/

/-- `Finset.mem_filter` for an arbitrary decidability instance (the instance is unified, not
synthesized). -/
theorem mem_filter_any {α : Type*} {p : α → Prop} {inst : DecidablePred p} {s : Finset α}
    {a : α} : a ∈ @Finset.filter α p inst s ↔ a ∈ s ∧ p a :=
  @Finset.mem_filter α p inst s a

open scoped Classical in
/-- Fibre-split counting bound: the accepted pairs are split over the first coordinate. -/
theorem card_filter_prod_le {U V : Type*} [Fintype U] [Fintype V]
    (Acc : U × V → Prop) (Bad : U → Prop) (b : ℚ) (hb : 0 ≤ b)
    (h : ∀ x, ¬ Bad x → ((Finset.univ.filter (fun y : V => Acc (x, y))).card : ℚ) ≤ b) :
    ((Finset.univ.filter Acc).card : ℚ) ≤
      ((Finset.univ.filter Bad).card : ℚ) * (Fintype.card V : ℚ) + (Fintype.card U : ℚ) * b := by
  have hfib : ((Finset.univ.filter Acc).card : ℚ)
      = ∑ x : U, ((Finset.univ.filter (fun y : V => Acc (x, y))).card : ℚ) := by
    simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
    rw [Fintype.sum_prod_type]
  have hpt : ∀ x : U, ((Finset.univ.filter (fun y : V => Acc (x, y))).card : ℚ)
      ≤ (if Bad x then (Fintype.card V : ℚ) else 0) + b := by
    intro x
    by_cases hx : Bad x
    · rw [if_pos hx]
      have h1 : (Finset.univ.filter (fun y : V => Acc (x, y))).card ≤ Fintype.card V := by
        calc _ ≤ (Finset.univ : Finset V).card := Finset.card_filter_le _ _
          _ = Fintype.card V := Finset.card_univ
      have h2 : ((Finset.univ.filter (fun y : V => Acc (x, y))).card : ℚ)
          ≤ (Fintype.card V : ℚ) := by exact_mod_cast h1
      linarith
    · rw [if_neg hx]
      have := h x hx
      linarith
  rw [hfib]
  calc ∑ x : U, ((Finset.univ.filter (fun y : V => Acc (x, y))).card : ℚ)
      ≤ ∑ x : U, ((if Bad x then (Fintype.card V : ℚ) else 0) + b) :=
        Finset.sum_le_sum (fun x _ => hpt x)
    _ = ((Finset.univ.filter Bad).card : ℚ) * (Fintype.card V : ℚ)
          + (Fintype.card U : ℚ) * b := by
      rw [Finset.sum_add_distrib, ← Finset.sum_filter, Finset.sum_const, Finset.sum_const,
        Finset.card_univ, nsmul_eq_mul, nsmul_eq_mul]

open scoped Classical in
/-- Three-level union bound over `X1 × X2 × X3 × Ω`. -/
theorem nested_count {X1 X2 X3 Ω : Type*} [Fintype X1] [Fintype X2] [Fintype X3] [Fintype Ω]
    (Acc : X1 → X2 → X3 → Ω → Prop)
    (Bad1 : X1 → Prop) (Bad2 : X1 → X2 → Prop) (Bad3 : X1 → X2 → X3 → Prop)
    (n1 n2 n3 n4 : ℚ) (hn2 : 0 ≤ n2) (hn3 : 0 ≤ n3) (hn4 : 0 ≤ n4)
    (hB1 : ((Finset.univ.filter Bad1).card : ℚ) ≤ n1)
    (hB2 : ∀ x1, ¬ Bad1 x1 → ((Finset.univ.filter (Bad2 x1)).card : ℚ) ≤ n2)
    (hB3 : ∀ x1 x2, ¬ Bad1 x1 → ¬ Bad2 x1 x2 →
      ((Finset.univ.filter (Bad3 x1 x2)).card : ℚ) ≤ n3)
    (hAcc : ∀ x1 x2 x3, ¬ Bad1 x1 → ¬ Bad2 x1 x2 → ¬ Bad3 x1 x2 x3 →
      ((Finset.univ.filter (fun w : Ω => Acc x1 x2 x3 w)).card : ℚ) ≤ n4) :
    ((Finset.univ.filter (fun p : X1 × X2 × X3 × Ω => Acc p.1 p.2.1 p.2.2.1 p.2.2.2)).card : ℚ)
      ≤ n1 * ((Fintype.card X2 : ℚ) * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ)))
        + (Fintype.card X1 : ℚ) * (n2 * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ))
          + (Fintype.card X2 : ℚ) * (n3 * (Fintype.card Ω : ℚ)
            + (Fintype.card X3 : ℚ) * n4)) := by
  have c3 : ∀ x1 x2, ¬ Bad1 x1 → ¬ Bad2 x1 x2 →
      ((Finset.univ.filter (fun p : X3 × Ω => Acc x1 x2 p.1 p.2)).card : ℚ)
        ≤ n3 * (Fintype.card Ω : ℚ) + (Fintype.card X3 : ℚ) * n4 := by
    intro x1 x2 h1 h2
    have key := card_filter_prod_le (fun p : X3 × Ω => Acc x1 x2 p.1 p.2) (Bad3 x1 x2) n4 hn4
      (fun x3 h3 => hAcc x1 x2 x3 h1 h2 h3)
    have hb := hB3 x1 x2 h1 h2
    have hΩ : (0 : ℚ) ≤ (Fintype.card Ω : ℚ) := Nat.cast_nonneg _
    have := mul_le_mul_of_nonneg_right hb hΩ
    linarith
  have c2 : ∀ x1, ¬ Bad1 x1 →
      ((Finset.univ.filter (fun p : X2 × X3 × Ω => Acc x1 p.1 p.2.1 p.2.2)).card : ℚ)
        ≤ n2 * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ))
          + (Fintype.card X2 : ℚ) * (n3 * (Fintype.card Ω : ℚ)
            + (Fintype.card X3 : ℚ) * n4) := by
    intro x1 h1
    have hb3 : (0 : ℚ) ≤ n3 * (Fintype.card Ω : ℚ) + (Fintype.card X3 : ℚ) * n4 :=
      add_nonneg (mul_nonneg hn3 (Nat.cast_nonneg _)) (mul_nonneg (Nat.cast_nonneg _) hn4)
    have key := card_filter_prod_le (fun p : X2 × X3 × Ω => Acc x1 p.1 p.2.1 p.2.2) (Bad2 x1)
      (n3 * (Fintype.card Ω : ℚ) + (Fintype.card X3 : ℚ) * n4) hb3
      (fun x2 h2 => c3 x1 x2 h1 h2)
    have hcard : (Fintype.card (X3 × Ω) : ℚ) = (Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ) := by
      rw [Fintype.card_prod, Nat.cast_mul]
    rw [hcard] at key
    have hb := hB2 x1 h1
    have hnn : (0 : ℚ) ≤ (Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ) :=
      mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    have := mul_le_mul_of_nonneg_right hb hnn
    linarith
  have hb2 : (0 : ℚ) ≤ n2 * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ))
      + (Fintype.card X2 : ℚ) * (n3 * (Fintype.card Ω : ℚ) + (Fintype.card X3 : ℚ) * n4) :=
    add_nonneg (mul_nonneg hn2 (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)))
      (mul_nonneg (Nat.cast_nonneg _) (add_nonneg (mul_nonneg hn3 (Nat.cast_nonneg _))
        (mul_nonneg (Nat.cast_nonneg _) hn4)))
  have key := card_filter_prod_le
    (fun p : X1 × X2 × X3 × Ω => Acc p.1 p.2.1 p.2.2.1 p.2.2.2) Bad1 _ hb2
    (fun x1 h1 => c2 x1 h1)
  have hcard : (Fintype.card (X2 × X3 × Ω) : ℚ)
      = (Fintype.card X2 : ℚ) * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ)) := by
    rw [Fintype.card_prod, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul]
  rw [hcard] at key
  have hnn : (0 : ℚ) ≤ (Fintype.card X2 : ℚ) * ((Fintype.card X3 : ℚ) * (Fintype.card Ω : ℚ)) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  have := mul_le_mul_of_nonneg_right hB1 hnn
  linarith

/-! ### 2. Field arithmetic -/

theorem mu_arith (L' : ℕ) (κ : ℚ) :
    κ * (if 2 ≤ L' + 1 then κ ^ (L' - 1) else 0)
      = (if 2 ≤ L' + 1 then (1 : ℚ) else 0) * κ ^ L' := by
  rcases Nat.eq_zero_or_pos L' with h | h
  · subst h
    simp
  · obtain ⟨n, rfl⟩ : ∃ n, L' = n + 1 := ⟨L' - 1, by omega⟩
    have h2 : 2 ≤ n + 1 + 1 := by omega
    have h3 : n + 1 - 1 = n := by omega
    rw [if_pos h2, if_pos h2, h3, pow_succ]
    ring

theorem final_arith_eq {κ cX1 cX2 cX3 cΩ cTot n1 n2 n3 n4 mR S βQ iota : ℚ}
    (hκ : 0 < κ) (hcX1 : cX1 = κ) (hcTot : cTot = cX1 * (cX2 * (cX3 * cΩ)))
    (hn2 : n2 = 2 * mR / κ * cX2) (hn4 : n4 = (S / κ + βQ) * cΩ)
    (hμ : κ * n3 = iota * cX3) :
    n1 * (cX2 * (cX3 * cΩ)) + cX1 * (n2 * (cX3 * cΩ) + cX2 * (n3 * cΩ + cX3 * n4))
      = ((n1 + 2 * mR + iota + S) / κ + βQ) * cTot := by
  have hκ0 : κ ≠ 0 := hκ.ne'
  have hn3 : n3 = iota * cX3 / κ := eq_div_of_mul_eq hκ0 ((mul_comm n3 κ).trans hμ)
  subst hcTot
  subst hn2
  subst hn4
  subst hn3
  rw [hcX1]
  first
    | (field_simp; ring)
    | field_simp
    | (simp only [div_eq_mul_inv]
       linear_combination (-(n1 * cX2 * cX3 * cΩ)) * (mul_inv_cancel₀ hκ0))

/-! ### 3. Definitions -/

/-- The batched Phase-0 claim `C₀ = Σ_{a<L, k<d} γ^(d·a+k) · y_{a,k}` (flattened index `d·a+k`). -/
def claim0 {K : Type*} [Field K] {L' : ℕ} (d : ℕ) (γ : K) (y : Fin (L' + 1) → Fin d → K) : K :=
  ∑ a : Fin (L' + 1), ∑ k : Fin d, γ ^ (d * (a : ℕ) + (k : ℕ)) * y a k

/-- `E_a = Σ_k γ^(d·a+k) · eq(σ^{-k} r_a, ρ)`: the weight of `z_a` in the final Phase-A identity. -/
def Estar {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {m L' : ℕ} (d : ℕ)
    (σ : K ≃ₐ[F] K) (rhi : Fin (L' + 1) → Fin m → K) (γ : K) (ρ : Fin m → K)
    (a : Fin (L' + 1)) : K :=
  ∑ k : Fin d, γ ^ (d * (a : ℕ) + (k : ℕ)) *
    ∏ t : Fin m, eq1 ((σ ^ (k : ℕ)).symm (rhi a t)) (ρ t)

/-- The honest conjugate evaluations `y*_{a,k} = ω_a~(σ^{-k} r_a)`. -/
noncomputable def ystar {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K) :
    Fin (L' + 1) → Fin d → K :=
  fun a k => Packing.mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W a j x))
    (fun l => (σ ^ (k : ℕ)).symm (rhi a l))

/-- The Phase-A sum-check polynomial `F_γ(ρ) = Σ_{a,k} γ^(d·a+k) · eq(σ^{-k} r_a, ρ) · ω_a~(ρ)`. -/
noncomputable def phaseAF {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K) (γ : K) :
    (Fin m → K) → K :=
  chainF (fun x : Fin (L' + 1) × Fin d => γ ^ (d * (x.1 : ℕ) + (x.2 : ℕ)))
    (fun x l => (σ ^ (x.2 : ℕ)).symm (rhi x.1 l))
    (fun x u => ∑ j : Fin d, b j * algebraMap F K (W x.1 j u))

/-- **The verifier's acceptance predicate**.
For the prover `(y, gA, z, SB)` and the verifier challenges `γ, ρ, μ', α` and queries `q`, the
verifier accepts iff all of the following hold (in this order):
1. (Phase 0) the conjugate check `v_a = Σ_k c_{a,k} σ^k(y_{a,k})` for every matrix `a`,
   `c_{a,k} = Σ_j λ_{a,j} M'_{j,k}`;
2. (Phase A, rounds) every sum-check round check passes, starting from the batched claim
   `claim0 = Σ_{a,k} γ^(d·a+k) y_{a,k}` (`scOk`);
3. (Phase A, final) `Σ_a z_a E_a` equals the last running claim (`scClaim`);
4. (Phase B, non-query part) `Accept0` for the Phase-B instance on the layers `D_t = ⟨ζ^(2^t)⟩`
   with virtual word `π_μ = Σ_a μ_a P_a` and initial claim `c₀ = Σ_a μ_a z_a`, `μ = (1, μ')`;
5. (queries) every query `q i ∈ D_{t₁}` passes the whole chain of consistency checks. -/
def accepts {F : Type*} [Field F] {K : Type*} [Field K] [DecidableEq K] [Algebra F K]
    {d : ℕ} (σ : K ≃ₐ[F] K) (M' : Matrix (Fin d) (Fin d) K)
    (ζ : K) (m s : ℕ) (isEnd : ℕ → Bool) (t1 : ℕ) {L' : ℕ}
    (lam : Fin (L' + 1) → Fin d → K) (rhi : Fin (L' + 1) → Fin m → K) (v : Fin (L' + 1) → K)
    (P : Fin (L' + 1) → Polynomial K)
    (y : Fin (L' + 1) → Fin d → K) (gA : K → SCStrat K)
    (z : K → (Fin m → K) → Fin (L' + 1) → K)
    (SB : K → (Fin m → K) → (Fin L' → K) → Strategy K m)
    (Q : ℕ) (γ : K) (ρ : Fin m → K) (μ' : Fin L' → K) (α : Fin m → K)
    (q : Fin Q → ↥(layerDomain ζ m s t1)) : Prop :=
  (∀ a, v a = ∑ k : Fin d, (∑ j, lam a j * M' j k) * (σ ^ (k : ℕ)) (y a k)) ∧
  scOk (gA γ) (claim0 d γ y) m ρ ∧
  ∑ a, z γ ρ a * Estar d σ rhi γ ρ a = scClaim (gA γ) (claim0 d γ y) m ρ ∧
  (PhaseBInst.ofRoot ζ m s (List.ofFn ρ) isEnd t1
      (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
      (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * z γ ρ a)).Accept0 (SB γ ρ μ') α ∧
  ∀ i, ((PhaseBInst.ofRoot ζ m s (List.ofFn ρ) isEnd t1
      (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
      (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * z γ ρ a)).final (SB γ ρ μ') α).chk (q i)

/-! ### 4. Algebraic lemmas (Steps 0-3 of the proof) -/

/-- Step 0: a passing conjugate check on a false instance forces `y ≠ y*`. -/
theorem y_ne_ystar {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K) (M' : Matrix (Fin d) (Fin d) K)
    (hM : ∀ (m' : ℕ) (W' : Fin d → (Fin m' → Bool) → F) (r : Fin m' → K) (i : Fin d),
      Packing.mleTable (fun x => algebraMap F K (W' i x)) r =
        ∑ k : Fin d, M' i k * (σ ^ (k : ℕ)) (Packing.mleTable
          (fun x => ∑ j : Fin d, b j * algebraMap F K (W' j x))
          (fun l => (σ ^ (k : ℕ)).symm (r l))))
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F)
    (lam : Fin (L' + 1) → Fin d → K) (rhi : Fin (L' + 1) → Fin m → K) (v : Fin (L' + 1) → K)
    (hfalse : ∃ a, v a ≠ ∑ j, lam a j *
      Packing.mleTable (fun x => algebraMap F K (W a j x)) (rhi a))
    (y : Fin (L' + 1) → Fin d → K)
    (hconj : ∀ a, v a = ∑ k : Fin d, (∑ j, lam a j * M' j k) * (σ ^ (k : ℕ)) (y a k)) :
    y ≠ ystar b σ W rhi := by
  intro hy
  obtain ⟨a0, ha0⟩ := hfalse
  apply ha0
  have hyk : ∀ k : Fin d, y a0 k = Packing.mleTable
      (fun x => ∑ j : Fin d, b j * algebraMap F K (W a0 j x))
      (fun l => (σ ^ (k : ℕ)).symm (rhi a0 l)) := fun k => congrFun (congrFun hy a0) k
  have hj : ∀ j : Fin d, Packing.mleTable (fun x => algebraMap F K (W a0 j x)) (rhi a0) =
      ∑ k : Fin d, M' j k * (σ ^ (k : ℕ)) (y a0 k) := by
    intro j
    rw [hM m (W a0) (rhi a0) j]
    refine Finset.sum_congr rfl (fun k _ => ?_)
    rw [hyk k]
  rw [hconj a0]
  calc ∑ k : Fin d, (∑ j, lam a0 j * M' j k) * (σ ^ (k : ℕ)) (y a0 k)
      = ∑ k : Fin d, ∑ j, lam a0 j * (M' j k * (σ ^ (k : ℕ)) (y a0 k)) := by
        refine Finset.sum_congr rfl (fun k _ => ?_)
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl (fun j _ => by ring)
    _ = ∑ j, ∑ k : Fin d, lam a0 j * (M' j k * (σ ^ (k : ℕ)) (y a0 k)) := Finset.sum_comm
    _ = ∑ j, lam a0 j * Packing.mleTable (fun x => algebraMap F K (W a0 j x)) (rhi a0) := by
        refine Finset.sum_congr rfl (fun j _ => ?_)
        rw [hj j, Finset.mul_sum]

theorem aux_flat_val {L' d : ℕ} (a : Fin (L' + 1)) (k : Fin d) :
    ((finProdFinEquiv (a, k) : Fin ((L' + 1) * d)) : ℕ) = d * (a : ℕ) + (k : ℕ) := by
  first
    | (rw [finProdFinEquiv_apply_val]
       show (k : ℕ) + d * (a : ℕ) = d * (a : ℕ) + (k : ℕ)
       exact Nat.add_comm _ _)
    | (simp [finProdFinEquiv]; ring)
    | (simp [finProdFinEquiv]; omega)

/-- Step 1: at most `d·L - 1` values of `γ` make the batched claims of `y ≠ y*` agree. -/
theorem claim0_bad_card {K : Type*} [Field K] [Fintype K] [DecidableEq K] {d L' : ℕ}
    (y ys : Fin (L' + 1) → Fin d → K) (hne : y ≠ ys) :
    (Finset.univ.filter (fun γ : K => claim0 d γ y = claim0 d γ ys)).card ≤ d * (L' + 1) - 1 := by
  classical
  let e : Fin (L' + 1) × Fin d ≃ Fin ((L' + 1) * d) := finProdFinEquiv
  let yf : Fin ((L' + 1) * d) → K := fun j => y (e.symm j).1 (e.symm j).2
  let ysf : Fin ((L' + 1) * d) → K := fun j => ys (e.symm j).1 (e.symm j).2
  have hne' : yf ≠ ysf := by
    intro h
    apply hne
    funext a k
    have h1 := congrFun h (e (a, k))
    first
      | (simp only [yf, ysf, Equiv.symm_apply_apply] at h1; exact h1)
      | simpa [yf, ysf] using h1
  have hroots := gamma_roots yf ysf hne'
  have hsum : ∀ γ : K, ∑ j : Fin ((L' + 1) * d), γ ^ (j : ℕ) * (yf j - ysf j)
      = claim0 d γ y - claim0 d γ ys := by
    intro γ
    calc ∑ j : Fin ((L' + 1) * d), γ ^ (j : ℕ) * (yf j - ysf j)
        = ∑ x : Fin (L' + 1) × Fin d,
            γ ^ ((e x : Fin ((L' + 1) * d)) : ℕ) * (yf (e x) - ysf (e x)) :=
          (Equiv.sum_comp e (fun j : Fin ((L' + 1) * d) => γ ^ (j : ℕ) * (yf j - ysf j))).symm
      _ = ∑ x : Fin (L' + 1) × Fin d,
            γ ^ (d * (x.1 : ℕ) + (x.2 : ℕ)) * (y x.1 x.2 - ys x.1 x.2) := by
          refine Finset.sum_congr rfl (fun x _ => ?_)
          have h1 : yf (e x) = y x.1 x.2 := by simp only [yf, Equiv.symm_apply_apply]
          have h2 : ysf (e x) = ys x.1 x.2 := by simp only [ysf, Equiv.symm_apply_apply]
          have h3 : ((e x : Fin ((L' + 1) * d)) : ℕ) = d * (x.1 : ℕ) + (x.2 : ℕ) := by
            obtain ⟨a, k⟩ := x
            exact aux_flat_val a k
          rw [h1, h2, h3]
      _ = ∑ a : Fin (L' + 1), ∑ k : Fin d, γ ^ (d * (a : ℕ) + (k : ℕ)) * (y a k - ys a k) :=
          Fintype.sum_prod_type _
      _ = claim0 d γ y - claim0 d γ ys := by
          unfold claim0
          rw [← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl (fun a _ => ?_)
          rw [← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl (fun k _ => ?_)
          ring
  calc (Finset.univ.filter (fun γ : K => claim0 d γ y = claim0 d γ ys)).card
      ≤ (Finset.univ.filter (fun γ : K =>
          ∑ j : Fin ((L' + 1) * d), γ ^ (j : ℕ) * (yf j - ysf j) = 0)).card := by
        apply Finset.card_le_card
        intro γ hγ
        rw [Finset.mem_filter] at hγ ⊢
        refine ⟨hγ.1, ?_⟩
        rw [hsum γ, hγ.2, sub_self]
    _ ≤ (L' + 1) * d - 1 := hroots
    _ = d * (L' + 1) - 1 := by rw [Nat.mul_comm]

/-- Step 2 (part): the Boolean-cube sum of `F_γ` is the honest batched claim. -/
theorem sum_bool_phaseAF {F : Type*} [Field F] {K : Type*} [Field K] [DecidableEq K]
    [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K) (γ : K) :
    ∑ u : Fin m → Bool, phaseAF b σ W rhi γ (boolPt u) = claim0 d γ (ystar b σ W rhi) := by
  have h := sum_bool_chainF (fun x : Fin (L' + 1) × Fin d => γ ^ (d * (x.1 : ℕ) + (x.2 : ℕ)))
    (fun x l => (σ ^ (x.2 : ℕ)).symm (rhi x.1 l))
    (fun x u => ∑ j : Fin d, b j * algebraMap F K (W x.1 j u))
  refine h.trans ?_
  rw [Fintype.sum_prod_type]
  first | rfl | (simp only [claim0, ystar]) | (unfold claim0 ystar; rfl)

theorem phaseAF_coordDeg {F : Type*} [Field F] {K : Type*} [Field K] [DecidableEq K]
    [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K) (γ : K) :
    CoordDeg 2 (phaseAF b σ W rhi γ) :=
  coordDeg_chainF _ _ _

/-- Step 3 (part): `F_γ(ρ) = Σ_a E_a · ω_a~(ρ)`. -/
theorem phaseAF_eq {F : Type*} [Field F] {K : Type*} [Field K] [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K)
    (γ : K) (ρ : Fin m → K) :
    phaseAF b σ W rhi γ ρ = ∑ a : Fin (L' + 1), Estar d σ rhi γ ρ a *
      Packing.mleTable (fun u => ∑ j : Fin d, b j * algebraMap F K (W a j u)) ρ := by
  show ∑ x : Fin (L' + 1) × Fin d, γ ^ (d * (x.1 : ℕ) + (x.2 : ℕ)) *
      ((∏ t : Fin m, eq1 ((σ ^ (x.2 : ℕ)).symm (rhi x.1 t)) (ρ t)) *
        Packing.mleTable (fun u => ∑ j : Fin d, b j * algebraMap F K (W x.1 j u)) ρ) = _
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  unfold Estar
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  try dsimp only
  ring

/-- Step 3: if the final Phase-A identity holds but the running claim is not `F_γ(ρ)`, then the
prover's `z` differs from the honest `z*_a = mle (P a) ρ`. -/
theorem exists_z_ne {F : Type*} [Field F] {K : Type*} [Field K] [DecidableEq K] [Algebra F K]
    {d m L' : ℕ} (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K)
    (P : Fin (L' + 1) → Polynomial K)
    (hP : ∀ (a : Fin (L' + 1)) (r : Fin m → K), mle (P a) (List.ofFn r) =
      Packing.mleTable (fun x => ∑ j : Fin d, b j * algebraMap F K (W a j x)) r)
    (γ : K) (ρ : Fin m → K) (zz : Fin (L' + 1) → K) (c : K)
    (hid : ∑ a, zz a * Estar d σ rhi γ ρ a = c) (hne : c ≠ phaseAF b σ W rhi γ ρ) :
    ∃ a, zz a ≠ mle (P a) (List.ofFn ρ) := by
  by_contra hcon
  push_neg at hcon
  apply hne
  rw [← hid, phaseAF_eq b σ W rhi γ ρ]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  rw [hcon a, hP a ρ]
  exact mul_comm _ _

open scoped Classical in
/-- Step 2: for `γ` with `claim0 ≠ S*`, at most `2m|K|^(m-1)` challenge vectors `ρ` pass all
round checks and end on `F_γ(ρ)`. -/
theorem rho_bad_card {F : Type*} [Field F] {K : Type*} [Field K] [Fintype K] [DecidableEq K]
    [Algebra F K] {d m L' : ℕ}
    (b : Module.Basis (Fin d) F K) (σ : K ≃ₐ[F] K)
    (W : Fin (L' + 1) → Fin d → (Fin m → Bool) → F) (rhi : Fin (L' + 1) → Fin m → K)
    (y : Fin (L' + 1) → Fin d → K) (gA : K → SCStrat K)
    (hgA : ∀ γ t pre, (gA γ t pre).natDegree ≤ 2) (γ : K)
    (hne : claim0 d γ y ≠ claim0 d γ (ystar b σ W rhi)) :
    ((Finset.univ.filter (fun ρ : Fin m → K => scOk (gA γ) (claim0 d γ y) m ρ ∧
        scClaim (gA γ) (claim0 d γ y) m ρ = phaseAF b σ W rhi γ ρ)).card : ℚ)
      ≤ (2 * (m : ℚ) / Fintype.card K) * (Fintype.card K : ℚ) ^ m := by
  have hfalse : ∑ u : Fin m → Bool, phaseAF b σ W rhi γ (boolPt u) ≠ claim0 d γ y := by
    rw [sum_bool_phaseAF b σ W rhi γ]
    exact fun h => hne h.symm
  have hsc := sumcheck_sound (phaseAF b σ W rhi γ) (phaseAF_coordDeg b σ W rhi γ)
    (claim0 d γ y) hfalse (gA γ) (hgA γ)
  refine le_trans (Nat.cast_le.mpr (Finset.card_le_card ?_)) hsc
  intro ρ hρ
  rw [Finset.mem_filter] at hρ ⊢
  exact hρ

/-! ### 5. The `μ`-step and the Phase-B step -/

open scoped Classical in
/-- Step 4: if `z ≠ z*`, the state relation `R_0` holds for at most `|K|^(L-2)` of the `|K|^(L-1)`
vectors `μ = (1, μ')` when `L ≥ 2`, and for none when `L = 1`. -/
theorem badMu_card {K : Type*} [Field K] [DecidableEq K] [Fintype K] (D : Finset K) (k : ℕ)
    (β : ℚ) (hrate : (k : ℚ) ≤ (2 * β - 1) * D.card) {L' : ℕ}
    (P : Fin (L' + 1) → Polynomial K) (hP : ∀ a, P a ∈ Polynomial.degreeLT K k)
    (ρl : List K) (zz : Fin (L' + 1) → K) (hz : ∃ a, zz a ≠ mle (P a) ρl) :
    ((Finset.univ.filter (fun μ' : Fin L' → K =>
        Rel D k β ρl (fun _ => 1)
          (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
          (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * zz a) 1)).card : ℚ)
      ≤ if 2 ≤ L' + 1 then (Fintype.card K : ℚ) ^ (L' - 1) else 0 := by
  rcases Nat.eq_zero_or_pos L' with hL0 | hLpos
  · -- `L' = 0`: the filter is empty
    have hL : ¬ (2 ≤ L' + 1) := by omega
    rw [if_neg hL]
    have h0 : ∀ a : Fin (L' + 1), a = 0 := by
      intro a
      apply Fin.ext
      have h1 := a.isLt
      have h2 : ((0 : Fin (L' + 1)) : ℕ) = 0 := by first | rfl | simp
      omega
    have hs : ∀ f : Fin (L' + 1) → K, ∑ a, f a = f 0 := by
      intro f
      rw [Finset.sum_eq_single (0 : Fin (L' + 1))]
      · intro a _ ha
        exact absurd (h0 a) ha
      · intro h
        exact absurd (Finset.mem_univ _) h
    obtain ⟨a0, ha0⟩ := hz
    refine le_trans (Nat.cast_le.mpr
      (Finset.card_le_card (t := (∅ : Finset (Fin L' → K))) ?_)) ?_
    · intro μ' hμ'
      rw [Finset.mem_filter] at hμ'
      have hkey := aux_phaseA_mu_key D k β hrate P hP ρl zz (Fin.cons 1 μ') hμ'.2
      rw [hs] at hkey
      have hc0 : (Fin.cons 1 μ' : Fin (L' + 1) → K) 0 = 1 := by
        first | exact Fin.cons_zero _ _ | exact Fin.cons_zero | simp
      rw [hc0, one_mul, sub_eq_zero] at hkey
      exact (ha0 (by rw [h0 a0]; exact hkey)).elim
    · simp
  · -- `L' = n + 1`
    obtain ⟨n, rfl⟩ : ∃ n, L' = n + 1 := ⟨L' - 1, by omega⟩
    have hL : 2 ≤ n + 1 + 1 := by omega
    rw [if_pos hL]
    have hstep := mu_step D k β hrate (L := n + 1 + 1) hL P hP ρl zz hz
    have e : n + 1 + 1 - 2 = n := by omega
    rw [e] at hstep
    have e2 : n + 1 - 1 = n := by omega
    rw [e2]
    have hcast : ((Fintype.card K ^ n : ℕ) : ℚ) = (Fintype.card K : ℚ) ^ n := by
      push_cast
      rfl
    rw [← hcast]
    refine Nat.cast_le.mpr (le_trans (Finset.card_le_card_of_injOn
      (fun μ' : Fin (n + 1) → K => (Fin.cons 1 μ' : Fin (n + 1 + 1) → K)) ?_ ?_) hstep)
    · intro μ' hμ'
      rw [Finset.mem_coe, Finset.mem_filter] at hμ' ⊢
      refine ⟨Finset.mem_univ _, ?_, hμ'.2⟩
      first | exact Fin.cons_zero _ _ | exact Fin.cons_zero | simp
    · intro μ₁ _ μ₂ _ h
      funext i
      have h' := congrFun h i.succ
      simpa using h'

theorem hrate_lemma {K : Type*} [Field K] [DecidableEq K] (ζ : K) (m s : ℕ)
    (hζ : orderOf ζ = 2 ^ (m + s)) (β : ℚ) (hβ : β = (2 ^ s + 1) / 2 ^ (s + 1)) :
    ((2 ^ m : ℕ) : ℚ) ≤ (2 * β - 1) * ((layerDomain ζ m s 0).card : ℚ) := by
  have hD0 : (layerDomain ζ m s 0).card = 2 ^ (m + s) := by
    have h := layerDomain_card ζ m s hζ (t := 0) (Nat.zero_le _)
    simpa using h
  rw [hD0]
  have h2 : (2 * β - 1) * ((2 ^ (m + s) : ℕ) : ℚ) = ((2 ^ m : ℕ) : ℚ) := by
    rw [hβ]
    push_cast
    first
      | (field_simp; ring)
      | field_simp
  exact le_of_eq h2.symm

open scoped Classical in
/-- Step 5: Theorem B′ (`PhaseBInst.soundness_unconditional`) on the instance
`PhaseBInst.ofRoot`, in product form. -/
theorem phaseB_fibre_le {K : Type} [Field K] [Fintype K] [DecidableEq K]
    (ζ : K) (m s : ℕ) (hζ : orderOf ζ = 2 ^ (m + s)) (hs : 1 ≤ s)
    (isEnd : ℕ → Bool) (t1 : ℕ) (ht1pos : 0 < t1) (ht1m : t1 ≤ m)
    (hend : ∀ t, isEnd t = true → t1 ≤ t ∧ t < m)
    (hend1 : t1 < m → isEnd t1 = true) (hno : ∀ t, 0 < t → t < t1 → isEnd t = false)
    (ρ : Fin m → K) (π : K → K) (c0 : K) (β : ℚ) (hβ : β = (2 ^ s + 1) / 2 ^ (s + 1))
    (hfalse : ¬ Rel (layerDomain ζ m s 0) (2 ^ m) β (List.ofFn ρ) (fun _ => 1) π c0 1)
    (S : Strategy K m) (Q : ℕ) :
    ((Finset.univ.filter (fun p : (Fin m → K) × (Fin Q → ↥(layerDomain ζ m s t1)) =>
        (PhaseBInst.ofRoot ζ m s (List.ofFn ρ) isEnd t1 π c0).Accept0 S p.1 ∧
          ∀ i, ((PhaseBInst.ofRoot ζ m s (List.ofFn ρ) isEnd t1 π c0).final S p.1).chk
            (p.2 i))).card : ℚ)
      ≤ ((∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ)) / Fintype.card K
          + β ^ Q)
        * ((Fintype.card K : ℚ) ^ m * ((layerDomain ζ m s t1).card : ℚ) ^ Q) := by
  have hWF := PhaseBInst.ofRoot_wf ζ m s hζ hs (List.ofFn ρ) List.length_ofFn isEnd t1 ht1pos
    ht1m hend hend1 hno π c0
  have h := PhaseBInst.soundness_unconditional
    (PhaseBInst.ofRoot ζ m s (List.ofFn ρ) isEnd t1 π c0) hWF β hβ hfalse S Q
  have hκ : (0 : ℚ) < (Fintype.card K : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(0 : K)⟩
  have hD : (0 : ℚ) < ((layerDomain ζ m s t1).card : ℚ) := by
    have hc := layerDomain_card ζ m s hζ (t := t1) (by omega)
    rw [hc]
    positivity
  have hpos : (0 : ℚ) < (Fintype.card K : ℚ) ^ m * ((layerDomain ζ m s t1).card : ℚ) ^ Q :=
    mul_pos (pow_pos hκ _) (pow_pos hD _)
  first
    | exact (div_le_iff₀ hpos).mp h
    | (convert (div_le_iff₀ hpos).mp h using 2)

/-! ### 6. The theorem -/

open scoped Classical in
/-- **Interactive soundness of the CPRS opening protocol (Model H).**

Setting: `K/F` finite extension of degree `d` with `F`-basis `b` (packing), `σ` an `F`-automorphism
of `K` of order `d` (Frobenius); `ζ ∈ K` of order `2^(m+s)` (layers `D_t = ⟨ζ^(2^t)⟩`); `isEnd`,
`t1` the block structure of Phase B; `L = L'+1` registered matrices with base-field sub-tables
`W a j : {0,1}^m → F`, packed table `ω_a = Σ_j b_j W_{a,j}`; public weights `lam`, points `rhi`,
claims `v`.  Model H: `P a` has degree `< 2^m` and `P~_a = ω_a~` (`hP`).  The instance is false:
some claim `v a` differs from `Σ_j lam a j · W_{a,j}~(rhi a)`.

For every deterministic adaptive prover `(y, gA, z, SB)` (Phase-0 vector, Phase-A round polynomials
of degree `≤ 2`, final claims `z`, Phase-B strategy), the fraction of tuples
`(γ, ρ, μ', α, queries)` (uniform in `K × K^m × K^(L-1) × K^m × D_{t1}^Q`) accepted by the verifier
`accepts` is at most
`((d·L - 1) + 2m + [L ≥ 2] + Σ_{t<m} |D_{t+1}|)/|K| + β^Q`, `β = (2^s+1)/2^(s+1)`.
(`d·L ≥ 1`, so the natural-number subtraction is exact.)

Hypotheses: `hM` says that `M'` is a matrix recovering the base-field evaluations from the
conjugate evaluations (the conclusion of `Packing.conjugate_identity`, which shows that such an
`M'`, the inverse of the Moore matrix, exists whenever `hd` and `hσ` hold); `hd`, `hσ` document the
setting and are not used by the proof. -/
theorem chain_soundness
    {F : Type*} [Field F] {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra F K]
    [FiniteDimensional F K]
    (d : ℕ) (hd : Module.finrank F K = d) (b : Module.Basis (Fin d) F K)
    (σ : K ≃ₐ[F] K) (hσ : orderOf σ = d)
    (M' : Matrix (Fin d) (Fin d) K)
    (hM : ∀ (m' : ℕ) (W' : Fin d → (Fin m' → Bool) → F) (r : Fin m' → K) (i : Fin d),
      Packing.mleTable (fun x => algebraMap F K (W' i x)) r =
        ∑ k : Fin d, M' i k * (σ ^ (k : ℕ)) (Packing.mleTable
          (fun x => ∑ j : Fin d, b j * algebraMap F K (W' j x))
          (fun l => (σ ^ (k : ℕ)).symm (r l))))
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
        accepts σ M' ζ m s isEnd t1 lam rhi v P y gA z SB Q
          p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2)).card : ℚ) /
      (Fintype.card (K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1))) : ℚ)
    ≤ (((d * (L' + 1) - 1 : ℕ) : ℚ) + 2 * (m : ℚ) + (if 2 ≤ L' + 1 then (1 : ℚ) else 0) +
        ∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ)) / Fintype.card K
      + ((2 ^ s + 1) / 2 ^ (s + 1) : ℚ) ^ Q := by
  have hκ : (0 : ℚ) < (Fintype.card K : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(0 : K)⟩
  have hDpos : (0 : ℚ) < ((layerDomain ζ m s t1).card : ℚ) := by
    have h := layerDomain_card ζ m s hζ (t := t1) (by omega)
    rw [h]
    positivity
  obtain ⟨β, hβ⟩ : ∃ β : ℚ, β = (2 ^ s + 1) / 2 ^ (s + 1) := ⟨_, rfl⟩
  have hβ0 : 0 < β := by rw [hβ]; positivity
  rw [← hβ]
  have hX2 : (Fintype.card (Fin m → K) : ℚ) = (Fintype.card K : ℚ) ^ m := by simp
  have hX3 : (Fintype.card (Fin L' → K) : ℚ) = (Fintype.card K : ℚ) ^ L' := by simp
  have hΩc : (Fintype.card ((Fin m → K) × (Fin Q → ↥(layerDomain ζ m s t1))) : ℚ) =
      (Fintype.card K : ℚ) ^ m * ((layerDomain ζ m s t1).card : ℚ) ^ Q := by simp
  have hcTotEq : (Fintype.card (K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1))) : ℚ) =
      (Fintype.card K : ℚ) * ((Fintype.card K : ℚ) ^ m * ((Fintype.card K : ℚ) ^ L' *
        ((Fintype.card K : ℚ) ^ m * ((layerDomain ζ m s t1).card : ℚ) ^ Q))) := by
    first
      | (simp; done)
      | (simp; ring)
  have hpos : (0 : ℚ) < (Fintype.card (K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1))) : ℚ) := by
    rw [hcTotEq]
    exact mul_pos hκ (mul_pos (pow_pos hκ _) (mul_pos (pow_pos hκ _)
      (mul_pos (pow_pos hκ _) (pow_pos hDpos _))))
  rw [div_le_iff₀ hpos]
  by_cases hconj : ∀ a, v a = ∑ k : Fin d, (∑ j, lam a j * M' j k) * (σ ^ (k : ℕ)) (y a k)
  · -- the Phase-0 check passes: `y ≠ y*`
    have hy : y ≠ ystar b σ W rhi := y_ne_ystar b σ M' hM W lam rhi v hfalse y hconj
    have hrate := hrate_lemma ζ m s hζ β hβ
    obtain ⟨n2, hn2def⟩ : ∃ n2 : ℚ,
        n2 = (2 * (m : ℚ) / Fintype.card K) * (Fintype.card K : ℚ) ^ m := ⟨_, rfl⟩
    obtain ⟨n3, hn3def⟩ : ∃ n3 : ℚ,
        n3 = if 2 ≤ L' + 1 then (Fintype.card K : ℚ) ^ (L' - 1) else 0 := ⟨_, rfl⟩
    obtain ⟨n4, hn4def⟩ : ∃ n4 : ℚ,
        n4 = ((∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ)) /
              Fintype.card K + β ^ Q) *
            ((Fintype.card K : ℚ) ^ m * ((layerDomain ζ m s t1).card : ℚ) ^ Q) := ⟨_, rfl⟩
    have hn2 : 0 ≤ n2 := by rw [hn2def]; positivity
    have hn3 : 0 ≤ n3 := by rw [hn3def]; positivity
    have hn4 : 0 ≤ n4 := by
      rw [hn4def]
      exact mul_nonneg (add_nonneg (div_nonneg (Finset.sum_nonneg
        (fun _ _ => Nat.cast_nonneg _)) hκ.le) (pow_nonneg hβ0.le Q))
        (mul_nonneg (pow_nonneg hκ.le _) (pow_nonneg hDpos.le _))
    refine le_trans (nested_count (X1 := K) (X2 := Fin m → K) (X3 := Fin L' → K)
      (Ω := (Fin m → K) × (Fin Q → ↥(layerDomain ζ m s t1)))
      (fun γ ρ μ' ω => accepts σ M' ζ m s isEnd t1 lam rhi v P y gA z SB Q γ ρ μ' ω.1 ω.2)
      (fun γ => claim0 d γ y = claim0 d γ (ystar b σ W rhi))
      (fun γ ρ => scOk (gA γ) (claim0 d γ y) m ρ ∧
        scClaim (gA γ) (claim0 d γ y) m ρ = phaseAF b σ W rhi γ ρ)
      (fun γ ρ μ' => (scOk (gA γ) (claim0 d γ y) m ρ ∧
          ∑ a, z γ ρ a * Estar d σ rhi γ ρ a = scClaim (gA γ) (claim0 d γ y) m ρ) ∧
        Rel (layerDomain ζ m s 0) (2 ^ m) β (List.ofFn ρ) (fun _ => 1)
          (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
          (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * z γ ρ a) 1)
      ((d * (L' + 1) - 1 : ℕ) : ℚ) n2 n3 n4 hn2 hn3 hn4 ?_ ?_ ?_ ?_) (le_of_eq ?_)
    · -- hB1: few bad `γ`
      have h := claim0_bad_card y (ystar b σ W rhi) hy
      refine Nat.cast_le.mpr (le_trans (Finset.card_le_card ?_) h)
      intro γ hγ
      rw [mem_filter_any] at hγ ⊢
      exact hγ
    · -- hB2: few bad `ρ`
      intro γ hγ
      have hγ' : claim0 d γ y ≠ claim0 d γ (ystar b σ W rhi) := hγ
      have h := rho_bad_card b σ W rhi y gA hgA γ hγ'
      rw [hn2def]
      refine le_trans (Nat.cast_le.mpr (Finset.card_le_card ?_)) h
      intro ρ hρ
      rw [mem_filter_any] at hρ ⊢
      exact hρ
    · -- hB3: few bad `μ'`
      intro γ ρ hb1 hb2
      by_cases hpass : scOk (gA γ) (claim0 d γ y) m ρ ∧
          ∑ a, z γ ρ a * Estar d σ rhi γ ρ a = scClaim (gA γ) (claim0 d γ y) m ρ
      · have hne : scClaim (gA γ) (claim0 d γ y) m ρ ≠ phaseAF b σ W rhi γ ρ :=
          fun h => hb2 ⟨hpass.1, h⟩
        have hz := exists_z_ne b σ W rhi P hP γ ρ (z γ ρ) _ hpass.2 hne
        have hbm := badMu_card (layerDomain ζ m s 0) (2 ^ m) β hrate P hPdeg (List.ofFn ρ)
          (z γ ρ) hz
        rw [hn3def]
        refine le_trans (Nat.cast_le.mpr (Finset.card_le_card ?_)) hbm
        intro μ' hμ'
        rw [mem_filter_any] at hμ' ⊢
        exact ⟨hμ'.1, hμ'.2.2⟩
      · refine le_trans (Nat.cast_le.mpr
          (Finset.card_le_card (t := (∅ : Finset (Fin L' → K))) ?_)) ?_
        · intro μ' hμ'
          rw [mem_filter_any] at hμ'
          exact (hpass hμ'.2.1).elim
        · rw [Finset.card_empty, Nat.cast_zero]
          exact hn3
    · -- hAcc: Phase B
      intro γ ρ μ' hb1 hb2 hb3
      by_cases hpass : scOk (gA γ) (claim0 d γ y) m ρ ∧
          ∑ a, z γ ρ a * Estar d σ rhi γ ρ a = scClaim (gA γ) (claim0 d γ y) m ρ
      · have hnot : ¬ Rel (layerDomain ζ m s 0) (2 ^ m) β (List.ofFn ρ) (fun _ => 1)
            (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
            (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * z γ ρ a) 1 :=
          fun h => hb3 ⟨hpass, h⟩
        have hB := phaseB_fibre_le ζ m s hζ hs isEnd t1 ht1pos ht1m hend hend1 hno ρ
          (fun x => ∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * (P a).eval x)
          (∑ a, (Fin.cons 1 μ' : Fin (L' + 1) → K) a * z γ ρ a) β hβ hnot (SB γ ρ μ') Q
        rw [hn4def]
        refine le_trans (Nat.cast_le.mpr (Finset.card_le_card ?_)) hB
        intro w hw
        rw [mem_filter_any] at hw ⊢
        refine ⟨Finset.mem_univ _, ?_⟩
        have hacc := hw.2
        unfold accepts at hacc
        obtain ⟨_, _, _, hB0, hC⟩ := hacc
        exact ⟨hB0, hC⟩
      · refine le_trans (Nat.cast_le.mpr (Finset.card_le_card
          (t := (∅ : Finset ((Fin m → K) × (Fin Q → ↥(layerDomain ζ m s t1))))) ?_)) ?_
        · intro w hw
          rw [mem_filter_any] at hw
          have hacc := hw.2
          unfold accepts at hacc
          obtain ⟨_, hA, hI, _, _⟩ := hacc
          exact (hpass ⟨hA, hI⟩).elim
        · rw [Finset.card_empty, Nat.cast_zero]
          exact hn4
    · -- the final arithmetic
      refine final_arith_eq ?_ ?_ ?_ ?_ ?_ ?_
      · exact hκ
      · rfl
      · first
          | (simp only [Fintype.card_prod, Nat.cast_mul]; done)
          | (simp only [Fintype.card_prod, Nat.cast_mul]; ring)
          | (simp [Fintype.card_prod]; ring)
      · rw [hn2def, hX2]
      · rw [hn4def, hΩc]
      · rw [hn3def, hX3]
        exact mu_arith L' _
  · -- the Phase-0 check fails: nothing is accepted
    have h0 : ∀ p : K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1)),
        ¬ accepts σ M' ζ m s isEnd t1 lam rhi v P y gA z SB Q
          p.1 p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2 := by
      intro p hp
      unfold accepts at hp
      exact hconj hp.1
    refine le_trans (Nat.cast_le.mpr (Finset.card_le_card
      (t := (∅ : Finset (K × (Fin m → K) × (Fin L' → K) × (Fin m → K) ×
        (Fin Q → ↥(layerDomain ζ m s t1))))) ?_)) ?_
    · intro p hp
      rw [mem_filter_any] at hp
      exact (h0 p hp.2).elim
    · rw [Finset.card_empty, Nat.cast_zero]
      have h1 : (0 : ℚ) ≤ (if 2 ≤ L' + 1 then (1 : ℚ) else 0) := by
        split_ifs <;> norm_num
      have h2 : (0 : ℚ) ≤ ∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ) :=
        Finset.sum_nonneg (fun _ _ => Nat.cast_nonneg _)
      have h3 : (0 : ℚ) ≤ ((d * (L' + 1) - 1 : ℕ) : ℚ) := Nat.cast_nonneg _
      have h4 : (0 : ℚ) ≤ 2 * (m : ℚ) := by
        have := (Nat.cast_nonneg m : (0 : ℚ) ≤ (m : ℚ))
        linarith
      have hS : (0 : ℚ) ≤ ((d * (L' + 1) - 1 : ℕ) : ℚ) + 2 * (m : ℚ) +
          (if 2 ≤ L' + 1 then (1 : ℚ) else 0) +
          ∑ t ∈ Finset.range m, ((layerDomain ζ m s (t + 1)).card : ℚ) := by
        linarith
      exact mul_nonneg (add_nonneg (div_nonneg hS hκ.le) (pow_nonneg hβ0.le Q)) hpos.le

#print axioms Cprs.Chain.chain_soundness

end Cprs.Chain
