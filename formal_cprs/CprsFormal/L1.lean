import CprsFormal.Basic
import ArkLib.Data.CodingTheory.ProximityGap.BCIKS20.AffineLines.UniqueDecoding

/-!
# L1: unique-decoding correlated agreement for lines (BCIKS20 Theorem 4.1)

`CorrAgree D' k' β` is the (weak) form of BCIKS20 Theorem 4.1 that Lemma F' uses.  It is *implied* by
the published statement, and by ArkLib's machine-checked proof
`ProximityGap.RS_jointAgreement_of_goodCoeffs_card_gt` (vendored, unmodified, in `vendor/ArkLib`).
`corrAgree_of_arklib` is the bridge; it makes Lemma F' / Theorem B' free of any cited-but-unchecked
coding-theory hypothesis.
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- **L1 (weak form).**  Let `D'` be a set of evaluation points and `k'` a number of coefficients.
If for more than `|D'|` values `a` the word `u₀ + a u₁` agrees with some polynomial of degree `< k'`
on at least `β |D'|` points, then there are two polynomials `P₀, P₁` of degree `< k'` such that
`u₀ + a u₁` agrees with `P₀ + a P₁` on at least `β |D'|` points for every such `a`. -/
def CorrAgree (D' : Finset K) (k' : ℕ) (β : ℚ) : Prop :=
  ∀ (u0 u1 : K → K) (S : Finset K), D'.card < S.card →
    (∀ a ∈ S, ∃ P ∈ degreeLT K k',
      β * D'.card ≤ (agreeCount D' (fun y => u0 y + a * u1 y) P : ℚ)) →
    ∃ P0 ∈ degreeLT K k', ∃ P1 ∈ degreeLT K k', ∀ a ∈ S,
      β * D'.card ≤ (agreeCount D' (fun y => u0 y + a * u1 y) (P0 + C a * P1) : ℚ)

namespace Aux

/-- Counting a filter over the subtype of a finset. -/
theorem aux_l1_card_filter_coe {α : Type*} (D' : Finset α) (p : α → Prop) [DecidablePred p] :
    (Finset.univ.filter (fun i : ↥D' => p i.1)).card = (D'.filter p).card := by
  rw [Finset.card_filter, Finset.card_filter]
  exact Finset.sum_coe_sort D' (fun x => if p x then 1 else 0)

/-- A word with an agreement set of relative size `≥ 1 - δ` with an RS codeword is
`δ`-relatively close to the code. -/
theorem aux_l1_relClose_of_agree {ι : Type} [Fintype ι] [Nonempty ι] {F : Type} [Field F]
    [DecidableEq F] {domain : ι ↪ F} {deg : ℕ} (w : ι → F) (P : F[X])
    (hP : P ∈ degreeLT F deg) (δ : NNReal)
    (h : (1 - δ) * (Fintype.card ι : NNReal) ≤
      ((Finset.univ.filter (fun i => w i = P.eval (domain i))).card : NNReal)) :
    Code.relDistFromCode w (ReedSolomon.code domain deg : Set (ι → F)) ≤ (δ : ENNReal) := by
  rw [Code.relCloseToCode_iff_relCloseToCodeword_of_minDist]
  refine ⟨ReedSolomon.evalOnPoints domain P, ?_, ?_⟩
  · exact Submodule.mem_map_of_mem hP
  · rw [Code.relCloseToWord_iff_exists_agreementCols]
    refine ⟨Finset.univ.filter (fun i => w i = P.eval (domain i)), ?_, ?_⟩
    · exact (Code.relDist_floor_bound_iff_complement_bound _ _ _).2 h
    · intro i
      simp [ReedSolomon.evalOnPoints]

end Aux

/-- Bridge to ArkLib: for a finite field `K : Type`, `1 ≤ k' ≤ |D'|`, and `β` at least
`(1 + k'/|D'|)/2` (i.e. radius `1 - β` at most the unique-decoding radius `(1 - k'/|D'|)/2`),
`CorrAgree D' k' β` holds. -/
theorem corrAgree_of_arklib {K : Type} [Field K] [Fintype K] [DecidableEq K]
    (D' : Finset K) (k' : ℕ) (β : ℚ) (hk' : 0 < k') (hkn : k' ≤ D'.card) (hβ1 : β ≤ 1)
    (hβ : (D'.card + k' : ℚ) ≤ 2 * β * D'.card) :
    CorrAgree D' k' β := by
  intro u0 u1 S hS hagree
  have hn0 : 0 < D'.card := lt_of_lt_of_le hk' hkn
  obtain ⟨x0, hx0⟩ := Finset.card_pos.mp hn0
  have : Nonempty ↥D' := ⟨⟨x0, hx0⟩⟩
  have : NeZero k' := ⟨hk'.ne'⟩
  have hcard : Fintype.card ↥D' = D'.card := Fintype.card_coe D'
  have hnQ : (0 : ℚ) < D'.card := by exact_mod_cast hn0
  have hβ0 : (0 : ℚ) < β := by
    by_contra hneg
    have hneg := not_lt.mp hneg
    have hk'Q : (0 : ℚ) < k' := by exact_mod_cast hk'
    nlinarith
  let domain : ↥D' ↪ K := Function.Embedding.subtype (· ∈ D')
  let b : NNReal := ⟨(β : ℝ), by exact_mod_cast hβ0.le⟩
  have hbR : (b : ℝ) = (β : ℝ) := rfl
  have hb1 : b ≤ 1 := by
    rw [← NNReal.coe_le_coe, hbR]
    simpa using (by exact_mod_cast hβ1 : (β : ℝ) ≤ 1)
  let δ : NNReal := 1 - b
  have hδ' : 1 - δ = b := tsub_tsub_cancel_of_le hb1
  let u : Code.WordStack K (Fin 2) ↥D' := Matrix.of ![fun i => u0 i.1, fun i => u1 i.1]
  have hu0 : u 0 = fun i => u0 i.1 := rfl
  have hu1 : u 1 = fun i => u1 i.1 := rfl
  -- δ ≤ relative unique decoding radius
  have hδ : δ ≤ Code.relativeUniqueDecodingRadius
      (ι := ↥D') (F := K) (C := (ReedSolomon.code domain k' : Set (↥D' → K))) := by
    rw [ReedSolomon.relativeUniqueDecodingRadius_RS_eq (by rw [hcard]; exact hkn)]
    rw [← NNReal.coe_le_coe]
    have hkn' : ((k' : NNReal) / (Fintype.card ↥D' : NNReal)) ≤ 1 := by
      rw [div_le_one (by exact_mod_cast (hcard ▸ hn0))]
      exact_mod_cast (hcard ▸ hkn)
    rw [NNReal.coe_sub hb1, NNReal.coe_div, NNReal.coe_sub hkn', NNReal.coe_div]
    simp only [NNReal.coe_one, NNReal.coe_natCast, NNReal.coe_ofNat, hbR, hcard]
    have hnR : (0 : ℝ) < (D'.card : ℝ) := by exact_mod_cast hn0
    have hβR : ((D'.card : ℝ) + k') ≤ 2 * (β : ℝ) * D'.card := by exact_mod_cast hβ
    have h1 : (k' : ℝ) / D'.card ≤ 2 * (β : ℝ) - 1 := by
      rw [div_le_iff₀ hnR]
      linarith
    linarith
  -- the set of good coefficients contains S
  have hsub : S ⊆ ProximityGap.RS_goodCoeffs (deg := k') (domain := domain) u δ := by
    intro a ha
    obtain ⟨P, hP, hPa⟩ := hagree a ha
    unfold ProximityGap.RS_goodCoeffs
    rw [Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    refine Aux.aux_l1_relClose_of_agree (domain := domain) (u 0 + a • u 1) P hP δ ?_
    rw [hδ']
    have hfilt : (Finset.univ.filter
        (fun i : ↥D' => (u 0 + a • u 1) i = P.eval (domain i))).card
        = agreeCount D' (fun y => u0 y + a * u1 y) P := by
      unfold agreeCount
      rw [← Aux.aux_l1_card_filter_coe D' (fun x => u0 x + a * u1 x = P.eval x)]
      rfl
    rw [hfilt, hcard, ← NNReal.coe_le_coe]
    push_cast
    rw [hbR]
    exact_mod_cast hPa
  have hgood : Fintype.card ↥D' <
      (ProximityGap.RS_goodCoeffs (deg := k') (domain := domain) u δ).card := by
    calc Fintype.card ↥D' = D'.card := hcard
      _ < S.card := hS
      _ ≤ _ := Finset.card_le_card hsub
  obtain ⟨T, hTcard, v, hv⟩ := ProximityGap.RS_jointAgreement_of_goodCoeffs_card_gt
    (deg := k') (domain := domain) (δ := δ) hδ u hgood
  obtain ⟨hv0C, hT0⟩ := hv 0
  obtain ⟨hv1C, hT1⟩ := hv 1
  have hv0C' : v 0 ∈ ReedSolomon.code domain k' := hv0C
  have hv1C' : v 1 ∈ ReedSolomon.code domain k' := hv1C
  obtain ⟨P0, hP0, hP0v⟩ := Submodule.mem_map.1 hv0C'
  obtain ⟨P1, hP1, hP1v⟩ := Submodule.mem_map.1 hv1C'
  refine ⟨P0, hP0, P1, hP1, fun a _ => ?_⟩
  have hTsub : T.map (Function.Embedding.subtype (· ∈ D')) ⊆
      D'.filter (fun y => u0 y + a * u1 y = (P0 + C a * P1).eval y) := by
    intro y hy
    rw [Finset.mem_map] at hy
    obtain ⟨i, hi, rfl⟩ := hy
    rw [Finset.mem_filter]
    refine ⟨i.2, ?_⟩
    have h0 := (Finset.mem_filter.mp (hT0 hi)).2
    have h1 := (Finset.mem_filter.mp (hT1 hi)).2
    have e0 : P0.eval (i : K) = u0 i := by
      have h := congrFun hP0v i
      have h0' : v 0 i = u0 i := h0
      calc P0.eval (i : K) = ReedSolomon.evalOnPoints domain P0 i := rfl
        _ = v 0 i := h
        _ = u0 i := h0'
    have e1 : P1.eval (i : K) = u1 i := by
      have h := congrFun hP1v i
      have h1' : v 1 i = u1 i := h1
      calc P1.eval (i : K) = ReedSolomon.evalOnPoints domain P1 i := rfl
        _ = v 1 i := h
        _ = u1 i := h1'
    show u0 (i : K) + a * u1 (i : K) = (P0 + C a * P1).eval (i : K)
    rw [eval_add, eval_mul, eval_C, e0, e1]
  have hle : T.card ≤ agreeCount D' (fun y => u0 y + a * u1 y) (P0 + C a * P1) := by
    calc T.card = (T.map (Function.Embedding.subtype (· ∈ D'))).card :=
          (Finset.card_map _).symm
      _ ≤ _ := Finset.card_le_card hTsub
  rw [hδ'] at hTcard
  have hfin : b * (Fintype.card ↥D' : NNReal) ≤
      ((agreeCount D' (fun y => u0 y + a * u1 y) (P0 + C a * P1) : ℕ) : NNReal) :=
    le_trans hTcard (by exact_mod_cast hle)
  rw [hcard, ← NNReal.coe_le_coe] at hfin
  push_cast at hfin
  rw [hbR] at hfin
  exact_mod_cast hfin

#print axioms corrAgree_of_arklib

end Cprs
