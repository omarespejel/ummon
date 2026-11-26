import CprsFormal.Mle
import CprsFormal.Fold
import CprsFormal.WeightedAgreement
import CprsFormal.L1

/-!
# Lemma F' (one binary fold with a fixed weight)

One fold step of Phase B, relative to the correlated
agreement hypothesis `CorrAgree` (BCIKS20 Theorem 4.1, machine-checked in ArkLib: see `L1.lean`).
-/

open Polynomial Finset

namespace Cprs

namespace Aux

section FoldStepHelpers

variable {K : Type*} [Field K] [DecidableEq K]

/-- Uniqueness of the unique-decoding witness: two polynomials of degree `< k'` that both agree
with the same word `u` on at least `β |D'|` points of `D'`, where `k' ≤ (2β - 1) |D'|`, coincide
(their agreement sets share at least `k'` points). -/
theorem foldStep_unique (D' : Finset K) (k' : ℕ) (β : ℚ)
    (hrate : (k' : ℚ) ≤ (2 * β - 1) * D'.card) (u : K → K) (P Q : K[X])
    (hP : P ∈ degreeLT K k') (hQ : Q ∈ degreeLT K k')
    (hPu : β * D'.card ≤ (agreeCount D' u P : ℚ))
    (hQu : β * D'.card ≤ (agreeCount D' u Q : ℚ)) : P = Q := by
  set A : Finset K := D'.filter (fun x => u x = P.eval x) with hA
  set B : Finset K := D'.filter (fun x => u x = Q.eval x) with hB
  have hAc : agreeCount D' u P = A.card := rfl
  have hBc : agreeCount D' u Q = B.card := rfl
  have hUI := Finset.card_union_add_card_inter A B
  have hU : (A ∪ B).card ≤ D'.card :=
    Finset.card_le_card (Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _))
  have hk : k' ≤ (A ∩ B).card := by
    have e1 : ((A ∪ B).card : ℚ) + ((A ∩ B).card : ℚ) = (A.card : ℚ) + (B.card : ℚ) := by
      exact_mod_cast hUI
    have e2 : ((A ∪ B).card : ℚ) ≤ (D'.card : ℚ) := by exact_mod_cast hU
    rw [hAc] at hPu
    rw [hBc] at hQu
    have h1 : (k' : ℚ) ≤ ((A ∩ B).card : ℚ) := by linarith
    exact_mod_cast h1
  have hkW : ((k' : ℕ) : WithBot ℕ) ≤ (((A ∩ B).card : ℕ) : WithBot ℕ) := by exact_mod_cast hk
  have hdegP : P.degree < (((A ∩ B).card : ℕ) : WithBot ℕ) :=
    lt_of_lt_of_le (mem_degreeLT.1 hP) hkW
  have hdegQ : Q.degree < (((A ∩ B).card : ℕ) : WithBot ℕ) :=
    lt_of_lt_of_le (mem_degreeLT.1 hQ) hkW
  refine Polynomial.eq_of_degrees_lt_of_eval_finset_eq (A ∩ B) hdegP hdegQ ?_
  intro x hx
  rw [Finset.mem_inter, hA, hB, Finset.mem_filter, Finset.mem_filter] at hx
  rw [← hx.1.2, ← hx.2.2]

/-- In a squaring layer, both square roots of a point of `D'` lie in `D`. -/
theorem foldStep_sqrtK_mem {D D' : Finset K} (hL : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ∈ D ∧ -sqrtK y ∈ D := by
  obtain ⟨x, hxD, hxy⟩ := hL.sq_surj y hy
  have h1 : sqrtK y ^ 2 = y := sqrtK_sq ⟨x, hxy⟩
  have h2 : sqrtK y ^ 2 = x ^ 2 := by rw [h1, hxy]
  rcases sq_eq_sq_iff_eq_or_eq_neg.1 h2 with h | h
  · rw [h]
    exact ⟨hxD, hL.neg_mem x hxD⟩
  · rw [h, neg_neg]
    exact ⟨hL.neg_mem x hxD, hxD⟩

/-- A weight bounded by `1` on `D` has fibre average bounded by `1` on `D'`. -/
theorem foldStep_fibAvg_le_one {D D' : Finset K} (hL : SqLayer D D') (ν : K → ℚ)
    (hν1 : ∀ x ∈ D, ν x ≤ 1) {y : K} (hy : y ∈ D') : fibAvg ν y ≤ 1 := by
  obtain ⟨h1, h2⟩ := foldStep_sqrtK_mem hL hy
  have h3 := hν1 _ h1
  have h4 := hν1 _ h2
  unfold fibAvg
  linarith

/-- A weight bounded by `1` turns a weighted agreement `≥ β` into an honest agreement count
`≥ β |D'|`. -/
theorem foldStep_wagree_count (D' : Finset K) (hpos : 0 < D'.card) (w : K → ℚ)
    (hw : ∀ y ∈ D', w y ≤ 1) (β : ℚ) (u : K → K) (P : K[X])
    (h : β ≤ wagree D' w u (fun x => P.eval x)) :
    β * D'.card ≤ (agreeCount D' u P : ℚ) := by
  unfold wagree at h
  have hc : (0 : ℚ) < D'.card := by exact_mod_cast hpos
  have h1 := (le_div_iff₀ hc).1 h
  refine h1.trans ?_
  unfold agreeCount
  calc ∑ x ∈ D'.filter (fun x => u x = P.eval x), w x
      ≤ ∑ x ∈ D'.filter (fun x => u x = P.eval x), (1 : ℚ) :=
        Finset.sum_le_sum (fun x hx => hw x (Finset.mem_filter.1 hx).1)
    _ = ((D'.filter (fun x => u x = P.eval x)).card : ℚ) := by simp

omit [DecidableEq K] in
/-- The even/odd lift of two polynomials with `< k'` coefficients has `< 2 k'` coefficients. -/
theorem foldStep_lift_mem {k' : ℕ} {P0 P1 : K[X]} (h0 : P0 ∈ degreeLT K k')
    (h1 : P1 ∈ degreeLT K k') :
    (expand K 2 P0 + Polynomial.X * expand K 2 P1 : K[X]) ∈ degreeLT K (2 * k') := by
  rw [mem_degreeLT] at h0 h1 ⊢
  rw [degree_lt_iff_coeff_zero] at h0 h1 ⊢
  intro m hm
  rw [coeff_add, coeff_expand (by norm_num)]
  have e1 : (if 2 ∣ m then P0.coeff (m / 2) else 0) = 0 := by
    split_ifs with hd
    · exact h0 _ (by omega)
    · rfl
  rcases m with _ | n
  · rw [coeff_X_mul_zero, e1, add_zero]
  · rw [coeff_X_mul, coeff_expand (by norm_num), e1, zero_add]
    split_ifs with hd
    · exact h1 _ (by omega)
    · rfl

end FoldStepHelpers

end Aux

variable {K : Type*} [Field K] [DecidableEq K] [Fintype K]

open scoped Classical in
/-- **Lemma F'.**  Let `D → D'` be a squaring layer with `|D'| ≥ 2`, `k'` coefficients at layer `D'`
(`2k'` at `D`), agreement threshold `β` with `k' ≤ (2β - 1)|D'|` (uniqueness), and the correlated
agreement hypothesis for `(D', k', β)`.  Let `ν` be a weight with values in `[0,1]`,
`W` a word on `D`, `c, E ∈ K` and `g` a polynomial of degree `≤ 2` with `g(0) + g(1) = c`
(all fixed before the challenge).  If `R_t(W, ν, c, E)` is false, then the set of challenges `a` for
which `R_{t+1}(fold_a W, ν^v, g(a), E · eq₁(ρ_t, a))` holds has at most `|D'|` elements. -/
theorem lemmaF' (D D' : Finset K) (hL : SqLayer D D') (hD' : 2 ≤ D'.card) (k' : ℕ) (β : ℚ)
    (hk' : 0 < k') (hrate : (k' : ℚ) ≤ (2 * β - 1) * D'.card)
    (hCA : CorrAgree D' k' β)
    (ν : K → ℚ) (hν0 : ∀ x ∈ D, 0 ≤ ν x) (hν1 : ∀ x ∈ D, ν x ≤ 1)
    (W : K → K) (c E ρt : K) (ρ' : List K) (g : K[X]) (hg : g.natDegree ≤ 2)
    (hgc : g.eval 0 + g.eval 1 = c)
    (hnot : ¬ Rel D (2 * k') β (ρt :: ρ') ν W c E) :
    (Finset.univ.filter (fun a : K =>
        Rel D' k' β ρ' (fibAvg ν) (fold a W) (g.eval a) (E * eq1 ρt a))).card ≤ D'.card := by
  -- `hk'` is not needed below (the argument is valid for `k' = 0` as well)
  have _hk' := hk'
  set S : Finset K := Finset.univ.filter (fun a : K =>
      Rel D' k' β ρ' (fibAvg ν) (fold a W) (g.eval a) (E * eq1 ρt a)) with hS
  by_contra hcon
  have hlt : D'.card < S.card := not_le.1 hcon
  have hD'pos : 0 < D'.card := by omega
  have hmemS : ∀ a, a ∈ S ↔
      Rel D' k' β ρ' (fibAvg ν) (fold a W) (g.eval a) (E * eq1 ρt a) := by
    intro a
    rw [hS, Finset.mem_filter]
    simp
  -- (1) a witness `P^a` for every good challenge `a`
  have h1 : ∀ a ∈ S, ∃ P ∈ degreeLT K k',
      β ≤ wagree D' (fibAvg ν) (fold a W) (fun x => P.eval x) ∧
      g.eval a = E * eq1 ρt a * mle P ρ' := by
    intro a ha
    obtain ⟨P, hP, hag, hc⟩ := (hmemS a).1 ha
    exact ⟨P, hP, hag, hc⟩
  -- weighted agreement `≥ β` gives an honest agreement count `≥ β |D'|`
  have hcount : ∀ a ∈ S, ∀ P ∈ degreeLT K k',
      β ≤ wagree D' (fibAvg ν) (fold a W) (fun x => P.eval x) →
      β * D'.card ≤ (agreeCount D' (fun y => evenPart W y + a * oddPart W y) P : ℚ) := by
    intro a _ P _ hag
    exact Aux.foldStep_wagree_count D' hD'pos (fibAvg ν)
      (fun y hy => Aux.foldStep_fibAvg_le_one hL ν hν1 hy) β _ P hag
  -- (2) correlated agreement for the even/odd parts of `W`
  obtain ⟨P0, hP0, P1, hP1, hP01⟩ := hCA (evenPart W) (oddPart W) S hlt (by
    intro a ha
    obtain ⟨P, hP, hag, _⟩ := h1 a ha
    exact ⟨P, hP, hcount a ha P hP hag⟩)
  -- (3) uniqueness: `P^a = P0 + a P1` on `S`
  have h3 : ∀ a ∈ S,
      β ≤ wagree D' (fibAvg ν) (fun y => evenPart W y + a * oddPart W y)
        (fun y => P0.eval y + a * P1.eval y) ∧
      g.eval a = E * eq1 ρt a * (mle P0 ρ' + a * mle P1 ρ') := by
    intro a ha
    obtain ⟨P, hP, hag, hc⟩ := h1 a ha
    have hmem : P0 + C a * P1 ∈ degreeLT K k' := by
      rw [C_mul']
      exact Submodule.add_mem _ hP0 (Submodule.smul_mem _ a hP1)
    have hPeq : P = P0 + C a * P1 :=
      Aux.foldStep_unique D' k' β hrate (fun y => evenPart W y + a * oddPart W y) P
        (P0 + C a * P1) hP hmem (hcount a ha P hP hag) (hP01 a ha)
    subst hPeq
    refine ⟨?_, ?_⟩
    · have e : (fun x => (P0 + C a * P1).eval x) = (fun y => P0.eval y + a * P1.eval y) := by
        funext x
        simp
      rw [e] at hag
      exact hag
    · rw [hc, mle_add, mle_C_mul]
  -- (4) Lemma W' : the pair `(W_e, W_o)` agrees with `(P0, P1)` on weighted mass `≥ β`
  have hcardB : (D'.filter (fun y => ¬ (evenPart W y = P0.eval y ∧
      oddPart W y = P1.eval y))).card < S.card :=
    lt_of_le_of_lt (Finset.card_filter_le _ _) hlt
  have hAhat := lemmaW' D' (fibAvg ν) (evenPart W) (oddPart W) (fun y => P0.eval y)
    (fun y => P1.eval y) S β (fun a ha => (h3 a ha).1) hcardB
  set Ahat : Finset K := D'.filter (fun y => evenPart W y = P0.eval y ∧
      oddPart W y = P1.eval y) with hAhatdef
  have hAhat' : β ≤ (∑ y ∈ Ahat, fibAvg ν y) / D'.card := hAhat
  -- (5) lift: `P = P0(X²) + X P1(X²)`
  obtain ⟨P, hPdef⟩ : ∃ P : K[X], P = expand K 2 P0 + Polynomial.X * expand K 2 P1 := ⟨_, rfl⟩
  have hPmem : P ∈ degreeLT K (2 * k') := by
    rw [hPdef]
    exact Aux.foldStep_lift_mem hP0 hP1
  have hPeval : ∀ x : K, P.eval x = P0.eval (x ^ 2) + x * P1.eval (x ^ 2) := by
    intro x
    rw [hPdef]
    simp [eval_add, eval_mul, expand_eval]
  have hsub : D.filter (fun x => x ^ 2 ∈ Ahat) ⊆ D.filter (fun x => W x = P.eval x) := by
    intro x hx
    rw [Finset.mem_filter] at hx ⊢
    obtain ⟨hxD, hxA⟩ := hx
    refine ⟨hxD, ?_⟩
    have hx0 : x ≠ 0 := by
      rintro rfl
      exact hL.zero_notMem hxD
    have hxA' := (Finset.mem_filter.1 hxA).2
    rw [word_eq_even_add_odd hL.two_ne_zero W hx0, hxA'.1, hxA'.2, hPeval]
  have hwag : β ≤ wagree D ν W (fun x => P.eval x) := by
    have hsum1 : ∑ x ∈ D.filter (fun x => x ^ 2 ∈ Ahat), ν x ≤
        ∑ x ∈ D.filter (fun x => W x = P.eval x), ν x :=
      Finset.sum_le_sum_of_subset_of_nonneg hsub
        (fun x hx _ => hν0 x (Finset.mem_filter.1 hx).1)
    have hsum2 : ∑ x ∈ D.filter (fun x => x ^ 2 ∈ Ahat), ν x = 2 * ∑ y ∈ Ahat, fibAvg ν y := by
      rw [hL.sum_fibre Ahat (Finset.filter_subset _ _) ν, Finset.mul_sum]
      refine Finset.sum_congr rfl (fun y _ => ?_)
      unfold fibAvg
      ring
    have hDc : (D.card : ℚ) = 2 * D'.card := by exact_mod_cast hL.card_eq
    have hc' : (0 : ℚ) < D'.card := by exact_mod_cast hD'pos
    show β ≤ (∑ x ∈ D.filter (fun x => W x = P.eval x), ν x) / (D.card : ℚ)
    rw [hDc, le_div_iff₀ (by positivity)]
    have h5 := (le_div_iff₀ hc').1 hAhat'
    linarith
  -- (6) the claim: `g = h` on `S`, hence as polynomials
  set m0 : K := mle P0 ρ' with hm0
  set m1 : K := mle P1 ρ' with hm1
  obtain ⟨h, hhdef⟩ : ∃ h : K[X], h = C E * (C ρt * Polynomial.X + C (1 - ρt) *
      (1 - Polynomial.X)) * (C m0 + C m1 * Polynomial.X) := ⟨_, rfl⟩
  have hheval : ∀ a : K, h.eval a = E * eq1 ρt a * (m0 + a * m1) := by
    intro a
    rw [hhdef]
    simp only [eval_mul, eval_add, eval_sub, eval_C, eval_X, eval_one, eq1]
    ring
  have hhdeg : h.natDegree ≤ 2 := by
    rw [hhdef]
    compute_degree
  have hgh : g = h := by
    refine Polynomial.eq_of_natDegree_lt_card_of_eval_eq' g h S (fun a ha => ?_) ?_
    · rw [(h3 a ha).2, hheval]
    · have : max g.natDegree h.natDegree ≤ 2 := max_le hg hhdeg
      omega
  have hc : c = E * mle P (ρt :: ρ') := by
    rw [hPdef, mle_lift, ← hgc, hgh, hheval, hheval]
    simp only [eq1]
    ring
  -- (7) `P` witnesses `R_t`, contradicting `hnot`
  refine hnot ?_
  unfold Rel
  exact ⟨P, hPmem, hwag, hc⟩

#print axioms lemmaF'

end Cprs
