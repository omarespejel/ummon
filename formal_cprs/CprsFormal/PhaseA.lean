import CprsFormal.Mle
import CprsFormal.FoldStep

/-!
# Round-by-round errors of the rounds before Phase B (`γ`, the Phase-A sum-check, `μ`)

Lemma Π-RBR, for the three kinds of round before Phase B:
`γ`: `≤ (4L-1)/|K|`, `A_t`: `≤ 2/|K|`, `μ` (Model H): `≤ 1/|K|`.
Every statement here is a purely algebraic counting fact about the challenge.
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K] [Fintype K]

/-- **γ-batching.**  If the prover's Phase-0 vector `y` differs from the true vector `y*`
(in `n = 4L` coordinates), at most `n - 1` values of `γ` make the batched claims agree:
`Σ_j γ^j (y_j - y*_j)` is a nonzero polynomial of degree `< n` in `γ`. -/
theorem gamma_roots {n : ℕ} (y ystar : Fin n → K) (hne : y ≠ ystar) :
    (Finset.univ.filter (fun γ : K => ∑ j : Fin n, γ ^ (j : ℕ) * (y j - ystar j) = 0)).card
      ≤ n - 1 := by
  classical
  obtain ⟨j0, hj0⟩ : ∃ j, y j ≠ ystar j := by
    by_contra h
    exact hne (funext fun j => by_contra fun hj => h ⟨j, hj⟩)
  -- NB: the ArkLib import makes the bare token `X` a bivariate notation; use `Polynomial.X`.
  set Pl : K[X] := ∑ j : Fin n, C (y j - ystar j) * Polynomial.X ^ (j : ℕ) with hPl
  have hdeg : Pl.degree < n := degree_sum_fin_lt (fun j => y j - ystar j)
  have hcoeff : Pl.coeff (j0 : ℕ) = y j0 - ystar j0 := by
    simp only [hPl, finsetSum_coeff, coeff_C_mul, coeff_X_pow]
    rw [Finset.sum_eq_single j0]
    · simp
    · intro j _ hj
      have hjj : (j0 : ℕ) ≠ (j : ℕ) := fun h => hj (Fin.ext h).symm
      simp [hjj]
    · intro h
      exact absurd (Finset.mem_univ j0) h
  have hne0 : Pl ≠ 0 := by
    intro h0
    rw [h0, coeff_zero] at hcoeff
    exact hj0 (sub_eq_zero.mp hcoeff.symm)
  have hnat : Pl.natDegree ≤ n - 1 := by
    have := (natDegree_lt_iff_degree_lt hne0).mpr hdeg
    omega
  calc (Finset.univ.filter (fun γ : K => ∑ j : Fin n, γ ^ (j : ℕ) * (y j - ystar j) = 0)).card
      ≤ Pl.roots.toFinset.card := by
        apply Finset.card_le_card
        intro γ hγ
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hγ
        rw [Multiset.mem_toFinset, mem_roots hne0, IsRoot.def, hPl]
        simp only [eval_finsetSum, eval_mul, eval_C, eval_pow, eval_X]
        rw [← hγ]
        exact Finset.sum_congr rfl (fun j _ => mul_comm _ _)
    _ ≤ Multiset.card Pl.roots := Multiset.toFinset_card_le _
    _ ≤ Pl.natDegree := card_roots' _
    _ ≤ n - 1 := hnat

/-- **One sum-check round.**  If the prover's running claim `claim` differs from the true value
`cstar`, and both round polynomials (prover's `g`, honest `gstar`, degree `≤ 2`) satisfy the
sum-check identities `g(0) + g(1) = claim`, `gstar(0) + gstar(1) = cstar`, then they agree at
at most two challenges. -/
theorem sumcheck_roots (g gstar : K[X]) (hg : g.natDegree ≤ 2) (hgs : gstar.natDegree ≤ 2)
    (claim cstar : K) (h1 : g.eval 0 + g.eval 1 = claim) (h2 : gstar.eval 0 + gstar.eval 1 = cstar)
    (hne : claim ≠ cstar) :
    (Finset.univ.filter (fun x : K => g.eval x = gstar.eval x)).card ≤ 2 := by
  classical
  have hne0 : g - gstar ≠ 0 := by
    intro h
    have hgg : g = gstar := sub_eq_zero.mp h
    apply hne
    rw [← h1, ← h2, hgg]
  have hdeg : (g - gstar).natDegree ≤ 2 := (natDegree_sub_le g gstar).trans (max_le hg hgs)
  calc (Finset.univ.filter (fun x : K => g.eval x = gstar.eval x)).card
      ≤ (g - gstar).roots.toFinset.card := by
        apply Finset.card_le_card
        intro x hx
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
        rw [Multiset.mem_toFinset, mem_roots hne0, IsRoot.def, eval_sub, hx, sub_self]
    _ ≤ Multiset.card (g - gstar).roots := Multiset.toFinset_card_le _
    _ ≤ (g - gstar).natDegree := card_roots' _
    _ ≤ 2 := hdeg

set_option linter.unusedSectionVars false in
/-- `mleAux` of the zero coefficient function is `0`. -/
theorem aux_phaseA_mleAux_zero (ρ : List K) : mleAux ρ (fun _ => (0 : K)) = 0 := by
  induction ρ with
  | nil => simp [mleAux]
  | cons r ρ ih => simp [mleAux, ih]

/-- `mle 0 ρ = 0`. -/
theorem aux_phaseA_mle_zero (ρ : List K) : mle (0 : K[X]) ρ = 0 := by
  unfold mle
  simp only [coeff_zero]
  exact aux_phaseA_mleAux_zero ρ

/-- `mle` is additive over finite sums. -/
theorem mle_sum {ι : Type*} (s : Finset ι) (P : ι → K[X]) (ρ : List K) :
    mle (∑ i ∈ s, P i) ρ = ∑ i ∈ s, mle (P i) ρ := by
  classical
  induction s using Finset.induction_on with
  | empty => rw [Finset.sum_empty, Finset.sum_empty]; exact aux_phaseA_mle_zero ρ
  | insert a s ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, mle_add, ih]

set_option linter.unusedSectionVars false in
/-- Uniqueness: two polynomials of `degreeLT K k` agreeing on at least `k` points are equal. -/
theorem aux_phaseA_uniq {k : ℕ} {P Q : K[X]} (hP : P ∈ degreeLT K k) (hQ : Q ∈ degreeLT K k)
    (A : Finset K) (hA : k ≤ A.card) (h : ∀ x ∈ A, P.eval x = Q.eval x) : P = Q := by
  have hPQ : P - Q ∈ degreeLT K k := Submodule.sub_mem _ hP hQ
  rw [mem_degreeLT] at hPQ
  by_cases h0 : P - Q = 0
  · exact sub_eq_zero.mp h0
  · exfalso
    have hlt : (P - Q).natDegree < k := (natDegree_lt_iff_degree_lt h0).mpr hPQ
    have hz := eq_zero_of_natDegree_lt_card_of_eval_eq_zero' (P - Q) A
      (fun x hx => by rw [eval_sub, h x hx, sub_self]) (lt_of_lt_of_le hlt hA)
    exact h0 hz

/-- Counting lemma: for `d ≠ 0` the set of `μ` with `μ₀ = 1` and `Σ μ_a d_a = 0` has at most
`|K|^(L-2)` elements. -/
theorem aux_phaseA_count {L : ℕ} (hL : 2 ≤ L) (d : Fin L → K) (hd : ∃ a, d a ≠ 0) :
    (Finset.univ.filter (fun μ : Fin L → K =>
        μ ⟨0, by omega⟩ = 1 ∧ ∑ a, μ a * d a = 0)).card ≤ Fintype.card K ^ (L - 2) := by
  classical
  by_cases h : ∃ a1 : Fin L, a1 ≠ ⟨0, by omega⟩ ∧ d a1 ≠ 0
  · obtain ⟨a1, ha1, hda1⟩ := h
    have hcardT : Fintype.card {a : Fin L // a ≠ ⟨0, by omega⟩ ∧ a ≠ a1} = L - 2 := by
      rw [Fintype.card_subtype]
      have hset : (Finset.univ.filter (fun a : Fin L => a ≠ ⟨0, by omega⟩ ∧ a ≠ a1))
          = (Finset.univ.erase (⟨0, by omega⟩ : Fin L)).erase a1 := by
        ext a
        simp [and_comm]
      rw [hset, Finset.card_erase_of_mem, Finset.card_erase_of_mem (Finset.mem_univ _)]
      · simp only [Finset.card_univ, Fintype.card_fin]
        omega
      · simp [ha1]
    calc (Finset.univ.filter (fun μ : Fin L → K =>
            μ ⟨0, by omega⟩ = 1 ∧ ∑ a, μ a * d a = 0)).card
        ≤ (Finset.univ : Finset ({a : Fin L // a ≠ ⟨0, by omega⟩ ∧ a ≠ a1} → K)).card := by
          apply Finset.card_le_card_of_injOn
            (fun μ (t : {a : Fin L // a ≠ ⟨0, by omega⟩ ∧ a ≠ a1}) => μ t.1)
          · intro μ _
            exact Finset.mem_coe.mpr (Finset.mem_univ _)
          · intro μ hμ μ' hμ' hf
            obtain ⟨h1, h2⟩ := (Finset.mem_filter.mp (Finset.mem_coe.mp hμ)).2
            obtain ⟨h1', h2'⟩ := (Finset.mem_filter.mp (Finset.mem_coe.mp hμ')).2
            have hagree : ∀ a, a ≠ a1 → μ a = μ' a := by
              intro a ha
              by_cases hz : a = ⟨0, by omega⟩
              · rw [hz, h1, h1']
              · exact congrFun hf ⟨a, hz, ha⟩
            have hsum : ∀ ν : Fin L → K,
                ∑ a, ν a * d a = ν a1 * d a1 + ∑ a ∈ Finset.univ.erase a1, ν a * d a :=
              fun ν => (Finset.add_sum_erase _ _ (Finset.mem_univ a1)).symm
            have hrest : ∑ a ∈ Finset.univ.erase a1, μ a * d a
                = ∑ a ∈ Finset.univ.erase a1, μ' a * d a :=
              Finset.sum_congr rfl (fun a ha => by rw [hagree a (Finset.ne_of_mem_erase ha)])
            have h3 : μ a1 * d a1 = μ' a1 * d a1 := by
              have e1 := hsum μ
              have e2 := hsum μ'
              rw [h2] at e1
              rw [h2'] at e2
              rw [hrest] at e1
              exact add_right_cancel (e1.symm.trans e2)
            have h4 : μ a1 = μ' a1 := mul_right_cancel₀ hda1 h3
            funext a
            by_cases ha : a = a1
            · rw [ha]; exact h4
            · exact hagree a ha
      _ = Fintype.card ({a : Fin L // a ≠ ⟨0, by omega⟩ ∧ a ≠ a1} → K) := Finset.card_univ
      _ = Fintype.card K ^ (L - 2) := by rw [Fintype.card_fun, hcardT]
  · have h' : ∀ a1 : Fin L, a1 ≠ ⟨0, by omega⟩ → d a1 = 0 :=
      fun a1 ha1 => by_contra fun hd1 => h ⟨a1, ha1, hd1⟩
    have hz : d ⟨0, by omega⟩ ≠ 0 := by
      obtain ⟨a, ha⟩ := hd
      by_cases hae : a = ⟨0, by omega⟩
      · rw [← hae]; exact ha
      · exact absurd (h' a hae) ha
    have hempty : (Finset.univ.filter (fun μ : Fin L → K =>
        μ ⟨0, by omega⟩ = 1 ∧ ∑ a, μ a * d a = 0)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro μ _ hμ
      obtain ⟨h1, h2⟩ := hμ
      rw [Finset.sum_eq_single (⟨0, by omega⟩ : Fin L)] at h2
      · rw [h1, one_mul] at h2
        exact hz h2
      · intro b _ hb
        rw [h' b hb, mul_zero]
      · intro hb
        exact absurd (Finset.mem_univ _) hb
    rw [hempty, Finset.card_empty]
    exact Nat.zero_le _

set_option linter.unusedSectionVars false in
/-- Weighted agreement for the constant weight `1` is a plain agreement ratio. -/
theorem aux_phaseA_wagree_one (D : Finset K) (u v : K → K) :
    wagree D (fun _ => (1 : ℚ)) u v = ((D.filter (fun x => u x = v x)).card : ℚ) / D.card := by
  unfold wagree
  simp

/-- Core of the `μ`-step: if `Rel` holds for the combination `Σ μ_a P_a` (with the uniqueness
inequality), then `Σ μ_a (z_a - P_a~(ρ)) = 0`. -/
theorem aux_phaseA_mu_key (D : Finset K) (k : ℕ) (β : ℚ) (hrate : (k : ℚ) ≤ (2 * β - 1) * D.card)
    {L : ℕ} (P : Fin L → K[X]) (hP : ∀ a, P a ∈ degreeLT K k) (ρ : List K)
    (z : Fin L → K) (μ : Fin L → K)
    (hRel : Rel D k β ρ (fun _ => 1) (fun x => ∑ a, μ a * (P a).eval x) (∑ a, μ a * z a) 1) :
    ∑ a, μ a * (z a - mle (P a) ρ) = 0 := by
  classical
  obtain ⟨P', hP', hβ, hc⟩ := hRel
  set Q : K[X] := ∑ a, C (μ a) * P a with hQ
  have hQmem : Q ∈ degreeLT K k := by
    apply Submodule.sum_mem
    intro a _
    rw [← smul_eq_C_mul]
    exact Submodule.smul_mem _ _ (hP a)
  have hQeval : ∀ x, Q.eval x = ∑ a, μ a * (P a).eval x := by
    intro x
    simp only [hQ, eval_finsetSum, eval_mul, eval_C]
  have hβ' : β ≤ ((D.filter (fun x => (∑ a, μ a * (P a).eval x) = P'.eval x)).card : ℚ) / D.card := by
    rw [← aux_phaseA_wagree_one D (fun x => ∑ a, μ a * (P a).eval x) (fun x => P'.eval x)]
    exact hβ
  set A : Finset K := D.filter (fun x => (∑ a, μ a * (P a).eval x) = P'.eval x) with hA
  have hDnn : (0 : ℚ) ≤ D.card := Nat.cast_nonneg _
  have hAle : (A.card : ℚ) ≤ D.card := by exact_mod_cast Finset.card_filter_le _ _
  have hβ1 : β ≤ 1 := by
    refine hβ'.trans ?_
    rcases Nat.eq_zero_or_pos D.card with h0 | hpos
    · simp [h0]
    · rw [div_le_one (by exact_mod_cast hpos)]
      exact hAle
  have hβD : β * D.card ≤ A.card := by
    rcases Nat.eq_zero_or_pos D.card with h0 | hpos
    · simp [h0]
    · exact (le_div_iff₀ (by exact_mod_cast hpos)).mp hβ'
  have hkA : k ≤ A.card := by
    have hq : (k : ℚ) ≤ A.card := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hβ1) hDnn]
    exact_mod_cast hq
  have hPQ : P' = Q := aux_phaseA_uniq hP' hQmem A hkA (fun x hx => by
    rw [hQeval]
    exact ((Finset.mem_filter.mp hx).2).symm)
  rw [hPQ, one_mul] at hc
  have hmleQ : mle Q ρ = ∑ a, μ a * mle (P a) ρ := by
    rw [hQ, mle_sum]
    exact Finset.sum_congr rfl (fun a _ => mle_C_mul _ _ _)
  simp only [mul_sub, Finset.sum_sub_distrib]
  rw [hc, hmleQ, sub_self]

open scoped Classical in
/-- **μ-combination (Model H).**  Let `P a` (`a < L`) be the honest registered polynomials
(degree `< k`), `z*_a = P_a~(ρ)` the true claims and `z` the prover's claims, not all equal.
For the combination `π_μ = Σ_a μ_a P_a` (with `μ_0 = 1`) the relation `R_0(π_μ, 1, Σ μ_a z_a, 1)`
holds for at most `|K|^(L-2)` of the `|K|^(L-1)` vectors `μ` (probability `≤ 1/|K|`), provided
the agreement threshold satisfies the uniqueness inequality `k ≤ (2β-1)|D|`. -/
theorem mu_step (D : Finset K) (k : ℕ) (β : ℚ) (hrate : (k : ℚ) ≤ (2 * β - 1) * D.card)
    {L : ℕ} (hL : 2 ≤ L) (P : Fin L → K[X]) (hP : ∀ a, P a ∈ degreeLT K k) (ρ : List K)
    (z : Fin L → K) (hz : ∃ a, z a ≠ mle (P a) ρ) :
    (Finset.univ.filter (fun μ : Fin L → K =>
        μ ⟨0, by omega⟩ = 1 ∧
        Rel D k β ρ (fun _ => 1) (fun x => ∑ a, μ a * (P a).eval x) (∑ a, μ a * z a) 1)).card
      ≤ Fintype.card K ^ (L - 2) := by
  obtain ⟨a0, ha0⟩ := hz
  have hd : ∃ a, (z a - mle (P a) ρ) ≠ 0 := ⟨a0, sub_ne_zero.mpr ha0⟩
  refine le_trans (Finset.card_le_card ?_) (aux_phaseA_count hL (fun a => z a - mle (P a) ρ) hd)
  intro μ hμ
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hμ ⊢
  exact ⟨hμ.1, aux_phaseA_mu_key D k β hrate P hP ρ z μ hμ.2⟩

#print axioms gamma_roots
#print axioms sumcheck_roots
#print axioms mle_sum
#print axioms mu_step

end Cprs
