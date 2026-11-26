import CprsFormal.ChainDefs
import CprsFormal.Rbr

/-!
# Sum-check soundness in counting form

Lund–Fortnow–Karloff–Nisan: if the claimed total `c0` differs from the Boolean sum of `F` and every
round polynomial has degree `≤ 2`, then at most `2 m |K|^(m-1)` challenge vectors satisfy all round
checks and make the final claim equal `F ρ`.

Proof architecture (round-by-round, via the union bound `rbr_prob`):
* `aux_sumcheck_pt pre b`  : the point whose first `t` coordinates are `pre` and whose remaining
  coordinates are the Boolean vector `b` (junk-padded, so that no cast `m - t = r + 1` is needed);
* `aux_sumcheck_S F pre r` : the true partial sum `Σ_{b ∈ {0,1}^r} F(pre, b)`; the truth after the
  prefix `pre` of length `t` is `aux_sumcheck_S F pre (m - t)`;
* `aux_sumcheck_S_succ`    : splitting the Boolean sum on its first free coordinate;
* `aux_sumcheck_poly`, `aux_sumcheck_honest` : the honest round polynomial `gstar_t` (degree `≤ 2`,
  `gstar_t(0) + gstar_t(1) = truth t pre`, `gstar_t(x) = truth (t+1) (pre, x)`);
* `aux_sumcheck_step`      : per-round counting (at most `2` good extensions of a doomed prefix),
  via `sumcheck_roots`;
* `sumcheck_sound`         : the statement, from `rbr_prob`.
-/

open Polynomial Finset

namespace Cprs

section SumCheckAux

variable {K : Type*} [Field K]

/-- The point of `K^m` whose first `t` coordinates are `pre` and whose coordinates `t, t+1, …`
are the Boolean vector `b : Fin r → Bool` (as `0/1`); coordinates `≥ t + r` are padded with `0`
(a junk value that is never used when `r = m - t`). -/
def aux_sumcheck_pt {m t r : ℕ} (pre : Fin t → K) (b : Fin r → Bool) : Fin m → K :=
  fun i => if h : (i : ℕ) < t then pre ⟨i, h⟩ else
    if h' : (i : ℕ) - t < r then (if b ⟨(i : ℕ) - t, h'⟩ then 1 else 0) else 0

/-- The partial Boolean sum `Σ_{b ∈ {0,1}^r} F(pre, b)`. -/
def aux_sumcheck_S {m t : ℕ} (F : (Fin m → K) → K) (pre : Fin t → K) (r : ℕ) : K :=
  ∑ b : Fin r → Bool, F (aux_sumcheck_pt pre b)

/-- For `r = m - t` the junk branch of `aux_sumcheck_pt` is never taken: coordinates `i ≥ t` are
read from the Boolean vector `b` (as `0/1`).  So `aux_sumcheck_S F pre (m - t)` is exactly
`Σ_{b : Fin (m - t) → Bool} F(pre, b)`. -/
theorem aux_sumcheck_pt_no_junk {m t : ℕ} (pre : Fin t → K) (b : Fin (m - t) → Bool) (i : Fin m)
    (hi : ¬ (i : ℕ) < t) :
    aux_sumcheck_pt pre b i =
      if b ⟨(i : ℕ) - t, by have := i.isLt; omega⟩ then 1 else 0 := by
  unfold aux_sumcheck_pt
  have h' : (i : ℕ) - t < m - t := by have := i.isLt; omega
  rw [dite_eq_right hi, dite_eq_left h']

omit [Field K] in
theorem aux_sumcheck_snoc_val {t : ℕ} (pre : Fin t → K) (x : K) (i : ℕ) (h : i < t + 1) :
    (Fin.snoc (α := fun _ => K) pre x : Fin (t + 1) → K) ⟨i, h⟩ =
      if h' : i < t then pre ⟨i, h'⟩ else x := by
  by_cases h' : i < t
  · rw [dite_eq_left h']
    exact Fin.snoc_castSucc (α := fun _ => K) x pre ⟨i, h'⟩
  · rw [dite_eq_right h']
    have hi : (⟨i, h⟩ : Fin (t + 1)) = Fin.last t := Fin.ext (by simp; omega)
    rw [hi]
    exact Fin.snoc_last (α := fun _ => K) x pre

theorem aux_sumcheck_pt_cons {m t r : ℕ} (pre : Fin t → K) (b0 : Bool) (b : Fin r → Bool) :
    (aux_sumcheck_pt (m := m) pre (Fin.cons b0 b : Fin (r + 1) → Bool)) =
      aux_sumcheck_pt (Fin.snoc (α := fun _ => K) pre (if b0 then 1 else 0)) b := by
  funext i
  unfold aux_sumcheck_pt
  by_cases h1 : (i : ℕ) < t
  · have h2 : (i : ℕ) < t + 1 := by omega
    rw [dite_eq_left h1, dite_eq_left h2, aux_sumcheck_snoc_val _ _ _ h2, dite_eq_left h1]
  · rw [dite_eq_right h1]
    by_cases h3 : (i : ℕ) = t
    · have h2 : (i : ℕ) < t + 1 := by omega
      have h4 : (i : ℕ) - t < r + 1 := by omega
      rw [dite_eq_left h2, aux_sumcheck_snoc_val _ _ _ h2, dite_eq_right h1, dite_eq_left h4]
      have h5 : (⟨(i : ℕ) - t, h4⟩ : Fin (r + 1)) = 0 := Fin.ext (by simp; omega)
      rw [h5, Fin.cons_zero]
    · have h2 : ¬ (i : ℕ) < t + 1 := by omega
      rw [dite_eq_right h2]
      by_cases h5 : (i : ℕ) - (t + 1) < r
      · have h6 : (i : ℕ) - t < r + 1 := by omega
        rw [dite_eq_left h5, dite_eq_left h6]
        have h7 : (⟨(i : ℕ) - t, h6⟩ : Fin (r + 1)) = Fin.succ ⟨(i : ℕ) - (t + 1), h5⟩ :=
          Fin.ext (by simp; omega)
        rw [h7, Fin.cons_succ]
      · have h6 : ¬ (i : ℕ) - t < r + 1 := by omega
        rw [dite_eq_right h5, dite_eq_right h6]

theorem aux_sumcheck_S_succ {m t : ℕ} (F : (Fin m → K) → K) (pre : Fin t → K) (r : ℕ) :
    aux_sumcheck_S F pre (r + 1) =
      aux_sumcheck_S F (Fin.snoc (α := fun _ => K) pre 0) r +
        aux_sumcheck_S F (Fin.snoc (α := fun _ => K) pre 1) r := by
  unfold aux_sumcheck_S
  rw [← (Fin.consEquiv (fun _ : Fin (r + 1) => Bool)).sum_comp, Fintype.sum_prod_type,
    Fintype.sum_bool]
  have e : ∀ (b0 : Bool) (y : Fin r → Bool),
      (Fin.consEquiv (fun _ : Fin (r + 1) => Bool)) (b0, y) = (Fin.cons b0 y : Fin (r + 1) → Bool) :=
    fun _ _ => rfl
  simp only [e, aux_sumcheck_pt_cons]
  simp only [ite_true, Bool.false_eq_true, ite_false]
  rw [add_comm]

theorem aux_sumcheck_S_zero_full {m : ℕ} (F : (Fin m → K) → K) (pre : Fin m → K) :
    aux_sumcheck_S F pre 0 = F pre := by
  unfold aux_sumcheck_S
  have h : ∀ b : Fin 0 → Bool, aux_sumcheck_pt (m := m) pre b = pre := by
    intro b
    funext i
    unfold aux_sumcheck_pt
    rw [dite_eq_left i.isLt]
  simp [h]

theorem aux_sumcheck_S_zero_pre {m : ℕ} (F : (Fin m → K) → K) (pre : Fin 0 → K) :
    aux_sumcheck_S F pre m = ∑ b : Fin m → Bool, F (boolPt b) := by
  unfold aux_sumcheck_S
  refine Finset.sum_congr rfl (fun b _ => ?_)
  congr 1
  funext i
  unfold aux_sumcheck_pt boolPt
  have h : (i : ℕ) - 0 < m := by simp
  rw [dite_eq_right (by simp), dite_eq_left h]
  have hi : (⟨(i : ℕ) - 0, h⟩ : Fin m) = i := Fin.ext (by simp)
  rw [hi]

theorem aux_sumcheck_pt_update {m t r : ℕ} (ht : t < m) (pre : Fin t → K) (x : K)
    (b : Fin r → Bool) :
    aux_sumcheck_pt (m := m) (Fin.snoc (α := fun _ => K) pre x) b =
      Function.update (aux_sumcheck_pt (m := m) (Fin.snoc (α := fun _ => K) pre (0 : K)) b)
        ⟨t, ht⟩ x := by
  funext i
  unfold aux_sumcheck_pt
  by_cases hi : i = ⟨t, ht⟩
  · subst hi
    simp only [Function.update_self]
    have h2 : t < t + 1 := Nat.lt_succ_self t
    rw [dite_eq_left h2, aux_sumcheck_snoc_val _ _ _ h2, dite_eq_right (lt_irrefl t)]
  · rw [Function.update_of_ne hi]
    have hne : (i : ℕ) ≠ t := fun h => hi (Fin.ext h)
    by_cases h1 : (i : ℕ) < t + 1
    · have h2 : (i : ℕ) < t := by omega
      rw [dite_eq_left h1, dite_eq_left h1, aux_sumcheck_snoc_val _ _ _ h1, aux_sumcheck_snoc_val _ _ _ h1,
        dite_eq_left h2, dite_eq_left h2]
    · rw [dite_eq_right h1, dite_eq_right h1]

theorem aux_sumcheck_poly {m : ℕ} (F : (Fin m → K) → K) (hF : CoordDeg 2 F) {t : ℕ} (ht : t < m)
    (pre : Fin t → K) (r : ℕ) :
    ∃ q : K[X], q.natDegree ≤ 2 ∧ ∀ x : K,
      q.eval x = aux_sumcheck_S F (Fin.snoc (α := fun _ => K) pre x) r := by
  have hb : ∀ b : Fin r → Bool, ∃ q : K[X], q.natDegree ≤ 2 ∧ ∀ x : K,
      F (aux_sumcheck_pt (Fin.snoc (α := fun _ => K) pre x) b) = q.eval x := by
    intro b
    obtain ⟨q, hq, hqx⟩ := hF ⟨t, ht⟩ (aux_sumcheck_pt (Fin.snoc (α := fun _ => K) pre (0 : K)) b)
    refine ⟨q, hq, fun x => ?_⟩
    rw [← hqx x, aux_sumcheck_pt_update ht pre x b]
  choose q hqd hqe using hb
  refine ⟨∑ b, q b, natDegree_sum_le_of_forall_le _ _ (fun b _ => hqd b), fun x => ?_⟩
  simp only [aux_sumcheck_S, eval_finsetSum, hqe]

/-- **The honest round polynomial.**  For `t < m` and a challenge prefix `pre` there is `gstar` of
degree `≤ 2` with `gstar(0) + gstar(1) = truth t pre` and `gstar(x) = truth (t+1) (pre, x)`, where
`truth t pre = Σ_{b ∈ {0,1}^(m-t)} F(pre, b)`. -/
theorem aux_sumcheck_honest {m : ℕ} (F : (Fin m → K) → K) (hF : CoordDeg 2 F) {t : ℕ} (ht : t < m)
    (pre : Fin t → K) :
    ∃ q : K[X], q.natDegree ≤ 2 ∧ q.eval 0 + q.eval 1 = aux_sumcheck_S F pre (m - t) ∧
      ∀ x : K, q.eval x = aux_sumcheck_S F (Fin.snoc (α := fun _ => K) pre x) (m - (t + 1)) := by
  obtain ⟨q, hqd, hqe⟩ := aux_sumcheck_poly F hF ht pre (m - (t + 1))
  refine ⟨q, hqd, ?_, hqe⟩
  rw [hqe, hqe]
  have h1 : m - t = (m - (t + 1)) + 1 := by omega
  rw [h1, aux_sumcheck_S_succ]

/-- Goodness of a challenge prefix: all round checks passed and the running claim equals the true
partial sum. -/
def aux_sumcheck_Good {m : ℕ} (F : (Fin m → K) → K) (g : SCStrat K) (c0 : K) (t : ℕ)
    (pre : Fin t → K) : Prop :=
  scOk g c0 t pre ∧ scClaim g c0 t pre = aux_sumcheck_S F pre (m - t)

end SumCheckAux

section SumCheckMain

variable {K : Type*} [Field K] [DecidableEq K]

open scoped Classical in
/-- One round: from a doomed prefix at most two challenges make the next prefix good. -/
theorem aux_sumcheck_step [Fintype K] {m : ℕ} (F : (Fin m → K) → K) (hF : CoordDeg 2 F) (c0 : K)
    (g : SCStrat K) (hg : ∀ t pre, (g t pre).natDegree ≤ 2) {t : ℕ} (ht : t < m)
    (pre : Fin t → K) (hnot : ¬ aux_sumcheck_Good F g c0 t pre) :
    (Finset.univ.filter (fun x : K =>
        aux_sumcheck_Good F g c0 (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card ≤ 2 := by
  by_cases hok : scOk g c0 t pre ∧ (g t pre).eval 0 + (g t pre).eval 1 = scClaim g c0 t pre
  · have hne : scClaim g c0 t pre ≠ aux_sumcheck_S F pre (m - t) :=
      fun h => hnot ⟨hok.1, h⟩
    obtain ⟨q, hqd, hsplit, hqe⟩ := aux_sumcheck_honest F hF ht pre
    have hroots := sumcheck_roots (g t pre) q (hg t pre) hqd (scClaim g c0 t pre)
      (aux_sumcheck_S F pre (m - t)) hok.2 hsplit hne
    refine le_trans (Finset.card_le_card ?_) hroots
    intro x hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    obtain ⟨_, hcl⟩ := hx
    simp only [scClaim, Fin.init_snoc, Fin.snoc_last] at hcl
    rw [hqe]
    exact hcl
  · have hempty : (Finset.univ.filter (fun x : K =>
        aux_sumcheck_Good F g c0 (t + 1) (Fin.snoc (α := fun _ => K) pre x))) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro x _ hx
      apply hok
      obtain ⟨hx1, _⟩ := hx
      simpa only [scOk, Fin.init_snoc] using hx1
    rw [hempty]
    simp

open scoped Classical in
/-- **Sum-check soundness (counting form).**  If the claimed total `c0` is not the true Boolean sum of
`F`, then for every (adaptive, deterministic) prover strategy `g` at most `2 m |K|^(m-1)` challenge
vectors satisfy all round checks and make the final claim equal `F ρ`. -/
theorem sumcheck_sound [Fintype K] {m : ℕ} (F : (Fin m → K) → K) (hF : CoordDeg 2 F) (c0 : K)
    (hfalse : ∑ b : Fin m → Bool, F (boolPt b) ≠ c0)
    (g : SCStrat K) (hg : ∀ t pre, (g t pre).natDegree ≤ 2) :
    ((Finset.univ.filter (fun ρ : Fin m → K => scOk g c0 m ρ ∧ scClaim g c0 m ρ = F ρ)).card : ℚ)
      ≤ (2 * (m : ℚ) / Fintype.card K) * (Fintype.card K : ℚ) ^ m := by
  have hκ : (0 : ℚ) < (Fintype.card K : ℚ) := by exact_mod_cast Fintype.card_pos
  have h0 : ∀ pre : Fin 0 → K, ¬ aux_sumcheck_Good F g c0 0 pre := by
    intro pre hgood
    apply hfalse
    have h1 := hgood.2
    simp only [scClaim] at h1
    rw [Nat.sub_zero, aux_sumcheck_S_zero_pre] at h1
    exact h1.symm
  have hstep : ∀ t, t < m → ∀ pre : Fin t → K, ¬ aux_sumcheck_Good F g c0 t pre →
      ((Finset.univ.filter (fun x : K =>
        aux_sumcheck_Good F g c0 (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card : ℚ)
        ≤ (fun _ : ℕ => 2 / (Fintype.card K : ℚ)) t * Fintype.card K := by
    intro t ht pre hnot
    have h := aux_sumcheck_step F hF c0 g hg ht pre hnot
    have h' : (((Finset.univ.filter (fun x : K =>
        aux_sumcheck_Good F g c0 (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card : ℕ) : ℚ)
          ≤ 2 := by exact_mod_cast h
    show _ ≤ 2 / (Fintype.card K : ℚ) * Fintype.card K
    rw [div_mul_cancel₀ _ hκ.ne']
    exact h'
  have hbad := rbr_prob m (fun t pre => aux_sumcheck_Good F g c0 t pre)
    (fun _ => 2 / (Fintype.card K : ℚ)) (fun _ => by positivity) h0 hstep m le_rfl
  have hset : Finset.univ.filter (fun ρ : Fin m → K => scOk g c0 m ρ ∧ scClaim g c0 m ρ = F ρ) =
      Finset.univ.filter (fun pre : Fin m → K => aux_sumcheck_Good F g c0 m pre) := by
    apply Finset.filter_congr
    intro ρ _
    unfold aux_sumcheck_Good
    rw [Nat.sub_self, aux_sumcheck_S_zero_full]
  rw [hset]
  refine hbad.trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  ring

end SumCheckMain

#print axioms aux_sumcheck_pt_no_junk
#print axioms aux_sumcheck_snoc_val
#print axioms aux_sumcheck_pt_cons
#print axioms aux_sumcheck_S_succ
#print axioms aux_sumcheck_S_zero_full
#print axioms aux_sumcheck_S_zero_pre
#print axioms aux_sumcheck_pt_update
#print axioms aux_sumcheck_poly
#print axioms aux_sumcheck_honest
#print axioms aux_sumcheck_step
#print axioms sumcheck_sound

end Cprs
