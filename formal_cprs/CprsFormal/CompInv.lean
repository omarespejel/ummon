import CprsFormal.CompAlg

/-!
# Completeness of the Phase-B model: the invariant of the honest prover
-/

open Polynomial Finset

namespace Cprs
namespace Complete

variable {K : Type*} [Field K] [DecidableEq K]

/-- The honest prover strategy for the polynomial `P`: it keeps the folded polynomial
`P_t = Pt P t pre`, sends `g_t = E_t * eq₁(ρ_t, X) * (m₀ + m₁ X)`, commits the oracle
`O_t = P_t` on the whole field, and `c_final = (P_m).coeff 0`. -/
noncomputable def honest (I : PhaseBInst K) (P : K[X]) : Strategy K I.m where
  g := fun t pre => gPoly (Et I.ρ t pre) (I.ρ.getD t 0)
    (mle (evenP (Pt P t pre)) (I.ρ.drop (t + 1))) (mle (oddP (Pt P t pre)) (I.ρ.drop (t + 1)))
  gdeg := fun t pre => gPoly_natDegree _ _ _ _
  O := fun t pre => fun y => (Pt P t pre).eval y
  cfin := fun ch => (Pt P I.m ch).coeff 0

/-- The defining equations of the honest strategy (so that the proofs below never unfold
`honest`). -/
structure IsHonest (I : PhaseBInst K) (P : K[X]) (S : Strategy K I.m) : Prop where
  g : ∀ (t : ℕ) (pre : Fin t → K), S.g t pre =
    gPoly (Et I.ρ t pre) (I.ρ.getD t 0)
      (mle (evenP (Pt P t pre)) (I.ρ.drop (t + 1))) (mle (oddP (Pt P t pre)) (I.ρ.drop (t + 1)))
  O : ∀ (t : ℕ) (pre : Fin t → K) (y : K), S.O t pre y = (Pt P t pre).eval y
  cfin : ∀ ch : Fin I.m → K, S.cfin ch = (Pt P I.m ch).coeff 0

theorem isHonest_honest (I : PhaseBInst K) (P : K[X]) : IsHonest I P (honest I P) :=
  ⟨fun _ _ => rfl, fun _ _ _ => rfl, fun _ => rfl⟩

/-- The invariant maintained by the honest prover along every challenge prefix. -/
structure HInv (I : PhaseBInst K) (P : K[X]) (S : Strategy K I.m) (t : ℕ)
    (pre : Fin t → K) : Prop where
  ok : (I.state S t pre).ok
  E : (I.state S t pre).E = Et I.ρ t pre
  c : (I.state S t pre).c = Et I.ρ t pre * mle (Pt P t pre) (I.ρ.drop t)
  W : ∀ y ∈ I.D t, (I.state S t pre).W y = (Pt P t pre).eval y
  ν : ∀ y ∈ I.D t, (I.state S t pre).ν y = 1
  chk : ∀ q ∈ I.D I.t1, (I.state S t pre).chk q

theorem hinv_zero (I : PhaseBInst K) (P : K[X]) (S : Strategy K I.m)
    (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x) (hc : I.c0 = mle P I.ρ) (pre : Fin 0 → K) :
    HInv I P S 0 pre := by
  have e0 : I.state S 0 pre =
      { W := I.π, ν := fun _ => 1, c := I.c0, E := 1, ok := True, chk := fun _ => True } := by
    first
      | rfl
      | simp [PhaseBInst.state]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · first
      | exact True.intro
      | (rw [e0]; exact True.intro)
  · first
      | rfl
      | (rw [e0]; rfl)
  · rw [e0]
    show I.c0 = 1 * mle P (I.ρ.drop 0)
    rw [List.drop_zero, one_mul]
    exact hc
  · intro y hy
    first
      | exact hpi y hy
      | (rw [e0]; exact hpi y hy)
  · intro y _
    first
      | rfl
      | (rw [e0]; rfl)
  · intro q _
    first
      | exact True.intro
      | (rw [e0]; exact True.intro)

theorem hinv_step (I : PhaseBInst K) (hWF : I.WF) (P : K[X]) (S : Strategy K I.m)
    (hS : IsHonest I P S) (t : ℕ) (ht : t < I.m) (pre : Fin t → K) (x : K)
    (IH : HInv I P S t pre) :
    HInv I P S (t + 1) (Fin.snoc (α := fun _ => K) pre x) := by
  have hst : I.state S (t + 1) (Fin.snoc (α := fun _ => K) pre x) =
      I.step S t pre (I.state S t pre) x := Aux.GoodStep.state_succ_snoc I S t pre x
  have hPt : Pt P (t + 1) (Fin.snoc (α := fun _ => K) pre x) = foldPoly x (Pt P t pre) :=
    Pt_snoc P t pre x
  have hEt : Et I.ρ (t + 1) (Fin.snoc (α := fun _ => K) pre x) =
      Et I.ρ t pre * eq1 (I.ρ.getD t 0) x := Et_snoc I.ρ t pre x
  have hlen : t < I.ρ.length := by rw [hWF.ρ_len]; exact ht
  have hdrop : I.ρ.drop t = I.ρ.getD t 0 :: I.ρ.drop (t + 1) := by
    rw [List.getD_eq_getElem _ _ hlen]
    exact List.drop_eq_getElem_cons hlen
  have hL : SqLayer (I.D t) (I.D (t + 1)) := hWF.layer t ht
  have hsum : (S.g t pre).eval 0 + (S.g t pre).eval 1 = (I.state S t pre).c := by
    rw [hS.g, gPoly_sum, IH.c, hdrop, mle_cons_even_odd]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- the sum-check test passes
    rw [hst, Aux.GoodStep.step_ok]
    exact ⟨IH.ok, hsum⟩
  · -- E
    rw [hst, Aux.GoodStep.step_E, IH.E, hEt]
  · -- c
    rw [hst, Aux.GoodStep.step_c, hEt, hPt, mle_foldPoly, hS.g, gPoly_eval]
  · -- the word agrees with the folded polynomial
    intro y hy
    rw [hst, hPt]
    by_cases he : I.isEnd (t + 1) = true
    · rw [Aux.GoodStep.step_W_end I S t pre _ x he, hS.O, hPt]
    · have he' : I.isEnd (t + 1) = false := by simpa using he
      rw [Aux.GoodStep.step_W_nonend I S t pre _ x he']
      exact fold_eval hL x (Pt P t pre) _ IH.W hy
  · -- the weight stays `1`
    intro y hy
    obtain ⟨hs0, hs1, -⟩ := Aux.LemmaQ.sqrt_mem hL hy
    have hfa : fibAvg (I.state S t pre).ν y = 1 := by
      unfold fibAvg
      rw [IH.ν _ hs0, IH.ν _ hs1]
      norm_num
    rw [hst]
    by_cases he : I.isEnd (t + 1) = true
    · rw [Aux.LemmaQ.step_end_ν I S t pre _ x he y]
      have hcond : S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) y =
          fold x (I.state S t pre).W y := by
        rw [hS.O, hPt]
        exact (fold_eval hL x (Pt P t pre) _ IH.W hy).symm
      rw [if_pos hcond]
      exact hfa
    · have he' : I.isEnd (t + 1) = false := by simpa using he
      rw [Aux.LemmaQ.step_ne_ν I S t pre _ x he' y]
      exact hfa
  · -- the query-chain checks pass
    intro q hq
    rw [hst]
    by_cases he : I.isEnd (t + 1) = true
    · rw [Aux.LemmaQ.step_end_chk I S t pre _ x he q]
      refine ⟨IH.chk q hq, ?_⟩
      obtain ⟨h1, h2⟩ := hWF.end_range (t + 1) he
      have hmem := Aux.LemmaQ.pow_mem I hWF hq (t + 1) h1 h2.le
      rw [hS.O, hPt]
      exact (fold_eval hL x (Pt P t pre) _ IH.W hmem).symm
    · have he' : I.isEnd (t + 1) = false := by simpa using he
      rw [Aux.LemmaQ.step_ne_chk I S t pre _ x he' q]
      exact IH.chk q hq

theorem hinv_all (I : PhaseBInst K) (hWF : I.WF) (P : K[X]) (S : Strategy K I.m)
    (hS : IsHonest I P S) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x) (hc : I.c0 = mle P I.ρ) :
    ∀ (t : ℕ), t ≤ I.m → ∀ pre : Fin t → K, HInv I P S t pre := by
  intro t
  induction t with
  | zero =>
    intro _ pre
    exact hinv_zero I P S hpi hc pre
  | succ t ih =>
    intro hle pre
    have h := hinv_step I hWF P S hS t (by omega) (Fin.init pre) (pre (Fin.last t))
      (ih (by omega) (Fin.init pre))
    have e : (Fin.snoc (α := fun _ => K) (Fin.init pre) (pre (Fin.last t)) : Fin (t + 1) → K) =
        pre := Fin.snoc_init_self pre
    rw [e] at h
    exact h

/-- Final step: the verifier's last identity, the query chains and the final weight. -/
theorem final_facts (I : PhaseBInst K) (hWF : I.WF) (P : K[X]) (hP : P ∈ degreeLT K (2 ^ I.m))
    (S : Strategy K I.m) (hS : IsHonest I P S) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) (ch : Fin I.m → K) :
    I.Accept0 S ch ∧ (∀ q ∈ I.D I.t1, (I.final S ch).chk q) ∧
      (∀ y ∈ I.D I.m, (I.final S ch).ν y = 1) := by
  have IH := hinv_all I hWF P S hS hpi hc I.m le_rfl ch
  have hmem : Pt P I.m ch ∈ degreeLT K 1 := by
    have h := Pt_mem P I.m hP I.m le_rfl ch
    rw [Nat.sub_self, pow_zero] at h
    exact h
  have hconst : ∀ z, (Pt P I.m ch).eval z = S.cfin ch := by
    intro z
    rw [hS.cfin ch]
    exact eval_const_of_mem _ hmem z
  have hρ : I.ρ.drop I.m = [] := List.drop_eq_nil_of_le (by rw [hWF.ρ_len])
  have hνdef : ∀ y, (I.final S ch).ν y =
      if S.cfin ch = (I.state S I.m ch).W y then (I.state S I.m ch).ν y else 0 := fun y => rfl
  have hchk : ∀ q, (I.final S ch).chk q ↔
      ((I.state S I.m ch).chk q ∧ S.cfin ch = (I.state S I.m ch).W (q ^ 2 ^ (I.m - I.t1))) :=
    fun q => Iff.rfl
  have hWc : ∀ y ∈ I.D I.m, S.cfin ch = (I.state S I.m ch).W y := by
    intro y hy
    rw [IH.W y hy, hconst y]
  have hacc : I.Accept0 S ch := by
    unfold PhaseBInst.Accept0
    refine ⟨?_, ?_⟩
    · exact IH.ok
    · show (I.state S I.m ch).c = (I.state S I.m ch).E * S.cfin ch
      rw [IH.c, IH.E, hρ, mle_nil, hS.cfin ch]
  refine ⟨hacc, ?_, ?_⟩
  · intro q hq
    rw [hchk]
    exact ⟨IH.chk q hq, hWc _ (Aux.LemmaQ.pow_mem I hWF hq I.m hWF.t1_le le_rfl)⟩
  · intro y hy
    rw [hνdef, if_pos (hWc y hy)]
    exact IH.ν y hy

end Complete
end Cprs
