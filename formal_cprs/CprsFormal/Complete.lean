import CprsFormal.CompInv

/-!
# Completeness of the Phase-B model

The state machine of `CprsFormal.PhaseB` accepts the honest prover: for every instance with a true
initial claim `R_0(π, 1, c₀, 1)` witnessed by a polynomial `P` (`< 2^m` coefficients, `π = P` on
`D_0`, `c₀ = P~(ρ)`) there is a prover strategy that is accepted for every challenge vector and
passes every query-chain check, whatever the queries are.  The final weight mass is `1`, so the
acceptance fraction in `PhaseBInst.soundness` is exactly `1` for the honest strategy.
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- **Completeness.**  A true initial claim is accepted by the honest prover for every challenge
vector, and every query point of `D_(t₁)` passes the chain checks. -/
theorem PhaseBInst.completeness (I : PhaseBInst K) (hWF : I.WF) (P : K[X])
    (hP : P ∈ degreeLT K (2 ^ I.m)) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) :
    ∃ S : Strategy K I.m, ∀ ch : Fin I.m → K,
      I.Accept0 S ch ∧ ∀ q ∈ I.D I.t1, (I.final S ch).chk q := by
  refine ⟨Complete.honest I P, fun ch => ?_⟩
  obtain ⟨h1, h2, -⟩ :=
    Complete.final_facts I hWF P hP _ (Complete.isHonest_honest I P) hpi hc ch
  exact ⟨h1, h2⟩

/-- The initial relation `R_0(π, 1, c₀, 1)` holds for every threshold `β ≤ 1` (so the hypothesis
`hfalse` of `PhaseBInst.soundness` is refutable for a true claim). -/
theorem PhaseBInst.rel_init (I : PhaseBInst K) (hWF : I.WF) (P : K[X])
    (hP : P ∈ degreeLT K (2 ^ I.m)) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) (β : ℚ) (hβ : β ≤ 1) :
    Rel (I.D 0) (2 ^ I.m) β I.ρ (fun _ => 1) I.π I.c0 1 := by
  have hcard : (I.D 0).card ≠ 0 := by
    rw [hWF.card 0 (Nat.zero_le _)]
    exact pow_ne_zero _ (by norm_num)
  refine ⟨P, hP, ?_, ?_⟩
  · exact hβ.trans_eq
      (Complete.wagree_eq_one hcard (fun _ => (1 : ℚ)) I.π (fun x => P.eval x)
        (fun _ _ => rfl) hpi).symm
  · rw [one_mul]
    exact hc

/-- The honest strategy is never doomed: `Good β S t pre` holds for every prefix, every
`t ≤ m` and every `β ≤ 1`. -/
theorem PhaseBInst.honest_good (I : PhaseBInst K) (hWF : I.WF) (P : K[X])
    (hP : P ∈ degreeLT K (2 ^ I.m)) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) (β : ℚ) (hβ : β ≤ 1) (t : ℕ) (ht : t ≤ I.m)
    (pre : Fin t → K) : I.Good β (Complete.honest I P) t pre := by
  have IH := Complete.hinv_all I hWF P _ (Complete.isHonest_honest I P) hpi hc t ht pre
  have hcard : (I.D t).card ≠ 0 := by
    rw [hWF.card t ht]
    exact pow_ne_zero _ (by norm_num)
  refine ⟨IH.ok, Complete.Pt P t pre, Complete.Pt_mem P I.m hP t ht pre, ?_, ?_⟩
  · exact hβ.trans_eq (Complete.wagree_eq_one hcard _ _ _ IH.ν IH.W).symm
  · rw [IH.c, IH.E]

/-- The final weight of the honest strategy has mass exactly `1`. -/
theorem PhaseBInst.honest_final_mass (I : PhaseBInst K) (hWF : I.WF) (P : K[X])
    (hP : P ∈ degreeLT K (2 ^ I.m)) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) (ch : Fin I.m → K) :
    (∑ y ∈ I.D I.m, (I.final (Complete.honest I P) ch).ν y) / (I.D I.m).card = 1 := by
  obtain ⟨-, -, h3⟩ :=
    Complete.final_facts I hWF P hP _ (Complete.isHonest_honest I P) hpi hc ch
  have hc0 : (I.D I.m).card ≠ 0 := by
    rw [hWF.card I.m le_rfl]
    exact pow_ne_zero _ (by norm_num)
  have hcard : ((I.D I.m).card : ℚ) ≠ 0 := by exact_mod_cast hc0
  have hsum : ∑ y ∈ I.D I.m, (I.final (Complete.honest I P) ch).ν y =
      ((I.D I.m).card : ℚ) := by
    calc ∑ y ∈ I.D I.m, (I.final (Complete.honest I P) ch).ν y = ∑ y ∈ I.D I.m, (1 : ℚ) :=
          Finset.sum_congr rfl h3
      _ = ((I.D I.m).card : ℚ) := by simp
  rw [hsum]
  exact div_self hcard

open scoped Classical in
/-- Perfect completeness in the form of the soundness theorem: for the honest strategy and a true
claim, **every** pair (challenge vector, query vector) is accepted, so the accepted fraction is
exactly `1`. -/
theorem PhaseBInst.honest_accept_fraction [Fintype K] (I : PhaseBInst K) (hWF : I.WF) (P : K[X])
    (hP : P ∈ degreeLT K (2 ^ I.m)) (hpi : ∀ x ∈ I.D 0, I.π x = P.eval x)
    (hc : I.c0 = mle P I.ρ) (Q : ℕ) :
    ((Finset.univ.filter (fun p : (Fin I.m → K) × (Fin Q → ↥(I.D I.t1)) =>
        I.Accept0 (Complete.honest I P) p.1 ∧
          ∀ i, (I.final (Complete.honest I P) p.1).chk (p.2 i))).card : ℚ) /
      ((Fintype.card K : ℚ) ^ I.m * ((I.D I.t1).card : ℚ) ^ Q) = 1 := by
  have hall : Finset.univ.filter (fun p : (Fin I.m → K) × (Fin Q → ↥(I.D I.t1)) =>
        I.Accept0 (Complete.honest I P) p.1 ∧
          ∀ i, (I.final (Complete.honest I P) p.1).chk (p.2 i)) = Finset.univ := by
    apply Finset.filter_true_of_mem
    intro p _
    obtain ⟨h1, h2, -⟩ :=
      Complete.final_facts I hWF P hP _ (Complete.isHonest_honest I P) hpi hc p.1
    exact ⟨h1, fun i => h2 _ (p.2 i).2⟩
  rw [hall, Finset.card_univ]
  have hcardK : (0 : ℚ) < (Fintype.card K : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(0 : K)⟩
  have hn : (0 : ℚ) < ((I.D I.t1).card : ℚ) := by
    have hc1 := hWF.card I.t1 hWF.t1_le
    rw [hc1]
    positivity
  have hcardpow : (Fintype.card ((Fin I.m → K) × (Fin Q → ↥(I.D I.t1))) : ℚ) =
      (Fintype.card K : ℚ) ^ I.m * ((I.D I.t1).card : ℚ) ^ Q := by
    simp [Fintype.card_prod, Fintype.card_fun]
  rw [hcardpow]
  exact div_self (mul_pos (pow_pos hcardK _) (pow_pos hn _)).ne'

#print axioms PhaseBInst.completeness
#print axioms PhaseBInst.rel_init
#print axioms PhaseBInst.honest_good
#print axioms PhaseBInst.honest_final_mass
#print axioms PhaseBInst.honest_accept_fraction

end Cprs
