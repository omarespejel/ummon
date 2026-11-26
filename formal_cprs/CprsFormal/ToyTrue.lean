import CprsFormal.Complete
import CprsFormal.ToyFalse

/-!
# Non-vacuity, part 2: a true claim on the toy instance is accepted by the honest prover

`P = 5 + 3 X` over `F_17` with the instance of `ToyFalse` (`m = s = 1`, `ζ = 4`, `ρ = (7)`,
`t₁ = 1`): `π = P|_{D_0}`, `c₀ = P~(ρ)`.
-/

open Polynomial Finset

namespace Cprs
namespace Toy

/-- `P = 5 + 3 X` over `F_17`. -/
noncomputable def toyP : Polynomial (ZMod 17) := C 5 + C 3 * Polynomial.X

theorem toyP_mem : toyP ∈ degreeLT (ZMod 17) (2 ^ 1) := by
  rw [mem_degreeLT, degree_lt_iff_coeff_zero]
  intro n hn
  have hn2 : 2 ≤ n := by simpa using hn
  have h0 : n ≠ 0 := by omega
  have h1 : 1 ≠ n := by omega
  simp [toyP, coeff_add, coeff_C, coeff_X, h0, h1]

/-- The true toy instance: `π = P`, `c₀ = P~(ρ)` (which equals `9` in `F_17`). -/
noncomputable def toyTrue : PhaseBInst (ZMod 17) :=
  toyInst (fun x => toyP.eval x) (mle toyP [7])

theorem toyTrue_wf : toyTrue.WF := toyInst_wf _ _

theorem toyTrue_rel :
    Rel (toyTrue.D 0) (2 ^ toyTrue.m) (3 / 4) toyTrue.ρ (fun _ => 1) toyTrue.π toyTrue.c0 1 :=
  PhaseBInst.rel_init toyTrue toyTrue_wf toyP toyP_mem (fun _ _ => rfl) rfl (3 / 4) (by norm_num)

/-- **Non-vacuity (true claim, honest prover accepted).** -/
theorem nonvacuity_true : ∃ I : PhaseBInst (ZMod 17), I.WF ∧
    Rel (I.D 0) (2 ^ I.m) (3 / 4) I.ρ (fun _ => 1) I.π I.c0 1 ∧
    ∃ S : Strategy (ZMod 17) I.m, ∀ ch : Fin I.m → ZMod 17,
      I.Accept0 S ch ∧ ∀ q ∈ I.D I.t1, (I.final S ch).chk q :=
  ⟨toyTrue, toyTrue_wf, toyTrue_rel,
    PhaseBInst.completeness toyTrue toyTrue_wf toyP toyP_mem (fun _ _ => rfl) rfl⟩

open scoped Classical in
/-- For the honest strategy on the true toy instance **every** (challenge, query) pair is
accepted: the accepted fraction in the statement of `PhaseBInst.soundness` is exactly `1`. -/
theorem toy_honest_accept_fraction (Q : ℕ) :
    ((Finset.univ.filter (fun p : (Fin toyTrue.m → ZMod 17) × (Fin Q → ↥(toyTrue.D toyTrue.t1)) =>
        toyTrue.Accept0 (Complete.honest toyTrue toyP) p.1 ∧
          ∀ i, (toyTrue.final (Complete.honest toyTrue toyP) p.1).chk (p.2 i))).card : ℚ) /
      ((Fintype.card (ZMod 17) : ℚ) ^ toyTrue.m * ((toyTrue.D toyTrue.t1).card : ℚ) ^ Q) = 1 :=
  PhaseBInst.honest_accept_fraction toyTrue toyTrue_wf toyP toyP_mem (fun _ _ => rfl) rfl Q

/-- The initial claim of the true toy instance, computed by hand: `P~(7) = 5 + 7 * 3 = 9`. -/
theorem toyP_c0 : mle toyP [7] = (9 : ZMod 17) := by
  first
    | (simp [mle, mleAux, toyP]; done)
    | (simp [mle, mleAux, toyP]; decide)

#print axioms nonvacuity_true
#print axioms toy_honest_accept_fraction

end Toy
end Cprs
