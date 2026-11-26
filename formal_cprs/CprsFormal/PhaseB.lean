import CprsFormal.FoldStep
import CprsFormal.Commit
import CprsFormal.Rbr
import CprsFormal.L1

/-!
# Phase B as a state machine, and the round-by-round Theorem B'

The prover's moves, the verifier's checks and the state relation of Phase B, and Theorem B'.

An *instance* fixes the layers `D_0, …, D_m`, the Phase-A point `ρ`, the virtual layer-0 word `π`
and the initial claim `c₀`.  A *strategy* of the (deterministic, adaptive) prover gives, for every
challenge prefix, the round polynomial `g_t`, the block-end oracle `O_t` (only used when `t` is a
block end) and the final constant `c_final`.  `state` is the pair `(W, ν)`, claim `c`, factor `E`
and "all checks so far passed" flag, together with the accumulated query-chain checks `chk` (a
predicate on a query point of the first oracle layer `D_{t₁}`).
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- The data of a Phase-B instance. -/
structure PhaseBInst (K : Type*) [Field K] where
  /-- number of binary rounds (`m = n - 2`) -/
  m : ℕ
  /-- rate exponent `s ≥ 1` (rate `2^(-s)`) -/
  s : ℕ
  /-- layer domains `D_t = ⟨ζ^(2^t)⟩`, `t = 0..m` -/
  D : ℕ → Finset K
  /-- the Phase-A point `ρ = (ρ_0, …, ρ_{m-1})` -/
  ρ : List K
  /-- block ends strictly inside `(0, m)`: rounds after which an oracle is committed -/
  isEnd : ℕ → Bool
  /-- the first committed layer: the first block end, or `m` if there is a single block -/
  t1 : ℕ
  /-- the virtual layer-0 word `π_μ` -/
  π : K → K
  /-- the initial claim `c₀` -/
  c0 : K

/-- Well-formedness of an instance (all of this holds for `D_t = ⟨ζ^(2^t)⟩`, see `Layers.lean`). -/
structure PhaseBInst.WF (I : PhaseBInst K) : Prop where
  s_pos : 1 ≤ I.s
  ρ_len : I.ρ.length = I.m
  layer : ∀ t, t < I.m → SqLayer (I.D t) (I.D (t + 1))
  card : ∀ t, t ≤ I.m → (I.D t).card = 2 ^ (I.m + I.s - t)
  t1_pos : 0 < I.t1
  t1_le : I.t1 ≤ I.m
  end_range : ∀ t, I.isEnd t = true → I.t1 ≤ t ∧ t < I.m
  end_t1 : I.t1 < I.m → I.isEnd I.t1 = true
  no_end_before : ∀ t, 0 < t → t < I.t1 → I.isEnd t = false

/-- A deterministic adaptive prover strategy. -/
structure Strategy (K : Type*) [Field K] (m : ℕ) where
  /-- the round polynomial `g_t` (sent as its values at `0, 1, 2`: degree `≤ 2`) -/
  g : (t : ℕ) → (Fin t → K) → K[X]
  gdeg : ∀ t pre, (g t pre).natDegree ≤ 2
  O : (t : ℕ) → (Fin t → K) → (K → K)
  cfin : (Fin m → K) → K

/-- Running state of the protocol after a challenge prefix. -/
structure St (K : Type*) where
  W : K → K
  ν : K → ℚ
  c : K
  E : K
  ok : Prop
  chk : K → Prop

/-- One round: the prover sends `g_t`, the verifier checks `g_t(0) + g_t(1) = c_t` and draws `x`;
the word is folded, the weight fibre-averaged, `c_{t+1} = g_t(x)`, `E_{t+1} = E_t eq₁(ρ_t, x)`.
If `t+1` is a block end the prover commits the oracle `O`: the new base word is `O` and the new
base weight is `ν^v · 1[O = V]`; the query-chain check `O = V` is recorded in `chk`. -/
noncomputable def PhaseBInst.step (I : PhaseBInst K) {m : ℕ}
    (S : Strategy K m) (t : ℕ) (pre : Fin t → K) (st : St K) (x : K) : St K :=
  let g := S.g t pre
  let W' := fold x st.W
  let ν' := fibAvg st.ν
  if I.isEnd (t + 1) then
    let O := S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x)
    { W := O
      ν := fun y => if O y = W' y then ν' y else 0
      c := g.eval x
      E := st.E * eq1 (I.ρ.getD t 0) x
      ok := st.ok ∧ g.eval 0 + g.eval 1 = st.c
      chk := fun q => st.chk q ∧ O (q ^ 2 ^ (t + 1 - I.t1)) = W' (q ^ 2 ^ (t + 1 - I.t1)) }
  else
    { W := W'
      ν := ν'
      c := g.eval x
      E := st.E * eq1 (I.ρ.getD t 0) x
      ok := st.ok ∧ g.eval 0 + g.eval 1 = st.c
      chk := st.chk }

/-- The state after a challenge prefix of length `t`. -/
noncomputable def PhaseBInst.state (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) :
    (t : ℕ) → (Fin t → K) → St K
  | 0, _ => { W := I.π, ν := fun _ => 1, c := I.c0, E := 1, ok := True, chk := fun _ => True }
  | t + 1, pre =>
      I.step S t (Fin.init pre) (PhaseBInst.state I S t (Fin.init pre)) (pre (Fin.last t))

/-- The state after the last challenge and the commitment of `c_final`: the base word is the
constant `c_final` on `D_m`, the weight is `ν · 1[c_final = V_m]`, and the check
`c_final = V_m(q^(2^(m-t₁)))` is added to the query chain. -/
noncomputable def PhaseBInst.final (I : PhaseBInst K) (S : Strategy K I.m) (ch : Fin I.m → K) :
    St K :=
  let st := I.state S I.m ch
  let cf := S.cfin ch
  { W := fun _ => cf
    ν := fun y => if cf = st.W y then st.ν y else 0
    c := st.c
    E := st.E
    ok := st.ok
    chk := fun q => st.chk q ∧ cf = st.W (q ^ 2 ^ (I.m - I.t1)) }

/-- `Good t pre`: all checks so far passed and the state relation `R_t` holds for the current
(post-commitment) pair.  "Doomed" means `¬ Good`. -/
def PhaseBInst.Good (I : PhaseBInst K) (β : ℚ) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin t → K) : Prop :=
  (I.state S t pre).ok ∧
    Rel (I.D t) (2 ^ (I.m - t)) β (I.ρ.drop t) (I.state S t pre).ν (I.state S t pre).W
      (I.state S t pre).c (I.state S t pre).E

/-- The non-query part of the verifier: all `g_t` checks and the final identity
`c_m = c_final · eq(ρ, a)`. -/
def PhaseBInst.Accept0 (I : PhaseBInst K) (S : Strategy K I.m) (ch : Fin I.m → K) : Prop :=
  (I.final S ch).ok ∧ (I.final S ch).c = (I.final S ch).E * S.cfin ch

/-! ### Helper lemmas for `state_weight_mem` and `good_step` -/

namespace Aux.GoodStep

/-- Appending the challenge `x` to the prefix applies one `step`. -/
theorem state_succ_snoc (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin t → K) (x : K) :
    I.state S (t + 1) (Fin.snoc (α := fun _ => K) pre x) =
      I.step S t pre (I.state S t pre) x := by
  simp only [PhaseBInst.state, Fin.init_snoc, Fin.snoc_last]

/-- Unfolding of the state for an arbitrary prefix of length `t + 1`. -/
theorem state_succ (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin (t + 1) → K) :
    I.state S (t + 1) pre =
      I.step S t (Fin.init pre) (I.state S t (Fin.init pre)) (pre (Fin.last t)) := by
  simp only [PhaseBInst.state]

/-- Both square roots of a point of `D'` lie in `D`. -/
theorem sqrtK_mem {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ∈ D ∧ -sqrtK y ∈ D := by
  obtain ⟨x, hxD, hx⟩ := h.sq_surj y hy
  have h1 : sqrtK y ^ 2 = y := sqrtK_sq ⟨x, hx⟩
  have h2 : sqrtK y ^ 2 = x ^ 2 := by rw [h1, hx]
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp h2 with h3 | h3
  · rw [h3]; exact ⟨hxD, h.neg_mem x hxD⟩
  · rw [h3, neg_neg]; exact ⟨h.neg_mem x hxD, hxD⟩

/-- The fibre average of a `[0,1]`-valued weight on `D` is `[0,1]`-valued on `D'`. -/
theorem fibAvg_mem {D D' : Finset K} (h : SqLayer D D') {ν : K → ℚ}
    (h0 : ∀ x ∈ D, 0 ≤ ν x) (h1 : ∀ x ∈ D, ν x ≤ 1) {y : K} (hy : y ∈ D') :
    0 ≤ fibAvg ν y ∧ fibAvg ν y ≤ 1 := by
  obtain ⟨ha, hb⟩ := sqrtK_mem h hy
  have a0 := h0 _ ha
  have a1 := h1 _ ha
  have b0 := h0 _ hb
  have b1 := h1 _ hb
  unfold fibAvg
  constructor <;> linarith

theorem step_ok (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) :
    (I.step S t pre st x).ok = (st.ok ∧ (S.g t pre).eval 0 + (S.g t pre).eval 1 = st.c) := by
  unfold PhaseBInst.step
  split_ifs <;> rfl

theorem step_c (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) :
    (I.step S t pre st x).c = (S.g t pre).eval x := by
  unfold PhaseBInst.step
  split_ifs <;> rfl

theorem step_E (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) :
    (I.step S t pre st x).E = st.E * eq1 (I.ρ.getD t 0) x := by
  unfold PhaseBInst.step
  split_ifs <;> rfl

theorem step_W_nonend (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = false) :
    (I.step S t pre st x).W = fold x st.W := by
  simp only [PhaseBInst.step, h, Bool.false_eq_true, ite_false]

theorem step_ν_nonend (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = false) :
    (I.step S t pre st x).ν = fibAvg st.ν := by
  simp only [PhaseBInst.step, h, Bool.false_eq_true, ite_false]

theorem step_W_end (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = true) :
    (I.step S t pre st x).W = S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) := by
  simp only [PhaseBInst.step, h, ite_true]

theorem step_ν_end (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = true) :
    (I.step S t pre st x).ν = fun y =>
      if S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) y = fold x st.W y
      then fibAvg st.ν y else 0 := by
  simp only [PhaseBInst.step, h, ite_true]

/-- One step keeps the weights in `[0,1]` on the next layer. -/
theorem step_weight_mem (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (hL : SqLayer (I.D t) (I.D (t + 1)))
    (h0 : ∀ y ∈ I.D t, 0 ≤ st.ν y) (h1 : ∀ y ∈ I.D t, st.ν y ≤ 1) :
    ∀ y ∈ I.D (t + 1), 0 ≤ (I.step S t pre st x).ν y ∧ (I.step S t pre st x).ν y ≤ 1 := by
  intro y hy
  have hf := fibAvg_mem hL h0 h1 hy
  cases hE : I.isEnd (t + 1)
  · rw [step_ν_nonend I S t pre st x hE]; exact hf
  · rw [step_ν_end I S t pre st x hE]
    simp only
    split_ifs
    · exact hf
    · exact ⟨le_refl _, zero_le_one⟩

end Aux.GoodStep

/-! ### Helper lemmas for the query invariant (Lemma Q) and the final commitment step -/

namespace Aux.LemmaQ

/-- `state` at a snoc'd prefix is the step from the state of the prefix. -/
theorem state_snoc (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (x : K) :
    I.state S (t + 1) (Fin.snoc (α := fun _ => K) pre x) = I.step S t pre (I.state S t pre) x := by
  rw [PhaseBInst.state.eq_2, Fin.init_snoc, Fin.snoc_last]

/-- Both square roots of a point of the upper layer lie in the lower layer, are nonzero and
distinct. -/
theorem sqrt_mem {D D' : Finset K} (h : SqLayer D D') {y : K} (hy : y ∈ D') :
    sqrtK y ∈ D ∧ -sqrtK y ∈ D ∧ sqrtK y ^ 2 = y ∧ sqrtK y ≠ 0 ∧ sqrtK y ≠ -sqrtK y := by
  obtain ⟨x, hx, hxy⟩ := h.sq_surj y hy
  have hs : sqrtK y ^ 2 = y := sqrtK_sq ⟨x, hxy⟩
  have hsx : sqrtK y = x ∨ sqrtK y = -x :=
    sq_eq_sq_iff_eq_or_eq_neg.mp (by rw [hs, hxy])
  have hx0 : x ≠ 0 := fun h0 => h.zero_notMem (h0 ▸ hx)
  have hnx : -x ∈ D := h.neg_mem x hx
  refine ⟨?_, ?_, hs, ?_, ?_⟩
  · rcases hsx with e | e
    · rw [e]; exact hx
    · rw [e]; exact hnx
  · rcases hsx with e | e
    · rw [e]; exact hnx
    · rw [e, neg_neg]; exact hx
  · intro h0
    rcases hsx with e | e
    · exact hx0 (e ▸ h0)
    · exact hx0 (neg_eq_zero.mp (e ▸ h0))
  · intro h0
    have h2 : (2 : K) * sqrtK y = 0 := by linear_combination h0
    rcases mul_eq_zero.mp h2 with h3 | h3
    · exact h.two_ne_zero h3
    · rcases hsx with e | e
      · exact hx0 (e ▸ h3)
      · exact hx0 (neg_eq_zero.mp (e ▸ h3))

omit [DecidableEq K] in
/-- Query points of `D_{t₁}` are sent into `D_t` by `q ↦ q ^ 2 ^ (t - t₁)`. -/
theorem pow_mem (I : PhaseBInst K) (hWF : I.WF) {q : K} (hq : q ∈ I.D I.t1) :
    ∀ t, I.t1 ≤ t → t ≤ I.m → q ^ 2 ^ (t - I.t1) ∈ I.D t := by
  intro t ht
  induction t, ht using Nat.le_induction with
  | base => intro _; simpa using hq
  | succ t ht ih =>
    intro hle
    have h1 : q ^ 2 ^ (t - I.t1) ∈ I.D t := ih (by omega)
    have h2 := (hWF.layer t (by omega)).sq_mem _ h1
    have h3 : q ^ 2 ^ (t + 1 - I.t1) = (q ^ 2 ^ (t - I.t1)) ^ 2 := by
      rw [show t + 1 - I.t1 = (t - I.t1) + 1 by omega, pow_succ, pow_mul]
    rw [h3]; exact h2

theorem step_end_ν (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = true) (y : K) :
    (I.step S t pre st x).ν y =
      if S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) y = fold x st.W y
        then fibAvg st.ν y else 0 := by
  simp [PhaseBInst.step, h]

theorem step_end_chk (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = true) (q : K) :
    (I.step S t pre st x).chk q ↔
      st.chk q ∧ S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) (q ^ 2 ^ (t + 1 - I.t1)) =
        fold x st.W (q ^ 2 ^ (t + 1 - I.t1)) := by
  simp [PhaseBInst.step, h]

theorem step_ne_ν (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = false) (y : K) :
    (I.step S t pre st x).ν y = fibAvg st.ν y := by
  simp [PhaseBInst.step, h]

theorem step_ne_chk (I : PhaseBInst K) {m : ℕ} (S : Strategy K m) (t : ℕ) (pre : Fin t → K)
    (st : St K) (x : K) (h : I.isEnd (t + 1) = false) (q : K) :
    (I.step S t pre st x).chk q ↔ st.chk q := by
  simp [PhaseBInst.step, h]

open scoped Classical in
/-- Splitting the query points above `y ∈ D_{t+1}` according to their image in `D_t`. -/
theorem count_split (I : PhaseBInst K) (hWF : I.WF) (P : K → Prop) (t : ℕ) (ht1 : I.t1 ≤ t)
    (htm : t < I.m) {y : K} (hy : y ∈ I.D (t + 1)) :
    ((I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧ P q)).card =
      ((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = sqrtK y ∧ P q)).card +
      ((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = -sqrtK y ∧ P q)).card := by
  obtain ⟨hs0, hs1, hs2, hs3, hs4⟩ := sqrt_mem (hWF.layer t htm) hy
  rw [← Finset.card_union_of_disjoint]
  · congr 1
    ext q
    simp only [Finset.mem_filter, Finset.mem_union]
    constructor
    · rintro ⟨hq, hqy, hP⟩
      have h1 : (q ^ 2 ^ (t - I.t1)) ^ 2 = (sqrtK y) ^ 2 := by
        rw [hs2, ← hqy, ← pow_mul, ← pow_succ, show t - I.t1 + 1 = t + 1 - I.t1 by omega]
      rcases sq_eq_sq_iff_eq_or_eq_neg.mp h1 with h | h
      · exact Or.inl ⟨hq, h, hP⟩
      · exact Or.inr ⟨hq, h, hP⟩
    · rintro (⟨hq, h, hP⟩ | ⟨hq, h, hP⟩)
      · refine ⟨hq, ?_, hP⟩
        rw [show t + 1 - I.t1 = (t - I.t1) + 1 by omega, pow_succ, pow_mul, h, hs2]
      · refine ⟨hq, ?_, hP⟩
        rw [show t + 1 - I.t1 = (t - I.t1) + 1 by omega, pow_succ, pow_mul, h, neg_sq, hs2]
  · rw [Finset.disjoint_left]
    intro q hq1 hq2
    simp only [Finset.mem_filter] at hq1 hq2
    exact hs4 (hq1.2.1.symm.trans hq2.2.1)

open scoped Classical in
/-- The fibre average of a weight satisfying the query invariant at layer `t` satisfies it at
layer `t + 1` for the same chain checks. -/
theorem count_fibAvg (I : PhaseBInst K) (hWF : I.WF) (st : St K) (t : ℕ) (ht1 : I.t1 ≤ t)
    (htm : t < I.m)
    (IH : ∀ y ∈ I.D t, st.ν y * 2 ^ (t - I.t1) =
      (((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = y ∧ st.chk q)).card : ℚ))
    {y : K} (hy : y ∈ I.D (t + 1)) :
    fibAvg st.ν y * 2 ^ (t + 1 - I.t1) =
      (((I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧ st.chk q)).card : ℚ) := by
  obtain ⟨hs0, hs1, hs2, hs3, hs4⟩ := sqrt_mem (hWF.layer t htm) hy
  rw [count_split I hWF st.chk t ht1 htm hy, Nat.cast_add, ← IH _ hs0, ← IH _ hs1]
  unfold fibAvg
  rw [show t + 1 - I.t1 = (t - I.t1) + 1 by omega, pow_succ]
  ring

open scoped Classical in
/-- Inductive step of the query invariant (one round, with or without a commitment). -/
theorem count_step (I : PhaseBInst K) (hWF : I.WF) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin t → K) (st : St K) (x : K) (ht1 : I.t1 ≤ t) (htm : t < I.m)
    (IH : ∀ y ∈ I.D t, st.ν y * 2 ^ (t - I.t1) =
      (((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = y ∧ st.chk q)).card : ℚ)) :
    ∀ y ∈ I.D (t + 1), (I.step S t pre st x).ν y * 2 ^ (t + 1 - I.t1) =
      (((I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧
        (I.step S t pre st x).chk q)).card : ℚ) := by
  intro y hy
  have hfa := count_fibAvg I hWF st t ht1 htm IH hy
  by_cases he : I.isEnd (t + 1) = true
  · rw [step_end_ν I S t pre st x he]
    by_cases hOW : S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) y = fold x st.W y
    · rw [ite_eq_left hOW, hfa]
      have hAB : (I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧ st.chk q) =
          (I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧
            (I.step S t pre st x).chk q) := by
        ext q
        simp only [Finset.mem_filter, step_end_chk I S t pre st x he]
        constructor
        · rintro ⟨hq, hqy, hc⟩
          exact ⟨hq, hqy, hc, by rw [hqy]; exact hOW⟩
        · rintro ⟨hq, hqy, hc, _⟩
          exact ⟨hq, hqy, hc⟩
      rw [hAB]
    · rw [ite_eq_right hOW, zero_mul]
      have hE : (I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧
          (I.step S t pre st x).chk q) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        rintro q _ ⟨hqy, hc⟩
        rw [step_end_chk I S t pre st x he, hqy] at hc
        exact hOW hc.2
      rw [hE]; simp
  · have he' : I.isEnd (t + 1) = false := by simpa using he
    rw [step_ne_ν I S t pre st x he', hfa]
    have hAB : (I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧ st.chk q) =
        (I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧
          (I.step S t pre st x).chk q) := by
      ext q
      simp only [Finset.mem_filter, step_ne_chk I S t pre st x he']
    rw [hAB]

/-- Before the first committed layer the weight is `1` and no check has been recorded. -/
theorem pre_step (I : PhaseBInst K) (hWF : I.WF) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin t → K) (st : St K) (x : K) (ht : t + 1 < I.t1)
    (hν : ∀ y ∈ I.D t, st.ν y = 1) (hc : ∀ q, st.chk q) :
    (∀ y ∈ I.D (t + 1), (I.step S t pre st x).ν y = 1) ∧ ∀ q, (I.step S t pre st x).chk q := by
  have he : I.isEnd (t + 1) = false := hWF.no_end_before (t + 1) (by omega) ht
  have htm : t < I.m := by have := hWF.t1_le; omega
  refine ⟨fun y hy => ?_, fun q => ?_⟩
  · obtain ⟨hs0, hs1, -⟩ := sqrt_mem (hWF.layer t htm) hy
    rw [step_ne_ν I S t pre st x he]
    unfold fibAvg
    rw [hν _ hs0, hν _ hs1]; norm_num
  · exact (step_ne_chk I S t pre st x he q).mpr (hc q)

open scoped Classical in
/-- The step into the first committed layer `t₁ = t + 1`. -/
theorem base_step (I : PhaseBInst K) (hWF : I.WF) {m : ℕ} (S : Strategy K m) (t : ℕ)
    (pre : Fin t → K) (st : St K) (x : K) (hk : I.t1 = t + 1)
    (hν : ∀ y ∈ I.D t, st.ν y = 1) (hc : ∀ q, st.chk q) :
    ∀ y ∈ I.D (t + 1), (I.step S t pre st x).ν y * 2 ^ (t + 1 - I.t1) =
      (((I.D I.t1).filter (fun q => q ^ 2 ^ (t + 1 - I.t1) = y ∧
        (I.step S t pre st x).chk q)).card : ℚ) := by
  intro y hy
  have htm : t < I.m := by have := hWF.t1_le; omega
  obtain ⟨hs0, hs1, -⟩ := sqrt_mem (hWF.layer t htm) hy
  have hfa : fibAvg st.ν y = 1 := by
    unfold fibAvg; rw [hν _ hs0, hν _ hs1]; norm_num
  have e0 : t + 1 - I.t1 = 0 := by omega
  have eD : I.D I.t1 = I.D (t + 1) := by rw [hk]
  rw [e0, eD]
  by_cases he : I.isEnd (t + 1) = true
  · rw [step_end_ν I S t pre st x he]
    by_cases hOW : S.O (t + 1) (Fin.snoc (α := fun _ => K) pre x) y = fold x st.W y
    · rw [ite_eq_left hOW, hfa]
      have hsing : (I.D (t + 1)).filter (fun q => q ^ 2 ^ 0 = y ∧
          (I.step S t pre st x).chk q) = {y} := by
        ext q
        simp only [Finset.mem_filter, Finset.mem_singleton, step_end_chk I S t pre st x he, e0,
          pow_zero, pow_one]
        constructor
        · rintro ⟨_, h, _⟩; exact h
        · rintro rfl; exact ⟨hy, rfl, hc _, hOW⟩
      rw [hsing, Finset.card_singleton]; simp
    · rw [ite_eq_right hOW]
      have hE : (I.D (t + 1)).filter (fun q => q ^ 2 ^ 0 = y ∧
          (I.step S t pre st x).chk q) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        rintro q _ ⟨hqy, hc'⟩
        rw [step_end_chk I S t pre st x he, e0, pow_zero, pow_one] at hc'
        rw [pow_zero, pow_one] at hqy
        rw [hqy] at hc'
        exact hOW hc'.2
      rw [hE]; simp
  · have he' : I.isEnd (t + 1) = false := by simpa using he
    rw [step_ne_ν I S t pre st x he', hfa]
    have hsing : (I.D (t + 1)).filter (fun q => q ^ 2 ^ 0 = y ∧
        (I.step S t pre st x).chk q) = {y} := by
      ext q
      simp only [Finset.mem_filter, Finset.mem_singleton, step_ne_chk I S t pre st x he',
        pow_zero, pow_one]
      constructor
      · rintro ⟨_, h, _⟩; exact h
      · rintro rfl; exact ⟨hy, rfl, hc _⟩
    rw [hsing, Finset.card_singleton]; simp

open scoped Classical in
/-- The full invariant, by induction on the number of rounds. -/
theorem inv_all (I : PhaseBInst K) (hWF : I.WF) {m : ℕ} (S : Strategy K m) :
    ∀ (t : ℕ) (pre : Fin t → K), t ≤ I.m →
      (t < I.t1 → (∀ y ∈ I.D t, (I.state S t pre).ν y = 1) ∧ ∀ q, (I.state S t pre).chk q) ∧
      (I.t1 ≤ t → ∀ y ∈ I.D t, (I.state S t pre).ν y * 2 ^ (t - I.t1) =
        (((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = y ∧
          (I.state S t pre).chk q)).card : ℚ)) := by
  intro t
  induction t with
  | zero =>
    intro pre _
    refine ⟨fun _ => ⟨fun y _ => ?_, fun q => ?_⟩, fun h => ?_⟩
    · simp [PhaseBInst.state.eq_1]
    · simp [PhaseBInst.state.eq_1]
    · exact absurd h (by have := hWF.t1_pos; omega)
  | succ t ih =>
    intro pre hle
    have ih' := ih (Fin.init pre) (by omega)
    rw [PhaseBInst.state.eq_2]
    refine ⟨fun hlt => ?_, fun hge => ?_⟩
    · exact pre_step I hWF S t (Fin.init pre) _ (pre (Fin.last t)) hlt (ih'.1 (by omega)).1
        (ih'.1 (by omega)).2
    · rcases Nat.lt_or_ge t I.t1 with h | h
      · exact base_step I hWF S t (Fin.init pre) _ (pre (Fin.last t)) (by omega)
          (ih'.1 h).1 (ih'.1 h).2
      · exact count_step I hWF S t (Fin.init pre) _ (pre (Fin.last t)) h (by omega)
          (ih'.2 h)

end Aux.LemmaQ

section Weights

variable (I : PhaseBInst K)

/-- The weights of the state are `[0,1]`-valued on the current layer. -/
theorem PhaseBInst.state_weight_mem (hWF : I.WF) {m : ℕ} (S : Strategy K m) :
    ∀ (t : ℕ) (pre : Fin t → K), t ≤ I.m → ∀ x ∈ I.D t,
      0 ≤ (I.state S t pre).ν x ∧ (I.state S t pre).ν x ≤ 1 := by
  intro t
  induction t with
  | zero =>
    intro pre _ x _
    simp only [PhaseBInst.state]
    exact ⟨zero_le_one, le_refl _⟩
  | succ t ih =>
    intro pre ht x hx
    rw [Aux.GoodStep.state_succ]
    have hih := ih (Fin.init pre) (by omega)
    exact Aux.GoodStep.step_weight_mem I S t (Fin.init pre) (I.state S t (Fin.init pre))
      (pre (Fin.last t)) (hWF.layer t (by omega)) (fun y hy => (hih y hy).1)
      (fun y hy => (hih y hy).2) x hx

open scoped Classical in
/-- **Round-by-round error (Theorem B (iii), with the improved bound of Lemma F').**  If the state
is doomed before the challenge `a_t`, at most `|D_{t+1}|` values of `a_t` make it good. -/
theorem PhaseBInst.good_step [Fintype K] (hWF : I.WF) (β : ℚ)
    (hβ : β = (2 ^ I.s + 1) / 2 ^ (I.s + 1)) {m : ℕ} (S : Strategy K m)
    (hCA : ∀ t, 1 ≤ t → t ≤ I.m → CorrAgree (I.D t) (2 ^ (I.m - t)) β)
    (t : ℕ) (ht : t < I.m) (pre : Fin t → K) (hnot : ¬ I.Good β S t pre) :
    (Finset.univ.filter (fun x : K => I.Good β S (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card
      ≤ (I.D (t + 1)).card := by
  by_cases hc : (I.state S t pre).ok ∧
      (S.g t pre).eval 0 + (S.g t pre).eval 1 = (I.state S t pre).c
  · obtain ⟨hok, hgc⟩ := hc
    have hlen : t < I.ρ.length := by rw [hWF.ρ_len]; exact ht
    have hdrop : I.ρ.drop t = I.ρ.getD t 0 :: I.ρ.drop (t + 1) := by
      rw [List.getD_eq_getElem _ _ hlen]; exact List.drop_eq_getElem_cons hlen
    have hk : 2 * 2 ^ (I.m - (t + 1)) = 2 ^ (I.m - t) := by
      have : I.m - t = (I.m - (t + 1)) + 1 := by omega
      rw [this, pow_succ]; ring
    have hnot' : ¬ Rel (I.D t) (2 * 2 ^ (I.m - (t + 1))) β (I.ρ.getD t 0 :: I.ρ.drop (t + 1))
        (I.state S t pre).ν (I.state S t pre).W (I.state S t pre).c (I.state S t pre).E := by
      rw [hk, ← hdrop]
      intro h
      exact hnot ⟨hok, h⟩
    have hcard : (I.D (t + 1)).card = 2 ^ (I.m + I.s - (t + 1)) := hWF.card (t + 1) (by omega)
    have hs := hWF.s_pos
    have hD' : 2 ≤ (I.D (t + 1)).card := by
      rw [hcard]
      calc 2 = 2 ^ 1 := by norm_num
        _ ≤ 2 ^ (I.m + I.s - (t + 1)) := Nat.pow_le_pow_right (by norm_num) (by omega)
    have hk' : 0 < 2 ^ (I.m - (t + 1)) := by positivity
    have hrate : ((2 ^ (I.m - (t + 1)) : ℕ) : ℚ) ≤ (2 * β - 1) * ((I.D (t + 1)).card : ℚ) := by
      rw [hcard, hβ]
      have h1 : I.m + I.s - (t + 1) = (I.m - (t + 1)) + I.s := by omega
      rw [h1]
      push_cast
      apply le_of_eq
      field_simp
      ring
    have hweights := PhaseBInst.state_weight_mem I hWF S t pre ht.le
    have hF := lemmaF' (I.D t) (I.D (t + 1)) (hWF.layer t ht) hD' (2 ^ (I.m - (t + 1))) β hk' hrate
      (hCA (t + 1) (by omega) (by omega)) (I.state S t pre).ν
      (fun x hx => (hweights x hx).1) (fun x hx => (hweights x hx).2)
      (I.state S t pre).W (I.state S t pre).c (I.state S t pre).E (I.ρ.getD t 0)
      (I.ρ.drop (t + 1)) (S.g t pre) (S.gdeg t pre) hgc hnot'
    refine le_trans (Finset.card_le_card ?_) hF
    intro x hx
    rw [Finset.mem_filter] at hx ⊢
    refine ⟨Finset.mem_univ _, ?_⟩
    have hGood := hx.2
    unfold PhaseBInst.Good at hGood
    rw [Aux.GoodStep.state_succ_snoc] at hGood
    obtain ⟨_, hRel⟩ := hGood
    rw [Aux.GoodStep.step_c, Aux.GoodStep.step_E] at hRel
    cases hE : I.isEnd (t + 1)
    · rw [Aux.GoodStep.step_W_nonend I S t pre _ x hE,
        Aux.GoodStep.step_ν_nonend I S t pre _ x hE] at hRel
      exact hRel
    · rw [Aux.GoodStep.step_W_end I S t pre _ x hE,
        Aux.GoodStep.step_ν_end I S t pre _ x hE] at hRel
      refine lemmaC (I.D (t + 1)) (2 ^ (I.m - (t + 1))) β (I.ρ.drop (t + 1))
        (fibAvg (I.state S t pre).ν) ?_ (fold x (I.state S t pre).W) _ _ _ hRel
      intro y hy
      exact (Aux.GoodStep.fibAvg_mem (hWF.layer t ht) (fun x hx => (hweights x hx).1)
        (fun x hx => (hweights x hx).2) hy).1
  · have hempty : Finset.univ.filter
        (fun x : K => I.Good β S (t + 1) (Fin.snoc (α := fun _ => K) pre x)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro x _ hx
      apply hc
      have h1 := hx.1
      rw [Aux.GoodStep.state_succ_snoc, Aux.GoodStep.step_ok] at h1
      exact h1
    rw [hempty]
    simp

#print axioms PhaseBInst.state_weight_mem
#print axioms PhaseBInst.good_step

open scoped Classical in
/-- **Query invariant (Lemma Q).** For `t ≥ t₁` the weight at `y ∈ D_t` is the fraction of the
`2^(t-t₁)` query points of `D_{t₁}` above `y` that pass the chain checks so far. -/
theorem PhaseBInst.weight_eq_chk_fraction (hWF : I.WF) {m : ℕ} (S : Strategy K m) :
    ∀ (t : ℕ) (pre : Fin t → K), I.t1 ≤ t → t ≤ I.m → ∀ y ∈ I.D t,
      (I.state S t pre).ν y * 2 ^ (t - I.t1) =
        (((I.D I.t1).filter (fun q => q ^ 2 ^ (t - I.t1) = y ∧ (I.state S t pre).chk q)).card : ℚ) := by
  intro t pre h1 h2 y hy
  exact (Aux.LemmaQ.inv_all I hWF S t pre h2).2 h1 y hy


open scoped Classical in
/-- The mass of the final weight equals the fraction of query points of `D_{t₁}` that pass
every chain check (Lemma Q, final form). -/
theorem PhaseBInst.final_mass_eq_chk_fraction (hWF : I.WF) {m : ℕ} (S : Strategy K I.m) (ch : Fin I.m → K) :
    (∑ y ∈ I.D I.m, (I.final S ch).ν y) / (I.D I.m).card =
      (((I.D I.t1).filter (fun q => (I.final S ch).chk q)).card : ℚ) / (I.D I.t1).card := by
  have hle : I.t1 ≤ I.m := hWF.t1_le
  -- `|D_{t₁}| = |D_m| · 2^(m - t₁)`
  have hcard : ((I.D I.t1).card : ℚ) = ((I.D I.m).card : ℚ) * 2 ^ (I.m - I.t1) := by
    rw [hWF.card I.t1 hle, hWF.card I.m le_rfl]
    push_cast
    rw [← pow_add]
    congr 1
    omega
  have hνdef : ∀ y, (I.final S ch).ν y =
      if S.cfin ch = (I.state S I.m ch).W y then (I.state S I.m ch).ν y else 0 := fun y => rfl
  have hchk : ∀ q, (I.final S ch).chk q ↔
      (I.state S I.m ch).chk q ∧
        S.cfin ch = (I.state S I.m ch).W (q ^ 2 ^ (I.m - I.t1)) := fun q => Iff.rfl
  -- partition the query points passing every check by their image in `D_m`
  have key : (((I.D I.t1).filter (fun q => (I.final S ch).chk q)).card : ℚ) =
      (∑ y ∈ I.D I.m, (I.final S ch).ν y) * 2 ^ (I.m - I.t1) := by
    have hmaps : Set.MapsTo (fun q : K => q ^ 2 ^ (I.m - I.t1))
        (((I.D I.t1).filter (fun q => (I.final S ch).chk q) : Finset K) : Set K)
        ((I.D I.m : Finset K) : Set K) := by
      intro q hq
      rw [Finset.mem_coe, Finset.mem_filter] at hq
      exact Aux.LemmaQ.pow_mem I hWF hq.1 I.m hle le_rfl
    rw [Finset.card_eq_sum_card_fiberwise hmaps, Nat.cast_sum, Finset.sum_mul]
    refine Finset.sum_congr rfl (fun y hy => ?_)
    have hW := I.weight_eq_chk_fraction hWF S I.m ch hle le_rfl y hy
    by_cases hc : S.cfin ch = (I.state S I.m ch).W y
    · rw [hνdef, ite_eq_left hc, hW]
      have hAB : ((I.D I.t1).filter (fun q => (I.final S ch).chk q)).filter
            (fun a => a ^ 2 ^ (I.m - I.t1) = y) =
          (I.D I.t1).filter (fun q => q ^ 2 ^ (I.m - I.t1) = y ∧ (I.state S I.m ch).chk q) := by
        ext q
        simp only [Finset.mem_filter, hchk]
        constructor
        · rintro ⟨⟨hq, hq1, _⟩, hqy⟩
          exact ⟨hq, hqy, hq1⟩
        · rintro ⟨hq, hqy, hq1⟩
          exact ⟨⟨hq, hq1, by rw [hqy]; exact hc⟩, hqy⟩
      rw [hAB]
    · rw [hνdef, ite_eq_right hc, zero_mul]
      have hE : ((I.D I.t1).filter (fun q => (I.final S ch).chk q)).filter
            (fun a => a ^ 2 ^ (I.m - I.t1) = y) = ∅ := by
        rw [Finset.filter_eq_empty_iff]
        intro q hq hqy
        rw [Finset.mem_filter, hchk] at hq
        rw [hqy] at hq
        exact hc hq.2.2
      rw [hE]; simp
  rw [key, hcard, mul_div_mul_right _ _ (pow_ne_zero _ (two_ne_zero))]

open scoped Classical in
/-- If the pre-commitment state is doomed and the verifier's non-query checks pass, the mass of
the post-commitment weight of `c_final` is below `β` (Lemma C at the final commitment, plus the
fact that the only candidate witness for a constant oracle is the constant itself). -/
theorem PhaseBInst.final_mass_lt (hWF : I.WF) (β : ℚ) (hβ0 : 0 < β) (S : Strategy K I.m)
    (ch : Fin I.m → K) (hacc : I.Accept0 S ch) (hdoom : ¬ I.Good β S I.m ch) :
    (∑ y ∈ I.D I.m, (I.final S ch).ν y) / (I.D I.m).card < β := by
  have _ := hβ0
  by_contra hge
  rw [not_lt] at hge
  apply hdoom
  have hν0 : ∀ x ∈ I.D I.m, 0 ≤ (I.state S I.m ch).ν x :=
    fun x hx => (I.state_weight_mem hWF S I.m ch le_rfl x hx).1
  have hρ : I.ρ.drop I.m = [] := List.drop_eq_nil_of_le (by rw [hWF.ρ_len])
  refine And.intro hacc.1 ?_
  refine lemmaC (I.D I.m) (2 ^ (I.m - I.m)) β (I.ρ.drop I.m) (I.state S I.m ch).ν hν0
    (I.state S I.m ch).W (fun _ => S.cfin ch) (I.state S I.m ch).c (I.state S I.m ch).E ?_
  refine ⟨Polynomial.C (S.cfin ch), ?_, ?_, ?_⟩
  · rw [mem_degreeLT]
    exact lt_of_le_of_lt degree_C_le (by simp)
  · unfold wagree
    rw [Finset.filter_true_of_mem (fun x _ => by simp)]
    exact hge
  · rw [hρ]
    simp only [mle, mleAux, coeff_C_zero]
    exact hacc.2

#print axioms Cprs.PhaseBInst.weight_eq_chk_fraction
#print axioms Cprs.PhaseBInst.final_mass_eq_chk_fraction
#print axioms Cprs.PhaseBInst.final_mass_lt

end Weights

/-! ### Auxiliary counting lemmas for Theorem B′ (helpers). -/

namespace Aux

/-- The number of query vectors all of whose coordinates pass a check is the `Q`-th power of the
number of passing query points. -/
theorem aux_thmB_card_all_pass {α : Type*} [Fintype α] (Q : ℕ) (χ : α → Prop) [DecidablePred χ] :
    (Finset.univ.filter (fun qs : Fin Q → α => ∀ i, χ (qs i))).card
      = (Finset.univ.filter χ).card ^ Q := by
  have h : (Finset.univ.filter (fun qs : Fin Q → α => ∀ i, χ (qs i)))
      = Fintype.piFinset (fun _ : Fin Q => Finset.univ.filter χ) := by
    ext qs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
  rw [h, Fintype.card_piFinset_const]

/-- Counting the points of a finset through its coercion to a type. -/
theorem aux_thmB_card_subtype_filter {α : Type*} (D : Finset α) (χ : α → Prop) [DecidablePred χ] :
    (Finset.univ.filter (fun q : ↥D => χ q.1)).card = (D.filter χ).card := by
  rw [Finset.card_filter, Finset.card_filter]
  exact Finset.sum_coe_sort D (fun x => if χ x then 1 else 0)

open scoped Classical in
/-- The union-bound split of the accepted pairs: pairs whose challenge vector lies in the
exceptional set `bad` are bounded by `|bad| · |D|^Q`; for the other challenge vectors the number
of all-passing query vectors is at most `(β |D|)^Q`. -/
theorem aux_thmB_count_split {α Ω : Type*} [Fintype Ω] (D : Finset α) (Q : ℕ) (β : ℚ)
    (hβ0 : 0 ≤ β) (bad acc : Ω → Prop) (chk : Ω → α → Prop)
    (h : ∀ ω, acc ω → ¬ bad ω → ((D.filter (chk ω)).card : ℚ) ≤ β * D.card) :
    ((Finset.univ.filter (fun p : Ω × (Fin Q → ↥D) =>
        acc p.1 ∧ ∀ i, chk p.1 (p.2 i))).card : ℚ)
      ≤ ((Finset.univ.filter bad).card : ℚ) * (D.card : ℚ) ^ Q
        + (Fintype.card Ω : ℚ) * (β * D.card) ^ Q := by
  have hfib : ((Finset.univ.filter (fun p : Ω × (Fin Q → ↥D) =>
        acc p.1 ∧ ∀ i, chk p.1 (p.2 i))).card : ℚ)
      = ∑ ω : Ω, ((Finset.univ.filter (fun qs : Fin Q → ↥D =>
          acc ω ∧ ∀ i, chk ω (qs i))).card : ℚ) := by
    simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
    rw [Fintype.sum_prod_type]
  have hpt : ∀ ω : Ω, ((Finset.univ.filter (fun qs : Fin Q → ↥D =>
          acc ω ∧ ∀ i, chk ω (qs i))).card : ℚ)
      ≤ (if bad ω then (D.card : ℚ) ^ Q else 0) + (β * D.card) ^ Q := by
    intro ω
    have hnn : (0 : ℚ) ≤ (β * D.card) ^ Q :=
      pow_nonneg (mul_nonneg hβ0 (Nat.cast_nonneg _)) Q
    by_cases hb : bad ω
    · rw [ite_eq_left hb]
      have hle : ((Finset.univ.filter (fun qs : Fin Q → ↥D =>
          acc ω ∧ ∀ i, chk ω (qs i))).card : ℚ) ≤ (D.card : ℚ) ^ Q := by
        have h1 : (Finset.univ.filter (fun qs : Fin Q → ↥D =>
            acc ω ∧ ∀ i, chk ω (qs i))).card ≤ D.card ^ Q := by
          calc _ ≤ (Finset.univ : Finset (Fin Q → ↥D)).card := Finset.card_filter_le _ _
            _ = D.card ^ Q := by simp [Finset.card_univ]
        exact_mod_cast h1
      linarith
    · rw [ite_eq_right hb]
      by_cases ha : acc ω
      · have hfilt : (Finset.univ.filter (fun qs : Fin Q → ↥D =>
            acc ω ∧ ∀ i, chk ω (qs i)))
            = Finset.univ.filter (fun qs : Fin Q → ↥D => ∀ i, chk ω (qs i)) := by
          ext qs
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, ha, true_and]
        rw [hfilt, aux_thmB_card_all_pass Q (fun q : ↥D => chk ω q.1),
          aux_thmB_card_subtype_filter D (chk ω)]
        have hh := h ω ha hb
        have h0 : (0 : ℚ) ≤ ((D.filter (chk ω)).card : ℚ) := Nat.cast_nonneg _
        rw [zero_add, Nat.cast_pow]
        exact pow_le_pow_left₀ h0 hh Q
      · have hempty : (Finset.univ.filter (fun qs : Fin Q → ↥D =>
            acc ω ∧ ∀ i, chk ω (qs i))) = ∅ := by
          ext qs
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, ha, false_and,
            Finset.notMem_empty]
        rw [hempty, Finset.card_empty, Nat.cast_zero, zero_add]
        exact hnn
  rw [hfib]
  calc ∑ ω : Ω, ((Finset.univ.filter (fun qs : Fin Q → ↥D =>
          acc ω ∧ ∀ i, chk ω (qs i))).card : ℚ)
      ≤ ∑ ω : Ω, ((if bad ω then (D.card : ℚ) ^ Q else 0) + (β * D.card) ^ Q) :=
        Finset.sum_le_sum (fun ω _ => hpt ω)
    _ = ((Finset.univ.filter bad).card : ℚ) * (D.card : ℚ) ^ Q
        + (Fintype.card Ω : ℚ) * (β * D.card) ^ Q := by
      rw [Finset.sum_add_distrib, ← Finset.sum_filter, Finset.sum_const, Finset.sum_const,
        Finset.card_univ, nsmul_eq_mul, nsmul_eq_mul]

end Aux

open scoped Classical in
/-- **Theorem B′ (Phase B, unique decoding; interactive form).**  For every deterministic
adaptive prover strategy, if the initial state relation `R_0(π, 1, c₀, 1)` is false, then the
fraction of (challenge vector, query vector) pairs accepted by the verifier is at most
`(Σ_{t<m} |D_{t+1}|) / |K| + β^Q` with `β = (1 + 2^(-s))/2`.  Relative only to the correlated
agreement hypothesis `hCA`, which `corrAgree_of_arklib` discharges (see
`PhaseBInst.soundness_unconditional`). -/
theorem PhaseBInst.soundness [Fintype K] (I : PhaseBInst K) (hWF : I.WF) (β : ℚ)
    (hβ : β = (2 ^ I.s + 1) / 2 ^ (I.s + 1))
    (hCA : ∀ t, 1 ≤ t → t ≤ I.m → CorrAgree (I.D t) (2 ^ (I.m - t)) β)
    (hfalse : ¬ Rel (I.D 0) (2 ^ I.m) β I.ρ (fun _ => 1) I.π I.c0 1)
    (S : Strategy K I.m) (Q : ℕ) :
    ((Finset.univ.filter (fun p : (Fin I.m → K) × (Fin Q → ↥(I.D I.t1)) =>
        I.Accept0 S p.1 ∧ ∀ i, (I.final S p.1).chk (p.2 i))).card : ℚ) /
      ((Fintype.card K : ℚ) ^ I.m * ((I.D I.t1).card : ℚ) ^ Q)
    ≤ (∑ t ∈ Finset.range I.m, ((I.D (t + 1)).card : ℚ)) / Fintype.card K + β ^ Q := by
  have hβ0 : 0 < β := by rw [hβ]; positivity
  have hκ : (0 : ℚ) < (Fintype.card K : ℚ) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(0 : K)⟩
  have hn : (0 : ℚ) < ((I.D I.t1).card : ℚ) := by
    have hc := hWF.card I.t1 hWF.t1_le
    rw [hc]
    positivity
  -- Step 1: round-by-round soundness gives the bound on the exceptional challenge vectors.
  obtain ⟨ε, hεdef⟩ : ∃ ε : ℕ → ℚ, ε = fun t => ((I.D (t + 1)).card : ℚ) / Fintype.card K :=
    ⟨_, rfl⟩
  have hε : ∀ t, 0 ≤ ε t := by
    intro t
    rw [hεdef]
    positivity
  have h0 : ∀ pre : Fin 0 → K, ¬ I.Good β S 0 pre := by
    intro pre hg
    apply hfalse
    simpa [PhaseBInst.Good, PhaseBInst.state] using hg.2
  have hstep : ∀ t, t < I.m → ∀ pre : Fin t → K, ¬ I.Good β S t pre →
      ((Finset.univ.filter
          (fun x : K => I.Good β S (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card : ℚ)
        ≤ ε t * Fintype.card K := by
    intro t ht pre hnot
    have hgs := I.good_step hWF β hβ S hCA t ht pre hnot
    have hgs' : ((Finset.univ.filter
          (fun x : K => I.Good β S (t + 1) (Fin.snoc (α := fun _ => K) pre x))).card : ℚ)
        ≤ ((I.D (t + 1)).card : ℚ) := by exact_mod_cast hgs
    have heq : ε t * (Fintype.card K : ℚ) = ((I.D (t + 1)).card : ℚ) := by
      rw [hεdef]
      exact div_mul_cancel₀ _ hκ.ne'
    rw [heq]
    exact hgs'
  have hBad := rbr_prob I.m (fun t pre => I.Good β S t pre) ε hε h0 hstep I.m le_rfl
  have hsum : ∑ u ∈ Finset.range I.m, ε u
      = (∑ t ∈ Finset.range I.m, ((I.D (t + 1)).card : ℚ)) / Fintype.card K := by
    rw [hεdef]
    exact (Finset.sum_div _ _ _).symm
  rw [hsum] at hBad
  -- Step 2: for non-exceptional accepting challenge vectors, the query-chain pass fraction is `< β`.
  have hmass : ∀ ch : Fin I.m → K, I.Accept0 S ch → ¬ I.Good β S I.m ch →
      (((I.D I.t1).filter (fun q => (I.final S ch).chk q)).card : ℚ)
        ≤ β * ((I.D I.t1).card : ℚ) := by
    intro ch hacc hdoom
    have hlt := I.final_mass_lt hWF β hβ0 S ch hacc hdoom
    have heq := I.final_mass_eq_chk_fraction hWF (m := I.m) S ch
    rw [heq, div_lt_iff₀ hn] at hlt
    exact hlt.le
  have hAcc := Aux.aux_thmB_count_split (I.D I.t1) Q β hβ0.le
    (fun ch => I.Good β S I.m ch) (fun ch => I.Accept0 S ch)
    (fun ch q => (I.final S ch).chk q) hmass
  have hcardΩ : (Fintype.card (Fin I.m → K) : ℚ) = (Fintype.card K : ℚ) ^ I.m := by
    simp
  -- Step 3: assemble.
  rw [div_le_iff₀ (mul_pos (pow_pos hκ _) (pow_pos hn _))]
  refine hAcc.trans ?_
  rw [hcardΩ]
  have hnQ : (0 : ℚ) ≤ ((I.D I.t1).card : ℚ) ^ Q := pow_nonneg hn.le Q
  have h1 := mul_le_mul_of_nonneg_right hBad hnQ
  calc _ ≤ ((∑ t ∈ Finset.range I.m, ((I.D (t + 1)).card : ℚ)) / Fintype.card K
            * (Fintype.card K : ℚ) ^ I.m) * ((I.D I.t1).card : ℚ) ^ Q
          + (Fintype.card K : ℚ) ^ I.m * (β * ((I.D I.t1).card : ℚ)) ^ Q := by linarith
    _ = _ := by rw [mul_pow]; ring

open scoped Classical in
/-- The same with the correlated-agreement hypothesis discharged by ArkLib's theorem.

Model: the `m` challenges and the `Q` queries are uniform and independent (the real system derives
them by hashing, see `Bits.lean` for the bias factor `r1`), the committed oracles are ideal (no Merkle
tree), there is no proof of work and no Fiat–Shamir transformation, and `Q` is arbitrary. -/
theorem PhaseBInst.soundness_unconditional {K : Type} [Field K] [Fintype K] [DecidableEq K]
    (I : PhaseBInst K) (hWF : I.WF) (β : ℚ) (hβ : β = (2 ^ I.s + 1) / 2 ^ (I.s + 1))
    (hfalse : ¬ Rel (I.D 0) (2 ^ I.m) β I.ρ (fun _ => 1) I.π I.c0 1)
    (S : Strategy K I.m) (Q : ℕ) :
    ((Finset.univ.filter (fun p : (Fin I.m → K) × (Fin Q → ↥(I.D I.t1)) =>
        I.Accept0 S p.1 ∧ ∀ i, (I.final S p.1).chk (p.2 i))).card : ℚ) /
      ((Fintype.card K : ℚ) ^ I.m * ((I.D I.t1).card : ℚ) ^ Q)
    ≤ (∑ t ∈ Finset.range I.m, ((I.D (t + 1)).card : ℚ)) / Fintype.card K + β ^ Q := by
  have hCA : ∀ t, 1 ≤ t → t ≤ I.m → CorrAgree (I.D t) (2 ^ (I.m - t)) β := by
    intro t _ htm
    have hcard := hWF.card t htm
    have hpow : I.m + I.s - t = (I.m - t) + I.s := by omega
    have h2s : (0 : ℚ) < 2 ^ I.s := by positivity
    refine corrAgree_of_arklib (I.D t) (2 ^ (I.m - t)) β (by positivity) ?_ ?_ ?_
    · -- `k' ≤ |D t|`
      rw [hcard]
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
    · -- `β ≤ 1`
      rw [hβ, div_le_one (by positivity), pow_succ]
      have h1 : (1 : ℚ) ≤ 2 ^ I.s := one_le_pow₀ (by norm_num)
      linarith
    · -- `|D t| + k' ≤ 2 β |D t|` (in fact an equality)
      rw [hcard, hβ, hpow]
      push_cast
      have heq : 2 * (((2 : ℚ) ^ I.s + 1) / 2 ^ (I.s + 1)) * (2 ^ (I.m - t) * 2 ^ I.s)
          = 2 ^ (I.m - t) * 2 ^ I.s + 2 ^ (I.m - t) := by
        rw [pow_succ]
        field_simp
      rw [pow_add, heq]
  exact I.soundness hWF β hβ hCA hfalse S Q

#print axioms Cprs.PhaseBInst.soundness
#print axioms Cprs.PhaseBInst.soundness_unconditional

end Cprs
