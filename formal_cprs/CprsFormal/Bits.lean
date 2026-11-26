import Mathlib

/-!
# `CprsFormal/Bits.lean` — exact-rational certificate for the final security arithmetic of CPRS

## What is certified, and what is not

For the CPRS argument compiled with Fiat–Shamir/BCS, the soundness accounting uses the expression

    eps_FS(T) ≤ (T + R) · ε_max + H*(T)                                                              (★)

    R       = 2 (n-2) + 2 + [L ≥ 2] = 2n - 2 + [L ≥ 2]                                  (Lemma R)
    ε_max   = max { (4L-1) r1/|K| , 2 r1/|K| , ε_mu , ε_fold , 2^-b β^q }
    ε_mu    = r1/|K|  (Model H; the mu round exists only if L ≥ 2)
    ε_fold  = r1 N/|K|  (Theorem B)      or      r1 (N/2)/|K|  (Theorem B', n_1 = N/2)
    r1      = (1 + (p-4)/2^64)^4 ,  |K| = p^4 ,  p = 2^31 - 1 ,  N = 2^(n-2+s) ,  β = (1 + 2^-s)/2     (Lemma S)
    H*(T)   = max { 3 (T^2+1) , 3.5 T^2 - 1.5 T + 2 (D+1) Σ_l + 4 l_0^2 } / 2^256                     (Model H)

This file transcribes (★) verbatim in `ℚ` (no floating point anywhere) and machine-checks the *arithmetic* of the 100-bit
statement in the convention `WORK_RATIO_T_GE_R`: `min over T ≥ R of log2 (T / min (1, g T)) ≥ 100`,
`g` the right-hand side of (★).  It does **not** prove that (★) is a valid bound on the success probability of a
`T`-query adversary: that rests on the random-oracle model, the fold theorem, honest registration and the other files
of this development.
The file imports only `Mathlib`, contains no placeholder proofs and no new postulates (the `#print axioms` block at the end lists
only `propext`, `Classical.choice`, `Quot.sound`), and relies on no compiled-code evaluation of propositions: the `decide +kernel`
spot checks of section 10 are evaluated by the kernel.  Checked with Lean 4.34.0 / Mathlib `5ed2965` (the build of this development).

## Method (all `T` at once, no enumeration)

For `t0 ≤ T ≤ 2^100` (rational `T`):
* `(T + R) ε_max ≤ T (1 + R/t0) ε_max`, since `T/t0 ≥ 1` (`fsBound_le`); for `t0 = R` this is the bound `(T+R) ε_max ≤ 2 T ε_max`;
* `H*(T) ≤ T · hslope(t0)`, `hslope(t0) = (3.5·2^100 + (C+3)/t0)/2^256`, `C = 2(D+1)Σ_l + 4 l_0^2`
  (`hashStar_le`: both branches of the `max` are `≤ T (3.5·2^100 + (C+3)/t0)` for `t0 ≤ T ≤ 2^100`).
Hence `bound T ≤ T (2 ε_max + hslope R) ≤ T/2^100` as soon as the *closed rational fact* `2 ε_max + hslope R ≤ 2^-100` holds.
That fact is checked by `norm_num` on exact rationals (numerators of ≈ 500 bits): each of the five round errors is compared with
`E0 = (2^-100 - 2^-154)/2`, and `E0 ≤ (2^-100 - hslope R)/2` because `hslope R ≤ 2^-154`.  Minimality is the opposite
inequality at the single point `T = R`: `bound R ≥ (R + R) ε_Q` and `(R+R) ε_Q(243, 0) > R/2^100`.

## Inputs

Profiles have `s = 1`, `f = 3`.  `R, D, Σ_l, l_0` are *derived* below from `(n, L, g, s, f)` by the layout definitions
and pinned by `decide`; section 10 evaluates the bound at the decisive points by the kernel.

## Index of theorems (`i ∈ {1,2,3}`, profiles P1, P2, P3)

* (1) `Pi_q244_b0`, `Pi_q196_b20`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100` (Model H, Theorem B);
  `Pi_q244_b0_rat`, `Pi_q196_b20_rat`: the same for every rational `T`; `Pi_workRatio_q244_b0`, `Pi_workRatio_q196_b20`:
  the `min`-form `min (1, bound T) · 2^100 ≤ T` for all `T ≥ R` (`WorkRatio100`); `Pi_atBudget_*`: `bound (2^100) ≤ 1`.
* (2) `Pi_q243_b0_fails_at_R`, `Pi_q195_b20_fails_at_R` (and `Cert.fail_b0/fail_b20` for all smaller `q`);
  `Pi_workRatio_iff_b0`: `WorkRatio100 Pi .thmB q 0 ↔ 244 ≤ q`; `Pi_workRatio_iff_b20`: `... q 20 ↔ 196 ≤ q`.
* (3) `Pi_Bp_*`: the same four statements with the Theorem-B' fold term (the thresholds do not move, the query round binds);
  `P4_Bp_fails_at_R`, `P4_Bp_not_workRatio`: **P4 (`n = 25`) meets the convention for no `(q, b)` under B'**;
  `P4_Bp_shortfall`: the failure is by a factor `< 1 + 2^-28` (3.36·10^-9 bits); `P4_Bp_atBudget_q241`, `P4_Bp_atBudget_q193_b20`:
  P4 under B' meets `AT_BUDGET`; `P4_Bp_q244_b0_from_R_plus_one`, `P4_Bp_q196_b20_from_R_plus_one`: it meets the convention on
  `R + 1 ≤ T ≤ 2^100`, i.e. everywhere except the single point `T = R`; `P4_B_fails_at_R`: under Theorem B it fails for all `(q, b)`.
* Extension: `P5_*` (`n = 24`): meets the convention under B' iff `q ≥ 244` (`b = 0`) / `q ≥ 196` (`b = 20`),
  and fails under B for every `(q, b)` (`P5_B_fails_at_R`): B' moves the largest admissible `n` from 23 to 24.

## Results (bits = -log2, from exact rationals, logarithms to 60 digits)

    profile fold   fold-term bits   2*fold bits    min q T>=R (b=0/20)  min q AT_BUDGET   min q T>=R+1
    P1 n=22 B      102.999999997    101.999999997  244 / 196            241 / 193         244 / 196
    P2 n=23 B      101.999999997    100.999999997  244 / 196            241 / 193         244 / 196
    P3 n=17 B      107.999999997    106.999999997  244 / 196            241 / 193         244 / 196
    P4 n=25 B       99.999999997     98.999999997  none                 none              none
    P4 n=25 B'     100.999999997     99.999999997  none                 241 / 193         244 / 196
    P5 n=24 B      100.999999997     99.999999997  none                 241 / 193         244 / 196
    P5 n=24 B'     101.999999997    100.999999997  244 / 196            241 / 193         244 / 196

(`T>=R` figure at `q = 244, b = 0`: 100.269149824 bits for P1–P3 = 101.269149824 - 1, the bit lost to the factor `T + R = 2R`
at `T = R`; at `q = 243`: 99.854112325; at `q = 196, b = 20`: 100.347349859; at `q = 195, b = 20`: 99.932312359.)

## Caveats

* Model H only (honest registration).  In Model A (`ε_mu = r1^(L-1) N/|K|`, `L ≥ 2`; no figure is claimed there)
  the mu round is not improved by Theorem B', so the P4/P5 statements about B' are Model-H statements.
* All `Bp` results are conditional on Theorem B'.  Theorem B (`r1 N/|K|`) gives the primary figures.
* `Profile.epsMax` takes all five kinds of rounds into account; the `A` and fold rounds exist because `m = n - 2 ≥ 1` for every
  profile used (not enforced by the type).
-/

namespace Cprs.Bits

/-! ## 1. Field constants and sampling bias -/

/-- `p = 2^31 - 1`, the M31 prime. -/
def pM31 : ℚ := 2 ^ 31 - 1

/-- `|K| = p^4`, `K = QM31 = F_{p^4}`. -/
def Ksz : ℚ := pM31 ^ 4

/-- `r_limb = 1 + (p - 4)/2^64`: the largest point mass of a 64-bit word reduced mod `p`, relative to `1/p`
(Lemma S; `2^64 ≡ 4 (mod p)`). -/
def rLimb : ℚ := 1 + (pM31 - 4) / 2 ^ 64

/-- `r1 = r_limb^4`: the largest point mass of one QM31 challenge relative to `1/|K|`. -/
def r1 : ℚ := rLimb ^ 4

/-- `β = (1 + 2^-s)/2 = 1 - θ`, `θ = (1 - 2^-s)/2` the unique-decoding radius at rate `2^-s`. -/
def betaUD (s : ℕ) : ℚ := (1 + (1 / 2) ^ s) / 2

/-! ## 2. A profile and its derived layout -/

/-- The parameters of a CPRS instance: matrix size `2^n`, number `L` of matrices, `g` matrices per registered Merkle
tree, rate exponent `s` (rate `2^-s`), fold-arity exponent `f`. -/
structure Profile where
  n : ℕ
  L : ℕ
  g : ℕ
  s : ℕ
  f : ℕ

/-- `m = n - 2` binary rounds (conjugate packing absorbs two variables). -/
def Profile.m (P : Profile) : ℕ := P.n - 2

/-- `log2 N = m + s`, `N = 2^(m+s)` the codeword length. -/
def Profile.logN (P : Profile) : ℕ := P.m + P.s

/-- Lemma R: `R = 1 (gamma) + m (A) + [L ≥ 2] (mu) + m (alpha) + 1 (query) = 2m + 2 + [L ≥ 2] = 2n - 2 + [L ≥ 2]`. -/
def Profile.R (P : Profile) : ℕ := 2 * P.m + 2 + (if 2 ≤ P.L then 1 else 0)

/-- Number of fold blocks `B = ⌈m / f⌉` (block sizes `f, f, …, f, m - (B-1) f`). -/
def Profile.B (P : Profile) : ℕ := (P.m + P.f - 1) / P.f

/-- Merkle depth of the oracle of block `r`: `(m + s - r f) - f_r` with `f_r = min (f, m - r f)`. -/
def Profile.depth (P : Profile) (r : ℕ) : ℕ := (P.logN - r * P.f) - min P.f (P.m - r * P.f)

/-- `G = L / g` registered Merkle trees. -/
def Profile.G (P : Profile) : ℕ := P.L / P.g

/-- The depths of the `B` blocks. -/
def Profile.depths (P : Profile) : List ℕ := (List.range P.B).map P.depth

/-- `D`: the largest Merkle depth. -/
def Profile.D (P : Profile) : ℕ := P.depths.foldr max 0

/-- Leaf counts of all Merkle trees, each tree counted separately: the `G` registered trees (`2^depth_0` leaves each), the
`B - 1` inner block-end oracles, and `G` one-leaf configurations for the commitment layer `C_j`. -/
def Profile.leafCounts (P : Profile) : List ℕ :=
  List.replicate P.G (2 ^ P.depth 0) ++ (P.depths.tail.map (fun d => 2 ^ d)) ++ List.replicate P.G 1

/-- `Σ_l`: the sum of the leaf counts of all Merkle trees of the instance. -/
def Profile.sumL (P : Profile) : ℕ := P.leafCounts.sum

/-- `l_0 = G 2^(depth_0)`: the number of registered leaves. -/
def Profile.l0 (P : Profile) : ℕ := P.G * 2 ^ P.depth 0

/-! ## 3. Round errors, `ε_max`, the hash term `H*` and the bound (★) (Model H) -/

/-- Which per-round field term the binary fold rounds carry: Theorem B (`r1 N/|K|`) or the strengthening Theorem B'
(`r1 (N/2)/|K|`: the round `t` carries `n_(t+1)/|K|`, `n_(t+1) = N/2^(t+1)`, and
`n_1 = N/2` is the largest per-round value). -/
inductive FoldTerm
  | thmB
  | thmBp

/-- `ε_gamma = (4L - 1) r1/|K|`. -/
def Profile.epsGamma (P : Profile) : ℚ := (4 * (P.L : ℚ) - 1) * r1 / Ksz

/-- `ε_A = 2 r1/|K|` (the `m` Phase-A rounds). -/
def epsA : ℚ := 2 * r1 / Ksz

/-- `ε_mu` in Model H: `r1/|K|`; the mu round exists only if `L ≥ 2` (else the term is absent, here `0`). -/
def Profile.epsMu (P : Profile) : ℚ := if 2 ≤ P.L then r1 / Ksz else 0

/-- Theorem B fold term `ε_B = r1 N/|K|`. -/
def Profile.epsB (P : Profile) : ℚ := r1 * 2 ^ P.logN / Ksz

/-- Theorem B' fold term `r1 (N/2)/|K|`. -/
def Profile.epsBp (P : Profile) : ℚ := r1 * (2 ^ P.logN / 2) / Ksz

/-- The fold term selected by `ft`. -/
def Profile.epsFold (P : Profile) : FoldTerm → ℚ
  | .thmB => P.epsB
  | .thmBp => P.epsBp

/-- `ε_Q = 2^-b β^q` (query round with `b` bits of proof of work). -/
def Profile.epsQ (P : Profile) (q b : ℕ) : ℚ := (1 / 2) ^ b * betaUD P.s ^ q

/-- `ε_max = max {ε_gamma, ε_A, ε_mu, ε_fold, ε_Q}` (all five kinds exist because `m ≥ 1` for the profiles used). -/
def Profile.epsMax (P : Profile) (ft : FoldTerm) (q b : ℕ) : ℚ :=
  max (max (max (max P.epsGamma epsA) P.epsMu) (P.epsFold ft)) (P.epsQ q b)

/-- `C = 2 (D + 1) Σ_l + 4 l_0^2` (the `T`-independent part of the hash term, Model H). -/
def Profile.hashC (P : Profile) : ℚ := 2 * ((P.D : ℚ) + 1) * (P.sumL : ℚ) + 4 * (P.l0 : ℚ) ^ 2

/-- `H*(T) = max {3 (T^2 + 1), 3.5 T^2 - 1.5 T + 2 (D + 1) Σ_l + 4 l_0^2} / 2^256` (Model H). -/
def Profile.hashStar (P : Profile) (T : ℚ) : ℚ :=
  max (3 * (T ^ 2 + 1)) (7 / 2 * T ^ 2 - 3 / 2 * T + P.hashC) / 2 ^ 256

/-- The Fiat–Shamir bound (★): `eps_FS(T) ≤ (T + R) ε_max + H*(T)`. -/
def Profile.fsBound (P : Profile) (ft : FoldTerm) (q b : ℕ) (T : ℚ) : ℚ :=
  (T + P.R) * P.epsMax ft q b + P.hashStar T

/-- The convention `WORK_RATIO_T_GE_R` at σ = 100: `min over T ≥ R of log2 (T / min (1, bound T)) ≥ 100`,
i.e. `min (1, bound T) · 2^100 ≤ T` for every natural `T ≥ R` (no logarithm, no floating point). -/
def WorkRatio100 (P : Profile) (ft : FoldTerm) (q b : ℕ) : Prop :=
  ∀ T : ℕ, P.R ≤ T → min 1 (P.fsBound ft q b T) * 2 ^ 100 ≤ (T : ℚ)

/-- The linear-in-`T` slope of the hash term on `[t0, 2^100]`: `H*(T) ≤ T · hslope t0` (see `hashStar_le`). -/
def Profile.hslope (P : Profile) (t0 : ℚ) : ℚ := (7 / 2 * 2 ^ 100 + (P.hashC + 3) / t0) / 2 ^ 256

/-- `target = (2^-100 - hslope R)/2`: the largest `ε_max` for which the proof below closes. -/
def Profile.target (P : Profile) : ℚ := (1 / 2 ^ 100 - P.hslope P.R) / 2

/-- A uniform target `E0 = (2^-100 - 2^-154)/2 ≈ 2^-101 (1 - 2^-54)` for every round error (see `E0_le_target`). -/
def E0 : ℚ := (1 / 2 ^ 100 - 1 / 2 ^ 154) / 2

/-! ## 4. Generic lemmas (valid for every profile) -/

theorem betaUD_nonneg (s : ℕ) : 0 ≤ betaUD s := by
  unfold betaUD; positivity

theorem betaUD_le_one (s : ℕ) : betaUD s ≤ 1 := by
  unfold betaUD
  have : ((1 : ℚ) / 2) ^ s ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  linarith

theorem r1_pos : 0 < r1 := by norm_num [r1, rLimb, pM31]
theorem Ksz_pos : 0 < Ksz := by norm_num [Ksz, pM31]

theorem epsQ_nonneg (P : Profile) (q b : ℕ) : 0 ≤ P.epsQ q b := by
  unfold Profile.epsQ
  exact mul_nonneg (by positivity) (pow_nonneg (betaUD_nonneg _) _)

theorem epsQ_anti (P : Profile) (b : ℕ) {q q' : ℕ} (h : q ≤ q') : P.epsQ q' b ≤ P.epsQ q b := by
  unfold Profile.epsQ
  exact mul_le_mul_of_nonneg_left (pow_le_pow_of_le_one (betaUD_nonneg _) (betaUD_le_one _) h)
    (by positivity)

theorem epsQ_le_epsMax (P : Profile) (ft : FoldTerm) (q b : ℕ) : P.epsQ q b ≤ P.epsMax ft q b :=
  le_max_right _ _

theorem epsFold_le_epsMax (P : Profile) (ft : FoldTerm) (q b : ℕ) :
    P.epsFold ft ≤ P.epsMax ft q b :=
  le_trans (le_max_right _ _) (le_max_left _ _)

theorem epsMax_nonneg (P : Profile) (ft : FoldTerm) (q b : ℕ) : 0 ≤ P.epsMax ft q b :=
  le_trans (epsQ_nonneg P q b) (epsQ_le_epsMax P ft q b)

theorem epsMax_le_iff (P : Profile) (ft : FoldTerm) (q b : ℕ) (E : ℚ) :
    P.epsMax ft q b ≤ E ↔
      (P.epsGamma ≤ E ∧ epsA ≤ E ∧ P.epsMu ≤ E ∧ P.epsFold ft ≤ E ∧ P.epsQ q b ≤ E) := by
  unfold Profile.epsMax
  simp only [max_le_iff]
  tauto

/-- More queries never increase `ε_max`. -/
theorem epsMax_anti_q (P : Profile) (ft : FoldTerm) (b : ℕ) {q q' : ℕ} (h : q ≤ q') :
    P.epsMax ft q' b ≤ P.epsMax ft q b := by
  unfold Profile.epsMax
  exact max_le_max le_rfl (epsQ_anti P b h)

theorem epsBp_le_epsB (P : Profile) : P.epsBp ≤ P.epsB := by
  unfold Profile.epsBp Profile.epsB
  have h : 0 ≤ r1 * 2 ^ P.logN / Ksz :=
    div_nonneg (mul_nonneg r1_pos.le (by positivity)) Ksz_pos.le
  have e : r1 * (2 ^ P.logN / 2) / Ksz = (r1 * 2 ^ P.logN / Ksz) / 2 := by ring
  rw [e]
  linarith

/-- Theorem B' only improves the fold term, hence `ε_max`. -/
theorem epsMax_Bp_le_B (P : Profile) (q b : ℕ) :
    P.epsMax .thmBp q b ≤ P.epsMax .thmB q b := by
  unfold Profile.epsMax
  exact max_le_max_right _ (max_le_max_left _ (epsBp_le_epsB P))

theorem hashStar_nonneg (P : Profile) (T : ℚ) : 0 ≤ P.hashStar T := by
  unfold Profile.hashStar
  apply div_nonneg _ (by positivity)
  exact le_trans (by positivity) (le_max_left _ _)

theorem hashC_nonneg (P : Profile) : 0 ≤ P.hashC := by
  unfold Profile.hashC; positivity

/-- The hash term is at most linear in `T` on `[t0, 2^100]`: `H*(T) ≤ T · hslope t0`.  (`3 (T²+1) ≤ T (3.5·2^100 + (C+3)/t0)` and
`3.5 T² - 1.5 T + C ≤ T (3.5·2^100 + (C+3)/t0)` for `t0 ≤ T ≤ 2^100`.) -/
theorem hashStar_le (P : Profile) (t0 T : ℚ) (ht0 : 0 < t0) (hT0 : t0 ≤ T) (hT : T ≤ 2 ^ 100) :
    P.hashStar T ≤ T * P.hslope t0 := by
  have hTpos : 0 < T := lt_of_lt_of_le ht0 hT0
  have hC := hashC_nonneg P
  have hu0 : 0 ≤ (P.hashC + 3) / t0 := div_nonneg (by linarith) ht0.le
  have htu : P.hashC + 3 ≤ T * ((P.hashC + 3) / t0) := by
    have h1 : t0 * ((P.hashC + 3) / t0) = P.hashC + 3 := by field_simp
    nlinarith
  have h1 : T * T ≤ T * 2 ^ 100 := mul_le_mul_of_nonneg_left hT hTpos.le
  have h2 : (0 : ℚ) ≤ T * 2 ^ 100 := by positivity
  unfold Profile.hashStar Profile.hslope
  rw [← mul_div_assoc]
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply max_le
  · nlinarith
  · nlinarith

/-- Core reduction: if `ε_max ≤ E` and `(1 + R/t0) E + hslope t0 ≤ 2^-100`, then `bound T ≤ T/2^100` for all
(rational) `T` with `t0 ≤ T ≤ 2^100`.  Uses only `(T + R) ε_max ≤ T (1 + R/t0) ε_max` (as `T/t0 ≥ 1`) and `hashStar_le`. -/
theorem fsBound_le (P : Profile) (ft : FoldTerm) (q b : ℕ) (t0 E : ℚ) (ht0 : 0 < t0)
    (hE : P.epsMax ft q b ≤ E)
    (hnum : (1 + (P.R : ℚ) / t0) * E + P.hslope t0 ≤ 1 / 2 ^ 100) :
    ∀ T : ℚ, t0 ≤ T → T ≤ 2 ^ 100 → P.fsBound ft q b T ≤ T / 2 ^ 100 := by
  intro T hT0 hT
  have hTpos : 0 < T := lt_of_lt_of_le ht0 hT0
  have hε0 := epsMax_nonneg P ft q b
  have hR0 : (0 : ℚ) ≤ P.R := Nat.cast_nonneg _
  have hratio : 1 ≤ T / t0 := by rw [le_div_iff₀ ht0]; linarith
  have h1 : (T + P.R) * P.epsMax ft q b ≤ T * ((1 + P.R / t0) * P.epsMax ft q b) := by
    have key : T * ((1 + P.R / t0) * P.epsMax ft q b) - (T + P.R) * P.epsMax ft q b
        = (T / t0 - 1) * (P.R * P.epsMax ft q b) := by
      field_simp
      ring
    have : 0 ≤ (T / t0 - 1) * (P.R * P.epsMax ft q b) :=
      mul_nonneg (by linarith) (mul_nonneg hR0 hε0)
    linarith
  have h2 : T * ((1 + P.R / t0) * P.epsMax ft q b) ≤ T * ((1 + P.R / t0) * E) := by
    apply mul_le_mul_of_nonneg_left _ hTpos.le
    apply mul_le_mul_of_nonneg_left hE
    positivity
  have h3 := hashStar_le P t0 T ht0 hT0 hT
  have h4 : T * ((1 + P.R / t0) * E) + T * P.hslope t0 ≤ T * (1 / 2 ^ 100) := by
    rw [← mul_add]
    exact mul_le_mul_of_nonneg_left hnum hTpos.le
  unfold Profile.fsBound
  calc (T + P.R) * P.epsMax ft q b + P.hashStar T
      ≤ T * ((1 + P.R / t0) * E) + T * P.hslope t0 := by linarith
    _ ≤ T * (1 / 2 ^ 100) := h4
    _ = T / 2 ^ 100 := by ring

/-- The `t0 = R` case: `ε_max ≤ target` suffices (`(1 + R/R) target + hslope R = 2^-100`). -/
theorem fsBound_le_target (P : Profile) (ft : FoldTerm) (q b : ℕ) (hR : 0 < P.R)
    (hE : P.epsMax ft q b ≤ P.target) :
    ∀ T : ℚ, (P.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P.fsBound ft q b T ≤ T / 2 ^ 100 := by
  apply fsBound_le P ft q b (P.R) P.target (by exact_mod_cast hR) hE
  have hRne : (P.R : ℚ) ≠ 0 := by exact_mod_cast hR.ne'
  rw [div_self hRne]
  unfold Profile.target
  linarith

/-- Natural-number form of `fsBound_le_target`. -/
theorem workRatio_nat (P : Profile) (ft : FoldTerm) (q b : ℕ) (hR : 0 < P.R)
    (hE : P.epsMax ft q b ≤ P.target) :
    ∀ T : ℕ, P.R ≤ T → T ≤ 2 ^ 100 → P.fsBound ft q b T ≤ (T : ℚ) / 2 ^ 100 := by
  intro T h1 h2
  apply fsBound_le_target P ft q b hR hE
  · exact_mod_cast h1
  · exact_mod_cast h2

/-- `bound T ≤ T/2^100` on `[R, 2^100]` gives the `min`-form of the convention for all `T ≥ R` (for `T ≥ 2^100` it is trivial). -/
theorem workRatio100_of_le (P : Profile) (ft : FoldTerm) (q b : ℕ)
    (h : ∀ T : ℕ, P.R ≤ T → T ≤ 2 ^ 100 → P.fsBound ft q b T ≤ (T : ℚ) / 2 ^ 100) :
    WorkRatio100 P ft q b := by
  intro T hT
  by_cases hle : T ≤ 2 ^ 100
  · have h1 := h T hT hle
    calc min 1 (P.fsBound ft q b T) * 2 ^ 100 ≤ P.fsBound ft q b T * 2 ^ 100 :=
          mul_le_mul_of_nonneg_right (min_le_right _ _) (by positivity)
      _ ≤ (T : ℚ) / 2 ^ 100 * 2 ^ 100 := mul_le_mul_of_nonneg_right h1 (by positivity)
      _ = T := by field_simp
  · have hlt : (2 : ℚ) ^ 100 ≤ T := by
      have := (not_le.mp hle).le
      exact_mod_cast this
    calc min 1 (P.fsBound ft q b T) * 2 ^ 100 ≤ 1 * 2 ^ 100 :=
          mul_le_mul_of_nonneg_right (min_le_left _ _) (by positivity)
      _ ≤ T := by linarith

/-- The point `T = 2^100` of the statement is the `AT_BUDGET` convention: `bound (2^100) ≤ 1`. -/
theorem atBudget_of_nat (P : Profile) (ft : FoldTerm) (q b : ℕ) (hR : P.R ≤ 2 ^ 100)
    (h : ∀ T : ℕ, P.R ≤ T → T ≤ 2 ^ 100 → P.fsBound ft q b T ≤ (T : ℚ) / 2 ^ 100) :
    P.fsBound ft q b (2 ^ 100) ≤ 1 := by
  have h1 := h (2 ^ 100) hR le_rfl
  have e : (((2 ^ 100 : ℕ) : ℚ)) = 2 ^ 100 := by push_cast; rfl
  rw [e] at h1
  rwa [div_self (by positivity)] at h1

/-- `AT_BUDGET` directly: `ε_max ≤ E` and `(2^100 + R) E + 2^100 hslope(2^100) ≤ 1` give `bound (2^100) ≤ 1`. -/
theorem fsBound_atBudget (P : Profile) (ft : FoldTerm) (q b : ℕ) (E : ℚ)
    (hE : P.epsMax ft q b ≤ E)
    (hnum : (2 ^ 100 + (P.R : ℚ)) * E + 2 ^ 100 * P.hslope (2 ^ 100) ≤ 1) :
    P.fsBound ft q b (2 ^ 100) ≤ 1 := by
  unfold Profile.fsBound
  have h1 : (2 ^ 100 + (P.R : ℚ)) * P.epsMax ft q b ≤ (2 ^ 100 + (P.R : ℚ)) * E :=
    mul_le_mul_of_nonneg_left hE (by positivity)
  have h2 := hashStar_le P (2 ^ 100) (2 ^ 100) (by positivity) le_rfl le_rfl
  linarith

/-- At `T = R` the bound is at least `(R + R) e` for every `e ≤ ε_max`; so `(R + R) e > R/2^100` refutes the convention there. -/
theorem not_le_at_R (P : Profile) (ft : FoldTerm) (q b : ℕ) (e : ℚ) (he : e ≤ P.epsMax ft q b)
    (hlow : (P.R : ℚ) / 2 ^ 100 < ((P.R : ℚ) + P.R) * e) :
    ¬ (P.fsBound ft q b (P.R : ℚ) ≤ (P.R : ℚ) / 2 ^ 100) := by
  intro hle
  have h0 := hashStar_nonneg P (P.R : ℚ)
  have h : ((P.R : ℚ) + P.R) * e ≤ ((P.R : ℚ) + P.R) * P.epsMax ft q b :=
    mul_le_mul_of_nonneg_left he (by positivity)
  have h1 : ((P.R : ℚ) + P.R) * e ≤ P.fsBound ft q b (P.R : ℚ) := by
    unfold Profile.fsBound
    linarith
  linarith

/-- Failure at `T = R` for every `q ≤ q₀` from a lower bound on the query term only (`ε_Q` is decreasing in `q`). -/
theorem not_le_at_R_of_query (P : Profile) (ft : FoldTerm) (b q₀ : ℕ)
    (hlow : (P.R : ℚ) / 2 ^ 100 < ((P.R : ℚ) + P.R) * P.epsQ q₀ b) :
    ∀ q, q ≤ q₀ → ¬ (P.fsBound ft q b (P.R : ℚ) ≤ (P.R : ℚ) / 2 ^ 100) := by
  intro q hq
  exact not_le_at_R P ft q b (P.epsQ q₀ b) (le_trans (epsQ_anti P b hq) (epsQ_le_epsMax P ft q b))
    hlow

/-- If the bound fails at `T = R` (and `R < 2^100`) then the `min`-form convention fails. -/
theorem not_workRatio100_of_fail_at_R (P : Profile) (ft : FoldTerm) (q b : ℕ) (hRlt : P.R < 2 ^ 100)
    (hfail : ¬ (P.fsBound ft q b (P.R : ℚ) ≤ (P.R : ℚ) / 2 ^ 100)) : ¬ WorkRatio100 P ft q b := by
  intro h
  have hR1 := h P.R le_rfl
  apply hfail
  by_contra hgt
  have hgt' := not_le.mp hgt
  have hlt1 : (P.R : ℚ) / 2 ^ 100 < 1 := by
    rw [div_lt_one (by positivity)]
    exact_mod_cast hRlt
  have hmin : (P.R : ℚ) / 2 ^ 100 < min 1 (P.fsBound ft q b P.R) := lt_min hlt1 hgt'
  have h3 : (P.R : ℚ) < min 1 (P.fsBound ft q b P.R) * 2 ^ 100 := by
    rwa [div_lt_iff₀ (by positivity)] at hmin
  linarith

/-- Threshold characterisation: if `ε_max ≤ target` at `q₁ + 1` and the query term alone refutes `T = R` at `q₁`, then the
convention holds exactly for `q ≥ q₁ + 1`. -/
theorem workRatio100_iff (P : Profile) (ft : FoldTerm) (b q₁ : ℕ) (hR : 0 < P.R)
    (hRlt : P.R < 2 ^ 100) (hhi : P.epsMax ft (q₁ + 1) b ≤ P.target)
    (hlow : (P.R : ℚ) / 2 ^ 100 < ((P.R : ℚ) + P.R) * P.epsQ q₁ b) (q : ℕ) :
    WorkRatio100 P ft q b ↔ q₁ + 1 ≤ q := by
  constructor
  · intro h
    by_contra hq
    have hq' : q ≤ q₁ := by omega
    exact not_workRatio100_of_fail_at_R P ft q b hRlt (not_le_at_R_of_query P ft b q₁ hlow q hq') h
  · intro hq
    apply workRatio100_of_le
    exact workRatio_nat P ft q b hR (le_trans (epsMax_anti_q P ft b hq) hhi)

theorem hslope_le (P : Profile) (t0 : ℚ) (ht0 : 0 < t0) (hC : P.hashC + 3 ≤ 2 ^ 60 * t0) :
    P.hslope t0 ≤ 1 / 2 ^ 154 := by
  unfold Profile.hslope
  have h1 : (P.hashC + 3) / t0 ≤ 2 ^ 60 := by
    rw [div_le_iff₀ ht0]; linarith
  rw [div_le_iff₀ (by positivity)]
  have e : (1 : ℚ) / 2 ^ 154 * 2 ^ 256 = 2 ^ 102 := by norm_num
  rw [e]
  have h3 : (7 / 2 * 2 ^ 100 + 2 ^ 60 : ℚ) ≤ 2 ^ 102 := by norm_num
  linarith

theorem E0_le_target (P : Profile) (hR : 0 < P.R) (hC : P.hashC + 3 ≤ 2 ^ 60 * (P.R : ℚ)) :
    E0 ≤ P.target := by
  have hR' : (0 : ℚ) < P.R := by exact_mod_cast hR
  have h := hslope_le P P.R hR' hC
  unfold E0 Profile.target
  linarith

/-- The six facts certified for a profile and a fold term (Model H, `s = 1`, `f = 3`). -/
structure Cert (P : Profile) (ft : FoldTerm) : Prop where
  /-- `(q, b) = (244, 0)`: `bound T ≤ T/2^100` for every natural `T` with `R ≤ T ≤ 2^100`. -/
  at244 : ∀ T : ℕ, P.R ≤ T → T ≤ 2 ^ 100 → P.fsBound ft 244 0 T ≤ (T : ℚ) / 2 ^ 100
  /-- `(q, b) = (196, 20)`: the same. -/
  at196 : ∀ T : ℕ, P.R ≤ T → T ≤ 2 ^ 100 → P.fsBound ft 196 20 T ≤ (T : ℚ) / 2 ^ 100
  /-- the `min`-form convention (all `T ≥ R`) holds iff `q ≥ 244` (`b = 0`). -/
  iff0 : ∀ q : ℕ, WorkRatio100 P ft q 0 ↔ 244 ≤ q
  /-- the `min`-form convention (all `T ≥ R`) holds iff `q ≥ 196` (`b = 20`). -/
  iff20 : ∀ q : ℕ, WorkRatio100 P ft q 20 ↔ 196 ≤ q
  /-- minimality, `b = 0`: for every `q ≤ 243` the inequality fails at `T = R`. -/
  fail_b0 : ∀ q : ℕ, q ≤ 243 → ¬ (P.fsBound ft q 0 (P.R : ℚ) ≤ (P.R : ℚ) / 2 ^ 100)
  /-- minimality, `b = 20`: for every `q ≤ 195` the inequality fails at `T = R`. -/
  fail_b20 : ∀ q : ℕ, q ≤ 195 → ¬ (P.fsBound ft q 20 (P.R : ℚ) ≤ (P.R : ℚ) / 2 ^ 100)

/-- Assemble a `Cert` from closed rational facts. -/
theorem cert_of_numerics (P : Profile) (ft : FoldTerm) (hR : 0 < P.R) (hRlt : P.R < 2 ^ 100)
    (hT : E0 ≤ P.target)
    (h1 : P.epsGamma ≤ E0) (h2 : epsA ≤ E0) (h3 : P.epsMu ≤ E0) (h4 : P.epsFold ft ≤ E0)
    (hq244 : P.epsQ 244 0 ≤ E0) (hq196 : P.epsQ 196 20 ≤ E0)
    (hl243 : (P.R : ℚ) / 2 ^ 100 < ((P.R : ℚ) + P.R) * P.epsQ 243 0)
    (hl195 : (P.R : ℚ) / 2 ^ 100 < ((P.R : ℚ) + P.R) * P.epsQ 195 20) : Cert P ft where
  at244 := workRatio_nat P ft 244 0 hR
    (le_trans ((epsMax_le_iff P ft 244 0 E0).2 ⟨h1, h2, h3, h4, hq244⟩) hT)
  at196 := workRatio_nat P ft 196 20 hR
    (le_trans ((epsMax_le_iff P ft 196 20 E0).2 ⟨h1, h2, h3, h4, hq196⟩) hT)
  iff0 := workRatio100_iff P ft 0 243 hR hRlt
    (le_trans ((epsMax_le_iff P ft 244 0 E0).2 ⟨h1, h2, h3, h4, hq244⟩) hT) hl243
  iff20 := workRatio100_iff P ft 20 195 hR hRlt
    (le_trans ((epsMax_le_iff P ft 196 20 E0).2 ⟨h1, h2, h3, h4, hq196⟩) hT) hl195
  fail_b0 := not_le_at_R_of_query P ft 0 243 hl243
  fail_b20 := not_le_at_R_of_query P ft 20 195 hl195

/-- Unfold the definitions of the round errors and evaluate the resulting closed rational inequality. -/
local macro "bits_eps_norm" : tactic =>
  `(tactic| norm_num [Profile.epsGamma, epsA, Profile.epsMu, Profile.epsFold, Profile.epsB,
      Profile.epsBp, Profile.epsQ, betaUD, r1, rLimb, Ksz, pM31, E0])

/-- The two expressions of the number of rounds agree (for `n ≥ 2`): `2 (n-2) + 2 + [L ≥ 2] = 2 n - 2 + [L ≥ 2]`. -/
theorem Profile.R_eq (P : Profile) (h : 2 ≤ P.n) :
    P.R = 2 * P.n - 2 + (if 2 ≤ P.L then 1 else 0) := by
  unfold Profile.R Profile.m
  split_ifs <;> omega

/-- If the query round is the largest, `ε_max = ε_Q`. -/
theorem epsMax_eq_epsQ_of (P : Profile) (ft : FoldTerm) (q b : ℕ)
    (h1 : P.epsGamma ≤ P.epsQ q b) (h2 : epsA ≤ P.epsQ q b) (h3 : P.epsMu ≤ P.epsQ q b)
    (h4 : P.epsFold ft ≤ P.epsQ q b) : P.epsMax ft q b = P.epsQ q b :=
  le_antisymm ((epsMax_le_iff P ft q b _).2 ⟨h1, h2, h3, h4, le_rfl⟩) (epsQ_le_epsMax P ft q b)

/-- If the fold round is the largest, `ε_max = ε_fold`. -/
theorem epsMax_eq_fold_of (P : Profile) (ft : FoldTerm) (q b : ℕ)
    (h1 : P.epsGamma ≤ P.epsFold ft) (h2 : epsA ≤ P.epsFold ft) (h3 : P.epsMu ≤ P.epsFold ft)
    (h5 : P.epsQ q b ≤ P.epsFold ft) : P.epsMax ft q b = P.epsFold ft :=
  le_antisymm ((epsMax_le_iff P ft q b _).2 ⟨h1, h2, h3, le_rfl, h5⟩) (epsFold_le_epsMax P ft q b)

/-! ## 5. Pinning the constants -/

theorem pM31_eq : pM31 = 2147483647 := by norm_num [pM31]
theorem Ksz_eq : Ksz = 21267647892944572736998860269687930881 := by norm_num [Ksz, pM31]
theorem r1_eq : r1 = 115792089291236088641745882450254045641044892790509726805359207743822218920561 /
    115792089237316195423570985008687907853269984665640564039457584007913129639936 := by
  norm_num [r1, rLimb, pM31]
theorem one_lt_r1 : 1 < r1 := by norm_num [r1, rLimb, pM31]
theorem r1_lt : r1 < 1 + 1 / 2 ^ 31 := by norm_num [r1, rLimb, pM31]
theorem Ksz_lt_two_pow_124 : Ksz < 2 ^ 124 := by norm_num [Ksz, pM31]
theorem Ksz_lt_r1_mul : Ksz < r1 * 2 ^ 124 := by norm_num [Ksz, r1, rLimb, pM31]
theorem betaUD_one : betaUD 1 = 3 / 4 := by norm_num [betaUD]

/-! ## 6. The profiles `P1`–`P5` (inputs and derived layout) -/

/-- Profile P1 (`n = 22, L = 18, g = 3`), `s = 1, f = 3`, with `R = 43`, `Σ_l = 1610320`, `D = 18`, `l_0 = 1572864`.
These numbers are *derived* from `(n, L, g, s, f)` by the layout definitions above and checked by `decide`. -/
def P1 : Profile := ⟨22, 18, 3, 1, 3⟩
theorem P1_L : P1.L = 18 := rfl
theorem P1_s : P1.s = 1 := rfl
theorem P1_R : P1.R = 43 := by decide
theorem P1_logN : P1.logN = 21 := by decide
theorem P1_depths : P1.depths = [18, 15, 12, 9, 6, 3, 1] := by decide
theorem P1_D : P1.D = 18 := by decide
theorem P1_sumL : P1.sumL = 1610320 := by decide
theorem P1_l0 : P1.l0 = 1572864 := by decide
theorem P1_hashC : P1.hashC = 9895665842144 := by
  norm_num [Profile.hashC, P1_D, P1_sumL, P1_l0]
attribute [local simp] P1_L P1_s P1_R P1_logN

/-- Profile P2 (`n = 23, L = 3, g = 3`), `s = 1, f = 3`, with `R = 45`, `Σ_l = 599187`, `D = 19`, `l_0 = 524288`. -/
def P2 : Profile := ⟨23, 3, 3, 1, 3⟩
theorem P2_L : P2.L = 3 := rfl
theorem P2_s : P2.s = 1 := rfl
theorem P2_R : P2.R = 45 := by decide
theorem P2_logN : P2.logN = 22 := by decide
theorem P2_depths : P2.depths = [19, 16, 13, 10, 7, 4, 1] := by decide
theorem P2_D : P2.D = 19 := by decide
theorem P2_sumL : P2.sumL = 599187 := by decide
theorem P2_l0 : P2.l0 = 524288 := by decide
theorem P2_hashC : P2.hashC = 1099535595256 := by
  norm_num [Profile.hashC, P2_D, P2_sumL, P2_l0]
attribute [local simp] P2_L P2_s P2_R P2_logN

/-- Profile P3 (`n = 17, L = 1, g = 1`), `s = 1, f = 3`, with `R = 32`, `Σ_l = 9363`, `D = 13`, `l_0 = 8192`. -/
def P3 : Profile := ⟨17, 1, 1, 1, 3⟩
theorem P3_L : P3.L = 1 := rfl
theorem P3_s : P3.s = 1 := rfl
theorem P3_R : P3.R = 32 := by decide
theorem P3_logN : P3.logN = 16 := by decide
theorem P3_depths : P3.depths = [13, 10, 7, 4, 1] := by decide
theorem P3_D : P3.D = 13 := by decide
theorem P3_sumL : P3.sumL = 9363 := by decide
theorem P3_l0 : P3.l0 = 8192 := by decide
theorem P3_hashC : P3.hashC = 268697620 := by
  norm_num [Profile.hashC, P3_D, P3_sumL, P3_l0]
attribute [local simp] P3_L P3_s P3_R P3_logN

/-- Profile P4 (`n = 25, L = 18, g = 3`), `s = 1, f = 3`: the contrast profile in which the field term itself fails,
with `R = 49`, `Σ_l = 12882512`, `D = 21`, `l_0 = 12582912`. -/
def P4 : Profile := ⟨25, 18, 3, 1, 3⟩
theorem P4_L : P4.L = 18 := rfl
theorem P4_s : P4.s = 1 := rfl
theorem P4_R : P4.R = 49 := by decide
theorem P4_logN : P4.logN = 24 := by decide
theorem P4_depths : P4.depths = [21, 18, 15, 12, 9, 6, 3, 1] := by decide
theorem P4_D : P4.D = 21 := by decide
theorem P4_sumL : P4.sumL = 12882512 := by decide
theorem P4_l0 : P4.l0 = 12582912 := by decide
theorem P4_hashC : P4.hashC = 633319264429504 := by
  norm_num [Profile.hashC, P4_D, P4_sumL, P4_l0]
attribute [local simp] P4_L P4_s P4_R P4_logN

/-- Profile P5 (an extension; `n = 24, L = 18, g = 3`, `s = 1, f = 3`), with `R = 47`, `Σ_l = 6441260`, `D = 20`,
`l_0 = 6291456`. -/
def P5 : Profile := ⟨24, 18, 3, 1, 3⟩
theorem P5_L : P5.L = 18 := rfl
theorem P5_s : P5.s = 1 := rfl
theorem P5_R : P5.R = 47 := by decide
theorem P5_logN : P5.logN = 23 := by decide
theorem P5_depths : P5.depths = [20, 17, 14, 11, 8, 5, 2, 1] := by decide
theorem P5_D : P5.D = 20 := by decide
theorem P5_sumL : P5.sumL = 6441260 := by decide
theorem P5_l0 : P5.l0 = 6291456 := by decide
theorem P5_hashC : P5.hashC = 158329944932664 := by
  norm_num [Profile.hashC, P5_D, P5_sumL, P5_l0]
attribute [local simp] P5_L P5_s P5_R P5_logN

/-! ## 7. Theorems (1), (2) for P1, P2, P3 — Model H, fold term Theorem B (`r1 N/|K|`); and (3) with Theorem B' -/

theorem P1_target : E0 ≤ P1.target :=
  E0_le_target P1 (by simp) (by norm_num [P1_hashC])

/-- All six facts for `P1`, Theorem B. -/
theorem P1_cert : Cert P1 .thmB :=
  cert_of_numerics P1 .thmB (by simp) (by norm_num) P1_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- All six facts for `P1`, Theorem B' (the fold term only gets smaller). -/
theorem P1_Bp_cert : Cert P1 .thmBp :=
  cert_of_numerics P1 .thmBp (by simp) (by norm_num) P1_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- `P1`, Model H, `(q, b) = (244, 0)`: the query round is the largest, `ε_max = (3/4)^244`. -/
theorem P1_epsMax_q244_b0 : P1.epsMax .thmB 244 0 = (3 / 4) ^ 244 :=
  (epsMax_eq_epsQ_of P1 .thmB 244 0 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- `P1`, Model H, `(q, b) = (196, 20)`: `ε_max = 2^-20 (3/4)^196`. -/
theorem P1_epsMax_q196_b20 : P1.epsMax .thmB 196 20 = (1 / 2) ^ 20 * (3 / 4) ^ 196 :=
  (epsMax_eq_epsQ_of P1 .thmB 196 20 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- (1) `P1`, Model H, `(q, b) = (244, 0)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P1_q244_b0 :
    ∀ T : ℕ, P1.R ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmB 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P1_cert.at244

/-- (1) `P1`, Model H, `(q, b) = (196, 20)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P1_q196_b20 :
    ∀ T : ℕ, P1.R ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmB 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P1_cert.at196

/-- (1) the same for every rational (real-valued query budget) `T` with `R ≤ T ≤ 2^100`, `(244, 0)`. -/
theorem P1_q244_b0_rat :
    ∀ T : ℚ, (P1.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmB 244 0 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P1 .thmB 244 0 (by simp)
    (le_trans (by rw [P1_epsMax_q244_b0]; bits_eps_norm) P1_target)

/-- (1) the same for every rational `T` with `R ≤ T ≤ 2^100`, `(196, 20)`. -/
theorem P1_q196_b20_rat :
    ∀ T : ℚ, (P1.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmB 196 20 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P1 .thmB 196 20 (by simp)
    (le_trans (by rw [P1_epsMax_q196_b20]; bits_eps_norm) P1_target)

/-- (1) `min`-form of the convention: `min (1, bound T) · 2^100 ≤ T` for every natural `T ≥ R`, `(q, b) = (244, 0)`. -/
theorem P1_workRatio_q244_b0 : WorkRatio100 P1 .thmB 244 0 := (P1_cert.iff0 244).2 le_rfl

/-- (1) the same for `(q, b) = (196, 20)`. -/
theorem P1_workRatio_q196_b20 : WorkRatio100 P1 .thmB 196 20 := (P1_cert.iff20 196).2 le_rfl

/-- `AT_BUDGET` (the point `T = 2^100`), `(244, 0)`. -/
theorem P1_atBudget_q244_b0 : P1.fsBound .thmB 244 0 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P1 .thmB 244 0 (by norm_num) P1_q244_b0

/-- `AT_BUDGET`, `(196, 20)`. -/
theorem P1_atBudget_q196_b20 : P1.fsBound .thmB 196 20 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P1 .thmB 196 20 (by norm_num) P1_q196_b20

/-- (2) minimality, `(q, b) = (243, 0)`: the inequality fails at `T = R` (indeed for every `q ≤ 243`, `P1_cert.fail_b0`). -/
theorem P1_q243_b0_fails_at_R :
    ¬ (P1.fsBound .thmB 243 0 (P1.R : ℚ) ≤ (P1.R : ℚ) / 2 ^ 100) := P1_cert.fail_b0 243 le_rfl

/-- (2) minimality, `(q, b) = (195, 20)`. -/
theorem P1_q195_b20_fails_at_R :
    ¬ (P1.fsBound .thmB 195 20 (P1.R : ℚ) ≤ (P1.R : ℚ) / 2 ^ 100) := P1_cert.fail_b20 195 le_rfl

/-- (2) `q = 244` is the smallest query count meeting the convention with `b = 0`. -/
theorem P1_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P1 .thmB q 0 ↔ 244 ≤ q := P1_cert.iff0 q

/-- (2) `q = 196` is the smallest query count meeting the convention with `b = 20`. -/
theorem P1_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P1 .thmB q 20 ↔ 196 ≤ q := P1_cert.iff20 q

/-- (3) Theorem B' fold term, `(244, 0)`. -/
theorem P1_Bp_q244_b0 :
    ∀ T : ℕ, P1.R ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmBp 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P1_Bp_cert.at244

/-- (3) Theorem B' fold term, `(196, 20)`. -/
theorem P1_Bp_q196_b20 :
    ∀ T : ℕ, P1.R ≤ T → T ≤ 2 ^ 100 → P1.fsBound .thmBp 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P1_Bp_cert.at196

/-- (3) Theorem B': the threshold is again `q = 244` (`b = 0`). -/
theorem P1_Bp_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P1 .thmBp q 0 ↔ 244 ≤ q := P1_Bp_cert.iff0 q

/-- (3) Theorem B': the threshold is again `q = 196` (`b = 20`). -/
theorem P1_Bp_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P1 .thmBp q 20 ↔ 196 ≤ q := P1_Bp_cert.iff20 q

theorem P2_target : E0 ≤ P2.target :=
  E0_le_target P2 (by simp) (by norm_num [P2_hashC])

/-- All six facts for `P2`, Theorem B. -/
theorem P2_cert : Cert P2 .thmB :=
  cert_of_numerics P2 .thmB (by simp) (by norm_num) P2_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- All six facts for `P2`, Theorem B' (the fold term only gets smaller). -/
theorem P2_Bp_cert : Cert P2 .thmBp :=
  cert_of_numerics P2 .thmBp (by simp) (by norm_num) P2_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- `P2`, Model H, `(q, b) = (244, 0)`: the query round is the largest, `ε_max = (3/4)^244`. -/
theorem P2_epsMax_q244_b0 : P2.epsMax .thmB 244 0 = (3 / 4) ^ 244 :=
  (epsMax_eq_epsQ_of P2 .thmB 244 0 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- `P2`, Model H, `(q, b) = (196, 20)`: `ε_max = 2^-20 (3/4)^196`. -/
theorem P2_epsMax_q196_b20 : P2.epsMax .thmB 196 20 = (1 / 2) ^ 20 * (3 / 4) ^ 196 :=
  (epsMax_eq_epsQ_of P2 .thmB 196 20 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- (1) `P2`, Model H, `(q, b) = (244, 0)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P2_q244_b0 :
    ∀ T : ℕ, P2.R ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmB 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P2_cert.at244

/-- (1) `P2`, Model H, `(q, b) = (196, 20)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P2_q196_b20 :
    ∀ T : ℕ, P2.R ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmB 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P2_cert.at196

/-- (1) the same for every rational (real-valued query budget) `T` with `R ≤ T ≤ 2^100`, `(244, 0)`. -/
theorem P2_q244_b0_rat :
    ∀ T : ℚ, (P2.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmB 244 0 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P2 .thmB 244 0 (by simp)
    (le_trans (by rw [P2_epsMax_q244_b0]; bits_eps_norm) P2_target)

/-- (1) the same for every rational `T` with `R ≤ T ≤ 2^100`, `(196, 20)`. -/
theorem P2_q196_b20_rat :
    ∀ T : ℚ, (P2.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmB 196 20 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P2 .thmB 196 20 (by simp)
    (le_trans (by rw [P2_epsMax_q196_b20]; bits_eps_norm) P2_target)

/-- (1) `min`-form of the convention: `min (1, bound T) · 2^100 ≤ T` for every natural `T ≥ R`, `(q, b) = (244, 0)`. -/
theorem P2_workRatio_q244_b0 : WorkRatio100 P2 .thmB 244 0 := (P2_cert.iff0 244).2 le_rfl

/-- (1) the same for `(q, b) = (196, 20)`. -/
theorem P2_workRatio_q196_b20 : WorkRatio100 P2 .thmB 196 20 := (P2_cert.iff20 196).2 le_rfl

/-- `AT_BUDGET` (the point `T = 2^100`), `(244, 0)`. -/
theorem P2_atBudget_q244_b0 : P2.fsBound .thmB 244 0 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P2 .thmB 244 0 (by norm_num) P2_q244_b0

/-- `AT_BUDGET`, `(196, 20)`. -/
theorem P2_atBudget_q196_b20 : P2.fsBound .thmB 196 20 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P2 .thmB 196 20 (by norm_num) P2_q196_b20

/-- (2) minimality, `(q, b) = (243, 0)`: the inequality fails at `T = R` (indeed for every `q ≤ 243`, `P2_cert.fail_b0`). -/
theorem P2_q243_b0_fails_at_R :
    ¬ (P2.fsBound .thmB 243 0 (P2.R : ℚ) ≤ (P2.R : ℚ) / 2 ^ 100) := P2_cert.fail_b0 243 le_rfl

/-- (2) minimality, `(q, b) = (195, 20)`. -/
theorem P2_q195_b20_fails_at_R :
    ¬ (P2.fsBound .thmB 195 20 (P2.R : ℚ) ≤ (P2.R : ℚ) / 2 ^ 100) := P2_cert.fail_b20 195 le_rfl

/-- (2) `q = 244` is the smallest query count meeting the convention with `b = 0`. -/
theorem P2_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P2 .thmB q 0 ↔ 244 ≤ q := P2_cert.iff0 q

/-- (2) `q = 196` is the smallest query count meeting the convention with `b = 20`. -/
theorem P2_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P2 .thmB q 20 ↔ 196 ≤ q := P2_cert.iff20 q

/-- (3) Theorem B' fold term, `(244, 0)`. -/
theorem P2_Bp_q244_b0 :
    ∀ T : ℕ, P2.R ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmBp 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P2_Bp_cert.at244

/-- (3) Theorem B' fold term, `(196, 20)`. -/
theorem P2_Bp_q196_b20 :
    ∀ T : ℕ, P2.R ≤ T → T ≤ 2 ^ 100 → P2.fsBound .thmBp 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P2_Bp_cert.at196

/-- (3) Theorem B': the threshold is again `q = 244` (`b = 0`). -/
theorem P2_Bp_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P2 .thmBp q 0 ↔ 244 ≤ q := P2_Bp_cert.iff0 q

/-- (3) Theorem B': the threshold is again `q = 196` (`b = 20`). -/
theorem P2_Bp_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P2 .thmBp q 20 ↔ 196 ≤ q := P2_Bp_cert.iff20 q

theorem P3_target : E0 ≤ P3.target :=
  E0_le_target P3 (by simp) (by norm_num [P3_hashC])

/-- All six facts for `P3`, Theorem B. -/
theorem P3_cert : Cert P3 .thmB :=
  cert_of_numerics P3 .thmB (by simp) (by norm_num) P3_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- All six facts for `P3`, Theorem B' (the fold term only gets smaller). -/
theorem P3_Bp_cert : Cert P3 .thmBp :=
  cert_of_numerics P3 .thmBp (by simp) (by norm_num) P3_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- `P3`, Model H, `(q, b) = (244, 0)`: the query round is the largest, `ε_max = (3/4)^244`. -/
theorem P3_epsMax_q244_b0 : P3.epsMax .thmB 244 0 = (3 / 4) ^ 244 :=
  (epsMax_eq_epsQ_of P3 .thmB 244 0 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- `P3`, Model H, `(q, b) = (196, 20)`: `ε_max = 2^-20 (3/4)^196`. -/
theorem P3_epsMax_q196_b20 : P3.epsMax .thmB 196 20 = (1 / 2) ^ 20 * (3 / 4) ^ 196 :=
  (epsMax_eq_epsQ_of P3 .thmB 196 20 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)).trans (by norm_num [Profile.epsQ, betaUD])

/-- (1) `P3`, Model H, `(q, b) = (244, 0)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P3_q244_b0 :
    ∀ T : ℕ, P3.R ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmB 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P3_cert.at244

/-- (1) `P3`, Model H, `(q, b) = (196, 20)`: for every natural `T` with `R ≤ T ≤ 2^100`, `bound T ≤ T/2^100`. -/
theorem P3_q196_b20 :
    ∀ T : ℕ, P3.R ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmB 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P3_cert.at196

/-- (1) the same for every rational (real-valued query budget) `T` with `R ≤ T ≤ 2^100`, `(244, 0)`. -/
theorem P3_q244_b0_rat :
    ∀ T : ℚ, (P3.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmB 244 0 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P3 .thmB 244 0 (by simp)
    (le_trans (by rw [P3_epsMax_q244_b0]; bits_eps_norm) P3_target)

/-- (1) the same for every rational `T` with `R ≤ T ≤ 2^100`, `(196, 20)`. -/
theorem P3_q196_b20_rat :
    ∀ T : ℚ, (P3.R : ℚ) ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmB 196 20 T ≤ T / 2 ^ 100 :=
  fsBound_le_target P3 .thmB 196 20 (by simp)
    (le_trans (by rw [P3_epsMax_q196_b20]; bits_eps_norm) P3_target)

/-- (1) `min`-form of the convention: `min (1, bound T) · 2^100 ≤ T` for every natural `T ≥ R`, `(q, b) = (244, 0)`. -/
theorem P3_workRatio_q244_b0 : WorkRatio100 P3 .thmB 244 0 := (P3_cert.iff0 244).2 le_rfl

/-- (1) the same for `(q, b) = (196, 20)`. -/
theorem P3_workRatio_q196_b20 : WorkRatio100 P3 .thmB 196 20 := (P3_cert.iff20 196).2 le_rfl

/-- `AT_BUDGET` (the point `T = 2^100`), `(244, 0)`. -/
theorem P3_atBudget_q244_b0 : P3.fsBound .thmB 244 0 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P3 .thmB 244 0 (by norm_num) P3_q244_b0

/-- `AT_BUDGET`, `(196, 20)`. -/
theorem P3_atBudget_q196_b20 : P3.fsBound .thmB 196 20 (2 ^ 100) ≤ 1 :=
  atBudget_of_nat P3 .thmB 196 20 (by norm_num) P3_q196_b20

/-- (2) minimality, `(q, b) = (243, 0)`: the inequality fails at `T = R` (indeed for every `q ≤ 243`, `P3_cert.fail_b0`). -/
theorem P3_q243_b0_fails_at_R :
    ¬ (P3.fsBound .thmB 243 0 (P3.R : ℚ) ≤ (P3.R : ℚ) / 2 ^ 100) := P3_cert.fail_b0 243 le_rfl

/-- (2) minimality, `(q, b) = (195, 20)`. -/
theorem P3_q195_b20_fails_at_R :
    ¬ (P3.fsBound .thmB 195 20 (P3.R : ℚ) ≤ (P3.R : ℚ) / 2 ^ 100) := P3_cert.fail_b20 195 le_rfl

/-- (2) `q = 244` is the smallest query count meeting the convention with `b = 0`. -/
theorem P3_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P3 .thmB q 0 ↔ 244 ≤ q := P3_cert.iff0 q

/-- (2) `q = 196` is the smallest query count meeting the convention with `b = 20`. -/
theorem P3_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P3 .thmB q 20 ↔ 196 ≤ q := P3_cert.iff20 q

/-- (3) Theorem B' fold term, `(244, 0)`. -/
theorem P3_Bp_q244_b0 :
    ∀ T : ℕ, P3.R ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmBp 244 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P3_Bp_cert.at244

/-- (3) Theorem B' fold term, `(196, 20)`. -/
theorem P3_Bp_q196_b20 :
    ∀ T : ℕ, P3.R ≤ T → T ≤ 2 ^ 100 → P3.fsBound .thmBp 196 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P3_Bp_cert.at196

/-- (3) Theorem B': the threshold is again `q = 244` (`b = 0`). -/
theorem P3_Bp_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P3 .thmBp q 0 ↔ 244 ≤ q := P3_Bp_cert.iff0 q

/-- (3) Theorem B': the threshold is again `q = 196` (`b = 20`). -/
theorem P3_Bp_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P3 .thmBp q 20 ↔ 196 ≤ q := P3_Bp_cert.iff20 q

/-! ## 8. Theorem (3): the profile P4 (`n = 25`) under Theorem B'

Result: **no** `(q, b)` meets the convention (the failing point is `T = R`); `AT_BUDGET` and the variant with
minimum over `T ≥ R + 1` are met.  Model H only (see the module documentation on Model A). -/

theorem P4_Bp_fails_at_R_aux : (P4.R : ℚ) / 2 ^ 100 < ((P4.R : ℚ) + P4.R) * P4.epsBp := by
  norm_num [Profile.epsBp, r1, rLimb, Ksz, pM31]

/-- (3) P4 (`n = 25, L = 18, g = 3`), Model H, Theorem B': for EVERY `q` and `b` the inequality `bound R ≤ R/2^100`
fails at `T = R` (the fold round alone gives `2 ε_fold = r1 2^24/|K| > 2^-100`, because `|K| < 2^124`). -/
theorem P4_Bp_fails_at_R (q b : ℕ) :
    ¬ (P4.fsBound .thmBp q b (P4.R : ℚ) ≤ (P4.R : ℚ) / 2 ^ 100) :=
  not_le_at_R P4 .thmBp q b P4.epsBp (epsFold_le_epsMax P4 .thmBp q b) P4_Bp_fails_at_R_aux

/-- (3) hence P4 under B' meets the `min`-form convention for no `q` and no `b`. -/
theorem P4_Bp_not_workRatio (q b : ℕ) : ¬ WorkRatio100 P4 .thmBp q b :=
  not_workRatio100_of_fail_at_R P4 .thmBp q b (by norm_num) (P4_Bp_fails_at_R q b)

/-- (3) quantitative sharpness: the failure is by a factor `< 1 + 2^-28` (`3.4·10^-9` bits):
`2^-100 < 2 ε_fold < (1 + 2^-28) 2^-100` for `ε_fold = r1 (N/2)/|K|`, `N = 2^24`. -/
theorem P4_Bp_shortfall : 1 / 2 ^ 100 < 2 * P4.epsBp ∧ 2 * P4.epsBp < (1 + 1 / 2 ^ 28) / 2 ^ 100 := by
  constructor <;> norm_num [Profile.epsBp, r1, rLimb, Ksz, pM31]

/-- The same for Theorem B (one bit worse: `2 ε_fold = r1 2^25/|K|`). -/
theorem P4_B_fails_at_R (q b : ℕ) :
    ¬ (P4.fsBound .thmB q b (P4.R : ℚ) ≤ (P4.R : ℚ) / 2 ^ 100) := by
  apply not_le_at_R P4 .thmB q b P4.epsB (epsFold_le_epsMax P4 .thmB q b)
  have h := P4_Bp_fails_at_R_aux
  have h2 : P4.epsBp ≤ P4.epsB := epsBp_le_epsB P4
  have hR : (0 : ℚ) ≤ (P4.R : ℚ) + P4.R := by positivity
  nlinarith

/-- For P4 under B' (`q = 244`, `b = 0`) the fold round is the largest. -/
theorem P4_epsMax_Bp_q244_b0 : P4.epsMax .thmBp 244 0 = P4.epsBp :=
  epsMax_eq_fold_of P4 .thmBp 244 0 (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm)

/-- The non-query errors of P4 are at most the Theorem-B' fold term. -/
theorem P4_nonquery_le_epsBp : P4.epsGamma ≤ P4.epsBp ∧ epsA ≤ P4.epsBp ∧ P4.epsMu ≤ P4.epsBp := by
  refine ⟨?_, ?_, ?_⟩ <;>
    norm_num [Profile.epsGamma, epsA, Profile.epsMu, Profile.epsBp, r1, rLimb, Ksz, pM31]

theorem P4_Bp_from_R_plus_one (q b : ℕ) (hq : P4.epsQ q b ≤ P4.epsBp) :
    ∀ T : ℕ, P4.R + 1 ≤ T → T ≤ 2 ^ 100 → P4.fsBound .thmBp q b T ≤ (T : ℚ) / 2 ^ 100 := by
  intro T h1 h2
  obtain ⟨g1, g2, g3⟩ := P4_nonquery_le_epsBp
  have hE : P4.epsMax .thmBp q b ≤ P4.epsBp :=
    (epsMax_le_iff P4 .thmBp q b P4.epsBp).2 ⟨g1, g2, g3, le_rfl, hq⟩
  refine fsBound_le P4 .thmBp q b ((P4.R : ℚ) + 1) P4.epsBp (by positivity) hE ?_ (T : ℚ) ?_ ?_
  · norm_num [Profile.hslope, Profile.hashC, Profile.epsBp, P4_D, P4_sumL, P4_l0, r1, rLimb, Ksz,
      pM31]
  · exact_mod_cast h1
  · exact_mod_cast h2

/-- (3) P4 under B', `b = 0`: the convention holds for every natural `T` with `R + 1 ≤ T ≤ 2^100` as soon as `q ≥ 244`;
only the single point `T = R` fails (`P4_Bp_fails_at_R`). -/
theorem P4_Bp_q244_b0_from_R_plus_one (q : ℕ) (hq : 244 ≤ q) :
    ∀ T : ℕ, P4.R + 1 ≤ T → T ≤ 2 ^ 100 → P4.fsBound .thmBp q 0 T ≤ (T : ℚ) / 2 ^ 100 :=
  P4_Bp_from_R_plus_one q 0 (le_trans (epsQ_anti P4 0 hq)
    (by norm_num [Profile.epsQ, betaUD, Profile.epsBp, r1, rLimb, Ksz, pM31]))

/-- (3) P4 under B', `b = 20`: the same for `q ≥ 196`. -/
theorem P4_Bp_q196_b20_from_R_plus_one (q : ℕ) (hq : 196 ≤ q) :
    ∀ T : ℕ, P4.R + 1 ≤ T → T ≤ 2 ^ 100 → P4.fsBound .thmBp q 20 T ≤ (T : ℚ) / 2 ^ 100 :=
  P4_Bp_from_R_plus_one q 20 (le_trans (epsQ_anti P4 20 hq)
    (by norm_num [Profile.epsQ, betaUD, Profile.epsBp, r1, rLimb, Ksz, pM31]))

/-- (3) P4 under B' meets `AT_BUDGET` (`bound (2^100) ≤ 1`) with `q = 241`, `b = 0`
(under Theorem B no `q` does: `P4_B_fails_at_R` and the fold term `2^-99.9999999966`). -/
theorem P4_Bp_atBudget_q241 : P4.fsBound .thmBp 241 0 (2 ^ 100) ≤ 1 := by
  refine fsBound_atBudget P4 .thmBp 241 0 ((3 / 4 : ℚ) ^ 241) ?_ ?_
  · rw [epsMax_le_iff]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [Profile.epsGamma, epsA, Profile.epsMu, Profile.epsFold, Profile.epsBp, Profile.epsQ,
      betaUD, r1, rLimb, Ksz, pM31]
  · norm_num [Profile.hslope, Profile.hashC, P4_D, P4_sumL, P4_l0]

/-- (3) `AT_BUDGET` for P4 under B' with `q = 193`, `b = 20`. -/
theorem P4_Bp_atBudget_q193_b20 : P4.fsBound .thmBp 193 20 (2 ^ 100) ≤ 1 := by
  refine fsBound_atBudget P4 .thmBp 193 20 ((1 / 2 : ℚ) ^ 20 * (3 / 4 : ℚ) ^ 193) ?_ ?_
  · rw [epsMax_le_iff]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [Profile.epsGamma, epsA, Profile.epsMu, Profile.epsFold, Profile.epsBp, Profile.epsQ,
      betaUD, r1, rLimb, Ksz, pM31]
  · norm_num [Profile.hslope, Profile.hashC, P4_D, P4_sumL, P4_l0]

/-! ## 9. Extension: P5 (`n = 24`) is the largest `n` meeting the convention under B' (and fails under B) -/

theorem P5_target : E0 ≤ P5.target :=
  E0_le_target P5 (by simp) (by norm_num [P5_hashC])

theorem P5_Bp_cert : Cert P5 .thmBp :=
  cert_of_numerics P5 .thmBp (by simp) (by norm_num) P5_target
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)
    (by bits_eps_norm) (by bits_eps_norm) (by bits_eps_norm)

/-- P5 (`n = 24`), Model H, Theorem B': the convention holds iff `q ≥ 244` (`b = 0`). -/
theorem P5_Bp_workRatio_iff_b0 (q : ℕ) : WorkRatio100 P5 .thmBp q 0 ↔ 244 ≤ q := P5_Bp_cert.iff0 q

/-- P5, Theorem B': the convention holds iff `q ≥ 196` (`b = 20`). -/
theorem P5_Bp_workRatio_iff_b20 (q : ℕ) : WorkRatio100 P5 .thmBp q 20 ↔ 196 ≤ q := P5_Bp_cert.iff20 q

/-- P5 under Theorem B (`r1 N/|K|`, `N = 2^23`): fails at `T = R` for every `q` and `b`. -/
theorem P5_B_fails_at_R (q b : ℕ) :
    ¬ (P5.fsBound .thmB q b (P5.R : ℚ) ≤ (P5.R : ℚ) / 2 ^ 100) :=
  not_le_at_R P5 .thmB q b P5.epsB (epsFold_le_epsMax P5 .thmB q b)
    (by norm_num [Profile.epsB, r1, rLimb, Ksz, pM31])

/-! ## 10. Spot checks: kernel evaluation of the definitions at single points

These do not use the generic lemmas above; they evaluate `fsBound` (exact `ℚ` arithmetic) at the decisive points and must agree
with the Results table of the module comment.  `decide +kernel` is evaluation by the kernel (no extra assumptions). -/

theorem spot_P1_244_at_R : P1.fsBound .thmB 244 0 43 ≤ 43 / 2 ^ 100 := by decide +kernel
theorem spot_P1_243_at_R : ¬ (P1.fsBound .thmB 243 0 43 ≤ 43 / 2 ^ 100) := by decide +kernel
theorem spot_P1_244_budget : P1.fsBound .thmB 244 0 (2 ^ 100) ≤ 1 := by decide +kernel
theorem spot_P1_196_at_R_b20 : P1.fsBound .thmB 196 20 43 ≤ 43 / 2 ^ 100 := by decide +kernel
theorem spot_P1_195_at_R_b20 : ¬ (P1.fsBound .thmB 195 20 43 ≤ 43 / 2 ^ 100) := by decide +kernel
theorem spot_P3_244_at_R : P3.fsBound .thmB 244 0 32 ≤ 32 / 2 ^ 100 := by decide +kernel
theorem spot_P3_243_at_R : ¬ (P3.fsBound .thmB 243 0 32 ≤ 32 / 2 ^ 100) := by decide +kernel
theorem spot_P4_Bp_at_R : ¬ (P4.fsBound .thmBp 244 0 49 ≤ 49 / 2 ^ 100) := by decide +kernel
theorem spot_P4_Bp_at_R_plus_one : P4.fsBound .thmBp 244 0 50 ≤ 50 / 2 ^ 100 := by decide +kernel
theorem spot_P4_B_budget : ¬ (P4.fsBound .thmB 244 0 (2 ^ 100) ≤ 1) := by decide +kernel
theorem spot_P4_Bp_budget : P4.fsBound .thmBp 241 0 (2 ^ 100) ≤ 1 := by decide +kernel
theorem spot_P5_Bp_at_R : P5.fsBound .thmBp 244 0 47 ≤ 47 / 2 ^ 100 := by decide +kernel
theorem spot_P5_B_at_R : ¬ (P5.fsBound .thmB 244 0 47 ≤ 47 / 2 ^ 100) := by decide +kernel

/-! ## 11. Axioms (each must print only `propext`, `Classical.choice`, `Quot.sound`) -/

#print axioms P1_epsMax_q244_b0
#print axioms P1_epsMax_q196_b20
#print axioms P1_q244_b0
#print axioms P1_q196_b20
#print axioms P1_q244_b0_rat
#print axioms P1_q196_b20_rat
#print axioms P1_workRatio_q244_b0
#print axioms P1_workRatio_q196_b20
#print axioms P1_atBudget_q244_b0
#print axioms P1_atBudget_q196_b20
#print axioms P1_q243_b0_fails_at_R
#print axioms P1_q195_b20_fails_at_R
#print axioms P1_workRatio_iff_b0
#print axioms P1_workRatio_iff_b20
#print axioms P1_Bp_q244_b0
#print axioms P1_Bp_q196_b20
#print axioms P1_Bp_workRatio_iff_b0
#print axioms P1_Bp_workRatio_iff_b20
#print axioms P2_epsMax_q244_b0
#print axioms P2_epsMax_q196_b20
#print axioms P2_q244_b0
#print axioms P2_q196_b20
#print axioms P2_q244_b0_rat
#print axioms P2_q196_b20_rat
#print axioms P2_workRatio_q244_b0
#print axioms P2_workRatio_q196_b20
#print axioms P2_atBudget_q244_b0
#print axioms P2_atBudget_q196_b20
#print axioms P2_q243_b0_fails_at_R
#print axioms P2_q195_b20_fails_at_R
#print axioms P2_workRatio_iff_b0
#print axioms P2_workRatio_iff_b20
#print axioms P2_Bp_q244_b0
#print axioms P2_Bp_q196_b20
#print axioms P2_Bp_workRatio_iff_b0
#print axioms P2_Bp_workRatio_iff_b20
#print axioms P3_epsMax_q244_b0
#print axioms P3_epsMax_q196_b20
#print axioms P3_q244_b0
#print axioms P3_q196_b20
#print axioms P3_q244_b0_rat
#print axioms P3_q196_b20_rat
#print axioms P3_workRatio_q244_b0
#print axioms P3_workRatio_q196_b20
#print axioms P3_atBudget_q244_b0
#print axioms P3_atBudget_q196_b20
#print axioms P3_q243_b0_fails_at_R
#print axioms P3_q195_b20_fails_at_R
#print axioms P3_workRatio_iff_b0
#print axioms P3_workRatio_iff_b20
#print axioms P3_Bp_q244_b0
#print axioms P3_Bp_q196_b20
#print axioms P3_Bp_workRatio_iff_b0
#print axioms P3_Bp_workRatio_iff_b20
#print axioms P4_Bp_fails_at_R
#print axioms P4_Bp_not_workRatio
#print axioms P4_Bp_shortfall
#print axioms P4_B_fails_at_R
#print axioms P4_epsMax_Bp_q244_b0
#print axioms P4_Bp_q244_b0_from_R_plus_one
#print axioms P4_Bp_q196_b20_from_R_plus_one
#print axioms P4_Bp_atBudget_q241
#print axioms P4_Bp_atBudget_q193_b20
#print axioms P5_Bp_workRatio_iff_b0
#print axioms P5_Bp_workRatio_iff_b20
#print axioms P5_B_fails_at_R
#print axioms pM31_eq
#print axioms Ksz_eq
#print axioms r1_eq
#print axioms one_lt_r1
#print axioms r1_lt
#print axioms Ksz_lt_two_pow_124
#print axioms Ksz_lt_r1_mul
#print axioms betaUD_one
#print axioms P1_hashC
#print axioms P2_hashC
#print axioms P3_hashC
#print axioms P4_hashC
#print axioms P5_hashC
#print axioms Profile.R_eq
#print axioms spot_P1_244_at_R
#print axioms spot_P1_243_at_R
#print axioms spot_P1_244_budget
#print axioms spot_P1_196_at_R_b20
#print axioms spot_P1_195_at_R_b20
#print axioms spot_P3_244_at_R
#print axioms spot_P3_243_at_R
#print axioms spot_P4_Bp_at_R
#print axioms spot_P4_Bp_at_R_plus_one
#print axioms spot_P4_B_budget
#print axioms spot_P4_Bp_budget
#print axioms spot_P5_Bp_at_R
#print axioms spot_P5_B_at_R
#print axioms hashStar_le
#print axioms fsBound_le
#print axioms fsBound_le_target
#print axioms workRatio_nat
#print axioms workRatio100_of_le
#print axioms fsBound_atBudget
#print axioms not_le_at_R
#print axioms not_le_at_R_of_query
#print axioms not_workRatio100_of_fail_at_R
#print axioms workRatio100_iff
#print axioms cert_of_numerics
#print axioms P1_cert
#print axioms P2_cert
#print axioms P3_cert
#print axioms P1_Bp_cert
#print axioms P2_Bp_cert
#print axioms P3_Bp_cert
#print axioms P5_Bp_cert
#print axioms P1_depths
#print axioms P4_depths

end Cprs.Bits
