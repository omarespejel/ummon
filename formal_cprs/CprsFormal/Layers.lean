import CprsFormal.PhaseB

/-!
# The layer domains `D_t = ⟨ζ^(2^t)⟩` of CPRS satisfy the hypotheses of `PhaseBInst.WF`

`D_t = {ζ^(2^t j) : j < 2^(m+s-t)}` for an element `ζ` of multiplicative order `N = 2^(m+s)`
(in the reference implementation
`D_t[j] = ζ^(2^t j)` and squaring is `j ↦ j mod n_{t+1}`).
-/

open Polynomial Finset

namespace Cprs

variable {K : Type*} [Field K] [DecidableEq K]

/-- `D_t = {ζ^(2^t j) : j < 2^(m+s-t)}`. -/
def layerDomain (ζ : K) (m s t : ℕ) : Finset K :=
  (Finset.range (2 ^ (m + s - t))).image (fun j => ζ ^ (2 ^ t * j))

namespace Aux

omit [DecidableEq K] in
/-- A root of unity of order `2 ^ n` is nonzero. -/
theorem aux_ne_zero (ζ : K) (n : ℕ) (hζ : orderOf ζ = 2 ^ n) : ζ ≠ 0 := by
  intro h
  have h1 : ζ ^ (2 ^ n) = 1 := by
    rw [← hζ]
    exact pow_orderOf_eq_one ζ
  rw [h, zero_pow (pow_ne_zero n (by norm_num))] at h1
  exact zero_ne_one h1

omit [DecidableEq K] in
/-- The element of order `2` in `⟨ζ⟩` is `-1`: `ζ ^ (2 ^ (n - 1)) = -1` when `orderOf ζ = 2 ^ n`,
`n ≥ 1`. -/
theorem aux_half (ζ : K) (n : ℕ) (hζ : orderOf ζ = 2 ^ n) (hn : 1 ≤ n) :
    ζ ^ (2 ^ (n - 1)) = -1 := by
  have hpow : 2 ^ (n - 1) * 2 = 2 ^ n := by
    rw [← pow_succ]
    congr 1
    omega
  have hsq : (ζ ^ (2 ^ (n - 1))) ^ 2 = 1 := by
    rw [← pow_mul, hpow, ← hζ]
    exact pow_orderOf_eq_one ζ
  have hn0 : (2 : ℕ) ^ (n - 1) ≠ 0 := pow_ne_zero _ (by norm_num)
  have hlt : (2 : ℕ) ^ (n - 1) < orderOf ζ := by
    rw [hζ]
    exact Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hne : ζ ^ (2 ^ (n - 1)) ≠ 1 := pow_ne_one_of_lt_orderOf hn0 hlt
  have h0 : (ζ ^ (2 ^ (n - 1)) - 1) * (ζ ^ (2 ^ (n - 1)) + 1) = 0 := by
    linear_combination hsq
  rcases mul_eq_zero.1 h0 with h | h
  · exact absurd (sub_eq_zero.1 h) hne
  · linear_combination h

omit [DecidableEq K] in
/-- The characteristic is not `2` when `K` contains a root of unity of order `2 ^ n`, `n ≥ 1`. -/
theorem aux_two_ne_zero (ζ : K) (n : ℕ) (hζ : orderOf ζ = 2 ^ n) (hn : 1 ≤ n) :
    (2 : K) ≠ 0 := by
  intro h2
  have hhalf := aux_half ζ n hζ hn
  have hn0 : (2 : ℕ) ^ (n - 1) ≠ 0 := pow_ne_zero _ (by norm_num)
  have hlt : (2 : ℕ) ^ (n - 1) < orderOf ζ := by
    rw [hζ]
    exact Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hne : ζ ^ (2 ^ (n - 1)) ≠ 1 := pow_ne_one_of_lt_orderOf hn0 hlt
  apply hne
  rw [hhalf]
  linear_combination (-1 : K) * h2

/-- Membership in `D_t` (for `t ≤ m + s`): `D_t` is the set of all natural powers of `ζ ^ (2 ^ t)`
(the exponent `j` may be reduced modulo `2 ^ (m + s - t)`). -/
theorem aux_mem_layerDomain (ζ : K) (m s : ℕ) (hζ : orderOf ζ = 2 ^ (m + s)) {t : ℕ}
    (ht : t ≤ m + s) (x : K) :
    x ∈ layerDomain ζ m s t ↔ ∃ j : ℕ, ζ ^ (2 ^ t * j) = x := by
  unfold layerDomain
  rw [Finset.mem_image]
  constructor
  · rintro ⟨j, -, rfl⟩
    exact ⟨j, rfl⟩
  · rintro ⟨j, rfl⟩
    refine ⟨j % 2 ^ (m + s - t), Finset.mem_range.2 (Nat.mod_lt _ (by positivity)), ?_⟩
    show ζ ^ (2 ^ t * (j % 2 ^ (m + s - t))) = ζ ^ (2 ^ t * j)
    have h2 : 2 ^ t * 2 ^ (m + s - t) = 2 ^ (m + s) := by
      rw [← pow_add]
      congr 1
      omega
    rw [← pow_mod_orderOf ζ (2 ^ t * j), hζ, ← h2, Nat.mul_mod_mul_left]

end Aux

/-- Cardinality: for `t ≤ m + s`, `|D_t| = 2^(m+s-t)`. -/
theorem layerDomain_card (ζ : K) (m s : ℕ) (hζ : orderOf ζ = 2 ^ (m + s)) {t : ℕ}
    (ht : t ≤ m + s) : (layerDomain ζ m s t).card = 2 ^ (m + s - t) := by
  have hfin : IsOfFinOrder ζ := orderOf_pos_iff.1 (by rw [hζ]; positivity)
  unfold layerDomain
  rw [Finset.card_image_of_injOn]
  · exact Finset.card_range _
  · intro j hj j' hj' hjj'
    have hj1 : j < 2 ^ (m + s - t) := Finset.mem_range.1 (Finset.mem_coe.1 hj)
    have hj2 : j' < 2 ^ (m + s - t) := Finset.mem_range.1 (Finset.mem_coe.1 hj')
    have h1 : Nat.ModEq (2 ^ (m + s)) (2 ^ t * j) (2 ^ t * j') := by
      rw [← hζ]
      exact hfin.pow_eq_pow_iff_modEq.1 hjj'
    have h2 : 2 ^ (m + s) = 2 ^ t * 2 ^ (m + s - t) := by
      rw [← pow_add]
      congr 1
      omega
    rw [h2] at h1
    exact Nat.ModEq.eq_of_lt_of_lt
      (Nat.ModEq.mul_left_cancel' (pow_ne_zero _ (by norm_num)) h1) hj1 hj2

/-- Consecutive layers form a squaring layer (negation closed, `0 ∉ D`, `2 ≠ 0`, `x ↦ x²` onto). -/
theorem layerDomain_sqLayer (ζ : K) (m s : ℕ) (hζ : orderOf ζ = 2 ^ (m + s)) (hs : 1 ≤ s)
    {t : ℕ} (ht : t < m) : SqLayer (layerDomain ζ m s t) (layerDomain ζ m s (t + 1)) := by
  have hmt : t + 1 ≤ m + s := by omega
  have ht' : t ≤ m + s := by omega
  have hζ0 : ζ ≠ 0 := Aux.aux_ne_zero ζ (m + s) hζ
  refine ⟨Aux.aux_two_ne_zero ζ (m + s) hζ (by omega), ?_, ?_, ?_, ?_⟩
  · -- `0 ∉ D_t`
    intro h0
    obtain ⟨j, hj⟩ := (Aux.aux_mem_layerDomain ζ m s hζ ht' 0).1 h0
    exact pow_ne_zero _ hζ0 hj
  · -- `D_t` is closed under negation: `-1 = ζ ^ (2 ^ (m + s - 1))` and `2 ^ t ∣ 2 ^ (m + s - 1)`
    intro x hx
    obtain ⟨j, rfl⟩ := (Aux.aux_mem_layerDomain ζ m s hζ ht' x).1 hx
    rw [Aux.aux_mem_layerDomain ζ m s hζ ht']
    refine ⟨j + 2 ^ (m + s - 1 - t), ?_⟩
    have h2 : 2 ^ t * 2 ^ (m + s - 1 - t) = 2 ^ (m + s - 1) := by
      rw [← pow_add]
      congr 1
      omega
    have h1 : 2 ^ t * (j + 2 ^ (m + s - 1 - t)) = 2 ^ t * j + 2 ^ (m + s - 1) := by
      rw [mul_add, h2]
    rw [h1, pow_add, Aux.aux_half ζ (m + s) hζ (by omega)]
    ring
  · -- squaring maps `D_t` into `D_(t+1)`
    intro x hx
    obtain ⟨j, rfl⟩ := (Aux.aux_mem_layerDomain ζ m s hζ ht' x).1 hx
    rw [Aux.aux_mem_layerDomain ζ m s hζ hmt]
    refine ⟨j, ?_⟩
    rw [← pow_mul]
    congr 1
    ring
  · -- squaring is onto `D_(t+1)`
    intro y hy
    obtain ⟨j, rfl⟩ := (Aux.aux_mem_layerDomain ζ m s hζ hmt y).1 hy
    refine ⟨ζ ^ (2 ^ t * j), ?_, ?_⟩
    · rw [Aux.aux_mem_layerDomain ζ m s hζ ht']
      exact ⟨j, rfl⟩
    · rw [← pow_mul]
      congr 1
      ring

/-- A Phase-B instance on the CPRS layers. -/
def PhaseBInst.ofRoot (ζ : K) (m s : ℕ) (ρ : List K) (isEnd : ℕ → Bool) (t1 : ℕ) (π : K → K)
    (c0 : K) : PhaseBInst K where
  m := m
  s := s
  D := layerDomain ζ m s
  ρ := ρ
  isEnd := isEnd
  t1 := t1
  π := π
  c0 := c0

/-- The instance built from a root of unity of order `2^(m+s)` is well formed. -/
theorem PhaseBInst.ofRoot_wf (ζ : K) (m s : ℕ) (hζ : orderOf ζ = 2 ^ (m + s)) (hs : 1 ≤ s)
    (ρ : List K) (hρ : ρ.length = m) (isEnd : ℕ → Bool) (t1 : ℕ) (ht1pos : 0 < t1)
    (ht1m : t1 ≤ m) (hend : ∀ t, isEnd t = true → t1 ≤ t ∧ t < m)
    (hend1 : t1 < m → isEnd t1 = true) (hno : ∀ t, 0 < t → t < t1 → isEnd t = false)
    (π : K → K) (c0 : K) : (PhaseBInst.ofRoot ζ m s ρ isEnd t1 π c0).WF := by
  refine ⟨hs, hρ, ?_, ?_, ht1pos, ht1m, hend, hend1, hno⟩
  · intro t ht
    have ht' : t < m := ht
    show SqLayer (layerDomain ζ m s t) (layerDomain ζ m s (t + 1))
    exact layerDomain_sqLayer ζ m s hζ hs ht'
  · intro t ht
    have ht' : t ≤ m := ht
    show (layerDomain ζ m s t).card = 2 ^ (m + s - t)
    exact layerDomain_card ζ m s hζ (by omega)

end Cprs

#print axioms Cprs.layerDomain_card
#print axioms Cprs.layerDomain_sqLayer
#print axioms Cprs.PhaseBInst.ofRoot_wf
