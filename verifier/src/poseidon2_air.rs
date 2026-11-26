//! EXPERIMENTAL (`exp_pos2_*` profiles, not production-ready): one Poseidon2-M31 permutation per
//! row (see `poseidon2_m31.rs` for the instance), the in-circuit hash of the Poseidon2 profile.
//!
//! A row is one of three modes, chosen by static (`Tree 0`) selectors:
//! - sponge start (`s_init`): `inp = (blk, cap)` with `cap = [tag, len, ctr, 0, 0, 0, 0, 0]`;
//! - sponge continuation (`s_cont`): `inp = prev + (blk, 0)`, with `prev` consumed on the chain
//!   relation as `(in_id, inp[0..8] - blk, inp[8..16])`;
//! - Merkle compression (`s_cmp`): `blk` is the current digest, consumed on the chain relation;
//!   `d` is a boolean read from the memory bus; `inp = (blk, sibling)` if `d = 0`, else
//!   `(sibling, blk)`.
//!
//! The block `blk` comes from two `QM31` memory reads (`mem_id0`, `mem_id1`) or from a chain digest
//! (`blk_id`, consumed `blk_mult` times at once). The output `out = P(inp)` is produced on the chain
//! relation as a full state, a digest (`out[0..8]`) or a compression digest (`out[0..8] +
//! inp[0..8]`), with static multiplicities, and `out[0..4]`, `out[4..8]` can be yielded on the
//! memory bus. Every S-box is two columns (`x^2`, `x^5`), so all constraints have degree at most 3
//! and the component keeps expansion 1 like every other component of the outer STARK.
//!
//! Rows with all static columns zero (padding, blinding) still hold a valid permutation and
//! contribute nothing to any relation.

use num_traits::{One, Zero};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, RelationEntry,
};

use crate::blake2s_air::MemoryBusElements;
use crate::poseidon2_m31::{
    P2_HALF_FULL_ROUNDS, P2_INTERNAL_DIAG_SHIFTS, P2_RC_EXTERNAL_FINAL, P2_RC_EXTERNAL_INITIAL,
    P2_RC_INTERNAL, P2_SBOX_COLS,
};

relation!(Pos2ChainElements, 17);

pub const POS2_STATIC_COLS: usize = 21;
/// `inp` (16), `blk` (8), `d` (1), then the S-box columns.
pub const POS2_WITNESS_COLS: usize = 16 + 8 + 1 + P2_SBOX_COLS;

/// Static column indices.
pub mod st {
    pub const S_INIT: usize = 0;
    pub const S_CONT: usize = 1;
    pub const S_CMP: usize = 2;
    pub const CAP_TAG: usize = 3;
    pub const CAP_LEN: usize = 4;
    pub const CAP_CTR: usize = 5;
    pub const IN_ID: usize = 6;
    pub const BLK_ID: usize = 7;
    pub const BLK_MULT: usize = 8;
    pub const OUT_ID: usize = 9;
    pub const M_STATE: usize = 10;
    pub const M_DIG: usize = 11;
    pub const M_FF: usize = 12;
    pub const MEM_ID0: usize = 13;
    pub const MEM_MULT0: usize = 14;
    pub const MEM_ID1: usize = 15;
    pub const MEM_MULT1: usize = 16;
    pub const SQ_ID0: usize = 17;
    pub const SQ_MULT0: usize = 18;
    pub const SQ_ID1: usize = 19;
    pub const SQ_MULT1: usize = 20;
}

pub fn pos2_static_id(log_size: u32, col_idx: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("sparseprove_pos2_static_{log_size}_{col_idx}"),
    }
}

pub fn pos2_static_ids(log_size: u32) -> [PreProcessedColumnId; POS2_STATIC_COLS] {
    core::array::from_fn(|c| pos2_static_id(log_size, c))
}

fn m31(x: u32) -> BaseField {
    BaseField::from_u32_unchecked(x)
}

/// Constraint families, for the mutation tests (`mutation::SKIP` drops one in a test build).
pub mod family {
    pub const INIT_RATE: u8 = 1;
    pub const INIT_CAPACITY: u8 = 2;
    pub const DIR_BOOLEAN: u8 = 3;
    pub const CMP_SELECT: u8 = 4;
    pub const SBOX_SQUARE: u8 = 5;
    pub const SBOX_FIFTH: u8 = 6;
}

/// Test-only: the constraint family to drop on this thread (`None`: keep all).
#[cfg(test)]
pub mod mutation {
    thread_local! {
        pub static SKIP: core::cell::Cell<Option<u8>> = const { core::cell::Cell::new(None) };
    }
}

#[inline]
fn kept(f: u8) -> bool {
    #[cfg(test)]
    {
        mutation::SKIP.with(|s| s.get() != Some(f))
    }
    #[cfg(not(test))]
    {
        let _ = f;
        true
    }
}

/// `M4 = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]` (the addition chain of Plonky3's `apply_mat4`).
fn apply_m4<F: Clone + core::ops::Add<F, Output = F>>(x: [F; 4]) -> [F; 4] {
    let [x0, x1, x2, x3] = x;
    let t01 = x0.clone() + x1.clone();
    let t23 = x2.clone() + x3.clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x1;
    let t01233 = t0123 + x3;
    let y3 = t01233.clone() + x0.clone() + x0;
    let y1 = t01123.clone() + x2.clone() + x2;
    let y0 = t01123 + t01;
    let y2 = t01233 + t23;
    [y0, y1, y2, y3]
}

/// External linear layer `circ(2 M4, M4, M4, M4)` over any additive type.
pub fn external_layer_expr<F: Clone + core::ops::Add<F, Output = F>>(state: &mut [F; 16]) {
    for c in 0..4 {
        let y = apply_m4([
            state[4 * c].clone(),
            state[4 * c + 1].clone(),
            state[4 * c + 2].clone(),
            state[4 * c + 3].clone(),
        ]);
        for (k, v) in y.into_iter().enumerate() {
            state[4 * c + k] = v;
        }
    }
    let sums: [F; 4] = core::array::from_fn(|k| {
        state[k].clone() + state[4 + k].clone() + state[8 + k].clone() + state[12 + k].clone()
    });
    for (i, s) in state.iter_mut().enumerate() {
        *s = s.clone() + sums[i % 4].clone();
    }
}

/// Internal linear layer `1 + Diag(V)` over any type with `+`, `-` and scaling by `BaseField`.
pub fn internal_layer_expr<F>(state: &mut [F; 16])
where
    F: Clone
        + core::ops::Add<F, Output = F>
        + core::ops::Sub<F, Output = F>
        + core::ops::Mul<BaseField, Output = F>,
{
    let mut part = state[1].clone();
    for s in &state[2..] {
        part = part + s.clone();
    }
    let full = part.clone() + state[0].clone();
    state[0] = part - state[0].clone();
    for i in 1..16 {
        state[i] = full.clone() + state[i].clone() * m31(1u32 << P2_INTERNAL_DIAG_SHIFTS[i - 1]);
    }
}

pub type Pos2Component = FrameworkComponent<Pos2Eval>;

#[derive(Clone, Debug)]
pub struct Pos2Eval {
    pub log_size: u32,
    pub chain_elements: Pos2ChainElements,
    pub mem_bus_elements: MemoryBusElements,
    pub claimed_sum: SecureField,
}

/// One S-box `y = x^5` as two columns: `x2 = x^2` (degree 2) and `y = x2^2 * x` (degree 3).
fn sbox_constrained<E: EvalAtRow>(eval: &mut E, x: E::F) -> E::F {
    let x2 = eval.next_trace_mask();
    let y = eval.next_trace_mask();
    if kept(family::SBOX_SQUARE) {
        eval.add_constraint(x2.clone() - x.clone() * x.clone());
    }
    if kept(family::SBOX_FIFTH) {
        eval.add_constraint(y.clone() - x2.clone() * x2 * x);
    }
    y
}

/// The permutation's constraints on `inp`; returns `out = P(inp)` as linear expressions in the
/// last round's S-box columns.
fn permutation_constraints<E: EvalAtRow>(eval: &mut E, inp: &[E::F; 16]) -> [E::F; 16] {
    let mut state = inp.clone();
    external_layer_expr(&mut state);
    for r in 0..2 * P2_HALF_FULL_ROUNDS {
        let rc = if r < P2_HALF_FULL_ROUNDS {
            &P2_RC_EXTERNAL_INITIAL[r]
        } else {
            &P2_RC_EXTERNAL_FINAL[r - P2_HALF_FULL_ROUNDS]
        };
        if r == P2_HALF_FULL_ROUNDS {
            for &rc_i in &P2_RC_INTERNAL {
                let x = state[0].clone() + E::F::from(m31(rc_i));
                state[0] = sbox_constrained(eval, x);
                internal_layer_expr(&mut state);
            }
        }
        for i in 0..16 {
            let x = state[i].clone() + E::F::from(m31(rc[i]));
            state[i] = sbox_constrained(eval, x);
        }
        external_layer_expr(&mut state);
    }
    state
}

impl FrameworkEval for Pos2Eval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let s: [E::F; POS2_STATIC_COLS] =
            core::array::from_fn(|c| eval.get_preprocessed_column(pos2_static_id(self.log_size, c)));
        let inp: [E::F; 16] = core::array::from_fn(|_| eval.next_trace_mask());
        let blk: [E::F; 8] = core::array::from_fn(|_| eval.next_trace_mask());
        let d = eval.next_trace_mask();
        let zero = E::F::zero();

        // Sponge start: inp = (blk, tag, len, ctr, 0, 0, 0, 0, 0).
        if kept(family::INIT_RATE) {
            for i in 0..8 {
                eval.add_constraint(s[st::S_INIT].clone() * (inp[i].clone() - blk[i].clone()));
            }
        }
        if kept(family::INIT_CAPACITY) {
            let cap = [st::CAP_TAG, st::CAP_LEN, st::CAP_CTR];
            for k in 0..8 {
                let want = if k < 3 { s[cap[k]].clone() } else { zero.clone() };
                eval.add_constraint(s[st::S_INIT].clone() * (inp[8 + k].clone() - want));
            }
        }
        // Compression: d boolean; the child selected by d is the consumed digest blk.
        if kept(family::DIR_BOOLEAN) {
            eval.add_constraint(s[st::S_CMP].clone() * d.clone() * (d.clone() - E::F::one()));
        }
        if kept(family::CMP_SELECT) {
            for i in 0..8 {
                let sel = inp[i].clone() - blk[i].clone() + d.clone() * (inp[8 + i].clone() - inp[i].clone());
                eval.add_constraint(s[st::S_CMP].clone() * sel);
            }
        }

        let out = permutation_constraints(&mut eval, &inp);
        self.add_relations(&mut eval, &s, &inp, &blk, d, &out);
        eval.finalize_logup_in_pairs();
        eval
    }
}

impl Pos2Eval {
    fn add_relations<E: EvalAtRow>(
        &self,
        eval: &mut E,
        s: &[E::F; POS2_STATIC_COLS],
        inp: &[E::F; 16],
        blk: &[E::F; 8],
        d: E::F,
        out: &[E::F; 16],
    ) {
        let zero = E::F::zero();
        let chain = |id: &E::F, v: [E::F; 16]| -> Vec<E::F> {
            let mut t = Vec::with_capacity(17);
            t.push(id.clone());
            t.extend(v);
            t
        };
        let digest = |lo: [E::F; 8]| -> [E::F; 16] {
            core::array::from_fn(|i| if i < 8 { lo[i].clone() } else { zero.clone() })
        };
        let prev: [E::F; 16] = core::array::from_fn(|i| {
            if i < 8 { inp[i].clone() - blk[i].clone() } else { inp[i].clone() }
        });
        let out_lo: [E::F; 8] = core::array::from_fn(|i| out[i].clone());
        let ff: [E::F; 8] = core::array::from_fn(|i| out[i].clone() + inp[i].clone());
        let entries: [(E::EF, Vec<E::F>); 5] = [
            (E::EF::from(s[st::S_CONT].clone()), chain(&s[st::IN_ID], prev)),
            (E::EF::from(s[st::BLK_MULT].clone()), chain(&s[st::BLK_ID], digest(blk.clone()))),
            (-E::EF::from(s[st::M_STATE].clone()), chain(&s[st::OUT_ID], out.clone())),
            (-E::EF::from(s[st::M_DIG].clone()), chain(&s[st::OUT_ID], digest(out_lo))),
            (-E::EF::from(s[st::M_FF].clone()), chain(&s[st::OUT_ID], digest(ff))),
        ];
        for (mult, tuple) in entries {
            eval.add_to_relation(RelationEntry::new(&self.chain_elements, mult, &tuple));
        }
        let mem = |id: &E::F, v: [E::F; 4]| -> [E::F; 5] {
            [id.clone(), v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()]
        };
        let q = |a: &[E::F], o: usize| -> [E::F; 4] { core::array::from_fn(|i| a[o + i].clone()) };
        let mem_entries: [(E::EF, [E::F; 5]); 5] = [
            (E::EF::from(s[st::MEM_MULT0].clone()), mem(&s[st::MEM_ID0], q(blk, 0))),
            (E::EF::from(s[st::MEM_MULT1].clone()), mem(&s[st::MEM_ID1], q(blk, 4))),
            (
                E::EF::from(s[st::S_CMP].clone()),
                mem(&s[st::MEM_ID0], [d, zero.clone(), zero.clone(), zero.clone()]),
            ),
            (-E::EF::from(s[st::SQ_MULT0].clone()), mem(&s[st::SQ_ID0], q(out, 0))),
            (-E::EF::from(s[st::SQ_MULT1].clone()), mem(&s[st::SQ_ID1], q(out, 4))),
        ];
        for (mult, tuple) in mem_entries {
            eval.add_to_relation(RelationEntry::new(&self.mem_bus_elements, mult, &tuple));
        }
    }
}

pub fn pos2_info(log_size: u32) -> InfoEvaluator {
    let component = Pos2Eval {
        log_size,
        chain_elements: Pos2ChainElements::dummy(),
        mem_bus_elements: MemoryBusElements::dummy(),
        claimed_sum: SecureField::zero(),
    };
    component.evaluate(InfoEvaluator::empty())
}
