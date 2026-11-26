//! Compact, pure-stable-Rust (`wasm32`-compatible) RFC 7693 `Blake2s-256` AIR for
//! standalone receipts (`STWOREC2` and `STWOMDL1`).
//!
//! ### Architectural Highlights vs `stwo-examples::blake`
//! - **Zero nightly SIMD in verifier**: compiles cleanly on `wasm32-unknown-unknown` and stable Rust.
//! - **Eliminates `Xor12` (`-768` trace + LogUp columns)**: decomposes `rotr 12` 16-bit limbs into
//!   `(8, 4, 4)` non-overlapping bit slices (`255 + 256*15 + 4096*15 = 65535 < P`), verified via
//!   `Xor8` + two `Xor4` lookups.
//! - **Compact PreProcessed XOR Tables**: uses only `Xor9<9,1>`, `Xor8<8,0>`, `Xor7<7,0>`, and
//!   `Xor4<4,0>` (`12` `Tree 0` columns max `65,536` rows, `7` `Tree 1` multiplicity columns).
//! - **Full RFC 7693 `Blake2s-256` Compression**: enforces `IV`, `t_lo`, `f0 = is_last ? ~0 : 0`,
//!   10 `SIGMA`-permuted rounds, and finalization `h_out = h_in ^ v[0..8] ^ v[8..16]` directly over
//!   byte columns `(h_in_b[0..32], msg_b[0..64], t_b[0..4], is_last, h_out_b[0..32])`.

use num_traits::{One, Zero};
use stwo::core::channel::Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, RelationEntry,
};

pub const BLAKE2S_STATE_WORDS: usize = 16;
pub const BLAKE2S_MSG_WORDS: usize = 16;
pub const BLAKE2S_ROUNDS: usize = 10;

/// RFC 7693 Section 2.6 `Blake2s-256` Initialization Vector.
pub const BLAKE2S_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Unkeyed `Blake2s-256` parameter-block initial state `IV ^ [0x01010020, 0, .., 0]`.
pub const BLAKE2S_H_INIT: [u32; 8] = [
    0x6A09_E667 ^ 0x0101_0020,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// Little-endian byte representation of [`BLAKE2S_H_INIT`].
pub const BLAKE2S_H_INIT_BYTES: [u8; 32] = [
    0x47, 0xE6, 0x08, 0x6B, 0x85, 0xAE, 0x67, 0xBB, 0x72, 0xF3, 0x6E, 0x3C, 0x3A, 0xF5, 0x4F, 0xA5,
    0x7F, 0x52, 0x0E, 0x51, 0x8C, 0x68, 0x05, 0x9B, 0xAB, 0xD9, 0x83, 0x1F, 0x19, 0xCD, 0xE0, 0x5B,
];

/// RFC 7693 Section 2.7 message schedule permutations `SIGMA[0..10]`.
pub const BLAKE2S_SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Exact record of one RFC 7693 `Blake2s-256` 64-byte compression call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blake2sBlockRecord {
    pub h_in: [u8; 32],
    pub msg: [u8; 64],
    pub t_lo: u32,
    pub is_last: bool,
    pub h_out: [u8; 32],
}

#[inline]
fn blake2s_g_ref(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// Pure RFC 7693 single-block compression returning `(h_out_bytes, round_states[0..=10])`.
pub fn blake2s_compress_block(
    h_in: &[u8; 32],
    msg: &[u8; 64],
    t_lo: u32,
    is_last: bool,
) -> ([u8; 32], [[u32; 16]; 11]) {
    let mut h_words = [0u32; 8];
    for i in 0..8 {
        h_words[i] = u32::from_le_bytes([
            h_in[4 * i],
            h_in[4 * i + 1],
            h_in[4 * i + 2],
            h_in[4 * i + 3],
        ]);
    }
    let mut m_words = [0u32; 16];
    for i in 0..16 {
        m_words[i] = u32::from_le_bytes([
            msg[4 * i],
            msg[4 * i + 1],
            msg[4 * i + 2],
            msg[4 * i + 3],
        ]);
    }
    let mut v = [0u32; 16];
    v[..8].copy_from_slice(&h_words);
    v[8..16].copy_from_slice(&BLAKE2S_IV);
    v[12] ^= t_lo;
    if is_last {
        v[14] ^= 0xFFFF_FFFF;
    }

    let mut states = [[0u32; 16]; 11];
    states[0] = v;
    for r in 0..BLAKE2S_ROUNDS {
        let s = &BLAKE2S_SIGMA[r];
        blake2s_g_ref(&mut v, 0, 4, 8, 12, m_words[s[0]], m_words[s[1]]);
        blake2s_g_ref(&mut v, 1, 5, 9, 13, m_words[s[2]], m_words[s[3]]);
        blake2s_g_ref(&mut v, 2, 6, 10, 14, m_words[s[4]], m_words[s[5]]);
        blake2s_g_ref(&mut v, 3, 7, 11, 15, m_words[s[6]], m_words[s[7]]);
        blake2s_g_ref(&mut v, 0, 5, 10, 15, m_words[s[8]], m_words[s[9]]);
        blake2s_g_ref(&mut v, 1, 6, 11, 12, m_words[s[10]], m_words[s[11]]);
        blake2s_g_ref(&mut v, 2, 7, 8, 13, m_words[s[12]], m_words[s[13]]);
        blake2s_g_ref(&mut v, 3, 4, 9, 14, m_words[s[14]], m_words[s[15]]);
        states[r + 1] = v;
    }

    let mut h_out = [0u8; 32];
    for i in 0..8 {
        let w = h_words[i] ^ v[i] ^ v[i + 8];
        h_out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    (h_out, states)
}

/// Decompose an arbitrary byte slice (`<= u32::MAX` bytes) into exact RFC 7693 `Blake2s-256`
/// compression blocks starting from [`BLAKE2S_H_INIT_BYTES`].
pub fn blake2s_record_message(bytes: &[u8]) -> ([u8; 32], Vec<Blake2sBlockRecord>) {
    assert!(
        u64::try_from(bytes.len()).unwrap_or(u64::MAX) <= u64::from(u32::MAX),
        "Blake2s AIR message exceeds u32 byte counter"
    );
    if bytes.is_empty() {
        let msg = [0u8; 64];
        let (h_out, _) = blake2s_compress_block(&BLAKE2S_H_INIT_BYTES, &msg, 0, true);
        return (
            h_out,
            vec![Blake2sBlockRecord {
                h_in: BLAKE2S_H_INIT_BYTES,
                msg,
                t_lo: 0,
                is_last: true,
                h_out,
            }],
        );
    }
    let num_blocks = bytes.len().div_ceil(64);
    let mut records = Vec::with_capacity(num_blocks);
    let mut h_cur = BLAKE2S_H_INIT_BYTES;
    for blk_idx in 0..num_blocks {
        let start = blk_idx * 64;
        let end = (start + 64).min(bytes.len());
        let mut msg = [0u8; 64];
        msg[..end - start].copy_from_slice(&bytes[start..end]);
        let is_last = blk_idx + 1 == num_blocks;
        let t_lo = if is_last {
            bytes.len() as u32
        } else {
            ((blk_idx + 1) * 64) as u32
        };
        let (h_next, _) = blake2s_compress_block(&h_cur, &msg, t_lo, is_last);
        records.push(Blake2sBlockRecord {
            h_in: h_cur,
            msg,
            t_lo,
            is_last,
            h_out: h_next,
        });
        h_cur = h_next;
    }
    (h_cur, records)
}

/// Identifier and sizing metadata for a constant PreProcessed XOR lookup table column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blake2sXorTableSpec {
    pub elem_bits: u32,
    pub expand_bits: u32,
    pub col_index: usize,
}

impl Blake2sXorTableSpec {
    pub const fn new(elem_bits: u32, expand_bits: u32, col_index: usize) -> Self {
        Self {
            elem_bits,
            expand_bits,
            col_index,
        }
    }

    pub const fn limb_bits(&self) -> u32 {
        self.elem_bits - self.expand_bits
    }

    pub const fn column_bits(&self) -> u32 {
        2 * self.limb_bits()
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!(
                "sparseprove_blake2s_xor_{}_{}_{}",
                self.elem_bits, self.expand_bits, self.col_index
            ),
        }
    }
}

pub fn preprocessed_blake2s_xor_specs() -> [Blake2sXorTableSpec; 12] {
    [
        Blake2sXorTableSpec::new(9, 1, 0),
        Blake2sXorTableSpec::new(9, 1, 1),
        Blake2sXorTableSpec::new(9, 1, 2),
        Blake2sXorTableSpec::new(8, 0, 0),
        Blake2sXorTableSpec::new(8, 0, 1),
        Blake2sXorTableSpec::new(8, 0, 2),
        Blake2sXorTableSpec::new(7, 0, 0),
        Blake2sXorTableSpec::new(7, 0, 1),
        Blake2sXorTableSpec::new(7, 0, 2),
        Blake2sXorTableSpec::new(4, 0, 0),
        Blake2sXorTableSpec::new(4, 0, 1),
        Blake2sXorTableSpec::new(4, 0, 2),
    ]
}

pub fn preprocessed_blake2s_xor_ids() -> [PreProcessedColumnId; 12] {
    preprocessed_blake2s_xor_specs().map(|s| s.id())
}

pub const fn preprocessed_blake2s_xor_log_sizes() -> [u32; 12] {
    [16, 16, 16, 16, 16, 16, 14, 14, 14, 8, 8, 8]
}

relation!(XorElements9, 3);
relation!(XorElements8, 3);
relation!(XorElements7, 3);
relation!(XorElements4, 3);
relation!(Blake2sRoundElements, 96);
relation!(Blake2sCompressElements, 133);

#[derive(Clone, Debug, PartialEq)]
pub struct Blake2sXorElements {
    pub xor9: XorElements9,
    pub xor8: XorElements8,
    pub xor7: XorElements7,
    pub xor4: XorElements4,
}

impl Blake2sXorElements {
    pub fn draw(channel: &mut impl Channel) -> Self {
        Self {
            xor9: XorElements9::draw(channel),
            xor8: XorElements8::draw(channel),
            xor7: XorElements7::draw(channel),
            xor4: XorElements4::draw(channel),
        }
    }

    pub fn dummy() -> Self {
        Self {
            xor9: XorElements9::dummy(),
            xor8: XorElements8::dummy(),
            xor7: XorElements7::dummy(),
            xor4: XorElements4::dummy(),
        }
    }

    #[inline]
    pub fn add_xor<E: EvalAtRow>(&self, eval: &mut E, w: u32, a: E::F, b: E::F, c: E::F) {
        let vals = [a, b, c];
        match w {
            9 => eval.add_to_relation(RelationEntry::new(&self.xor9, E::EF::one(), &vals)),
            8 => eval.add_to_relation(RelationEntry::new(&self.xor8, E::EF::one(), &vals)),
            7 => eval.add_to_relation(RelationEntry::new(&self.xor7, E::EF::one(), &vals)),
            4 => eval.add_to_relation(RelationEntry::new(&self.xor4, E::EF::one(), &vals)),
            _ => unreachable!("unsupported Blake2s XOR width"),
        }
    }
}

#[derive(Clone, Debug)]
struct Fu32<F> {
    l: F,
    h: F,
}

macro_rules! define_blake2s_xor_table_component {
    ($eval_name:ident, $comp_name:ident, $info_fn:ident, $elems_ty:ty, $elem_bits:literal, $expand_bits:literal) => {
        pub type $comp_name = FrameworkComponent<$eval_name>;

        #[derive(Clone, Debug)]
        pub struct $eval_name {
            pub lookup_elements: $elems_ty,
            pub claimed_sum: SecureField,
        }

        impl FrameworkEval for $eval_name {
            fn log_size(&self) -> u32 {
                Blake2sXorTableSpec::new($elem_bits, $expand_bits, 0).column_bits()
            }

            fn max_constraint_log_degree_bound(&self) -> u32 {
                self.log_size() + 1
            }

            fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
                let limb_bits = Blake2sXorTableSpec::new($elem_bits, $expand_bits, 0).limb_bits();
                let al = eval.get_preprocessed_column(
                    Blake2sXorTableSpec::new($elem_bits, $expand_bits, 0).id(),
                );
                let bl = eval.get_preprocessed_column(
                    Blake2sXorTableSpec::new($elem_bits, $expand_bits, 1).id(),
                );
                let cl = eval.get_preprocessed_column(
                    Blake2sXorTableSpec::new($elem_bits, $expand_bits, 2).id(),
                );

                for idx in 0..(1usize << (2 * $expand_bits)) {
                    let hi_a = (idx >> $expand_bits) as u32;
                    let hi_b = (idx & ((1usize << $expand_bits) - 1)) as u32;
                    let mult = eval.next_trace_mask();
                    let a = al.clone() + E::F::from(BaseField::from_u32_unchecked(hi_a << limb_bits));
                    let b = bl.clone() + E::F::from(BaseField::from_u32_unchecked(hi_b << limb_bits));
                    let c = cl.clone()
                        + E::F::from(BaseField::from_u32_unchecked((hi_a ^ hi_b) << limb_bits));
                    eval.add_to_relation(RelationEntry::new(
                        &self.lookup_elements,
                        -E::EF::from(mult),
                        &[a, b, c],
                    ));
                }
                eval.finalize_logup_in_pairs();
                eval
            }
        }

        pub fn $info_fn() -> InfoEvaluator {
            let component = $eval_name {
                lookup_elements: <$elems_ty>::dummy(),
                claimed_sum: SecureField::zero(),
            };
            component.evaluate(InfoEvaluator::empty())
        }
    };
}

define_blake2s_xor_table_component!(
    Blake2sXor9Eval,
    Blake2sXor9Component,
    blake2s_xor9_info,
    XorElements9,
    9,
    1
);
define_blake2s_xor_table_component!(
    Blake2sXor8Eval,
    Blake2sXor8Component,
    blake2s_xor8_info,
    XorElements8,
    8,
    0
);
define_blake2s_xor_table_component!(
    Blake2sXor7Eval,
    Blake2sXor7Component,
    blake2s_xor7_info,
    XorElements7,
    7,
    0
);
define_blake2s_xor_table_component!(
    Blake2sXor4Eval,
    Blake2sXor4Component,
    blake2s_xor4_info,
    XorElements4,
    4,
    0
);

const INV16: BaseField = BaseField::from_u32_unchecked(1 << 15);
const TWO: BaseField = BaseField::from_u32_unchecked(2);

/// Ids of the outer circuit's constraints and checks
/// (`A01`..`A21` `StandaloneAuditAirEval`, `B01`..`B15` `Blake2sMemoryBridgeEval`, `C01`..`C04` `Blake2sSchedulerEval`,
/// `C05`..`C07` `Blake2sRoundEval`; `D01`..`D14` the in-circuit CPRS and LogUp-GKR checks the prover emits, with
/// `D05B` (the base-field check of the Phase-0 conjugate coordinates); `D15`..`D19` the
/// verifier's checks). Each id is a bit of the test-only mask of `outer_skip`; outside tests every constraint and check
/// is always kept.
pub mod outer_ids {
    pub const A01: u32 = 0;
    pub const A02: u32 = 1;
    pub const A03: u32 = 2;
    pub const A04: u32 = 3;
    pub const A05: u32 = 4;
    pub const A06: u32 = 5;
    pub const A07: u32 = 6;
    pub const A08: u32 = 7;
    pub const A09: u32 = 8;
    pub const A10: u32 = 9;
    pub const A11: u32 = 10;
    pub const A12: u32 = 11;
    pub const A13: u32 = 12;
    pub const A14: u32 = 13;
    pub const A15: u32 = 14;
    pub const A16: u32 = 15;
    pub const A17: u32 = 16;
    pub const A18: u32 = 17;
    pub const A19: u32 = 18;
    pub const A20: u32 = 19;
    pub const A21: u32 = 20;
    pub const B01: u32 = 21;
    pub const B02: u32 = 22;
    pub const B03: u32 = 23;
    pub const B04: u32 = 24;
    pub const B05: u32 = 25;
    pub const B06: u32 = 26;
    pub const B07: u32 = 27;
    pub const B08: u32 = 28;
    pub const B09: u32 = 29;
    pub const B10: u32 = 30;
    pub const B11: u32 = 31;
    pub const B12: u32 = 32;
    pub const B13: u32 = 33;
    pub const B14: u32 = 34;
    pub const B15: u32 = 35;
    pub const C01: u32 = 36;
    pub const C02: u32 = 37;
    pub const C03: u32 = 38;
    pub const C04: u32 = 39;
    pub const C05: u32 = 40;
    pub const C06: u32 = 41;
    pub const C07: u32 = 42;
    pub const D16: u32 = 43;
    pub const D03: u32 = 44;
    pub const D04: u32 = 45;
    pub const D01: u32 = 46;
    pub const D02: u32 = 47;
    pub const D05: u32 = 48;
    pub const D05B: u32 = 49;
    pub const D06: u32 = 50;
    pub const D07: u32 = 51;
    pub const D08: u32 = 52;
    pub const D09: u32 = 53;
    pub const D10: u32 = 54;
    pub const D11: u32 = 55;
    pub const D12: u32 = 56;
    pub const D13: u32 = 57;
    pub const D14: u32 = 58;
    pub const D15: u32 = 59;
    pub const D17: u32 = 60;
    pub const D18: u32 = 61;
    pub const D19: u32 = 62;
}

/// Test-only switch of the paired constraint tests (in the prover, not released): while a `outer_skip::Scope`
/// lives, every outer evaluator built on this thread drops the constraints whose `outer_ids` bit is set, and the
/// verifier on this thread skips the checks so marked. The evaluators keep the mask they were built with, so the
/// prover's and the verifier's rayon workers see it too.
#[cfg(test)]
pub mod outer_skip {
    use std::cell::Cell;

    thread_local! {
        static MASK: Cell<u64> = const { Cell::new(0) };
    }

    /// The mask of the current thread (`0`: every constraint kept).
    pub fn current() -> u64 {
        MASK.with(Cell::get)
    }

    /// Restores the previous mask on drop.
    pub struct Scope(u64);

    /// Drop the constraints of `ids` on this thread until the returned scope is dropped.
    pub fn scope(ids: &[u32]) -> Scope {
        let mask = ids.iter().fold(0u64, |m, &id| m | (1u64 << id));
        Scope(MASK.with(|m| m.replace(mask)))
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            MASK.with(|m| m.set(self.0));
        }
    }
}

/// `false` when a test dropped outer constraint or check `id` (mask `mask`); always `true` outside tests.
#[inline(always)]
pub fn outer_keep(mask: u64, id: u32) -> bool {
    (mask >> id) & 1 == 0
}

/// The current thread's test mask for a check the verifier runs on the calling thread (`0` outside tests).
#[inline(always)]
pub fn outer_thread_mask() -> u64 {
    #[cfg(test)]
    let mask = outer_skip::current();
    #[cfg(not(test))]
    let mask = 0u64;
    mask
}

pub type Blake2sRoundComponent = FrameworkComponent<Blake2sRoundEval>;

#[derive(Clone, Debug)]
pub struct Blake2sRoundEval {
    pub log_size: u32,
    pub xor_elements: Blake2sXorElements,
    pub round_elements: Blake2sRoundElements,
    pub claimed_sum: SecureField,
    /// Test-only mask of dropped constraints (`outer_skip`), captured when the evaluator is built.
    #[cfg(test)]
    pub skip: u64,
}

impl Blake2sRoundEval {
    /// `false` when a test built this evaluator with outer constraint `id` dropped (`outer_skip`).
    #[inline(always)]
    fn keep(&self, id: u32) -> bool {
        #[cfg(test)]
        let mask = self.skip;
        #[cfg(not(test))]
        let mask = 0u64;
        outer_keep(mask, id)
    }
}

impl FrameworkEval for Blake2sRoundEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real_round = eval.next_trace_mask();
        if self.keep(outer_ids::C05) {
            eval.add_constraint(is_real_round.clone() * (is_real_round.clone() - E::F::one()));
        }

        let mut v: [Fu32<E::F>; BLAKE2S_STATE_WORDS] =
            core::array::from_fn(|_| Self::next_u32(&mut eval));
        let input_v = v.clone();
        let m: [Fu32<E::F>; BLAKE2S_MSG_WORDS] =
            core::array::from_fn(|_| Self::next_u32(&mut eval));

        const G_IDX: [[usize; 4]; 8] = [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [1, 6, 11, 12],
            [2, 7, 8, 13],
            [3, 4, 9, 14],
        ];
        for (k, &[ia, ib, ic, id]) in G_IDX.iter().enumerate() {
            let (va, vb, vc, vd) = self.eval_g(
                &mut eval,
                v[ia].clone(),
                v[ib].clone(),
                v[ic].clone(),
                v[id].clone(),
                m[2 * k].clone(),
                m[2 * k + 1].clone(),
            );
            v[ia] = va;
            v[ib] = vb;
            v[ic] = vc;
            v[id] = vd;
        }

        let mut round_tuple = Vec::with_capacity(96);
        for w in &input_v {
            round_tuple.push(w.l.clone());
            round_tuple.push(w.h.clone());
        }
        for w in &v {
            round_tuple.push(w.l.clone());
            round_tuple.push(w.h.clone());
        }
        for w in &m {
            round_tuple.push(w.l.clone());
            round_tuple.push(w.h.clone());
        }
        eval.add_to_relation(RelationEntry::new(
            &self.round_elements,
            -E::EF::from(is_real_round),
            &round_tuple,
        ));
        eval.finalize_logup_in_pairs();
        eval
    }
}

impl Blake2sRoundEval {
    #[inline]
    fn next_u32<E: EvalAtRow>(eval: &mut E) -> Fu32<E::F> {
        let l = eval.next_trace_mask();
        let h = eval.next_trace_mask();
        Fu32 { l, h }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_g<E: EvalAtRow>(
        &self,
        eval: &mut E,
        mut a: Fu32<E::F>,
        mut b: Fu32<E::F>,
        mut c: Fu32<E::F>,
        mut d: Fu32<E::F>,
        m0: Fu32<E::F>,
        m1: Fu32<E::F>,
    ) -> (Fu32<E::F>, Fu32<E::F>, Fu32<E::F>, Fu32<E::F>) {
        let xor_elems = &self.xor_elements;
        let (keep2, keep3) = (self.keep(outer_ids::C06), self.keep(outer_ids::C07));
        a = Self::add3_u32_unchecked(eval, a, b.clone(), m0, keep3);
        d = Self::xor_rotr16_u32(eval, xor_elems, a.clone(), d);
        c = Self::add2_u32_unchecked(eval, c, d.clone(), keep2);
        b = Self::xor_rotr12_u32(eval, xor_elems, b, c.clone());
        a = Self::add3_u32_unchecked(eval, a, b.clone(), m1, keep3);
        d = Self::xor_rotr_u32(eval, xor_elems, a.clone(), d, 8);
        c = Self::add2_u32_unchecked(eval, c, d.clone(), keep2);
        b = Self::xor_rotr_u32(eval, xor_elems, b, c.clone(), 7);
        (a, b, c, d)
    }

    /// `keep`: `false` drops the two carry constraints (`C06`, test only).
    fn add2_u32_unchecked<E: EvalAtRow>(
        eval: &mut E,
        a: Fu32<E::F>,
        b: Fu32<E::F>,
        keep: bool,
    ) -> Fu32<E::F> {
        let sl = eval.next_trace_mask();
        let sh = eval.next_trace_mask();
        let carry_l = (a.l + b.l - sl.clone()) * E::F::from(INV16);
        if keep {
            eval.add_constraint(carry_l.clone() * carry_l.clone() - carry_l.clone());
        }
        let carry_h = (a.h + b.h + carry_l - sh.clone()) * E::F::from(INV16);
        if keep {
            eval.add_constraint(carry_h.clone() * carry_h.clone() - carry_h);
        }
        Fu32 { l: sl, h: sh }
    }

    /// `keep`: `false` drops the two carry constraints (`C07`, test only).
    fn add3_u32_unchecked<E: EvalAtRow>(
        eval: &mut E,
        a: Fu32<E::F>,
        b: Fu32<E::F>,
        c: Fu32<E::F>,
        keep: bool,
    ) -> Fu32<E::F> {
        let sl = eval.next_trace_mask();
        let sh = eval.next_trace_mask();
        let carry_l = (a.l + b.l + c.l - sl.clone()) * E::F::from(INV16);
        if keep {
            eval.add_constraint(
                carry_l.clone()
                    * (carry_l.clone() - E::F::one())
                    * (carry_l.clone() - E::F::from(TWO)),
            );
        }
        let carry_h = (a.h + b.h + c.h + carry_l - sh.clone()) * E::F::from(INV16);
        if keep {
            eval.add_constraint(
                carry_h.clone() * (carry_h.clone() - E::F::one()) * (carry_h - E::F::from(TWO)),
            );
        }
        Fu32 { l: sl, h: sh }
    }

    #[inline]
    fn split_unchecked<E: EvalAtRow>(eval: &mut E, val: E::F, r: u32) -> (E::F, E::F) {
        let hi = eval.next_trace_mask();
        let lo = val - hi.clone() * E::F::from(BaseField::from_u32_unchecked(1 << r));
        (lo, hi)
    }

    #[inline]
    fn split_8_4_4_unchecked<E: EvalAtRow>(eval: &mut E, val: E::F) -> (E::F, E::F, E::F) {
        let mid4 = eval.next_trace_mask();
        let hi4 = eval.next_trace_mask();
        let lo8 = val
            - mid4.clone() * E::F::from(BaseField::from_u32_unchecked(1 << 8))
            - hi4.clone() * E::F::from(BaseField::from_u32_unchecked(1 << 12));
        (lo8, mid4, hi4)
    }

    fn xor_rotr16_u32<E: EvalAtRow>(
        eval: &mut E,
        xor_elems: &Blake2sXorElements,
        a: Fu32<E::F>,
        b: Fu32<E::F>,
    ) -> Fu32<E::F> {
        let (all, alh) = Self::split_unchecked(eval, a.l, 8);
        let (ahl, ahh) = Self::split_unchecked(eval, a.h, 8);
        let (bll, blh) = Self::split_unchecked(eval, b.l, 8);
        let (bhl, bhh) = Self::split_unchecked(eval, b.h, 8);

        let xorll = eval.next_trace_mask();
        let xorhl = eval.next_trace_mask();
        let xorlh = eval.next_trace_mask();
        let xorhh = eval.next_trace_mask();
        xor_elems.add_xor(eval, 8, all, bll, xorll.clone());
        xor_elems.add_xor(eval, 8, ahl, bhl, xorhl.clone());
        xor_elems.add_xor(eval, 8, alh, blh, xorlh.clone());
        xor_elems.add_xor(eval, 8, ahh, bhh, xorhh.clone());

        let c256 = E::F::from(BaseField::from_u32_unchecked(1 << 8));
        Fu32 {
            l: xorhh * c256.clone() + xorhl,
            h: xorlh * c256 + xorll,
        }
    }

    /// Computes `(a ^ b).rotate_right(12)` and range-checks `a.{l,h}, b.{l,h} in [0, 2^16)`
    /// without `Xor12`, splitting each 16-bit limb into `(8, 4, 4)` bits (`Xor8` + two `Xor4`s).
    fn xor_rotr12_u32<E: EvalAtRow>(
        eval: &mut E,
        xor_elems: &Blake2sXorElements,
        a: Fu32<E::F>,
        b: Fu32<E::F>,
    ) -> Fu32<E::F> {
        let (al0, al1, al2) = Self::split_8_4_4_unchecked(eval, a.l);
        let (ah0, ah1, ah2) = Self::split_8_4_4_unchecked(eval, a.h);
        let (bl0, bl1, bl2) = Self::split_8_4_4_unchecked(eval, b.l);
        let (bh0, bh1, bh2) = Self::split_8_4_4_unchecked(eval, b.h);

        let xl0 = eval.next_trace_mask();
        let xh0 = eval.next_trace_mask();
        let xl1 = eval.next_trace_mask();
        let xh1 = eval.next_trace_mask();
        let xl2 = eval.next_trace_mask();
        let xh2 = eval.next_trace_mask();

        xor_elems.add_xor(eval, 8, al0, bl0, xl0.clone());
        xor_elems.add_xor(eval, 8, ah0, bh0, xh0.clone());
        xor_elems.add_xor(eval, 4, al1, bl1, xl1.clone());
        xor_elems.add_xor(eval, 4, ah1, bh1, xh1.clone());
        xor_elems.add_xor(eval, 4, al2, bl2, xl2.clone());
        xor_elems.add_xor(eval, 4, ah2, bh2, xh2.clone());

        let c16 = E::F::from(BaseField::from_u32_unchecked(1 << 4));
        let c4096 = E::F::from(BaseField::from_u32_unchecked(1 << 12));
        Fu32 {
            l: xl2 + xh0 * c16.clone() + xh1 * c4096.clone(),
            h: xh2 + xl0 * c16 + xl1 * c4096,
        }
    }

    fn xor_rotr_u32<E: EvalAtRow>(
        eval: &mut E,
        xor_elems: &Blake2sXorElements,
        a: Fu32<E::F>,
        b: Fu32<E::F>,
        r: u32,
    ) -> Fu32<E::F> {
        let (all, alh) = Self::split_unchecked(eval, a.l, r);
        let (ahl, ahh) = Self::split_unchecked(eval, a.h, r);
        let (bll, blh) = Self::split_unchecked(eval, b.l, r);
        let (bhl, bhh) = Self::split_unchecked(eval, b.h, r);

        let xorll = eval.next_trace_mask();
        let xorhl = eval.next_trace_mask();
        let xorlh = eval.next_trace_mask();
        let xorhh = eval.next_trace_mask();
        xor_elems.add_xor(eval, r, all, bll, xorll.clone());
        xor_elems.add_xor(eval, r, ahl, bhl, xorhl.clone());
        xor_elems.add_xor(eval, 16 - r, alh, blh, xorlh.clone());
        xor_elems.add_xor(eval, 16 - r, ahh, bhh, xorhh.clone());

        let shift = E::F::from(BaseField::from_u32_unchecked(1 << (16 - r)));
        Fu32 {
            l: xorhl * shift.clone() + xorlh,
            h: xorll * shift + xorhh,
        }
    }
}

/// Lookups per row into each XOR relation, the counts `StandaloneBlockStatement::validate` sums against `p`
/// before any proof is checked. Every lookup has multiplicity one on every row,
/// padding rows included; a bridge lookup is weighted by a static selector, so its count is an upper bound.
/// The test `outer_xor_lookups_per_row_match_the_air` recounts them with `InfoEvaluator`.
///
/// `Blake2sRoundEval`: 8 G functions, each `xor_rotr16` (4 `Xor8`), `xor_rotr12` (2 `Xor8`, 4 `Xor4`),
/// `xor_rotr(8)` (4 `Xor8`) and `xor_rotr(7)` (2 `Xor7`, 2 `Xor9`).
pub const ROUND_XOR8_PER_ROW: u64 = 80;
pub const ROUND_XOR7_PER_ROW: u64 = 16;
pub const ROUND_XOR9_PER_ROW: u64 = 16;
pub const ROUND_XOR4_PER_ROW: u64 = 32;
/// `Blake2sSchedulerEval`: 4 (`t` XOR `iv`) + 32 (message range pairs) + 64 (message byte pairs) `Xor8`.
pub const SCHED_XOR8_PER_ROW: u64 = 100;
/// `Blake2sMemoryBridgeEval`: 8 `Xor8` under `s_qm31_bytes` plus 16 under `s_pack_range_check`; 4 `Xor7`
/// under `s_qm31_bytes` plus 8 in the digest words.
pub const BRIDGE_XOR8_PER_ROW_MAX: u64 = 24;
pub const BRIDGE_XOR7_PER_ROW_MAX: u64 = 12;

pub fn blake2s_round_info() -> InfoEvaluator {
    let component = Blake2sRoundEval {
        log_size: 4,
        xor_elements: Blake2sXorElements::dummy(),
        round_elements: Blake2sRoundElements::dummy(),
        claimed_sum: SecureField::zero(),
        #[cfg(test)]
        skip: outer_skip::current(),
    };
    component.evaluate(InfoEvaluator::empty())
}

pub type Blake2sSchedulerComponent = FrameworkComponent<Blake2sSchedulerEval>;

#[derive(Clone, Debug)]
pub struct Blake2sSchedulerEval {
    pub log_size: u32,
    pub xor_elements: Blake2sXorElements,
    pub round_elements: Blake2sRoundElements,
    pub compress_elements: Blake2sCompressElements,
    pub claimed_sum: SecureField,
    /// Test-only mask of dropped constraints (`outer_skip`), captured when the evaluator is built.
    #[cfg(test)]
    pub skip: u64,
}

impl Blake2sSchedulerEval {
    /// `false` when a test built this evaluator with outer constraint `id` dropped (`outer_skip`).
    #[inline(always)]
    fn keep(&self, id: u32) -> bool {
        #[cfg(test)]
        let mask = self.skip;
        #[cfg(not(test))]
        let mask = 0u64;
        outer_keep(mask, id)
    }
}

impl FrameworkEval for Blake2sSchedulerEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real = eval.next_trace_mask();
        if self.keep(outer_ids::C01) {
            eval.add_constraint(is_real.clone() * (is_real.clone() - E::F::one()));
        }
        let is_last = eval.next_trace_mask();
        if self.keep(outer_ids::C02) {
            eval.add_constraint(is_last.clone() * (is_last.clone() - E::F::one()));
        }
        // On padded rows (`is_real == 0`), force `is_last == 0`.
        if self.keep(outer_ids::C03) {
            eval.add_constraint((E::F::one() - is_real.clone()) * is_last.clone());
        }

        let t_b: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let v0_12_b: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let iv4_bytes = BLAKE2S_IV[4].to_le_bytes();
        for k in 0..4 {
            let iv_f = E::F::from(BaseField::from_u32_unchecked(u32::from(iv4_bytes[k])));
            self.xor_elements
                .add_xor(&mut eval, 8, t_b[k].clone(), iv_f, v0_12_b[k].clone());
        }

        let h_in_b: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());
        let msg_b: [E::F; 64] = core::array::from_fn(|_| eval.next_trace_mask());
        let states_1_to_10: Vec<[Fu32<E::F>; BLAKE2S_STATE_WORDS]> = (0..BLAKE2S_ROUNDS)
            .map(|_| {
                core::array::from_fn(|_| Fu32 {
                    l: eval.next_trace_mask(),
                    h: eval.next_trace_mask(),
                })
            })
            .collect();
        let v10_b: [E::F; 64] = core::array::from_fn(|_| eval.next_trace_mask());
        let mid_xor_b: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());
        let h_out_b: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());
        let msg_pair_xor_b: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());

        let c256 = E::F::from(BaseField::from_u32_unchecked(256));

        // Range-check `msg_b[0..64]` to `[0, 256)` via 32 paired `Xor8` lookups.
        for k in 0..32 {
            self.xor_elements.add_xor(
                &mut eval,
                8,
                msg_b[2 * k].clone(),
                msg_b[2 * k + 1].clone(),
                msg_pair_xor_b[k].clone(),
            );
        }

        // Range-check `h_in_b[0..32]` and `v10_b[0..64]` and compute RFC 7693 finalization
        // `h_out_b[k] = h_in_b[k] ^ v10_b[k] ^ v10_b[k + 32]`.
        for k in 0..32 {
            self.xor_elements.add_xor(
                &mut eval,
                8,
                h_in_b[k].clone(),
                v10_b[k].clone(),
                mid_xor_b[k].clone(),
            );
            self.xor_elements.add_xor(
                &mut eval,
                8,
                mid_xor_b[k].clone(),
                v10_b[k + 32].clone(),
                h_out_b[k].clone(),
            );
        }

        // Bind `v10_b[0..64]` to `states_1_to_10[9]` (round 10 output state).
        for w in 0..BLAKE2S_STATE_WORDS {
            let l_from_b = v10_b[4 * w].clone() + v10_b[4 * w + 1].clone() * c256.clone();
            let h_from_b = v10_b[4 * w + 2].clone() + v10_b[4 * w + 3].clone() * c256.clone();
            if self.keep(outer_ids::C04) {
                eval.add_constraint(states_1_to_10[9][w].l.clone() - l_from_b);
                eval.add_constraint(states_1_to_10[9][w].h.clone() - h_from_b);
            }
        }

        // Construct `v_0` and `m_words` as 16-bit limbs.
        let m_words: [Fu32<E::F>; BLAKE2S_MSG_WORDS] = core::array::from_fn(|w| Fu32 {
            l: msg_b[4 * w].clone() + msg_b[4 * w + 1].clone() * c256.clone(),
            h: msg_b[4 * w + 2].clone() + msg_b[4 * w + 3].clone() * c256.clone(),
        });

        let iv6_l = BLAKE2S_IV[6] & 0xFFFF;
        let iv6_h = BLAKE2S_IV[6] >> 16;
        let v0: [Fu32<E::F>; BLAKE2S_STATE_WORDS] = core::array::from_fn(|w| {
            if w < 8 {
                Fu32 {
                    l: h_in_b[4 * w].clone() + h_in_b[4 * w + 1].clone() * c256.clone(),
                    h: h_in_b[4 * w + 2].clone() + h_in_b[4 * w + 3].clone() * c256.clone(),
                }
            } else if w == 12 {
                Fu32 {
                    l: v0_12_b[0].clone() + v0_12_b[1].clone() * c256.clone(),
                    h: v0_12_b[2].clone() + v0_12_b[3].clone() * c256.clone(),
                }
            } else if w == 14 {
                let one_minus_last = E::F::one() - is_last.clone();
                Fu32 {
                    l: E::F::from(BaseField::from_u32_unchecked(iv6_l)) * one_minus_last.clone()
                        + E::F::from(BaseField::from_u32_unchecked(0xFFFF - iv6_l))
                            * is_last.clone(),
                    h: E::F::from(BaseField::from_u32_unchecked(iv6_h)) * one_minus_last
                        + E::F::from(BaseField::from_u32_unchecked(0xFFFF - iv6_h))
                            * is_last.clone(),
                }
            } else {
                let iv_w = BLAKE2S_IV[w - 8];
                Fu32 {
                    l: E::F::from(BaseField::from_u32_unchecked(iv_w & 0xFFFF)),
                    h: E::F::from(BaseField::from_u32_unchecked(iv_w >> 16)),
                }
            }
        });

        // Demand all 10 rounds on `Blake2sRoundElements` with multiplicity `+is_real`.
        for r in 0..BLAKE2S_ROUNDS {
            let in_state = if r == 0 { &v0 } else { &states_1_to_10[r - 1] };
            let out_state = &states_1_to_10[r];
            let sigma = &BLAKE2S_SIGMA[r];
            let mut tuple = Vec::with_capacity(96);
            for w in in_state {
                tuple.push(w.l.clone());
                tuple.push(w.h.clone());
            }
            for w in out_state {
                tuple.push(w.l.clone());
                tuple.push(w.h.clone());
            }
            for &idx in sigma {
                tuple.push(m_words[idx].l.clone());
                tuple.push(m_words[idx].h.clone());
            }
            eval.add_to_relation(RelationEntry::new(
                &self.round_elements,
                E::EF::from(is_real.clone()),
                &tuple,
            ));
        }

        // Yield `(h_in_b, msg_b, t_b, is_last, h_out_b)` on `Blake2sCompressElements`.
        let mut comp_tuple = Vec::with_capacity(133);
        comp_tuple.extend_from_slice(&h_in_b);
        comp_tuple.extend_from_slice(&msg_b);
        comp_tuple.extend_from_slice(&t_b);
        comp_tuple.push(is_last);
        comp_tuple.extend_from_slice(&h_out_b);
        eval.add_to_relation(RelationEntry::new(
            &self.compress_elements,
            -E::EF::from(is_real),
            &comp_tuple,
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

pub fn blake2s_scheduler_info() -> InfoEvaluator {
    let component = Blake2sSchedulerEval {
        log_size: 4,
        xor_elements: Blake2sXorElements::dummy(),
        round_elements: Blake2sRoundElements::dummy(),
        compress_elements: Blake2sCompressElements::dummy(),
        claimed_sum: SecureField::zero(),
        #[cfg(test)]
        skip: outer_skip::current(),
    };
    component.evaluate(InfoEvaluator::empty())
}

relation!(MemoryBusElements, 5);
relation!(Blake2sChainElements, 33);
relation!(ByteBusElements, 2);

pub const BLAKE2S_BRIDGE_STATIC_COLS: usize = 29;
pub const BLAKE2S_BRIDGE_WITNESS_COLS: usize = 194;

pub fn blake2s_bridge_static_id(log_size: u32, col_idx: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("sparseprove_blake2s_bridge_{log_size}_{col_idx}"),
    }
}

pub fn blake2s_bridge_static_ids(log_size: u32) -> [PreProcessedColumnId; BLAKE2S_BRIDGE_STATIC_COLS] {
    core::array::from_fn(|col_idx| blake2s_bridge_static_id(log_size, col_idx))
}

pub type Blake2sMemoryBridgeComponent = FrameworkComponent<Blake2sMemoryBridgeEval>;

#[derive(Clone, Debug)]
pub struct Blake2sMemoryBridgeEval {
    pub log_size: u32,
    pub xor_elements: Blake2sXorElements,
    pub compress_elements: Blake2sCompressElements,
    pub chain_elements: Blake2sChainElements,
    pub byte_bus_elements: ByteBusElements,
    pub mem_bus_elements: MemoryBusElements,
    pub claimed_sum: SecureField,
    /// Test-only mask of dropped constraints (`outer_skip`), captured when the evaluator is built.
    #[cfg(test)]
    pub skip: u64,
}

impl Blake2sMemoryBridgeEval {
    /// `false` when a test built this evaluator with outer constraint `id` dropped (`outer_skip`).
    #[inline(always)]
    fn keep(&self, id: u32) -> bool {
        #[cfg(test)]
        let mask = self.skip;
        #[cfg(not(test))]
        let mask = 0u64;
        outer_keep(mask, id)
    }
}

impl FrameworkEval for Blake2sMemoryBridgeEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let st: [E::F; BLAKE2S_BRIDGE_STATIC_COLS] = core::array::from_fn(|c| {
            eval.get_preprocessed_column(blake2s_bridge_static_id(self.log_size, c))
        });
        let s_compress_gen = st[0].clone();
        let s_merkle_b0 = st[1].clone();
        let s_merkle_b1 = st[2].clone();
        let s_qm31_bytes = st[3].clone();
        let s_digest_to_bytes = st[4].clone();
        let s_digest_pack_mem = st[5].clone();
        let s_cprs_sq_qm = st[6].clone();
        let s_stwo_sq_qm = st[7].clone();
        let s_digest_bits32 = st[8].clone();
        let s_bytes8 = st[9].clone();
        let in_chain_id = st[10].clone();
        let in_chain_mult = st[11].clone();
        let out_chain_id = st[12].clone();
        let out_chain_mult = st[13].clone();
        let byte_base_addr = st[14].clone();
        let m_yield_0_8 = st[15].clone();
        let m_yield_8_16 = st[16].clone();
        let m_yield_16_32 = st[17].clone();
        let s_init_h0 = st[18].clone();
        let s_copy_chain_to_b = st[19].clone();
        let s_pack_range_check = st[20].clone();
        let k_par: [E::F; 8] = core::array::from_fn(|i| st[21 + i].clone());

        let b: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());
        let msg: [E::F; 64] = core::array::from_fn(|_| eval.next_trace_mask());
        let hout: [E::F; 32] = core::array::from_fn(|_| eval.next_trace_mask());
        let aux: [E::F; 64] = core::array::from_fn(|_| eval.next_trace_mask());
        let sc0 = eval.next_trace_mask();
        let sc1 = eval.next_trace_mask();

        // 1. Optional H_INIT initialization or copy from `aux[0..32]` (`chain_in`) into `b[0..32]`.
        for i in 0..32 {
            let h0_i = E::F::from(BaseField::from_u32_unchecked(u32::from(
                BLAKE2S_H_INIT_BYTES[i],
            )));
            if self.keep(outer_ids::B01) {
                eval.add_constraint(s_init_h0.clone() * (b[i].clone() - h0_i));
            }
            if self.keep(outer_ids::B02) {
                eval.add_constraint(s_copy_chain_to_b.clone() * (b[i].clone() - aux[i].clone()));
            }
        }

        // 2. `Blake2sChainElements` input (`aux[0..32]`) & output (`hout[0..32]`).
        {
            let mut in_tuple = Vec::with_capacity(33);
            in_tuple.push(in_chain_id.clone());
            in_tuple.extend_from_slice(&aux[..32]);
            eval.add_to_relation(RelationEntry::new(
                &self.chain_elements,
                E::EF::from(in_chain_mult),
                &in_tuple,
            ));

            let mut out_tuple = Vec::with_capacity(33);
            out_tuple.push(out_chain_id);
            out_tuple.extend_from_slice(&hout);
            eval.add_to_relation(RelationEntry::new(
                &self.chain_elements,
                -E::EF::from(out_chain_mult),
                &out_tuple,
            ));
        }

        // 3. Merkle Block 0 (`0x02 || left_32B || right_32B` bytes 0..64) & Block 1 (`byte 64, 0..0`).
        // On `s_merkle_b0`: `aux[0..32] = cur_b`, `aux[32..64] = sib_b`, `sc0 = dir_bit`, `sc1 = carry_b31`.
        if self.keep(outer_ids::B03) {
            eval.add_constraint(s_merkle_b0.clone() * sc0.clone() * (sc0.clone() - E::F::one()));
        }
        if self.keep(outer_ids::B04) {
            eval.add_constraint(
                s_merkle_b0.clone()
                    * (msg[0].clone() - E::F::from(BaseField::from_u32_unchecked(2))),
            );
        }
        for i in 0..32 {
            let left_i =
                aux[i].clone() + sc0.clone() * (aux[32 + i].clone() - aux[i].clone());
            if self.keep(outer_ids::B05) {
                eval.add_constraint(s_merkle_b0.clone() * (msg[1 + i].clone() - left_i));
            }
        }
        for i in 0..31 {
            let right_i =
                aux[32 + i].clone() + sc0.clone() * (aux[i].clone() - aux[32 + i].clone());
            if self.keep(outer_ids::B06) {
                eval.add_constraint(s_merkle_b0.clone() * (msg[33 + i].clone() - right_i));
            }
        }
        let right_31 = aux[63].clone() + sc0.clone() * (aux[31].clone() - aux[63].clone());
        if self.keep(outer_ids::B07) {
            eval.add_constraint(s_merkle_b0.clone() * (sc1.clone() - right_31));
        }
        let zero_f = E::F::zero();
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(s_merkle_b0.clone()),
            &[
                k_par[0].clone(),
                sc0,
                zero_f.clone(),
                zero_f.clone(),
                zero_f.clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.byte_bus_elements,
            -E::EF::from(s_merkle_b0.clone()),
            &[k_par[1].clone(), sc1],
        ));

        for i in 1..64 {
            if self.keep(outer_ids::B08) {
                eval.add_constraint(s_merkle_b1.clone() * msg[i].clone());
            }
        }

        // 4. `Blake2sCompressElements` call for `s_compress_gen + s_merkle_b0 + s_merkle_b1`.
        // On `s_merkle_b0`: `k_par[2..6] = [64, 0, 0, 0], k_par[6] = 0`.
        // On `s_merkle_b1`: `k_par[2..6] = [65, 0, 0, 0], k_par[6] = 1`.
        // On `s_compress_gen`: `k_par[2..6] = t_b[0..4], k_par[6] = is_last`.
        let s_any_compress =
            s_compress_gen.clone() + s_merkle_b0 + s_merkle_b1.clone();
        {
            let mut comp_tuple = Vec::with_capacity(133);
            comp_tuple.extend_from_slice(&b);
            comp_tuple.extend_from_slice(&msg);
            comp_tuple.extend_from_slice(&k_par[2..6]);
            comp_tuple.push(k_par[6].clone());
            comp_tuple.extend_from_slice(&hout);
            eval.add_to_relation(RelationEntry::new(
                &self.compress_elements,
                E::EF::from(s_any_compress),
                &comp_tuple,
            ));
        }

        // 5. `ByteBus` consumer (`msg[0..64]`) and producer (`b[0..32]`).
        let s_consume_b0 = s_compress_gen.clone() + s_merkle_b1;
        eval.add_to_relation(RelationEntry::new(
            &self.byte_bus_elements,
            E::EF::from(s_consume_b0),
            &[byte_base_addr.clone(), msg[0].clone()],
        ));
        for i in 1..64 {
            let addr_i =
                byte_base_addr.clone() + E::F::from(BaseField::from_u32_unchecked(i as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.byte_bus_elements,
                E::EF::from(s_compress_gen.clone()),
                &[addr_i, msg[i].clone()],
            ));
        }
        let s_contig_yield = s_qm31_bytes.clone() + s_digest_to_bytes.clone();
        for i in 0..8 {
            let addr_i =
                byte_base_addr.clone() + E::F::from(BaseField::from_u32_unchecked(i as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.byte_bus_elements,
                -E::EF::from(s_contig_yield.clone() * m_yield_0_8.clone()),
                &[addr_i, b[i].clone()],
            ));
        }
        for i in 8..16 {
            let addr_i =
                byte_base_addr.clone() + E::F::from(BaseField::from_u32_unchecked(i as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.byte_bus_elements,
                -E::EF::from(s_contig_yield.clone() * m_yield_8_16.clone()),
                &[addr_i, b[i].clone()],
            ));
        }
        for i in 16..32 {
            let addr_i =
                byte_base_addr.clone() + E::F::from(BaseField::from_u32_unchecked(i as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.byte_bus_elements,
                -E::EF::from(s_contig_yield.clone() * m_yield_16_32.clone()),
                &[addr_i, b[i].clone()],
            ));
        }

        // 6. `s_bytes8` (4-slot indexed byte supplier): `k_par[0..4]` = arbitrary byte addresses,
        // `k_par[4..8]` = constant byte values (enforced when `in_chain_id == 1`),
        // `[byte_base_addr, m_yield_0_8, m_yield_8_16, m_yield_16_32]` = per-slot multiplicities.
        let slot_mults = [byte_base_addr, m_yield_0_8, m_yield_8_16, m_yield_16_32];
        for i in 0..4 {
            if self.keep(outer_ids::B09) {
                eval.add_constraint(
                    s_bytes8.clone() * in_chain_id.clone() * (b[i].clone() - k_par[4 + i].clone()),
                );
            }
            eval.add_to_relation(RelationEntry::new(
                &self.byte_bus_elements,
                -E::EF::from(s_bytes8.clone() * slot_mults[i].clone()),
                &[k_par[i].clone(), b[i].clone()],
            ));
        }

        let c256 = E::F::from(BaseField::from_u32_unchecked(256));
        let c65536 = E::F::from(BaseField::from_u32_unchecked(65536));
        let c16m = E::F::from(BaseField::from_u32_unchecked(1 << 24));
        let c892 = E::F::from(BaseField::from_u32_unchecked(892));

        // 7. `s_qm31_bytes`: canonical `QM31` (`4 x M31 < P`) <-> `b[0..16]`.
        let mut qm_coords: [E::F; 4] = core::array::from_fn(|_| E::F::zero());
        for c in 0..4 {
            let b0 = b[4 * c].clone();
            let b1 = b[4 * c + 1].clone();
            let b2 = b[4 * c + 2].clone();
            let b3 = b[4 * c + 3].clone();
            let inv_non_p = msg[c].clone();
            let xor01 = msg[4 + c].clone();

            eval.add_to_relation(RelationEntry::new(
                &self.xor_elements.xor8,
                E::EF::from(s_qm31_bytes.clone()),
                &[b0.clone(), b1.clone(), xor01],
            ));
            eval.add_to_relation(RelationEntry::new(
                &self.xor_elements.xor8,
                E::EF::from(s_qm31_bytes.clone()),
                &[b2.clone(), zero_f.clone(), b2.clone()],
            ));
            eval.add_to_relation(RelationEntry::new(
                &self.xor_elements.xor7,
                E::EF::from(s_qm31_bytes.clone()),
                &[b3.clone(), zero_f.clone(), b3.clone()],
            ));
            let diff_p =
                c892.clone() - b0.clone() - b1.clone() - b2.clone() - b3.clone();
            if self.keep(outer_ids::B10) {
                eval.add_constraint(s_qm31_bytes.clone() * (diff_p * inv_non_p - E::F::one()));
            }
            qm_coords[c] = b0
                + b1 * c256.clone()
                + b2 * c65536.clone()
                + b3 * c16m.clone();
        }
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(s_qm31_bytes * k_par[1].clone()),
            &[
                k_par[0].clone(),
                qm_coords[0].clone(),
                qm_coords[1].clone(),
                qm_coords[2].clone(),
                qm_coords[3].clone(),
            ],
        ));

        // 8. `s_digest_pack_mem`: pack `b[0..32]` into 4 `QM31`s of 16-bit limbs on `MemoryBus`.
        for i in 0..16 {
            eval.add_to_relation(RelationEntry::new(
                &self.xor_elements.xor8,
                E::EF::from(s_pack_range_check.clone()),
                &[
                    b[2 * i].clone(),
                    b[2 * i + 1].clone(),
                    msg[8 + i].clone(),
                ],
            ));
        }
        for q in 0..4 {
            let limbs: [E::F; 4] = core::array::from_fn(|j| {
                let idx = 8 * q + 2 * j;
                b[idx].clone() + b[idx + 1].clone() * c256.clone()
            });
            eval.add_to_relation(RelationEntry::new(
                &self.mem_bus_elements,
                E::EF::from(s_digest_pack_mem.clone() * k_par[4 + q].clone()),
                &[
                    k_par[q].clone(),
                    limbs[0].clone(),
                    limbs[1].clone(),
                    limbs[2].clone(),
                    limbs[3].clone(),
                ],
            ));
        }

        // 9. `s_cprs_sq_qm`: CPRS `canon(u64)` reduction of `aux[0..32]` -> `QM31` on `MemoryBus`.
        let c2 = E::F::from(BaseField::from_u32_unchecked(2));
        let c512 = E::F::from(BaseField::from_u32_unchecked(512));
        let c131k = E::F::from(BaseField::from_u32_unchecked(131_072));
        let c33m = E::F::from(BaseField::from_u32_unchecked(1 << 25));
        let cprs_sq: [E::F; 4] = core::array::from_fn(|c| {
            let o = 8 * c;
            aux[o].clone()
                + aux[o + 1].clone() * c256.clone()
                + aux[o + 2].clone() * c65536.clone()
                + aux[o + 3].clone() * c16m.clone()
                + aux[o + 4].clone() * c2.clone()
                + aux[o + 5].clone() * c512.clone()
                + aux[o + 6].clone() * c131k.clone()
                + aux[o + 7].clone() * c33m.clone()
        });
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            -E::EF::from(s_cprs_sq_qm * k_par[7].clone()),
            &[
                k_par[0].clone(),
                cprs_sq[0].clone(),
                cprs_sq[1].clone(),
                cprs_sq[2].clone(),
                cprs_sq[3].clone(),
            ],
        ));

        // 10. `s_stwo_sq_qm`: Stwo `Blake2sChannel` `< 2*P` check + reduction on `aux[0..32]`.
        let mut stwo_felts: [E::F; 8] = core::array::from_fn(|_| E::F::zero());
        for w in 0..8 {
            let bit0 = msg[24 + w].clone();
            let hi7 = msg[32 + w].clone();
            let inv_lt = msg[40 + w].clone();
            if self.keep(outer_ids::B11) {
                eval.add_constraint(
                    s_stwo_sq_qm.clone() * bit0.clone() * (bit0.clone() - E::F::one()),
                );
            }
            if self.keep(outer_ids::B12) {
                eval.add_constraint(
                    s_stwo_sq_qm.clone()
                        * (aux[4 * w].clone() - (bit0 + hi7.clone() * c2.clone())),
                );
            }
            eval.add_to_relation(RelationEntry::new(
                &self.xor_elements.xor7,
                E::EF::from(s_stwo_sq_qm.clone()),
                &[hi7.clone(), zero_f.clone(), hi7.clone()],
            ));
            let diff_2p = c892.clone()
                - hi7
                - aux[4 * w + 1].clone()
                - aux[4 * w + 2].clone()
                - aux[4 * w + 3].clone();
            if self.keep(outer_ids::B13) {
                eval.add_constraint(s_stwo_sq_qm.clone() * (diff_2p * inv_lt - E::F::one()));
            }
            stwo_felts[w] = aux[4 * w].clone()
                + aux[4 * w + 1].clone() * c256.clone()
                + aux[4 * w + 2].clone() * c65536.clone()
                + aux[4 * w + 3].clone() * c16m.clone();
        }
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            -E::EF::from(s_stwo_sq_qm.clone() * k_par[2].clone()),
            &[
                k_par[0].clone(),
                stwo_felts[0].clone(),
                stwo_felts[1].clone(),
                stwo_felts[2].clone(),
                stwo_felts[3].clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            -E::EF::from(s_stwo_sq_qm * k_par[3].clone()),
            &[
                k_par[1].clone(),
                stwo_felts[4].clone(),
                stwo_felts[5].clone(),
                stwo_felts[6].clone(),
                stwo_felts[7].clone(),
            ],
        ));

        // 11. `s_digest_bits32`: decompose `aux[0..4]` into 32 boolean bits `aux[32..64]`.
        for byte_i in 0..4 {
            let mut acc_b = E::F::zero();
            for bit_j in 0..8 {
                let bit_val = aux[32 + 8 * byte_i + bit_j].clone();
                if self.keep(outer_ids::B14) {
                    eval.add_constraint(
                        s_digest_bits32.clone() * bit_val.clone() * (bit_val.clone() - E::F::one()),
                    );
                }
                acc_b += bit_val * E::F::from(BaseField::from_u32_unchecked(1u32 << bit_j));
            }
            if self.keep(outer_ids::B15) {
                eval.add_constraint(s_digest_bits32.clone() * (aux[byte_i].clone() - acc_b));
            }
        }
        for bit_idx in 0..32 {
            let var_id =
                k_par[0].clone() + E::F::from(BaseField::from_u32_unchecked(bit_idx as u32));
            eval.add_to_relation(RelationEntry::new(
                &self.mem_bus_elements,
                -E::EF::from(s_digest_bits32.clone() * k_par[2].clone()),
                &[
                    var_id,
                    aux[32 + bit_idx].clone(),
                    zero_f.clone(),
                    zero_f.clone(),
                    zero_f.clone(),
                ],
            ));
        }

        let _ = s_digest_to_bytes;
        eval.finalize_logup_in_pairs();
        eval
    }
}

pub fn blake2s_bridge_info(log_size: u32) -> InfoEvaluator {
    let component = Blake2sMemoryBridgeEval {
        log_size,
        xor_elements: Blake2sXorElements::dummy(),
        compress_elements: Blake2sCompressElements::dummy(),
        chain_elements: Blake2sChainElements::dummy(),
        byte_bus_elements: ByteBusElements::dummy(),
        mem_bus_elements: MemoryBusElements::dummy(),
        claimed_sum: SecureField::zero(),
        #[cfg(test)]
        skip: outer_skip::current(),
    };
    component.evaluate(InfoEvaluator::empty())
}
