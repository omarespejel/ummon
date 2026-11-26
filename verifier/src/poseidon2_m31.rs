//! Poseidon2 over M31, width 16, for the EXPERIMENTAL `exp_pos2_*` profiles (not production-ready).
//!
//! Instance: S-box `x^5`, 8 full rounds (4 + 4), 14 partial rounds, external layer
//! `circ(2 M4, M4, M4, M4)`, internal layer `1 + Diag(V)` with
//! `V = [-2, 2^0, 2^1, ..., 2^8, 2^10, 2^12, 2^13, 2^14, 2^15, 2^16]`. Round constants copied
//! verbatim from Plonky3 `p3-mersenne-31` 0.8.0 `src/poseidon2.rs` (SHA-256 `4331699e...be2720`),
//! which states that they come from the Grain LFSR (`field_type=1, alpha=5, n=31, t=16, R_F=8,
//! R_P=14`). The test `plonky3_known_answer` checks that file's known answer.
//!
//! Modes (all inputs are canonical field elements `< P`):
//! - [`hash`]: sponge, rate 8, capacity 8. The capacity starts as `cap` (a tag, the message length
//!   and a counter); messages have a length fixed by the caller, so there is no padding rule. At
//!   least one permutation is applied. Output: the first 8 elements.
//! - [`compress`]: `first 8 of (P(l || r) + (l || r))` for Merkle nodes.
//!
//! Self-contained (no dependency) so that the verifier crate, the CPRS library and the prover can
//! all mount this one file.

pub const P2_P: u32 = (1u32 << 31) - 1;
pub const P2_WIDTH: usize = 16;
pub const P2_RATE: usize = 8;
pub const P2_HALF_FULL_ROUNDS: usize = 4;
pub const P2_PARTIAL_ROUNDS: usize = 14;

pub const P2_RC_EXTERNAL_INITIAL: [[u32; 16]; 4] = [
    [
        0x768bab52, 0x70e0ab7d, 0x3d266c8a, 0x6da42045, 0x600fef22, 0x41dace6b, 0x64f9bdd4,
        0x5d42d4fe, 0x76b1516d, 0x6fc9a717, 0x70ac4fb6, 0x00194ef6, 0x22b644e2, 0x1f7916d5,
        0x47581be2, 0x2710a123,
    ],
    [
        0x6284e867, 0x018d3afe, 0x5df99ef3, 0x4c1e467b, 0x566f6abc, 0x2994e427, 0x538a6d42,
        0x5d7bf2cf, 0x7fda2dab, 0x0fd854c4, 0x46922fca, 0x3d7763a1, 0x19fd05ca, 0x0a4bbb43,
        0x15075851, 0x3d903d76,
    ],
    [
        0x2d290ff7, 0x40809fa0, 0x59dac6ec, 0x127927a2, 0x6bbf0ea0, 0x0294140f, 0x24742976,
        0x6e84c081, 0x22484f4a, 0x354cae59, 0x0453ffe1, 0x3f47a3cc, 0x0088204e, 0x6066e109,
        0x3b7c4b80, 0x6b55665d,
    ],
    [
        0x3bc4b897, 0x735bf378, 0x508daf42, 0x1884fc2b, 0x7214f24c, 0x7498be0a, 0x1a60e640,
        0x3303f928, 0x29b46376, 0x5c96bb68, 0x65d097a5, 0x1d358e9f, 0x4a9a9017, 0x4724cf76,
        0x347af70f, 0x1e77e59a,
    ],
];

pub const P2_RC_EXTERNAL_FINAL: [[u32; 16]; 4] = [
    [
        0x57090613, 0x1fa42108, 0x17bbef50, 0x1ff7e11c, 0x047b24ca, 0x4e140275, 0x4fa086f5,
        0x079b309c, 0x1159bd47, 0x6d37e4e5, 0x075d8dce, 0x12121ca0, 0x7f6a7c40, 0x68e182ba,
        0x5493201b, 0x0444a80e,
    ],
    [
        0x0064f4c6, 0x6467abe6, 0x66975762, 0x2af68f9b, 0x345b33be, 0x1b70d47f, 0x053db717,
        0x381189cb, 0x43b915f8, 0x20df3694, 0x0f459d26, 0x77a0e97b, 0x2f73e739, 0x1876c2f9,
        0x65a0e29a, 0x4cabefbe,
    ],
    [
        0x5abd1268, 0x4d34a760, 0x12771799, 0x69a0c9ac, 0x39091e55, 0x7f611cd0, 0x3af055da,
        0x7ac0bbdf, 0x6e0f3a24, 0x41e3b6f7, 0x49b3756d, 0x568bc538, 0x20c079d8, 0x1701c72c,
        0x7670dc6c, 0x5a439035,
    ],
    [
        0x7c93e00e, 0x561fbb4d, 0x1178907b, 0x02737406, 0x32fb24f1, 0x6323b60a, 0x6ab12418,
        0x42c99cea, 0x155a0b97, 0x53d1c6aa, 0x2bd20347, 0x279b3d73, 0x4f5f3c70, 0x0245af6c,
        0x238359d3, 0x49966a59,
    ],
];

pub const P2_RC_INTERNAL: [u32; 14] = [
    0x7f7ec4bf, 0x0421926f, 0x5198e669, 0x34db3148, 0x4368bafd, 0x66685c7f, 0x78d3249a, 0x60187881,
    0x76dad67a, 0x0690b437, 0x1ea95311, 0x40e5369a, 0x38f103fc, 0x1d226a21,
];

/// Diagonal `V` of the internal matrix `1 + Diag(V)`, entries `1..16` as powers of two
/// (entry 0 is `-2`).
pub const P2_INTERNAL_DIAG_SHIFTS: [u32; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

#[inline]
fn reduce64(x: u64) -> u32 {
    let lo = (x & P2_P as u64) + (x >> 31);
    let lo = (lo & P2_P as u64) + (lo >> 31);
    let v = lo as u32;
    if v >= P2_P {
        v - P2_P
    } else {
        v
    }
}

#[inline]
pub fn p2_add(a: u32, b: u32) -> u32 {
    reduce64(a as u64 + b as u64)
}

#[inline]
pub fn p2_mul(a: u32, b: u32) -> u32 {
    reduce64(a as u64 * b as u64)
}

#[inline]
fn neg(a: u32) -> u32 {
    if a == 0 {
        0
    } else {
        P2_P - a
    }
}

#[inline]
fn sbox(x: u32) -> u32 {
    let x2 = p2_mul(x, x);
    p2_mul(p2_mul(x2, x2), x)
}

/// `M4` of the external layer: `[[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]`.
#[inline]
fn apply_m4(x: [u32; 4]) -> [u32; 4] {
    let lin = |c: [u64; 4]| reduce64(c[0] * x[0] as u64 + c[1] * x[1] as u64 + c[2] * x[2] as u64 + c[3] * x[3] as u64);
    [lin([2, 3, 1, 1]), lin([1, 2, 3, 1]), lin([1, 1, 2, 3]), lin([3, 1, 1, 2])]
}

/// External linear layer `circ(2 M4, M4, M4, M4)`.
pub fn external_layer(state: &mut [u32; 16]) {
    for c in 0..4 {
        let y = apply_m4([state[4 * c], state[4 * c + 1], state[4 * c + 2], state[4 * c + 3]]);
        state[4 * c..4 * c + 4].copy_from_slice(&y);
    }
    let sums: [u64; 4] = core::array::from_fn(|k| (0..4).map(|j| state[4 * j + k] as u64).sum());
    for (i, s) in state.iter_mut().enumerate() {
        *s = reduce64(*s as u64 + sums[i % 4]);
    }
}

/// Internal linear layer `1 + Diag(V)`.
pub fn internal_layer(state: &mut [u32; 16]) {
    let part: u64 = state[1..].iter().map(|&x| x as u64).sum();
    let full = part + state[0] as u64;
    state[0] = reduce64(part + neg(state[0]) as u64);
    for i in 1..16 {
        state[i] = reduce64(full + ((state[i] as u64) << P2_INTERNAL_DIAG_SHIFTS[i - 1]));
    }
}

/// The Poseidon2 permutation (Plonky3 order: external layer first, then 4 full, 14 partial and
/// 4 full rounds; each full round adds constants, applies the S-box to every lane, then the
/// external layer).
pub fn permute(state: &mut [u32; 16]) {
    external_layer(state);
    for rc in &P2_RC_EXTERNAL_INITIAL {
        for i in 0..16 {
            state[i] = sbox(p2_add(state[i], rc[i]));
        }
        external_layer(state);
    }
    for &rc in &P2_RC_INTERNAL {
        state[0] = sbox(p2_add(state[0], rc));
        internal_layer(state);
    }
    for rc in &P2_RC_EXTERNAL_FINAL {
        for i in 0..16 {
            state[i] = sbox(p2_add(state[i], rc[i]));
        }
        external_layer(state);
    }
}

/// Columns the AIR commits per permutation for the S-boxes: `x^2` and `x^5` of every lane of the
/// 8 full rounds and of lane 0 of the 14 partial rounds.
pub const P2_SBOX_COLS: usize = 2 * (2 * P2_HALF_FULL_ROUNDS * P2_WIDTH + P2_PARTIAL_ROUNDS);

/// The permutation, also returning its S-box columns in AIR order (round by round, lane by lane,
/// `x^2` then `x^5`).
pub fn permute_with_trace(state: &mut [u32; 16], cols: &mut Vec<u32>) {
    fn sbox_cols(x: u32, cols: &mut Vec<u32>) -> u32 {
        let x2 = p2_mul(x, x);
        let y = p2_mul(p2_mul(x2, x2), x);
        cols.push(x2);
        cols.push(y);
        y
    }
    cols.clear();
    external_layer(state);
    for rc in &P2_RC_EXTERNAL_INITIAL {
        for i in 0..16 {
            state[i] = sbox_cols(p2_add(state[i], rc[i]), cols);
        }
        external_layer(state);
    }
    for &rc in &P2_RC_INTERNAL {
        state[0] = sbox_cols(p2_add(state[0], rc), cols);
        internal_layer(state);
    }
    for rc in &P2_RC_EXTERNAL_FINAL {
        for i in 0..16 {
            state[i] = sbox_cols(p2_add(state[i], rc[i]), cols);
        }
        external_layer(state);
    }
    debug_assert_eq!(cols.len(), P2_SBOX_COLS);
}

/// Number of permutations of [`hash`] for a message of `len` elements.
#[inline]
pub fn hash_blocks(len: usize) -> usize {
    len.div_ceil(P2_RATE).max(1)
}

/// Capacity of a hash: `[tag, message length, counter, 0, 0, 0, 0, 0]`.
#[inline]
pub fn capacity(tag: u32, len: usize, ctr: u32) -> [u32; 8] {
    let len = u32::try_from(len).expect("Poseidon2 message length fits a u32");
    assert!(tag < P2_P && len < P2_P && ctr < P2_P, "Poseidon2 capacity values must be < p");
    [tag, len, ctr, 0, 0, 0, 0, 0]
}

/// Sponge hash with rate 8 and the given initial capacity; the message length is the caller's.
pub fn hash(cap: [u32; 8], msg: &[u32]) -> [u32; 8] {
    debug_assert!(msg.iter().all(|&x| x < P2_P));
    let mut state = [0u32; 16];
    state[8..].copy_from_slice(&cap);
    for b in 0..hash_blocks(msg.len()) {
        for i in 0..P2_RATE {
            if let Some(&x) = msg.get(P2_RATE * b + i) {
                state[i] = p2_add(state[i], x);
            }
        }
        permute(&mut state);
    }
    core::array::from_fn(|i| state[i])
}

/// 2-to-1 compression `first 8 of (P(l || r) + (l || r))`.
pub fn compress(l: &[u32; 8], r: &[u32; 8]) -> [u32; 8] {
    let mut state = [0u32; 16];
    state[..8].copy_from_slice(l);
    state[8..].copy_from_slice(r);
    permute(&mut state);
    core::array::from_fn(|i| p2_add(state[i], l[i]))
}

/// A digest as 32 bytes: 8 little-endian words, each `< P`.
pub fn digest_to_bytes(d: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in d.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// Inverse of [`digest_to_bytes`]; `None` when a word is `>= P`.
pub fn digest_from_bytes(b: &[u8; 32]) -> Option<[u32; 8]> {
    let mut out = [0u32; 8];
    for (i, w) in out.iter_mut().enumerate() {
        *w = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        if *w >= P2_P {
            return None;
        }
    }
    Some(out)
}

/// Roles of the hashes of the experimental profile (first capacity element is
/// `role << 16 | label`). One role per use; labels separate the steps inside a transcript.
pub mod role {
    pub const LEAF: u32 = 1;
    pub const TR_INIT: u32 = 3;
    pub const TR_ABSORB: u32 = 4;
    pub const TR_SQUEEZE: u32 = 5;
    pub const POW: u32 = 6;
    pub const COMMIT: u32 = 7;
    pub const EXEC_INIT: u32 = 16;
    pub const EXEC_ABSORB: u32 = 17;
    pub const EXEC_SQUEEZE: u32 = 18;
    pub const WEIGHTS_DIGEST: u32 = 19;
    pub const FINAL_CHAIN: u32 = 20;
    pub const PUBLIC_IO: u32 = 21;
}

/// `role << 16 | label` with `label < 2^16`.
#[inline]
pub fn tag(role: u32, label: u32) -> u32 {
    assert!(role < (1 << 14) && label < (1 << 16), "Poseidon2 tag out of range");
    (role << 16) | label
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plonky3_known_answer() {
        // `test_default_mersenne31_poseidon2_width_16` of the Plonky3 file named in the header.
        let mut s: [u32; 16] = core::array::from_fn(|i| i as u32);
        permute(&mut s);
        assert_eq!(
            s,
            [
                0x0b2c803a, 0x5b1ee4d1, 0x49c6b1e3, 0x2cdc280c, 0x310a60c8, 0x530a729e, 0x4e61bcb4,
                0x2e84d3c3, 0x58709c08, 0x7e82ac42, 0x2162bcef, 0x6d153ab6, 0x742cf0e3, 0x2f21632d,
                0x61adce1e, 0x1973d6f1,
            ]
        );
    }

    #[test]
    fn permute_with_trace_matches_permute() {
        let mut a: [u32; 16] = core::array::from_fn(|i| 3 * i as u32 + 5);
        let mut b = a;
        let mut cols = Vec::new();
        permute(&mut a);
        permute_with_trace(&mut b, &mut cols);
        assert_eq!(a, b);
        assert_eq!(cols.len(), P2_SBOX_COLS);
    }

    #[test]
    fn hash_binds_capacity_length_and_every_element() {
        let msg: [u32; 12] = core::array::from_fn(|i| 1000 + i as u32);
        let base = hash(capacity(tag(role::LEAF, 0), 12, 0), &msg);
        assert_ne!(base, hash(capacity(tag(role::LEAF, 1), 12, 0), &msg));
        assert_ne!(base, hash(capacity(tag(role::LEAF, 0), 12, 1), &msg));
        assert_ne!(base, hash(capacity(tag(role::LEAF, 0), 16, 0), &msg));
        for i in 0..12 {
            let mut m2 = msg;
            m2[i] = p2_add(m2[i], 1);
            assert_ne!(base, hash(capacity(tag(role::LEAF, 0), 12, 0), &m2));
        }
        // An empty message still permutes once.
        assert_ne!(hash(capacity(tag(role::TR_INIT, 0), 0, 0), &[]), [0u32; 8]);
    }

    #[test]
    fn compress_feeds_forward_and_orders_children() {
        let l: [u32; 8] = core::array::from_fn(|i| 7 * i as u32 + 1);
        let r: [u32; 8] = core::array::from_fn(|i| 11 * i as u32 + 3);
        let mut s = [0u32; 16];
        s[..8].copy_from_slice(&l);
        s[8..].copy_from_slice(&r);
        permute(&mut s);
        let want: [u32; 8] = core::array::from_fn(|i| p2_add(s[i], l[i]));
        assert_eq!(compress(&l, &r), want);
        assert_ne!(compress(&l, &r), compress(&r, &l));
    }

    #[test]
    fn digest_bytes_round_trip_and_reject_non_canonical() {
        let d: [u32; 8] = core::array::from_fn(|i| P2_P - 1 - i as u32);
        assert_eq!(digest_from_bytes(&digest_to_bytes(&d)), Some(d));
        let mut b = digest_to_bytes(&d);
        b[0..4].copy_from_slice(&P2_P.to_le_bytes());
        assert_eq!(digest_from_bytes(&b), None);
    }

    #[test]
    fn field_ops_match_u64_reference() {
        let samples = [0u32, 1, 2, 5, 65535, 1 << 30, P2_P - 2, P2_P - 1, 0x1234_5678 % P2_P];
        for &a in &samples {
            for &b in &samples {
                assert_eq!(p2_add(a, b) as u64, (a as u64 + b as u64) % P2_P as u64);
                assert_eq!(p2_mul(a, b) as u64, (a as u64 * b as u64) % P2_P as u64);
            }
        }
    }
}
