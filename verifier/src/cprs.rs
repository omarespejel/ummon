// Copyright 2026 StarkWare Industries Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Standalone CPRS v1 verifier (sequential, `wasm32`-compatible, no prover or GPU dependencies).

use std::fmt;
use std::sync::OnceLock;

use blake2::{Blake2s256, Digest};
use stwo::core::fields::qm31::SecureField;

pub type Hash = [u8; 32];

pub const P: u32 = 0x7FFF_FFFF;
pub const P64: u64 = P as u64;
pub const TWO_ADICITY: usize = 33;

#[inline]
pub fn to_cprs_qm31(value: SecureField) -> QM31 {
    let [a, b, c, d] = value.to_m31_array();
    QM31::new(a.0, b.0, c.0, d.0)
}

#[inline]
pub fn stwo_point_to_cprs(point: &[SecureField]) -> Vec<QM31> {
    point.iter().rev().copied().map(to_cprs_qm31).collect()
}

#[inline]
pub fn i8_to_canonical_m31_u32(value: i8) -> u32 {
    let v = i32::from(value);
    (v + ((v >> 31) & (P as i32))) as u32
}

#[inline(always)]
pub fn add31(a: u32, b: u32) -> u32 {
    let t = a + b;
    t.min(t.wrapping_sub(P))
}

#[inline(always)]
pub fn sub31(a: u32, b: u32) -> u32 {
    let t = a.wrapping_sub(b);
    t.min(t.wrapping_add(P))
}

#[inline(always)]
pub fn neg31(a: u32) -> u32 {
    let t = P - a;
    t.min(t.wrapping_sub(P))
}

#[inline(always)]
pub fn fold(x: u64) -> u64 {
    (x & P64) + (x >> 31)
}

#[inline(always)]
pub fn canon(x: u64) -> u32 {
    let a = fold(fold(x)) as u32;
    a.min(a.wrapping_sub(P))
}

#[inline(always)]
pub fn canon128(x: u128) -> u32 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    canon(fold(lo) + 4 * fold(hi))
}

#[inline(always)]
pub fn mul31(a: u32, b: u32) -> u32 {
    canon(a as u64 * b as u64)
}

pub fn pow31(mut a: u32, mut e: u64) -> u32 {
    let mut r = 1u32;
    while e > 0 {
        if e & 1 == 1 {
            r = mul31(r, a);
        }
        a = mul31(a, a);
        e >>= 1;
    }
    r
}

pub fn inv31(a: u32) -> u32 {
    assert!(a != 0, "inverse of zero in M31");
    pow31(a, (P - 2) as u64)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct QM31 {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

impl fmt::Debug for QM31 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QM({}, {}, {}, {})", self.a, self.b, self.c, self.d)
    }
}

#[inline]
pub fn pack_view(w: &[u32]) -> &[QM31] {
    assert!(w.len() % 4 == 0, "weight length must be a multiple of 4");
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const QM31, w.len() / 4) }
}

pub fn weights_canonical(w: &[u32]) -> bool {
    w.iter().all(|&x| x < P)
}

impl QM31 {
    pub const ZERO: QM31 = QM31 { a: 0, b: 0, c: 0, d: 0 };
    pub const ONE: QM31 = QM31 { a: 1, b: 0, c: 0, d: 0 };

    #[inline(always)]
    pub const fn new(a: u32, b: u32, c: u32, d: u32) -> QM31 {
        QM31 { a, b, c, d }
    }

    #[inline(always)]
    pub const fn is_canonical(&self) -> bool {
        self.a < P && self.b < P && self.c < P && self.d < P
    }

    #[inline]
    pub fn from_u64s(v: [u64; 4]) -> QM31 {
        QM31 {
            a: (v[0] % P64) as u32,
            b: (v[1] % P64) as u32,
            c: (v[2] % P64) as u32,
            d: (v[3] % P64) as u32,
        }
    }

    #[inline(always)]
    pub const fn from_m31(a: u32) -> QM31 {
        QM31 { a, b: 0, c: 0, d: 0 }
    }

    #[inline(always)]
    pub fn coords(self) -> [u32; 4] {
        [self.a, self.b, self.c, self.d]
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self == QM31::ZERO
    }

    #[inline(always)]
    pub fn add(self, o: QM31) -> QM31 {
        QM31 {
            a: add31(self.a, o.a),
            b: add31(self.b, o.b),
            c: add31(self.c, o.c),
            d: add31(self.d, o.d),
        }
    }

    #[inline(always)]
    pub fn sub(self, o: QM31) -> QM31 {
        QM31 {
            a: sub31(self.a, o.a),
            b: sub31(self.b, o.b),
            c: sub31(self.c, o.c),
            d: sub31(self.d, o.d),
        }
    }

    #[inline(always)]
    pub fn neg(self) -> QM31 {
        QM31 {
            a: neg31(self.a),
            b: neg31(self.b),
            c: neg31(self.c),
            d: neg31(self.d),
        }
    }

    #[inline(always)]
    pub fn mul_m31(self, s: u32) -> QM31 {
        QM31 {
            a: mul31(self.a, s),
            b: mul31(self.b, s),
            c: mul31(self.c, s),
            d: mul31(self.d, s),
        }
    }

    #[inline(always)]
    pub fn mul(self, y: QM31) -> QM31 {
        let (a0, a1, a2, a3) = (self.a as u64, self.b as u64, self.c as u64, self.d as u64);
        let (b0, b1, b2, b3) = (y.a as u64, y.b as u64, y.c as u64, y.d as u64);
        let (n1, n2, n3) = (P64 - b1, P64 - b2, P64 - b3);
        let s0a = a0 * b0 + a1 * n1 + a2 * n3 + a3 * n2;
        let s0b = (a2 * b2 + a3 * n3) << 1;
        let s1a = a0 * b1 + a1 * b0 + a2 * b2 + a3 * n3;
        let s1b = (a2 * b3 + a3 * b2) << 1;
        let s2 = a0 * b2 + a1 * n3 + a2 * b0 + a3 * n1;
        let s3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        QM31 {
            a: canon(fold(s0a) + fold(s0b)),
            b: canon(fold(s1a) + fold(s1b)),
            c: canon(s2),
            d: canon(s3),
        }
    }

    pub fn pow(self, mut e: u128) -> QM31 {
        let mut r = QM31::ONE;
        let mut b = self;
        while e > 0 {
            if e & 1 == 1 {
                r = r.mul(b);
            }
            b = b.mul(b);
            e >>= 1;
        }
        r
    }

    pub fn inv(self) -> QM31 {
        assert!(!self.is_zero(), "inverse of zero in QM31");
        let p = P as u128;
        let p2 = p * p;
        self.pow(p2 * p2 - 2)
    }

    #[inline]
    pub fn frob(self, k: usize) -> QM31 {
        match k & 3 {
            0 => self,
            k => apply_m31_matrix(&consts().frob[k], self),
        }
    }

    #[inline]
    pub fn frob_inv(self, i: usize) -> QM31 {
        self.frob((4 - (i & 3)) & 3)
    }

    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 16] {
        let mut o = [0u8; 16];
        o[0..4].copy_from_slice(&self.a.to_le_bytes());
        o[4..8].copy_from_slice(&self.b.to_le_bytes());
        o[8..12].copy_from_slice(&self.c.to_le_bytes());
        o[12..16].copy_from_slice(&self.d.to_le_bytes());
        o
    }

    #[inline]
    pub fn from_bytes(b: &[u8]) -> Option<QM31> {
        if b.len() != 16 {
            return None;
        }
        let a = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let bb = u32::from_le_bytes(b[4..8].try_into().unwrap());
        let c = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let d = u32::from_le_bytes(b[12..16].try_into().unwrap());
        if a >= P || bb >= P || c >= P || d >= P {
            None
        } else {
            Some(QM31 { a, b: bb, c, d })
        }
    }
}

#[inline(always)]
pub fn mul_acc(acc: &mut [u128; 4], x: QM31, y: QM31) {
    let (a0, a1, a2, a3) = (x.a as u64, x.b as u64, x.c as u64, x.d as u64);
    let (b0, b1, b2, b3) = (y.a as u64, y.b as u64, y.c as u64, y.d as u64);
    let (n1, n2, n3) = (P64 - b1, P64 - b2, P64 - b3);
    let s0a = a0 * b0 + a1 * n1 + a2 * n3 + a3 * n2;
    let s0b = (a2 * b2 + a3 * n3) << 1;
    let s1a = a0 * b1 + a1 * b0 + a2 * b2 + a3 * n3;
    let s1b = (a2 * b3 + a3 * b2) << 1;
    let s2 = a0 * b2 + a1 * n3 + a2 * b0 + a3 * n1;
    let s3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
    acc[0] += s0a as u128 + s0b as u128;
    acc[1] += s1a as u128 + s1b as u128;
    acc[2] += s2 as u128;
    acc[3] += s3 as u128;
}

#[inline(always)]
pub fn wide_finish(acc: &[u128; 4]) -> QM31 {
    QM31 {
        a: canon128(acc[0]),
        b: canon128(acc[1]),
        c: canon128(acc[2]),
        d: canon128(acc[3]),
    }
}

#[inline(always)]
pub fn apply_m31_matrix(m: &[[u32; 4]; 4], x: QM31) -> QM31 {
    let v = [x.a as u64, x.b as u64, x.c as u64, x.d as u64];
    let row = |r: &[u32; 4]| -> u32 {
        canon(r[0] as u64 * v[0] + r[1] as u64 * v[1] + r[2] as u64 * v[2] + r[3] as u64 * v[3])
    };
    QM31 {
        a: row(&m[0]),
        b: row(&m[1]),
        c: row(&m[2]),
        d: row(&m[3]),
    }
}

pub struct Consts {
    pub frob: [[[u32; 4]; 4]; 4],
    pub moore_inv: [[QM31; 4]; 4],
    pub generator: QM31,
    pub roots: Vec<QM31>,
    pub inv2: u32,
}

pub const BETA: [QM31; 4] = [
    QM31::new(1, 0, 0, 0),
    QM31::new(0, 1, 0, 0),
    QM31::new(0, 0, 1, 0),
    QM31::new(0, 0, 0, 1),
];

static CONSTS: OnceLock<Consts> = OnceLock::new();

pub fn consts() -> &'static Consts {
    CONSTS.get_or_init(build_consts)
}

fn build_consts() -> Consts {
    let p = P as u128;
    let mut frob = [[[0u32; 4]; 4]; 4];
    let mut pk: u128 = 1;
    for k in 0..4 {
        for col in 0..4 {
            let img = BETA[col].pow(pk);
            let co = img.coords();
            for row in 0..4 {
                frob[k][row][col] = co[row];
            }
        }
        pk *= p;
    }
    let col_vec = |k: usize, j: usize| -> QM31 {
        QM31::new(frob[k][0][j], frob[k][1][j], frob[k][2][j], frob[k][3][j])
    };
    let mut m = [[QM31::ZERO; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            m[j][i] = col_vec(i, j);
        }
    }
    let minv = invert4(&m);
    let cofactor = (p * p * p * p - 1) >> TWO_ADICITY;
    let mut k = 0u32;
    let generator = loop {
        let c = QM31::new(k, 1, 1, 0);
        let g = c.pow(cofactor);
        let mut t = g;
        for _ in 0..(TWO_ADICITY - 1) {
            t = t.mul(t);
        }
        if t != QM31::ONE {
            break g;
        }
        k += 1;
    };
    let mut roots = vec![QM31::ONE; TWO_ADICITY + 1];
    roots[TWO_ADICITY] = generator;
    for log in (0..TWO_ADICITY).rev() {
        roots[log] = roots[log + 1].mul(roots[log + 1]);
    }
    Consts {
        frob,
        moore_inv: minv,
        generator,
        roots,
        inv2: inv31(2),
    }
}

fn invert4(m: &[[QM31; 4]; 4]) -> [[QM31; 4]; 4] {
    let mut a = [[QM31::ZERO; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m[r][c];
        }
        a[r][4 + r] = QM31::ONE;
    }
    for c in 0..4 {
        let piv = (c..4).find(|&r| !a[r][c].is_zero()).expect("singular Moore matrix");
        a.swap(c, piv);
        let inv = a[c][c].inv();
        for x in a[c].iter_mut() {
            *x = x.mul(inv);
        }
        for r in 0..4 {
            if r != c && !a[r][c].is_zero() {
                let f = a[r][c];
                let rowc = a[c];
                for k in 0..8 {
                    a[r][k] = a[r][k].sub(f.mul(rowc[k]));
                }
            }
        }
    }
    let mut out = [[QM31::ZERO; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = a[r][4 + c];
        }
    }
    out
}

#[inline]
pub fn root_of_unity(log: usize) -> QM31 {
    assert!(log <= TWO_ADICITY);
    consts().roots[log]
}

#[inline]
pub fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS as usize - bits)
    }
}

pub const ROLE_LEAF: u8 = 1;
pub const ROLE_NODE: u8 = 2;
pub const ROLE_INIT: u8 = 3;
pub const ROLE_ABSORB: u8 = 4;
pub const ROLE_SQUEEZE: u8 = 5;
pub const ROLE_POW: u8 = 6;
pub const ROLE_COMMIT: u8 = 7;

#[inline]
pub fn h_role(role: u8, parts: &[&[u8]]) -> Hash {
    let mut h = Blake2s256::new();
    h.update([role]);
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

#[inline]
pub fn leaf_hash(symbols: &[QM31]) -> Hash {
    let mut h = Blake2s256::new();
    h.update([ROLE_LEAF]);
    for s in symbols {
        h.update(s.to_bytes());
    }
    h.finalize().into()
}

#[inline]
pub fn node_hash(l: &Hash, r: &Hash) -> Hash {
    let mut buf = [0u8; 65];
    buf[0] = ROLE_NODE;
    buf[1..33].copy_from_slice(l);
    buf[33..65].copy_from_slice(r);
    Blake2s256::digest(buf).into()
}

pub fn commitment_digest(n: u8, s: u8, f: u8, g: u8, depth: u8, root: &Hash) -> Hash {
    h_role(ROLE_COMMIT, &[&[n, s, f, g, depth], root])
}

#[inline]
pub fn pow_ok(state: &Hash, nonce: u64, pow_bits: usize) -> bool {
    if pow_bits == 0 {
        return true;
    }
    let d = h_role(ROLE_POW, &[state, &nonce.to_le_bytes()]);
    let mut zeros = 0usize;
    for b in d {
        if b == 0 {
            zeros += 8;
        } else {
            zeros += b.leading_zeros() as usize;
            break;
        }
        if zeros >= pow_bits {
            break;
        }
    }
    zeros >= pow_bits
}

#[inline]
pub fn verify_path(root: &Hash, idx: usize, leaf_digest: Hash, path: &[Hash], depth: usize) -> bool {
    if path.len() != depth || (depth < usize::BITS as usize && (idx >> depth) != 0) {
        return false;
    }
    let mut cur = leaf_digest;
    let mut i = idx;
    for sib in path {
        cur = if i & 1 == 0 { node_hash(&cur, sib) } else { node_hash(sib, &cur) };
        i >>= 1;
    }
    &cur == root
}

pub fn merkle_root_from_leaves(mut level: Vec<Hash>) -> Hash {
    assert!(level.len().is_power_of_two(), "number of leaves must be a power of two");
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|p| node_hash(&p[0], &p[1]))
            .collect();
    }
    level[0]
}

pub const MAX_LOG_N: usize = 32;
pub const MAX_POW_BITS: usize = 40;
pub const MAX_L: u64 = u32::MAX as u64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    pub(crate) n: usize,
    pub(crate) s: usize,
    pub(crate) f: usize,
    pub(crate) q: usize,
    pub(crate) pow_bits: usize,
    pub(crate) g: usize,
    pub(crate) m: usize,
    pub(crate) log_n: usize,
    pub(crate) lo_vars: usize,
    _seal: (),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamError(pub &'static str);

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid parameters: {}", self.0)
    }
}

impl Params {
    pub fn try_new(
        n: usize,
        s: usize,
        f: usize,
        q: usize,
        pow_bits: usize,
        g: usize,
    ) -> Result<Params, ParamError> {
        if !(3..=255).contains(&n) {
            return Err(ParamError("n must be in 3..=255"));
        }
        let m = n - 2;
        if s < 1 {
            return Err(ParamError("s must be >= 1"));
        }
        if !(1..=m).contains(&f) {
            return Err(ParamError("f must be in 1..=m = n - 2"));
        }
        if !(1..=255).contains(&g) {
            return Err(ParamError("g must be in 1..=255"));
        }
        if !(1..=u32::MAX as u64).contains(&(q as u64)) {
            return Err(ParamError("q must be in 1..=2^32 - 1"));
        }
        if pow_bits > MAX_POW_BITS {
            return Err(ParamError("pow_bits must be <= 40"));
        }
        let log_n = match m.checked_add(s) {
            Some(t) if t <= MAX_LOG_N => t,
            _ => return Err(ParamError("evaluation domain larger than 2^32")),
        };
        Ok(Params {
            n,
            s,
            f,
            q,
            pow_bits,
            g,
            m,
            log_n,
            lo_vars: 2,
            _seal: (),
        })
    }

    pub fn with_lo_vars(&self, lo_vars: usize) -> Params {
        assert!(lo_vars == 0 || lo_vars == 2, "lo_vars must be 0 or 2");
        let mut p = self.clone();
        p.lo_vars = lo_vars;
        p
    }

    #[inline]
    pub fn lo_vars(&self) -> usize {
        self.lo_vars
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
    #[inline]
    pub fn s(&self) -> usize {
        self.s
    }
    #[inline]
    pub fn f(&self) -> usize {
        self.f
    }
    #[inline]
    pub fn q(&self) -> usize {
        self.q
    }
    #[inline]
    pub fn pow_bits(&self) -> usize {
        self.pow_bits
    }
    #[inline]
    pub fn g(&self) -> usize {
        self.g
    }
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }
    #[inline]
    pub fn log_n(&self) -> usize {
        self.log_n
    }

    pub fn fold_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut left = self.m;
        while left > 0 {
            let x = self.f.min(left);
            sizes.push(x);
            left -= x;
        }
        sizes
    }

    pub fn shape(&self) -> Shape {
        Shape::new(self)
    }

    pub fn depth(&self) -> usize {
        self.log_n - self.f.min(self.m)
    }

    pub fn commitment(&self, root: &Hash) -> Hash {
        let b = |x: usize| u8::try_from(x).expect("Params invariant: one-byte header field");
        commitment_digest(b(self.n), b(self.s), b(self.f), b(self.g), b(self.depth()), root)
    }
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub sizes: Vec<usize>,
    pub log_sizes: Vec<usize>,
}

impl Shape {
    fn new(p: &Params) -> Shape {
        let sizes = p.fold_sizes();
        let mut log_sizes = vec![p.log_n];
        for &f in &sizes {
            let last = *log_sizes.last().unwrap();
            log_sizes.push(last - f);
        }
        Shape { sizes, log_sizes }
    }

    pub fn inner(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn reg_leaf_log(&self) -> usize {
        self.sizes[0]
    }
    pub fn reg_depth(&self) -> usize {
        self.log_sizes[0] - self.sizes[0]
    }
    pub fn inner_leaf_log(&self, r: usize) -> usize {
        self.sizes[r + 1]
    }
    pub fn inner_depth(&self, r: usize) -> usize {
        self.log_sizes[r + 1] - self.sizes[r + 1]
    }
}

pub const TAG_HDR: &[u8] = b"hdr";
pub const TAG_COM: &[u8] = b"com";
pub const TAG_PT: &[u8] = b"pt";
pub const TAG_V: &[u8] = b"v";
pub const TAG_Y: &[u8] = b"y";
pub const TAG_A: &[u8] = b"A";
pub const TAG_Z: &[u8] = b"z";
pub const TAG_B: &[u8] = b"B";
pub const TAG_ORACLE: &[u8] = b"oracle";
pub const TAG_FINAL: &[u8] = b"final";
pub const TAG_NONCE: &[u8] = b"nonce";
pub const SQ_GAMMA: &[u8] = b"gamma";
pub const SQ_RHOA: &[u8] = b"rhoA";
pub const SQ_MU: &[u8] = b"mu";
pub const SQ_ALPHA: &[u8] = b"alpha";
pub const SQ_QUERY: &[u8] = b"query";

pub fn mu_tag(a: usize) -> [u8; 6] {
    let mut t = [0u8; 6];
    t[..2].copy_from_slice(SQ_MU);
    t[2..].copy_from_slice(&(a as u32).to_le_bytes());
    t
}

pub struct Transcript {
    state: Hash,
    ctr: u64,
}

impl Transcript {
    pub fn new() -> Transcript {
        Transcript {
            state: h_role(ROLE_INIT, &[]),
            ctr: 0,
        }
    }

    pub fn state(&self) -> &Hash {
        &self.state
    }

    pub fn absorb(&mut self, tag: &[u8], data: &[u8]) {
        let len = (data.len() as u64).to_le_bytes();
        self.state = h_role(ROLE_ABSORB, &[&self.state, tag, &len, data]);
        self.ctr = 0;
    }

    pub fn absorb_qm(&mut self, tag: &[u8], xs: &[QM31]) {
        let mut buf = Vec::with_capacity(16 * xs.len());
        for x in xs {
            buf.extend_from_slice(&x.to_bytes());
        }
        self.absorb(tag, &buf);
    }

    fn squeeze_bytes(&mut self, tag: &[u8]) -> Hash {
        let ctr = self.ctr;
        let out = h_role(ROLE_SQUEEZE, &[&self.state, tag, &ctr.to_le_bytes()]);
        self.ctr += 1;
        out
    }

    pub fn squeeze_qm(&mut self, tag: &[u8]) -> QM31 {
        let b = self.squeeze_bytes(tag);
        let mut limbs = [0u64; 4];
        for k in 0..4 {
            limbs[k] = u64::from_le_bytes(b[8 * k..8 * k + 8].try_into().unwrap());
        }
        QM31::from_u64s(limbs)
    }

    pub fn squeeze_index(&mut self, tag: &[u8], log_range: usize) -> u64 {
        let b = self.squeeze_bytes(tag);
        let v = u64::from_le_bytes(b[0..8].try_into().unwrap());
        if log_range >= 64 {
            v
        } else {
            v & ((1u64 << log_range) - 1)
        }
    }
}

pub struct FoldTables {
    pub f: usize,
    pub levels: Vec<Vec<QM31>>,
    pub scale: u32,
}

impl FoldTables {
    pub fn new(alphas: &[QM31]) -> FoldTables {
        let f = alphas.len();
        let mut levels = Vec::with_capacity(f);
        for u in 0..f {
            let size = 1usize << (f - u - 1);
            let winv = root_of_unity(f - u).inv();
            let mut tab = Vec::with_capacity(size);
            let mut cur = alphas[u];
            for _ in 0..size {
                tab.push(cur);
                cur = cur.mul(winv);
            }
            levels.push(tab);
        }
        FoldTables {
            f,
            levels,
            scale: inv31(pow31(2, f as u64)),
        }
    }
}

#[inline]
pub fn fold_coset(v: &mut [QM31], mut xinv: QM31, ft: &FoldTables) -> QM31 {
    let f = ft.f;
    let mut len = 1usize << f;
    for u in 0..f {
        let half = len >> 1;
        let tab = &ft.levels[u];
        for s in 0..half {
            let (v0, v1) = (v[s], v[s + half]);
            let sum = v0.add(v1);
            let diff = v0.sub(v1);
            v[s] = sum.add(diff.mul(tab[s]).mul(xinv));
        }
        len = half;
        xinv = xinv.mul(xinv);
    }
    v[0].mul_m31(ft.scale)
}

pub fn eq_index(r: &[QM31], idx: usize) -> QM31 {
    let mut out = QM31::ONE;
    for (t, &rt) in r.iter().enumerate() {
        out = out.mul(if (idx >> t) & 1 == 1 { rt } else { QM31::ONE.sub(rt) });
    }
    out
}

pub fn eq_point(r: &[QM31], b: &[QM31]) -> QM31 {
    assert_eq!(r.len(), b.len());
    let mut out = QM31::ONE;
    for (&rt, &bt) in r.iter().zip(b) {
        let term = bt.mul(rt).add(QM31::ONE.sub(bt).mul(QM31::ONE.sub(rt)));
        out = out.mul(term);
    }
    out
}

pub fn conjugate_coeffs(r_lo: &[QM31]) -> [QM31; 4] {
    assert_eq!(r_lo.len(), 2);
    let e: [QM31; 4] = [
        eq_index(r_lo, 0),
        eq_index(r_lo, 1),
        eq_index(r_lo, 2),
        eq_index(r_lo, 3),
    ];
    let mi = &consts().moore_inv;
    let mut c = [QM31::ZERO; 4];
    for i in 0..4 {
        let mut acc = QM31::ZERO;
        for k in 0..4 {
            acc = acc.add(mi[i][k].mul(e[k]));
        }
        c[i] = acc;
    }
    c
}

pub fn mobius_seq(g: &mut [QM31]) {
    let n = g.len();
    let mut half = 1;
    while half < n {
        for blk in g.chunks_exact_mut(2 * half) {
            let (lo, hi) = blk.split_at_mut(half);
            for (h, l) in hi.iter_mut().zip(lo.iter()) {
                *h = h.sub(*l);
            }
        }
        half *= 2;
    }
}

pub fn rs_encode_bitrev_seq(g: &[QM31], log_n: usize) -> Vec<QM31> {
    assert!(g.len().is_power_of_two());
    let m = g.len().trailing_zeros() as usize;
    assert!(log_n >= m);
    let s = log_n - m;
    let n = 1usize << log_n;
    let half = (n / 2).max(1);
    let zeta = root_of_unity(log_n);
    let mut tw = Vec::with_capacity(half);
    let mut cur = QM31::ONE;
    for _ in 0..half {
        tw.push(cur);
        cur = cur.mul(zeta);
    }
    let mut out = vec![QM31::ZERO; n];
    let blk = 1usize << m;
    let step0 = 1usize << s;
    for (b, chunk) in out.chunks_exact_mut(blk).enumerate() {
        let c = bit_reverse(b, s);
        if c == 0 {
            chunk.copy_from_slice(g);
        } else {
            for (k, (o, &x)) in chunk.iter_mut().zip(g.iter()).enumerate() {
                let e = c * k;
                let w = if e < half { tw[e] } else { tw[e - half].neg() };
                *o = x.mul(w);
            }
        }
        let mut b_len = chunk.len();
        let mut step = step0;
        while b_len >= 2 {
            let h = b_len / 2;
            for sub in chunk.chunks_exact_mut(b_len) {
                let (lo, hi) = sub.split_at_mut(h);
                let (u0, v0) = (lo[0], hi[0]);
                lo[0] = u0.add(v0);
                hi[0] = u0.sub(v0);
                for i in 1..h {
                    let (u, v) = (lo[i], hi[i]);
                    lo[i] = u.add(v);
                    hi[i] = u.sub(v).mul(tw[i * step]);
                }
            }
            b_len >>= 1;
            step <<= 1;
        }
    }
    out
}

/// Sequential CPRS group commitment computation used only when verifying a Stage B batch that
/// contains a decode rollover token (`(position + 1) % layout.chunk == 0`).
pub fn register_group_commitment_seq(matrices: &[&[u32]], prm: &Params) -> Result<Hash, ParamError> {
    if matrices.len() != prm.g {
        return Err(ParamError("register_group requires exactly g matrices"));
    }
    let len = 1usize << prm.n;
    for w in matrices {
        if w.len() != len {
            return Err(ParamError("every matrix must have 2^n entries"));
        }
        if !weights_canonical(w) {
            return Err(ParamError("all weights must be canonical M31 residues (< 2^31 - 1)"));
        }
    }
    let mut codes = Vec::with_capacity(prm.g);
    for w in matrices {
        let mut g = pack_view(w).to_vec();
        mobius_seq(&mut g);
        codes.push(rs_encode_bitrev_seq(&g, prm.log_n));
    }
    let f0 = prm.fold_sizes()[0];
    let lb = prm.log_n - f0;
    let nleaves = 1usize << lb;
    let width = 1usize << f0;
    let g = codes.len();
    let mut by_block = vec![[0u8; 32]; nleaves];
    let mut buf = vec![QM31::ZERO; g * width];
    for (b, hh) in by_block.iter_mut().enumerate() {
        for (t, code) in codes.iter().enumerate() {
            let blk = &code[b << f0..(b + 1) << f0];
            for u in 0..width {
                buf[t * width + u] = blk[bit_reverse(u, f0)];
            }
        }
        *hh = leaf_hash(&buf);
    }
    let hashes: Vec<Hash> = (0..nleaves)
        .map(|j| by_block[bit_reverse(j, lb)])
        .collect();
    let root = merkle_root_from_leaves(hashes);
    Ok(prm.commitment(&root))
}

pub mod register {
    pub use super::register_group_commitment_seq as register_group_commitment;
}

pub const PROTOCOL_ID: &[u8; 7] = b"cprs-v1";
pub const FIELD_ID: &[u8; 30] = b"M31|QM31:u2=2+i|basis=1,i,u,iu";
pub const HASH_ID: &[u8; 11] = b"BLAKE2s-256";
pub const HEADER_LEN: usize = 109;
pub const ZERO_CONTEXT: Hash = [0u8; 32];

pub fn header_bytes(prm: &Params, l: usize, outer: &Hash, depth: usize) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..7].copy_from_slice(PROTOCOL_ID);
    h[7..37].copy_from_slice(FIELD_ID);
    h[37..48].copy_from_slice(HASH_ID);
    h[48..64].copy_from_slice(&root_of_unity(TWO_ADICITY).to_bytes());
    h[64] = u8::try_from(prm.n).expect("Params invariant: n <= 255");
    h[65] = u8::try_from(prm.s).expect("Params invariant: s <= 31");
    h[66] = u8::try_from(prm.f).expect("Params invariant: f <= 253");
    h[67..71].copy_from_slice(&u32::try_from(prm.q).expect("Params invariant: q <= 2^32 - 1").to_le_bytes());
    h[71] = u8::try_from(prm.pow_bits).expect("Params invariant: pow_bits <= 40");
    h[72..76].copy_from_slice(&u32::try_from(l).expect("L must fit a u32").to_le_bytes());
    h[76] = u8::try_from(depth).expect("depth must fit a byte");
    h[77..109].copy_from_slice(outer);
    h
}

pub fn init_transcript(
    prm: &Params,
    l: usize,
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
) -> Transcript {
    let mut tr = Transcript::new();
    tr.absorb(TAG_HDR, &header_bytes(prm, l, outer, prm.depth()));
    let g = prm.g;
    for (j, com) in commitments.iter().enumerate() {
        tr.absorb(TAG_COM, com);
        for t in 0..g {
            let a = j * g + t;
            tr.absorb_qm(TAG_PT, &points[a]);
            tr.absorb_qm(TAG_V, std::slice::from_ref(&claims[a]));
        }
    }
    tr
}

pub fn master_params_for_groups(group_prms: &[Params]) -> Result<Params, Reject> {
    if group_prms.is_empty() {
        return Err(Reject::Statement("number of commitments out of range"));
    }
    let total_l: u64 = group_prms.iter().map(|gp| gp.g as u64).sum();
    if total_l < 1 || total_l > MAX_L {
        return Err(Reject::Statement("number of commitments out of range"));
    }
    let p0 = &group_prms[0];
    let f0 = p0.fold_sizes()[0];
    let mut n_max = p0.n;
    for gp in group_prms {
        if gp.s != p0.s
            || gp.f != p0.f
            || gp.q != p0.q
            || gp.pow_bits != p0.pow_bits
            || gp.fold_sizes()[0] != f0
        {
            return Err(Reject::Statement("heterogeneous groups must share (s, f, q, pow_bits, f0)"));
        }
        if gp.n > n_max {
            n_max = gp.n;
        }
    }
    Params::try_new(n_max, p0.s, p0.f, p0.q, p0.pow_bits, 1)
        .map_err(|_| Reject::Statement("invalid master parameters"))
}

pub fn lift_point_heterogeneous(pt: &[QM31], f0: usize, delta: usize) -> Vec<QM31> {
    if delta == 0 {
        return pt.to_vec();
    }
    let mut out = Vec::with_capacity(pt.len() + delta);
    out.extend_from_slice(&pt[..2 + f0]);
    out.resize(2 + f0 + delta, QM31::ZERO);
    out.extend_from_slice(&pt[2 + f0..]);
    out
}

pub fn init_transcript_heterogeneous(
    master_prm: &Params,
    group_prms: &[Params],
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
) -> Transcript {
    let l: usize = group_prms.iter().map(|gp| gp.g).sum();
    let mut tr = Transcript::new();
    tr.absorb(TAG_HDR, &header_bytes(master_prm, l, outer, master_prm.depth()));
    let mut a = 0usize;
    for (com, gp) in commitments.iter().zip(group_prms.iter()) {
        tr.absorb(TAG_COM, com);
        for _ in 0..gp.g {
            tr.absorb_qm(TAG_PT, &points[a]);
            tr.absorb_qm(TAG_V, std::slice::from_ref(&claims[a]));
            a += 1;
        }
    }
    tr
}

#[inline]
pub fn interp2(g: &[QM31; 3], x: QM31) -> QM31 {
    let inv2 = consts().inv2;
    let two = QM31::from_m31(2);
    let xm1 = x.sub(QM31::ONE);
    let xm2 = x.sub(two);
    let l0 = xm1.mul(xm2).mul_m31(inv2);
    let l1 = x.mul(xm2).neg();
    let l2 = x.mul(xm1).mul_m31(inv2);
    g[0].mul(l0).add(g[1].mul(l1)).add(g[2].mul(l2))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    Statement(&'static str),
    ShorterThanRoots,
    CommitmentMismatch(usize),
    Length { expected: u128, got: usize },
    NonCanonical(&'static str),
    ConjugateClaim(usize),
    PhaseARound(usize),
    PhaseAFinal,
    PhaseBRound(usize),
    PhaseBFinal,
    NonceNotZero,
    ProofOfWork,
    RegisteredPath(usize),
    OraclePath(usize),
    FoldConsistency(usize),
    FinalConstant(usize),
}

impl Reject {
    pub fn stage(&self) -> &'static str {
        match self {
            Reject::Statement(_) => "statement",
            Reject::ShorterThanRoots => "root block",
            Reject::CommitmentMismatch(_) => "commitment mismatch",
            Reject::Length { .. } => "length",
            Reject::NonCanonical(_) => "non-canonical",
            Reject::ConjugateClaim(_) => "conjugate claim mismatch",
            Reject::PhaseARound(_) => "phase A round",
            Reject::PhaseAFinal => "phase A final identity",
            Reject::PhaseBRound(_) => "phase B round",
            Reject::PhaseBFinal => "phase B final identity",
            Reject::NonceNotZero => "nonce must be 0",
            Reject::ProofOfWork => "proof of work",
            Reject::RegisteredPath(_) => "registered path",
            Reject::OraclePath(_) => "oracle path",
            Reject::FoldConsistency(_) => "fold consistency",
            Reject::FinalConstant(_) => "final constant",
        }
    }
}

impl fmt::Display for Reject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:?})", self.stage(), self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opening {
    pub leaf: Vec<QM31>,
    pub path: Vec<Hash>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryEntry {
    pub reg: Vec<Opening>,
    pub layers: Vec<Opening>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub roots: Vec<Hash>,
    pub y: Vec<[QM31; 4]>,
    pub a_rounds: Vec<[QM31; 3]>,
    pub z: Vec<QM31>,
    pub b_rounds: Vec<[QM31; 3]>,
    pub oracles: Vec<Hash>,
    pub c_final: QM31,
    pub nonce: u64,
    pub queries: Vec<QueryEntry>,
}

pub fn proof_len_checked(prm: &Params, l: usize) -> Result<u128, Reject> {
    if l < 1 || l % prm.g != 0 {
        return Err(Reject::Statement("L must be a positive multiple of g"));
    }
    let sh = prm.shape();
    let groups = (l / prm.g) as u128;
    let m = prm.m as u128;
    let l128 = l as u128;
    let y_cols = if prm.lo_vars == 0 { 0u128 } else { 4u128 };
    let fixed = 32 * groups + 16 * ((1 + y_cols) * l128 + 6 * m) + 32 * sh.inner() as u128 + 16 + 8;
    let mut per_q =
        groups * (16 * prm.g as u128 * (1u128 << sh.reg_leaf_log()) + 32 * sh.reg_depth() as u128);
    for r in 0..sh.inner() {
        per_q += 16 * (1u128 << sh.inner_leaf_log(r)) + 32 * sh.inner_depth(r) as u128;
    }
    Ok(fixed + prm.q as u128 * per_q)
}

pub fn proof_len_heterogeneous_checked(
    master_prm: &Params,
    group_prms: &[Params],
) -> Result<u128, Reject> {
    if group_prms.is_empty() {
        return Err(Reject::Statement("at least one group required"));
    }
    let sh = master_prm.shape();
    let f0 = sh.reg_leaf_log();
    let groups = group_prms.len() as u128;
    let m = master_prm.m as u128;
    let l128: u128 = group_prms.iter().map(|gp| gp.g as u128).sum();
    if l128 < 1 {
        return Err(Reject::Statement("L must be positive"));
    }
    let fixed = 32 * groups + 16 * (5 * l128 + 6 * m) + 32 * sh.inner() as u128 + 16 + 8;
    let mut per_q = 0u128;
    for gp in group_prms {
        let gsh = gp.shape();
        if gsh.reg_leaf_log() != f0 {
            return Err(Reject::Statement("all groups in a heterogeneous proof must share f0"));
        }
        per_q += 16 * (gp.g as u128) * (1u128 << f0) + 32 * (gsh.reg_depth() as u128);
    }
    for r in 0..sh.inner() {
        per_q += 16 * (1u128 << sh.inner_leaf_log(r)) + 32 * sh.inner_depth(r) as u128;
    }
    Ok(fixed + master_prm.q as u128 * per_q)
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn qm(&mut self, what: &'static str) -> Result<QM31, Reject> {
        let v = QM31::from_bytes(&self.b[self.pos..self.pos + 16]).ok_or(Reject::NonCanonical(what))?;
        self.pos += 16;
        Ok(v)
    }
    fn hash(&mut self) -> Hash {
        let mut h = [0u8; 32];
        h.copy_from_slice(&self.b[self.pos..self.pos + 32]);
        self.pos += 32;
        h
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
}

pub fn parse_proof(bytes: &[u8], prm: &Params, l: usize) -> Result<Proof, Reject> {
    let want = proof_len_checked(prm, l)?;
    if bytes.len() as u128 != want {
        return Err(Reject::Length {
            expected: want,
            got: bytes.len(),
        });
    }
    let sh: Shape = prm.shape();
    let m = prm.m;
    let groups = l / prm.g;
    let mut rd = Reader { b: bytes, pos: 0 };
    let roots: Vec<Hash> = (0..groups).map(|_| rd.hash()).collect();
    let mut y = Vec::with_capacity(if prm.lo_vars == 0 { 0 } else { l });
    if prm.lo_vars != 0 {
        for _ in 0..l {
            let mut row = [QM31::ZERO; 4];
            for v in row.iter_mut() {
                *v = rd.qm("y")?;
            }
            y.push(row);
        }
    }
    let mut a_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("A")?;
        }
        a_rounds.push(g);
    }
    let mut z = Vec::with_capacity(l);
    for _ in 0..l {
        z.push(rd.qm("z")?);
    }
    let mut b_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("B")?;
        }
        b_rounds.push(g);
    }
    let oracles: Vec<Hash> = (0..sh.inner()).map(|_| rd.hash()).collect();
    let c_final = rd.qm("c_final")?;
    let nonce = rd.u64();
    let mut queries = Vec::with_capacity(prm.q);
    for _ in 0..prm.q {
        let mut reg = Vec::with_capacity(groups);
        for _ in 0..groups {
            let mut leaf = Vec::with_capacity(prm.g << sh.reg_leaf_log());
            for _ in 0..(prm.g << sh.reg_leaf_log()) {
                leaf.push(rd.qm("registered leaf")?);
            }
            let path: Vec<Hash> = (0..sh.reg_depth()).map(|_| rd.hash()).collect();
            reg.push(Opening { leaf, path });
        }
        let mut layers = Vec::with_capacity(sh.inner());
        for r in 0..sh.inner() {
            let mut leaf = Vec::with_capacity(1usize << sh.inner_leaf_log(r));
            for _ in 0..(1usize << sh.inner_leaf_log(r)) {
                leaf.push(rd.qm("inner leaf")?);
            }
            let path: Vec<Hash> = (0..sh.inner_depth(r)).map(|_| rd.hash()).collect();
            layers.push(Opening { leaf, path });
        }
        queries.push(QueryEntry { reg, layers });
    }
    debug_assert_eq!(rd.pos, bytes.len());
    Ok(Proof {
        roots,
        y,
        a_rounds,
        z,
        b_rounds,
        oracles,
        c_final,
        nonce,
        queries,
    })
}

pub fn parse_proof_heterogeneous(
    bytes: &[u8],
    master_prm: &Params,
    group_prms: &[Params],
) -> Result<Proof, Reject> {
    let want = proof_len_heterogeneous_checked(master_prm, group_prms)?;
    if bytes.len() as u128 != want {
        return Err(Reject::Length {
            expected: want,
            got: bytes.len(),
        });
    }
    let sh: Shape = master_prm.shape();
    let f0 = sh.reg_leaf_log();
    let m = master_prm.m;
    let groups = group_prms.len();
    let l: usize = group_prms.iter().map(|gp| gp.g).sum();
    let mut rd = Reader { b: bytes, pos: 0 };
    let roots: Vec<Hash> = (0..groups).map(|_| rd.hash()).collect();
    let mut y = Vec::with_capacity(if master_prm.lo_vars == 0 { 0 } else { l });
    if master_prm.lo_vars != 0 {
        for _ in 0..l {
            let mut row = [QM31::ZERO; 4];
            for v in row.iter_mut() {
                *v = rd.qm("y")?;
            }
            y.push(row);
        }
    }
    let mut a_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("A")?;
        }
        a_rounds.push(g);
    }
    let mut z = Vec::with_capacity(l);
    for _ in 0..l {
        z.push(rd.qm("z")?);
    }
    let mut b_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("B")?;
        }
        b_rounds.push(g);
    }
    let oracles: Vec<Hash> = (0..sh.inner()).map(|_| rd.hash()).collect();
    let c_final = rd.qm("c_final")?;
    let nonce = rd.u64();
    let mut queries = Vec::with_capacity(master_prm.q);
    for _ in 0..master_prm.q {
        let mut reg = Vec::with_capacity(groups);
        for gp in group_prms {
            let gsh = gp.shape();
            let leaf_len = gp.g << f0;
            let mut leaf = Vec::with_capacity(leaf_len);
            for _ in 0..leaf_len {
                leaf.push(rd.qm("registered leaf")?);
            }
            let path: Vec<Hash> = (0..gsh.reg_depth()).map(|_| rd.hash()).collect();
            reg.push(Opening { leaf, path });
        }
        let mut layers = Vec::with_capacity(sh.inner());
        for r in 0..sh.inner() {
            let mut leaf = Vec::with_capacity(1usize << sh.inner_leaf_log(r));
            for _ in 0..(1usize << sh.inner_leaf_log(r)) {
                leaf.push(rd.qm("inner leaf")?);
            }
            let path: Vec<Hash> = (0..sh.inner_depth(r)).map(|_| rd.hash()).collect();
            layers.push(Opening { leaf, path });
        }
        queries.push(QueryEntry { reg, layers });
    }
    debug_assert_eq!(rd.pos, bytes.len());
    Ok(Proof {
        roots,
        y,
        a_rounds,
        z,
        b_rounds,
        oracles,
        c_final,
        nonce,
        queries,
    })
}

struct QueryCtx<'a> {
    roots: &'a [Hash],
    sh: &'a Shape,
    g: usize,
    mus: &'a [QM31],
    fts: &'a [FoldTables],
    zinvs: &'a [QM31],
    oracle_roots: &'a [Hash],
    c_final: QM31,
}

fn check_query(qi: usize, entry: &QueryEntry, j0: u64, cx: &QueryCtx) -> Result<(), Reject> {
    let sh = cx.sh;
    let f0 = sh.reg_leaf_log();
    let width0 = 1usize << f0;
    let mut acc = vec![[0u128; 4]; width0];
    for (j, op) in entry.reg.iter().enumerate() {
        if op.leaf.len() != cx.g * width0
            || !verify_path(&cx.roots[j], j0 as usize, leaf_hash(&op.leaf), &op.path, sh.reg_depth())
        {
            return Err(Reject::RegisteredPath(qi));
        }
        for t in 0..cx.g {
            let mu = cx.mus[j * cx.g + t];
            let slice = &op.leaf[t * width0..(t + 1) * width0];
            for u in 0..width0 {
                mul_acc(&mut acc[u], slice[u], mu);
            }
        }
    }
    let mut comb: Vec<QM31> = acc.iter().map(wide_finish).collect();
    let xinv = cx.zinvs[0].pow(j0 as u128);
    let mut expected = fold_coset(&mut comb, xinv, &cx.fts[0]);
    let mut k = j0 as usize;
    for li in 0..sh.inner() {
        let op = &entry.layers[li];
        let f_next = sh.inner_leaf_log(li);
        let stride = 1usize << sh.inner_depth(li);
        let jl = k % stride;
        let off = k / stride;
        if op.leaf.len() != (1usize << f_next)
            || !verify_path(&cx.oracle_roots[li], jl, leaf_hash(&op.leaf), &op.path, sh.inner_depth(li))
        {
            return Err(Reject::OraclePath(qi));
        }
        if op.leaf[off] != expected {
            return Err(Reject::FoldConsistency(qi));
        }
        let mut v = op.leaf.clone();
        let xinv = cx.zinvs[li + 1].pow(jl as u128);
        expected = fold_coset(&mut v, xinv, &cx.fts[li + 1]);
        k = jl;
    }
    if expected != cx.c_final {
        return Err(Reject::FinalConstant(qi));
    }
    Ok(())
}

fn check_statement(
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    prm: &Params,
) -> Result<(usize, usize), Reject> {
    let gcount = commitments.len();
    if gcount < 1 || gcount as u64 > MAX_L / prm.g as u64 {
        return Err(Reject::Statement("number of commitments out of range"));
    }
    let l = gcount * prm.g;
    if points.len() != l || claims.len() != l {
        return Err(Reject::Statement(
            "G commitments of groups of g matrices need G g points and claims",
        ));
    }
    if points.iter().any(|p| p.len() != prm.m + prm.lo_vars) {
        return Err(Reject::Statement("point must have n QM31 coordinates"));
    }
    if !points.iter().all(|p| p.iter().all(|x| x.is_canonical()))
        || !claims.iter().all(|c| c.is_canonical())
    {
        return Err(Reject::Statement(
            "point and claim coordinates must be canonical (each M31 component < 2^31 - 1)",
        ));
    }
    Ok((l, gcount))
}

pub fn verify(
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
    prm: &Params,
    proof: &[u8],
) -> Result<(), Reject> {
    let (l, gcount) = check_statement(commitments, points, claims, prm)?;
    if proof.len() < 32 * gcount {
        return Err(Reject::ShorterThanRoots);
    }
    for j in 0..gcount {
        let mut root = [0u8; 32];
        root.copy_from_slice(&proof[32 * j..32 * j + 32]);
        if prm.commitment(&root) != commitments[j] {
            return Err(Reject::CommitmentMismatch(j));
        }
    }
    let pf = parse_proof(proof, prm, l)?;
    let m = prm.m;
    let sh = prm.shape();
    let sizes = &sh.sizes;
    let mut tr = init_transcript(prm, l, commitments, points, claims, outer);

    if prm.lo_vars != 0 {
        for a in 0..l {
            let c = conjugate_coeffs(&points[a][..2]);
            let mut rhs = QM31::ZERO;
            for i in 0..4 {
                rhs = rhs.add(c[i].mul(pf.y[a][i].frob(i)));
            }
            if rhs != claims[a] {
                return Err(Reject::ConjugateClaim(a));
            }
        }
        let y_flat: Vec<QM31> = pf.y.iter().flat_map(|r| r.iter().copied()).collect();
        tr.absorb_qm(TAG_Y, &y_flat);
    }
    let gamma = tr.squeeze_qm(SQ_GAMMA);
    let gpow: Vec<QM31> = if prm.lo_vars == 0 {
        let mut gp = vec![QM31::ZERO; 4 * l + 1];
        let mut cur = QM31::ONE;
        for a in 0..l {
            gp[4 * a] = cur;
            cur = cur.mul(gamma);
        }
        gp
    } else {
        let mut gp = vec![QM31::ONE];
        for _ in 0..4 * l {
            gp.push(gp.last().unwrap().mul(gamma));
        }
        gp
    };
    let mut claim = QM31::ZERO;
    if prm.lo_vars == 0 {
        for a in 0..l {
            claim = claim.add(gpow[4 * a].mul(claims[a]));
        }
    } else {
        for a in 0..l {
            for i in 0..4 {
                claim = claim.add(gpow[4 * a + i].mul(pf.y[a][i]));
            }
        }
    }

    let mut rho: Vec<QM31> = Vec::with_capacity(m);
    for t in 0..m {
        let gg = pf.a_rounds[t];
        if gg[0].add(gg[1]) != claim {
            return Err(Reject::PhaseARound(t));
        }
        tr.absorb_qm(TAG_A, &gg);
        let r_t = tr.squeeze_qm(SQ_RHOA);
        rho.push(r_t);
        claim = interp2(&gg, r_t);
    }

    let mut total = QM31::ZERO;
    for a in 0..l {
        let mut ea = QM31::ZERO;
        for i in 0..4 {
            if gpow[4 * a + i].is_zero() {
                continue;
            }
            let pt: Vec<QM31> = points[a][prm.lo_vars..].iter().map(|x| x.frob_inv(i)).collect();
            ea = ea.add(gpow[4 * a + i].mul(eq_point(&pt, &rho)));
        }
        total = total.add(pf.z[a].mul(ea));
    }
    if total != claim {
        return Err(Reject::PhaseAFinal);
    }
    tr.absorb_qm(TAG_Z, &pf.z);
    let mut mus = vec![QM31::ONE];
    for a in 1..l {
        mus.push(tr.squeeze_qm(&mu_tag(a)));
    }
    let mut claim = QM31::ZERO;
    for a in 0..l {
        claim = claim.add(mus[a].mul(pf.z[a]));
    }

    let mut alphas_all: Vec<QM31> = Vec::with_capacity(m);
    let mut fts: Vec<FoldTables> = Vec::with_capacity(sizes.len());
    let mut pos = 0usize;
    for (r, &f_r) in sizes.iter().enumerate() {
        let mut alphas = Vec::with_capacity(f_r);
        for _ in 0..f_r {
            let gg = pf.b_rounds[pos];
            if gg[0].add(gg[1]) != claim {
                return Err(Reject::PhaseBRound(pos));
            }
            tr.absorb_qm(TAG_B, &gg);
            let al = tr.squeeze_qm(SQ_ALPHA);
            pos += 1;
            alphas.push(al);
            claim = interp2(&gg, al);
        }
        alphas_all.extend_from_slice(&alphas);
        fts.push(FoldTables::new(&alphas));
        if r < sizes.len() - 1 {
            tr.absorb(TAG_ORACLE, &pf.oracles[r]);
        }
    }
    let c_final = pf.c_final;
    let eqv = eq_point(&rho, &alphas_all);
    if claim != c_final.mul(eqv) {
        return Err(Reject::PhaseBFinal);
    }
    tr.absorb_qm(TAG_FINAL, std::slice::from_ref(&c_final));

    let nonce = pf.nonce;
    if prm.pow_bits == 0 {
        if nonce != 0 {
            return Err(Reject::NonceNotZero);
        }
    } else if !pow_ok(tr.state(), nonce, prm.pow_bits) {
        return Err(Reject::ProofOfWork);
    }
    tr.absorb(TAG_NONCE, &nonce.to_le_bytes());

    let log_leaf0 = prm.log_n - sizes[0];
    let j0s: Vec<u64> = (0..prm.q)
        .map(|_| tr.squeeze_index(SQ_QUERY, log_leaf0))
        .collect();
    let zinvs: Vec<QM31> = (0..sizes.len())
        .map(|r| root_of_unity(sh.log_sizes[r]).inv())
        .collect();
    let cx = QueryCtx {
        roots: &pf.roots,
        sh: &sh,
        g: prm.g,
        mus: &mus,
        fts: &fts,
        zinvs: &zinvs,
        oracle_roots: &pf.oracles,
        c_final,
    };
    for (qi, (entry, &j0)) in pf.queries.iter().zip(j0s.iter()).enumerate() {
        check_query(qi, entry, j0, &cx)?;
    }
    Ok(())
}

pub fn verify_strict_context(
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
    prm: &Params,
    proof: &[u8],
) -> Result<(), Reject> {
    if *outer == ZERO_CONTEXT {
        return Err(Reject::Statement("strict context: the outer digest must not be all zero"));
    }
    verify(commitments, points, claims, outer, prm, proof)
}

pub const CPRSMUL1_MAGIC: &[u8; 8] = b"CPRSMUL1";

struct CheckedReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> CheckedReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Reject> {
        let end = self.pos.checked_add(n).ok_or(Reject::Length {
            expected: u128::MAX,
            got: self.b.len(),
        })?;
        if end > self.b.len() {
            return Err(Reject::Length {
                expected: end as u128,
                got: self.b.len(),
            });
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn qm(&mut self, what: &'static str) -> Result<QM31, Reject> {
        let s = self.take(16)?;
        QM31::from_bytes(s).ok_or(Reject::NonCanonical(what))
    }
    fn hash(&mut self) -> Result<Hash, Reject> {
        let s = self.take(32)?;
        let mut h = [0u8; 32];
        h.copy_from_slice(s);
        Ok(h)
    }
    fn u64(&mut self) -> Result<u64, Reject> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }
}

#[inline]
fn low_bits_mask(bits: usize) -> usize {
    if bits == 0 {
        0
    } else if bits >= usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << bits) - 1
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_query_indices_heterogeneous(
    master_prm: &Params,
    group_prms: &[Params],
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
    y: &[[QM31; 4]],
    a_rounds: &[[QM31; 3]],
    z: &[QM31],
    b_rounds: &[[QM31; 3]],
    oracles: &[Hash],
    c_final: QM31,
    nonce: u64,
) -> Vec<u64> {
    let sh = master_prm.shape();
    let sizes = &sh.sizes;
    let m = master_prm.m;
    let l = z.len();
    let mut tr = init_transcript_heterogeneous(
        master_prm,
        group_prms,
        commitments,
        points,
        claims,
        outer,
    );
    let y_flat: Vec<QM31> = y.iter().flat_map(|r| r.iter().copied()).collect();
    tr.absorb_qm(TAG_Y, &y_flat);
    let _gamma = tr.squeeze_qm(SQ_GAMMA);
    for gg in a_rounds.iter().take(m) {
        tr.absorb_qm(TAG_A, gg);
        let _r_t = tr.squeeze_qm(SQ_RHOA);
    }
    tr.absorb_qm(TAG_Z, z);
    for a in 1..l {
        let _mu = tr.squeeze_qm(&mu_tag(a));
    }
    let mut pos = 0usize;
    for (r, &f_r) in sizes.iter().enumerate() {
        for _ in 0..f_r {
            tr.absorb_qm(TAG_B, &b_rounds[pos]);
            let _al = tr.squeeze_qm(SQ_ALPHA);
            pos += 1;
        }
        if r < sizes.len() - 1 {
            tr.absorb(TAG_ORACLE, &oracles[r]);
        }
    }
    tr.absorb_qm(TAG_FINAL, std::slice::from_ref(&c_final));
    tr.absorb(TAG_NONCE, &nonce.to_le_bytes());
    let log_leaf0 = master_prm.log_n - sizes[0];
    (0..master_prm.q)
        .map(|_| tr.squeeze_index(SQ_QUERY, log_leaf0))
        .collect()
}

fn tree_multiproof_byte_len(depth: usize, leaf_width: usize, leaf_indices: &[usize]) -> usize {
    let mut cur = leaf_indices.to_vec();
    cur.sort_unstable();
    cur.dedup();
    let leaf_bytes = cur.len() * leaf_width * 16;
    let mut sib_count = 0usize;
    for _lvl in 0..depth {
        let mut next_cur = Vec::with_capacity(cur.len());
        let mut i = 0usize;
        while i < cur.len() {
            let idx = cur[i];
            if i + 1 < cur.len() && cur[i + 1] == (idx ^ 1) {
                next_cur.push(idx >> 1);
                i += 2;
            } else {
                sib_count += 1;
                next_cur.push(idx >> 1);
                i += 1;
            }
        }
        cur = next_cur;
    }
    leaf_bytes + sib_count * 32
}

fn decode_tree_multiproof(
    rd: &mut CheckedReader<'_>,
    expected_root: &Hash,
    depth: usize,
    leaf_width: usize,
    leaf_indices: &[usize],
    what: &'static str,
    mismatch_err: Reject,
) -> Result<Vec<Opening>, Reject> {
    let mut uniq_leaves = leaf_indices.to_vec();
    uniq_leaves.sort_unstable();
    uniq_leaves.dedup();

    let mut uniq_leaf_data: Vec<Vec<QM31>> = Vec::with_capacity(uniq_leaves.len());
    let mut cur: Vec<(usize, Hash)> = Vec::with_capacity(uniq_leaves.len());
    for &u in &uniq_leaves {
        let mut leaf = Vec::with_capacity(leaf_width);
        for _ in 0..leaf_width {
            leaf.push(rd.qm(what)?);
        }
        let h = leaf_hash(&leaf);
        uniq_leaf_data.push(leaf);
        cur.push((u, h));
    }

    let mut level_nodes: Vec<Vec<(usize, Hash)>> = Vec::with_capacity(depth);
    for _lvl in 0..depth {
        let mut nodes_at_lvl: Vec<(usize, Hash)> = Vec::with_capacity(cur.len() * 2);
        let mut next_cur: Vec<(usize, Hash)> = Vec::with_capacity(cur.len());
        let mut i = 0usize;
        while i < cur.len() {
            let (idx, h_idx) = cur[i];
            if i + 1 < cur.len() && cur[i + 1].0 == (idx ^ 1) {
                let (odd_idx, h_odd) = cur[i + 1];
                nodes_at_lvl.push((idx, h_idx));
                nodes_at_lvl.push((odd_idx, h_odd));
                next_cur.push((idx >> 1, node_hash(&h_idx, &h_odd)));
                i += 2;
            } else {
                let h_sib = rd.hash()?;
                let sib_idx = idx ^ 1;
                if idx & 1 == 0 {
                    nodes_at_lvl.push((idx, h_idx));
                    nodes_at_lvl.push((sib_idx, h_sib));
                    next_cur.push((idx >> 1, node_hash(&h_idx, &h_sib)));
                } else {
                    nodes_at_lvl.push((sib_idx, h_sib));
                    nodes_at_lvl.push((idx, h_idx));
                    next_cur.push((idx >> 1, node_hash(&h_sib, &h_idx)));
                }
                i += 1;
            }
        }
        level_nodes.push(nodes_at_lvl);
        cur = next_cur;
    }

    if cur.len() != 1 || cur[0].0 != 0 || &cur[0].1 != expected_root {
        return Err(mismatch_err);
    }

    let mut openings = Vec::with_capacity(leaf_indices.len());
    for &li in leaf_indices {
        let leaf_pos = uniq_leaves
            .binary_search(&li)
            .expect("query leaf index present in uniq_leaves");
        let leaf = uniq_leaf_data[leaf_pos].clone();
        let mut path = Vec::with_capacity(depth);
        for (lvl, nodes_at_lvl) in level_nodes.iter().enumerate() {
            let sib_idx = (li >> lvl) ^ 1;
            let pos = nodes_at_lvl
                .binary_search_by_key(&sib_idx, |&(k, _)| k)
                .expect("sibling index present in multiproof level");
            path.push(nodes_at_lvl[pos].1);
        }
        openings.push(Opening { leaf, path });
    }
    Ok(openings)
}

pub fn parse_or_decompress_proof_heterogeneous(
    bytes: &[u8],
    master_prm: &Params,
    group_prms: &[Params],
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
) -> Result<(Proof, bool), Reject> {
    if !(bytes.starts_with(CPRSMUL1_MAGIC) && bytes.len().is_multiple_of(16)) {
        let pf = parse_proof_heterogeneous(bytes, master_prm, group_prms)?;
        return Ok((pf, false));
    }

    let sh = master_prm.shape();
    let f0 = sh.reg_leaf_log();
    let m = master_prm.m;
    let groups = group_prms.len();
    let l: usize = group_prms.iter().map(|gp| gp.g).sum();

    let mut rd = CheckedReader {
        b: bytes,
        pos: CPRSMUL1_MAGIC.len(),
    };
    let mut roots = Vec::with_capacity(groups);
    for _ in 0..groups {
        roots.push(rd.hash()?);
    }
    let mut y = Vec::with_capacity(l);
    let mut a = 0usize;
    for gp in group_prms {
        for _ in 0..gp.g {
            if gp.lo_vars == 0 {
                y.push([claims[a], QM31::ZERO, QM31::ZERO, QM31::ZERO]);
            } else {
                let mut row = [QM31::ZERO; 4];
                for v in row.iter_mut() {
                    *v = rd.qm("y")?;
                }
                y.push(row);
            }
            a += 1;
        }
    }
    let mut a_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("A")?;
        }
        a_rounds.push(g);
    }
    let mut z = Vec::with_capacity(l);
    for _ in 0..l {
        z.push(rd.qm("z")?);
    }
    let mut b_rounds = Vec::with_capacity(m);
    for _ in 0..m {
        let mut g = [QM31::ZERO; 3];
        for v in g.iter_mut() {
            *v = rd.qm("B")?;
        }
        b_rounds.push(g);
    }
    let mut oracles = Vec::with_capacity(sh.inner());
    for _ in 0..sh.inner() {
        oracles.push(rd.hash()?);
    }
    let c_final = rd.qm("c_final")?;
    let nonce = rd.u64()?;

    let j0s = derive_query_indices_heterogeneous(
        master_prm,
        group_prms,
        commitments,
        points,
        claims,
        outer,
        &y,
        &a_rounds,
        &z,
        &b_rounds,
        &oracles,
        c_final,
        nonce,
    );

    let mut reg_tasks = Vec::with_capacity(groups);
    let mut cursor = rd.pos;
    for (j, gp) in group_prms.iter().enumerate() {
        let depth_j = gp.depth();
        let mask_j = low_bits_mask(depth_j);
        let leaf_indices: Vec<usize> = j0s.iter().map(|&j0| (j0 as usize) & mask_j).collect();
        let leaf_w = gp.g << f0;
        let blen = tree_multiproof_byte_len(depth_j, leaf_w, &leaf_indices);
        let end = cursor.checked_add(blen).ok_or(Reject::Length {
            expected: u128::MAX,
            got: bytes.len(),
        })?;
        reg_tasks.push((j, depth_j, leaf_w, leaf_indices, cursor..end));
        cursor = end;
    }

    let mut inner_tasks = Vec::with_capacity(sh.inner());
    for (r, _oracle_root) in oracles.iter().enumerate() {
        let depth_r = sh.inner_depth(r);
        let mask_r = low_bits_mask(depth_r);
        let leaf_indices: Vec<usize> = j0s.iter().map(|&j0| (j0 as usize) & mask_r).collect();
        let leaf_w = 1usize << sh.inner_leaf_log(r);
        let blen = tree_multiproof_byte_len(depth_r, leaf_w, &leaf_indices);
        let end = cursor.checked_add(blen).ok_or(Reject::Length {
            expected: u128::MAX,
            got: bytes.len(),
        })?;
        inner_tasks.push((r, depth_r, leaf_w, leaf_indices, cursor..end));
        cursor = end;
    }

    if cursor != bytes.len() {
        return Err(Reject::Length {
            expected: cursor as u128,
            got: bytes.len(),
        });
    }

    let mut reg_by_group: Vec<Vec<Opening>> = Vec::with_capacity(groups);
    for (j, depth_j, leaf_w, leaf_indices, range) in reg_tasks {
        let mut trd = CheckedReader {
            b: &bytes[range],
            pos: 0,
        };
        let ops = decode_tree_multiproof(
            &mut trd,
            &roots[j],
            depth_j,
            leaf_w,
            &leaf_indices,
            "registered leaf",
            Reject::RegisteredPath(0),
        )?;
        debug_assert_eq!(trd.pos, trd.b.len());
        reg_by_group.push(ops);
    }

    let mut inner_by_layer: Vec<Vec<Opening>> = Vec::with_capacity(sh.inner());
    for (r, depth_r, leaf_w, leaf_indices, range) in inner_tasks {
        let mut trd = CheckedReader {
            b: &bytes[range],
            pos: 0,
        };
        let ops = decode_tree_multiproof(
            &mut trd,
            &oracles[r],
            depth_r,
            leaf_w,
            &leaf_indices,
            "inner leaf",
            Reject::OraclePath(0),
        )?;
        debug_assert_eq!(trd.pos, trd.b.len());
        inner_by_layer.push(ops);
    }

    let q = master_prm.q;
    let mut reg_iters: Vec<std::vec::IntoIter<Opening>> =
        reg_by_group.into_iter().map(Vec::into_iter).collect();
    let mut inner_iters: Vec<std::vec::IntoIter<Opening>> =
        inner_by_layer.into_iter().map(Vec::into_iter).collect();
    let mut queries = Vec::with_capacity(q);
    for _ in 0..q {
        let reg = reg_iters
            .iter_mut()
            .map(|it| it.next().expect("q openings per group"))
            .collect();
        let layers = inner_iters
            .iter_mut()
            .map(|it| it.next().expect("q openings per inner layer"))
            .collect();
        queries.push(QueryEntry { reg, layers });
    }

    Ok((
        Proof {
            roots,
            y,
            a_rounds,
            z,
            b_rounds,
            oracles,
            c_final,
            nonce,
            queries,
        },
        true,
    ))
}

struct HeteroQueryCtx<'a> {
    roots: &'a [Hash],
    sh: &'a Shape,
    group_prms: &'a [Params],
    mus: &'a [QM31],
    fts: &'a [FoldTables],
    zinvs: &'a [QM31],
    oracle_roots: &'a [Hash],
    c_final: QM31,
    paths_preverified: bool,
}

fn check_query_heterogeneous(
    qi: usize,
    entry: &QueryEntry,
    j0: u64,
    cx: &HeteroQueryCtx,
) -> Result<(), Reject> {
    let sh = cx.sh;
    let f0 = sh.reg_leaf_log();
    let width0 = 1usize << f0;
    let mut buckets: Vec<(usize, Vec<[u128; 4]>)> = Vec::with_capacity(4);
    let mut a_off = 0usize;
    for (j, op) in entry.reg.iter().enumerate() {
        let gp = &cx.group_prms[j];
        let depth_j = gp.depth();
        let j0_j = (j0 as usize) & low_bits_mask(depth_j);
        if op.leaf.len() != gp.g * width0
            || (!cx.paths_preverified
                && !verify_path(&cx.roots[j], j0_j, leaf_hash(&op.leaf), &op.path, depth_j))
        {
            return Err(Reject::RegisteredPath(qi));
        }
        let acc = match buckets.iter_mut().find(|(ln, _)| *ln == gp.log_n) {
            Some((_, a)) => a,
            None => {
                buckets.push((gp.log_n, vec![[0u128; 4]; width0]));
                &mut buckets.last_mut().unwrap().1
            }
        };
        for t in 0..gp.g {
            let mu = cx.mus[a_off + t];
            let slice = &op.leaf[t * width0..(t + 1) * width0];
            for u in 0..width0 {
                mul_acc(&mut acc[u], slice[u], mu);
            }
        }
        a_off += gp.g;
    }
    let mut expected = QM31::ZERO;
    for (log_n_b, acc_b) in &buckets {
        let mut comb: Vec<QM31> = acc_b.iter().map(wide_finish).collect();
        let depth_b = log_n_b - f0;
        let j0_b = (j0 as usize) & low_bits_mask(depth_b);
        let xinv = root_of_unity(*log_n_b).inv().pow(j0_b as u128);
        expected = expected.add(fold_coset(&mut comb, xinv, &cx.fts[0]));
    }
    let mut k = j0 as usize;
    for li in 0..sh.inner() {
        let op = &entry.layers[li];
        let f_next = sh.inner_leaf_log(li);
        let stride = 1usize << sh.inner_depth(li);
        let jl = k % stride;
        let off = k / stride;
        if op.leaf.len() != (1usize << f_next)
            || (!cx.paths_preverified
                && !verify_path(&cx.oracle_roots[li], jl, leaf_hash(&op.leaf), &op.path, sh.inner_depth(li)))
        {
            return Err(Reject::OraclePath(qi));
        }
        if op.leaf[off] != expected {
            return Err(Reject::FoldConsistency(qi));
        }
        let mut v = op.leaf.clone();
        let xinv = cx.zinvs[li + 1].pow(jl as u128);
        expected = fold_coset(&mut v, xinv, &cx.fts[li + 1]);
        k = jl;
    }
    if expected != cx.c_final {
        return Err(Reject::FinalConstant(qi));
    }
    Ok(())
}

fn check_statement_heterogeneous(
    commitments: &[Hash],
    points: &[Vec<QM31>],
    claims: &[QM31],
    group_prms: &[Params],
) -> Result<(usize, usize), Reject> {
    let gcount = commitments.len();
    if gcount < 1 || gcount != group_prms.len() {
        return Err(Reject::Statement("number of commitments must match group_prms"));
    }
    let l_u64: u64 = group_prms.iter().map(|gp| gp.g as u64).sum();
    if l_u64 < 1 || l_u64 > MAX_L {
        return Err(Reject::Statement("number of commitments out of range"));
    }
    let l = l_u64 as usize;
    if points.len() != l || claims.len() != l {
        return Err(Reject::Statement(
            "heterogeneous groups need sum_j g_j points and claims",
        ));
    }
    let mut a = 0usize;
    for gp in group_prms {
        for _ in 0..gp.g {
            if points[a].len() != gp.m + gp.lo_vars {
                return Err(Reject::Statement("point must have n_j QM31 coordinates"));
            }
            a += 1;
        }
    }
    if !points.iter().all(|p| p.iter().all(|x| x.is_canonical()))
        || !claims.iter().all(|c| c.is_canonical())
    {
        return Err(Reject::Statement(
            "point and claim coordinates must be canonical (each M31 component < 2^31 - 1)",
        ));
    }
    Ok((l, gcount))
}

pub fn verify_heterogeneous(
    commitments: &[Hash],
    group_prms: &[Params],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
    proof: &[u8],
) -> Result<(), Reject> {
    let master_prm = master_params_for_groups(group_prms)?;
    let (l, gcount) = check_statement_heterogeneous(commitments, points, claims, group_prms)?;
    let is_multiproof = proof.starts_with(CPRSMUL1_MAGIC) && proof.len().is_multiple_of(16);
    let roots_off = if is_multiproof { CPRSMUL1_MAGIC.len() } else { 0 };
    if proof.len() < roots_off + 32 * gcount {
        return Err(Reject::ShorterThanRoots);
    }
    for j in 0..gcount {
        let mut root = [0u8; 32];
        root.copy_from_slice(&proof[roots_off + 32 * j..roots_off + 32 * j + 32]);
        if group_prms[j].commitment(&root) != commitments[j] {
            return Err(Reject::CommitmentMismatch(j));
        }
    }
    let (pf, paths_preverified) = parse_or_decompress_proof_heterogeneous(
        proof,
        &master_prm,
        group_prms,
        commitments,
        points,
        claims,
        outer,
    )?;
    let m = master_prm.m;
    let sh = master_prm.shape();
    let sizes = &sh.sizes;
    let f0 = sizes[0];
    let mut tr = init_transcript_heterogeneous(&master_prm, group_prms, commitments, points, claims, outer);

    let mut a = 0usize;
    for gp in group_prms {
        for _ in 0..gp.g {
            if gp.lo_vars == 0 {
                if pf.y[a] != [claims[a], QM31::ZERO, QM31::ZERO, QM31::ZERO] {
                    return Err(Reject::ConjugateClaim(a));
                }
            } else {
                let c = conjugate_coeffs(&points[a][..2]);
                let mut rhs = QM31::ZERO;
                for i in 0..4 {
                    rhs = rhs.add(c[i].mul(pf.y[a][i].frob(i)));
                }
                if rhs != claims[a] {
                    return Err(Reject::ConjugateClaim(a));
                }
            }
            a += 1;
        }
    }
    let y_flat: Vec<QM31> = pf.y.iter().flat_map(|r| r.iter().copied()).collect();
    tr.absorb_qm(TAG_Y, &y_flat);
    let gamma = tr.squeeze_qm(SQ_GAMMA);
    let mut gpow = vec![QM31::ONE];
    for _ in 0..4 * l {
        gpow.push(gpow.last().unwrap().mul(gamma));
    }
    let mut a = 0usize;
    for gp in group_prms {
        for _ in 0..gp.g {
            if gp.lo_vars == 0 {
                gpow[4 * a + 1] = QM31::ZERO;
                gpow[4 * a + 2] = QM31::ZERO;
                gpow[4 * a + 3] = QM31::ZERO;
            }
            a += 1;
        }
    }
    let mut claim = QM31::ZERO;
    for a in 0..l {
        for i in 0..4 {
            claim = claim.add(gpow[4 * a + i].mul(pf.y[a][i]));
        }
    }

    let mut rho: Vec<QM31> = Vec::with_capacity(m);
    for t in 0..m {
        let gg = pf.a_rounds[t];
        if gg[0].add(gg[1]) != claim {
            return Err(Reject::PhaseARound(t));
        }
        tr.absorb_qm(TAG_A, &gg);
        let r_t = tr.squeeze_qm(SQ_RHOA);
        rho.push(r_t);
        claim = interp2(&gg, r_t);
    }

    let mut total = QM31::ZERO;
    let mut a_cursor = 0usize;
    for gp in group_prms {
        let delta = m - gp.m;
        for _ in 0..gp.g {
            let ea = if gp.lo_vars == 0 {
                let mut lifted_hi = Vec::with_capacity(m);
                lifted_hi.extend_from_slice(&points[a_cursor][..f0]);
                lifted_hi.resize(f0 + delta, QM31::ZERO);
                lifted_hi.extend_from_slice(&points[a_cursor][f0..]);
                gpow[4 * a_cursor].mul(eq_point(&lifted_hi, &rho))
            } else {
                let lifted_pt = lift_point_heterogeneous(&points[a_cursor], f0, delta);
                let mut ea = QM31::ZERO;
                for i in 0..4 {
                    let pt: Vec<QM31> = lifted_pt[2..].iter().map(|x| x.frob_inv(i)).collect();
                    ea = ea.add(gpow[4 * a_cursor + i].mul(eq_point(&pt, &rho)));
                }
                ea
            };
            total = total.add(pf.z[a_cursor].mul(ea));
            a_cursor += 1;
        }
    }
    if total != claim {
        return Err(Reject::PhaseAFinal);
    }
    tr.absorb_qm(TAG_Z, &pf.z);
    let mut mus = vec![QM31::ONE];
    for a in 1..l {
        mus.push(tr.squeeze_qm(&mu_tag(a)));
    }
    let mut claim = QM31::ZERO;
    for a in 0..l {
        claim = claim.add(mus[a].mul(pf.z[a]));
    }

    let mut alphas_all: Vec<QM31> = Vec::with_capacity(m);
    let mut fts: Vec<FoldTables> = Vec::with_capacity(sizes.len());
    let mut pos = 0usize;
    for (r, &f_r) in sizes.iter().enumerate() {
        let mut alphas = Vec::with_capacity(f_r);
        for _ in 0..f_r {
            let gg = pf.b_rounds[pos];
            if gg[0].add(gg[1]) != claim {
                return Err(Reject::PhaseBRound(pos));
            }
            tr.absorb_qm(TAG_B, &gg);
            let al = tr.squeeze_qm(SQ_ALPHA);
            pos += 1;
            alphas.push(al);
            claim = interp2(&gg, al);
        }
        alphas_all.extend_from_slice(&alphas);
        fts.push(FoldTables::new(&alphas));
        if r < sizes.len() - 1 {
            tr.absorb(TAG_ORACLE, &pf.oracles[r]);
        }
    }
    let c_final = pf.c_final;
    let eqv = eq_point(&rho, &alphas_all);
    if claim != c_final.mul(eqv) {
        return Err(Reject::PhaseBFinal);
    }
    tr.absorb_qm(TAG_FINAL, std::slice::from_ref(&c_final));

    let nonce = pf.nonce;
    if master_prm.pow_bits == 0 {
        if nonce != 0 {
            return Err(Reject::NonceNotZero);
        }
    } else if !pow_ok(tr.state(), nonce, master_prm.pow_bits) {
        return Err(Reject::ProofOfWork);
    }
    tr.absorb(TAG_NONCE, &nonce.to_le_bytes());

    let log_leaf0 = master_prm.log_n - sizes[0];
    let j0s: Vec<u64> = (0..master_prm.q)
        .map(|_| tr.squeeze_index(SQ_QUERY, log_leaf0))
        .collect();
    let zinvs: Vec<QM31> = (0..sizes.len())
        .map(|r| root_of_unity(sh.log_sizes[r]).inv())
        .collect();
    let cx = HeteroQueryCtx {
        roots: &pf.roots,
        sh: &sh,
        group_prms,
        mus: &mus,
        fts: &fts,
        zinvs: &zinvs,
        oracle_roots: &pf.oracles,
        c_final,
        paths_preverified,
    };
    for (qi, (entry, &j0)) in pf.queries.iter().zip(j0s.iter()).enumerate() {
        check_query_heterogeneous(qi, entry, j0, &cx)?;
    }
    Ok(())
}

pub fn verify_heterogeneous_strict_context(
    commitments: &[Hash],
    group_prms: &[Params],
    points: &[Vec<QM31>],
    claims: &[QM31],
    outer: &Hash,
    proof: &[u8],
) -> Result<(), Reject> {
    if *outer == ZERO_CONTEXT {
        return Err(Reject::Statement("strict context: the outer digest must not be all zero"));
    }
    verify_heterogeneous(commitments, group_prms, points, claims, outer, proof)
}
