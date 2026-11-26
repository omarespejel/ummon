use std::collections::BTreeMap;

use blake2::{Blake2s256, Digest};
use num_traits::{One, Zero};
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;

use crate::cprs::{
    self, i8_to_canonical_m31_u32, stwo_point_to_cprs, to_cprs_qm31, verify_heterogeneous_strict_context,
    verify_strict_context, Hash, Params, QM31,
};
use crate::stage_a::{
    cprs_profile_parameters, put_gkr_batch_proof, read_gkr_batch_proof, requant_with_rounding,
    RequantSpec, StageARoundingMode,
};
use crate::{
    acc_to_secure, bytemuck_cast_i8, cfs_deg3_accumulate_weights, eq, eq_evals, eq_evals_i64,
    i8_to_m31, make_channel, mle_at, partially_verify, partially_verify_batch,
    partially_verify_batch_with_libra_masks, partially_verify_with_lo_basis, put_bytes,
    put_secure_field, put_secure_fields, put_sumcheck, put_u32, put_u64, restrict_cols_i8_fast,
    restrict_rows_i8_fast, zk_lo_basis_omega_scale, Gate, GkrBatchProof, ReceiptReader,
    SecurityProfile, SumcheckProof, MAX_WIRE_RECEIPT_BYTES,
};

pub const ATTENTION_POLICY_ID: &str = "attention-i8-v0";
pub const EXP_TABLE_LEN: usize = 256;
pub const LOG_EXP_TABLE_LEN: usize = 8;
pub const EXP_TABLE_SCALE_BITS: u32 = 15;
pub const EXP_TABLE_STEP_RECIP: f64 = 16.0;
pub const WEIGHT_BITS: u32 = 11;
pub const ROPE_BITS: u32 = 14;
pub const ROPE_THETA: f64 = 10_000.0;
pub const GAMMA_BITS: u32 = 12;

pub const STATEMENT_DOMAIN: &[u8] = b"stwo-ml:stage-b:statement:v0";
pub const EMBEDDING_DOMAIN: &[u8] = b"stwo-ml:stage-b:embedding:v0";
pub const LM_HEAD_DOMAIN: &[u8] = b"stwo-ml:stage-b:lm-head:v0";
pub const CACHE_DIGEST_DOMAIN: &[u8] = b"stwo-ml:stage-b:cache:v0";
pub const STEP_CACHE_DOMAIN: &[u8] = b"stwo-ml:stage-b:step-cache:v1";
pub const OUTER_DOMAIN: &[u8] = b"stwo-ml:stage-b:cprs-outer:v0";
pub const STAGE_B_RECEIPT_MAGIC: &[u8; 8] = b"STWOBRC1";
pub const STAGE_B_RECEIPT_WIRE_VERSION: u32 = 1;
pub const STAGE_B_RECEIPT_WIRE_VERSION_GKR: u32 = 2;
pub const STAGE_B_RECEIPT_WIRE_VERSION_ZK_GKR: u32 = 3;
pub const ZK_B_MASK_LOG_ROWS: usize = 4;
pub const ZK_B_MASK_ROWS: usize = 1 << ZK_B_MASK_LOG_ROWS;

pub const STAGE_B_WITNESS_COLS: usize = 11;
pub const STAGE_B_WITNESS_COLS_ZK_DATA: usize = 14;
pub const STAGE_B_WITNESS_COLS_ZK: usize = 15;
pub const COL_B_X1: usize = 0;
pub const COL_B_Q_RAW: usize = 1;
pub const COL_B_K_RAW: usize = 2;
pub const COL_B_V_RAW: usize = 3;
pub const COL_B_Q_ROT: usize = 4;
pub const COL_B_CTX_RAW: usize = 5;
pub const COL_B_CTX: usize = 6;
pub const COL_B_O_RAW: usize = 7;
pub const COL_B_EXP_IDX: usize = 8;
pub const COL_B_EXP_VAL: usize = 9;
pub const COL_B_MULT: usize = 10;
pub const COL_B_H_IN: usize = 11;
pub const COL_B_A: usize = 12;
pub const COL_B_X: usize = 13;
pub const COL_B_BLIND: usize = 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    pub dimension: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionShape {
    pub const fn kv_dim(self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub const fn group_size(self) -> usize {
        self.heads / self.kv_heads
    }

    pub fn validate(self) -> Result<(), String> {
        if self.dimension == 0 || !self.dimension.is_power_of_two() || self.dimension > 1 << 13 {
            return Err("attention dimension must be a power of two up to 8192".into());
        }
        if self.heads == 0 || self.head_dim == 0 || self.heads * self.head_dim != self.dimension {
            return Err("attention heads * head_dim must equal the dimension".into());
        }
        if self.kv_heads == 0
            || !self.kv_heads.is_power_of_two()
            || self.kv_heads > self.heads
            || self.heads % self.kv_heads != 0
        {
            return Err("attention kv_heads must be a power of two dividing heads".into());
        }
        if self.head_dim % 2 != 0 || !self.head_dim.is_power_of_two() {
            return Err("attention head_dim must be an even power of two".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionPolicy {
    pub rounding: StageARoundingMode,
    pub qkv: RequantSpec,
    pub output: RequantSpec,
    pub score_shift: u32,
    pub norm1_fraction_bits: i32,
    pub norm2_fraction_bits: i32,
}

impl AttentionPolicy {
    pub fn id(&self) -> String {
        format!(
            "{}/qkv{}:{}/o{}:{}/sh{}/n{}:{}/{}",
            ATTENTION_POLICY_ID,
            self.qkv.multiplier,
            self.qkv.shift,
            self.output.multiplier,
            self.output.shift,
            self.score_shift,
            self.norm1_fraction_bits,
            self.norm2_fraction_bits,
            match self.rounding {
                StageARoundingMode::TiesToEven => "ties-even",
                StageARoundingMode::HalfTowardPositiveInfinity => "half-up",
            }
        )
    }

    pub const fn v0() -> Self {
        Self {
            rounding: StageARoundingMode::TiesToEven,
            qkv: RequantSpec {
                multiplier: 1,
                shift: 10,
                output_bits: 8,
            },
            output: RequantSpec {
                multiplier: 1,
                shift: 10,
                output_bits: 12,
            },
            score_shift: 6,
            norm1_fraction_bits: 5,
            norm2_fraction_bits: 7,
        }
    }

    pub const fn trained_v1() -> Self {
        Self {
            rounding: StageARoundingMode::TiesToEven,
            qkv: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 8,
            },
            output: RequantSpec {
                multiplier: 7,
                shift: 11,
                output_bits: 12,
            },
            score_shift: 8,
            norm1_fraction_bits: 5,
            norm2_fraction_bits: 5,
        }
    }

    pub fn from_id(id: &str) -> Result<Self, String> {
        let parts: Vec<&str> = id.split('/').collect();
        if parts.len() != 6 || parts[0] != ATTENTION_POLICY_ID {
            return Err(format!("unknown Stage B attention policy id {id:?}"));
        }
        let parse_pair = |raw: &str, prefix: &str| -> Result<(i64, u32), String> {
            let body = raw
                .strip_prefix(prefix)
                .ok_or_else(|| format!("invalid {prefix} in Stage B policy id {id:?}"))?;
            let (m, s) = body
                .split_once(':')
                .ok_or_else(|| format!("invalid {prefix} pair in Stage B policy id {id:?}"))?;
            Ok((
                m.parse::<i64>()
                    .map_err(|_| format!("invalid {prefix} multiplier in {id:?}"))?,
                s.parse::<u32>()
                    .map_err(|_| format!("invalid {prefix} shift in {id:?}"))?,
            ))
        };
        let (qkv_m, qkv_s) = parse_pair(parts[1], "qkv")?;
        let (out_m, out_s) = parse_pair(parts[2], "o")?;
        let score_shift = parts[3]
            .strip_prefix("sh")
            .ok_or_else(|| format!("invalid score_shift in {id:?}"))?
            .parse::<u32>()
            .map_err(|_| format!("invalid score_shift in {id:?}"))?;
        let n_body = parts[4]
            .strip_prefix("n")
            .ok_or_else(|| format!("invalid norm bits in {id:?}"))?;
        let (n1_s, n2_s) = n_body
            .split_once(':')
            .ok_or_else(|| format!("invalid norm bits pair in {id:?}"))?;
        let norm1_fraction_bits = n1_s
            .parse::<i32>()
            .map_err(|_| format!("invalid norm1 in {id:?}"))?;
        let norm2_fraction_bits = n2_s
            .parse::<i32>()
            .map_err(|_| format!("invalid norm2 in {id:?}"))?;
        let rounding = match parts[5] {
            "ties-even" => StageARoundingMode::TiesToEven,
            "half-up" => StageARoundingMode::HalfTowardPositiveInfinity,
            _ => return Err(format!("invalid rounding mode in {id:?}")),
        };
        let policy = Self {
            rounding,
            qkv: RequantSpec {
                multiplier: qkv_m,
                shift: qkv_s,
                output_bits: 8,
            },
            output: RequantSpec {
                multiplier: out_m,
                shift: out_s,
                output_bits: 12,
            },
            score_shift,
            norm1_fraction_bits,
            norm2_fraction_bits,
        };
        if policy.id() != id {
            return Err(format!("non-canonical Stage B attention policy id {id:?}"));
        }
        Ok(policy)
    }
}

pub fn exp_table_static() -> &'static [u16; EXP_TABLE_LEN] {
    static TABLE: std::sync::OnceLock<[u16; EXP_TABLE_LEN]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|a| {
            ((1u32 << EXP_TABLE_SCALE_BITS) as f64 * (-(a as f64) / EXP_TABLE_STEP_RECIP).exp())
                .floor() as u16
        })
    })
}

pub fn exp_table() -> Vec<u16> {
    exp_table_static().to_vec()
}

pub fn rope_tables(head_dim: usize, position: u64) -> (Vec<i32>, Vec<i32>) {
    let half = head_dim / 2;
    let scale = (1u32 << ROPE_BITS) as f64;
    (0..half)
        .map(|i| {
            let freq = ROPE_THETA.powf(-2.0 * i as f64 / head_dim as f64);
            let angle = position as f64 * freq;
            (
                (angle.cos() * scale).round() as i32,
                (angle.sin() * scale).round() as i32,
            )
        })
        .unzip()
}

fn round_ratio_ties_even_signed(numerator: i128, denominator: i128) -> i128 {
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator.rem_euclid(denominator);
    match (2 * remainder).cmp(&denominator) {
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal if quotient.rem_euclid(2) != 0 => quotient + 1,
        _ => quotient,
    }
}

fn isqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut x = (value as f64).sqrt() as u128;
    while x * x > value {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= value {
        x += 1;
    }
    x
}

fn requant_i64(value: i64, spec: RequantSpec, rounding: StageARoundingMode) -> Result<i32, String> {
    requant_with_rounding(i128::from(value), spec, rounding).map(|r| r.value)
}

pub fn rms_norm(
    h: &[i16],
    gamma: &[i16],
    fraction_bits: i32,
    output_bits: u32,
) -> Result<Vec<i32>, String> {
    if h.len() != gamma.len() || h.is_empty() {
        return Err("rms_norm input and gamma lengths differ".into());
    }
    if !(-16..=16).contains(&fraction_bits) {
        return Err("rms_norm fraction bits must be in -16..=16".into());
    }
    let sum_squares: u128 = h.iter().map(|&v| (i128::from(v) * i128::from(v)) as u128).sum();
    let rms = isqrt(sum_squares / h.len() as u128).max(1) as i128;
    let (numerator_shift, denominator_shift) = if fraction_bits >= 0 {
        (fraction_bits as u32, GAMMA_BITS)
    } else {
        (0, GAMMA_BITS + fraction_bits.unsigned_abs())
    };
    let denominator = rms << denominator_shift;
    let min = -(1i128 << (output_bits - 1));
    let max = (1i128 << (output_bits - 1)) - 1;
    h.iter()
        .zip(gamma)
        .map(|(&hi, &gi)| {
            let value = round_ratio_ties_even_signed(
                (i128::from(hi) * i128::from(gi)) << numerator_shift,
                denominator,
            );
            if !(min..=max).contains(&value) {
                return Err(format!(
                    "rms_norm output {value} exceeds signed {output_bits}-bit range (h = {hi}, gamma = {gi}, rms = {rms}, d = {})",
                    h.len()
                ));
            }
            Ok(value as i32)
        })
        .collect()
}

pub fn rope(values: &[i8], shape: AttentionShape, position: u64) -> Result<Vec<i8>, String> {
    if values.is_empty() || values.len() % shape.head_dim != 0 {
        return Err("rope input length must be a multiple of head_dim".into());
    }
    let heads = values.len() / shape.head_dim;
    let (cos, sin) = rope_tables(shape.head_dim, position);
    let half = shape.head_dim / 2;
    let round = |v: i64| -> Result<i8, String> {
        let r = (v + (1i64 << (ROPE_BITS - 1))) >> ROPE_BITS;
        i8::try_from(r).map_err(|_| "rope output exceeds i8".to_string())
    };
    let mut out = vec![0i8; values.len()];
    for head in 0..heads {
        let base = head * shape.head_dim;
        for i in 0..half {
            let a = i64::from(values[base + i]);
            let b = i64::from(values[base + half + i]);
            let (c, s) = (i64::from(cos[i]), i64::from(sin[i]));
            out[base + i] = round(a * c - b * s)?;
            out[base + half + i] = round(b * c + a * s)?;
        }
    }
    Ok(out)
}

pub fn softmax_weights(scores: &[i32], score_shift: u32) -> Result<Vec<u16>, String> {
    let max = *scores
        .iter()
        .max()
        .ok_or_else(|| "softmax over an empty context".to_string())?;
    let table = exp_table_static();
    let e: Vec<u32> = scores
        .iter()
        .map(|&s| {
            let index = (i64::from(max) - i64::from(s)) >> score_shift;
            u32::from(table[index.min(EXP_TABLE_LEN as i64 - 1) as usize])
        })
        .collect();
    let z: u64 = e.iter().map(|&v| u64::from(v)).sum();
    Ok(e.iter()
        .map(|&v| ((u64::from(v) << WEIGHT_BITS) / z) as u16)
        .collect())
}

pub fn residual_out(a: &[Vec<i16>], out: &[Vec<i16>]) -> Result<Vec<Vec<i16>>, String> {
    if a.len() != out.len() {
        return Err("residual add needs one output row per residual row".into());
    }
    a.iter()
        .zip(out)
        .map(|(ra, ro)| {
            if ra.len() != ro.len() {
                return Err("residual add row length mismatch".into());
            }
            ra.iter()
                .zip(ro)
                .map(|(&x, &y)| {
                    i16::try_from(i32::from(x) + i32::from(y))
                        .map_err(|_| "residual exceeds i16".to_string())
                })
                .collect()
        })
        .collect()
}

fn requant_to_i8(values: &[i32], policy: AttentionPolicy) -> Result<Vec<i8>, String> {
    values
        .iter()
        .map(|&v| {
            requant_i64(i64::from(v), policy.qkv, policy.rounding)
                .and_then(|r| i8::try_from(r).map_err(|_| "i8".to_string()))
        })
        .collect()
}

fn ctx_from_raw(ctx_raw: &[i32]) -> Result<Vec<i8>, String> {
    let spec = RequantSpec {
        multiplier: 1,
        shift: WEIGHT_BITS,
        output_bits: 8,
    };
    ctx_raw
        .iter()
        .map(|&v| {
            requant_i64(i64::from(v), spec, StageARoundingMode::TiesToEven)
                .and_then(|r| i8::try_from(r).map_err(|_| "i8".to_string()))
        })
        .collect()
}

fn residual_add(h_in: &[i16], o_raw: &[i32], policy: AttentionPolicy) -> Result<Vec<i16>, String> {
    h_in.iter()
        .zip(o_raw)
        .map(|(&h, &o)| {
            let o = requant_i64(i64::from(o), policy.output, policy.rounding)?;
            i16::try_from(i32::from(h) + o).map_err(|_| "residual exceeds i16".to_string())
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicStep {
    pub position: u64,
    pub h_in: Vec<i16>,
    pub q_raw: Vec<i32>,
    pub k_raw: Vec<i32>,
    pub v_raw: Vec<i32>,
    pub scores: Vec<Vec<i32>>,
    pub ctx_raw: Vec<i32>,
    pub o_raw: Vec<i32>,
    pub a: Vec<i16>,
    pub x: Vec<i16>,
}

impl PublicStep {
    pub fn is_committed_witness(&self) -> bool {
        self.q_raw.is_empty() && self.k_raw.is_empty() && self.o_raw.is_empty()
    }

    pub fn has_hidden_residuals(&self) -> bool {
        self.is_committed_witness()
            && (self.h_in.is_empty() || self.a.is_empty() || self.x.is_empty())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecomputedStep {
    pub x1: Vec<i8>,
    pub q_rot: Vec<i8>,
    pub k_rot: Vec<i8>,
    pub v_q: Vec<i8>,
    pub weights: Vec<Vec<u16>>,
    pub ctx: Vec<i8>,
    pub a: Vec<i16>,
    pub x: Vec<i16>,
}

pub fn check_public_step(
    shape: AttentionShape,
    policy: AttentionPolicy,
    gamma1: &[i16],
    gamma2: &[i16],
    step: &PublicStep,
) -> Result<RecomputedStep, String> {
    shape.validate()?;
    let d_lengths = [
        step.h_in.len(),
        step.q_raw.len(),
        step.ctx_raw.len(),
        step.o_raw.len(),
    ];
    let kv_lengths = [step.k_raw.len(), step.v_raw.len()];
    if d_lengths.iter().any(|&l| l != shape.dimension)
        || kv_lengths.iter().any(|&l| l != shape.kv_dim())
    {
        return Err("attention step vector length is not the dimension".into());
    }
    let context = step.position as usize + 1;
    if step.scores.len() != shape.heads {
        return Err("attention step must carry one score vector per head".into());
    }
    let x1: Vec<i8> = rms_norm(&step.h_in, gamma1, policy.norm1_fraction_bits, 8)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let q_rot = rope(&requant_to_i8(&step.q_raw, policy)?, shape, step.position)?;
    let k_rot = rope(&requant_to_i8(&step.k_raw, policy)?, shape, step.position)?;
    let v_q = requant_to_i8(&step.v_raw, policy)?;
    let mut weights = Vec::with_capacity(shape.heads);
    for s in &step.scores {
        if s.len() != context {
            return Err("score vector length is not the context length".into());
        }
        weights.push(softmax_weights(s, policy.score_shift)?);
    }
    let ctx = ctx_from_raw(&step.ctx_raw)?;
    let a = residual_add(&step.h_in, &step.o_raw, policy)?;
    let x: Vec<i16> = rms_norm(&a, gamma2, policy.norm2_fraction_bits, 12)?
        .into_iter()
        .map(|v| v as i16)
        .collect();
    Ok(RecomputedStep {
        x1,
        q_rot,
        k_rot,
        v_q,
        weights,
        ctx,
        a,
        x,
    })
}

#[derive(Clone, Debug)]
pub struct ProjectionProof {
    pub proof: SumcheckProof,
    pub weight_openings: Vec<SecureField>,
}

pub struct ProjectionOpeningClaims {
    pub point: Vec<SecureField>,
    pub openings: Vec<SecureField>,
}

#[inline(always)]
fn i128_mod_p(v: i128) -> BaseField {
    const P64: u64 = P as u64;
    let abs = v.unsigned_abs();
    let w0 = (abs as u64) & P64;
    let w1 = ((abs >> 31) as u64) & P64;
    let w2 = ((abs >> 62) as u64) & P64;
    let w3 = (abs >> 93) as u64;
    let mut s = w0 + w1 + w2 + (w3 & P64) + (w3 >> 31);
    s = (s & P64) + (s >> 31);
    s = (s & P64) + (s >> 31);
    let r = if s >= P64 { (s - P64) as u32 } else { s as u32 };
    let canonical = if v < 0 && r != 0 { P - r } else { r };
    BaseField::from_u32_unchecked(canonical)
}

fn acc_i128_to_secure(a: [i128; 4]) -> SecureField {
    SecureField::from_m31_array([
        i128_mod_p(a[0]),
        i128_mod_p(a[1]),
        i128_mod_p(a[2]),
        i128_mod_p(a[3]),
    ])
}

fn dot_i32_eq_i64(eq_w: &[[i64; 4]], values: &[i32]) -> SecureField {
    let mut a = [0i128; 4];
    for (w, &v) in eq_w.iter().zip(values) {
        let v = i128::from(v);
        a[0] += i128::from(w[0]) * v;
        a[1] += i128::from(w[1]) * v;
        a[2] += i128::from(w[2]) * v;
        a[3] += i128::from(w[3]) * v;
    }
    acc_i128_to_secure(a)
}

fn dot_u16_eq_i64(eq_w: &[[i64; 4]], values: &[u16]) -> SecureField {
    let mut a = [0i64; 4];
    for (w, &v) in eq_w.iter().zip(values) {
        let v = i64::from(v);
        a[0] += w[0] * v;
        a[1] += w[1] * v;
        a[2] += w[2] * v;
        a[3] += w[3] * v;
    }
    acc_to_secure(a)
}

fn projection_claims_fast(
    outputs: &[&[Vec<i32>]],
    r_tokens: &[SecureField],
    r_out: &[SecureField],
) -> Vec<SecureField> {
    let tok_w = eq_evals(r_tokens);
    let out_w = eq_evals_i64(r_out);
    outputs
        .iter()
        .map(|matrix| {
            matrix
                .iter()
                .zip(&tok_w)
                .map(|(row, &tw)| tw * dot_i32_eq_i64(&out_w, row))
                .sum()
        })
        .collect()
}

fn restrict_inputs(inputs: &[Vec<i8>], r_tokens: &[SecureField]) -> Vec<SecureField> {
    let mut out = restrict_rows_i8_fast(inputs, r_tokens);
    let padded = out.len().next_power_of_two();
    if out.len() < padded {
        out.resize(padded, SecureField::from(i8_to_m31(0)));
    }
    out
}

pub fn eq_at_index(point: &[SecureField], index: usize) -> SecureField {
    let one = SecureField::from(i8_to_m31(1));
    let n = point.len();
    let mut acc = one;
    for (j, &p) in point.iter().enumerate() {
        let bit = (index >> (n - 1 - j)) & 1;
        acc *= if bit == 1 { p } else { one - p };
    }
    acc
}

fn projection_dims(
    inputs: &[Vec<i8>],
    outputs: &[&[Vec<i32>]],
) -> Result<(usize, usize, usize), String> {
    let tokens = inputs.len();
    let dimension = inputs.first().map_or(0, Vec::len);
    if !tokens.is_power_of_two() || dimension < 2 {
        return Err("projection batch must be a power of two and dimension >= 2".into());
    }
    if inputs.iter().any(|row| row.len() != dimension) {
        return Err("projection inputs must be tokens x dimension".into());
    }
    if outputs.is_empty() {
        return Err("projection needs at least one matrix".into());
    }
    let max_cols = outputs
        .iter()
        .map(|out| out.first().map_or(0, Vec::len))
        .max()
        .unwrap_or(0);
    if max_cols < 2 {
        return Err("projection outputs must be tokens x dimension".into());
    }
    for out in outputs {
        let cols = out.first().map_or(0, Vec::len);
        if out.len() != tokens
            || cols < 2
            || cols > max_cols
            || out.iter().any(|row| row.len() != cols)
        {
            return Err("projection outputs must be tokens x dimension".into());
        }
    }
    let dim_padded = dimension.next_power_of_two();
    let cols_padded = max_cols.next_power_of_two();
    Ok((
        tokens.ilog2() as usize,
        dim_padded.ilog2() as usize,
        cols_padded.ilog2() as usize,
    ))
}

fn draw_points(
    channel: &mut Blake2sChannel,
    log_tokens: usize,
    log_out: usize,
) -> (Vec<SecureField>, Vec<SecureField>) {
    (
        channel.draw_secure_felts(log_tokens),
        channel.draw_secure_felts(log_out),
    )
}

fn combine(values: &[SecureField], lambda: SecureField) -> SecureField {
    values
        .iter()
        .rev()
        .fold(SecureField::from(i8_to_m31(0)), |acc, &v| acc * lambda + v)
}

pub fn verify_projections(
    inputs: &[Vec<i8>],
    outputs: &[&[Vec<i32>]],
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<ProjectionOpeningClaims, String> {
    let (log_tokens, log_dimension, log_out) = projection_dims(inputs, outputs)?;
    if proof.proof.round_polys.len() != log_dimension
        || proof.weight_openings.len() != outputs.len()
    {
        return Err("projection sum-check shape mismatch".into());
    }
    let (r_tokens, r_out) = draw_points(channel, log_tokens, log_out);
    let claims = projection_claims_fast(outputs, &r_tokens, &r_out);
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s_dimension, final_eval) =
        partially_verify(combine(&claims, lambda), &proof.proof, channel)?;
    if s_dimension.len() != log_dimension {
        return Err("projection sum-check assignment length mismatch".into());
    }
    let x_opening = mle_at(&restrict_inputs(inputs, &r_tokens), &s_dimension);
    if final_eval != x_opening * combine(&proof.weight_openings, lambda) {
        return Err("projection sum-check final check failed".into());
    }
    channel.mix_felts(&proof.weight_openings);
    let mut point = s_dimension;
    point.extend_from_slice(&r_out);
    Ok(ProjectionOpeningClaims {
        point,
        openings: proof.weight_openings.clone(),
    })
}

pub struct ProjectionGkrClaims {
    pub opening_claims: ProjectionOpeningClaims,
    pub r_tokens: Vec<SecureField>,
    pub s_dimension: Vec<SecureField>,
    pub r_out: Vec<SecureField>,
    pub lambda: SecureField,
}

pub fn verify_projections_gkr(
    tokens: usize,
    in_dim: usize,
    max_out_cols: usize,
    output_claims: &[SecureField],
    input_opening: SecureField,
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<ProjectionGkrClaims, String> {
    verify_projections_gkr_with_zk(
        tokens,
        in_dim,
        max_out_cols,
        output_claims,
        input_opening,
        proof,
        channel,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_projections_gkr_with_zk(
    tokens: usize,
    in_dim: usize,
    max_out_cols: usize,
    output_claims: &[SecureField],
    input_opening: SecureField,
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
    zk_cfs: Option<(SecureField, SecureField)>,
) -> Result<ProjectionGkrClaims, String> {
    if !tokens.is_power_of_two() || in_dim < 2 || max_out_cols < 2 || output_claims.is_empty() {
        return Err("projection batch must be a power of two and dimension >= 2".into());
    }
    let log_tokens = tokens.ilog2() as usize;
    let log_dimension = in_dim.next_power_of_two().ilog2() as usize;
    let log_out = max_out_cols.next_power_of_two().ilog2() as usize;
    if proof.proof.round_polys.len() != log_dimension
        || proof.weight_openings.len() != output_claims.len()
    {
        return Err("projection sum-check shape mismatch".into());
    }
    let (r_tokens, r_out) = draw_points(channel, log_tokens, log_out);
    channel.mix_felts(output_claims);
    let lambda = channel.draw_secure_felt();
    let init_claim = match zk_cfs {
        Some((cfs_masked_sum, _)) => {
            channel.mix_felts(&[cfs_masked_sum]);
            combine(output_claims, lambda) + cfs_masked_sum
        }
        None => combine(output_claims, lambda),
    };
    let (s_dimension, final_eval) = partially_verify(init_claim, &proof.proof, channel)?;
    if s_dimension.len() != log_dimension {
        return Err("projection sum-check assignment length mismatch".into());
    }
    channel.mix_felts(&proof.weight_openings);
    let expected_final = match zk_cfs {
        Some((_, cfs_masked_eval)) => {
            channel.mix_felts(&[cfs_masked_eval]);
            input_opening * combine(&proof.weight_openings, lambda) + cfs_masked_eval
        }
        None => input_opening * combine(&proof.weight_openings, lambda),
    };
    if final_eval != expected_final {
        return Err("projection sum-check final check failed".into());
    }
    let mut point = s_dimension.clone();
    point.extend_from_slice(&r_out);
    Ok(ProjectionGkrClaims {
        opening_claims: ProjectionOpeningClaims {
            point,
            openings: proof.weight_openings.clone(),
        },
        r_tokens,
        s_dimension,
        r_out,
        lambda,
    })
}

pub fn verify_onehot_projection(
    token_ids: &[u32],
    vocab_size: usize,
    output: &[Vec<i32>],
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<ProjectionOpeningClaims, String> {
    let tokens = token_ids.len();
    let dimension = output.first().map_or(0, Vec::len);
    if !tokens.is_power_of_two() || vocab_size < 2 || dimension < 2 {
        return Err("onehot projection requires power-of-two batch and vocab_size, dimension >= 2".into());
    }
    if output.len() != tokens || output.iter().any(|r| r.len() != dimension) {
        return Err("onehot projection output shape mismatch".into());
    }
    if token_ids.iter().any(|&id| (id as usize) >= vocab_size) {
        return Err("onehot projection token_id exceeds vocab_size".into());
    }
    let vocab_padded = vocab_size.next_power_of_two();
    let dim_padded = dimension.next_power_of_two();
    let log_tokens = tokens.ilog2() as usize;
    let log_vocab = vocab_padded.ilog2() as usize;
    let log_d = dim_padded.ilog2() as usize;
    if proof.proof.round_polys.len() != log_vocab || proof.weight_openings.len() != 1 {
        return Err("onehot projection sum-check shape mismatch".into());
    }
    let (r_tokens, r_d) = draw_points(channel, log_tokens, log_d);
    let claims = projection_claims_fast(&[output], &r_tokens, &r_d);
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s_vocab, final_eval) = partially_verify(combine(&claims, lambda), &proof.proof, channel)?;
    if s_vocab.len() != log_vocab {
        return Err("onehot projection sum-check assignment length mismatch".into());
    }
    let tok_eq = eq_evals(&r_tokens);
    let x_opening: SecureField = token_ids
        .iter()
        .zip(&tok_eq)
        .map(|(&id, &tw)| tw * eq_at_index(&s_vocab, id as usize))
        .sum();
    if final_eval != x_opening * combine(&proof.weight_openings, lambda) {
        return Err("onehot projection sum-check final check failed".into());
    }
    channel.mix_felts(&proof.weight_openings);
    let mut point = s_vocab;
    point.extend_from_slice(&r_d);
    Ok(ProjectionOpeningClaims {
        point,
        openings: proof.weight_openings.clone(),
    })
}

pub fn verify_transposed_projection(
    inputs: &[Vec<i8>],
    output: &[Vec<i32>],
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<ProjectionOpeningClaims, String> {
    let (log_tokens, log_dimension, log_vocab) = projection_dims(inputs, &[output])?;
    if proof.proof.round_polys.len() != log_dimension || proof.weight_openings.len() != 1 {
        return Err("transposed projection sum-check shape mismatch".into());
    }
    let (r_tokens, r_vocab) = draw_points(channel, log_tokens, log_vocab);
    let claims = projection_claims_fast(&[output], &r_tokens, &r_vocab);
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s_dimension, final_eval) =
        partially_verify(combine(&claims, lambda), &proof.proof, channel)?;
    if s_dimension.len() != log_dimension {
        return Err("transposed projection sum-check assignment length mismatch".into());
    }
    let x_opening = mle_at(&restrict_inputs(inputs, &r_tokens), &s_dimension);
    if final_eval != x_opening * combine(&proof.weight_openings, lambda) {
        return Err("transposed projection sum-check final check failed".into());
    }
    channel.mix_felts(&proof.weight_openings);
    let mut point = r_vocab;
    point.extend_from_slice(&s_dimension);
    Ok(ProjectionOpeningClaims {
        point,
        openings: proof.weight_openings.clone(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheLayout {
    pub chunk: usize,
}

impl CacheLayout {
    pub fn validate(self) -> Result<(), String> {
        if self.chunk < 2 || !self.chunk.is_power_of_two() {
            return Err("cache chunk must be a power of two of at least two".into());
        }
        Ok(())
    }

    pub fn padded_positions(self, rows: usize) -> usize {
        rows.max(1).next_power_of_two().max(2 * self.chunk)
    }

    pub fn full_chunks(self, rows: usize) -> usize {
        rows.saturating_sub(1) / self.chunk
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheOpeningClaim {
    pub point: Vec<SecureField>,
    pub log_chunks: usize,
    pub full_chunk_openings: Vec<SecureField>,
    pub value: SecureField,
}

#[derive(Clone, Debug)]
pub struct ContractionProof {
    pub proof: SumcheckProof,
    pub claims: Vec<CacheOpeningClaim>,
}

fn log_chunks_of(layout: CacheLayout, padded_rows: usize) -> usize {
    padded_rows.ilog2() as usize - layout.chunk.ilog2() as usize
}

fn partial_chunk_mle_at(
    partial_rows: &[Vec<i8>],
    full_chunks: usize,
    layout: CacheLayout,
    padded_rows: usize,
    point: &[SecureField],
    chunk_weights: &[SecureField],
) -> SecureField {
    if partial_rows.is_empty() {
        return SecureField::from(i8_to_m31(0));
    }
    let log_rows = padded_rows.ilog2() as usize;
    let log_chunks = log_chunks_of(layout, padded_rows);
    let within_weights = eq_evals(&point[log_chunks..log_rows]);
    let row_dots = restrict_cols_i8_fast(partial_rows, &point[log_rows..]);
    let within_sum: SecureField = row_dots
        .iter()
        .zip(&within_weights)
        .map(|(&d, &w)| d * w)
        .sum();
    chunk_weights[full_chunks] * within_sum
}

pub fn verify_cache_claim(
    partial_rows: &[Vec<i8>],
    full_chunks: usize,
    layout: CacheLayout,
    padded_rows: usize,
    claim: &CacheOpeningClaim,
) -> Result<(), String> {
    if partial_rows.is_empty() || partial_rows.len() > layout.chunk {
        return Err("partial chunk row count out of range".into());
    }
    if claim.full_chunk_openings.len() != full_chunks {
        return Err("cache claim does not open every full chunk".into());
    }
    let log_chunks = log_chunks_of(layout, padded_rows);
    if claim.log_chunks != log_chunks {
        return Err("cache claim chunk coordinate count mismatch".into());
    }
    let chunk_weights = eq_evals(&claim.point[..log_chunks]);
    let mut value = partial_chunk_mle_at(
        partial_rows,
        full_chunks,
        layout,
        padded_rows,
        &claim.point,
        &chunk_weights,
    );
    for (c, &opening) in claim.full_chunk_openings.iter().enumerate() {
        value += chunk_weights[c] * opening;
    }
    if value != claim.value {
        return Err("cache openings do not recombine to the claimed cache value".into());
    }
    Ok(())
}

pub fn verify_cache_claims_batch(
    chunk_ids_per_token: &[&[usize]],
    partial_rows_per_token: &[&[Vec<i8>]],
    layout: CacheLayout,
    padded_rows: usize,
    claims: &[CacheOpeningClaim],
) -> Result<(), String> {
    let b = claims.len();
    if chunk_ids_per_token.len() != b || partial_rows_per_token.len() != b {
        return Err("cache claim batch length mismatch".into());
    }
    if b == 0 {
        return Ok(());
    }
    let log_rows = padded_rows.ilog2() as usize;
    let log_chunks = log_chunks_of(layout, padded_rows);
    let shared_point = &claims[0].point;
    if shared_point.len() < log_rows {
        return Err("cache claim point shorter than log_rows".into());
    }
    if claims.iter().any(|c| c.point != *shared_point) {
        for t in 0..b {
            verify_cache_claim(
                partial_rows_per_token[t],
                chunk_ids_per_token[t].len(),
                layout,
                padded_rows,
                &claims[t],
            )?;
        }
        return Ok(());
    }
    let chunk_weights = eq_evals(&shared_point[..log_chunks]);
    let within_weights = eq_evals(&shared_point[log_chunks..log_rows]);
    let col_point = &shared_point[log_rows..];

    let mut groups: BTreeMap<&[usize], Vec<usize>> = BTreeMap::new();
    for t in 0..b {
        let rows = partial_rows_per_token[t];
        let full = chunk_ids_per_token[t].len();
        let claim = &claims[t];
        if rows.is_empty() || rows.len() > layout.chunk {
            return Err("partial chunk row count out of range".into());
        }
        if claim.full_chunk_openings.len() != full {
            return Err("cache claim does not open every full chunk".into());
        }
        if claim.log_chunks != log_chunks {
            return Err("cache claim chunk coordinate count mismatch".into());
        }
        groups.entry(chunk_ids_per_token[t]).or_default().push(t);
    }

    for (cids, idxs) in groups {
        let full = cids.len();
        let &t_max = idxs
            .iter()
            .max_by_key(|&&t| partial_rows_per_token[t].len())
            .unwrap();
        let max_rows = partial_rows_per_token[t_max];
        let row_dots = restrict_cols_i8_fast(max_rows, col_point);
        let mut prefix_sums = Vec::with_capacity(max_rows.len() + 1);
        let mut acc = SecureField::from(i8_to_m31(0));
        prefix_sums.push(acc);
        for (d, &w) in row_dots.iter().zip(&within_weights) {
            acc += *d * w;
            prefix_sums.push(acc);
        }
        for &t in &idxs {
            let claim = &claims[t];
            let r_t = partial_rows_per_token[t].len();
            let mut value = chunk_weights[full] * prefix_sums[r_t];
            for (c, &opening) in claim.full_chunk_openings.iter().enumerate() {
                value += chunk_weights[c] * opening;
            }
            if value != claim.value {
                return Err("cache openings do not recombine to the claimed cache value".into());
            }
        }
    }
    Ok(())
}

fn scores_claims_fast(
    scores: &[&[Vec<i32>]],
    r_heads: &[SecureField],
    r_rows: &[SecureField],
) -> Vec<SecureField> {
    let head_w = eq_evals(r_heads);
    let row_w = eq_evals_i64(r_rows);
    scores
        .iter()
        .map(|token_scores| {
            token_scores
                .iter()
                .zip(&head_w)
                .map(|(head_scores, &hw)| hw * dot_i32_eq_i64(&row_w, head_scores))
                .sum()
        })
        .collect()
}

fn scores_left_table_with_head_weights(
    q_rot: &[i8],
    shape: AttentionShape,
    head_weights: &[SecureField],
) -> Vec<SecureField> {
    let group = shape.group_size();
    let mut out = vec![SecureField::from(i8_to_m31(0)); shape.kv_dim()];
    for head in 0..shape.heads {
        let hw = head_weights[head];
        let kv_base = (head / group) * shape.head_dim;
        let q_base = head * shape.head_dim;
        for i in 0..shape.head_dim {
            out[kv_base + i] += hw * i8_to_m31(q_rot[q_base + i]);
        }
    }
    out
}

fn context_left_evals_fast(
    weights: &[&[Vec<u16>]],
    shape: AttentionShape,
    r_heads: &[SecureField],
    s_kv_heads: &[SecureField],
    s_rows: &[SecureField],
) -> Vec<SecureField> {
    let group = shape.group_size();
    let s_kv_eq = eq_evals(s_kv_heads);
    let head_factors: Vec<SecureField> = eq_evals(r_heads)
        .into_iter()
        .enumerate()
        .map(|(h, r)| r * s_kv_eq[h / group])
        .collect();
    let row_w = eq_evals_i64(s_rows);
    weights
        .iter()
        .map(|token_weights| {
            token_weights
                .iter()
                .zip(&head_factors)
                .map(|(head_weights, &hf)| hf * dot_u16_eq_i64(&row_w, head_weights))
                .sum()
        })
        .collect()
}

pub fn verify_scores(
    shape: AttentionShape,
    layout: CacheLayout,
    q_rot: &[&[i8]],
    cache_lengths: &[usize],
    scores: &[&[Vec<i32>]],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
) -> Result<(), String> {
    let tokens = q_rot.len();
    if tokens == 0
        || cache_lengths.len() != tokens
        || scores.len() != tokens
        || proof.claims.len() != tokens
    {
        return Err("scores proof shape mismatch".into());
    }
    let log_kv_dim = shape.kv_dim().ilog2() as usize;
    if proof.proof.round_polys.len() != log_kv_dim {
        return Err("scores sum-check round count mismatch".into());
    }
    for t in 0..tokens {
        if q_rot[t].len() != shape.dimension
            || scores[t].len() != shape.heads
            || scores[t].iter().any(|s| s.len() != cache_lengths[t])
        {
            return Err("scores must be heads x cache length over a query of the dimension".into());
        }
    }
    let padded_all = cache_lengths
        .iter()
        .map(|&c| layout.padded_positions(c))
        .max()
        .unwrap();
    let r_heads = channel.draw_secure_felts(shape.heads.ilog2() as usize);
    let r_rows = channel.draw_secure_felts(padded_all.ilog2() as usize);
    let claims = scores_claims_fast(scores, &r_heads, &r_rows);
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s_cols, final_eval) =
        partially_verify(combine(&claims, lambda), &proof.proof, channel)?;
    if s_cols.len() != log_kv_dim {
        return Err("scores sum-check assignment length mismatch".into());
    }
    let mut point = r_rows;
    point.extend_from_slice(&s_cols);
    for t in 0..tokens {
        if proof.claims[t].point != point {
            return Err("scores cache claim is not at the transcript's point".into());
        }
    }
    let head_weights = eq_evals(&r_heads);
    let col_weights = eq_evals(&s_cols);
    let lefts: Vec<SecureField> = q_rot
        .iter()
        .map(|q| {
            let left_table = scores_left_table_with_head_weights(q, shape, &head_weights);
            col_weights
                .iter()
                .zip(&left_table)
                .map(|(&w, &v)| w * v)
                .sum()
        })
        .collect();
    let mut expected = Vec::with_capacity(tokens);
    for t in 0..tokens {
        expected.push(lefts[t] * proof.claims[t].value);
        channel.mix_felts(&[proof.claims[t].value]);
        channel.mix_felts(&proof.claims[t].full_chunk_openings);
    }
    if final_eval != combine(&expected, lambda) {
        return Err("scores sum-check final check failed".into());
    }
    Ok(())
}

pub struct ScoresGkrClaims {
    pub r_heads: Vec<SecureField>,
    pub s_cols: Vec<SecureField>,
    pub token_alphas: Vec<SecureField>,
    pub final_eval: SecureField,
}

pub fn scores_left_weight_mle_at(
    shape: AttentionShape,
    r_heads: &[SecureField],
    s_cols: &[SecureField],
    r_i_star: &[SecureField],
) -> SecureField {
    let log_kv_heads = shape.kv_heads.ilog2() as usize;
    let log_group = shape.group_size().ilog2() as usize;
    let (r_kv_h, r_grp) = r_heads.split_at(log_kv_heads);
    let (s_kv_h, s_hd) = s_cols.split_at(log_kv_heads);
    let (ri_kv_h, rest) = r_i_star.split_at(log_kv_heads);
    let (ri_grp, ri_hd) = rest.split_at(log_group);
    let one = SecureField::from(i8_to_m31(1));
    let mut acc = one;
    for b in 0..log_kv_heads {
        let w0 = (one - r_kv_h[b]) * (one - s_kv_h[b]);
        let w1 = r_kv_h[b] * s_kv_h[b];
        acc *= w0 * (one - ri_kv_h[b]) + w1 * ri_kv_h[b];
    }
    for b in 0..log_group {
        acc *= (one - r_grp[b]) * (one - ri_grp[b]) + r_grp[b] * ri_grp[b];
    }
    for b in 0..ri_hd.len() {
        acc *= (one - s_hd[b]) * (one - ri_hd[b]) + s_hd[b] * ri_hd[b];
    }
    acc
}

pub fn verify_scores_gkr(
    shape: AttentionShape,
    layout: CacheLayout,
    cache_lengths: &[usize],
    scores_claims: &[SecureField],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
) -> Result<ScoresGkrClaims, String> {
    verify_scores_gkr_with_zk(
        shape,
        layout,
        cache_lengths,
        scores_claims,
        proof,
        channel,
        None,
    )
}

pub fn verify_scores_gkr_with_zk(
    shape: AttentionShape,
    layout: CacheLayout,
    cache_lengths: &[usize],
    scores_claims: &[SecureField],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
    zk_cfs: Option<(SecureField, SecureField)>,
) -> Result<ScoresGkrClaims, String> {
    let tokens = cache_lengths.len();
    let expected_claims_len = if zk_cfs.is_some() { 1 } else { tokens };
    if tokens == 0 || scores_claims.len() != expected_claims_len || proof.claims.len() != tokens {
        return Err("scores proof shape mismatch".into());
    }
    let log_kv_dim = shape.kv_dim().ilog2() as usize;
    if proof.proof.round_polys.len() != log_kv_dim {
        return Err("scores sum-check round count mismatch".into());
    }
    let padded_all = cache_lengths
        .iter()
        .map(|&c| layout.padded_positions(c))
        .max()
        .unwrap();
    let r_heads = channel.draw_secure_felts(shape.heads.ilog2() as usize);
    let r_rows = channel.draw_secure_felts(padded_all.ilog2() as usize);
    let (lambda, init_claim) = match zk_cfs {
        Some((cfs_masked_sum, _)) => {
            let lam = channel.draw_secure_felt();
            channel.mix_felts(&[scores_claims[0], cfs_masked_sum]);
            (lam, scores_claims[0] + cfs_masked_sum)
        }
        None => {
            channel.mix_felts(scores_claims);
            let lam = channel.draw_secure_felt();
            (lam, combine(scores_claims, lam))
        }
    };
    let (s_cols, final_eval) = partially_verify(init_claim, &proof.proof, channel)?;
    if s_cols.len() != log_kv_dim {
        return Err("scores sum-check assignment length mismatch".into());
    }
    let mut point = r_rows;
    point.extend_from_slice(&s_cols);
    let mut token_alphas = Vec::with_capacity(tokens);
    let mut pow = SecureField::from(i8_to_m31(1));
    for t in 0..tokens {
        if proof.claims[t].point != point {
            return Err("scores cache claim is not at the transcript's point".into());
        }
        token_alphas.push(pow * proof.claims[t].value);
        pow *= lambda;
        channel.mix_felts(&[proof.claims[t].value]);
        channel.mix_felts(&proof.claims[t].full_chunk_openings);
    }
    let reported_final_eval = match zk_cfs {
        Some((_, cfs_masked_eval)) => {
            channel.mix_felts(&[cfs_masked_eval]);
            final_eval - cfs_masked_eval
        }
        None => final_eval,
    };
    Ok(ScoresGkrClaims {
        r_heads,
        s_cols,
        token_alphas,
        final_eval: reported_final_eval,
    })
}

pub fn verify_context(
    shape: AttentionShape,
    layout: CacheLayout,
    weights: &[&[Vec<u16>]],
    cache_lengths: &[usize],
    ctx_raw: &[&[i32]],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
) -> Result<(), String> {
    let tokens = weights.len();
    if tokens == 0
        || cache_lengths.len() != tokens
        || ctx_raw.len() != tokens
        || proof.claims.len() != tokens
    {
        return Err("context proof shape mismatch".into());
    }
    let log_heads = shape.heads.ilog2() as usize;
    let log_kv_heads = shape.kv_heads.ilog2() as usize;
    let padded_all = cache_lengths
        .iter()
        .map(|&c| layout.padded_positions(c))
        .max()
        .unwrap();
    let log_rows = padded_all.ilog2() as usize;
    if proof.proof.round_polys.len() != log_kv_heads + log_rows {
        return Err("context sum-check round count mismatch".into());
    }
    for t in 0..tokens {
        if weights[t].len() != shape.heads
            || weights[t].iter().any(|w| w.len() != cache_lengths[t])
            || ctx_raw[t].len() != shape.dimension
        {
            return Err("weights must be heads x cache length and the context the dimension".into());
        }
    }
    let r_heads = channel.draw_secure_felts(log_heads);
    let r_i = channel.draw_secure_felts(shape.head_dim.ilog2() as usize);
    let mut claim_point = r_heads.clone();
    claim_point.extend_from_slice(&r_i);
    let claim_weights = eq_evals_i64(&claim_point);
    let claims: Vec<SecureField> = ctx_raw
        .iter()
        .map(|c| dot_i32_eq_i64(&claim_weights, c))
        .collect();
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s, final_eval) = partially_verify(combine(&claims, lambda), &proof.proof, channel)?;
    if s.len() != log_kv_heads + log_rows {
        return Err("context sum-check assignment length mismatch".into());
    }
    let (s_kv_heads, s_rows) = s.split_at(log_kv_heads);
    let mut point = s_rows.to_vec();
    point.extend_from_slice(s_kv_heads);
    point.extend_from_slice(&r_i);
    for t in 0..tokens {
        if proof.claims[t].point != point {
            return Err("context cache claim is not at the transcript's point".into());
        }
    }
    let lefts = context_left_evals_fast(weights, shape, &r_heads, s_kv_heads, s_rows);
    let mut expected = Vec::with_capacity(tokens);
    for t in 0..tokens {
        expected.push(lefts[t] * proof.claims[t].value);
        channel.mix_felts(&[proof.claims[t].value]);
        channel.mix_felts(&proof.claims[t].full_chunk_openings);
    }
    if final_eval != combine(&expected, lambda) {
        return Err("context sum-check final check failed".into());
    }
    Ok(())
}

pub struct ContextGkrClaims {
    pub r_heads: Vec<SecureField>,
    pub r_i: Vec<SecureField>,
    pub s_point: Vec<SecureField>,
    pub lambda_ctx: SecureField,
}

pub fn verify_context_gkr(
    shape: AttentionShape,
    layout: CacheLayout,
    cache_lengths: &[usize],
    context_claims: &[SecureField],
    context_left_openings: &[SecureField],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
) -> Result<ContextGkrClaims, String> {
    verify_context_gkr_with_zk(
        shape,
        layout,
        cache_lengths,
        context_claims,
        context_left_openings,
        proof,
        channel,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_context_gkr_with_zk(
    shape: AttentionShape,
    layout: CacheLayout,
    cache_lengths: &[usize],
    context_claims: &[SecureField],
    context_left_openings: &[SecureField],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
    zk_cfs: Option<(SecureField, SecureField)>,
) -> Result<ContextGkrClaims, String> {
    let tokens = cache_lengths.len();
    let expected_claims_len = if zk_cfs.is_some() { 1 } else { tokens };
    if tokens == 0
        || context_claims.len() != expected_claims_len
        || context_left_openings.len() != tokens
        || proof.claims.len() != tokens
    {
        return Err("context proof shape mismatch".into());
    }
    let log_heads = shape.heads.ilog2() as usize;
    let log_kv_heads = shape.kv_heads.ilog2() as usize;
    let padded_all = cache_lengths
        .iter()
        .map(|&c| layout.padded_positions(c))
        .max()
        .unwrap();
    let log_rows = padded_all.ilog2() as usize;
    if proof.proof.round_polys.len() != log_kv_heads + log_rows {
        return Err("context sum-check round count mismatch".into());
    }
    let r_heads = channel.draw_secure_felts(log_heads);
    let r_i = channel.draw_secure_felts(shape.head_dim.ilog2() as usize);
    let (lambda, init_claim) = match zk_cfs {
        Some((cfs_masked_sum, _)) => {
            let lam = channel.draw_secure_felt();
            channel.mix_felts(&[context_claims[0], cfs_masked_sum]);
            (lam, context_claims[0] + cfs_masked_sum)
        }
        None => {
            channel.mix_felts(context_claims);
            let lam = channel.draw_secure_felt();
            (lam, combine(context_claims, lam))
        }
    };
    let (s, final_eval) = partially_verify(init_claim, &proof.proof, channel)?;
    if s.len() != log_kv_heads + log_rows {
        return Err("context sum-check assignment length mismatch".into());
    }
    let (s_kv_heads, s_rows) = s.split_at(log_kv_heads);
    let mut point = s_rows.to_vec();
    point.extend_from_slice(s_kv_heads);
    point.extend_from_slice(&r_i);
    let mut expected = Vec::with_capacity(tokens);
    for t in 0..tokens {
        if proof.claims[t].point != point {
            return Err("context cache claim is not at the transcript's point".into());
        }
        expected.push(context_left_openings[t] * proof.claims[t].value);
        channel.mix_felts(&[proof.claims[t].value]);
        channel.mix_felts(&proof.claims[t].full_chunk_openings);
    }
    let expected_final = match zk_cfs {
        Some((_, cfs_masked_eval)) => {
            channel.mix_felts(&[cfs_masked_eval]);
            combine(&expected, lambda) + cfs_masked_eval
        }
        None => combine(&expected, lambda),
    };
    if final_eval != expected_final {
        return Err("context sum-check final check failed".into());
    }
    Ok(ContextGkrClaims {
        r_heads,
        r_i,
        s_point: s,
        lambda_ctx: lambda,
    })
}

pub fn group_params(n: usize, g: usize, profile: &SecurityProfile) -> Result<Params, String> {
    group_params_with_f(n, g, None, profile)
}

pub fn group_params_with_f(
    n: usize,
    g: usize,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    let m = n
        .checked_sub(2)
        .ok_or_else(|| "CPRS n must be >= 3".to_string())?;
    let (s, default_f, q, pow_bits) = cprs_profile_parameters(profile, m);
    let f = f_override.unwrap_or(default_f).clamp(1, m);
    Params::try_new(n, s, f, q, pow_bits, g).map_err(|e| e.to_string())
}

pub fn outer_digest(tag: &[u8], statement_digest: &[u8; 32]) -> Hash {
    let mut h = Blake2s256::new();
    h.update(OUTER_DOMAIN);
    h.update([tag.len() as u8]);
    h.update(tag);
    h.update(statement_digest);
    h.finalize().into()
}

pub fn cache_chunk_params(
    layout: CacheLayout,
    dimension: usize,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    cache_chunk_params_with_f(layout, dimension, None, profile)
}

pub fn cache_chunk_params_with_f(
    layout: CacheLayout,
    dimension: usize,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    layout.validate()?;
    if !dimension.is_power_of_two() {
        return Err("attention dimension must be a power of two".into());
    }
    group_params_with_f(
        (layout.chunk.ilog2() + dimension.ilog2()) as usize,
        2,
        f_override,
        profile,
    )
}

pub fn projection_leaf_params(
    rows: usize,
    cols: usize,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    if rows == 0 || cols == 0 {
        return Err("projection matrix dimensions must be positive".into());
    }
    let rows_padded = rows.next_power_of_two();
    let cols_padded = cols.next_power_of_two();
    let n = (rows_padded.ilog2() + cols_padded.ilog2()) as usize;
    group_params(n, 1, profile)
}

pub fn register_cache_chunk_commitment(
    k_rows: &[Vec<i8>],
    v_rows: &[Vec<i8>],
    layout: CacheLayout,
    params: &Params,
) -> Result<Hash, String> {
    if k_rows.len() != layout.chunk || v_rows.len() != layout.chunk {
        return Err("a cache chunk has exactly `chunk` rows".into());
    }
    let cols = k_rows[0].len();
    if k_rows.iter().chain(v_rows).any(|r| r.len() != cols) {
        return Err("cache rows must all have the dimension".into());
    }
    let rows_padded = layout.chunk;
    let cols_padded = cols.next_power_of_two();
    let pad = |rows: &[Vec<i8>]| -> Vec<u32> {
        let mut out = vec![0u32; rows_padded * cols_padded];
        for (r, row) in rows.iter().enumerate() {
            let base = r * cols_padded;
            for (c, &v) in row.iter().enumerate() {
                out[base + c] = i8_to_canonical_m31_u32(v);
            }
        }
        out
    };
    let k_u32 = pad(k_rows);
    let v_u32 = pad(v_rows);
    cprs::register::register_group_commitment(&[&k_u32, &v_u32], params)
        .map_err(|e| e.to_string())
}

#[derive(Clone, Debug)]
pub struct CacheChunkOpening {
    pub chunk: usize,
    pub k_point: Vec<SecureField>,
    pub k_value: SecureField,
    pub v_point: Vec<SecureField>,
    pub v_value: SecureField,
}

fn leaf_points(qkv_point: &[SecureField], o_point: &[SecureField]) -> Vec<Vec<QM31>> {
    let qkv = stwo_point_to_cprs(qkv_point);
    vec![qkv.clone(), qkv.clone(), qkv, stwo_point_to_cprs(o_point)]
}

fn chunk_points_and_claims(openings: &[CacheChunkOpening]) -> (Vec<Vec<QM31>>, Vec<QM31>) {
    let mut points = Vec::with_capacity(2 * openings.len());
    let mut claims = Vec::with_capacity(2 * openings.len());
    for opening in openings {
        points.push(stwo_point_to_cprs(&opening.k_point));
        claims.push(to_cprs_qm31(opening.k_value));
        points.push(stwo_point_to_cprs(&opening.v_point));
        claims.push(to_cprs_qm31(opening.v_value));
    }
    (points, claims)
}

pub fn verify_attention_leaf_opening(
    commitment: &Hash,
    params: &Params,
    qkv_point: &[SecureField],
    o_point: &[SecureField],
    claims: &[SecureField; 4],
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    let points = leaf_points(qkv_point, o_point);
    let claims: Vec<QM31> = claims.iter().copied().map(to_cprs_qm31).collect();
    let outer = outer_digest(b"attention-leaf", statement_digest);
    verify_strict_context(&[*commitment], &points, &claims, &outer, params, proof)
        .map_err(|e| format!("Stage B attention leaf opening rejected: {e}"))
}

pub fn verify_cache_chunk_openings(
    commitments: &[Hash],
    openings: &[CacheChunkOpening],
    params: &Params,
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    if openings.is_empty() {
        return if proof.is_empty() {
            Ok(())
        } else {
            Err("cache chunk proof without openings".into())
        };
    }
    let group_commitments = openings
        .iter()
        .map(|o| {
            commitments
                .get(o.chunk)
                .copied()
                .ok_or_else(|| "cache opening names an unknown chunk commitment".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (points, claims) = chunk_points_and_claims(openings);
    let outer = outer_digest(b"kv-cache-chunks", statement_digest);
    verify_strict_context(&group_commitments, &points, &claims, &outer, params, proof)
        .map_err(|e| format!("Stage B cache chunk opening rejected: {e}"))
}

pub fn stage_b_witness_n_with_queries(
    batch: usize,
    dimension: usize,
    exp_queries: usize,
) -> usize {
    let b_rows = batch
        .next_power_of_two()
        .saturating_mul(dimension.next_power_of_two());
    let min_rows = b_rows.max(exp_queries).max(512);
    min_rows.next_power_of_two().ilog2() as usize
}

pub fn stage_b_zk_cfs_log_rows(batch: usize) -> usize {
    let log_b = batch.max(1).next_power_of_two().ilog2() as usize;
    13usize.max(log_b + 3)
}

pub fn stage_b_witness_n_for_zk(
    batch: usize,
    dimension: usize,
    exp_queries: usize,
) -> usize {
    let min_base = stage_b_zk_cfs_log_rows(batch) + 2;
    stage_b_witness_n_with_queries(batch, dimension, exp_queries).max(min_base) + 1
}

pub fn stage_b_witness_n(batch: usize, dimension: usize) -> usize {
    stage_b_witness_n_with_queries(batch, dimension, 0)
}

pub fn stage_b_witness_params_with_queries(
    batch: usize,
    dimension: usize,
    exp_queries: usize,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    stage_b_witness_params_with_queries_and_f(batch, dimension, exp_queries, None, profile)
}

pub fn stage_b_witness_params_with_queries_and_f(
    batch: usize,
    dimension: usize,
    exp_queries: usize,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    let n = stage_b_witness_n_with_queries(batch, dimension, exp_queries);
    let m = n
        .checked_sub(2)
        .ok_or_else(|| "CPRS witness n must be >= 3".to_string())?;
    let (s, default_f, q, pow_bits) = match profile {
        SecurityProfile::Production100 | SecurityProfile::Dev13 => {
            cprs_profile_parameters(profile, m)
        }
        _ => (2, 4usize.min(m).max(1), 32, 0),
    };
    let f = f_override.unwrap_or(default_f).clamp(1, m);
    Params::try_new(n, s, f, q, pow_bits, STAGE_B_WITNESS_COLS).map_err(|e| e.to_string())
}

pub fn stage_b_witness_params_for_zk(
    batch: usize,
    dimension: usize,
    exp_queries: usize,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    let n = stage_b_witness_n_for_zk(batch, dimension, exp_queries);
    let m = n
        .checked_sub(2)
        .ok_or_else(|| "CPRS ZK witness n must be >= 3".to_string())?;
    let (s, default_f, q, pow_bits) = match profile {
        SecurityProfile::Production100 | SecurityProfile::Dev13 => {
            cprs_profile_parameters(profile, m)
        }
        _ => (2, 4usize.min(m).max(1), 32, 0),
    };
    let f = f_override.unwrap_or(default_f).clamp(1, m);
    Params::try_new(n, s, f, q, pow_bits, STAGE_B_WITNESS_COLS_ZK)
        .map(|p| p.with_lo_vars(0))
        .map_err(|e| e.to_string())
}

pub fn stage_b_witness_params(
    batch: usize,
    dimension: usize,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    stage_b_witness_params_with_queries(batch, dimension, 0, profile)
}

fn stage_b_exp_query_count(statement: &StageBStatement) -> usize {
    statement
        .tokens
        .iter()
        .map(|t| statement.shape.heads * (t.step.position as usize + 1))
        .sum()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageBZkDeferredOpenings {
    pub h_a_x_openings: [SecureField; 3],
    pub cfs_hi_point: Vec<SecureField>,
    pub blind_opening: SecureField,
}

#[derive(Clone, Debug)]
pub struct StageBDeferredWitnessOpening {
    pub witness_commitment: Hash,
    pub witness_params: Params,
    pub witness_point: Vec<SecureField>,
    pub witness_openings: [SecureField; STAGE_B_WITNESS_COLS],
    pub zk_extra_openings: Option<StageBZkDeferredOpenings>,
}

impl StageBDeferredWitnessOpening {
    pub fn cprs_points_and_claims(&self) -> (Vec<Vec<QM31>>, Vec<QM31>) {
        if let Some(ref zk) = self.zk_extra_openings {
            let m_ext = self.witness_point.len().saturating_sub(2);
            let omega_scale = zk_lo_basis_omega_scale();
            let w_pt_hi = stwo_point_to_cprs(&self.witness_point[..m_ext]);
            let mut pts = vec![w_pt_hi; STAGE_B_WITNESS_COLS_ZK_DATA];
            pts.push(stwo_point_to_cprs(&zk.cfs_hi_point));
            let mut cls: Vec<QM31> = self
                .witness_openings
                .iter()
                .chain(zk.h_a_x_openings.iter())
                .map(|&v| to_cprs_qm31(omega_scale * v))
                .collect();
            cls.push(to_cprs_qm31(omega_scale * zk.blind_opening));
            (pts, cls)
        } else {
            let w_pt = stwo_point_to_cprs(&self.witness_point);
            let pts = vec![w_pt; STAGE_B_WITNESS_COLS];
            let cls = self
                .witness_openings
                .iter()
                .copied()
                .map(to_cprs_qm31)
                .collect();
            (pts, cls)
        }
    }

    pub fn update_outer_hasher(&self, h: &mut Blake2s256) {
        h.update(self.witness_commitment);
        h.update((self.witness_point.len() as u64).to_le_bytes());
        for &coord in &self.witness_point {
            for m in coord.to_m31_array() {
                h.update(m.0.to_le_bytes());
            }
        }
        if let Some(ref zk) = self.zk_extra_openings {
            h.update(b"stage-b-zk-v3");
            h.update((zk.cfs_hi_point.len() as u64).to_le_bytes());
            for &coord in &zk.cfs_hi_point {
                for m in coord.to_m31_array() {
                    h.update(m.0.to_le_bytes());
                }
            }
        }
    }
}

pub fn verify_stage_b_witness_opening(
    commitment: &Hash,
    params: &Params,
    point: &[SecureField],
    openings: &[SecureField; STAGE_B_WITNESS_COLS],
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    let w = StageBDeferredWitnessOpening {
        witness_commitment: *commitment,
        witness_params: params.clone(),
        witness_point: point.to_vec(),
        witness_openings: *openings,
        zk_extra_openings: None,
    };
    verify_stage_b_witness_opening_deferred(&w, statement_digest, proof)
}

pub fn verify_stage_b_witness_opening_deferred(
    w: &StageBDeferredWitnessOpening,
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    let (points, claims) = w.cprs_points_and_claims();
    let mut h = Blake2s256::new();
    h.update(OUTER_DOMAIN);
    h.update([b"stage-b-witness".len() as u8]);
    h.update(b"stage-b-witness");
    h.update(statement_digest);
    w.update_outer_hasher(&mut h);
    let outer: Hash = h.finalize().into();
    verify_strict_context(
        &[w.witness_commitment],
        &points,
        &claims,
        &outer,
        &w.witness_params,
        proof,
    )
    .map_err(|e| format!("Stage B witness opening rejected: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_b_cprs_unified(
    attention_commitment: &Hash,
    leaf_params: &Params,
    chunk_commitments: &[Hash],
    openings: &[CacheChunkOpening],
    chunk_params: &Params,
    qkv_point: &[SecureField],
    o_point: &[SecureField],
    leaf_claims: &[SecureField; 4],
    witness_opening: Option<&StageBDeferredWitnessOpening>,
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    if openings.is_empty() && witness_opening.is_none() {
        return verify_attention_leaf_opening(
            attention_commitment,
            leaf_params,
            qkv_point,
            o_point,
            leaf_claims,
            statement_digest,
            proof,
        );
    }
    let extra_wit = usize::from(witness_opening.is_some());
    let mut commitments = Vec::with_capacity(1 + openings.len() + extra_wit);
    let mut group_params_list = Vec::with_capacity(1 + openings.len() + extra_wit);
    commitments.push(*attention_commitment);
    group_params_list.push(leaf_params.clone());
    for opening in openings {
        let c = chunk_commitments
            .get(opening.chunk)
            .copied()
            .ok_or_else(|| "cache opening names an unknown chunk commitment".to_string())?;
        commitments.push(c);
        group_params_list.push(chunk_params.clone());
    }
    let mut points = leaf_points(qkv_point, o_point);
    let mut claims: Vec<QM31> = leaf_claims.iter().copied().map(to_cprs_qm31).collect();
    let (chunk_pts, chunk_clms) = chunk_points_and_claims(openings);
    points.extend(chunk_pts);
    claims.extend(chunk_clms);
    if let Some(w) = witness_opening {
        commitments.push(w.witness_commitment);
        group_params_list.push(w.witness_params.clone());
        let (w_pts, w_cls) = w.cprs_points_and_claims();
        points.extend(w_pts);
        claims.extend(w_cls);
    }
    let outer = outer_digest(b"stage-b-layer-cprs", statement_digest);
    verify_heterogeneous_strict_context(
        &commitments,
        &group_params_list,
        &points,
        &claims,
        &outer,
        proof,
    )
    .map_err(|e| format!("Stage B unified CPRS opening rejected: {e}"))
}

pub fn verify_projection_leaf_opening(
    commitment: &Hash,
    params: &Params,
    tag: &[u8],
    point: &[SecureField],
    claim: SecureField,
    statement_digest: &[u8; 32],
    proof: &[u8],
) -> Result<(), String> {
    let points = vec![stwo_point_to_cprs(point)];
    let claims = vec![to_cprs_qm31(claim)];
    let outer = outer_digest(tag, statement_digest);
    verify_strict_context(&[*commitment], &points, &claims, &outer, params, proof)
        .map_err(|e| format!("registered projection opening rejected: {e}"))
}

pub fn stage_b_step_cache_commitment(
    chunk_commitments: &[Hash],
    k_rows: &[Vec<i8>],
    v_rows: &[Vec<i8>],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(STEP_CACHE_DOMAIN);
    h.update((chunk_commitments.len() as u64).to_le_bytes());
    for c in chunk_commitments {
        h.update(c);
    }
    update_i8_rows(&mut h, k_rows);
    update_i8_rows(&mut h, v_rows);
    h.finalize().into()
}

pub fn stage_b_cache_digest(
    chunk_commitments: &[Hash],
    partial_rows: &[(Vec<Vec<i8>>, Vec<Vec<i8>>)],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(CACHE_DIGEST_DOMAIN);
    h.update((chunk_commitments.len() as u64).to_le_bytes());
    for c in chunk_commitments {
        h.update(c);
    }
    h.update((partial_rows.len() as u64).to_le_bytes());
    for (k_rows, v_rows) in partial_rows {
        update_i8_rows(&mut h, k_rows);
        update_i8_rows(&mut h, v_rows);
    }
    h.finalize().into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageBToken {
    pub step: PublicStep,
    pub chunk_ids: Vec<usize>,
    pub cache_commitment_before: [u8; 32],
    pub cache_commitment_after: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageBStatement {
    pub shape: AttentionShape,
    pub layout: CacheLayout,
    pub policy_id: String,
    pub attention_commitment: Hash,
    pub gamma1: Vec<i16>,
    pub gamma2: Vec<i16>,
    pub chunk_commitments: Vec<Hash>,
    pub tokens: Vec<StageBToken>,
    pub partial_rows: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)>,
    pub binding: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct StageBZkWitnessAux {
    pub gkr_libra_sums: Vec<SecureField>,
    pub gkr_libra_evals: Vec<SecureField>,
    pub qkv_cfs: [SecureField; 2],
    pub scores_cfs: [SecureField; 2],
    pub ctx_cfs: [SecureField; 2],
    pub o_cfs: [SecureField; 2],
    pub gkr_mult_masked_claim: SecureField,
    pub wit_cfs: [SecureField; 2],
    pub h_a_x_openings: [SecureField; 3],
    pub cont_claims: Vec<SecureField>,
    pub cfs_pad_claim: SecureField,
    pub cfs_binding_sumcheck: SumcheckProof,
    pub blind_col_opening: SecureField,
}

#[derive(Clone, Debug)]
pub struct StageBGkrWitnessProof {
    pub witness_commitment: Hash,
    pub qkv_claims: [SecureField; 3],
    pub x1_opening: SecureField,
    pub scores_claims: Vec<SecureField>,
    pub context_claims: Vec<SecureField>,
    pub context_left_openings: Vec<SecureField>,
    pub o_claim: SecureField,
    pub ctx_opening: SecureField,
    pub gkr_proof: GkrBatchProof,
    pub witness_sumcheck: SumcheckProof,
    pub witness_openings: [SecureField; STAGE_B_WITNESS_COLS],
    pub zk_aux: Option<StageBZkWitnessAux>,
    pub cprs_witness_proof: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct StageBProof {
    pub qkv: ProjectionProof,
    pub scores: ContractionProof,
    pub context: ContractionProof,
    pub o: ProjectionProof,
    pub attention_leaf_proof: Vec<u8>,
    pub cache_chunk_proof: Vec<u8>,
    pub gkr_witness: Option<StageBGkrWitnessProof>,
}

#[inline]
fn update_i16(h: &mut Blake2s256, values: &[i16]) {
    h.update((values.len() as u64).to_le_bytes());
    let mut buf = [0u8; 2048];
    for chunk in values.chunks(1024) {
        let len = chunk.len() * 2;
        for (i, &v) in chunk.iter().enumerate() {
            buf[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
        h.update(&buf[..len]);
    }
}

#[inline]
fn update_i8_rows(h: &mut Blake2s256, rows: &[Vec<i8>]) {
    h.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        h.update((row.len() as u64).to_le_bytes());
        h.update(bytemuck_cast_i8(row));
    }
}

#[inline]
fn update_i32(h: &mut Blake2s256, values: &[i32]) {
    h.update((values.len() as u64).to_le_bytes());
    let mut buf = [0u8; 2048];
    for chunk in values.chunks(512) {
        let len = chunk.len() * 4;
        for (i, &v) in chunk.iter().enumerate() {
            buf[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        h.update(&buf[..len]);
    }
}

fn update_step(h: &mut Blake2s256, step: &PublicStep) {
    h.update(step.position.to_le_bytes());
    update_i16(h, &step.h_in);
    update_i32(h, &step.q_raw);
    update_i32(h, &step.k_raw);
    update_i32(h, &step.v_raw);
    h.update((step.scores.len() as u64).to_le_bytes());
    for head in &step.scores {
        update_i32(h, head);
    }
    update_i32(h, &step.ctx_raw);
    update_i32(h, &step.o_raw);
    if !step.a.is_empty() || !step.x.is_empty() {
        update_i16(h, &step.a);
        update_i16(h, &step.x);
    }
}

impl StageBStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(STATEMENT_DOMAIN);
        for value in [
            self.shape.dimension,
            self.shape.heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.layout.chunk,
        ] {
            h.update((value as u64).to_le_bytes());
        }
        h.update((self.policy_id.len() as u64).to_le_bytes());
        h.update(self.policy_id.as_bytes());
        h.update(self.attention_commitment);
        h.update(self.binding);
        update_i16(&mut h, &self.gamma1);
        update_i16(&mut h, &self.gamma2);
        h.update((self.chunk_commitments.len() as u64).to_le_bytes());
        for c in &self.chunk_commitments {
            h.update(c);
        }
        h.update((self.tokens.len() as u64).to_le_bytes());
        for (token, (k_rows, v_rows)) in self.tokens.iter().zip(&self.partial_rows) {
            update_step(&mut h, &token.step);
            h.update((token.chunk_ids.len() as u64).to_le_bytes());
            for &id in &token.chunk_ids {
                h.update((id as u64).to_le_bytes());
            }
            h.update(token.cache_commitment_before);
            h.update(token.cache_commitment_after);
            update_i8_rows(&mut h, k_rows);
            update_i8_rows(&mut h, v_rows);
        }
        h.finalize().into()
    }

    pub fn cache_digest(&self) -> [u8; 32] {
        stage_b_cache_digest(&self.chunk_commitments, &self.partial_rows)
    }

    pub fn channel_from_digest(digest: &[u8; 32]) -> Blake2sChannel {
        let words: [u64; 4] =
            std::array::from_fn(|i| u64::from_le_bytes(digest[8 * i..8 * i + 8].try_into().expect("8 bytes")));
        make_channel(&words)
    }

    fn cache_lengths(&self) -> Vec<usize> {
        self.tokens
            .iter()
            .map(|t| t.step.position as usize + 1)
            .collect()
    }

    pub fn validate(&self, policy: AttentionPolicy) -> Result<Vec<RecomputedStep>, String> {
        self.shape.validate()?;
        self.layout.validate()?;
        if self.policy_id != policy.id() {
            return Err(format!(
                "Stage B statement policy {} is not the verifier's {}",
                self.policy_id,
                policy.id()
            ));
        }
        if self.tokens.is_empty() || !self.tokens.len().is_power_of_two() {
            return Err("Stage B batch must be a power of two".into());
        }
        if self.partial_rows.len() != self.tokens.len()
            || self.gamma1.len() != self.shape.dimension
            || self.gamma2.len() != self.shape.dimension
        {
            return Err("Stage B statement shape mismatch".into());
        }
        for (token, (k_rows, v_rows)) in self.tokens.iter().zip(&self.partial_rows) {
            let len = token.step.position as usize + 1;
            let full = self.layout.full_chunks(len);
            if token.chunk_ids.len() != full
                || token
                    .chunk_ids
                    .iter()
                    .any(|&id| id >= self.chunk_commitments.len())
            {
                return Err("token's chunk ids do not cover its full chunks".into());
            }
            if k_rows.len() != len - full * self.layout.chunk || v_rows.len() != k_rows.len() {
                return Err(
                    "token's partial rows are not the rows after its last full chunk".into(),
                );
            }
        }
        let is_committed = self.tokens[0].step.is_committed_witness();
        if self
            .tokens
            .iter()
            .any(|t| t.step.is_committed_witness() != is_committed)
        {
            return Err("Stage B tokens must all share the same witness mode".into());
        }
        let recomputed: Vec<RecomputedStep> = if is_committed {
            let d = self.shape.dimension;
            let kv_d = self.shape.kv_dim();
            let has_h_in = !self.tokens[0].step.h_in.is_empty();
            let has_a = !self.tokens[0].step.a.is_empty();
            let has_x = !self.tokens[0].step.x.is_empty();
            let mut out = Vec::with_capacity(self.tokens.len());
            for (token, (k_rows, v_rows)) in self.tokens.iter().zip(&self.partial_rows) {
                let s = &token.step;
                let h_in_ok = if has_h_in { s.h_in.len() == d } else { s.h_in.is_empty() };
                let a_ok = if has_a { s.a.len() == d } else { s.a.is_empty() };
                let x_ok = if has_x { s.x.len() == d } else { s.x.is_empty() };
                if !h_in_ok
                    || !a_ok
                    || !x_ok
                    || !s.q_raw.is_empty()
                    || !s.k_raw.is_empty()
                    || !s.v_raw.is_empty()
                    || !s.scores.is_empty()
                    || !s.ctx_raw.is_empty()
                    || !s.o_raw.is_empty()
                {
                    return Err("committed Stage B step shape mismatch".into());
                }
                let k_rot = k_rows
                    .last()
                    .ok_or_else(|| "token partial_rows k_rows is empty".to_string())?;
                let v_q = v_rows
                    .last()
                    .ok_or_else(|| "token partial_rows v_rows is empty".to_string())?;
                if k_rot.len() != kv_d || v_q.len() != kv_d {
                    return Err("token partial_rows row length does not match kv_dim".into());
                }
                out.push(RecomputedStep {
                    x1: Vec::new(),
                    q_rot: Vec::new(),
                    k_rot: k_rot.clone(),
                    v_q: v_q.clone(),
                    weights: Vec::new(),
                    ctx: Vec::new(),
                    a: s.a.clone(),
                    x: s.x.clone(),
                });
            }
            out
        } else {
            self.tokens
                .iter()
                .map(|token| {
                    check_public_step(
                        self.shape,
                        policy,
                        &self.gamma1,
                        &self.gamma2,
                        &token.step,
                    )
                })
                .collect::<Result<_, _>>()?
        };
        let k_rots: Vec<&[i8]> = recomputed.iter().map(|r| r.k_rot.as_slice()).collect();
        let v_qs: Vec<&[i8]> = recomputed.iter().map(|r| r.v_q.as_slice()).collect();
        self.validate_partial_rows(&k_rots, &v_qs)?;
        Ok(recomputed)
    }

    fn validate_partial_rows(&self, k_rots: &[&[i8]], v_qs: &[&[i8]]) -> Result<(), String> {
        for (t, (k_rows, v_rows)) in self.partial_rows.iter().enumerate() {
            if k_rows.last().map(Vec::as_slice) != Some(k_rots[t])
                || v_rows.last().map(Vec::as_slice) != Some(v_qs[t])
            {
                return Err("token's own appended row is not the last partial row".into());
            }
        }
        let mut groups: BTreeMap<&[usize], Vec<usize>> = BTreeMap::new();
        for (t, token) in self.tokens.iter().enumerate() {
            groups
                .entry(token.chunk_ids.as_slice())
                .or_default()
                .push(t);
        }
        for (_, mut idxs) in groups {
            if idxs.len() < 2 {
                continue;
            }
            idxs.sort_by_key(|&t| (self.tokens[t].step.position, self.partial_rows[t].0.len()));
            for pair in idxs.windows(2) {
                let (lo, hi) = (pair[0], pair[1]);
                let r_lo = self.partial_rows[lo].0.len();
                let r_hi = self.partial_rows[hi].0.len();
                if r_lo >= r_hi {
                    return Err(
                        "tokens in the same partial chunk have inconsistent positions".into(),
                    );
                }
            }
            let max_hi = *idxs.last().unwrap();
            for (pos_idx, &lo) in idxs[..idxs.len() - 1].iter().enumerate() {
                let next_hi = idxs[pos_idx + 1];
                let r_lo = self.partial_rows[lo].0.len();
                for hi in [next_hi, max_hi] {
                    if self.partial_rows[hi].0[r_lo - 1].as_slice() != k_rots[lo]
                        || self.partial_rows[hi].1[r_lo - 1].as_slice() != v_qs[lo]
                    {
                        return Err(
                            "token's prefix partial row does not match an earlier batch token's proven row"
                                .into(),
                        );
                    }
                    if next_hi == max_hi {
                        break;
                    }
                }
            }
            for &lo in &idxs[..idxs.len() - 1] {
                let r_lo = self.partial_rows[lo].0.len();
                if self.partial_rows[max_hi].0[..r_lo] != self.partial_rows[lo].0[..]
                    || self.partial_rows[max_hi].1[..r_lo] != self.partial_rows[lo].1[..]
                {
                    return Err(
                        "tokens in the same partial chunk do not share a common prefix".into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate_chunk_rollovers(&self, chunk_params: &Params) -> Result<(), String> {
        let mut rollover_comms: Vec<Option<Hash>> = vec![None; self.tokens.len()];
        for (a, (k_rows, v_rows)) in self.partial_rows.iter().enumerate() {
            if k_rows.len() != self.layout.chunk {
                continue;
            }
            let full = self.tokens[a].chunk_ids.len();
            let comm =
                register_cache_chunk_commitment(k_rows, v_rows, self.layout, chunk_params)?;
            rollover_comms[a] = Some(comm);
            if full > 0 {
                for (b, token_b) in self.tokens.iter().enumerate() {
                    if b == a
                        || token_b.chunk_ids.len() <= full
                        || token_b.chunk_ids[..full] != self.tokens[a].chunk_ids[..]
                    {
                        continue;
                    }
                    if self.chunk_commitments[token_b.chunk_ids[full]] != comm {
                        return Err(
                            "rolled-over partial chunk does not match the next full-chunk commitment"
                                .into(),
                        );
                    }
                }
            } else {
                let candidates: Vec<usize> = self
                    .tokens
                    .iter()
                    .enumerate()
                    .filter(|&(b, tb)| b != a && !tb.chunk_ids.is_empty())
                    .map(|(_, tb)| tb.chunk_ids[0])
                    .collect();
                if !candidates.is_empty()
                    && !candidates
                        .iter()
                        .any(|&cid| self.chunk_commitments[cid] == comm)
                {
                    return Err(
                        "rolled-over partial chunk does not match the next full-chunk commitment"
                            .into(),
                    );
                }
            }
        }
        let mut prev_after: Option<[u8; 32]> = None;
        for (t, (token, (k_rows, v_rows))) in
            self.tokens.iter().zip(&self.partial_rows).enumerate()
        {
            let token_comms: Vec<Hash> = token
                .chunk_ids
                .iter()
                .map(|&id| self.chunk_commitments[id])
                .collect();
            let expected_before = if t > 0
                && prev_after.is_some()
                && rollover_comms[t - 1].is_none()
                && self.tokens[t - 1].chunk_ids == token.chunk_ids
                && k_rows.len() == self.partial_rows[t - 1].0.len() + 1
            {
                prev_after.unwrap()
            } else {
                stage_b_step_cache_commitment(
                    &token_comms,
                    &k_rows[..k_rows.len() - 1],
                    &v_rows[..v_rows.len() - 1],
                )
            };
            if token.cache_commitment_before != expected_before {
                return Err(
                    "token's cache_commitment_before does not match its pre-step KV-cache state"
                        .into(),
                );
            }
            if let Some(comm) = rollover_comms[t] {
                let mut post_rollover_comms = token_comms;
                post_rollover_comms.push(comm);
                let expected_after =
                    stage_b_step_cache_commitment(&post_rollover_comms, &[], &[]);
                if token.cache_commitment_after != expected_after {
                    return Err(
                        "rollover token's cache_commitment_after does not match registered rollover chunk commitment"
                            .into(),
                    );
                }
                prev_after = Some(expected_after);
            } else {
                let expected_after =
                    stage_b_step_cache_commitment(&token_comms, k_rows, v_rows);
                if token.cache_commitment_after != expected_after {
                    return Err(
                        "token's cache_commitment_after does not match its post-step KV-cache state"
                            .into(),
                    );
                }
                prev_after = Some(expected_after);
            }
        }
        Ok(())
    }
}

fn chunk_openings(
    statement: &StageBStatement,
    scores: &[CacheOpeningClaim],
    context: &[CacheOpeningClaim],
) -> Result<Vec<CacheChunkOpening>, String> {
    let mut openings: BTreeMap<usize, CacheChunkOpening> = BTreeMap::new();
    for (t, token) in statement.tokens.iter().enumerate() {
        let (s, c) = (&scores[t], &context[t]);
        if s.full_chunk_openings.len() != token.chunk_ids.len()
            || c.full_chunk_openings.len() != token.chunk_ids.len()
        {
            return Err("cache claims do not open every full chunk of the token".into());
        }
        for (i, &chunk) in token.chunk_ids.iter().enumerate() {
            let opening = CacheChunkOpening {
                chunk,
                k_point: s.point[s.log_chunks..].to_vec(),
                k_value: s.full_chunk_openings[i],
                v_point: c.point[c.log_chunks..].to_vec(),
                v_value: c.full_chunk_openings[i],
            };
            match openings.get(&chunk) {
                None => {
                    openings.insert(chunk, opening);
                }
                Some(existing) => {
                    if existing.k_point != opening.k_point
                        || existing.v_point != opening.v_point
                        || existing.k_value != opening.k_value
                        || existing.v_value != opening.v_value
                    {
                        return Err(
                            "two tokens claim the same chunk at different points or values".into(),
                        );
                    }
                }
            }
        }
    }
    Ok(openings.into_values().collect())
}

#[derive(Clone, Debug)]
pub struct StageBDeferredCprsClaims {
    pub statement_digest: [u8; 32],
    pub attention_commitment: Hash,
    pub leaf_params: Params,
    pub qkv_point: Vec<SecureField>,
    pub o_point: Vec<SecureField>,
    pub leaf_claims: [SecureField; 4],
    pub chunk_commitments: Vec<Hash>,
    pub chunk_params: Params,
    pub chunk_openings: Vec<CacheChunkOpening>,
    pub witness_opening: Option<StageBDeferredWitnessOpening>,
}

pub struct StageBVerifierRun {
    pub recompute_ms: f64,
    pub sumcheck_ms: f64,
    pub cache_recombine_ms: f64,
    pub cprs_ms: f64,
    pub recomputed: Vec<RecomputedStep>,
}

fn pad_token_dim_point(
    n_b_wit: usize,
    r_tokens: &[SecureField],
    r_dim: &[SecureField],
) -> Vec<SecureField> {
    let n_pad = n_b_wit - (r_tokens.len() + r_dim.len());
    let mut pt = Vec::with_capacity(n_b_wit);
    pt.resize(n_pad, SecureField::zero());
    pt.extend_from_slice(r_tokens);
    pt.extend_from_slice(r_dim);
    pt
}

fn eval_scores_left_weight_at(
    n_b_wit: usize,
    shape: AttentionShape,
    token_alphas: &[SecureField],
    r_heads: &[SecureField],
    s_cols: &[SecureField],
    r_b_wit: &[SecureField],
) -> SecureField {
    let log_tokens = token_alphas.len().ilog2() as usize;
    let log_d = shape.dimension.ilog2() as usize;
    let n_pad = n_b_wit - (log_tokens + log_d);
    let mut pad_factor = SecureField::one();
    for &p in &r_b_wit[..n_pad] {
        pad_factor *= SecureField::one() - p;
    }
    let r_tok_star = &r_b_wit[n_pad..n_pad + log_tokens];
    let r_i_star = &r_b_wit[n_pad + log_tokens..];
    let tok_eq = eq_evals(r_tok_star);
    let tok_factor: SecureField = token_alphas
        .iter()
        .zip(&tok_eq)
        .map(|(&a, &e)| a * e)
        .sum();
    pad_factor * tok_factor * scores_left_weight_mle_at(shape, r_heads, s_cols, r_i_star)
}

fn eval_ctx_weight_at(
    n_b_wit: usize,
    token_betas: &[SecureField],
    r_heads: &[SecureField],
    r_i: &[SecureField],
    r_b_wit: &[SecureField],
) -> SecureField {
    let log_tokens = token_betas.len().ilog2() as usize;
    let log_d = r_heads.len() + r_i.len();
    let n_pad = n_b_wit - (log_tokens + log_d);
    let mut pad_factor = SecureField::one();
    for &p in &r_b_wit[..n_pad] {
        pad_factor *= SecureField::one() - p;
    }
    let r_tok_star = &r_b_wit[n_pad..n_pad + log_tokens];
    let r_i_star = &r_b_wit[n_pad + log_tokens..];
    let tok_eq = eq_evals(r_tok_star);
    let tok_factor: SecureField = token_betas
        .iter()
        .zip(&tok_eq)
        .map(|(&b, &e)| b * e)
        .sum();
    let mut ctx_i_pt = Vec::with_capacity(log_d);
    ctx_i_pt.extend_from_slice(r_heads);
    ctx_i_pt.extend_from_slice(r_i);
    pad_factor * tok_factor * eq(&ctx_i_pt, r_i_star)
}

#[derive(Clone, Copy, Debug)]
pub struct StageBZkCfsLayout {
    pub cfs_log_rows: usize,
    pub cfs_rows: usize,
    pub half_cfs: usize,
    pub qkv_cfs_off: usize,
    pub scores_cfs_off: usize,
    pub ctx_cfs_off: usize,
    pub o_cfs_off: usize,
    pub gkr_libra_off: usize,
    pub wit_cfs_off: usize,
    pub m_qkv_off: usize,
    pub m_x1_off: usize,
    pub m_scores_comb_off: usize,
    pub m_scores_final_off: usize,
    pub m_ctx_comb_off: usize,
    pub m_ctx_left_off: usize,
    pub m_o_off: usize,
    pub m_ctx_in_off: usize,
    pub delta0_off: usize,
    pub delta1_off: usize,
    pub m_mult_off: usize,
}

impl StageBZkCfsLayout {
    pub fn compute(
        b: usize,
        shape: AttentionShape,
        padded_all: usize,
        ext_n_b_wit: usize,
    ) -> Result<Self, String> {
        let cfs_log_rows = stage_b_zk_cfs_log_rows(b);
        let cfs_rows = 1usize << cfs_log_rows;
        let half_cfs = cfs_rows / 2;
        let log_d = shape.dimension.ilog2() as usize;
        let log_kv_dim = shape.kv_dim().ilog2() as usize;
        let log_ctx_rounds = shape.kv_heads.ilog2() as usize + padded_all.ilog2() as usize;

        let mut cur = 0usize;
        let qkv_cfs_off = cur;
        cur += 4 * log_d;
        let scores_cfs_off = cur;
        cur += 4 * log_kv_dim;
        let ctx_cfs_off = cur;
        cur += 4 * log_ctx_rounds;
        let o_cfs_off = cur;
        cur += 4 * log_d;
        let gkr_libra_off = cur;
        for layer in 0..ext_n_b_wit {
            cur += 4 * layer;
        }
        let wit_cfs_off = cur;
        cur += 4 * ext_n_b_wit;

        let m_qkv_off = cur;
        cur += 3;
        let m_x1_off = cur;
        cur += 1;
        let m_scores_comb_off = cur;
        cur += 1;
        let m_scores_final_off = cur;
        cur += 1;
        let m_ctx_comb_off = cur;
        cur += 1;
        let m_ctx_left_off = cur;
        cur += b;
        let m_o_off = cur;
        cur += 1;
        let m_ctx_in_off = cur;
        cur += 1;
        let delta0_off = cur;
        cur += 2;
        let delta1_off = cur;
        cur += 2;
        let m_mult_off = cur;
        cur += 1;

        if cur > half_cfs {
            return Err(format!(
                "Stage B ZK CFS layout overflow: need {cur} slots, half_cfs={half_cfs}"
            ));
        }

        Ok(Self {
            cfs_log_rows,
            cfs_rows,
            half_cfs,
            qkv_cfs_off,
            scores_cfs_off,
            ctx_cfs_off,
            o_cfs_off,
            gkr_libra_off,
            wit_cfs_off,
            m_qkv_off,
            m_x1_off,
            m_scores_comb_off,
            m_scores_final_off,
            m_ctx_comb_off,
            m_ctx_left_off,
            m_o_off,
            m_ctx_in_off,
            delta0_off,
            delta1_off,
            m_mult_off,
        })
    }
}

#[inline]
fn i32_to_canonical_m31_u32(v: i32) -> u32 {
    if v >= 0 {
        let u = v as u32;
        if u >= P { u - P } else { u }
    } else {
        let mag = (-i64::from(v)) as u32;
        let rem = if mag >= P { mag - P } else { mag };
        if rem == 0 { 0 } else { P - rem }
    }
}

pub fn eval_pub_i16_mle_at<'a, F>(
    tokens_len: usize,
    d: usize,
    get_vec: F,
    rc: &[SecureField],
) -> SecureField
where
    F: Fn(usize) -> &'a [i16],
{
    let log_tokens = tokens_len.ilog2() as usize;
    let tok_eq = eq_evals(&rc[..log_tokens]);
    let dim_eq = eq_evals(&rc[log_tokens..]);
    let mut total = SecureField::zero();
    for t in 0..tokens_len {
        let row = get_vec(t);
        let mut row_dot = SecureField::zero();
        for j in 0..d {
            row_dot += dim_eq[j]
                * BaseField::from_u32_unchecked(i32_to_canonical_m31_u32(i32::from(row[j])));
        }
        total += tok_eq[t] * row_dot;
    }
    total
}

pub fn eval_zk_cont_mask_weight_at(
    ext_n_b_wit: usize,
    gamma_hax: &[SecureField; 3],
    r_b_wit: &[SecureField],
) -> SecureField {
    let mut mid_zero = r_b_wit[0];
    for &rj in &r_b_wit[1..ext_n_b_wit - 4] {
        mid_zero *= SecureField::one() - rj;
    }
    let low16_eq = eq_evals(&r_b_wit[ext_n_b_wit - 4..]);
    let beta = [
        SecureField::from_u32_unchecked(1, 0, 0, 0),
        SecureField::from_u32_unchecked(0, 1, 0, 0),
        SecureField::from_u32_unchecked(0, 0, 1, 0),
        SecureField::from_u32_unchecked(0, 0, 0, 1),
    ];
    let mut sum = SecureField::zero();
    for j in 0..3 {
        let mut inner = SecureField::zero();
        for c in 0..4 {
            inner += low16_eq[4 * j + c] * beta[c];
        }
        sum += gamma_hax[j] * inner;
    }
    mid_zero * sum
}

pub fn verify_stage_b_defer_cprs(
    statement: &StageBStatement,
    policy: AttentionPolicy,
    profile: &SecurityProfile,
    leaf_params: &Params,
    proof: &StageBProof,
) -> Result<(StageBVerifierRun, StageBDeferredCprsClaims), String> {
    verify_stage_b_defer_cprs_with_witness_f(statement, policy, profile, leaf_params, proof, None)
}

pub fn verify_stage_b_defer_cprs_with_witness_f(
    statement: &StageBStatement,
    policy: AttentionPolicy,
    profile: &SecurityProfile,
    leaf_params: &Params,
    proof: &StageBProof,
    witness_f_override: Option<usize>,
) -> Result<(StageBVerifierRun, StageBDeferredCprsClaims), String> {
    verify_stage_b_defer_cprs_with_cont(
        statement,
        policy,
        profile,
        leaf_params,
        proof,
        witness_f_override,
        None,
    )
}

pub fn verify_stage_b_defer_cprs_with_cont(
    statement: &StageBStatement,
    policy: AttentionPolicy,
    profile: &SecurityProfile,
    leaf_params: &Params,
    proof: &StageBProof,
    witness_f_override: Option<usize>,
    r_cont: Option<&[SecureField]>,
) -> Result<(StageBVerifierRun, StageBDeferredCprsClaims), String> {
    if !proof.attention_leaf_proof.is_empty() || !proof.cache_chunk_proof.is_empty() {
        return Err(
            "deferred Stage B verification expects empty attention_leaf_proof and cache_chunk_proof"
                .into(),
        );
    }
    if let Some(ref gkr) = proof.gkr_witness {
        if !gkr.cprs_witness_proof.is_empty() {
            return Err(
                "deferred Stage B GKR verification expects empty cprs_witness_proof".into(),
            );
        }
    }
    let recomputed = statement.validate(policy)?;
    let weight_f_override = (leaf_params.f() != 4).then_some(leaf_params.f());
    let f_override = witness_f_override.or(weight_f_override);
    let chunk_params = cache_chunk_params_with_f(
        statement.layout,
        statement.shape.kv_dim(),
        weight_f_override,
        profile,
    )?;
    statement.validate_chunk_rollovers(&chunk_params)?;

    let digest = statement.digest();
    let mut channel = StageBStatement::channel_from_digest(&digest);
    let tokens = &statement.tokens;
    let lengths = statement.cache_lengths();
    let is_committed = tokens[0].step.is_committed_witness();
    if is_committed != proof.gkr_witness.is_some() {
        return Err(
            "Stage B statement witness mode does not match proof gkr_witness presence".into(),
        );
    }
    if tokens[0].step.has_hidden_residuals() && r_cont.is_none() {
        return Err("Stage B hidden residuals require external r_cont".into());
    }
    if r_cont.is_some()
        && !proof
            .gkr_witness
            .as_ref()
            .is_some_and(|g| g.zk_aux.is_some())
    {
        return Err("Stage B external r_cont requires v3 ZK witness proof".into());
    }

    let (qkv_claims, o_claims, witness_opening) = if let Some(ref gkr) = proof.gkr_witness {
        if let Some(ref zk) = gkr.zk_aux {
            let b = tokens.len();
            let d = statement.shape.dimension;
            let log_tokens = b.ilog2() as usize;
            let log_d = d.ilog2() as usize;
            if let Some(rc) = r_cont {
                if rc.len() != log_tokens + log_d + 1 {
                    return Err("Stage B v3 ZK r_cont length mismatch".into());
                }
            }
            let exp_queries = stage_b_exp_query_count(statement);
            let ext_n_b_wit = stage_b_witness_n_for_zk(b, d, exp_queries);
            let m_ext = ext_n_b_wit - 2;
            let padded_all = lengths
                .iter()
                .map(|&l| statement.layout.padded_positions(l))
                .max()
                .unwrap();
            let layout = StageBZkCfsLayout::compute(b, statement.shape, padded_all, ext_n_b_wit)?;

            let wit_words: [u32; 8] = std::array::from_fn(|i| {
                u32::from_le_bytes(gkr.witness_commitment[4 * i..4 * i + 4].try_into().unwrap())
            });
            channel.mix_u32s(&wit_words);
            if let Some(rc) = r_cont {
                channel.mix_felts(rc);
            }

            let r_cfs_anchor = channel.draw_secure_felts(m_ext - layout.cfs_log_rows);

            let qkv_gkr = verify_projections_gkr_with_zk(
                b,
                d,
                d,
                &gkr.qkv_claims,
                gkr.x1_opening,
                &proof.qkv,
                &mut channel,
                Some((zk.qkv_cfs[0], zk.qkv_cfs[1])),
            )?;
            let scores_gkr = verify_scores_gkr_with_zk(
                statement.shape,
                statement.layout,
                &lengths,
                &gkr.scores_claims,
                &proof.scores,
                &mut channel,
                Some((zk.scores_cfs[0], zk.scores_cfs[1])),
            )?;
            let ctx_gkr = verify_context_gkr_with_zk(
                statement.shape,
                statement.layout,
                &lengths,
                &gkr.context_claims,
                &gkr.context_left_openings,
                &proof.context,
                &mut channel,
                Some((zk.ctx_cfs[0], zk.ctx_cfs[1])),
            )?;
            let o_gkr = verify_projections_gkr_with_zk(
                b,
                d,
                d,
                std::slice::from_ref(&gkr.o_claim),
                gkr.ctx_opening,
                &proof.o,
                &mut channel,
                Some((zk.o_cfs[0], zk.o_cfs[1])),
            )?;

            channel.mix_felts(&[gkr.x1_opening, gkr.ctx_opening]);
            channel.mix_felts(&gkr.context_left_openings);

            let exp_alphas = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let exp_z = channel.draw_secure_felt();
            if exp_z.is_zero() {
                return Err("Stage B ZK GKR exp_z challenge is zero".into());
            }
            if gkr.gkr_proof.output_claims_by_instance.len() != 3
                || gkr.gkr_proof.layer_masks_by_instance.len() != 3
                || gkr.gkr_proof.layer_masks_by_instance[0].len() != ext_n_b_wit
                || gkr.gkr_proof.layer_masks_by_instance[1].len() != LOG_EXP_TABLE_LEN + 1
                || gkr.gkr_proof.layer_masks_by_instance[2].len() != ZK_B_MASK_LOG_ROWS
                || gkr
                    .gkr_proof
                    .output_claims_by_instance
                    .iter()
                    .any(|c| c.len() != 2)
                || zk.gkr_libra_sums.len() != ext_n_b_wit
                || zk.gkr_libra_evals.len() != ext_n_b_wit
            {
                return Err("Stage B v3 ZK GKR witness proof shape mismatch".into());
            }
            let (gkr_artifact, gkr_layer_ood_points) = partially_verify_batch_with_libra_masks(
                vec![Gate::LogUp; 3],
                &gkr.gkr_proof,
                Some((&zk.gkr_libra_sums, &zk.gkr_libra_evals)),
                &mut channel,
            )
            .map_err(|e| format!("Stage B v3 ZK GKR batch proof rejected: {e}"))?;
            let mut gkr_frac_sum = SecureField::zero();
            for claims in &gkr.gkr_proof.output_claims_by_instance {
                let (num, den) = (claims[0], claims[1]);
                if den.is_zero() {
                    return Err("Stage B v3 ZK GKR output denominator is zero".into());
                }
                gkr_frac_sum += num / den;
            }
            if !gkr_frac_sum.is_zero() {
                return Err("Stage B v3 ZK GKR LogUp fractions do not cancel to zero".into());
            }

            let r_gkr_w = &gkr_artifact.ood_point[..ext_n_b_wit];
            let tau_0 = r_gkr_w[0];
            let r_inst1 = &gkr_artifact.ood_point[ext_n_b_wit - (LOG_EXP_TABLE_LEN + 1)..];
            let tau_1 = r_inst1[0];
            let r_gkr_tbl = &r_inst1[1..];
            let r_mask = &gkr_artifact.ood_point[ext_n_b_wit - ZK_B_MASK_LOG_ROWS..];
            if tau_0.is_zero() || tau_1.is_zero() || tau_1 == SecureField::one() {
                return Err("Stage B v3 ZK GKR degenerate half-cube selector challenge".into());
            }

            let [num0, den0]: [SecureField; 2] = gkr_artifact.claims_to_verify_by_instance[0]
                .as_slice()
                .try_into()
                .map_err(|_| "Stage B v3 ZK GKR instance 0 claim shape".to_string())?;
            let gkr_w_claim = den0 + exp_z;
            let [num1, den1]: [SecureField; 2] = gkr_artifact.claims_to_verify_by_instance[1]
                .as_slice()
                .try_into()
                .map_err(|_| "Stage B v3 ZK GKR instance 1 claim shape".to_string())?;
            let [num2, den2]: [SecureField; 2] = gkr_artifact.claims_to_verify_by_instance[2]
                .as_slice()
                .try_into()
                .map_err(|_| "Stage B v3 ZK GKR instance 2 claim shape".to_string())?;

            let exp_tbl = exp_table_static();
            let exp_idx_mle: SecureField = r_gkr_tbl
                .iter()
                .enumerate()
                .map(|(bit, &rb)| {
                    let w = 1u32 << (LOG_EXP_TABLE_LEN - 1 - bit);
                    rb * BaseField::from_u32_unchecked(w)
                })
                .sum();
            let exp_val_mle = dot_u16_eq_i64(&eq_evals_i64(r_gkr_tbl), exp_tbl);
            let expected_tbl_den = exp_alphas[0] * exp_idx_mle + exp_alphas[1] * exp_val_mle - exp_z;
            if den1 != expected_tbl_den {
                return Err("Stage B v3 ZK GKR T_EXP table denominator mismatch".into());
            }

            let den_tbl0 =
                exp_alphas[1] * BaseField::from_u32_unchecked(u32::from(exp_tbl[0])) - exp_z;
            let den_tbl1 = exp_alphas[0] * BaseField::one()
                + exp_alphas[1] * BaseField::from_u32_unchecked(u32::from(exp_tbl[1]))
                - exp_z;
            if den_tbl0 == den_tbl1 || den_tbl0.is_zero() || den_tbl1.is_zero() {
                return Err("Stage B v3 ZK GKR degenerate mask denominators".into());
            }
            let mut den2_arr = [-exp_z; ZK_B_MASK_ROWS];
            den2_arr[0] = den_tbl0;
            den2_arr[1] = den_tbl1;
            den2_arr[2] = den_tbl0;
            den2_arr[3] = den_tbl1;
            let mask_eq = eq_evals(r_mask);
            let expected_den2: SecureField = den2_arr
                .iter()
                .zip(&mask_eq)
                .map(|(&a, &b)| a * b)
                .sum();
            if den2 != expected_den2 {
                return Err("Stage B v3 ZK GKR masking instance denominator mismatch".into());
            }

            channel.mix_felts(&[zk.gkr_mult_masked_claim]);

            let has_h_in = !tokens[0].step.h_in.is_empty();
            let has_a = !tokens[0].step.a.is_empty();
            let has_x = !tokens[0].step.x.is_empty();
            let (rc, tau_cont, has_wire_cont, hax_pub_claims) = if let Some(ext_rc) = r_cont {
                if ext_rc.len() != log_tokens + log_d + 1 {
                    return Err("Stage B v3 ZK external r_cont length mismatch".into());
                }
                let rc_pt = &ext_rc[..log_tokens + log_d];
                let tc = ext_rc[log_tokens + log_d];
                if tc.is_zero() {
                    return Err("Stage B v3 ZK external tau_cont cannot be zero".into());
                }
                if zk.cont_claims.len() != 3 {
                    return Err("Stage B v3 ZK cont_claims length mismatch".into());
                }
                channel.mix_felts(&zk.cont_claims);
                if has_h_in {
                    let expected = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.h_in, rc_pt);
                    if zk.cont_claims[0] != expected {
                        return Err("Stage B v3 ZK public h_in continuity claim mismatch".into());
                    }
                }
                if has_a {
                    let expected = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.a, rc_pt);
                    if zk.cont_claims[1] != expected {
                        return Err("Stage B v3 ZK public a continuity claim mismatch".into());
                    }
                }
                if has_x {
                    let expected = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.x, rc_pt);
                    if zk.cont_claims[2] != expected {
                        return Err("Stage B v3 ZK public x continuity claim mismatch".into());
                    }
                }
                (
                    rc_pt.to_vec(),
                    tc,
                    true,
                    [zk.cont_claims[0], zk.cont_claims[1], zk.cont_claims[2]],
                )
            } else {
                if !zk.cont_claims.is_empty() {
                    return Err("Stage B v3 ZK unexpected cont_claims without r_cont".into());
                }
                let drawn_rc = channel.draw_secure_felts(log_tokens + log_d);
                let h_eval = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.h_in, &drawn_rc);
                let a_eval = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.a, &drawn_rc);
                let x_eval = eval_pub_i16_mle_at(b, d, |t| &tokens[t].step.x, &drawn_rc);
                (
                    drawn_rc,
                    SecureField::one(),
                    false,
                    [h_eval, a_eval, x_eval],
                )
            };

            let gamma_qkv = [
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
            ];
            let gamma_hax = [
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
            ];
            let lambda_wit = channel.draw_secure_felt();

            if gkr.witness_sumcheck.round_polys.len() != ext_n_b_wit {
                return Err("Stage B v3 ZK witness sum-check round count mismatch".into());
            }

            let mut token_betas = Vec::with_capacity(b);
            let mut pow = SecureField::one();
            for _ in 0..b {
                token_betas.push(pow);
                pow *= ctx_gkr.lambda_ctx;
            }

            let hax_comb_claim = gamma_hax[0] * hax_pub_claims[0]
                + gamma_hax[1] * hax_pub_claims[1]
                + gamma_hax[2] * hax_pub_claims[2];
            let mut sent_wit_claims = vec![
                gkr.x1_opening,
                gamma_qkv[0] * gkr.qkv_claims[0]
                    + gamma_qkv[1] * gkr.qkv_claims[1]
                    + gamma_qkv[2] * gkr.qkv_claims[2],
                scores_gkr.final_eval,
                gkr.context_claims[0],
                gkr.ctx_opening,
                gkr.o_claim,
                gkr_w_claim,
                zk.gkr_mult_masked_claim,
            ];
            if has_wire_cont {
                sent_wit_claims.push(SecureField::zero());
                sent_wit_claims.push(hax_comb_claim);
            } else {
                sent_wit_claims.push(hax_comb_claim);
            }

            channel.mix_felts(&[zk.wit_cfs[0]]);
            let (r_b_wit, wit_final_eval) = partially_verify_with_lo_basis(
                combine(&sent_wit_claims, lambda_wit) + zk.wit_cfs[0],
                &gkr.witness_sumcheck,
                &mut channel,
                true,
            )?;
            if r_b_wit.len() != ext_n_b_wit || r_b_wit[0].is_zero() {
                return Err("Stage B v3 ZK witness sum-check assignment invalid".into());
            }
            channel.mix_felts(&gkr.witness_openings);
            channel.mix_felts(&zk.h_a_x_openings);
            channel.mix_felts(&[zk.wit_cfs[1]]);

            let p_x1 = pad_token_dim_point(ext_n_b_wit, &qkv_gkr.r_tokens, &qkv_gkr.s_dimension);
            let p_qkv = pad_token_dim_point(ext_n_b_wit, &qkv_gkr.r_tokens, &qkv_gkr.r_out);
            let p_ctx_in = pad_token_dim_point(ext_n_b_wit, &o_gkr.r_tokens, &o_gkr.s_dimension);
            let p_o = pad_token_dim_point(ext_n_b_wit, &o_gkr.r_tokens, &o_gkr.r_out);
            let r_gkr_tbl_padded = pad_token_dim_point(ext_n_b_wit, &[], r_gkr_tbl);
            let p_rc = pad_token_dim_point(ext_n_b_wit, &[], &rc);
            let c = &gkr.witness_openings;
            let hax_open_comb = gamma_hax[0] * zk.h_a_x_openings[0]
                + gamma_hax[1] * zk.h_a_x_openings[1]
                + gamma_hax[2] * zk.h_a_x_openings[2];
            let mut expected_wit = vec![
                eq(&p_x1, &r_b_wit) * c[COL_B_X1],
                eq(&p_qkv, &r_b_wit)
                    * (gamma_qkv[0] * c[COL_B_Q_RAW]
                        + gamma_qkv[1] * c[COL_B_K_RAW]
                        + gamma_qkv[2] * c[COL_B_V_RAW]),
                eval_scores_left_weight_at(
                    ext_n_b_wit,
                    statement.shape,
                    &scores_gkr.token_alphas,
                    &scores_gkr.r_heads,
                    &scores_gkr.s_cols,
                    &r_b_wit,
                ) * c[COL_B_Q_ROT],
                eval_ctx_weight_at(
                    ext_n_b_wit,
                    &token_betas,
                    &ctx_gkr.r_heads,
                    &ctx_gkr.r_i,
                    &r_b_wit,
                ) * c[COL_B_CTX_RAW],
                eq(&p_ctx_in, &r_b_wit) * c[COL_B_CTX],
                eq(&p_o, &r_b_wit) * c[COL_B_O_RAW],
                eq(r_gkr_w, &r_b_wit)
                    * (exp_alphas[0] * c[COL_B_EXP_IDX] + exp_alphas[1] * c[COL_B_EXP_VAL]),
                eq(&r_gkr_tbl_padded, &r_b_wit) * c[COL_B_MULT],
            ];
            if has_wire_cont {
                expected_wit.push((lambda_wit * eq(&p_rc, &r_b_wit)) * hax_open_comb);
                expected_wit.push(
                    tau_cont
                        * eval_zk_cont_mask_weight_at(ext_n_b_wit, &gamma_hax, &r_b_wit)
                        * c[COL_B_MULT],
                );
            } else {
                expected_wit.push(eq(&p_rc, &r_b_wit) * hax_open_comb);
            }
            if wit_final_eval != combine(&expected_wit, lambda_wit) + zk.wit_cfs[1] {
                return Err("Stage B v3 ZK witness sum-check final check failed".into());
            }

            let r_cfs_pad = channel.draw_secure_felts(layout.cfs_log_rows - 1);
            channel.mix_felts(&[zk.cfs_pad_claim]);

            let gamma_libra_sums = channel.draw_secure_felts(ext_n_b_wit);
            let gamma_libra_evals = channel.draw_secure_felts(ext_n_b_wit);
            let g_qkv = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let g_sc = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let g_ctx = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let g_o = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let g_wit = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let g_num = [
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
            ];
            let g_pad = channel.draw_secure_felt();

            if zk.cfs_binding_sumcheck.round_polys.len() != layout.cfs_log_rows {
                return Err("Stage B v3 ZK CFS binding sum-check round count mismatch".into());
            }

            let lam1 = lambda_wit;
            let lam2 = lam1 * lambda_wit;
            let lam3 = lam2 * lambda_wit;
            let lam4 = lam3 * lambda_wit;
            let lam5 = lam4 * lambda_wit;
            let lam7 = lam5 * lambda_wit * lambda_wit;

            let mut w_cfs = vec![SecureField::zero(); layout.cfs_rows];
            let mut gkr_off = layout.gkr_libra_off;
            for (layer, pt) in gkr_layer_ood_points.iter().enumerate() {
                cfs_deg3_accumulate_weights(
                    &mut w_cfs,
                    gkr_off,
                    pt,
                    gamma_libra_sums[layer],
                    gamma_libra_evals[layer],
                );
                gkr_off += 4 * pt.len();
            }
            cfs_deg3_accumulate_weights(
                &mut w_cfs,
                layout.qkv_cfs_off,
                &qkv_gkr.s_dimension,
                g_qkv[0],
                g_qkv[1],
            );
            let qkv_lam = qkv_gkr.lambda;
            w_cfs[layout.m_qkv_off] -= g_qkv[0];
            w_cfs[layout.m_qkv_off + 1] -= g_qkv[0] * qkv_lam;
            w_cfs[layout.m_qkv_off + 2] -= g_qkv[0] * qkv_lam * qkv_lam;
            let qkv_w_comb = combine(&proof.qkv.weight_openings, qkv_lam);
            w_cfs[layout.m_x1_off] -= g_qkv[1] * qkv_w_comb;

            cfs_deg3_accumulate_weights(
                &mut w_cfs,
                layout.scores_cfs_off,
                &scores_gkr.s_cols,
                g_sc[0],
                g_sc[1],
            );
            w_cfs[layout.m_scores_comb_off] -= g_sc[0];
            w_cfs[layout.m_scores_final_off] -= g_sc[1];

            cfs_deg3_accumulate_weights(
                &mut w_cfs,
                layout.ctx_cfs_off,
                &ctx_gkr.s_point,
                g_ctx[0],
                g_ctx[1],
            );
            w_cfs[layout.m_ctx_comb_off] -= g_ctx[0];
            for t in 0..b {
                w_cfs[layout.m_ctx_left_off + t] -=
                    g_ctx[1] * token_betas[t] * proof.context.claims[t].value;
            }

            cfs_deg3_accumulate_weights(
                &mut w_cfs,
                layout.o_cfs_off,
                &o_gkr.s_dimension,
                g_o[0],
                g_o[1],
            );
            w_cfs[layout.m_o_off] -= g_o[0];
            w_cfs[layout.m_ctx_in_off] -= g_o[1] * proof.o.weight_openings[0];

            cfs_deg3_accumulate_weights(
                &mut w_cfs,
                layout.wit_cfs_off,
                &r_b_wit,
                g_wit[0],
                g_wit[1],
            );
            w_cfs[layout.m_x1_off] -= g_wit[0];
            for j in 0..3 {
                w_cfs[layout.m_qkv_off + j] -= g_wit[0] * lam1 * gamma_qkv[j];
            }
            w_cfs[layout.m_scores_final_off] -= g_wit[0] * lam2;
            w_cfs[layout.m_ctx_comb_off] -= g_wit[0] * lam3;
            w_cfs[layout.m_ctx_in_off] -= g_wit[0] * lam4;
            w_cfs[layout.m_o_off] -= g_wit[0] * lam5;
            w_cfs[layout.m_mult_off] -= g_wit[0] * lam7;

            let r_last = r_gkr_w[ext_n_b_wit - 1];
            let one_minus_last = SecureField::one() - r_last;
            let mut prod_zero_0 = SecureField::one();
            for &rj in &r_gkr_w[1..ext_n_b_wit - 1] {
                prod_zero_0 *= SecureField::one() - rj;
            }
            w_cfs[layout.delta0_off] += g_num[0] * tau_0 * prod_zero_0 * one_minus_last;
            w_cfs[layout.delta0_off + 1] += g_num[0] * tau_0 * prod_zero_0 * r_last;

            let mut prod_zero_1 = SecureField::one();
            for &rj in &r_gkr_tbl[..LOG_EXP_TABLE_LEN - 1] {
                prod_zero_1 *= SecureField::one() - rj;
            }
            w_cfs[layout.m_mult_off] += g_num[1] * (SecureField::one() - tau_1);
            w_cfs[layout.delta1_off] += g_num[1] * tau_1 * prod_zero_1 * one_minus_last;
            w_cfs[layout.delta1_off + 1] += g_num[1] * tau_1 * prod_zero_1 * r_last;

            w_cfs[layout.delta0_off] -= g_num[2] * mask_eq[0];
            w_cfs[layout.delta0_off + 1] -= g_num[2] * mask_eq[1];
            w_cfs[layout.delta1_off] -= g_num[2] * mask_eq[2];
            w_cfs[layout.delta1_off + 1] -= g_num[2] * mask_eq[3];

            let pad_eq = eq_evals(&r_cfs_pad);
            for (idx, &pe) in pad_eq.iter().enumerate() {
                w_cfs[layout.half_cfs + idx] += g_pad * pe;
            }

            let mut cfs_binding_claim = g_qkv[0] * zk.qkv_cfs[0]
                + g_qkv[1] * zk.qkv_cfs[1]
                + g_sc[0] * zk.scores_cfs[0]
                + g_sc[1] * zk.scores_cfs[1]
                + g_ctx[0] * zk.ctx_cfs[0]
                + g_ctx[1] * zk.ctx_cfs[1]
                + g_o[0] * zk.o_cfs[0]
                + g_o[1] * zk.o_cfs[1]
                + g_wit[0] * zk.wit_cfs[0]
                + g_wit[1] * zk.wit_cfs[1]
                + g_num[0] * (num0 - (SecureField::one() - tau_0))
                + g_num[1] * (num1 + (SecureField::one() - tau_1) * zk.gkr_mult_masked_claim)
                + g_num[2] * num2
                + g_pad * zk.cfs_pad_claim;
            for layer in 0..ext_n_b_wit {
                cfs_binding_claim += gamma_libra_sums[layer] * zk.gkr_libra_sums[layer]
                    + gamma_libra_evals[layer] * zk.gkr_libra_evals[layer];
            }

            let (r_cfs_sub, cfs_final_eval) =
                partially_verify(cfs_binding_claim, &zk.cfs_binding_sumcheck, &mut channel)
                    .map_err(|e| format!("Stage B v3 ZK CFS binding sum-check rejected: {e}"))?;
            channel.mix_felts(&[zk.blind_col_opening]);

            let w_cfs_eval = mle_at(&w_cfs, &r_cfs_sub);
            if cfs_final_eval != w_cfs_eval * zk.blind_col_opening {
                return Err("Stage B v3 ZK CFS binding sum-check final check failed".into());
            }

            let mut cfs_hi_point = r_cfs_anchor;
            cfs_hi_point.extend_from_slice(&r_cfs_sub);
            let witness_params =
                stage_b_witness_params_for_zk(b, d, exp_queries, f_override, profile)?;
            (
                qkv_gkr.opening_claims,
                o_gkr.opening_claims,
                Some(StageBDeferredWitnessOpening {
                    witness_commitment: gkr.witness_commitment,
                    witness_params,
                    witness_point: r_b_wit,
                    witness_openings: gkr.witness_openings,
                    zk_extra_openings: Some(StageBZkDeferredOpenings {
                        h_a_x_openings: zk.h_a_x_openings,
                        cfs_hi_point,
                        blind_opening: zk.blind_col_opening,
                    }),
                }),
            )
        } else {
            let b = tokens.len();
            let d = statement.shape.dimension;
            let exp_queries = stage_b_exp_query_count(statement);
            let n_b_wit = stage_b_witness_n_with_queries(b, d, exp_queries);
            let wit_words: [u32; 8] = std::array::from_fn(|i| {
                u32::from_le_bytes(gkr.witness_commitment[4 * i..4 * i + 4].try_into().unwrap())
            });
            channel.mix_u32s(&wit_words);

            let qkv_gkr = verify_projections_gkr(
                b,
                d,
                d,
                &gkr.qkv_claims,
                gkr.x1_opening,
                &proof.qkv,
                &mut channel,
            )?;
            let scores_gkr = verify_scores_gkr(
                statement.shape,
                statement.layout,
                &lengths,
                &gkr.scores_claims,
                &proof.scores,
                &mut channel,
            )?;
            let ctx_gkr = verify_context_gkr(
                statement.shape,
                statement.layout,
                &lengths,
                &gkr.context_claims,
                &gkr.context_left_openings,
                &proof.context,
                &mut channel,
            )?;
            let o_gkr = verify_projections_gkr(
                b,
                d,
                d,
                std::slice::from_ref(&gkr.o_claim),
                gkr.ctx_opening,
                &proof.o,
                &mut channel,
            )?;

            channel.mix_felts(&[gkr.x1_opening, gkr.ctx_opening]);
            channel.mix_felts(&gkr.context_left_openings);

            let exp_alphas = [channel.draw_secure_felt(), channel.draw_secure_felt()];
            let exp_z = channel.draw_secure_felt();
            if gkr.gkr_proof.output_claims_by_instance.len() != 2
                || gkr.gkr_proof.layer_masks_by_instance.len() != 2
                || gkr.gkr_proof.layer_masks_by_instance[0].len() != n_b_wit
                || gkr.gkr_proof.layer_masks_by_instance[1].len() != LOG_EXP_TABLE_LEN
                || gkr.gkr_proof.output_claims_by_instance[0].len() != 2
                || gkr.gkr_proof.output_claims_by_instance[1].len() != 2
            {
                return Err("Stage B GKR witness proof shape mismatch".into());
            }
            let gkr_artifact =
                partially_verify_batch(vec![Gate::LogUp; 2], &gkr.gkr_proof, &mut channel)
                    .map_err(|e| format!("Stage B GKR batch proof rejected: {e}"))?;
            let mut gkr_frac_sum = SecureField::zero();
            for claims in &gkr.gkr_proof.output_claims_by_instance {
                let (num, den): (SecureField, SecureField) = (claims[0], claims[1]);
                if den.is_zero() {
                    return Err("Stage B GKR output denominator is zero".into());
                }
                gkr_frac_sum += num / den;
            }
            if !gkr_frac_sum.is_zero() {
                return Err("Stage B GKR LogUp fractions do not cancel to zero".into());
            }
            let r_gkr_w = &gkr_artifact.ood_point[..n_b_wit];
            let r_gkr_tbl = &gkr_artifact.ood_point[n_b_wit - LOG_EXP_TABLE_LEN..];
            let [num0, den0]: [SecureField; 2] = gkr_artifact.claims_to_verify_by_instance[0]
                .as_slice()
                .try_into()
                .map_err(|_| "Stage B GKR instance 0 claim shape".to_string())?;
            if num0 != SecureField::one() {
                return Err("Stage B GKR query numerator is not one".into());
            }
            let gkr_w_claim = den0 + exp_z;
            let [num1, den1]: [SecureField; 2] = gkr_artifact.claims_to_verify_by_instance[1]
                .as_slice()
                .try_into()
                .map_err(|_| "Stage B GKR instance 1 claim shape".to_string())?;
            let exp_idx_mle: SecureField = r_gkr_tbl
                .iter()
                .enumerate()
                .map(|(bit, &rb)| {
                    let w = 1u32 << (LOG_EXP_TABLE_LEN - 1 - bit);
                    rb * BaseField::from_u32_unchecked(w)
                })
                .sum();
            let exp_val_mle = dot_u16_eq_i64(&eq_evals_i64(r_gkr_tbl), exp_table_static());
            let expected_tbl_den = exp_alphas[0] * exp_idx_mle + exp_alphas[1] * exp_val_mle - exp_z;
            if den1 != expected_tbl_den {
                return Err("Stage B GKR T_EXP table denominator mismatch".into());
            }
            let gkr_mult_claim = -num1;

            let gamma_qkv = [
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
                channel.draw_secure_felt(),
            ];
            let beta_ctx = channel.draw_secure_felt();
            let lambda_wit = channel.draw_secure_felt();
            if gkr.witness_sumcheck.round_polys.len() != n_b_wit {
                return Err("Stage B witness sum-check round count mismatch".into());
            }
            let mut token_betas = Vec::with_capacity(b);
            let mut pow = SecureField::one();
            for _ in 0..b {
                token_betas.push(pow);
                pow *= beta_ctx;
            }
            let ctx_combined_claim: SecureField = token_betas
                .iter()
                .zip(&gkr.context_claims)
                .map(|(&bt, &cl)| bt * cl)
                .sum();
            let claims_8 = [
                gkr.x1_opening,
                gamma_qkv[0] * gkr.qkv_claims[0]
                    + gamma_qkv[1] * gkr.qkv_claims[1]
                    + gamma_qkv[2] * gkr.qkv_claims[2],
                scores_gkr.final_eval,
                ctx_combined_claim,
                gkr.ctx_opening,
                gkr.o_claim,
                gkr_w_claim,
                gkr_mult_claim,
            ];
            let (r_b_wit, wit_final_eval) =
                partially_verify(combine(&claims_8, lambda_wit), &gkr.witness_sumcheck, &mut channel)?;
            if r_b_wit.len() != n_b_wit {
                return Err("Stage B witness sum-check assignment length mismatch".into());
            }
            channel.mix_felts(&gkr.witness_openings);

            let p_x1 = pad_token_dim_point(n_b_wit, &qkv_gkr.r_tokens, &qkv_gkr.s_dimension);
            let p_qkv = pad_token_dim_point(n_b_wit, &qkv_gkr.r_tokens, &qkv_gkr.r_out);
            let p_ctx_in = pad_token_dim_point(n_b_wit, &o_gkr.r_tokens, &o_gkr.s_dimension);
            let p_o = pad_token_dim_point(n_b_wit, &o_gkr.r_tokens, &o_gkr.r_out);
            let r_gkr_tbl_padded = pad_token_dim_point(n_b_wit, &[], r_gkr_tbl);
            let c = &gkr.witness_openings;
            let expected_8 = [
                eq(&p_x1, &r_b_wit) * c[COL_B_X1],
                eq(&p_qkv, &r_b_wit)
                    * (gamma_qkv[0] * c[COL_B_Q_RAW]
                        + gamma_qkv[1] * c[COL_B_K_RAW]
                        + gamma_qkv[2] * c[COL_B_V_RAW]),
                eval_scores_left_weight_at(
                    n_b_wit,
                    statement.shape,
                    &scores_gkr.token_alphas,
                    &scores_gkr.r_heads,
                    &scores_gkr.s_cols,
                    &r_b_wit,
                ) * c[COL_B_Q_ROT],
                eval_ctx_weight_at(
                    n_b_wit,
                    &token_betas,
                    &ctx_gkr.r_heads,
                    &ctx_gkr.r_i,
                    &r_b_wit,
                ) * c[COL_B_CTX_RAW],
                eq(&p_ctx_in, &r_b_wit) * c[COL_B_CTX],
                eq(&p_o, &r_b_wit) * c[COL_B_O_RAW],
                eq(r_gkr_w, &r_b_wit)
                    * (exp_alphas[0] * c[COL_B_EXP_IDX] + exp_alphas[1] * c[COL_B_EXP_VAL]),
                eq(&r_gkr_tbl_padded, &r_b_wit) * c[COL_B_MULT],
            ];
            if wit_final_eval != combine(&expected_8, lambda_wit) {
                return Err("Stage B witness sum-check final check failed".into());
            }
            let witness_params =
                stage_b_witness_params_with_queries_and_f(b, d, exp_queries, f_override, profile)?;
            (
                qkv_gkr.opening_claims,
                o_gkr.opening_claims,
                Some(StageBDeferredWitnessOpening {
                    witness_commitment: gkr.witness_commitment,
                    witness_params,
                    witness_point: r_b_wit,
                    witness_openings: gkr.witness_openings,
                    zk_extra_openings: None,
                }),
            )
        }
    } else {
        let inputs: Vec<Vec<i8>> = recomputed.iter().map(|r| r.x1.clone()).collect();
        let q: Vec<Vec<i32>> = tokens.iter().map(|t| t.step.q_raw.clone()).collect();
        let k: Vec<Vec<i32>> = tokens.iter().map(|t| t.step.k_raw.clone()).collect();
        let v: Vec<Vec<i32>> = tokens.iter().map(|t| t.step.v_raw.clone()).collect();
        let ctx_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.ctx.clone()).collect();
        let o: Vec<Vec<i32>> = tokens.iter().map(|t| t.step.o_raw.clone()).collect();
        let qkv_claims = verify_projections(&inputs, &[&q, &k, &v], &proof.qkv, &mut channel)?;
        let q_rot: Vec<&[i8]> = recomputed.iter().map(|r| r.q_rot.as_slice()).collect();
        let scores_pub: Vec<&[Vec<i32>]> = tokens.iter().map(|t| t.step.scores.as_slice()).collect();
        let attn: Vec<&[Vec<u16>]> = recomputed.iter().map(|r| r.weights.as_slice()).collect();
        let ctx_raw: Vec<&[i32]> = tokens.iter().map(|t| t.step.ctx_raw.as_slice()).collect();
        verify_scores(
            statement.shape,
            statement.layout,
            &q_rot,
            &lengths,
            &scores_pub,
            &proof.scores,
            &mut channel,
        )?;
        verify_context(
            statement.shape,
            statement.layout,
            &attn,
            &lengths,
            &ctx_raw,
            &proof.context,
            &mut channel,
        )?;
        let o_claims = verify_projections(&ctx_in, &[&o], &proof.o, &mut channel)?;
        (qkv_claims, o_claims, None)
    };

    let padded_all = lengths
        .iter()
        .map(|&l| statement.layout.padded_positions(l))
        .max()
        .unwrap();
    let chunk_ids_refs: Vec<&[usize]> = statement
        .tokens
        .iter()
        .map(|t| t.chunk_ids.as_slice())
        .collect();
    let k_rows_refs: Vec<&[Vec<i8>]> = statement
        .partial_rows
        .iter()
        .map(|(kr, _)| kr.as_slice())
        .collect();
    let v_rows_refs: Vec<&[Vec<i8>]> = statement
        .partial_rows
        .iter()
        .map(|(_, vr)| vr.as_slice())
        .collect();
    verify_cache_claims_batch(
        &chunk_ids_refs,
        &k_rows_refs,
        statement.layout,
        padded_all,
        &proof.scores.claims,
    )?;
    verify_cache_claims_batch(
        &chunk_ids_refs,
        &v_rows_refs,
        statement.layout,
        padded_all,
        &proof.context.claims,
    )?;

    let leaf_claims = [
        qkv_claims.openings[0],
        qkv_claims.openings[1],
        qkv_claims.openings[2],
        o_claims.openings[0],
    ];
    let openings = chunk_openings(statement, &proof.scores.claims, &proof.context.claims)?;
    let deferred = StageBDeferredCprsClaims {
        statement_digest: digest,
        attention_commitment: statement.attention_commitment,
        leaf_params: leaf_params.clone(),
        qkv_point: qkv_claims.point,
        o_point: o_claims.point,
        leaf_claims,
        chunk_commitments: statement.chunk_commitments.clone(),
        chunk_params,
        chunk_openings: openings,
        witness_opening,
    };
    Ok((
        StageBVerifierRun {
            recompute_ms: 0.0,
            sumcheck_ms: 0.0,
            cache_recombine_ms: 0.0,
            cprs_ms: 0.0,
            recomputed,
        },
        deferred,
    ))
}

pub fn verify_stage_b(
    statement: &StageBStatement,
    policy: AttentionPolicy,
    profile: &SecurityProfile,
    leaf_params: &Params,
    proof: &StageBProof,
) -> Result<StageBVerifierRun, String> {
    let mut deferred_proof = proof.clone();
    deferred_proof.attention_leaf_proof = Vec::new();
    deferred_proof.cache_chunk_proof = Vec::new();
    if let Some(ref mut gkr) = deferred_proof.gkr_witness {
        gkr.cprs_witness_proof = Vec::new();
    }
    let (run, deferred) =
        verify_stage_b_defer_cprs(statement, policy, profile, leaf_params, &deferred_proof)?;
    let wit_separate = proof
        .gkr_witness
        .as_ref()
        .is_some_and(|g| !g.cprs_witness_proof.is_empty());
    if wit_separate {
        let gkr = proof.gkr_witness.as_ref().unwrap();
        let w = deferred.witness_opening.as_ref().unwrap();
        verify_stage_b_witness_opening_deferred(
            w,
            &deferred.statement_digest,
            &gkr.cprs_witness_proof,
        )?;
    }
    if proof.cache_chunk_proof.is_empty() {
        verify_stage_b_cprs_unified(
            &deferred.attention_commitment,
            &deferred.leaf_params,
            &deferred.chunk_commitments,
            &deferred.chunk_openings,
            &deferred.chunk_params,
            &deferred.qkv_point,
            &deferred.o_point,
            &deferred.leaf_claims,
            if wit_separate {
                None
            } else {
                deferred.witness_opening.as_ref()
            },
            &deferred.statement_digest,
            &proof.attention_leaf_proof,
        )?;
    } else {
        if deferred.witness_opening.is_some() && !wit_separate {
            return Err("Stage B witness opening requires cprs_witness_proof or unified CPRS".into());
        }
        verify_attention_leaf_opening(
            &deferred.attention_commitment,
            &deferred.leaf_params,
            &deferred.qkv_point,
            &deferred.o_point,
            &deferred.leaf_claims,
            &deferred.statement_digest,
            &proof.attention_leaf_proof,
        )?;
        verify_cache_chunk_openings(
            &deferred.chunk_commitments,
            &deferred.chunk_openings,
            &deferred.chunk_params,
            &deferred.statement_digest,
            &proof.cache_chunk_proof,
        )?;
    }
    Ok(run)
}

pub fn put_projection_proof(out: &mut Vec<u8>, proof: &ProjectionProof) -> Result<(), String> {
    put_sumcheck(out, &proof.proof)?;
    put_secure_fields(out, &proof.weight_openings)
}

fn put_contraction_proof(
    out: &mut Vec<u8>,
    proof: &ContractionProof,
    version: u32,
) -> Result<(), String> {
    put_sumcheck(out, &proof.proof)?;
    put_u64(out, proof.claims.len())?;
    if version >= STAGE_B_RECEIPT_WIRE_VERSION_GKR {
        if let Some(first) = proof.claims.first() {
            for claim in &proof.claims[1..] {
                if claim.point != first.point || claim.log_chunks != first.log_chunks {
                    return Err(
                        "Stage B v2 contraction claims must share point and log_chunks".into(),
                    );
                }
            }
            put_secure_fields(out, &first.point)?;
            put_u64(out, first.log_chunks)?;
            let mut running: Vec<SecureField> = Vec::new();
            for claim in &proof.claims {
                let fco = &claim.full_chunk_openings;
                if fco.len() <= running.len() && fco.as_slice() == &running[..fco.len()] {
                    out.push(0u8);
                } else if fco.len() > running.len() && fco[..running.len()] == running[..] {
                    out.push(1u8);
                    for &sf in &fco[running.len()..] {
                        put_secure_field(out, sf);
                    }
                    running.extend_from_slice(&fco[running.len()..]);
                } else {
                    out.push(2u8);
                    for &sf in fco {
                        put_secure_field(out, sf);
                    }
                    running.clear();
                    running.extend_from_slice(fco);
                }
                put_secure_field(out, claim.value);
            }
        }
    } else {
        for claim in &proof.claims {
            put_secure_fields(out, &claim.point)?;
            put_u64(out, claim.log_chunks)?;
            put_secure_fields(out, &claim.full_chunk_openings)?;
            put_secure_field(out, claim.value);
        }
    }
    Ok(())
}

pub fn stage_b_receipt_bytes(proof: &StageBProof) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(STAGE_B_RECEIPT_MAGIC);
    let version = match proof.gkr_witness {
        Some(ref gkr) if gkr.zk_aux.is_some() => STAGE_B_RECEIPT_WIRE_VERSION_ZK_GKR,
        Some(_) => STAGE_B_RECEIPT_WIRE_VERSION_GKR,
        None => STAGE_B_RECEIPT_WIRE_VERSION,
    };
    put_u32(&mut out, version);
    put_projection_proof(&mut out, &proof.qkv)?;
    put_contraction_proof(&mut out, &proof.scores, version)?;
    put_contraction_proof(&mut out, &proof.context, version)?;
    put_projection_proof(&mut out, &proof.o)?;
    put_bytes(&mut out, &proof.attention_leaf_proof)?;
    put_bytes(&mut out, &proof.cache_chunk_proof)?;
    if let Some(ref gkr) = proof.gkr_witness {
        out.extend_from_slice(&gkr.witness_commitment);
        put_secure_fields(&mut out, &gkr.qkv_claims)?;
        put_secure_field(&mut out, gkr.x1_opening);
        put_secure_fields(&mut out, &gkr.scores_claims)?;
        put_secure_fields(&mut out, &gkr.context_claims)?;
        put_secure_fields(&mut out, &gkr.context_left_openings)?;
        put_secure_field(&mut out, gkr.o_claim);
        put_secure_field(&mut out, gkr.ctx_opening);
        put_gkr_batch_proof(&mut out, &gkr.gkr_proof)?;
        put_sumcheck(&mut out, &gkr.witness_sumcheck)?;
        put_secure_fields(&mut out, &gkr.witness_openings)?;
        if let Some(ref zk) = gkr.zk_aux {
            put_secure_fields(&mut out, &zk.gkr_libra_sums)?;
            put_secure_fields(&mut out, &zk.gkr_libra_evals)?;
            put_secure_fields(&mut out, &zk.qkv_cfs)?;
            put_secure_fields(&mut out, &zk.scores_cfs)?;
            put_secure_fields(&mut out, &zk.ctx_cfs)?;
            put_secure_fields(&mut out, &zk.o_cfs)?;
            put_secure_field(&mut out, zk.gkr_mult_masked_claim);
            put_secure_fields(&mut out, &zk.wit_cfs)?;
            put_secure_fields(&mut out, &zk.h_a_x_openings)?;
            put_secure_fields(&mut out, &zk.cont_claims)?;
            put_secure_field(&mut out, zk.cfs_pad_claim);
            put_sumcheck(&mut out, &zk.cfs_binding_sumcheck)?;
            put_secure_field(&mut out, zk.blind_col_opening);
        }
        put_bytes(&mut out, &gkr.cprs_witness_proof)?;
    }
    Ok(out)
}

pub fn read_projection_proof(
    reader: &mut ReceiptReader<'_>,
    expected_openings: usize,
    label: &str,
) -> Result<ProjectionProof, String> {
    let proof = reader.read_sumcheck(&format!("{label} sumcheck"))?;
    let weight_openings = reader.read_secure_fields_exact(
        expected_openings,
        &format!("{label} weight openings"),
    )?;
    Ok(ProjectionProof {
        proof,
        weight_openings,
    })
}

fn read_contraction_proof(
    reader: &mut ReceiptReader<'_>,
    statement: &StageBStatement,
    expected_point_len: usize,
    expected_log_chunks: usize,
    version: u32,
    label: &str,
) -> Result<ContractionProof, String> {
    let proof = reader.read_sumcheck(&format!("{label} sumcheck"))?;
    let tokens = statement.tokens.len();
    let count = reader.read_len(tokens, &format!("{label} claim count"))?;
    if count != tokens {
        return Err(format!(
            "{label} claim count mismatch: expected {tokens}, got {count}"
        ));
    }
    let mut claims = Vec::with_capacity(tokens);
    if version >= STAGE_B_RECEIPT_WIRE_VERSION_GKR {
        if tokens > 0 {
            let shared_point = reader.read_secure_fields_exact(
                expected_point_len,
                &format!("{label} shared point"),
            )?;
            let shared_log_chunks = reader.read_len(
                expected_log_chunks,
                &format!("{label} shared log_chunks"),
            )?;
            if shared_log_chunks != expected_log_chunks {
                return Err(format!(
                    "{label} shared log_chunks mismatch: expected {expected_log_chunks}, got {shared_log_chunks}"
                ));
            }
            let mut running: Vec<SecureField> = Vec::new();
            for (t, token) in statement.tokens.iter().enumerate() {
                let expected_fco_len = token.chunk_ids.len();
                let tag = reader.read_u32_byte(&format!("{label} claim {t} fco tag"))?;
                let full_chunk_openings = match tag {
                    0 => {
                        if expected_fco_len > running.len() {
                            return Err(format!(
                                "{label} claim {t} prefix fco tag exceeds running length"
                            ));
                        }
                        running[..expected_fco_len].to_vec()
                    }
                    1 => {
                        if expected_fco_len <= running.len() {
                            return Err(format!(
                                "{label} claim {t} extend fco tag does not grow running length"
                            ));
                        }
                        let delta = expected_fco_len - running.len();
                        for idx in 0..delta {
                            running.push(reader.read_secure_field(&format!(
                                "{label} claim {t} fco ext {idx}"
                            ))?);
                        }
                        running.clone()
                    }
                    2 => {
                        let mut fresh = Vec::with_capacity(expected_fco_len);
                        for idx in 0..expected_fco_len {
                            fresh.push(reader.read_secure_field(&format!(
                                "{label} claim {t} fco reset {idx}"
                            ))?);
                        }
                        if (expected_fco_len <= running.len()
                            && fresh.as_slice() == &running[..expected_fco_len])
                            || (expected_fco_len > running.len()
                                && fresh[..running.len()] == running[..])
                        {
                            return Err(format!(
                                "non-canonical {label} claim {t} reset fco encoding"
                            ));
                        }
                        running = fresh.clone();
                        fresh
                    }
                    other => {
                        return Err(format!(
                            "invalid {label} claim {t} fco tag {other}; expected 0, 1, or 2"
                        ));
                    }
                };
                let value = reader.read_secure_field(&format!("{label} claim {t} value"))?;
                claims.push(CacheOpeningClaim {
                    point: shared_point.clone(),
                    log_chunks: shared_log_chunks,
                    full_chunk_openings,
                    value,
                });
            }
        }
    } else {
        for (t, token) in statement.tokens.iter().enumerate() {
            let point = reader.read_secure_fields_exact(
                expected_point_len,
                &format!("{label} claim {t} point"),
            )?;
            let log_chunks = reader.read_len(
                expected_log_chunks,
                &format!("{label} claim {t} log_chunks"),
            )?;
            if log_chunks != expected_log_chunks {
                return Err(format!(
                    "{label} claim {t} log_chunks mismatch: expected {expected_log_chunks}, got {log_chunks}"
                ));
            }
            let full_chunk_openings = reader.read_secure_fields_exact(
                token.chunk_ids.len(),
                &format!("{label} claim {t} full_chunk_openings"),
            )?;
            let value = reader.read_secure_field(&format!("{label} claim {t} value"))?;
            claims.push(CacheOpeningClaim {
                point,
                log_chunks,
                full_chunk_openings,
                value,
            });
        }
    }
    Ok(ContractionProof { proof, claims })
}

pub fn decode_stage_b_receipt_bytes(
    bytes: &[u8],
    statement: &StageBStatement,
) -> Result<StageBProof, String> {
    statement.shape.validate()?;
    statement.layout.validate()?;
    if statement.tokens.is_empty() || !statement.tokens.len().is_power_of_two() {
        return Err("Stage B batch must be a power of two".into());
    }
    let lengths = statement.cache_lengths();
    let padded_all = lengths
        .iter()
        .map(|&l| statement.layout.padded_positions(l))
        .max()
        .unwrap();
    let log_rows = padded_all.ilog2() as usize;
    let log_chunks = log_rows - statement.layout.chunk.ilog2() as usize;
    let point_len = log_rows + statement.shape.kv_dim().ilog2() as usize;

    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(STAGE_B_RECEIPT_MAGIC, "Stage B receipt magic")?;
    let version = reader.read_u32("Stage B receipt version")?;
    if version != STAGE_B_RECEIPT_WIRE_VERSION
        && version != STAGE_B_RECEIPT_WIRE_VERSION_GKR
        && version != STAGE_B_RECEIPT_WIRE_VERSION_ZK_GKR
    {
        return Err(format!(
            "unsupported Stage B receipt version {version}; expected {STAGE_B_RECEIPT_WIRE_VERSION}, {STAGE_B_RECEIPT_WIRE_VERSION_GKR}, or {STAGE_B_RECEIPT_WIRE_VERSION_ZK_GKR}"
        ));
    }
    let qkv = read_projection_proof(&mut reader, 3, "Stage B qkv")?;
    let scores = read_contraction_proof(
        &mut reader,
        statement,
        point_len,
        log_chunks,
        version,
        "Stage B scores",
    )?;
    let context = read_contraction_proof(
        &mut reader,
        statement,
        point_len,
        log_chunks,
        version,
        "Stage B context",
    )?;
    let o = read_projection_proof(&mut reader, 1, "Stage B o")?;
    let attention_leaf_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "Stage B attention leaf proof")?
        .to_vec();
    let cache_chunk_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "Stage B cache chunk proof")?
        .to_vec();
    let gkr_witness = if version >= STAGE_B_RECEIPT_WIRE_VERSION_GKR {
        let b = statement.tokens.len();
        let d = statement.shape.dimension;
        let exp_queries = stage_b_exp_query_count(statement);
        let is_zk_v3 = version == STAGE_B_RECEIPT_WIRE_VERSION_ZK_GKR;
        let n_b_wit = if is_zk_v3 {
            stage_b_witness_n_for_zk(b, d, exp_queries)
        } else {
            stage_b_witness_n_with_queries(b, d, exp_queries)
        };
        let mut witness_commitment = [0u8; 32];
        for byte in &mut witness_commitment {
            *byte = reader.read_u32_byte("Stage B witness commitment byte")?;
        }
        let qkv_claims_vec = reader.read_secure_fields_exact(3, "Stage B qkv_claims")?;
        let qkv_claims: [SecureField; 3] = qkv_claims_vec
            .try_into()
            .map_err(|_| "Stage B qkv_claims count mismatch".to_string())?;
        let x1_opening = reader.read_secure_field("Stage B x1_opening")?;
        let expected_sc_len = if is_zk_v3 { 1 } else { b };
        let scores_claims =
            reader.read_secure_fields_exact(expected_sc_len, "Stage B scores_claims")?;
        let context_claims =
            reader.read_secure_fields_exact(expected_sc_len, "Stage B context_claims")?;
        let context_left_openings =
            reader.read_secure_fields_exact(b, "Stage B context_left_openings")?;
        let o_claim = reader.read_secure_field("Stage B o_claim")?;
        let ctx_opening = reader.read_secure_field("Stage B ctx_opening")?;
        let gkr_proof = if is_zk_v3 {
            read_gkr_batch_proof(
                &mut reader,
                &[n_b_wit, LOG_EXP_TABLE_LEN + 1, ZK_B_MASK_LOG_ROWS],
            )?
        } else {
            read_gkr_batch_proof(&mut reader, &[n_b_wit, LOG_EXP_TABLE_LEN])?
        };
        let witness_sumcheck = reader.read_sumcheck("Stage B witness sumcheck")?;
        let witness_openings_vec = reader.read_secure_fields_exact(
            STAGE_B_WITNESS_COLS,
            "Stage B witness openings",
        )?;
        let witness_openings: [SecureField; STAGE_B_WITNESS_COLS] = witness_openings_vec
            .try_into()
            .map_err(|_| "Stage B witness openings count mismatch".to_string())?;
        let zk_aux = if is_zk_v3 {
            let gkr_libra_sums =
                reader.read_secure_fields_exact(n_b_wit, "Stage B v3 gkr_libra_sums")?;
            let gkr_libra_evals =
                reader.read_secure_fields_exact(n_b_wit, "Stage B v3 gkr_libra_evals")?;
            let qkv_cfs: [SecureField; 2] = reader
                .read_secure_fields_exact(2, "Stage B v3 qkv_cfs")?
                .try_into()
                .unwrap();
            let scores_cfs: [SecureField; 2] = reader
                .read_secure_fields_exact(2, "Stage B v3 scores_cfs")?
                .try_into()
                .unwrap();
            let ctx_cfs: [SecureField; 2] = reader
                .read_secure_fields_exact(2, "Stage B v3 ctx_cfs")?
                .try_into()
                .unwrap();
            let o_cfs: [SecureField; 2] = reader
                .read_secure_fields_exact(2, "Stage B v3 o_cfs")?
                .try_into()
                .unwrap();
            let gkr_mult_masked_claim =
                reader.read_secure_field("Stage B v3 gkr_mult_masked_claim")?;
            let wit_cfs: [SecureField; 2] = reader
                .read_secure_fields_exact(2, "Stage B v3 wit_cfs")?
                .try_into()
                .unwrap();
            let h_a_x_openings: [SecureField; 3] = reader
                .read_secure_fields_exact(3, "Stage B v3 h_a_x_openings")?
                .try_into()
                .unwrap();
            let cont_len = reader.read_len(3, "Stage B v3 cont_claims count")?;
            if cont_len != 0 && cont_len != 3 {
                return Err(format!(
                    "Stage B v3 cont_claims count must be 0 or 3, got {cont_len}"
                ));
            }
            let cont_claims = if cont_len == 0 {
                Vec::new()
            } else {
                vec![
                    reader.read_secure_field("Stage B v3 cont_claims 0")?,
                    reader.read_secure_field("Stage B v3 cont_claims 1")?,
                    reader.read_secure_field("Stage B v3 cont_claims 2")?,
                ]
            };
            let cfs_pad_claim = reader.read_secure_field("Stage B v3 cfs_pad_claim")?;
            let cfs_binding_sumcheck =
                reader.read_sumcheck("Stage B v3 cfs_binding_sumcheck")?;
            let blind_col_opening =
                reader.read_secure_field("Stage B v3 blind_col_opening")?;
            Some(StageBZkWitnessAux {
                gkr_libra_sums,
                gkr_libra_evals,
                qkv_cfs,
                scores_cfs,
                ctx_cfs,
                o_cfs,
                gkr_mult_masked_claim,
                wit_cfs,
                h_a_x_openings,
                cont_claims,
                cfs_pad_claim,
                cfs_binding_sumcheck,
                blind_col_opening,
            })
        } else {
            None
        };
        let cprs_witness_proof = reader
            .read_bytes(MAX_WIRE_RECEIPT_BYTES, "Stage B cprs witness proof")?
            .to_vec();
        Some(StageBGkrWitnessProof {
            witness_commitment,
            qkv_claims,
            x1_opening,
            scores_claims,
            context_claims,
            context_left_openings,
            o_claim,
            ctx_opening,
            gkr_proof,
            witness_sumcheck,
            witness_openings,
            zk_aux,
            cprs_witness_proof,
        })
    } else {
        None
    };
    reader.finish()?;
    let proof = StageBProof {
        qkv,
        scores,
        context,
        o,
        attention_leaf_proof,
        cache_chunk_proof,
        gkr_witness,
    };
    if stage_b_receipt_bytes(&proof)? != bytes {
        return Err("Stage B receipt bytes are not canonical".into());
    }
    Ok(proof)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingStatement {
    pub vocab_size: usize,
    pub dimension: usize,
    pub embedding_shift: u32,
    pub embedding_commitment: Hash,
    pub token_ids: Vec<u32>,
    pub h_out: Vec<Vec<i16>>,
    pub binding: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct EmbeddingProof {
    pub projection: ProjectionProof,
    pub leaf_proof: Vec<u8>,
}

impl EmbeddingStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(EMBEDDING_DOMAIN);
        h.update((self.vocab_size as u64).to_le_bytes());
        h.update((self.dimension as u64).to_le_bytes());
        h.update(self.embedding_shift.to_le_bytes());
        h.update(self.embedding_commitment);
        h.update(self.binding);
        h.update((self.token_ids.len() as u64).to_le_bytes());
        for &id in &self.token_ids {
            h.update(id.to_le_bytes());
        }
        h.update((self.h_out.len() as u64).to_le_bytes());
        for row in &self.h_out {
            update_i16(&mut h, row);
        }
        h.finalize().into()
    }

    fn validate_and_build_inputs(&self) -> Result<Vec<Vec<i32>>, String> {
        if self.vocab_size < 2 || self.dimension < 2 {
            return Err("embedding vocab_size and dimension must be at least 2".into());
        }
        if self.embedding_shift > 7 {
            return Err("embedding_shift must be in 0..=7".into());
        }
        let batch = self.token_ids.len();
        if batch == 0 || !batch.is_power_of_two() || self.h_out.len() != batch {
            return Err("embedding batch must be a power of two matching h_out".into());
        }
        let mut emb_raw = Vec::with_capacity(batch);
        for (&token_id, row) in self.token_ids.iter().zip(&self.h_out) {
            let idx = token_id as usize;
            if idx >= self.vocab_size {
                return Err("embedding token_id exceeds vocab_size".into());
            }
            if row.len() != self.dimension {
                return Err("embedding h_out row length does not match dimension".into());
            }
            let mut raw_row = Vec::with_capacity(self.dimension);
            for &val in row {
                let unshifted = val >> self.embedding_shift;
                if (unshifted << self.embedding_shift) != val || !(-128..=127).contains(&unshifted)
                {
                    return Err("embedding h_out entry is not a shifted i8 table value".into());
                }
                raw_row.push(i32::from(unshifted));
            }
            emb_raw.push(raw_row);
        }
        Ok(emb_raw)
    }
}

pub fn verify_embedding_lookup(
    statement: &EmbeddingStatement,
    profile: &SecurityProfile,
    proof: &EmbeddingProof,
) -> Result<(), String> {
    let emb_raw = statement.validate_and_build_inputs()?;
    let params = projection_leaf_params(statement.vocab_size, statement.dimension, profile)?;
    let digest = statement.digest();
    let mut channel = StageBStatement::channel_from_digest(&digest);
    let claims = verify_onehot_projection(
        &statement.token_ids,
        statement.vocab_size,
        &emb_raw,
        &proof.projection,
        &mut channel,
    )?;
    verify_projection_leaf_opening(
        &statement.embedding_commitment,
        &params,
        b"embedding-leaf",
        &claims.point,
        claims.openings[0],
        &digest,
        &proof.leaf_proof,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LmHeadStatement {
    pub dimension: usize,
    pub vocab_size: usize,
    pub norm_fraction_bits: i32,
    pub lm_head_commitment: Hash,
    pub gamma_final: Vec<i16>,
    pub h_in: Vec<Vec<i16>>,
    pub logits_raw: Vec<Vec<i32>>,
    pub binding: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct LmHeadProof {
    pub projection: ProjectionProof,
    pub leaf_proof: Vec<u8>,
}

impl LmHeadStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(LM_HEAD_DOMAIN);
        h.update((self.dimension as u64).to_le_bytes());
        h.update((self.vocab_size as u64).to_le_bytes());
        h.update(self.norm_fraction_bits.to_le_bytes());
        h.update(self.lm_head_commitment);
        h.update(self.binding);
        update_i16(&mut h, &self.gamma_final);
        h.update((self.h_in.len() as u64).to_le_bytes());
        for row in &self.h_in {
            update_i16(&mut h, row);
        }
        h.update((self.logits_raw.len() as u64).to_le_bytes());
        let row_digests: Vec<[u8; 32]> = self
            .logits_raw
            .iter()
            .map(|row| {
                let mut rh = Blake2s256::new();
                rh.update(b"stwo-ml:stage-b:lm-head-row:v1");
                update_i32(&mut rh, row);
                rh.finalize().into()
            })
            .collect();
        for rd in &row_digests {
            h.update(rd);
        }
        h.finalize().into()
    }

    pub fn argmax_token_ids(&self) -> Result<Vec<u32>, String> {
        if self.vocab_size < 2 {
            return Err("LM head vocab_size must be at least 2".into());
        }
        let batch = self.h_in.len();
        if batch == 0 || !batch.is_power_of_two() || self.logits_raw.len() != batch {
            return Err("LM head batch must be a power of two matching logits_raw".into());
        }
        let mut out = Vec::with_capacity(batch);
        for row in &self.logits_raw {
            if row.len() != self.vocab_size {
                return Err("LM head row shape mismatch".into());
            }
            let mut best_id = 0u32;
            let mut best_logit = row[0];
            for (v, &logit) in row.iter().enumerate().skip(1) {
                if logit > best_logit {
                    best_logit = logit;
                    best_id = v as u32;
                }
            }
            out.push(best_id);
        }
        Ok(out)
    }

    pub fn emitted_next_token(&self) -> Result<u32, String> {
        let ids = self.argmax_token_ids()?;
        ids.last()
            .copied()
            .ok_or_else(|| "LM head has no emitted token rows".into())
    }

    pub fn validate_and_recompute_norm(&self) -> Result<Vec<Vec<i8>>, String> {
        if self.dimension < 2 || self.vocab_size < 2 {
            return Err("LM head dimension and vocab_size must be at least 2".into());
        }
        if self.gamma_final.len() != self.dimension {
            return Err("LM head gamma_final length does not match dimension".into());
        }
        let _ = self.argmax_token_ids()?;
        let batch = self.h_in.len();
        let mut x_final = Vec::with_capacity(batch);
        for (h_row, logit_row) in self.h_in.iter().zip(&self.logits_raw) {
            if h_row.len() != self.dimension || logit_row.len() != self.vocab_size {
                return Err("LM head row shape mismatch".into());
            }
            let normed = rms_norm(h_row, &self.gamma_final, self.norm_fraction_bits, 8)?;
            x_final.push(normed.into_iter().map(|v| v as i8).collect());
        }
        Ok(x_final)
    }
}

pub fn verify_lm_head(
    statement: &LmHeadStatement,
    profile: &SecurityProfile,
    proof: &LmHeadProof,
) -> Result<(), String> {
    let x_final = statement.validate_and_recompute_norm()?;
    let params = projection_leaf_params(statement.vocab_size, statement.dimension, profile)?;
    let digest = statement.digest();
    let mut channel = StageBStatement::channel_from_digest(&digest);
    let claims = verify_transposed_projection(
        &x_final,
        &statement.logits_raw,
        &proof.projection,
        &mut channel,
    )?;
    verify_projection_leaf_opening(
        &statement.lm_head_commitment,
        &params,
        b"lm-head-leaf",
        &claims.point,
        claims.openings[0],
        &digest,
        &proof.leaf_proof,
    )
}
