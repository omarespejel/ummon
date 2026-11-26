//! Standalone `wasm32`-compatible verifier for Absorbed Multi-Head Latent Attention (`StageBMla`,
//! wire magic `STWOMLA1`), covering `DeepSeek-V3 / R1` and `Kimi K2` without ever expanding into
//! dense `(K, V)` heads (`576 B/token` compressed `(c_kv, k_pe)` latent KV cache, `72 KiB` per
//! 128-token chunk, and 7-matrix compact CPRS leaf opening).
//!
//! The wire format and every check are defined in this file.

use std::collections::BTreeMap;

use blake2::{Blake2s256, Digest};
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::qm31::SecureField;

use crate::cprs::{
    self, i8_to_canonical_m31_u32, stwo_point_to_cprs, to_cprs_qm31,
    verify_heterogeneous_strict_context, Hash, Params, QM31,
};
use crate::stage_a::{requant_with_rounding, RequantSpec, StageARoundingMode};
use crate::stage_b::{
    group_params_with_f, put_projection_proof, read_projection_proof, rms_norm, rope_tables,
    softmax_weights, verify_cache_claims_batch, verify_context, verify_projections, verify_scores,
    AttentionShape, CacheLayout, CacheOpeningClaim, ContractionProof, ProjectionProof,
    StageBStatement, ROPE_BITS, WEIGHT_BITS,
};
use crate::{
    eq_evals, eq_evals_i64, i8_to_m31, mle_at, partially_verify, put_bytes,
    put_secure_field, put_secure_fields, put_sumcheck, put_u32, put_u64, restrict_rows_i8_fast,
    ReceiptReader, SecurityProfile, SumcheckProof, MAX_WIRE_RECEIPT_BYTES,
};

pub const STAGE_B_MLA_RECEIPT_MAGIC: &[u8; 8] = b"STWOMLA1";
pub const STAGE_B_MLA_RECEIPT_VERSION: u32 = 1;
pub const MLA_STATEMENT_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:statement:v1";
pub const MLA_REGISTERED_ANCHOR_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:registered-anchor:v1";
pub const MLA_STEP_CACHE_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:step-cache:v1";
pub const MLA_LATENT_LEAF_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:latent-leaf:v1";
pub const MLA_LATENT_CACHE_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:latent-cache:v1";
pub const MLA_OUTER_DOMAIN: &[u8] = b"stwo-ml:stage-b-mla:cprs-outer:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaShape {
    /// Model residual dimension `d` (`7168` on DeepSeek-V3/R1 & Kimi K2; may be non-power-of-two).
    pub dimension: usize,
    /// Query attention head count `H` (`128` on DeepSeek-V3/R1; power of two).
    pub heads: usize,
    /// Query low-rank compression dimension `d_c^Q` (`1536` on DeepSeek-V3/R1; may be non-power-of-two).
    pub q_lora_rank: usize,
    /// Joint KV latent compression rank `d_c` (`512` on DeepSeek-V3/R1; power of two).
    pub kv_lora_rank: usize,
    /// Shared decoupled RoPE key/query head dimension `d_h^R` (`64` on DeepSeek-V3/R1; even power of two).
    pub qk_rope_head_dim: usize,
    /// Per-head latent value projection dimension `d_h^V` (`128` on DeepSeek-V3/R1; power of two).
    pub v_head_dim: usize,
}

impl MlaShape {
    pub const fn q_abs_dim(self) -> usize {
        self.heads * self.kv_lora_rank
    }

    pub const fn q_pe_dim(self) -> usize {
        self.heads * self.qk_rope_head_dim
    }

    pub const fn v_dim(self) -> usize {
        self.heads * self.v_head_dim
    }

    pub const fn cached_bytes_per_token(self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    pub const fn nope_attention_shape(self) -> AttentionShape {
        AttentionShape {
            dimension: self.heads * self.kv_lora_rank,
            heads: self.heads,
            kv_heads: 1,
            head_dim: self.kv_lora_rank,
        }
    }

    pub const fn rope_attention_shape(self) -> AttentionShape {
        AttentionShape {
            dimension: self.heads * self.qk_rope_head_dim,
            heads: self.heads,
            kv_heads: 1,
            head_dim: self.qk_rope_head_dim,
        }
    }

    pub fn validate(self) -> Result<(), String> {
        if self.dimension < 4 || self.dimension % 2 != 0 || self.dimension > (1 << 15) {
            return Err("MLA dimension must be an even integer in 4..=32768".into());
        }
        if self.heads == 0 || !self.heads.is_power_of_two() || self.heads > 256 {
            return Err("MLA heads must be a power of two in 1..=256".into());
        }
        if self.q_lora_rank < 4 || self.q_lora_rank > (1 << 14) {
            return Err("MLA q_lora_rank must be in 4..=16384".into());
        }
        if self.kv_lora_rank < 4 || !self.kv_lora_rank.is_power_of_two() || self.kv_lora_rank > 4096
        {
            return Err("MLA kv_lora_rank must be a power of two in 4..=4096".into());
        }
        if self.qk_rope_head_dim < 4
            || self.qk_rope_head_dim % 2 != 0
            || !self.qk_rope_head_dim.is_power_of_two()
            || self.qk_rope_head_dim > 512
        {
            return Err("MLA qk_rope_head_dim must be an even power of two in 4..=512".into());
        }
        if self.v_head_dim < 4 || !self.v_head_dim.is_power_of_two() || self.v_head_dim > 1024 {
            return Err("MLA v_head_dim must be a power of two in 4..=1024".into());
        }
        Ok(())
    }

    pub const fn deepseek_v3() -> Self {
        Self {
            dimension: 7168,
            heads: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaAttentionPolicy {
    pub rounding: StageARoundingMode,
    /// `c_q_raw (i32) -> c_q_pre (i16 at Q7)` feeding `RMSNorm_Q`.
    pub dq_pre: RequantSpec,
    /// `c_kv_raw (i32) -> c_kv_pre (i16 at Q7)` feeding `RMSNorm_KV`.
    pub dkv_pre: RequantSpec,
    /// `k_pe_raw (i32) -> i8` feeding decoupled `RoPE_p`.
    pub kr: RequantSpec,
    /// `q_abs_raw (i32) -> q_abs (i8)`.
    pub q_abs: RequantSpec,
    /// `q_pe_raw (i32) -> i8` feeding query `RoPE_p`.
    pub q_pe: RequantSpec,
    /// `v_raw (i32) -> v_q8 (i8)`.
    pub uv: RequantSpec,
    /// `o_raw (i32) -> i12 at Q7` residual add.
    pub output: RequantSpec,
    /// Per-head integer multipliers reconciling absorbed `W_Q_abs` scales (`len == heads`, each `> 0`).
    pub alpha_nope: Vec<u16>,
    /// Shared RoPE score multiplier (`> 0`).
    pub alpha_rope: u16,
    /// Right shift reconciling `s_nope * alpha_nope[h] + s_rope * alpha_rope` into `s_{h,j}`.
    pub reconcile_shift: u32,
    /// Softmax `T_exp` shift (`(max - s) >> score_shift`).
    pub score_shift: u32,
    pub norm1_fraction_bits: i32,
    pub norm_q_fraction_bits: i32,
    pub norm_kv_fraction_bits: i32,
    pub norm2_fraction_bits: i32,
}

impl MlaAttentionPolicy {
    pub fn calibrated_v1(heads: usize) -> Self {
        Self {
            rounding: StageARoundingMode::TiesToEven,
            dq_pre: RequantSpec {
                multiplier: 1,
                shift: 6,
                output_bits: 16,
            },
            dkv_pre: RequantSpec {
                multiplier: 1,
                shift: 6,
                output_bits: 16,
            },
            kr: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 8,
            },
            q_abs: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 8,
            },
            q_pe: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 8,
            },
            uv: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 8,
            },
            output: RequantSpec {
                multiplier: 7,
                shift: 11,
                output_bits: 12,
            },
            alpha_nope: vec![16u16; heads],
            alpha_rope: 16,
            reconcile_shift: 4,
            score_shift: 8,
            norm1_fraction_bits: 5,
            norm_q_fraction_bits: 5,
            norm_kv_fraction_bits: 5,
            norm2_fraction_bits: 5,
        }
    }

    pub fn validate(&self, shape: MlaShape) -> Result<(), String> {
        if self.dq_pre.output_bits != 16 || self.dkv_pre.output_bits != 16 {
            return Err("MLA dq_pre and dkv_pre must target 16-bit pre-norm latents".into());
        }
        if self.kr.output_bits != 8
            || self.q_abs.output_bits != 8
            || self.q_pe.output_bits != 8
            || self.uv.output_bits != 8
        {
            return Err("MLA kr, q_abs, q_pe, and uv specs must target 8-bit outputs".into());
        }
        if self.output.output_bits != 12 {
            return Err("MLA output spec must target 12-bit residual increment".into());
        }
        if self.alpha_nope.len() != shape.heads || self.alpha_nope.contains(&0) {
            return Err("MLA alpha_nope must have one positive multiplier per head".into());
        }
        if self.alpha_rope == 0 {
            return Err("MLA alpha_rope must be positive".into());
        }
        if !(1..=30).contains(&self.reconcile_shift) || !(1..=30).contains(&self.score_shift) {
            return Err("MLA reconcile_shift and score_shift must be in 1..=30".into());
        }
        Ok(())
    }

    pub fn update_digest(&self, h: &mut Blake2s256) {
        let r_tag: u8 = match self.rounding {
            StageARoundingMode::TiesToEven => 0,
            StageARoundingMode::HalfTowardPositiveInfinity => 1,
        };
        h.update([r_tag]);
        for spec in [
            self.dq_pre,
            self.dkv_pre,
            self.kr,
            self.q_abs,
            self.q_pe,
            self.uv,
            self.output,
        ] {
            h.update(spec.multiplier.to_le_bytes());
            h.update(spec.shift.to_le_bytes());
            h.update(spec.output_bits.to_le_bytes());
        }
        h.update((self.alpha_nope.len() as u64).to_le_bytes());
        for &a in &self.alpha_nope {
            h.update(a.to_le_bytes());
        }
        h.update(self.alpha_rope.to_le_bytes());
        h.update(self.reconcile_shift.to_le_bytes());
        h.update(self.score_shift.to_le_bytes());
        for fb in [
            self.norm1_fraction_bits,
            self.norm_q_fraction_bits,
            self.norm_kv_fraction_bits,
            self.norm2_fraction_bits,
        ] {
            h.update(fb.to_le_bytes());
        }
    }
}

pub fn rope_dim(values: &[i8], head_dim: usize, position: u64) -> Result<Vec<i8>, String> {
    if head_dim == 0 || head_dim % 2 != 0 || values.is_empty() || values.len() % head_dim != 0 {
        return Err("MLA rope_dim input length must be a positive multiple of even head_dim".into());
    }
    let heads = values.len() / head_dim;
    let (cos, sin) = rope_tables(head_dim, position);
    let half = head_dim / 2;
    let round = |v: i64| -> Result<i8, String> {
        let r = (v + (1i64 << (ROPE_BITS - 1))) >> ROPE_BITS;
        i8::try_from(r).map_err(|_| "MLA rope output exceeds i8".to_string())
    };
    let mut out = vec![0i8; values.len()];
    for head in 0..heads {
        let base = head * head_dim;
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

pub fn requant_to_i16_spec(
    values: &[i32],
    spec: RequantSpec,
    rounding: StageARoundingMode,
) -> Result<Vec<i16>, String> {
    values
        .iter()
        .map(|&v| {
            let r = requant_with_rounding(i128::from(v), spec, rounding)?.value;
            i16::try_from(r).map_err(|_| "MLA pre-norm latent exceeds i16".to_string())
        })
        .collect()
}

pub fn requant_to_i8_spec(
    values: &[i32],
    spec: RequantSpec,
    rounding: StageARoundingMode,
) -> Result<Vec<i8>, String> {
    values
        .iter()
        .map(|&v| {
            let r = requant_with_rounding(i128::from(v), spec, rounding)?.value;
            i8::try_from(r).map_err(|_| "MLA requant output exceeds i8".to_string())
        })
        .collect()
}

pub fn mla_u_q8_from_raw(u_raw: &[i32]) -> Result<Vec<i8>, String> {
    let spec = RequantSpec {
        multiplier: 1,
        shift: WEIGHT_BITS,
        output_bits: 8,
    };
    requant_to_i8_spec(u_raw, spec, StageARoundingMode::TiesToEven)
}

pub fn mla_residual_add(
    h_in: &[i16],
    o_raw: &[i32],
    policy: &MlaAttentionPolicy,
) -> Result<Vec<i16>, String> {
    h_in.iter()
        .zip(o_raw)
        .map(|(&h, &o)| {
            let dq = requant_with_rounding(i128::from(o), policy.output, policy.rounding)?.value;
            i16::try_from(i32::from(h) + dq).map_err(|_| "MLA residual exceeds i16".to_string())
        })
        .collect()
}

pub fn reconcile_mla_head_scores(
    s_nope: &[i32],
    s_rope: &[i32],
    alpha_nope: u16,
    alpha_rope: u16,
    reconcile_shift: u32,
) -> Result<Vec<i32>, String> {
    if s_nope.len() != s_rope.len() {
        return Err("MLA s_nope and s_rope lengths differ".into());
    }
    let half = 1i64 << (reconcile_shift - 1);
    s_nope
        .iter()
        .zip(s_rope)
        .map(|(&sn, &sr)| {
            let num = i64::from(sn) * i64::from(alpha_nope)
                + i64::from(sr) * i64::from(alpha_rope)
                + half;
            i32::try_from(num >> reconcile_shift)
                .map_err(|_| "MLA reconciled score exceeds i32".to_string())
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaPublicStep {
    pub position: u64,
    pub h_in: Vec<i16>,
    pub c_q_raw: Vec<i32>,
    pub c_kv_raw: Vec<i32>,
    pub k_pe_raw: Vec<i32>,
    pub q_abs_raw: Vec<i32>,
    pub q_pe_raw: Vec<i32>,
    pub s_nope: Vec<Vec<i32>>,
    pub s_rope: Vec<Vec<i32>>,
    pub u_raw: Vec<i32>,
    pub v_raw: Vec<i32>,
    pub o_raw: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaRecomputedStep {
    pub x1: Vec<i8>,
    pub c_q: Vec<i8>,
    /// Compressed KV latent appended to `cache_C[position]` (`512` bytes on DeepSeek-V3/R1).
    pub c_kv: Vec<i8>,
    /// Shared decoupled RoPE key appended to `cache_K[position]` (`64` bytes on DeepSeek-V3/R1).
    pub k_pe: Vec<i8>,
    pub q_abs: Vec<i8>,
    pub q_pe: Vec<i8>,
    pub weights: Vec<Vec<u16>>,
    pub u_q8: Vec<i8>,
    pub v_q8: Vec<i8>,
    pub a: Vec<i16>,
    pub x: Vec<i16>,
}

pub fn check_mla_public_step(
    shape: MlaShape,
    policy: &MlaAttentionPolicy,
    gamma1: &[i16],
    gamma_q: &[i16],
    gamma_kv: &[i16],
    gamma2: &[i16],
    step: &MlaPublicStep,
) -> Result<MlaRecomputedStep, String> {
    shape.validate()?;
    policy.validate(shape)?;
    if step.h_in.len() != shape.dimension
        || step.o_raw.len() != shape.dimension
        || step.c_q_raw.len() != shape.q_lora_rank
        || step.c_kv_raw.len() != shape.kv_lora_rank
        || step.k_pe_raw.len() != shape.qk_rope_head_dim
        || step.q_abs_raw.len() != shape.q_abs_dim()
        || step.q_pe_raw.len() != shape.q_pe_dim()
        || step.u_raw.len() != shape.q_abs_dim()
        || step.v_raw.len() != shape.v_dim()
    {
        return Err("MLA step vector length mismatch".into());
    }
    let context = step.position as usize + 1;
    if step.s_nope.len() != shape.heads || step.s_rope.len() != shape.heads {
        return Err("MLA step must carry one s_nope and s_rope vector per head".into());
    }
    let x1: Vec<i8> = rms_norm(&step.h_in, gamma1, policy.norm1_fraction_bits, 8)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let c_q_pre = requant_to_i16_spec(&step.c_q_raw, policy.dq_pre, policy.rounding)?;
    let c_q: Vec<i8> = rms_norm(&c_q_pre, gamma_q, policy.norm_q_fraction_bits, 8)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let c_kv_pre = requant_to_i16_spec(&step.c_kv_raw, policy.dkv_pre, policy.rounding)?;
    let c_kv: Vec<i8> = rms_norm(&c_kv_pre, gamma_kv, policy.norm_kv_fraction_bits, 8)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let k_pe = rope_dim(
        &requant_to_i8_spec(&step.k_pe_raw, policy.kr, policy.rounding)?,
        shape.qk_rope_head_dim,
        step.position,
    )?;
    let q_abs = requant_to_i8_spec(&step.q_abs_raw, policy.q_abs, policy.rounding)?;
    let q_pe = rope_dim(
        &requant_to_i8_spec(&step.q_pe_raw, policy.q_pe, policy.rounding)?,
        shape.qk_rope_head_dim,
        step.position,
    )?;
    let mut weights = Vec::with_capacity(shape.heads);
    for h in 0..shape.heads {
        if step.s_nope[h].len() != context || step.s_rope[h].len() != context {
            return Err("MLA score vector length does not match context length".into());
        }
        let s = reconcile_mla_head_scores(
            &step.s_nope[h],
            &step.s_rope[h],
            policy.alpha_nope[h],
            policy.alpha_rope,
            policy.reconcile_shift,
        )?;
        weights.push(softmax_weights(&s, policy.score_shift)?);
    }
    let u_q8 = mla_u_q8_from_raw(&step.u_raw)?;
    let v_q8 = requant_to_i8_spec(&step.v_raw, policy.uv, policy.rounding)?;
    let a = mla_residual_add(&step.h_in, &step.o_raw, policy)?;
    let x: Vec<i16> = rms_norm(&a, gamma2, policy.norm2_fraction_bits, 12)?
        .into_iter()
        .map(|v| v as i16)
        .collect();
    Ok(MlaRecomputedStep {
        x1,
        c_q,
        c_kv,
        k_pe,
        q_abs,
        q_pe,
        weights,
        u_q8,
        v_q8,
        a,
        x,
    })
}

pub fn mla_step_cache_commitment(
    chunk_commitments: &[Hash],
    c_kv_rows: &[Vec<i8>],
    k_pe_rows: &[Vec<i8>],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(MLA_STEP_CACHE_DOMAIN);
    h.update((chunk_commitments.len() as u64).to_le_bytes());
    for c in chunk_commitments {
        h.update(c);
    }
    h.update((c_kv_rows.len() as u64).to_le_bytes());
    for row in c_kv_rows {
        h.update((row.len() as u64).to_le_bytes());
        for &v in row {
            h.update([v as u8]);
        }
    }
    h.update((k_pe_rows.len() as u64).to_le_bytes());
    for row in k_pe_rows {
        h.update((row.len() as u64).to_le_bytes());
        for &v in row {
            h.update([v as u8]);
        }
    }
    h.finalize().into()
}

pub fn compute_mla_leaf_commitment(sub_commitments: &[Hash; 7]) -> Hash {
    let mut h = Blake2s256::new();
    h.update(MLA_LATENT_LEAF_DOMAIN);
    for c in sub_commitments {
        h.update(c);
    }
    h.finalize().into()
}

pub fn compute_mla_chunk_commitment(com_c_kv: &Hash, com_k_pe: &Hash) -> Hash {
    let mut h = Blake2s256::new();
    h.update(MLA_LATENT_CACHE_DOMAIN);
    h.update(com_c_kv);
    h.update(com_k_pe);
    h.finalize().into()
}

pub fn mla_leaf_group_params(
    shape: MlaShape,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<[Params; 7], String> {
    shape.validate()?;
    let log_d = shape.dimension.next_power_of_two().ilog2() as usize;
    let log_cq = shape.q_lora_rank.next_power_of_two().ilog2() as usize;
    let log_ckv = shape.kv_lora_rank.ilog2() as usize;
    let log_kr = shape.qk_rope_head_dim.ilog2() as usize;
    let log_h = shape.heads.ilog2() as usize;
    let log_vh = shape.v_head_dim.ilog2() as usize;
    let ns = [
        log_d + log_cq,            // 0: W_DQ
        log_d + log_ckv,           // 1: W_DKV
        log_d + log_kr,            // 2: W_KR
        log_cq + log_h + log_ckv,  // 3: W_Q_abs
        log_cq + log_h + log_kr,   // 4: W_QR
        log_h + log_ckv + log_vh,  // 5: W_UV
        log_h + log_vh + log_d,    // 6: W_O
    ];
    let mut out = Vec::with_capacity(7);
    for n in ns {
        out.push(group_params_with_f(n, 1, f_override, profile)?);
    }
    out.try_into().map_err(|_| "MLA leaf group count mismatch".into())
}

pub fn mla_cache_chunk_group_params(
    layout: CacheLayout,
    shape: MlaShape,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<(Params, Params), String> {
    layout.validate()?;
    shape.validate()?;
    let log_c = layout.chunk.ilog2() as usize;
    let p_ckv = group_params_with_f(
        log_c + shape.kv_lora_rank.ilog2() as usize,
        1,
        f_override,
        profile,
    )?;
    let p_kpe = group_params_with_f(
        log_c + shape.qk_rope_head_dim.ilog2() as usize,
        1,
        f_override,
        profile,
    )?;
    Ok((p_ckv, p_kpe))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaToken {
    pub step: MlaPublicStep,
    pub chunk_ids: Vec<usize>,
    pub cache_commitment_before: [u8; 32],
    pub cache_commitment_after: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageBMlaStatement {
    pub shape: MlaShape,
    pub policy: MlaAttentionPolicy,
    pub layout: CacheLayout,
    pub cprs_f: Option<usize>,
    pub mla_leaf_commitment: Hash,
    pub gamma1: Vec<i16>,
    pub gamma_q: Vec<i16>,
    pub gamma_kv: Vec<i16>,
    pub gamma2: Vec<i16>,
    pub chunk_commitments: Vec<Hash>,
    /// Public partial-chunk latent rows `(c_kv_rows, k_pe_rows)` per token.
    pub partial_rows: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)>,
    pub tokens: Vec<MlaToken>,
    pub binding: [u8; 32],
}

/// Computes the immutable model-registration anchor for an MLA layer (`Mythos-M4`),
/// pinning `(shape, policy, layout, cprs_f, mla_leaf_commitment, gamma1, gamma_q, gamma_kv, gamma2)`.
#[allow(clippy::too_many_arguments)]
pub fn compute_mla_registered_anchor(
    shape: MlaShape,
    policy: &MlaAttentionPolicy,
    layout: CacheLayout,
    cprs_f: Option<usize>,
    mla_leaf_commitment: &Hash,
    gamma1: &[i16],
    gamma_q: &[i16],
    gamma_kv: &[i16],
    gamma2: &[i16],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(MLA_REGISTERED_ANCHOR_DOMAIN);
    for v in [
        shape.dimension,
        shape.heads,
        shape.q_lora_rank,
        shape.kv_lora_rank,
        shape.qk_rope_head_dim,
        shape.v_head_dim,
        layout.chunk,
        cprs_f.unwrap_or(0),
    ] {
        h.update((v as u64).to_le_bytes());
    }
    policy.update_digest(&mut h);
    h.update(mla_leaf_commitment);
    for g in [gamma1, gamma_q, gamma_kv, gamma2] {
        h.update((g.len() as u64).to_le_bytes());
        for &x in g {
            h.update(x.to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Deterministically checks at model registration time that the absorbed query matrix
/// `W_Q_abs` (`q_lora_rank x (heads * kv_lora_rank)`) equals `requant(W_UQ^(h) * (W_UK^(h))^T)`
/// across every head `h in 0..heads` (`Mythos-M4`).
pub fn verify_mla_absorbed_q_weights(
    shape: MlaShape,
    qk_nope_head_dim: usize,
    w_uq_per_head: &[Vec<i8>],
    w_uk_per_head: &[Vec<i8>],
    w_q_abs_values: &[i8],
    absorb_spec: RequantSpec,
    rounding: StageARoundingMode,
) -> Result<(), String> {
    shape.validate()?;
    if qk_nope_head_dim == 0 {
        return Err("MLA qk_nope_head_dim must be positive".into());
    }
    if w_uq_per_head.len() != shape.heads || w_uk_per_head.len() != shape.heads {
        return Err("MLA absorbed-weight audit requires one W_UQ and W_UK slice per head".into());
    }
    let cols_total = shape.q_abs_dim();
    if w_q_abs_values.len() != shape.q_lora_rank * cols_total {
        return Err("MLA W_Q_abs length does not match q_lora_rank * heads * kv_lora_rank".into());
    }
    for h in 0..shape.heads {
        let uq = &w_uq_per_head[h];
        let uk = &w_uk_per_head[h];
        if uq.len() != shape.q_lora_rank * qk_nope_head_dim {
            return Err(format!("MLA W_UQ[{h}] shape mismatch"));
        }
        if uk.len() != shape.kv_lora_rank * qk_nope_head_dim {
            return Err(format!("MLA W_UK[{h}] shape mismatch"));
        }
        for r in 0..shape.q_lora_rank {
            let uq_row = &uq[r * qk_nope_head_dim..(r + 1) * qk_nope_head_dim];
            for c in 0..shape.kv_lora_rank {
                let uk_row = &uk[c * qk_nope_head_dim..(c + 1) * qk_nope_head_dim];
                let dot: i128 = uq_row
                    .iter()
                    .zip(uk_row)
                    .map(|(&a, &b)| i128::from(a) * i128::from(b))
                    .sum();
                let expected = i8::try_from(requant_with_rounding(dot, absorb_spec, rounding)?.value)
                    .map_err(|_| "MLA absorbed W_Q_abs entry exceeds i8".to_string())?;
                let actual = w_q_abs_values[r * cols_total + h * shape.kv_lora_rank + c];
                if actual != expected {
                    return Err(format!(
                        "MLA absorbed W_Q_abs mismatch at head {h}, row {r}, col {c}: got {actual}, expected {expected}"
                    ));
                }
            }
        }
    }
    Ok(())
}

impl StageBMlaStatement {
    pub fn registered_anchor(&self) -> [u8; 32] {
        compute_mla_registered_anchor(
            self.shape,
            &self.policy,
            self.layout,
            self.cprs_f,
            &self.mla_leaf_commitment,
            &self.gamma1,
            &self.gamma_q,
            &self.gamma_kv,
            &self.gamma2,
        )
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(MLA_STATEMENT_DOMAIN);
        for v in [
            self.shape.dimension,
            self.shape.heads,
            self.shape.q_lora_rank,
            self.shape.kv_lora_rank,
            self.shape.qk_rope_head_dim,
            self.shape.v_head_dim,
            self.layout.chunk,
            self.cprs_f.unwrap_or(0),
        ] {
            h.update((v as u64).to_le_bytes());
        }
        self.policy.update_digest(&mut h);
        h.update(self.mla_leaf_commitment);
        h.update(self.binding);
        for g in [&self.gamma1, &self.gamma_q, &self.gamma_kv, &self.gamma2] {
            h.update((g.len() as u64).to_le_bytes());
            for &x in g {
                h.update(x.to_le_bytes());
            }
        }
        h.update((self.chunk_commitments.len() as u64).to_le_bytes());
        for c in &self.chunk_commitments {
            h.update(c);
        }
        h.update((self.partial_rows.len() as u64).to_le_bytes());
        for (ckv, kpe) in &self.partial_rows {
            h.update((ckv.len() as u64).to_le_bytes());
            for r in ckv {
                h.update((r.len() as u64).to_le_bytes());
                for &b in r {
                    h.update([b as u8]);
                }
            }
            h.update((kpe.len() as u64).to_le_bytes());
            for r in kpe {
                h.update((r.len() as u64).to_le_bytes());
                for &b in r {
                    h.update([b as u8]);
                }
            }
        }
        h.update((self.tokens.len() as u64).to_le_bytes());
        for tok in &self.tokens {
            h.update(tok.step.position.to_le_bytes());
            h.update(tok.cache_commitment_before);
            h.update(tok.cache_commitment_after);
            h.update((tok.chunk_ids.len() as u64).to_le_bytes());
            for &cid in &tok.chunk_ids {
                h.update((cid as u64).to_le_bytes());
            }
            for &x in &tok.step.h_in {
                h.update(x.to_le_bytes());
            }
            for vec_i32 in [
                &tok.step.c_q_raw,
                &tok.step.c_kv_raw,
                &tok.step.k_pe_raw,
                &tok.step.q_abs_raw,
                &tok.step.q_pe_raw,
                &tok.step.u_raw,
                &tok.step.v_raw,
                &tok.step.o_raw,
            ] {
                h.update((vec_i32.len() as u64).to_le_bytes());
                for &v in vec_i32 {
                    h.update(v.to_le_bytes());
                }
            }
            for s_mat in [&tok.step.s_nope, &tok.step.s_rope] {
                h.update((s_mat.len() as u64).to_le_bytes());
                for s_row in s_mat {
                    h.update((s_row.len() as u64).to_le_bytes());
                    for &v in s_row {
                        h.update(v.to_le_bytes());
                    }
                }
            }
        }
        h.finalize().into()
    }

    pub fn validate_and_recompute(
        &self,
        f_override: Option<usize>,
        profile: &SecurityProfile,
    ) -> Result<Vec<MlaRecomputedStep>, String> {
        self.shape.validate()?;
        self.policy.validate(self.shape)?;
        self.layout.validate()?;
        if self.gamma1.len() != self.shape.dimension
            || self.gamma2.len() != self.shape.dimension
            || self.gamma_q.len() != self.shape.q_lora_rank
            || self.gamma_kv.len() != self.shape.kv_lora_rank
        {
            return Err("MLA gamma vector lengths do not match shape".into());
        }
        let batch = self.tokens.len();
        if batch == 0 || !batch.is_power_of_two() || self.partial_rows.len() != batch {
            return Err("MLA batch must be a positive power of two matching partial_rows".into());
        }
        let (p_ckv, p_kpe) = mla_cache_chunk_group_params(self.layout, self.shape, f_override, profile)?;
        let mut recomputed = Vec::with_capacity(batch);
        let mut by_chunks: BTreeMap<&[usize], Vec<usize>> = BTreeMap::new();
        for (t, tok) in self.tokens.iter().enumerate() {
            let rec = check_mla_public_step(
                self.shape,
                &self.policy,
                &self.gamma1,
                &self.gamma_q,
                &self.gamma_kv,
                &self.gamma2,
                &tok.step,
            )?;
            let context = tok.step.position as usize + 1;
            let expected_full = self.layout.full_chunks(context);
            if tok.chunk_ids.len() != expected_full
                || tok.chunk_ids.iter().any(|&c| c >= self.chunk_commitments.len())
            {
                return Err("MLA token chunk_ids mismatch".into());
            }
            let expected_partial = context - expected_full * self.layout.chunk;
            let (ref ckv_rows, ref kpe_rows) = self.partial_rows[t];
            if ckv_rows.len() != expected_partial
                || kpe_rows.len() != expected_partial
                || ckv_rows.iter().any(|r| r.len() != self.shape.kv_lora_rank)
                || kpe_rows.iter().any(|r| r.len() != self.shape.qk_rope_head_dim)
            {
                return Err("MLA partial_rows shape mismatch".into());
            }
            if ckv_rows.last() != Some(&rec.c_kv) || kpe_rows.last() != Some(&rec.k_pe) {
                return Err("MLA partial_rows last entry does not equal recomputed (c_kv, k_pe)".into());
            }
            let full_hashes: Vec<Hash> = tok
                .chunk_ids
                .iter()
                .map(|&id| self.chunk_commitments[id])
                .collect();
            let before = mla_step_cache_commitment(
                &full_hashes,
                &ckv_rows[..expected_partial - 1],
                &kpe_rows[..expected_partial - 1],
            );
            if tok.cache_commitment_before != before {
                return Err("MLA cache_commitment_before mismatch".into());
            }
            let after = if expected_partial == self.layout.chunk {
                let pad_mat = |rows: &[Vec<i8>], cols: usize| -> Vec<u32> {
                    rows.iter()
                        .flat_map(|r| {
                            assert_eq!(r.len(), cols);
                            r.iter().map(|&v| i8_to_canonical_m31_u32(v))
                        })
                        .collect()
                };
                let ckv_u32 = pad_mat(ckv_rows, self.shape.kv_lora_rank);
                let kpe_u32 = pad_mat(kpe_rows, self.shape.qk_rope_head_dim);
                let com_c = cprs::register::register_group_commitment(&[&ckv_u32], &p_ckv)
                    .map_err(|e| e.to_string())?;
                let com_k = cprs::register::register_group_commitment(&[&kpe_u32], &p_kpe)
                    .map_err(|e| e.to_string())?;
                let roll = compute_mla_chunk_commitment(&com_c, &com_k);
                let mut rolled = full_hashes;
                rolled.push(roll);
                mla_step_cache_commitment(&rolled, &[], &[])
            } else {
                mla_step_cache_commitment(&full_hashes, ckv_rows, kpe_rows)
            };
            if tok.cache_commitment_after != after {
                return Err("MLA cache_commitment_after mismatch".into());
            }
            by_chunks.entry(&tok.chunk_ids).or_default().push(t);
            recomputed.push(rec);
        }
        for (_, mut ts) in by_chunks {
            ts.sort_by_key(|&t| self.partial_rows[t].0.len());
            for pair in ts.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let la = self.partial_rows[a].0.len();
                if self.partial_rows[b].0[..la] != self.partial_rows[a].0[..]
                    || self.partial_rows[b].1[..la] != self.partial_rows[a].1[..]
                {
                    return Err("MLA partial_rows prefix mismatch across tokens sharing chunk_ids".into());
                }
            }
        }
        Ok(recomputed)
    }
}

#[inline(always)]
fn i128_mod_p_m31(v: i128) -> stwo::core::fields::m31::BaseField {
    use stwo::core::fields::m31::{BaseField, P};
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

fn dot_i32_eq(eq_w: &[[i64; 4]], values: &[i32]) -> SecureField {
    let mut a = [0i128; 4];
    for (w, &v) in eq_w.iter().zip(values) {
        let v = i128::from(v);
        a[0] += i128::from(w[0]) * v;
        a[1] += i128::from(w[1]) * v;
        a[2] += i128::from(w[2]) * v;
        a[3] += i128::from(w[3]) * v;
    }
    SecureField::from_m31_array([
        i128_mod_p_m31(a[0]),
        i128_mod_p_m31(a[1]),
        i128_mod_p_m31(a[2]),
        i128_mod_p_m31(a[3]),
    ])
}

fn combine_horner(values: &[SecureField], lambda: SecureField) -> SecureField {
    values
        .iter()
        .rev()
        .fold(SecureField::from(i8_to_m31(0)), |acc, &v| acc * lambda + v)
}

/// Verifies a shared-input projection batch where each matrix `m` has its own exact column width
/// (`outputs[m][0].len().next_power_of_two()`), avoiding zero-padding narrow matrices (`W_KR`, `W_QR`)
/// to the widest output.
pub fn verify_hetero_col_projections(
    inputs: &[Vec<i8>],
    outputs: &[&[Vec<i32>]],
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<Vec<(Vec<SecureField>, SecureField)>, String> {
    let tokens = inputs.len();
    let in_dim = inputs.first().map_or(0, Vec::len);
    if !tokens.is_power_of_two() || in_dim < 2 || outputs.is_empty() {
        return Err("hetero projection requires power-of-two batch and in_dim >= 2".into());
    }
    if inputs.iter().any(|r| r.len() != in_dim) {
        return Err("hetero projection input row length mismatch".into());
    }
    let log_tokens = tokens.ilog2() as usize;
    let log_in = in_dim.next_power_of_two().ilog2() as usize;
    if proof.proof.round_polys.len() != log_in || proof.weight_openings.len() != outputs.len() {
        return Err("hetero projection proof shape mismatch".into());
    }
    let r_tokens = channel.draw_secure_felts(log_tokens);
    let tok_w = eq_evals(&r_tokens);
    let mut r_outs = Vec::with_capacity(outputs.len());
    let mut claims = Vec::with_capacity(outputs.len());
    for out in outputs {
        let cols = out.first().map_or(0, Vec::len);
        if out.len() != tokens || cols < 2 || out.iter().any(|r| r.len() != cols) {
            return Err("hetero projection output shape mismatch".into());
        }
        let log_out = cols.next_power_of_two().ilog2() as usize;
        let r_out = channel.draw_secure_felts(log_out);
        let out_w = eq_evals_i64(&r_out);
        let claim: SecureField = out
            .iter()
            .zip(&tok_w)
            .map(|(row, &tw)| tw * dot_i32_eq(&out_w, row))
            .sum();
        r_outs.push(r_out);
        claims.push(claim);
    }
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s_in, final_eval) = partially_verify(combine_horner(&claims, lambda), &proof.proof, channel)?;
    if s_in.len() != log_in {
        return Err("hetero projection assignment length mismatch".into());
    }
    let mut x_rest = restrict_rows_i8_fast(inputs, &r_tokens);
    x_rest.resize(in_dim.next_power_of_two(), SecureField::from(i8_to_m31(0)));
    let x_opening = mle_at(&x_rest, &s_in);
    if final_eval != x_opening * combine_horner(&proof.weight_openings, lambda) {
        return Err("hetero projection final evaluation check failed".into());
    }
    channel.mix_felts(&proof.weight_openings);
    Ok(r_outs
        .into_iter()
        .zip(&proof.weight_openings)
        .map(|(r_out, &op)| {
            let mut pt = s_in.clone();
            pt.extend_from_slice(&r_out);
            (pt, op)
        })
        .collect())
}

/// Verifies the head-batched latent-to-value contraction `v_raw[t, h, i] = sum_{k < d_c} u_q8[t, h, k] * W_UV[h, k, i]`
/// in `log2(H * d_c)` rounds, returning ONE opening on `W_UV` at `(s_hk, r_v_dim)`.
pub fn verify_mla_head_batched_uv(
    shape: MlaShape,
    u_q8: &[Vec<i8>],
    v_raw: &[Vec<i32>],
    proof: &ProjectionProof,
    channel: &mut Blake2sChannel,
) -> Result<(Vec<SecureField>, SecureField), String> {
    let tokens = u_q8.len();
    if !tokens.is_power_of_two()
        || v_raw.len() != tokens
        || u_q8.iter().any(|r| r.len() != shape.q_abs_dim())
        || v_raw.iter().any(|r| r.len() != shape.v_dim())
    {
        return Err("MLA head-batched W_UV input/output shape mismatch".into());
    }
    let log_tokens = tokens.ilog2() as usize;
    let log_h = shape.heads.ilog2() as usize;
    let log_dc = shape.kv_lora_rank.ilog2() as usize;
    let log_hk = log_h + log_dc;
    let log_vh = shape.v_head_dim.ilog2() as usize;
    if proof.proof.round_polys.len() != log_hk || proof.weight_openings.len() != 1 {
        return Err("MLA head-batched W_UV proof shape mismatch".into());
    }
    let r_tokens = channel.draw_secure_felts(log_tokens);
    let r_heads = channel.draw_secure_felts(log_h);
    let r_vh = channel.draw_secure_felts(log_vh);
    let tok_w = eq_evals(&r_tokens);
    let mut hv_pt = r_heads.clone();
    hv_pt.extend_from_slice(&r_vh);
    let hv_w = eq_evals_i64(&hv_pt);
    let claim: SecureField = v_raw
        .iter()
        .zip(&tok_w)
        .map(|(row, &tw)| tw * dot_i32_eq(&hv_w, row))
        .sum();
    channel.mix_felts(&[claim]);
    let (s_hk, final_eval) = partially_verify(claim, &proof.proof, channel)?;
    if s_hk.len() != log_hk {
        return Err("MLA head-batched W_UV assignment length mismatch".into());
    }
    let head_w = eq_evals(&r_heads);
    let mut u_left = restrict_rows_i8_fast(u_q8, &r_tokens);
    for h in 0..shape.heads {
        let hw = head_w[h];
        for val in &mut u_left[h * shape.kv_lora_rank..(h + 1) * shape.kv_lora_rank] {
            *val *= hw;
        }
    }
    let u_eval = mle_at(&u_left, &s_hk);
    let w_uv_op = proof.weight_openings[0];
    if final_eval != u_eval * w_uv_op {
        return Err("MLA head-batched W_UV final evaluation check failed".into());
    }
    channel.mix_felts(&[w_uv_op]);
    let mut pt = s_hk;
    pt.extend_from_slice(&r_vh);
    Ok((pt, w_uv_op))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaOpenedChunkSubCommitments {
    pub chunk: usize,
    pub com_c_kv: Hash,
    pub com_k_pe: Hash,
}

#[derive(Clone, Debug)]
pub struct StageBMlaProof {
    pub f_override: Option<usize>,
    pub down_proj: ProjectionProof,
    pub up_q_proj: ProjectionProof,
    pub scores_nope: ContractionProof,
    pub scores_rope: ContractionProof,
    pub latent_ctx: ContractionProof,
    pub uv_proj: ProjectionProof,
    pub o_proj: ProjectionProof,
    pub leaf_sub_commitments: [Hash; 7],
    pub opened_chunks: Vec<MlaOpenedChunkSubCommitments>,
    pub unified_cprs_proof: Vec<u8>,
}

pub fn mla_outer_digest(statement_digest: &[u8; 32]) -> Hash {
    let mut h = Blake2s256::new();
    h.update(MLA_OUTER_DOMAIN);
    h.update(statement_digest);
    h.finalize().into()
}

#[derive(Clone, Debug)]
struct MlaCacheTripleOpening {
    chunk: usize,
    c_nope_point: Vec<SecureField>,
    c_nope_val: SecureField,
    c_ctx_point: Vec<SecureField>,
    c_ctx_val: SecureField,
    k_rope_point: Vec<SecureField>,
    k_rope_val: SecureField,
}

fn collect_mla_chunk_openings(
    statement: &StageBMlaStatement,
    nope_claims: &[CacheOpeningClaim],
    ctx_claims: &[CacheOpeningClaim],
    rope_claims: &[CacheOpeningClaim],
) -> Result<Vec<MlaCacheTripleOpening>, String> {
    let mut map: BTreeMap<usize, MlaCacheTripleOpening> = BTreeMap::new();
    for (t, tok) in statement.tokens.iter().enumerate() {
        let nc = &nope_claims[t];
        let cc = &ctx_claims[t];
        let rc = &rope_claims[t];
        let c_nope_pt = nc.point[nc.log_chunks..].to_vec();
        let c_ctx_pt = cc.point[cc.log_chunks..].to_vec();
        let k_rope_pt = rc.point[rc.log_chunks..].to_vec();
        for (pos, &chunk) in tok.chunk_ids.iter().enumerate() {
            let entry = MlaCacheTripleOpening {
                chunk,
                c_nope_point: c_nope_pt.clone(),
                c_nope_val: nc.full_chunk_openings[pos],
                c_ctx_point: c_ctx_pt.clone(),
                c_ctx_val: cc.full_chunk_openings[pos],
                k_rope_point: k_rope_pt.clone(),
                k_rope_val: rc.full_chunk_openings[pos],
            };
            match map.get(&chunk) {
                None => {
                    map.insert(chunk, entry);
                }
                Some(prev) => {
                    if prev.c_nope_point != entry.c_nope_point
                        || prev.c_nope_val != entry.c_nope_val
                        || prev.c_ctx_point != entry.c_ctx_point
                        || prev.c_ctx_val != entry.c_ctx_val
                        || prev.k_rope_point != entry.k_rope_point
                        || prev.k_rope_val != entry.k_rope_val
                    {
                        return Err("MLA tokens disagree on shared latent cache chunk opening".into());
                    }
                }
            }
        }
    }
    Ok(map.into_values().collect())
}

pub fn verify_stage_b_mla_against_anchor(
    expected_anchor: &[u8; 32],
    statement: &StageBMlaStatement,
    profile: &SecurityProfile,
    proof: &StageBMlaProof,
) -> Result<Vec<MlaRecomputedStep>, String> {
    if &statement.registered_anchor() != expected_anchor {
        return Err("MLA statement does not match expected registered layer anchor".into());
    }
    verify_stage_b_mla(statement, profile, proof)
}

pub fn verify_stage_b_mla(
    statement: &StageBMlaStatement,
    profile: &SecurityProfile,
    proof: &StageBMlaProof,
) -> Result<Vec<MlaRecomputedStep>, String> {
    if proof.f_override != statement.cprs_f {
        return Err("MLA proof f_override does not match statement.cprs_f".into());
    }
    let recomputed = statement.validate_and_recompute(statement.cprs_f, profile)?;
    if compute_mla_leaf_commitment(&proof.leaf_sub_commitments) != statement.mla_leaf_commitment {
        return Err("MLA leaf sub-commitments do not hash to mla_leaf_commitment".into());
    }
    let stmt_digest = statement.digest();
    let mut channel = StageBStatement::channel_from_digest(&stmt_digest);

    let x1_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.x1.clone()).collect();
    let cq_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.c_q_raw.clone()).collect();
    let ckv_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.c_kv_raw.clone()).collect();
    let kr_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.k_pe_raw.clone()).collect();
    let down_claims = verify_hetero_col_projections(
        &x1_in,
        &[&cq_out, &ckv_out, &kr_out],
        &proof.down_proj,
        &mut channel,
    )?;

    let cq_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.c_q.clone()).collect();
    let qabs_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.q_abs_raw.clone()).collect();
    let qpe_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.q_pe_raw.clone()).collect();
    let up_q_claims = verify_hetero_col_projections(
        &cq_in,
        &[&qabs_out, &qpe_out],
        &proof.up_q_proj,
        &mut channel,
    )?;

    let cache_lengths: Vec<usize> = statement
        .tokens
        .iter()
        .map(|t| t.step.position as usize + 1)
        .collect();
    let q_abs_refs: Vec<&[i8]> = recomputed.iter().map(|r| r.q_abs.as_slice()).collect();
    let s_nope_refs: Vec<&[Vec<i32>]> = statement.tokens.iter().map(|t| t.step.s_nope.as_slice()).collect();
    verify_scores(
        statement.shape.nope_attention_shape(),
        statement.layout,
        &q_abs_refs,
        &cache_lengths,
        &s_nope_refs,
        &proof.scores_nope,
        &mut channel,
    )?;

    let q_pe_refs: Vec<&[i8]> = recomputed.iter().map(|r| r.q_pe.as_slice()).collect();
    let s_rope_refs: Vec<&[Vec<i32>]> = statement.tokens.iter().map(|t| t.step.s_rope.as_slice()).collect();
    verify_scores(
        statement.shape.rope_attention_shape(),
        statement.layout,
        &q_pe_refs,
        &cache_lengths,
        &s_rope_refs,
        &proof.scores_rope,
        &mut channel,
    )?;

    let w_refs: Vec<&[Vec<u16>]> = recomputed.iter().map(|r| r.weights.as_slice()).collect();
    let u_raw_refs: Vec<&[i32]> = statement.tokens.iter().map(|t| t.step.u_raw.as_slice()).collect();
    verify_context(
        statement.shape.nope_attention_shape(),
        statement.layout,
        &w_refs,
        &cache_lengths,
        &u_raw_refs,
        &proof.latent_ctx,
        &mut channel,
    )?;

    let u_q8_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.u_q8.clone()).collect();
    let v_raw_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.v_raw.clone()).collect();
    let (uv_pt, uv_op) = verify_mla_head_batched_uv(
        statement.shape,
        &u_q8_in,
        &v_raw_out,
        &proof.uv_proj,
        &mut channel,
    )?;

    let v_q8_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.v_q8.clone()).collect();
    let o_raw_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.o_raw.clone()).collect();
    let o_claims = verify_projections(&v_q8_in, &[&o_raw_out], &proof.o_proj, &mut channel)?;

    let padded_all = cache_lengths
        .iter()
        .map(|&c| statement.layout.padded_positions(c))
        .max()
        .unwrap();
    let cids_refs: Vec<&[usize]> = statement.tokens.iter().map(|t| t.chunk_ids.as_slice()).collect();
    let ckv_partial_refs: Vec<&[Vec<i8>]> =
        statement.partial_rows.iter().map(|(c, _)| c.as_slice()).collect();
    let kpe_partial_refs: Vec<&[Vec<i8>]> =
        statement.partial_rows.iter().map(|(_, k)| k.as_slice()).collect();
    verify_cache_claims_batch(
        &cids_refs,
        &ckv_partial_refs,
        statement.layout,
        padded_all,
        &proof.scores_nope.claims,
    )?;
    verify_cache_claims_batch(
        &cids_refs,
        &ckv_partial_refs,
        statement.layout,
        padded_all,
        &proof.latent_ctx.claims,
    )?;
    verify_cache_claims_batch(
        &cids_refs,
        &kpe_partial_refs,
        statement.layout,
        padded_all,
        &proof.scores_rope.claims,
    )?;

    let chunk_triples = collect_mla_chunk_openings(
        statement,
        &proof.scores_nope.claims,
        &proof.latent_ctx.claims,
        &proof.scores_rope.claims,
    )?;
    if proof.opened_chunks.len() != chunk_triples.len() {
        return Err("MLA opened_chunks count mismatch".into());
    }
    let leaf_params = mla_leaf_group_params(statement.shape, proof.f_override, profile)?;
    let (p_ckv, p_kpe) =
        mla_cache_chunk_group_params(statement.layout, statement.shape, proof.f_override, profile)?;

    let mut group_coms: Vec<Hash> = proof.leaf_sub_commitments.to_vec();
    let mut group_params: Vec<Params> = leaf_params.to_vec();
    let mut points: Vec<Vec<QM31>> = vec![
        stwo_point_to_cprs(&down_claims[0].0),
        stwo_point_to_cprs(&down_claims[1].0),
        stwo_point_to_cprs(&down_claims[2].0),
        stwo_point_to_cprs(&up_q_claims[0].0),
        stwo_point_to_cprs(&up_q_claims[1].0),
        stwo_point_to_cprs(&uv_pt),
        stwo_point_to_cprs(&o_claims.point),
    ];
    let mut claims_cprs: Vec<QM31> = vec![
        to_cprs_qm31(down_claims[0].1),
        to_cprs_qm31(down_claims[1].1),
        to_cprs_qm31(down_claims[2].1),
        to_cprs_qm31(up_q_claims[0].1),
        to_cprs_qm31(up_q_claims[1].1),
        to_cprs_qm31(uv_op),
        to_cprs_qm31(o_claims.openings[0]),
    ];

    for (triple, sub) in chunk_triples.iter().zip(&proof.opened_chunks) {
        if sub.chunk != triple.chunk {
            return Err("MLA opened_chunks ordering mismatch".into());
        }
        let expected_com = statement
            .chunk_commitments
            .get(triple.chunk)
            .ok_or_else(|| "MLA chunk index out of range".to_string())?;
        if compute_mla_chunk_commitment(&sub.com_c_kv, &sub.com_k_pe) != *expected_com {
            return Err("MLA chunk sub-commitments do not match chunk_commitments".into());
        }
        group_coms.push(sub.com_c_kv);
        group_params.push(p_ckv.clone());
        points.push(stwo_point_to_cprs(&triple.c_nope_point));
        claims_cprs.push(to_cprs_qm31(triple.c_nope_val));

        group_coms.push(sub.com_c_kv);
        group_params.push(p_ckv.clone());
        points.push(stwo_point_to_cprs(&triple.c_ctx_point));
        claims_cprs.push(to_cprs_qm31(triple.c_ctx_val));

        group_coms.push(sub.com_k_pe);
        group_params.push(p_kpe.clone());
        points.push(stwo_point_to_cprs(&triple.k_rope_point));
        claims_cprs.push(to_cprs_qm31(triple.k_rope_val));
    }

    let outer = mla_outer_digest(&stmt_digest);
    verify_heterogeneous_strict_context(
        &group_coms,
        &group_params,
        &points,
        &claims_cprs,
        &outer,
        &proof.unified_cprs_proof,
    )
    .map_err(|e| format!("StageBMla unified CPRS opening rejected: {e}"))?;

    Ok(recomputed)
}

fn read_hash_32(reader: &mut ReceiptReader<'_>, label: &str) -> Result<Hash, String> {
    let mut out = [0u8; 32];
    for b in &mut out {
        *b = reader.read_u32_byte(label)?;
    }
    Ok(out)
}

fn put_mla_contraction_proof(out: &mut Vec<u8>, proof: &ContractionProof) -> Result<(), String> {
    put_sumcheck(out, &proof.proof)?;
    put_u64(out, proof.claims.len())?;
    for c in &proof.claims {
        put_secure_fields(out, &c.point)?;
        put_u64(out, c.log_chunks)?;
        put_secure_fields(out, &c.full_chunk_openings)?;
        put_secure_field(out, c.value);
    }
    Ok(())
}

fn read_mla_contraction_proof(
    reader: &mut ReceiptReader<'_>,
    expected_tokens: usize,
    expected_point_len: usize,
    expected_log_chunks: usize,
    chunk_lens: &[usize],
    label: &str,
) -> Result<ContractionProof, String> {
    let proof = reader.read_sumcheck(&format!("{label} sumcheck"))?;
    let count = reader.read_len(expected_tokens, &format!("{label} claim count"))?;
    if count != expected_tokens {
        return Err(format!("{label} claim count mismatch"));
    }
    let mut claims = Vec::with_capacity(count);
    for (t, &fco_len) in chunk_lens.iter().enumerate() {
        let point = reader.read_secure_fields_exact(expected_point_len, &format!("{label} pt {t}"))?;
        let log_chunks = reader.read_len(expected_log_chunks, &format!("{label} lc {t}"))?;
        if log_chunks != expected_log_chunks {
            return Err(format!("{label} log_chunks mismatch"));
        }
        let full_chunk_openings =
            reader.read_secure_fields_exact(fco_len, &format!("{label} fco {t}"))?;
        let value = reader.read_secure_field(&format!("{label} val {t}"))?;
        claims.push(CacheOpeningClaim {
            point,
            log_chunks,
            full_chunk_openings,
            value,
        });
    }
    Ok(ContractionProof { proof, claims })
}

pub fn stage_b_mla_receipt_bytes(proof: &StageBMlaProof) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(STAGE_B_MLA_RECEIPT_MAGIC);
    put_u32(&mut out, STAGE_B_MLA_RECEIPT_VERSION);
    put_u32(&mut out, proof.f_override.unwrap_or(0) as u32);
    put_projection_proof(&mut out, &proof.down_proj)?;
    put_projection_proof(&mut out, &proof.up_q_proj)?;
    put_mla_contraction_proof(&mut out, &proof.scores_nope)?;
    put_mla_contraction_proof(&mut out, &proof.scores_rope)?;
    put_mla_contraction_proof(&mut out, &proof.latent_ctx)?;
    put_projection_proof(&mut out, &proof.uv_proj)?;
    put_projection_proof(&mut out, &proof.o_proj)?;
    for c in &proof.leaf_sub_commitments {
        out.extend_from_slice(c);
    }
    put_u64(&mut out, proof.opened_chunks.len())?;
    for oc in &proof.opened_chunks {
        put_u64(&mut out, oc.chunk)?;
        out.extend_from_slice(&oc.com_c_kv);
        out.extend_from_slice(&oc.com_k_pe);
    }
    put_bytes(&mut out, &proof.unified_cprs_proof)?;
    Ok(out)
}

pub fn decode_stage_b_mla_receipt_bytes(
    bytes: &[u8],
    statement: &StageBMlaStatement,
) -> Result<StageBMlaProof, String> {
    if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOMLA1 receipt exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(STAGE_B_MLA_RECEIPT_MAGIC, "STWOMLA1")?;
    let ver = reader.read_u32("STWOMLA1 version")?;
    if ver != STAGE_B_MLA_RECEIPT_VERSION {
        return Err(format!("unsupported STWOMLA1 version {ver}"));
    }
    let f_raw = reader.read_u32("STWOMLA1 f_override")? as usize;
    let f_override = if f_raw == 0 { None } else { Some(f_raw) };
    if f_override != statement.cprs_f {
        return Err("STWOMLA1 wire f_override does not match statement.cprs_f".into());
    }
    let tokens = statement.tokens.len();
    let padded_all = statement
        .tokens
        .iter()
        .map(|t| statement.layout.padded_positions(t.step.position as usize + 1))
        .max()
        .unwrap_or(2 * statement.layout.chunk);
    let log_rows = padded_all.ilog2() as usize;
    let log_chunks = log_rows - statement.layout.chunk.ilog2() as usize;
    let chunk_lens: Vec<usize> = statement.tokens.iter().map(|t| t.chunk_ids.len()).collect();
    let log_ckv = statement.shape.kv_lora_rank.ilog2() as usize;
    let log_kr = statement.shape.qk_rope_head_dim.ilog2() as usize;

    let down_proj = read_projection_proof(&mut reader, 3, "STWOMLA1 down_proj")?;
    let up_q_proj = read_projection_proof(&mut reader, 2, "STWOMLA1 up_q_proj")?;
    let scores_nope = read_mla_contraction_proof(
        &mut reader,
        tokens,
        log_rows + log_ckv,
        log_chunks,
        &chunk_lens,
        "STWOMLA1 scores_nope",
    )?;
    let scores_rope = read_mla_contraction_proof(
        &mut reader,
        tokens,
        log_rows + log_kr,
        log_chunks,
        &chunk_lens,
        "STWOMLA1 scores_rope",
    )?;
    let latent_ctx = read_mla_contraction_proof(
        &mut reader,
        tokens,
        log_rows + log_ckv,
        log_chunks,
        &chunk_lens,
        "STWOMLA1 latent_ctx",
    )?;
    let uv_proj = read_projection_proof(&mut reader, 1, "STWOMLA1 uv_proj")?;
    let o_proj = read_projection_proof(&mut reader, 1, "STWOMLA1 o_proj")?;
    let mut leaf_sub_commitments = [[0u8; 32]; 7];
    for slot in &mut leaf_sub_commitments {
        *slot = read_hash_32(&mut reader, "STWOMLA1 leaf sub-commitment")?;
    }
    let oc_count = reader.read_len(statement.chunk_commitments.len(), "STWOMLA1 opened_chunks")?;
    let mut opened_chunks = Vec::with_capacity(oc_count);
    for _ in 0..oc_count {
        let chunk = reader.read_len(
            statement.chunk_commitments.len().saturating_sub(1),
            "STWOMLA1 chunk id",
        )?;
        let com_c_kv = read_hash_32(&mut reader, "STWOMLA1 com_c_kv")?;
        let com_k_pe = read_hash_32(&mut reader, "STWOMLA1 com_k_pe")?;
        opened_chunks.push(MlaOpenedChunkSubCommitments {
            chunk,
            com_c_kv,
            com_k_pe,
        });
    }
    let unified_cprs_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "STWOMLA1 cprs proof")?
        .to_vec();
    let _ = SumcheckProof { round_polys: Vec::new() };
    reader.finish()?;
    Ok(StageBMlaProof {
        f_override,
        down_proj,
        up_q_proj,
        scores_nope,
        scores_rope,
        latent_ctx,
        uv_proj,
        o_proj,
        leaf_sub_commitments,
        opened_chunks,
        unified_cprs_proof,
    })
}
