//! Standalone `wasm32`-compatible verifier for Alternating Sliding-Window Attention + Learnable
//! Per-Head Attention Sinks (`StageBWindowSinks`, wire magic `STWOSWA1`), covering `GPT-OSS-120B`
//! (`d = 2880`, `H = 64`, `H_kv = 8`, `d_h = 64`, `W = 128` sliding window, per-head `sinks[64]`)
//! with O(W) score wire size and KV-cache chunk pruning at arbitrary sequence lengths.
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
    cache_chunk_params_with_f, exp_table_static, group_params_with_f, put_projection_proof,
    read_projection_proof, rms_norm, stage_b_step_cache_commitment, verify_cache_claims_batch,
    verify_context, verify_projections, AttentionPolicy, AttentionShape, CacheChunkOpening,
    CacheLayout, CacheOpeningClaim, ContractionProof, ProjectionProof, StageBStatement,
    EXP_TABLE_LEN, WEIGHT_BITS,
};
use crate::stage_b_mla::{rope_dim, verify_hetero_col_projections};
use crate::{
    eq_evals, eq_evals_i64, i8_to_m31, partially_verify, put_bytes, put_secure_field,
    put_secure_fields, put_sumcheck, put_u32, put_u64, ReceiptReader, SecurityProfile,
    MAX_WIRE_RECEIPT_BYTES,
};

pub const STAGE_B_WINDOW_SINK_RECEIPT_MAGIC: &[u8; 8] = b"STWOSWA1";
pub const STAGE_B_WINDOW_SINK_RECEIPT_VERSION: u32 = 1;
pub const SWA_STATEMENT_DOMAIN: &[u8] = b"stwo-ml:stage-b-swa:statement:v1";
pub const SWA_REGISTERED_ANCHOR_DOMAIN: &[u8] = b"stwo-ml:stage-b-swa:registered-anchor:v1";
pub const SWA_LEAF_DOMAIN: &[u8] = b"stwo-ml:stage-b-swa:leaf:v1";
pub const SWA_OUTER_DOMAIN: &[u8] = b"stwo-ml:stage-b-swa:cprs-outer:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlidingWindowSpec {
    /// Sliding window length `W` in tokens (`128` on GPT-OSS-120B even layers).
    /// Token at position `p` attends only to `j in max(0, p + 1 - W)..=p`.
    pub window_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionSinkSpec {
    /// Per-head sink logits `sigma_{h,int}` quantised to the integer score scale (`len == heads`).
    pub head_sink_logits_i32: Vec<i32>,
}

/// Supports both standard power-of-two GQA (`d == H * d_h`) and decoupled residual-width GQA
/// such as `openai/gpt-oss-120b` (`d = 2880`, `H = 64`, `H_kv = 8`, `d_h = 64` -> `q_dim = 4096`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GptOssAttentionShape {
    pub dimension: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl GptOssAttentionShape {
    pub const fn q_dim(self) -> usize {
        self.heads * self.head_dim
    }

    pub const fn kv_dim(self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub const fn group_size(self) -> usize {
        self.heads / self.kv_heads
    }

    pub const fn inner_attention_shape(self) -> AttentionShape {
        AttentionShape {
            dimension: self.heads * self.head_dim,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }

    pub fn validate(self) -> Result<(), String> {
        if self.dimension < 4 || self.dimension % 2 != 0 || self.dimension > (1 << 15) {
            return Err("GptOssAttentionShape dimension must be even in 4..=32768".into());
        }
        if self.heads == 0 || !self.heads.is_power_of_two() || self.heads > 256 {
            return Err("GptOssAttentionShape heads must be a power of two in 1..=256".into());
        }
        if self.kv_heads == 0
            || !self.kv_heads.is_power_of_two()
            || self.kv_heads > self.heads
            || self.heads % self.kv_heads != 0
        {
            return Err("GptOssAttentionShape kv_heads must be a power of two dividing heads".into());
        }
        if self.head_dim < 4 || self.head_dim % 2 != 0 || !self.head_dim.is_power_of_two() {
            return Err("GptOssAttentionShape head_dim must be an even power of two >= 4".into());
        }
        Ok(())
    }

    pub const fn gpt_oss_120b() -> Self {
        Self {
            dimension: 2880,
            heads: 64,
            kv_heads: 8,
            head_dim: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowSinkAttentionPolicy {
    pub base: AttentionPolicy,
    pub sliding_window: Option<SlidingWindowSpec>,
    pub attention_sinks: Option<AttentionSinkSpec>,
}

impl WindowSinkAttentionPolicy {
    pub fn gpt_oss_layer(layer_idx: usize, head_sink_logits_i32: Vec<i32>) -> Self {
        Self {
            base: AttentionPolicy::trained_v1(),
            sliding_window: if layer_idx % 2 == 0 {
                Some(SlidingWindowSpec { window_size: 128 })
            } else {
                None
            },
            attention_sinks: Some(AttentionSinkSpec {
                head_sink_logits_i32,
            }),
        }
    }

    pub fn validate(&self, shape: GptOssAttentionShape) -> Result<(), String> {
        if let Some(w) = self.sliding_window {
            if w.window_size == 0 {
                return Err("SlidingWindowSpec window_size must be positive".into());
            }
        }
        if let Some(ref s) = self.attention_sinks {
            if s.head_sink_logits_i32.len() != shape.heads {
                return Err("AttentionSinkSpec head_sink_logits_i32 length must equal heads".into());
            }
        }
        Ok(())
    }

    pub fn window_start(&self, position: u64) -> usize {
        let end = position as usize + 1;
        match self.sliding_window {
            Some(w) => end.saturating_sub(w.window_size),
            None => 0,
        }
    }

    pub fn update_digest(&self, h: &mut Blake2s256) {
        let id = self.base.id();
        h.update((id.len() as u64).to_le_bytes());
        h.update(id.as_bytes());
        match self.sliding_window {
            Some(w) => {
                h.update([1u8]);
                h.update((w.window_size as u64).to_le_bytes());
            }
            None => h.update([0u8]),
        }
        match self.attention_sinks {
            Some(ref s) => {
                h.update([1u8]);
                h.update((s.head_sink_logits_i32.len() as u64).to_le_bytes());
                for &v in &s.head_sink_logits_i32 {
                    h.update(v.to_le_bytes());
                }
            }
            None => h.update([0u8]),
        }
    }
}

/// Fixed-point softmax over active window scores `scores_active` with optional virtual attention sink
/// `sink_logit` (`v_sink = 0`, so `sum_j w_j <= 2^WEIGHT_BITS`, preserving M31 non-overflow).
pub fn softmax_weights_with_sink(
    scores_active: &[i32],
    score_shift: u32,
    sink_logit: Option<i32>,
) -> Result<Vec<u16>, String> {
    let max_real = *scores_active
        .iter()
        .max()
        .ok_or_else(|| "softmax over an empty window".to_string())?;
    let max_all = match sink_logit {
        Some(sigma) => max_real.max(sigma),
        None => max_real,
    };
    let table = exp_table_static();
    let lookup = |s: i32| -> u32 {
        let idx = ((i64::from(max_all) - i64::from(s)) >> score_shift)
            .clamp(0, EXP_TABLE_LEN as i64 - 1) as usize;
        u32::from(table[idx])
    };
    let e: Vec<u32> = scores_active.iter().map(|&s| lookup(s)).collect();
    let e_sink: u64 = sink_logit.map_or(0, |sigma| u64::from(lookup(sigma)));
    let z: u64 = e_sink + e.iter().map(|&v| u64::from(v)).sum::<u64>();
    if z == 0 {
        return Err("softmax partition function Z is zero".into());
    }
    Ok(e.iter()
        .map(|&v| ((u64::from(v) << WEIGHT_BITS) / z) as u16)
        .collect())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedPublicStep {
    pub position: u64,
    pub h_in: Vec<i16>,
    pub q_raw: Vec<i32>,
    pub k_raw: Vec<i32>,
    pub v_raw: Vec<i32>,
    /// Active-window scores ONLY (`scores[h].len() == (position + 1) - window_start(position)`).
    pub scores: Vec<Vec<i32>>,
    pub ctx_raw: Vec<i32>,
    pub o_raw: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedRecomputedStep {
    pub x1: Vec<i8>,
    pub q_rot: Vec<i8>,
    pub k_rot: Vec<i8>,
    pub v_q: Vec<i8>,
    /// Active-window softmax weights (`weights[h].len() == scores[h].len()`).
    pub weights: Vec<Vec<u16>>,
    pub ctx: Vec<i8>,
    pub a: Vec<i16>,
    pub x: Vec<i16>,
}

fn requant_qkv_i8(values: &[i32], policy: AttentionPolicy) -> Result<Vec<i8>, String> {
    values
        .iter()
        .map(|&v| {
            let r = requant_with_rounding(i128::from(v), policy.qkv, policy.rounding)?.value;
            i8::try_from(r).map_err(|_| "SWA qkv requant exceeds i8".to_string())
        })
        .collect()
}

fn swa_ctx_from_raw(ctx_raw: &[i32]) -> Result<Vec<i8>, String> {
    let spec = RequantSpec {
        multiplier: 1,
        shift: WEIGHT_BITS,
        output_bits: 8,
    };
    ctx_raw
        .iter()
        .map(|&v| {
            let r =
                requant_with_rounding(i128::from(v), spec, StageARoundingMode::TiesToEven)?.value;
            i8::try_from(r).map_err(|_| "SWA ctx exceeds i8".to_string())
        })
        .collect()
}

fn swa_residual_add(
    h_in: &[i16],
    o_raw: &[i32],
    policy: AttentionPolicy,
) -> Result<Vec<i16>, String> {
    h_in.iter()
        .zip(o_raw)
        .map(|(&h, &o)| {
            let dq = requant_with_rounding(i128::from(o), policy.output, policy.rounding)?.value;
            i16::try_from(i32::from(h) + dq).map_err(|_| "SWA residual exceeds i16".to_string())
        })
        .collect()
}

pub fn check_windowed_public_step(
    shape: GptOssAttentionShape,
    policy: &WindowSinkAttentionPolicy,
    gamma1: &[i16],
    gamma2: &[i16],
    step: &WindowedPublicStep,
) -> Result<WindowedRecomputedStep, String> {
    shape.validate()?;
    policy.validate(shape)?;
    if step.h_in.len() != shape.dimension
        || step.o_raw.len() != shape.dimension
        || step.q_raw.len() != shape.q_dim()
        || step.ctx_raw.len() != shape.q_dim()
        || step.k_raw.len() != shape.kv_dim()
        || step.v_raw.len() != shape.kv_dim()
    {
        return Err("windowed attention step vector length mismatch".into());
    }
    let win_start = policy.window_start(step.position);
    let win_len = (step.position as usize + 1) - win_start;
    if step.scores.len() != shape.heads || step.scores.iter().any(|s| s.len() != win_len) {
        return Err("windowed attention scores must be heads x active_window_len".into());
    }
    let x1: Vec<i8> = rms_norm(&step.h_in, gamma1, policy.base.norm1_fraction_bits, 8)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let q_rot = rope_dim(
        &requant_qkv_i8(&step.q_raw, policy.base)?,
        shape.head_dim,
        step.position,
    )?;
    let k_rot = rope_dim(
        &requant_qkv_i8(&step.k_raw, policy.base)?,
        shape.head_dim,
        step.position,
    )?;
    let v_q = requant_qkv_i8(&step.v_raw, policy.base)?;
    let mut weights = Vec::with_capacity(shape.heads);
    for (h, s) in step.scores.iter().enumerate() {
        let sink = policy
            .attention_sinks
            .as_ref()
            .map(|sp| sp.head_sink_logits_i32[h]);
        weights.push(softmax_weights_with_sink(s, policy.base.score_shift, sink)?);
    }
    let ctx = swa_ctx_from_raw(&step.ctx_raw)?;
    let a = swa_residual_add(&step.h_in, &step.o_raw, policy.base)?;
    let x: Vec<i16> = rms_norm(&a, gamma2, policy.base.norm2_fraction_bits, 12)?
        .into_iter()
        .map(|v| v as i16)
        .collect();
    Ok(WindowedRecomputedStep {
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

pub fn compute_swa_leaf_commitment(sub_commitments: &[Hash; 3]) -> Hash {
    let mut h = Blake2s256::new();
    h.update(SWA_LEAF_DOMAIN);
    for c in sub_commitments {
        h.update(c);
    }
    h.finalize().into()
}

/// Group 0: `W_Q` (`g = 1`, `d_pad x q_dim`), Group 1: `(W_K, W_V)` (`g = 2`, `d_pad x kv_dim`),
/// Group 2: `W_O` (`g = 1`, `q_dim x d_pad`).
pub fn swa_leaf_group_params(
    shape: GptOssAttentionShape,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<[Params; 3], String> {
    shape.validate()?;
    let log_d = shape.dimension.next_power_of_two().ilog2() as usize;
    let log_q = shape.q_dim().ilog2() as usize;
    let log_kv = shape.kv_dim().ilog2() as usize;
    Ok([
        group_params_with_f(log_d + log_q, 1, f_override, profile)?,
        group_params_with_f(log_d + log_kv, 2, f_override, profile)?,
        group_params_with_f(log_q + log_d, 1, f_override, profile)?,
    ])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedToken {
    pub step: WindowedPublicStep,
    pub chunk_ids: Vec<usize>,
    pub cache_commitment_before: [u8; 32],
    pub cache_commitment_after: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageBWindowSinkStatement {
    pub shape: GptOssAttentionShape,
    pub policy: WindowSinkAttentionPolicy,
    pub layout: CacheLayout,
    pub cprs_f: Option<usize>,
    pub leaf_commitment: Hash,
    pub gamma1: Vec<i16>,
    pub gamma2: Vec<i16>,
    pub chunk_commitments: Vec<Hash>,
    pub partial_rows: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)>,
    pub tokens: Vec<WindowedToken>,
    pub binding: [u8; 32],
}

/// Computes the immutable model-registration anchor for a Sliding-Window + Attention-Sink layer
/// (`Mythos-M4`), pinning `(shape, policy, layout, cprs_f, leaf_commitment, gamma1, gamma2)`.
pub fn compute_swa_registered_anchor(
    shape: GptOssAttentionShape,
    policy: &WindowSinkAttentionPolicy,
    layout: CacheLayout,
    cprs_f: Option<usize>,
    leaf_commitment: &Hash,
    gamma1: &[i16],
    gamma2: &[i16],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(SWA_REGISTERED_ANCHOR_DOMAIN);
    for v in [
        shape.dimension,
        shape.heads,
        shape.kv_heads,
        shape.head_dim,
        layout.chunk,
        cprs_f.unwrap_or(0),
    ] {
        h.update((v as u64).to_le_bytes());
    }
    policy.update_digest(&mut h);
    h.update(leaf_commitment);
    for g in [gamma1, gamma2] {
        h.update((g.len() as u64).to_le_bytes());
        for &x in g {
            h.update(x.to_le_bytes());
        }
    }
    h.finalize().into()
}

impl StageBWindowSinkStatement {
    pub fn registered_anchor(&self) -> [u8; 32] {
        compute_swa_registered_anchor(
            self.shape,
            &self.policy,
            self.layout,
            self.cprs_f,
            &self.leaf_commitment,
            &self.gamma1,
            &self.gamma2,
        )
    }

    /// Earliest full KV-cache chunk overlapped by any token's active window in the batch.
    /// Chunks `< first_active_chunk` have identically zero attention weight and zero score claims,
    /// so they are pruned from contraction sumchecks and never opened in CPRS.
    pub fn first_active_chunk(&self) -> usize {
        self.tokens
            .iter()
            .map(|t| self.policy.window_start(t.step.position) / self.layout.chunk)
            .min()
            .unwrap_or(0)
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(SWA_STATEMENT_DOMAIN);
        for v in [
            self.shape.dimension,
            self.shape.heads,
            self.shape.kv_heads,
            self.shape.head_dim,
            self.layout.chunk,
            self.cprs_f.unwrap_or(0),
        ] {
            h.update((v as u64).to_le_bytes());
        }
        self.policy.update_digest(&mut h);
        h.update(self.leaf_commitment);
        h.update(self.binding);
        for g in [&self.gamma1, &self.gamma2] {
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
        for (kr, vr) in &self.partial_rows {
            for mat in [kr, vr] {
                h.update((mat.len() as u64).to_le_bytes());
                for r in mat {
                    h.update((r.len() as u64).to_le_bytes());
                    for &b in r {
                        h.update([b as u8]);
                    }
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
                &tok.step.q_raw,
                &tok.step.k_raw,
                &tok.step.v_raw,
                &tok.step.ctx_raw,
                &tok.step.o_raw,
            ] {
                h.update((vec_i32.len() as u64).to_le_bytes());
                for &v in vec_i32 {
                    h.update(v.to_le_bytes());
                }
            }
            h.update((tok.step.scores.len() as u64).to_le_bytes());
            for s_row in &tok.step.scores {
                h.update((s_row.len() as u64).to_le_bytes());
                for &v in s_row {
                    h.update(v.to_le_bytes());
                }
            }
        }
        h.finalize().into()
    }
}

impl StageBWindowSinkStatement {
    pub fn validate_and_recompute(
        &self,
        f_override: Option<usize>,
        profile: &SecurityProfile,
    ) -> Result<Vec<WindowedRecomputedStep>, String> {
        self.shape.validate()?;
        self.policy.validate(self.shape)?;
        self.layout.validate()?;
        if self.gamma1.len() != self.shape.dimension || self.gamma2.len() != self.shape.dimension {
            return Err("SWA gamma lengths must match shape.dimension".into());
        }
        let batch = self.tokens.len();
        if batch == 0 || !batch.is_power_of_two() || self.partial_rows.len() != batch {
            return Err("SWA batch must be a positive power of two matching partial_rows".into());
        }
        let chunk_params = cache_chunk_params_with_f(
            self.layout,
            self.shape.kv_dim(),
            f_override,
            profile,
        )?;
        let mut recomputed = Vec::with_capacity(batch);
        let mut by_chunks: BTreeMap<&[usize], Vec<usize>> = BTreeMap::new();
        for (t, tok) in self.tokens.iter().enumerate() {
            let rec = check_windowed_public_step(
                self.shape,
                &self.policy,
                &self.gamma1,
                &self.gamma2,
                &tok.step,
            )?;
            let context = tok.step.position as usize + 1;
            let expected_full = self.layout.full_chunks(context);
            if tok.chunk_ids.len() != expected_full
                || tok.chunk_ids.iter().any(|&c| c >= self.chunk_commitments.len())
            {
                return Err("SWA token chunk_ids mismatch".into());
            }
            let expected_partial = context - expected_full * self.layout.chunk;
            let (ref k_rows, ref v_rows) = self.partial_rows[t];
            if k_rows.len() != expected_partial
                || v_rows.len() != expected_partial
                || k_rows.iter().chain(v_rows).any(|r| r.len() != self.shape.kv_dim())
            {
                return Err("SWA partial_rows shape mismatch".into());
            }
            if k_rows.last() != Some(&rec.k_rot) || v_rows.last() != Some(&rec.v_q) {
                return Err("SWA partial_rows last entry does not equal recomputed (k_rot, v_q)".into());
            }
            let full_hashes: Vec<Hash> = tok
                .chunk_ids
                .iter()
                .map(|&id| self.chunk_commitments[id])
                .collect();
            let before = stage_b_step_cache_commitment(
                &full_hashes,
                &k_rows[..expected_partial - 1],
                &v_rows[..expected_partial - 1],
            );
            if tok.cache_commitment_before != before {
                return Err("SWA cache_commitment_before mismatch".into());
            }
            let after = if expected_partial == self.layout.chunk {
                let pad_mat = |rows: &[Vec<i8>]| -> Vec<u32> {
                    rows.iter()
                        .flat_map(|r| r.iter().map(|&v| i8_to_canonical_m31_u32(v)))
                        .collect()
                };
                let k_u32 = pad_mat(k_rows);
                let v_u32 = pad_mat(v_rows);
                let roll = cprs::register::register_group_commitment(&[&k_u32, &v_u32], &chunk_params)
                    .map_err(|e| e.to_string())?;
                let mut rolled = full_hashes;
                rolled.push(roll);
                stage_b_step_cache_commitment(&rolled, &[], &[])
            } else {
                stage_b_step_cache_commitment(&full_hashes, k_rows, v_rows)
            };
            if tok.cache_commitment_after != after {
                return Err("SWA cache_commitment_after mismatch".into());
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
                    return Err("SWA partial_rows prefix mismatch".into());
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

fn dot_i32_eq_slice(eq_w: &[[i64; 4]], values: &[i32]) -> SecureField {
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

/// Verifies active-window score dots `scores[t][h][k] = <q_rot[t,h], K_active[local_start_t + k, h/G]>`
/// via a factored tensor-indicator degree-2 sumcheck over `(r_pos, kv_col)` (`log_rows + log_kv_dim` rounds).
/// Evaluates the left tensor factor in `O(W + H * d_h)` per token and leaves ONE shared evaluation
/// point `(s_rows, s_cols)` on `K_active` across all tokens in the batch.
pub fn verify_windowed_scores(
    shape: GptOssAttentionShape,
    padded_active_rows: usize,
    q_rot: &[&[i8]],
    local_starts: &[usize],
    scores_win: &[&[Vec<i32>]],
    proof: &ContractionProof,
    channel: &mut Blake2sChannel,
) -> Result<(), String> {
    let tokens = q_rot.len();
    if tokens == 0
        || local_starts.len() != tokens
        || scores_win.len() != tokens
        || proof.claims.len() != tokens
    {
        return Err("windowed scores proof shape mismatch".into());
    }
    let log_rows = padded_active_rows.ilog2() as usize;
    let log_kv_dim = shape.kv_dim().ilog2() as usize;
    if proof.proof.round_polys.len() != log_rows + log_kv_dim {
        return Err("windowed scores sumcheck round count mismatch".into());
    }
    let r_heads = channel.draw_secure_felts(shape.heads.ilog2() as usize);
    let r_win = channel.draw_secure_felts(log_rows);
    let head_w = eq_evals(&r_heads);
    let win_w_i64 = eq_evals_i64(&r_win);
    let win_w = eq_evals(&r_win);

    let mut claims = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let ls = local_starts[t];
        let win_len = scores_win[t].first().map_or(0, Vec::len);
        if ls + win_len > padded_active_rows {
            return Err("windowed scores active range exceeds padded_active_rows".into());
        }
        let slice_w = &win_w_i64[ls..ls + win_len];
        let claim: SecureField = scores_win[t]
            .iter()
            .zip(&head_w)
            .map(|(hs, &hw)| hw * dot_i32_eq_slice(slice_w, hs))
            .sum();
        claims.push(claim);
    }
    channel.mix_felts(&claims);
    let lambda = channel.draw_secure_felt();
    let (s, final_eval) = partially_verify(combine_horner(&claims, lambda), &proof.proof, channel)?;
    if s.len() != log_rows + log_kv_dim {
        return Err("windowed scores assignment length mismatch".into());
    }
    let (s_rows, s_cols) = s.split_at(log_rows);
    for t in 0..tokens {
        if proof.claims[t].point != s {
            return Err("windowed scores cache claim point mismatch".into());
        }
    }
    let s_row_eq = eq_evals(s_rows);
    let s_col_eq = eq_evals(s_cols);
    let group = shape.group_size();
    let mut expected = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let ls = local_starts[t];
        let win_len = scores_win[t][0].len();
        let row_factor: SecureField = (ls..ls + win_len)
            .map(|r| s_row_eq[r] * win_w[r])
            .sum();
        let mut q_left = vec![SecureField::from(i8_to_m31(0)); shape.kv_dim()];
        for h in 0..shape.heads {
            let hw = head_w[h];
            let kv_base = (h / group) * shape.head_dim;
            let q_base = h * shape.head_dim;
            for i in 0..shape.head_dim {
                q_left[kv_base + i] += hw * i8_to_m31(q_rot[t][q_base + i]);
            }
        }
        let col_factor: SecureField = s_col_eq.iter().zip(&q_left).map(|(&w, &v)| w * v).sum();
        expected.push(row_factor * col_factor * proof.claims[t].value);
        channel.mix_felts(&[proof.claims[t].value]);
        channel.mix_felts(&proof.claims[t].full_chunk_openings);
    }
    if final_eval != combine_horner(&expected, lambda) {
        return Err("windowed scores sumcheck final check failed".into());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct StageBWindowSinkProof {
    pub f_override: Option<usize>,
    pub qkv: ProjectionProof,
    pub scores: ContractionProof,
    pub context: ContractionProof,
    pub o: ProjectionProof,
    pub leaf_sub_commitments: [Hash; 3],
    pub unified_cprs_proof: Vec<u8>,
}

pub fn swa_outer_digest(statement_digest: &[u8; 32]) -> Hash {
    let mut h = Blake2s256::new();
    h.update(SWA_OUTER_DOMAIN);
    h.update(statement_digest);
    h.finalize().into()
}

fn collect_pruned_chunk_openings(
    active_cids_per_token: &[&[usize]],
    scores_claims: &[CacheOpeningClaim],
    ctx_claims: &[CacheOpeningClaim],
) -> Result<Vec<CacheChunkOpening>, String> {
    let mut map: BTreeMap<usize, CacheChunkOpening> = BTreeMap::new();
    for (t, cids) in active_cids_per_token.iter().enumerate() {
        let sc = &scores_claims[t];
        let cc = &ctx_claims[t];
        let k_pt = sc.point[sc.log_chunks..].to_vec();
        let v_pt = cc.point[cc.log_chunks..].to_vec();
        for (pos, &chunk) in cids.iter().enumerate() {
            let entry = CacheChunkOpening {
                chunk,
                k_point: k_pt.clone(),
                k_value: sc.full_chunk_openings[pos],
                v_point: v_pt.clone(),
                v_value: cc.full_chunk_openings[pos],
            };
            match map.get(&chunk) {
                None => {
                    map.insert(chunk, entry);
                }
                Some(prev) => {
                    if prev.k_point != entry.k_point
                        || prev.k_value != entry.k_value
                        || prev.v_point != entry.v_point
                        || prev.v_value != entry.v_value
                    {
                        return Err("SWA tokens disagree on active KV-cache chunk opening".into());
                    }
                }
            }
        }
    }
    Ok(map.into_values().collect())
}

pub fn verify_stage_b_window_sinks_against_anchor(
    expected_anchor: &[u8; 32],
    statement: &StageBWindowSinkStatement,
    profile: &SecurityProfile,
    proof: &StageBWindowSinkProof,
) -> Result<Vec<WindowedRecomputedStep>, String> {
    if &statement.registered_anchor() != expected_anchor {
        return Err("SWA statement does not match expected registered layer anchor".into());
    }
    verify_stage_b_window_sinks(statement, profile, proof)
}

pub fn verify_stage_b_window_sinks(
    statement: &StageBWindowSinkStatement,
    profile: &SecurityProfile,
    proof: &StageBWindowSinkProof,
) -> Result<Vec<WindowedRecomputedStep>, String> {
    if proof.f_override != statement.cprs_f {
        return Err("SWA proof f_override does not match statement.cprs_f".into());
    }
    let recomputed = statement.validate_and_recompute(statement.cprs_f, profile)?;
    if compute_swa_leaf_commitment(&proof.leaf_sub_commitments) != statement.leaf_commitment {
        return Err("SWA leaf sub-commitments do not hash to leaf_commitment".into());
    }
    let stmt_digest = statement.digest();
    let mut channel = StageBStatement::channel_from_digest(&stmt_digest);

    let x1_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.x1.clone()).collect();
    let q_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.q_raw.clone()).collect();
    let k_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.k_raw.clone()).collect();
    let v_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.v_raw.clone()).collect();
    let qkv_claims =
        verify_hetero_col_projections(&x1_in, &[&q_out, &k_out, &v_out], &proof.qkv, &mut channel)?;

    let first_chunk = statement.first_active_chunk();
    let pruned_rows = first_chunk * statement.layout.chunk;
    let active_cids: Vec<&[usize]> = statement
        .tokens
        .iter()
        .map(|t| &t.chunk_ids[first_chunk..])
        .collect();
    let active_cache_lengths: Vec<usize> = statement
        .tokens
        .iter()
        .map(|t| (t.step.position as usize + 1) - pruned_rows)
        .collect();
    let local_starts: Vec<usize> = statement
        .tokens
        .iter()
        .map(|t| statement.policy.window_start(t.step.position) - pruned_rows)
        .collect();
    let padded_active = active_cache_lengths
        .iter()
        .map(|&c| statement.layout.padded_positions(c))
        .max()
        .unwrap();

    let q_rot_refs: Vec<&[i8]> = recomputed.iter().map(|r| r.q_rot.as_slice()).collect();
    let scores_refs: Vec<&[Vec<i32>]> =
        statement.tokens.iter().map(|t| t.step.scores.as_slice()).collect();
    verify_windowed_scores(
        statement.shape,
        padded_active,
        &q_rot_refs,
        &local_starts,
        &scores_refs,
        &proof.scores,
        &mut channel,
    )?;

    let local_weights: Vec<Vec<Vec<u16>>> = recomputed
        .iter()
        .enumerate()
        .map(|(t, rec)| {
            let ls = local_starts[t];
            let act_len = active_cache_lengths[t];
            rec.weights
                .iter()
                .map(|hw| {
                    let mut v = vec![0u16; act_len];
                    v[ls..ls + hw.len()].copy_from_slice(hw);
                    v
                })
                .collect()
        })
        .collect();
    let lw_refs: Vec<&[Vec<u16>]> = local_weights.iter().map(Vec::as_slice).collect();
    let ctx_raw_refs: Vec<&[i32]> =
        statement.tokens.iter().map(|t| t.step.ctx_raw.as_slice()).collect();
    verify_context(
        statement.shape.inner_attention_shape(),
        statement.layout,
        &lw_refs,
        &active_cache_lengths,
        &ctx_raw_refs,
        &proof.context,
        &mut channel,
    )?;

    let ctx_in: Vec<Vec<i8>> = recomputed.iter().map(|r| r.ctx.clone()).collect();
    let o_out: Vec<Vec<i32>> = statement.tokens.iter().map(|t| t.step.o_raw.clone()).collect();
    let o_claims = verify_projections(&ctx_in, &[&o_out], &proof.o, &mut channel)?;

    let k_partial_refs: Vec<&[Vec<i8>]> =
        statement.partial_rows.iter().map(|(k, _)| k.as_slice()).collect();
    let v_partial_refs: Vec<&[Vec<i8>]> =
        statement.partial_rows.iter().map(|(_, v)| v.as_slice()).collect();
    verify_cache_claims_batch(
        &active_cids,
        &k_partial_refs,
        statement.layout,
        padded_active,
        &proof.scores.claims,
    )?;
    verify_cache_claims_batch(
        &active_cids,
        &v_partial_refs,
        statement.layout,
        padded_active,
        &proof.context.claims,
    )?;

    let opened_chunks =
        collect_pruned_chunk_openings(&active_cids, &proof.scores.claims, &proof.context.claims)?;
    let leaf_params = swa_leaf_group_params(statement.shape, proof.f_override, profile)?;
    let chunk_params = cache_chunk_params_with_f(
        statement.layout,
        statement.shape.kv_dim(),
        proof.f_override,
        profile,
    )?;

    let mut group_coms: Vec<Hash> = proof.leaf_sub_commitments.to_vec();
    let mut group_params: Vec<Params> = leaf_params.to_vec();
    let mut points: Vec<Vec<QM31>> = vec![
        stwo_point_to_cprs(&qkv_claims[0].0),
        stwo_point_to_cprs(&qkv_claims[1].0),
        stwo_point_to_cprs(&qkv_claims[2].0),
        stwo_point_to_cprs(&o_claims.point),
    ];
    let mut claims_cprs: Vec<QM31> = vec![
        to_cprs_qm31(qkv_claims[0].1),
        to_cprs_qm31(qkv_claims[1].1),
        to_cprs_qm31(qkv_claims[2].1),
        to_cprs_qm31(o_claims.openings[0]),
    ];
    for oc in &opened_chunks {
        let com = statement
            .chunk_commitments
            .get(oc.chunk)
            .copied()
            .ok_or_else(|| "SWA active chunk commitment out of bounds".to_string())?;
        group_coms.push(com);
        group_params.push(chunk_params.clone());
        points.push(stwo_point_to_cprs(&oc.k_point));
        claims_cprs.push(to_cprs_qm31(oc.k_value));
        points.push(stwo_point_to_cprs(&oc.v_point));
        claims_cprs.push(to_cprs_qm31(oc.v_value));
    }

    let outer = swa_outer_digest(&stmt_digest);
    verify_heterogeneous_strict_context(
        &group_coms,
        &group_params,
        &points,
        &claims_cprs,
        &outer,
        &proof.unified_cprs_proof,
    )
    .map_err(|e| format!("StageBWindowSinks unified CPRS opening rejected: {e}"))?;

    Ok(recomputed)
}

fn read_swa_hash_32(reader: &mut ReceiptReader<'_>, label: &str) -> Result<Hash, String> {
    let mut out = [0u8; 32];
    for b in &mut out {
        *b = reader.read_u32_byte(label)?;
    }
    Ok(out)
}

fn put_swa_contraction(out: &mut Vec<u8>, proof: &ContractionProof) -> Result<(), String> {
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

fn read_swa_contraction(
    reader: &mut ReceiptReader<'_>,
    expected_tokens: usize,
    expected_point_len: usize,
    expected_log_chunks: usize,
    active_chunk_lens: &[usize],
    label: &str,
) -> Result<ContractionProof, String> {
    let proof = reader.read_sumcheck(&format!("{label} sumcheck"))?;
    let count = reader.read_len(expected_tokens, &format!("{label} claim count"))?;
    if count != expected_tokens {
        return Err(format!("{label} claim count mismatch"));
    }
    let mut claims = Vec::with_capacity(count);
    for (t, &fco_len) in active_chunk_lens.iter().enumerate() {
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

pub fn stage_b_window_sink_receipt_bytes(proof: &StageBWindowSinkProof) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(STAGE_B_WINDOW_SINK_RECEIPT_MAGIC);
    put_u32(&mut out, STAGE_B_WINDOW_SINK_RECEIPT_VERSION);
    put_u32(&mut out, proof.f_override.unwrap_or(0) as u32);
    put_projection_proof(&mut out, &proof.qkv)?;
    put_swa_contraction(&mut out, &proof.scores)?;
    put_swa_contraction(&mut out, &proof.context)?;
    put_projection_proof(&mut out, &proof.o)?;
    for c in &proof.leaf_sub_commitments {
        out.extend_from_slice(c);
    }
    put_bytes(&mut out, &proof.unified_cprs_proof)?;
    Ok(out)
}

pub fn decode_stage_b_window_sink_receipt_bytes(
    bytes: &[u8],
    statement: &StageBWindowSinkStatement,
) -> Result<StageBWindowSinkProof, String> {
    if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOSWA1 receipt exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(STAGE_B_WINDOW_SINK_RECEIPT_MAGIC, "STWOSWA1")?;
    let ver = reader.read_u32("STWOSWA1 version")?;
    if ver != STAGE_B_WINDOW_SINK_RECEIPT_VERSION {
        return Err(format!("unsupported STWOSWA1 version {ver}"));
    }
    let f_raw = reader.read_u32("STWOSWA1 f_override")? as usize;
    let f_override = if f_raw == 0 { None } else { Some(f_raw) };
    if f_override != statement.cprs_f {
        return Err("STWOSWA1 wire f_override does not match statement.cprs_f".into());
    }
    let tokens = statement.tokens.len();
    let first_chunk = statement.first_active_chunk();
    let pruned_rows = first_chunk * statement.layout.chunk;
    let padded_active = statement
        .tokens
        .iter()
        .map(|t| statement.layout.padded_positions((t.step.position as usize + 1) - pruned_rows))
        .max()
        .unwrap_or(2 * statement.layout.chunk);
    let log_rows = padded_active.ilog2() as usize;
    let log_chunks = log_rows - statement.layout.chunk.ilog2() as usize;
    let log_kv_dim = statement.shape.kv_dim().ilog2() as usize;
    let active_chunk_lens: Vec<usize> = statement
        .tokens
        .iter()
        .map(|t| t.chunk_ids.len().saturating_sub(first_chunk))
        .collect();

    let qkv = read_projection_proof(&mut reader, 3, "STWOSWA1 qkv")?;
    let scores = read_swa_contraction(
        &mut reader,
        tokens,
        log_rows + log_kv_dim,
        log_chunks,
        &active_chunk_lens,
        "STWOSWA1 scores",
    )?;
    let context = read_swa_contraction(
        &mut reader,
        tokens,
        log_rows + log_kv_dim,
        log_chunks,
        &active_chunk_lens,
        "STWOSWA1 context",
    )?;
    let o = read_projection_proof(&mut reader, 1, "STWOSWA1 o")?;
    let mut leaf_sub_commitments = [[0u8; 32]; 3];
    for slot in &mut leaf_sub_commitments {
        *slot = read_swa_hash_32(&mut reader, "STWOSWA1 leaf sub-commitment")?;
    }
    let unified_cprs_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "STWOSWA1 cprs proof")?
        .to_vec();
    reader.finish()?;
    Ok(StageBWindowSinkProof {
        f_override,
        qkv,
        scores,
        context,
        o,
        leaf_sub_commitments,
        unified_cprs_proof,
    })
}
