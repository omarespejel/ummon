//! Standalone verifier library (`ummon-verifier`) for Ummon Leaf v4 spot-check bundles.
//!
//! Zero prover dependencies (no `stwo` prover feature, no `rayon`, no GPU libraries).
//! Builds cleanly to native targets (`aarch64`/`x86_64`) and WebAssembly (`wasm32-unknown-unknown`,
//! `wasm32-wasip1`).

pub mod blake2s_air;
pub mod cprs;
pub mod poseidon2_air;
pub mod poseidon2_m31;
pub mod recursive_audit;
pub mod stage_a;
pub mod stage_b;
pub mod stage_b_mla;
pub mod stage_b_window_sinks;
pub mod standalone_audit;
pub use stage_a::zk_full as stage_a_zk_full;

use std::ops::{Add, Deref, Mul, Neg, Sub};

use blake2::{Blake2s256, Digest};
use num_traits::{One, Zero};
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::{ExtensionOf, Field};
use stwo::core::Fraction;

pub use crate::recursive_audit::{
    compute_recursive_audit_block_anchor, decode_recursive_audit_receipt, expected_is_first_root,
    is_first_column_id, mix_recursive_audit_statement, pcs_config_for_profile,
    recursive_audit_receipt_bytes, verify_recursive_audit_stark_against_anchor,
    RecursiveAuditAirEval, RecursiveAuditComponent, RecursiveAuditStarkProof,
    RecursiveAuditStatement, RecursiveAuditVerifierRun, IS_FIRST_PREPROCESSED_ROOTS_BLOWUP_1,
    MEM_BUS_TRACE_COLS, MEM_BUS_TRACE_IDX, RECURSIVE_AUDIT_DOMAIN_TAG, RECURSIVE_AUDIT_MAGIC,
    RECURSIVE_AUDIT_VERSION, WIRING_TRACE_COLS, WITNESS_TRACE_COLS,
};
pub use crate::stage_b_mla::{
    check_mla_public_step, compute_mla_chunk_commitment, compute_mla_leaf_commitment,
    compute_mla_registered_anchor, decode_stage_b_mla_receipt_bytes, mla_cache_chunk_group_params,
    mla_leaf_group_params, mla_step_cache_commitment, stage_b_mla_receipt_bytes,
    verify_mla_absorbed_q_weights, verify_stage_b_mla, verify_stage_b_mla_against_anchor,
    MlaAttentionPolicy, MlaOpenedChunkSubCommitments, MlaPublicStep, MlaRecomputedStep, MlaShape,
    MlaToken, StageBMlaProof, StageBMlaStatement, MLA_REGISTERED_ANCHOR_DOMAIN,
    STAGE_B_MLA_RECEIPT_MAGIC, STAGE_B_MLA_RECEIPT_VERSION,
};
pub use crate::stage_b_window_sinks::{
    check_windowed_public_step, compute_swa_leaf_commitment, compute_swa_registered_anchor,
    decode_stage_b_window_sink_receipt_bytes, softmax_weights_with_sink,
    stage_b_window_sink_receipt_bytes, swa_leaf_group_params, verify_stage_b_window_sinks,
    verify_stage_b_window_sinks_against_anchor, AttentionSinkSpec, GptOssAttentionShape,
    SlidingWindowSpec, StageBWindowSinkProof, StageBWindowSinkStatement, WindowSinkAttentionPolicy,
    WindowedPublicStep, WindowedRecomputedStep, WindowedToken, STAGE_B_WINDOW_SINK_RECEIPT_MAGIC,
    STAGE_B_WINDOW_SINK_RECEIPT_VERSION, SWA_REGISTERED_ANCHOR_DOMAIN,
};
pub use crate::standalone_audit::{
    blake2s_bytes_to_u32_words, cairo_short_string_to_u256_le, compute_salted_tokens_commitment,
    compute_standalone_block_anchor, compute_standalone_block_public_io_digest,
    compute_standalone_block_public_io_digest_for_profile,
    compute_standalone_chain_anchor, compute_standalone_ends_public_io_digest,
    compute_standalone_model_anchor,
    compute_whole_model_full_stack_anchor,
    decode_standalone_block_receipt, decode_whole_model_receipt,
    pack_stark_proof_to_cairo_felts_u256, pack_stark_proof_to_cairo_u32_words,
    pack_u32_words_to_cairo_felts_u256, standalone_block_receipt_bytes,
    standalone_lookup_pow_bits_for_profile_id, standalone_min_zk_blind_rows_for_queries,
    EXP_POS2_PROFILES, STANDALONE_PROFILE_IDS,
    standalone_pcs_config_for_profile_id, PRODUCTION_HARDENED_LOOKUP_POW_BITS,
    PRODUCTION_HARDENED_PROFILE_ID,
    standalone_zk_blind_rows_for_pcs, standalone_zk_blind_rows_for_profile_id,
    starknet_compute_activation_leaf_hash, starknet_compute_committed_io_binding_digest,
    starknet_compute_escrow_request_digest, starknet_compute_public_io_binding_digest,
    starknet_compute_request_bound_initial_chain_digest, validate_standalone_zk_blind_rows,
    verify_standalone_block_chain, verify_standalone_block_chain_against_anchor,
    verify_standalone_block_chain_against_anchor_timed, verify_standalone_block_chain_timed,
    verify_standalone_block_stark_against_anchor,
    verify_standalone_block_stark_against_anchor_and_io,
    verify_standalone_model_chain_against_anchor, verify_standalone_model_chain_against_anchor_timed,
    verify_standalone_model_gen_chain_against_anchor,
    verify_standalone_model_gen_chain_against_anchor_timed,
    verify_whole_model_committed_stark_against_anchor, verify_whole_model_stark_against_anchor,
    whole_model_receipt_bytes, StandaloneAuditAirEval, StandaloneAuditComponent,
    StandaloneAuditVerifierRun, StandaloneBlockStarkProof, StandaloneBlockStatement,
    StandaloneChainLink, StandaloneChainVerdict, StandaloneChainedBlock, StandaloneClaimedSums,
    StandaloneEndsDecl, StandaloneEndsIo, StandaloneModelMultiVerdict, StandaloneModelVerdict,
    StandaloneTransformerDims, TransformerBlockShape, WholeModelStandaloneStarkProof,
    WholeModelStatement, EMITTED_COMMITMENT_DOMAIN, ENDS_FLAG_EMBED, ENDS_FLAG_HEAD,
    ENDS_FLAG_MULTI, MAX_ENDS_VOCAB, MAX_MULTI_HEAD_BV, MAX_STANDALONE_BLOCK_LAYERS,
    PROMPT_COMMITMENT_DOMAIN, STANDALONE_AUDIT_VERSION, STANDALONE_BLOCK_MAGIC,
    STANDALONE_CHAIN_ANCHOR_DOMAIN, STANDALONE_ENDS_IO_DOMAIN, STANDALONE_ENDS_MULTI_IO_DOMAIN,
    STANDALONE_MODEL_ANCHOR_DOMAIN, STANDALONE_PUBLIC_IO_DOMAIN, WHOLE_MODEL_MAGIC,
};
pub use crate::stage_a::{
    cross_layer_gkr_block_bytes, decode_cross_layer_gkr_block, decode_stage_a_gkr_lane_receipt,
    decode_stage_a_gkr_lane_receipt_with_mode, decode_stage_a_receipt_bytes,
    encode_stage_a_receipt_bytes_matching_header, stage_a_gkr_lane_receipt_bytes,
    stage_a_gkr_lane_receipt_bytes_with_mode, verify_cross_layer_gkr_block,
    verify_cross_layer_gkr_block_with_policies,
    verify_stage_a_gkr_lane_batch_against_anchor,
    verify_stage_a_gkr_lane_batch_against_anchor_mode,
    verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b,
    verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode,
    verify_stage_a_public_witness_batch_against_anchor_with_stage_b,
    verify_stage_a_receipt_against_anchor, verify_stage_a_receipt_against_anchor_with_stage_b,
    CrossLayerGkrBlockProof, CrossLayerGkrBlockVerifierRun, StageAChainLink,
    StageAGkrLaneBatchProof, StageANonlinearBatchStatement, StageANumericPolicy,
    StageAReceiptProof, StageARegisteredContext, StageARegisteredExpertOpening,
    StageARegisteredVerifierRun, StageAShape, StageAZkLaneAux, STAGE_A_RECEIPT_MODE_GKR_LANE,
    STAGE_A_RECEIPT_MODE_ZK_FULL, STAGE_A_RECEIPT_MODE_ZK_GKR_LANE,
};
use crate::stage_b::{
    decode_stage_b_receipt_bytes, group_params, put_projection_proof, read_projection_proof,
    residual_out, stage_b_receipt_bytes, verify_embedding_lookup, verify_lm_head, verify_stage_b,
    verify_stage_b_defer_cprs, AttentionPolicy, AttentionShape, CacheLayout, EmbeddingProof,
    EmbeddingStatement, LmHeadProof, LmHeadStatement, PublicStep, RecomputedStep, StageBProof,
    StageBStatement, StageBToken,
};

pub const MAX_DEGREE: usize = 3;
pub const MAX_WIRE_RECEIPT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_WIRE_SUMCHECK_ROUNDS: usize = 64;
const SECURE_FIELD_WIRE_BYTES: usize = 4 * 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnivariatePoly<F: Field>(Vec<F>);

impl<F: Field> UnivariatePoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut polynomial = Self(coeffs);
        polynomial.truncate_leading_zeros();
        polynomial
    }

    pub fn eval_at_point(&self, x: F) -> F {
        self.0
            .iter()
            .rfold(F::zero(), |acc, coeff| acc * x + *coeff)
    }

    pub fn degree(&self) -> usize {
        let mut coeffs = self.0.iter().rev();
        let _ = (&mut coeffs).take_while(|v| v.is_zero());
        coeffs.len().saturating_sub(1)
    }

    fn truncate_leading_zeros(&mut self) {
        while self.0.last() == Some(&F::zero()) {
            self.0.pop();
        }
    }
}

impl<F: Field> From<F> for UnivariatePoly<F> {
    fn from(value: F) -> Self {
        Self::new(vec![value])
    }
}

impl<F: Field> Mul<F> for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        self.0.iter_mut().for_each(|coeff| *coeff *= rhs);
        self
    }
}

impl<F: Field> Mul for UnivariatePoly<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        self.truncate_leading_zeros();
        rhs.truncate_leading_zeros();
        let mut res = vec![F::zero(); self.0.len() + rhs.0.len() - 1];
        for (i, coeff_a) in self.0.into_iter().enumerate() {
            for (j, coeff_b) in rhs.0.iter().enumerate() {
                res[i + j] += coeff_a * *coeff_b;
            }
        }
        Self::new(res)
    }
}

impl<F: Field> Add for UnivariatePoly<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.0.len().max(rhs.0.len());
        let mut res = Vec::with_capacity(n);
        for i in 0..n {
            res.push(match (self.0.get(i), rhs.0.get(i)) {
                (Some(a), Some(b)) => *a + *b,
                (Some(a), None) | (None, Some(a)) => *a,
                _ => unreachable!(),
            });
        }
        Self(res)
    }
}

impl<F: Field> Sub for UnivariatePoly<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl<F: Field> Neg for UnivariatePoly<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.into_iter().map(|v| -v).collect())
    }
}

impl<F: Field> Zero for UnivariatePoly<F> {
    fn zero() -> Self {
        Self(vec![])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(F::is_zero)
    }
}

impl<F: Field> Deref for UnivariatePoly<F> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct SumcheckProof {
    pub round_polys: Vec<UnivariatePoly<SecureField>>,
}

#[inline]
pub fn zk_lo_basis_r1_star() -> SecureField {
    use stwo::core::fields::FieldExpOps;
    let u = SecureField::from_u32_unchecked(0, 0, 1, 0);
    let one_plus_u = SecureField::from_u32_unchecked(1, 0, 1, 0);
    u * one_plus_u.inverse()
}

#[inline]
pub fn zk_lo_basis_r0_star() -> SecureField {
    use stwo::core::fields::FieldExpOps;
    let i = SecureField::from_u32_unchecked(0, 1, 0, 0);
    let one_plus_i = SecureField::from_u32_unchecked(1, 1, 0, 0);
    i * one_plus_i.inverse()
}

#[inline]
pub fn zk_lo_basis_omega_scale() -> SecureField {
    SecureField::from_u32_unchecked(1, 1, 1, 1)
}

#[inline]
pub fn cfs_deg3_accumulate_weights(
    w_cfs: &mut [SecureField],
    base_u: usize,
    point: &[SecureField],
    gamma_sum: SecureField,
    gamma_eval: SecureField,
) {
    let n_vars = point.len();
    if n_vars == 0 {
        return;
    }
    let two_n_m1 = SecureField::from(BaseField::from_u32_unchecked(1u32 << (n_vars - 1)));
    let two = SecureField::from(BaseField::from_u32_unchecked(2));
    let w_sum_0 = gamma_sum * two_n_m1 * two;
    let w_sum_123 = gamma_sum * two_n_m1;
    for (k, &r) in point.iter().enumerate() {
        let r2 = r * r;
        let r3 = r2 * r;
        let off = base_u + 4 * k;
        w_cfs[off] += w_sum_0 + gamma_eval;
        w_cfs[off + 1] += w_sum_123 + gamma_eval * r;
        w_cfs[off + 2] += w_sum_123 + gamma_eval * r2;
        w_cfs[off + 3] += w_sum_123 + gamma_eval * r3;
    }
}

pub fn partially_verify(
    claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
) -> Result<(Vec<SecureField>, SecureField), String> {
    partially_verify_with_lo_basis(claim, proof, channel, false)
}

pub fn partially_verify_with_lo_basis(
    mut claim: SecureField,
    proof: &SumcheckProof,
    channel: &mut impl Channel,
    fold_lo_basis: bool,
) -> Result<(Vec<SecureField>, SecureField), String> {
    let n_rounds = proof.round_polys.len();
    let mut assignment = Vec::with_capacity(n_rounds);
    for (round, round_poly) in proof.round_polys.iter().enumerate() {
        if round_poly.degree() > MAX_DEGREE {
            return Err(format!(
                "degree of the polynomial in round {round} is too high"
            ));
        }
        let sum = round_poly.eval_at_point(Zero::zero()) + round_poly.eval_at_point(One::one());
        if claim != sum {
            return Err(format!(
                "sum does not match the claim in round {round} (sum {sum}, claim {claim})"
            ));
        }
        channel.mix_felts(round_poly);
        let drawn = channel.draw_secure_felt();
        let challenge = if fold_lo_basis && n_rounds >= 2 && round == n_rounds - 2 {
            zk_lo_basis_r1_star()
        } else if fold_lo_basis && n_rounds >= 2 && round == n_rounds - 1 {
            zk_lo_basis_r0_star()
        } else {
            drawn
        };
        claim = round_poly.eval_at_point(challenge);
        assignment.push(challenge);
    }
    Ok((assignment, claim))
}

pub fn eq<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    std::iter::zip(x, y)
        .map(|(xi, yi)| *xi * *yi + (F::one() - *xi) * (F::one() - *yi))
        .product()
}

pub fn fold_mle_evals<F>(assignment: SecureField, eval0: F, eval1: F) -> SecureField
where
    F: Field,
    SecureField: ExtensionOf<F>,
{
    assignment * (eval1 - eval0) + eval0
}

pub fn random_linear_combination(v: &[SecureField], alpha: SecureField) -> SecureField {
    v.iter()
        .rfold(SecureField::zero(), |acc, coeff| acc * alpha + *coeff)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GkrMask {
    columns: Vec<[SecureField; 2]>,
}

impl GkrMask {
    pub fn new(columns: Vec<[SecureField; 2]>) -> Self {
        Self { columns }
    }

    pub fn to_rows(&self) -> [Vec<SecureField>; 2] {
        self.columns.iter().copied().map( Into::<(_, _)>::into ).unzip().into()
    }

    pub fn columns(&self) -> &[[SecureField; 2]] {
        &self.columns
    }

    pub fn reduce_at_point(&self, x: SecureField) -> Vec<SecureField> {
        self.columns
            .iter()
            .map(|[v0, v1]| fold_mle_evals(x, *v0, *v1))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Gate {
    LogUp,
    GrandProduct,
}

impl Gate {
    pub fn eval(&self, mask: &GkrMask) -> Result<Vec<SecureField>, String> {
        match self {
            Self::LogUp => {
                if mask.columns.len() != 2 {
                    return Err("invalid mask length for LogUp".into());
                }
                let [num_a, num_b] = mask.columns[0];
                let [denom_a, denom_b] = mask.columns[1];
                let a = Fraction::new(num_a, denom_a);
                let b = Fraction::new(num_b, denom_b);
                let res = a + b;
                Ok(vec![res.numerator, res.denominator])
            }
            Self::GrandProduct => {
                if mask.columns.len() != 1 {
                    return Err("invalid mask length for GrandProduct".into());
                }
                let [a, b] = mask.columns[0];
                Ok(vec![a * b])
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GkrBatchProof {
    pub sumcheck_proofs: Vec<SumcheckProof>,
    pub layer_masks_by_instance: Vec<Vec<GkrMask>>,
    pub output_claims_by_instance: Vec<Vec<SecureField>>,
}

#[derive(Debug, Clone)]
pub struct GkrArtifact {
    pub ood_point: Vec<SecureField>,
    pub claims_to_verify_by_instance: Vec<Vec<SecureField>>,
    pub n_variables_by_instance: Vec<usize>,
}

pub fn partially_verify_batch(
    gate_by_instance: Vec<Gate>,
    proof: &GkrBatchProof,
    channel: &mut impl Channel,
) -> Result<GkrArtifact, String> {
    let GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    } = proof;

    if layer_masks_by_instance.len() != output_claims_by_instance.len() {
        return Err("GKR batch proof instance lengths do not match".into());
    }

    let n_instances = layer_masks_by_instance.len();
    let instance_n_layers = |instance: usize| layer_masks_by_instance[instance].len();
    let n_layers = (0..n_instances).map(instance_n_layers).max().unwrap_or(0);

    if n_layers != sumcheck_proofs.len() {
        return Err("GKR batch proof sumcheck count does not match max layers".into());
    }

    if gate_by_instance.len() != n_instances {
        return Err("GKR batch proof gate count does not match instance count".into());
    }

    let mut ood_point = vec![];
    let mut claims_to_verify_by_instance: Vec<Option<Vec<SecureField>>> = vec![None; n_instances];

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let n_remaining_layers = n_layers - layer;

        for instance in 0..n_instances {
            if instance_n_layers(instance) == n_remaining_layers {
                let output_claims = output_claims_by_instance[instance].clone();
                claims_to_verify_by_instance[instance] = Some(output_claims);
            }
        }

        for claims_to_verify in claims_to_verify_by_instance.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }

        let sumcheck_alpha = channel.draw_secure_felt();
        let instance_lambda = channel.draw_secure_felt();

        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();

        for (instance, claims_to_verify) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let n_unused_variables = n_layers - instance_n_layers(instance);
                let doubling_factor = BaseField::from_u32_unchecked(1u32 << n_unused_variables);
                let claim =
                    random_linear_combination(claims_to_verify, instance_lambda) * doubling_factor;
                sumcheck_claims.push(claim);
                sumcheck_instances.push(instance);
            }
        }

        let sumcheck_claim = random_linear_combination(&sumcheck_claims, sumcheck_alpha);
        let (sumcheck_ood_point, sumcheck_eval) =
            partially_verify(sumcheck_claim, sumcheck_proof, channel)?;

        let mut layer_evals = Vec::new();

        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            let gate = &gate_by_instance[instance];
            let gate_output = gate.eval(mask)?;
            let eq_eval = eq(&ood_point[n_unused..], &sumcheck_ood_point[n_unused..]);
            layer_evals.push(eq_eval * random_linear_combination(&gate_output, instance_lambda));
        }

        let layer_eval = random_linear_combination(&layer_evals, sumcheck_alpha);

        if sumcheck_eval != layer_eval {
            return Err(format!("GKR sumcheck evaluation mismatch in layer {layer}"));
        }

        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            channel.mix_felts(mask.columns().as_flattened());
        }

        let challenge = channel.draw_secure_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);

        for instance in sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(challenge));
        }
    }

    let claims_to_verify_by_instance = claims_to_verify_by_instance
        .into_iter()
        .map(Option::unwrap)
        .collect();

    Ok(GkrArtifact {
        ood_point,
        claims_to_verify_by_instance,
        n_variables_by_instance: (0..n_instances).map(instance_n_layers).collect(),
    })
}

pub fn partially_verify_batch_with_libra_masks(
    gate_by_instance: Vec<Gate>,
    proof: &GkrBatchProof,
    libra_masks: Option<(&[SecureField], &[SecureField])>,
    channel: &mut impl Channel,
) -> Result<(GkrArtifact, Vec<Vec<SecureField>>), String> {
    if libra_masks.is_none() {
        let artifact = partially_verify_batch(gate_by_instance, proof, channel)?;
        return Ok((artifact, Vec::new()));
    }
    let (libra_sums, libra_evals) = libra_masks.unwrap();
    let GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    } = proof;

    if layer_masks_by_instance.len() != output_claims_by_instance.len() {
        return Err("GKR batch proof instance lengths do not match".into());
    }
    let n_instances = layer_masks_by_instance.len();
    let instance_n_layers = |instance: usize| layer_masks_by_instance[instance].len();
    let n_layers = (0..n_instances).map(instance_n_layers).max().unwrap_or(0);
    if n_layers != sumcheck_proofs.len()
        || libra_sums.len() != n_layers
        || libra_evals.len() != n_layers
    {
        return Err("GKR batch proof sumcheck or Libra mask count mismatch".into());
    }
    if gate_by_instance.len() != n_instances {
        return Err("GKR batch proof gate count does not match instance count".into());
    }

    let mut ood_point = Vec::new();
    let mut sumcheck_ood_points_by_layer = Vec::with_capacity(n_layers);
    let mut claims_to_verify_by_instance: Vec<Option<Vec<SecureField>>> = vec![None; n_instances];

    for (layer, sumcheck_proof) in sumcheck_proofs.iter().enumerate() {
        let n_remaining_layers = n_layers - layer;
        for instance in 0..n_instances {
            if instance_n_layers(instance) == n_remaining_layers {
                claims_to_verify_by_instance[instance] =
                    Some(output_claims_by_instance[instance].clone());
            }
        }
        for claims_to_verify in claims_to_verify_by_instance.iter().flatten() {
            channel.mix_felts(claims_to_verify);
        }
        let sumcheck_alpha = channel.draw_secure_felt();
        let instance_lambda = channel.draw_secure_felt();
        channel.mix_felts(&[libra_sums[layer]]);
        let rho_libra = channel.draw_secure_felt();

        let mut sumcheck_claims = Vec::new();
        let mut sumcheck_instances = Vec::new();
        for (instance, claims_to_verify) in claims_to_verify_by_instance.iter().enumerate() {
            if let Some(claims_to_verify) = claims_to_verify {
                let n_unused_variables = n_layers - instance_n_layers(instance);
                let doubling_factor = BaseField::from_u32_unchecked(1u32 << n_unused_variables);
                let claim =
                    random_linear_combination(claims_to_verify, instance_lambda) * doubling_factor;
                sumcheck_claims.push(claim);
                sumcheck_instances.push(instance);
            }
        }
        let sumcheck_claim = random_linear_combination(&sumcheck_claims, sumcheck_alpha)
            + rho_libra * libra_sums[layer];
        let (sumcheck_ood_point, sumcheck_eval) =
            partially_verify(sumcheck_claim, sumcheck_proof, channel)?;
        channel.mix_felts(&[libra_evals[layer]]);

        let mut layer_evals = Vec::new();
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            let gate = &gate_by_instance[instance];
            let gate_output = gate.eval(mask)?;
            let eq_eval = eq(&ood_point[n_unused..], &sumcheck_ood_point[n_unused..]);
            layer_evals.push(eq_eval * random_linear_combination(&gate_output, instance_lambda));
        }
        let layer_eval = random_linear_combination(&layer_evals, sumcheck_alpha)
            + rho_libra * libra_evals[layer];
        if sumcheck_eval != layer_eval {
            return Err(format!("GKR sumcheck evaluation mismatch in layer {layer}"));
        }
        for &instance in &sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            channel.mix_felts(mask.columns().as_flattened());
        }
        sumcheck_ood_points_by_layer.push(sumcheck_ood_point.clone());
        let challenge = channel.draw_secure_felt();
        ood_point = sumcheck_ood_point;
        ood_point.push(challenge);
        for instance in sumcheck_instances {
            let n_unused = n_layers - instance_n_layers(instance);
            let mask = &layer_masks_by_instance[instance][layer - n_unused];
            claims_to_verify_by_instance[instance] = Some(mask.reduce_at_point(challenge));
        }
    }

    Ok((
        GkrArtifact {
            ood_point,
            claims_to_verify_by_instance: claims_to_verify_by_instance
                .into_iter()
                .map(Option::unwrap)
                .collect(),
            n_variables_by_instance: (0..n_instances).map(instance_n_layers).collect(),
        },
        sumcheck_ood_points_by_layer,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityProfile {
    Dev13,
    Hardened96,
    SmallProof96,
    FastProve96,
    LongQuery,
    Production100,
}

impl SecurityProfile {
    pub fn id(self) -> &'static str {
        match self {
            Self::Dev13 => "dev13",
            Self::Hardened96 => "hardened96",
            Self::SmallProof96 => "smallproof96",
            Self::FastProve96 => "fastprove96",
            Self::LongQuery => "longquery",
            Self::Production100 => "production100",
        }
    }

    pub fn from_id(id: &str) -> Result<Self, String> {
        match id {
            "dev13" => Ok(Self::Dev13),
            "hardened96" => Ok(Self::Hardened96),
            "smallproof96" | "smallproof" => Ok(Self::SmallProof96),
            "fastprove96" => Ok(Self::FastProve96),
            "longquery" | "conditional100" => Ok(Self::LongQuery),
            "production100" => Ok(Self::Production100),
            other => Err(format!("unknown Stage A security profile {other:?}")),
        }
    }
}

#[inline]
pub fn bytemuck_cast_i8(v: &[i8]) -> &[u8] {
    // SAFETY: `i8` and `u8` have identical size (1) and alignment (1).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

#[inline]
pub fn hash_i16_le_slice(h: &mut Blake2s256, v: &[i16]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `i16` is 2 bytes with no padding, and on little-endian targets
        // its in-memory representation matches `to_le_bytes()`.
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
        h.update(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &val in v {
            h.update(val.to_le_bytes());
        }
    }
}

pub fn make_channel(digest: &[u64; 4]) -> Blake2sChannel {
    let mut ch = Blake2sChannel::default();
    for &d in digest {
        ch.mix_u64(d);
    }
    ch
}

pub fn eq_evals(point: &[SecureField]) -> Vec<SecureField> {
    let mut table = vec![SecureField::zero(); 1usize << point.len()];
    table[0] = SecureField::one();
    let mut half = 1usize;
    for &coordinate in point.iter().rev() {
        let (lo, hi) = table[..2 * half].split_at_mut(half);
        for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
            let v = *l;
            let prod = v * coordinate;
            *l = v - prod;
            *h = prod;
        }
        half *= 2;
    }
    table
}

pub fn dot(values: &[SecureField], weights: &[SecureField]) -> SecureField {
    let length = values.len().min(weights.len());
    values[..length]
        .iter()
        .zip(&weights[..length])
        .map(|(&value, &weight)| value * weight)
        .sum()
}

pub fn mle_at(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    dot(evals, &eq_evals(point))
}

#[inline]
pub fn i8_to_m31(v: i8) -> BaseField {
    if v >= 0 {
        BaseField::from_u32_unchecked(v as u32)
    } else {
        BaseField::from_u32_unchecked(P - (-(v as i32)) as u32)
    }
}

#[inline]
pub fn i64_mod_p(v: i64) -> BaseField {
    let p = P as i64;
    BaseField::from_u32_unchecked(((v % p + p) % p) as u32)
}

pub fn eq_evals_i64(point: &[SecureField]) -> Vec<[i64; 4]> {
    eq_evals(point)
        .into_iter()
        .map(|w| {
            let c = w.to_m31_array();
            [c[0].0 as i64, c[1].0 as i64, c[2].0 as i64, c[3].0 as i64]
        })
        .collect()
}

pub fn acc_to_secure(a: [i64; 4]) -> SecureField {
    SecureField::from_m31_array([
        i64_mod_p(a[0]),
        i64_mod_p(a[1]),
        i64_mod_p(a[2]),
        i64_mod_p(a[3]),
    ])
}

pub fn restrict_rows_i8_fast(m: &[Vec<i8>], point: &[SecureField]) -> Vec<SecureField> {
    if m.is_empty() {
        return Vec::new();
    }
    let n_cols = m[0].len();
    let eq_w = eq_evals_i64(point);
    let mut acc = vec![[0i64; 4]; n_cols];
    for (row, m_row) in m.iter().enumerate() {
        let w = eq_w[row];
        for (k, &v) in m_row.iter().enumerate() {
            let v = v as i64;
            if v == 0 {
                continue;
            }
            let a = &mut acc[k];
            a[0] += w[0] * v;
            a[1] += w[1] * v;
            a[2] += w[2] * v;
            a[3] += w[3] * v;
        }
    }
    acc.into_iter().map(acc_to_secure).collect()
}

pub fn restrict_cols_i8_fast(m: &[Vec<i8>], point: &[SecureField]) -> Vec<SecureField> {
    if m.is_empty() {
        return Vec::new();
    }
    let eq_w = eq_evals_i64(point);
    m.iter()
        .map(|m_row| {
            let mut a = [0i64; 4];
            for (w, &v) in eq_w.iter().zip(m_row) {
                let v = i64::from(v);
                a[0] += w[0] * v;
                a[1] += w[1] * v;
                a[2] += w[2] * v;
                a[3] += w[3] * v;
            }
            acc_to_secure(a)
        })
        .collect()
}

#[derive(Clone)]
pub struct ReceiptReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ReceiptReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
            return Err(format!(
                "receipt exceeds the {} byte wire limit",
                MAX_WIRE_RECEIPT_BYTES
            ));
        }
        Ok(Self { bytes, cursor: 0 })
    }

    fn take(&mut self, len: usize, label: &str) -> Result<&'a [u8], String> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| format!("{label} length overflow"))?;
        let value = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| format!("truncated {label}"))?;
        self.cursor = end;
        Ok(value)
    }

    pub fn expect_raw(&mut self, expected: &[u8], label: &str) -> Result<(), String> {
        if self.take(expected.len(), label)? != expected {
            return Err(format!("invalid {label}"));
        }
        Ok(())
    }

    pub fn read_u32(&mut self, label: &str) -> Result<u32, String> {
        let bytes: [u8; 4] = self
            .take(4, label)?
            .try_into()
            .map_err(|_| format!("invalid {label}"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn read_u32_byte(&mut self, label: &str) -> Result<u8, String> {
        Ok(self.take(1, label)?[0])
    }

    pub fn read_u64(&mut self, label: &str) -> Result<u64, String> {
        let bytes: [u8; 8] = self
            .take(8, label)?
            .try_into()
            .map_err(|_| format!("invalid {label}"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn read_len(&mut self, max: usize, label: &str) -> Result<usize, String> {
        let bytes: [u8; 8] = self
            .take(8, label)?
            .try_into()
            .map_err(|_| format!("invalid {label}"))?;
        let value = u64::from_le_bytes(bytes);
        let value = usize::try_from(value).map_err(|_| format!("{label} exceeds usize"))?;
        if value > max {
            return Err(format!("{label} exceeds the {max} element limit"));
        }
        Ok(value)
    }

    pub fn read_bytes(&mut self, max: usize, label: &str) -> Result<&'a [u8], String> {
        let len = self.read_len(max, &format!("{label} length"))?;
        self.take(len, label)
    }

    pub fn read_secure_field(&mut self, label: &str) -> Result<SecureField, String> {
        let mut coordinates = [BaseField::zero(); 4];
        for (index, coordinate) in coordinates.iter_mut().enumerate() {
            let raw = self.read_u32(&format!("{label} coordinate {index}"))?;
            if raw >= P {
                return Err(format!("non-canonical {label} coordinate {index}"));
            }
            *coordinate = BaseField::from_u32_unchecked(raw);
        }
        Ok(SecureField::from_m31_array(coordinates))
    }

    pub fn read_secure_fields_exact(
        &mut self,
        expected: usize,
        label: &str,
    ) -> Result<Vec<SecureField>, String> {
        let len = self.read_len(expected, &format!("{label} count"))?;
        if len != expected {
            return Err(format!(
                "{label} count mismatch: expected {expected}, got {len}"
            ));
        }
        let encoded_len = len
            .checked_mul(SECURE_FIELD_WIRE_BYTES)
            .ok_or_else(|| format!("{label} byte length overflow"))?;
        if encoded_len > self.bytes.len().saturating_sub(self.cursor) {
            return Err(format!("truncated {label}"));
        }
        (0..len)
            .map(|index| self.read_secure_field(&format!("{label} {index}")))
            .collect()
    }

    pub fn read_sumcheck(&mut self, label: &str) -> Result<SumcheckProof, String> {
        let round_count =
            self.read_len(MAX_WIRE_SUMCHECK_ROUNDS, &format!("{label} round count"))?;
        let mut round_polys = Vec::with_capacity(round_count);
        for round in 0..round_count {
            let coefficient_count = self.read_len(
                MAX_DEGREE + 1,
                &format!("{label} round {round} coefficient count"),
            )?;
            let mut coefficients = Vec::with_capacity(coefficient_count);
            for coefficient in 0..coefficient_count {
                coefficients.push(self.read_secure_field(&format!(
                    "{label} round {round} coefficient {coefficient}"
                ))?);
            }
            if coefficients.last().is_some_and(Zero::is_zero) {
                return Err(format!("non-canonical {label} round {round} polynomial"));
            }
            round_polys.push(UnivariatePoly::new(coefficients));
        }
        Ok(SumcheckProof { round_polys })
    }

    pub fn finish(self) -> Result<(), String> {
        if self.cursor != self.bytes.len() {
            return Err("trailing bytes after receipt".into());
        }
        Ok(())
    }
}

pub fn put_u64(out: &mut Vec<u8>, value: usize) -> Result<(), String> {
    let value = u64::try_from(value).map_err(|_| "receipt length exceeds u64".to_string())?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

pub fn put_raw_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn i64_wire_word(value: i64) -> u64 {
    u64::from_le_bytes(value.to_le_bytes())
}

pub fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&i64_wire_word(value).to_le_bytes());
}

pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), String> {
    put_u64(out, bytes.len())?;
    out.extend_from_slice(bytes);
    Ok(())
}

pub fn put_secure_field(out: &mut Vec<u8>, value: SecureField) {
    for coordinate in value.to_m31_array() {
        out.extend_from_slice(&coordinate.0.to_le_bytes());
    }
}

pub fn put_secure_fields(out: &mut Vec<u8>, values: &[SecureField]) -> Result<(), String> {
    put_u64(out, values.len())?;
    for &value in values {
        put_secure_field(out, value);
    }
    Ok(())
}

pub fn put_sumcheck(out: &mut Vec<u8>, proof: &SumcheckProof) -> Result<(), String> {
    put_u64(out, proof.round_polys.len())?;
    for polynomial in &proof.round_polys {
        put_u64(out, polynomial.len())?;
        for &coefficient in polynomial.iter() {
            put_secure_field(out, coefficient);
        }
    }
    Ok(())
}

pub const BEACON_DOMAIN: &[u8] = b"stwo-ml:stage-a-beacon:v4";
pub const BEACON_BUNDLE_V3_MAGIC: &[u8; 8] = b"STWOBCN3";
pub const BEACON_BUNDLE_V3_VERSION: u32 = 3;
pub const BEACON_BUNDLE_MAGIC: &[u8; 8] = b"STWOBCN4";
pub const BEACON_BUNDLE_VERSION: u32 = 4;
const MAX_BUNDLE_LABEL_BYTES: usize = 128;
const MAX_BUNDLE_ROWS: usize = 1 << 20;
const MAX_BUNDLE_SIBLINGS: usize = 64;

fn beacon_hasher(tag: &[u8]) -> Blake2s256 {
    let mut h = Blake2s256::new();
    h.update(BEACON_DOMAIN);
    h.update([tag.len() as u8]);
    h.update(tag);
    h
}

fn update_rows(h: &mut Blake2s256, rows: &[Vec<i16>]) {
    h.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        h.update((row.len() as u64).to_le_bytes());
        hash_i16_le_slice(h, row);
    }
}

pub fn state_digest(rows: &[Vec<i16>]) -> [u8; 32] {
    let mut h = beacon_hasher(b"state");
    update_rows(&mut h, rows);
    h.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerDigests {
    pub h_in: [u8; 32],
    pub a: [u8; 32],
    pub x: [u8; 32],
    pub out: [u8; 32],
    pub cache: [u8; 32],
}

impl LayerDigests {
    pub fn of(
        h_in: &[Vec<i16>],
        a: &[Vec<i16>],
        x: &[Vec<i16>],
        out: &[Vec<i16>],
        cache: [u8; 32],
    ) -> Self {
        Self {
            h_in: state_digest(h_in),
            a: state_digest(a),
            x: state_digest(x),
            out: state_digest(out),
            cache,
        }
    }
}

pub fn activation_leaf(layer: u32, digests: &LayerDigests) -> [u8; 32] {
    let mut h = beacon_hasher(b"leaf");
    h.update(layer.to_le_bytes());
    h.update(digests.h_in);
    h.update(digests.a);
    h.update(digests.x);
    h.update(digests.out);
    h.update(digests.cache);
    h.finalize().into()
}

pub fn public_transition(a: &[Vec<i16>], out: &[Vec<i16>]) -> Result<Vec<Vec<i16>>, String> {
    residual_out(a, out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryAnchors {
    pub embedding_commitment: [u8; 32],
    pub lm_head_commitment: [u8; 32],
    pub gamma_final: Vec<i16>,
}

pub fn layer_anchor_with_boundaries(
    stage_a_anchor: &[u8; 32],
    attention_commitment: &[u8; 32],
    attention_policy_id: &str,
    gamma1: &[i16],
    gamma2: &[i16],
    boundaries: Option<&BoundaryAnchors>,
) -> [u8; 32] {
    let mut h = beacon_hasher(b"layer-anchor");
    h.update(stage_a_anchor);
    h.update(attention_commitment);
    h.update((attention_policy_id.len() as u64).to_le_bytes());
    h.update(attention_policy_id.as_bytes());
    for gamma in [gamma1, gamma2] {
        h.update((gamma.len() as u64).to_le_bytes());
        for &g in gamma {
            h.update(g.to_le_bytes());
        }
    }
    if let Some(b) = boundaries {
        h.update([1u8]);
        h.update(b.embedding_commitment);
        h.update(b.lm_head_commitment);
        h.update((b.gamma_final.len() as u64).to_le_bytes());
        for &g in &b.gamma_final {
            h.update(g.to_le_bytes());
        }
    }
    h.finalize().into()
}

pub fn layer_anchor(
    stage_a_anchor: &[u8; 32],
    attention_commitment: &[u8; 32],
    attention_policy_id: &str,
    gamma1: &[i16],
    gamma2: &[i16],
) -> [u8; 32] {
    layer_anchor_with_boundaries(
        stage_a_anchor,
        attention_commitment,
        attention_policy_id,
        gamma1,
        gamma2,
        None,
    )
}

fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = beacon_hasher(b"node");
    h.update(left);
    h.update(right);
    h.finalize().into()
}

pub fn activation_depth(layers: u32) -> u32 {
    (layers.max(1) as u64).next_power_of_two().trailing_zeros()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuityOpening {
    pub a_digest: [u8; 32],
    pub x_digest: [u8; 32],
    pub output_digest: [u8; 32],
    pub cache_digest: [u8; 32],
    pub path: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalLeafOpening {
    pub h_in_digest: [u8; 32],
    pub x_digest: [u8; 32],
    pub cache_digest: [u8; 32],
    pub last_a: Vec<Vec<i16>>,
    pub last_out: Vec<Vec<i16>>,
    pub path: Vec<[u8; 32]>,
}

pub fn verify_activation_opening(
    root: &[u8; 32],
    layers: u32,
    layer: u32,
    leaf: &[u8; 32],
    path: &[[u8; 32]],
) -> bool {
    let depth = activation_depth(layers) as usize;
    if layer >= layers.max(1) || path.len() != depth {
        return false;
    }
    let mut acc = *leaf;
    let mut index = layer;
    for sibling in path {
        acc = if index & 1 == 0 {
            node_hash(&acc, sibling)
        } else {
            node_hash(sibling, &acc)
        };
        index >>= 1;
    }
    acc == *root
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconPolicy {
    pub layers: u32,
    pub sample_den: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarknetBeacon {
    pub block_number: u64,
    pub block_hash: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeaconChallenge {
    pub beacon: StarknetBeacon,
    pub response_root: [u8; 32],
    pub policy: BeaconPolicy,
    pub challenged: bool,
    pub layer: u32,
}

pub fn derive_challenge(
    beacon: StarknetBeacon,
    response_root: [u8; 32],
    policy: BeaconPolicy,
) -> Result<BeaconChallenge, String> {
    if policy.layers == 0 {
        return Err("beacon policy needs at least one layer".into());
    }
    if policy.sample_den == 0 {
        return Err("beacon policy sample_den must be positive".into());
    }
    let mut h = beacon_hasher(b"challenge");
    h.update(beacon.block_number.to_le_bytes());
    h.update(beacon.block_hash);
    h.update(response_root);
    h.update(policy.layers.to_le_bytes());
    h.update(policy.sample_den.to_le_bytes());
    let digest: [u8; 32] = h.finalize().into();
    let draw_sample = u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"));
    let draw_layer = u64::from_le_bytes(digest[8..16].try_into().expect("8 bytes"));
    let challenged = draw_sample % policy.sample_den == 0;
    let layer = (draw_layer % u64::from(policy.layers)) as u32;
    Ok(BeaconChallenge {
        beacon,
        response_root,
        policy,
        challenged,
        layer,
    })
}

pub fn beacon_link(challenge: &BeaconChallenge) -> StageAChainLink {
    let mut h = beacon_hasher(b"link");
    h.update(challenge.beacon.block_number.to_le_bytes());
    h.update(challenge.beacon.block_hash);
    h.update(challenge.response_root);
    h.update(challenge.policy.layers.to_le_bytes());
    h.update(challenge.policy.sample_den.to_le_bytes());
    h.update(challenge.layer.to_le_bytes());
    StageAChainLink {
        layer: challenge.layer,
        prev_receipt_digest: h.finalize().into(),
    }
}

fn layer_states(
    steps: &[PublicStep],
    recomputed: &[RecomputedStep],
) -> (Vec<Vec<i16>>, Vec<Vec<i16>>, Vec<Vec<i16>>) {
    (
        steps.iter().map(|s| s.h_in.clone()).collect(),
        recomputed.iter().map(|r| r.a.clone()).collect(),
        recomputed.iter().map(|r| r.x.clone()).collect(),
    )
}

#[derive(Clone)]
pub struct ForwardPassBoundaryBundle {
    pub boundary_anchors: BoundaryAnchors,
    pub embedding: EmbeddingStatement,
    pub embedding_proof: EmbeddingProof,
    pub leaf0_opening: ContinuityOpening,
    pub lm_head: LmHeadStatement,
    pub lm_head_proof: LmHeadProof,
    pub terminal_opening: TerminalLeafOpening,
}

#[derive(Clone)]
pub struct SpotCheckVerificationBundle {
    pub profile: SecurityProfile,
    pub policy: StageANumericPolicy,
    pub attention_policy: AttentionPolicy,
    pub anchor: [u8; 32],
    pub beacon: StarknetBeacon,
    pub response_root: [u8; 32],
    pub beacon_policy: BeaconPolicy,
    pub path: Vec<[u8; 32]>,
    pub continuity: Option<ContinuityOpening>,
    pub attention: StageBStatement,
    pub attention_receipt: Vec<u8>,
    pub statement: StageANonlinearBatchStatement,
    pub context: StageARegisteredContext,
    pub receipt: Vec<u8>,
    pub boundaries: Option<ForwardPassBoundaryBundle>,
}

pub fn stage_a_policy_from_id(id: &str) -> Result<StageANumericPolicy, String> {
    bcn_policy_from_id(id)
}

fn bcn_policy_from_id(id: &str) -> Result<StageANumericPolicy, String> {
    [
        StageANumericPolicy::v1(),
        StageANumericPolicy::deep_prove_comparison_v1(),
        StageANumericPolicy::trained_i12_q7_v1(),
        StageANumericPolicy::trained_i12_q7_half_up_v1(),
    ]
    .into_iter()
    .find(|policy| policy.id() == id)
    .ok_or_else(|| format!("unknown Stage A numeric policy id {id:?}"))
}

fn put_bcn_label(out: &mut Vec<u8>, label: &str) -> Result<(), String> {
    if label.len() > MAX_BUNDLE_LABEL_BYTES {
        return Err(format!(
            "label {label:?} exceeds {MAX_BUNDLE_LABEL_BYTES} bytes"
        ));
    }
    put_bytes(out, label.as_bytes())
}

fn put_bcn_digests(out: &mut Vec<u8>, digests: &[[u8; 32]]) -> Result<(), String> {
    put_u64(out, digests.len())?;
    for d in digests {
        out.extend_from_slice(d);
    }
    Ok(())
}

fn put_bcn_shape(out: &mut Vec<u8>, shape: StageAShape) -> Result<(), String> {
    for value in [
        shape.batch,
        shape.dimension,
        shape.expert_width,
        shape.experts,
        shape.top_k,
        shape.shared_factor,
    ] {
        put_u64(out, value)?;
    }
    Ok(())
}

fn put_bcn_i16_vec(out: &mut Vec<u8>, values: &[i16]) -> Result<(), String> {
    put_u64(out, values.len())?;
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

fn put_bcn_i32_vec(out: &mut Vec<u8>, values: &[i32]) -> Result<(), String> {
    put_u64(out, values.len())?;
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

fn put_bcn_i8_rows(out: &mut Vec<u8>, rows: &[Vec<i8>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_u64(out, row.len())?;
        for &v in row {
            out.push(v as u8);
        }
    }
    Ok(())
}

fn put_bcn_i16_rows(out: &mut Vec<u8>, rows: &[Vec<i16>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_bcn_i16_vec(out, row)?;
    }
    Ok(())
}

fn put_bcn_i32_rows(out: &mut Vec<u8>, rows: &[Vec<i32>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_bcn_i32_vec(out, row)?;
    }
    Ok(())
}

fn put_bcn_i64_rows(out: &mut Vec<u8>, rows: &[Vec<i64>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_u64(out, row.len())?;
        for &v in row {
            put_i64(out, v);
        }
    }
    Ok(())
}

fn put_bcn_u16_rows(out: &mut Vec<u8>, rows: &[Vec<u16>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_u64(out, row.len())?;
        for &v in row {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}

fn put_bcn_usize_rows(out: &mut Vec<u8>, rows: &[Vec<usize>]) -> Result<(), String> {
    put_u64(out, rows.len())?;
    for row in rows {
        put_u64(out, row.len())?;
        for &v in row {
            put_u64(out, v)?;
        }
    }
    Ok(())
}

fn put_bcn_continuity_opening(out: &mut Vec<u8>, c: &ContinuityOpening) -> Result<(), String> {
    out.extend_from_slice(&c.a_digest);
    out.extend_from_slice(&c.x_digest);
    out.extend_from_slice(&c.output_digest);
    out.extend_from_slice(&c.cache_digest);
    put_bcn_digests(out, &c.path)
}

fn read_bcn_continuity_opening(
    reader: &mut ReceiptReader<'_>,
    label: &str,
) -> Result<ContinuityOpening, String> {
    Ok(ContinuityOpening {
        a_digest: read_bcn_digest(reader, &format!("{label} a_digest"))?,
        x_digest: read_bcn_digest(reader, &format!("{label} x_digest"))?,
        output_digest: read_bcn_digest(reader, &format!("{label} output_digest"))?,
        cache_digest: read_bcn_digest(reader, &format!("{label} cache_digest"))?,
        path: read_bcn_digests(reader, MAX_BUNDLE_SIBLINGS, &format!("{label} path"))?,
    })
}

fn put_bcn_boundaries(out: &mut Vec<u8>, b: &ForwardPassBoundaryBundle) -> Result<(), String> {
    out.extend_from_slice(&b.boundary_anchors.embedding_commitment);
    out.extend_from_slice(&b.boundary_anchors.lm_head_commitment);
    put_bcn_i16_vec(out, &b.boundary_anchors.gamma_final)?;

    put_u64(out, b.embedding.vocab_size)?;
    put_u64(out, b.embedding.dimension)?;
    put_u32(out, b.embedding.embedding_shift);
    out.extend_from_slice(&b.embedding.embedding_commitment);
    put_u64(out, b.embedding.token_ids.len())?;
    for &id in &b.embedding.token_ids {
        put_u32(out, id);
    }
    put_bcn_i16_rows(out, &b.embedding.h_out)?;
    out.extend_from_slice(&b.embedding.binding);
    put_projection_proof(out, &b.embedding_proof.projection)?;
    put_bytes(out, &b.embedding_proof.leaf_proof)?;
    put_bcn_continuity_opening(out, &b.leaf0_opening)?;

    put_u64(out, b.lm_head.dimension)?;
    put_u64(out, b.lm_head.vocab_size)?;
    out.extend_from_slice(&b.lm_head.norm_fraction_bits.to_le_bytes());
    out.extend_from_slice(&b.lm_head.lm_head_commitment);
    put_bcn_i16_vec(out, &b.lm_head.gamma_final)?;
    put_bcn_i16_rows(out, &b.lm_head.h_in)?;
    put_bcn_i32_rows(out, &b.lm_head.logits_raw)?;
    out.extend_from_slice(&b.lm_head.binding);
    put_projection_proof(out, &b.lm_head_proof.projection)?;
    put_bytes(out, &b.lm_head_proof.leaf_proof)?;

    out.extend_from_slice(&b.terminal_opening.h_in_digest);
    out.extend_from_slice(&b.terminal_opening.x_digest);
    out.extend_from_slice(&b.terminal_opening.cache_digest);
    put_bcn_i16_rows(out, &b.terminal_opening.last_a)?;
    put_bcn_i16_rows(out, &b.terminal_opening.last_out)?;
    put_bcn_digests(out, &b.terminal_opening.path)?;
    Ok(())
}

fn read_bcn_boundaries(
    reader: &mut ReceiptReader<'_>,
) -> Result<ForwardPassBoundaryBundle, String> {
    let embedding_commitment = read_bcn_digest(reader, "boundary embedding_commitment")?;
    let lm_head_commitment = read_bcn_digest(reader, "boundary lm_head_commitment")?;
    let gamma_final = read_bcn_vec(reader, "boundary gamma_final", read_bcn_i16)?;
    let boundary_anchors = BoundaryAnchors {
        embedding_commitment,
        lm_head_commitment,
        gamma_final,
    };

    let emb_vocab_size = reader.read_len(MAX_BUNDLE_ROWS, "embedding vocab_size")?;
    let emb_dimension = reader.read_len(MAX_BUNDLE_ROWS, "embedding dimension")?;
    let embedding_shift = reader.read_u32("embedding_shift")?;
    let emb_stmt_comm = read_bcn_digest(reader, "embedding statement commitment")?;
    let token_ids = read_bcn_vec(reader, "embedding token_ids", |r, l| r.read_u32(l))?;
    let h_out = read_bcn_rows(reader, "embedding h_out", read_bcn_i16)?;
    let emb_binding = read_bcn_digest(reader, "embedding binding")?;
    let embedding = EmbeddingStatement {
        vocab_size: emb_vocab_size,
        dimension: emb_dimension,
        embedding_shift,
        embedding_commitment: emb_stmt_comm,
        token_ids,
        h_out,
        binding: emb_binding,
    };
    let emb_proj = read_projection_proof(reader, 1, "embedding projection")?;
    let emb_leaf_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "embedding leaf proof")?
        .to_vec();
    let embedding_proof = EmbeddingProof {
        projection: emb_proj,
        leaf_proof: emb_leaf_proof,
    };
    let leaf0_opening = read_bcn_continuity_opening(reader, "leaf0_opening")?;

    let lm_dimension = reader.read_len(MAX_BUNDLE_ROWS, "lm_head dimension")?;
    let lm_vocab_size = reader.read_len(MAX_BUNDLE_ROWS, "lm_head vocab_size")?;
    let norm_fraction_bits = read_bcn_i32(reader, "lm_head norm_fraction_bits")?;
    let lm_stmt_comm = read_bcn_digest(reader, "lm_head statement commitment")?;
    let lm_gamma_final = read_bcn_vec(reader, "lm_head gamma_final", read_bcn_i16)?;
    let h_in = read_bcn_rows(reader, "lm_head h_in", read_bcn_i16)?;
    let logits_raw = read_bcn_rows(reader, "lm_head logits_raw", read_bcn_i32)?;
    let lm_binding = read_bcn_digest(reader, "lm_head binding")?;
    let lm_head = LmHeadStatement {
        dimension: lm_dimension,
        vocab_size: lm_vocab_size,
        norm_fraction_bits,
        lm_head_commitment: lm_stmt_comm,
        gamma_final: lm_gamma_final,
        h_in,
        logits_raw,
        binding: lm_binding,
    };
    let lm_proj = read_projection_proof(reader, 1, "lm_head projection")?;
    let lm_leaf_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "lm_head leaf proof")?
        .to_vec();
    let lm_head_proof = LmHeadProof {
        projection: lm_proj,
        leaf_proof: lm_leaf_proof,
    };

    let h_in_digest = read_bcn_digest(reader, "terminal_opening h_in_digest")?;
    let x_digest = read_bcn_digest(reader, "terminal_opening x_digest")?;
    let cache_digest = read_bcn_digest(reader, "terminal_opening cache_digest")?;
    let last_a = read_bcn_rows(reader, "terminal_opening last_a", read_bcn_i16)?;
    let last_out = read_bcn_rows(reader, "terminal_opening last_out", read_bcn_i16)?;
    let path = read_bcn_digests(reader, MAX_BUNDLE_SIBLINGS, "terminal_opening path")?;
    let terminal_opening = TerminalLeafOpening {
        h_in_digest,
        x_digest,
        cache_digest,
        last_a,
        last_out,
        path,
    };

    Ok(ForwardPassBoundaryBundle {
        boundary_anchors,
        embedding,
        embedding_proof,
        leaf0_opening,
        lm_head,
        lm_head_proof,
        terminal_opening,
    })
}

fn spot_check_verification_bundle_bytes_with_version(
    bundle: &SpotCheckVerificationBundle,
    version: u32,
) -> Result<Vec<u8>, String> {
    if bundle.attention.partial_rows.len() != bundle.attention.tokens.len() {
        return Err("Stage B statement partial_rows length mismatch".into());
    }
    if version == BEACON_BUNDLE_V3_VERSION && bundle.boundaries.is_some() {
        return Err("STWOBCN3 v3 bundle cannot carry forward-pass boundaries".into());
    }
    let mut out = Vec::new();
    match version {
        BEACON_BUNDLE_V3_VERSION => {
            out.extend_from_slice(BEACON_BUNDLE_V3_MAGIC);
            put_u32(&mut out, BEACON_BUNDLE_V3_VERSION);
        }
        BEACON_BUNDLE_VERSION => {
            out.extend_from_slice(BEACON_BUNDLE_MAGIC);
            put_u32(&mut out, BEACON_BUNDLE_VERSION);
        }
        other => return Err(format!("unsupported beacon bundle version {other}")),
    }
    put_bcn_label(&mut out, bundle.profile.id())?;
    put_bcn_label(&mut out, bundle.policy.id())?;
    put_bcn_label(&mut out, &bundle.attention_policy.id())?;
    out.extend_from_slice(&bundle.anchor);
    put_raw_u64(&mut out, bundle.beacon.block_number);
    out.extend_from_slice(&bundle.beacon.block_hash);
    out.extend_from_slice(&bundle.response_root);
    put_u32(&mut out, bundle.beacon_policy.layers);
    put_raw_u64(&mut out, bundle.beacon_policy.sample_den);
    put_bcn_digests(&mut out, &bundle.path)?;
    match &bundle.continuity {
        None => out.push(0),
        Some(c) => {
            out.push(1);
            put_bcn_continuity_opening(&mut out, c)?;
        }
    }

    let a = &bundle.attention;
    for value in [
        a.shape.dimension,
        a.shape.heads,
        a.shape.kv_heads,
        a.shape.head_dim,
        a.layout.chunk,
    ] {
        put_u64(&mut out, value)?;
    }
    put_bcn_label(&mut out, &a.policy_id)?;
    out.extend_from_slice(&a.attention_commitment);
    out.extend_from_slice(&a.binding);
    put_bcn_i16_vec(&mut out, &a.gamma1)?;
    put_bcn_i16_vec(&mut out, &a.gamma2)?;
    put_bcn_digests(&mut out, &a.chunk_commitments)?;
    put_u64(&mut out, a.tokens.len())?;
    for (token, (k_rows, v_rows)) in a.tokens.iter().zip(&a.partial_rows) {
        put_raw_u64(&mut out, token.step.position);
        put_bcn_i16_vec(&mut out, &token.step.h_in)?;
        put_bcn_i32_vec(&mut out, &token.step.q_raw)?;
        put_bcn_i32_vec(&mut out, &token.step.k_raw)?;
        put_bcn_i32_vec(&mut out, &token.step.v_raw)?;
        put_bcn_i32_rows(&mut out, &token.step.scores)?;
        put_bcn_i32_vec(&mut out, &token.step.ctx_raw)?;
        put_bcn_i32_vec(&mut out, &token.step.o_raw)?;
        if token.step.q_raw.is_empty() {
            put_bcn_i16_vec(&mut out, &token.step.a)?;
            put_bcn_i16_vec(&mut out, &token.step.x)?;
        }
        put_u64(&mut out, token.chunk_ids.len())?;
        for &cid in &token.chunk_ids {
            put_u64(&mut out, cid)?;
        }
        out.extend_from_slice(&token.cache_commitment_before);
        out.extend_from_slice(&token.cache_commitment_after);
        put_bcn_i8_rows(&mut out, k_rows)?;
        put_bcn_i8_rows(&mut out, v_rows)?;
    }
    put_bytes(&mut out, &bundle.attention_receipt)?;

    let s = &bundle.statement;
    put_bcn_shape(&mut out, s.shape)?;
    put_bcn_label(&mut out, &s.policy_id)?;
    for d in [
        &s.policy_digest,
        &s.model_digest,
        &s.model_registry_manifest_digest,
        &s.table_digest,
        &s.input_digest,
    ] {
        out.extend_from_slice(d);
    }
    put_bcn_i64_rows(&mut out, &s.router_outputs_raw)?;
    put_bcn_usize_rows(&mut out, &s.routes)?;
    put_bcn_u16_rows(&mut out, &s.route_weights_q15)?;
    put_bcn_i16_rows(&mut out, &s.output_q8)?;
    put_u64(&mut out, s.wide_log_columns)?;

    let c = &bundle.context;
    put_bcn_shape(&mut out, c.shape)?;
    for d in [
        &c.model_digest,
        &c.manifest_digest,
        &c.scheme_digest,
        &c.router_commitment,
        &c.shared_commitment,
        &c.registered_expert_root,
    ] {
        out.extend_from_slice(d);
    }
    put_u64(&mut out, c.active_expert_openings.len())?;
    for opening in &c.active_expert_openings {
        put_u64(&mut out, opening.expert_id)?;
        out.extend_from_slice(&opening.slot_roles);
        out.extend_from_slice(&opening.group_commitment);
        put_bcn_digests(&mut out, &opening.siblings)?;
    }
    put_bytes(&mut out, &bundle.receipt)?;
    if version == BEACON_BUNDLE_VERSION {
        match &bundle.boundaries {
            None => out.push(0),
            Some(b) => {
                out.push(1);
                put_bcn_boundaries(&mut out, b)?;
            }
        }
    }
    Ok(out)
}

pub fn spot_check_verification_bundle_bytes(
    bundle: &SpotCheckVerificationBundle,
) -> Result<Vec<u8>, String> {
    spot_check_verification_bundle_bytes_with_version(bundle, BEACON_BUNDLE_VERSION)
}

pub fn spot_check_verification_bundle_bytes_matching_header(
    bundle: &SpotCheckVerificationBundle,
    reference_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    if reference_bytes.len() >= 12 && &reference_bytes[..8] == BEACON_BUNDLE_V3_MAGIC {
        spot_check_verification_bundle_bytes_with_version(bundle, BEACON_BUNDLE_V3_VERSION)
    } else {
        spot_check_verification_bundle_bytes_with_version(bundle, BEACON_BUNDLE_VERSION)
    }
}

fn read_bcn_label<'a>(reader: &mut ReceiptReader<'a>, label: &str) -> Result<&'a str, String> {
    let bytes = reader.read_bytes(MAX_BUNDLE_LABEL_BYTES, label)?;
    std::str::from_utf8(bytes).map_err(|_| format!("{label} is not UTF-8"))
}

fn read_bcn_digest(reader: &mut ReceiptReader<'_>, label: &str) -> Result<[u8; 32], String> {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = reader.read_u32_byte(&format!("{label} byte {i}"))?;
    }
    Ok(out)
}

fn read_bcn_digests(
    reader: &mut ReceiptReader<'_>,
    max: usize,
    label: &str,
) -> Result<Vec<[u8; 32]>, String> {
    let count = reader.read_len(max, &format!("{label} count"))?;
    (0..count)
        .map(|i| read_bcn_digest(reader, &format!("{label} {i}")))
        .collect()
}

fn read_bcn_shape(reader: &mut ReceiptReader<'_>, label: &str) -> Result<StageAShape, String> {
    let mut values = [0usize; 6];
    for (i, v) in values.iter_mut().enumerate() {
        *v = reader.read_len(MAX_BUNDLE_ROWS, &format!("{label} field {i}"))?;
    }
    Ok(StageAShape {
        batch: values[0],
        dimension: values[1],
        expert_width: values[2],
        experts: values[3],
        top_k: values[4],
        shared_factor: values[5],
    })
}

fn read_bcn_i16(reader: &mut ReceiptReader<'_>, label: &str) -> Result<i16, String> {
    let lo = reader.read_u32_byte(label)?;
    let hi = reader.read_u32_byte(label)?;
    Ok(i16::from_le_bytes([lo, hi]))
}

fn read_bcn_u16(reader: &mut ReceiptReader<'_>, label: &str) -> Result<u16, String> {
    let lo = reader.read_u32_byte(label)?;
    let hi = reader.read_u32_byte(label)?;
    Ok(u16::from_le_bytes([lo, hi]))
}

fn read_bcn_i32(reader: &mut ReceiptReader<'_>, label: &str) -> Result<i32, String> {
    let raw = reader.read_u32(label)?;
    Ok(i32::from_le_bytes(raw.to_le_bytes()))
}

fn read_bcn_vec<T>(
    reader: &mut ReceiptReader<'_>,
    label: &str,
    mut read_one: impl FnMut(&mut ReceiptReader<'_>, &str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let len = reader.read_len(MAX_BUNDLE_ROWS, &format!("{label} length"))?;
    (0..len).map(|_| read_one(reader, label)).collect()
}

fn read_bcn_rows<T>(
    reader: &mut ReceiptReader<'_>,
    label: &str,
    mut read_one: impl FnMut(&mut ReceiptReader<'_>, &str) -> Result<T, String>,
) -> Result<Vec<Vec<T>>, String> {
    let rows = reader.read_len(MAX_BUNDLE_ROWS, &format!("{label} row count"))?;
    (0..rows)
        .map(|r| {
            let len = reader.read_len(MAX_BUNDLE_ROWS, &format!("{label} row {r} length"))?;
            (0..len).map(|_| read_one(reader, label)).collect()
        })
        .collect()
}

pub fn decode_spot_check_verification_bundle(
    bytes: &[u8],
) -> Result<SpotCheckVerificationBundle, String> {
    let mut reader = ReceiptReader::new(bytes)?;
    let is_v3 = bytes.len() >= 8 && &bytes[..8] == BEACON_BUNDLE_V3_MAGIC;
    if is_v3 {
        reader.expect_raw(BEACON_BUNDLE_V3_MAGIC, "beacon bundle magic")?;
        let version = reader.read_u32("beacon bundle version")?;
        if version != BEACON_BUNDLE_V3_VERSION {
            return Err(format!(
                "unsupported beacon bundle version {version}; expected {BEACON_BUNDLE_V3_VERSION}"
            ));
        }
    } else {
        reader.expect_raw(BEACON_BUNDLE_MAGIC, "beacon bundle magic")?;
        let version = reader.read_u32("beacon bundle version")?;
        if version != BEACON_BUNDLE_VERSION {
            return Err(format!(
                "unsupported beacon bundle version {version}; expected {BEACON_BUNDLE_VERSION}"
            ));
        }
    }
    let profile = SecurityProfile::from_id(read_bcn_label(&mut reader, "beacon bundle profile")?)?;
    let policy = bcn_policy_from_id(read_bcn_label(&mut reader, "beacon bundle policy")?)?;
    let attention_policy =
        AttentionPolicy::from_id(read_bcn_label(&mut reader, "beacon bundle attention policy")?)?;
    let anchor = read_bcn_digest(&mut reader, "beacon bundle anchor")?;
    let beacon = StarknetBeacon {
        block_number: reader.read_u64("beacon block number")?,
        block_hash: read_bcn_digest(&mut reader, "beacon block hash")?,
    };
    let response_root = read_bcn_digest(&mut reader, "response root")?;
    let beacon_policy = BeaconPolicy {
        layers: reader.read_u32("beacon policy layers")?,
        sample_den: reader.read_u64("beacon policy sample_den")?,
    };
    let path = read_bcn_digests(&mut reader, MAX_BUNDLE_SIBLINGS, "activation path")?;
    let has_continuity = reader.read_u32_byte("continuity flag")?;
    let continuity = match has_continuity {
        0 => None,
        1 => Some(read_bcn_continuity_opening(&mut reader, "continuity")?),
        other => return Err(format!("invalid continuity flag {other}")),
    };

    let b_shape = AttentionShape {
        dimension: reader.read_len(MAX_BUNDLE_ROWS, "Stage B dimension")?,
        heads: reader.read_len(MAX_BUNDLE_ROWS, "Stage B heads")?,
        kv_heads: reader.read_len(MAX_BUNDLE_ROWS, "Stage B kv_heads")?,
        head_dim: reader.read_len(MAX_BUNDLE_ROWS, "Stage B head_dim")?,
    };
    let b_layout = CacheLayout {
        chunk: reader.read_len(MAX_BUNDLE_ROWS, "Stage B chunk")?,
    };
    let b_policy_id = read_bcn_label(&mut reader, "Stage B policy_id")?.to_string();
    let attention_commitment = read_bcn_digest(&mut reader, "Stage B attention_commitment")?;
    let binding = read_bcn_digest(&mut reader, "Stage B binding")?;
    let gamma1 = read_bcn_vec(&mut reader, "Stage B gamma1", read_bcn_i16)?;
    let gamma2 = read_bcn_vec(&mut reader, "Stage B gamma2", read_bcn_i16)?;
    let chunk_commitments =
        read_bcn_digests(&mut reader, MAX_BUNDLE_ROWS, "Stage B chunk_commitments")?;
    let token_count = reader.read_len(MAX_BUNDLE_ROWS, "Stage B token count")?;
    let mut tokens = Vec::with_capacity(token_count);
    let mut partial_rows = Vec::with_capacity(token_count);
    for t in 0..token_count {
        let position = reader.read_u64(&format!("Stage B token {t} position"))?;
        let h_in = read_bcn_vec(&mut reader, &format!("Stage B token {t} h_in"), read_bcn_i16)?;
        let q_raw = read_bcn_vec(&mut reader, &format!("Stage B token {t} q_raw"), read_bcn_i32)?;
        let k_raw = read_bcn_vec(&mut reader, &format!("Stage B token {t} k_raw"), read_bcn_i32)?;
        let v_raw = read_bcn_vec(&mut reader, &format!("Stage B token {t} v_raw"), read_bcn_i32)?;
        let scores =
            read_bcn_rows(&mut reader, &format!("Stage B token {t} scores"), read_bcn_i32)?;
        let ctx_raw =
            read_bcn_vec(&mut reader, &format!("Stage B token {t} ctx_raw"), read_bcn_i32)?;
        let o_raw = read_bcn_vec(&mut reader, &format!("Stage B token {t} o_raw"), read_bcn_i32)?;
        let (a, x) = if q_raw.is_empty() {
            let a = read_bcn_vec(&mut reader, &format!("Stage B token {t} a"), read_bcn_i16)?;
            let x = read_bcn_vec(&mut reader, &format!("Stage B token {t} x"), read_bcn_i16)?;
            (a, x)
        } else {
            (Vec::new(), Vec::new())
        };
        let chunk_ids = read_bcn_vec(&mut reader, &format!("Stage B token {t} chunk_ids"), |r, l| {
            r.read_len(MAX_BUNDLE_ROWS, l)
        })?;
        let cache_commitment_before = read_bcn_digest(
            &mut reader,
            &format!("Stage B token {t} cache_commitment_before"),
        )?;
        let cache_commitment_after = read_bcn_digest(
            &mut reader,
            &format!("Stage B token {t} cache_commitment_after"),
        )?;
        let k_rows = read_bcn_rows(&mut reader, &format!("Stage B token {t} k_rows"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        let v_rows = read_bcn_rows(&mut reader, &format!("Stage B token {t} v_rows"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        tokens.push(StageBToken {
            step: PublicStep {
                position,
                h_in,
                q_raw,
                k_raw,
                v_raw,
                scores,
                ctx_raw,
                o_raw,
                a,
                x,
            },
            chunk_ids,
            cache_commitment_before,
            cache_commitment_after,
        });
        partial_rows.push((k_rows, v_rows));
    }
    let attention = StageBStatement {
        shape: b_shape,
        layout: b_layout,
        policy_id: b_policy_id,
        attention_commitment,
        gamma1,
        gamma2,
        chunk_commitments,
        tokens,
        partial_rows,
        binding,
    };
    let attention_receipt = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "Stage B receipt")?
        .to_vec();

    let shape = read_bcn_shape(&mut reader, "Stage A statement shape")?;
    let policy_id = read_bcn_label(&mut reader, "Stage A statement policy_id")?.to_string();
    let policy_digest = read_bcn_digest(&mut reader, "policy digest")?;
    let model_digest = read_bcn_digest(&mut reader, "model digest")?;
    let model_registry_manifest_digest = read_bcn_digest(&mut reader, "manifest digest")?;
    let table_digest = read_bcn_digest(&mut reader, "table digest")?;
    let input_digest = read_bcn_digest(&mut reader, "input digest")?;
    let router_outputs_raw = read_bcn_rows(&mut reader, "router outputs", |r, l| {
        r.read_u64(l).map(|v| i64::from_le_bytes(v.to_le_bytes()))
    })?;
    let routes = read_bcn_rows(&mut reader, "routes", |r, l| r.read_len(MAX_BUNDLE_ROWS, l))?;
    let route_weights_q15 = read_bcn_rows(&mut reader, "route weights", read_bcn_u16)?;
    let output_q8 = read_bcn_rows(&mut reader, "output_q8", read_bcn_i16)?;
    let wide_log_columns = reader.read_len(MAX_BUNDLE_ROWS, "wide_log_columns")?;
    let statement = StageANonlinearBatchStatement {
        shape,
        policy_id,
        policy_digest,
        model_digest,
        model_registry_manifest_digest,
        table_digest,
        input_digest,
        router_outputs_raw,
        routes,
        route_weights_q15,
        output_q8,
        wide_log_columns,
    };

    let context_shape = read_bcn_shape(&mut reader, "context shape")?;
    let c_model_digest = read_bcn_digest(&mut reader, "context model digest")?;
    let manifest_digest = read_bcn_digest(&mut reader, "context manifest digest")?;
    let scheme_digest = read_bcn_digest(&mut reader, "context scheme digest")?;
    let router_commitment = read_bcn_digest(&mut reader, "router commitment")?;
    let shared_commitment = read_bcn_digest(&mut reader, "shared commitment")?;
    let registered_expert_root = read_bcn_digest(&mut reader, "registered expert root")?;
    let openings = reader.read_len(MAX_BUNDLE_ROWS, "active expert opening count")?;
    let mut active_expert_openings = Vec::with_capacity(openings);
    for idx in 0..openings {
        let expert_id = reader.read_len(MAX_BUNDLE_ROWS, &format!("opening {idx} expert id"))?;
        let mut slot_roles = [0u8; 3];
        for role in slot_roles.iter_mut() {
            *role = reader.read_u32_byte(&format!("opening {idx} slot role"))?;
        }
        let group_commitment =
            read_bcn_digest(&mut reader, &format!("opening {idx} group commitment"))?;
        let siblings = read_bcn_digests(
            &mut reader,
            MAX_BUNDLE_SIBLINGS,
            &format!("opening {idx} siblings"),
        )?;
        active_expert_openings.push(StageARegisteredExpertOpening {
            expert_id,
            slot_roles,
            group_commitment,
            siblings,
        });
    }
    let context = StageARegisteredContext {
        shape: context_shape,
        model_digest: c_model_digest,
        manifest_digest,
        scheme_digest,
        router_commitment,
        shared_commitment,
        registered_expert_root,
        active_expert_openings,
    };
    let receipt = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "Stage A receipt")?
        .to_vec();
    let boundaries = if is_v3 {
        None
    } else {
        match reader.read_u32_byte("boundaries flag")? {
            0 => None,
            1 => Some(read_bcn_boundaries(&mut reader)?),
            other => return Err(format!("invalid boundaries flag {other}")),
        }
    };
    reader.finish()?;
    let bundle = SpotCheckVerificationBundle {
        profile,
        policy,
        attention_policy,
        anchor,
        beacon,
        response_root,
        beacon_policy,
        path,
        continuity,
        attention,
        attention_receipt,
        statement,
        context,
        receipt,
        boundaries,
    };
    if spot_check_verification_bundle_bytes_matching_header(&bundle, bytes)? != bytes {
        return Err("beacon bundle bytes are not canonical".into());
    }
    Ok(bundle)
}

fn verify_continuity(
    root: &[u8; 32],
    layers: u32,
    layer: u32,
    a: &[Vec<i16>],
    out: &[Vec<i16>],
    continuity: &Option<ContinuityOpening>,
) -> Result<(), String> {
    let next = layer + 1;
    match (continuity, next < layers) {
        (None, false) => Ok(()),
        (Some(_), false) => Err("spot check opens a leaf after the last layer".into()),
        (None, true) => Err("spot check does not open the next layer's leaf".into()),
        (Some(opening), true) => {
            let next_h_in = public_transition(a, out)?;
            let next_leaf = activation_leaf(
                next,
                &LayerDigests {
                    h_in: state_digest(&next_h_in),
                    a: opening.a_digest,
                    x: opening.x_digest,
                    out: opening.output_digest,
                    cache: opening.cache_digest,
                },
            );
            if verify_activation_opening(root, layers, next, &next_leaf, &opening.path) {
                Ok(())
            } else {
                Err(
                    "the next layer's committed residual is not the proved layer's transition"
                        .into(),
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_base_spot_check(
    anchor: &[u8; 32],
    beacon: StarknetBeacon,
    response_root: [u8; 32],
    beacon_policy: BeaconPolicy,
    path: &[[u8; 32]],
    continuity: &Option<ContinuityOpening>,
    attention: &StageBStatement,
    attention_proof: &StageBProof,
    attention_receipt: &[u8],
    statement: &StageANonlinearBatchStatement,
    context: &StageARegisteredContext,
    proof: &StageAReceiptProof,
    receipt: &[u8],
    policy: StageANumericPolicy,
    attention_policy: AttentionPolicy,
    profile: &SecurityProfile,
) -> Result<StageARegisteredVerifierRun, String> {
    let expected = derive_challenge(beacon, response_root, beacon_policy)?;
    if !expected.challenged {
        return Err("the beacon does not challenge this response".into());
    }
    let link = beacon_link(&expected);
    if attention.binding != link.prev_receipt_digest {
        return Err("attention receipt is not bound to this challenge".into());
    }
    if layer_anchor(
        &context.anchor_digest(),
        &attention.attention_commitment,
        &attention_policy.id(),
        &attention.gamma1,
        &attention.gamma2,
    ) != *anchor
    {
        return Err("spot check anchor is not the layer's public anchor".into());
    }
    let base_pos = attention
        .tokens
        .first()
        .ok_or_else(|| "attention receipt has no tokens".to_string())?
        .step
        .position;
    if attention
        .tokens
        .iter()
        .enumerate()
        .any(|(t, token)| token.step.position != base_pos + t as u64)
    {
        return Err("attention receipt tokens are not consecutive sequence positions".into());
    }
    if stage_b_receipt_bytes(attention_proof)? != attention_receipt {
        return Err("attention receipt bytes do not serialise the attention proof".into());
    }
    let decoded_b = decode_stage_b_receipt_bytes(attention_receipt, attention)?;
    let leaf_params = group_params(2 * attention.shape.dimension.ilog2() as usize, 4, profile)?;
    if encode_stage_a_receipt_bytes_matching_header(proof, receipt)? != receipt {
        return Err("receipt bytes do not serialise the proof".into());
    }
    let decoded_a = decode_stage_a_receipt_bytes(receipt, statement, policy)?;
    let steps: Vec<PublicStep> = attention.tokens.iter().map(|t| t.step.clone()).collect();
    if decoded_b.attention_leaf_proof.is_empty() && decoded_b.cache_chunk_proof.is_empty() {
        let (b_run, deferred_b) = verify_stage_b_defer_cprs(
            attention,
            attention_policy,
            profile,
            &leaf_params,
            &decoded_b,
        )?;
        let (h_in, a, x) = layer_states(&steps, &b_run.recomputed);
        let leaf = activation_leaf(
            expected.layer,
            &LayerDigests::of(
                &h_in,
                &a,
                &x,
                &statement.output_q8,
                attention.cache_digest(),
            ),
        );
        if !verify_activation_opening(
            &response_root,
            beacon_policy.layers,
            expected.layer,
            &leaf,
            path,
        ) {
            return Err("opened activations are not in the committed response".into());
        }
        verify_continuity(
            &response_root,
            beacon_policy.layers,
            expected.layer,
            &a,
            &statement.output_q8,
            continuity,
        )?;
        verify_stage_a_receipt_against_anchor_with_stage_b(
            &context.anchor_digest(),
            context,
            &x,
            policy,
            profile,
            statement,
            &decoded_a,
            &link,
            Some(&deferred_b),
        )
    } else {
        let b_run = verify_stage_b(
            attention,
            attention_policy,
            profile,
            &leaf_params,
            &decoded_b,
        )?;
        let (h_in, a, x) = layer_states(&steps, &b_run.recomputed);
        let leaf = activation_leaf(
            expected.layer,
            &LayerDigests::of(
                &h_in,
                &a,
                &x,
                &statement.output_q8,
                attention.cache_digest(),
            ),
        );
        if !verify_activation_opening(
            &response_root,
            beacon_policy.layers,
            expected.layer,
            &leaf,
            path,
        ) {
            return Err("opened activations are not in the committed response".into());
        }
        verify_continuity(
            &response_root,
            beacon_policy.layers,
            expected.layer,
            &a,
            &statement.output_q8,
            continuity,
        )?;
        verify_stage_a_receipt_against_anchor(
            &context.anchor_digest(),
            context,
            &x,
            policy,
            profile,
            statement,
            &decoded_a,
            &link,
        )
    }
}

pub struct BeaconBundleVerdict {
    pub stage_a_mode: &'static str,
    pub decode_ms: f64,
    pub verify_ms: f64,
    pub layer: u32,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub emitted_token_ids: Option<Vec<u32>>,
    pub emitted_next_token: Option<u32>,
    pub run: StageARegisteredVerifierRun,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
struct Timer(std::time::Instant);

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
impl Timer {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }
    fn elapsed_ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1_000.0
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct Timer;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Timer {
    fn start() -> Self {
        Self
    }
    fn elapsed_ms(&self) -> f64 {
        0.0
    }
}

pub fn verify_spot_check_verification_bundle(
    bundle: &SpotCheckVerificationBundle,
) -> Result<BeaconBundleVerdict, String> {
    let decode_start = Timer::start();
    let attention_proof =
        decode_stage_b_receipt_bytes(&bundle.attention_receipt, &bundle.attention)?;
    let proof = decode_stage_a_receipt_bytes(
        &bundle.receipt,
        &bundle.statement,
        bundle.policy,
    )?;
    if encode_stage_a_receipt_bytes_matching_header(&proof, &bundle.receipt)?
        != bundle.receipt
    {
        return Err("Stage A receipt bytes are not canonical".into());
    }
    let stage_a_mode = proof.mode_id();
    // The GKR-lane modes (9 and 10) neither mix `router_outputs_raw` into their transcript nor check the routes
    // against it (only the public-witness mode does both), so a bundle carrying it would assert raw router outputs
    // that nothing verifies. Mode 11 refuses it at decode.
    if matches!(stage_a_mode, "gkr_lane" | "zk_gkr_lane")
        && bundle.statement.router_outputs_raw.iter().any(|row| !row.is_empty())
    {
        return Err(format!(
            "STWOBCN4 {stage_a_mode} bundle carries router_outputs_raw, which its receipt neither binds nor checks (finding D1); refused"
        ));
    }
    let challenge = derive_challenge(bundle.beacon, bundle.response_root, bundle.beacon_policy)?;
    let decode_ms = decode_start.elapsed_ms();
    let verify_start = Timer::start();
    let (run, prompt_token_ids, emitted_token_ids, emitted_next_token) =
        if let Some(ref b) = bundle.boundaries {
            let link = beacon_link(&challenge);
            if b.embedding.binding != link.prev_receipt_digest {
                return Err("embedding lookup receipt is not bound to this challenge".into());
            }
            if b.lm_head.binding != link.prev_receipt_digest {
                return Err("LM head receipt is not bound to this challenge".into());
            }
            if b.embedding.embedding_commitment != b.boundary_anchors.embedding_commitment
                || b.lm_head.lm_head_commitment != b.boundary_anchors.lm_head_commitment
                || b.lm_head.gamma_final != b.boundary_anchors.gamma_final
            {
                return Err("boundary statements do not match boundary anchors".into());
            }
            let expected_anchor = layer_anchor_with_boundaries(
                &bundle.context.anchor_digest(),
                &bundle.attention.attention_commitment,
                &bundle.attention_policy.id(),
                &bundle.attention.gamma1,
                &bundle.attention.gamma2,
                Some(&b.boundary_anchors),
            );
            if bundle.anchor != expected_anchor {
                return Err(
                    "forward-pass spot check anchor does not match the layer's public anchor"
                        .into(),
                );
            }
            verify_embedding_lookup(&b.embedding, &bundle.profile, &b.embedding_proof)?;
            let leaf0 = activation_leaf(
                0,
                &LayerDigests {
                    h_in: state_digest(&b.embedding.h_out),
                    a: b.leaf0_opening.a_digest,
                    x: b.leaf0_opening.x_digest,
                    out: b.leaf0_opening.output_digest,
                    cache: b.leaf0_opening.cache_digest,
                },
            );
            if !verify_activation_opening(
                &bundle.response_root,
                bundle.beacon_policy.layers,
                0,
                &leaf0,
                &b.leaf0_opening.path,
            ) {
                return Err(
                    "Leaf 0 in the activation commitment does not consume the proven embedding output"
                        .into(),
                );
            }
            if challenge.layer == 0 {
                let spot_h_in: Vec<Vec<i16>> = bundle
                    .attention
                    .tokens
                    .iter()
                    .map(|t| t.step.h_in.clone())
                    .collect();
                if spot_h_in != b.embedding.h_out {
                    return Err(
                        "challenged Layer 0 h_in does not match the proven embedding output"
                            .into(),
                    );
                }
            }
            verify_lm_head(&b.lm_head, &bundle.profile, &b.lm_head_proof)?;
            let prompt_tokens = b.embedding.token_ids.clone();
            let emitted_tokens = b.lm_head.argmax_token_ids()?;
            let next_token = b.lm_head.emitted_next_token()?;
            if prompt_tokens.len() != bundle.statement.shape.batch
                || emitted_tokens.len() != bundle.statement.shape.batch
            {
                return Err(
                    "forward-pass boundary token batch does not match Stage A batch".into(),
                );
            }
            let last_idx = bundle.beacon_policy.layers.saturating_sub(1);
            let h_final_from_terminal =
                public_transition(&b.terminal_opening.last_a, &b.terminal_opening.last_out)?;
            if h_final_from_terminal != b.lm_head.h_in {
                return Err("LM head input does not match the terminal layer's transition".into());
            }
            let last_leaf = activation_leaf(
                last_idx,
                &LayerDigests {
                    h_in: b.terminal_opening.h_in_digest,
                    a: state_digest(&b.terminal_opening.last_a),
                    x: b.terminal_opening.x_digest,
                    out: state_digest(&b.terminal_opening.last_out),
                    cache: b.terminal_opening.cache_digest,
                },
            );
            if !verify_activation_opening(
                &bundle.response_root,
                bundle.beacon_policy.layers,
                last_idx,
                &last_leaf,
                &b.terminal_opening.path,
            ) {
                return Err(
                    "terminal leaf in the activation commitment does not produce the LM head input"
                        .into(),
                );
            }
            if challenge.layer == last_idx
                && b.terminal_opening.last_out != bundle.statement.output_q8
            {
                return Err(
                    "challenged terminal layer output does not match terminal opening".into(),
                );
            }
            let base_anchor = layer_anchor(
                &bundle.context.anchor_digest(),
                &bundle.attention.attention_commitment,
                &bundle.attention_policy.id(),
                &bundle.attention.gamma1,
                &bundle.attention.gamma2,
            );
            let run = verify_base_spot_check(
                &base_anchor,
                bundle.beacon,
                bundle.response_root,
                bundle.beacon_policy,
                &bundle.path,
                &bundle.continuity,
                &bundle.attention,
                &attention_proof,
                &bundle.attention_receipt,
                &bundle.statement,
                &bundle.context,
                &proof,
                &bundle.receipt,
                bundle.policy,
                bundle.attention_policy,
                &bundle.profile,
            )?;
            (
                run,
                Some(prompt_tokens),
                Some(emitted_tokens),
                Some(next_token),
            )
        } else {
            let run = verify_base_spot_check(
                &bundle.anchor,
                bundle.beacon,
                bundle.response_root,
                bundle.beacon_policy,
                &bundle.path,
                &bundle.continuity,
                &bundle.attention,
                &attention_proof,
                &bundle.attention_receipt,
                &bundle.statement,
                &bundle.context,
                &proof,
                &bundle.receipt,
                bundle.policy,
                bundle.attention_policy,
                &bundle.profile,
            )?;
            (run, None, None, None)
        };
    Ok(BeaconBundleVerdict {
        stage_a_mode,
        decode_ms,
        verify_ms: verify_start.elapsed_ms(),
        layer: challenge.layer,
        prompt_token_ids,
        emitted_token_ids,
        emitted_next_token,
        run,
    })
}

pub fn verify_spot_check_verification_bundle_bytes(
    bytes: &[u8],
) -> Result<BeaconBundleVerdict, String> {
    let bundle = decode_spot_check_verification_bundle(bytes)?;
    verify_spot_check_verification_bundle(&bundle)
}

pub fn verify_spot_check_bundle_for_emitted_token(
    bundle: &SpotCheckVerificationBundle,
    expected_prompt_tokens: &[u32],
    expected_next_token: u32,
) -> Result<BeaconBundleVerdict, String> {
    let verdict = verify_spot_check_verification_bundle(bundle)?;
    let prompt_tokens = verdict
        .prompt_token_ids
        .as_deref()
        .ok_or_else(|| "spot-check bundle does not include forward-pass boundary proofs".to_string())?;
    if prompt_tokens != expected_prompt_tokens {
        return Err(format!(
            "prompt token mismatch: bundle proved {:?}, expected {:?}",
            prompt_tokens, expected_prompt_tokens
        ));
    }
    let next_token = verdict
        .emitted_next_token
        .ok_or_else(|| "spot-check bundle does not include an emitted next token".to_string())?;
    if next_token != expected_next_token {
        return Err(format!(
            "emitted next token mismatch: bundle proved token {}, expected {}",
            next_token, expected_next_token
        ));
    }
    Ok(verdict)
}

pub fn verify_spot_check_bundle_bytes_for_emitted_token(
    bytes: &[u8],
    expected_prompt_tokens: &[u32],
    expected_next_token: u32,
) -> Result<BeaconBundleVerdict, String> {
    let bundle = decode_spot_check_verification_bundle(bytes)?;
    verify_spot_check_bundle_for_emitted_token(&bundle, expected_prompt_tokens, expected_next_token)
}

pub fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn parse_hex32(raw: &str) -> Result<[u8; 32], String> {
    let s = raw.trim().strip_prefix("0x").unwrap_or_else(|| raw.trim());
    if !s.is_ascii() {
        return Err("hex32 digest must contain only ASCII hex characters".into());
    }
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars for 32-byte digest, got {}", s.len()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)
            .map_err(|_| format!("invalid hex byte at offset {}", 2 * i))?;
    }
    Ok(out)
}

pub fn parse_chain_link_spec(raw: &str) -> Result<(u32, [u8; 32]), String> {
    let (l_str, h_str) = raw
        .trim()
        .split_once(':')
        .ok_or_else(|| "expected <layer:hex32> chain-link format".to_string())?;
    let layer: u32 = l_str
        .parse()
        .map_err(|_| format!("invalid chain-link layer '{l_str}'"))?;
    let digest = parse_hex32(h_str)?;
    Ok((layer, digest))
}

/// Decode and verify a public-tokens `STWOMDL1` whole-model standalone receipt directly from wire
/// bytes against **only** `expected_full_stack_anchor` (`32 B`), `prompt_token_ids`, and
/// `emitted_token_ids`.
pub fn verify_whole_model_receipt_bytes(
    expected_full_stack_anchor: &[u8; 32],
    prompt_token_ids: &[u32],
    emitted_token_ids: &[u32],
    receipt_bytes: &[u8],
) -> Result<(WholeModelStatement, StandaloneAuditVerifierRun), String> {
    let proof = decode_whole_model_receipt(receipt_bytes)?;
    let run = verify_whole_model_stark_against_anchor(
        expected_full_stack_anchor,
        prompt_token_ids,
        emitted_token_ids,
        &proof,
    )?;
    Ok((proof.statement, run))
}

/// Decode and verify a salted-`Blake2s`-committed-tokens `STWOMDL1` whole-model receipt from wire
/// bytes against `expected_full_stack_anchor`, `expected_prompt_commitment`, and
/// `expected_emitted_commitment`.
pub fn verify_whole_model_committed_receipt_bytes(
    expected_full_stack_anchor: &[u8; 32],
    expected_prompt_commitment: &[u8; 32],
    expected_emitted_commitment: &[u8; 32],
    receipt_bytes: &[u8],
) -> Result<(WholeModelStatement, StandaloneAuditVerifierRun), String> {
    let proof = decode_whole_model_receipt(receipt_bytes)?;
    let run = verify_whole_model_committed_stark_against_anchor(
        expected_full_stack_anchor,
        expected_prompt_commitment,
        expected_emitted_commitment,
        &proof,
    )?;
    Ok((proof.statement, run))
}

/// Decode and verify a `STWOREC2` standalone single-block receipt directly from wire bytes against
/// **only** `expected_standalone_anchor` (`32 B`) and `expected_initial_chain`.
pub fn verify_standalone_block_receipt_bytes(
    expected_standalone_anchor: &[u8; 32],
    expected_initial_chain: &StandaloneChainLink,
    receipt_bytes: &[u8],
) -> Result<(StandaloneBlockStatement, StandaloneAuditVerifierRun), String> {
    verify_standalone_block_receipt_bytes_with_io(
        expected_standalone_anchor,
        expected_initial_chain,
        None,
        None,
        receipt_bytes,
    )
}

/// Decode and verify a `STWOREC2` standalone single-block receipt with optional caller-pinned
/// `expected_final_chain` and `expected_public_io_digest`.
///
/// A receipt with an end stage (kind byte 2) is refused: its public values are token ids, and only the
/// model verifier (`verify_standalone_model_chain_against_anchor`, `ummon-verify model`) checks them
/// against the vocabulary; the circuit bounds them only below the padded vocabulary
/// (`V_pad`, the vocabulary padded to a power of two).
pub fn verify_standalone_block_receipt_bytes_with_io(
    expected_standalone_anchor: &[u8; 32],
    expected_initial_chain: &StandaloneChainLink,
    expected_final_chain: Option<&StandaloneChainLink>,
    expected_public_io_digest: Option<&[u8; 32]>,
    receipt_bytes: &[u8],
) -> Result<(StandaloneBlockStatement, StandaloneAuditVerifierRun), String> {
    let proof = decode_standalone_block_receipt(receipt_bytes)?;
    if proof.statement.ends.is_some() {
        return Err(
            "this receipt has an end stage (embedding or head): verify it in its model chain (`ummon-verify model`), which checks the tokens and the output token against the vocabulary"
                .into(),
        );
    }
    let run = verify_standalone_block_stark_against_anchor_and_io(
        expected_standalone_anchor,
        expected_initial_chain,
        expected_final_chain,
        expected_public_io_digest,
        &proof,
    )?;
    Ok((proof.statement, run))
}

/// Decode and verify a `STWOREC1` M4 split-anchor-and-wiring-root receipt from wire bytes.
pub fn verify_recursive_audit_receipt_bytes(
    expected_block_anchor: &[u8; 32],
    expected_wiring_root: &[u8; 32],
    expected_initial_chain: &StageAChainLink,
    receipt_bytes: &[u8],
) -> Result<(RecursiveAuditStatement, RecursiveAuditVerifierRun), String> {
    let proof = decode_recursive_audit_receipt(receipt_bytes)?;
    let run = verify_recursive_audit_stark_against_anchor(
        expected_block_anchor,
        expected_wiring_root,
        expected_initial_chain,
        &proof,
    )?;
    Ok((proof.statement, run))
}

pub const STANDALONE_CHAIN_MANIFEST_MAGIC: &[u8; 8] = b"STWOCHN2";
pub const MAX_STANDALONE_CHAIN_BLOCKS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneChainManifestEntry {
    pub block_index: usize,
    pub standalone_anchor: [u8; 32],
    pub com_h_in: [u8; 32],
    pub com_h_out: [u8; 32],
    pub receipt_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneChainManifest {
    pub initial_chain: StandaloneChainLink,
    pub blocks: Vec<StandaloneChainManifestEntry>,
}

fn validate_relative_receipt_subpath(rel: &str) -> Result<(), String> {
    if rel.is_empty() || rel.len() > 1024 {
        return Err("chain manifest receipt_path must be 1..=1024 bytes".into());
    }
    if rel.bytes().any(|b| !b.is_ascii_graphic() || b == b'\\') {
        return Err(
            "chain manifest receipt_path must contain only printable ASCII without whitespace or '\\'"
                .into(),
        );
    }
    if rel.starts_with('/') || rel.ends_with('/') {
        return Err(format!(
            "chain manifest receipt_path must be a normalized relative path, got '{rel}'"
        ));
    }
    if !rel
        .split('/')
        .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
    {
        return Err(format!(
            "chain manifest receipt_path contains empty, '.' or '..' segment in '{rel}'"
        ));
    }
    let p = std::path::Path::new(rel);
    if p.is_absolute() {
        return Err(format!(
            "chain manifest receipt_path must be relative, got '{rel}'"
        ));
    }
    let mut count = 0usize;
    for comp in p.components() {
        match comp {
            std::path::Component::Normal(_) => count += 1,
            _ => {
                return Err(format!(
                    "chain manifest receipt_path contains disallowed component in '{rel}'"
                ));
            }
        }
    }
    if count == 0 {
        return Err("chain manifest receipt_path has zero path components".into());
    }
    Ok(())
}

pub fn encode_standalone_chain_manifest(
    manifest: &StandaloneChainManifest,
) -> Result<String, String> {
    if manifest.blocks.is_empty() {
        return Err("standalone chain manifest must contain at least one block".into());
    }
    if manifest.blocks.len() > MAX_STANDALONE_CHAIN_BLOCKS {
        return Err(format!(
            "standalone chain manifest exceeds MAX_STANDALONE_CHAIN_BLOCKS ({MAX_STANDALONE_CHAIN_BLOCKS})"
        ));
    }
    let mut out = String::new();
    out.push_str("STWOCHN2\n");
    out.push_str(&format!(
        "initial_chain {}:{}\n",
        manifest.initial_chain.layer,
        hex32(&manifest.initial_chain.prev_receipt_digest)
    ));
    for (expected_idx, blk) in manifest.blocks.iter().enumerate() {
        if blk.block_index != expected_idx {
            return Err(format!(
                "standalone chain manifest block_index {} != expected {expected_idx}",
                blk.block_index
            ));
        }
        validate_relative_receipt_subpath(&blk.receipt_path)?;
        out.push_str(&format!(
            "block {} {} {} {} {}\n",
            blk.block_index,
            hex32(&blk.standalone_anchor),
            hex32(&blk.com_h_in),
            hex32(&blk.com_h_out),
            blk.receipt_path,
        ));
    }
    Ok(out)
}

pub fn decode_standalone_chain_manifest(bytes: &[u8]) -> Result<StandaloneChainManifest, String> {
    if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOCHN2 manifest exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    if !bytes.starts_with(STANDALONE_CHAIN_MANIFEST_MAGIC) {
        return Err("invalid STWOCHN2 manifest magic".into());
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "STWOCHN2 manifest is not valid UTF-8".to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some("STWOCHN2") {
        return Err("STWOCHN2 header line must be exact 'STWOCHN2'".into());
    }
    let mut initial_chain: Option<StandaloneChainLink> = None;
    let mut blocks = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["initial_chain", spec] => {
                if initial_chain.is_some() {
                    return Err(format!(
                        "STWOCHN2 line {}: duplicate initial_chain directive",
                        line_no + 2
                    ));
                }
                if !blocks.is_empty() {
                    return Err(format!(
                        "STWOCHN2 line {}: initial_chain must precede block entries",
                        line_no + 2
                    ));
                }
                let (layer, prev_receipt_digest) = parse_chain_link_spec(spec)?;
                initial_chain = Some(StandaloneChainLink {
                    layer,
                    prev_receipt_digest,
                });
            }
            ["block", idx_s, anchor_s, com_in_s, com_out_s, rel_path] => {
                if initial_chain.is_none() {
                    return Err(format!(
                        "STWOCHN2 line {}: block entry before initial_chain",
                        line_no + 2
                    ));
                }
                if blocks.len() >= MAX_STANDALONE_CHAIN_BLOCKS {
                    return Err(format!(
                        "STWOCHN2 exceeds MAX_STANDALONE_CHAIN_BLOCKS ({MAX_STANDALONE_CHAIN_BLOCKS})"
                    ));
                }
                let block_index: usize = idx_s.parse().map_err(|_| {
                    format!("STWOCHN2 line {}: invalid block index '{idx_s}'", line_no + 2)
                })?;
                if block_index != blocks.len() {
                    return Err(format!(
                        "STWOCHN2 line {}: non-contiguous block index {block_index}, expected {}",
                        line_no + 2,
                        blocks.len()
                    ));
                }
                validate_relative_receipt_subpath(rel_path)?;
                blocks.push(StandaloneChainManifestEntry {
                    block_index,
                    standalone_anchor: parse_hex32(anchor_s)?,
                    com_h_in: parse_hex32(com_in_s)?,
                    com_h_out: parse_hex32(com_out_s)?,
                    receipt_path: (*rel_path).to_string(),
                });
            }
            _ => {
                return Err(format!(
                    "STWOCHN2 line {}: unrecognized directive '{line}'",
                    line_no + 2
                ));
            }
        }
    }
    let initial_chain = initial_chain.ok_or("STWOCHN2 manifest missing initial_chain")?;
    if blocks.is_empty() {
        return Err("STWOCHN2 manifest contains zero blocks".into());
    }
    let manifest = StandaloneChainManifest {
        initial_chain,
        blocks,
    };
    // One manifest, one byte string: the digest of the file then names the manifest.
    if encode_standalone_chain_manifest(&manifest)?.as_bytes() != bytes {
        return Err("STWOCHN2 manifest bytes are non-canonical".into());
    }
    Ok(manifest)
}

/// Largest total size of the receipt files of one `STWOCHN2` chain.
pub const MAX_CHAIN_TOTAL_RECEIPT_BYTES: usize = 512 * 1024 * 1024;

/// Read a non-symlink regular file up to `cap` bytes, rejecting symlinks, FIFOs, sockets, and
/// device nodes both before and (on Unix) via post-open `(dev, ino)` descriptor verification.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn read_capped_regular(path: &std::path::Path, cap: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let meta = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if meta.file_type().is_symlink() {
        return Err("refusing to follow symlink".into());
    }
    if !meta.file_type().is_file() {
        return Err("path is not a regular file".into());
    }
    if meta.len() > cap as u64 {
        return Err(format!("larger than the {cap} byte wire limit"));
    }
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let fmeta = f.metadata().map_err(|e| e.to_string())?;
    if !fmeta.file_type().is_file() {
        return Err("opened descriptor is not a regular file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if fmeta.dev() != meta.dev() || fmeta.ino() != meta.ino() {
            return Err("file changed between metadata check and open (TOCTOU detected)".into());
        }
    }
    if fmeta.len() > cap as u64 {
        return Err(format!("larger than the {cap} byte wire limit"));
    }
    let mut out = Vec::with_capacity(fmeta.len() as usize);
    f.take(cap as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| e.to_string())?;
    if out.len() > cap {
        return Err(format!("larger than the {cap} byte wire limit"));
    }
    Ok(out)
}

/// Resolve `rel` inside `base_dir`, rejecting symlinks at every intermediate directory segment and
/// verifying canonical path containment inside `base_dir`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn resolve_contained_receipt_path(
    base_dir: &std::path::Path,
    rel: &str,
) -> Result<std::path::PathBuf, String> {
    let canon_base = base_dir
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize bundle directory {}: {e}", base_dir.display()))?;
    let mut cur = base_dir.to_path_buf();
    for seg in rel.split('/') {
        if seg.is_empty() || seg == "." || seg == ".." {
            return Err(format!("disallowed path segment in '{rel}'"));
        }
        cur.push(seg);
        let meta = std::fs::symlink_metadata(&cur)
            .map_err(|e| format!("{}: {e}", cur.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!("refusing to follow symlink segment '{}'", cur.display()));
        }
    }
    let canon_target = cur
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e}", cur.display()))?;
    if !canon_target.starts_with(&canon_base) {
        return Err(format!(
            "resolved receipt path '{}' escapes bundle directory '{}'",
            canon_target.display(),
            canon_base.display()
        ));
    }
    Ok(cur)
}

/// Median of `xs` (the upper middle element for an even count); `0` for an empty list.
pub fn median_f64(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Run `verify_once` once cold, then `bench_runs` more times when `bench_runs > 1`. Returns the
/// first result, the cold time and the median warm time (the cold time when there is no warm run).
pub fn run_timed_verifies<T, F>(bench_runs: usize, mut verify_once: F) -> Result<(T, f64, f64), String>
where
    F: FnMut() -> Result<(T, f64), String>,
{
    let (first_val, cold_ms) = verify_once()?;
    if bench_runs <= 1 {
        return Ok((first_val, cold_ms, cold_ms));
    }
    let mut warm = Vec::with_capacity(bench_runs);
    for _ in 0..bench_runs {
        let (_, ms) = verify_once()?;
        warm.push(ms);
    }
    let warm_med = median_f64(warm);
    Ok((first_val, cold_ms, warm_med))
}

/// Flag values of the `chn2` command of `ummon-verify`, as given on the command line or in the
/// matching environment variable, not yet parsed.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandaloneChainCommandArgs<'a> {
    /// `--chain-anchor` (`STWO_ML_CHAIN_ANCHOR`), required.
    pub chain_anchor: Option<&'a str>,
    /// `--initial-chain` (`STWO_ML_INITIAL_CHAIN`), required.
    pub initial_chain: Option<&'a str>,
    /// `--final-chain` (`STWO_ML_FINAL_CHAIN`), optional.
    pub final_chain: Option<&'a str>,
    /// `--public-io` (`STWO_ML_PUBLIC_IO`), optional.
    pub public_io: Option<&'a str>,
    /// `--profile` (`STWO_ML_PROFILE`), optional: the security profile the caller requires; every block must be
    /// proved under it.
    /// The profile id is a label: the inner parameters (the CPRS query count, the query and fold grinding bits) are
    /// fixed by the registered circuit, so the caller's anchor, not this pin, fixes them (8 and 16 bits of fold
    /// grinding both carry `production_hardened`).
    pub profile: Option<&'a str>,
    pub bench_runs: usize,
}

/// `--profile` of `rec2` and `chn2`: the id must be a standalone profile id (checked before any receipt is read).
pub fn require_known_standalone_profile(profile: &str, format: &str) -> Result<(), String> {
    if STANDALONE_PROFILE_IDS.contains(&profile) {
        Ok(())
    } else {
        Err(format!("--profile '{profile}' is not a profile id of {format}"))
    }
}

/// `--profile` of `rec2`: the receipt must be proved under the profile the caller
/// requires. Decodes the receipt's statement only, before any proof is checked. It compares a label: the inner
/// parameters (query count, query and fold grinding bits) are fixed by the anchor, not by the profile id.
pub fn require_standalone_receipt_profile(receipt_bytes: &[u8], profile: &str) -> Result<(), String> {
    require_known_standalone_profile(profile, "STWOREC2")?;
    let proof = decode_standalone_block_receipt(receipt_bytes)?;
    if proof.statement.profile_id != profile {
        return Err(format!(
            "the receipt is proved under profile '{}', not the profile '{profile}' the caller requires (--profile)",
            proof.statement.profile_id
        ));
    }
    Ok(())
}

/// What the `chn2` command reports for an accepted chain.
#[derive(Clone, Debug)]
pub struct StandaloneChainCommandReport {
    pub chain_anchor: [u8; 32],
    /// The manifest's initial chain link, equal to `--initial-chain`.
    pub initial_chain: StandaloneChainLink,
    pub verdict: StandaloneChainVerdict,
    pub receipts_sha256: [u8; 32],
    pub total_receipt_bytes: usize,
    pub per_block_receipt_bytes: Vec<usize>,
    pub per_block_verify_ms: Vec<f64>,
    pub num_blocks: usize,
    pub total_layers: usize,
    pub profile_id: String,
    pub mode_tag: u32,
    pub zk_trace_blinded: bool,
    pub batch: usize,
    pub cold_ms: f64,
    pub warm_ms: f64,
}

/// The `chn2` command of `ummon-verify`: decode a `STWOCHN2` manifest, check it against the
/// caller's chain anchor and initial chain link before reading any receipt, read and decode the
/// receipts under `base_dir`, and verify the chain against the chain anchor. The checks, their
/// order and their messages are those of the command; it prints the report or the error.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn verify_standalone_chain_command(
    manifest_bytes: &[u8],
    base_dir: &std::path::Path,
    args: &StandaloneChainCommandArgs<'_>,
) -> Result<StandaloneChainCommandReport, String> {
    let manifest = decode_standalone_chain_manifest(manifest_bytes)?;
    // The manifest is the prover's. Verify caller-pinned `--chain-anchor` and `--initial-chain`
    // BEFORE opening or decoding any `.rec2` file on disk so an unauthenticated manifest cannot
    // trigger 4,096 x 64 MiB allocations or file reads.
    let chain_anchor = args
        .chain_anchor
        .ok_or_else(|| "missing --chain-anchor <hex32> (or STWO_ML_CHAIN_ANCHOR) for STWOCHN2".to_string())
        .and_then(parse_hex32)?;
    if chain_anchor == [0u8; 32] {
        return Err("expected_chain_anchor must not be all-zero".into());
    }
    let manifest_anchors: Vec<[u8; 32]> = manifest
        .blocks
        .iter()
        .map(|b| b.standalone_anchor)
        .collect();
    if compute_standalone_chain_anchor(&manifest_anchors) != chain_anchor {
        return Err(
            "standalone block chain: 32B chain anchor mismatch (a block anchor, the block count or the order differs from registration)"
                .into(),
        );
    }
    let (init_layer, init_prev) = args
        .initial_chain
        .ok_or_else(|| "missing --initial-chain <layer:hex32> for STWOCHN2".to_string())
        .and_then(parse_chain_link_spec)?;
    if manifest.initial_chain.layer != init_layer
        || manifest.initial_chain.prev_receipt_digest != init_prev
    {
        return Err("manifest initial_chain disagrees with --initial-chain".into());
    }
    if let Some(p) = args.profile {
        require_known_standalone_profile(p, "STWOCHN2")?;
    }
    let expected_final_link = match args.final_chain {
        Some(s) => {
            let (l, d) = parse_chain_link_spec(s)?;
            Some(StandaloneChainLink {
                layer: l,
                prev_receipt_digest: d,
            })
        }
        None => None,
    };
    let expected_pub_io = match args.public_io {
        Some(s) => Some(parse_hex32(s)?),
        None => None,
    };

    let mut seen_paths: std::collections::HashSet<&str> =
        std::collections::HashSet::with_capacity(manifest.blocks.len());
    let mut decoded_proofs = Vec::with_capacity(manifest.blocks.len());
    let mut per_block_bytes = Vec::with_capacity(manifest.blocks.len());
    let mut total_receipt_bytes = 0usize;
    let mut total_layers = 0usize;
    let mut rcpt_hasher = sha2::Sha256::new();
    for entry in &manifest.blocks {
        if !seen_paths.insert(entry.receipt_path.as_str()) {
            return Err(format!(
                "duplicate receipt_path '{}' in chain manifest",
                entry.receipt_path
            ));
        }
        let rec_path = resolve_contained_receipt_path(base_dir, &entry.receipt_path).map_err(|e| {
            format!(
                "invalid block {} receipt path '{}': {e}",
                entry.block_index, entry.receipt_path
            )
        })?;
        let rec_bytes = read_capped_regular(&rec_path, MAX_WIRE_RECEIPT_BYTES).map_err(|e| {
            format!(
                "failed to read block {} receipt {}: {e}",
                entry.block_index,
                rec_path.display()
            )
        })?;
        total_receipt_bytes = match total_receipt_bytes.checked_add(rec_bytes.len()) {
            Some(tot) if tot <= MAX_CHAIN_TOTAL_RECEIPT_BYTES => tot,
            _ => {
                return Err(format!(
                    "cumulative chain receipt size exceeds {} byte limit",
                    MAX_CHAIN_TOTAL_RECEIPT_BYTES
                ))
            }
        };
        rcpt_hasher.update((entry.block_index as u64).to_le_bytes());
        rcpt_hasher.update((rec_bytes.len() as u64).to_le_bytes());
        rcpt_hasher.update(&rec_bytes);
        let proof = decode_standalone_block_receipt(&rec_bytes)
            .map_err(|r| format!("block {} decode failed: {r}", entry.block_index))?;
        match standalone_block_receipt_bytes(&proof) {
            Ok(reenc) if reenc == rec_bytes => {}
            Ok(_) => {
                return Err(format!(
                    "block {} receipt bytes are non-canonical",
                    entry.block_index
                ))
            }
            Err(r) => {
                return Err(format!(
                    "block {} re-encode check failed: {r}",
                    entry.block_index
                ))
            }
        }
        total_layers += proof.statement.num_layers;
        per_block_bytes.push(rec_bytes.len());
        decoded_proofs.push(proof);
    }
    let receipts_sha256: [u8; 32] = rcpt_hasher.finalize().into();
    if let Some(p) = args.profile {
        if let Some((k, proof)) = decoded_proofs.iter().enumerate().find(|(_, pr)| pr.statement.profile_id != p) {
            return Err(format!(
                "block {k} is proved under profile '{}', not the profile '{p}' the caller requires (--profile)",
                proof.statement.profile_id
            ));
        }
    }

    let chained_blocks: Vec<StandaloneChainedBlock<'_>> = manifest
        .blocks
        .iter()
        .zip(decoded_proofs.iter())
        .map(|(entry, proof)| StandaloneChainedBlock {
            standalone_anchor: &entry.standalone_anchor,
            proof,
            com_h_in: entry.com_h_in,
            com_h_out: entry.com_h_out,
        })
        .collect();

    let mut per_block_samples: Vec<Vec<f64>> =
        vec![Vec::with_capacity(args.bench_runs.max(1)); chained_blocks.len()];
    let (verdict, cold_ms, warm_ms) = run_timed_verifies(args.bench_runs, || {
        let t0 = std::time::Instant::now();
        let (v, blk_ms) = verify_standalone_block_chain_against_anchor_timed(
            &chain_anchor,
            &manifest.initial_chain,
            &chained_blocks,
        )?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        for (dst, &m) in per_block_samples.iter_mut().zip(blk_ms.iter()) {
            dst.push(m);
        }
        Ok((v, ms))
    })?;
    if let Some(exp_final) = expected_final_link {
        if verdict.final_chain != exp_final {
            return Err("chain final_chain disagrees with --final-chain".into());
        }
    }
    if let Some(exp_io) = expected_pub_io {
        let profile = &chained_blocks.first().ok_or("empty chain")?.proof.statement.profile_id;
        let actual_io = compute_standalone_block_public_io_digest_for_profile(
            profile,
            &verdict.com_h_in,
            &verdict.com_h_out,
        )?;
        if actual_io != exp_io {
            return Err("chain endpoint public_io_digest disagrees with --public-io".into());
        }
    }

    let per_block_verify_ms: Vec<f64> = per_block_samples
        .into_iter()
        .map(|samples| {
            let slice = if samples.len() > 1 {
                samples[1..].to_vec()
            } else {
                samples
            };
            (median_f64(slice) * 1000.0).round() / 1000.0
        })
        .collect();

    let st0 = &decoded_proofs[0].statement;
    Ok(StandaloneChainCommandReport {
        chain_anchor,
        initial_chain: manifest.initial_chain,
        verdict,
        receipts_sha256,
        total_receipt_bytes,
        per_block_receipt_bytes: per_block_bytes,
        per_block_verify_ms,
        num_blocks: chained_blocks.len(),
        total_layers,
        profile_id: st0.profile_id.clone(),
        mode_tag: st0.mode_tag,
        zk_trace_blinded: st0.zk_trace_blinded,
        batch: st0.batch,
        cold_ms,
        warm_ms,
    })
}

/// A model bundle: receipts `[embedding, blocks.., head]` with the public statement
/// that the manifest carries.
pub const STANDALONE_MODEL_MANIFEST_MAGIC: &[u8; 8] = b"STWOMDL2";

/// The `STWOMDL2` manifest: the initial chain link, the tokens, the output token, then one line per
/// receipt as in `STWOCHN2` (an absent boundary is written as 64 zeros).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneModelManifest {
    pub initial_chain: StandaloneChainLink,
    pub tokens: Vec<u32>,
    pub out_token: u32,
    pub receipts: Vec<StandaloneChainManifestEntry>,
}

fn csv_u32(v: &[u32]) -> String {
    v.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
}

pub fn encode_standalone_model_manifest(m: &StandaloneModelManifest) -> Result<String, String> {
    if m.receipts.is_empty() || m.receipts.len() > MAX_STANDALONE_CHAIN_BLOCKS {
        return Err(format!("model manifest needs 1..={MAX_STANDALONE_CHAIN_BLOCKS} receipts"));
    }
    if m.tokens.is_empty() || m.tokens.len() > 1 << 20 {
        return Err("model manifest needs 1..=2^20 tokens".into());
    }
    let mut out = String::from("STWOMDL2\n");
    out.push_str(&format!(
        "initial_chain {}:{}\n",
        m.initial_chain.layer,
        hex32(&m.initial_chain.prev_receipt_digest)
    ));
    out.push_str(&format!("tokens {}\n", csv_u32(&m.tokens)));
    out.push_str(&format!("out_token {}\n", m.out_token));
    for (k, r) in m.receipts.iter().enumerate() {
        if r.block_index != k {
            return Err(format!("model manifest receipt index {} != expected {k}", r.block_index));
        }
        validate_relative_receipt_subpath(&r.receipt_path)?;
        out.push_str(&format!(
            "receipt {} {} {} {} {}\n",
            r.block_index,
            hex32(&r.standalone_anchor),
            hex32(&r.com_h_in),
            hex32(&r.com_h_out),
            r.receipt_path
        ));
    }
    Ok(out)
}

/// Decode a `STWOMDL2` manifest. The lines come in one order and the bytes must re-encode exactly.
pub fn decode_standalone_model_manifest(bytes: &[u8]) -> Result<StandaloneModelManifest, String> {
    if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOMDL2 manifest exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    if !bytes.starts_with(STANDALONE_MODEL_MANIFEST_MAGIC) {
        return Err("invalid STWOMDL2 manifest magic".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "STWOMDL2 manifest is not valid UTF-8".to_string())?;
    let mut lines = text.lines();
    if lines.next() != Some("STWOMDL2") {
        return Err("STWOMDL2 header line must be exact 'STWOMDL2'".into());
    }
    let mut field = |name: &str| -> Result<String, String> {
        let line = lines.next().ok_or_else(|| format!("STWOMDL2 manifest ends before '{name}'"))?;
        line.strip_prefix(name)
            .and_then(|rest| rest.strip_prefix(' '))
            .map(|v| v.to_string())
            .ok_or_else(|| format!("STWOMDL2 manifest: expected '{name} ...', got '{line}'"))
    };
    let (layer, prev_receipt_digest) = parse_chain_link_spec(&field("initial_chain")?)?;
    let tokens = field("tokens")?
        .split(',')
        .map(|t| t.parse::<u32>().map_err(|_| format!("STWOMDL2: invalid token '{t}'")))
        .collect::<Result<Vec<u32>, String>>()?;
    let out_token: u32 = field("out_token")?
        .parse()
        .map_err(|_| "STWOMDL2: invalid out_token".to_string())?;
    let mut receipts = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(' ').collect();
        let ["receipt", idx_s, anchor_s, com_in_s, com_out_s, rel_path] = parts.as_slice() else {
            return Err(format!("STWOMDL2: unrecognized line '{line}'"));
        };
        if receipts.len() >= MAX_STANDALONE_CHAIN_BLOCKS {
            return Err(format!("STWOMDL2 exceeds {MAX_STANDALONE_CHAIN_BLOCKS} receipts"));
        }
        let block_index: usize = idx_s.parse().map_err(|_| format!("STWOMDL2: invalid receipt index '{idx_s}'"))?;
        validate_relative_receipt_subpath(rel_path)?;
        receipts.push(StandaloneChainManifestEntry {
            block_index,
            standalone_anchor: parse_hex32(anchor_s)?,
            com_h_in: parse_hex32(com_in_s)?,
            com_h_out: parse_hex32(com_out_s)?,
            receipt_path: (*rel_path).to_string(),
        });
    }
    let manifest = StandaloneModelManifest {
        initial_chain: StandaloneChainLink { layer, prev_receipt_digest },
        tokens,
        out_token,
        receipts,
    };
    if encode_standalone_model_manifest(&manifest)?.as_bytes() != bytes {
        return Err("STWOMDL2 manifest bytes are non-canonical".into());
    }
    Ok(manifest)
}

/// A multi-token model bundle: the `STWOMDL2`
/// manifest with a `gen_tokens` line in place of `out_token`, for receipts registered with `multi_token` end stages.
/// The prompt (`tokens`, the `L` prompt tokens) and the `K` generated tokens are the prover's record only: the
/// `model` command checks them against the caller's `--tokens` and `--gen-tokens` and verifies the caller's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneModelGenManifest {
    pub initial_chain: StandaloneChainLink,
    pub prompt_tokens: Vec<u32>,
    pub gen_tokens: Vec<u32>,
    pub receipts: Vec<StandaloneChainManifestEntry>,
}

/// The fourth line of a multi-token `STWOMDL2` manifest starts with this; a single-token one has `out_token `.
const MODEL_GEN_TOKENS_LINE: &str = "gen_tokens ";

pub fn encode_standalone_model_gen_manifest(m: &StandaloneModelGenManifest) -> Result<String, String> {
    if m.prompt_tokens.is_empty() || m.gen_tokens.is_empty() || m.prompt_tokens.len() + m.gen_tokens.len() > 1 << 20 {
        return Err("multi-token model manifest needs >= 1 prompt token, >= 1 generated token, at most 2^20 in all".into());
    }
    // The receipt lines are the single-token manifest's: encode one with a placeholder output and keep its tail.
    let single = StandaloneModelManifest {
        initial_chain: m.initial_chain,
        tokens: m.prompt_tokens.clone(),
        out_token: 0,
        receipts: m.receipts.clone(),
    };
    let text = encode_standalone_model_manifest(&single)?;
    let (head, tail) = text.split_once("\nout_token 0\n").ok_or("model manifest layout")?;
    Ok(format!("{head}\n{MODEL_GEN_TOKENS_LINE}{}\n{tail}", csv_u32(&m.gen_tokens)))
}

/// Whether `bytes` is a multi-token `STWOMDL2` manifest (its fourth line is `gen_tokens ...`).
pub fn is_standalone_model_gen_manifest(bytes: &[u8]) -> bool {
    bytes.starts_with(STANDALONE_MODEL_MANIFEST_MAGIC)
        && std::str::from_utf8(bytes).is_ok_and(|s| s.lines().nth(3).is_some_and(|l| l.starts_with(MODEL_GEN_TOKENS_LINE)))
}

/// Decode a multi-token `STWOMDL2` manifest: the single-token decoder on the same bytes with the `gen_tokens` line
/// swapped for a placeholder `out_token 0` line, so the two forms share one parser; the bytes must re-encode exactly.
pub fn decode_standalone_model_gen_manifest(bytes: &[u8]) -> Result<StandaloneModelGenManifest, String> {
    if !is_standalone_model_gen_manifest(bytes) {
        return Err("not a multi-token STWOMDL2 manifest (no gen_tokens line)".into());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "STWOMDL2 manifest is not valid UTF-8".to_string())?;
    let line = text.lines().nth(3).expect("checked above");
    let gen_tokens = line[MODEL_GEN_TOKENS_LINE.len()..]
        .split(',')
        .map(|t| t.parse::<u32>().map_err(|_| format!("STWOMDL2: invalid generated token '{t}'")))
        .collect::<Result<Vec<u32>, String>>()?;
    let single = text.replacen(&format!("\n{line}\n"), "\nout_token 0\n", 1);
    let m = decode_standalone_model_manifest(single.as_bytes())?;
    let manifest = StandaloneModelGenManifest {
        initial_chain: m.initial_chain,
        prompt_tokens: m.tokens,
        gen_tokens,
        receipts: m.receipts,
    };
    if encode_standalone_model_gen_manifest(&manifest)?.as_bytes() != bytes {
        return Err("STWOMDL2 manifest bytes are non-canonical".into());
    }
    Ok(manifest)
}

/// Read and decode the receipts a manifest names, under `base_dir`: unique paths, contained in the
/// directory, no links, each within the wire limit, the total within `MAX_CHAIN_TOTAL_RECEIPT_BYTES`,
/// canonical encodings. Returns the receipts, their sizes and a SHA-256 over all of them.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn read_manifest_receipts(
    entries: &[StandaloneChainManifestEntry],
    base_dir: &std::path::Path,
) -> Result<(Vec<StandaloneBlockStarkProof>, Vec<usize>, usize, [u8; 32]), String> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::with_capacity(entries.len());
    let mut proofs = Vec::with_capacity(entries.len());
    let mut sizes = Vec::with_capacity(entries.len());
    let mut total = 0usize;
    let mut hasher = sha2::Sha256::new();
    for entry in entries {
        if !seen.insert(entry.receipt_path.as_str()) {
            return Err(format!("duplicate receipt_path '{}' in manifest", entry.receipt_path));
        }
        let rec_path = resolve_contained_receipt_path(base_dir, &entry.receipt_path)
            .map_err(|e| format!("invalid receipt {} path '{}': {e}", entry.block_index, entry.receipt_path))?;
        let rec_bytes = read_capped_regular(&rec_path, MAX_WIRE_RECEIPT_BYTES)
            .map_err(|e| format!("failed to read receipt {} {}: {e}", entry.block_index, rec_path.display()))?;
        total = match total.checked_add(rec_bytes.len()) {
            Some(t) if t <= MAX_CHAIN_TOTAL_RECEIPT_BYTES => t,
            _ => return Err(format!("cumulative receipt size exceeds {MAX_CHAIN_TOTAL_RECEIPT_BYTES} bytes")),
        };
        hasher.update((entry.block_index as u64).to_le_bytes());
        hasher.update((rec_bytes.len() as u64).to_le_bytes());
        hasher.update(&rec_bytes);
        let proof = decode_standalone_block_receipt(&rec_bytes)
            .map_err(|r| format!("receipt {} decode failed: {r}", entry.block_index))?;
        sizes.push(rec_bytes.len());
        proofs.push(proof);
    }
    Ok((proofs, sizes, total, hasher.finalize().into()))
}

/// Flag values of the `model` command of `ummon-verify`.
#[derive(Clone, Copy, Debug, Default)]
pub struct StandaloneModelCommandArgs<'a> {
    /// `--model-anchor` (`STWO_ML_MODEL_ANCHOR`), required.
    pub model_anchor: Option<&'a str>,
    /// `--initial-chain` (`STWO_ML_INITIAL_CHAIN`), required.
    pub initial_chain: Option<&'a str>,
    /// `--tokens` (`STWO_ML_TOKENS`), required: the caller's prompt tokens, which must be the manifest's.
    /// Without them the command would verify only the prover's own statement.
    pub tokens: Option<&'a str>,
    /// `--out-token` (`STWO_ML_OUT_TOKEN`), required for a single-token bundle: the output token the caller was given.
    pub out_token: Option<&'a str>,
    /// `--gen-tokens` (`STWO_ML_GEN_TOKENS`), required for a multi-token bundle and refused for a single-token one: the
    /// `K` generated tokens the caller was given (`--tokens` is then the `L` prompt tokens).
    pub gen_tokens: Option<&'a str>,
    /// `--profile` (`STWO_ML_PROFILE`), optional: the security profile the caller requires; every receipt must be
    /// proved under exactly this profile (one of `STANDALONE_PROFILE_IDS`), weaker or not.
    /// The profile id is a label: the inner parameters (the CPRS query count, the query and fold grinding bits) are
    /// fixed by the registered circuit, so the caller's anchor, not this pin, fixes them (8 and 16 bits of fold
    /// grinding both carry `production_hardened`).
    pub profile: Option<&'a str>,
    pub bench_runs: usize,
}

#[derive(Clone, Debug)]
pub struct StandaloneModelCommandReport {
    /// For a multi-token bundle: `tokens` the prompt and `out_token` the last generated token; `gen_tokens` has all.
    pub verdict: StandaloneModelVerdict,
    /// The `K` generated tokens of a multi-token bundle; `None` for a single-token one.
    pub gen_tokens: Option<Vec<u32>>,
    pub initial_chain: StandaloneChainLink,
    pub receipts_sha256: [u8; 32],
    pub total_receipt_bytes: usize,
    pub per_receipt_bytes: Vec<usize>,
    pub per_receipt_verify_ms: Vec<f64>,
    pub num_receipts: usize,
    pub total_layers: usize,
    pub vocab_size: usize,
    pub profile_id: String,
    pub mode_tag: u32,
    pub zk_trace_blinded: bool,
    pub batch: usize,
    pub cold_ms: f64,
    pub warm_ms: f64,
}

/// The `model` command: decode a `STWOMDL2` manifest, check the caller's model anchor, initial link,
/// tokens and output token (and profile id, when given) before reading any receipt, then verify the
/// model chain. All of them come from the caller: the manifest is the prover's, and a statement taken
/// from it would be checked against nothing.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub fn verify_standalone_model_command(
    manifest_bytes: &[u8],
    base_dir: &std::path::Path,
    args: &StandaloneModelCommandArgs<'_>,
) -> Result<StandaloneModelCommandReport, String> {
    if is_standalone_model_gen_manifest(manifest_bytes) {
        return verify_standalone_model_gen_command(manifest_bytes, base_dir, args);
    }
    if args.gen_tokens.is_some() {
        return Err("--gen-tokens is for a multi-token bundle; this bundle is single-token: give its --out-token".into());
    }
    let manifest = decode_standalone_model_manifest(manifest_bytes)?;
    let model_anchor = args
        .model_anchor
        .ok_or_else(|| "missing --model-anchor <hex32> (or STWO_ML_MODEL_ANCHOR) for STWOMDL2".to_string())
        .and_then(parse_hex32)?;
    if model_anchor == [0u8; 32] {
        return Err("expected_model_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = manifest.receipts.iter().map(|r| r.standalone_anchor).collect();
    if compute_standalone_model_anchor(&anchors) != model_anchor {
        return Err("standalone model: 32B model anchor mismatch (a receipt anchor, the receipt count or the order differs from registration)".into());
    }
    let (init_layer, init_prev) = args
        .initial_chain
        .ok_or_else(|| "missing --initial-chain <layer:hex32> for STWOMDL2".to_string())
        .and_then(parse_chain_link_spec)?;
    if manifest.initial_chain.layer != init_layer || manifest.initial_chain.prev_receipt_digest != init_prev {
        return Err("manifest initial_chain disagrees with --initial-chain".into());
    }
    let toks = args
        .tokens
        .ok_or_else(|| "missing --tokens <csv> (or STWO_ML_TOKENS) for STWOMDL2: the caller's prompt, never the manifest's alone".to_string())?
        .split(',')
        .map(|x| x.trim().parse::<u32>().map_err(|_| format!("invalid --tokens entry '{x}'")))
        .collect::<Result<Vec<u32>, String>>()?;
    if toks != manifest.tokens {
        return Err("manifest tokens disagree with --tokens".into());
    }
    let y = args
        .out_token
        .ok_or_else(|| "missing --out-token <u32> (or STWO_ML_OUT_TOKEN) for STWOMDL2: the output the caller was given".to_string())?;
    let y: u32 = y.trim().parse().map_err(|_| format!("invalid --out-token '{y}'"))?;
    if y != manifest.out_token {
        return Err("manifest out_token disagrees with --out-token".into());
    }
    if let Some(p) = args.profile {
        if !STANDALONE_PROFILE_IDS.contains(&p) {
            return Err(format!("--profile '{p}' is not a profile id of STWOREC2"));
        }
    }
    let (proofs, per_receipt_bytes, total_receipt_bytes, receipts_sha256) =
        read_manifest_receipts(&manifest.receipts, base_dir)?;
    if let Some(p) = args.profile {
        if let Some((k, proof)) = proofs.iter().enumerate().find(|(_, pr)| pr.statement.profile_id != p) {
            return Err(format!(
                "receipt {k} is proved under profile '{}', not the profile '{p}' the caller requires (--profile)",
                proof.statement.profile_id
            ));
        }
    }
    let chained: Vec<StandaloneChainedBlock<'_>> = manifest
        .receipts
        .iter()
        .zip(proofs.iter())
        .map(|(e, proof)| StandaloneChainedBlock {
            standalone_anchor: &e.standalone_anchor,
            proof,
            com_h_in: e.com_h_in,
            com_h_out: e.com_h_out,
        })
        .collect();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(args.bench_runs.max(1)); chained.len()];
    let (verdict, cold_ms, warm_ms) = run_timed_verifies(args.bench_runs, || {
        let t0 = std::time::Instant::now();
        let (v, ms) = verify_standalone_model_chain_against_anchor_timed(
            &model_anchor,
            &manifest.initial_chain,
            &chained,
            &manifest.tokens,
            manifest.out_token,
        )?;
        for (dst, &m) in samples.iter_mut().zip(ms.iter()) {
            dst.push(m);
        }
        Ok((v, t0.elapsed().as_secs_f64() * 1000.0))
    })?;
    let per_receipt_verify_ms = samples
        .into_iter()
        .map(|s| {
            let s = if s.len() > 1 { s[1..].to_vec() } else { s };
            (median_f64(s) * 1000.0).round() / 1000.0
        })
        .collect();
    let st0 = &proofs[0].statement;
    Ok(StandaloneModelCommandReport {
        verdict,
        gen_tokens: None,
        initial_chain: manifest.initial_chain,
        receipts_sha256,
        total_receipt_bytes,
        per_receipt_bytes,
        per_receipt_verify_ms,
        num_receipts: proofs.len(),
        total_layers: proofs.iter().map(|p| p.statement.num_layers).sum(),
        vocab_size: st0.ends.map_or(0, |e| e.vocab_size),
        profile_id: st0.profile_id.clone(),
        mode_tag: st0.mode_tag,
        zk_trace_blinded: st0.zk_trace_blinded,
        batch: st0.batch,
        cold_ms,
        warm_ms,
    })
}

/// A caller's token list (`--tokens`, `--gen-tokens`): required, comma-separated `u32`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn caller_token_csv(value: Option<&str>, flag: &str, env: &str, what: &str) -> Result<Vec<u32>, String> {
    value
        .ok_or_else(|| format!("missing {flag} <csv> (or {env}) for a multi-token STWOMDL2: {what}, never the manifest's alone"))?
        .split(',')
        .map(|x| x.trim().parse::<u32>().map_err(|_| format!("invalid {flag} entry '{x}'")))
        .collect()
}

/// The `model` command on a multi-token bundle (`StandaloneModelGenManifest`): the caller's model anchor, initial link,
/// `L` prompt tokens (`--tokens`) and `K` generated tokens (`--gen-tokens`), and profile when given, are checked
/// before any receipt is read, then `verify_standalone_model_gen_chain_against_anchor` verifies the caller's
/// statement. `--out-token` is refused (it is the single-token form), and so is a manifest whose end stages are not
/// multi-token.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn verify_standalone_model_gen_command(
    manifest_bytes: &[u8],
    base_dir: &std::path::Path,
    args: &StandaloneModelCommandArgs<'_>,
) -> Result<StandaloneModelCommandReport, String> {
    if args.out_token.is_some() {
        return Err("--out-token is for a single-token bundle; this bundle is multi-token: give its --gen-tokens".into());
    }
    let manifest = decode_standalone_model_gen_manifest(manifest_bytes)?;
    let model_anchor = args
        .model_anchor
        .ok_or_else(|| "missing --model-anchor <hex32> (or STWO_ML_MODEL_ANCHOR) for STWOMDL2".to_string())
        .and_then(parse_hex32)?;
    if model_anchor == [0u8; 32] {
        return Err("expected_model_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = manifest.receipts.iter().map(|r| r.standalone_anchor).collect();
    if compute_standalone_model_anchor(&anchors) != model_anchor {
        return Err("standalone model: 32B model anchor mismatch (a receipt anchor, the receipt count or the order differs from registration)".into());
    }
    let (init_layer, init_prev) = args
        .initial_chain
        .ok_or_else(|| "missing --initial-chain <layer:hex32> for STWOMDL2".to_string())
        .and_then(parse_chain_link_spec)?;
    if manifest.initial_chain.layer != init_layer || manifest.initial_chain.prev_receipt_digest != init_prev {
        return Err("manifest initial_chain disagrees with --initial-chain".into());
    }
    let prompt = caller_token_csv(args.tokens, "--tokens", "STWO_ML_TOKENS", "the caller's L prompt tokens")?;
    if prompt != manifest.prompt_tokens {
        return Err("manifest tokens disagree with --tokens".into());
    }
    let gen = caller_token_csv(args.gen_tokens, "--gen-tokens", "STWO_ML_GEN_TOKENS", "the K generated tokens the caller was given")?;
    if gen != manifest.gen_tokens {
        return Err("manifest gen_tokens disagree with --gen-tokens".into());
    }
    if let Some(p) = args.profile {
        if !STANDALONE_PROFILE_IDS.contains(&p) {
            return Err(format!("--profile '{p}' is not a profile id of STWOREC2"));
        }
    }
    let (proofs, per_receipt_bytes, total_receipt_bytes, receipts_sha256) =
        read_manifest_receipts(&manifest.receipts, base_dir)?;
    if let Some(p) = args.profile {
        if let Some((k, proof)) = proofs.iter().enumerate().find(|(_, pr)| pr.statement.profile_id != p) {
            return Err(format!(
                "receipt {k} is proved under profile '{}', not the profile '{p}' the caller requires (--profile)",
                proof.statement.profile_id
            ));
        }
    }
    if let Some(k) = proofs.iter().position(|pr| pr.statement.ends.is_some_and(|e| !e.multi_token)) {
        return Err(format!("receipt {k} has a single-token end stage; a single-token bundle is verified with --out-token"));
    }
    let chained: Vec<StandaloneChainedBlock<'_>> = manifest
        .receipts
        .iter()
        .zip(proofs.iter())
        .map(|(e, proof)| StandaloneChainedBlock { standalone_anchor: &e.standalone_anchor, proof, com_h_in: e.com_h_in, com_h_out: e.com_h_out })
        .collect();
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(args.bench_runs.max(1)); chained.len()];
    let (verdict, cold_ms, warm_ms) = run_timed_verifies(args.bench_runs, || {
        let t0 = std::time::Instant::now();
        let (v, ms) = verify_standalone_model_gen_chain_against_anchor_timed(&model_anchor, &manifest.initial_chain, &chained, &prompt, &gen)?;
        for (dst, &m) in samples.iter_mut().zip(ms.iter()) {
            dst.push(m);
        }
        Ok((v, t0.elapsed().as_secs_f64() * 1000.0))
    })?;
    let per_receipt_verify_ms = samples
        .into_iter()
        .map(|s| (median_f64(if s.len() > 1 { s[1..].to_vec() } else { s }) * 1000.0).round() / 1000.0)
        .collect();
    let st0 = &proofs[0].statement;
    let out_token = *verdict.gen_tokens.last().expect("K >= 1");
    Ok(StandaloneModelCommandReport {
        verdict: StandaloneModelVerdict { model_anchor: verdict.model_anchor, tokens: verdict.prompt_tokens, out_token, final_chain: verdict.final_chain },
        gen_tokens: Some(verdict.gen_tokens),
        initial_chain: manifest.initial_chain,
        receipts_sha256,
        total_receipt_bytes,
        per_receipt_bytes,
        per_receipt_verify_ms,
        num_receipts: proofs.len(),
        total_layers: proofs.iter().map(|p| p.statement.num_layers).sum(),
        vocab_size: st0.ends.map_or(0, |e| e.vocab_size),
        profile_id: st0.profile_id.clone(),
        mode_tag: st0.mode_tag,
        zk_trace_blinded: st0.zk_trace_blinded,
        batch: st0.batch,
        cold_ms,
        warm_ms,
    })
}

pub const CROSS_LAYER_BLOCK_BUNDLE_MAGIC: &[u8; 8] = b"STWOBLKB";
pub const CROSS_LAYER_BLOCK_BUNDLE_VERSION: u32 = 3;
/// Domain of the digest a caller pins a `STWOBLKB` bundle with (`--anchor`).
pub const CROSS_LAYER_BLOCK_ANCHORS_DOMAIN: &[u8] = b"SPARSEPROVE_STWOBLKB_ANCHORS_V1";

/// `Blake2s(STWOBLKB_ANCHORS_V1 || k || anchor_0 .. anchor_{k-1})` over a `STWOBLKB` bundle's per-layer anchors, in
/// order: the one value a caller pins the bundle with. The bundle carries those anchors itself, so without it the
/// command would check the bundle only against the prover's own anchors
/// (the ones the bundle carries).
pub fn compute_cross_layer_block_anchors_digest(anchors: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(CROSS_LAYER_BLOCK_ANCHORS_DOMAIN);
    h.update((anchors.len() as u64).to_le_bytes());
    for anchor in anchors {
        h.update(anchor);
    }
    h.finalize().into()
}
const MAX_BUNDLE_PARTIAL_ROWS_CELLS: usize = 256 * 1024 * 1024;

/// Self-contained CLI/WASM verification bundle wrapping an M3 `STWOBLK1` cross-layer block receipt
/// together with its per-layer `StageARegisteredContext`, optional stacked expert bank anchors,
/// `StageANonlinearBatchStatement`, optional Stage B attention commitments/policies/statements, and
/// initial chain link.
#[derive(Clone, Debug)]
pub struct CrossLayerBlockVerificationBundle {
    pub profile: SecurityProfile,
    pub policy: StageANumericPolicy,
    pub attention_policy: AttentionPolicy,
    pub expected_anchors: Vec<[u8; 32]>,
    pub contexts: Vec<StageARegisteredContext>,
    pub stacked_bank_commitments: Option<Vec<[u8; 32]>>,
    pub stage_a_statements: Vec<StageANonlinearBatchStatement>,
    pub layer_inputs: Option<Vec<Vec<Vec<i16>>>>,
    pub stage_b: Option<(Vec<[u8; 32]>, Vec<AttentionPolicy>, Vec<StageBStatement>)>,
    pub initial_chain: StageAChainLink,
    pub block_receipt: Vec<u8>,
}

fn put_bcn_stage_b_stmt_versioned(
    out: &mut Vec<u8>,
    a: &StageBStatement,
    ver: u32,
) -> Result<(), String> {
    if a.partial_rows.len() != a.tokens.len() {
        return Err("Stage B statement partial_rows length mismatch".into());
    }
    for value in [
        a.shape.dimension,
        a.shape.heads,
        a.shape.kv_heads,
        a.shape.head_dim,
        a.layout.chunk,
    ] {
        put_u64(out, value)?;
    }
    put_bcn_label(out, &a.policy_id)?;
    out.extend_from_slice(&a.attention_commitment);
    out.extend_from_slice(&a.binding);
    put_bcn_i16_vec(out, &a.gamma1)?;
    put_bcn_i16_vec(out, &a.gamma2)?;
    put_bcn_digests(out, &a.chunk_commitments)?;
    put_u64(out, a.tokens.len())?;
    for (t, (token, (k_rows, v_rows))) in a.tokens.iter().zip(&a.partial_rows).enumerate() {
        put_raw_u64(out, token.step.position);
        put_bcn_i16_vec(out, &token.step.h_in)?;
        put_bcn_i32_vec(out, &token.step.q_raw)?;
        put_bcn_i32_vec(out, &token.step.k_raw)?;
        put_bcn_i32_vec(out, &token.step.v_raw)?;
        put_bcn_i32_rows(out, &token.step.scores)?;
        put_bcn_i32_vec(out, &token.step.ctx_raw)?;
        put_bcn_i32_vec(out, &token.step.o_raw)?;
        if token.step.q_raw.is_empty() {
            put_bcn_i16_vec(out, &token.step.a)?;
            put_bcn_i16_vec(out, &token.step.x)?;
        }
        put_u64(out, token.chunk_ids.len())?;
        for &cid in &token.chunk_ids {
            put_u64(out, cid)?;
        }
        out.extend_from_slice(&token.cache_commitment_before);
        out.extend_from_slice(&token.cache_commitment_after);
        if ver >= 3 {
            let reuse_prefix = if t > 0 {
                let (prev_k, prev_v) = &a.partial_rows[t - 1];
                prev_k.len() <= k_rows.len()
                    && prev_v.len() <= v_rows.len()
                    && k_rows[..prev_k.len()] == prev_k[..]
                    && v_rows[..prev_v.len()] == prev_v[..]
            } else {
                false
            };
            if reuse_prefix {
                let (prev_k, prev_v) = &a.partial_rows[t - 1];
                out.push(1);
                put_bcn_i8_rows(out, &k_rows[prev_k.len()..])?;
                put_bcn_i8_rows(out, &v_rows[prev_v.len()..])?;
            } else {
                out.push(0);
                put_bcn_i8_rows(out, k_rows)?;
                put_bcn_i8_rows(out, v_rows)?;
            }
        } else {
            put_bcn_i8_rows(out, k_rows)?;
            put_bcn_i8_rows(out, v_rows)?;
        }
    }
    Ok(())
}

fn read_bcn_stage_b_stmt_versioned(
    reader: &mut ReceiptReader<'_>,
    ver: u32,
    reconstructed_cells: &mut usize,
) -> Result<StageBStatement, String> {
    let b_shape = AttentionShape {
        dimension: reader.read_len(MAX_BUNDLE_ROWS, "Stage B dimension")?,
        heads: reader.read_len(MAX_BUNDLE_ROWS, "Stage B heads")?,
        kv_heads: reader.read_len(MAX_BUNDLE_ROWS, "Stage B kv_heads")?,
        head_dim: reader.read_len(MAX_BUNDLE_ROWS, "Stage B head_dim")?,
    };
    let b_layout = CacheLayout {
        chunk: reader.read_len(MAX_BUNDLE_ROWS, "Stage B chunk")?,
    };
    let b_policy_id = read_bcn_label(reader, "Stage B policy_id")?.to_string();
    let attention_commitment = read_bcn_digest(reader, "Stage B attention_commitment")?;
    let binding = read_bcn_digest(reader, "Stage B binding")?;
    let gamma1 = read_bcn_vec(reader, "Stage B gamma1", read_bcn_i16)?;
    let gamma2 = read_bcn_vec(reader, "Stage B gamma2", read_bcn_i16)?;
    let chunk_commitments = read_bcn_digests(reader, MAX_BUNDLE_ROWS, "Stage B chunk_commitments")?;
    let token_count = reader.read_len(MAX_BUNDLE_ROWS, "Stage B token count")?;
    let mut tokens = Vec::with_capacity(token_count);
    let mut partial_rows: Vec<(Vec<Vec<i8>>, Vec<Vec<i8>>)> = Vec::with_capacity(token_count);
    for t in 0..token_count {
        let position = reader.read_u64(&format!("Stage B token {t} position"))?;
        let h_in = read_bcn_vec(reader, &format!("Stage B token {t} h_in"), read_bcn_i16)?;
        let q_raw = read_bcn_vec(reader, &format!("Stage B token {t} q_raw"), read_bcn_i32)?;
        let k_raw = read_bcn_vec(reader, &format!("Stage B token {t} k_raw"), read_bcn_i32)?;
        let v_raw = read_bcn_vec(reader, &format!("Stage B token {t} v_raw"), read_bcn_i32)?;
        let scores = read_bcn_rows(reader, &format!("Stage B token {t} scores"), read_bcn_i32)?;
        let ctx_raw = read_bcn_vec(reader, &format!("Stage B token {t} ctx_raw"), read_bcn_i32)?;
        let o_raw = read_bcn_vec(reader, &format!("Stage B token {t} o_raw"), read_bcn_i32)?;
        let (a, x) = if q_raw.is_empty() {
            let a = read_bcn_vec(reader, &format!("Stage B token {t} a"), read_bcn_i16)?;
            let x = read_bcn_vec(reader, &format!("Stage B token {t} x"), read_bcn_i16)?;
            (a, x)
        } else {
            (Vec::new(), Vec::new())
        };
        let chunk_ids = read_bcn_vec(reader, &format!("Stage B token {t} chunk_ids"), |r, l| {
            r.read_len(MAX_BUNDLE_ROWS, l)
        })?;
        let cache_commitment_before = read_bcn_digest(
            reader,
            &format!("Stage B token {t} cache_commitment_before"),
        )?;
        let cache_commitment_after =
            read_bcn_digest(reader, &format!("Stage B token {t} cache_commitment_after"))?;
        let reuse_flag = if ver >= 3 {
            reader.read_u32_byte(&format!("Stage B token {t} partial_rows reuse"))?
        } else {
            0
        };
        let (k_rows, v_rows) = match reuse_flag {
            0 => {
                let k = read_bcn_rows(reader, &format!("Stage B token {t} k_rows"), |r, l| {
                    r.read_u32_byte(l).map(|b| b as i8)
                })?;
                let v = read_bcn_rows(reader, &format!("Stage B token {t} v_rows"), |r, l| {
                    r.read_u32_byte(l).map(|b| b as i8)
                })?;
                let cells = k.iter().map(Vec::len).sum::<usize>()
                    + v.iter().map(Vec::len).sum::<usize>();
                *reconstructed_cells = reconstructed_cells
                    .checked_add(cells)
                    .filter(|&n| n <= MAX_BUNDLE_PARTIAL_ROWS_CELLS)
                    .ok_or_else(|| {
                        format!(
                            "Stage B reconstructed partial_rows exceeds {MAX_BUNDLE_PARTIAL_ROWS_CELLS} cell limit"
                        )
                    })?;
                (k, v)
            }
            1 => {
                if t == 0 {
                    return Err("Stage B token 0 cannot reuse prior partial_rows prefix".into());
                }
                let k_suffix =
                    read_bcn_rows(reader, &format!("Stage B token {t} k_rows suffix"), |r, l| {
                        r.read_u32_byte(l).map(|b| b as i8)
                    })?;
                let v_suffix =
                    read_bcn_rows(reader, &format!("Stage B token {t} v_rows suffix"), |r, l| {
                        r.read_u32_byte(l).map(|b| b as i8)
                    })?;
                let (prev_k, prev_v) = &partial_rows[t - 1];
                let next_k_len = prev_k
                    .len()
                    .checked_add(k_suffix.len())
                    .filter(|&n| n <= MAX_BUNDLE_ROWS)
                    .ok_or_else(|| format!("Stage B token {t} k_rows length exceeds limit"))?;
                let next_v_len = prev_v
                    .len()
                    .checked_add(v_suffix.len())
                    .filter(|&n| n <= MAX_BUNDLE_ROWS)
                    .ok_or_else(|| format!("Stage B token {t} v_rows length exceeds limit"))?;
                let prev_cells = prev_k.iter().map(Vec::len).sum::<usize>()
                    + prev_v.iter().map(Vec::len).sum::<usize>();
                let suffix_cells = k_suffix.iter().map(Vec::len).sum::<usize>()
                    + v_suffix.iter().map(Vec::len).sum::<usize>();
                let token_cells = prev_cells
                    .checked_add(suffix_cells)
                    .ok_or_else(|| "Stage B partial_rows cell count overflow".to_string())?;
                *reconstructed_cells = reconstructed_cells
                    .checked_add(token_cells)
                    .filter(|&n| n <= MAX_BUNDLE_PARTIAL_ROWS_CELLS)
                    .ok_or_else(|| {
                        format!(
                            "Stage B reconstructed partial_rows exceeds {MAX_BUNDLE_PARTIAL_ROWS_CELLS} cell limit"
                        )
                    })?;
                let mut k = Vec::with_capacity(next_k_len);
                k.extend_from_slice(prev_k);
                k.extend(k_suffix);
                let mut v = Vec::with_capacity(next_v_len);
                v.extend_from_slice(prev_v);
                v.extend(v_suffix);
                (k, v)
            }
            other => {
                return Err(format!(
                    "invalid Stage B token {t} partial_rows reuse flag {other}"
                ));
            }
        };
        tokens.push(StageBToken {
            step: PublicStep {
                position,
                h_in,
                q_raw,
                k_raw,
                v_raw,
                scores,
                ctx_raw,
                o_raw,
                a,
                x,
            },
            chunk_ids,
            cache_commitment_before,
            cache_commitment_after,
        });
        partial_rows.push((k_rows, v_rows));
    }
    Ok(StageBStatement {
        shape: b_shape,
        layout: b_layout,
        policy_id: b_policy_id,
        attention_commitment,
        gamma1,
        gamma2,
        chunk_commitments,
        tokens,
        partial_rows,
        binding,
    })
}

pub fn cross_layer_block_bundle_bytes(
    b: &CrossLayerBlockVerificationBundle,
) -> Result<Vec<u8>, String> {
    cross_layer_block_bundle_bytes_with_version(b, CROSS_LAYER_BLOCK_BUNDLE_VERSION)
}

fn cross_layer_block_bundle_bytes_with_version(
    b: &CrossLayerBlockVerificationBundle,
    version: u32,
) -> Result<Vec<u8>, String> {
    if version != 2 && version != CROSS_LAYER_BLOCK_BUNDLE_VERSION {
        return Err(format!("unsupported STWOBLKB encode version {version}"));
    }
    let k = b.contexts.len();
    if k == 0 || b.expected_anchors.len() != k || b.stage_a_statements.len() != k {
        return Err("STWOBLKB requires non-empty matching layer slices".into());
    }
    if let Some(ref sb) = b.stacked_bank_commitments {
        if sb.len() != k {
            return Err("STWOBLKB stacked_bank_commitments length mismatch".into());
        }
    }
    if let Some(ref li) = b.layer_inputs {
        if li.len() != k {
            return Err("STWOBLKB layer_inputs length mismatch".into());
        }
    }
    if let Some((ref ac, ref ap, ref bst)) = b.stage_b {
        if ac.len() != k || ap.len() != k || bst.len() != k {
            return Err("STWOBLKB stage_b slice length mismatch".into());
        }
    }
    let mut out = Vec::with_capacity(1024 + b.block_receipt.len());
    out.extend_from_slice(CROSS_LAYER_BLOCK_BUNDLE_MAGIC);
    put_u32(&mut out, version);
    put_bytes(&mut out, b.profile.id().as_bytes())?;
    put_bytes(&mut out, b.policy.id().as_bytes())?;
    put_bytes(&mut out, b.attention_policy.id().as_bytes())?;
    put_u64(&mut out, k)?;
    match b.stacked_bank_commitments {
        None => out.push(0),
        Some(ref sb) => {
            out.push(1);
            for d in sb {
                out.extend_from_slice(d);
            }
        }
    }
    out.push(u8::from(b.layer_inputs.is_some()));
    out.push(u8::from(b.stage_b.is_some()));
    for l in 0..k {
        out.extend_from_slice(&b.expected_anchors[l]);
        let c = &b.contexts[l];
        put_bcn_shape(&mut out, c.shape)?;
        for d in [
            &c.model_digest,
            &c.manifest_digest,
            &c.scheme_digest,
            &c.router_commitment,
            &c.shared_commitment,
            &c.registered_expert_root,
        ] {
            out.extend_from_slice(d);
        }
        put_u64(&mut out, c.active_expert_openings.len())?;
        for op in &c.active_expert_openings {
            put_u64(&mut out, op.expert_id)?;
            out.extend_from_slice(&op.slot_roles);
            out.extend_from_slice(&op.group_commitment);
            put_bcn_digests(&mut out, &op.siblings)?;
        }
        let s = &b.stage_a_statements[l];
        put_bcn_shape(&mut out, s.shape)?;
        put_bytes(&mut out, s.policy_id.as_bytes())?;
        for d in [
            &s.policy_digest,
            &s.model_digest,
            &s.model_registry_manifest_digest,
            &s.table_digest,
            &s.input_digest,
        ] {
            out.extend_from_slice(d);
        }
        put_bcn_i64_rows(&mut out, &s.router_outputs_raw)?;
        put_bcn_usize_rows(&mut out, &s.routes)?;
        put_bcn_u16_rows(&mut out, &s.route_weights_q15)?;
        put_bcn_i16_rows(&mut out, &s.output_q8)?;
        put_u64(&mut out, s.wide_log_columns)?;
        if let Some(ref li) = b.layer_inputs {
            put_bcn_i16_rows(&mut out, &li[l])?;
        }
        if let Some((ref ac, ref ap, ref bst)) = b.stage_b {
            out.extend_from_slice(&ac[l]);
            put_bcn_label(&mut out, &ap[l].id())?;
            put_bcn_stage_b_stmt_versioned(&mut out, &bst[l], version)?;
        }
    }
    put_u32(&mut out, b.initial_chain.layer);
    out.extend_from_slice(&b.initial_chain.prev_receipt_digest);
    put_bytes(&mut out, &b.block_receipt)?;
    Ok(out)
}

pub fn decode_cross_layer_block_bundle(
    bytes: &[u8],
) -> Result<CrossLayerBlockVerificationBundle, String> {
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(CROSS_LAYER_BLOCK_BUNDLE_MAGIC, "STWOBLKB magic")?;
    let ver = reader.read_u32("STWOBLKB version")?;
    if ver != 1 && ver != 2 && ver != CROSS_LAYER_BLOCK_BUNDLE_VERSION {
        return Err(format!("unsupported STWOBLKB version {ver}"));
    }
    let profile = SecurityProfile::from_id(read_bcn_label(&mut reader, "profile")?)?;
    let policy = bcn_policy_from_id(read_bcn_label(&mut reader, "policy")?)?;
    let attention_policy =
        AttentionPolicy::from_id(read_bcn_label(&mut reader, "attention_policy")?)?;
    let num_layers = reader.read_len(256, "num_layers")?;
    if num_layers == 0 {
        return Err("STWOBLKB num_layers must be >= 1".into());
    }
    let (stacked_bank_commitments, has_layer_inputs, has_stage_b) = if ver == 1 {
        (None, true, false)
    } else {
        let sb_opt = match reader.read_u32_byte("stacked_bank flag")? {
            0 => None,
            1 => {
                let mut sb = Vec::with_capacity(num_layers);
                for l in 0..num_layers {
                    sb.push(read_bcn_digest(&mut reader, &format!("stacked_com {l}"))?);
                }
                Some(sb)
            }
            other => return Err(format!("invalid STWOBLKB stacked_bank flag {other}")),
        };
        let li = match reader.read_u32_byte("layer_inputs flag")? {
            0 => false,
            1 => true,
            other => return Err(format!("invalid STWOBLKB layer_inputs flag {other}")),
        };
        let sb = match reader.read_u32_byte("stage_b flag")? {
            0 => false,
            1 => true,
            other => return Err(format!("invalid STWOBLKB stage_b flag {other}")),
        };
        (sb_opt, li, sb)
    };
    let mut expected_anchors = Vec::with_capacity(num_layers);
    let mut contexts = Vec::with_capacity(num_layers);
    let mut stage_a_statements = Vec::with_capacity(num_layers);
    let mut layer_inputs = if has_layer_inputs {
        Some(Vec::with_capacity(num_layers))
    } else {
        None
    };
    let mut stage_b = if has_stage_b {
        Some((
            Vec::with_capacity(num_layers),
            Vec::with_capacity(num_layers),
            Vec::with_capacity(num_layers),
        ))
    } else {
        None
    };
    let mut reconstructed_partial_cells = 0usize;
    for l in 0..num_layers {
        expected_anchors.push(read_bcn_digest(&mut reader, &format!("anchor {l}"))?);
        let shape = read_bcn_shape(&mut reader, &format!("ctx {l} shape"))?;
        let model_digest = read_bcn_digest(&mut reader, "model_digest")?;
        let manifest_digest = read_bcn_digest(&mut reader, "manifest_digest")?;
        let scheme_digest = read_bcn_digest(&mut reader, "scheme_digest")?;
        let router_commitment = read_bcn_digest(&mut reader, "router_commitment")?;
        let shared_commitment = read_bcn_digest(&mut reader, "shared_commitment")?;
        let registered_expert_root = read_bcn_digest(&mut reader, "registered_expert_root")?;
        let n_ops = reader.read_len(MAX_BUNDLE_ROWS, "openings")?;
        let mut active_expert_openings = Vec::with_capacity(n_ops);
        for idx in 0..n_ops {
            let expert_id = reader.read_len(MAX_BUNDLE_ROWS, &format!("op {idx} expert"))?;
            let mut slot_roles = [0u8; 3];
            for r in &mut slot_roles {
                *r = reader.read_u32_byte("slot_role")?;
            }
            let group_commitment = read_bcn_digest(&mut reader, "group_com")?;
            let siblings = read_bcn_digests(&mut reader, MAX_BUNDLE_SIBLINGS, "siblings")?;
            active_expert_openings.push(StageARegisteredExpertOpening {
                expert_id,
                slot_roles,
                group_commitment,
                siblings,
            });
        }
        contexts.push(StageARegisteredContext {
            shape,
            model_digest,
            manifest_digest,
            scheme_digest,
            router_commitment,
            shared_commitment,
            registered_expert_root,
            active_expert_openings,
        });
        let s_shape = read_bcn_shape(&mut reader, "stmt shape")?;
        let policy_id = read_bcn_label(&mut reader, "stmt policy_id")?.to_string();
        let policy_digest = read_bcn_digest(&mut reader, "policy_digest")?;
        let s_model_digest = read_bcn_digest(&mut reader, "s_model_digest")?;
        let model_registry_manifest_digest = read_bcn_digest(&mut reader, "s_manifest_digest")?;
        let table_digest = read_bcn_digest(&mut reader, "table_digest")?;
        let input_digest = read_bcn_digest(&mut reader, "input_digest")?;
        let router_outputs_raw = read_bcn_rows(&mut reader, "router_outputs", |r, lb| {
            r.read_u64(lb).map(|v| i64::from_le_bytes(v.to_le_bytes()))
        })?;
        let routes = read_bcn_rows(&mut reader, "routes", |r, lb| r.read_len(MAX_BUNDLE_ROWS, lb))?;
        let route_weights_q15 = read_bcn_rows(&mut reader, "weights_q15", read_bcn_u16)?;
        let output_q8 = read_bcn_rows(&mut reader, "output_q8", read_bcn_i16)?;
        let wide_log_columns = reader.read_len(MAX_BUNDLE_ROWS, "wide_log_columns")?;
        stage_a_statements.push(StageANonlinearBatchStatement {
            shape: s_shape,
            policy_id,
            policy_digest,
            model_digest: s_model_digest,
            model_registry_manifest_digest,
            table_digest,
            input_digest,
            router_outputs_raw,
            routes,
            route_weights_q15,
            output_q8,
            wide_log_columns,
        });
        if let Some(ref mut li) = layer_inputs {
            li.push(read_bcn_rows(&mut reader, "layer_inputs", read_bcn_i16)?);
        }
        if let Some((ref mut ac, ref mut ap, ref mut bst)) = stage_b {
            ac.push(read_bcn_digest(&mut reader, "attn_com")?);
            ap.push(AttentionPolicy::from_id(read_bcn_label(
                &mut reader,
                "layer attn_policy",
            )?)?);
            bst.push(read_bcn_stage_b_stmt_versioned(
                &mut reader,
                ver,
                &mut reconstructed_partial_cells,
            )?);
        }
    }
    let init_layer = reader.read_u32("initial_chain.layer")?;
    let init_prev = read_bcn_digest(&mut reader, "initial_chain.prev")?;
    let block_receipt = reader.read_bytes(MAX_WIRE_RECEIPT_BYTES, "block_receipt")?.to_vec();
    reader.finish()?;
    Ok(CrossLayerBlockVerificationBundle {
        profile,
        policy,
        attention_policy,
        expected_anchors,
        contexts,
        stacked_bank_commitments,
        stage_a_statements,
        layer_inputs,
        stage_b,
        initial_chain: StageAChainLink {
            layer: init_layer,
            prev_receipt_digest: init_prev,
        },
        block_receipt,
    })
}

pub fn verify_cross_layer_block_bundle_bytes(
    bytes: &[u8],
) -> Result<(CrossLayerBlockVerificationBundle, CrossLayerGkrBlockVerifierRun), String> {
    let b = decode_cross_layer_block_bundle(bytes)?;
    let sb_stmts = b.stage_b.as_ref().map(|(_, _, s)| s.as_slice());
    let block = decode_cross_layer_gkr_block(&b.block_receipt, &b.stage_a_statements, b.policy, sb_stmts)?;
    let sb_tuple = b
        .stage_b
        .as_ref()
        .map(|(ac, ap, bst)| (ac.as_slice(), ap.as_slice(), bst.as_slice()));
    let run = verify_cross_layer_gkr_block_with_policies(
        &b.expected_anchors,
        &b.contexts,
        b.stacked_bank_commitments.as_deref(),
        b.layer_inputs.as_deref(),
        b.policy,
        &b.profile,
        &b.stage_a_statements,
        sb_tuple,
        &block,
        &b.initial_chain,
    )?;
    Ok((b, run))
}

pub const MLA_BUNDLE_MAGIC: &[u8; 8] = b"STWOMLAB";
pub const MLA_BUNDLE_VERSION: u32 = 1;
pub const SWA_BUNDLE_MAGIC: &[u8; 8] = b"STWOSWAB";
pub const SWA_BUNDLE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct MlaVerificationBundle {
    pub profile: SecurityProfile,
    pub expected_anchor: [u8; 32],
    pub statement: StageBMlaStatement,
    pub receipt_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct SwaVerificationBundle {
    pub profile: SecurityProfile,
    pub expected_anchor: [u8; 32],
    pub statement: StageBWindowSinkStatement,
    pub receipt_bytes: Vec<u8>,
}

fn put_requant_spec_raw(out: &mut Vec<u8>, s: stage_a::RequantSpec) {
    out.extend_from_slice(&s.multiplier.to_le_bytes());
    put_u32(out, s.shift);
    put_u32(out, s.output_bits);
}

fn read_requant_spec_raw(
    reader: &mut ReceiptReader<'_>,
    label: &str,
) -> Result<stage_a::RequantSpec, String> {
    let raw_mult = reader.read_u64(label)?;
    let multiplier = i64::from_le_bytes(raw_mult.to_le_bytes());
    let shift = reader.read_u32(label)?;
    let output_bits = reader.read_u32(label)?;
    Ok(stage_a::RequantSpec {
        multiplier,
        shift,
        output_bits,
    })
}

pub fn mla_verification_bundle_bytes(b: &MlaVerificationBundle) -> Result<Vec<u8>, String> {
    let s = &b.statement;
    if s.partial_rows.len() != s.tokens.len() {
        return Err("STWOMLAB partial_rows length must equal tokens length".into());
    }
    let mut out = Vec::with_capacity(1024 + b.receipt_bytes.len());
    out.extend_from_slice(MLA_BUNDLE_MAGIC);
    put_u32(&mut out, MLA_BUNDLE_VERSION);
    put_bcn_label(&mut out, b.profile.id())?;
    out.extend_from_slice(&b.expected_anchor);
    for v in [
        s.shape.dimension,
        s.shape.heads,
        s.shape.q_lora_rank,
        s.shape.kv_lora_rank,
        s.shape.qk_rope_head_dim,
        s.shape.v_head_dim,
        s.layout.chunk,
        s.cprs_f.unwrap_or(0),
    ] {
        put_u64(&mut out, v)?;
    }
    let r_tag: u8 = match s.policy.rounding {
        stage_a::StageARoundingMode::TiesToEven => 0,
        stage_a::StageARoundingMode::HalfTowardPositiveInfinity => 1,
    };
    out.push(r_tag);
    for spec in [
        s.policy.dq_pre,
        s.policy.dkv_pre,
        s.policy.kr,
        s.policy.q_abs,
        s.policy.q_pe,
        s.policy.uv,
        s.policy.output,
    ] {
        put_requant_spec_raw(&mut out, spec);
    }
    put_u64(&mut out, s.policy.alpha_nope.len())?;
    for &a in &s.policy.alpha_nope {
        out.extend_from_slice(&a.to_le_bytes());
    }
    out.extend_from_slice(&s.policy.alpha_rope.to_le_bytes());
    put_u32(&mut out, s.policy.reconcile_shift);
    put_u32(&mut out, s.policy.score_shift);
    for fb in [
        s.policy.norm1_fraction_bits,
        s.policy.norm_q_fraction_bits,
        s.policy.norm_kv_fraction_bits,
        s.policy.norm2_fraction_bits,
    ] {
        out.extend_from_slice(&fb.to_le_bytes());
    }
    out.extend_from_slice(&s.mla_leaf_commitment);
    put_bcn_i16_vec(&mut out, &s.gamma1)?;
    put_bcn_i16_vec(&mut out, &s.gamma_q)?;
    put_bcn_i16_vec(&mut out, &s.gamma_kv)?;
    put_bcn_i16_vec(&mut out, &s.gamma2)?;
    put_bcn_digests(&mut out, &s.chunk_commitments)?;
    put_u64(&mut out, s.tokens.len())?;
    for (tok, (ckv_rows, kpe_rows)) in s.tokens.iter().zip(&s.partial_rows) {
        put_raw_u64(&mut out, tok.step.position);
        put_bcn_i16_vec(&mut out, &tok.step.h_in)?;
        put_bcn_i32_vec(&mut out, &tok.step.c_q_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.c_kv_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.k_pe_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.q_abs_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.q_pe_raw)?;
        put_bcn_i32_rows(&mut out, &tok.step.s_nope)?;
        put_bcn_i32_rows(&mut out, &tok.step.s_rope)?;
        put_bcn_i32_vec(&mut out, &tok.step.u_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.v_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.o_raw)?;
        put_u64(&mut out, tok.chunk_ids.len())?;
        for &cid in &tok.chunk_ids {
            put_u64(&mut out, cid)?;
        }
        out.extend_from_slice(&tok.cache_commitment_before);
        out.extend_from_slice(&tok.cache_commitment_after);
        put_bcn_i8_rows(&mut out, ckv_rows)?;
        put_bcn_i8_rows(&mut out, kpe_rows)?;
    }
    out.extend_from_slice(&s.binding);
    put_bytes(&mut out, &b.receipt_bytes)?;
    Ok(out)
}

pub fn decode_mla_verification_bundle(bytes: &[u8]) -> Result<MlaVerificationBundle, String> {
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(MLA_BUNDLE_MAGIC, "STWOMLAB magic")?;
    let ver = reader.read_u32("STWOMLAB version")?;
    if ver != MLA_BUNDLE_VERSION {
        return Err(format!("unsupported STWOMLAB version {ver}"));
    }
    let profile = SecurityProfile::from_id(read_bcn_label(&mut reader, "MLA profile")?)?;
    let expected_anchor = read_bcn_digest(&mut reader, "MLA expected_anchor")?;
    let shape = MlaShape {
        dimension: reader.read_len(MAX_BUNDLE_ROWS, "MLA dimension")?,
        heads: reader.read_len(MAX_BUNDLE_ROWS, "MLA heads")?,
        q_lora_rank: reader.read_len(MAX_BUNDLE_ROWS, "MLA q_lora_rank")?,
        kv_lora_rank: reader.read_len(MAX_BUNDLE_ROWS, "MLA kv_lora_rank")?,
        qk_rope_head_dim: reader.read_len(MAX_BUNDLE_ROWS, "MLA qk_rope_head_dim")?,
        v_head_dim: reader.read_len(MAX_BUNDLE_ROWS, "MLA v_head_dim")?,
    };
    let layout = CacheLayout {
        chunk: reader.read_len(MAX_BUNDLE_ROWS, "MLA chunk")?,
    };
    let raw_f = reader.read_len(MAX_BUNDLE_ROWS, "MLA cprs_f")?;
    let cprs_f = if raw_f == 0 { None } else { Some(raw_f) };
    let rounding = match reader.read_u32_byte("MLA rounding")? {
        0 => stage_a::StageARoundingMode::TiesToEven,
        1 => stage_a::StageARoundingMode::HalfTowardPositiveInfinity,
        other => return Err(format!("invalid MLA rounding tag {other}")),
    };
    let dq_pre = read_requant_spec_raw(&mut reader, "MLA dq_pre")?;
    let dkv_pre = read_requant_spec_raw(&mut reader, "MLA dkv_pre")?;
    let kr = read_requant_spec_raw(&mut reader, "MLA kr")?;
    let q_abs = read_requant_spec_raw(&mut reader, "MLA q_abs")?;
    let q_pe = read_requant_spec_raw(&mut reader, "MLA q_pe")?;
    let uv = read_requant_spec_raw(&mut reader, "MLA uv")?;
    let output = read_requant_spec_raw(&mut reader, "MLA output")?;
    let alpha_nope = read_bcn_vec(&mut reader, "MLA alpha_nope", read_bcn_u16)?;
    let alpha_rope = read_bcn_u16(&mut reader, "MLA alpha_rope")?;
    let reconcile_shift = reader.read_u32("MLA reconcile_shift")?;
    let score_shift = reader.read_u32("MLA score_shift")?;
    let norm1_fraction_bits = read_bcn_i32(&mut reader, "MLA norm1_fb")?;
    let norm_q_fraction_bits = read_bcn_i32(&mut reader, "MLA norm_q_fb")?;
    let norm_kv_fraction_bits = read_bcn_i32(&mut reader, "MLA norm_kv_fb")?;
    let norm2_fraction_bits = read_bcn_i32(&mut reader, "MLA norm2_fb")?;
    let policy = MlaAttentionPolicy {
        rounding,
        dq_pre,
        dkv_pre,
        kr,
        q_abs,
        q_pe,
        uv,
        output,
        alpha_nope,
        alpha_rope,
        reconcile_shift,
        score_shift,
        norm1_fraction_bits,
        norm_q_fraction_bits,
        norm_kv_fraction_bits,
        norm2_fraction_bits,
    };
    let mla_leaf_commitment = read_bcn_digest(&mut reader, "MLA leaf_commitment")?;
    let gamma1 = read_bcn_vec(&mut reader, "MLA gamma1", read_bcn_i16)?;
    let gamma_q = read_bcn_vec(&mut reader, "MLA gamma_q", read_bcn_i16)?;
    let gamma_kv = read_bcn_vec(&mut reader, "MLA gamma_kv", read_bcn_i16)?;
    let gamma2 = read_bcn_vec(&mut reader, "MLA gamma2", read_bcn_i16)?;
    let chunk_commitments = read_bcn_digests(&mut reader, MAX_BUNDLE_ROWS, "MLA chunk_coms")?;
    let n_toks = reader.read_len(MAX_BUNDLE_ROWS, "MLA tokens")?;
    let mut tokens = Vec::with_capacity(n_toks);
    let mut partial_rows = Vec::with_capacity(n_toks);
    for t in 0..n_toks {
        let position = reader.read_u64("MLA position")?;
        let h_in = read_bcn_vec(&mut reader, "MLA h_in", read_bcn_i16)?;
        let c_q_raw = read_bcn_vec(&mut reader, "MLA c_q_raw", read_bcn_i32)?;
        let c_kv_raw = read_bcn_vec(&mut reader, "MLA c_kv_raw", read_bcn_i32)?;
        let k_pe_raw = read_bcn_vec(&mut reader, "MLA k_pe_raw", read_bcn_i32)?;
        let q_abs_raw = read_bcn_vec(&mut reader, "MLA q_abs_raw", read_bcn_i32)?;
        let q_pe_raw = read_bcn_vec(&mut reader, "MLA q_pe_raw", read_bcn_i32)?;
        let s_nope = read_bcn_rows(&mut reader, "MLA s_nope", read_bcn_i32)?;
        let s_rope = read_bcn_rows(&mut reader, "MLA s_rope", read_bcn_i32)?;
        let u_raw = read_bcn_vec(&mut reader, "MLA u_raw", read_bcn_i32)?;
        let v_raw = read_bcn_vec(&mut reader, "MLA v_raw", read_bcn_i32)?;
        let o_raw = read_bcn_vec(&mut reader, "MLA o_raw", read_bcn_i32)?;
        let chunk_ids = read_bcn_vec(&mut reader, "MLA chunk_ids", |r, l| {
            r.read_len(MAX_BUNDLE_ROWS, l)
        })?;
        let cache_commitment_before = read_bcn_digest(&mut reader, "MLA before")?;
        let cache_commitment_after = read_bcn_digest(&mut reader, "MLA after")?;
        let ckv_rows = read_bcn_rows(&mut reader, &format!("MLA {t} ckv"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        let kpe_rows = read_bcn_rows(&mut reader, &format!("MLA {t} kpe"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        tokens.push(MlaToken {
            step: MlaPublicStep {
                position,
                h_in,
                c_q_raw,
                c_kv_raw,
                k_pe_raw,
                q_abs_raw,
                q_pe_raw,
                s_nope,
                s_rope,
                u_raw,
                v_raw,
                o_raw,
            },
            chunk_ids,
            cache_commitment_before,
            cache_commitment_after,
        });
        partial_rows.push((ckv_rows, kpe_rows));
    }
    let binding = read_bcn_digest(&mut reader, "MLA binding")?;
    let receipt_bytes = reader.read_bytes(MAX_WIRE_RECEIPT_BYTES, "MLA receipt")?.to_vec();
    reader.finish()?;
    Ok(MlaVerificationBundle {
        profile,
        expected_anchor,
        statement: StageBMlaStatement {
            shape,
            policy,
            layout,
            cprs_f,
            mla_leaf_commitment,
            gamma1,
            gamma_q,
            gamma_kv,
            gamma2,
            chunk_commitments,
            partial_rows,
            tokens,
            binding,
        },
        receipt_bytes,
    })
}

pub fn verify_mla_verification_bundle(
    b: &MlaVerificationBundle,
) -> Result<(usize, f64), String> {
    let t0 = Timer::start();
    let proof = decode_stage_b_mla_receipt_bytes(&b.receipt_bytes, &b.statement)?;
    let steps = verify_stage_b_mla_against_anchor(
        &b.expected_anchor,
        &b.statement,
        &b.profile,
        &proof,
    )?;
    Ok((steps.len(), t0.elapsed_ms()))
}

pub fn swa_verification_bundle_bytes(b: &SwaVerificationBundle) -> Result<Vec<u8>, String> {
    let s = &b.statement;
    if s.partial_rows.len() != s.tokens.len() {
        return Err("STWOSWAB partial_rows length must equal tokens length".into());
    }
    let mut out = Vec::with_capacity(1024 + b.receipt_bytes.len());
    out.extend_from_slice(SWA_BUNDLE_MAGIC);
    put_u32(&mut out, SWA_BUNDLE_VERSION);
    put_bcn_label(&mut out, b.profile.id())?;
    out.extend_from_slice(&b.expected_anchor);
    for v in [
        s.shape.dimension,
        s.shape.heads,
        s.shape.kv_heads,
        s.shape.head_dim,
        s.layout.chunk,
        s.cprs_f.unwrap_or(0),
    ] {
        put_u64(&mut out, v)?;
    }
    put_bcn_label(&mut out, &s.policy.base.id())?;
    match s.policy.sliding_window {
        None => out.push(0),
        Some(w) => {
            out.push(1);
            put_u64(&mut out, w.window_size)?;
        }
    }
    match s.policy.attention_sinks {
        None => out.push(0),
        Some(ref sk) => {
            out.push(1);
            put_bcn_i32_vec(&mut out, &sk.head_sink_logits_i32)?;
        }
    }
    out.extend_from_slice(&s.leaf_commitment);
    put_bcn_i16_vec(&mut out, &s.gamma1)?;
    put_bcn_i16_vec(&mut out, &s.gamma2)?;
    put_bcn_digests(&mut out, &s.chunk_commitments)?;
    put_u64(&mut out, s.tokens.len())?;
    for (tok, (k_rows, v_rows)) in s.tokens.iter().zip(&s.partial_rows) {
        put_raw_u64(&mut out, tok.step.position);
        put_bcn_i16_vec(&mut out, &tok.step.h_in)?;
        put_bcn_i32_vec(&mut out, &tok.step.q_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.k_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.v_raw)?;
        put_bcn_i32_rows(&mut out, &tok.step.scores)?;
        put_bcn_i32_vec(&mut out, &tok.step.ctx_raw)?;
        put_bcn_i32_vec(&mut out, &tok.step.o_raw)?;
        put_u64(&mut out, tok.chunk_ids.len())?;
        for &cid in &tok.chunk_ids {
            put_u64(&mut out, cid)?;
        }
        out.extend_from_slice(&tok.cache_commitment_before);
        out.extend_from_slice(&tok.cache_commitment_after);
        put_bcn_i8_rows(&mut out, k_rows)?;
        put_bcn_i8_rows(&mut out, v_rows)?;
    }
    out.extend_from_slice(&s.binding);
    put_bytes(&mut out, &b.receipt_bytes)?;
    Ok(out)
}

pub fn decode_swa_verification_bundle(bytes: &[u8]) -> Result<SwaVerificationBundle, String> {
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(SWA_BUNDLE_MAGIC, "STWOSWAB magic")?;
    let ver = reader.read_u32("STWOSWAB version")?;
    if ver != SWA_BUNDLE_VERSION {
        return Err(format!("unsupported STWOSWAB version {ver}"));
    }
    let profile = SecurityProfile::from_id(read_bcn_label(&mut reader, "SWA profile")?)?;
    let expected_anchor = read_bcn_digest(&mut reader, "SWA expected_anchor")?;
    let shape = GptOssAttentionShape {
        dimension: reader.read_len(MAX_BUNDLE_ROWS, "SWA dimension")?,
        heads: reader.read_len(MAX_BUNDLE_ROWS, "SWA heads")?,
        kv_heads: reader.read_len(MAX_BUNDLE_ROWS, "SWA kv_heads")?,
        head_dim: reader.read_len(MAX_BUNDLE_ROWS, "SWA head_dim")?,
    };
    let layout = CacheLayout {
        chunk: reader.read_len(MAX_BUNDLE_ROWS, "SWA chunk")?,
    };
    let raw_f = reader.read_len(MAX_BUNDLE_ROWS, "SWA cprs_f")?;
    let cprs_f = if raw_f == 0 { None } else { Some(raw_f) };
    let base = AttentionPolicy::from_id(read_bcn_label(&mut reader, "SWA base policy")?)?;
    let sliding_window = match reader.read_u32_byte("SWA sliding_window flag")? {
        0 => None,
        1 => Some(SlidingWindowSpec {
            window_size: reader.read_len(MAX_BUNDLE_ROWS, "SWA window_size")?,
        }),
        other => return Err(format!("invalid SWA sliding_window flag {other}")),
    };
    let attention_sinks = match reader.read_u32_byte("SWA sinks flag")? {
        0 => None,
        1 => Some(AttentionSinkSpec {
            head_sink_logits_i32: read_bcn_vec(&mut reader, "SWA sinks", read_bcn_i32)?,
        }),
        other => return Err(format!("invalid SWA sinks flag {other}")),
    };
    let policy = WindowSinkAttentionPolicy {
        base,
        sliding_window,
        attention_sinks,
    };
    let leaf_commitment = read_bcn_digest(&mut reader, "SWA leaf_commitment")?;
    let gamma1 = read_bcn_vec(&mut reader, "SWA gamma1", read_bcn_i16)?;
    let gamma2 = read_bcn_vec(&mut reader, "SWA gamma2", read_bcn_i16)?;
    let chunk_commitments = read_bcn_digests(&mut reader, MAX_BUNDLE_ROWS, "SWA chunk_coms")?;
    let n_toks = reader.read_len(MAX_BUNDLE_ROWS, "SWA tokens")?;
    let mut tokens = Vec::with_capacity(n_toks);
    let mut partial_rows = Vec::with_capacity(n_toks);
    for t in 0..n_toks {
        let position = reader.read_u64("SWA position")?;
        let h_in = read_bcn_vec(&mut reader, "SWA h_in", read_bcn_i16)?;
        let q_raw = read_bcn_vec(&mut reader, "SWA q_raw", read_bcn_i32)?;
        let k_raw = read_bcn_vec(&mut reader, "SWA k_raw", read_bcn_i32)?;
        let v_raw = read_bcn_vec(&mut reader, "SWA v_raw", read_bcn_i32)?;
        let scores = read_bcn_rows(&mut reader, "SWA scores", read_bcn_i32)?;
        let ctx_raw = read_bcn_vec(&mut reader, "SWA ctx_raw", read_bcn_i32)?;
        let o_raw = read_bcn_vec(&mut reader, "SWA o_raw", read_bcn_i32)?;
        let chunk_ids = read_bcn_vec(&mut reader, "SWA chunk_ids", |r, l| {
            r.read_len(MAX_BUNDLE_ROWS, l)
        })?;
        let cache_commitment_before = read_bcn_digest(&mut reader, "SWA before")?;
        let cache_commitment_after = read_bcn_digest(&mut reader, "SWA after")?;
        let k_rows = read_bcn_rows(&mut reader, &format!("SWA {t} k_rows"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        let v_rows = read_bcn_rows(&mut reader, &format!("SWA {t} v_rows"), |r, l| {
            r.read_u32_byte(l).map(|b| b as i8)
        })?;
        tokens.push(WindowedToken {
            step: WindowedPublicStep {
                position,
                h_in,
                q_raw,
                k_raw,
                v_raw,
                scores,
                ctx_raw,
                o_raw,
            },
            chunk_ids,
            cache_commitment_before,
            cache_commitment_after,
        });
        partial_rows.push((k_rows, v_rows));
    }
    let binding = read_bcn_digest(&mut reader, "SWA binding")?;
    let receipt_bytes = reader.read_bytes(MAX_WIRE_RECEIPT_BYTES, "SWA receipt")?.to_vec();
    reader.finish()?;
    Ok(SwaVerificationBundle {
        profile,
        expected_anchor,
        statement: StageBWindowSinkStatement {
            shape,
            policy,
            layout,
            cprs_f,
            leaf_commitment,
            gamma1,
            gamma2,
            chunk_commitments,
            partial_rows,
            tokens,
            binding,
        },
        receipt_bytes,
    })
}

pub fn verify_swa_verification_bundle(
    b: &SwaVerificationBundle,
) -> Result<(usize, f64), String> {
    let t0 = Timer::start();
    let proof = decode_stage_b_window_sink_receipt_bytes(&b.receipt_bytes, &b.statement)?;
    let steps = verify_stage_b_window_sinks_against_anchor(
        &b.expected_anchor,
        &b.statement,
        &b.profile,
        &proof,
    )?;
    Ok((steps.len(), t0.elapsed_ms()))
}

/// Zero-dependency WebAssembly C-ABI exports (`wasm32-unknown-unknown` / `wasm32-wasip1`) for
/// browser and edge verification of `STWOMDL1`, `STWOREC2`, and `STWOBCN4` receipts.
#[cfg(target_arch = "wasm32")]
pub mod wasm_abi {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    const WASM_MAX_TOKENS: usize = 1 << 20;

    #[inline]
    fn wasm_in_bounds(ptr: *const u8, byte_len: usize) -> bool {
        if byte_len == 0 {
            return true;
        }
        if ptr.is_null() || byte_len > isize::MAX as usize {
            return false;
        }
        let Some(end) = (ptr as usize).checked_add(byte_len) else {
            return false;
        };
        let mem_bytes = core::arch::wasm32::memory_size(0).saturating_mul(65_536);
        end <= mem_bytes
    }

    unsafe fn read_fixed32(ptr: *const u8) -> Option<[u8; 32]> {
        if !wasm_in_bounds(ptr, 32) {
            return None;
        }
        core::slice::from_raw_parts(ptr, 32).try_into().ok()
    }

    unsafe fn read_bounded_bytes(ptr: *const u8, len: usize) -> Option<Vec<u8>> {
        if len > MAX_WIRE_RECEIPT_BYTES {
            return None;
        }
        if len == 0 {
            return Some(Vec::new());
        }
        if !wasm_in_bounds(ptr, len) {
            return None;
        }
        Some(core::slice::from_raw_parts(ptr, len).to_vec())
    }

    unsafe fn read_unaligned_u32_vec(ptr: *const u32, len: usize) -> Option<Vec<u32>> {
        if len > WASM_MAX_TOKENS {
            return None;
        }
        if len == 0 {
            return Some(Vec::new());
        }
        let byte_len = len.checked_mul(4)?;
        let raw_ptr = ptr.cast::<u8>();
        if !wasm_in_bounds(raw_ptr, byte_len) {
            return None;
        }
        let raw = core::slice::from_raw_parts(raw_ptr, byte_len);
        let mut out = Vec::with_capacity(len);
        for chunk in raw.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Some(out)
    }

    #[no_mangle]
    pub extern "C" fn wasm_alloc(size: usize) -> *mut u8 {
        if size == 0 || size > MAX_WIRE_RECEIPT_BYTES {
            return core::ptr::null_mut();
        }
        let Ok(layout) = Layout::from_size_align(size, 4) else {
            return core::ptr::null_mut();
        };
        // SAFETY: `size > 0` and `layout` is valid.
        unsafe { alloc(layout) }
    }

    /// # Safety
    /// `ptr` and `cap` must originate from `wasm_alloc(cap)`.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_dealloc(ptr: *mut u8, cap: usize) {
        if ptr.is_null() || cap == 0 || cap > MAX_WIRE_RECEIPT_BYTES {
            return;
        }
        if let Ok(layout) = Layout::from_size_align(cap, 4) {
            dealloc(ptr, layout);
        }
    }

    /// Verify a public-tokens `STWOMDL1` whole-model standalone receipt (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// All pointers must reference valid linear-memory slices for their respective lengths.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_whole_model_mdl1(
        anchor_ptr: *const u8,
        prompts_ptr: *const u32,
        prompts_len: usize,
        emitted_ptr: *const u32,
        emitted_len: usize,
        receipt_ptr: *const u8,
        receipt_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(anchor) = read_fixed32(anchor_ptr) else {
                return 0;
            };
            let Some(prompts) = read_unaligned_u32_vec(prompts_ptr, prompts_len) else {
                return 0;
            };
            let Some(emitted) = read_unaligned_u32_vec(emitted_ptr, emitted_len) else {
                return 0;
            };
            let Some(receipt) = read_bounded_bytes(receipt_ptr, receipt_len) else {
                return 0;
            };
            i32::from(verify_whole_model_receipt_bytes(&anchor, &prompts, &emitted, &receipt).is_ok())
        }))
        .unwrap_or(0)
    }

    /// Verify a salted-`Blake2s`-committed-tokens `STWOMDL1` receipt (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// All pointers must reference valid linear-memory slices for their respective lengths.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_whole_model_committed_mdl1(
        anchor_ptr: *const u8,
        prompt_com_ptr: *const u8,
        emitted_com_ptr: *const u8,
        receipt_ptr: *const u8,
        receipt_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(anchor) = read_fixed32(anchor_ptr) else {
                return 0;
            };
            let Some(prompt_com) = read_fixed32(prompt_com_ptr) else {
                return 0;
            };
            let Some(emitted_com) = read_fixed32(emitted_com_ptr) else {
                return 0;
            };
            let Some(receipt) = read_bounded_bytes(receipt_ptr, receipt_len) else {
                return 0;
            };
            i32::from(
                verify_whole_model_committed_receipt_bytes(
                    &anchor,
                    &prompt_com,
                    &emitted_com,
                    &receipt,
                )
                .is_ok(),
            )
        }))
        .unwrap_or(0)
    }

    /// Verify a `STWOREC2` standalone block receipt (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// All pointers must reference valid linear-memory slices for their respective lengths.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_standalone_block_rec2(
        anchor_ptr: *const u8,
        init_layer: u32,
        init_prev_ptr: *const u8,
        receipt_ptr: *const u8,
        receipt_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(anchor) = read_fixed32(anchor_ptr) else {
                return 0;
            };
            let Some(prev_receipt_digest) = read_fixed32(init_prev_ptr) else {
                return 0;
            };
            let Some(receipt) = read_bounded_bytes(receipt_ptr, receipt_len) else {
                return 0;
            };
            let link = StandaloneChainLink {
                layer: init_layer,
                prev_receipt_digest,
            };
            i32::from(verify_standalone_block_receipt_bytes(&anchor, &link, &receipt).is_ok())
        }))
        .unwrap_or(0)
    }

    /// Verify a `STWOREC1` split-anchor-and-wiring-root receipt (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// All pointers must reference valid linear-memory slices for their respective lengths.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_recursive_audit_rec1(
        anchor_ptr: *const u8,
        wiring_root_ptr: *const u8,
        init_layer: u32,
        init_prev_ptr: *const u8,
        receipt_ptr: *const u8,
        receipt_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(anchor) = read_fixed32(anchor_ptr) else {
                return 0;
            };
            let Some(wiring_root) = read_fixed32(wiring_root_ptr) else {
                return 0;
            };
            let Some(prev_receipt_digest) = read_fixed32(init_prev_ptr) else {
                return 0;
            };
            let Some(receipt) = read_bounded_bytes(receipt_ptr, receipt_len) else {
                return 0;
            };
            let link = StageAChainLink {
                layer: init_layer,
                prev_receipt_digest,
            };
            i32::from(
                verify_recursive_audit_receipt_bytes(&anchor, &wiring_root, &link, &receipt).is_ok(),
            )
        }))
        .unwrap_or(0)
    }

    /// Verify a `STWOBLKB` (`STWOBLK1` + `CPRSMUL1`) cross-layer block bundle (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// `bundle_ptr..bundle_ptr + bundle_len` must be valid WASM linear memory.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_cross_layer_block_blk1(
        bundle_ptr: *const u8,
        bundle_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(bytes) = read_bounded_bytes(bundle_ptr, bundle_len) else {
                return 0;
            };
            i32::from(verify_cross_layer_block_bundle_bytes(&bytes).is_ok())
        }))
        .unwrap_or(0)
    }

    /// Verify a `STWOBCN4` spot-check bundle (`1` = accepted, `0` = rejected).
    ///
    /// # Safety
    /// `bundle_ptr..bundle_ptr + bundle_len` must be valid WASM linear memory.
    #[no_mangle]
    pub unsafe extern "C" fn wasm_verify_beacon_bcn4(
        bundle_ptr: *const u8,
        bundle_len: usize,
    ) -> i32 {
        catch_unwind(AssertUnwindSafe(|| {
            let Some(bytes) = read_bounded_bytes(bundle_ptr, bundle_len) else {
                return 0;
            };
            i32::from(verify_spot_check_verification_bundle_bytes(&bytes).is_ok())
        }))
        .unwrap_or(0)
    }
}

#[cfg(test)]
#[cfg(feature = "fixture-tests")]
mod fixture_receipt_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "fixture-tests")]
    const FIXTURE_BUNDLE: &[u8] = include_bytes!("../fixtures/leaf_v4_forward_pass_bundle.bcn4");
    #[cfg(feature = "fixture-tests")]
    const MAVERICK_MODE9_FIXTURE_BUNDLE: &[u8] =
        include_bytes!("../fixtures/maverick_mode9_fastprove96_bundle.bcn4");

    #[test]
    #[cfg(feature = "fixture-tests")]
    fn leaf_v4_forward_pass_bundle_fixture_verifies_and_rejects_tampering() {
        assert_eq!(&FIXTURE_BUNDLE[..8], b"STWOBCN4");
        let verdict = verify_spot_check_verification_bundle_bytes(FIXTURE_BUNDLE)
            .expect("committed Leaf-v4 forward-pass bundle fixture verifies");
        assert_eq!(verdict.stage_a_mode, "public_witness");
        assert_eq!(verdict.layer, 0);

        let bundle = decode_spot_check_verification_bundle(FIXTURE_BUNDLE).unwrap();
        assert!(bundle.continuity.is_some());
        let b = bundle.boundaries.as_ref().expect("boundaries present");
        let expected_prompt = b.embedding.token_ids.clone();
        let expected_emitted = b.lm_head.argmax_token_ids().unwrap();
        let expected_next = b.lm_head.emitted_next_token().unwrap();
        assert_eq!(
            verdict.prompt_token_ids.as_deref(),
            Some(expected_prompt.as_slice())
        );
        assert_eq!(
            verdict.emitted_token_ids.as_deref(),
            Some(expected_emitted.as_slice())
        );
        assert_eq!(verdict.emitted_next_token, Some(expected_next));

        verify_spot_check_bundle_for_emitted_token(&bundle, &expected_prompt, expected_next)
            .expect("emitted next token and prompt token IDs match committed forward-pass bundle");
        verify_spot_check_bundle_bytes_for_emitted_token(
            FIXTURE_BUNDLE,
            &expected_prompt,
            expected_next,
        )
        .expect("emitted next token helper verifies from raw bundle bytes");

        // Tamper case 0a: non-argmax emitted next token is rejected.
        let wrong_next = (expected_next + 1) % (b.lm_head.vocab_size as u32);
        assert!(
            verify_spot_check_bundle_for_emitted_token(&bundle, &expected_prompt, wrong_next)
                .is_err()
        );

        // Tamper case 0b: modified prompt token list is rejected.
        let mut wrong_prompt = expected_prompt.clone();
        wrong_prompt[0] = (wrong_prompt[0] + 1) % (b.embedding.vocab_size as u32);
        assert!(
            verify_spot_check_bundle_for_emitted_token(&bundle, &wrong_prompt, expected_next)
                .is_err()
        );

        // Tamper case 1: tamper Stage B attention receipt byte.
        let mut bad_stage_b = decode_spot_check_verification_bundle(FIXTURE_BUNDLE).unwrap();
        let last = bad_stage_b.attention_receipt.len() - 1;
        bad_stage_b.attention_receipt[last] ^= 0x01;
        assert!(verify_spot_check_verification_bundle(&bad_stage_b).is_err());

        // Tamper case 2: tamper Stage A receipt byte.
        let mut bad_stage_a = decode_spot_check_verification_bundle(FIXTURE_BUNDLE).unwrap();
        let last = bad_stage_a.receipt.len() - 1;
        bad_stage_a.receipt[last] ^= 0x01;
        assert!(verify_spot_check_verification_bundle(&bad_stage_a).is_err());

        // Tamper case 3: tamper inter-layer continuity opening.
        let mut bad_continuity = decode_spot_check_verification_bundle(FIXTURE_BUNDLE).unwrap();
        bad_continuity.continuity.as_mut().unwrap().output_digest[0] ^= 0x01;
        assert!(verify_spot_check_verification_bundle(&bad_continuity).is_err());

        // Tamper case 4: tamper forward-pass boundary LM head output.
        let mut bad_boundary = decode_spot_check_verification_bundle(FIXTURE_BUNDLE).unwrap();
        bad_boundary.boundaries.as_mut().unwrap().lm_head.logits_raw[0][0] += 1;
        assert!(verify_spot_check_verification_bundle(&bad_boundary).is_err());
    }

    #[test]
    #[cfg(feature = "fixture-tests")]
    fn maverick_mode9_fastprove96_fixture_verifies_and_rejects_tampering() {
        assert_eq!(&MAVERICK_MODE9_FIXTURE_BUNDLE[..8], b"STWOBCN4");
        // The bundle's statement carries `router_outputs_raw`, which the Mode 9 receipt neither binds
        // nor checks, so the bundle verifier refuses it (it accepted it, and 12 mutants of it, before). The receipt
        // itself still verifies directly against its anchor below, and still refuses every tampered byte.
        let err = verify_spot_check_verification_bundle_bytes(MAVERICK_MODE9_FIXTURE_BUNDLE)
            .err()
            .expect("a GKR-lane bundle with unbound router_outputs_raw is refused (finding D1)");
        assert!(err.contains("router_outputs_raw") && err.contains("gkr_lane"), "{err}");

        let bundle = decode_spot_check_verification_bundle(MAVERICK_MODE9_FIXTURE_BUNDLE).unwrap();
        assert_eq!(bundle.profile, SecurityProfile::FastProve96);
        assert_eq!(bundle.statement.shape.batch, 32);
        assert_eq!(bundle.statement.shape.dimension, 8192);
        assert_eq!(bundle.statement.shape.expert_width, 2048);
        assert_eq!(bundle.statement.shape.experts, 128);
        assert_eq!(bundle.statement.shape.top_k, 1);
        assert_eq!(bundle.statement.shape.shared_factor, 1);

        // Verify the committed Mode 9 receipt bit-for-bit via `decode_stage_a_gkr_lane_receipt`,
        // `stage_a_gkr_lane_receipt_bytes`, and `verify_stage_a_gkr_lane_batch_against_anchor`.
        assert_eq!(&bundle.receipt[..8], b"STWOREG1");
        let wire_mode = u32::from_le_bytes(bundle.receipt[8..12].try_into().unwrap());
        assert_eq!(wire_mode, STAGE_A_RECEIPT_MODE_GKR_LANE);

        let gkr_proof = decode_stage_a_gkr_lane_receipt(
            &bundle.receipt,
            &bundle.statement,
            bundle.policy,
        )
        .expect("Mode 9 Maverick receipt decodes cleanly");
        let reencoded = stage_a_gkr_lane_receipt_bytes(&gkr_proof)
            .expect("Mode 9 Maverick receipt re-encodes cleanly");
        assert_eq!(reencoded, bundle.receipt, "Mode 9 receipt must round-trip bit-for-bit");

        let challenge =
            derive_challenge(bundle.beacon, bundle.response_root, bundle.beacon_policy).unwrap();
        let link = beacon_link(&challenge);
        let decoded_b =
            decode_stage_b_receipt_bytes(&bundle.attention_receipt, &bundle.attention).unwrap();
        let leaf_params = group_params(
            2 * bundle.attention.shape.dimension.ilog2() as usize,
            4,
            &bundle.profile,
        )
        .unwrap();
        let b_run = verify_stage_b(
            &bundle.attention,
            bundle.attention_policy,
            &bundle.profile,
            &leaf_params,
            &decoded_b,
        )
        .unwrap();
        let steps: Vec<PublicStep> = bundle.attention.tokens.iter().map(|t| t.step.clone()).collect();
        let (_, _, inputs) = layer_states(&steps, &b_run.recomputed);
        verify_stage_a_gkr_lane_batch_against_anchor(
            &bundle.context.anchor_digest(),
            &bundle.context,
            &inputs,
            bundle.policy,
            &bundle.profile,
            &bundle.statement,
            &gkr_proof,
            &link,
        )
        .expect("Mode 9 Maverick receipt verifies directly against anchor");

        // Reject on one flipped byte in the Mode 9 receipt (across header, sumcheck/GKR, witness openings, and CPRS tail).
        for offset in [
            0usize,
            8,
            32,
            1024,
            bundle.receipt.len() / 2,
            bundle.receipt.len() - 1,
        ] {
            let mut tampered_receipt = bundle.receipt.clone();
            tampered_receipt[offset] ^= 0x01;
            let direct_ok = decode_stage_a_gkr_lane_receipt(
                &tampered_receipt,
                &bundle.statement,
                bundle.policy,
            )
            .and_then(|p| {
                if stage_a_gkr_lane_receipt_bytes(&p)? != tampered_receipt {
                    return Err("non-canonical Mode 9 receipt bytes".into());
                }
                verify_stage_a_gkr_lane_batch_against_anchor(
                    &bundle.context.anchor_digest(),
                    &bundle.context,
                    &inputs,
                    bundle.policy,
                    &bundle.profile,
                    &bundle.statement,
                    &p,
                    &link,
                )
            });
            assert!(
                direct_ok.is_err(),
                "flipped byte at offset {offset} must be rejected by Mode 9 verifier"
            );

            let mut bad_bundle = bundle.clone();
            bad_bundle.receipt = tampered_receipt;
            assert!(
                verify_spot_check_verification_bundle(&bad_bundle).is_err(),
                "flipped byte at offset {offset} must be rejected by bundle verifier"
            );
        }
    }

    #[test]
    fn cairo_whole_model_anchor_and_logup_vector_matches_rust_bit_for_bit() {
        use stwo::core::channel::MerkleChannel;
        use stwo::core::vcs::blake2_hash::Blake2sHash;
        use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
        use stwo_constraint_framework::Relation;
        use crate::blake2s_air::{
            Blake2sChainElements, Blake2sCompressElements, Blake2sRoundElements,
            Blake2sXorElements, ByteBusElements, MemoryBusElements,
        };
        use crate::standalone_audit::{
            build_whole_model_public_memory_items, mix_whole_model_statement,
        };

        let u32s_to_bytes = |words: [u32; 8]| -> [u8; 32] {
            let mut out = [0u8; 32];
            for (i, w) in words.into_iter().enumerate() {
                out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            out
        };

        // Exact inputs from `cairo/src/lib.cairo::build_valid_whole_model_fixture`.
        let weights_dig = u32s_to_bytes([0x11112222, 0x33334444, 5, 6, 7, 8, 9, 10]);
        let t0 = u32s_to_bytes([0xAABBCCDD, 1, 2, 3, 4, 5, 6, 7]);
        let t1 = u32s_to_bytes([0x55667788, 11, 12, 13, 14, 15, 16, 17]);
        let t2 = u32s_to_bytes([0x99AABBCC, 21, 22, 23, 24, 25, 26, 27]);

        let stmt = WholeModelStatement {
            mode_tag: 11,
            profile_id: "fastprove96".to_string(),
            batch: 2,
            num_blocks: 6,
            layers_per_block: 8,
            vocab_size: 256,
            d_model: 16,
            cprs_weight_f: 2,
            cprs_witness_f: 2,
            stacked_expert_pack: 8,
            zk_trace_blinded: true,
            tokens_committed: false,
            audit_log_size: 12,
            sched_log_size: 10,
            round_log_size: 14,
            bridge_log_size: 12,
            weights_and_boundaries_digest: weights_dig,
            prompt_token_ids: vec![7, 42],
            emitted_token_ids: vec![19, 108],
            prompt_commitment: [0u8; 32],
            emitted_commitment: [0u8; 32],
        };
        let anchor = compute_whole_model_full_stack_anchor(&stmt.policy_bytes(), &weights_dig, &t0);
        let hex = hex32(&anchor);
        assert_eq!(parse_hex32(&format!("0x{hex}")).unwrap(), anchor);
        assert_eq!(
            parse_chain_link_spec(&format!("8:{hex}")).unwrap(),
            (8, anchor)
        );

        let mut ch = Blake2sChannel::default();
        mix_whole_model_statement(&mut ch, &stmt);
        Blake2sMerkleChannel::mix_root(&mut ch, Blake2sHash(t0));
        Blake2sMerkleChannel::mix_root(&mut ch, Blake2sHash(t1));
        let _ = Blake2sXorElements::draw(&mut ch);
        let _ = Blake2sRoundElements::draw(&mut ch);
        let _ = Blake2sCompressElements::draw(&mut ch);
        let _ = Blake2sChainElements::draw(&mut ch);
        let _ = ByteBusElements::draw(&mut ch);
        let mem = MemoryBusElements::draw(&mut ch);
        Blake2sMerkleChannel::mix_root(&mut ch, Blake2sHash(t2));

        let mut pub_sum = SecureField::zero();
        for (id, coords) in build_whole_model_public_memory_items(&stmt) {
            let denom: SecureField = mem.combine(&[
                BaseField::from_u32_unchecked(id),
                coords[0],
                coords[1],
                coords[2],
                coords[3],
            ]);
            pub_sum -= SecureField::one() / denom;
        }
        let z0 = SecureField::zero();
        let sums = StandaloneClaimedSums {
            audit_sum: -pub_sum,
            sched_sum: z0,
            round_sum: z0,
            bridge_sum: z0,
            xor9_sum: z0,
            xor8_sum: z0,
            xor7_sum: z0,
            xor4_sum: z0,
        };
        ch.mix_felts(&sums.as_array());
        assert!((sums.total() + pub_sum).is_zero());
        assert_ne!(ch.digest().0, [0u8; 32]);
    }

    #[cfg(feature = "fixture-tests")]
    fn build_synthetic_blkb_bundle(
        num_layers: usize,
        batch: usize,
        kv_dim: usize,
        receipt_len: usize,
    ) -> CrossLayerBlockVerificationBundle {
        let base = decode_spot_check_verification_bundle(MAVERICK_MODE9_FIXTURE_BUNDLE).unwrap();
        let mut expected_anchors = Vec::with_capacity(num_layers);
        let mut contexts = Vec::with_capacity(num_layers);
        let mut stage_a_statements = Vec::with_capacity(num_layers);
        let mut attn_coms = Vec::with_capacity(num_layers);
        let mut attn_pols = Vec::with_capacity(num_layers);
        let mut sb_stmts = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            expected_anchors.push([l as u8 + 1; 32]);
            contexts.push(base.context.clone());
            stage_a_statements.push(base.statement.clone());
            attn_coms.push([0xA0 | (l as u8); 32]);
            attn_pols.push(base.attention_policy);

            let mut all_k = Vec::with_capacity(batch);
            let mut all_v = Vec::with_capacity(batch);
            let mut tokens = Vec::with_capacity(batch);
            let mut partial_rows = Vec::with_capacity(batch);
            for t in 0..batch {
                all_k.push(vec![((l * 17 + t) & 0x7f) as i8; kv_dim]);
                all_v.push(vec![-((l * 13 + t) as i8 & 0x3f); kv_dim]);
                tokens.push(StageBToken {
                    step: PublicStep {
                        position: t as u64,
                        h_in: if l == 0 { vec![t as i16; 16] } else { Vec::new() },
                        q_raw: Vec::new(),
                        k_raw: Vec::new(),
                        v_raw: Vec::new(),
                        scores: Vec::new(),
                        ctx_raw: Vec::new(),
                        o_raw: Vec::new(),
                        a: Vec::new(),
                        x: Vec::new(),
                    },
                    chunk_ids: Vec::new(),
                    cache_commitment_before: [t as u8; 32],
                    cache_commitment_after: [(t + 1) as u8; 32],
                });
                partial_rows.push((all_k.clone(), all_v.clone()));
            }
            sb_stmts.push(StageBStatement {
                shape: AttentionShape {
                    dimension: 2048,
                    heads: 16,
                    kv_heads: 16,
                    head_dim: kv_dim / 16,
                },
                layout: CacheLayout { chunk: batch.max(128) },
                policy_id: base.attention.policy_id.clone(),
                attention_commitment: [0xB0 | (l as u8); 32],
                gamma1: vec![256; 16],
                gamma2: vec![256; 16],
                chunk_commitments: Vec::new(),
                tokens,
                partial_rows,
                binding: [0xC0 | (l as u8); 32],
            });
        }

        CrossLayerBlockVerificationBundle {
            profile: SecurityProfile::FastProve96,
            policy: base.policy,
            attention_policy: base.attention_policy,
            expected_anchors,
            contexts,
            stacked_bank_commitments: Some(vec![[0x55; 32]; num_layers]),
            stage_a_statements,
            layer_inputs: None,
            stage_b: Some((attn_coms, attn_pols, sb_stmts)),
            initial_chain: StageAChainLink {
                layer: 0,
                prev_receipt_digest: [0u8; 32],
            },
            block_receipt: vec![0x42; receipt_len],
        }
    }

    #[test]
    #[cfg(feature = "fixture-tests")]
    fn stwoblkb_v3_delta_encodes_cumulative_partial_rows_and_preserves_v2_decode() {
        // 4 layers, B = 128, kv_dim = 2048, 3.43 MiB STWOBLK1 receipt (exact ds16b M3 geometry).
        let big = build_synthetic_blkb_bundle(4, 128, 2048, 3_426_094);
        let v2_bytes = cross_layer_block_bundle_bytes_with_version(&big, 2).unwrap();
        assert!(
            v2_bytes.len() > 135 * 1024 * 1024,
            "legacy v2 encoding duplicates O(T^2) partial_rows prefixes ({} bytes)",
            v2_bytes.len()
        );
        let v3_bytes = cross_layer_block_bundle_bytes(&big).unwrap();
        assert!(
            v3_bytes.len() < 9 * 1024 * 1024,
            "v3 delta encoding must stay well inside 64 MiB wire limit, got {} bytes",
            v3_bytes.len()
        );
        let decoded = decode_cross_layer_block_bundle(&v3_bytes)
            .expect("v3 STWOBLKB bundle within MAX_WIRE_RECEIPT_BYTES decodes cleanly");
        let (_, _, orig_sb) = big.stage_b.as_ref().unwrap();
        let (_, _, dec_sb) = decoded.stage_b.as_ref().unwrap();
        for l in 0..4 {
            assert_eq!(dec_sb[l].partial_rows, orig_sb[l].partial_rows);
        }

        // Backward compatibility: small v2 bundles remain decodable.
        let small = build_synthetic_blkb_bundle(2, 8, 64, 1024);
        let small_v2 = cross_layer_block_bundle_bytes_with_version(&small, 2).unwrap();
        let dec_small = decode_cross_layer_block_bundle(&small_v2)
            .expect("legacy v2 STWOBLKB bundle within wire limit still decodes");
        assert_eq!(
            dec_small.stage_b.unwrap().2[0].partial_rows,
            small.stage_b.unwrap().2[0].partial_rows
        );
    }

    #[test]
    #[cfg(feature = "fixture-tests")]
    fn stwoblkb_v3_rejects_token0_reuse_and_partial_rows_decompression_bomb() {
        let small = build_synthetic_blkb_bundle(1, 2, 16, 64);
        let mut bad_tok0 = cross_layer_block_bundle_bytes(&small).unwrap();
        // Locate the first token's `reuse_flag == 0` right before its `k_rows` count (1u64) + len (16u64)
        // and flip it to `1`.
        let mut needle = vec![0u8];
        needle.extend_from_slice(&1u64.to_le_bytes());
        needle.extend_from_slice(&16u64.to_le_bytes());
        let pos = bad_tok0
            .windows(needle.len())
            .position(|w| w == needle.as_slice())
            .expect("locate token 0 partial_rows header");
        bad_tok0[pos] = 1;
        let err = decode_cross_layer_block_bundle(&bad_tok0).unwrap_err();
        assert!(
            err.contains("token 0 cannot reuse"),
            "unexpected error for token 0 reuse: {err}"
        );

        // Decompression bomb: 1 layer, 512 tokens, kv_dim = 2048 (`512 * 513 / 2 * 4096`
        // = 537,919,488 reconstructed cells > MAX_BUNDLE_PARTIAL_ROWS_CELLS = 268,435,456), while
        // wire payload is < 4.5 MiB!
        let bomb = build_synthetic_blkb_bundle(1, 512, 2048, 64);
        let bomb_bytes = cross_layer_block_bundle_bytes(&bomb).unwrap();
        assert!(bomb_bytes.len() < 5 * 1024 * 1024);
        let bomb_err = decode_cross_layer_block_bundle(&bomb_bytes).unwrap_err();
        assert!(
            bomb_err.contains("reconstructed partial_rows exceeds"),
            "unexpected error for decompression bomb: {bomb_err}"
        );
    }

    #[test]
    fn standalone_zk_blind_rows_exceeds_two_queries_plus_two_across_all_profiles() {
        for profile_id in [
            "dev13",
            "hardened96",
            "smallproof96",
            "fastprove96",
            "longquery",
            "production100",
        ] {
            let pcs = standalone_pcs_config_for_profile_id(profile_id)
                .unwrap_or_else(|e| panic!("profile {profile_id}: {e}"));
            let q = pcs.fri_config.n_queries;
            let rows = standalone_zk_blind_rows_for_profile_id(profile_id)
                .unwrap_or_else(|e| panic!("blind rows for {profile_id}: {e}"));
            assert!(
                rows > 2 * q + 2,
                "profile {profile_id}: blind_rows {rows} must strictly exceed 2 * {q} + 2 = {}",
                2 * q + 2
            );
            assert_eq!(rows % 2, 0, "profile {profile_id}: blind_rows {rows} must be even");
            assert!(validate_standalone_zk_blind_rows(rows, q).is_ok());
            assert!(
                validate_standalone_zk_blind_rows(2 * q + 2, q).is_err(),
                "profile {profile_id}: boundary 2*q+2={} must fail strict inequality",
                2 * q + 2
            );
        }
        // Regression pin for `production100` (`n_queries = 224` -> `2 * 224 + 2 = 450`):
        // the legacy constant `256` must fail, and the derived count must be `512`.
        assert!(validate_standalone_zk_blind_rows(256, 224).is_err());
        assert!(validate_standalone_zk_blind_rows(450, 224).is_err());
        assert!(validate_standalone_zk_blind_rows(451, 224).is_err());
        assert!(validate_standalone_zk_blind_rows(452, 224).is_ok());
        assert_eq!(standalone_zk_blind_rows_for_profile_id("production100").unwrap(), 512);
        assert_eq!(standalone_zk_blind_rows_for_profile_id("fastprove96").unwrap(), 256);
    }

    #[test]
    fn standalone_profiles_enforce_blowup8_fold8_last_deg64_fri_and_wire_version_11() {
        assert_eq!(STANDALONE_AUDIT_VERSION, 11);
        let expected_queries = [
            ("dev13", 3usize, 0u32),
            ("hardened96", 35, 0),
            ("smallproof96", 34, 0),
            ("fastprove96", 40, 0),
            ("longquery", 80, 0),
            ("production100", 112, 16),
            (PRODUCTION_HARDENED_PROFILE_ID, 112, 16),
        ];
        for (profile_id, n_q, first_fold_pow) in expected_queries {
            let pcs = standalone_pcs_config_for_profile_id(profile_id).unwrap();
            assert_eq!(pcs.fri_config.log_last_layer_degree_bound, 6, "{profile_id}");
            assert_eq!(pcs.fri_config.log_blowup_factor, 3, "{profile_id}");
            assert_eq!(pcs.fri_config.n_queries, n_q, "{profile_id}");
            assert_eq!(pcs.fri_config.fold_step, 3, "{profile_id}");
            assert_eq!(pcs.fri_config.first_fold_pow_bits, first_fold_pow, "{profile_id}");
        }
    }

    /// The hardened profile runs `production100`'s outer STARK and adds 16 bits of proof of work
    /// before the outer lookup draws; no other profile grinds there, so their transcripts and wire
    /// bytes are unchanged. The profile is defined for transformer blocks only.
    #[test]
    fn standalone_hardened_profile_pins_outer_config_and_lookup_grinding() {
        assert_eq!(PRODUCTION_HARDENED_PROFILE_ID, "production_hardened");
        assert_eq!(
            standalone_pcs_config_for_profile_id(PRODUCTION_HARDENED_PROFILE_ID).unwrap(),
            standalone_pcs_config_for_profile_id("production100").unwrap()
        );
        assert_eq!(PRODUCTION_HARDENED_LOOKUP_POW_BITS, 16);
        assert_eq!(standalone_lookup_pow_bits_for_profile_id(PRODUCTION_HARDENED_PROFILE_ID), 16);
        for p in ["dev13", "hardened96", "smallproof96", "fastprove96", "longquery", "production100"] {
            assert_eq!(standalone_lookup_pow_bits_for_profile_id(p), 0, "{p}");
        }
        assert_eq!(
            standalone_zk_blind_rows_for_profile_id(PRODUCTION_HARDENED_PROFILE_ID).unwrap(),
            standalone_zk_blind_rows_for_profile_id("production100").unwrap()
        );
    }

    /// Every listed profile id has an outer configuration, and every `exp_pos2` profile has its
    /// base profile's outer configuration and lookup grinding (16 bits for the hardened one).
    #[test]
    fn standalone_profile_ids_resolve_and_exp_pos2_profiles_follow_their_base() {
        for id in STANDALONE_PROFILE_IDS {
            assert!(standalone_pcs_config_for_profile_id(id).is_ok(), "{id}");
        }
        for (exp, base) in EXP_POS2_PROFILES {
            assert!(STANDALONE_PROFILE_IDS.contains(&exp), "{exp}");
            assert_eq!(
                standalone_pcs_config_for_profile_id(exp).unwrap(),
                standalone_pcs_config_for_profile_id(base).unwrap()
            );
            assert_eq!(
                standalone_lookup_pow_bits_for_profile_id(exp),
                standalone_lookup_pow_bits_for_profile_id(base),
                "{exp}"
            );
        }
        assert_eq!(standalone_lookup_pow_bits_for_profile_id("exp_pos2_production_hardened"), 16);
    }

    #[test]
    fn standalone_chain_manifest_roundtrips_and_rejects_invalid_paths_or_indices() {
        let m = StandaloneChainManifest {
            initial_chain: StandaloneChainLink {
                layer: 0,
                prev_receipt_digest: [0x24; 32],
            },
            blocks: vec![
                StandaloneChainManifestEntry {
                    block_index: 0,
                    standalone_anchor: [0x11; 32],
                    com_h_in: [0x22; 32],
                    com_h_out: [0x33; 32],
                    receipt_path: "block_00.rec2".to_string(),
                },
                StandaloneChainManifestEntry {
                    block_index: 1,
                    standalone_anchor: [0xaa; 32],
                    com_h_in: [0x33; 32],
                    com_h_out: [0x55; 32],
                    receipt_path: "receipts/block_01.rec2".to_string(),
                },
            ],
        };
        let encoded = encode_standalone_chain_manifest(&m).expect("encode STWOCHN2");
        assert!(encoded.starts_with("STWOCHN2\n"));
        let decoded = decode_standalone_chain_manifest(encoded.as_bytes()).expect("decode STWOCHN2");
        assert_eq!(decoded, m);

        let bad_trav = encoded.replace("block_00.rec2", "../block_00.rec2");
        assert!(decode_standalone_chain_manifest(bad_trav.as_bytes()).is_err());

        let bad_abs = encoded.replace("block_00.rec2", "/tmp/block_00.rec2");
        assert!(decode_standalone_chain_manifest(bad_abs.as_bytes()).is_err());

        let bad_idx = encoded.replace("\nblock 1 ", "\nblock 2 ");
        assert!(decode_standalone_chain_manifest(bad_idx.as_bytes()).is_err());

        // Other spellings of the same manifest are rejected, so its digest identifies it.
        for other in [
            encoded.replace("\nblock 1 ", "\nblock +1 "),
            encoded.replace("\nblock 1 ", "\nblock  1 "),
            encoded.replace("\nblock 1 ", "\n# note\nblock 1 "),
            encoded.replace('\n', "\r\n"),
            encoded.replace(&"11".repeat(32), &format!("0x{}", "11".repeat(32))),
            encoded.replace(&"aa".repeat(32), &"AA".repeat(32)),
            format!("{encoded}\n"),
            encoded.trim_end().to_string(),
            encoded.replace("receipts/block_01.rec2", "receipts//block_01.rec2"),
            encoded.replace("receipts/block_01.rec2", "receipts/./block_01.rec2"),
            encoded.replace("block_00.rec2", "./block_00.rec2"),
            encoded.replace("receipts/block_01.rec2", "receipts\\block_01.rec2"),
        ] {
            assert_ne!(other, encoded);
            assert!(
                decode_standalone_chain_manifest(other.as_bytes()).is_err(),
                "accepted a second spelling: {other:?}"
            );
        }

        // Non-ASCII 64-byte UTF-8 input must return Err cleanly without slicing inside a codepoint.
        let multibyte_64 = format!("aé{}", "a".repeat(61));
        assert_eq!(multibyte_64.len(), 64);
        assert!(parse_hex32(&multibyte_64).is_err());
        assert!(parse_chain_link_spec(&format!("0:{multibyte_64}")).is_err());
    }

    /// A chain anchor names the registered blocks in order. Another order, a prefix, an extra
    /// block or a repeated block is another anchor; otherwise a prover could chain the registered
    /// blocks in another order under the same anchor.
    /// `STWOMDL2` has one spelling: a manifest round-trips, and every second spelling of the same
    /// content (blank line, other order, `+1`, upper-case hex, spaces) is refused.
    #[test]
    fn standalone_model_manifest_has_one_spelling() {
        let m = StandaloneModelManifest {
            initial_chain: StandaloneChainLink { layer: 0, prev_receipt_digest: [0x24; 32] },
            tokens: vec![200_000, 5, 9, 1],
            out_token: 7,
            receipts: (0..3)
                .map(|k| StandaloneChainManifestEntry {
                    block_index: k,
                    standalone_anchor: [k as u8 + 1; 32],
                    com_h_in: if k == 0 { [0; 32] } else { [0x10 + k as u8; 32] },
                    com_h_out: if k == 2 { [0; 32] } else { [0x20 + k as u8; 32] },
                    receipt_path: format!("r{k}.rec2"),
                })
                .collect(),
        };
        let enc = encode_standalone_model_manifest(&m).expect("encode");
        assert!(enc.starts_with("STWOMDL2\ninitial_chain 0:") && enc.contains("\ntokens 200000,5,9,1\nout_token 7\n"));
        assert_eq!(decode_standalone_model_manifest(enc.as_bytes()).expect("decode"), m);
        let others = [
            enc.replacen("out_token 7", "out_token +7", 1),
            enc.replacen("tokens 200000", "tokens 0200000", 1),
            enc.replacen("tokens 200000,5", "tokens 200000, 5", 1),
            enc.replacen("receipt 1 ", "receipt  1 ", 1),
            enc.replacen("receipt 1", "receipt 01", 1),
            enc.replacen("\ntokens", "\n\ntokens", 1),
            enc.to_uppercase(),
            format!("{enc}\n"),
            enc.replacen("out_token 7\n", "", 1),
        ];
        for other in others {
            assert!(decode_standalone_model_manifest(other.as_bytes()).is_err(), "accepted: {other:?}");
        }
        let swapped = {
            let mut lines: Vec<&str> = enc.lines().collect();
            lines.swap(2, 3);
            lines.join("\n") + "\n"
        };
        assert!(decode_standalone_model_manifest(swapped.as_bytes()).is_err());
        let mut bad_idx = m.clone();
        bad_idx.receipts[1].block_index = 2;
        assert!(encode_standalone_model_manifest(&bad_idx).is_err());
        let mut no_tokens = m;
        no_tokens.tokens.clear();
        assert!(encode_standalone_model_manifest(&no_tokens).is_err());
    }

    #[test]
    fn standalone_chain_anchor_binds_block_order_and_count() {
        let (a, b) = ([0x11u8; 32], [0x22u8; 32]);
        let ab = compute_standalone_chain_anchor(&[a, b]);
        assert_ne!(ab, compute_standalone_chain_anchor(&[b, a]), "order");
        assert_ne!(ab, compute_standalone_chain_anchor(&[a]), "prefix");
        assert_ne!(ab, compute_standalone_chain_anchor(&[a, b, b]), "extra block");
        assert_ne!(
            compute_standalone_chain_anchor(&[a, a]),
            compute_standalone_chain_anchor(&[a]),
            "repeated block"
        );
    }
}
