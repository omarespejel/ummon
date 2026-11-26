//! Stage A Mode 11 (`STAGE_A_RECEIPT_MODE_ZK_FULL = 11`) standalone verifier:
//! Full Zero-Knowledge MoE FFN with private expert routing (`routes` & `route_weights_q15` hidden),
//! private in-witness softmax Q0.15 + canonical Top-K verification, blinded private-operand
//! matmul sum-checks, and correlated cross-layer hidden-residual continuity (`Option R7-B`).

use super::*;

pub const ZK_FULL_EXTRA_WITNESS_COLS: usize = 12;
pub const ZK_FULL_GKR_TABLES: usize = 5;
pub const ZK_FULL_UNMASKED_GKR_INSTANCES: usize = 44;
pub const ZK_FULL_CFS_LOG_ROWS: usize = 14;
pub const ZK_FULL_CFS_ROWS: usize = 1 << ZK_FULL_CFS_LOG_ROWS;
pub const ZK_FULL_AUX_MASK_SUB_BITS: usize = 14;

pub const COL_ZF_X_IN: usize = 17;
pub const COL_ZF_OUT_Q: usize = 18;
pub const COL_ZF_ONE_HOT: usize = 19;
pub const COL_ZF_OH_RW: usize = 20;
pub const COL_ZF_ROUTE_W: usize = 21;
pub const COL_ZF_ACTIVE: usize = 22;
pub const COL_ZF_RQ_EXP: usize = 23;
pub const COL_ZF_MAX_Q_SM: usize = 24;
pub const COL_ZF_THRESH_SM: usize = 25;
pub const COL_ZF_OH_LAST: usize = 26;
pub const COL_ZF_SOFTMAX_W: usize = 27;
pub const COL_ZF_DENOM_BC: usize = 28;
pub const COL_ZK_FULL_CFS_BLIND: usize = 29;

pub const Q_MU_XIN: usize = 2 * ZK_FULL_UNMASKED_GKR_INSTANCES; // 88
pub const Q_MU_OUT: usize = Q_MU_XIN + 1; // 89
pub const Q_MU_RT_CLAIM: usize = Q_MU_XIN + 2; // 90
pub const Q_MU_RT_X: usize = Q_MU_XIN + 3; // 91
pub const Q_MU_SH_GU_GATE: usize = Q_MU_XIN + 4; // 92
pub const Q_MU_SH_GU_UP: usize = Q_MU_XIN + 5; // 93
pub const Q_MU_SH_GU_X: usize = Q_MU_XIN + 6; // 94
pub const Q_MU_SH_DN_CLAIM: usize = Q_MU_XIN + 7; // 95
pub const Q_MU_SH_DN_X: usize = Q_MU_XIN + 8; // 96
pub const Q_MU_REG_DN_CLAIM: usize = Q_MU_XIN + 9; // 97
pub const Q_MU_REG_DN_X: usize = Q_MU_XIN + 10; // 98
pub const Q_MU_DN_ROUTED_L: usize = Q_MU_XIN + 11; // 99
pub const Q_MU_DN_ROUTED_R: usize = Q_MU_XIN + 12; // 100
pub const Q_MU_REG_GU_X: usize = Q_MU_XIN + 13; // 101
pub const Q_MU_GU_ROUTED_L: usize = Q_MU_XIN + 14; // 102
pub const Q_MU_GU_ROUTED_R: usize = Q_MU_XIN + 15; // 103
pub const Q_MU_GU_GATE_BASE: usize = Q_MU_XIN + 16; // 104 .. 104 + 2*E

#[derive(Clone, Copy, Debug)]
pub struct ZkFullCfsLayout {
    pub rt_cfs_off: usize,
    pub reg_gu_cfs_off: usize,
    pub gu_routed_cfs_off: usize,
    pub sh_gu_cfs_off: usize,
    pub reg_dn_cfs_off: usize,
    pub dn_routed_cfs_off: usize,
    pub sh_dn_cfs_off: usize,
    pub gkr_libra_off: usize,
    pub wit_cfs_off: usize,
    pub pad_off: usize,
}

impl ZkFullCfsLayout {
    pub fn compute(
        log_d: usize,
        log_e: usize,
        log_b: usize,
        log_m: usize,
        log_m_sh: usize,
        log_k_pad: usize,
        n_gkr_layers: usize,
        ext_n_wit: usize,
    ) -> Result<Self, String> {
        let mut cur = 0usize;
        let rt_cfs_off = cur;
        cur += 4 * (2 + log_d);
        let reg_gu_cfs_off = cur;
        cur += 4 * (2 + log_e + log_d);
        let gu_routed_cfs_off = cur;
        cur += 4 * (2 + log_b);
        let sh_gu_cfs_off = cur;
        cur += 4 * (2 + log_d);
        let reg_dn_cfs_off = cur;
        cur += 4 * (2 + log_e + log_m);
        let dn_routed_cfs_off = cur;
        cur += 4 * (2 + log_b + log_k_pad);
        let sh_dn_cfs_off = cur;
        cur += 4 * (2 + log_m_sh);
        let gkr_libra_off = cur;
        cur += 2 * n_gkr_layers * (n_gkr_layers + 1);
        let wit_cfs_off = cur;
        cur += 4 * ext_n_wit;
        let pad_off = ZK_FULL_CFS_ROWS / 2;
        if cur > pad_off {
            return Err("Stage A Mode 11 CFS layout exceeds half-table capacity".into());
        }
        Ok(Self {
            rt_cfs_off,
            reg_gu_cfs_off,
            gu_routed_cfs_off,
            sh_gu_cfs_off,
            reg_dn_cfs_off,
            dn_routed_cfs_off,
            sh_dn_cfs_off,
            gkr_libra_off,
            wit_cfs_off,
            pad_off,
        })
    }
}

#[derive(Clone, Debug)]
pub struct StageAZkFullAux {
    pub rt_cfs_sum: SecureField,
    pub rt_cfs_eval: SecureField,
    pub rt_x_claim: SecureField,
    pub reg_gu_cfs_sum: SecureField,
    pub reg_gu_cfs_eval: SecureField,
    pub reg_gu_x_claim: SecureField,
    pub gu_routed_cfs_sum: SecureField,
    pub gu_routed_sumcheck: SumcheckProof,
    pub gu_routed_cfs_eval: SecureField,
    pub gu_routed_l_claim: SecureField,
    pub gu_routed_r_claim: SecureField,
    pub sh_gu_cfs_sum: SecureField,
    pub sh_gu_cfs_eval: SecureField,
    pub sh_gu_x_claim: SecureField,
    pub reg_dn_cfs_sum: SecureField,
    pub reg_dn_cfs_eval: SecureField,
    pub dn_routed_cfs_sum: SecureField,
    pub dn_routed_sumcheck: SumcheckProof,
    pub dn_routed_cfs_eval: SecureField,
    pub dn_routed_l_claim: SecureField,
    pub dn_routed_r_claim: SecureField,
    pub sh_dn_cfs_sum: SecureField,
    pub sh_dn_cfs_eval: SecureField,
    pub gkr_mask_evals: [SecureField; ZK_FULL_UNMASKED_GKR_INSTANCES],
    pub gkr_libra_sums: Vec<SecureField>,
    pub gkr_libra_evals: Vec<SecureField>,
    pub cont_claims: Vec<SecureField>,
    pub wit_cfs_sum: SecureField,
    pub wit_cfs_eval: SecureField,
    pub extra_witness_openings: [SecureField; ZK_FULL_EXTRA_WITNESS_COLS],
    pub cfs_pad_claim: SecureField,
    pub cfs_binding_sumcheck: SumcheckProof,
    pub blind_col_opening: SecureField,
}

pub fn validate_zk_full_statement_shape(
    statement: &StageANonlinearBatchStatement,
) -> Result<(), String> {
    let shape = statement.shape;
    shape.validate()?;
    if !shape.batch.is_power_of_two() {
        return Err("Stage A Mode 11 batch size must be a power of two".into());
    }
    if !shape.dimension.is_power_of_two() {
        return Err("Stage A Mode 11 dimension must be a power of two".into());
    }
    if !shape.experts.is_power_of_two() || shape.experts > 1024 {
        return Err("Stage A Mode 11 experts must be a power of two <= 1024".into());
    }
    if !statement.routes.is_empty()
        || !statement.route_weights_q15.is_empty()
        || !statement.router_outputs_raw.is_empty()
    {
        return Err(
            "Stage A Mode 11 statement must omit public routes, route_weights_q15, and router_outputs_raw"
                .into(),
        );
    }
    if !statement.output_q8.is_empty() {
        if statement.output_q8.len() != shape.batch {
            return Err("Stage A Mode 11 public output_q8 batch mismatch".into());
        }
        for row in &statement.output_q8 {
            if row.len() != shape.dimension
                || row.iter().any(|&v| !(-2048..=2047).contains(&v))
            {
                return Err("Stage A Mode 11 public output_q8 row shape or i12 range invalid".into());
            }
        }
    }
    Ok(())
}

/// Mirrors prover-side `validate_zk_full_arithmetic_domain_with_l1_bound`:
/// enforces that in-circuit LogUp-GKR bounds (`x_in in [-128, 127]` via `IoRangeQueries`,
/// `q in [-2048, 2047]` via `T_I12`/`T_SILU`, and `rem in [-2^{s-1}, 2^{s-1}]` via `T_REM`)
/// prevent any modular wrap-around in `m * raw == q * 2^s + rem` over `M31`.
fn validate_zk_full_arithmetic_domain_with_l1_bound(
    shape: StageAShape,
    policy: StageANumericPolicy,
    input_l1_bound: u128,
) -> Result<(), String> {
    validate_stage_a_arithmetic_domain_with_l1_bound(shape, policy, 1)?;
    let raw_bound = input_l1_bound
        .checked_mul(128u128)
        .ok_or_else(|| "Stage A Mode 11 unscaled projection bound overflow".to_string())?;
    if raw_bound > u128::from(P / 2) {
        return Err(format!(
            "Stage A unscaled projection bound {raw_bound} exceeds centered M31 range"
        ));
    }
    for (label, spec) in [
        ("router projection", policy.router),
        ("gate projection", policy.gate),
        ("up projection", policy.up),
    ] {
        if spec.output_bits != 12 {
            return Err(format!(
                "Stage A Mode 11 {label} output_bits {} must equal 12 to match in-circuit T_I12/T_SILU range tables",
                spec.output_bits
            ));
        }
        if !(1..=28).contains(&spec.shift) {
            return Err(format!(
                "Stage A Mode 11 {label} shift {} must lie in 1..=28 to match in-circuit T_REM range tables",
                spec.shift
            ));
        }
        let m = u128::try_from(spec.multiplier)
            .map_err(|_| format!("Stage A {label} multiplier must be positive"))?;
        let lhs_bound = raw_bound
            .checked_mul(m)
            .ok_or_else(|| format!("Stage A {label} scaled bound overflow"))?;
        let q_bound = 1u128 << (spec.output_bits - 1);
        let denom = 1u128 << spec.shift;
        let rhs_bound = q_bound * denom + denom / 2;
        if lhs_bound
            .checked_add(rhs_bound)
            .map_or(true, |sum| sum >= u128::from(P))
        {
            return Err(format!(
                "Stage A {label} modular-aliasing bound {} + {rhs_bound} reaches M31 modulus P",
                lhs_bound
            ));
        }
    }
    Ok(())
}

pub fn gkr_lane_geometry_for_zk_full(shape: StageAShape) -> Result<GkrLaneGeometry, String> {
    let base = GkrLaneGeometry::for_shape(shape)?;
    let k_pad = shape.top_k.next_power_of_two();
    let oh_rows = k_pad
        .checked_mul(base.rt_rows)
        .ok_or_else(|| "Stage A Mode 11 one_hot row overflow".to_string())?;
    let io_rows2 = 2usize
        .checked_mul(base.out_rows)
        .ok_or_else(|| "Stage A Mode 11 io row overflow".to_string())?;
    let min_wit = base
        .wit_len
        .max(oh_rows)
        .max(io_rows2)
        .max(8 * base.rt_rows)
        .max(64 * TABLE_ROWS)
        .next_power_of_two();
    let n_wit = min_wit.ilog2() as usize;
    Ok(GkrLaneGeometry {
        n_wit,
        wit_len: min_wit,
        sub_len: min_wit / 8,
        ..base
    })
}

pub(crate) fn gkr_lane_plan_for_zk_full(geom: &GkrLaneGeometry) -> Vec<GkrPlannedInstance> {
    let n_chunks = 1usize << LOG_NL_ROW_CHUNKS;
    let log_nl_chunk_rows = (geom.n_wit as u32) - LOG_NL_ROW_CHUNKS + 1;
    let log_out_rows = geom.out_rows.ilog2() + 1;
    let log_rt_rows = geom.rt_rows.ilog2() + 1;
    let log_table_rows = LOG_TABLE_ROWS + 1;
    let mut plan = Vec::with_capacity(ZK_FULL_UNMASKED_GKR_INSTANCES + 1);
    for row_chunk in 0..n_chunks {
        plan.push(GkrPlannedInstance {
            kind: GkrInstanceKind::NonlinearChunk {
                row_chunk,
                q_chunk: 0,
            },
            log_slots: 2,
            log_rows: log_nl_chunk_rows,
        });
        plan.push(GkrPlannedInstance {
            kind: GkrInstanceKind::NonlinearChunk {
                row_chunk,
                q_chunk: 1,
            },
            log_slots: 1,
            log_rows: log_nl_chunk_rows,
        });
    }
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::OutputChunk { q_chunk: 0 },
        log_slots: 2,
        log_rows: log_out_rows,
    });
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::OutputChunk { q_chunk: 1 },
        log_slots: 1,
        log_rows: log_out_rows,
    });
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::RoutingQueries,
        log_slots: 2,
        log_rows: log_rt_rows,
    });
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::SoftmaxQueriesA,
        log_slots: 2,
        log_rows: log_rt_rows,
    });
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::SoftmaxQueriesB,
        log_slots: 2,
        log_rows: log_rt_rows,
    });
    // Second softmax range chunk (`rw_half_hi`, `rw_par`): reuse `SoftmaxQueriesB` with `log_slots = 1`.
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::SoftmaxQueriesB,
        log_slots: 1,
        log_rows: log_rt_rows,
    });
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::IoRangeQueries,
        log_slots: 1,
        log_rows: log_out_rows,
    });
    for t in 0..ZK_FULL_GKR_TABLES {
        plan.push(GkrPlannedInstance {
            kind: GkrInstanceKind::Table(t),
            log_slots: 0,
            log_rows: log_table_rows,
        });
    }
    debug_assert_eq!(plan.len(), ZK_FULL_UNMASKED_GKR_INSTANCES);
    plan.push(GkrPlannedInstance {
        kind: GkrInstanceKind::MaskingInstance,
        log_slots: 0,
        log_rows: ZK_MASK_INSTANCE_LOG_ROWS,
    });
    plan
}

pub fn put_stage_a_zk_full_aux(out: &mut Vec<u8>, z: &StageAZkFullAux) -> Result<(), String> {
    if !z.cont_claims.is_empty() && z.cont_claims.len() != 2 {
        return Err("Stage A Mode 11 cont_claims must have length 0 or 2".into());
    }
    put_secure_field(out, z.rt_cfs_sum);
    put_secure_field(out, z.rt_cfs_eval);
    put_secure_field(out, z.rt_x_claim);
    put_secure_field(out, z.reg_gu_cfs_sum);
    put_secure_field(out, z.reg_gu_cfs_eval);
    put_secure_field(out, z.reg_gu_x_claim);
    put_secure_field(out, z.gu_routed_cfs_sum);
    put_sumcheck(out, &z.gu_routed_sumcheck)?;
    put_secure_field(out, z.gu_routed_cfs_eval);
    put_secure_field(out, z.gu_routed_l_claim);
    put_secure_field(out, z.gu_routed_r_claim);
    put_secure_field(out, z.sh_gu_cfs_sum);
    put_secure_field(out, z.sh_gu_cfs_eval);
    put_secure_field(out, z.sh_gu_x_claim);
    put_secure_field(out, z.reg_dn_cfs_sum);
    put_secure_field(out, z.reg_dn_cfs_eval);
    put_secure_field(out, z.dn_routed_cfs_sum);
    put_sumcheck(out, &z.dn_routed_sumcheck)?;
    put_secure_field(out, z.dn_routed_cfs_eval);
    put_secure_field(out, z.dn_routed_l_claim);
    put_secure_field(out, z.dn_routed_r_claim);
    put_secure_field(out, z.sh_dn_cfs_sum);
    put_secure_field(out, z.sh_dn_cfs_eval);
    put_secure_fields(out, &z.gkr_mask_evals)?;
    put_secure_fields(out, &z.gkr_libra_sums)?;
    put_secure_fields(out, &z.gkr_libra_evals)?;
    put_u32(out, z.cont_claims.len() as u32);
    for &c in &z.cont_claims {
        put_secure_field(out, c);
    }
    put_secure_field(out, z.wit_cfs_sum);
    put_secure_field(out, z.wit_cfs_eval);
    put_secure_fields(out, &z.extra_witness_openings)?;
    put_secure_field(out, z.cfs_pad_claim);
    put_sumcheck(out, &z.cfs_binding_sumcheck)?;
    put_secure_field(out, z.blind_col_opening);
    Ok(())
}

pub fn read_stage_a_zk_full_aux(
    reader: &mut ReceiptReader<'_>,
    n_gkr_layers: usize,
) -> Result<StageAZkFullAux, String> {
    let rt_cfs_sum = reader.read_secure_field("zf rt_cfs_sum")?;
    let rt_cfs_eval = reader.read_secure_field("zf rt_cfs_eval")?;
    let rt_x_claim = reader.read_secure_field("zf rt_x_claim")?;
    let reg_gu_cfs_sum = reader.read_secure_field("zf reg_gu_cfs_sum")?;
    let reg_gu_cfs_eval = reader.read_secure_field("zf reg_gu_cfs_eval")?;
    let reg_gu_x_claim = reader.read_secure_field("zf reg_gu_x_claim")?;
    let gu_routed_cfs_sum = reader.read_secure_field("zf gu_routed_cfs_sum")?;
    let gu_routed_sumcheck = reader.read_sumcheck("zf gu_routed_sumcheck")?;
    let gu_routed_cfs_eval = reader.read_secure_field("zf gu_routed_cfs_eval")?;
    let gu_routed_l_claim = reader.read_secure_field("zf gu_routed_l_claim")?;
    let gu_routed_r_claim = reader.read_secure_field("zf gu_routed_r_claim")?;
    let sh_gu_cfs_sum = reader.read_secure_field("zf sh_gu_cfs_sum")?;
    let sh_gu_cfs_eval = reader.read_secure_field("zf sh_gu_cfs_eval")?;
    let sh_gu_x_claim = reader.read_secure_field("zf sh_gu_x_claim")?;
    let reg_dn_cfs_sum = reader.read_secure_field("zf reg_dn_cfs_sum")?;
    let reg_dn_cfs_eval = reader.read_secure_field("zf reg_dn_cfs_eval")?;
    let dn_routed_cfs_sum = reader.read_secure_field("zf dn_routed_cfs_sum")?;
    let dn_routed_sumcheck = reader.read_sumcheck("zf dn_routed_sumcheck")?;
    let dn_routed_cfs_eval = reader.read_secure_field("zf dn_routed_cfs_eval")?;
    let dn_routed_l_claim = reader.read_secure_field("zf dn_routed_l_claim")?;
    let dn_routed_r_claim = reader.read_secure_field("zf dn_routed_r_claim")?;
    let sh_dn_cfs_sum = reader.read_secure_field("zf sh_dn_cfs_sum")?;
    let sh_dn_cfs_eval = reader.read_secure_field("zf sh_dn_cfs_eval")?;
    let mask_vec =
        reader.read_secure_fields_exact(ZK_FULL_UNMASKED_GKR_INSTANCES, "zf gkr_mask_evals")?;
    let gkr_mask_evals: [SecureField; ZK_FULL_UNMASKED_GKR_INSTANCES] = mask_vec
        .try_into()
        .map_err(|_| "zf gkr_mask_evals count mismatch".to_string())?;
    let gkr_libra_sums = reader.read_secure_fields_exact(n_gkr_layers, "zf gkr_libra_sums")?;
    let gkr_libra_evals = reader.read_secure_fields_exact(n_gkr_layers, "zf gkr_libra_evals")?;
    let cont_len = reader.read_u32("zf cont_claims len")? as usize;
    if cont_len != 0 && cont_len != 2 {
        return Err(format!("invalid Mode 11 cont_claims length {cont_len}"));
    }
    let mut cont_claims = Vec::with_capacity(cont_len);
    for _ in 0..cont_len {
        cont_claims.push(reader.read_secure_field("zf cont_claim")?);
    }
    let wit_cfs_sum = reader.read_secure_field("zf wit_cfs_sum")?;
    let wit_cfs_eval = reader.read_secure_field("zf wit_cfs_eval")?;
    let extra_vec =
        reader.read_secure_fields_exact(ZK_FULL_EXTRA_WITNESS_COLS, "zf extra_witness_openings")?;
    let extra_witness_openings: [SecureField; ZK_FULL_EXTRA_WITNESS_COLS] = extra_vec
        .try_into()
        .map_err(|_| "zf extra_witness_openings count mismatch".to_string())?;
    let cfs_pad_claim = reader.read_secure_field("zf cfs_pad_claim")?;
    let cfs_binding_sumcheck = reader.read_sumcheck("zf cfs_binding_sumcheck")?;
    let blind_col_opening = reader.read_secure_field("zf blind_col_opening")?;
    Ok(StageAZkFullAux {
        rt_cfs_sum,
        rt_cfs_eval,
        rt_x_claim,
        reg_gu_cfs_sum,
        reg_gu_cfs_eval,
        reg_gu_x_claim,
        gu_routed_cfs_sum,
        gu_routed_sumcheck,
        gu_routed_cfs_eval,
        gu_routed_l_claim,
        gu_routed_r_claim,
        sh_gu_cfs_sum,
        sh_gu_cfs_eval,
        sh_gu_x_claim,
        reg_dn_cfs_sum,
        reg_dn_cfs_eval,
        dn_routed_cfs_sum,
        dn_routed_sumcheck,
        dn_routed_cfs_eval,
        dn_routed_l_claim,
        dn_routed_r_claim,
        sh_dn_cfs_sum,
        sh_dn_cfs_eval,
        gkr_mask_evals,
        gkr_libra_sums,
        gkr_libra_evals,
        cont_claims,
        wit_cfs_sum,
        wit_cfs_eval,
        extra_witness_openings,
        cfs_pad_claim,
        cfs_binding_sumcheck,
        blind_col_opening,
    })
}

#[inline]
fn eq_all_zeros(pt: &[SecureField]) -> SecureField {
    pt.iter().map(|&r| SecureField::one() - r).product()
}

#[inline]
fn eval_rev_e_mle(r_exp: &[SecureField]) -> SecureField {
    let n = r_exp.len();
    let mut acc = SecureField::zero();
    for (b, &coord) in r_exp.iter().enumerate() {
        let w = 1u32 << (n - 1 - b);
        acc += (SecureField::one() - coord) * BaseField::from_u32_unchecked(w);
    }
    acc
}

fn eval_i16_rows_mle_at_pt(rows: &[Vec<i16>], r_tok: &[SecureField], r_dim: &[SecureField]) -> SecureField {
    let eq_tok = if r_tok.is_empty() {
        vec![SecureField::one()]
    } else {
        eq_evals(r_tok)
    };
    let eq_dim_i64 = eq_evals_i64(r_dim);
    eval_i16_matrix_mle_with_eq(rows, &eq_tok, &eq_dim_i64)
}

#[allow(clippy::too_many_arguments)]
fn verify_masked_weight_matmul_prefix(
    masked_claim: SecureField,
    cfs_sum: SecureField,
    cfs_eval: SecureField,
    x_claim: SecureField,
    sumcheck: &SumcheckProof,
    expected_inner_vars: usize,
    channel: &mut Production100Channel,
    label: &str,
) -> Result<(SecureField, SecureField, Vec<SecureField>, SecureField, SecureField), String> {
    channel.mix_felts(&[cfs_sum]);
    let rho = channel.draw_secure_felt();
    let init = masked_claim + rho * cfs_sum;
    let (assign, final_eval) = partially_verify(init, sumcheck, channel)
        .map_err(|e| format!("Stage A Mode 11 {label} sumcheck rejected: {e}"))?;
    if assign.len() != 2 + expected_inner_vars {
        return Err(format!("Stage A Mode 11 {label} assignment length mismatch"));
    }
    let tau_0 = assign[0];
    let tau_1 = assign[1];
    let inner = assign[2..].to_vec();
    if ((SecureField::one() - tau_0) * tau_1).is_zero() {
        return Err(format!(
            "Stage A Mode 11 {label} degenerate blinding challenge"
        ));
    }
    channel.mix_felts(&[cfs_eval, x_claim]);
    Ok((tau_0, tau_1, inner, final_eval, rho))
}

#[allow(clippy::too_many_arguments)]
fn finish_masked_weight_matmul_check(
    tau_0: SecureField,
    tau_1: SecureField,
    inner: &[SecureField],
    final_eval: SecureField,
    rho: SecureField,
    cfs_eval: SecureField,
    x_claim: SecureField,
    w_true_eval: SecureField,
    w_openings_to_mix: &[SecureField],
    channel: &mut Production100Channel,
    label: &str,
) -> Result<(), String> {
    channel.mix_felts(w_openings_to_mix);
    let a = (SecureField::one() - tau_0) * (SecureField::one() - tau_1);
    let b = tau_0 * (SecureField::one() - tau_1);
    let w_ext_eval = a * w_true_eval + b * eq_all_zeros(inner);
    let expected = x_claim * w_ext_eval + rho * cfs_eval;
    if final_eval != expected {
        return Err(format!(
            "Stage A Mode 11 {label} sumcheck final evaluation mismatch"
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_routed_quadratic_check(
    target_claim: SecureField,
    a_prev: SecureField,
    eq_pub_tok_point: &[SecureField],
    cfs_sum: SecureField,
    cfs_eval: SecureField,
    l_claim: SecureField,
    r_claim: SecureField,
    sumcheck: &SumcheckProof,
    log_tok: usize,
    extra_vars: usize,
    channel: &mut Production100Channel,
    label: &str,
) -> Result<(SecureField, SecureField, Vec<SecureField>), String> {
    channel.mix_felts(&[cfs_sum]);
    let rho = channel.draw_secure_felt();
    let init = target_claim + rho * cfs_sum;
    let (assign, final_eval) = partially_verify(init, sumcheck, channel)
        .map_err(|e| format!("Stage A Mode 11 {label} sumcheck rejected: {e}"))?;
    if assign.len() != 2 + log_tok + extra_vars {
        return Err(format!("Stage A Mode 11 {label} assignment length mismatch"));
    }
    let tau_0 = assign[0];
    let tau_1 = assign[1];
    let inner = assign[2..].to_vec();
    if ((SecureField::one() - tau_0) * tau_1).is_zero() || (tau_0 * tau_1).is_zero() {
        return Err(format!(
            "Stage A Mode 11 {label} degenerate routed blinding challenge"
        ));
    }
    channel.mix_felts(&[cfs_eval, l_claim, r_claim]);
    let tok_pt = &inner[..log_tok];
    let eq_tok_val = eq(eq_pub_tok_point, tok_pt);
    let e_pub = (SecureField::one() - tau_1)
        * ((SecureField::one() - tau_0) * a_prev * eq_tok_val + tau_0 * eq_all_zeros(&inner));
    let expected = l_claim * r_claim * e_pub + rho * cfs_eval;
    if final_eval != expected {
        return Err(format!(
            "Stage A Mode 11 {label} sumcheck final evaluation mismatch"
        ));
    }
    Ok((tau_0, tau_1, inner))
}

struct ZkFullLookupBinding {
    tau_nl: SecureField,
    tau_out: SecureField,
    tau_rt: SecureField,
    tau_tbl: SecureField,
    r_nl: Vec<SecureField>,
    r_gkr_out: Vec<SecureField>,
    r_gkr_rt: Vec<SecureField>,
    r_gkr_tbl: Vec<SecureField>,
    nl_col_weights: [SecureField; 10],
    gkr_nl_claim: SecureField,
    out_limb_weights: [SecureField; 6],
    gkr_out_claim: SecureField,
    rt_sub_weights: [SecureField; 7],
    gkr_rt_claim: SecureField,
    sma_rq_exp_weights: [SecureField; 5],
    sma_maxq_u2_weight: SecureField,
    gkr_sma_claim: SecureField,
    smb_maxq_weights: [SecureField; 6],
    smb_th_weights: [SecureField; 3],
    gkr_smb_claim: SecureField,
    io_xin_weights: [SecureField; 2],
    io_outq_weights: [SecureField; 2],
    gkr_io_claim: SecureField,
    tbl_mult_claims: [SecureField; ZK_FULL_GKR_TABLES],
}

fn zk_full_slot0_mask_denominators(
    policy: StageANumericPolicy,
    tables: &GkrLanePublicTables,
    challenges: &GkrLaneLookupChallenges,
    exp_alphas: [SecureField; 2],
    exp_z: SecureField,
) -> [[SecureField; 2]; ZK_FULL_UNMASKED_GKR_INSTANCES] {
    let out_widths = out_limb_widths(policy.output.shift);
    let d_nl_q0 = -challenges.silu_z;
    let d_nl_q1 =
        challenges.rem_alphas[0] * BaseField::from_u32_unchecked(policy.up.shift) - challenges.rem_z;
    let d_out_q0 = challenges.width_alphas[0]
        * BaseField::from_u32_unchecked(out_widths[0] as u32)
        - challenges.width_z;
    let d_out_q1 = challenges.width_alphas[0]
        * BaseField::from_u32_unchecked(out_widths[1] as u32)
        - challenges.width_z;
    let d_rt = -challenges.range_z;
    let d_sma = -exp_z;
    let d_smb0 =
        challenges.width_alphas[0] * BaseField::from_u32_unchecked(10) - challenges.width_z;
    let d_smb1 =
        challenges.width_alphas[0] * BaseField::from_u32_unchecked(8) - challenges.width_z;
    let d_io = challenges.width_alphas[0] * BaseField::from_u32_unchecked(8)
        - challenges.width_alphas[1] * BaseField::from_u32_unchecked(1920)
        - challenges.width_z;

    let tbl_den = |t_idx: usize, row: usize| -> SecureField {
        let sc = BaseField::from_u32_unchecked(row as u32);
        let sp = BaseField::from_u32_unchecked((row & 1) as u32);
        match t_idx {
            0 => {
                challenges.silu_alphas[0] * sc
                    + challenges.silu_alphas[1] * tables.silu_val[row]
                    + challenges.silu_alphas[2] * sp
                    - challenges.silu_z
            }
            1 => challenges.range_alphas[0] * sc + challenges.range_alphas[1] * sp - challenges.range_z,
            2 => {
                challenges.width_alphas[0] * tables.width_tag[row]
                    + challenges.width_alphas[1] * tables.width_val[row]
                    - challenges.width_z
            }
            3 => {
                challenges.rem_alphas[0] * tables.rem_shift[row]
                    + challenges.rem_alphas[1] * tables.rem_val[row]
                    + challenges.rem_alphas[2] * tables.rem_par[row]
                    - challenges.rem_z
            }
            4 => {
                exp_alphas[0] * tables.exp_code[row]
                    + exp_alphas[1] * tables.exp_val[row]
                    - exp_z
            }
            _ => unreachable!(),
        }
    };

    std::array::from_fn(|k| {
        if k < 32 {
            let d = if k % 2 == 0 { d_nl_q0 } else { d_nl_q1 };
            [d, d]
        } else if k == 32 {
            [d_out_q0, d_out_q0]
        } else if k == 33 {
            [d_out_q1, d_out_q1]
        } else if k == 34 {
            [d_rt, d_rt]
        } else if k == 35 {
            [d_sma, d_sma]
        } else if k == 36 {
            [d_smb0, d_smb0]
        } else if k == 37 {
            [d_smb1, d_smb1]
        } else if k == 38 {
            [d_io, d_io]
        } else {
            let t_idx = k - 39;
            [tbl_den(t_idx, 0), tbl_den(t_idx, 1)]
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_zk_full_gkr_binding(
    artifact: &GkrArtifact,
    geom: &GkrLaneGeometry,
    policy: StageANumericPolicy,
    tables: &GkrLanePublicTables,
    challenges: &GkrLaneLookupChallenges,
    exp_alphas: [SecureField; 2],
    exp_z: SecureField,
    channel: &mut Production100Channel,
    gkr_mask_evals: &[SecureField; ZK_FULL_UNMASKED_GKR_INSTANCES],
) -> Result<ZkFullLookupBinding, String> {
    let plan = gkr_lane_plan_for_zk_full(geom);
    if artifact.claims_to_verify_by_instance.len() != plan.len() {
        return Err("Stage A Mode 11 GKR instance count mismatch".into());
    }
    let chunk_point = channel.draw_secure_felts(LOG_NL_ROW_CHUNKS as usize);
    let nl_qchunk_weights = channel.draw_secure_felts(2);
    let out_qchunk_weights = channel.draw_secure_felts(2);
    let smb_qchunk_weights = channel.draw_secure_felts(2);

    let n_layers = artifact.ood_point.len();
    let log_nl_chunk_rows = (geom.n_wit as u32 - LOG_NL_ROW_CHUNKS) as usize;
    let log_out_rows = geom.out_rows.ilog2() as usize;
    let log_rt_rows = geom.rt_rows.ilog2() as usize;

    let tau_nl = artifact.ood_point[n_layers - log_nl_chunk_rows - 1];
    let tau_out = artifact.ood_point[n_layers - log_out_rows - 1];
    let tau_rt = artifact.ood_point[n_layers - log_rt_rows - 1];
    let tau_tbl = artifact.ood_point[n_layers - (LOG_TABLE_ROWS as usize) - 1];
    let expected_num_nl = SecureField::one() - tau_nl;
    let expected_num_out = SecureField::one() - tau_out;
    let expected_num_rt = SecureField::one() - tau_rt;

    let mut r_nl = chunk_point.clone();
    r_nl.extend_from_slice(&artifact.ood_point[n_layers - log_nl_chunk_rows..]);
    let r_gkr_out = artifact.ood_point[n_layers - log_out_rows..].to_vec();
    let r_gkr_rt = artifact.ood_point[n_layers - log_rt_rows..].to_vec();
    let r_gkr_tbl = artifact.ood_point[n_layers - (LOG_TABLE_ROWS as usize)..].to_vec();

    let tbl_eq_i64 = eq_evals_i64(&r_gkr_tbl);
    let eval_bf = |col: &[BaseField]| -> SecureField {
        let mut a = [0i64; 4];
        for (v, w) in col.iter().zip(&tbl_eq_i64) {
            let v64 = i64::from(v.0);
            a[0] += w[0] * v64;
            a[1] += w[1] * v64;
            a[2] += w[2] * v64;
            a[3] += w[3] * v64;
        }
        acc_to_secure(a)
    };
    let mut sc_mle = SecureField::zero();
    for (b, &coord) in r_gkr_tbl.iter().enumerate() {
        let bit_weight = 1u32 << ((LOG_TABLE_ROWS as usize) - 1 - b);
        sc_mle += coord * BaseField::from_u32_unchecked(bit_weight);
    }
    let sp_mle = *r_gkr_tbl.last().expect("LOG_TABLE_ROWS > 0");
    let sv_mle = eval_bf(&tables.silu_val);
    let wt_mle = eval_bf(&tables.width_tag);
    let wv_mle = eval_bf(&tables.width_val);
    let rs_mle = eval_bf(&tables.rem_shift);
    let rv_mle = eval_bf(&tables.rem_val);
    let rp_mle = eval_bf(&tables.rem_par);
    let ec_mle = eval_bf(&tables.exp_code);
    let ev_mle = eval_bf(&tables.exp_val);

    let expected_tbl_den = [
        challenges.silu_alphas[0] * sc_mle
            + challenges.silu_alphas[1] * sv_mle
            + challenges.silu_alphas[2] * sp_mle
            - challenges.silu_z,
        challenges.range_alphas[0] * sc_mle + challenges.range_alphas[1] * sp_mle
            - challenges.range_z,
        challenges.width_alphas[0] * wt_mle + challenges.width_alphas[1] * wv_mle
            - challenges.width_z,
        challenges.rem_alphas[0] * rs_mle
            + challenges.rem_alphas[1] * rv_mle
            + challenges.rem_alphas[2] * rp_mle
            - challenges.rem_z,
        exp_alphas[0] * ec_mle + exp_alphas[1] * ev_mle - exp_z,
    ];

    let inst_prefix = |inst: &GkrPlannedInstance| -> &[SecureField] {
        let start = n_layers - inst.n_variables();
        &artifact.ood_point[start..start + inst.log_slots as usize]
    };
    let inst_mask_term = |idx: usize, inst: &GkrPlannedInstance| -> SecureField {
        let start = n_layers - inst.n_variables();
        let slot_pt = &artifact.ood_point[start..start + inst.log_slots as usize];
        let slot0_eq: SecureField = slot_pt.iter().map(|&s| SecureField::one() - s).product();
        let tau_k = artifact.ood_point[start + inst.log_slots as usize];
        let log_base_rows = (inst.log_rows - 1) as usize;
        let row_hi_pt = &artifact.ood_point[n_layers - log_base_rows..n_layers - 1];
        let row0_hi_eq: SecureField = row_hi_pt.iter().map(|&r| SecureField::one() - r).product();
        slot0_eq * tau_k * row0_hi_eq * gkr_mask_evals[idx]
    };

    let mut nl_col_weights = [SecureField::zero(); 10];
    let p_nl0 = inst_prefix(&plan[0]);
    for slot in 0..4usize {
        let sw = nl_qchunk_weights[0] * eq(p_nl0, &slot_bits_felt(slot, 2));
        match slot {
            0 => {
                nl_col_weights[COL_WIT_GATE_Q] += sw * challenges.silu_alphas[0];
                nl_col_weights[COL_WIT_SILU] += sw * challenges.silu_alphas[1];
                nl_col_weights[COL_WIT_GATE_PARITY] += sw * challenges.silu_alphas[2];
            }
            1 => {
                nl_col_weights[COL_WIT_UP_Q] += sw * challenges.range_alphas[0];
                nl_col_weights[COL_WIT_UP_PARITY] += sw * challenges.range_alphas[1];
            }
            2 => {
                nl_col_weights[COL_WIT_HIDDEN_Q] += sw * challenges.range_alphas[0];
                nl_col_weights[COL_WIT_HIDDEN_PARITY] += sw * challenges.range_alphas[1];
            }
            3 => {
                nl_col_weights[COL_WIT_GATE_REM] += sw * challenges.rem_alphas[1];
                nl_col_weights[COL_WIT_GATE_PARITY] += sw * challenges.rem_alphas[2];
            }
            _ => unreachable!(),
        }
    }
    let nl0_const = {
        let s0 = eq(p_nl0, &slot_bits_felt(0, 2)) * (-challenges.silu_z);
        let s1 = eq(p_nl0, &slot_bits_felt(1, 2)) * (-challenges.range_z);
        let s2 = eq(p_nl0, &slot_bits_felt(2, 2)) * (-challenges.range_z);
        let s3 = eq(p_nl0, &slot_bits_felt(3, 2))
            * (challenges.rem_alphas[0] * BaseField::from_u32_unchecked(policy.gate.shift)
                - challenges.rem_z);
        s0 + s1 + s2 + s3
    };
    let p_nl1 = inst_prefix(&plan[1]);
    for slot in 0..2usize {
        let sw = nl_qchunk_weights[1] * eq(p_nl1, &slot_bits_felt(slot, 1));
        match slot {
            0 => {
                nl_col_weights[COL_WIT_UP_REM] += sw * challenges.rem_alphas[1];
                nl_col_weights[COL_WIT_UP_PARITY] += sw * challenges.rem_alphas[2];
            }
            1 => {
                nl_col_weights[COL_WIT_PROD_REM] += sw * challenges.rem_alphas[1];
                nl_col_weights[COL_WIT_HIDDEN_PARITY] += sw * challenges.rem_alphas[2];
            }
            _ => unreachable!(),
        }
    }
    let nl1_const = {
        let s0 = eq(p_nl1, &slot_bits_felt(0, 1))
            * (challenges.rem_alphas[0] * BaseField::from_u32_unchecked(policy.up.shift)
                - challenges.rem_z);
        let s1 = eq(p_nl1, &slot_bits_felt(1, 1))
            * (challenges.rem_alphas[0] * BaseField::from_u32_unchecked(policy.product.shift)
                - challenges.rem_z);
        s0 + s1
    };

    let out_widths = out_limb_widths(policy.output.shift);
    let mut out_limb_weights = [SecureField::zero(); 6];
    let mut gkr_nl_claim = SecureField::zero();
    let mut gkr_out_claim = SecureField::zero();
    let mut rt_sub_weights = [SecureField::zero(); 7];
    let mut gkr_rt_claim = SecureField::zero();
    let mut sma_rq_exp_weights = [SecureField::zero(); 5];
    let mut sma_maxq_u2_weight = SecureField::zero();
    let mut gkr_sma_claim = SecureField::zero();
    let mut smb_maxq_weights = [SecureField::zero(); 6];
    let mut smb_th_weights = [SecureField::zero(); 3];
    let mut gkr_smb_claim = SecureField::zero();
    let mut io_xin_weights = [SecureField::zero(); 2];
    let mut io_outq_weights = [SecureField::zero(); 2];
    let mut gkr_io_claim = SecureField::zero();
    let mut tbl_mult_claims = [SecureField::zero(); ZK_FULL_GKR_TABLES];

    for (idx, inst) in plan.iter().enumerate() {
        let [num, den]: [SecureField; 2] = artifact.claims_to_verify_by_instance[idx]
            .as_slice()
            .try_into()
            .map_err(|_| "Stage A Mode 11 GKR instance claim shape".to_string())?;
        match inst.kind {
            GkrInstanceKind::NonlinearChunk { row_chunk, q_chunk } => {
                if num != expected_num_nl + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR nonlinear query numerator mismatch".into());
                }
                let cw =
                    eq(&chunk_point, &slot_bits_felt(row_chunk, LOG_NL_ROW_CHUNKS as usize));
                let c_part = if q_chunk == 0 { nl0_const } else { nl1_const };
                gkr_nl_claim += cw * nl_qchunk_weights[q_chunk] * (den - c_part);
            }
            GkrInstanceKind::OutputChunk { q_chunk } => {
                if num != expected_num_out + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR output query numerator mismatch".into());
                }
                let pref = inst_prefix(inst);
                let n_slots = 1usize << inst.log_slots;
                let mut c_part = SecureField::zero();
                for slot in 0..n_slots {
                    let limb_idx = q_chunk * 4 + slot;
                    let sw = eq(pref, &slot_bits_felt(slot, inst.log_slots as usize));
                    out_limb_weights[limb_idx] +=
                        out_qchunk_weights[q_chunk] * sw * challenges.width_alphas[1];
                    c_part += sw
                        * (challenges.width_alphas[0]
                            * BaseField::from_u32_unchecked(out_widths[limb_idx % 3] as u32)
                            - challenges.width_z);
                }
                gkr_out_claim += out_qchunk_weights[q_chunk] * (den - c_part);
            }
            GkrInstanceKind::RoutingQueries => {
                if num != expected_num_rt + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR routing query numerator mismatch".into());
                }
                let pref = inst_prefix(inst);
                let mut c_part = SecureField::zero();
                for slot in 0..4usize {
                    let sw = eq(pref, &slot_bits_felt(slot, 2));
                    match slot {
                        0 => {
                            rt_sub_weights[0] += sw * challenges.range_alphas[0];
                            rt_sub_weights[2] += sw * challenges.range_alphas[1];
                            c_part += sw * (-challenges.range_z);
                        }
                        1 => {
                            rt_sub_weights[1] += sw * challenges.rem_alphas[1];
                            rt_sub_weights[2] += sw * challenges.rem_alphas[2];
                            c_part += sw
                                * (challenges.rem_alphas[0]
                                    * BaseField::from_u32_unchecked(policy.router.shift)
                                    - challenges.rem_z);
                        }
                        2 => {
                            rt_sub_weights[5] += sw * challenges.width_alphas[1];
                            c_part += sw
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(11)
                                    - challenges.width_z);
                        }
                        3 => {
                            rt_sub_weights[6] += sw * challenges.width_alphas[1];
                            c_part += sw
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(11)
                                    - challenges.width_z);
                        }
                        _ => unreachable!(),
                    }
                }
                gkr_rt_claim = den - c_part;
            }
            GkrInstanceKind::SoftmaxQueriesA => {
                if num != expected_num_rt + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR SoftmaxQueriesA numerator mismatch".into());
                }
                let pref = inst_prefix(inst);
                let mut c_part = SecureField::zero();
                for slot in 0..4usize {
                    let sw = eq(pref, &slot_bits_felt(slot, 2));
                    match slot {
                        0 => {
                            sma_rq_exp_weights[1] += sw * exp_alphas[0];
                            sma_rq_exp_weights[2] += sw * exp_alphas[1];
                            c_part += sw * (-exp_z);
                        }
                        1 => {
                            sma_rq_exp_weights[3] += sw * challenges.width_alphas[1];
                            c_part += sw
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(10)
                                    - challenges.width_z);
                        }
                        2 => {
                            sma_rq_exp_weights[4] += sw * challenges.width_alphas[1];
                            c_part += sw
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(9)
                                    - challenges.width_z);
                        }
                        3 => {
                            sma_maxq_u2_weight += sw * challenges.width_alphas[1];
                            c_part += sw
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(9)
                                    - challenges.width_z);
                        }
                        _ => unreachable!(),
                    }
                }
                gkr_sma_claim = den - c_part;
            }
            GkrInstanceKind::SoftmaxQueriesB => {
                if num != expected_num_rt + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR SoftmaxQueriesB numerator mismatch".into());
                }
                let pref = inst_prefix(inst);
                let mut c_part = SecureField::zero();
                if inst.log_slots == 2 {
                    let qw = smb_qchunk_weights[0];
                    for slot in 0..4usize {
                        let sw = eq(pref, &slot_bits_felt(slot, 2));
                        let width_tag = match slot {
                            0 => {
                                smb_maxq_weights[2] += qw * sw * challenges.width_alphas[1];
                                10u32
                            }
                            1 => {
                                smb_maxq_weights[3] += qw * sw * challenges.width_alphas[1];
                                9u32
                            }
                            2 => {
                                smb_maxq_weights[4] += qw * sw * challenges.width_alphas[1];
                                9u32
                            }
                            3 => {
                                smb_maxq_weights[5] += qw * sw * challenges.width_alphas[1];
                                7u32
                            }
                            _ => unreachable!(),
                        };
                        c_part += sw
                            * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(width_tag)
                                - challenges.width_z);
                    }
                    gkr_smb_claim += qw * (den - c_part);
                } else {
                    let qw = smb_qchunk_weights[1];
                    for slot in 0..2usize {
                        let sw = eq(pref, &slot_bits_felt(slot, 1));
                        let width_tag = match slot {
                            0 => {
                                smb_th_weights[1] += qw * sw * challenges.width_alphas[1];
                                8u32
                            }
                            1 => {
                                smb_th_weights[2] += qw * sw * challenges.width_alphas[1];
                                1u32
                            }
                            _ => unreachable!(),
                        };
                        c_part += sw
                            * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(width_tag)
                                - challenges.width_z);
                    }
                    gkr_smb_claim += qw * (den - c_part);
                }
            }
            GkrInstanceKind::IoRangeQueries => {
                if num != expected_num_out + inst_mask_term(idx, inst) {
                    return Err("Stage A Mode 11 GKR IoRangeQueries numerator mismatch".into());
                }
                let pref = inst_prefix(inst);
                let s0 = eq(pref, &slot_bits_felt(0, 1));
                let s1 = eq(pref, &slot_bits_felt(1, 1));
                let c1920 = BaseField::from_u32_unchecked(1920);
                io_xin_weights[0] += s0 * challenges.width_alphas[1];
                io_outq_weights[0] += s1 * challenges.range_alphas[0];
                io_outq_weights[1] += s1 * challenges.range_alphas[1];
                let c_part = s0
                    * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(8)
                        - challenges.width_alphas[1] * c1920
                        - challenges.width_z)
                    + s1 * (-challenges.range_z);
                gkr_io_claim = den - c_part;
            }
            GkrInstanceKind::Table(t_idx) => {
                if den != expected_tbl_den[t_idx] {
                    return Err(format!(
                        "Stage A Mode 11 GKR table {t_idx} denominator mismatch"
                    ));
                }
                tbl_mult_claims[t_idx] = -num + inst_mask_term(idx, inst);
            }
            GkrInstanceKind::MaskingInstance => {
                let r_mask = &artifact.ood_point[n_layers - (ZK_MASK_INSTANCE_LOG_ROWS as usize)..];
                let mask_dens = zk_full_slot0_mask_denominators(
                    policy, tables, challenges, exp_alphas, exp_z,
                );
                let mut den_vec = vec![-challenges.silu_z; ZK_MASK_INSTANCE_ROWS];
                for k in 0..ZK_FULL_UNMASKED_GKR_INSTANCES {
                    den_vec[2 * k] = mask_dens[k][0];
                    den_vec[2 * k + 1] = mask_dens[k][1];
                }
                let expected_mask_den = mle_at(&den_vec, r_mask);
                if den != expected_mask_den {
                    return Err(
                        "Stage A Mode 11 GKR masking instance denominator mismatch".into(),
                    );
                }
                let eq9 = eq_evals(&r_mask[..(ZK_MASK_INSTANCE_LOG_ROWS as usize) - 1]);
                let mut expected_mask_num = SecureField::zero();
                for k in 0..ZK_FULL_UNMASKED_GKR_INSTANCES {
                    expected_mask_num -= eq9[k] * gkr_mask_evals[k];
                }
                if num != expected_mask_num {
                    return Err("Stage A Mode 11 GKR masking instance numerator mismatch".into());
                }
            }
        }
    }

    Ok(ZkFullLookupBinding {
        tau_nl,
        tau_out,
        tau_rt,
        tau_tbl,
        r_nl,
        r_gkr_out,
        r_gkr_rt,
        r_gkr_tbl,
        nl_col_weights,
        gkr_nl_claim,
        out_limb_weights,
        gkr_out_claim,
        rt_sub_weights,
        gkr_rt_claim,
        sma_rq_exp_weights,
        sma_maxq_u2_weight,
        gkr_sma_claim,
        smb_maxq_weights,
        smb_th_weights,
        gkr_smb_claim,
        io_xin_weights,
        io_outq_weights,
        gkr_io_claim,
        tbl_mult_claims,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_zk_full_core(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
    defer_cprs: bool,
    witness_f_override: Option<usize>,
    r_cont: Option<&[SecureField]>,
) -> Result<(StageARegisteredVerifierRun, Option<StageAGkrLaneDeferredCprs>), String> {
    if &context.anchor_digest() != anchor {
        return Err("Stage A Mode 11 context does not match the public model anchor".into());
    }
    if proof.legacy_11col || proof.zk_aux.is_some() {
        return Err("Stage A Mode 11 rejects legacy_11col and Mode 10 zk_aux".into());
    }
    if !proof.r4_routing_sumcheck.round_polys.is_empty() {
        return Err("Stage A Mode 11 requires empty r4_routing_sumcheck".into());
    }
    if defer_cprs
        && (!proof.cprs_router_proof.is_empty()
            || !proof.cprs_regular_proof.is_empty()
            || !proof.cprs_shared_proof.is_empty()
            || !proof.cprs_witness_proof.is_empty())
    {
        return Err("deferred Stage A Mode 11 verification expects empty per-layer CPRS proofs".into());
    }
    policy.validate()?;
    validate_zk_full_statement_shape(statement)?;
    let shape = statement.shape;
    if !inputs.is_empty() {
        validate_fused_arithmetic_domain(shape, inputs, policy)?;
    } else {
        validate_zk_full_arithmetic_domain_with_l1_bound(
            shape,
            policy,
            128u128 * (shape.dimension as u128),
        )?;
    }
    if statement.policy_id != policy.id()
        || statement.policy_digest != policy.digest()?
        || statement.table_digest != public_tables_for_policy(policy)?.digest
        || statement.input_digest != stage_a_input_digest(inputs)
    {
        return Err("Stage A Mode 11 statement metadata mismatch".into());
    }
    if context.shape != shape
        || context.model_digest != statement.model_digest
        || context.manifest_digest != statement.model_registry_manifest_digest
    {
        return Err("Stage A Mode 11 context metadata mismatch".into());
    }

    let zf = proof
        .zk_full_aux
        .as_ref()
        .ok_or_else(|| "Stage A Mode 11 proof missing zk_full_aux".to_string())?;

    let all_experts: Vec<usize> = (0..shape.experts).collect();
    let cprs_params = context.validate_for_active_experts_any_f(&all_experts, profile)?;
    let geom = gkr_lane_geometry_for_zk_full(shape)?;
    let ext_n_wit = geom.n_wit + 1;
    let m_ext = ext_n_wit - 2;
    let eff_wit_f = witness_f_override
        .or_else(|| (cprs_params.router.f() != 4).then_some(cprs_params.router.f()));
    let witness_params = witness_cprs_params_for_mode_cols_and_f(
        ext_n_wit,
        profile,
        STAGE_A_RECEIPT_MODE_ZK_FULL,
        WITNESS_GROUP_COLS_ZK_FULL,
        eff_wit_f,
    )?;

    let e_count = shape.experts;
    if proof.regular_gate_up_claims.len() != 2 * e_count
        || proof.shared_gate_up_claims.len() != 2
        || proof.regular_gate_weight_openings.len() != e_count
        || proof.regular_up_weight_openings.len() != e_count
        || proof.regular_down_weight_openings.len() != e_count
    {
        return Err("Stage A Mode 11 claim or weight opening count mismatch".into());
    }

    let log_b = shape.batch.ilog2() as usize;
    let log_d = shape.dimension.ilog2() as usize;
    let log_e = shape.experts.ilog2() as usize;
    let log_m = geom.m_pad.ilog2() as usize;
    let log_m_sh = geom.sh_pad.ilog2() as usize;
    let k_pad = shape.top_k.next_power_of_two();
    let log_k_pad = k_pad.ilog2() as usize;

    let plan = gkr_lane_plan_for_zk_full(&geom);
    let n_gkr_layers = plan
        .iter()
        .map(GkrPlannedInstance::n_variables)
        .max()
        .unwrap_or(0);
    let cfs_layout = ZkFullCfsLayout::compute(
        log_d,
        log_e,
        log_b,
        log_m,
        log_m_sh,
        log_k_pad,
        n_gkr_layers,
        ext_n_wit,
    )?;

    let channel = &mut Production100Channel::default();
    channel.enter_scope(TranscriptScope::Statement);
    mix_gkr_lane_statement_with_mode(
        channel,
        statement,
        context,
        profile,
        chain,
        &proof.witness_commitment,
        STAGE_A_RECEIPT_MODE_ZK_FULL,
    );
    if let Some(ext_rc) = r_cont {
        channel.mix_felts(ext_rc);
    }

    // Draw `r_cfs_anchor` upfront so `v_cfs` masks all matmul, GKR, and witness sum-checks.
    channel.enter_scope(TranscriptScope::RouterBinding);
    let r_cfs_anchor = channel.draw_secure_felts(m_ext - ZK_FULL_CFS_LOG_ROWS);

    // 1. Blinded private-input Router matmul (`2 + log_D` rounds).
    let r_rt_tok = channel.draw_secure_felts(log_b);
    let r_rt_exp = channel.draw_secure_felts(log_e);
    let mut r_rt_init = r_rt_tok.clone();
    r_rt_init.extend_from_slice(&r_rt_exp);
    channel.mix_felts(&[proof.router.gate_claim]);
    let (tau_rt_0, tau_rt_1, s_rt_dim, rt_final, rt_rho) = verify_masked_weight_matmul_prefix(
        proof.router.gate_claim,
        zf.rt_cfs_sum,
        zf.rt_cfs_eval,
        zf.rt_x_claim,
        &proof.router.gate_proof,
        log_d,
        channel,
        "router",
    )?;
    finish_masked_weight_matmul_check(
        tau_rt_0,
        tau_rt_1,
        &s_rt_dim,
        rt_final,
        rt_rho,
        zf.rt_cfs_eval,
        zf.rt_x_claim,
        proof.router.wg_opening,
        &[proof.router.wg_opening],
        channel,
        "router",
    )?;
    let mut router_weight_point = s_rt_dim.clone();
    router_weight_point.extend_from_slice(&r_rt_exp);

    // 2. Blinded Gate/Up matmul + routed sum-checks.
    channel.enter_scope(TranscriptScope::BatchAggregation);
    let r_gu_tok = channel.draw_secure_felts(log_b);
    let r_gu_slot = channel.draw_secure_felts(log_k_pad);
    let regular_projection_point = channel.draw_secure_felts(log_m);
    let shared_projection_point = channel.draw_secure_felts(log_m_sh);
    let reg_proj_eq = eq_evals(&regular_projection_point);
    let sh_proj_eq = eq_evals(&shared_projection_point);

    channel.enter_scope(TranscriptScope::RegularGateUp);
    channel.mix_felts(&proof.regular_gate_up_claims);
    let reg_gu_batch_weights = channel.draw_secure_felts(2 * e_count);
    let mut reg_gu_masked_claim = SecureField::zero();
    for e in 0..e_count {
        reg_gu_masked_claim += reg_gu_batch_weights[e] * proof.regular_gate_up_claims[e]
            + reg_gu_batch_weights[e_count + e] * proof.regular_gate_up_claims[e_count + e];
    }
    let (tau_gu_0, tau_gu_1, reg_gu_inner, reg_gu_final, reg_gu_rho) =
        verify_masked_weight_matmul_prefix(
            reg_gu_masked_claim,
            zf.reg_gu_cfs_sum,
            zf.reg_gu_cfs_eval,
            zf.reg_gu_x_claim,
            &proof.regular_gate_up_sumcheck,
            log_e + log_d,
            channel,
            "regular_gate_up",
        )?;
    let r_gu_exp = reg_gu_inner[..log_e].to_vec();
    let r_gu_dim = reg_gu_inner[log_e..].to_vec();
    let reg_gu_exp_eq = eq_evals(&r_gu_exp);
    let mut reg_gu_w_eval = SecureField::zero();
    for e in 0..e_count {
        let comb_w = reg_gu_batch_weights[e] * proof.regular_gate_weight_openings[e]
            + reg_gu_batch_weights[e_count + e] * proof.regular_up_weight_openings[e];
        reg_gu_w_eval += reg_gu_exp_eq[e] * comb_w;
    }
    channel.mix_felts(&proof.regular_gate_weight_openings);
    finish_masked_weight_matmul_check(
        tau_gu_0,
        tau_gu_1,
        &reg_gu_inner,
        reg_gu_final,
        reg_gu_rho,
        zf.reg_gu_cfs_eval,
        zf.reg_gu_x_claim,
        reg_gu_w_eval,
        &proof.regular_up_weight_openings,
        channel,
        "regular_gate_up",
    )?;
    let a_gu = (SecureField::one() - tau_gu_0) * (SecureField::one() - tau_gu_1);
    let b_gu = tau_gu_0 * (SecureField::one() - tau_gu_1);
    let c_gu = (SecureField::one() - tau_gu_0) * tau_gu_1;
    let (tau_gur_0, tau_gur_1, r_gur_tok) = verify_routed_quadratic_check(
        zf.reg_gu_x_claim,
        a_gu,
        &r_gu_tok,
        zf.gu_routed_cfs_sum,
        zf.gu_routed_cfs_eval,
        zf.gu_routed_l_claim,
        zf.gu_routed_r_claim,
        &zf.gu_routed_sumcheck,
        log_b,
        0,
        channel,
        "gu_routed",
    )?;

    channel.enter_scope(TranscriptScope::SharedGateUp);
    channel.mix_felts(&proof.shared_gate_up_claims);
    let sh_gu_batch_weights = channel.draw_secure_felts(2);
    let sh_gu_masked_claim = sh_gu_batch_weights[0] * proof.shared_gate_up_claims[0]
        + sh_gu_batch_weights[1] * proof.shared_gate_up_claims[1];
    let sh_gu_w_eval = sh_gu_batch_weights[0] * proof.shared_gate_weight_opening
        + sh_gu_batch_weights[1] * proof.shared_up_weight_opening;
    let (tau_sgu_0, tau_sgu_1, s_sh_gu_dim, sgu_final, sgu_rho) =
        verify_masked_weight_matmul_prefix(
            sh_gu_masked_claim,
            zf.sh_gu_cfs_sum,
            zf.sh_gu_cfs_eval,
            zf.sh_gu_x_claim,
            &proof.shared_gate_up_sumcheck,
            log_d,
            channel,
            "shared_gate_up",
        )?;
    finish_masked_weight_matmul_check(
        tau_sgu_0,
        tau_sgu_1,
        &s_sh_gu_dim,
        sgu_final,
        sgu_rho,
        zf.sh_gu_cfs_eval,
        zf.sh_gu_x_claim,
        sh_gu_w_eval,
        &[
            proof.shared_gate_weight_opening,
            proof.shared_up_weight_opening,
        ],
        channel,
        "shared_gate_up",
    )?;

    // 3. Blinded Down matmul + routed sum-checks.
    channel.enter_scope(TranscriptScope::DownProjection);
    let r_dn_tok = channel.draw_secure_felts(log_b);
    let output_point = channel.draw_secure_felts(log_d);
    channel.mix_felts(&[proof.regular_down_claim, proof.shared_down_claim]);

    channel.enter_scope(TranscriptScope::RegularDown);
    let (tau_rdn_0, tau_rdn_1, reg_dn_inner, rdn_final, rdn_rho) =
        verify_masked_weight_matmul_prefix(
            proof.regular_down_claim,
            zf.reg_dn_cfs_sum,
            zf.reg_dn_cfs_eval,
            proof.regular_grouped_claim,
            &proof.regular_down_sumcheck,
            log_e + log_m,
            channel,
            "regular_down",
        )?;
    let r_dn_exp = reg_dn_inner[..log_e].to_vec();
    let r_dn_hid = reg_dn_inner[log_e..].to_vec();
    let expected_reg_w_eval = dot(&eq_evals(&r_dn_exp), &proof.regular_down_weight_openings);
    if proof.regular_weight_opening != expected_reg_w_eval {
        return Err("Stage A Mode 11 regular_weight_opening mismatch".into());
    }
    channel.mix_felts(&proof.regular_down_weight_openings);
    finish_masked_weight_matmul_check(
        tau_rdn_0,
        tau_rdn_1,
        &reg_dn_inner,
        rdn_final,
        rdn_rho,
        zf.reg_dn_cfs_eval,
        proof.regular_grouped_claim,
        proof.regular_weight_opening,
        &[proof.regular_weight_opening],
        channel,
        "regular_down",
    )?;
    let a_rdn = (SecureField::one() - tau_rdn_0) * (SecureField::one() - tau_rdn_1);
    let b_rdn = tau_rdn_0 * (SecureField::one() - tau_rdn_1);
    let c_rdn = (SecureField::one() - tau_rdn_0) * tau_rdn_1;
    let (tau_dnr_0, tau_dnr_1, dn_routed_inner) = verify_routed_quadratic_check(
        proof.regular_grouped_claim,
        a_rdn,
        &r_dn_tok,
        zf.dn_routed_cfs_sum,
        zf.dn_routed_cfs_eval,
        zf.dn_routed_l_claim,
        zf.dn_routed_r_claim,
        &zf.dn_routed_sumcheck,
        log_b,
        log_k_pad,
        channel,
        "dn_routed",
    )?;
    let r_dnr_tok = dn_routed_inner[..log_b].to_vec();
    let r_dnr_slot = dn_routed_inner[log_b..].to_vec();

    channel.enter_scope(TranscriptScope::SharedDown);
    let (tau_sdn_0, tau_sdn_1, s_sh_dn_hid, sdn_final, sdn_rho) =
        verify_masked_weight_matmul_prefix(
            proof.shared_down_claim,
            zf.sh_dn_cfs_sum,
            zf.sh_dn_cfs_eval,
            proof.shared_grouped_claim,
            &proof.shared_down_sumcheck,
            log_m_sh,
            channel,
            "shared_down",
        )?;
    finish_masked_weight_matmul_check(
        tau_sdn_0,
        tau_sdn_1,
        &s_sh_dn_hid,
        sdn_final,
        sdn_rho,
        zf.sh_dn_cfs_eval,
        proof.shared_grouped_claim,
        proof.shared_weight_opening,
        &[proof.shared_weight_opening],
        channel,
        "shared_down",
    )?;

    // 4. Batch-unified LogUp-GKR (`44` unmasked + `1` MaskingInstance).
    channel.enter_scope(TranscriptScope::Lookup);
    let tables = GkrLanePublicTables::cached(policy)?;
    let lookup_challenges = GkrLaneLookupChallenges::draw(channel);
    let exp_ch = channel.draw_secure_felts(3);
    let exp_alphas = [exp_ch[0], exp_ch[1]];
    let exp_z = exp_ch[2];
    if proof.gkr_proof.output_claims_by_instance.len() != plan.len()
        || proof.gkr_proof.layer_masks_by_instance.len() != plan.len()
    {
        return Err("Stage A Mode 11 GKR proof instance count mismatch".into());
    }
    for (idx, inst) in plan.iter().enumerate() {
        if proof.gkr_proof.layer_masks_by_instance[idx].len() != inst.n_variables()
            || proof.gkr_proof.output_claims_by_instance[idx].len() != 2
        {
            return Err("Stage A Mode 11 GKR proof layer mask or output claim mismatch".into());
        }
    }
    let gates = vec![Gate::LogUp; plan.len()];
    let (gkr_artifact, gkr_layer_ood_points) = partially_verify_batch_with_libra_masks(
        gates,
        &proof.gkr_proof,
        Some((zf.gkr_libra_sums.as_slice(), zf.gkr_libra_evals.as_slice())),
        channel,
    )
    .map_err(|e| format!("Stage A Mode 11 GKR batch proof rejected: {e}"))?;
    channel.mix_felts(&zf.gkr_mask_evals);
    let r_gkr_last = *gkr_artifact.ood_point.last().expect("non-empty ood_point");
    let mut gkr_frac_sum = SecureField::zero();
    for claims in &proof.gkr_proof.output_claims_by_instance {
        if claims[1].is_zero() {
            return Err("Stage A Mode 11 GKR output denominator is zero".into());
        }
        gkr_frac_sum += claims[0] / claims[1];
    }
    if !gkr_frac_sum.is_zero() {
        return Err("Stage A Mode 11 LogUp fractions do not cancel to zero".into());
    }
    let binding = derive_zk_full_gkr_binding(
        &gkr_artifact,
        &geom,
        policy,
        &tables,
        &lookup_challenges,
        exp_alphas,
        exp_z,
        channel,
        &zf.gkr_mask_evals,
    )?;

    // Continuity claims validation when `r_cont` is supplied.
    let (rc_tok_pt, rc_dim_pt, tau_cont) = if let Some(ext_rc) = r_cont {
        if ext_rc.len() != log_b + log_d + 1 {
            return Err("Stage A Mode 11 external r_cont length mismatch".into());
        }
        let tc = ext_rc[log_b + log_d];
        if tc.is_zero() {
            return Err("Stage A Mode 11 external tau_cont cannot be zero".into());
        }
        if zf.cont_claims.len() != 2 {
            return Err("Stage A Mode 11 missing 2 continuity claims for r_cont".into());
        }
        channel.mix_felts(&zf.cont_claims);
        let rct = ext_rc[..log_b].to_vec();
        let rcd = ext_rc[log_b..log_b + log_d].to_vec();
        if !inputs.is_empty() {
            let expected_in = eval_i16_rows_mle_at_pt(inputs, &rct, &rcd);
            if zf.cont_claims[0] != expected_in {
                return Err("Stage A Mode 11 public input continuity claim mismatch".into());
            }
        }
        if !statement.output_q8.is_empty() {
            let expected_out = eval_i16_rows_mle_at_pt(&statement.output_q8, &rct, &rcd);
            if zf.cont_claims[1] != expected_out {
                return Err("Stage A Mode 11 public output continuity claim mismatch".into());
            }
        }
        (rct, rcd, tc)
    } else {
        if !zf.cont_claims.is_empty() {
            return Err("Stage A Mode 11 unexpected cont_claims without r_cont".into());
        }
        (Vec::new(), Vec::new(), SecureField::zero())
    };

    // 5. Unified `r_witness_sumcheck` over `(tau_wit, r_wit)`.
    let r_prod = channel.draw_secure_felts(geom.n_wit);
    let r_oh_bool = channel.draw_secure_felts(log_k_pad + log_b + log_e);
    let r_oh_rw = channel.draw_secure_felts(log_k_pad + log_b + log_e);
    let r_act_bool = channel.draw_secure_felts(log_b + log_e);
    let r_rw_act = channel.draw_secure_felts(log_b + log_e);
    let r_max_att = channel.draw_secure_felts(log_b + log_e);
    let r_th_att = channel.draw_secure_felts(log_b + log_e);
    let r_topk_ord = channel.draw_secure_felts(log_b + log_e);
    let r_sm = channel.draw_secure_felts(log_b + log_e);

    let r_rq_eq = channel.draw_secure_felts(log_b + log_e);
    let r_exp_diff = channel.draw_secure_felts(log_b + log_e);
    let r_sm_w = channel.draw_secure_felts(log_b + log_e);
    let r_rwc = channel.draw_secure_felts(log_k_pad + log_b + log_e);
    let r_mc = channel.draw_secure_felts(log_b + log_e);
    let r_tc = channel.draw_secure_felts(log_b + log_e);
    let r_dc = channel.draw_secure_felts(log_b + log_e);
    let r_ds_t = channel.draw_secure_felts(log_b);
    let r_ohs = channel.draw_secure_felts(log_k_pad + log_b);
    let r_ad = channel.draw_secure_felts(log_b + log_e);
    let r_ohl = channel.draw_secure_felts(log_b + log_e);
    let r_pub_in = channel.draw_secure_felts(log_b + log_d);
    let r_pub_out = channel.draw_secure_felts(log_b + log_d);

    let gammas_quad = channel.draw_secure_felts(10); // prod, oh_bool, oh_rw, act_bool, rw_act, max_att, th_att, topk_ord, sm_u, sm_v
    let gammas_lin = channel.draw_secure_felts(28);
    let gamma_gkr_tbl = channel.draw_secure_felts(ZK_FULL_GKR_TABLES);
    let gamma_gkr_mask = channel.draw_secure_felts(ZK_FULL_UNMASKED_GKR_INSTANCES);

    let g_rq_eq = gammas_lin[0];
    let g_exp_diff = gammas_lin[1];
    let g_sm_w = gammas_lin[2];
    let g_rwc = gammas_lin[3];
    let g_mc = gammas_lin[4];
    let g_tc = gammas_lin[5];
    let g_dc = gammas_lin[6];
    let g_ds = gammas_lin[7];
    let g_ohs = gammas_lin[8];
    let g_ad = gammas_lin[9];
    let g_ohl = gammas_lin[10];
    let g_rt_rel1 = gammas_lin[11];
    let g_rt_x = gammas_lin[12];
    let g_reg_gu_gate = gammas_lin[13];
    let g_reg_gu_up = gammas_lin[14];
    let g_sh_gu_gate = gammas_lin[15];
    let g_sh_gu_up = gammas_lin[16];
    let g_gur_l = gammas_lin[17];
    let g_gur_r = gammas_lin[18];
    let g_sh_gu_x = gammas_lin[19];
    let g_dnr_l = gammas_lin[20];
    let g_dnr_r = gammas_lin[21];
    let g_sh_dn_x = gammas_lin[22];
    let g_r5a = gammas_lin[23];
    let g_r5b = gammas_lin[24];
    let g_gkr_nl = gammas_lin[25];
    let g_gkr_out = gammas_lin[26];
    let g_gkr_rt_sm_io = channel.draw_secure_felts(4); // rt, sma, smb, io
    let g_pub_cont = channel.draw_secure_felts(4); // pub_in, pub_out, cont_xin, cont_out
    let _ = gammas_lin[27];

    let m_gate = SecureField::from(BaseField::from_u32_unchecked(policy.gate.multiplier as u32));
    let m_up = SecureField::from(BaseField::from_u32_unchecked(policy.up.multiplier as u32));
    let m_prod =
        SecureField::from(BaseField::from_u32_unchecked(policy.product.multiplier as u32));
    let m_rt = SecureField::from(BaseField::from_u32_unchecked(policy.router.multiplier as u32));
    let m_out = SecureField::from(BaseField::from_u32_unchecked(policy.output.multiplier as u32));
    let two_s_gate = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.gate.shift));
    let half_s_gate =
        SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.gate.shift - 1)));
    let two_s_up = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.up.shift));
    let half_s_up = SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.up.shift - 1)));
    let two_s_prod = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.product.shift));
    let half_s_prod =
        SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.product.shift - 1)));
    let two_s_rt = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.router.shift));
    let half_s_rt =
        SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.router.shift - 1)));
    let two_s_out = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.output.shift));
    let half_s_out =
        SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.output.shift - 1)));
    let two_15 = SecureField::from(BaseField::from_u32_unchecked(1u32 << 15));
    let c2048 = SecureField::from(BaseField::from_u32_unchecked(2048));
    let c2 = SecureField::from(BaseField::from_u32_unchecked(2));

    let a_rt = (SecureField::one() - tau_rt_0) * (SecureField::one() - tau_rt_1);
    let b_rt = tau_rt_0 * (SecureField::one() - tau_rt_1);
    let c_rt = (SecureField::one() - tau_rt_0) * tau_rt_1;

    let a_gur = (SecureField::one() - tau_gur_0) * (SecureField::one() - tau_gur_1);
    let b_gur = tau_gur_0 * (SecureField::one() - tau_gur_1);
    let c_gur = (SecureField::one() - tau_gur_0) * tau_gur_1;
    let d_gur = tau_gur_0 * tau_gur_1;

    let a_sgu = (SecureField::one() - tau_sgu_0) * (SecureField::one() - tau_sgu_1);
    let b_sgu = tau_sgu_0 * (SecureField::one() - tau_sgu_1);
    let c_sgu = (SecureField::one() - tau_sgu_0) * tau_sgu_1;

    let a_dnr = (SecureField::one() - tau_dnr_0) * (SecureField::one() - tau_dnr_1);
    let b_dnr = tau_dnr_0 * (SecureField::one() - tau_dnr_1);
    let c_dnr = (SecureField::one() - tau_dnr_0) * tau_dnr_1;
    let d_dnr = tau_dnr_0 * tau_dnr_1;

    let a_sdn = (SecureField::one() - tau_sdn_0) * (SecureField::one() - tau_sdn_1);
    let b_sdn = tau_sdn_0 * (SecureField::one() - tau_sdn_1);
    let c_sdn = (SecureField::one() - tau_sdn_0) * tau_sdn_1;

    // Build RHS `r_witness_claim`.
    let mut r_witness_claim = SecureField::zero();
    // 9. `sum_e one_hot[s, t, e] == I[s < K]`
    let ohs_rhs: SecureField = (0..shape.top_k)
        .map(|s| eq_index(&r_ohs[..log_k_pad], s))
        .sum();
    r_witness_claim += g_ohs * ohs_rhs;
    // 12. `rt_rel1`
    r_witness_claim += g_rt_rel1 * (m_rt * proof.router.gate_claim + c2048 * two_s_rt + half_s_rt);
    // 13. `rt_x_claim` (`c_x_in_u - 2048`)
    r_witness_claim += g_rt_x * (zf.rt_x_claim + c2048 * a_rt);
    // 14. `gu_rel1` (regular sum over `0..E` + shared)
    let reg_gate_sum: SecureField = proof.regular_gate_up_claims[..e_count].iter().copied().sum();
    let reg_up_sum: SecureField = proof.regular_gate_up_claims[e_count..].iter().copied().sum();
    r_witness_claim += g_reg_gu_gate * m_gate * reg_gate_sum + g_reg_gu_up * m_up * reg_up_sum;
    r_witness_claim += g_sh_gu_gate * m_gate * proof.shared_gate_up_claims[0]
        + g_sh_gu_up * m_up * proof.shared_gate_up_claims[1];
    // 15. `gu_routed_l/r`
    r_witness_claim += g_gur_l * zf.gu_routed_l_claim;
    r_witness_claim += g_gur_r * (zf.gu_routed_r_claim - b_gur + c2048 * a_gur);
    // 16. `sh_gu_x_claim`
    r_witness_claim += g_sh_gu_x * (zf.sh_gu_x_claim + c2048 * a_sgu);
    // 17. `dn_routed_l/r`
    r_witness_claim += g_dnr_l * zf.dn_routed_l_claim;
    r_witness_claim += g_dnr_r * (zf.dn_routed_r_claim - b_dnr);
    // 18. `shared_grouped_claim`
    r_witness_claim += g_sh_dn_x * proof.shared_grouped_claim;
    // 19. `R5` (`r5a`, `r5b`)
    let acc_masked = proof.regular_down_claim + two_15 * proof.shared_down_claim;
    r_witness_claim += g_r5a * (m_out * acc_masked + half_s_out);
    match policy.rounding {
        StageARoundingMode::TiesToEven => {
            r_witness_claim += g_r5b * two_s_out;
        }
        StageARoundingMode::HalfTowardPositiveInfinity => {
            r_witness_claim += g_r5b
                * SecureField::from(BaseField::from_u32_unchecked(
                    (1u32 << policy.output.shift) - 1,
                ));
        }
    }
    // 20. LogUp-GKR claims
    r_witness_claim += g_gkr_nl * binding.gkr_nl_claim
        + g_gkr_out * binding.gkr_out_claim
        + g_gkr_rt_sm_io[0] * binding.gkr_rt_claim
        + g_gkr_rt_sm_io[1] * binding.gkr_sma_claim
        + g_gkr_rt_sm_io[2] * binding.gkr_smb_claim
        + g_gkr_rt_sm_io[3] * binding.gkr_io_claim;
    for t in 0..ZK_FULL_GKR_TABLES {
        r_witness_claim += gamma_gkr_tbl[t] * binding.tbl_mult_claims[t];
    }
    for k in 0..ZK_FULL_UNMASKED_GKR_INSTANCES {
        r_witness_claim += gamma_gkr_mask[k] * zf.gkr_mask_evals[k];
    }
    // 21. Public boundary & cross-layer continuity claims
    if !inputs.is_empty() {
        let pub_in_mle = eval_i16_rows_mle_at_pt(inputs, &r_pub_in[..log_b], &r_pub_in[log_b..]);
        r_witness_claim += g_pub_cont[0] * (pub_in_mle + c2048);
    }
    if !statement.output_q8.is_empty() {
        let pub_out_mle = eval_i16_rows_mle_at_pt(
            &statement.output_q8,
            &r_pub_out[..log_b],
            &r_pub_out[log_b..],
        );
        r_witness_claim += g_pub_cont[1] * (pub_out_mle + c2048);
    }
    if r_cont.is_some() {
        r_witness_claim += g_pub_cont[2] * (zf.cont_claims[0] + c2048);
        r_witness_claim += g_pub_cont[3] * (zf.cont_claims[1] + c2048);
    }

    channel.mix_felts(&[zf.wit_cfs_sum]);
    let rho_wit = channel.draw_secure_felt();
    let wit_init_claim = r_witness_claim + rho_wit * zf.wit_cfs_sum;
    let (r_wit_full, r_wit_final) = partially_verify_with_lo_basis(
        wit_init_claim,
        &proof.r_witness_sumcheck,
        channel,
        true,
    )?;
    channel.mix_felts(&[zf.wit_cfs_eval]);
    if r_wit_full.len() != ext_n_wit {
        return Err("Stage A Mode 11 witness sumcheck assignment length mismatch".into());
    }
    let tau_wit = r_wit_full[0];
    let r_wit = &r_wit_full[1..];
    if tau_wit.is_zero()
        || binding.tau_nl.is_zero()
        || binding.tau_out.is_zero()
        || binding.tau_rt.is_zero()
        || binding.tau_tbl.is_zero()
    {
        return Err("Stage A Mode 11 rejects degenerate tau == 0 challenge".into());
    }
    channel.mix_felts(&proof.witness_openings);
    channel.mix_felts(&zf.extra_witness_openings);

    let sel_wit = SecureField::one() - tau_wit;
    let eq_tau_nl = eq(&[binding.tau_nl], &[tau_wit]);
    let eq_tau_out = eq(&[binding.tau_out], &[tau_wit]);
    let eq_tau_rt = eq(&[binding.tau_rt], &[tau_wit]);
    let sel_tbl = sel_wit * (SecureField::one() - binding.tau_tbl);

    let c = &proof.witness_openings;
    let ex = &zf.extra_witness_openings;
    let c_xin = ex[COL_ZF_X_IN - 17];
    let c_outq = ex[COL_ZF_OUT_Q - 17];
    let c_oh = ex[COL_ZF_ONE_HOT - 17];
    let c_oh_rw = ex[COL_ZF_OH_RW - 17];
    let c_rw_rep = ex[COL_ZF_ROUTE_W - 17];
    let c_act = ex[COL_ZF_ACTIVE - 17];
    let c_rq_exp = ex[COL_ZF_RQ_EXP - 17];
    let c_maxq_sm = ex[COL_ZF_MAX_Q_SM - 17];
    let c_th_sm = ex[COL_ZF_THRESH_SM - 17];
    let c_oh_last = ex[COL_ZF_OH_LAST - 17];
    let c_sm_w = ex[COL_ZF_SOFTMAX_W - 17];
    let c_den_bc = ex[COL_ZF_DENOM_BC - 17];

    // Helper for aligned subcube prefix eq at `r_wit`.
    let subcube_pref_eq = |sub_bits: usize, sub_idx: usize| -> SecureField {
        let pref_bits = geom.n_wit - sub_bits;
        eq(&r_wit[..pref_bits], &slot_bits_felt(sub_idx, pref_bits))
    };
    let log_rt = log_b + log_e;
    let log_oh = log_k_pad + log_rt;
    let log_out = log_b + log_d;
    let r_wit_rt = &r_wit[geom.n_wit - log_rt..];
    let r_wit_oh = &r_wit[geom.n_wit - log_oh..];
    let r_wit_out = &r_wit[geom.n_wit - log_out..];
    let pref_rt_0 = subcube_pref_eq(log_rt, 0);
    let pref_oh_0 = subcube_pref_eq(log_oh, 0);
    let pref_out_0 = subcube_pref_eq(log_out, 0);
    let pref_out_1 = subcube_pref_eq(log_out, 1);

    let mut expected_wit_final = SecureField::zero();

    // A. Quadratic terms (`sel_wit * ...`):
    let eq_prod_wit = sel_wit * eq(&r_prod, r_wit);
    expected_wit_final += gammas_quad[0]
        * eq_prod_wit
        * (m_prod * (c[COL_WIT_SILU] - c2048) * (c[COL_WIT_UP_Q] - c2048)
            - (two_s_prod * (c[COL_WIT_HIDDEN_Q] - c2048)
                + (c[COL_WIT_PROD_REM] - half_s_prod)));
    expected_wit_final +=
        gammas_quad[1] * sel_wit * pref_oh_0 * eq(&r_oh_bool, r_wit_oh) * c_oh * (SecureField::one() - c_oh);
    expected_wit_final +=
        gammas_quad[2] * sel_wit * pref_oh_0 * eq(&r_oh_rw, r_wit_oh) * (c_oh_rw - c_oh * c_rw_rep);
    expected_wit_final +=
        gammas_quad[3] * sel_wit * pref_rt_0 * eq(&r_act_bool, r_wit_rt) * c_act * (SecureField::one() - c_act);
    expected_wit_final +=
        gammas_quad[4] * sel_wit * pref_rt_0 * eq(&r_rw_act, r_wit_rt) * (c_rw_rep - c_act * c_sm_w);
    expected_wit_final +=
        gammas_quad[5] * sel_wit * pref_rt_0 * eq(&r_max_att, r_wit_rt) * c_oh * (c_rq_exp - c_maxq_sm);
    let e_felt = SecureField::from(BaseField::from_u32_unchecked(e_count as u32));
    let rev_e_at_r_wit = eval_rev_e_mle(&r_wit_rt[log_b..]);
    let curr_minus_thresh = e_felt * c_rq_exp + rev_e_at_r_wit - c_th_sm;
    expected_wit_final +=
        gammas_quad[6] * sel_wit * pref_rt_0 * eq(&r_th_att, r_wit_rt) * c_oh_last * curr_minus_thresh;
    expected_wit_final += gammas_quad[7]
        * sel_wit
        * pref_rt_0
        * eq(&r_topk_ord, r_wit_rt)
        * ((c2 * c_act - SecureField::one()) * curr_minus_thresh - (SecureField::one() - c_act));
    let eq_sm = sel_wit * eq(&r_sm, r_wit_rt);
    expected_wit_final +=
        eq_sm * pref_rt_0 * c2 * (gammas_quad[9] - gammas_quad[8]) * c_sm_w * c_den_bc;

    // B1. Nonlinear chunk GKR + `gu_rel1` + `dn_routed_r` + `shared_down_x` on `0..10`:
    let mut nl_dot = SecureField::zero();
    for j in 0..10 {
        nl_dot += binding.nl_col_weights[j] * c[j];
    }
    expected_wit_final += eq_tau_nl * eq(&binding.r_nl, r_wit) * g_gkr_nl * nl_dot;

    // `gu_rel1` & `dn_routed_r` & `sh_dn_x` on `0..nl_total_rows`:
    {
        let log_nl_total = geom.nl_total_rows.ilog2() as usize;
        let nl_pref = subcube_pref_eq(log_nl_total, 0);
        let r_nl_tok = &r_wit[geom.n_wit - log_nl_total..geom.n_wit - log_nl_total + log_b];
        let r_nl_within = &r_wit[geom.n_wit - log_nl_total + log_b..];
        let within_eq = eq_evals(r_nl_within);
        let reg_dn_hid_eq = eq_evals(&r_dn_hid);
        let sh_dn_hid_eq = eq_evals(&s_sh_dn_hid);

        let mut slot_gu_sum = SecureField::zero();
        let mut slot_dn_sum = SecureField::zero();
        for s in 0..shape.top_k {
            let base = s * geom.m_pad;
            slot_gu_sum += eq_index(&r_gu_slot, s)
                * dot(
                    &within_eq[base..base + shape.expert_width],
                    &reg_proj_eq[..shape.expert_width],
                );
            slot_dn_sum += eq_index(&r_dnr_slot, s)
                * dot(
                    &within_eq[base..base + shape.expert_width],
                    &reg_dn_hid_eq[..shape.expert_width],
                );
        }
        let sh_base = shape.top_k * geom.m_pad;
        let sh_width = shape.shared_width();
        let sh_gu_within = dot(
            &within_eq[sh_base..sh_base + sh_width],
            &sh_proj_eq[..sh_width],
        );
        let sh_dn_within = dot(
            &within_eq[sh_base..sh_base + sh_width],
            &sh_dn_hid_eq[..sh_width],
        );

        let eq_gu_tok = eq(&r_gu_tok, r_nl_tok);
        let gate_requant_y = two_s_gate * (c[COL_WIT_GATE_Q] - c2048) + (c[COL_WIT_GATE_REM] - half_s_gate);
        let up_requant_y = two_s_up * (c[COL_WIT_UP_Q] - c2048) + (c[COL_WIT_UP_REM] - half_s_up);

        let w_reg_gu = sel_wit * nl_pref * eq_gu_tok * slot_gu_sum;
        let w_sh_gu = sel_wit * nl_pref * eq_gu_tok * sh_gu_within;
        expected_wit_final += w_reg_gu * (g_reg_gu_gate * gate_requant_y + g_reg_gu_up * up_requant_y);
        expected_wit_final += w_sh_gu * (g_sh_gu_gate * gate_requant_y + g_sh_gu_up * up_requant_y);

        let w_dnr_r = sel_wit * nl_pref * eq(&r_dnr_tok, r_nl_tok) * a_dnr * slot_dn_sum;
        let w_sdn_x = sel_wit * nl_pref * eq(&r_dn_tok, r_nl_tok) * a_sdn * sh_dn_within;
        expected_wit_final += (g_dnr_r * w_dnr_r + g_sh_dn_x * w_sdn_x) * (c[COL_WIT_HIDDEN_Q] - c2048);
    }

    // B2. Output limbs `11..17` (`GKR` + `R5`):
    {
        let out_widths = out_limb_widths(policy.output.shift);
        let limb_shifts = [0u32, out_widths[0] as u32, (out_widths[0] + out_widths[1]) as u32];
        let gkr_out_eq = pref_out_0 * eq(r_wit_out, &binding.r_gkr_out);
        let r5_eq = pref_out_0 * eq(&r_wit_out[..log_b], &r_dn_tok) * eq(&r_wit_out[log_b..], &output_point);
        for k in 0..6 {
            let shift_scale =
                SecureField::from(BaseField::from_u32_unchecked(1u32 << limb_shifts[k % 3]));
            let r5_coeff = if k < 3 {
                (g_r5a + g_r5b) * shift_scale
            } else {
                g_r5b * shift_scale
            };
            let wk = eq_tau_out * g_gkr_out * binding.out_limb_weights[k] * gkr_out_eq
                + sel_wit * r5_coeff * r5_eq;
            expected_wit_final += wk * c[COL_WIT_OUT_LIMB_0 + k];
        }
    }

    // B3. Extra columns `17..29` linear weights:
    let mut w_xin = SecureField::zero();
    let mut w_outq = SecureField::zero();
    let mut w_oh = SecureField::zero();
    let mut w_oh_rw = SecureField::zero();
    let mut w_rw_rep = SecureField::zero();
    let mut w_act = SecureField::zero();
    let mut w_rq_exp = SecureField::zero();
    let mut w_maxq_sm = SecureField::zero();
    let mut w_th_sm = SecureField::zero();
    let mut w_oh_last = SecureField::zero();
    let mut w_sm_w = SecureField::zero();
    let mut w_den_bc = SecureField::zero();

    // IoRangeQueries GKR on `COL_ZF_X_IN` & `COL_ZF_OUT_Q`:
    let gkr_io_eq = eq_tau_out * g_gkr_rt_sm_io[3] * eq(r_wit_out, &binding.r_gkr_out);
    w_xin += gkr_io_eq * (pref_out_0 * binding.io_xin_weights[0] + pref_out_1 * binding.io_xin_weights[1]);
    w_outq += gkr_io_eq * (pref_out_0 * binding.io_outq_weights[0] + pref_out_1 * binding.io_outq_weights[1]);

    // SoftmaxQueriesA & SoftmaxQueriesB GKR on `COL_ZF_RQ_EXP`, `COL_ZF_MAX_Q_SM`, `COL_ZF_THRESH_SM`:
    let gkr_rt_sub_eq = eq(r_wit_rt, &binding.r_gkr_rt);
    let gkr_sma_factor = eq_tau_rt * g_gkr_rt_sm_io[1] * gkr_rt_sub_eq;
    for u in 1..5 {
        w_rq_exp += gkr_sma_factor * subcube_pref_eq(log_rt, u) * binding.sma_rq_exp_weights[u];
    }
    w_maxq_sm += gkr_sma_factor * subcube_pref_eq(log_rt, 1) * binding.sma_maxq_u2_weight;
    let gkr_smb_factor = eq_tau_rt * g_gkr_rt_sm_io[2] * gkr_rt_sub_eq;
    for u in 2..6 {
        w_maxq_sm += gkr_smb_factor * subcube_pref_eq(log_rt, u) * binding.smb_maxq_weights[u];
    }
    for u in 1..3 {
        w_th_sm += gkr_smb_factor * subcube_pref_eq(log_rt, u) * binding.smb_th_weights[u];
    }

    // Softmax `u_sm` & `v_sm` linear parts at `r_sm`:
    let c1024 = SecureField::from(BaseField::from_u32_unchecked(1024));
    let c524288 = SecureField::from(BaseField::from_u32_unchecked(524_288));
    let c65536 = SecureField::from(BaseField::from_u32_unchecked(65_536));
    let g_sm_u = gammas_quad[8];
    let g_sm_v = gammas_quad[9];
    w_den_bc += eq_sm * pref_rt_0 * (g_sm_u + g_sm_v);
    w_rq_exp += eq_sm * subcube_pref_eq(log_rt, 2) * c65536 * (g_sm_u - g_sm_v);
    w_th_sm -= eq_sm * subcube_pref_eq(log_rt, 2) * (g_sm_u + g_sm_v);
    w_rq_exp -= eq_sm * subcube_pref_eq(log_rt, 3) * g_sm_u;
    w_rq_exp -= eq_sm * subcube_pref_eq(log_rt, 4) * g_sm_u * c1024;
    w_maxq_sm -= eq_sm * subcube_pref_eq(log_rt, 1) * g_sm_u * c524288;
    w_maxq_sm -= eq_sm * subcube_pref_eq(log_rt, 2) * g_sm_v;
    w_maxq_sm -= eq_sm * subcube_pref_eq(log_rt, 3) * g_sm_v * c1024;
    w_maxq_sm -= eq_sm * subcube_pref_eq(log_rt, 4) * g_sm_v * c524288;

    // 1. `rq_u` consistency: `COL_ZF_RQ_EXP[0] - COL_WIT_AUX[0] == 0`
    w_rq_exp += sel_wit * g_rq_eq * pref_rt_0 * eq(&r_rq_eq, r_wit_rt);
    // 2. `exp_diff`: `exp_diff - max_q + rq_u == 0`
    {
        let e_ed = sel_wit * g_exp_diff * eq(&r_exp_diff, r_wit_rt);
        w_rq_exp += e_ed * (subcube_pref_eq(log_rt, 1) + pref_rt_0);
        w_maxq_sm -= e_ed * pref_rt_0;
    }
    // 3. `softmax_w` limb decomposition: `sm_w - (2*rw_half_lo + 256*rw_half_hi + rw_par) == 0`
    {
        let e_sw = sel_wit * g_sm_w * eq(&r_sm_w, r_wit_rt);
        let c256 = SecureField::from(BaseField::from_u32_unchecked(256));
        w_sm_w += e_sw * pref_rt_0;
        w_maxq_sm -= e_sw * subcube_pref_eq(log_rt, 5) * c2;
        w_th_sm -= e_sw * (subcube_pref_eq(log_rt, 1) * c256 + subcube_pref_eq(log_rt, 2));
    }
    // 4. `route_w_rep` constant in `s`:
    {
        let r_s = &r_rwc[..log_k_pad];
        let r_te = &r_rwc[log_k_pad..];
        let r_wit_s = &r_wit_oh[..log_k_pad];
        w_rw_rep += sel_wit
            * g_rwc
            * pref_oh_0
            * (eq(r_s, r_wit_s) - eq_all_zeros(r_wit_s))
            * eq(r_te, r_wit_rt);
    }
    // 5, 6, 7. `max_q_u_bc`, `thresh_key_bc`, `denom_bc` constant in `e`:
    {
        let r_wit_t = &r_wit_rt[..log_b];
        let r_wit_e = &r_wit_rt[log_b..];
        let zero_e = eq_all_zeros(r_wit_e);
        w_maxq_sm += sel_wit
            * g_mc
            * pref_rt_0
            * eq(&r_mc[..log_b], r_wit_t)
            * (eq(&r_mc[log_b..], r_wit_e) - zero_e);
        w_th_sm += sel_wit
            * g_tc
            * pref_rt_0
            * eq(&r_tc[..log_b], r_wit_t)
            * (eq(&r_tc[log_b..], r_wit_e) - zero_e);
        w_den_bc += sel_wit
            * g_dc
            * pref_rt_0
            * eq(&r_dc[..log_b], r_wit_t)
            * (eq(&r_dc[log_b..], r_wit_e) - zero_e);
        // 8. `denom_bc[t, 0] - sum_e exp_val[t, e] == 0`
        let e_ds = sel_wit * g_ds * eq(&r_ds_t, r_wit_t);
        w_den_bc += e_ds * pref_rt_0 * zero_e;
        w_rq_exp -= e_ds * subcube_pref_eq(log_rt, 2);
    }
    // 9. `sum_e one_hot[s, t, e] == I[s < K]`
    w_oh += sel_wit
        * g_ohs
        * pref_oh_0
        * eq(&r_ohs[..log_k_pad], &r_wit_oh[..log_k_pad])
        * eq(&r_ohs[log_k_pad..], &r_wit_rt[..log_b]);
    // 10. `active[t, e] - sum_s one_hot[s, t, e] == 0`
    {
        let e_ad = sel_wit * g_ad * eq(&r_ad, r_wit_rt);
        w_act += e_ad * pref_rt_0;
        w_oh -= e_ad * pref_oh_0;
    }
    // 11. `one_hot_last[t, e] - one_hot[K - 1, t, e] == 0`
    {
        let e_ohl = sel_wit * g_ohl * eq(&r_ohl, r_wit_rt);
        w_oh_last += e_ohl * pref_rt_0;
        w_oh -= e_ohl * pref_oh_0 * eq_index(&r_wit_oh[..log_k_pad], shape.top_k - 1);
    }
    // 13, 15, 16. `x_in_u` linear claims (`rt_x`, `gur_r`, `sh_gu_x`) & `gur_l` & `dnr_l`:
    {
        let r_wit_out_t = &r_wit_out[..log_b];
        let r_wit_out_d = &r_wit_out[log_b..];
        w_xin += sel_wit
            * pref_out_0
            * (g_rt_x * a_rt * eq(&r_rt_tok, r_wit_out_t) * eq(&s_rt_dim, r_wit_out_d)
                + g_gur_r * a_gur * eq(&r_gur_tok, r_wit_out_t) * eq(&r_gu_dim, r_wit_out_d)
                + g_sh_gu_x * a_sgu * eq(&r_gu_tok, r_wit_out_t) * eq(&s_sh_gu_dim, r_wit_out_d));
        w_oh += sel_wit
            * g_gur_l
            * a_gur
            * pref_oh_0
            * eq(&r_gu_slot, &r_wit_oh[..log_k_pad])
            * eq(&r_gur_tok, &r_wit_rt[..log_b])
            * eq(&r_gu_exp, &r_wit_rt[log_b..]);
        w_oh_rw += sel_wit
            * g_dnr_l
            * a_dnr
            * pref_oh_0
            * eq(&r_dnr_slot, &r_wit_oh[..log_k_pad])
            * eq(&r_dnr_tok, &r_wit_rt[..log_b])
            * eq(&r_dn_exp, &r_wit_rt[log_b..]);
    }
    // 19. `R5` (`out_q_u` and `out_q_par`):
    {
        let r5_eq_base =
            sel_wit * eq(&r_wit_out[..log_b], &r_dn_tok) * eq(&r_wit_out[log_b..], &output_point);
        // Note `2^{s_out} * (c_out_q_u - 2048)`: `-2048 * 2^{s_out}` cancels by moving `+2048 * 2^{s_out}` to RHS!
        w_outq += r5_eq_base * pref_out_0 * g_r5a * two_s_out;
        if policy.rounding == StageARoundingMode::TiesToEven {
            w_outq += r5_eq_base * pref_out_1 * (g_r5a + g_r5b * c2);
        }
    }
    // Adjust `r_witness_claim` note: `-2048 * two_s_out` on LHS means we subtract `-g_r5a * c2048 * two_s_out` from LHS directly!
    expected_wit_final -= sel_wit
        * pref_out_0
        * eq(&r_wit_out[..log_b], &r_dn_tok)
        * eq(&r_wit_out[log_b..], &output_point)
        * g_r5a
        * c2048
        * two_s_out;

    // 21. Public boundary & continuity linear weights on `x_in_u` and `out_q_u`:
    if !inputs.is_empty() {
        w_xin += sel_wit * g_pub_cont[0] * pref_out_0 * eq(&r_pub_in, r_wit_out);
    }
    if !statement.output_q8.is_empty() {
        w_outq += sel_wit * g_pub_cont[1] * pref_out_0 * eq(&r_pub_out, r_wit_out);
    }
    if r_cont.is_some() {
        let eq_rc_t = eq(&rc_tok_pt, &r_wit_out[..log_b]);
        let eq_rc_d = eq(&rc_dim_pt, &r_wit_out[log_b..]);
        let rc_eq = sel_wit * pref_out_0 * eq_rc_t * eq_rc_d;
        w_xin += g_pub_cont[2] * rc_eq;
        w_outq += g_pub_cont[3] * rc_eq;
    }

    expected_wit_final += w_xin * c_xin
        + w_outq * c_outq
        + w_oh * c_oh
        + w_oh_rw * c_oh_rw
        + w_rw_rep * c_rw_rep
        + w_act * c_act
        + w_rq_exp * c_rq_exp
        + w_maxq_sm * c_maxq_sm
        + w_th_sm * c_th_sm
        + w_oh_last * c_oh_last
        + w_sm_w * c_sm_w
        + w_den_bc * c_den_bc;

    // B4. `COL_WIT_AUX` (`c[10]`) linear weight `w_c10`:
    let mut w_c10 = SecureField::zero();
    // RoutingQueries GKR (`rt_sub_weights[0,1,2,5,6]`):
    for u in 0..7 {
        if !binding.rt_sub_weights[u].is_zero() {
            w_c10 += eq_tau_rt
                * g_gkr_rt_sm_io[0]
                * binding.rt_sub_weights[u]
                * (subcube_pref_eq(log_rt, u) * gkr_rt_sub_eq);
        }
    }
    // `rq_u` consistency (`-COL_WIT_AUX[0]`):
    w_c10 -= sel_wit * g_rq_eq * pref_rt_0 * eq(&r_rq_eq, r_wit_rt);
    // `topk_ord` linear remainder limbs `-diff_lo - 2048 * diff_hi`:
    {
        let e_tk = sel_wit * gammas_quad[7] * eq(&r_topk_ord, r_wit_rt);
        w_c10 -= e_tk * (subcube_pref_eq(log_rt, 5) + c2048 * subcube_pref_eq(log_rt, 6));
    }
    // `rt_rel1` (`2^{s_rt} * rq_u + rq_rem` at `r_rt_init`):
    {
        let e_rt = sel_wit * g_rt_rel1 * eq(&r_rt_init, r_wit_rt);
        w_c10 += e_rt * (two_s_rt * pref_rt_0 + subcube_pref_eq(log_rt, 1));
    }
    // Table multiplicities `0..5`:
    {
        let tbl_bits = LOG_TABLE_ROWS as usize;
        let tbl_pref_bits = geom.n_wit - tbl_bits;
        let tbl_sub_eq = eq(&r_wit[tbl_pref_bits..], &binding.r_gkr_tbl);
        for v in 0..ZK_FULL_GKR_TABLES {
            let pref_idx = geom.tbl_subcube_offset(v) >> tbl_bits;
            let p_eq = eq(&r_wit[..tbl_pref_bits], &slot_bits_felt(pref_idx, tbl_pref_bits));
            w_c10 += sel_tbl * gamma_gkr_tbl[v] * (p_eq * tbl_sub_eq);
        }
    }
    // `tau_wit = 1` scalar `QM31` masks in `COL_WIT_AUX[hi_base + 7 * sub_len + 0..16384]`:
    {
        let mask_pref_bits = geom.n_wit - ZK_FULL_AUX_MASK_SUB_BITS;
        let mask_pref_idx = geom.tbl_subcube_offset(0) >> ZK_FULL_AUX_MASK_SUB_BITS;
        let mask_pref_eq = tau_wit
            * eq(
                &r_wit[..mask_pref_bits],
                &slot_bits_felt(mask_pref_idx, mask_pref_bits),
            );
        let mask_sub_eq = eq_evals(&r_wit[mask_pref_bits..]);
        let beta = [
            SecureField::from_u32_unchecked(1, 0, 0, 0),
            SecureField::from_u32_unchecked(0, 1, 0, 0),
            SecureField::from_u32_unchecked(0, 0, 1, 0),
            SecureField::from_u32_unchecked(0, 0, 0, 1),
        ];
        let mut mask_acc = SecureField::zero();
        let mut add_qmask = |q_idx: usize, coeff: SecureField| {
            if !coeff.is_zero() {
                let base = 4 * q_idx;
                for c_idx in 0..4 {
                    mask_acc += coeff * mask_sub_eq[base + c_idx] * beta[c_idx];
                }
            }
        };

        let one_minus_r_last = SecureField::one() - r_gkr_last;
        for (k, &gm) in gamma_gkr_mask.iter().enumerate() {
            add_qmask(2 * k, gm * one_minus_r_last);
            add_qmask(2 * k + 1, gm * r_gkr_last);
        }
        if r_cont.is_some() {
            add_qmask(Q_MU_XIN, g_pub_cont[2] * tau_cont);
            add_qmask(Q_MU_OUT, g_pub_cont[3] * tau_cont);
        }
        add_qmask(Q_MU_RT_CLAIM, g_rt_rel1 * m_rt + g_rt_x * b_rt);
        add_qmask(Q_MU_RT_X, g_rt_x * c_rt);
        add_qmask(
            Q_MU_SH_GU_GATE,
            g_sh_gu_gate * m_gate + g_sh_gu_x * b_sgu * sh_gu_batch_weights[0],
        );
        add_qmask(
            Q_MU_SH_GU_UP,
            g_sh_gu_up * m_up + g_sh_gu_x * b_sgu * sh_gu_batch_weights[1],
        );
        add_qmask(Q_MU_SH_GU_X, g_sh_gu_x * c_sgu);
        add_qmask(
            Q_MU_SH_DN_CLAIM,
            g_sh_dn_x * b_sdn + g_r5a * m_out * two_15,
        );
        add_qmask(Q_MU_SH_DN_X, g_sh_dn_x * c_sdn);
        add_qmask(
            Q_MU_REG_DN_CLAIM,
            g_dnr_l * b_dnr * b_rdn + g_r5a * m_out,
        );
        add_qmask(Q_MU_REG_DN_X, g_dnr_l * b_dnr * c_rdn);
        add_qmask(Q_MU_DN_ROUTED_L, g_dnr_l * c_dnr);
        add_qmask(Q_MU_DN_ROUTED_R, g_dnr_r * d_dnr);
        add_qmask(Q_MU_REG_GU_X, g_gur_l * b_gur * c_gu);
        add_qmask(Q_MU_GU_ROUTED_L, g_gur_l * c_gur);
        add_qmask(Q_MU_GU_ROUTED_R, g_gur_r * d_gur);
        for e in 0..e_count {
            add_qmask(
                Q_MU_GU_GATE_BASE + e,
                g_reg_gu_gate * m_gate + g_gur_l * b_gur * b_gu * reg_gu_batch_weights[e],
            );
            add_qmask(
                Q_MU_GU_GATE_BASE + e_count + e,
                g_reg_gu_up * m_up + g_gur_l * b_gur * b_gu * reg_gu_batch_weights[e_count + e],
            );
        }
        w_c10 += mask_pref_eq * mask_acc;
    }
    expected_wit_final += w_c10 * c[COL_WIT_AUX];
    expected_wit_final += rho_wit * zf.wit_cfs_eval;

    if r_wit_final != expected_wit_final {
        return Err("Stage A Mode 11 witness sumcheck final evaluation mismatch".into());
    }

    // 5b. CFS binding sumcheck over `ZK_FULL_CFS_LOG_ROWS` (`16384` QM31s).
    let r_cfs_pad = channel.draw_secure_felts(ZK_FULL_CFS_LOG_ROWS - 1);
    channel.mix_felts(&[zf.cfs_pad_claim]);
    let gamma_sc_cfs = channel.draw_secure_felts(16); // 8 `(sum, eval)` pairs for the 8 non-GKR sumchecks
    let gamma_libra_sums = channel.draw_secure_felts(n_gkr_layers);
    let gamma_libra_evals = channel.draw_secure_felts(n_gkr_layers);
    let gamma_cfs_pad = channel.draw_secure_felt();

    let mut rt_pt = vec![tau_rt_0, tau_rt_1];
    rt_pt.extend_from_slice(&s_rt_dim);
    let mut reg_gu_pt = vec![tau_gu_0, tau_gu_1];
    reg_gu_pt.extend_from_slice(&reg_gu_inner);
    let mut gur_pt = vec![tau_gur_0, tau_gur_1];
    gur_pt.extend_from_slice(&r_gur_tok);
    let mut sgu_pt = vec![tau_sgu_0, tau_sgu_1];
    sgu_pt.extend_from_slice(&s_sh_gu_dim);
    let mut rdn_pt = vec![tau_rdn_0, tau_rdn_1];
    rdn_pt.extend_from_slice(&reg_dn_inner);
    let mut dnr_pt = vec![tau_dnr_0, tau_dnr_1];
    dnr_pt.extend_from_slice(&dn_routed_inner);
    let mut sdn_pt = vec![tau_sdn_0, tau_sdn_1];
    sdn_pt.extend_from_slice(&s_sh_dn_hid);

    let mut w_cfs = vec![SecureField::zero(); ZK_FULL_CFS_ROWS];
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.rt_cfs_off,
        &rt_pt,
        gamma_sc_cfs[0],
        gamma_sc_cfs[1],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.reg_gu_cfs_off,
        &reg_gu_pt,
        gamma_sc_cfs[2],
        gamma_sc_cfs[3],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.gu_routed_cfs_off,
        &gur_pt,
        gamma_sc_cfs[4],
        gamma_sc_cfs[5],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.sh_gu_cfs_off,
        &sgu_pt,
        gamma_sc_cfs[6],
        gamma_sc_cfs[7],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.reg_dn_cfs_off,
        &rdn_pt,
        gamma_sc_cfs[8],
        gamma_sc_cfs[9],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.dn_routed_cfs_off,
        &dnr_pt,
        gamma_sc_cfs[10],
        gamma_sc_cfs[11],
    );
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        cfs_layout.sh_dn_cfs_off,
        &sdn_pt,
        gamma_sc_cfs[12],
        gamma_sc_cfs[13],
    );
    let mut gkr_off = cfs_layout.gkr_libra_off;
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
        cfs_layout.wit_cfs_off,
        &r_wit_full,
        gamma_sc_cfs[14],
        gamma_sc_cfs[15],
    );
    let pad_eq = eq_evals(&r_cfs_pad);
    for (v, &e) in pad_eq.iter().enumerate() {
        w_cfs[cfs_layout.pad_off + v] += gamma_cfs_pad * e;
    }

    let mut cfs_binding_claim = gamma_sc_cfs[0] * zf.rt_cfs_sum
        + gamma_sc_cfs[1] * zf.rt_cfs_eval
        + gamma_sc_cfs[2] * zf.reg_gu_cfs_sum
        + gamma_sc_cfs[3] * zf.reg_gu_cfs_eval
        + gamma_sc_cfs[4] * zf.gu_routed_cfs_sum
        + gamma_sc_cfs[5] * zf.gu_routed_cfs_eval
        + gamma_sc_cfs[6] * zf.sh_gu_cfs_sum
        + gamma_sc_cfs[7] * zf.sh_gu_cfs_eval
        + gamma_sc_cfs[8] * zf.reg_dn_cfs_sum
        + gamma_sc_cfs[9] * zf.reg_dn_cfs_eval
        + gamma_sc_cfs[10] * zf.dn_routed_cfs_sum
        + gamma_sc_cfs[11] * zf.dn_routed_cfs_eval
        + gamma_sc_cfs[12] * zf.sh_dn_cfs_sum
        + gamma_sc_cfs[13] * zf.sh_dn_cfs_eval
        + gamma_sc_cfs[14] * zf.wit_cfs_sum
        + gamma_sc_cfs[15] * zf.wit_cfs_eval
        + gamma_cfs_pad * zf.cfs_pad_claim;
    for layer in 0..n_gkr_layers {
        cfs_binding_claim += gamma_libra_sums[layer] * zf.gkr_libra_sums[layer]
            + gamma_libra_evals[layer] * zf.gkr_libra_evals[layer];
    }
    let (r_cfs_sub, cfs_final) =
        partially_verify(cfs_binding_claim, &zf.cfs_binding_sumcheck, channel)
            .map_err(|e| format!("Stage A Mode 11 CFS binding sumcheck rejected: {e}"))?;
    if r_cfs_sub.len() != ZK_FULL_CFS_LOG_ROWS {
        return Err("Stage A Mode 11 CFS binding assignment length mismatch".into());
    }
    channel.mix_felts(&[zf.blind_col_opening]);
    if cfs_final != mle_at(&w_cfs, &r_cfs_sub) * zf.blind_col_opening {
        return Err("Stage A Mode 11 CFS binding final evaluation mismatch".into());
    }

    // 6. CPRS openings (`30` witness cols + `router` + `E` experts + `shared`).
    let binding_seed = draw_cprs_binding_seed(channel);
    let d_wit = cprs_witness_outer_digest_for_roles_and_extra(
        &binding_seed,
        statement,
        &proof.witness_commitment,
        &r_wit_full,
        &WITNESS_SLOT_ROLES_ZK_FULL,
        Some(&r_cfs_sub),
    );
    let omega_scale = zk_lo_basis_omega_scale();
    let r_wit_hi_cprs = stwo_point_to_cprs(&r_wit_full[..m_ext]);
    let mut r_cfs_hi = r_cfs_anchor;
    r_cfs_hi.extend_from_slice(&r_cfs_sub);
    let r_cfs_hi_cprs = stwo_point_to_cprs(&r_cfs_hi);
    let mut wit_points = vec![r_wit_hi_cprs; WITNESS_GROUP_COLS + ZK_FULL_EXTRA_WITNESS_COLS];
    wit_points.push(r_cfs_hi_cprs);
    let mut wit_claims: Vec<_> = proof
        .witness_openings
        .iter()
        .chain(zf.extra_witness_openings.iter())
        .map(|&v| to_cprs_qm31(omega_scale * v))
        .collect();
    wit_claims.push(to_cprs_qm31(omega_scale * zf.blind_col_opening));

    let r_router_cprs = stwo_point_to_cprs(&router_weight_point);
    let mut r_reg_gu_stwo = r_gu_dim;
    r_reg_gu_stwo.extend_from_slice(&regular_projection_point);
    let mut r_reg_down_stwo = r_dn_hid;
    r_reg_down_stwo.extend_from_slice(&output_point);
    let r_reg_gu_cprs = stwo_point_to_cprs(&r_reg_gu_stwo);
    let r_reg_down_cprs = stwo_point_to_cprs(&r_reg_down_stwo);
    let mut r_sh_gu_stwo = s_sh_gu_dim;
    r_sh_gu_stwo.extend_from_slice(&shared_projection_point);
    let mut r_sh_down_stwo = s_sh_dn_hid;
    r_sh_down_stwo.extend_from_slice(&output_point);
    let r_sh_gu_cprs = stwo_point_to_cprs(&r_sh_gu_stwo);
    let r_sh_down_cprs = stwo_point_to_cprs(&r_sh_down_stwo);

    if defer_cprs {
        let mut reg_commitments = Vec::with_capacity(e_count);
        let mut reg_points = Vec::with_capacity(3 * e_count);
        let mut reg_claims = Vec::with_capacity(3 * e_count);
        for e in 0..e_count {
            reg_commitments.push(context.active_expert_openings[e].group_commitment);
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_down_cprs.clone());
            reg_claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[e]));
            reg_claims.push(to_cprs_qm31(proof.regular_up_weight_openings[e]));
            reg_claims.push(to_cprs_qm31(proof.regular_down_weight_openings[e]));
        }
        let sh_points = vec![r_sh_gu_cprs.clone(), r_sh_gu_cprs, r_sh_down_cprs];
        let sh_claims = [
            to_cprs_qm31(proof.shared_gate_weight_opening),
            to_cprs_qm31(proof.shared_up_weight_opening),
            to_cprs_qm31(proof.shared_weight_opening),
        ];
        let cprs_digest = cprs_gkr_lane_proofs_transcript_digest(
            &proof.cprs_router_proof,
            &proof.cprs_regular_proof,
            &proof.cprs_shared_proof,
            &proof.cprs_witness_proof,
        );
        mix_digest(channel, &cprs_digest);
        return Ok((
            StageARegisteredVerifierRun {
                verifier_context_bytes: context.footprint_bytes(),
                transcript_census: channel.census(),
                transcript_trace: channel.trace(),
            },
            Some(StageAGkrLaneDeferredCprs {
                binding_seed,
                active_experts: all_experts,
                witness_commitment: proof.witness_commitment,
                witness_params,
                wit_points,
                wit_claims,
                d_wit,
                router_commitment: context.router_commitment,
                router_params: cprs_params.router,
                r_router_cprs,
                wg_opening: to_cprs_qm31(proof.router.wg_opening),
                r_reg_gu_cprs,
                r_reg_down_cprs,
                reg_commitments,
                reg_params: cprs_params.regular,
                reg_points,
                reg_claims,
                shared_commitment: context.shared_commitment,
                shared_params: cprs_params.shared,
                sh_points,
                sh_claims,
            }),
        ));
    }

    if let Some(w_b) = stage_b.and_then(|b| b.witness_opening.as_ref()) {
        let mut h = Blake2s256::new();
        h.update(b"stwo-ml:stage-ab:cprs-witness-outer:v1");
        h.update(d_wit);
        w_b.update_outer_hasher(&mut h);
        let d_ab_wit: [u8; 32] = h.finalize().into();
        let (mut wit_all_points, mut wit_all_claims) = w_b.cprs_points_and_claims();
        wit_all_points.extend(wit_points);
        wit_all_claims.extend(wit_claims);
        verify_heterogeneous_strict_context(
            &[w_b.witness_commitment, proof.witness_commitment],
            &[w_b.witness_params.clone(), witness_params],
            &wit_all_points,
            &wit_all_claims,
            &d_ab_wit,
            &proof.cprs_witness_proof,
        )
        .map_err(|e| format!("Stage A+B Mode 11 witness CPRS verification failed: {e}"))?;
    } else {
        verify_strict_context(
            &[proof.witness_commitment],
            &wit_points,
            &wit_claims,
            &d_wit,
            &witness_params,
            &proof.cprs_witness_proof,
        )
        .map_err(|e| format!("Stage A Mode 11 witness CPRS verification failed: {e}"))?;
    }

    let is_unified = proof.cprs_regular_proof.is_empty() && proof.cprs_shared_proof.is_empty();
    if stage_b.is_some() && !is_unified {
        return Err(
            "Stage A Mode 11 + Stage B unified CPRS requires empty cprs_regular_proof and cprs_shared_proof"
                .into(),
        );
    }
    if is_unified {
        let d_layer = cprs_layer_outer_digest(&binding_seed, statement, &all_experts, stage_b);
        let extra_b_groups = stage_b.map_or(0, |b| 1 + b.chunk_openings.len());
        let mut commitments = Vec::with_capacity(extra_b_groups + 2 + e_count);
        let mut group_params_list = Vec::with_capacity(extra_b_groups + 2 + e_count);
        let mut points = Vec::new();
        let mut claims = Vec::new();
        if let Some(b_deferred) = stage_b {
            commitments.push(b_deferred.attention_commitment);
            group_params_list.push(b_deferred.leaf_params.clone());
            let qkv_cprs = stwo_point_to_cprs(&b_deferred.qkv_point);
            let o_cprs = stwo_point_to_cprs(&b_deferred.o_point);
            points.push(qkv_cprs.clone());
            points.push(qkv_cprs.clone());
            points.push(qkv_cprs);
            points.push(o_cprs);
            for &cl in &b_deferred.leaf_claims {
                claims.push(to_cprs_qm31(cl));
            }
            for op in &b_deferred.chunk_openings {
                let chunk_commit = b_deferred
                    .chunk_commitments
                    .get(op.chunk)
                    .copied()
                    .ok_or_else(|| "Stage B cache opening unknown chunk commitment".to_string())?;
                commitments.push(chunk_commit);
                group_params_list.push(b_deferred.chunk_params.clone());
                points.push(stwo_point_to_cprs(&op.k_point));
                points.push(stwo_point_to_cprs(&op.v_point));
                claims.push(to_cprs_qm31(op.k_value));
                claims.push(to_cprs_qm31(op.v_value));
            }
        }
        commitments.push(context.router_commitment);
        group_params_list.push(cprs_params.router.clone());
        points.push(r_router_cprs);
        claims.push(to_cprs_qm31(proof.router.wg_opening));
        for e in 0..e_count {
            commitments.push(context.active_expert_openings[e].group_commitment);
            group_params_list.push(cprs_params.regular.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_down_cprs.clone());
            claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[e]));
            claims.push(to_cprs_qm31(proof.regular_up_weight_openings[e]));
            claims.push(to_cprs_qm31(proof.regular_down_weight_openings[e]));
        }
        commitments.push(context.shared_commitment);
        group_params_list.push(cprs_params.shared.clone());
        points.push(r_sh_gu_cprs.clone());
        points.push(r_sh_gu_cprs);
        points.push(r_sh_down_cprs);
        claims.push(to_cprs_qm31(proof.shared_gate_weight_opening));
        claims.push(to_cprs_qm31(proof.shared_up_weight_opening));
        claims.push(to_cprs_qm31(proof.shared_weight_opening));

        verify_heterogeneous_strict_context(
            &commitments,
            &group_params_list,
            &points,
            &claims,
            &d_layer,
            &proof.cprs_router_proof,
        )
        .map_err(|e| format!("Stage A Mode 11 unified CPRS opening rejected: {e}"))?;
    } else {
        if proof.cprs_regular_proof.is_empty() || proof.cprs_shared_proof.is_empty() {
            return Err("Stage A Mode 11 separate CPRS requires regular and shared proofs".into());
        }
        let d_router = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            shape.batch,
            &all_experts,
            CPRS_CLASS_TAG_ROUTER,
            &ROUTER_SLOT_ROLES,
        );
        let d_reg = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            shape.batch,
            &all_experts,
            CPRS_CLASS_TAG_REGULAR,
            &REGULAR_SLOT_ROLES,
        );
        let d_sh = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            shape.batch,
            &all_experts,
            CPRS_CLASS_TAG_SHARED,
            &SHARED_SLOT_ROLES,
        );
        verify_strict_context(
            &[context.router_commitment],
            std::slice::from_ref(&r_router_cprs),
            &[to_cprs_qm31(proof.router.wg_opening)],
            &d_router,
            &cprs_params.router,
            &proof.cprs_router_proof,
        )
        .map_err(|e| format!("Stage A Mode 11 router CPRS failed: {e}"))?;
        let reg_commitments: Vec<[u8; 32]> = context
            .active_expert_openings
            .iter()
            .map(|op| op.group_commitment)
            .collect();
        let mut reg_points = Vec::with_capacity(3 * e_count);
        let mut reg_claims = Vec::with_capacity(3 * e_count);
        for e in 0..e_count {
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_down_cprs.clone());
            reg_claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[e]));
            reg_claims.push(to_cprs_qm31(proof.regular_up_weight_openings[e]));
            reg_claims.push(to_cprs_qm31(proof.regular_down_weight_openings[e]));
        }
        verify_strict_context(
            &reg_commitments,
            &reg_points,
            &reg_claims,
            &d_reg,
            &cprs_params.regular,
            &proof.cprs_regular_proof,
        )
        .map_err(|e| format!("Stage A Mode 11 regular CPRS failed: {e}"))?;
        let sh_points = vec![r_sh_gu_cprs.clone(), r_sh_gu_cprs, r_sh_down_cprs];
        let sh_claims = [
            to_cprs_qm31(proof.shared_gate_weight_opening),
            to_cprs_qm31(proof.shared_up_weight_opening),
            to_cprs_qm31(proof.shared_weight_opening),
        ];
        verify_strict_context(
            &[context.shared_commitment],
            &sh_points,
            &sh_claims,
            &d_sh,
            &cprs_params.shared,
            &proof.cprs_shared_proof,
        )
        .map_err(|e| format!("Stage A Mode 11 shared CPRS failed: {e}"))?;
    }

    let cprs_digest = cprs_gkr_lane_proofs_transcript_digest(
        &proof.cprs_router_proof,
        &proof.cprs_regular_proof,
        &proof.cprs_shared_proof,
        &proof.cprs_witness_proof,
    );
    mix_digest(channel, &cprs_digest);

    Ok((
        StageARegisteredVerifierRun {
            verifier_context_bytes: context.footprint_bytes(),
            transcript_census: channel.census(),
            transcript_trace: channel.trace(),
        },
        None,
    ))
}

pub fn draw_zk_block_cont_point(
    initial_chain: &StageAChainLink,
    stage_a_anchors: &[[u8; 32]],
    stage_a_statements: &[StageANonlinearBatchStatement],
    stage_b: Option<(&[[u8; 32]], &[crate::stage_b::StageBStatement])>,
    proof: &CrossLayerGkrBlockProof,
) -> Result<Vec<SecureField>, String> {
    let k = stage_a_statements.len();
    let batch = stage_a_statements[0].shape.batch;
    let dim = stage_a_statements[0].shape.dimension;
    if !batch.is_power_of_two() || !dim.is_power_of_two() {
        return Err("Mode 11 cross-layer block requires power-of-two batch and dimension".into());
    }
    let mut h = Blake2s256::new();
    h.update(b"stwo-ml:zk-full:block-cont-point:v1");
    h.update(initial_chain.layer.to_le_bytes());
    h.update(initial_chain.prev_receipt_digest);
    h.update(proof.mode_tag.to_le_bytes());
    h.update((k as u64).to_le_bytes());
    h.update([
        u8::from(stage_b.is_some()),
        proof.cprs_weight_f,
        proof.cprs_witness_f,
    ]);
    h.update(proof.stacked_expert_pack.to_le_bytes());
    for l in 0..k {
        h.update(stage_a_anchors[l]);
        h.update(stage_a_statements[l].model_digest);
        h.update(stage_a_statements[l].policy_digest);
        h.update(stage_a_statements[l].table_digest);
        h.update(stage_a_statements[l].input_digest);
        h.update(gkr_lane_output_digest(&stage_a_statements[l].output_q8));
        h.update(proof.stage_a_proofs[l].witness_commitment);
        if let Some((attn_comms, b_stmts)) = stage_b {
            h.update(attn_comms[l]);
            for v in [
                b_stmts[l].shape.dimension,
                b_stmts[l].shape.heads,
                b_stmts[l].shape.kv_heads,
                b_stmts[l].shape.head_dim,
            ] {
                h.update((v as u64).to_le_bytes());
            }
            h.update((b_stmts[l].chunk_commitments.len() as u64).to_le_bytes());
            for cc in &b_stmts[l].chunk_commitments {
                h.update(cc);
            }
            let b_wit_comm = proof.stage_b_proofs[l]
                .gkr_witness
                .as_ref()
                .ok_or_else(|| format!("layer {l} Mode 11 Stage B proof missing gkr_witness"))?
                .witness_commitment;
            h.update(b_wit_comm);
        }
    }
    let seed: [u8; 32] = h.finalize().into();
    let mut ch = Production100Channel::default();
    mix_digest(&mut ch, &seed);
    let log_b = batch.ilog2() as usize;
    let log_d = dim.ilog2() as usize;
    let mut r_cont = ch.draw_secure_felts(log_b + log_d + 1);
    if r_cont[log_b + log_d].is_zero() {
        r_cont[log_b + log_d] = SecureField::one();
    }
    Ok(r_cont)
}

pub fn verify_zk_full_block_continuity(
    block_channel: &mut Production100Channel,
    k: usize,
    has_stage_b: bool,
    layer_inputs_override: Option<&[Vec<Vec<i16>>]>,
    stage_a_statements: &[StageANonlinearBatchStatement],
    r_cont: &[SecureField],
    proof: &CrossLayerGkrBlockProof,
) -> Result<(), String> {
    block_channel.mix_felts(r_cont);
    for l in 0..k {
        let zf_l = proof.stage_a_proofs[l]
            .zk_full_aux
            .as_ref()
            .ok_or_else(|| format!("layer {l} missing Mode 11 zk_full_aux"))?;
        if zf_l.cont_claims.len() != 2 {
            return Err(format!("layer {l} Mode 11 missing 2 cont_claims"));
        }
        block_channel.mix_felts(&zf_l.cont_claims);
        if has_stage_b {
            let zb_l = proof.stage_b_proofs[l]
                .gkr_witness
                .as_ref()
                .and_then(|g| g.zk_aux.as_ref())
                .ok_or_else(|| format!("layer {l} missing Stage B v3 zk_aux"))?;
            if zb_l.cont_claims.len() != 3 {
                return Err(format!("layer {l} Stage B v3 missing 3 cont_claims"));
            }
            block_channel.mix_felts(&zb_l.cont_claims);
            // Intra-layer handoff: Stage B post-RMSNorm `x` == Stage A `x_in`.
            if zf_l.cont_claims[0] != zb_l.cont_claims[2] {
                return Err(format!(
                    "Mode 11 intra-layer x -> x_in continuity mismatch at layer {l}"
                ));
            }
        }
    }
    for l in 0..k.saturating_sub(1) {
        let zf_l = proof.stage_a_proofs[l].zk_full_aux.as_ref().unwrap();
        if has_stage_b {
            let zb_l = proof.stage_b_proofs[l]
                .gkr_witness
                .as_ref()
                .unwrap()
                .zk_aux
                .as_ref()
                .unwrap();
            let zb_next = proof.stage_b_proofs[l + 1]
                .gkr_witness
                .as_ref()
                .unwrap()
                .zk_aux
                .as_ref()
                .unwrap();
            // Inter-layer residual handoff: `h_in[l + 1] == a[l] + out_q8[l]`.
            if zb_next.cont_claims[0] != zb_l.cont_claims[1] + zf_l.cont_claims[1] {
                return Err(format!(
                    "Mode 11 inter-layer residual continuity mismatch between layers {l} and {}",
                    l + 1
                ));
            }
        } else {
            // Stage-A-only Mode 11 block: internal `inputs_override[l+1]` must be empty (hidden) or match,
            // and `x_in[l + 1] == out_q8[l]`.
            if let Some(ov) = layer_inputs_override {
                if !ov[l + 1].is_empty() && !stage_a_statements[l].output_q8.is_empty() && ov[l + 1] != stage_a_statements[l].output_q8 {
                    return Err(format!(
                        "Mode 11 Stage-A-only public override mismatch between layers {l} and {}",
                        l + 1
                    ));
                }
            }
            let zf_next = proof.stage_a_proofs[l + 1].zk_full_aux.as_ref().unwrap();
            if zf_next.cont_claims[0] != zf_l.cont_claims[1] {
                return Err(format!(
                    "Mode 11 Stage-A-only continuity mismatch between layers {l} and {}",
                    l + 1
                ));
            }
        }
    }
    Ok(())
}
