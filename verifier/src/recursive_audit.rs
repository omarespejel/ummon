//! Standalone `wasm32`-compatible verifier for Recursive / Succinct Audit Proofs
//! (`STWOREC1`), verifying a native M31 + Blake2s Stwo Circle-STARK (`FrameworkComponent`) over
//! the complete multi-layer algebraic IOP, LogUp-GKR fraction checks, cross-layer continuity, and
//! CPRS Reed-Solomon proximity-folding circuit of a `CrossLayerGkrBlockProof`.
//!
//! ### Split-Verification & Soundness Model
//! - **Tree 0 (`PREPROCESSED_TRACE_IDX = 0`, 1 M31 col)**: `IsFirst(log_size)`, pinned directly
//!   against the constant table [`IS_FIRST_PREPROCESSED_ROOTS_BLOWUP_1`].
//! - **Tree 1 (`ORIGINAL_TRACE_IDX = 1`, 29 M31 cols)**: Deterministic circuit selector schedule,
//!   copy-consistency variable IDs & multiplicities, public boundary claims, and Fiat-Shamir
//!   challenges, pinned directly against the caller-supplied `expected_wiring_root`.
//! - **Tree 2 (`INTERACTION_TRACE_IDX = 2`, 48 M31 cols)**: Prover witness execution trace
//!   (univariate polynomial coefficients, Horner intermediates, bilinear products, accumulators,
//!   and wrap-around active-gate prefix counters) with zero unconstrained slack.
//! - **Tree 3 (`MEM_BUS_TRACE_IDX = 3`, 12 M31 cols)**: Verifier-challenged degree-2 LogUp
//!   copy-consistency memory bus (`alpha_mem, z_mem` drawn after Tree 2 is committed), enforcing
//!   that every shared witness opening (`x1_opening`, `ctx_opening`, Stage B/A `witness_openings`,
//!   CPRS `y`, `z`, `c_final`, and inner oracle leaves) takes the exact same `QM31` value across
//!   every algebraic gate where it is referenced.
//! - **Tree 4 (`COMPOSITION_TRACE`, 8 M31 cols)**: Degree-2 Circle-STARK composition polynomial
//!   (`max_constraint_log_degree_bound = log_size + 1`).

use std::io::Cursor;

use bincode::Options;
use blake2::{Blake2s256, Digest};
use num_traits::{One, Zero};
use stwo::core::air::Components;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::line::LinePoly;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator, INTERACTION_TRACE_IDX,
    ORIGINAL_TRACE_IDX,
};

use crate::{SecurityProfile, StageAChainLink, MAX_WIRE_RECEIPT_BYTES};

pub const RECURSIVE_AUDIT_MAGIC: &[u8; 8] = b"STWOREC1";
pub const RECURSIVE_AUDIT_VERSION: u32 = 1;
pub const RECURSIVE_AUDIT_DOMAIN_TAG: u64 = 0x5354_574f_5245_4331;
pub const MIN_RECURSIVE_AUDIT_LOG_SIZE: u32 = 4;
pub const MAX_RECURSIVE_AUDIT_LOG_SIZE: u32 = 22;

pub const WIRING_TRACE_COLS: usize = 29;
pub const WITNESS_TRACE_COLS: usize = 48;
pub const MEM_BUS_TRACE_IDX: usize = 3;
pub const MEM_BUS_TRACE_COLS: usize = 12;

/// Precomputed `Blake2sMerkleHasher` roots of `IsFirst::new(log_size)` lifted at
/// `fri_log_blowup = 1` (`log_size = 4..=22`, 19 entries).
/// Pinned bit-for-bit against `CommitmentSchemeProver` in
/// `is_first_preprocessed_roots_table_matches_prover`.
pub static IS_FIRST_PREPROCESSED_ROOTS_BLOWUP_1: [[u8; 32]; 19] = [
    [74, 111, 147, 210, 178, 208, 101, 133, 81, 26, 198, 195, 125, 95, 54, 20, 91, 122, 252, 35, 69, 140, 170, 93, 136, 55, 132, 46, 46, 61, 51, 53], // log_size = 4
    [229, 206, 164, 86, 195, 95, 99, 184, 40, 163, 163, 202, 5, 104, 134, 134, 103, 55, 6, 79, 111, 125, 154, 136, 199, 25, 66, 25, 198, 182, 123, 247], // log_size = 5
    [110, 44, 173, 61, 18, 37, 239, 179, 244, 18, 62, 173, 34, 45, 210, 250, 53, 115, 122, 149, 147, 215, 92, 157, 108, 251, 129, 135, 209, 128, 104, 120], // log_size = 6
    [73, 102, 191, 88, 210, 220, 202, 114, 156, 62, 144, 174, 226, 119, 154, 204, 160, 20, 125, 160, 240, 139, 174, 14, 75, 63, 209, 33, 59, 33, 191, 164], // log_size = 7
    [8, 52, 119, 105, 192, 9, 199, 112, 127, 65, 193, 123, 35, 237, 249, 77, 59, 109, 248, 71, 90, 166, 238, 145, 43, 47, 117, 144, 92, 105, 21, 146], // log_size = 8
    [225, 51, 53, 250, 233, 211, 222, 109, 31, 100, 220, 99, 83, 36, 84, 116, 200, 71, 175, 87, 188, 95, 165, 129, 188, 197, 173, 49, 59, 71, 75, 119], // log_size = 9
    [95, 216, 140, 139, 53, 108, 93, 94, 195, 5, 250, 181, 101, 3, 153, 189, 232, 179, 128, 179, 131, 127, 107, 220, 160, 142, 211, 141, 147, 17, 120, 154], // log_size = 10
    [173, 161, 14, 144, 177, 208, 25, 121, 228, 186, 104, 14, 5, 46, 180, 3, 75, 33, 43, 254, 66, 33, 144, 220, 168, 245, 54, 186, 112, 2, 169, 249], // log_size = 11
    [239, 49, 242, 115, 172, 189, 232, 139, 90, 158, 254, 12, 216, 211, 212, 191, 200, 109, 106, 71, 187, 209, 228, 103, 241, 14, 54, 190, 217, 230, 121, 112], // log_size = 12
    [250, 136, 93, 40, 201, 229, 129, 69, 30, 20, 115, 196, 20, 146, 98, 133, 190, 71, 252, 165, 229, 208, 37, 9, 94, 8, 102, 82, 50, 113, 51, 194], // log_size = 13
    [59, 146, 158, 3, 120, 146, 160, 212, 148, 219, 52, 101, 199, 148, 166, 129, 193, 7, 65, 11, 80, 179, 83, 197, 4, 223, 82, 74, 250, 48, 45, 121], // log_size = 14
    [179, 15, 239, 121, 170, 124, 113, 41, 120, 165, 210, 69, 212, 163, 133, 98, 179, 183, 162, 212, 118, 212, 38, 207, 203, 0, 237, 104, 237, 232, 244, 142], // log_size = 15
    [245, 222, 219, 211, 143, 78, 203, 109, 147, 64, 232, 162, 171, 37, 166, 221, 132, 170, 82, 132, 135, 130, 186, 108, 44, 32, 238, 129, 132, 33, 160, 30], // log_size = 16
    [25, 196, 4, 100, 19, 48, 107, 152, 120, 154, 140, 98, 72, 103, 212, 17, 222, 252, 77, 202, 136, 143, 16, 154, 31, 148, 37, 21, 109, 174, 161, 252], // log_size = 17
    [97, 152, 112, 50, 222, 142, 183, 9, 28, 41, 252, 253, 91, 91, 100, 249, 66, 103, 41, 53, 76, 163, 166, 162, 239, 232, 199, 58, 223, 115, 89, 149], // log_size = 18
    [211, 154, 89, 112, 134, 153, 70, 130, 180, 169, 4, 172, 206, 84, 176, 61, 25, 219, 249, 159, 11, 149, 168, 234, 183, 93, 20, 53, 132, 55, 50, 102], // log_size = 19
    [152, 162, 242, 87, 203, 200, 222, 47, 193, 215, 117, 171, 113, 19, 109, 117, 96, 244, 107, 26, 73, 12, 216, 185, 127, 51, 224, 165, 92, 109, 45, 196], // log_size = 20
    [104, 171, 130, 184, 229, 71, 149, 224, 136, 216, 56, 6, 67, 240, 180, 204, 198, 151, 228, 152, 180, 183, 88, 146, 39, 233, 84, 234, 130, 17, 40, 239], // log_size = 21
    [186, 210, 185, 140, 79, 89, 83, 179, 235, 130, 244, 42, 197, 241, 48, 119, 35, 210, 137, 205, 196, 100, 157, 190, 217, 53, 207, 228, 112, 212, 107, 241], // log_size = 22
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveAuditStatement {
    pub mode_tag: u32,
    pub profile_id: String,
    pub batch: usize,
    pub num_layers: usize,
    pub cprs_weight_f: u8,
    pub cprs_witness_f: u8,
    pub stacked_expert_pack: u32,
    pub log_size: u32,
    pub num_sumcheck_rounds: u32,
    pub num_chain_links: u32,
    pub num_mul_gates: u32,
    pub num_eq_checks: u32,
    pub block_anchor_digest: [u8; 32],
    pub initial_chain: StageAChainLink,
    pub final_chain: StageAChainLink,
    pub public_io_digest: [u8; 32],
    pub emitted_token_id: Option<u32>,
    pub circuit_wiring_root: [u8; 32],
}

impl RecursiveAuditStatement {
    pub fn validate(&self) -> Result<SecurityProfile, String> {
        if self.mode_tag != 9 && self.mode_tag != 10 && self.mode_tag != 11 {
            return Err(format!(
                "unsupported recursive audit Stage A mode_tag {}",
                self.mode_tag
            ));
        }
        let profile = SecurityProfile::from_id(&self.profile_id)?;
        if self.batch == 0 || !self.batch.is_power_of_two() {
            return Err("recursive audit batch must be a positive power of two".into());
        }
        if self.num_layers == 0 || self.num_layers > 256 {
            return Err("recursive audit num_layers out of range".into());
        }
        if !(1..=8).contains(&self.cprs_weight_f) || !(1..=8).contains(&self.cprs_witness_f) {
            return Err("recursive audit CPRS leaf arity out of range".into());
        }
        if !(MIN_RECURSIVE_AUDIT_LOG_SIZE..=MAX_RECURSIVE_AUDIT_LOG_SIZE).contains(&self.log_size) {
            return Err(format!(
                "recursive audit log_size {} outside supported range {}..={}",
                self.log_size, MIN_RECURSIVE_AUDIT_LOG_SIZE, MAX_RECURSIVE_AUDIT_LOG_SIZE
            ));
        }
        let capacity = 1u64 << self.log_size;
        let active_rows = u64::from(self.num_sumcheck_rounds) + u64::from(self.num_mul_gates);
        if self.num_mul_gates == 0 || self.num_eq_checks == 0 || active_rows > capacity {
            return Err(
                "recursive audit active row counts exceed trace capacity or omit required gates"
                    .into(),
            );
        }
        if self.num_chain_links > self.num_sumcheck_rounds {
            return Err("recursive audit num_chain_links exceeds num_sumcheck_rounds".into());
        }
        if u64::from(self.num_eq_checks) > active_rows {
            return Err("recursive audit num_eq_checks exceeds active rows".into());
        }
        if self.block_anchor_digest == [0u8; 32] || self.circuit_wiring_root == [0u8; 32] {
            return Err("recursive audit anchor or circuit_wiring_root cannot be all-zero".into());
        }
        let expected_final_layer = self
            .initial_chain
            .layer
            .checked_add(self.num_layers as u32)
            .ok_or_else(|| "recursive audit chain layer overflow".to_string())?;
        if self.final_chain.layer != expected_final_layer {
            return Err(
                "recursive audit final_chain.layer does not advance by num_layers".into(),
            );
        }
        Ok(profile)
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(b"stwo-ml:recursive-audit-statement:v1");
        h.update(self.mode_tag.to_le_bytes());
        h.update((self.profile_id.len() as u64).to_le_bytes());
        h.update(self.profile_id.as_bytes());
        h.update((self.batch as u64).to_le_bytes());
        h.update((self.num_layers as u64).to_le_bytes());
        h.update([self.cprs_weight_f, self.cprs_witness_f]);
        h.update(self.stacked_expert_pack.to_le_bytes());
        h.update(self.log_size.to_le_bytes());
        h.update(self.num_sumcheck_rounds.to_le_bytes());
        h.update(self.num_chain_links.to_le_bytes());
        h.update(self.num_mul_gates.to_le_bytes());
        h.update(self.num_eq_checks.to_le_bytes());
        h.update(self.block_anchor_digest);
        h.update(self.initial_chain.layer.to_le_bytes());
        h.update(self.initial_chain.prev_receipt_digest);
        h.update(self.final_chain.layer.to_le_bytes());
        h.update(self.final_chain.prev_receipt_digest);
        h.update(self.public_io_digest);
        match self.emitted_token_id {
            Some(tok) => {
                h.update([1u8]);
                h.update(tok.to_le_bytes());
            }
            None => h.update([0u8]),
        }
        h.update(self.circuit_wiring_root);
        h.finalize().into()
    }

    pub fn statement_fingerprints(&self) -> (SecureField, SecureField) {
        let d = self.digest();
        let word = |idx: usize| -> BaseField {
            let raw = u32::from_le_bytes(d[4 * idx..4 * idx + 4].try_into().unwrap());
            let bounded = (raw & 0x3fff_ffff).max(1);
            BaseField::from_u32_unchecked(bounded)
        };
        let fp0 = SecureField::from_m31_array([word(0), word(1), word(2), word(3)]);
        let fp1 = SecureField::from_m31_array([word(4), word(5), word(6), word(7)]);
        (fp0, fp1)
    }
}

#[derive(Clone)]
pub struct RecursiveAuditStarkProof {
    pub statement: RecursiveAuditStatement,
    pub stark: StarkProof<Blake2sMerkleHasher>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveAuditVerifierRun {
    pub verify_ms: f64,
    pub receipt_bytes: usize,
}

pub fn pcs_config_for_profile(profile: SecurityProfile) -> PcsConfig {
    match profile {
        SecurityProfile::Dev13 => PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(0, 1, 3, 1),
            ..PcsConfig::default()
        },
        SecurityProfile::Hardened96 => PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(0, 1, 70, 1),
            ..PcsConfig::default()
        },
        SecurityProfile::SmallProof96 => PcsConfig {
            pow_bits: 28,
            fri_config: FriConfig::new(0, 1, 68, 1),
            ..PcsConfig::default()
        },
        SecurityProfile::FastProve96 => PcsConfig {
            pow_bits: 16,
            fri_config: FriConfig::new(0, 1, 80, 1),
            ..PcsConfig::default()
        },
        SecurityProfile::LongQuery => PcsConfig {
            pow_bits: 20,
            fri_config: FriConfig::new(0, 1, 160, 1),
            ..PcsConfig::default()
        },
        SecurityProfile::Production100 => PcsConfig {
            pow_bits: 26,
            composition_pow_bits: 4,
            oods_pow_bits: 16,
            fri_batch_pow_bits: 24,
            fri_config: FriConfig::new(0, 1, 224, 1).with_first_fold_pow_bits(16),
            ..PcsConfig::default()
        },
    }
}

pub fn is_first_column_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("preprocessed_is_first_{log_size}"),
    }
}

pub fn compute_recursive_audit_block_anchor(
    stage_a_anchors: &[[u8; 32]],
    attention_commitments: Option<&[[u8; 32]]>,
    stacked_bank_commitments: Option<&[[u8; 32]]>,
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"stwo-ml:recursive-audit-block-anchor:v1");
    h.update((stage_a_anchors.len() as u64).to_le_bytes());
    for a in stage_a_anchors {
        h.update(a);
    }
    match attention_commitments {
        Some(comms) => {
            h.update([1u8]);
            h.update((comms.len() as u64).to_le_bytes());
            for c in comms {
                h.update(c);
            }
        }
        None => h.update([0u8]),
    }
    match stacked_bank_commitments {
        Some(comms) => {
            h.update([1u8]);
            h.update((comms.len() as u64).to_le_bytes());
            for c in comms {
                h.update(c);
            }
        }
        None => h.update([0u8]),
    }
    h.finalize().into()
}

pub fn expected_is_first_root(log_size: u32) -> Result<[u8; 32], String> {
    if !(MIN_RECURSIVE_AUDIT_LOG_SIZE..=MAX_RECURSIVE_AUDIT_LOG_SIZE).contains(&log_size) {
        return Err(format!("unsupported IsFirst log_size {log_size}"));
    }
    Ok(IS_FIRST_PREPROCESSED_ROOTS_BLOWUP_1[(log_size - MIN_RECURSIVE_AUDIT_LOG_SIZE) as usize])
}

#[derive(Clone)]
pub struct RecursiveAuditAirEval {
    pub log_size: u32,
    pub stmt_fp_0: SecureField,
    pub stmt_fp_1: SecureField,
    pub expected_sc_count: BaseField,
    pub expected_chain_count: BaseField,
    pub expected_mul_count: BaseField,
    pub expected_eq_count: BaseField,
    pub alpha_mem: SecureField,
    pub z_mem: SecureField,
}

pub type RecursiveAuditComponent = FrameworkComponent<RecursiveAuditAirEval>;

impl RecursiveAuditAirEval {
    pub fn from_statement(
        statement: &RecursiveAuditStatement,
        alpha_mem: SecureField,
        z_mem: SecureField,
    ) -> Self {
        let (stmt_fp_0, stmt_fp_1) = statement.statement_fingerprints();
        Self {
            log_size: statement.log_size,
            stmt_fp_0,
            stmt_fp_1,
            expected_sc_count: BaseField::from_u32_unchecked(statement.num_sumcheck_rounds),
            expected_chain_count: BaseField::from_u32_unchecked(statement.num_chain_links),
            expected_mul_count: BaseField::from_u32_unchecked(statement.num_mul_gates),
            expected_eq_count: BaseField::from_u32_unchecked(statement.num_eq_checks),
            alpha_mem,
            z_mem,
        }
    }
}

impl FrameworkEval for RecursiveAuditAirEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::from(BaseField::one());
        let ef_one = E::EF::from(SecureField::one());
        let is_first = eval.get_preprocessed_column(is_first_column_id(self.log_size));

        // Tree 1 (`ORIGINAL_TRACE_IDX = 1`, 29 M31 columns): deterministic circuit wiring,
        // copy-bus variable IDs/multiplicities, Fiat-Shamir challenges, and wired constants.
        let [sel_sc, sel_sc_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [sel_deg2] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_chain, sel_chain_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [sel_mul, sel_mul_prev] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [sel_acc_chain] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_term_link] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_carry_claim] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_eq_claim, sel_eq_claim_prev] =
            eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let [sel_pin_claim] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_pin_w0] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_pin_w1] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_link_w0] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [sel_base_w0] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [mem_id_w0] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [mem_mult_w0] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [mem_id_w1] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [mem_mult_w1] = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [k0] = eval.next_extension_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [k1] = eval.next_extension_interaction_mask(ORIGINAL_TRACE_IDX, [0]);
        let [c_wire] = eval.next_extension_interaction_mask(ORIGINAL_TRACE_IDX, [0]);

        // Tree 2 (`INTERACTION_TRACE_IDX = 2`, 48 M31 columns): witness execution & counters.
        let [w0_c0] = eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [w0_c1] = eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [w0_c2] = eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [w0_c3] = eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let w0 = E::combine_ef([w0_c0, w0_c1.clone(), w0_c2.clone(), w0_c3.clone()]);
        let [w1] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [w2] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [w3] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [q_aux] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [h_aux] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [horner_val] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [prod_ab] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [acc_in] = eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0]);
        let [claim_val, claim_val_prev] =
            eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);
        let [eval_val, eval_val_prev] =
            eval.next_extension_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);
        let [sc_prefix, sc_prefix_prev] =
            eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);
        let [chain_prefix, chain_prefix_prev] =
            eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);
        let [mul_prefix, mul_prefix_prev] =
            eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);
        let [eq_prefix, eq_prefix_prev] =
            eval.next_interaction_mask(INTERACTION_TRACE_IDX, [0, -1]);

        // Tree 3 (`MEM_BUS_TRACE_IDX = 3`, 12 M31 columns): degree-2 LogUp copy-consistency bus.
        let [mem_inv_w0] = eval.next_extension_interaction_mask(MEM_BUS_TRACE_IDX, [0]);
        let [mem_inv_w1] = eval.next_extension_interaction_mask(MEM_BUS_TRACE_IDX, [0]);
        let [mem_cum, mem_cum_prev] =
            eval.next_extension_interaction_mask(MEM_BUS_TRACE_IDX, [0, -1]);

        // 1. Booleanity and structural exclusivity (all degree <= 2).
        for s in [
            sel_sc.clone(),
            sel_deg2.clone(),
            sel_chain.clone(),
            sel_mul.clone(),
            sel_acc_chain.clone(),
            sel_term_link.clone(),
            sel_carry_claim.clone(),
            sel_eq_claim.clone(),
            sel_pin_claim.clone(),
            sel_pin_w0.clone(),
            sel_pin_w1.clone(),
            sel_link_w0.clone(),
            sel_base_w0.clone(),
        ] {
            eval.add_constraint(s.clone() * (s - one.clone()));
        }
        eval.add_constraint(sel_sc.clone() * sel_mul.clone());
        eval.add_constraint(sel_deg2.clone() * (one.clone() - sel_sc.clone()));
        eval.add_constraint(sel_chain * (one.clone() - sel_sc.clone()));
        eval.add_constraint(sel_acc_chain.clone() * (one.clone() - sel_mul.clone()));
        eval.add_constraint(is_first.clone() * sel_chain_prev.clone());
        eval.add_constraint(is_first.clone() * sel_acc_chain.clone());
        eval.add_constraint(is_first.clone() * sel_term_link.clone());
        eval.add_constraint(is_first.clone() * sel_carry_claim.clone());
        eval.add_constraint(is_first.clone() * sel_link_w0.clone());
        eval.add_constraint(is_first.clone() * mem_mult_w0.clone());
        eval.add_constraint(is_first.clone() * mem_mult_w1.clone());

        // 2. Row 0 public statement anchor gate (`w0 = stmt_fp_0`, `w1 = stmt_fp_1`).
        eval.add_constraint(is_first.clone() * (one - sel_mul.clone()));
        eval.add_constraint((w0.clone() - self.stmt_fp_0) * is_first.clone());
        eval.add_constraint((w1.clone() - self.stmt_fp_1) * is_first.clone());

        // 3. Unconditional degree-2 Horner, bilinear, and accumulator intermediates.
        eval.add_constraint(q_aux.clone() - (w3.clone() * k1.clone() + w2.clone()));
        eval.add_constraint(h_aux.clone() - (q_aux * k1.clone() + w1.clone()));
        eval.add_constraint(horner_val.clone() - (h_aux * k1.clone() + w0.clone()));
        eval.add_constraint(prod_ab.clone() - w0.clone() * w1.clone());
        eval.add_constraint(acc_in.clone() - eval_val_prev.clone() * sel_acc_chain);

        // 4. Degree-2 sumcheck, bilinear accumulator, operand pins, and continuity links.
        eval.add_constraint(w3.clone() * sel_deg2);
        let poly_sum_01 = w0.clone() + w0.clone() + w1.clone() + w2 + w3;
        eval.add_constraint((claim_val.clone() - poly_sum_01) * sel_sc.clone());
        eval.add_constraint((eval_val.clone() - horner_val) * sel_sc.clone());
        eval.add_constraint((eval_val.clone() - (acc_in + prod_ab + c_wire)) * sel_mul.clone());
        let sel_inactive = E::F::from(BaseField::one()) - sel_sc - sel_mul;
        eval.add_constraint(eval_val.clone() * sel_inactive);

        eval.add_constraint((claim_val.clone() - eval_val_prev.clone()) * sel_chain_prev.clone());
        eval.add_constraint((claim_val.clone() - eval_val_prev.clone()) * sel_term_link);
        eval.add_constraint((claim_val.clone() - claim_val_prev) * sel_carry_claim);
        eval.add_constraint((claim_val.clone() - k0.clone()) * sel_pin_claim);
        eval.add_constraint((eval_val - claim_val) * sel_eq_claim);
        eval.add_constraint((w0.clone() - k0) * sel_pin_w0);
        eval.add_constraint((w1.clone() - k1) * sel_pin_w1);
        eval.add_constraint((w0.clone() - eval_val_prev) * sel_link_w0);
        eval.add_constraint(w0_c1 * sel_base_w0.clone());
        eval.add_constraint(w0_c2 * sel_base_w0.clone());
        eval.add_constraint(w0_c3 * sel_base_w0);

        // 5. Exact active-row wrap-around prefix counters (anti-zeroing / anti-truncation).
        eval.add_constraint(sc_prefix.clone() * is_first.clone());
        eval.add_constraint(
            sc_prefix - sc_prefix_prev - sel_sc_prev + is_first.clone() * self.expected_sc_count,
        );
        eval.add_constraint(chain_prefix.clone() * is_first.clone());
        eval.add_constraint(
            chain_prefix - chain_prefix_prev - sel_chain_prev
                + is_first.clone() * self.expected_chain_count,
        );
        eval.add_constraint(mul_prefix.clone() * is_first.clone());
        eval.add_constraint(
            mul_prefix - mul_prefix_prev - sel_mul_prev
                + is_first.clone() * self.expected_mul_count,
        );
        eval.add_constraint(eq_prefix.clone() * is_first.clone());
        eval.add_constraint(
            eq_prefix - eq_prefix_prev - sel_eq_claim_prev
                + is_first.clone() * self.expected_eq_count,
        );

        // 6. Verifier-challenged degree-2 LogUp shared-witness copy-consistency bus.
        let denom0 = (w0 * self.alpha_mem + mem_id_w0) - self.z_mem;
        let denom1 = (w1 * self.alpha_mem + mem_id_w1) - self.z_mem;
        eval.add_constraint(denom0 * mem_inv_w0.clone() - ef_one.clone());
        eval.add_constraint(denom1 * mem_inv_w1.clone() - ef_one);
        eval.add_constraint(mem_cum.clone() * is_first);
        eval.add_constraint(
            mem_cum - mem_cum_prev - mem_inv_w0 * mem_mult_w0 - mem_inv_w1 * mem_mult_w1,
        );

        eval
    }
}

pub fn mix_recursive_audit_statement(
    channel: &mut Blake2sChannel,
    statement: &RecursiveAuditStatement,
) {
    channel.mix_u64(RECURSIVE_AUDIT_DOMAIN_TAG);
    channel.mix_u32s(&[
        RECURSIVE_AUDIT_VERSION,
        statement.mode_tag,
        statement.log_size,
        statement.num_sumcheck_rounds,
        statement.num_chain_links,
        statement.num_mul_gates,
        statement.num_eq_checks,
        u32::from(statement.cprs_weight_f),
        u32::from(statement.cprs_witness_f),
        statement.stacked_expert_pack,
    ]);
    let digest = statement.digest();
    let mut words = [0u32; 8];
    for i in 0..8 {
        words[i] = u32::from_le_bytes(digest[4 * i..4 * i + 4].try_into().unwrap());
    }
    channel.mix_u32s(&words);
}

/// Verify a `STWOREC1` (`RecursiveAuditStarkProof`) against caller-supplied trusted anchors:
/// - `expected_block_anchor`: 32-byte anchor digest binding the block's registered weight & context anchors.
/// - `expected_wiring_root`: 32-byte Merkle root of Tree 1 (`circuit_wiring_root`).
/// - `expected_initial_chain`: expected entry `StageAChainLink` at the start of the `K`-layer block.
pub fn verify_recursive_audit_stark_against_anchor(
    expected_block_anchor: &[u8; 32],
    expected_wiring_root: &[u8; 32],
    expected_initial_chain: &StageAChainLink,
    proof: &RecursiveAuditStarkProof,
) -> Result<RecursiveAuditVerifierRun, String> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let start = std::time::Instant::now();
    if *expected_block_anchor == [0u8; 32] || *expected_wiring_root == [0u8; 32] {
        return Err("recursive audit expected anchors must not be all-zero".into());
    }
    let profile = proof.statement.validate()?;
    if &proof.statement.block_anchor_digest != expected_block_anchor {
        return Err("recursive audit statement block_anchor_digest mismatch".into());
    }
    if &proof.statement.circuit_wiring_root != expected_wiring_root {
        return Err("recursive audit statement circuit_wiring_root mismatch".into());
    }
    if &proof.statement.initial_chain != expected_initial_chain {
        return Err("recursive audit statement initial_chain mismatch".into());
    }
    let expected_config = pcs_config_for_profile(profile);
    if proof.stark.config != expected_config {
        return Err("recursive audit STARK proof PcsConfig mismatch".into());
    }
    if proof.stark.commitments.len() != 5 {
        return Err("recursive audit STARK proof must contain exactly 5 tree commitments".into());
    }
    let expected_tree0 = expected_is_first_root(proof.statement.log_size)?;
    if proof.stark.commitments[0].0 != expected_tree0 {
        return Err("recursive audit STARK Tree 0 (IsFirst) commitment mismatch".into());
    }
    if &proof.stark.commitments[1].0 != expected_wiring_root {
        return Err("recursive audit STARK Tree 1 (circuit_wiring_root) commitment mismatch".into());
    }

    let channel = &mut Blake2sChannel::default();
    mix_recursive_audit_statement(channel, &proof.statement);

    let log_size = proof.statement.log_size;
    let scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(expected_config);
    scheme.commit(proof.stark.commitments[0], &[log_size; 1], channel);
    scheme.commit(
        proof.stark.commitments[1],
        &[log_size; WIRING_TRACE_COLS],
        channel,
    );
    scheme.commit(
        proof.stark.commitments[2],
        &[log_size; WITNESS_TRACE_COLS],
        channel,
    );

    let alpha_mem = channel.draw_secure_felt();
    let z_mem = channel.draw_secure_felt();
    scheme.commit(
        proof.stark.commitments[3],
        &[log_size; MEM_BUS_TRACE_COLS],
        channel,
    );

    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
        is_first_column_id(proof.statement.log_size),
    ]);
    let component = RecursiveAuditComponent::new(
        &mut allocator,
        RecursiveAuditAirEval::from_statement(&proof.statement, alpha_mem, z_mem),
        SecureField::zero(),
    );
    let components = Components {
        components: vec![&component],
        n_preprocessed_columns: 1,
    };
    debug_assert_eq!(components.column_log_sizes().len(), 4);
    // Tree 4 (composition polynomial) is committed and mixed inside `stwo::core::verifier::verify`.
    verify(&components.components, channel, scheme, proof.stark.clone())
        .map_err(|e| format!("recursive audit STARK rejected: {e}"))?;

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let verify_ms = start.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let verify_ms = 0.0;

    Ok(RecursiveAuditVerifierRun {
        verify_ms,
        receipt_bytes: proof.stark.size_estimate(),
    })
}

pub(crate) fn validate_stark_field_encodings(
    proof: &StarkProof<Blake2sMerkleHasher>,
    label: &str,
) -> Result<(), String> {
    for (tree_idx, tree) in proof.sampled_values.iter().enumerate() {
        for (col_idx, col) in tree.iter().enumerate() {
            for (val_idx, &val) in col.iter().enumerate() {
                for c in val.to_m31_array() {
                    if c.0 >= P {
                        return Err(format!(
                            "non-canonical {label} sampled value {tree_idx}/{col_idx}/{val_idx}"
                        ));
                    }
                }
            }
        }
    }
    for (tree_idx, tree) in proof.queried_values.iter().enumerate() {
        for (col_idx, col) in tree.iter().enumerate() {
            for (val_idx, &val) in col.iter().enumerate() {
                if val.0 >= P {
                    return Err(format!(
                        "non-canonical {label} queried value {tree_idx}/{col_idx}/{val_idx}"
                    ));
                }
            }
        }
    }
    for (idx, &val) in proof.fri_proof.first_layer.fri_witness.iter().enumerate() {
        for c in val.to_m31_array() {
            if c.0 >= P {
                return Err(format!("non-canonical {label} first FRI witness {idx}"));
            }
        }
    }
    for (l_idx, layer) in proof.fri_proof.inner_layers.iter().enumerate() {
        for (idx, &val) in layer.fri_witness.iter().enumerate() {
            for c in val.to_m31_array() {
                if c.0 >= P {
                    return Err(format!(
                        "non-canonical {label} inner FRI witness {l_idx}/{idx}"
                    ));
                }
            }
        }
    }
    for (idx, &val) in proof.fri_proof.last_layer_poly.iter().enumerate() {
        for c in val.to_m31_array() {
            if c.0 >= P {
                return Err(format!("non-canonical {label} last-layer coeff {idx}"));
            }
        }
    }
    Ok(())
}

pub fn recursive_audit_receipt_bytes(proof: &RecursiveAuditStarkProof) -> Result<Vec<u8>, String> {
    let _ = proof.statement.validate()?;
    let stark_bytes = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&proof.stark)
        .map_err(|e| format!("failed to serialize recursive audit StarkProof: {e}"))?;
    if stark_bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("recursive audit StarkProof exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    let mut out = Vec::with_capacity(232 + stark_bytes.len());
    out.extend_from_slice(RECURSIVE_AUDIT_MAGIC);
    out.extend_from_slice(&RECURSIVE_AUDIT_VERSION.to_le_bytes());
    out.extend_from_slice(&proof.statement.mode_tag.to_le_bytes());
    out.extend_from_slice(&(proof.statement.profile_id.len() as u64).to_le_bytes());
    out.extend_from_slice(proof.statement.profile_id.as_bytes());
    out.extend_from_slice(&(proof.statement.batch as u64).to_le_bytes());
    out.extend_from_slice(&(proof.statement.num_layers as u64).to_le_bytes());
    out.push(proof.statement.cprs_weight_f);
    out.push(proof.statement.cprs_witness_f);
    out.extend_from_slice(&proof.statement.stacked_expert_pack.to_le_bytes());
    out.extend_from_slice(&proof.statement.log_size.to_le_bytes());
    out.extend_from_slice(&proof.statement.num_sumcheck_rounds.to_le_bytes());
    out.extend_from_slice(&proof.statement.num_chain_links.to_le_bytes());
    out.extend_from_slice(&proof.statement.num_mul_gates.to_le_bytes());
    out.extend_from_slice(&proof.statement.num_eq_checks.to_le_bytes());
    out.extend_from_slice(&proof.statement.block_anchor_digest);
    out.extend_from_slice(&proof.statement.initial_chain.layer.to_le_bytes());
    out.extend_from_slice(&proof.statement.initial_chain.prev_receipt_digest);
    out.extend_from_slice(&proof.statement.final_chain.layer.to_le_bytes());
    out.extend_from_slice(&proof.statement.final_chain.prev_receipt_digest);
    out.extend_from_slice(&proof.statement.public_io_digest);
    match proof.statement.emitted_token_id {
        Some(tok) => {
            out.push(1u8);
            out.extend_from_slice(&tok.to_le_bytes());
        }
        None => out.push(0u8),
    }
    out.extend_from_slice(&proof.statement.circuit_wiring_root);
    out.extend_from_slice(&(stark_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&stark_bytes);
    Ok(out)
}

pub fn decode_recursive_audit_receipt(bytes: &[u8]) -> Result<RecursiveAuditStarkProof, String> {
    if bytes.len() > MAX_WIRE_RECEIPT_BYTES {
        return Err("recursive audit receipt exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    let mut cur = 0usize;
    let take = |cur: &mut usize, n: usize, label: &str| -> Result<&[u8], String> {
        let end = cur
            .checked_add(n)
            .ok_or_else(|| format!("overflow reading {label}"))?;
        if end > bytes.len() {
            return Err(format!("unexpected EOF reading {label}"));
        }
        let s = &bytes[*cur..end];
        *cur = end;
        Ok(s)
    };
    let read_u32 = |cur: &mut usize, label: &str| -> Result<u32, String> {
        Ok(u32::from_le_bytes(take(cur, 4, label)?.try_into().unwrap()))
    };
    let read_u64 = |cur: &mut usize, label: &str| -> Result<u64, String> {
        Ok(u64::from_le_bytes(take(cur, 8, label)?.try_into().unwrap()))
    };
    let read_hash = |cur: &mut usize, label: &str| -> Result<[u8; 32], String> {
        Ok(take(cur, 32, label)?.try_into().unwrap())
    };

    if take(&mut cur, 8, "magic")? != RECURSIVE_AUDIT_MAGIC {
        return Err("invalid recursive audit receipt magic (expected STWOREC1)".into());
    }
    let version = read_u32(&mut cur, "version")?;
    if version != RECURSIVE_AUDIT_VERSION {
        return Err(format!(
            "unsupported recursive audit receipt version {version}"
        ));
    }
    let mode_tag = read_u32(&mut cur, "mode_tag")?;
    let prof_len = usize::try_from(read_u64(&mut cur, "profile_id len")?)
        .map_err(|_| "profile_id len overflow".to_string())?;
    if prof_len > 64 {
        return Err("profile_id too long".into());
    }
    let profile_id = std::str::from_utf8(take(&mut cur, prof_len, "profile_id")?)
        .map_err(|_| "invalid utf-8 in profile_id".to_string())?
        .to_string();
    let batch = usize::try_from(read_u64(&mut cur, "batch")?)
        .map_err(|_| "batch overflow".to_string())?;
    let num_layers = usize::try_from(read_u64(&mut cur, "num_layers")?)
        .map_err(|_| "num_layers overflow".to_string())?;
    let f_pair = take(&mut cur, 2, "cprs_f")?;
    let cprs_weight_f = f_pair[0];
    let cprs_witness_f = f_pair[1];
    let stacked_expert_pack = read_u32(&mut cur, "stacked_expert_pack")?;
    let log_size = read_u32(&mut cur, "log_size")?;
    let num_sumcheck_rounds = read_u32(&mut cur, "num_sumcheck_rounds")?;
    let num_chain_links = read_u32(&mut cur, "num_chain_links")?;
    let num_mul_gates = read_u32(&mut cur, "num_mul_gates")?;
    let num_eq_checks = read_u32(&mut cur, "num_eq_checks")?;
    let block_anchor_digest = read_hash(&mut cur, "block_anchor_digest")?;
    let initial_layer = read_u32(&mut cur, "initial_chain.layer")?;
    let initial_prev = read_hash(&mut cur, "initial_chain.prev_receipt_digest")?;
    let final_layer = read_u32(&mut cur, "final_chain.layer")?;
    let final_prev = read_hash(&mut cur, "final_chain.prev_receipt_digest")?;
    let public_io_digest = read_hash(&mut cur, "public_io_digest")?;
    let has_tok = take(&mut cur, 1, "has_emitted_token")?[0];
    let emitted_token_id = match has_tok {
        0 => None,
        1 => Some(read_u32(&mut cur, "emitted_token_id")?),
        _ => return Err("non-canonical emitted_token_id tag".into()),
    };
    let circuit_wiring_root = read_hash(&mut cur, "circuit_wiring_root")?;

    let stark_len = usize::try_from(read_u64(&mut cur, "stark_len")?)
        .map_err(|_| "stark_len overflow".to_string())?;
    if stark_len > MAX_WIRE_RECEIPT_BYTES {
        return Err("stark_len exceeds MAX_WIRE_RECEIPT_BYTES".into());
    }
    let stark_bytes = take(&mut cur, stark_len, "stark_bytes")?;
    if cur != bytes.len() {
        return Err("trailing bytes after STWOREC1 receipt".into());
    }

    let mut bcursor = Cursor::new(stark_bytes);
    let stark: StarkProof<Blake2sMerkleHasher> = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(stark_bytes.len() as u64)
        .deserialize_from(&mut bcursor)
        .map_err(|e| format!("failed to decode STWOREC1 StarkProof: {e}"))?;
    if bcursor.position() != stark_bytes.len() as u64 {
        return Err("trailing bytes inside STWOREC1 StarkProof".into());
    }
    validate_stark_field_encodings(&stark, "STWOREC1")?;
    let last_coeffs: Vec<_> = stark.fri_proof.last_layer_poly.iter().copied().collect();
    if last_coeffs.is_empty() || !last_coeffs.len().is_power_of_two() {
        return Err("invalid STWOREC1 last-layer coefficient count".into());
    }
    if LinePoly::new(last_coeffs) != stark.fri_proof.last_layer_poly {
        return Err("inconsistent STWOREC1 last-layer polynomial metadata".into());
    }

    let statement = RecursiveAuditStatement {
        mode_tag,
        profile_id,
        batch,
        num_layers,
        cprs_weight_f,
        cprs_witness_f,
        stacked_expert_pack,
        log_size,
        num_sumcheck_rounds,
        num_chain_links,
        num_mul_gates,
        num_eq_checks,
        block_anchor_digest,
        initial_chain: StageAChainLink {
            layer: initial_layer,
            prev_receipt_digest: initial_prev,
        },
        final_chain: StageAChainLink {
            layer: final_layer,
            prev_receipt_digest: final_prev,
        },
        public_io_digest,
        emitted_token_id,
        circuit_wiring_root,
    };
    let _ = statement.validate()?;
    let decoded = RecursiveAuditStarkProof { statement, stark };
    if recursive_audit_receipt_bytes(&decoded)? != bytes {
        return Err("non-canonical STWOREC1 receipt encoding".into());
    }
    Ok(decoded)
}
