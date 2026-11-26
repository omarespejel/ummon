//! Standalone `wasm32`-compatible verifier for Single-Anchor Receipts:
//! - **`STWOREC2` ([`StandaloneBlockStarkProof`])**: Standalone single-block Circle-STARK receipt
//!   verified against **only** the `32 B` `expected_standalone_anchor` + `expected_initial_chain`
//!   (`verification_model = "standalone_anchor_only"`, **zero `expected_wiring_root`**).
//! - **`STWOMDL1` ([`WholeModelStandaloneStarkProof`])**: Whole-model `N_blocks -> 1` folded +
//!   registered `Embedding` (`prompt_token_ids -> h_0`) + final `RMSNorm` / `LM Head` / `Argmax`
//!   (`h_L -> emitted_token_ids`) standalone Circle-STARK receipt verified against **only** the
//!   `32 B` `expected_full_stack_anchor`, `prompt_token_ids`, and `emitted_token_ids`.
//!
//! ### Four-Tree Standalone Soundness Architecture
//! - **Tree 0 (`PREPROCESSED_TRACE_IDX = 0`)**: 100% static (prompt-, routing-, witness-, and
//!   transcript-independent) preprocessed & circuit-topology columns (`IsFirst`, `Xor9/8/7/4`,
//!   `Blake2sMemoryBridge` static schedule, and `StandaloneAudit` static gate/bus schedule).
//!   `proof.stark.commitments[0]` is cryptographically bound directly inside the `32 B`
//!   `standalone_anchor` / `full_stack_anchor` at model registration time.
//! - **Tree 1 (`ORIGINAL_TRACE_IDX = 1`)**: Prover witness columns across `StandaloneAuditAirEval`,
//!   `Blake2sSchedulerEval`, `Blake2sRoundEval`, `Blake2sMemoryBridgeEval`, and `Xor9/8/7/4`
//!   multiplicities, with profile-derived (`> 2 * n_queries + 2`, e.g. `512 > 450` at
//!   `production100`) per-component OS-CSPRNG trace-blinding rows when `zk_trace_blinded == true`.
//! - **Tree 2 (`INTERACTION_TRACE_IDX = 2`)**: Verifier-challenged LogUp cumulative-sum columns
//!   across `MemoryBusElements`, `Blake2sCompressElements`, `Blake2sRoundElements`,
//!   `Blake2sChainElements`, `ByteBusElements`, and `Blake2sXorElements`, with public inputs bound
//!   purely via the verifier-computed `public_memory_sum`.
//! - **Tree 3 (`COMPOSITION_TRACE_IDX = 3`)**: Circle-STARK quotient composition polynomial under
//!   unchanged `SecurityProfile` (`FastProve96` or `Production100`).

use bincode::Options;
use blake2::{Blake2s256, Digest};
use num_traits::{One, Zero};
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, InfoEvaluator, Relation, RelationEntry,
    TraceLocationAllocator, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
};

use crate::blake2s_air::{
    blake2s_bridge_info, blake2s_bridge_static_ids, blake2s_round_info, blake2s_scheduler_info,
    blake2s_xor4_info, blake2s_xor7_info, blake2s_xor8_info, blake2s_xor9_info,
    preprocessed_blake2s_xor_ids, preprocessed_blake2s_xor_log_sizes, Blake2sChainElements,
    Blake2sCompressElements, Blake2sMemoryBridgeComponent, Blake2sMemoryBridgeEval,
    Blake2sRoundComponent, Blake2sRoundElements, Blake2sRoundEval, Blake2sSchedulerComponent,
    Blake2sSchedulerEval, Blake2sXor4Component, Blake2sXor4Eval, Blake2sXor7Component,
    Blake2sXor7Eval, Blake2sXor8Component, Blake2sXor8Eval, Blake2sXor9Component, Blake2sXor9Eval,
    Blake2sXorElements, ByteBusElements, MemoryBusElements, BRIDGE_XOR7_PER_ROW_MAX, BRIDGE_XOR8_PER_ROW_MAX,
    ROUND_XOR4_PER_ROW, ROUND_XOR7_PER_ROW, ROUND_XOR8_PER_ROW, ROUND_XOR9_PER_ROW, SCHED_XOR8_PER_ROW,
};
use crate::blake2s_air::{outer_ids, outer_keep, outer_thread_mask};

pub const STANDALONE_MAX_WIRE_RECEIPT_BYTES: usize = 64 * 1024 * 1024;

pub fn standalone_is_first_id(log_size: u32) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("preprocessed_is_first_{log_size}"),
    }
}

/// EXPERIMENTAL profiles (not production-ready): every hash computed inside the circuit is
/// Poseidon2 over M31 (`poseidon2_m31.rs`) and the outer AIR is `[audit, Poseidon2]`, with the
/// outer STARK, the inner CPRS parameters, the grinding, the LogUp challenge sets and the blinding
/// of the named base profile. Only the `STWOREC2` transformer block supports them.
/// `exp_pos2_dev13` exists for fast tests, like `dev13`, and carries no security claim.
/// `exp_pos2_production_hardened` is `production_hardened` with the Poseidon2 in-circuit hash
/// in place of Blake2s.
pub const EXP_POS2_PROFILES: [(&str, &str); 4] = [
    ("exp_pos2_fastprove96", "fastprove96"),
    ("exp_pos2_production100", "production100"),
    ("exp_pos2_dev13", "dev13"),
    ("exp_pos2_production_hardened", "production_hardened"),
];

/// The base profile of an `exp_pos2_*` profile, `None` for every other id.
pub fn pos2_base_profile(profile_id: &str) -> Option<&'static str> {
    EXP_POS2_PROFILES
        .iter()
        .find(|(exp, _)| *exp == profile_id)
        .map(|(_, base)| *base)
}

pub fn is_pos2_profile(profile_id: &str) -> bool {
    pos2_base_profile(profile_id).is_some()
}

pub fn standalone_pcs_config_for_profile_id(profile_id: &str) -> Result<PcsConfig, String> {
    let profile_id = pos2_base_profile(profile_id).unwrap_or(profile_id);
    match profile_id {
        "dev13" => Ok(PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(6, 3, 3, 3),
            ..PcsConfig::default()
        }),
        "hardened96" => Ok(PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig::new(6, 3, 35, 3),
            ..PcsConfig::default()
        }),
        "smallproof96" => Ok(PcsConfig {
            pow_bits: 28,
            fri_config: FriConfig::new(6, 3, 34, 3),
            ..PcsConfig::default()
        }),
        "fastprove96" => Ok(PcsConfig {
            pow_bits: 16,
            fri_config: FriConfig::new(6, 3, 40, 3),
            ..PcsConfig::default()
        }),
        "longquery" => Ok(PcsConfig {
            pow_bits: 20,
            fri_config: FriConfig::new(6, 3, 80, 3),
            ..PcsConfig::default()
        }),
        "production100" | PRODUCTION_HARDENED_PROFILE_ID => Ok(PcsConfig {
            pow_bits: 26,
            composition_pow_bits: 4,
            oods_pow_bits: 16,
            fri_batch_pow_bits: 24,
            fri_config: FriConfig::new(6, 3, 112, 3).with_first_fold_pow_bits(16),
            ..PcsConfig::default()
        }),
        // EXPERIMENT (not a production profile): the
        // production100 outer STARK at log blowup 3..6. Query counts keep the unique-decoding
        // query bits of production100, query round only (112 * 0.8301 = 92.97):
        // 102 * 0.9125 = 93.08, 98 * 0.9556 = 93.65, 96 * 0.9776 = 93.85. All grinding is
        // unchanged. `exp_p100_lb*` keep the production100 inner CPRS; `exp_outer100_lb*` pair
        // the same outer STARK with the dev13 inner CPRS only to shrink the trace for
        // measurement, and carry no security claim.
        "exp_p100_lb4" | "exp_outer100_lb4" => Ok(exp_outer100_pcs(4, 102)),
        "exp_p100_lb5" | "exp_outer100_lb5" => Ok(exp_outer100_pcs(5, 98)),
        "exp_p100_lb6" | "exp_outer100_lb6" => Ok(exp_outer100_pcs(6, 96)),
        "exp_outer100_lb3" => Ok(exp_outer100_pcs(3, 112)),
        other => Err(format!("unsupported standalone STARK SecurityProfile '{other}'")),
    }
}

/// The hardened `STWOREC2` profile (transformer blocks only). Its outer STARK is `production100`'s;
/// what differs is checked elsewhere: 139 inner CPRS queries instead of 118
/// (`cprs_s_q_pow_for_profile_id`) and a proof of work before the six outer lookup draws
/// ([`standalone_lookup_pow_bits_for_profile_id`]). Its security figures are in
/// the paper (Section 5 and Appendix A); the name states no level.
pub const PRODUCTION_HARDENED_PROFILE_ID: &str = "production_hardened";

/// Leading zero bits of the proof of work the prover mixes into the outer channel after the
/// tree-1 commitment and before the six LogUp relation draws. Zero for every other profile, whose
/// transcript and wire bytes stay as they were.
pub const PRODUCTION_HARDENED_LOOKUP_POW_BITS: u32 = 16;

/// Every profile id the standalone STARK accepts: the match arms of
/// [`standalone_pcs_config_for_profile_id`] and the `exp_pos2` table. The security checker
/// (not released) reads this list and those arms, requires them to agree,
/// and requires every id to be modeled or listed with the reason it is not.
pub const STANDALONE_PROFILE_IDS: [&str; 18] = [
    "dev13",
    "hardened96",
    "smallproof96",
    "fastprove96",
    "longquery",
    "production100",
    PRODUCTION_HARDENED_PROFILE_ID,
    "exp_p100_lb4",
    "exp_p100_lb5",
    "exp_p100_lb6",
    "exp_outer100_lb3",
    "exp_outer100_lb4",
    "exp_outer100_lb5",
    "exp_outer100_lb6",
    "exp_pos2_fastprove96",
    "exp_pos2_production100",
    "exp_pos2_dev13",
    "exp_pos2_production_hardened",
];

/// Lookup-draw proof-of-work bits of a standalone profile (zero when the profile has none); an
/// `exp_pos2_*` profile has its base profile's.
pub fn standalone_lookup_pow_bits_for_profile_id(profile_id: &str) -> u32 {
    if pos2_base_profile(profile_id).unwrap_or(profile_id) == PRODUCTION_HARDENED_PROFILE_ID {
        PRODUCTION_HARDENED_LOOKUP_POW_BITS
    } else {
        0
    }
}

fn exp_outer100_pcs(log_blowup: u32, n_queries: usize) -> PcsConfig {
    PcsConfig {
        pow_bits: 26,
        composition_pow_bits: 4,
        oods_pow_bits: 16,
        fri_batch_pow_bits: 24,
        fri_config: FriConfig::new(6, log_blowup, n_queries, 3).with_first_fold_pow_bits(16),
        ..PcsConfig::default()
    }
}

pub const STANDALONE_BLOCK_MAGIC: &[u8; 8] = b"STWOREC2";
pub const WHOLE_MODEL_MAGIC: &[u8; 8] = b"STWOMDL1";
/// Wire and anchor version of `STWOREC2`/`STWOMDL1`. Version 11: per-row exponents at every weight
/// site (`C8R_rowpow2_all`); a version 10 receipt is refused on decode, never read as 11.
pub const STANDALONE_AUDIT_VERSION: u32 = 11;

/// Experimental transformer-block mode: mode 11 with the lookups proved by LogUp-GKR
/// (`mode_tag` 12).
pub const STANDALONE_MODE_LOGUP_GKR: u32 = 12;

/// Most layers one `STWOREC2` block may hold.
pub const MAX_STANDALONE_BLOCK_LAYERS: usize = 256;

pub const STANDALONE_BLOCK_ANCHOR_DOMAIN: &[u8] = b"SPARSEPROVE_STWOREC2_ANCHOR_V1";
pub const WHOLE_MODEL_ANCHOR_DOMAIN: &[u8] = b"SPARSEPROVE_STWOMDL1_ANCHOR_V1";
pub const STANDALONE_WEIGHTS_DIGEST_DOMAIN: &[u8] = b"SPARSEPROVE_STANDALONE_WEIGHTS_V1";
pub const STANDALONE_PUBLIC_IO_DOMAIN: &[u8] = b"SPARSEPROVE_STWOREC2_PUB_IO_V2";
pub const STANDALONE_CHAIN_ANCHOR_DOMAIN: &[u8] = b"SPARSEPROVE_STWOCHN2_ANCHOR_V1";
/// Domain of the public input/output digest of a block with an end stage.
pub const STANDALONE_ENDS_IO_DOMAIN: &[u8] = b"SPARSEPROVE_STWOREC2_ENDS_IO_V1";
/// Domain of the public input/output digest of a multi-token causal decode end stage.
pub const STANDALONE_ENDS_MULTI_IO_DOMAIN: &[u8] = b"SPARSEPROVE_STWOREC2_ENDS_MULTI_IO_V1";
/// Domain of the anchor of a model registered as `[embedding receipt, blocks.., head receipt]` (7.4).
pub const STANDALONE_MODEL_ANCHOR_DOMAIN: &[u8] = b"SPARSEPROVE_STWOMDL2_MODEL_ANCHOR_V1";
pub const PROMPT_COMMITMENT_DOMAIN: &[u8] = b"SPARSEPROVE_PROMPT_COM_V2";
pub const EMITTED_COMMITMENT_DOMAIN: &[u8] = b"SPARSEPROVE_EMITTED_COM_V2";

pub const MIN_STANDALONE_AUDIT_LOG_SIZE: u32 = 4;
pub const MAX_STANDALONE_AUDIT_LOG_SIZE: u32 = 24;

/// Minimum blinding row count strictly exceeding `2 * n_queries + 2` (two evaluations per FRI
/// query due to offset `-1` row reads in `StandaloneAuditAirEval`, plus 2 OODS evaluation points).
pub fn standalone_min_zk_blind_rows_for_queries(n_queries: usize) -> usize {
    2 * n_queries + 3
}

/// Validate that `blind_rows` is even (`(sel_sc, sel_mul)` pairs) and strictly exceeds
/// `2 * n_queries + 2`.
pub fn validate_standalone_zk_blind_rows(
    blind_rows: usize,
    n_queries: usize,
) -> Result<(), String> {
    let min_required = standalone_min_zk_blind_rows_for_queries(n_queries);
    if blind_rows < min_required {
        return Err(format!(
            "standalone trace-blinding rows {blind_rows} <= 2 * n_queries + 2 ({}) for n_queries={n_queries}",
            2 * n_queries + 2
        ));
    }
    if blind_rows % 2 != 0 {
        return Err(format!(
            "standalone trace-blinding rows {blind_rows} must be even ((sel_sc, sel_mul) pairs)"
        ));
    }
    Ok(())
}

/// Derive the power-of-two per-component blinding row count from the active profile's `PcsConfig`
/// while preserving the earlier blowup-2 floor (`512` at `production100`, `256` at
/// `fastprove96`) when `log_blowup_factor >= 3` halves `n_queries`.
pub fn standalone_zk_blind_rows_for_pcs(pcs: &PcsConfig) -> usize {
    let q = pcs.fri_config.n_queries;
    let floor_queries = if pcs.fri_config.log_blowup_factor >= 3 && q > 3 {
        q.saturating_mul(2)
    } else {
        q
    };
    standalone_min_zk_blind_rows_for_queries(floor_queries)
        .max(16)
        .next_power_of_two()
}

pub fn standalone_zk_blind_rows_for_profile_id(profile_id: &str) -> Result<usize, String> {
    let pcs = standalone_pcs_config_for_profile_id(profile_id)?;
    let blind_rows = standalone_zk_blind_rows_for_pcs(&pcs);
    validate_standalone_zk_blind_rows(blind_rows, pcs.fri_config.n_queries)?;
    Ok(blind_rows)
}

pub const STANDALONE_AUDIT_STATIC_COLS: usize = 25;
pub const STANDALONE_AUDIT_WITNESS_COLS: usize = 48;

pub const PUB_ID_WEIGHTS_DIGEST_BASE: u32 = 0x4000_0000;
pub const PUB_ID_INITIAL_CHAIN_BASE: u32 = 0x4000_0010;
pub const PUB_ID_FINAL_CHAIN_BASE: u32 = 0x4000_0020;
pub const PUB_ID_PUBLIC_IO_BASE: u32 = 0x4000_0030;
pub const PUB_ID_PROMPT_COM_BASE: u32 = 0x4000_0040;
pub const PUB_ID_EMITTED_COM_BASE: u32 = 0x4000_0050;
pub const PUB_ID_PROMPT_TOK_BASE: u32 = 0x4001_0000;
pub const PUB_ID_EMITTED_TOK_BASE: u32 = 0x4002_0000;

/// Compute the canonical salted `Blake2s-256` commitment to a token slice (`prompt` or `emitted`)
/// matching in-AIR `Blake2sBridgeBuilder::hash_segments` (`domain || 32B_salt || 16B_qm31_tokens`).
pub fn compute_salted_tokens_commitment(
    domain: &[u8],
    salt: &[u8; 32],
    tokens: &[u32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(domain);
    h.update(salt);
    for &tok in tokens {
        h.update(tok.to_le_bytes());
        h.update([0u8; 12]);
    }
    h.finalize().into()
}

pub fn standalone_audit_static_id(log_size: u32, col_idx: usize) -> PreProcessedColumnId {
    PreProcessedColumnId {
        id: format!("sparseprove_standalone_audit_static_{log_size}_{col_idx}"),
    }
}

pub fn standalone_audit_static_ids(
    log_size: u32,
) -> [PreProcessedColumnId; STANDALONE_AUDIT_STATIC_COLS] {
    core::array::from_fn(|c| standalone_audit_static_id(log_size, c))
}

pub type StandaloneAuditComponent = FrameworkComponent<StandaloneAuditAirEval>;

#[derive(Clone, Debug)]
pub struct StandaloneAuditAirEval {
    pub log_size: u32,
    pub mem_bus_elements: MemoryBusElements,
    pub claimed_sum: SecureField,
    /// Test-only mask of dropped constraints (`outer_skip`), captured when the evaluator is built.
    #[cfg(test)]
    pub skip: u64,
}

impl StandaloneAuditAirEval {
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

impl FrameworkEval for StandaloneAuditAirEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + 1
    }

    #[inline(never)]
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let one = E::F::one();
        let is_first = eval.get_preprocessed_column(standalone_is_first_id(self.log_size));
        let st: [E::F; STANDALONE_AUDIT_STATIC_COLS] = core::array::from_fn(|c| {
            eval.get_preprocessed_column(standalone_audit_static_id(self.log_size, c))
        });

        let sel_sc = st[0].clone();
        let sel_deg2 = st[1].clone();
        let sel_chain_from_prev = st[2].clone();
        let sel_mul = st[3].clone();
        let sel_acc_chain = st[4].clone();
        let sel_eq_claim = st[5].clone();
        let sel_static_w0 = st[6].clone();
        let sel_static_w1 = st[7].clone();
        let sel_static_claim = st[8].clone();
        let sel_base_w0 = st[9].clone();
        let sel_bool_w0 = st[10].clone();
        let mem_id_w0 = st[11].clone();
        let mem_mult_w0 = st[12].clone();
        let mem_id_w1 = st[13].clone();
        let mem_mult_w1 = st[14].clone();
        let mem_id_chal = st[15].clone();
        let mem_mult_chal = st[16].clone();
        let mem_id_eval = st[17].clone();
        let mem_mult_eval = st[18].clone();
        let k_static = E::combine_ef([
            st[19].clone(),
            st[20].clone(),
            st[21].clone(),
            st[22].clone(),
        ]);
        let sel_add_k_static = st[23].clone();
        let sel_carry_claim = st[24].clone();

        let w0_coords: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let w1_coords: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let w2_coords: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let w3_coords: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let chal_coords: [E::F; 4] = core::array::from_fn(|_| eval.next_trace_mask());
        let q_aux = E::combine_ef(core::array::from_fn(|_| eval.next_trace_mask()));
        let h_aux = E::combine_ef(core::array::from_fn(|_| eval.next_trace_mask()));
        let horner_val = E::combine_ef(core::array::from_fn(|_| eval.next_trace_mask()));
        let prod_ab = E::combine_ef(core::array::from_fn(|_| eval.next_trace_mask()));
        let acc_in = E::combine_ef(core::array::from_fn(|_| eval.next_trace_mask()));
        let [claim_val, claim_val_prev] =
            eval.next_extension_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]);
        let eval_coords_both: [[E::F; 2]; 4] =
            core::array::from_fn(|_| eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, -1]));
        let eval_coords: [E::F; 4] = core::array::from_fn(|c| eval_coords_both[c][0].clone());
        let eval_prev_coords: [E::F; 4] = core::array::from_fn(|c| eval_coords_both[c][1].clone());

        let w0 = E::combine_ef(w0_coords.clone());
        let w1 = E::combine_ef(w1_coords.clone());
        let w2 = E::combine_ef(w2_coords);
        let w3 = E::combine_ef(w3_coords);
        let chal = E::combine_ef(chal_coords.clone());
        let eval_val = E::combine_ef(eval_coords.clone());
        let eval_val_prev = E::combine_ef(eval_prev_coords);

        // Guard row 0 (`is_first == 1`) from wrapping around to row `N - 1`.
        if self.keep(outer_ids::A01) {
            eval.add_constraint(is_first.clone() * sel_chain_from_prev.clone());
        }
        if self.keep(outer_ids::A02) {
            eval.add_constraint(is_first.clone() * sel_acc_chain.clone());
        }
        if self.keep(outer_ids::A03) {
            eval.add_constraint(is_first * sel_carry_claim.clone());
        }

        // Algebraic Horner & Bilinear Multiplication intermediates.
        if self.keep(outer_ids::A04) {
            eval.add_constraint(q_aux.clone() - (w3.clone() * chal.clone() + w2.clone()));
        }
        if self.keep(outer_ids::A05) {
            eval.add_constraint(h_aux.clone() - (q_aux * chal.clone() + w1.clone()));
        }
        if self.keep(outer_ids::A06) {
            eval.add_constraint(horner_val.clone() - (h_aux * chal + w0.clone()));
        }
        if self.keep(outer_ids::A07) {
            eval.add_constraint(prod_ab.clone() - w0.clone() * w1.clone());
        }
        if self.keep(outer_ids::A08) {
            eval.add_constraint(acc_in.clone() - eval_val_prev.clone() * sel_acc_chain);
        }

        // Sumcheck round gate (`sel_sc == 1`): `claim == p(0) + p(1)` and `eval == p(chal)`.
        if self.keep(outer_ids::A09) {
            eval.add_constraint(w3.clone() * sel_deg2);
        }
        let poly_sum_01 = w0.clone() + w0.clone() + w1.clone() + w2 + w3;
        if self.keep(outer_ids::A10) {
            eval.add_constraint((claim_val.clone() - poly_sum_01) * sel_sc.clone());
        }
        if self.keep(outer_ids::A11) {
            eval.add_constraint((eval_val.clone() - horner_val) * sel_sc.clone());
        }

        // Bilinear accumulator gate (`sel_mul == 1`): `eval == acc_in + w0*w1 + sel_add_k*k_static`.
        let add_term = k_static.clone() * sel_add_k_static;
        if self.keep(outer_ids::A12) {
            eval.add_constraint((eval_val.clone() - (acc_in + prod_ab + add_term)) * sel_mul.clone());
        }
        let sel_inactive = one.clone() - sel_sc - sel_mul;
        if self.keep(outer_ids::A13) {
            eval.add_constraint(eval_val.clone() * sel_inactive);
        }

        // Inter-row chaining & static constant pinning.
        if self.keep(outer_ids::A14) {
            eval.add_constraint((claim_val.clone() - eval_val_prev) * sel_chain_from_prev);
        }
        if self.keep(outer_ids::A15) {
            eval.add_constraint((claim_val.clone() - claim_val_prev) * sel_carry_claim);
        }
        if self.keep(outer_ids::A16) {
            eval.add_constraint((eval_val - claim_val.clone()) * sel_eq_claim);
        }
        if self.keep(outer_ids::A17) {
            eval.add_constraint((w0 - k_static.clone()) * sel_static_w0);
        }
        if self.keep(outer_ids::A18) {
            eval.add_constraint((w1 - k_static.clone()) * sel_static_w1);
        }
        if self.keep(outer_ids::A19) {
            eval.add_constraint((claim_val - k_static) * sel_static_claim);
        }

        // Base-field (`M31`) & Boolean (`{0, 1}`) projections on `w0`.
        let sel_base_any = sel_base_w0 + sel_bool_w0.clone();
        if self.keep(outer_ids::A20) {
            eval.add_constraint(w0_coords[1].clone() * sel_base_any.clone());
            eval.add_constraint(w0_coords[2].clone() * sel_base_any.clone());
            eval.add_constraint(w0_coords[3].clone() * sel_base_any);
        }
        if self.keep(outer_ids::A21) {
            eval.add_constraint(
                sel_bool_w0 * w0_coords[0].clone() * (w0_coords[0].clone() - one),
            );
        }

        // 4 shared `MemoryBusElements` ports (`w0`, `w1`, `chal`, `eval_val`).
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(mem_mult_w0),
            &[
                mem_id_w0,
                w0_coords[0].clone(),
                w0_coords[1].clone(),
                w0_coords[2].clone(),
                w0_coords[3].clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(mem_mult_w1),
            &[
                mem_id_w1,
                w1_coords[0].clone(),
                w1_coords[1].clone(),
                w1_coords[2].clone(),
                w1_coords[3].clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(mem_mult_chal),
            &[
                mem_id_chal,
                chal_coords[0].clone(),
                chal_coords[1].clone(),
                chal_coords[2].clone(),
                chal_coords[3].clone(),
            ],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.mem_bus_elements,
            E::EF::from(mem_mult_eval),
            &[
                mem_id_eval,
                eval_coords[0].clone(),
                eval_coords[1].clone(),
                eval_coords[2].clone(),
                eval_coords[3].clone(),
            ],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

pub fn standalone_audit_info(log_size: u32) -> InfoEvaluator {
    let comp = StandaloneAuditAirEval {
        log_size,
        mem_bus_elements: MemoryBusElements::dummy(),
        claimed_sum: SecureField::zero(),
        #[cfg(test)]
        skip: crate::blake2s_air::outer_skip::current(),
    };
    comp.evaluate(InfoEvaluator::empty())
}

/// Preprocessed `Tree 0` column IDs and log-sizes in exact commitment order:
/// `[IsFirst(audit), AuditStatic[0..25], Xor9/8/7/4[0..12], BridgeStatic[0..29]]` (`67` columns).
pub fn standalone_preprocessed_ids_and_log_sizes(
    audit_log_size: u32,
    bridge_log_size: u32,
) -> (Vec<PreProcessedColumnId>, Vec<u32>) {
    let mut ids = Vec::with_capacity(67);
    let mut sizes = Vec::with_capacity(67);
    ids.push(standalone_is_first_id(audit_log_size));
    sizes.push(audit_log_size);
    for id in standalone_audit_static_ids(audit_log_size) {
        ids.push(id);
        sizes.push(audit_log_size);
    }
    let xor_ids = preprocessed_blake2s_xor_ids();
    let xor_sizes = preprocessed_blake2s_xor_log_sizes();
    for (id, sz) in xor_ids.into_iter().zip(xor_sizes.into_iter()) {
        ids.push(id);
        sizes.push(sz);
    }
    for id in blake2s_bridge_static_ids(bridge_log_size) {
        ids.push(id);
        sizes.push(bridge_log_size);
    }
    (ids, sizes)
}

/// Pack a 32-byte digest into four `[BaseField; 4]` 16-bit LE limb tuples (`< 65536 < P`).
pub fn bytes_32_to_digest_limbs(bytes: &[u8; 32]) -> [[BaseField; 4]; 4] {
    core::array::from_fn(|q| {
        core::array::from_fn(|j| {
            let idx = 8 * q + 2 * j;
            BaseField::from_u32_unchecked(u32::from(bytes[idx]) + 256 * u32::from(bytes[idx + 1]))
        })
    })
}

/// Canonical self-contained chain link `(layer, prev_receipt_digest)` matching `StageAChainLink`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandaloneChainLink {
    pub layer: u32,
    pub prev_receipt_digest: [u8; 32],
}

/// Per-component LogUp claimed sums for the 4-tree standalone Circle-STARK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandaloneClaimedSums {
    pub audit_sum: SecureField,
    pub sched_sum: SecureField,
    pub round_sum: SecureField,
    pub bridge_sum: SecureField,
    pub xor9_sum: SecureField,
    pub xor8_sum: SecureField,
    pub xor7_sum: SecureField,
    pub xor4_sum: SecureField,
}

impl StandaloneClaimedSums {
    pub fn as_array(&self) -> [SecureField; 8] {
        [
            self.audit_sum,
            self.sched_sum,
            self.round_sum,
            self.bridge_sum,
            self.xor9_sum,
            self.xor8_sum,
            self.xor7_sum,
            self.xor4_sum,
        ]
    }

    pub fn from_array(arr: [SecureField; 8]) -> Self {
        Self {
            audit_sum: arr[0],
            sched_sum: arr[1],
            round_sum: arr[2],
            bridge_sum: arr[3],
            xor9_sum: arr[4],
            xor8_sum: arr[5],
            xor7_sum: arr[6],
            xor4_sum: arr[7],
        }
    }

    pub fn total(&self) -> SecureField {
        self.as_array()
            .into_iter()
            .fold(SecureField::zero(), |acc, s| acc + s)
    }

    pub fn validate_canonical(&self) -> Result<(), String> {
        for (i, s) in self.as_array().iter().enumerate() {
            for c in s.to_m31_array() {
                if c.0 >= P {
                    return Err(format!("non-canonical StandaloneClaimedSums[{i}]"));
                }
            }
        }
        Ok(())
    }
}

fn validate_log_sizes(
    audit_log_size: u32,
    sched_log_size: u32,
    round_log_size: u32,
    bridge_log_size: u32,
) -> Result<(), String> {
    for (name, sz) in [
        ("audit_log_size", audit_log_size),
        ("sched_log_size", sched_log_size),
        ("round_log_size", round_log_size),
        ("bridge_log_size", bridge_log_size),
    ] {
        if !(MIN_STANDALONE_AUDIT_LOG_SIZE..=MAX_STANDALONE_AUDIT_LOG_SIZE).contains(&sz) {
            return Err(format!(
                "standalone {name}={sz} outside {}..={}",
                MIN_STANDALONE_AUDIT_LOG_SIZE, MAX_STANDALONE_AUDIT_LOG_SIZE
            ));
        }
    }
    validate_outer_lookup_counts(sched_log_size, round_log_size, bridge_log_size)
}

/// Every outer XOR relation must receive fewer than `p` lookups: past `p`, `p` copies of a false entry
/// cancel in LogUp. The log-size limits alone admit `2^24 x (80 + 100 + 24)` lookups into `Xor8`, above
/// `p`. Per-row counts: `blake2s_air.rs`.
fn validate_outer_lookup_counts(
    sched_log_size: u32,
    round_log_size: u32,
    bridge_log_size: u32,
) -> Result<(), String> {
    let (s, r, b) = (1u64 << sched_log_size, 1u64 << round_log_size, 1u64 << bridge_log_size);
    for (relation, lookups) in [
        ("Xor8", ROUND_XOR8_PER_ROW * r + SCHED_XOR8_PER_ROW * s + BRIDGE_XOR8_PER_ROW_MAX * b),
        ("Xor7", ROUND_XOR7_PER_ROW * r + BRIDGE_XOR7_PER_ROW_MAX * b),
        ("Xor9", ROUND_XOR9_PER_ROW * r),
        ("Xor4", ROUND_XOR4_PER_ROW * r),
    ] {
        if lookups >= u64::from(P) {
            return Err(format!(
                "standalone outer relation {relation} would receive {lookups} lookups (sched_log_size={sched_log_size}, round_log_size={round_log_size}, bridge_log_size={bridge_log_size}), not below p = {P}"
            ));
        }
    }
    Ok(())
}

/// Shape of a Phase 2c transformer block.
///
/// `validate` is the shape check of both registration and the verifier. Each bound keeps one
/// accumulator of the wide activations or of
/// the floating norm inside the field window.
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformerBlockShape {
    pub batch: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub dense_ffn_dim: usize,
    pub moe_ffn_dim: usize,
    pub num_experts: usize,
    pub stacked_expert_pack: usize,
}

/// Largest batch: the softmax relation `1024 u = P Z + r` stays below `2^30` only while
/// `Z <= 1023 B < 2^20` (design 3.1).
pub const MAX_TRANSFORMER_BATCH: usize = 1024;
/// Largest `d_model`: the floating norm's sums and `|h w| < 2^29.51` are derived for `D <= 2^13`,
/// and every projection of the norm output is bounded by `127 sqrt(D) 32767 < 2^28.5`.
pub const MAX_TRANSFORMER_D_MODEL: usize = 8192;
/// Largest `head_dim`: the score accumulator `|q . k| <= head_dim * 2048^2` must stay `<= 2^29`
/// (design 3.1).
pub const MAX_TRANSFORMER_HEAD_DIM: usize = 128;
/// Largest row of a two-limb contraction (`W_down`): `255 * 127 * K < 2^29` (design 3.4, end).
pub const MAX_TRANSFORMER_FFN_DIM: usize = 16384;
/// Largest expert count. The one-hot check `sum_e sel_e = 1` is an equation over the integers
/// only while `E < p`; this cap matches the cap on `stacked_expert_pack`.
pub const MAX_TRANSFORMER_EXPERTS: usize = 1 << 16;

impl TransformerBlockShape {
    pub fn validate(&self) -> Result<(), String> {
        let pow2_in = |v: usize, lo: usize, hi: usize| v.is_power_of_two() && (lo..=hi).contains(&v);
        if !pow2_in(self.batch, 2, MAX_TRANSFORMER_BATCH) {
            return Err(format!(
                "TransformerBlockShape batch {} must be a power of two in 2..={MAX_TRANSFORMER_BATCH} (softmax Z <= 1023 * B < 2^20)",
                self.batch
            ));
        }
        if !pow2_in(self.d_model, 4, MAX_TRANSFORMER_D_MODEL) {
            return Err(format!(
                "TransformerBlockShape d_model {} must be a power of two in 4..={MAX_TRANSFORMER_D_MODEL} (non-wrapping floating norm)",
                self.d_model
            ));
        }
        if !pow2_in(self.num_heads, 1, self.d_model)
            || !pow2_in(self.num_kv_heads, 1, self.num_heads)
        {
            return Err("TransformerBlockShape heads/kv_heads must be powers of two with kv_heads <= heads <= d_model".into());
        }
        let head_dim = self.head_dim();
        if !pow2_in(head_dim, 2, MAX_TRANSFORMER_HEAD_DIM) {
            return Err(format!(
                "TransformerBlockShape head_dim {head_dim} = d_model / num_heads must be a power of two in 2..={MAX_TRANSFORMER_HEAD_DIM} (score |q . k| <= head_dim * 2048^2 <= 2^29)"
            ));
        }
        for (name, k) in [("dense_ffn_dim", self.dense_ffn_dim), ("moe_ffn_dim", self.moe_ffn_dim)] {
            if !pow2_in(k, 4, MAX_TRANSFORMER_FFN_DIM) {
                return Err(format!(
                    "TransformerBlockShape {name} {k} must be a power of two in 4..={MAX_TRANSFORMER_FFN_DIM} (W_down row length; acc_lo <= 255 * 127 * K < 2^29)"
                ));
            }
        }
        if !pow2_in(self.num_experts, 1, MAX_TRANSFORMER_EXPERTS)
            || !pow2_in(self.stacked_expert_pack, 1, self.num_experts)
        {
            return Err(format!(
                "TransformerBlockShape num_experts {} and stacked_expert_pack {} must be powers of two with pack <= experts <= {MAX_TRANSFORMER_EXPERTS}",
                self.num_experts, self.stacked_expert_pack
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// The fields a block statement carries besides `batch`, `d_model` and `stacked_expert_pack`.
    pub fn statement_dims(&self) -> StandaloneTransformerDims {
        StandaloneTransformerDims {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            dense_ffn_dim: self.dense_ffn_dim,
            moe_ffn_dim: self.moe_ffn_dim,
            num_experts: self.num_experts,
        }
    }
}

/// Heads, widths and experts of a transformer block statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandaloneTransformerDims {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub dense_ffn_dim: usize,
    pub moe_ffn_dim: usize,
    pub num_experts: usize,
}

/// Output of a weight site with per-row exponents (`C8R_rowpow2_all`, `STWOTFM1` version 5,
/// the exporter fixture format): a `b`-bit saturating output or the
/// 24-bit two-limb wide output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSiteOutput {
    Saturating(u32),
    Wide,
}

/// Smallest row shift of a wide row in case A (`acc_top` requantised at `s_j - 8`); below it the
/// row is in case B.
pub const WIDE_ROW_CASE_A_MIN_SHIFT: u32 = 9;

impl RowSiteOutput {
    /// Largest row shift `s_j = s + e_j`: `30 - b` at a `b`-bit site (`b + s_j <= 30`, design
    /// 3.4) and 20 at a wide site (`R_j >= 2`, design 3.5).
    pub fn shift_max(self) -> u32 {
        match self {
            RowSiteOutput::Saturating(b) => 30 - b,
            RowSiteOutput::Wide => 20,
        }
    }
}

/// Top-limb cap `R_j = min(128, 2^(21 - s_j))` of a case A wide row: `u = y + 2^23 + 2^15` has
/// `floor(u / 2^16)` in `[128 - R_j, 128 + R_j)`, which keeps `|y 2^(s_j - 8)| < 2^29`. Case B rows
/// (`s_j <= 8`) take the full 24-bit range, `R_j = 128`.
pub fn wide_row_top_cap(row_shift: u32) -> u32 {
    if row_shift < WIDE_ROW_CASE_A_MIN_SHIFT {
        128
    } else {
        128u32.min(1 << (21 - row_shift))
    }
}

/// Registration's check of one exponent vector: one `e_j` per row, every row shift `s + e_j` in
/// `[1, shift_max]`, and `e_j = 0` on every row `zero_row` names (the rows the fixture layout pads
/// with zeros). The exponents are model data bound by the weight root; the circuit derives every
/// per-row constant from them at registration, so these bounds are the only check they get.
pub fn validate_row_exponents(
    name: &str,
    out: RowSiteOutput,
    shift: u32,
    exps: &[u8],
    zero_row: &dyn Fn(usize) -> bool,
) -> Result<(), String> {
    let max = out.shift_max();
    if !(1..=max).contains(&shift) {
        return Err(format!("{name}: base shift {shift} outside [1, {max}]"));
    }
    for (j, &e) in exps.iter().enumerate() {
        if shift + u32::from(e) > max {
            return Err(format!("{name}: row {j} exponent {e} puts its shift {} above {max}", shift + u32::from(e)));
        }
        if e != 0 && zero_row(j) {
            return Err(format!("{name}: padded row {j} has exponent {e}, must be 0"));
        }
    }
    Ok(())
}

/// Flag of the embedding stage in [`StandaloneEndsDecl::flags`].
pub const ENDS_FLAG_EMBED: u8 = 1;
/// Flag of the head stage in [`StandaloneEndsDecl::flags`].
pub const ENDS_FLAG_HEAD: u8 = 2;
/// Flag of multi-token causal autoregressive decoding in [`StandaloneEndsDecl::flags`].
pub const ENDS_FLAG_MULTI: u8 = 4;
/// Largest vocabulary of an end stage: the token bits stay within `decompose_u32_bits` and the head's
/// lookups far below `p`. Maverick needs `2^18` after padding.
pub const MAX_ENDS_VOCAB: usize = 1 << 20;
/// Largest `batch * V_pad` of a multi-token head stage: keeps `[B, V_pad]` witness columns within CPRS
/// `m <= 28` and head range lookups `8 * B * V_pad <= 2^29 < p` (`32 * 2^18 = 2^23` on Maverick).
pub const MAX_MULTI_HEAD_BV: u64 = 1 << 26;

/// The end stages of a transformer block.
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandaloneEndsDecl {
    pub embed: bool,
    pub head: bool,
    /// The vocabulary `V` (raw, before padding to a power of two).
    pub vocab_size: usize,
    /// `true` for dynamic `(L, K)` multi-token causal decoding (`flags` bit `ENDS_FLAG_MULTI = 4`).
    pub multi_token: bool,
}

impl StandaloneEndsDecl {
    pub fn flags(&self) -> u8 {
        (if self.embed { ENDS_FLAG_EMBED } else { 0 })
            | (if self.head { ENDS_FLAG_HEAD } else { 0 })
            | (if self.multi_token { ENDS_FLAG_MULTI } else { 0 })
    }

    pub fn from_flags(flags: u8, vocab_size: usize) -> Result<Self, String> {
        let base = flags & (ENDS_FLAG_EMBED | ENDS_FLAG_HEAD);
        if base == 0 || flags > (ENDS_FLAG_EMBED | ENDS_FLAG_HEAD | ENDS_FLAG_MULTI) {
            return Err(format!("non-canonical end-stage flags {flags}"));
        }
        Ok(Self {
            embed: flags & ENDS_FLAG_EMBED != 0,
            head: flags & ENDS_FLAG_HEAD != 0,
            vocab_size,
            multi_token: flags & ENDS_FLAG_MULTI != 0,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.embed && !self.head {
            return Err("an end-stage declaration needs the embedding, the head or both".into());
        }
        if !(2..=MAX_ENDS_VOCAB).contains(&self.vocab_size) {
            return Err(format!(
                "end-stage vocab {} outside [2, {MAX_ENDS_VOCAB}]",
                self.vocab_size
            ));
        }
        Ok(())
    }

    /// `V_pad`, the power of two the tables and the logit columns are padded to.
    pub fn vocab_pad(&self) -> usize {
        self.vocab_size.next_power_of_two()
    }

    /// `log2(V_pad)`: the bits of a token id in the AIR.
    pub fn log_vocab(&self) -> usize {
        self.vocab_pad().ilog2() as usize
    }
}

/// Block kind byte, then for a transformer its five dimensions as `u64` little-endian: kind 1
/// without an end stage, kind 2 with one, followed by the flags byte and the vocabulary (`u64`).
fn push_transformer_dims(
    out: &mut Vec<u8>,
    dims: &Option<StandaloneTransformerDims>,
    ends: &Option<StandaloneEndsDecl>,
) {
    match dims {
        None => out.push(0),
        Some(t) => {
            out.push(if ends.is_some() { 2 } else { 1 });
            for v in [t.num_heads, t.num_kv_heads, t.dense_ffn_dim, t.moe_ffn_dim, t.num_experts] {
                out.extend_from_slice(&(v as u64).to_le_bytes());
            }
            if let Some(e) = ends {
                out.push(e.flags());
                out.extend_from_slice(&(e.vocab_size as u64).to_le_bytes());
            }
        }
    }
}

/// Statement for `STWOREC2` standalone single-block receipts (`verification_model = "standalone_anchor_only"`).
///
/// The verifier checks the shape the statement declares; the anchor binds that declaration to
/// the registered circuit (`Tree 0`), which the verifier does not rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneBlockStatement {
    pub mode_tag: u32,
    pub profile_id: String,
    pub batch: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub cprs_weight_f: u8,
    pub cprs_witness_f: u8,
    pub stacked_expert_pack: u32,
    pub zk_trace_blinded: bool,
    pub audit_log_size: u32,
    pub sched_log_size: u32,
    pub round_log_size: u32,
    pub bridge_log_size: u32,
    /// `Some` for a transformer block (`batch`, `d_model` and `stacked_expert_pack` are the fields
    /// above); `None` for the two-matrix block of `standalone_model_prover`.
    pub transformer: Option<StandaloneTransformerDims>,
    /// `Some` for a transformer block with an end stage (kind byte 2); only with `transformer`.
    pub ends: Option<StandaloneEndsDecl>,
    pub weights_and_boundaries_digest: [u8; 32],
    pub initial_chain: StandaloneChainLink,
    pub final_chain: StandaloneChainLink,
    pub public_io_digest: [u8; 32],
}

impl StandaloneBlockStatement {
    /// The declared transformer shape, if this is a transformer block.
    pub fn transformer_shape(&self) -> Option<TransformerBlockShape> {
        self.transformer.map(|t| TransformerBlockShape {
            batch: self.batch,
            d_model: self.d_model,
            num_heads: t.num_heads,
            num_kv_heads: t.num_kv_heads,
            dense_ffn_dim: t.dense_ffn_dim,
            moe_ffn_dim: t.moe_ffn_dim,
            num_experts: t.num_experts,
            stacked_expert_pack: self.stacked_expert_pack as usize,
        })
    }

    pub fn validate(&self) -> Result<PcsConfig, String> {
        // Mode 12 (experimental: the block's lookups proved by LogUp-GKR) exists only for
        // transformer blocks. Its circuit differs from mode 11's, so its Tree 0 root, and with
        // it the anchor, differ too.
        let gkr_block = self.mode_tag == STANDALONE_MODE_LOGUP_GKR && self.transformer.is_some();
        if !(9..=11).contains(&self.mode_tag) && !gkr_block {
            return Err(format!("unsupported standalone mode_tag {}", self.mode_tag));
        }
        let cfg = standalone_pcs_config_for_profile_id(&self.profile_id)?;
        // `production_hardened` in mode 11 or 12 is registered with a blinded trace (the model runner sets
        // `zk_trace_blinded = mode_tag >= 11`); an unblinded statement is refused.
        if self.profile_id == PRODUCTION_HARDENED_PROFILE_ID && self.mode_tag >= 11 && !self.zk_trace_blinded {
            return Err(format!(
                "profile {PRODUCTION_HARDENED_PROFILE_ID} in mode {} requires zk_trace_blinded = true",
                self.mode_tag
            ));
        }
        if self.batch == 0 || !self.batch.is_power_of_two() || self.batch > 2048 {
            return Err("standalone batch must be a positive power of two <= 2048".into());
        }
        if self.d_model < 4 || !self.d_model.is_power_of_two() || self.d_model > (1 << 20) {
            return Err("standalone d_model must be a power of two in 4..=2^20".into());
        }
        if let Some(ends) = self.ends {
            if self.transformer.is_none() {
                return Err("an end stage needs a transformer block statement".into());
            }
            ends.validate()?;
            if ends.multi_token && ends.head {
                let bv = (self.batch as u64).saturating_mul(ends.vocab_pad() as u64);
                if bv > MAX_MULTI_HEAD_BV {
                    return Err(format!(
                        "multi-token head batch * vocab_pad = {bv} exceeds MAX_MULTI_HEAD_BV ({MAX_MULTI_HEAD_BV})"
                    ));
                }
            }
        }
        let min_layers = usize::from(self.ends.is_none());
        if self.num_layers < min_layers || self.num_layers > MAX_STANDALONE_BLOCK_LAYERS {
            return Err(format!(
                "standalone num_layers {} outside {min_layers}..={MAX_STANDALONE_BLOCK_LAYERS}",
                self.num_layers
            ));
        }
        if !(1..=8).contains(&self.cprs_weight_f) || !(1..=8).contains(&self.cprs_witness_f) {
            return Err("standalone CPRS leaf arity out of range".into());
        }
        if self.stacked_expert_pack == 0
            || !self.stacked_expert_pack.is_power_of_two()
            || self.stacked_expert_pack > (1 << 16)
        {
            return Err("standalone stacked_expert_pack must be a power of two in 1..=65536".into());
        }
        if let Some(shape) = self.transformer_shape() {
            shape.validate()?;
        } else if self.profile_id == PRODUCTION_HARDENED_PROFILE_ID {
            return Err(format!("profile {PRODUCTION_HARDENED_PROFILE_ID} is defined for transformer blocks only"));
        }
        let expected_final_layer = self
            .initial_chain
            .layer
            .checked_add(self.num_layers as u32)
            .ok_or_else(|| "standalone block chain layer u32 overflow".to_string())?;
        if self.initial_chain.layer >= P
            || self.final_chain.layer >= P
            || self.final_chain.layer != expected_final_layer
        {
            return Err(format!(
                "standalone block chain layer out of canonical M31 range or != initial + num_layers (initial={}, num_layers={}, final={})",
                self.initial_chain.layer, self.num_layers, self.final_chain.layer
            ));
        }
        if is_pos2_profile(&self.profile_id) {
            self.validate_pos2()?;
        } else {
            validate_log_sizes(
                self.audit_log_size,
                self.sched_log_size,
                self.round_log_size,
                self.bridge_log_size,
            )?;
        }
        Ok(cfg)
    }

    /// `exp_pos2_*` statements: a transformer block, `bridge_log_size` is the height of the
    /// Poseidon2 component and there are no Blake2s components (`sched_log_size = round_log_size =
    /// 0`); every digest is 8 canonical little-endian words (`< p`), as the circuit exposes them.
    fn validate_pos2(&self) -> Result<(), String> {
        if self.transformer.is_none() {
            return Err("exp_pos2 profiles exist only for the transformer block".into());
        }
        // Registration refuses end stages under exp_pos2 (`register_with_ends`); so does the statement,
        // so a receipt that carries them is refused.
        if self.ends.is_some() {
            return Err("exp_pos2 statement must not declare an end stage".into());
        }
        if self.sched_log_size != 0 || self.round_log_size != 0 {
            return Err("exp_pos2 statement must have sched_log_size = round_log_size = 0".into());
        }
        for (name, sz) in [("audit_log_size", self.audit_log_size), ("pos2_log_size", self.bridge_log_size)] {
            if !(MIN_STANDALONE_AUDIT_LOG_SIZE..=MAX_STANDALONE_AUDIT_LOG_SIZE).contains(&sz) {
                return Err(format!("exp_pos2 {name}={sz} out of range"));
            }
        }
        for (name, d) in [
            ("weights_and_boundaries_digest", &self.weights_and_boundaries_digest),
            ("initial_chain digest", &self.initial_chain.prev_receipt_digest),
            ("final_chain digest", &self.final_chain.prev_receipt_digest),
            ("public_io_digest", &self.public_io_digest),
        ] {
            if crate::poseidon2_m31::digest_from_bytes(d).is_none() {
                return Err(format!("exp_pos2 {name} has a word >= p"));
            }
        }
        Ok(())
    }

    /// Static registration policy bytes (100% prompt- and invocation-independent).
    pub fn policy_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(72 + self.profile_id.len());
        out.extend_from_slice(&STANDALONE_AUDIT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.mode_tag.to_le_bytes());
        out.extend_from_slice(&(self.profile_id.len() as u64).to_le_bytes());
        out.extend_from_slice(self.profile_id.as_bytes());
        out.extend_from_slice(&(self.batch as u64).to_le_bytes());
        out.extend_from_slice(&(self.d_model as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_layers as u64).to_le_bytes());
        out.push(self.cprs_weight_f);
        out.push(self.cprs_witness_f);
        out.push(u8::from(self.zk_trace_blinded));
        out.extend_from_slice(&self.stacked_expert_pack.to_le_bytes());
        out.extend_from_slice(&self.audit_log_size.to_le_bytes());
        out.extend_from_slice(&self.sched_log_size.to_le_bytes());
        out.extend_from_slice(&self.round_log_size.to_le_bytes());
        out.extend_from_slice(&self.bridge_log_size.to_le_bytes());
        push_transformer_dims(&mut out, &self.transformer, &self.ends);
        out
    }
}

/// Statement for `STWOMDL1` whole-model folded + `Embedding` + `LM Head` standalone receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WholeModelStatement {
    pub mode_tag: u32,
    pub profile_id: String,
    pub batch: usize,
    pub num_blocks: usize,
    pub layers_per_block: usize,
    pub vocab_size: usize,
    pub d_model: usize,
    pub cprs_weight_f: u8,
    pub cprs_witness_f: u8,
    pub stacked_expert_pack: u32,
    pub zk_trace_blinded: bool,
    pub tokens_committed: bool,
    pub audit_log_size: u32,
    pub sched_log_size: u32,
    pub round_log_size: u32,
    pub bridge_log_size: u32,
    pub weights_and_boundaries_digest: [u8; 32],
    pub prompt_token_ids: Vec<u32>,
    pub emitted_token_ids: Vec<u32>,
    pub prompt_commitment: [u8; 32],
    pub emitted_commitment: [u8; 32],
}

impl WholeModelStatement {
    pub fn validate(&self) -> Result<PcsConfig, String> {
        if self.mode_tag != 9 && self.mode_tag != 10 && self.mode_tag != 11 {
            return Err(format!("unsupported whole-model mode_tag {}", self.mode_tag));
        }
        if self.profile_id == PRODUCTION_HARDENED_PROFILE_ID {
            return Err(format!("profile {PRODUCTION_HARDENED_PROFILE_ID} is defined for STWOREC2 transformer blocks only"));
        }
        if is_pos2_profile(&self.profile_id) {
            return Err("exp_pos2 profiles exist only for the transformer block".into());
        }
        let cfg = standalone_pcs_config_for_profile_id(&self.profile_id)?;
        if self.batch == 0 || !self.batch.is_power_of_two() || self.batch > (1 << 20) {
            return Err("whole-model batch must be a positive power of two <= 2^20".into());
        }
        if self.num_blocks == 0 || self.num_blocks > 64 {
            return Err("whole-model num_blocks out of range".into());
        }
        if self.layers_per_block == 0 || self.layers_per_block > 256 {
            return Err("whole-model layers_per_block out of range".into());
        }
        if self.vocab_size < 2 || !self.vocab_size.is_power_of_two() || self.vocab_size as u64 >= u64::from(P) {
            return Err("whole-model vocab_size must be a power of two in 2..P".into());
        }
        if self.d_model < 4 || !self.d_model.is_power_of_two() || self.d_model > (1 << 20) {
            return Err("whole-model d_model must be a power of two in 4..=2^20".into());
        }
        if !(1..=8).contains(&self.cprs_weight_f) || !(1..=8).contains(&self.cprs_witness_f) {
            return Err("whole-model CPRS leaf arity out of range".into());
        }
        if self.stacked_expert_pack == 0
            || !self.stacked_expert_pack.is_power_of_two()
            || self.stacked_expert_pack > (1 << 16)
        {
            return Err("whole-model stacked_expert_pack must be a power of two in 1..=65536".into());
        }
        if self.tokens_committed {
            if !self.zk_trace_blinded || self.mode_tag < 11 {
                return Err(
                    "committed-tokens WholeModelStatement requires Mode 11 with zk_trace_blinded=true".into(),
                );
            }
            if !self.prompt_token_ids.is_empty() || !self.emitted_token_ids.is_empty() {
                return Err("committed-tokens WholeModelStatement must not carry plaintext token IDs".into());
            }
            if self.prompt_commitment == [0u8; 32] || self.emitted_commitment == [0u8; 32] {
                return Err("committed-tokens WholeModelStatement must carry non-zero 32B commitments".into());
            }
        } else {
            if self.prompt_token_ids.len() != self.batch || self.emitted_token_ids.len() != self.batch {
                return Err("whole-model token ID slices must have length == batch".into());
            }
            for &tok in self.prompt_token_ids.iter().chain(self.emitted_token_ids.iter()) {
                if (tok as usize) >= self.vocab_size {
                    return Err(format!("whole-model token ID {tok} >= vocab_size {}", self.vocab_size));
                }
            }
            if self.prompt_commitment != [0u8; 32] || self.emitted_commitment != [0u8; 32] {
                return Err("public-tokens WholeModelStatement must have zero commitment fields".into());
            }
        }
        validate_log_sizes(
            self.audit_log_size,
            self.sched_log_size,
            self.round_log_size,
            self.bridge_log_size,
        )?;
        Ok(cfg)
    }

    /// Static registration policy bytes (excludes prompt/emitted tokens or per-invocation salted
    /// commitments so `full_stack_anchor` is 100% static across all model invocations).
    pub fn policy_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96 + self.profile_id.len());
        out.extend_from_slice(&STANDALONE_AUDIT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.mode_tag.to_le_bytes());
        out.extend_from_slice(&(self.profile_id.len() as u64).to_le_bytes());
        out.extend_from_slice(self.profile_id.as_bytes());
        out.extend_from_slice(&(self.batch as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_blocks as u64).to_le_bytes());
        out.extend_from_slice(&(self.layers_per_block as u64).to_le_bytes());
        out.extend_from_slice(&(self.vocab_size as u64).to_le_bytes());
        out.extend_from_slice(&(self.d_model as u64).to_le_bytes());
        out.push(self.cprs_weight_f);
        out.push(self.cprs_witness_f);
        out.push(u8::from(self.zk_trace_blinded));
        out.push(u8::from(self.tokens_committed));
        out.extend_from_slice(&self.stacked_expert_pack.to_le_bytes());
        out.extend_from_slice(&self.audit_log_size.to_le_bytes());
        out.extend_from_slice(&self.sched_log_size.to_le_bytes());
        out.extend_from_slice(&self.round_log_size.to_le_bytes());
        out.extend_from_slice(&self.bridge_log_size.to_le_bytes());
        out
    }
}

/// Compute the standalone `32 B` block anchor (`STWOREC2`), binding static policy, all static
/// CPRS weight/attention commitments (`weights_and_boundaries_digest`), and `Tree 0` root.
pub fn compute_standalone_block_anchor(
    policy_bytes: &[u8],
    weights_and_boundaries_digest: &[u8; 32],
    tree_0_root: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(STANDALONE_BLOCK_ANCHOR_DOMAIN);
    h.update((policy_bytes.len() as u64).to_le_bytes());
    h.update(policy_bytes);
    h.update(weights_and_boundaries_digest);
    h.update(tree_0_root);
    h.finalize().into()
}

/// Compute the standalone `32 B` whole-model anchor (`STWOMDL1`), binding static policy, all
/// static block + `Embedding` + `LM Head` CPRS commitments, and `Tree 0` root.
pub fn compute_whole_model_full_stack_anchor(
    policy_bytes: &[u8],
    weights_and_boundaries_digest: &[u8; 32],
    tree_0_root: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(WHOLE_MODEL_ANCHOR_DOMAIN);
    h.update((policy_bytes.len() as u64).to_le_bytes());
    h.update(policy_bytes);
    h.update(weights_and_boundaries_digest);
    h.update(tree_0_root);
    h.finalize().into()
}

#[derive(Clone, Debug)]
pub struct StandaloneBlockStarkProof {
    pub statement: StandaloneBlockStatement,
    pub claimed_sums: StandaloneClaimedSums,
    pub stark: StarkProof<Blake2sMerkleHasher>,
    /// Nonce of the proof of work before the outer lookup draws; zero, and absent from the wire,
    /// when the profile has none ([`standalone_lookup_pow_bits_for_profile_id`]).
    pub lookup_pow_nonce: u64,
}

#[derive(Clone, Debug)]
pub struct WholeModelStandaloneStarkProof {
    pub statement: WholeModelStatement,
    pub claimed_sums: StandaloneClaimedSums,
    pub stark: StarkProof<Blake2sMerkleHasher>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandaloneAuditVerifierRun {
    pub verify_ms: f64,
    pub receipt_bytes: usize,
}

fn validate_standalone_stark_encodings(
    proof: &StarkProof<Blake2sMerkleHasher>,
) -> Result<(), String> {
    for (t_i, tree) in proof.sampled_values.iter().enumerate() {
        for (c_i, col) in tree.iter().enumerate() {
            for (v_i, &val) in col.iter().enumerate() {
                for c in val.to_m31_array() {
                    if c.0 >= P {
                        return Err(format!("non-canonical sampled value {t_i}/{c_i}/{v_i}"));
                    }
                }
            }
        }
    }
    for (t_i, tree) in proof.queried_values.iter().enumerate() {
        for (c_i, col) in tree.iter().enumerate() {
            for (v_i, &val) in col.iter().enumerate() {
                if val.0 >= P {
                    return Err(format!("non-canonical queried value {t_i}/{c_i}/{v_i}"));
                }
            }
        }
    }
    for (i, &val) in proof.fri_proof.first_layer.fri_witness.iter().enumerate() {
        for c in val.to_m31_array() {
            if c.0 >= P {
                return Err(format!("non-canonical first FRI witness {i}"));
            }
        }
    }
    for (l_i, layer) in proof.fri_proof.inner_layers.iter().enumerate() {
        for (i, &val) in layer.fri_witness.iter().enumerate() {
            for c in val.to_m31_array() {
                if c.0 >= P {
                    return Err(format!("non-canonical inner FRI witness {l_i}/{i}"));
                }
            }
        }
    }
    for (i, &val) in proof.fri_proof.last_layer_poly.iter().enumerate() {
        for c in val.to_m31_array() {
            if c.0 >= P {
                return Err(format!("non-canonical last-layer coeff {i}"));
            }
        }
    }
    let last_coeffs: Vec<_> = proof.fri_proof.last_layer_poly.iter().copied().collect();
    if last_coeffs.is_empty()
        || !last_coeffs.len().is_power_of_two()
        || stwo::core::poly::line::LinePoly::new(last_coeffs) != proof.fri_proof.last_layer_poly
    {
        return Err("non-canonical FRI last_layer_poly log_size".into());
    }
    Ok(())
}

pub fn mix_bytes_into_channel(channel: &mut Blake2sChannel, tag: &[u8], payload: &[u8]) {
    let mut h = Blake2s256::new();
    h.update(tag);
    h.update((payload.len() as u64).to_le_bytes());
    h.update(payload);
    let dig: [u8; 32] = h.finalize().into();
    let words: [u32; 8] =
        core::array::from_fn(|i| u32::from_le_bytes(dig[4 * i..4 * i + 4].try_into().unwrap()));
    channel.mix_u32s(&words);
}

#[allow(clippy::too_many_arguments)]
fn verify_standalone_stark_core(
    expected_config: PcsConfig,
    lookup_pow: (u32, u64),
    audit_log_size: u32,
    sched_log_size: u32,
    round_log_size: u32,
    bridge_log_size: u32,
    public_memory_items: &[(u32, [BaseField; 4])],
    claimed_sums: &StandaloneClaimedSums,
    stark: &StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), String> {
    if stark.config != expected_config {
        return Err("standalone STARK PcsConfig mismatch".into());
    }
    if stark.commitments.len() != 4 {
        return Err(format!(
            "standalone STARK expected 4 tree commitments, got {}",
            stark.commitments.len()
        ));
    }
    claimed_sums.validate_canonical()?;
    validate_standalone_stark_encodings(stark)?;

    let (prep_ids, prep_log_sizes) =
        standalone_preprocessed_ids_and_log_sizes(audit_log_size, bridge_log_size);
    let mut scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(expected_config);
    scheme.commit(stark.commitments[0], &prep_log_sizes, channel);

    let mut comp_sizes = vec![
        standalone_audit_info(audit_log_size)
            .mask_offsets
            .as_cols_ref()
            .map_cols(|_| audit_log_size),
        blake2s_scheduler_info()
            .mask_offsets
            .as_cols_ref()
            .map_cols(|_| sched_log_size),
        blake2s_round_info()
            .mask_offsets
            .as_cols_ref()
            .map_cols(|_| round_log_size),
        blake2s_bridge_info(bridge_log_size)
            .mask_offsets
            .as_cols_ref()
            .map_cols(|_| bridge_log_size),
        blake2s_xor9_info().mask_offsets.as_cols_ref().map_cols(|_| 16),
        blake2s_xor8_info().mask_offsets.as_cols_ref().map_cols(|_| 16),
        blake2s_xor7_info().mask_offsets.as_cols_ref().map_cols(|_| 14),
        blake2s_xor4_info().mask_offsets.as_cols_ref().map_cols(|_| 8),
    ];
    let tree_sizes = TreeVec::concat_cols(comp_sizes.drain(..));
    scheme.commit(
        stark.commitments[1],
        &tree_sizes[ORIGINAL_TRACE_IDX],
        channel,
    );

    // One proof of work covers all six relation draws: they follow it in the same round.
    check_lookup_pow(channel, lookup_pow)?;
    let xor_elements = Blake2sXorElements::draw(channel);
    let round_elements = Blake2sRoundElements::draw(channel);
    let compress_elements = Blake2sCompressElements::draw(channel);
    let chain_elements = Blake2sChainElements::draw(channel);
    let byte_bus_elements = ByteBusElements::draw(channel);
    let mem_bus_elements = MemoryBusElements::draw(channel);

    scheme.commit(
        stark.commitments[2],
        &tree_sizes[INTERACTION_TRACE_IDX],
        channel,
    );
    channel.mix_felts(&claimed_sums.as_array());

    // Public inputs LogUp cancellation check: every `(id, coords)` exposed by the circuit with
    // multiplicity `+1` must cancel the verifier-evaluated `-1 / combine(id, coords)`.
    let mut public_memory_sum = SecureField::zero();
    for &(id, coords) in public_memory_items {
        let denom: SecureField = mem_bus_elements.combine(&[
            BaseField::from_u32_unchecked(id),
            coords[0],
            coords[1],
            coords[2],
            coords[3],
        ]);
        if denom.is_zero() {
            return Err("degenerate zero MemoryBus denominator on public input".into());
        }
        public_memory_sum -= SecureField::one() / denom;
    }
    if !(claimed_sums.total() + public_memory_sum).is_zero()
        && outer_keep(outer_thread_mask(), outer_ids::D16)
    {
        return Err("standalone STARK global + public MemoryBus LogUp sum non-zero".into());
    }

    let mut alloc = TraceLocationAllocator::new_with_preprocessed_columns(&prep_ids);
    let c_audit = StandaloneAuditComponent::new(
        &mut alloc,
        StandaloneAuditAirEval {
            log_size: audit_log_size,
            mem_bus_elements: mem_bus_elements.clone(),
            claimed_sum: claimed_sums.audit_sum,
            #[cfg(test)]
            skip: crate::blake2s_air::outer_skip::current(),
        },
        claimed_sums.audit_sum,
    );
    let c_sched = Blake2sSchedulerComponent::new(
        &mut alloc,
        Blake2sSchedulerEval {
            log_size: sched_log_size,
            xor_elements: xor_elements.clone(),
            round_elements: round_elements.clone(),
            compress_elements: compress_elements.clone(),
            claimed_sum: claimed_sums.sched_sum,
            #[cfg(test)]
            skip: crate::blake2s_air::outer_skip::current(),
        },
        claimed_sums.sched_sum,
    );
    let c_round = Blake2sRoundComponent::new(
        &mut alloc,
        Blake2sRoundEval {
            log_size: round_log_size,
            xor_elements: xor_elements.clone(),
            round_elements,
            claimed_sum: claimed_sums.round_sum,
            #[cfg(test)]
            skip: crate::blake2s_air::outer_skip::current(),
        },
        claimed_sums.round_sum,
    );
    let c_bridge = Blake2sMemoryBridgeComponent::new(
        &mut alloc,
        Blake2sMemoryBridgeEval {
            log_size: bridge_log_size,
            xor_elements: xor_elements.clone(),
            compress_elements,
            chain_elements,
            byte_bus_elements,
            mem_bus_elements,
            claimed_sum: claimed_sums.bridge_sum,
            #[cfg(test)]
            skip: crate::blake2s_air::outer_skip::current(),
        },
        claimed_sums.bridge_sum,
    );
    let c_x9 = Blake2sXor9Component::new(
        &mut alloc,
        Blake2sXor9Eval {
            lookup_elements: xor_elements.xor9,
            claimed_sum: claimed_sums.xor9_sum,
        },
        claimed_sums.xor9_sum,
    );
    let c_x8 = Blake2sXor8Component::new(
        &mut alloc,
        Blake2sXor8Eval {
            lookup_elements: xor_elements.xor8,
            claimed_sum: claimed_sums.xor8_sum,
        },
        claimed_sums.xor8_sum,
    );
    let c_x7 = Blake2sXor7Component::new(
        &mut alloc,
        Blake2sXor7Eval {
            lookup_elements: xor_elements.xor7,
            claimed_sum: claimed_sums.xor7_sum,
        },
        claimed_sums.xor7_sum,
    );
    let c_x4 = Blake2sXor4Component::new(
        &mut alloc,
        Blake2sXor4Eval {
            lookup_elements: xor_elements.xor4,
            claimed_sum: claimed_sums.xor4_sum,
        },
        claimed_sums.xor4_sum,
    );

    let comps: [&dyn Component; 8] = [
        &c_audit, &c_sched, &c_round, &c_bridge, &c_x9, &c_x8, &c_x7, &c_x4,
    ];
    verify(&comps, channel, &mut scheme, stark.clone())
        .map_err(|e| format!("standalone STARK verification rejected: {e}"))
}

/// The proof of work before the outer lookup draws, for both outer AIRs: with `bits > 0` the nonce
/// must satisfy the work condition on the channel and is mixed into it; with `bits = 0` it must be
/// zero and nothing is mixed (`standalone_lookup_pow_bits_for_profile_id`).
fn check_lookup_pow(channel: &mut Blake2sChannel, (bits, nonce): (u32, u64)) -> Result<(), String> {
    if bits == 0 {
        if nonce != 0 {
            return Err("standalone STARK lookup nonce must be zero when the profile has no lookup proof of work".into());
        }
        return Ok(());
    }
    if !channel.verify_pow_nonce(bits, nonce) && outer_keep(outer_thread_mask(), outer_ids::D15) {
        return Err("standalone STARK lookup proof of work is invalid".into());
    }
    channel.mix_u64(nonce);
    Ok(())
}

/// Preprocessed `Tree 0` of an `exp_pos2` receipt: `[IsFirst(audit), AuditStatic[0..25],
/// Pos2Static[0..21]]`.
pub fn standalone_preprocessed_ids_and_log_sizes_pos2(
    audit_log_size: u32,
    pos2_log_size: u32,
) -> (Vec<PreProcessedColumnId>, Vec<u32>) {
    let mut ids = vec![standalone_is_first_id(audit_log_size)];
    let mut sizes = vec![audit_log_size];
    for id in standalone_audit_static_ids(audit_log_size) {
        ids.push(id);
        sizes.push(audit_log_size);
    }
    for id in crate::poseidon2_air::pos2_static_ids(pos2_log_size) {
        ids.push(id);
        sizes.push(pos2_log_size);
    }
    (ids, sizes)
}

/// The outer STARK of an `exp_pos2` receipt: components `[audit, Poseidon2]`; the Poseidon2
/// claimed sum travels in `bridge_sum` and every other Blake2s sum must be zero.
fn verify_standalone_stark_core_pos2(
    expected_config: PcsConfig,
    lookup_pow: (u32, u64),
    audit_log_size: u32,
    pos2_log_size: u32,
    public_memory_items: &[(u32, [BaseField; 4])],
    claimed_sums: &StandaloneClaimedSums,
    stark: &StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), String> {
    use crate::poseidon2_air::{pos2_info, Pos2ChainElements, Pos2Component, Pos2Eval};
    if stark.config != expected_config {
        return Err("standalone STARK PcsConfig mismatch".into());
    }
    if stark.commitments.len() != 4 {
        return Err(format!("standalone STARK expected 4 tree commitments, got {}", stark.commitments.len()));
    }
    claimed_sums.validate_canonical()?;
    let unused = [claimed_sums.sched_sum, claimed_sums.round_sum, claimed_sums.xor9_sum, claimed_sums.xor8_sum, claimed_sums.xor7_sum, claimed_sums.xor4_sum];
    if unused.iter().any(|s| !s.is_zero()) {
        return Err("exp_pos2 receipt carries a non-zero Blake2s claimed sum".into());
    }
    validate_standalone_stark_encodings(stark)?;

    let (prep_ids, prep_log_sizes) = standalone_preprocessed_ids_and_log_sizes_pos2(audit_log_size, pos2_log_size);
    let mut scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(expected_config);
    scheme.commit(stark.commitments[0], &prep_log_sizes, channel);
    let mut comp_sizes = vec![
        standalone_audit_info(audit_log_size).mask_offsets.as_cols_ref().map_cols(|_| audit_log_size),
        pos2_info(pos2_log_size).mask_offsets.as_cols_ref().map_cols(|_| pos2_log_size),
    ];
    let tree_sizes = TreeVec::concat_cols(comp_sizes.drain(..));
    scheme.commit(stark.commitments[1], &tree_sizes[ORIGINAL_TRACE_IDX], channel);
    // One proof of work covers both relation draws: they follow it in the same round.
    check_lookup_pow(channel, lookup_pow)?;
    let chain_elements = Pos2ChainElements::draw(channel);
    let mem_bus_elements = MemoryBusElements::draw(channel);
    scheme.commit(stark.commitments[2], &tree_sizes[INTERACTION_TRACE_IDX], channel);
    channel.mix_felts(&claimed_sums.as_array());

    let mut public_memory_sum = SecureField::zero();
    for &(id, coords) in public_memory_items {
        let denom: SecureField = mem_bus_elements.combine(&[BaseField::from_u32_unchecked(id), coords[0], coords[1], coords[2], coords[3]]);
        if denom.is_zero() {
            return Err("degenerate zero MemoryBus denominator on public input".into());
        }
        public_memory_sum -= SecureField::one() / denom;
    }
    if !(claimed_sums.audit_sum + claimed_sums.bridge_sum + public_memory_sum).is_zero() {
        return Err("standalone STARK global + public MemoryBus LogUp sum non-zero".into());
    }

    let mut alloc = TraceLocationAllocator::new_with_preprocessed_columns(&prep_ids);
    let c_audit = StandaloneAuditComponent::new(
        &mut alloc,
        StandaloneAuditAirEval { log_size: audit_log_size, mem_bus_elements: mem_bus_elements.clone(), claimed_sum: claimed_sums.audit_sum, #[cfg(test)] skip: crate::blake2s_air::outer_skip::current() },
        claimed_sums.audit_sum,
    );
    let c_pos2 = Pos2Component::new(
        &mut alloc,
        Pos2Eval { log_size: pos2_log_size, chain_elements, mem_bus_elements, claimed_sum: claimed_sums.bridge_sum },
        claimed_sums.bridge_sum,
    );
    let comps: [&dyn Component; 2] = [&c_audit, &c_pos2];
    verify(&comps, channel, &mut scheme, stark.clone()).map_err(|e| format!("standalone STARK verification rejected: {e}"))
}

/// The two `QM31` of an `exp_pos2` digest (8 little-endian words; canonical after `validate`).
pub fn pos2_digest_items(base: u32, d: &[u8; 32]) -> [(u32, [BaseField; 4]); 2] {
    let w = |i: usize| BaseField::from_u32_unchecked(u32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]) % P);
    [(base, [w(0), w(1), w(2), w(3)]), (base + 1, [w(4), w(5), w(6), w(7)])]
}

fn build_block_public_memory_items_pos2(stmt: &StandaloneBlockStatement) -> Vec<(u32, [BaseField; 4])> {
    let z = BaseField::zero();
    let mut out = Vec::with_capacity(10);
    out.extend(pos2_digest_items(PUB_ID_WEIGHTS_DIGEST_BASE, &stmt.weights_and_boundaries_digest));
    out.push((PUB_ID_INITIAL_CHAIN_BASE, [BaseField::from_u32_unchecked(stmt.initial_chain.layer), z, z, z]));
    out.extend(pos2_digest_items(PUB_ID_INITIAL_CHAIN_BASE + 1, &stmt.initial_chain.prev_receipt_digest));
    out.push((PUB_ID_FINAL_CHAIN_BASE, [BaseField::from_u32_unchecked(stmt.final_chain.layer), z, z, z]));
    out.extend(pos2_digest_items(PUB_ID_FINAL_CHAIN_BASE + 1, &stmt.final_chain.prev_receipt_digest));
    out.extend(pos2_digest_items(PUB_ID_PUBLIC_IO_BASE, &stmt.public_io_digest));
    out
}

pub fn build_block_public_memory_items(
    stmt: &StandaloneBlockStatement,
) -> Vec<(u32, [BaseField; 4])> {
    if is_pos2_profile(&stmt.profile_id) {
        return build_block_public_memory_items_pos2(stmt);
    }
    let z = BaseField::zero();
    let mut out = Vec::with_capacity(18);
    for (q, limbs) in bytes_32_to_digest_limbs(&stmt.weights_and_boundaries_digest)
        .into_iter()
        .enumerate()
    {
        out.push((PUB_ID_WEIGHTS_DIGEST_BASE + q as u32, limbs));
    }
    out.push((
        PUB_ID_INITIAL_CHAIN_BASE,
        [BaseField::from_u32_unchecked(stmt.initial_chain.layer), z, z, z],
    ));
    for (q, limbs) in bytes_32_to_digest_limbs(&stmt.initial_chain.prev_receipt_digest)
        .into_iter()
        .enumerate()
    {
        out.push((PUB_ID_INITIAL_CHAIN_BASE + 1 + q as u32, limbs));
    }
    out.push((
        PUB_ID_FINAL_CHAIN_BASE,
        [BaseField::from_u32_unchecked(stmt.final_chain.layer), z, z, z],
    ));
    for (q, limbs) in bytes_32_to_digest_limbs(&stmt.final_chain.prev_receipt_digest)
        .into_iter()
        .enumerate()
    {
        out.push((PUB_ID_FINAL_CHAIN_BASE + 1 + q as u32, limbs));
    }
    for (q, limbs) in bytes_32_to_digest_limbs(&stmt.public_io_digest)
        .into_iter()
        .enumerate()
    {
        out.push((PUB_ID_PUBLIC_IO_BASE + q as u32, limbs));
    }
    out
}

pub fn build_whole_model_public_memory_items(
    stmt: &WholeModelStatement,
) -> Vec<(u32, [BaseField; 4])> {
    let z = BaseField::zero();
    let cap = if stmt.tokens_committed {
        12
    } else {
        4 + 2 * stmt.batch
    };
    let mut out = Vec::with_capacity(cap);
    for (q, limbs) in bytes_32_to_digest_limbs(&stmt.weights_and_boundaries_digest)
        .into_iter()
        .enumerate()
    {
        out.push((PUB_ID_WEIGHTS_DIGEST_BASE + q as u32, limbs));
    }
    if stmt.tokens_committed {
        for (q, limbs) in bytes_32_to_digest_limbs(&stmt.prompt_commitment)
            .into_iter()
            .enumerate()
        {
            out.push((PUB_ID_PROMPT_COM_BASE + q as u32, limbs));
        }
        for (q, limbs) in bytes_32_to_digest_limbs(&stmt.emitted_commitment)
            .into_iter()
            .enumerate()
        {
            out.push((PUB_ID_EMITTED_COM_BASE + q as u32, limbs));
        }
    } else {
        for (t, &tok) in stmt.prompt_token_ids.iter().enumerate() {
            out.push((
                PUB_ID_PROMPT_TOK_BASE + t as u32,
                [BaseField::from_u32_unchecked(tok), z, z, z],
            ));
        }
        for (t, &tok) in stmt.emitted_token_ids.iter().enumerate() {
            out.push((
                PUB_ID_EMITTED_TOK_BASE + t as u32,
                [BaseField::from_u32_unchecked(tok), z, z, z],
            ));
        }
    }
    out
}

pub fn mix_standalone_block_statement(
    channel: &mut Blake2sChannel,
    stmt: &StandaloneBlockStatement,
) {
    mix_bytes_into_channel(channel, STANDALONE_BLOCK_MAGIC, &stmt.policy_bytes());
    mix_bytes_into_channel(channel, b"weights", &stmt.weights_and_boundaries_digest);
    channel.mix_u32s(&[stmt.initial_chain.layer, stmt.final_chain.layer]);
    mix_bytes_into_channel(channel, b"init_chain", &stmt.initial_chain.prev_receipt_digest);
    mix_bytes_into_channel(channel, b"final_chain", &stmt.final_chain.prev_receipt_digest);
    mix_bytes_into_channel(channel, b"pub_io", &stmt.public_io_digest);
}

pub fn mix_whole_model_statement(channel: &mut Blake2sChannel, stmt: &WholeModelStatement) {
    mix_bytes_into_channel(channel, WHOLE_MODEL_MAGIC, &stmt.policy_bytes());
    mix_bytes_into_channel(channel, b"weights", &stmt.weights_and_boundaries_digest);
    if stmt.tokens_committed {
        mix_bytes_into_channel(channel, b"prompt_com", &stmt.prompt_commitment);
        mix_bytes_into_channel(channel, b"emitted_com", &stmt.emitted_commitment);
    } else {
        channel.mix_u32s(&stmt.prompt_token_ids);
        channel.mix_u32s(&stmt.emitted_token_ids);
    }
}

/// Verify a `STWOREC2` standalone block receipt against **only** `expected_standalone_anchor`
/// (`32 B`) and `expected_initial_chain` (**zero `expected_wiring_root`**).
pub fn verify_standalone_block_stark_against_anchor(
    expected_standalone_anchor: &[u8; 32],
    expected_initial_chain: &StandaloneChainLink,
    proof: &StandaloneBlockStarkProof,
) -> Result<StandaloneAuditVerifierRun, String> {
    verify_standalone_block_stark_against_anchor_and_io(
        expected_standalone_anchor,
        expected_initial_chain,
        None,
        None,
        proof,
    )
}

/// `public_io_digest` of a `STWOREC2` block from its two boundary commitments: the CPRS commitment
/// of the block's input hidden state and of its output hidden state. Inside the proof each
/// commitment is hashed as its sixteen little-endian 16-bit limbs, each widened to four bytes.
pub fn compute_standalone_block_public_io_digest(
    com_h_in: &[u8; 32],
    com_h_out: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(STANDALONE_PUBLIC_IO_DOMAIN);
    for com in [com_h_in, com_h_out] {
        for limb in com.chunks_exact(2) {
            h.update([limb[0], limb[1], 0, 0]);
        }
    }
    h.finalize().into()
}

/// `public_io_digest` of a block of profile `profile_id` from its two boundary commitments:
/// [`compute_standalone_block_public_io_digest`] for the Blake2s profiles; for the EXPERIMENTAL
/// `exp_pos2` profiles the circuit's Poseidon2 digest `H([PUBLIC_IO], com_h_in || com_h_out)` over
/// the eight words of each commitment, which must all be below `p`.
pub fn compute_standalone_block_public_io_digest_for_profile(
    profile_id: &str,
    com_h_in: &[u8; 32],
    com_h_out: &[u8; 32],
) -> Result<[u8; 32], String> {
    if !is_pos2_profile(profile_id) {
        return Ok(compute_standalone_block_public_io_digest(com_h_in, com_h_out));
    }
    use crate::poseidon2_m31 as p2;
    let mut msg = Vec::with_capacity(16);
    for com in [com_h_in, com_h_out] {
        msg.extend(p2::digest_from_bytes(com).ok_or("exp_pos2 boundary commitment has a word >= p")?);
    }
    let cap = p2::capacity(p2::tag(p2::role::PUBLIC_IO, 0), msg.len(), 0);
    Ok(p2::digest_to_bytes(&p2::hash(cap, &msg)))
}

/// One side of the public input/output digest of a block with an end stage
/// (the embedding side or the head side).
#[derive(Clone, Copy, Debug)]
pub enum StandaloneEndsIo<'a> {
    /// Token ids: the embedding's input (`batch` of them) or the head's output (one).
    Tokens(&'a [u32]),
    /// Multi-token window `(l, k)` followed by length-`batch` zero-padded token array.
    MultiTokens {
        l: u32,
        k: u32,
        padded_tokens: &'a [u32],
    },
    /// A boundary commitment: `com_h_in` without the embedding, `com_h_out` without the head.
    Commitment(&'a [u8; 32]),
}

/// `public_io_digest` of a block with an end stage:
/// - Single-token (`flags & ENDS_FLAG_MULTI == 0`): `Blake2s(ENDS_IO_V1 || flags || in || out)`.
/// - Multi-token (`flags & ENDS_FLAG_MULTI != 0`): `Blake2s(ENDS_MULTI_IO_V1 || flags || in || out)`.
pub fn compute_standalone_ends_public_io_digest(
    flags: u8,
    input: StandaloneEndsIo<'_>,
    output: StandaloneEndsIo<'_>,
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    let domain = if flags & ENDS_FLAG_MULTI != 0 {
        STANDALONE_ENDS_MULTI_IO_DOMAIN
    } else {
        STANDALONE_ENDS_IO_DOMAIN
    };
    h.update(domain);
    h.update([flags]);
    for side in [input, output] {
        match side {
            StandaloneEndsIo::Tokens(toks) => {
                for &t in toks {
                    h.update(t.to_le_bytes());
                    h.update([0u8; 12]);
                }
            }
            StandaloneEndsIo::MultiTokens {
                l,
                k,
                padded_tokens,
            } => {
                for v in [l, k].into_iter().chain(padded_tokens.iter().copied()) {
                    h.update(v.to_le_bytes());
                    h.update([0u8; 12]);
                }
            }
            StandaloneEndsIo::Commitment(com) => {
                for limb in com.chunks_exact(2) {
                    h.update([limb[0], limb[1], 0, 0]);
                }
            }
        }
    }
    h.finalize().into()
}

/// One block of a chain: its anchor, its receipt and the two boundary commitments its prover
/// reports alongside it.
pub struct StandaloneChainedBlock<'a> {
    pub standalone_anchor: &'a [u8; 32],
    pub proof: &'a StandaloneBlockStarkProof,
    pub com_h_in: [u8; 32],
    pub com_h_out: [u8; 32],
}

/// What a verified chain of blocks establishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandaloneChainVerdict {
    pub final_chain: StandaloneChainLink,
    /// Commitment to the hidden state the first block consumed.
    pub com_h_in: [u8; 32],
    /// Commitment to the hidden state the last block produced.
    pub com_h_out: [u8; 32],
}

/// Verify consecutive `STWOREC2` block receipts as one computation.
///
/// The anchors in `blocks` are trusted: a caller that got them from the prover must use
/// `verify_standalone_block_chain_against_anchor`. Each receipt is verified against its own anchor. Three things then tie the blocks together:
/// the transcript chain (`final_chain` of one block is `initial_chain` of the next), the reported
/// boundary commitments (each pair must hash to the receipt's `public_io_digest`, which the proof
/// binds to the commitments it opened), and equality of `com_h_out` of one block with `com_h_in`
/// of the next. A boundary commitment covers three 24-bit residual limb columns `[a, u_b, u_c]`
/// under parameters fixed by the batch, the profile, the mode, `zk_trace_blinded` and the witness
/// fold (with hidden width `d_model` pinned by each block's registered `Tree 0` root when anchors
/// are bound via `verify_standalone_block_chain_against_anchor`), so the blocks must agree on
/// those; given that, equal commitments mean the next block consumed exactly the hidden state the
/// previous one produced.
///
/// It does not tie the first `com_h_in` to a prompt or the last `com_h_out` to an output. Those
/// are returned for the caller.
pub fn verify_standalone_block_chain(
    initial_chain: &StandaloneChainLink,
    blocks: &[StandaloneChainedBlock<'_>],
) -> Result<StandaloneChainVerdict, String> {
    verify_standalone_block_chain_timed(initial_chain, blocks).map(|(v, _)| v)
}

pub fn verify_standalone_block_chain_timed(
    initial_chain: &StandaloneChainLink,
    blocks: &[StandaloneChainedBlock<'_>],
) -> Result<(StandaloneChainVerdict, Vec<f64>), String> {
    let first = blocks.first().ok_or("standalone block chain is empty")?;
    let mut expected_initial = *initial_chain;
    let mut prev_out: Option<[u8; 32]> = None;
    let mut per_block_ms = Vec::with_capacity(blocks.len());
    for (k, blk) in blocks.iter().enumerate() {
        let st = &blk.proof.statement;
        let st0 = &first.proof.statement;
        if st.ends.is_some() {
            return Err(format!(
                "standalone block chain: block {k} has an end stage; verify a model with verify_standalone_model_chain_against_anchor"
            ));
        }
        if st.batch != st0.batch
            || st.d_model != st0.d_model
            || st.profile_id != st0.profile_id
            || st.mode_tag != st0.mode_tag
            || st.cprs_witness_f != st0.cprs_witness_f
            || st.zk_trace_blinded != st0.zk_trace_blinded
        {
            return Err(format!(
                "standalone block chain: block {k} differs from block 0 in batch, d_model, profile, mode, witness fold or zk_trace_blinded"
            ));
        }
        if let Some(po) = prev_out {
            if blk.com_h_in != po && outer_keep(outer_thread_mask(), outer_ids::D18) {
                return Err(format!(
                    "standalone block chain: block {k} does not consume the hidden state block {} produced",
                    k - 1
                ));
            }
        }
        let io = compute_standalone_block_public_io_digest_for_profile(&st.profile_id, &blk.com_h_in, &blk.com_h_out)
            .map_err(|e| format!("standalone block chain: block {k}: {e}"))?;
        let run = verify_standalone_block_stark_against_anchor_and_io(
            blk.standalone_anchor,
            &expected_initial,
            None,
            Some(&io),
            blk.proof,
        )
        .map_err(|e| format!("standalone block chain: block {k}: {e}"))?;
        per_block_ms.push(run.verify_ms);
        expected_initial = st.final_chain;
        prev_out = Some(blk.com_h_out);
    }
    Ok((
        StandaloneChainVerdict {
            final_chain: expected_initial,
            com_h_in: first.com_h_in,
            com_h_out: prev_out.expect("non-empty chain"),
        },
        per_block_ms,
    ))
}

/// The `32 B` anchor of a model registered as consecutive blocks: the number of blocks and the
/// anchor of each block, in order. It is what a verifier of a chain holds. A block anchor already
/// binds that block's policy, weights and `Tree 0`.
pub fn compute_standalone_chain_anchor(block_anchors: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(STANDALONE_CHAIN_ANCHOR_DOMAIN);
    h.update((block_anchors.len() as u64).to_le_bytes());
    for anchor in block_anchors {
        h.update(anchor);
    }
    h.finalize().into()
}

/// Verify a chain of blocks as the registered model: `verify_standalone_block_chain`, after
/// checking that the block anchors the prover supplied are exactly the registered ones.
///
/// `verify_standalone_block_chain` alone takes each anchor on trust. Given anchors a prover chose,
/// it accepts a chain over other weights, and a chain that stops early.
pub fn verify_standalone_block_chain_against_anchor(
    expected_chain_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    blocks: &[StandaloneChainedBlock<'_>],
) -> Result<StandaloneChainVerdict, String> {
    verify_standalone_block_chain_against_anchor_timed(expected_chain_anchor, initial_chain, blocks)
        .map(|(v, _)| v)
}

pub fn verify_standalone_block_chain_against_anchor_timed(
    expected_chain_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    blocks: &[StandaloneChainedBlock<'_>],
) -> Result<(StandaloneChainVerdict, Vec<f64>), String> {
    if *expected_chain_anchor == [0u8; 32] {
        return Err("expected_chain_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = blocks.iter().map(|b| *b.standalone_anchor).collect();
    if &compute_standalone_chain_anchor(&anchors) != expected_chain_anchor {
        return Err(
            "standalone block chain: 32B chain anchor mismatch (a block anchor, the block count or the order differs from registration)"
                .into(),
        );
    }
    verify_standalone_block_chain_timed(initial_chain, blocks)
}

/// The `32 B` anchor of a model registered as receipts `[embedding, blocks.., head]`: the count and
/// each receipt's anchor in order. Each receipt anchor binds that
/// receipt's kind byte and end-stage flags, so the anchor fixes which receipt holds which end.
pub fn compute_standalone_model_anchor(receipt_anchors: &[[u8; 32]]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(STANDALONE_MODEL_ANCHOR_DOMAIN);
    h.update((receipt_anchors.len() as u64).to_le_bytes());
    for anchor in receipt_anchors {
        h.update(anchor);
    }
    h.finalize().into()
}

/// What a verified model chain establishes: the registered model, run on `tokens`, gives
/// `out_token` as the lowest-index argmax at the last position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneModelVerdict {
    pub model_anchor: [u8; 32],
    pub tokens: Vec<u32>,
    pub out_token: u32,
    pub final_chain: StandaloneChainLink,
}

/// Verify a model: receipts `[embedding, blocks.., head]` (or one receipt with both ends) against
/// the model anchor the caller registered, the public tokens and the claimed output token
/// as public inputs. A receipt's unused boundary (`com_h_in` of the embedding
/// receipt, `com_h_out` of the head receipt) must be given as all zeros.
pub fn verify_standalone_model_chain_against_anchor(
    expected_model_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    receipts: &[StandaloneChainedBlock<'_>],
    tokens: &[u32],
    out_token: u32,
) -> Result<StandaloneModelVerdict, String> {
    verify_standalone_model_chain_against_anchor_timed(expected_model_anchor, initial_chain, receipts, tokens, out_token)
        .map(|(v, _)| v)
}

pub fn verify_standalone_model_chain_against_anchor_timed(
    expected_model_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    receipts: &[StandaloneChainedBlock<'_>],
    tokens: &[u32],
    out_token: u32,
) -> Result<(StandaloneModelVerdict, Vec<f64>), String> {
    if *expected_model_anchor == [0u8; 32] {
        return Err("expected_model_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = receipts.iter().map(|b| *b.standalone_anchor).collect();
    if &compute_standalone_model_anchor(&anchors) != expected_model_anchor {
        return Err(
            "standalone model: 32B model anchor mismatch (a receipt anchor, the receipt count or the order differs from registration)"
                .into(),
        );
    }
    let n = receipts.len();
    let first = receipts.first().ok_or("standalone model: no receipts")?;
    let st0 = &first.proof.statement;
    let decl = |k: usize| receipts[k].proof.statement.ends;
    let emb = decl(0).filter(|e| e.embed).ok_or("standalone model: receipt 0 has no embedding stage")?;
    let head = decl(n - 1).filter(|e| e.head).ok_or("standalone model: the last receipt has no head stage")?;
    for k in 0..n {
        let inner = k > 0 && k + 1 < n;
        let bad = match decl(k) {
            None => false,
            Some(e) => inner || (e.embed && k != 0) || (e.head && k != n - 1),
        };
        if bad {
            return Err(format!("standalone model: receipt {k} holds an end stage out of place"));
        }
    }
    if emb.vocab_size != head.vocab_size {
        return Err("standalone model: the embedding and the head declare different vocabularies".into());
    }
    if receipts
        .iter()
        .any(|r| r.proof.statement.ends.is_some_and(|e| e.multi_token))
    {
        return Err(
            "standalone model: multi-token end stage requires verify_standalone_model_gen_chain_against_anchor"
                .into(),
        );
    }
    if tokens.len() != st0.batch {
        return Err(format!("standalone model: {} tokens for a batch of {}", tokens.len(), st0.batch));
    }
    let keep_vocab = outer_keep(outer_thread_mask(), outer_ids::D19);
    if let Some(&t) = tokens.iter().find(|&&t| t as usize >= emb.vocab_size).filter(|_| keep_vocab) {
        return Err(format!("standalone model: token {t} >= vocab {}", emb.vocab_size));
    }
    if out_token as usize >= head.vocab_size && keep_vocab {
        return Err(format!("standalone model: output token {out_token} >= vocab {}", head.vocab_size));
    }
    check_one_weight_fold_and_dims(receipts)?;
    let mut expected_initial = *initial_chain;
    let mut prev_out: Option<[u8; 32]> = None;
    let mut per_ms = Vec::with_capacity(n);
    let out_slice = [out_token];
    for (k, blk) in receipts.iter().enumerate() {
        let st = &blk.proof.statement;
        if st.batch != st0.batch
            || st.d_model != st0.d_model
            || st.profile_id != st0.profile_id
            || st.mode_tag != st0.mode_tag
            || st.cprs_witness_f != st0.cprs_witness_f
            || st.zk_trace_blinded != st0.zk_trace_blinded
            || st.transformer.is_none()
        {
            return Err(format!(
                "standalone model: receipt {k} differs from receipt 0 in batch, d_model, profile, mode, witness fold or zk_trace_blinded, or is not a transformer block"
            ));
        }
        let ends = st.ends;
        let has_in = !ends.is_some_and(|e| e.embed);
        let has_out = !ends.is_some_and(|e| e.head);
        if (!has_in && blk.com_h_in != [0u8; 32]) || (!has_out && blk.com_h_out != [0u8; 32]) {
            return Err(format!("standalone model: receipt {k} reports a boundary it does not have"));
        }
        if let Some(po) = prev_out {
            if !has_in || blk.com_h_in != po {
                return Err(format!(
                    "standalone model: receipt {k} does not consume the hidden state receipt {} produced",
                    k - 1
                ));
            }
        }
        let io = match ends {
            None => compute_standalone_block_public_io_digest(&blk.com_h_in, &blk.com_h_out),
            Some(e) => compute_standalone_ends_public_io_digest(
                e.flags(),
                if e.embed { StandaloneEndsIo::Tokens(tokens) } else { StandaloneEndsIo::Commitment(&blk.com_h_in) },
                if e.head { StandaloneEndsIo::Tokens(&out_slice) } else { StandaloneEndsIo::Commitment(&blk.com_h_out) },
            ),
        };
        let run = verify_standalone_block_stark_against_anchor_and_io(
            blk.standalone_anchor,
            &expected_initial,
            None,
            Some(&io),
            blk.proof,
        )
        .map_err(|e| format!("standalone model: receipt {k}: {e}"))?;
        per_ms.push(run.verify_ms);
        expected_initial = st.final_chain;
        prev_out = has_out.then_some(blk.com_h_out);
    }
    check_model_starts_at_layer_0(initial_chain)?;
    Ok((
        StandaloneModelVerdict {
            model_anchor: *expected_model_anchor,
            tokens: tokens.to_vec(),
            out_token,
            final_chain: expected_initial,
        },
        per_ms,
    ))
}

/// One model has one CPRS weight fold and one set of transformer dimensions; checked for
/// every receipt before any proof, by both model verifiers.
fn check_one_weight_fold_and_dims(receipts: &[StandaloneChainedBlock<'_>]) -> Result<(), String> {
    let st0 = &receipts.first().ok_or("standalone model: no receipts")?.proof.statement;
    if let Some(k) = receipts.iter().position(|b| {
        b.proof.statement.cprs_weight_f != st0.cprs_weight_f || b.proof.statement.transformer != st0.transformer
    }) {
        return Err(format!(
            "standalone model: receipt {k} differs from receipt 0 in CPRS weight fold or transformer dimensions"
        ));
    }
    Ok(())
}

/// A model starts at layer 0: the embedding feeds layer 0 (a chain proved from layer 777
/// was accepted). Both model verifiers check it after the receipts, so a link the receipts disagree with is
/// reported as that mismatch.
fn check_model_starts_at_layer_0(initial_chain: &StandaloneChainLink) -> Result<(), String> {
    if initial_chain.layer != 0 {
        return Err(format!("standalone model: initial_chain.layer must be 0, got {}", initial_chain.layer));
    }
    Ok(())
}

/// What a verified multi-token causal autoregressive chain establishes: running the registered
/// integer model sequentially from `prompt_tokens` (`L >= 1`) greedily emits `gen_tokens` (`K >= 1`)
/// one token at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandaloneModelMultiVerdict {
    pub model_anchor: [u8; 32],
    pub prompt_tokens: Vec<u32>,
    pub gen_tokens: Vec<u32>,
    pub final_chain: StandaloneChainLink,
}

/// Verify a single-pass `K`-token causal autoregressive model proof (single receipt or `STWOMDL2`
/// `[embedding, blocks.., head]` chain) against `expected_model_anchor`, `prompt_tokens` (`L`), and
/// `gen_tokens` (`K`). For `!multi_token` receipts (`K = 1, L = batch`), delegates directly to
/// [`verify_standalone_model_chain_against_anchor_timed`].
pub fn verify_standalone_model_gen_chain_against_anchor(
    expected_model_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    receipts: &[StandaloneChainedBlock<'_>],
    prompt_tokens: &[u32],
    gen_tokens: &[u32],
) -> Result<StandaloneModelMultiVerdict, String> {
    verify_standalone_model_gen_chain_against_anchor_timed(
        expected_model_anchor,
        initial_chain,
        receipts,
        prompt_tokens,
        gen_tokens,
    )
    .map(|(v, _)| v)
}

pub fn verify_standalone_model_gen_chain_against_anchor_timed(
    expected_model_anchor: &[u8; 32],
    initial_chain: &StandaloneChainLink,
    receipts: &[StandaloneChainedBlock<'_>],
    prompt_tokens: &[u32],
    gen_tokens: &[u32],
) -> Result<(StandaloneModelMultiVerdict, Vec<f64>), String> {
    if *expected_model_anchor == [0u8; 32] {
        return Err("expected_model_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = receipts.iter().map(|b| *b.standalone_anchor).collect();
    if &compute_standalone_model_anchor(&anchors) != expected_model_anchor {
        return Err(
            "standalone model: 32B model anchor mismatch (a receipt anchor, the receipt count or the order differs from registration)"
                .into(),
        );
    }
    let n = receipts.len();
    let first = receipts.first().ok_or("standalone model: no receipts")?;
    let st0 = &first.proof.statement;
    let decl = |idx: usize| receipts[idx].proof.statement.ends;
    let emb = decl(0)
        .filter(|e| e.embed)
        .ok_or("standalone model: receipt 0 has no embedding stage")?;
    let head = decl(n - 1)
        .filter(|e| e.head)
        .ok_or("standalone model: the last receipt has no head stage")?;
    for k_idx in 0..n {
        let inner = k_idx > 0 && k_idx + 1 < n;
        let bad = match decl(k_idx) {
            None => false,
            Some(e) => inner || (e.embed && k_idx != 0) || (e.head && k_idx != n - 1),
        };
        if bad {
            return Err(format!("standalone model: receipt {k_idx} holds an end stage out of place"));
        }
    }
    if emb.vocab_size != head.vocab_size {
        return Err("standalone model: the embedding and the head declare different vocabularies".into());
    }
    if emb.multi_token != head.multi_token {
        return Err("standalone model: embedding and head disagree on multi_token flag".into());
    }
    if !emb.multi_token {
        if gen_tokens.len() != 1 {
            return Err(format!(
                "standalone model: single-token end stage requires K=1 generated token, got K={}",
                gen_tokens.len()
            ));
        }
        let (v, ms) = verify_standalone_model_chain_against_anchor_timed(
            expected_model_anchor,
            initial_chain,
            receipts,
            prompt_tokens,
            gen_tokens[0],
        )?;
        return Ok((
            StandaloneModelMultiVerdict {
                model_anchor: v.model_anchor,
                prompt_tokens: v.tokens,
                gen_tokens: vec![v.out_token],
                final_chain: v.final_chain,
            },
            ms,
        ));
    }

    let b_sz = st0.batch;
    let l_len = prompt_tokens.len();
    let k_len = gen_tokens.len();
    if l_len == 0 || k_len == 0 {
        return Err(format!(
            "standalone multi-token model: requires L >= 1 and K >= 1, got L={l_len}, K={k_len}"
        ));
    }
    let window_ok = if k_len >= 2 {
        l_len + k_len <= b_sz
    } else {
        l_len <= b_sz
    };
    if !window_ok {
        return Err(format!(
            "standalone multi-token model: window (L={l_len}, K={k_len}) exceeds batch={b_sz} (requires L + K <= batch for K >= 2, L <= batch for K == 1)"
        ));
    }
    if let Some(&t) = prompt_tokens.iter().find(|&&t| t as usize >= emb.vocab_size) {
        return Err(format!(
            "standalone multi-token model: prompt token {t} >= vocab {}",
            emb.vocab_size
        ));
    }
    if let Some(&y) = gen_tokens.iter().find(|&&y| y as usize >= head.vocab_size) {
        return Err(format!(
            "standalone multi-token model: generated token {y} >= vocab {}",
            head.vocab_size
        ));
    }
    check_one_weight_fold_and_dims(receipts)?;

    let l_u32 = l_len as u32;
    let k_u32 = k_len as u32;
    let mut prompt_pad = vec![0u32; b_sz];
    prompt_pad[..l_len].copy_from_slice(prompt_tokens);
    let mut t_in = vec![0u32; b_sz];
    t_in[..l_len].copy_from_slice(prompt_tokens);
    for i in 0..k_len.saturating_sub(1) {
        t_in[l_len + i] = gen_tokens[i];
    }
    let mut gen_pad = vec![0u32; b_sz];
    gen_pad[..k_len].copy_from_slice(gen_tokens);

    let mut expected_initial = *initial_chain;
    let mut prev_out: Option<[u8; 32]> = None;
    let mut per_ms = Vec::with_capacity(n);
    for (idx, blk) in receipts.iter().enumerate() {
        let st = &blk.proof.statement;
        if st.batch != st0.batch
            || st.d_model != st0.d_model
            || st.profile_id != st0.profile_id
            || st.mode_tag != st0.mode_tag
            || st.cprs_witness_f != st0.cprs_witness_f
            || st.zk_trace_blinded != st0.zk_trace_blinded
            || st.transformer.is_none()
        {
            return Err(format!(
                "standalone model: receipt {idx} differs from receipt 0 in batch, d_model, profile, mode, witness fold or zk_trace_blinded, or is not a transformer block"
            ));
        }
        let ends = st.ends;
        let has_in = !ends.is_some_and(|e| e.embed);
        let has_out = !ends.is_some_and(|e| e.head);
        if (!has_in && blk.com_h_in != [0u8; 32]) || (!has_out && blk.com_h_out != [0u8; 32]) {
            return Err(format!("standalone model: receipt {idx} reports a boundary it does not have"));
        }
        if let Some(po) = prev_out {
            if !has_in || blk.com_h_in != po {
                return Err(format!(
                    "standalone model: receipt {idx} does not consume the hidden state receipt {} produced",
                    idx - 1
                ));
            }
        }
        let io = match ends {
            None => compute_standalone_block_public_io_digest(&blk.com_h_in, &blk.com_h_out),
            Some(e) => {
                let in_side = if e.embed {
                    StandaloneEndsIo::MultiTokens {
                        l: l_u32,
                        k: k_u32,
                        padded_tokens: if e.head { &prompt_pad } else { &t_in },
                    }
                } else {
                    StandaloneEndsIo::Commitment(&blk.com_h_in)
                };
                let out_side = if e.head {
                    StandaloneEndsIo::MultiTokens {
                        l: l_u32,
                        k: k_u32,
                        padded_tokens: &gen_pad,
                    }
                } else {
                    StandaloneEndsIo::Commitment(&blk.com_h_out)
                };
                compute_standalone_ends_public_io_digest(e.flags(), in_side, out_side)
            }
        };
        let run = verify_standalone_block_stark_against_anchor_and_io(
            blk.standalone_anchor,
            &expected_initial,
            None,
            Some(&io),
            blk.proof,
        )
        .map_err(|e| format!("standalone model: receipt {idx}: {e}"))?;
        per_ms.push(run.verify_ms);
        expected_initial = st.final_chain;
        prev_out = has_out.then_some(blk.com_h_out);
    }
    check_model_starts_at_layer_0(initial_chain)?;
    Ok((
        StandaloneModelMultiVerdict {
            model_anchor: *expected_model_anchor,
            prompt_tokens: prompt_tokens.to_vec(),
            gen_tokens: gen_tokens.to_vec(),
            final_chain: expected_initial,
        },
        per_ms,
    ))
}

/// Verify a `STWOREC2` standalone block receipt against `expected_standalone_anchor` (`32 B`),
/// `expected_initial_chain`, and optional caller-pinned `expected_final_chain` and
/// `expected_public_io_digest`.
///
/// Internal: the building block of the chain and model verifiers, public only because the prover's
/// tests call it. For a receipt with an end stage it checks the proof and the digest only. It cannot check that
/// the prompt tokens and the output token are below the vocabulary: it receives `public_io_digest`, a Blake2s
/// digest of them, never the values, and the circuit bounds them only below the padded vocabulary (so
/// an output token in `[V, V_pad)` is accepted here). Bounding them in the circuit would change Tree 0 and
/// every anchor, which needs a new proof run. `verify_standalone_model_chain_against_anchor` (the `model`
/// command) and, for multi-token receipts, `verify_standalone_model_gen_chain_against_anchor` (the same command)
/// are the only statement-level entry points for receipts with an end stage, and they check every token.
pub fn verify_standalone_block_stark_against_anchor_and_io(
    expected_standalone_anchor: &[u8; 32],
    expected_initial_chain: &StandaloneChainLink,
    expected_final_chain: Option<&StandaloneChainLink>,
    expected_public_io_digest: Option<&[u8; 32]>,
    proof: &StandaloneBlockStarkProof,
) -> Result<StandaloneAuditVerifierRun, String> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let start = std::time::Instant::now();
    if *expected_standalone_anchor == [0u8; 32] {
        return Err("expected_standalone_anchor must not be all-zero".into());
    }
    let expected_cfg = proof.statement.validate()?;
    if &proof.statement.initial_chain != expected_initial_chain {
        return Err("standalone block initial_chain mismatch".into());
    }
    if let Some(exp_final) = expected_final_chain {
        if &proof.statement.final_chain != exp_final {
            return Err("standalone block final_chain mismatch".into());
        }
    }
    if let Some(exp_io) = expected_public_io_digest {
        if &proof.statement.public_io_digest != exp_io {
            return Err("standalone block public_io_digest mismatch".into());
        }
    }
    if proof.stark.commitments.is_empty() {
        return Err("standalone block STARK missing Tree 0 commitment".into());
    }
    let actual_anchor = compute_standalone_block_anchor(
        &proof.statement.policy_bytes(),
        &proof.statement.weights_and_boundaries_digest,
        &proof.stark.commitments[0].0,
    );
    if &actual_anchor != expected_standalone_anchor && outer_keep(outer_thread_mask(), outer_ids::D17) {
        return Err("standalone block 32B anchor mismatch (policy, weights, or Tree 0 forged)".into());
    }

    let pub_mem = build_block_public_memory_items(&proof.statement);
    let mut channel = Blake2sChannel::default();
    mix_standalone_block_statement(&mut channel, &proof.statement);
    let lookup_pow = (
        standalone_lookup_pow_bits_for_profile_id(&proof.statement.profile_id),
        proof.lookup_pow_nonce,
    );
    if is_pos2_profile(&proof.statement.profile_id) {
        verify_standalone_stark_core_pos2(
            expected_cfg,
            lookup_pow,
            proof.statement.audit_log_size,
            proof.statement.bridge_log_size,
            &pub_mem,
            &proof.claimed_sums,
            &proof.stark,
            &mut channel,
        )?;
    } else {
        verify_standalone_stark_core(
            expected_cfg,
            lookup_pow,
            proof.statement.audit_log_size,
            proof.statement.sched_log_size,
            proof.statement.round_log_size,
            proof.statement.bridge_log_size,
            &pub_mem,
            &proof.claimed_sums,
            &proof.stark,
            &mut channel,
        )?;
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let verify_ms = start.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let verify_ms = 0.0;
    Ok(StandaloneAuditVerifierRun {
        verify_ms,
        receipt_bytes: proof.stark.size_estimate(),
    })
}

/// Verify a public-tokens `STWOMDL1` whole-model standalone receipt (`N_blocks -> 1` + `Embedding`
/// + `LM Head`) against **only** `expected_full_stack_anchor` (`32 B`), `prompt_token_ids`, and
/// `emitted_token_ids`.
pub fn verify_whole_model_stark_against_anchor(
    expected_full_stack_anchor: &[u8; 32],
    prompt_token_ids: &[u32],
    emitted_token_ids: &[u32],
    proof: &WholeModelStandaloneStarkProof,
) -> Result<StandaloneAuditVerifierRun, String> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let start = std::time::Instant::now();
    if *expected_full_stack_anchor == [0u8; 32] {
        return Err("expected_full_stack_anchor must not be all-zero".into());
    }
    let expected_cfg = proof.statement.validate()?;
    if proof.statement.tokens_committed {
        return Err(
            "receipt uses salted_blake2s_committed_tokens; call verify_whole_model_committed_stark_against_anchor".into(),
        );
    }
    if proof.statement.prompt_token_ids.as_slice() != prompt_token_ids {
        return Err("whole-model prompt_token_ids mismatch".into());
    }
    if proof.statement.emitted_token_ids.as_slice() != emitted_token_ids {
        return Err("whole-model emitted_token_ids mismatch".into());
    }
    if proof.stark.commitments.is_empty() {
        return Err("whole-model STARK missing Tree 0 commitment".into());
    }
    let actual_anchor = compute_whole_model_full_stack_anchor(
        &proof.statement.policy_bytes(),
        &proof.statement.weights_and_boundaries_digest,
        &proof.stark.commitments[0].0,
    );
    if &actual_anchor != expected_full_stack_anchor {
        return Err("whole-model 32B full_stack_anchor mismatch (policy, weights, or Tree 0 forged)".into());
    }

    let pub_mem = build_whole_model_public_memory_items(&proof.statement);
    let mut channel = Blake2sChannel::default();
    mix_whole_model_statement(&mut channel, &proof.statement);
    verify_standalone_stark_core(
        expected_cfg,
        (0, 0),
        proof.statement.audit_log_size,
        proof.statement.sched_log_size,
        proof.statement.round_log_size,
        proof.statement.bridge_log_size,
        &pub_mem,
        &proof.claimed_sums,
        &proof.stark,
        &mut channel,
    )?;

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let verify_ms = start.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let verify_ms = 0.0;
    Ok(StandaloneAuditVerifierRun {
        verify_ms,
        receipt_bytes: proof.stark.size_estimate(),
    })
}

/// Verify a salted-`Blake2s`-committed-tokens `STWOMDL1` whole-model standalone receipt against
/// `expected_full_stack_anchor` (`32 B`), `expected_prompt_commitment` (`32 B`), and
/// `expected_emitted_commitment` (`32 B`) — zero plaintext prompt or emitted tokens on wire.
pub fn verify_whole_model_committed_stark_against_anchor(
    expected_full_stack_anchor: &[u8; 32],
    expected_prompt_commitment: &[u8; 32],
    expected_emitted_commitment: &[u8; 32],
    proof: &WholeModelStandaloneStarkProof,
) -> Result<StandaloneAuditVerifierRun, String> {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let start = std::time::Instant::now();
    if *expected_full_stack_anchor == [0u8; 32] {
        return Err("expected_full_stack_anchor must not be all-zero".into());
    }
    let expected_cfg = proof.statement.validate()?;
    if !proof.statement.tokens_committed {
        return Err(
            "receipt uses public_tokens; call verify_whole_model_stark_against_anchor".into(),
        );
    }
    if &proof.statement.prompt_commitment != expected_prompt_commitment {
        return Err("whole-model prompt_commitment mismatch".into());
    }
    if &proof.statement.emitted_commitment != expected_emitted_commitment {
        return Err("whole-model emitted_commitment mismatch".into());
    }
    if proof.stark.commitments.is_empty() {
        return Err("whole-model STARK missing Tree 0 commitment".into());
    }
    let actual_anchor = compute_whole_model_full_stack_anchor(
        &proof.statement.policy_bytes(),
        &proof.statement.weights_and_boundaries_digest,
        &proof.stark.commitments[0].0,
    );
    if &actual_anchor != expected_full_stack_anchor {
        return Err("whole-model 32B full_stack_anchor mismatch (policy, weights, or Tree 0 forged)".into());
    }

    let pub_mem = build_whole_model_public_memory_items(&proof.statement);
    let mut channel = Blake2sChannel::default();
    mix_whole_model_statement(&mut channel, &proof.statement);
    verify_standalone_stark_core(
        expected_cfg,
        (0, 0),
        proof.statement.audit_log_size,
        proof.statement.sched_log_size,
        proof.statement.round_log_size,
        proof.statement.bridge_log_size,
        &pub_mem,
        &proof.claimed_sums,
        &proof.stark,
        &mut channel,
    )?;

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let verify_ms = start.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let verify_ms = 0.0;
    Ok(StandaloneAuditVerifierRun {
        verify_ms,
        receipt_bytes: proof.stark.size_estimate(),
    })
}

fn push_claimed_sums(out: &mut Vec<u8>, sums: &StandaloneClaimedSums) {
    for sf in sums.as_array() {
        for c in sf.to_m31_array() {
            out.extend_from_slice(&c.0.to_le_bytes());
        }
    }
}

fn read_claimed_sums(raw: &[u8]) -> Result<StandaloneClaimedSums, String> {
    if raw.len() != 128 {
        return Err("invalid StandaloneClaimedSums byte slice".into());
    }
    let mut arr = [SecureField::zero(); 8];
    for i in 0..8 {
        let mut coords = [BaseField::zero(); 4];
        for c in 0..4 {
            let off = 16 * i + 4 * c;
            let u = u32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            if u >= P {
                return Err(format!("non-canonical claimed_sums M31 limb at {i}/{c}"));
            }
            coords[c] = BaseField::from_u32_unchecked(u);
        }
        arr[i] = SecureField::from_m31_array(coords);
    }
    Ok(StandaloneClaimedSums::from_array(arr))
}

pub fn standalone_block_receipt_bytes(proof: &StandaloneBlockStarkProof) -> Result<Vec<u8>, String> {
    let _ = proof.statement.validate()?;
    proof.claimed_sums.validate_canonical()?;
    let stark_bytes = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&proof.stark)
        .map_err(|e| format!("failed to serialize STWOREC2 StarkProof: {e}"))?;
    if stark_bytes.len() > STANDALONE_MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOREC2 StarkProof exceeds wire limit".into());
    }
    let s = &proof.statement;
    let mut out = Vec::with_capacity(384 + stark_bytes.len());
    out.extend_from_slice(STANDALONE_BLOCK_MAGIC);
    out.extend_from_slice(&STANDALONE_AUDIT_VERSION.to_le_bytes());
    out.extend_from_slice(&s.mode_tag.to_le_bytes());
    out.extend_from_slice(&(s.profile_id.len() as u64).to_le_bytes());
    out.extend_from_slice(s.profile_id.as_bytes());
    out.extend_from_slice(&(s.batch as u64).to_le_bytes());
    out.extend_from_slice(&(s.d_model as u64).to_le_bytes());
    out.extend_from_slice(&(s.num_layers as u64).to_le_bytes());
    out.push(s.cprs_weight_f);
    out.push(s.cprs_witness_f);
    out.push(u8::from(s.zk_trace_blinded));
    out.extend_from_slice(&s.stacked_expert_pack.to_le_bytes());
    out.extend_from_slice(&s.audit_log_size.to_le_bytes());
    out.extend_from_slice(&s.sched_log_size.to_le_bytes());
    out.extend_from_slice(&s.round_log_size.to_le_bytes());
    out.extend_from_slice(&s.bridge_log_size.to_le_bytes());
    push_transformer_dims(&mut out, &s.transformer, &s.ends);
    out.extend_from_slice(&s.weights_and_boundaries_digest);
    out.extend_from_slice(&s.initial_chain.layer.to_le_bytes());
    out.extend_from_slice(&s.initial_chain.prev_receipt_digest);
    out.extend_from_slice(&s.final_chain.layer.to_le_bytes());
    out.extend_from_slice(&s.final_chain.prev_receipt_digest);
    out.extend_from_slice(&s.public_io_digest);
    push_claimed_sums(&mut out, &proof.claimed_sums);
    if standalone_lookup_pow_bits_for_profile_id(&s.profile_id) > 0 {
        out.extend_from_slice(&proof.lookup_pow_nonce.to_le_bytes());
    } else if proof.lookup_pow_nonce != 0 {
        return Err("STWOREC2 lookup nonce must be zero when the profile has no lookup proof of work".into());
    }
    out.extend_from_slice(&(stark_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&stark_bytes);
    Ok(out)
}

pub fn decode_standalone_block_receipt(bytes: &[u8]) -> Result<StandaloneBlockStarkProof, String> {
    if bytes.len() > STANDALONE_MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOREC2 receipt exceeds wire limit".into());
    }
    let mut cur = 0usize;
    let take = |cur: &mut usize, n: usize, label: &str| -> Result<&[u8], String> {
        let end = cur.checked_add(n).ok_or_else(|| format!("overflow reading {label}"))?;
        if end > bytes.len() {
            return Err(format!("unexpected EOF reading {label}"));
        }
        let s = &bytes[*cur..end];
        *cur = end;
        Ok(s)
    };
    if take(&mut cur, 8, "magic")? != STANDALONE_BLOCK_MAGIC {
        return Err("invalid STWOREC2 magic".into());
    }
    let version = u32::from_le_bytes(take(&mut cur, 4, "version")?.try_into().unwrap());
    if version != STANDALONE_AUDIT_VERSION {
        return Err(format!("unsupported STWOREC2 version {version}"));
    }
    let mode_tag = u32::from_le_bytes(take(&mut cur, 4, "mode_tag")?.try_into().unwrap());
    let prof_len = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "prof_len")?.try_into().unwrap()))
        .map_err(|_| "profile_id len overflow".to_string())?;
    if prof_len > 64 {
        return Err("profile_id too long".into());
    }
    let profile_id = std::str::from_utf8(take(&mut cur, prof_len, "profile_id")?)
        .map_err(|_| "invalid utf-8 in profile_id".to_string())?
        .to_string();
    let batch = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "batch")?.try_into().unwrap()))
        .map_err(|_| "batch overflow".to_string())?;
    let d_model = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "d_model")?.try_into().unwrap()))
        .map_err(|_| "d_model overflow".to_string())?;
    let num_layers = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "num_layers")?.try_into().unwrap()))
        .map_err(|_| "num_layers overflow".to_string())?;
    let flags = take(&mut cur, 3, "flags")?;
    let cprs_weight_f = flags[0];
    let cprs_witness_f = flags[1];
    let zk_trace_blinded = match flags[2] {
        0 => false,
        1 => true,
        _ => return Err("non-canonical zk_trace_blinded byte".into()),
    };
    let stacked_expert_pack = u32::from_le_bytes(take(&mut cur, 4, "pack")?.try_into().unwrap());
    let audit_log_size = u32::from_le_bytes(take(&mut cur, 4, "audit_log")?.try_into().unwrap());
    let sched_log_size = u32::from_le_bytes(take(&mut cur, 4, "sched_log")?.try_into().unwrap());
    let round_log_size = u32::from_le_bytes(take(&mut cur, 4, "round_log")?.try_into().unwrap());
    let bridge_log_size = u32::from_le_bytes(take(&mut cur, 4, "bridge_log")?.try_into().unwrap());
    let kind = take(&mut cur, 1, "block_kind")?[0];
    let (transformer, ends) = match kind {
        0 => (None, None),
        1 | 2 => {
            let mut dim = |label: &str| -> Result<usize, String> {
                usize::try_from(u64::from_le_bytes(take(&mut cur, 8, label)?.try_into().unwrap()))
                    .map_err(|_| format!("{label} overflow"))
            };
            let dims = StandaloneTransformerDims {
                num_heads: dim("num_heads")?,
                num_kv_heads: dim("num_kv_heads")?,
                dense_ffn_dim: dim("dense_ffn_dim")?,
                moe_ffn_dim: dim("moe_ffn_dim")?,
                num_experts: dim("num_experts")?,
            };
            let ends = if kind == 2 {
                let flags = take(&mut cur, 1, "ends_flags")?[0];
                let vocab = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "vocab")?.try_into().unwrap()))
                    .map_err(|_| "vocab overflow".to_string())?;
                Some(StandaloneEndsDecl::from_flags(flags, vocab)?)
            } else {
                None
            };
            (Some(dims), ends)
        }
        _ => return Err("non-canonical STWOREC2 block kind byte".into()),
    };
    let weights_and_boundaries_digest: [u8; 32] = take(&mut cur, 32, "weights_dig")?.try_into().unwrap();
    let init_layer = u32::from_le_bytes(take(&mut cur, 4, "init_layer")?.try_into().unwrap());
    let init_prev: [u8; 32] = take(&mut cur, 32, "init_prev")?.try_into().unwrap();
    let final_layer = u32::from_le_bytes(take(&mut cur, 4, "final_layer")?.try_into().unwrap());
    let final_prev: [u8; 32] = take(&mut cur, 32, "final_prev")?.try_into().unwrap();
    let public_io_digest: [u8; 32] = take(&mut cur, 32, "pub_io")?.try_into().unwrap();
    let claimed_sums = read_claimed_sums(take(&mut cur, 128, "claimed_sums")?)?;
    let lookup_pow_nonce = if standalone_lookup_pow_bits_for_profile_id(&profile_id) > 0 {
        u64::from_le_bytes(take(&mut cur, 8, "lookup_pow_nonce")?.try_into().unwrap())
    } else {
        0
    };
    let stark_len = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "stark_len")?.try_into().unwrap()))
        .map_err(|_| "stark_len overflow".to_string())?;
    let stark_bytes = take(&mut cur, stark_len, "stark_bytes")?;
    if cur != bytes.len() {
        return Err("trailing bytes after STWOREC2 receipt".into());
    }
    let mut bcursor = std::io::Cursor::new(stark_bytes);
    let stark: StarkProof<Blake2sMerkleHasher> = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(stark_bytes.len() as u64)
        .deserialize_from(&mut bcursor)
        .map_err(|e| format!("failed to deserialize STWOREC2 StarkProof: {e}"))?;
    if bcursor.position() != stark_bytes.len() as u64 {
        return Err("trailing bytes inside STWOREC2 StarkProof bincode slice".into());
    }
    validate_standalone_stark_encodings(&stark)?;
    let statement = StandaloneBlockStatement {
        mode_tag,
        profile_id,
        batch,
        d_model,
        num_layers,
        cprs_weight_f,
        cprs_witness_f,
        stacked_expert_pack,
        zk_trace_blinded,
        audit_log_size,
        sched_log_size,
        round_log_size,
        bridge_log_size,
        transformer,
        ends,
        weights_and_boundaries_digest,
        initial_chain: StandaloneChainLink {
            layer: init_layer,
            prev_receipt_digest: init_prev,
        },
        final_chain: StandaloneChainLink {
            layer: final_layer,
            prev_receipt_digest: final_prev,
        },
        public_io_digest,
    };
    let _ = statement.validate()?;
    let decoded = StandaloneBlockStarkProof {
        statement,
        claimed_sums,
        stark,
        lookup_pow_nonce,
    };
    if standalone_block_receipt_bytes(&decoded)? != bytes {
        return Err("non-canonical STWOREC2 wire encoding".into());
    }
    Ok(decoded)
}

pub fn whole_model_receipt_bytes(proof: &WholeModelStandaloneStarkProof) -> Result<Vec<u8>, String> {
    let _ = proof.statement.validate()?;
    proof.claimed_sums.validate_canonical()?;
    let stark_bytes = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(&proof.stark)
        .map_err(|e| format!("failed to serialize STWOMDL1 StarkProof: {e}"))?;
    if stark_bytes.len() > STANDALONE_MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOMDL1 StarkProof exceeds wire limit".into());
    }
    let s = &proof.statement;
    let tok_bytes = if s.tokens_committed { 64 } else { 8 * s.batch };
    let mut out = Vec::with_capacity(384 + tok_bytes + stark_bytes.len());
    out.extend_from_slice(WHOLE_MODEL_MAGIC);
    out.extend_from_slice(&STANDALONE_AUDIT_VERSION.to_le_bytes());
    out.extend_from_slice(&s.mode_tag.to_le_bytes());
    out.extend_from_slice(&(s.profile_id.len() as u64).to_le_bytes());
    out.extend_from_slice(s.profile_id.as_bytes());
    out.extend_from_slice(&(s.batch as u64).to_le_bytes());
    out.extend_from_slice(&(s.num_blocks as u64).to_le_bytes());
    out.extend_from_slice(&(s.layers_per_block as u64).to_le_bytes());
    out.extend_from_slice(&(s.vocab_size as u64).to_le_bytes());
    out.extend_from_slice(&(s.d_model as u64).to_le_bytes());
    out.push(s.cprs_weight_f);
    out.push(s.cprs_witness_f);
    out.push(u8::from(s.zk_trace_blinded));
    out.push(u8::from(s.tokens_committed));
    out.extend_from_slice(&s.stacked_expert_pack.to_le_bytes());
    out.extend_from_slice(&s.audit_log_size.to_le_bytes());
    out.extend_from_slice(&s.sched_log_size.to_le_bytes());
    out.extend_from_slice(&s.round_log_size.to_le_bytes());
    out.extend_from_slice(&s.bridge_log_size.to_le_bytes());
    out.extend_from_slice(&s.weights_and_boundaries_digest);
    if s.tokens_committed {
        out.extend_from_slice(&s.prompt_commitment);
        out.extend_from_slice(&s.emitted_commitment);
    } else {
        for &tok in &s.prompt_token_ids {
            out.extend_from_slice(&tok.to_le_bytes());
        }
        for &tok in &s.emitted_token_ids {
            out.extend_from_slice(&tok.to_le_bytes());
        }
    }
    push_claimed_sums(&mut out, &proof.claimed_sums);
    out.extend_from_slice(&(stark_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&stark_bytes);
    Ok(out)
}

pub fn decode_whole_model_receipt(bytes: &[u8]) -> Result<WholeModelStandaloneStarkProof, String> {
    if bytes.len() > STANDALONE_MAX_WIRE_RECEIPT_BYTES {
        return Err("STWOMDL1 receipt exceeds wire limit".into());
    }
    let mut cur = 0usize;
    let take = |cur: &mut usize, n: usize, label: &str| -> Result<&[u8], String> {
        let end = cur.checked_add(n).ok_or_else(|| format!("overflow reading {label}"))?;
        if end > bytes.len() {
            return Err(format!("unexpected EOF reading {label}"));
        }
        let s = &bytes[*cur..end];
        *cur = end;
        Ok(s)
    };
    if take(&mut cur, 8, "magic")? != WHOLE_MODEL_MAGIC {
        return Err("invalid STWOMDL1 magic".into());
    }
    let version = u32::from_le_bytes(take(&mut cur, 4, "version")?.try_into().unwrap());
    if version != STANDALONE_AUDIT_VERSION {
        return Err(format!("unsupported STWOMDL1 version {version}"));
    }
    let mode_tag = u32::from_le_bytes(take(&mut cur, 4, "mode_tag")?.try_into().unwrap());
    let prof_len = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "prof_len")?.try_into().unwrap()))
        .map_err(|_| "profile_id len overflow".to_string())?;
    if prof_len > 64 {
        return Err("profile_id too long".into());
    }
    let profile_id = std::str::from_utf8(take(&mut cur, prof_len, "profile_id")?)
        .map_err(|_| "invalid utf-8 in profile_id".to_string())?
        .to_string();
    let batch = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "batch")?.try_into().unwrap()))
        .map_err(|_| "batch overflow".to_string())?;
    if batch > (1 << 20) {
        return Err("batch exceeds wire bound".into());
    }
    let num_blocks = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "num_blocks")?.try_into().unwrap()))
        .map_err(|_| "num_blocks overflow".to_string())?;
    let layers_per_block = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "lpb")?.try_into().unwrap()))
        .map_err(|_| "layers_per_block overflow".to_string())?;
    let vocab_size = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "vocab")?.try_into().unwrap()))
        .map_err(|_| "vocab_size overflow".to_string())?;
    let d_model = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "d_model")?.try_into().unwrap()))
        .map_err(|_| "d_model overflow".to_string())?;
    let flags = take(&mut cur, 4, "flags")?;
    let cprs_weight_f = flags[0];
    let cprs_witness_f = flags[1];
    let zk_trace_blinded = match flags[2] {
        0 => false,
        1 => true,
        _ => return Err("non-canonical zk_trace_blinded byte".into()),
    };
    let tokens_committed = match flags[3] {
        0 => false,
        1 => true,
        _ => return Err("non-canonical tokens_committed byte".into()),
    };
    let stacked_expert_pack = u32::from_le_bytes(take(&mut cur, 4, "pack")?.try_into().unwrap());
    let audit_log_size = u32::from_le_bytes(take(&mut cur, 4, "audit_log")?.try_into().unwrap());
    let sched_log_size = u32::from_le_bytes(take(&mut cur, 4, "sched_log")?.try_into().unwrap());
    let round_log_size = u32::from_le_bytes(take(&mut cur, 4, "round_log")?.try_into().unwrap());
    let bridge_log_size = u32::from_le_bytes(take(&mut cur, 4, "bridge_log")?.try_into().unwrap());
    let weights_and_boundaries_digest: [u8; 32] = take(&mut cur, 32, "weights_dig")?.try_into().unwrap();
    let (prompt_token_ids, emitted_token_ids, prompt_commitment, emitted_commitment) =
        if tokens_committed {
            let p_com: [u8; 32] = take(&mut cur, 32, "prompt_com")?.try_into().unwrap();
            let e_com: [u8; 32] = take(&mut cur, 32, "emitted_com")?.try_into().unwrap();
            (Vec::new(), Vec::new(), p_com, e_com)
        } else {
            let mut p_toks = Vec::with_capacity(batch);
            for _ in 0..batch {
                p_toks.push(u32::from_le_bytes(
                    take(&mut cur, 4, "prompt_tok")?.try_into().unwrap(),
                ));
            }
            let mut e_toks = Vec::with_capacity(batch);
            for _ in 0..batch {
                e_toks.push(u32::from_le_bytes(
                    take(&mut cur, 4, "emitted_tok")?.try_into().unwrap(),
                ));
            }
            (p_toks, e_toks, [0u8; 32], [0u8; 32])
        };
    let claimed_sums = read_claimed_sums(take(&mut cur, 128, "claimed_sums")?)?;
    let stark_len = usize::try_from(u64::from_le_bytes(take(&mut cur, 8, "stark_len")?.try_into().unwrap()))
        .map_err(|_| "stark_len overflow".to_string())?;
    let stark_bytes = take(&mut cur, stark_len, "stark_bytes")?;
    if cur != bytes.len() {
        return Err("trailing bytes after STWOMDL1 receipt".into());
    }
    let mut bcursor = std::io::Cursor::new(stark_bytes);
    let stark: StarkProof<Blake2sMerkleHasher> = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(stark_bytes.len() as u64)
        .deserialize_from(&mut bcursor)
        .map_err(|e| format!("failed to deserialize STWOMDL1 StarkProof: {e}"))?;
    if bcursor.position() != stark_bytes.len() as u64 {
        return Err("trailing bytes inside STWOMDL1 StarkProof bincode slice".into());
    }
    validate_standalone_stark_encodings(&stark)?;
    let statement = WholeModelStatement {
        mode_tag,
        profile_id,
        batch,
        num_blocks,
        layers_per_block,
        vocab_size,
        d_model,
        cprs_weight_f,
        cprs_witness_f,
        stacked_expert_pack,
        zk_trace_blinded,
        tokens_committed,
        audit_log_size,
        sched_log_size,
        round_log_size,
        bridge_log_size,
        weights_and_boundaries_digest,
        prompt_token_ids,
        emitted_token_ids,
        prompt_commitment,
        emitted_commitment,
    };
    let _ = statement.validate()?;
    let decoded = WholeModelStandaloneStarkProof {
        statement,
        claimed_sums,
        stark,
    };
    if whole_model_receipt_bytes(&decoded)? != bytes {
        return Err("non-canonical STWOMDL1 wire encoding".into());
    }
    Ok(decoded)
}

// ============================================================================
// Canonical Starknet Calldata Packer & Request-Bound Digest Helpers
// ============================================================================

/// Convert a 32-byte Blake2s digest into 8 little-endian `u32` words matching Cairo `[u32; 8]`.
pub fn blake2s_bytes_to_u32_words(bytes: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    out
}

/// Convert a Cairo short-string (`<= 31` ASCII bytes, big-endian integer) into 32-byte
/// little-endian `u256` bytes matching `append_u256_le(model_id.into())` in `beacon_slasher.cairo`.
pub fn cairo_short_string_to_u256_le(s: &str) -> Result<[u8; 32], String> {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 31 {
        return Err("Cairo short string must be 1..=31 bytes".into());
    }
    let mut be = [0u8; 32];
    be[32 - b.len()..].copy_from_slice(b);
    be.reverse();
    Ok(be)
}

pub fn starknet_compute_public_io_binding_digest(prompts: &[u32], emitted: &[u32]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"SPARSEPROVE_ESCROW_IO_PUB_V1");
    h.update((prompts.len() as u64).to_le_bytes());
    for &p in prompts {
        h.update(p.to_le_bytes());
    }
    h.update((emitted.len() as u64).to_le_bytes());
    for &e in emitted {
        h.update(e.to_le_bytes());
    }
    h.finalize().into()
}

pub fn starknet_compute_committed_io_binding_digest(
    prompt_commitment: &[u8; 32],
    emitted_commitment: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"SPARSEPROVE_ESCROW_IO_COM_V1");
    h.update(prompt_commitment);
    h.update(emitted_commitment);
    h.finalize().into()
}

pub fn starknet_compute_escrow_request_digest(
    model_id_le32: &[u8; 32],
    commitment_id_le32: &[u8; 32],
    io_binding_digest: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"SPARSEPROVE_ESCROW_REQ_V1");
    h.update(model_id_le32);
    h.update(commitment_id_le32);
    h.update(io_binding_digest);
    h.finalize().into()
}

pub fn starknet_compute_request_bound_initial_chain_digest(
    req_digest: &[u8; 32],
    block_index: u32,
    prev_state_digest: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"SPARSEPROVE_REQ_BOUND_INIT_V1");
    h.update(req_digest);
    h.update(block_index.to_le_bytes());
    h.update(prev_state_digest);
    h.finalize().into()
}

pub fn starknet_compute_activation_leaf_hash(
    block_index: u32,
    start_layer: u32,
    end_layer: u32,
    prev_state_digest: &[u8; 32],
    final_chain_digest: &[u8; 32],
    public_io_digest: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"SPARSEPROVE_ACT_LEAF_V1");
    h.update(block_index.to_le_bytes());
    h.update(start_layer.to_le_bytes());
    h.update(end_layer.to_le_bytes());
    h.update(prev_state_digest);
    h.update(final_chain_digest);
    h.update(public_io_digest);
    h.finalize().into()
}

fn push_u64_words(out: &mut Vec<u32>, v: u64) {
    out.push((v & 0xFFFF_FFFF) as u32);
    out.push((v >> 32) as u32);
}

fn push_qm31_words(out: &mut Vec<u32>, q: SecureField) {
    let coords = q.to_m31_array();
    out.push(coords[0].0);
    out.push(coords[1].0);
    out.push(coords[2].0);
    out.push(coords[3].0);
}

fn push_digest_words(out: &mut Vec<u32>, bytes: &[u8; 32]) {
    out.extend_from_slice(&blake2s_bytes_to_u32_words(bytes));
}

fn push_fri_layer_words(
    out: &mut Vec<u32>,
    layer: &stwo::core::fri::FriLayerProof<Blake2sMerkleHasher>,
) -> Result<(), String> {
    push_digest_words(out, &layer.commitment.0);
    let n_wit = u32::try_from(layer.fri_witness.len())
        .map_err(|_| "FRI witness length overflows u32".to_string())?;
    out.push(n_wit);
    for &q in &layer.fri_witness {
        push_qm31_words(out, q);
    }
    let n_hashes = u32::try_from(layer.decommitment.hash_witness.len())
        .map_err(|_| "FRI hash_witness length overflows u32".to_string())?;
    out.push(n_hashes);
    for h in &layer.decommitment.hash_witness {
        push_digest_words(out, &h.0);
    }
    Ok(())
}

/// Serialize a `StarkProof<Blake2sMerkleHasher>` into the exact canonical `u32` word stream
/// consumed by `cairo/src/circle_stark_verifier.cairo::verify_standalone_circle_stark`.
pub fn pack_stark_proof_to_cairo_u32_words(
    stark: &StarkProof<Blake2sMerkleHasher>,
) -> Result<Vec<u32>, String> {
    validate_standalone_stark_encodings(stark)?;
    if stark.config.lifting_log_size.is_some() {
        return Err("standalone Cairo packer expects lifting_log_size == None".into());
    }
    let mut out = Vec::with_capacity(4096);

    // 1. PcsConfig (10 u32 words)
    out.push(stark.config.pow_bits);
    out.push(stark.config.composition_pow_bits);
    out.push(stark.config.oods_pow_bits);
    out.push(stark.config.fri_batch_pow_bits);
    out.push(stark.config.fri_config.log_blowup_factor);
    out.push(stark.config.fri_config.log_last_layer_degree_bound);
    out.push(
        u32::try_from(stark.config.fri_config.n_queries)
            .map_err(|_| "n_queries overflows u32".to_string())?,
    );
    out.push(stark.config.fri_config.fold_step);
    out.push(stark.config.fri_config.first_fold_pow_bits);
    out.push(0u32); // has_lifting_log_size = 0

    // 2. PoW nonces (4 x u64 = 8 u32 words)
    push_u64_words(&mut out, stark.composition_proof_of_work);
    push_u64_words(&mut out, stark.oods_proof_of_work);
    push_u64_words(&mut out, stark.fri_batch_proof_of_work);
    push_u64_words(&mut out, stark.proof_of_work);

    // 3. sampled_values
    out.push(
        u32::try_from(stark.sampled_values.len())
            .map_err(|_| "sampled_values tree count overflows u32".to_string())?,
    );
    for tree in stark.sampled_values.iter() {
        out.push(
            u32::try_from(tree.len())
                .map_err(|_| "sampled_values column count overflows u32".to_string())?,
        );
        for col in tree.iter() {
            out.push(
                u32::try_from(col.len())
                    .map_err(|_| "sampled_values sample count overflows u32".to_string())?,
            );
            for &val in col.iter() {
                push_qm31_words(&mut out, val);
            }
        }
    }

    // 4. decommitments
    out.push(
        u32::try_from(stark.decommitments.len())
            .map_err(|_| "decommitments tree count overflows u32".to_string())?,
    );
    for dec in stark.decommitments.iter() {
        out.push(
            u32::try_from(dec.hash_witness.len())
                .map_err(|_| "decommitment hash_witness count overflows u32".to_string())?,
        );
        for h in &dec.hash_witness {
            push_digest_words(&mut out, &h.0);
        }
    }

    // 5. queried_values (column-major per tree)
    out.push(
        u32::try_from(stark.queried_values.len())
            .map_err(|_| "queried_values tree count overflows u32".to_string())?,
    );
    for tree in stark.queried_values.iter() {
        let n_cols = u32::try_from(tree.len())
            .map_err(|_| "queried_values column count overflows u32".to_string())?;
        let n_rows = if let Some(first_col) = tree.first() {
            let r = first_col.len();
            for col in tree.iter() {
                if col.len() != r {
                    return Err("ragged queried_values matrix within tree".into());
                }
            }
            u32::try_from(r).map_err(|_| "queried_values row count overflows u32".to_string())?
        } else {
            0u32
        };
        out.push(n_cols);
        out.push(n_rows);
        for col in tree.iter() {
            for &v in col.iter() {
                out.push(v.0);
            }
        }
    }

    // 6. fri_proof
    push_fri_layer_words(&mut out, &stark.fri_proof.first_layer)?;
    out.push(
        u32::try_from(stark.fri_proof.inner_layers.len())
            .map_err(|_| "FRI inner_layers count overflows u32".to_string())?,
    );
    for inner in &stark.fri_proof.inner_layers {
        push_fri_layer_words(&mut out, inner)?;
    }
    let last_coeffs: &[SecureField] = &stark.fri_proof.last_layer_poly;
    out.push(
        u32::try_from(last_coeffs.len())
            .map_err(|_| "FRI last_layer_poly length overflows u32".to_string())?,
    );
    for &c in last_coeffs {
        push_qm31_words(&mut out, c);
    }
    out.push(
        u32::try_from(stark.fri_proof.fold_proof_of_work.len())
            .map_err(|_| "FRI fold_proof_of_work count overflows u32".to_string())?,
    );
    for &nonce in &stark.fri_proof.fold_proof_of_work {
        push_u64_words(&mut out, nonce);
    }

    Ok(out)
}

/// Pack canonical `u32` words into `(low: u128, high: u128)` `felt252` integers (`< 2^224`)
/// matching `unpack_canonical_u32_words` in `cairo/src/circle_stark_verifier.cairo`.
pub fn pack_u32_words_to_cairo_felts_u256(words: &[u32]) -> Result<Vec<(u128, u128)>, String> {
    let total_words = u32::try_from(words.len())
        .map_err(|_| "word count overflows u32 header".to_string())?;
    let mut felts = Vec::with_capacity(1 + words.len().div_ceil(7));
    felts.push((total_words as u128, 0u128));
    for chunk in words.chunks(7) {
        let mut w = [0u32; 7];
        w[..chunk.len()].copy_from_slice(chunk);
        let low = (w[0] as u128)
            | ((w[1] as u128) << 32)
            | ((w[2] as u128) << 64)
            | ((w[3] as u128) << 96);
        let high = (w[4] as u128) | ((w[5] as u128) << 32) | ((w[6] as u128) << 64);
        felts.push((low, high));
    }
    Ok(felts)
}

/// Pack a `StarkProof<Blake2sMerkleHasher>` directly into canonical Starknet `felt252` calldata
/// words `(low: u128, high: u128)` (`value = low + high * 2^128 < 2^224`).
pub fn pack_stark_proof_to_cairo_felts_u256(
    stark: &StarkProof<Blake2sMerkleHasher>,
) -> Result<Vec<(u128, u128)>, String> {
    let words = pack_stark_proof_to_cairo_u32_words(stark)?;
    pack_u32_words_to_cairo_felts_u256(&words)
}

#[cfg(test)]
mod layer_canonicality_tests {
    use super::*;

    fn sample_stmt(init_layer: u32, num_layers: usize, final_layer: u32) -> StandaloneBlockStatement {
        StandaloneBlockStatement {
            mode_tag: 11,
            profile_id: "dev13".to_string(),
            batch: 4,
            d_model: 16,
            num_layers,
            cprs_weight_f: 4,
            cprs_witness_f: 4,
            stacked_expert_pack: 2,
            zk_trace_blinded: true,
            audit_log_size: 10,
            sched_log_size: 10,
            round_log_size: 10,
            bridge_log_size: 10,
            transformer: None,
            ends: None,
            weights_and_boundaries_digest: [0x11; 32],
            initial_chain: StandaloneChainLink {
                layer: init_layer,
                prev_receipt_digest: [0x22; 32],
            },
            final_chain: StandaloneChainLink {
                layer: final_layer,
                prev_receipt_digest: [0x33; 32],
            },
            public_io_digest: [0x44; 32],
        }
    }

    /// A transformer statement at every shape limit: 1,024 tokens, `d_model = 8192`,
    /// `head_dim = 128`, a 16,384-wide dense `W_down`.
    fn limit_transformer_stmt() -> StandaloneBlockStatement {
        let mut s = sample_stmt(0, 2, 2);
        s.batch = MAX_TRANSFORMER_BATCH;
        s.d_model = MAX_TRANSFORMER_D_MODEL;
        s.stacked_expert_pack = 8;
        s.transformer = Some(StandaloneTransformerDims {
            num_heads: 64,
            num_kv_heads: 8,
            dense_ffn_dim: MAX_TRANSFORMER_FFN_DIM,
            moe_ffn_dim: 8192,
            num_experts: 128,
        });
        s
    }

    fn dims(s: &mut StandaloneBlockStatement) -> &mut StandaloneTransformerDims {
        s.transformer.as_mut().expect("transformer statement")
    }

    /// The verifier rejects, from the statement alone, every transformer
    /// shape for which one accumulator of the design can leave the field window.
    #[test]
    fn transformer_statement_rejects_every_shape_past_a_no_wrap_bound() {
        assert!(limit_transformer_stmt().validate().is_ok());
        let mut experts_at_cap = limit_transformer_stmt();
        dims(&mut experts_at_cap).num_experts = MAX_TRANSFORMER_EXPERTS;
        assert!(experts_at_cap.validate().is_ok());

        type Edit = fn(&mut StandaloneBlockStatement);
        let cases: [(&str, Edit); 10] = [
            ("head_dim 256", |s| dims(s).num_heads = 32),
            ("head_dim 512", |s| dims(s).num_heads = 16),
            ("batch 2048", |s| s.batch = 2048),
            ("d_model 16384", |s| {
                s.d_model = 16384;
                dims(s).num_heads = 128;
            }),
            ("dense_ffn_dim 32768", |s| dims(s).dense_ffn_dim = 32768),
            ("moe_ffn_dim 32768", |s| dims(s).moe_ffn_dim = 32768),
            ("kv_heads <= heads", |s| dims(s).num_kv_heads = 128),
            ("num_experts 131072", |s| dims(s).num_experts = 1 << 17),
            ("stacked_expert_pack 256", |s| s.stacked_expert_pack = 256),
            ("num_layers 257", |s| {
                s.num_layers = 257;
                s.final_chain.layer = 257;
            }),
        ];
        for (want, edit) in cases {
            let mut s = limit_transformer_stmt();
            edit(&mut s);
            let err = s.validate().expect_err(want);
            assert!(err.contains(want), "{want}: unexpected error: {err}");
        }

        // Each dimension is bound in the policy bytes, so in the anchor.
        let base = limit_transformer_stmt().policy_bytes();
        let mut two_matrix = limit_transformer_stmt();
        two_matrix.transformer = None;
        assert_ne!(two_matrix.policy_bytes(), base);
        let dim_edits: [fn(&mut StandaloneTransformerDims); 5] = [
            |t| t.num_heads = 32,
            |t| t.num_kv_heads = 4,
            |t| t.dense_ffn_dim = 8192,
            |t| t.moe_ffn_dim = 4096,
            |t| t.num_experts = 64,
        ];
        for edit in dim_edits {
            let mut s = limit_transformer_stmt();
            edit(dims(&mut s));
            assert_ne!(s.policy_bytes(), base);
        }
    }

    #[test]
    fn standalone_block_statement_rejects_mod_p_layer_aliasing_and_mismatch() {
        assert!(sample_stmt(0, 2, 2).validate().is_ok());
        // Mod-P alias `2 + P` (same M31 element on MemoryBus!) MUST be rejected:
        assert!(sample_stmt(0, 2, 2 + P).validate().is_err());
        // Non-matching `final != initial + num_layers` MUST be rejected:
        assert!(sample_stmt(0, 2, 3).validate().is_err());
        // Wrap across `P` (`initial = P - 1`, `final = P + 1` or `1`) MUST be rejected:
        assert!(sample_stmt(P - 1, 2, P + 1).validate().is_err());
        assert!(sample_stmt(P - 1, 2, 1).validate().is_err());
        // `initial >= P` or `u32::MAX` overflow MUST be rejected without panic:
        assert!(sample_stmt(P, 2, P + 2).validate().is_err());
        assert!(sample_stmt(u32::MAX, 2, 1).validate().is_err());

        // `batch > 2048`, invalid `d_model`, and invalid `stacked_expert_pack` MUST be rejected:
        let mut s_big_b = sample_stmt(0, 2, 2);
        s_big_b.batch = 4096;
        assert!(s_big_b.validate().is_err());

        let mut s_bad_d = sample_stmt(0, 2, 2);
        s_bad_d.d_model = 2;
        assert!(s_bad_d.validate().is_err());
        s_bad_d.d_model = 32;
        assert_ne!(
            s_bad_d.policy_bytes(),
            sample_stmt(0, 2, 2).policy_bytes(),
            "d_model must be cryptographically bound in policy_bytes"
        );

        let mut s_bad_pack = sample_stmt(0, 2, 2);
        s_bad_pack.stacked_expert_pack = 0;
        assert!(s_bad_pack.validate().is_err());
        s_bad_pack.stacked_expert_pack = 3;
        assert!(s_bad_pack.validate().is_err());
    }

    fn ends_stmt(embed: bool, head: bool, vocab: usize, num_layers: usize) -> StandaloneBlockStatement {
        let mut s = limit_transformer_stmt();
        s.num_layers = num_layers;
        s.final_chain.layer = num_layers as u32;
        s.ends = Some(StandaloneEndsDecl {
            embed,
            head,
            vocab_size: vocab,
            multi_token: false,
        });
        s
    }

    /// Kind byte 2: an end stage needs a transformer, flags 1 to 3,
    /// a vocabulary in `[2, 2^20]`; zero layers only with an end stage; flags and vocabulary bound in
    /// the policy bytes.
    #[test]
    fn ends_statement_kind_2_bounds_and_policy_bytes() {
        assert!(ends_stmt(true, false, 202_048, 0).validate().is_ok());
        assert!(ends_stmt(false, true, 202_048, 0).validate().is_ok());
        assert!(ends_stmt(true, true, MAX_ENDS_VOCAB, 2).validate().is_ok());
        let cases: [(&str, StandaloneBlockStatement); 4] = [
            ("outside [2, 1048576]", ends_stmt(true, false, MAX_ENDS_VOCAB + 1, 0)),
            ("outside [2, 1048576]", ends_stmt(false, true, 1, 0)),
            ("the embedding, the head or both", ends_stmt(false, false, 64, 0)),
            ("num_layers 0 outside 1..", {
                let mut s = limit_transformer_stmt();
                s.num_layers = 0;
                s.final_chain.layer = 0;
                s
            }),
        ];
        for (want, st) in cases {
            let err = st.validate().expect_err(want);
            assert!(err.contains(want), "{want}: unexpected error: {err}");
        }
        let mut no_tfm = ends_stmt(true, false, 64, 0);
        no_tfm.transformer = None;
        assert!(no_tfm.validate().unwrap_err().contains("needs a transformer"));
        assert!(StandaloneEndsDecl::from_flags(0, 64).is_err() && StandaloneEndsDecl::from_flags(4, 64).is_err());
        assert!(StandaloneEndsDecl::from_flags(8, 64).is_err());
        assert_eq!(
            StandaloneEndsDecl::from_flags(3, 64).unwrap(),
            StandaloneEndsDecl {
                embed: true,
                head: true,
                vocab_size: 64,
                multi_token: false,
            }
        );
        assert_eq!(
            StandaloneEndsDecl::from_flags(7, 64).unwrap(),
            StandaloneEndsDecl {
                embed: true,
                head: true,
                vocab_size: 64,
                multi_token: true,
            }
        );
        // Both statements have batch 32, so only `multi_token` (flag bit 4) separates their policy bytes (with
        // batches 32 and 1024 the assertion held even without the flag).
        let mut single = ends_stmt(true, true, 202_048, 2);
        single.batch = 32;
        let mut multi_ok = single.clone();
        multi_ok.ends.as_mut().unwrap().multi_token = true;
        assert!(multi_ok.validate().is_ok());
        assert_ne!(multi_ok.policy_bytes(), single.policy_bytes(), "multi_token bound in the policy bytes");
        let mut multi_big = ends_stmt(false, true, MAX_ENDS_VOCAB, 0);
        multi_big.batch = 128; // 128 * 2^20 = 2^27 > MAX_MULTI_HEAD_BV (2^26)
        multi_big.ends.as_mut().unwrap().multi_token = true;
        assert!(multi_big.validate().unwrap_err().contains("MAX_MULTI_HEAD_BV"));

        let base = ends_stmt(true, false, 202_048, 0).policy_bytes();
        let mut kind1 = ends_stmt(true, false, 202_048, 0);
        kind1.ends = None;
        assert_ne!(kind1.policy_bytes(), base, "kind 1 and kind 2 differ");
        assert_ne!(ends_stmt(false, true, 202_048, 0).policy_bytes(), base, "flags bound");
        assert_ne!(ends_stmt(true, true, 202_048, 0).policy_bytes(), base, "flags bound");
        assert_ne!(ends_stmt(true, false, 202_049, 0).policy_bytes(), base, "vocab bound");
        let e = ends_stmt(true, false, 202_048, 0);
        assert_eq!((e.ends.unwrap().vocab_pad(), e.ends.unwrap().log_vocab()), (262_144, 18));
    }

    /// The public digest of a block with an end (7.2) binds the flags, each token, the output token
    /// and each commitment, and never equals the digest of a block without ends.
    #[test]
    fn ends_public_io_digest_binds_every_part() {
        let toks = [200_000u32, 5, 9, 1];
        let com = [0x5Au8; 32];
        let y = [7u32];
        let base = compute_standalone_ends_public_io_digest(3, StandaloneEndsIo::Tokens(&toks), StandaloneEndsIo::Tokens(&y));
        let mut t2 = toks;
        t2[2] = 10;
        assert_ne!(compute_standalone_ends_public_io_digest(3, StandaloneEndsIo::Tokens(&t2), StandaloneEndsIo::Tokens(&y)), base);
        assert_ne!(compute_standalone_ends_public_io_digest(3, StandaloneEndsIo::Tokens(&toks), StandaloneEndsIo::Tokens(&[8])), base);
        assert_ne!(compute_standalone_ends_public_io_digest(1, StandaloneEndsIo::Tokens(&toks), StandaloneEndsIo::Tokens(&y)), base);
        let emb = compute_standalone_ends_public_io_digest(1, StandaloneEndsIo::Tokens(&toks), StandaloneEndsIo::Commitment(&com));
        let mut com2 = com;
        com2[31] ^= 1;
        assert_ne!(compute_standalone_ends_public_io_digest(1, StandaloneEndsIo::Tokens(&toks), StandaloneEndsIo::Commitment(&com2)), emb);
        let head = compute_standalone_ends_public_io_digest(2, StandaloneEndsIo::Commitment(&com), StandaloneEndsIo::Tokens(&y));
        assert_ne!(head, compute_standalone_block_public_io_digest(&com, &com));
        assert_ne!(emb, head);
    }

    #[test]
    fn model_anchor_binds_count_and_order_and_differs_from_a_chain_anchor() {
        let (a, b, c) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let abc = compute_standalone_model_anchor(&[a, b, c]);
        assert_ne!(abc, compute_standalone_model_anchor(&[a, c, b]));
        assert_ne!(abc, compute_standalone_model_anchor(&[a, b]));
        assert_ne!(abc, compute_standalone_model_anchor(&[b, c]));
        assert_ne!(abc, compute_standalone_chain_anchor(&[a, b, c]));
    }

    #[test]
    fn whole_model_committed_tokens_requires_mode_11_zk_blinding() {
        let mut wm = WholeModelStatement {
            mode_tag: 11,
            profile_id: "dev13".to_string(),
            batch: 2,
            num_blocks: 1,
            layers_per_block: 2,
            vocab_size: 16,
            d_model: 8,
            cprs_weight_f: 2,
            cprs_witness_f: 2,
            stacked_expert_pack: 2,
            zk_trace_blinded: true,
            tokens_committed: true,
            audit_log_size: 10,
            sched_log_size: 10,
            round_log_size: 10,
            bridge_log_size: 10,
            weights_and_boundaries_digest: [0x11; 32],
            prompt_token_ids: Vec::new(),
            emitted_token_ids: Vec::new(),
            prompt_commitment: [0x22; 32],
            emitted_commitment: [0x33; 32],
        };
        assert!(wm.validate().is_ok());
        wm.zk_trace_blinded = false;
        assert!(wm.validate().is_err());
        wm.zk_trace_blinded = true;
        wm.mode_tag = 10;
        assert!(wm.validate().is_err());
    }
}
