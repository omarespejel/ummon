use core::array;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::OnceLock;

use blake2::{Blake2s256, Digest};
use num_traits::{One, Zero};
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::{BaseField, P};
use stwo::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use stwo::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};

use crate::cprs::{
    stwo_point_to_cprs, to_cprs_qm31, verify_heterogeneous_strict_context, verify_strict_context,
    Params,
};
use crate::stage_b::StageBDeferredCprsClaims;
use crate::{
    acc_to_secure, bytemuck_cast_i8, dot, eq, eq_evals, eq_evals_i64, hash_i16_le_slice,
    i64_mod_p, i64_wire_word, i8_to_m31, mle_at, partially_verify,
    partially_verify_batch_with_libra_masks, partially_verify_with_lo_basis, put_bytes,
    put_secure_field, put_secure_fields, put_sumcheck, put_u32, put_u64, zk_lo_basis_omega_scale,
    Gate, GkrArtifact, GkrBatchProof, GkrMask, ReceiptReader, SecurityProfile, SumcheckProof,
    MAX_WIRE_RECEIPT_BYTES,
};

pub const PRODUCTION100_TRANSCRIPT_VERSION: &str = "production100-v3";
const TRANSCRIPT_V3_PREFIX: &[u8] = b"sparseprove-production100-transcript-v3";
const MIX_FRAME_TAG: &[u8] = b"mix";
const DRAW_FRAME_TAG: &[u8] = b"draw";
const POW_FRAME_TAG: &[u8] = b"pow";

const REGISTERED_BATCH_NONLINEAR_DOMAIN_TAG: u64 = 0x5354_4741_5245_4731;
const REGISTERED_ROUTER_DOMAIN_TAG: u64 = 0x4752_5044_5247_5431;
const SLOT_RLC_DOMAIN_TAG: u64 = 0x534c_4f54_524c_4302;
pub const STAGE_A_REGISTERED_RECEIPT_MAGIC: &[u8; 8] = b"STWOREG1";
pub const STAGE_A_REGISTERED_RECEIPT_WIRE_VERSION: u32 = 6;
pub const STAGE_A_RECEIPT_MODE_STARK: u32 = STAGE_A_REGISTERED_RECEIPT_WIRE_VERSION;
pub const STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS: u32 = 7;
pub const STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8: u32 = 8;
const PUBLIC_WITNESS_MODE_DOMAIN_TAG: u64 = 0x5354_4741_5055_4257;

const CPRS_CLASS_TAG_LAYER: u8 = 0x04;
const LAYER_SLOT_ROLES: [u8; 7] = [0x01, 0x10, 0x11, 0x12, 0x20, 0x21, 0x22];
const STAGE_AB_LAYER_CPRS_OUTER_DOMAIN: &[u8] = b"stwo-ml:stage-ab:cprs-layer-outer:v1";

pub const WIDE_LAYOUT_MAX_LOG_COLUMNS: usize = 16;
pub const WIDE_LAYOUT_LOG_COLUMNS: usize = 7;
const WIDE_LAYOUT_MIN_LOG_ROWS: usize = 5;
pub const TABLE_BITS: usize = 12;
pub const TABLE_LEN: usize = 1 << TABLE_BITS;
pub const OUTPUT_BITS: usize = 12;
pub const WIDTH_TABLE_MAX_BITS: usize = 11;
pub const DOWN_ENCODE_BITS: usize = 31;
pub const DOWN_BIAS: i128 = 1 << 30;
pub const RHS_CARRY_BITS: usize = 2;
pub const ROUTE_SCALE: i128 = 1 << 15;
const Q15_ONE: i64 = 1 << 15;
pub const STAGE_A_SOFTMAX_MAX_LOGIT_SPREAD: i64 = 1 << 11;
pub const STAGE_A_POLICY_ID: &str = "deepseek-moe-stage-a-i12-swiglu-v1";
pub const STAGE_A_DEEPPROVE_COMPARISON_POLICY_ID: &str =
    "deepseek-moe-stage-a-i12-swiglu-deepprove-half-up-comparison-v1";
pub const STAGE_A_TRAINED_I12_Q7_POLICY_ID: &str =
    "deepseek-moe-stage-a-trained-i12-q7-proof-friendly-swiglu-v1";
pub const STAGE_A_TRAINED_I12_Q7_HALF_UP_POLICY_ID: &str =
    "deepseek-moe-stage-a-trained-i12-q7-half-up-comparison-v1";
pub const STAGE_A_TABLE_SHA256: &str =
    "9d7230759fc260cb635edf0bbb1f7c2d993d436543ab9229ecc83954f07e0221";
pub const STAGE_A_I12_Q7_TABLE_SHA256: &str =
    "ab2d84dc7e4ddab7c417e3ee9d4368eca8baf10c6d3a2cafee28eef30ca0b183";
const I12_Q8_TABLE_DATA: &str =
    include_str!("../../docs/evidence/stage_a_i12_silu_exp_table_v1.tsv");
const I12_Q7_TABLE_DATA: &str =
    include_str!("../../docs/evidence/stage_a_i12_q7_silu_exp_table_v1.tsv");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TranscriptScope {
    Bootstrap = 0,
    Statement = 1,
    PreprocessedCommitment = 2,
    OriginalCommitment = 3,
    RouterBinding = 4,
    BatchAggregation = 5,
    RegularGateUp = 6,
    SharedGateUp = 7,
    RawProjection = 8,
    DownProjection = 9,
    RegularDown = 10,
    SharedDown = 11,
    RegularGrouped = 12,
    SharedGrouped = 13,
    Lookup = 14,
    InteractionCommitment = 15,
    Stark = 16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TranscriptOperation {
    Scope = 1,
    MixU32s = 2,
    MixFelts = 3,
    MixU64 = 4,
    DrawSecureFelt = 5,
    DrawSecureFelts = 6,
    DrawU32s = 7,
    VerifyPow = 8,
    MixMerkleRoot = 9,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptEvent {
    pub sequence: u64,
    pub scope: TranscriptScope,
    pub operation: TranscriptOperation,
    pub payload_items: u64,
    pub raw_hashes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranscriptCensus {
    pub event_count: usize,
    pub challenge_draw_count: usize,
    pub challenge_element_count: u64,
    pub raw_hash_count: u64,
    scope_event_counts: [u64; 17],
    operation_event_counts: [u64; 10],
    pub trace_digest: [u8; 32],
}

impl TranscriptCensus {
    fn from_events(events: &[TranscriptEvent]) -> Self {
        let mut hasher = Blake2sHasher::new();
        hasher.update(b"sparseprove-production100-transcript-census-v1");
        let mut scope_event_counts = [0u64; 17];
        let mut operation_event_counts = [0u64; 10];
        let mut challenge_draw_count = 0usize;
        let mut challenge_element_count = 0u64;
        let mut raw_hash_count = 0u64;
        let mut buf = [0u8; 1024];
        for chunk in events.chunks(32) {
            let mut off = 0usize;
            for event in chunk {
                buf[off..off + 8].copy_from_slice(&event.sequence.to_le_bytes());
                buf[off + 8..off + 12].copy_from_slice(&(event.scope as u32).to_le_bytes());
                buf[off + 12..off + 16].copy_from_slice(&(event.operation as u32).to_le_bytes());
                buf[off + 16..off + 24].copy_from_slice(&event.payload_items.to_le_bytes());
                buf[off + 24..off + 32].copy_from_slice(&event.raw_hashes.to_le_bytes());
                off += 32;
                scope_event_counts[event.scope as usize] += 1;
                operation_event_counts[event.operation as usize] += 1;
                raw_hash_count += event.raw_hashes;
                if matches!(
                    event.operation,
                    TranscriptOperation::DrawSecureFelt
                        | TranscriptOperation::DrawSecureFelts
                        | TranscriptOperation::DrawU32s
                ) {
                    challenge_draw_count += 1;
                    challenge_element_count += event.payload_items;
                }
            }
            hasher.update(&buf[..off]);
        }
        Self {
            event_count: events.len(),
            challenge_draw_count,
            challenge_element_count,
            raw_hash_count,
            scope_event_counts,
            operation_event_counts,
            trace_digest: hasher.finalize().0,
        }
    }

    pub fn trace_digest_hex(&self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.trace_digest {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

#[derive(Debug)]
pub struct Production100Channel {
    digest: Blake2sHash,
    draw_counter: u32,
    scope: TranscriptScope,
    events: RefCell<Vec<TranscriptEvent>>,
    last_pow_frame: RefCell<Option<PowFrame>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PowFrame {
    scope: TranscriptScope,
    prior_digest: Blake2sHash,
    pow_bits: u32,
    sequence_after_record: u64,
    seed_digest: Blake2sHash,
}

impl Default for Production100Channel {
    fn default() -> Self {
        Self {
            digest: Blake2sHash::default(),
            draw_counter: 0,
            scope: TranscriptScope::Bootstrap,
            events: RefCell::new(Vec::new()),
            last_pow_frame: RefCell::new(None),
        }
    }
}

impl Clone for Production100Channel {
    fn clone(&self) -> Self {
        Self {
            digest: self.digest,
            draw_counter: self.draw_counter,
            scope: self.scope,
            events: RefCell::new(self.events.borrow().clone()),
            last_pow_frame: RefCell::new(*self.last_pow_frame.borrow()),
        }
    }
}

impl Production100Channel {
    pub fn enter_scope(&mut self, scope: TranscriptScope) {
        let old_scope = self.scope;
        self.scope = scope;
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&(old_scope as u32).to_le_bytes());
        payload.extend_from_slice(&(scope as u32).to_le_bytes());
        self.mix_frame(TranscriptOperation::Scope, 1, &payload);
    }

    pub fn trace(&self) -> Vec<TranscriptEvent> {
        self.events.borrow().clone()
    }

    pub fn census(&self) -> TranscriptCensus {
        let events = self.events.borrow();
        TranscriptCensus::from_events(&events)
    }

    fn next_sequence(&self) -> u64 {
        self.events.borrow().len() as u64
    }

    fn record(
        &self,
        sequence: u64,
        operation: TranscriptOperation,
        payload_items: u64,
        raw_hashes: u64,
    ) {
        self.events.borrow_mut().push(TranscriptEvent {
            sequence,
            scope: self.scope,
            operation,
            payload_items,
            raw_hashes,
        });
    }

    fn framed_hash(
        &self,
        frame_tag: &[u8],
        operation: TranscriptOperation,
        sequence: u64,
        payload_items: u64,
        payload: &[u8],
    ) -> Blake2sHash {
        let mut hasher = Blake2sHasher::new();
        let header_len = TRANSCRIPT_V3_PREFIX.len() + frame_tag.len() + 64;
        if header_len + payload.len() <= 160 {
            let mut buf = [0u8; 160];
            let mut off = 0usize;
            buf[off..off + TRANSCRIPT_V3_PREFIX.len()].copy_from_slice(TRANSCRIPT_V3_PREFIX);
            off += TRANSCRIPT_V3_PREFIX.len();
            buf[off..off + frame_tag.len()].copy_from_slice(frame_tag);
            off += frame_tag.len();
            buf[off..off + 4].copy_from_slice(&(self.scope as u32).to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&(operation as u32).to_le_bytes());
            buf[off + 8..off + 16].copy_from_slice(&sequence.to_le_bytes());
            buf[off + 16..off + 24].copy_from_slice(&payload_items.to_le_bytes());
            buf[off + 24..off + 32].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            buf[off + 32..off + 64].copy_from_slice(self.digest.as_ref());
            off += 64;
            buf[off..off + payload.len()].copy_from_slice(payload);
            off += payload.len();
            hasher.update(&buf[..off]);
        } else {
            hasher.update(TRANSCRIPT_V3_PREFIX);
            hasher.update(frame_tag);
            hasher.update(&(self.scope as u32).to_le_bytes());
            hasher.update(&(operation as u32).to_le_bytes());
            hasher.update(&sequence.to_le_bytes());
            hasher.update(&payload_items.to_le_bytes());
            hasher.update(&(payload.len() as u64).to_le_bytes());
            hasher.update(self.digest.as_ref());
            hasher.update(payload);
        }
        hasher.finalize()
    }

    fn mix_frame(&mut self, operation: TranscriptOperation, payload_items: u64, payload: &[u8]) {
        let sequence = self.next_sequence();
        self.digest = self.framed_hash(MIX_FRAME_TAG, operation, sequence, payload_items, payload);
        self.draw_counter = 0;
        self.record(sequence, operation, payload_items, 1);
    }

    fn draw_words(
        &mut self,
        operation: TranscriptOperation,
        sequence: u64,
        payload_items: u64,
    ) -> (Blake2sHash, [u32; 8]) {
        let mut payload = Vec::with_capacity(4);
        payload.extend_from_slice(&self.draw_counter.to_le_bytes());
        self.draw_counter = self
            .draw_counter
            .checked_add(1)
            .expect("Production100 challenge counter overflow");
        let digest = self.framed_hash(DRAW_FRAME_TAG, operation, sequence, payload_items, &payload);
        let words = array::from_fn(|index| {
            u32::from_le_bytes(
                digest.0[4 * index..4 * (index + 1)]
                    .try_into()
                    .expect("four-byte Blake2s word"),
            )
        });
        (digest, words)
    }

    fn draw_base_felts(
        &mut self,
        operation: TranscriptOperation,
        sequence: u64,
        payload_items: u64,
    ) -> ([BaseField; 8], u64, Blake2sHash) {
        let mut raw_hashes = 0;
        loop {
            raw_hashes += 1;
            let (digest, words) = self.draw_words(operation, sequence, payload_items);
            if words.iter().all(|word| *word < 2 * P) {
                return (
                    words.map(|word| BaseField::reduce(u64::from(word))),
                    raw_hashes,
                    digest,
                );
            }
        }
    }

    fn pow_channel(&self, pow_bits: u32) -> Blake2sChannel {
        let sequence = self.next_sequence();
        if let Some(frame) = *self.last_pow_frame.borrow() {
            if frame.scope == self.scope
                && frame.prior_digest == self.digest
                && frame.pow_bits == pow_bits
                && frame.sequence_after_record == sequence
            {
                let mut channel = Blake2sChannel::default();
                channel.update_digest(frame.seed_digest);
                return channel;
            }
        }
        let mut payload = Vec::with_capacity(4);
        payload.extend_from_slice(&pow_bits.to_le_bytes());
        let digest = self.framed_hash(
            POW_FRAME_TAG,
            TranscriptOperation::VerifyPow,
            sequence,
            1,
            &payload,
        );
        self.record(sequence, TranscriptOperation::VerifyPow, 1, 1);
        *self.last_pow_frame.borrow_mut() = Some(PowFrame {
            scope: self.scope,
            prior_digest: self.digest,
            pow_bits,
            sequence_after_record: sequence + 1,
            seed_digest: digest,
        });
        let mut channel = Blake2sChannel::default();
        channel.update_digest(digest);
        channel
    }
}

impl Channel for Production100Channel {
    const BYTES_PER_HASH: usize = 32;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        self.pow_channel(n_bits).verify_pow_nonce(n_bits, nonce)
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        let payload = data
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.mix_frame(TranscriptOperation::MixU32s, data.len() as u64, &payload);
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let payload = felts
            .iter()
            .flat_map(|felt| felt.to_m31_array())
            .flat_map(|coordinate| coordinate.0.to_le_bytes())
            .collect::<Vec<_>>();
        self.mix_frame(TranscriptOperation::MixFelts, felts.len() as u64, &payload);
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_frame(TranscriptOperation::MixU64, 1, &value.to_le_bytes());
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let sequence = self.next_sequence();
        let (felts, raw_hashes, digest) =
            self.draw_base_felts(TranscriptOperation::DrawSecureFelt, sequence, 1);
        self.digest = digest;
        self.draw_counter = 0;
        self.record(sequence, TranscriptOperation::DrawSecureFelt, 1, raw_hashes);
        SecureField::from_m31_array(
            felts[..SECURE_EXTENSION_DEGREE]
                .try_into()
                .expect("secure-field coordinates"),
        )
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        if n_felts == 0 {
            return Vec::new();
        }
        let sequence = self.next_sequence();
        let required_coordinates = n_felts
            .checked_mul(SECURE_EXTENSION_DEGREE)
            .expect("Production100 challenge length overflow");
        let mut coordinates = Vec::with_capacity(required_coordinates);
        let mut raw_hashes = 0;
        let mut last_digest = self.digest;
        while coordinates.len() < required_coordinates {
            let (felts, hashes, digest) = self.draw_base_felts(
                TranscriptOperation::DrawSecureFelts,
                sequence,
                n_felts as u64,
            );
            raw_hashes += hashes;
            last_digest = digest;
            coordinates.extend(felts);
        }
        let values = coordinates
            .chunks_exact(SECURE_EXTENSION_DEGREE)
            .take(n_felts)
            .map(|chunk| {
                SecureField::from_m31_array(
                    chunk.try_into().expect("secure-field coordinate chunk"),
                )
            })
            .collect();
        self.digest = last_digest;
        self.draw_counter = 0;
        self.record(
            sequence,
            TranscriptOperation::DrawSecureFelts,
            n_felts as u64,
            raw_hashes,
        );
        values
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        let sequence = self.next_sequence();
        let (digest, words) = self.draw_words(TranscriptOperation::DrawU32s, sequence, 8);
        self.digest = digest;
        self.draw_counter = 0;
        self.record(sequence, TranscriptOperation::DrawU32s, 8, 1);
        words.to_vec()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageAShape {
    pub batch: usize,
    pub dimension: usize,
    pub expert_width: usize,
    pub experts: usize,
    pub top_k: usize,
    pub shared_factor: usize,
}

impl StageAShape {
    pub fn validate(self) -> Result<(), String> {
        if self.batch == 0 {
            return Err("Stage A batch must be nonzero".into());
        }
        if self.dimension == 0 || !self.dimension.is_power_of_two() {
            return Err("Stage A dimension must be a nonzero power of two".into());
        }
        if self.expert_width == 0 {
            return Err("Stage A expert width must be nonzero".into());
        }
        if self.experts == 0 || !self.experts.is_power_of_two() {
            return Err("Stage A expert count must be a nonzero power of two".into());
        }
        if !(1..=self.experts).contains(&self.top_k) {
            return Err("Stage A top-k must be in 1..=E".into());
        }
        if self.shared_factor == 0 {
            return Err("Stage A shared factor must be nonzero".into());
        }
        self.expert_width
            .checked_mul(self.shared_factor)
            .ok_or_else(|| "Stage A shared width overflow".to_string())?
            .checked_next_power_of_two()
            .ok_or_else(|| "Stage A shared width padding overflow".to_string())?;
        Ok(())
    }

    pub const fn shared_width(self) -> usize {
        self.expert_width * self.shared_factor
    }

    pub const fn expert_width_padded(self) -> usize {
        self.expert_width.next_power_of_two()
    }

    pub const fn shared_width_padded(self) -> usize {
        self.shared_width().next_power_of_two()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequantSpec {
    pub multiplier: i64,
    pub shift: u32,
    pub output_bits: u32,
}

impl RequantSpec {
    pub fn validate(self) -> Result<(), String> {
        if self.multiplier <= 0 {
            return Err("requant multiplier must be positive".into());
        }
        if self.shift == 0 || self.shift >= 63 {
            return Err("requant shift must be in 1..63".into());
        }
        if !(2..=31).contains(&self.output_bits) {
            return Err("requant output bits must be in 2..=31".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageANumericPolicy {
    pub table_format: StageATableFormat,
    pub rounding: StageARoundingMode,
    pub router: RequantSpec,
    pub gate: RequantSpec,
    pub up: RequantSpec,
    pub product: RequantSpec,
    pub output: RequantSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageARoundingMode {
    TiesToEven,
    HalfTowardPositiveInfinity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageATableFormat {
    SignedI12Q8,
    SignedI12Q7,
}

impl StageATableFormat {
    pub const fn input_bits(self) -> u32 {
        12
    }

    pub const fn input_fraction_bits(self) -> u32 {
        match self {
            Self::SignedI12Q8 => 8,
            Self::SignedI12Q7 => 7,
        }
    }
}

impl StageANumericPolicy {
    pub const fn v1() -> Self {
        Self {
            table_format: StageATableFormat::SignedI12Q8,
            rounding: StageARoundingMode::TiesToEven,
            router: RequantSpec {
                multiplier: 1,
                shift: 1,
                output_bits: 12,
            },
            gate: RequantSpec {
                multiplier: 1,
                shift: 1,
                output_bits: 12,
            },
            up: RequantSpec {
                multiplier: 1,
                shift: 1,
                output_bits: 12,
            },
            product: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 12,
            },
            output: RequantSpec {
                multiplier: 1,
                shift: 15,
                output_bits: 12,
            },
        }
    }

    pub const fn deep_prove_comparison_v1() -> Self {
        Self {
            rounding: StageARoundingMode::HalfTowardPositiveInfinity,
            ..Self::v1()
        }
    }

    pub const fn trained_i12_q7_v1() -> Self {
        Self {
            table_format: StageATableFormat::SignedI12Q7,
            rounding: StageARoundingMode::TiesToEven,
            router: RequantSpec {
                multiplier: 10,
                shift: 9,
                output_bits: 12,
            },
            gate: RequantSpec {
                multiplier: 14,
                shift: 9,
                output_bits: 12,
            },
            up: RequantSpec {
                multiplier: 9,
                shift: 9,
                output_bits: 12,
            },
            product: RequantSpec {
                multiplier: 1,
                shift: 8,
                output_bits: 12,
            },
            output: RequantSpec {
                multiplier: 5,
                shift: 23,
                output_bits: 12,
            },
        }
    }

    pub const fn trained_i12_q7_half_up_v1() -> Self {
        Self {
            rounding: StageARoundingMode::HalfTowardPositiveInfinity,
            ..Self::trained_i12_q7_v1()
        }
    }

    pub const fn id(self) -> &'static str {
        match (self.table_format, self.rounding) {
            (StageATableFormat::SignedI12Q8, StageARoundingMode::TiesToEven) => STAGE_A_POLICY_ID,
            (StageATableFormat::SignedI12Q8, StageARoundingMode::HalfTowardPositiveInfinity) => {
                STAGE_A_DEEPPROVE_COMPARISON_POLICY_ID
            }
            (StageATableFormat::SignedI12Q7, StageARoundingMode::TiesToEven) => {
                STAGE_A_TRAINED_I12_Q7_POLICY_ID
            }
            (StageATableFormat::SignedI12Q7, StageARoundingMode::HalfTowardPositiveInfinity) => {
                STAGE_A_TRAINED_I12_Q7_HALF_UP_POLICY_ID
            }
        }
    }

    pub fn validate(self) -> Result<(), String> {
        for spec in [self.router, self.gate, self.up, self.product, self.output] {
            spec.validate()?;
        }
        if self == Self::v1()
            || self == Self::deep_prove_comparison_v1()
            || self == Self::trained_i12_q7_v1()
            || self == Self::trained_i12_q7_half_up_v1()
        {
            Ok(())
        } else {
            Err("unsupported Stage A numeric policy parameters".into())
        }
    }

    pub fn digest(self) -> Result<[u8; 32], String> {
        self.validate()?;
        let mut hasher = Blake2s256::new();
        hasher.update(b"stwo-ml:production-moe-stage-a:numeric-policy:v1");
        hasher.update(self.id().as_bytes());
        hasher.update(self.table_format.input_bits().to_le_bytes());
        hasher.update(self.table_format.input_fraction_bits().to_le_bytes());
        hasher.update([match self.rounding {
            StageARoundingMode::TiesToEven => 0,
            StageARoundingMode::HalfTowardPositiveInfinity => 1,
        }]);
        for spec in [self.router, self.gate, self.up, self.product, self.output] {
            hasher.update(spec.multiplier.to_le_bytes());
            hasher.update(spec.shift.to_le_bytes());
            hasher.update(spec.output_bits.to_le_bytes());
        }
        hasher.update(public_tables_for_format(self.table_format).digest);
        Ok(hasher.finalize().into())
    }
}

pub fn q15_route_weight_sum_upper_bound(top_k: usize) -> Result<u64, String> {
    let scale = 1u64 << 15;
    let rounding_slack = u64::try_from(top_k / 2)
        .map_err(|_| "Stage A top-k exceeds the route-weight sum representation")?;
    scale
        .checked_add(rounding_slack)
        .ok_or_else(|| "Stage A route-weight sum bound overflow".to_string())
}

#[derive(Debug)]
pub struct PublicTables {
    pub silu_q8: Vec<i16>,
    pub exp_q15: Vec<u16>,
    pub silu_max_abs: u128,
    pub digest: [u8; 32],
}

static I12_Q8_PUBLIC_TABLES: OnceLock<PublicTables> = OnceLock::new();
static I12_Q7_PUBLIC_TABLES: OnceLock<PublicTables> = OnceLock::new();

pub fn public_tables() -> &'static PublicTables {
    public_tables_for_format(StageATableFormat::SignedI12Q8)
}

pub fn public_tables_for_policy(policy: StageANumericPolicy) -> Result<&'static PublicTables, String> {
    policy.validate()?;
    Ok(public_tables_for_format(policy.table_format))
}

fn public_tables_for_format(format: StageATableFormat) -> &'static PublicTables {
    let (cell, data) = match format {
        StageATableFormat::SignedI12Q8 => (&I12_Q8_PUBLIC_TABLES, I12_Q8_TABLE_DATA),
        StageATableFormat::SignedI12Q7 => (&I12_Q7_PUBLIC_TABLES, I12_Q7_TABLE_DATA),
    };
    cell.get_or_init(|| parse_public_tables(data).expect("checked-in Stage A table is valid"))
}

fn parse_public_tables(data: &str) -> Result<PublicTables, String> {
    let mut silu_q8 = Vec::with_capacity(1 << 12);
    let mut exp_q15 = Vec::with_capacity(1 << 12);
    for line in data.lines() {
        if line.starts_with('#') || line.starts_with("address") || line.is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err("Stage A table row must have four fields".into());
        }
        let address = fields[0]
            .parse::<usize>()
            .map_err(|_| "invalid Stage A table address".to_string())?;
        let signed_input = fields[1]
            .parse::<i32>()
            .map_err(|_| "invalid Stage A table input".to_string())?;
        if address != silu_q8.len() || signed_input != address as i32 - 2048 {
            return Err("Stage A table address/input ordering mismatch".into());
        }
        silu_q8.push(
            fields[2]
                .parse::<i16>()
                .map_err(|_| "invalid Stage A SiLU table value".to_string())?,
        );
        exp_q15.push(
            fields[3]
                .parse::<u16>()
                .map_err(|_| "invalid Stage A exp table value".to_string())?,
        );
    }
    if silu_q8.len() != 1 << 12 || exp_q15.len() != 1 << 12 {
        return Err("Stage A table must contain 4096 rows".into());
    }
    let silu_max_abs = silu_q8
        .iter()
        .map(|&value| i128::from(value).unsigned_abs())
        .max()
        .ok_or_else(|| "Stage A SiLU table must not be empty".to_string())?;
    let mut hasher = Blake2s256::new();
    hasher.update(data.as_bytes());
    Ok(PublicTables {
        silu_q8,
        exp_q15,
        silu_max_abs,
        digest: hasher.finalize().into(),
    })
}

fn signed_i12_address(value: i64) -> Result<usize, String> {
    if !(-2048..=2047).contains(&value) {
        return Err(format!("signed i12 table input {value} is out of range"));
    }
    Ok((value + 2048) as usize)
}

pub fn silu_q8_for_policy(value: i64, policy: StageANumericPolicy) -> Result<i16, String> {
    Ok(public_tables_for_policy(policy)?.silu_q8[signed_i12_address(value)?])
}

fn exp_q15(value: i64, policy: StageANumericPolicy) -> Result<u16, String> {
    if value > 0 {
        return Err("softmax exp input must be non-positive after max subtraction".into());
    }
    Ok(public_tables_for_policy(policy)?.exp_q15[signed_i12_address(value)?])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequantResult {
    pub value: i32,
    pub remainder: i128,
}

fn round_ratio_ties_even(numerator: i128, denominator: i128) -> Result<i128, String> {
    if denominator <= 0 || numerator < 0 {
        return Err(
            "ties-even ratio expects a non-negative numerator and positive denominator".into(),
        );
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder
        .checked_mul(2)
        .ok_or_else(|| "ratio remainder overflow".to_string())?;
    Ok(
        if doubled > denominator || (doubled == denominator && quotient % 2 == 1) {
            quotient + 1
        } else {
            quotient
        },
    )
}

fn round_signed_pow2_ties_even(value: i128, shift: u32) -> Result<i128, String> {
    let denominator = 1i128
        .checked_shl(shift)
        .ok_or_else(|| "requant denominator overflow".to_string())?;
    let magnitude = value
        .checked_abs()
        .ok_or_else(|| "requant magnitude overflow".to_string())?;
    let rounded = round_ratio_ties_even(magnitude, denominator)?;
    Ok(if value < 0 { -rounded } else { rounded })
}

fn round_signed_pow2_half_toward_positive_infinity(
    value: i128,
    shift: u32,
) -> Result<i128, String> {
    let denominator = 1i128
        .checked_shl(shift)
        .ok_or_else(|| "requant denominator overflow".to_string())?;
    value
        .checked_add(denominator / 2)
        .ok_or_else(|| "requant rounding addition overflow".to_string())
        .map(|rounded| rounded.div_euclid(denominator))
}

pub fn requant_with_rounding(
    value: i128,
    spec: RequantSpec,
    rounding: StageARoundingMode,
) -> Result<RequantResult, String> {
    spec.validate()?;
    let scaled = value
        .checked_mul(i128::from(spec.multiplier))
        .ok_or_else(|| "requant multiplication overflow".to_string())?;
    let rounded = match rounding {
        StageARoundingMode::TiesToEven => round_signed_pow2_ties_even(scaled, spec.shift)?,
        StageARoundingMode::HalfTowardPositiveInfinity => {
            round_signed_pow2_half_toward_positive_infinity(scaled, spec.shift)?
        }
    };
    let min = -(1i128 << (spec.output_bits - 1));
    let max = (1i128 << (spec.output_bits - 1)) - 1;
    if !(min..=max).contains(&rounded) {
        return Err(format!(
            "requant output {rounded} exceeds signed {}-bit range",
            spec.output_bits
        ));
    }
    let denominator = 1i128 << spec.shift;
    let remainder = scaled - rounded * denominator;
    match rounding {
        StageARoundingMode::TiesToEven => {
            let doubled_abs = remainder
                .checked_abs()
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| "requant remainder overflow".to_string())?;
            if doubled_abs > denominator
                || (doubled_abs == denominator && rounded.rem_euclid(2) != 0)
            {
                return Err("requant ties-even witness is not canonical".into());
            }
        }
        StageARoundingMode::HalfTowardPositiveInfinity => {
            if remainder < -(denominator / 2) || remainder >= denominator / 2 {
                return Err("requant half-up witness is not canonical".into());
            }
        }
    }
    Ok(RequantResult {
        value: i32::try_from(rounded).map_err(|_| "requant i32 conversion failed")?,
        remainder,
    })
}

fn requant_vector(
    values: &[i64],
    spec: RequantSpec,
    rounding: StageARoundingMode,
) -> Result<Vec<i16>, String> {
    values
        .iter()
        .map(|&value| {
            let result = requant_with_rounding(i128::from(value), spec, rounding)?;
            i16::try_from(result.value).map_err(|_| "requant output exceeds i16".to_string())
        })
        .collect()
}

fn canonical_top_k(logits: &[i16], top_k: usize) -> Result<Vec<usize>, String> {
    if !(1..=logits.len()).contains(&top_k) {
        return Err("canonical top-k is outside the score domain".into());
    }
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| {
        logits[right]
            .cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(top_k);
    Ok(indices)
}

fn softmax_weights_q15(
    logits: &[i16],
    policy: StageANumericPolicy,
) -> Result<(Vec<u16>, Vec<u16>), String> {
    let max = logits
        .iter()
        .copied()
        .max()
        .ok_or_else(|| "softmax score row is empty".to_string())?;
    let min = logits
        .iter()
        .copied()
        .min()
        .ok_or_else(|| "softmax score row is empty".to_string())?;
    let spread = i64::from(max) - i64::from(min);
    if spread > STAGE_A_SOFTMAX_MAX_LOGIT_SPREAD {
        return Err(format!(
            "softmax logit spread {spread} exceeds the pinned signed-i12 exp-table domain {STAGE_A_SOFTMAX_MAX_LOGIT_SPREAD}"
        ));
    }
    let exp = logits
        .iter()
        .map(|&logit| exp_q15(i64::from(logit) - i64::from(max), policy))
        .collect::<Result<Vec<_>, _>>()?;
    let denominator = exp.iter().map(|&value| i128::from(value)).sum::<i128>();
    if denominator == 0 {
        return Err("softmax denominator is zero".into());
    }
    let weights = exp
        .iter()
        .map(|&value| {
            let numerator = i128::from(value)
                .checked_mul(i128::from(Q15_ONE))
                .ok_or_else(|| "softmax numerator overflow".to_string())?;
            let rounded = round_ratio_ties_even(numerator, denominator)?;
            u16::try_from(rounded).map_err(|_| "softmax Q0.15 weight exceeds u16".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((exp, weights))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageARouteTrace {
    pub router_output_raw: Vec<i64>,
    pub router_logits_q8: Vec<i16>,
    pub exp_q15: Vec<u16>,
    pub softmax_q15: Vec<u16>,
    pub ranked_route: Vec<usize>,
}

pub fn derive_stage_a_route_from_raw(
    router_output_raw: &[i64],
    shape: StageAShape,
    policy: StageANumericPolicy,
) -> Result<StageARouteTrace, String> {
    shape.validate()?;
    policy.validate()?;
    if router_output_raw.len() != shape.experts {
        return Err("Stage A router output length does not match the expert count".into());
    }
    let router_logits_q8 = requant_vector(router_output_raw, policy.router, policy.rounding)?;
    let ranked_route = canonical_top_k(&router_logits_q8, shape.top_k)?;
    let (exp_q15, softmax_q15) = softmax_weights_q15(&router_logits_q8, policy)?;
    Ok(StageARouteTrace {
        router_output_raw: router_output_raw.to_vec(),
        router_logits_q8,
        exp_q15,
        softmax_q15,
        ranked_route,
    })
}

pub const REGULAR_SLOT_ROLES: [u8; 3] = [0x00, 0x01, 0x02];
pub const SHARED_SLOT_ROLES: [u8; 3] = [0x00, 0x01, 0x02];
pub const ROUTER_SLOT_ROLES: [u8; 1] = [0x03];

pub const CPRS_CLASS_TAG_REGULAR: u8 = 0x00;
pub const CPRS_CLASS_TAG_SHARED: u8 = 0x01;
pub const CPRS_CLASS_TAG_ROUTER: u8 = 0x02;

const REGISTRY_V2_LEAF_DOMAIN: &[u8] = b"stwo-ml:expert-registry:leaf:v2";
const REGISTRY_V2_NODE_DOMAIN: &[u8] = b"stwo-ml:expert-registry:node:v2";
const REGISTRY_V2_ROOT_DOMAIN: &[u8] = b"stwo-ml:expert-registry:root:v2";
const CPRS_SCHEME_DOMAIN: &[u8] = b"stwo-ml:cprs-scheme:v1";
const CPRS_OUTER_DIGEST_DOMAIN: &[u8; 28] = b"stwo-ml:cprs-outer-digest:v1";
const CPRS_PROOFS_DIGEST_DOMAIN: &[u8] = b"stwo-ml:cprs-proofs-digest:v1";
const STAGE_A_REGISTERED_ANCHOR_DOMAIN: &[u8] = b"stwo-ml:stage-a-registered-anchor:v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageARegisteredGeometry {
    pub rows_padded: usize,
    pub experts_padded: usize,
    pub reg_cols_padded: usize,
    pub sh_cols_padded: usize,
    pub n_router: usize,
    pub n_regular: usize,
    pub n_shared: usize,
}

impl StageARegisteredGeometry {
    pub fn for_shape(shape: StageAShape) -> Result<Self, String> {
        shape.validate()?;
        let rows_padded = shape.dimension.next_power_of_two();
        let experts_padded = shape.experts.next_power_of_two();
        let reg_cols_padded = shape.expert_width_padded();
        let sh_cols_padded = shape.shared_width_padded();
        let n_router = (rows_padded.ilog2() + experts_padded.ilog2()) as usize;
        let n_regular = (rows_padded.ilog2() + reg_cols_padded.ilog2()) as usize;
        let n_shared = (rows_padded.ilog2() + sh_cols_padded.ilog2()) as usize;
        if n_router < 3 || n_regular < 3 || n_shared < 3 {
            return Err("Stage A CPRS matrix log-size must be at least 3".into());
        }
        Ok(Self {
            rows_padded,
            experts_padded,
            reg_cols_padded,
            sh_cols_padded,
            n_router,
            n_regular,
            n_shared,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageACprsParams {
    pub router: Params,
    pub regular: Params,
    pub shared: Params,
    pub scheme_digest: [u8; 32],
}

impl StageACprsParams {
    pub fn for_shape_and_profile(
        shape: StageAShape,
        profile: &SecurityProfile,
    ) -> Result<Self, String> {
        Self::for_shape_and_profile_with_f(shape, profile, None)
    }

    pub fn for_shape_and_profile_with_f(
        shape: StageAShape,
        profile: &SecurityProfile,
        f_override: Option<usize>,
    ) -> Result<Self, String> {
        let geom = StageARegisteredGeometry::for_shape(shape)?;
        let make = |n: usize, g: usize| -> Result<Params, String> {
            let m = n
                .checked_sub(2)
                .ok_or_else(|| "CPRS n must be >= 3".to_string())?;
            let (s, f_def, q, pow_bits) = cprs_profile_parameters(profile, m);
            let f = f_override.map(|fo| fo.min(m).max(1)).unwrap_or(f_def);
            Params::try_new(n, s, f, q, pow_bits, g).map_err(|e| e.to_string())
        };
        let router = make(geom.n_router, 1)?;
        let regular = make(geom.n_regular, 3)?;
        let shared = make(geom.n_shared, 3)?;
        let scheme_digest = compute_scheme_digest(&router, &regular, &shared);
        Ok(Self {
            router,
            regular,
            shared,
            scheme_digest,
        })
    }
}

pub fn cprs_profile_parameters(profile: &SecurityProfile, m: usize) -> (usize, usize, usize, usize) {
    match profile {
        SecurityProfile::Production100 => (2, 4usize.min(m).max(1), 118, 20),
        SecurityProfile::Dev13 => (1, m.clamp(1, 2), 8, 0),
        _ => (3, 4usize.min(m).max(1), 32, 0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CprsSoundnessBits {
    pub unique_decoding: f64,
    pub conjectured: f64,
}

pub fn cprs_soundness_bits(prm: &Params) -> CprsSoundnessBits {
    let rho = 2f64.powi(-(prm.s() as i32));
    let per_query_ud = -((1.0 + rho) / 2.0).log2();
    CprsSoundnessBits {
        unique_decoding: prm.q() as f64 * per_query_ud + prm.pow_bits() as f64,
        conjectured: (prm.q() * prm.s() + prm.pow_bits()) as f64,
    }
}

fn compute_scheme_digest(router: &Params, regular: &Params, shared: &Params) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(CPRS_SCHEME_DOMAIN);
    for prm in [router, regular, shared] {
        h.update([
            prm.n() as u8,
            prm.s() as u8,
            prm.f() as u8,
            prm.g() as u8,
            prm.pow_bits() as u8,
            prm.depth() as u8,
        ]);
        h.update((prm.q() as u32).to_le_bytes());
    }
    h.finalize().into()
}

pub fn registered_expert_leaf_hash(
    shape: StageAShape,
    expert_id: usize,
    slot_roles: &[u8; 3],
    scheme_digest: &[u8; 32],
    group_commitment: &[u8; 32],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(REGISTRY_V2_LEAF_DOMAIN);
    h.update((expert_id as u64).to_le_bytes());
    h.update((shape.dimension as u64).to_le_bytes());
    h.update((shape.expert_width as u64).to_le_bytes());
    h.update(slot_roles);
    h.update(scheme_digest);
    h.update(group_commitment);
    h.finalize().into()
}

fn registry_v2_node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(REGISTRY_V2_NODE_DOMAIN);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

fn registry_v2_root_hash(expert_count: usize, tree_root: &[u8; 32]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(REGISTRY_V2_ROOT_DOMAIN);
    h.update((expert_count as u64).to_le_bytes());
    h.update(tree_root);
    h.finalize().into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageARegisteredExpertOpening {
    pub expert_id: usize,
    pub slot_roles: [u8; 3],
    pub group_commitment: [u8; 32],
    pub siblings: Vec<[u8; 32]>,
}

impl StageARegisteredExpertOpening {
    pub fn verify(
        &self,
        shape: StageAShape,
        scheme_digest: &[u8; 32],
        expected_root: &[u8; 32],
    ) -> Result<(), String> {
        shape.validate()?;
        if self.expert_id >= shape.experts {
            return Err("Stage A registered expert opening index is out of range".into());
        }
        if self.slot_roles != REGULAR_SLOT_ROLES {
            return Err("Stage A registered expert opening has non-canonical slot roles".into());
        }
        let depth = shape.experts.ilog2() as usize;
        if self.siblings.len() != depth {
            return Err("Stage A registered expert opening Merkle path length mismatch".into());
        }
        let mut current = registered_expert_leaf_hash(
            shape,
            self.expert_id,
            &self.slot_roles,
            scheme_digest,
            &self.group_commitment,
        );
        let mut idx = self.expert_id;
        for sibling in &self.siblings {
            current = if idx & 1 == 0 {
                registry_v2_node_hash(&current, sibling)
            } else {
                registry_v2_node_hash(sibling, &current)
            };
            idx >>= 1;
        }
        let root = registry_v2_root_hash(shape.experts, &current);
        if &root != expected_root {
            return Err(format!(
                "Stage A registered expert {} failed registry v2 Merkle verification",
                self.expert_id
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageARegisteredContext {
    pub shape: StageAShape,
    pub model_digest: [u8; 32],
    pub manifest_digest: [u8; 32],
    pub scheme_digest: [u8; 32],
    pub router_commitment: [u8; 32],
    pub shared_commitment: [u8; 32],
    pub registered_expert_root: [u8; 32],
    pub active_expert_openings: Vec<StageARegisteredExpertOpening>,
}

impl StageARegisteredContext {
    pub fn anchor_digest(&self) -> [u8; 32] {
        let mut h = Blake2s256::new();
        h.update(STAGE_A_REGISTERED_ANCHOR_DOMAIN);
        for value in [
            self.shape.dimension,
            self.shape.expert_width,
            self.shape.experts,
            self.shape.top_k,
            self.shape.shared_factor,
        ] {
            h.update((value as u64).to_le_bytes());
        }
        h.update(self.model_digest);
        h.update(self.manifest_digest);
        h.update(self.scheme_digest);
        h.update(self.router_commitment);
        h.update(self.shared_commitment);
        h.update(self.registered_expert_root);
        h.finalize().into()
    }

    pub fn footprint_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .active_expert_openings
                .iter()
                .map(|op| {
                    std::mem::size_of::<StageARegisteredExpertOpening>()
                        + op.siblings.len() * 32
                })
                .sum::<usize>()
    }

    pub fn validate_for_active_experts(
        &self,
        active_experts: &[usize],
        profile: &SecurityProfile,
    ) -> Result<StageACprsParams, String> {
        self.validate_for_active_experts_with_f(active_experts, profile, None)
    }

    pub fn validate_for_active_experts_with_f(
        &self,
        active_experts: &[usize],
        profile: &SecurityProfile,
        f_override: Option<usize>,
    ) -> Result<StageACprsParams, String> {
        self.shape.validate()?;
        let params =
            StageACprsParams::for_shape_and_profile_with_f(self.shape, profile, f_override)?;
        if self.scheme_digest != params.scheme_digest {
            return Err("Stage A registered verifier context CPRS scheme digest mismatch".into());
        }
        if self.active_expert_openings.len() != active_experts.len() {
            return Err("Stage A registered verifier context active expert count mismatch".into());
        }
        for (opening, &expected_expert_id) in
            self.active_expert_openings.iter().zip(active_experts)
        {
            if opening.expert_id != expected_expert_id {
                return Err(format!(
                    "Stage A registered expert opening ID {} does not match routed active expert {}",
                    opening.expert_id, expected_expert_id
                ));
            }
            opening.verify(
                self.shape,
                &self.scheme_digest,
                &self.registered_expert_root,
            )?;
        }
        Ok(params)
    }

    pub fn validate_for_active_experts_any_f(
        &self,
        active_experts: &[usize],
        profile: &SecurityProfile,
    ) -> Result<StageACprsParams, String> {
        for cand in [None, Some(1), Some(2), Some(3), Some(4)] {
            let prm = StageACprsParams::for_shape_and_profile_with_f(self.shape, profile, cand)?;
            if prm.scheme_digest == self.scheme_digest {
                return self.validate_for_active_experts_with_f(active_experts, profile, cand);
            }
        }
        self.validate_for_active_experts_with_f(active_experts, profile, None)
    }
}

pub fn stacked_expert_bank_params(
    shape: StageAShape,
    pack: usize,
    f_override: Option<usize>,
    profile: &SecurityProfile,
) -> Result<Params, String> {
    let geom = StageARegisteredGeometry::for_shape(shape)?;
    if pack == 0 || !pack.is_power_of_two() || geom.experts_padded % pack != 0 {
        return Err(format!(
            "stacked_expert_bank_params: invalid pack={pack} for experts_padded={}",
            geom.experts_padded
        ));
    }
    let num_shards = geom.experts_padded / pack;
    let g_bank = 3 * num_shards;
    if g_bank == 0 || g_bank > 255 {
        return Err(format!(
            "stacked_expert_bank_params: group arity g={g_bank} out of CPRS [1, 255] bounds"
        ));
    }
    let log_pack = pack.ilog2() as usize;
    let n_bank = geom.n_regular + log_pack;
    let m_bank = n_bank
        .checked_sub(2)
        .ok_or_else(|| "CPRS n must be >= 3".to_string())?;
    let (s, f_def, q, pow_bits) = cprs_profile_parameters(profile, m_bank);
    let f = f_override.map(|fo| fo.min(m_bank).max(1)).unwrap_or(f_def);
    Params::try_new(n_bank, s, f, q, pow_bits, g_bank).map_err(|e| e.to_string())
}

pub fn cprs_outer_digest(
    binding_seed: &[u8; 32],
    model_digest: &[u8; 32],
    policy_digest: &[u8; 32],
    table_digest: &[u8; 32],
    input_digest: &[u8; 32],
    token_count: usize,
    active_expert_ids: &[usize],
    class_tag: u8,
    slot_roles: &[u8],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(CPRS_OUTER_DIGEST_DOMAIN);
    h.update(binding_seed);
    h.update(model_digest);
    h.update(policy_digest);
    h.update(table_digest);
    h.update(input_digest);
    h.update((token_count as u32).to_le_bytes());
    h.update((active_expert_ids.len() as u32).to_le_bytes());
    for &expert_id in active_expert_ids {
        h.update((expert_id as u32).to_le_bytes());
    }
    h.update([class_tag]);
    h.update((slot_roles.len() as u32).to_le_bytes());
    h.update(slot_roles);
    let digest: [u8; 32] = h.finalize().into();
    assert_ne!(
        digest, [0u8; 32],
        "BLAKE2s-256 outer digest collided with zero"
    );
    digest
}

pub fn cprs_proofs_transcript_digest(
    router_proof: &[u8],
    regular_proof: &[u8],
    shared_proof: &[u8],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(CPRS_PROOFS_DIGEST_DOMAIN);
    for proof_bytes in [router_proof, regular_proof, shared_proof] {
        h.update((proof_bytes.len() as u64).to_le_bytes());
        h.update(proof_bytes);
    }
    h.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageAChainLink {
    pub layer: u32,
    pub prev_receipt_digest: [u8; 32],
}

impl StageAChainLink {
    pub const GENESIS: Self = Self {
        layer: 0,
        prev_receipt_digest: [0u8; 32],
    };

    pub fn next(layer: u32, prev_receipt: &[u8]) -> Self {
        let mut h = Blake2s256::new();
        h.update(b"stwo-ml:stage-a-chain-link:v1");
        h.update(layer.to_le_bytes());
        h.update((prev_receipt.len() as u64).to_le_bytes());
        h.update(prev_receipt);
        Self {
            layer,
            prev_receipt_digest: h.finalize().into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageANonlinearBatchStatement {
    pub shape: StageAShape,
    pub policy_id: String,
    pub policy_digest: [u8; 32],
    pub model_digest: [u8; 32],
    pub model_registry_manifest_digest: [u8; 32],
    pub table_digest: [u8; 32],
    pub input_digest: [u8; 32],
    pub router_outputs_raw: Vec<Vec<i64>>,
    pub routes: Vec<Vec<usize>>,
    pub route_weights_q15: Vec<Vec<u16>>,
    pub output_q8: Vec<Vec<i16>>,
    pub wide_log_columns: usize,
}

#[derive(Clone, Debug)]
pub struct StageARegisteredRouterProof {
    pub gate_claim: SecureField,
    pub gate_proof: SumcheckProof,
    pub wg_opening: SecureField,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageAPublicExpertActivations {
    pub gate_raw: Vec<i32>,
    pub up_raw: Vec<i32>,
    pub gate_q8: Vec<i16>,
    pub up_q8: Vec<i16>,
    pub product: Vec<i32>,
    pub hidden_q8: Vec<i16>,
    pub down_raw: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageAPublicTokenActivations {
    pub regular: Vec<StageAPublicExpertActivations>,
    pub shared: StageAPublicExpertActivations,
}

#[derive(Clone, Debug)]
pub struct StageAPublicWitnessBatchProof {
    pub tokens: Vec<StageAPublicTokenActivations>,
    pub router: StageARegisteredRouterProof,
    pub regular_gate_up_claims: Vec<SecureField>,
    pub shared_gate_up_claims: Vec<SecureField>,
    pub regular_gate_up_sumcheck: SumcheckProof,
    pub regular_gate_weight_openings: Vec<SecureField>,
    pub regular_up_weight_openings: Vec<SecureField>,
    pub shared_gate_up_sumcheck: SumcheckProof,
    pub shared_gate_weight_opening: SecureField,
    pub shared_up_weight_opening: SecureField,
    pub regular_down_claim: SecureField,
    pub shared_down_claim: SecureField,
    pub regular_grouped_claim: SecureField,
    pub shared_grouped_claim: SecureField,
    pub regular_down_weight_openings: Vec<SecureField>,
    pub regular_weight_opening: SecureField,
    pub shared_weight_opening: SecureField,
    pub regular_down_sumcheck: SumcheckProof,
    pub shared_down_sumcheck: SumcheckProof,
    pub cprs_router_proof: Vec<u8>,
    pub cprs_regular_proof: Vec<u8>,
    pub cprs_shared_proof: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct StageARegisteredVerifierRun {
    pub verifier_context_bytes: usize,
    pub transcript_census: TranscriptCensus,
    pub transcript_trace: Vec<TranscriptEvent>,
}

pub fn stage_a_input_digest(inputs: &[Vec<i16>]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(b"stwo-ml:production-moe-stage-a:inputs:v1");
    hasher.update((inputs.len() as u64).to_le_bytes());
    for input in inputs {
        hasher.update((input.len() as u64).to_le_bytes());
        hash_i16_le_slice(&mut hasher, input);
    }
    hasher.finalize().into()
}

fn limb_widths(total_bits: usize, limb_bits: usize) -> Vec<usize> {
    let mut widths = vec![limb_bits; total_bits / limb_bits];
    if total_bits % limb_bits != 0 {
        widths.push(total_bits % limb_bits);
    }
    widths
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputLayout {
    slots: usize,
    limb_bits: usize,
    coefficient_sum_bits: usize,
    down_widths: Vec<usize>,
    remainder_widths: Vec<usize>,
    encoded_widths: Vec<usize>,
    target_widths: Vec<usize>,
    carry_widths: Vec<usize>,
}

impl OutputLayout {
    fn for_policy(slots: usize, policy: StageANumericPolicy) -> Result<Self, String> {
        policy.validate()?;
        if slots == 0 {
            return Err("Stage A output layout needs at least one slot".into());
        }
        let multiplier = u64::try_from(policy.output.multiplier)
            .map_err(|_| "Stage A output multiplier must be positive".to_string())?;
        let multiplier_bits = multiplier.next_power_of_two().ilog2() as usize;
        let coefficient_sum_bits =
            slots.next_power_of_two().ilog2() as usize + 15 + multiplier_bits;
        let limb_bits = 29usize
            .checked_sub(coefficient_sum_bits)
            .filter(|&bits| bits >= 1)
            .ok_or_else(|| "Stage A output coefficient sum leaves no limb headroom".to_string())?
            .min(WIDTH_TABLE_MAX_BITS);
        let shift = policy.output.shift as usize;
        Ok(Self {
            slots,
            limb_bits,
            coefficient_sum_bits,
            down_widths: limb_widths(DOWN_ENCODE_BITS, limb_bits),
            remainder_widths: limb_widths(shift, limb_bits),
            encoded_widths: limb_widths(shift + 1, limb_bits),
            target_widths: limb_widths(coefficient_sum_bits + 32, limb_bits),
            carry_widths: limb_widths(coefficient_sum_bits + 1, limb_bits),
        })
    }

    fn n_target_limbs(&self) -> usize {
        self.target_widths.len()
    }
}

fn checked_bound(label: &str, factors: &[u128]) -> Result<u128, String> {
    factors.iter().try_fold(1u128, |acc, &factor| {
        acc.checked_mul(factor)
            .ok_or_else(|| format!("Stage A {label} field-safety bound overflow"))
    })
}

fn require_centered_m31(bound: u128, label: &str) -> Result<(), String> {
    if bound > u128::from(P / 2) {
        return Err(format!(
            "Stage A {label} bound {bound} exceeds centered M31 range"
        ));
    }
    Ok(())
}

fn require_down_encoding_bound(bound: u128, label: &str) -> Result<(), String> {
    let first_ambiguous_value = u128::from(P / 2);
    if bound >= first_ambiguous_value {
        return Err(format!(
            "Stage A {label} bound {bound} reaches the ambiguous {DOWN_ENCODE_BITS}-bit M31 endpoint"
        ));
    }
    Ok(())
}

fn require_requant_rhs_centered(spec: RequantSpec, label: &str) -> Result<(), String> {
    let quotient_abs_bound = 1u128
        .checked_shl(spec.output_bits - 1)
        .ok_or_else(|| format!("Stage A {label} quotient bound overflow"))?;
    let denominator = 1u128
        .checked_shl(spec.shift)
        .ok_or_else(|| format!("Stage A {label} denominator bound overflow"))?;
    let remainder_abs_bound = denominator / 2;
    let rhs_bound = quotient_abs_bound
        .checked_mul(denominator)
        .and_then(|value| value.checked_add(remainder_abs_bound))
        .ok_or_else(|| format!("Stage A {label} requantized-rhs bound overflow"))?;
    require_centered_m31(rhs_bound, &format!("{label} requantized rhs"))
}

pub fn validate_stage_a_arithmetic_domain(
    shape: StageAShape,
    policy: StageANumericPolicy,
    input_abs_bound: u128,
) -> Result<(), String> {
    let dimension = u128::try_from(shape.dimension)
        .map_err(|_| "Stage A dimension exceeds the field-safety accounting range")?;
    let input_l1_bound = checked_bound("input L1", &[dimension, input_abs_bound])?;
    validate_stage_a_arithmetic_domain_with_l1_bound(shape, policy, input_l1_bound)
}

pub fn validate_stage_a_arithmetic_domain_with_l1_bound(
    shape: StageAShape,
    policy: StageANumericPolicy,
    input_l1_bound: u128,
) -> Result<(), String> {
    shape.validate()?;
    policy.validate()?;

    let expert_width = u128::try_from(shape.expert_width)
        .map_err(|_| "Stage A expert width exceeds the field-safety accounting range")?;
    let shared_width = u128::try_from(shape.shared_width())
        .map_err(|_| "Stage A shared width exceeds the field-safety accounting range")?;
    let i8_abs_bound = 128u128;
    let up_abs_bound = 1u128 << (policy.up.output_bits - 1);
    let hidden_abs_bound = 1u128 << (policy.product.output_bits - 1);
    let silu_abs_bound = public_tables_for_policy(policy)?.silu_max_abs;

    for (label, multiplier) in [
        ("router projection", policy.router.multiplier),
        ("gate projection", policy.gate.multiplier),
        ("up projection", policy.up.multiplier),
    ] {
        let multiplier = u128::try_from(multiplier)
            .map_err(|_| format!("Stage A {label} multiplier must be positive"))?;
        require_centered_m31(
            checked_bound(
                label,
                &[input_l1_bound, i8_abs_bound, multiplier],
            )?,
            label,
        )?;
    }
    require_requant_rhs_centered(policy.router, "router projection")?;
    require_requant_rhs_centered(policy.gate, "gate projection")?;
    require_requant_rhs_centered(policy.up, "up projection")?;

    let product_multiplier = u128::try_from(policy.product.multiplier)
        .map_err(|_| "Stage A product multiplier must be positive")?;
    require_centered_m31(
        checked_bound(
            "SwiGLU product",
            &[silu_abs_bound, up_abs_bound, product_multiplier],
        )?,
        "SwiGLU product",
    )?;
    require_requant_rhs_centered(policy.product, "SwiGLU product")?;

    let regular_down_bound = checked_bound(
        "regular down projection",
        &[expert_width, hidden_abs_bound, i8_abs_bound],
    )?;
    require_centered_m31(regular_down_bound, "regular down projection")?;
    require_down_encoding_bound(regular_down_bound, "regular down projection")?;
    let shared_down_bound = checked_bound(
        "shared down projection",
        &[shared_width, hidden_abs_bound, i8_abs_bound],
    )?;
    require_centered_m31(shared_down_bound, "shared down projection")?;
    require_down_encoding_bound(shared_down_bound, "shared down projection")?;

    let output_multiplier = u128::try_from(policy.output.multiplier)
        .map_err(|_| "Stage A output multiplier must be positive")?;
    let selected_sum_bound = u128::from(q15_route_weight_sum_upper_bound(shape.top_k)?);
    let coefficient_sum = selected_sum_bound
        .checked_add(ROUTE_SCALE as u128)
        .ok_or_else(|| "Stage A output coefficient-sum bound overflow".to_string())?;
    let slots_padded = (shape.top_k + 1).next_power_of_two();
    let layout = OutputLayout::for_policy(slots_padded, policy)?;
    let radix = 1u128 << layout.limb_bits;
    let max_limb = radix - 1;
    let max_lhs_carry = (1u128 << layout.carry_widths.iter().sum::<usize>()) - 1;
    let max_lhs_source = checked_bound(
        "output limb source",
        &[max_limb, output_multiplier, coefficient_sum],
    )?
    .checked_add(max_lhs_carry)
    .and_then(|value| value.checked_add(max_limb))
    .ok_or_else(|| "Stage A output limb-source bound overflow".to_string())?;
    let max_lhs_target = max_lhs_carry
        .checked_mul(radix)
        .and_then(|value| value.checked_add(max_limb))
        .ok_or_else(|| "Stage A output limb-target bound overflow".to_string())?;
    if max_lhs_source >= u128::from(P) || max_lhs_target >= u128::from(P) {
        return Err("Stage A output limb equation can wrap M31".into());
    }
    let required_lhs_carry = max_lhs_source.div_ceil(radix);
    if required_lhs_carry > max_lhs_carry {
        return Err("Stage A output carry width is insufficient for the admitted shape".into());
    }

    let max_rhs_carry = (1u128 << RHS_CARRY_BITS) - 1;
    let max_rhs_source = max_rhs_carry + 3 * max_limb;
    let max_rhs_target = max_rhs_carry * radix + max_limb;
    if max_rhs_source >= u128::from(P) || max_rhs_target >= u128::from(P) {
        return Err("Stage A output rhs limb equation can wrap M31".into());
    }

    let max_output_constant = checked_bound(
        "output constant",
        &[output_multiplier, DOWN_BIAS as u128, coefficient_sum],
    )?;
    if max_output_constant >= (1u128 << layout.target_widths.iter().sum::<usize>()) {
        return Err(format!(
            "Stage A output constant exceeds the {}-limb capacity",
            layout.n_target_limbs()
        ));
    }
    Ok(())
}

fn grouped_router_inputs(inputs: &[Vec<i16>]) -> Result<Vec<Vec<i8>>, String> {
    inputs
        .iter()
        .map(|input| {
            input
                .iter()
                .map(|&value| {
                    i8::try_from(value).map_err(|_| {
                        "Stage A fused router requires signed int8 inputs".to_string()
                    })
                })
                .collect()
        })
        .collect()
}

fn validate_fused_arithmetic_domain(
    shape: StageAShape,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
) -> Result<(), String> {
    if inputs.len() != shape.batch || inputs.iter().any(|input| input.len() != shape.dimension) {
        return Err("Stage A fused input shape mismatch".into());
    }
    let input_l1_bound = inputs.iter().try_fold(0u128, |max_l1, row| {
        let row_l1 = row.iter().try_fold(0u128, |acc, &value| {
            i8::try_from(value)
                .map(|value| acc + i128::from(value).unsigned_abs())
                .map_err(|_| "Stage A fused relation requires signed-int8 inputs".to_string())
        })?;
        Ok::<u128, String>(max_l1.max(row_l1))
    })?;
    validate_stage_a_arithmetic_domain_with_l1_bound(shape, policy, input_l1_bound)
}

fn nonlinear_domain_len(shape: StageAShape) -> Result<usize, String> {
    let shared_width = shape
        .expert_width
        .checked_mul(shape.shared_factor)
        .ok_or_else(|| "Stage A nonlinear shared-width overflow".to_string())?;
    let stride = shape
        .expert_width
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A nonlinear hidden padding overflow".to_string())?;
    let shared_padded = shared_width
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A nonlinear shared hidden padding overflow".to_string())?;
    let shared_slots = shared_padded.div_ceil(stride);
    let active_slots = shape
        .top_k
        .checked_add(shared_slots)
        .ok_or_else(|| "Stage A nonlinear active-slot count overflow".to_string())?;
    let padded_slots = active_slots
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A nonlinear slot padding overflow".to_string())?;
    let logical_len = padded_slots
        .checked_mul(stride)
        .ok_or_else(|| "Stage A nonlinear active domain overflow".to_string())?;
    Ok(logical_len.max(TABLE_LEN))
}

fn nonlinear_log_size(shape: StageAShape) -> Result<u32, String> {
    Ok(nonlinear_domain_len(shape)?.ilog2())
}

fn output_rows(shape: StageAShape) -> usize {
    shape.dimension.next_power_of_two()
}

fn router_supercolumn_vars(
    tokens: usize,
    dimension: usize,
    experts: usize,
) -> Result<usize, String> {
    let matrix_cells = [
        tokens
            .checked_mul(dimension)
            .ok_or_else(|| "grouped gating X cell count overflow".to_string())?,
        dimension
            .checked_mul(experts)
            .ok_or_else(|| "grouped gating Wg cell count overflow".to_string())?,
        tokens
            .checked_mul(experts)
            .ok_or_else(|| "grouped gating G cell count overflow".to_string())?,
    ];
    let max_cells = matrix_cells.into_iter().max().unwrap();
    let segment_len = max_cells
        .checked_next_power_of_two()
        .ok_or_else(|| "grouped gating segment length overflow".to_string())?;
    let segment_vars = segment_len.ilog2() as usize;
    segment_vars
        .checked_add(2)
        .ok_or_else(|| "grouped gating variable count overflow".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideLayoutGeometry {
    regular_log: usize,
    shared_log: usize,
    anchor_log: usize,
}

impl WideLayoutGeometry {
    fn from_lengths(
        shape: StageAShape,
        regular_len: usize,
        shared_len: usize,
        router_m_vars: usize,
    ) -> Result<Self, String> {
        if !regular_len.is_power_of_two() || !shared_len.is_power_of_two() {
            return Err("Stage A down-weight tensors must have power-of-two length".into());
        }
        let anchor_log = (nonlinear_log_size(shape)? as usize)
            .max(output_rows(shape).ilog2() as usize)
            .max(router_m_vars);
        Ok(Self {
            regular_log: regular_len.ilog2() as usize,
            shared_log: shared_len.ilog2() as usize,
            anchor_log,
        })
    }

    fn from_public(shape: StageAShape, active_regular_experts: usize) -> Result<Self, String> {
        let overflow = || "Stage A wide-layout geometry overflow".to_string();
        let active_padded = active_regular_experts
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(overflow)?;
        let hidden_padded = shape
            .expert_width
            .checked_next_power_of_two()
            .ok_or_else(overflow)?;
        let shared_padded = shape
            .shared_width()
            .checked_next_power_of_two()
            .ok_or_else(overflow)?;
        let output_padded = shape
            .dimension
            .checked_next_power_of_two()
            .ok_or_else(overflow)?;
        let regular_len = active_padded
            .checked_mul(hidden_padded)
            .and_then(|value| value.checked_mul(output_padded))
            .ok_or_else(overflow)?;
        let shared_len = shared_padded
            .checked_mul(output_padded)
            .ok_or_else(overflow)?;
        let router_m_vars = router_supercolumn_vars(shape.batch, shape.dimension, shape.experts)?;
        Self::from_lengths(shape, regular_len, shared_len, router_m_vars)
    }
}

fn default_wide_log_columns(geometry: &WideLayoutGeometry) -> Result<usize, String> {
    let tallest = geometry.regular_log.max(geometry.shared_log);
    let shortest = geometry.regular_log.min(geometry.shared_log);
    let lowest = tallest.saturating_sub(geometry.anchor_log).max(1);
    let highest = shortest
        .saturating_sub(WIDE_LAYOUT_MIN_LOG_ROWS)
        .min(WIDE_LAYOUT_MAX_LOG_COLUMNS);
    if lowest > highest {
        return Err(format!(
            "no admissible wide down-weight layout: tensors 2^{} and 2^{}, anchor height 2^{}",
            geometry.regular_log, geometry.shared_log, geometry.anchor_log
        ));
    }
    Ok(WIDE_LAYOUT_LOG_COLUMNS.clamp(lowest, highest))
}

fn validate_wide_layout(geometry: &WideLayoutGeometry, log_columns: usize) -> Result<(), String> {
    if log_columns == 0 || log_columns > WIDE_LAYOUT_MAX_LOG_COLUMNS {
        return Err(format!(
            "Stage A wide layout needs 1 <= k <= {WIDE_LAYOUT_MAX_LOG_COLUMNS}, got k={log_columns}"
        ));
    }
    for (label, tensor_log) in [
        ("regular", geometry.regular_log),
        ("shared", geometry.shared_log),
    ] {
        let rows_log = tensor_log
            .checked_sub(log_columns)
            .filter(|&rows_log| rows_log >= WIDE_LAYOUT_MIN_LOG_ROWS)
            .ok_or_else(|| {
                format!(
                    "Stage A wide layout leaves fewer than 2^{WIDE_LAYOUT_MIN_LOG_ROWS} rows for \
                     the {label} down-weight tensor: n={tensor_log}, k={log_columns}"
                )
            })?;
        if rows_log > geometry.anchor_log {
            return Err(format!(
                "Stage A wide layout column height 2^{rows_log} of the {label} down-weight \
                 tensor exceeds the tallest other component 2^{}: trace trees 1 and 2 would \
                 not reach the composition height",
                geometry.anchor_log
            ));
        }
    }
    Ok(())
}

pub fn wide_layout_log_columns_for_routes(
    shape: StageAShape,
    routes: &[Vec<usize>],
) -> Result<usize, String> {
    let active = routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let geometry = WideLayoutGeometry::from_public(shape, active)?;
    let log_columns = default_wide_log_columns(&geometry)?;
    validate_wide_layout(&geometry, log_columns)?;
    Ok(log_columns)
}

const MAX_FUSED_CIRCLE_DOMAIN_LOG_SIZE: u32 = 30;

fn validate_fused_circle_domain(label: &str, len: usize) -> Result<(), String> {
    if len < 2 || !len.is_power_of_two() {
        return Err(format!(
            "Stage A fused {label} domain must have power-of-two length at least 2"
        ));
    }
    if len.ilog2() > MAX_FUSED_CIRCLE_DOMAIN_LOG_SIZE {
        return Err(format!(
            "Stage A fused {label} domain exceeds the M31 circle-domain limit"
        ));
    }
    Ok(())
}

fn validate_fused_protocol_domain(shape: StageAShape) -> Result<(), String> {
    shape.validate()?;
    if shape.dimension < 2 {
        return Err("Stage A fused dimension must be a power of two at least 2".into());
    }
    if shape.expert_width < 2 {
        return Err("Stage A fused expert width must be at least 2".into());
    }
    if shape.experts < 2 {
        return Err("Stage A fused expert count must be a power of two at least 2".into());
    }

    let hidden_padded = shape
        .expert_width
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A fused hidden padding overflow".to_string())?;
    let shared_padded = shape
        .shared_width()
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A fused shared padding overflow".to_string())?;
    let output_padded = shape.dimension;
    let _maximum_active_experts = shape
        .batch
        .checked_mul(shape.top_k)
        .ok_or_else(|| "Stage A fused active-expert count overflow".to_string())?
        .min(shape.experts)
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A fused active-expert padding overflow".to_string())?;
    let router_weight_domain = shape
        .experts
        .checked_mul(shape.dimension)
        .ok_or_else(|| "Stage A fused router-weight domain overflow".to_string())?;
    let regular_weight_domain = hidden_padded
        .checked_mul(output_padded)
        .ok_or_else(|| "Stage A fused regular-weight domain overflow".to_string())?;
    let shared_weight_domain = shared_padded
        .checked_mul(output_padded)
        .ok_or_else(|| "Stage A fused shared-weight domain overflow".to_string())?;

    for (label, len) in [
        ("router", shape.experts),
        ("input", shape.dimension),
        ("hidden", hidden_padded),
        ("shared hidden", shared_padded),
        ("nonlinear trace", nonlinear_domain_len(shape)?),
        ("output", output_padded),
        ("router weight", router_weight_domain),
        ("regular down weight", regular_weight_domain),
        ("shared down weight", shared_weight_domain),
    ] {
        validate_fused_circle_domain(label, len)?;
    }
    Ok(())
}

pub fn validate_statement_shape(statement: &StageANonlinearBatchStatement) -> Result<(), String> {
    let shape = statement.shape;
    validate_fused_protocol_domain(shape)?;
    let unique_active_regular = statement
        .routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);
    let active_regular_padded = unique_active_regular
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A fused active-expert padding overflow".to_string())?;
    let hidden_padded = shape
        .expert_width
        .checked_next_power_of_two()
        .ok_or_else(|| "Stage A fused hidden padding overflow".to_string())?;
    let regular_weight_domain = active_regular_padded
        .checked_mul(hidden_padded)
        .and_then(|value| value.checked_mul(shape.dimension))
        .ok_or_else(|| "Stage A fused regular-weight domain overflow".to_string())?;
    if regular_weight_domain < 2 || !regular_weight_domain.is_power_of_two() {
        return Err(
            "Stage A fused regular down weight domain must have power-of-two length at least 2"
                .into(),
        );
    }
    if !(1..=WIDE_LAYOUT_MAX_LOG_COLUMNS).contains(&statement.wide_log_columns) {
        return Err(format!(
            "Stage A wide layout needs 1 <= k <= {WIDE_LAYOUT_MAX_LOG_COLUMNS}, got k={}",
            statement.wide_log_columns
        ));
    }
    if ![1, 4, 8, 16, 32, 64, 128, 256, 512, 1024].contains(&shape.batch) {
        return Err("Stage A nonlinear batch requires T in {1,4,8,16,32,64,128,256,512,1024}".into());
    }
    if statement.router_outputs_raw.len() != shape.batch
        || statement.routes.len() != shape.batch
        || statement.route_weights_q15.len() != shape.batch
        || statement.output_q8.len() != shape.batch
    {
        return Err("Stage A nonlinear batch statement count mismatch".into());
    }
    for token in 0..shape.batch {
        let route = &statement.routes[token];
        if statement.router_outputs_raw[token].len() != shape.experts
            || route.len() != shape.top_k
            || statement.route_weights_q15[token].len() != shape.top_k
            || statement.output_q8[token].len() != shape.dimension
        {
            return Err("Stage A nonlinear batch statement row shape mismatch".into());
        }
        let unique = route.iter().copied().collect::<BTreeSet<_>>();
        if route.iter().any(|&expert| expert >= shape.experts) || unique.len() != route.len() {
            return Err("Stage A nonlinear batch route is malformed".into());
        }
        if statement.route_weights_q15[token]
            .iter()
            .any(|&weight| i128::from(weight) > ROUTE_SCALE)
        {
            return Err("Stage A nonlinear batch route weight exceeds Q0.15".into());
        }
        let route_weight_sum = statement.route_weights_q15[token]
            .iter()
            .try_fold(0u64, |sum, &weight| sum.checked_add(u64::from(weight)))
            .ok_or_else(|| "Stage A nonlinear batch route weight sum overflow".to_string())?;
        if route_weight_sum > q15_route_weight_sum_upper_bound(shape.top_k)? {
            return Err(
                "Stage A nonlinear batch route weight sum exceeds the Q0.15 rounding budget".into(),
            );
        }
        if statement.output_q8[token]
            .iter()
            .any(|&value| !(-2048..=2047).contains(&value))
        {
            return Err("Stage A nonlinear batch output exceeds signed i12".into());
        }
    }
    Ok(())
}

fn validate_sumcheck_round_count(
    proof: &SumcheckProof,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let actual = proof.round_polys.len();
    if actual != expected {
        return Err(format!(
            "{label} round count mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn mix_digest(channel: &mut Production100Channel, digest: &[u8; 32]) {
    for chunk in digest.chunks_exact(8) {
        channel.mix_u64(u64::from_le_bytes(
            chunk.try_into().expect("eight-byte digest chunk"),
        ));
    }
}

fn rlc(values: &[SecureField], challenge: SecureField) -> SecureField {
    values
        .iter()
        .rev()
        .fold(SecureField::zero(), |acc, &value| acc * challenge + value)
}

fn token_powers(lambda: SecureField, batch: usize) -> Vec<SecureField> {
    let mut power = SecureField::one();
    (0..batch)
        .map(|_| {
            let current = power;
            power *= lambda;
            current
        })
        .collect()
}

fn eq_index(point: &[SecureField], index: usize) -> SecureField {
    point
        .iter()
        .enumerate()
        .map(|(bit, &challenge)| {
            let expected = (index >> (point.len() - 1 - bit)) & 1;
            if expected == 1 {
                challenge
            } else {
                SecureField::one() - challenge
            }
        })
        .product()
}

fn router_public_statement_digest(xs: &[Vec<i8>], scores: &[Vec<i64>]) -> [u64; 4] {
    let mut hasher = Blake2s256::new();
    hasher.update(b"stwo-ml:registered-router-public:v1");
    hasher.update((xs.len() as u64).to_le_bytes());
    hasher.update((xs.first().map_or(0, Vec::len) as u64).to_le_bytes());
    for row in xs {
        hasher.update(bytemuck_cast_i8(row));
    }
    hasher.update((scores.len() as u64).to_le_bytes());
    hasher.update((scores.first().map_or(0, Vec::len) as u64).to_le_bytes());
    for row in scores {
        for &score in row {
            hasher.update(score.to_le_bytes());
        }
    }
    let digest = hasher.finalize();
    let mut out = [0u64; 4];
    for (idx, chunk) in digest.chunks_exact(8).enumerate() {
        out[idx] = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
    }
    out
}

fn mix_registered_router_statement(
    channel: &mut Production100Channel,
    xs: &[Vec<i8>],
    scores: &[Vec<i64>],
    router_commitment: &[u8; 32],
) {
    channel.mix_u64(REGISTERED_ROUTER_DOMAIN_TAG);
    channel.mix_u64(xs.len() as u64);
    channel.mix_u64(xs.first().map_or(0, Vec::len) as u64);
    channel.mix_u64(scores.first().map_or(0, Vec::len) as u64);
    for word in router_public_statement_digest(xs, scores) {
        channel.mix_u64(word);
    }
    mix_digest(channel, router_commitment);
}

fn flatten_router_scores(scores: &[Vec<i64>]) -> Vec<SecureField> {
    scores
        .iter()
        .flat_map(|row| row.iter().map(|&value| SecureField::from(i64_mod_p(value))))
        .collect()
}

fn restrict_router_inputs(xs: &[Vec<i8>], r_tokens: &[SecureField]) -> Vec<SecureField> {
    let weights = eq_evals(r_tokens);
    let dimension = xs[0].len();
    (0..dimension)
        .map(|col| {
            xs.iter()
                .zip(&weights)
                .map(|(row, &w)| SecureField::from(i8_to_m31(row[col])) * w)
                .sum()
        })
        .collect()
}

fn verify_registered_router_sumcheck(
    xs: &[Vec<i8>],
    scores: &[Vec<i64>],
    proof: &StageARegisteredRouterProof,
    channel: &mut Production100Channel,
) -> Result<Vec<SecureField>, String> {
    let tokens = xs.len();
    let dimension = xs.first().map_or(0, Vec::len);
    let experts = scores.first().map_or(0, Vec::len);
    if !tokens.is_power_of_two() || !dimension.is_power_of_two() || !experts.is_power_of_two() {
        return Err("Stage A registered router dimensions must be powers of two".into());
    }
    let log_tokens = tokens.ilog2() as usize;
    let log_dimension = dimension.ilog2() as usize;
    let log_experts = experts.ilog2() as usize;
    validate_sumcheck_round_count(
        &proof.gate_proof,
        log_dimension,
        "Stage A registered router gate sumcheck",
    )?;
    let r_tokens = channel.draw_secure_felts(log_tokens);
    let r_experts = channel.draw_secure_felts(log_experts);
    let mut score_point = r_tokens.clone();
    score_point.extend_from_slice(&r_experts);
    let expected_gate_claim = mle_at(&flatten_router_scores(scores), &score_point);
    if proof.gate_claim != expected_gate_claim {
        return Err("Stage A registered router gate claim does not match public scores".into());
    }
    channel.mix_felts(&[proof.gate_claim]);
    let _lambda = channel.draw_secure_felt();
    let (s_dimension, final_gate) =
        partially_verify(proof.gate_claim, &proof.gate_proof, channel)?;
    if s_dimension.len() != log_dimension {
        return Err("Stage A registered router assignment length mismatch".into());
    }
    let x_opening = mle_at(&restrict_router_inputs(xs, &r_tokens), &s_dimension);
    if final_gate != x_opening * proof.wg_opening {
        return Err("Stage A registered router final gate check failed".into());
    }
    channel.mix_felts(&[proof.wg_opening]);
    let mut router_weight_point = s_dimension;
    router_weight_point.extend_from_slice(&r_experts);
    Ok(router_weight_point)
}

fn mix_registered_statement(
    channel: &mut Production100Channel,
    statement: &StageANonlinearBatchStatement,
    context: &StageARegisteredContext,
    profile: &SecurityProfile,
    chain: &StageAChainLink,
) {
    channel.mix_u64(REGISTERED_BATCH_NONLINEAR_DOMAIN_TAG);
    channel.mix_u64(u64::from(STAGE_A_REGISTERED_RECEIPT_WIRE_VERSION));
    let profile_id = profile.id().as_bytes();
    channel.mix_u64(profile_id.len() as u64);
    for chunk in profile_id.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        channel.mix_u64(u64::from_le_bytes(word));
    }
    channel.mix_u64(u64::from(chain.layer));
    mix_digest(channel, &chain.prev_receipt_digest);
    for value in [
        statement.shape.batch,
        statement.shape.dimension,
        statement.shape.expert_width,
        statement.shape.experts,
        statement.shape.top_k,
        statement.shape.shared_factor,
    ] {
        channel.mix_u64(value as u64);
    }
    mix_digest(channel, &statement.policy_digest);
    mix_digest(channel, &statement.model_digest);
    mix_digest(channel, &statement.model_registry_manifest_digest);
    mix_digest(channel, &statement.table_digest);
    mix_digest(channel, &statement.input_digest);
    mix_digest(channel, &context.scheme_digest);
    mix_digest(channel, &context.router_commitment);
    mix_digest(channel, &context.shared_commitment);
    mix_digest(channel, &context.registered_expert_root);
    for opening in &context.active_expert_openings {
        channel.mix_u64(opening.expert_id as u64);
        mix_digest(channel, &opening.group_commitment);
    }
    for token in 0..statement.shape.batch {
        channel.mix_u64(token as u64);
        for &value in &statement.router_outputs_raw[token] {
            channel.mix_u64(i64_wire_word(value));
        }
        for &expert in &statement.routes[token] {
            channel.mix_u64(expert as u64);
        }
        for &weight in &statement.route_weights_q15[token] {
            channel.mix_u64(u64::from(weight));
        }
        for &value in &statement.output_q8[token] {
            channel.mix_u64(i64_wire_word(i64::from(value)));
        }
    }
}

fn draw_cprs_binding_seed(channel: &mut Production100Channel) -> [u8; 32] {
    let felt0 = channel.draw_secure_felt().to_m31_array();
    let felt1 = channel.draw_secure_felt().to_m31_array();
    let mut seed = [0u8; 32];
    for (idx, coord) in felt0.into_iter().chain(felt1).enumerate() {
        seed[idx * 4..(idx + 1) * 4].copy_from_slice(&coord.0.to_le_bytes());
    }
    seed
}

fn cprs_layer_outer_digest(
    binding_seed: &[u8; 32],
    statement: &StageANonlinearBatchStatement,
    active_experts: &[usize],
    stage_b: Option<&StageBDeferredCprsClaims>,
) -> [u8; 32] {
    let d_a = cprs_outer_digest(
        binding_seed,
        &statement.model_digest,
        &statement.policy_digest,
        &statement.table_digest,
        &statement.input_digest,
        statement.shape.batch,
        active_experts,
        CPRS_CLASS_TAG_LAYER,
        &LAYER_SLOT_ROLES,
    );
    let Some(b) = stage_b else {
        return d_a;
    };
    let mut h = Blake2s256::new();
    h.update(STAGE_AB_LAYER_CPRS_OUTER_DOMAIN);
    h.update(d_a);
    h.update(b.statement_digest);
    h.update(b.attention_commitment);
    h.update((b.chunk_commitments.len() as u64).to_le_bytes());
    for c in &b.chunk_commitments {
        h.update(c);
    }
    h.update((b.chunk_openings.len() as u64).to_le_bytes());
    for op in &b.chunk_openings {
        h.update((op.chunk as u64).to_le_bytes());
    }
    if let Some(ref w) = b.witness_opening {
        w.update_outer_hasher(&mut h);
    }
    h.finalize().into()
}

struct FusedGroupedContext {
    active_regular_experts: Vec<usize>,
    active_regular_padded: usize,
    hidden_padded: usize,
    shared_padded: usize,
    output_padded: usize,
}

fn weight_free_grouped_context(statement: &StageANonlinearBatchStatement) -> FusedGroupedContext {
    let active_regular_experts = statement
        .routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let active_regular_padded = active_regular_experts.len().next_power_of_two();
    let hidden_padded = statement.shape.expert_width.next_power_of_two();
    let shared_padded = statement.shape.shared_width().next_power_of_two();
    let output_padded = statement.shape.dimension.next_power_of_two();
    FusedGroupedContext {
        active_regular_experts,
        active_regular_padded,
        hidden_padded,
        shared_padded,
        output_padded,
    }
}

struct GroupedGateUpInputs {
    regular_inputs: Vec<Vec<SecureField>>,
    shared_input: Vec<SecureField>,
}

struct GroupedGateUpOutputs {
    regular_gate_outputs: Vec<Vec<SecureField>>,
    regular_up_outputs: Vec<Vec<SecureField>>,
    shared_gate_output: Vec<SecureField>,
    shared_up_output: Vec<SecureField>,
}

fn verify_expert_activations(
    act: &StageAPublicExpertActivations,
    width: usize,
    dimension: usize,
    policy: StageANumericPolicy,
    token_idx: usize,
    slot_idx: Option<usize>,
) -> Result<(), String> {
    let label = || match slot_idx {
        Some(s) => format!("token {token_idx} regular slot {s}"),
        None => format!("token {token_idx} shared"),
    };
    if act.gate_raw.len() != width
        || act.up_raw.len() != width
        || act.gate_q8.len() != width
        || act.up_q8.len() != width
        || act.product.len() != width
        || act.hidden_q8.len() != width
        || act.down_raw.len() != dimension
    {
        return Err(format!("{}: public activation vector length mismatch", label()));
    }
    for i in 0..width {
        let g_req = requant_with_rounding(i128::from(act.gate_raw[i]), policy.gate, policy.rounding)
            .map_err(|e| format!("{} gate_q8[{i}]: {e}", label()))?;
        let expected_gate_q8 = i16::try_from(g_req.value)
            .map_err(|_| format!("{} gate_q8[{i}] exceeds i16", label()))?;
        if act.gate_q8[i] != expected_gate_q8 {
            return Err(format!(
                "{} gate_q8[{i}] mismatch: got {}, recomputed {}",
                label(),
                act.gate_q8[i],
                expected_gate_q8
            ));
        }

        let u_req = requant_with_rounding(i128::from(act.up_raw[i]), policy.up, policy.rounding)
            .map_err(|e| format!("{} up_q8[{i}]: {e}", label()))?;
        let expected_up_q8 = i16::try_from(u_req.value)
            .map_err(|_| format!("{} up_q8[{i}] exceeds i16", label()))?;
        if act.up_q8[i] != expected_up_q8 {
            return Err(format!(
                "{} up_q8[{i}] mismatch: got {}, recomputed {}",
                label(),
                act.up_q8[i],
                expected_up_q8
            ));
        }

        let silu = silu_q8_for_policy(i64::from(act.gate_q8[i]), policy)
            .map_err(|e| format!("{} silu[{i}]: {e}", label()))?;
        let expected_product = i32::from(silu) * i32::from(act.up_q8[i]);
        if act.product[i] != expected_product {
            return Err(format!(
                "{} product[{i}] mismatch: got {}, recomputed {}",
                label(),
                act.product[i],
                expected_product
            ));
        }

        let h_req =
            requant_with_rounding(i128::from(act.product[i]), policy.product, policy.rounding)
                .map_err(|e| format!("{} hidden_q8[{i}]: {e}", label()))?;
        let expected_hidden_q8 = i16::try_from(h_req.value)
            .map_err(|_| format!("{} hidden_q8[{i}] exceeds i16", label()))?;
        if act.hidden_q8[i] != expected_hidden_q8 {
            return Err(format!(
                "{} hidden_q8[{i}] mismatch: got {}, recomputed {}",
                label(),
                act.hidden_q8[i],
                expected_hidden_q8
            ));
        }
    }
    Ok(())
}

pub fn verify_public_token_activations(
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
    tokens: &[StageAPublicTokenActivations],
) -> Result<(), String> {
    let shape = statement.shape;
    if tokens.len() != shape.batch {
        return Err("Stage A public-witness token count does not match batch size".into());
    }
    let shared_width = shape.shared_width();
    for (t, tok) in tokens.iter().enumerate() {
        let route =
            derive_stage_a_route_from_raw(&statement.router_outputs_raw[t], shape, policy)?;
        if statement.routes[t] != route.ranked_route {
            return Err(format!(
                "Stage A public-witness token {t}: statement route does not match recomputed top-k route"
            ));
        }
        let expected_weights: Vec<u16> = route
            .ranked_route
            .iter()
            .map(|&e| route.softmax_q15[e])
            .collect();
        if statement.route_weights_q15[t] != expected_weights {
            return Err(format!(
                "Stage A public-witness token {t}: statement route_weights_q15 do not match recomputed softmax_q15"
            ));
        }
        if tok.regular.len() != shape.top_k {
            return Err(format!(
                "Stage A public-witness token {t}: regular slot count {} != top_k {}",
                tok.regular.len(),
                shape.top_k
            ));
        }
        for (slot, reg) in tok.regular.iter().enumerate() {
            verify_expert_activations(
                reg,
                shape.expert_width,
                shape.dimension,
                policy,
                t,
                Some(slot),
            )?;
        }
        verify_expert_activations(
            &tok.shared,
            shared_width,
            shape.dimension,
            policy,
            t,
            None,
        )?;
        for j in 0..shape.dimension {
            let mut acc = i128::from(tok.shared.down_raw[j]) << 15;
            for (slot, reg) in tok.regular.iter().enumerate() {
                let alpha = i128::from(statement.route_weights_q15[t][slot]);
                acc = acc
                    .checked_add(alpha * i128::from(reg.down_raw[j]))
                    .ok_or_else(|| "Stage A public-witness weighted output overflow".to_string())?;
            }
            let out_req = requant_with_rounding(acc, policy.output, policy.rounding)?;
            let expected_out = i16::try_from(out_req.value)
                .map_err(|_| "Stage A public-witness output exceeds i16".to_string())?;
            if statement.output_q8[t][j] != expected_out {
                return Err(format!(
                    "token {t} output_q8[{j}] mismatch: statement has {}, recomputed {}",
                    statement.output_q8[t][j], expected_out
                ));
            }
        }
    }
    Ok(())
}

fn hash_i32_slice_buffered(h: &mut Blake2s256, values: &[i32]) {
    for &v in values {
        h.update(v.to_le_bytes());
    }
}

fn hash_i16_slice_buffered(h: &mut Blake2s256, values: &[i16]) {
    for &v in values {
        h.update(v.to_le_bytes());
    }
}

fn public_tokens_digest(tokens: &[StageAPublicTokenActivations]) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"stwo-ml:stage-a-public-witness-tokens:v1");
    h.update((tokens.len() as u64).to_le_bytes());
    let hash_expert = |h: &mut Blake2s256, act: &StageAPublicExpertActivations| {
        h.update((act.gate_raw.len() as u64).to_le_bytes());
        hash_i32_slice_buffered(h, &act.gate_raw);
        hash_i32_slice_buffered(h, &act.up_raw);
        hash_i16_slice_buffered(h, &act.gate_q8);
        hash_i16_slice_buffered(h, &act.up_q8);
        hash_i32_slice_buffered(h, &act.product);
        hash_i16_slice_buffered(h, &act.hidden_q8);
        h.update((act.down_raw.len() as u64).to_le_bytes());
        hash_i32_slice_buffered(h, &act.down_raw);
    };
    for tok in tokens {
        h.update((tok.regular.len() as u64).to_le_bytes());
        for reg in &tok.regular {
            hash_expert(&mut h, reg);
        }
        hash_expert(&mut h, &tok.shared);
    }
    h.finalize().into()
}

#[inline]
fn coeff_coords_i128(c: SecureField) -> [i128; 4] {
    c.to_m31_array().map(|m| i128::from(m.0))
}

#[inline]
fn coeff_coords_i64(c: SecureField) -> [i64; 4] {
    c.to_m31_array().map(|m| i64::from(m.0))
}

#[inline]
pub fn reduce_i128_m31(v: i128) -> BaseField {
    const MASK62: u128 = (1u128 << 62) - 1;
    const MASK31: u64 = 0x7FFF_FFFF;
    let u = v.unsigned_abs();
    let s1 = ((u & MASK62) + ((u >> 62) & MASK62) + (u >> 124)) as u64;
    let x1 = (s1 & MASK31) + (s1 >> 31);
    let x2 = (x1 & MASK31) + (x1 >> 31);
    let x3 = (x2 & MASK31) + (x2 >> 31);
    let rem = if x3 >= MASK31 {
        (x3 - MASK31) as u32
    } else {
        x3 as u32
    };
    let canon = if v < 0 && rem != 0 {
        0x7FFF_FFFFu32 - rem
    } else {
        rem
    };
    BaseField::from_u32_unchecked(canon)
}

#[inline]
fn acc_i128_to_secure(a: [i128; 4]) -> SecureField {
    SecureField::from_m31_array(a.map(reduce_i128_m31))
}

#[inline]
fn check_i32_in_m31_signed_range(v: i32) -> Result<(), String> {
    if i64::from(v).unsigned_abs() >= u64::from(P) {
        return Err("Stage A integer magnitude exceeds the M31 signed embedding range".into());
    }
    Ok(())
}

fn public_grouped_gate_up_inputs(
    inputs: &[Vec<i16>],
    statement: &StageANonlinearBatchStatement,
    grouped: &FusedGroupedContext,
    lambda: SecureField,
) -> Result<GroupedGateUpInputs, String> {
    let shape = statement.shape;
    if inputs.len() != shape.batch
        || stage_a_input_digest(inputs) != statement.input_digest
        || inputs.iter().any(|input| input.len() != shape.dimension)
    {
        return Err("Stage A grouped gate/up inputs do not match the statement".into());
    }
    let active = grouped.active_regular_experts.len();
    let powers = token_powers(lambda, shape.batch);
    let power_coords: Vec<[i64; 4]> = powers.iter().copied().map(coeff_coords_i64).collect();
    let mut expert_tokens = vec![Vec::<usize>::new(); active];
    for token in 0..shape.batch {
        for &expert in &statement.routes[token] {
            let active_position = grouped
                .active_regular_experts
                .binary_search(&expert)
                .map_err(|_| "Stage A routed expert is absent from the active set".to_string())?;
            expert_tokens[active_position].push(token);
        }
    }

    let regular_inputs: Vec<Vec<SecureField>> = expert_tokens
        .iter()
        .map(|toks| {
            let mut acc = vec![[0i64; 4]; shape.dimension];
            for &token in toks {
                let c = power_coords[token];
                let input = &inputs[token];
                for (dst, &value) in acc.iter_mut().zip(input) {
                    let v = i64::from(value);
                    dst[0] += c[0] * v;
                    dst[1] += c[1] * v;
                    dst[2] += c[2] * v;
                    dst[3] += c[3] * v;
                }
            }
            acc.into_iter().map(acc_to_secure).collect()
        })
        .collect();

    let mut sh_acc = vec![[0i64; 4]; shape.dimension];
    for (c, input) in power_coords.iter().zip(inputs) {
        for (dst, &value) in sh_acc.iter_mut().zip(input) {
            let v = i64::from(value);
            dst[0] += c[0] * v;
            dst[1] += c[1] * v;
            dst[2] += c[2] * v;
            dst[3] += c[3] * v;
        }
    }
    let shared_input = sh_acc.into_iter().map(acc_to_secure).collect();

    Ok(GroupedGateUpInputs {
        regular_inputs,
        shared_input,
    })
}

fn public_grouped_gate_up_outputs(
    tokens: &[StageAPublicTokenActivations],
    statement: &StageANonlinearBatchStatement,
    grouped: &FusedGroupedContext,
    lambda: SecureField,
) -> Result<GroupedGateUpOutputs, String> {
    let shape = statement.shape;
    let active = grouped.active_regular_experts.len();
    let shared_width = shape.shared_width();
    let powers = token_powers(lambda, shape.batch);
    let power_coords: Vec<[i128; 4]> = powers.iter().copied().map(coeff_coords_i128).collect();

    let mut expert_hits = vec![Vec::<(usize, usize)>::new(); active];
    for token in 0..shape.batch {
        for (slot, &expert) in statement.routes[token].iter().enumerate() {
            let active_position = grouped
                .active_regular_experts
                .binary_search(&expert)
                .map_err(|_| "Stage A routed expert is absent from the active set".to_string())?;
            expert_hits[active_position].push((token, slot));
        }
    }

    let reg_pairs: Vec<(Vec<SecureField>, Vec<SecureField>)> = expert_hits
        .iter()
        .map(|hits| -> Result<(Vec<SecureField>, Vec<SecureField>), String> {
            let mut g_acc = vec![[0i128; 4]; shape.expert_width];
            let mut u_acc = vec![[0i128; 4]; shape.expert_width];
            for &(token, slot) in hits {
                let c = power_coords[token];
                let reg = &tokens[token].regular[slot];
                for hidden in 0..shape.expert_width {
                    let gv = reg.gate_raw[hidden];
                    let uv = reg.up_raw[hidden];
                    check_i32_in_m31_signed_range(gv)?;
                    check_i32_in_m31_signed_range(uv)?;
                    let gv = i128::from(gv);
                    let uv = i128::from(uv);
                    g_acc[hidden][0] += c[0] * gv;
                    g_acc[hidden][1] += c[1] * gv;
                    g_acc[hidden][2] += c[2] * gv;
                    g_acc[hidden][3] += c[3] * gv;
                    u_acc[hidden][0] += c[0] * uv;
                    u_acc[hidden][1] += c[1] * uv;
                    u_acc[hidden][2] += c[2] * uv;
                    u_acc[hidden][3] += c[3] * uv;
                }
            }
            let mut g_out = vec![SecureField::zero(); grouped.hidden_padded];
            let mut u_out = vec![SecureField::zero(); grouped.hidden_padded];
            for (dst, a) in g_out.iter_mut().zip(g_acc) {
                *dst = acc_i128_to_secure(a);
            }
            for (dst, a) in u_out.iter_mut().zip(u_acc) {
                *dst = acc_i128_to_secure(a);
            }
            Ok((g_out, u_out))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut regular_gate_outputs = Vec::with_capacity(active);
    let mut regular_up_outputs = Vec::with_capacity(active);
    for (g, u) in reg_pairs {
        regular_gate_outputs.push(g);
        regular_up_outputs.push(u);
    }

    let mut sh_gate_acc = vec![[0i128; 4]; shared_width];
    let mut sh_up_acc = vec![[0i128; 4]; shared_width];
    for (c, tok) in power_coords.iter().zip(tokens) {
        for hidden in 0..shared_width {
            let gv = tok.shared.gate_raw[hidden];
            let uv = tok.shared.up_raw[hidden];
            check_i32_in_m31_signed_range(gv)?;
            check_i32_in_m31_signed_range(uv)?;
            let gv = i128::from(gv);
            let uv = i128::from(uv);
            sh_gate_acc[hidden][0] += c[0] * gv;
            sh_gate_acc[hidden][1] += c[1] * gv;
            sh_gate_acc[hidden][2] += c[2] * gv;
            sh_gate_acc[hidden][3] += c[3] * gv;
            sh_up_acc[hidden][0] += c[0] * uv;
            sh_up_acc[hidden][1] += c[1] * uv;
            sh_up_acc[hidden][2] += c[2] * uv;
            sh_up_acc[hidden][3] += c[3] * uv;
        }
    }
    let mut shared_gate_output = vec![SecureField::zero(); grouped.shared_padded];
    let mut shared_up_output = vec![SecureField::zero(); grouped.shared_padded];
    for (dst, a) in shared_gate_output.iter_mut().zip(sh_gate_acc) {
        *dst = acc_i128_to_secure(a);
    }
    for (dst, a) in shared_up_output.iter_mut().zip(sh_up_acc) {
        *dst = acc_i128_to_secure(a);
    }

    Ok(GroupedGateUpOutputs {
        regular_gate_outputs,
        regular_up_outputs,
        shared_gate_output,
        shared_up_output,
    })
}

fn public_grouped_hidden_values(
    tokens: &[StageAPublicTokenActivations],
    statement: &StageANonlinearBatchStatement,
    grouped: &FusedGroupedContext,
    token_challenge: SecureField,
    slot_challenge: SecureField,
) -> Result<(Vec<SecureField>, Vec<SecureField>), String> {
    let shape = statement.shape;
    let token_coefficients = token_powers(token_challenge, shape.batch);
    let slot_coefficients = token_powers(slot_challenge, shape.top_k);
    let active = grouped.active_regular_experts.len();
    let shared_width = shape.shared_width();

    let mut expert_hits = vec![Vec::<(usize, usize, [i64; 4])>::new(); active];
    for (token, &token_coefficient) in token_coefficients.iter().enumerate() {
        for (slot, &slot_coefficient) in slot_coefficients.iter().enumerate() {
            let expert = statement.routes[token][slot];
            let active_position = grouped
                .active_regular_experts
                .binary_search(&expert)
                .map_err(|_| "Stage A routed expert is absent from the active set".to_string())?;
            let c = coeff_coords_i64(token_coefficient * slot_coefficient);
            expert_hits[active_position].push((token, slot, c));
        }
    }

    let mut regular =
        vec![SecureField::zero(); grouped.active_regular_padded * grouped.hidden_padded];
    for (dst_slice, hits) in regular
        .chunks_mut(grouped.hidden_padded)
        .take(active)
        .zip(&expert_hits)
    {
        let mut acc = vec![[0i64; 4]; shape.expert_width];
        for &(token, slot, c) in hits {
            let reg = &tokens[token].regular[slot];
            for coordinate in 0..shape.expert_width {
                let v = i64::from(reg.hidden_q8[coordinate]);
                acc[coordinate][0] += c[0] * v;
                acc[coordinate][1] += c[1] * v;
                acc[coordinate][2] += c[2] * v;
                acc[coordinate][3] += c[3] * v;
            }
        }
        for (coord, a) in acc.into_iter().enumerate() {
            dst_slice[coord] = acc_to_secure(a);
        }
    }

    let mut sh_acc = vec![[0i64; 4]; shared_width];
    for (&token_coefficient, tok) in token_coefficients.iter().zip(tokens) {
        let tc = coeff_coords_i64(token_coefficient);
        for coordinate in 0..shared_width {
            let v = i64::from(tok.shared.hidden_q8[coordinate]);
            sh_acc[coordinate][0] += tc[0] * v;
            sh_acc[coordinate][1] += tc[1] * v;
            sh_acc[coordinate][2] += tc[2] * v;
            sh_acc[coordinate][3] += tc[3] * v;
        }
    }
    let mut shared = vec![SecureField::zero(); grouped.shared_padded];
    for (coord, a) in sh_acc.into_iter().enumerate() {
        shared[coord] = acc_to_secure(a);
    }
    Ok((regular, shared))
}

fn public_combined_down_outputs(
    tokens: &[StageAPublicTokenActivations],
    statement: &StageANonlinearBatchStatement,
    grouped: &FusedGroupedContext,
    token_challenge: SecureField,
    slot_challenge: SecureField,
) -> Result<(Vec<SecureField>, Vec<SecureField>), String> {
    let shape = statement.shape;
    let output_padded = grouped.output_padded;
    let token_coefficients = token_powers(token_challenge, shape.batch);
    let slot_coefficients = token_powers(slot_challenge, shape.top_k);
    let tok_slot_coords: Vec<Vec<[i128; 4]>> = token_coefficients
        .iter()
        .map(|&tc| {
            slot_coefficients
                .iter()
                .map(|&sc| coeff_coords_i128(tc * sc))
                .collect()
        })
        .collect();
    let tok_coords: Vec<[i128; 4]> = token_coefficients
        .iter()
        .copied()
        .map(coeff_coords_i128)
        .collect();

    for tok in tokens {
        for reg in &tok.regular {
            for &v in &reg.down_raw[..shape.dimension] {
                check_i32_in_m31_signed_range(v)?;
            }
        }
        for &v in &tok.shared.down_raw[..shape.dimension] {
            check_i32_in_m31_signed_range(v)?;
        }
    }

    let mut reg_acc = vec![[0i128; 4]; shape.dimension];
    let mut sh_acc = vec![[0i128; 4]; shape.dimension];
    for (t, tok) in tokens.iter().enumerate() {
        for (slot, reg) in tok.regular.iter().enumerate() {
            let c = tok_slot_coords[t][slot];
            for (k, &v) in reg.down_raw[..shape.dimension].iter().enumerate() {
                let v = i128::from(v);
                reg_acc[k][0] += c[0] * v;
                reg_acc[k][1] += c[1] * v;
                reg_acc[k][2] += c[2] * v;
                reg_acc[k][3] += c[3] * v;
            }
        }
        let tc = tok_coords[t];
        for (k, &v) in tok.shared.down_raw[..shape.dimension].iter().enumerate() {
            let v = i128::from(v);
            sh_acc[k][0] += tc[0] * v;
            sh_acc[k][1] += tc[1] * v;
            sh_acc[k][2] += tc[2] * v;
            sh_acc[k][3] += tc[3] * v;
        }
    }
    let mut regular_out = vec![SecureField::zero(); output_padded];
    let mut shared_out = vec![SecureField::zero(); output_padded];
    for k in 0..shape.dimension {
        regular_out[k] = acc_i128_to_secure(reg_acc[k]);
        shared_out[k] = acc_i128_to_secure(sh_acc[k]);
    }
    Ok((regular_out, shared_out))
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_public_witness_batch_against_context_profiled_with_stage_b(
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAPublicWitnessBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
) -> Result<StageARegisteredVerifierRun, String> {
    policy.validate()?;
    validate_fused_arithmetic_domain(statement.shape, inputs, policy)?;
    validate_statement_shape(statement)?;
    StageACprsParams::for_shape_and_profile(statement.shape, profile)?;

    if context.shape != statement.shape
        || context.model_digest != statement.model_digest
        || context.manifest_digest != statement.model_registry_manifest_digest
    {
        return Err(
            "Stage A public-witness verifier context does not match statement digests".into(),
        );
    }
    if statement.input_digest != stage_a_input_digest(inputs) {
        return Err("Stage A public-witness verifier input digest mismatch".into());
    }
    let grouped = weight_free_grouped_context(statement);
    let active_experts = &grouped.active_regular_experts;
    let cprs_params = context.validate_for_active_experts(active_experts, profile)?;

    if inputs.len() != statement.shape.batch
        || inputs
            .iter()
            .any(|input| input.len() != statement.shape.dimension)
    {
        return Err("Stage A public-witness verifier input shape mismatch".into());
    }
    if proof.regular_gate_weight_openings.len() != active_experts.len()
        || proof.regular_up_weight_openings.len() != active_experts.len()
        || proof.regular_down_weight_openings.len() != active_experts.len()
        || proof.regular_gate_up_claims.len() != 2 * active_experts.len()
        || proof.shared_gate_up_claims.len() != 2
    {
        return Err("Stage A public-witness proof weight opening count mismatch".into());
    }

    let dimension_rounds = statement.shape.dimension.ilog2() as usize;
    let regular_down_rounds =
        grouped.active_regular_padded.ilog2() as usize + grouped.hidden_padded.ilog2() as usize;
    let shared_down_rounds = grouped.shared_padded.ilog2() as usize;
    for (sumcheck, expected_rounds, label) in [
        (
            &proof.regular_gate_up_sumcheck,
            dimension_rounds,
            "Stage A public-witness regular gate/up sumcheck",
        ),
        (
            &proof.shared_gate_up_sumcheck,
            dimension_rounds,
            "Stage A public-witness shared gate/up sumcheck",
        ),
        (
            &proof.regular_down_sumcheck,
            regular_down_rounds,
            "Stage A public-witness regular down sumcheck",
        ),
        (
            &proof.shared_down_sumcheck,
            shared_down_rounds,
            "Stage A public-witness shared down sumcheck",
        ),
    ] {
        validate_sumcheck_round_count(sumcheck, expected_rounds, label)?;
    }
    let router_inputs = grouped_router_inputs(inputs)?;

    verify_public_token_activations(statement, policy, &proof.tokens)?;

    let channel = &mut Production100Channel::default();
    channel.enter_scope(TranscriptScope::Statement);
    mix_registered_statement(channel, statement, context, profile, chain);
    channel.mix_u64(PUBLIC_WITNESS_MODE_DOMAIN_TAG);
    mix_digest(channel, &public_tokens_digest(&proof.tokens));
    mix_registered_router_statement(
        channel,
        &router_inputs,
        &statement.router_outputs_raw,
        &context.router_commitment,
    );

    channel.enter_scope(TranscriptScope::RouterBinding);
    let router_weight_point = verify_registered_router_sumcheck(
        &router_inputs,
        &statement.router_outputs_raw,
        &proof.router,
        channel,
    )?;

    channel.enter_scope(TranscriptScope::BatchAggregation);
    let lambda = channel.draw_secure_felt();
    channel.mix_u64(SLOT_RLC_DOMAIN_TAG);
    let slot_challenge = channel.draw_secure_felt();
    let gate_up_inputs = public_grouped_gate_up_inputs(inputs, statement, &grouped, lambda)?;
    let gate_up_outputs =
        public_grouped_gate_up_outputs(&proof.tokens, statement, &grouped, lambda)?;
    let regular_projection_point =
        channel.draw_secure_felts(grouped.hidden_padded.ilog2() as usize);
    let shared_projection_point =
        channel.draw_secure_felts(grouped.shared_padded.ilog2() as usize);
    let reg_proj_eq = eq_evals(&regular_projection_point);
    let expected_regular_gate_up_claims: Vec<SecureField> = gate_up_outputs
        .regular_gate_outputs
        .iter()
        .chain(&gate_up_outputs.regular_up_outputs)
        .map(|output| dot(output, &reg_proj_eq))
        .collect();
    if proof.regular_gate_up_claims != expected_regular_gate_up_claims {
        return Err(
            "Stage A public-witness regular gate/up claims do not match public gate_raw/up_raw"
                .into(),
        );
    }
    let sh_proj_eq = eq_evals(&shared_projection_point);
    let expected_shared_gate_up_claims = vec![
        dot(&gate_up_outputs.shared_gate_output, &sh_proj_eq),
        dot(&gate_up_outputs.shared_up_output, &sh_proj_eq),
    ];
    if proof.shared_gate_up_claims != expected_shared_gate_up_claims {
        return Err(
            "Stage A public-witness shared gate/up claims do not match public gate_raw/up_raw"
                .into(),
        );
    }

    channel.enter_scope(TranscriptScope::RegularGateUp);
    channel.mix_felts(&proof.regular_gate_up_claims);
    let regular_gate_up_lambda = channel.draw_secure_felt();
    let regular_combined = rlc(&proof.regular_gate_up_claims, regular_gate_up_lambda);
    let (regular_input_point, regular_gate_up_final) =
        partially_verify(regular_combined, &proof.regular_gate_up_sumcheck, channel)?;
    if regular_input_point.len() != dimension_rounds {
        return Err("Stage A public-witness regular gate/up input point arity mismatch".into());
    }
    channel.mix_felts(&proof.regular_gate_weight_openings);
    channel.mix_felts(&proof.regular_up_weight_openings);
    let reg_in_eq = eq_evals(&regular_input_point);
    let regular_input_openings = gate_up_inputs
        .regular_inputs
        .iter()
        .map(|input| dot(input, &reg_in_eq))
        .collect::<Vec<_>>();
    let mut regular_products = Vec::with_capacity(2 * active_experts.len());
    for (active, &in_eval) in regular_input_openings.iter().enumerate() {
        regular_products.push(in_eval * proof.regular_gate_weight_openings[active]);
    }
    for (active, &in_eval) in regular_input_openings.iter().enumerate() {
        regular_products.push(in_eval * proof.regular_up_weight_openings[active]);
    }
    if regular_gate_up_final != rlc(&regular_products, regular_gate_up_lambda) {
        return Err("Stage A public-witness regular grouped gate/up projection mismatch".into());
    }

    channel.enter_scope(TranscriptScope::SharedGateUp);
    channel.mix_felts(&proof.shared_gate_up_claims);
    let shared_gate_up_lambda = channel.draw_secure_felt();
    let shared_combined = rlc(&proof.shared_gate_up_claims, shared_gate_up_lambda);
    let (shared_input_point, shared_gate_up_final) =
        partially_verify(shared_combined, &proof.shared_gate_up_sumcheck, channel)?;
    if shared_input_point.len() != dimension_rounds {
        return Err("Stage A public-witness shared gate/up input point arity mismatch".into());
    }
    channel.mix_felts(&[
        proof.shared_gate_weight_opening,
        proof.shared_up_weight_opening,
    ]);
    let shared_in_eval = mle_at(&gate_up_inputs.shared_input, &shared_input_point);
    let shared_products = vec![
        shared_in_eval * proof.shared_gate_weight_opening,
        shared_in_eval * proof.shared_up_weight_opening,
    ];
    if shared_gate_up_final != rlc(&shared_products, shared_gate_up_lambda) {
        return Err("Stage A public-witness shared grouped gate/up projection mismatch".into());
    }

    let (grouped_regular, grouped_shared) = public_grouped_hidden_values(
        &proof.tokens,
        statement,
        &grouped,
        lambda,
        slot_challenge,
    )?;
    let (regular_output_values, shared_output_values) = public_combined_down_outputs(
        &proof.tokens,
        statement,
        &grouped,
        lambda,
        slot_challenge,
    )?;
    channel.enter_scope(TranscriptScope::DownProjection);
    let output_point = channel.draw_secure_felts(grouped.output_padded.ilog2() as usize);
    let expected_regular_down_claim = mle_at(&regular_output_values, &output_point);
    let expected_shared_down_claim = mle_at(&shared_output_values, &output_point);
    if proof.regular_down_claim != expected_regular_down_claim
        || proof.shared_down_claim != expected_shared_down_claim
    {
        return Err("Stage A public-witness down claims do not match public down_raw".into());
    }
    channel.mix_felts(&[proof.regular_down_claim, proof.shared_down_claim]);

    channel.enter_scope(TranscriptScope::RegularDown);
    let _regular_down_lambda = channel.draw_secure_felt();
    let (regular_down_assignment, regular_down_final) = partially_verify(
        proof.regular_down_claim,
        &proof.regular_down_sumcheck,
        channel,
    )?;
    if regular_down_assignment.len() != regular_down_rounds {
        return Err("Stage A public-witness regular down assignment arity mismatch".into());
    }
    let active_bits = grouped.active_regular_padded.ilog2() as usize;
    let (expert_point, hidden_point) = regular_down_assignment.split_at(active_bits);
    channel.mix_felts(&proof.regular_down_weight_openings);
    let expected_regular_grouped_claim = mle_at(&grouped_regular, &regular_down_assignment);
    let expected_regular_weight_opening: SecureField = proof
        .regular_down_weight_openings
        .iter()
        .enumerate()
        .map(|(active, &val)| eq_index(expert_point, active) * val)
        .sum();
    if proof.regular_grouped_claim != expected_regular_grouped_claim
        || proof.regular_weight_opening != expected_regular_weight_opening
        || regular_down_final != proof.regular_grouped_claim * proof.regular_weight_opening
    {
        return Err("Stage A public-witness regular down-projection reduction mismatch".into());
    }
    channel.mix_felts(&[proof.regular_grouped_claim, proof.regular_weight_opening]);

    channel.enter_scope(TranscriptScope::SharedDown);
    let _shared_down_lambda = channel.draw_secure_felt();
    let (shared_down_assignment, shared_down_final) = partially_verify(
        proof.shared_down_claim,
        &proof.shared_down_sumcheck,
        channel,
    )?;
    let expected_shared_grouped_claim = mle_at(&grouped_shared, &shared_down_assignment);
    if shared_down_assignment.len() != shared_down_rounds
        || proof.shared_grouped_claim != expected_shared_grouped_claim
        || shared_down_final != proof.shared_grouped_claim * proof.shared_weight_opening
    {
        return Err("Stage A public-witness shared down-projection reduction mismatch".into());
    }
    channel.mix_felts(&[proof.shared_grouped_claim, proof.shared_weight_opening]);

    let binding_seed = draw_cprs_binding_seed(channel);
    let r_router_cprs = stwo_point_to_cprs(&router_weight_point);
    let mut r_reg_gu_stwo = regular_input_point;
    r_reg_gu_stwo.extend_from_slice(&regular_projection_point);
    let mut r_reg_down_stwo = hidden_point.to_vec();
    r_reg_down_stwo.extend_from_slice(&output_point);
    let r_reg_gu_cprs = stwo_point_to_cprs(&r_reg_gu_stwo);
    let r_reg_down_cprs = stwo_point_to_cprs(&r_reg_down_stwo);

    let mut r_sh_gu_stwo = shared_input_point;
    r_sh_gu_stwo.extend_from_slice(&shared_projection_point);
    let mut r_sh_down_stwo = shared_down_assignment;
    r_sh_down_stwo.extend_from_slice(&output_point);
    let r_sh_gu_cprs = stwo_point_to_cprs(&r_sh_gu_stwo);
    let r_sh_down_cprs = stwo_point_to_cprs(&r_sh_down_stwo);

    let is_unified = proof.cprs_regular_proof.is_empty() && proof.cprs_shared_proof.is_empty();
    if stage_b.is_some() && !is_unified {
        return Err(
            "Stage A + Stage B unified CPRS mode requires cprs_regular_proof and cprs_shared_proof to be empty"
                .into(),
        );
    }
    if is_unified {
        let d_layer = cprs_layer_outer_digest(&binding_seed, statement, active_experts, stage_b);

        let extra_b_groups = stage_b.map_or(0, |b| {
            1 + b.chunk_openings.len() + usize::from(b.witness_opening.is_some())
        });
        let total_groups = extra_b_groups + 1 + active_experts.len() + 1;
        let mut commitments = Vec::with_capacity(total_groups);
        let mut group_params_list = Vec::with_capacity(total_groups);
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
            for &c in &b_deferred.leaf_claims {
                claims.push(to_cprs_qm31(c));
            }
            for op in &b_deferred.chunk_openings {
                let chunk_commit = b_deferred
                    .chunk_commitments
                    .get(op.chunk)
                    .copied()
                    .ok_or_else(|| {
                        "Stage B cache opening names an unknown chunk commitment".to_string()
                    })?;
                commitments.push(chunk_commit);
                group_params_list.push(b_deferred.chunk_params.clone());
                points.push(stwo_point_to_cprs(&op.k_point));
                points.push(stwo_point_to_cprs(&op.v_point));
                claims.push(to_cprs_qm31(op.k_value));
                claims.push(to_cprs_qm31(op.v_value));
            }
            if let Some(ref w) = b_deferred.witness_opening {
                commitments.push(w.witness_commitment);
                group_params_list.push(w.witness_params.clone());
                let (w_pts, w_cls) = w.cprs_points_and_claims();
                points.extend(w_pts);
                claims.extend(w_cls);
            }
        }

        commitments.push(context.router_commitment);
        group_params_list.push(cprs_params.router.clone());
        points.push(r_router_cprs);
        claims.push(to_cprs_qm31(proof.router.wg_opening));

        for active in 0..active_experts.len() {
            commitments.push(context.active_expert_openings[active].group_commitment);
            group_params_list.push(cprs_params.regular.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_down_cprs.clone());
            claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[active]));
            claims.push(to_cprs_qm31(proof.regular_up_weight_openings[active]));
            claims.push(to_cprs_qm31(proof.regular_down_weight_openings[active]));
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
        .map_err(|e| format!("Stage A public-witness unified CPRS opening rejected: {e}"))?;
    } else {
        if proof.cprs_regular_proof.is_empty() || proof.cprs_shared_proof.is_empty() {
            return Err(
                "Stage A separate CPRS mode requires non-empty cprs_regular_proof and cprs_shared_proof"
                    .into(),
            );
        }
        let d_router = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
            CPRS_CLASS_TAG_ROUTER,
            &ROUTER_SLOT_ROLES,
        );
        let d_reg = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
            CPRS_CLASS_TAG_REGULAR,
            &REGULAR_SLOT_ROLES,
        );
        let d_sh = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
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
        .map_err(|e| format!("Stage A public-witness router CPRS opening rejected: {e}"))?;

        let reg_commitments: Vec<[u8; 32]> = context
            .active_expert_openings
            .iter()
            .map(|o| o.group_commitment)
            .collect();
        let mut reg_points = Vec::with_capacity(3 * active_experts.len());
        let mut reg_claims = Vec::with_capacity(3 * active_experts.len());
        for active in 0..active_experts.len() {
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_down_cprs.clone());
            reg_claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_up_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_down_weight_openings[active]));
        }
        verify_strict_context(
            &reg_commitments,
            &reg_points,
            &reg_claims,
            &d_reg,
            &cprs_params.regular,
            &proof.cprs_regular_proof,
        )
        .map_err(|e| format!("Stage A public-witness regular CPRS opening rejected: {e}"))?;

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
        .map_err(|e| format!("Stage A public-witness shared CPRS opening rejected: {e}"))?;
    }

    let cprs_digest = cprs_proofs_transcript_digest(
        &proof.cprs_router_proof,
        &proof.cprs_regular_proof,
        &proof.cprs_shared_proof,
    );
    mix_digest(channel, &cprs_digest);

    Ok(StageARegisteredVerifierRun {
        verifier_context_bytes: context.footprint_bytes(),
        transcript_census: channel.census(),
        transcript_trace: channel.trace(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_public_witness_batch_against_anchor_with_stage_b(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAPublicWitnessBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
) -> Result<StageARegisteredVerifierRun, String> {
    if &context.anchor_digest() != anchor {
        return Err("Stage A registered context does not match the public model anchor".into());
    }
    verify_stage_a_public_witness_batch_against_context_profiled_with_stage_b(
        context, inputs, policy, profile, statement, proof, chain, stage_b,
    )
}

fn put_expert_activations(out: &mut Vec<u8>, act: &StageAPublicExpertActivations) {
    for &v in &act.gate_raw {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.up_raw {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.gate_q8 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.up_q8 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.product {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.hidden_q8 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &act.down_raw {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

const WIRE_V8_I24_MIN: i32 = -(1 << 23);
const WIRE_V8_I24_MAX: i32 = (1 << 23) - 1;

fn put_i32_slice_v8(out: &mut Vec<u8>, vals: &[i32]) {
    let fits_i24 = vals
        .iter()
        .all(|&v| (WIRE_V8_I24_MIN..=WIRE_V8_I24_MAX).contains(&v));
    if fits_i24 {
        out.push(0u8);
        for &v in vals {
            let b = v.to_le_bytes();
            out.extend_from_slice(&[b[0], b[1], b[2]]);
        }
    } else {
        out.push(1u8);
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn put_expert_activations_v8(out: &mut Vec<u8>, act: &StageAPublicExpertActivations) {
    put_i32_slice_v8(out, &act.gate_raw);
    put_i32_slice_v8(out, &act.up_raw);
    put_i32_slice_v8(out, &act.down_raw);
}

fn read_i32_slice_v8(
    reader: &mut ReceiptReader<'_>,
    len: usize,
    label: &str,
) -> Result<Vec<i32>, String> {
    let tag = reader.read_u32_byte(&format!("{label} i24/i32 tag"))?;
    let mut out = Vec::with_capacity(len);
    match tag {
        0 => {
            for _ in 0..len {
                let b0 = reader.read_u32_byte(label)?;
                let b1 = reader.read_u32_byte(label)?;
                let b2 = reader.read_u32_byte(label)?;
                let sign = if (b2 & 0x80) != 0 { 0xFFu8 } else { 0x00u8 };
                out.push(i32::from_le_bytes([b0, b1, b2, sign]));
            }
        }
        1 => {
            for _ in 0..len {
                out.push(read_i32_le(reader, label)?);
            }
            if out
                .iter()
                .all(|&v| (WIRE_V8_I24_MIN..=WIRE_V8_I24_MAX).contains(&v))
            {
                return Err(format!(
                    "non-canonical wire v8 i32 fallback tag in {label}: all values fit in i24"
                ));
            }
        }
        other => {
            return Err(format!("invalid wire v8 packing tag {other} in {label}"));
        }
    }
    Ok(out)
}

fn read_i16_le(reader: &mut ReceiptReader<'_>, label: &str) -> Result<i16, String> {
    let lo = reader.read_u32_byte(label)?;
    let hi = reader.read_u32_byte(label)?;
    Ok(i16::from_le_bytes([lo, hi]))
}

fn read_i32_le(reader: &mut ReceiptReader<'_>, label: &str) -> Result<i32, String> {
    let raw = reader.read_u32(label)?;
    Ok(i32::from_le_bytes(raw.to_le_bytes()))
}

fn read_expert_activations(
    reader: &mut ReceiptReader<'_>,
    width: usize,
    dimension: usize,
    label: &str,
) -> Result<StageAPublicExpertActivations, String> {
    let gate_raw_label = format!("{label} gate_raw");
    let gate_raw = (0..width)
        .map(|_| read_i32_le(reader, &gate_raw_label))
        .collect::<Result<Vec<_>, _>>()?;
    let up_raw_label = format!("{label} up_raw");
    let up_raw = (0..width)
        .map(|_| read_i32_le(reader, &up_raw_label))
        .collect::<Result<Vec<_>, _>>()?;
    let gate_q8_label = format!("{label} gate_q8");
    let gate_q8 = (0..width)
        .map(|_| read_i16_le(reader, &gate_q8_label))
        .collect::<Result<Vec<_>, _>>()?;
    let up_q8_label = format!("{label} up_q8");
    let up_q8 = (0..width)
        .map(|_| read_i16_le(reader, &up_q8_label))
        .collect::<Result<Vec<_>, _>>()?;
    let product_label = format!("{label} product");
    let product = (0..width)
        .map(|_| read_i32_le(reader, &product_label))
        .collect::<Result<Vec<_>, _>>()?;
    let hidden_q8_label = format!("{label} hidden_q8");
    let hidden_q8 = (0..width)
        .map(|_| read_i16_le(reader, &hidden_q8_label))
        .collect::<Result<Vec<_>, _>>()?;
    let down_raw_label = format!("{label} down_raw");
    let down_raw = (0..dimension)
        .map(|_| read_i32_le(reader, &down_raw_label))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StageAPublicExpertActivations {
        gate_raw,
        up_raw,
        gate_q8,
        up_q8,
        product,
        hidden_q8,
        down_raw,
    })
}

fn read_expert_activations_v8(
    reader: &mut ReceiptReader<'_>,
    width: usize,
    dimension: usize,
    policy: StageANumericPolicy,
    label: &str,
) -> Result<StageAPublicExpertActivations, String> {
    let gate_raw = read_i32_slice_v8(reader, width, &format!("{label} gate_raw"))?;
    let up_raw = read_i32_slice_v8(reader, width, &format!("{label} up_raw"))?;
    let down_raw = read_i32_slice_v8(reader, dimension, &format!("{label} down_raw"))?;
    let mut gate_q8 = Vec::with_capacity(width);
    let mut up_q8 = Vec::with_capacity(width);
    let mut product = Vec::with_capacity(width);
    let mut hidden_q8 = Vec::with_capacity(width);
    for i in 0..width {
        let g_req = requant_with_rounding(i128::from(gate_raw[i]), policy.gate, policy.rounding)
            .map_err(|e| format!("{label} gate_q8[{i}]: {e}"))?;
        let g_q8 = i16::try_from(g_req.value)
            .map_err(|_| format!("{label} gate_q8[{i}] exceeds i16"))?;
        let u_req = requant_with_rounding(i128::from(up_raw[i]), policy.up, policy.rounding)
            .map_err(|e| format!("{label} up_q8[{i}]: {e}"))?;
        let u_q8 = i16::try_from(u_req.value)
            .map_err(|_| format!("{label} up_q8[{i}] exceeds i16"))?;
        let silu = silu_q8_for_policy(i64::from(g_q8), policy)
            .map_err(|e| format!("{label} silu[{i}]: {e}"))?;
        let prod = i32::from(silu) * i32::from(u_q8);
        let h_req = requant_with_rounding(i128::from(prod), policy.product, policy.rounding)
            .map_err(|e| format!("{label} hidden_q8[{i}]: {e}"))?;
        let h_q8 = i16::try_from(h_req.value)
            .map_err(|_| format!("{label} hidden_q8[{i}] exceeds i16"))?;
        gate_q8.push(g_q8);
        up_q8.push(u_q8);
        product.push(prod);
        hidden_q8.push(h_q8);
    }
    Ok(StageAPublicExpertActivations {
        gate_raw,
        up_raw,
        gate_q8,
        up_q8,
        product,
        hidden_q8,
        down_raw,
    })
}

pub fn stage_a_public_witness_receipt_bytes_with_version(
    proof: &StageAPublicWitnessBatchProof,
    mode_tag: u32,
) -> Result<Vec<u8>, String> {
    if mode_tag != STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS
        && mode_tag != STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8
    {
        return Err(format!(
            "unsupported Stage A public-witness wire mode tag {mode_tag}"
        ));
    }
    let use_v8 = mode_tag == STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8;
    let mut out = Vec::new();
    out.extend_from_slice(STAGE_A_REGISTERED_RECEIPT_MAGIC);
    put_u32(&mut out, mode_tag);
    for tok in &proof.tokens {
        for reg in &tok.regular {
            if use_v8 {
                put_expert_activations_v8(&mut out, reg);
            } else {
                put_expert_activations(&mut out, reg);
            }
        }
        if use_v8 {
            put_expert_activations_v8(&mut out, &tok.shared);
        } else {
            put_expert_activations(&mut out, &tok.shared);
        }
    }
    put_secure_field(&mut out, proof.router.gate_claim);
    put_sumcheck(&mut out, &proof.router.gate_proof)?;
    put_secure_field(&mut out, proof.router.wg_opening);
    put_secure_fields(&mut out, &proof.regular_gate_up_claims)?;
    put_secure_fields(&mut out, &proof.shared_gate_up_claims)?;
    put_sumcheck(&mut out, &proof.regular_gate_up_sumcheck)?;
    put_secure_fields(&mut out, &proof.regular_gate_weight_openings)?;
    put_secure_fields(&mut out, &proof.regular_up_weight_openings)?;
    put_sumcheck(&mut out, &proof.shared_gate_up_sumcheck)?;
    put_secure_field(&mut out, proof.shared_gate_weight_opening);
    put_secure_field(&mut out, proof.shared_up_weight_opening);
    put_secure_field(&mut out, proof.regular_down_claim);
    put_secure_field(&mut out, proof.shared_down_claim);
    put_secure_field(&mut out, proof.regular_grouped_claim);
    put_secure_field(&mut out, proof.shared_grouped_claim);
    put_secure_fields(&mut out, &proof.regular_down_weight_openings)?;
    put_secure_field(&mut out, proof.regular_weight_opening);
    put_secure_field(&mut out, proof.shared_weight_opening);
    put_sumcheck(&mut out, &proof.regular_down_sumcheck)?;
    put_sumcheck(&mut out, &proof.shared_down_sumcheck)?;
    put_bytes(&mut out, &proof.cprs_router_proof)?;
    put_bytes(&mut out, &proof.cprs_regular_proof)?;
    put_bytes(&mut out, &proof.cprs_shared_proof)?;
    Ok(out)
}

pub fn decode_stage_a_public_witness_receipt(
    bytes: &[u8],
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
) -> Result<StageAPublicWitnessBatchProof, String> {
    validate_statement_shape(statement)?;
    if statement.policy_id != policy.id() {
        return Err("Stage A public-witness receipt policy does not match the statement".into());
    }
    let active = statement
        .routes
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let shape = statement.shape;
    let shared_width = shape.shared_width();

    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(STAGE_A_REGISTERED_RECEIPT_MAGIC, "Stage A receipt magic")?;
    let mode_tag = reader.read_u32("Stage A receipt mode tag")?;
    if mode_tag != STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS
        && mode_tag != STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8
    {
        return Err(format!(
            "unsupported Stage A public-witness receipt mode tag {mode_tag}; expected {STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS} or {STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8}"
        ));
    }
    let mut tokens = Vec::with_capacity(shape.batch);
    for t in 0..shape.batch {
        let mut regular = Vec::with_capacity(shape.top_k);
        for slot in 0..shape.top_k {
            let act = if mode_tag == STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8 {
                read_expert_activations_v8(
                    &mut reader,
                    shape.expert_width,
                    shape.dimension,
                    policy,
                    &format!("token {t} regular {slot}"),
                )?
            } else {
                read_expert_activations(
                    &mut reader,
                    shape.expert_width,
                    shape.dimension,
                    &format!("token {t} regular {slot}"),
                )?
            };
            regular.push(act);
        }
        let shared = if mode_tag == STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8 {
            read_expert_activations_v8(
                &mut reader,
                shared_width,
                shape.dimension,
                policy,
                &format!("token {t} shared"),
            )?
        } else {
            read_expert_activations(
                &mut reader,
                shared_width,
                shape.dimension,
                &format!("token {t} shared"),
            )?
        };
        tokens.push(StageAPublicTokenActivations { regular, shared });
    }
    let router = StageARegisteredRouterProof {
        gate_claim: reader.read_secure_field("router gate claim")?,
        gate_proof: reader.read_sumcheck("router gate sumcheck")?,
        wg_opening: reader.read_secure_field("router weight opening")?,
    };
    let regular_gate_up_claims =
        reader.read_secure_fields_exact(2 * active, "regular gate/up claim")?;
    let shared_gate_up_claims = reader.read_secure_fields_exact(2, "shared gate/up claim")?;
    let regular_gate_up_sumcheck = reader.read_sumcheck("regular gate/up sumcheck")?;
    let regular_gate_weight_openings =
        reader.read_secure_fields_exact(active, "regular gate weight opening")?;
    let regular_up_weight_openings =
        reader.read_secure_fields_exact(active, "regular up weight opening")?;
    let shared_gate_up_sumcheck = reader.read_sumcheck("shared gate/up sumcheck")?;
    let shared_gate_weight_opening = reader.read_secure_field("shared gate weight opening")?;
    let shared_up_weight_opening = reader.read_secure_field("shared up weight opening")?;
    let regular_down_claim = reader.read_secure_field("regular down claim")?;
    let shared_down_claim = reader.read_secure_field("shared down claim")?;
    let regular_grouped_claim = reader.read_secure_field("regular grouped claim")?;
    let shared_grouped_claim = reader.read_secure_field("shared grouped claim")?;
    let regular_down_weight_openings =
        reader.read_secure_fields_exact(active, "regular down weight opening")?;
    let regular_weight_opening = reader.read_secure_field("regular weight opening")?;
    let shared_weight_opening = reader.read_secure_field("shared weight opening")?;
    let regular_down_sumcheck = reader.read_sumcheck("regular down sumcheck")?;
    let shared_down_sumcheck = reader.read_sumcheck("shared down sumcheck")?;
    let cprs_router_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "CPRS router proof")?
        .to_vec();
    let cprs_regular_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "CPRS regular proof")?
        .to_vec();
    let cprs_shared_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "CPRS shared proof")?
        .to_vec();
    reader.finish()?;
    let proof = StageAPublicWitnessBatchProof {
        tokens,
        router,
        regular_gate_up_claims,
        shared_gate_up_claims,
        regular_gate_up_sumcheck,
        regular_gate_weight_openings,
        regular_up_weight_openings,
        shared_gate_up_sumcheck,
        shared_gate_weight_opening,
        shared_up_weight_opening,
        regular_down_claim,
        shared_down_claim,
        regular_grouped_claim,
        shared_grouped_claim,
        regular_down_weight_openings,
        regular_weight_opening,
        shared_weight_opening,
        regular_down_sumcheck,
        shared_down_sumcheck,
        cprs_router_proof,
        cprs_regular_proof,
        cprs_shared_proof,
    };
    if stage_a_public_witness_receipt_bytes_with_version(&proof, mode_tag)? != bytes {
        return Err("Stage A public-witness receipt bytes are not canonical".into());
    }
    Ok(proof)
}

pub fn encode_stage_a_public_witness_receipt_matching_header(
    proof: &StageAPublicWitnessBatchProof,
    reference_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let mode_tag = if reference_bytes.len() >= 12 {
        u32::from_le_bytes(reference_bytes[8..12].try_into().expect("4-byte mode tag"))
    } else {
        STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8
    };
    stage_a_public_witness_receipt_bytes_with_version(proof, mode_tag)
}

pub const STAGE_A_RECEIPT_MODE_GKR_LANE: u32 = 9;
pub const STAGE_A_RECEIPT_MODE_ZK_GKR_LANE: u32 = 10;
pub const STAGE_A_RECEIPT_MODE_ZK_FULL: u32 = 11;
pub const GKR_LANE_MODE_DOMAIN_TAG: u64 = 0x5354_4741_474b_5239;
pub const GKR_LANE_WITNESS_OUTER_DOMAIN: &[u8] = b"stwo-ml:stage-a:gkr-witness-cprs-outer:v1";
pub const GKR_LANE_PROOFS_DIGEST_DOMAIN: &[u8] = b"stwo-ml:stage-a:gkr-cprs-proofs-digest:v1";

pub const WITNESS_GROUP_COLS: usize = 17;
pub const WITNESS_GROUP_COLS_LEGACY: usize = 11;
pub const WITNESS_GROUP_COLS_ZK: usize = 18;
pub const WITNESS_GROUP_COLS_ZK_FULL: usize = 30;
pub const WITNESS_SLOT_ROLES: [u8; WITNESS_GROUP_COLS] = [
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3C, 0x3D, 0x3E, 0x3F,
    0x40, 0x41,
];
pub const WITNESS_SLOT_ROLES_LEGACY: [u8; WITNESS_GROUP_COLS_LEGACY] = [
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
];
pub const WITNESS_SLOT_ROLES_ZK: [u8; WITNESS_GROUP_COLS_ZK] = [
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3C, 0x3D, 0x3E, 0x3F,
    0x40, 0x41, 0x3B,
];
pub const WITNESS_SLOT_ROLES_ZK_FULL: [u8; WITNESS_GROUP_COLS_ZK_FULL] = [
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3C, 0x3D, 0x3E, 0x3F,
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x3B,
];

pub const ZK_UNMASKED_GKR_INSTANCES: usize = 39;
pub const ZK_MASK_INSTANCE_LOG_ROWS: u32 = 10;
pub const ZK_MASK_INSTANCE_ROWS: usize = 1 << ZK_MASK_INSTANCE_LOG_ROWS;
pub const ZK_CFS_TABLE_LOG_ROWS: usize = 12;
pub const ZK_CFS_TABLE_ROWS: usize = 1 << ZK_CFS_TABLE_LOG_ROWS;

pub const COL_WIT_GATE_Q: usize = 0;
pub const COL_WIT_UP_Q: usize = 1;
pub const COL_WIT_SILU: usize = 2;
pub const COL_WIT_HIDDEN_Q: usize = 3;
pub const COL_WIT_GATE_REM: usize = 4;
pub const COL_WIT_UP_REM: usize = 5;
pub const COL_WIT_PROD_REM: usize = 6;
pub const COL_WIT_GATE_PARITY: usize = 7;
pub const COL_WIT_UP_PARITY: usize = 8;
pub const COL_WIT_HIDDEN_PARITY: usize = 9;
pub const COL_WIT_AUX: usize = 10;
pub const COL_WIT_OUT_LIMB_0: usize = 11;
pub const COL_WIT_OUT_LIMB_1: usize = 12;
pub const COL_WIT_OUT_LIMB_2: usize = 13;
pub const COL_WIT_OUT_LIMB_3: usize = 14;
pub const COL_WIT_OUT_LIMB_4: usize = 15;
pub const COL_WIT_OUT_LIMB_5: usize = 16;

const GKR_TABLES: usize = 4;
const LOG_TABLE_ROWS: u32 = 12;
const TABLE_ROWS: usize = 1 << LOG_TABLE_ROWS;
const LOG_NL_ROW_CHUNKS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GkrLaneGeometry {
    pub n_wit: usize,
    pub wit_len: usize,
    pub sub_len: usize,
    pub nl_rows_per_token: usize,
    pub nl_total_rows: usize,
    pub out_rows: usize,
    pub rt_rows: usize,
    pub d_pad: usize,
    pub e_pad: usize,
    pub m_pad: usize,
    pub sh_pad: usize,
    pub legacy_11col: bool,
}

impl GkrLaneGeometry {
    pub fn for_shape(shape: StageAShape) -> Result<Self, String> {
        Self::for_shape_inner(shape, false)
    }

    pub fn for_shape_legacy_11col(shape: StageAShape) -> Result<Self, String> {
        Self::for_shape_inner(shape, true)
    }

    fn for_shape_inner(shape: StageAShape, legacy_11col: bool) -> Result<Self, String> {
        shape.validate()?;
        if !shape.batch.is_power_of_two() {
            return Err("Stage A GKR lane batch size must be a power of two".into());
        }
        let nl_rows_per_token = nonlinear_domain_len(shape)?;
        let nl_total_rows = shape
            .batch
            .checked_mul(nl_rows_per_token)
            .ok_or_else(|| "Stage A GKR lane nonlinear row count overflow".to_string())?;
        let d_pad = shape.dimension.next_power_of_two();
        let e_pad = shape.experts.next_power_of_two();
        let m_pad = shape.expert_width.next_power_of_two();
        let sh_pad = shape.shared_width().next_power_of_two();
        let out_rows = shape
            .batch
            .checked_mul(d_pad)
            .ok_or_else(|| "Stage A GKR lane output row count overflow".to_string())?;
        let rt_rows = shape
            .batch
            .checked_mul(e_pad)
            .ok_or_else(|| "Stage A GKR lane routing row count overflow".to_string())?;
        let min_wit = if legacy_11col {
            (8 * nl_total_rows)
                .max(8 * out_rows)
                .max(8 * rt_rows)
                .max(8 * 4 * TABLE_ROWS)
                .next_power_of_two()
        } else {
            nl_total_rows
                .max(out_rows)
                .max(8 * rt_rows)
                .max(8 * 4 * TABLE_ROWS)
                .next_power_of_two()
        };
        let n_wit = min_wit.ilog2() as usize;
        Ok(Self {
            n_wit,
            wit_len: min_wit,
            sub_len: min_wit / 8,
            nl_rows_per_token,
            nl_total_rows,
            out_rows,
            rt_rows,
            d_pad,
            e_pad,
            m_pad,
            sh_pad,
            legacy_11col,
        })
    }

    #[inline]
    pub const fn out_subcube_offset(&self, limb_idx: usize) -> usize {
        if self.legacy_11col {
            limb_idx * self.sub_len
        } else {
            0
        }
    }

    #[inline]
    pub const fn rt_subcube_offset(&self, sub_idx: usize) -> usize {
        if self.legacy_11col {
            6 * self.sub_len + sub_idx * self.rt_rows
        } else {
            sub_idx * self.rt_rows
        }
    }

    #[inline]
    pub const fn tbl_subcube_offset(&self, table_idx: usize) -> usize {
        7 * self.sub_len + table_idx * TABLE_ROWS
    }
}

pub const fn out_limb_widths(shift: u32) -> [usize; 3] {
    let total = (shift as usize) + 1;
    let w0 = if total / 3 > 11 {
        11
    } else if total / 3 < 1 {
        1
    } else {
        total / 3
    };
    let rem = total - w0;
    let w1 = if rem / 2 > 11 {
        11
    } else if rem / 2 < 1 {
        1
    } else {
        rem / 2
    };
    let w2 = rem - w1;
    [w0, w1, w2]
}

#[derive(Clone)]
pub(crate) struct GkrLanePublicTables {
    pub(crate) silu_val: Vec<BaseField>,
    pub(crate) width_tag: Vec<BaseField>,
    pub(crate) width_val: Vec<BaseField>,
    pub(crate) rem_shift: Vec<BaseField>,
    pub(crate) rem_val: Vec<BaseField>,
    pub(crate) rem_par: Vec<BaseField>,
    pub(crate) exp_code: Vec<BaseField>,
    pub(crate) exp_val: Vec<BaseField>,
}

impl GkrLanePublicTables {
    fn new(policy: StageANumericPolicy) -> Result<Self, String> {
        let pub_tables = public_tables_for_policy(policy)?;
        let mut silu_val = Vec::with_capacity(TABLE_ROWS);
        for row in 0..TABLE_ROWS {
            let q = (row as i64) - 2048;
            let s = silu_q8_for_policy(q, policy)?;
            let su = (i32::from(s) + 2048) as u32;
            silu_val.push(BaseField::from_u32_unchecked(su));
        }

        let mut exp_code = Vec::with_capacity(TABLE_ROWS);
        let mut exp_val = Vec::with_capacity(TABLE_ROWS);
        for row in 0..TABLE_ROWS {
            if row <= 2048 {
                exp_code.push(BaseField::from_u32_unchecked(row as u32));
                exp_val.push(BaseField::from_u32_unchecked(u32::from(
                    pub_tables.exp_q15[2048 - row],
                )));
            } else {
                exp_code.push(BaseField::zero());
                exp_val.push(BaseField::from_u32_unchecked(u32::from(
                    pub_tables.exp_q15[2048],
                )));
            }
        }

        let mut width_tag = vec![BaseField::zero(); TABLE_ROWS];
        let mut width_val = vec![BaseField::zero(); TABLE_ROWS];
        for w in 1..=11usize {
            let base = (1usize << w) - 1;
            for v in 0..(1usize << w) {
                width_tag[base + v] = BaseField::from_u32_unchecked(w as u32);
                width_val[base + v] = BaseField::from_u32_unchecked(v as u32);
            }
        }

        let mut distinct_shifts = vec![
            policy.gate.shift,
            policy.up.shift,
            policy.product.shift,
            policy.router.shift,
        ];
        distinct_shifts.sort_unstable();
        distinct_shifts.dedup();

        let mut rem_shift = Vec::with_capacity(TABLE_ROWS);
        let mut rem_val = Vec::with_capacity(TABLE_ROWS);
        let mut rem_par = Vec::with_capacity(TABLE_ROWS);
        for &s in &distinct_shifts {
            let max_rem = 1u32 << s;
            match policy.rounding {
                StageARoundingMode::HalfTowardPositiveInfinity => {
                    for par in 0..=1u32 {
                        for r in 0..max_rem {
                            rem_shift.push(BaseField::from_u32_unchecked(s));
                            rem_val.push(BaseField::from_u32_unchecked(r));
                            rem_par.push(BaseField::from_u32_unchecked(par));
                        }
                    }
                }
                StageARoundingMode::TiesToEven => {
                    for r in 0..=max_rem {
                        rem_shift.push(BaseField::from_u32_unchecked(s));
                        rem_val.push(BaseField::from_u32_unchecked(r));
                        rem_par.push(BaseField::zero());
                    }
                    for r in 1..max_rem {
                        rem_shift.push(BaseField::from_u32_unchecked(s));
                        rem_val.push(BaseField::from_u32_unchecked(r));
                        rem_par.push(BaseField::one());
                    }
                }
            }
        }
        if rem_shift.len() > TABLE_ROWS {
            return Err("Stage A GKR remainder table exceeds 4096 rows".into());
        }
        let pad_s = rem_shift[0];
        let pad_r = rem_val[0];
        let pad_p = rem_par[0];
        while rem_shift.len() < TABLE_ROWS {
            rem_shift.push(pad_s);
            rem_val.push(pad_r);
            rem_par.push(pad_p);
        }

        Ok(Self {
            silu_val,
            width_tag,
            width_val,
            rem_shift,
            rem_val,
            rem_par,
            exp_code,
            exp_val,
        })
    }

    pub(crate) fn cached(
        policy: StageANumericPolicy,
    ) -> Result<std::borrow::Cow<'static, Self>, String> {
        static V1_TABLES: OnceLock<GkrLanePublicTables> = OnceLock::new();
        static DP_TABLES: OnceLock<GkrLanePublicTables> = OnceLock::new();
        static I12_Q7_TABLES: OnceLock<GkrLanePublicTables> = OnceLock::new();
        static I12_Q7_HU_TABLES: OnceLock<GkrLanePublicTables> = OnceLock::new();

        if policy == StageANumericPolicy::v1() {
            return Ok(std::borrow::Cow::Borrowed(
                V1_TABLES.get_or_init(|| Self::new(policy).expect("valid v1 GKR tables")),
            ));
        }
        if policy == StageANumericPolicy::deep_prove_comparison_v1() {
            return Ok(std::borrow::Cow::Borrowed(
                DP_TABLES.get_or_init(|| Self::new(policy).expect("valid deep_prove GKR tables")),
            ));
        }
        if policy == StageANumericPolicy::trained_i12_q7_v1() {
            return Ok(std::borrow::Cow::Borrowed(
                I12_Q7_TABLES
                    .get_or_init(|| Self::new(policy).expect("valid trained_i12_q7 GKR tables")),
            ));
        }
        if policy == StageANumericPolicy::trained_i12_q7_half_up_v1() {
            return Ok(std::borrow::Cow::Borrowed(
                I12_Q7_HU_TABLES.get_or_init(|| {
                    Self::new(policy).expect("valid trained_i12_q7_half_up GKR tables")
                }),
            ));
        }
        Ok(std::borrow::Cow::Owned(Self::new(policy)?))
    }
}

#[derive(Clone, Debug)]
struct GkrLaneLookupChallenges {
    silu_alphas: [SecureField; 3],
    silu_z: SecureField,
    range_alphas: [SecureField; 2],
    range_z: SecureField,
    width_alphas: [SecureField; 2],
    width_z: SecureField,
    rem_alphas: [SecureField; 3],
    rem_z: SecureField,
}

impl GkrLaneLookupChallenges {
    fn draw(channel: &mut Production100Channel) -> Self {
        let s = channel.draw_secure_felts(4);
        let r = channel.draw_secure_felts(3);
        let w = channel.draw_secure_felts(3);
        let m = channel.draw_secure_felts(4);
        Self {
            silu_alphas: [s[0], s[1], s[2]],
            silu_z: s[3],
            range_alphas: [r[0], r[1]],
            range_z: r[2],
            width_alphas: [w[0], w[1]],
            width_z: w[2],
            rem_alphas: [m[0], m[1], m[2]],
            rem_z: m[3],
        }
    }
}

pub fn witness_cprs_params(n_wit: usize, profile: &SecurityProfile) -> Result<Params, String> {
    witness_cprs_params_for_cols(n_wit, profile, WITNESS_GROUP_COLS)
}

pub fn witness_cprs_params_for_cols(
    n_wit: usize,
    profile: &SecurityProfile,
    cols: usize,
) -> Result<Params, String> {
    witness_cprs_params_for_cols_and_f(n_wit, profile, cols, None)
}

pub fn witness_cprs_params_for_cols_and_f(
    n_wit: usize,
    profile: &SecurityProfile,
    cols: usize,
    f_override: Option<usize>,
) -> Result<Params, String> {
    witness_cprs_params_for_mode_cols_and_f(
        n_wit,
        profile,
        STAGE_A_RECEIPT_MODE_GKR_LANE,
        cols,
        f_override,
    )
}

pub fn witness_cprs_params_for_mode_cols_and_f(
    n_wit: usize,
    profile: &SecurityProfile,
    mode_tag: u32,
    cols: usize,
    f_override: Option<usize>,
) -> Result<Params, String> {
    let m = n_wit
        .checked_sub(2)
        .ok_or_else(|| "CPRS witness n_wit must be >= 3".to_string())?;
    let (s, f_def, q, pow_bits) = match profile {
        SecurityProfile::Production100 => cprs_profile_parameters(profile, m),
        SecurityProfile::Dev13 => cprs_profile_parameters(profile, m),
        _ => (2, 4usize.min(m).max(1), 32, 0),
    };
    let f = f_override.map(|fo| fo.min(m).max(1)).unwrap_or(f_def);
    if mode_tag == STAGE_A_RECEIPT_MODE_ZK_GKR_LANE {
        Params::try_new(n_wit, s, f, q, pow_bits, WITNESS_GROUP_COLS_ZK)
            .map(|p| p.with_lo_vars(0))
            .map_err(|e| e.to_string())
    } else if mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        Params::try_new(n_wit, s, f, q, pow_bits, WITNESS_GROUP_COLS_ZK_FULL)
            .map(|p| p.with_lo_vars(0))
            .map_err(|e| e.to_string())
    } else {
        Params::try_new(n_wit, s, f, q, pow_bits, cols).map_err(|e| e.to_string())
    }
}

pub fn cprs_witness_outer_digest(
    binding_seed: &[u8; 32],
    statement: &StageANonlinearBatchStatement,
    witness_commitment: &[u8; 32],
    r_wit: &[SecureField],
) -> [u8; 32] {
    cprs_witness_outer_digest_for_roles(
        binding_seed,
        statement,
        witness_commitment,
        r_wit,
        &WITNESS_SLOT_ROLES,
    )
}

pub fn cprs_witness_outer_digest_for_mode_and_legacy(
    binding_seed: &[u8; 32],
    statement: &StageANonlinearBatchStatement,
    witness_commitment: &[u8; 32],
    r_wit: &[SecureField],
    r_cfs_extra: Option<&[SecureField]>,
    legacy_11col: bool,
) -> [u8; 32] {
    cprs_witness_outer_digest_for_roles_and_extra(
        binding_seed,
        statement,
        witness_commitment,
        r_wit,
        if r_cfs_extra.is_some() {
            &WITNESS_SLOT_ROLES_ZK
        } else if legacy_11col {
            &WITNESS_SLOT_ROLES_LEGACY
        } else {
            &WITNESS_SLOT_ROLES
        },
        r_cfs_extra,
    )
}

pub fn cprs_witness_outer_digest_for_roles(
    binding_seed: &[u8; 32],
    statement: &StageANonlinearBatchStatement,
    witness_commitment: &[u8; 32],
    r_wit: &[SecureField],
    slot_roles: &[u8],
) -> [u8; 32] {
    cprs_witness_outer_digest_for_roles_and_extra(
        binding_seed,
        statement,
        witness_commitment,
        r_wit,
        slot_roles,
        None,
    )
}

pub fn cprs_witness_outer_digest_for_roles_and_extra(
    binding_seed: &[u8; 32],
    statement: &StageANonlinearBatchStatement,
    witness_commitment: &[u8; 32],
    r_wit: &[SecureField],
    slot_roles: &[u8],
    r_cfs_extra: Option<&[SecureField]>,
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(GKR_LANE_WITNESS_OUTER_DOMAIN);
    h.update(binding_seed);
    h.update(statement.model_digest);
    h.update(statement.policy_digest);
    h.update(statement.table_digest);
    h.update(statement.input_digest);
    h.update(witness_commitment);
    h.update(slot_roles);
    h.update((r_wit.len() as u64).to_le_bytes());
    for &c in r_wit {
        for m in c.to_m31_array() {
            h.update(m.0.to_le_bytes());
        }
    }
    if let Some(r_cfs) = r_cfs_extra {
        h.update((r_cfs.len() as u64).to_le_bytes());
        for &c in r_cfs {
            for m in c.to_m31_array() {
                h.update(m.0.to_le_bytes());
            }
        }
    }
    h.finalize().into()
}

pub fn cprs_gkr_lane_proofs_transcript_digest(
    router_proof: &[u8],
    regular_proof: &[u8],
    shared_proof: &[u8],
    witness_proof: &[u8],
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(GKR_LANE_PROOFS_DIGEST_DOMAIN);
    for p in [router_proof, regular_proof, shared_proof, witness_proof] {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    h.finalize().into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GkrInstanceKind {
    NonlinearChunk { row_chunk: usize, q_chunk: usize },
    OutputChunk { q_chunk: usize },
    RoutingQueries,
    SoftmaxQueriesA,
    SoftmaxQueriesB,
    IoRangeQueries,
    Table(usize),
    MaskingInstance,
}

#[derive(Clone, Debug)]
pub(crate) struct GkrPlannedInstance {
    pub(crate) kind: GkrInstanceKind,
    pub(crate) log_slots: u32,
    pub(crate) log_rows: u32,
}

impl GkrPlannedInstance {
    #[inline]
    pub(crate) const fn n_variables(&self) -> usize {
        (self.log_slots + self.log_rows) as usize
    }
}

fn gkr_lane_plan_for_mode(geom: &GkrLaneGeometry, mode_tag: u32) -> Vec<GkrPlannedInstance> {
    let is_zk = mode_tag == STAGE_A_RECEIPT_MODE_ZK_GKR_LANE;
    let ext_bit = u32::from(is_zk);
    let n_chunks = 1usize << LOG_NL_ROW_CHUNKS;
    let log_nl_chunk_rows = (geom.n_wit as u32) - LOG_NL_ROW_CHUNKS + ext_bit;
    let log_out_rows = geom.out_rows.ilog2() + ext_bit;
    let log_rt_rows = geom.rt_rows.ilog2() + ext_bit;
    let log_table_rows = LOG_TABLE_ROWS + ext_bit;
    let mut plan = Vec::with_capacity(2 * n_chunks + 2 + GKR_TABLES + usize::from(is_zk));
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
    for t in 0..GKR_TABLES {
        plan.push(GkrPlannedInstance {
            kind: GkrInstanceKind::Table(t),
            log_slots: 0,
            log_rows: log_table_rows,
        });
    }
    if is_zk {
        plan.push(GkrPlannedInstance {
            kind: GkrInstanceKind::MaskingInstance,
            log_slots: 0,
            log_rows: ZK_MASK_INSTANCE_LOG_ROWS,
        });
    }
    plan
}

fn zk_instance_slot0_mask_denominators(
    policy: StageANumericPolicy,
    tables: &GkrLanePublicTables,
    challenges: &GkrLaneLookupChallenges,
) -> [[SecureField; 2]; ZK_UNMASKED_GKR_INSTANCES] {
    let out_widths = out_limb_widths(policy.output.shift);
    let d_nl_q0 = -challenges.silu_z;
    let d_nl_q1 = challenges.rem_alphas[0] * BaseField::from_u32_unchecked(policy.up.shift)
        - challenges.rem_z;
    let d_out_q0 = challenges.width_alphas[0]
        * BaseField::from_u32_unchecked(out_widths[0] as u32)
        - challenges.width_z;
    let d_out_q1 = challenges.width_alphas[0]
        * BaseField::from_u32_unchecked(out_widths[1] as u32)
        - challenges.width_z;
    let d_rt = -challenges.range_z;

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
            _ => unreachable!(),
        }
    };

    array::from_fn(|k| {
        if k < 32 {
            let d = if k % 2 == 0 { d_nl_q0 } else { d_nl_q1 };
            [d, d]
        } else if k == 32 {
            [d_out_q0, d_out_q0]
        } else if k == 33 {
            [d_out_q1, d_out_q1]
        } else if k == 34 {
            [d_rt, d_rt]
        } else {
            let t_idx = k - 35;
            [tbl_den(t_idx, 0), tbl_den(t_idx, 1)]
        }
    })
}

#[inline]
fn slot_bits_felt(slot: usize, bits: usize) -> Vec<SecureField> {
    (0..bits)
        .map(|bit| {
            if (slot >> (bits - 1 - bit)) & 1 == 1 {
                SecureField::one()
            } else {
                SecureField::zero()
            }
        })
        .collect()
}

struct GkrLaneLookupBinding {
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
    tbl_mult_claims: [SecureField; GKR_TABLES],
}

#[allow(clippy::too_many_arguments)]
fn derive_gkr_lane_binding_for_mode(
    artifact: &GkrArtifact,
    geom: &GkrLaneGeometry,
    policy: StageANumericPolicy,
    tables: &GkrLanePublicTables,
    challenges: &GkrLaneLookupChallenges,
    channel: &mut Production100Channel,
    mode_tag: u32,
    gkr_mask_evals: Option<&[SecureField; ZK_UNMASKED_GKR_INSTANCES]>,
) -> Result<GkrLaneLookupBinding, String> {
    let plan = gkr_lane_plan_for_mode(geom, mode_tag);
    if artifact.claims_to_verify_by_instance.len() != plan.len() {
        return Err("Stage A GKR lane instance count mismatch".into());
    }
    let chunk_point = channel.draw_secure_felts(LOG_NL_ROW_CHUNKS as usize);
    let nl_qchunk_weights = channel.draw_secure_felts(2);
    let out_qchunk_weights = channel.draw_secure_felts(2);

    let n_layers = artifact.ood_point.len();
    let log_nl_chunk_rows = (geom.n_wit as u32 - LOG_NL_ROW_CHUNKS) as usize;
    let log_out_rows = geom.out_rows.ilog2() as usize;
    let log_rt_rows = geom.rt_rows.ilog2() as usize;
    let is_zk_mode = mode_tag == STAGE_A_RECEIPT_MODE_ZK_GKR_LANE;
    if is_zk_mode && gkr_mask_evals.is_none() {
        return Err("Stage A ZK GKR lane requires GKR mask evaluations".into());
    }

    let (tau_nl, tau_out, tau_rt, tau_tbl) = if is_zk_mode {
        (
            artifact.ood_point[n_layers - log_nl_chunk_rows - 1],
            artifact.ood_point[n_layers - log_out_rows - 1],
            artifact.ood_point[n_layers - log_rt_rows - 1],
            artifact.ood_point[n_layers - (LOG_TABLE_ROWS as usize) - 1],
        )
    } else {
        (
            SecureField::zero(),
            SecureField::zero(),
            SecureField::zero(),
            SecureField::zero(),
        )
    };
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
    ];

    let inst_prefix = |inst: &GkrPlannedInstance| -> &[SecureField] {
        let start = n_layers - inst.n_variables();
        &artifact.ood_point[start..start + inst.log_slots as usize]
    };
    let inst_mask_term = |idx: usize, inst: &GkrPlannedInstance| -> SecureField {
        let Some(evals) = gkr_mask_evals else {
            return SecureField::zero();
        };
        let start = n_layers - inst.n_variables();
        let slot_pt = &artifact.ood_point[start..start + inst.log_slots as usize];
        let slot0_eq: SecureField = slot_pt.iter().map(|&s| SecureField::one() - s).product();
        let tau_k = artifact.ood_point[start + inst.log_slots as usize];
        let log_base_rows = (inst.log_rows - 1) as usize;
        let row_hi_pt = &artifact.ood_point[n_layers - log_base_rows..n_layers - 1];
        let row0_hi_eq: SecureField = row_hi_pt.iter().map(|&r| SecureField::one() - r).product();
        slot0_eq * tau_k * row0_hi_eq * evals[idx]
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
    let mut tbl_mult_claims = [SecureField::zero(); GKR_TABLES];

    for (idx, inst) in plan.iter().enumerate() {
        let [num, den]: [SecureField; 2] = artifact.claims_to_verify_by_instance[idx]
            .as_slice()
            .try_into()
            .map_err(|_| "Stage A GKR instance claim shape".to_string())?;
        match inst.kind {
            GkrInstanceKind::NonlinearChunk { row_chunk, q_chunk } => {
                if num != expected_num_nl + inst_mask_term(idx, inst) {
                    return Err("Stage A GKR nonlinear query numerator is not one".into());
                }
                let cw = eq(&chunk_point, &slot_bits_felt(row_chunk, LOG_NL_ROW_CHUNKS as usize));
                let c_part = if q_chunk == 0 { nl0_const } else { nl1_const };
                gkr_nl_claim += cw * nl_qchunk_weights[q_chunk] * (den - c_part);
            }
            GkrInstanceKind::OutputChunk { q_chunk } => {
                if num != expected_num_out + inst_mask_term(idx, inst) {
                    return Err("Stage A GKR output query numerator is not one".into());
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
                    return Err("Stage A GKR routing query numerator is not one".into());
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
                                * (challenges.width_alphas[0] * BaseField::from_u32_unchecked(2)
                                    - challenges.width_z);
                        }
                        _ => unreachable!(),
                    }
                }
                gkr_rt_claim = den - c_part;
            }
            GkrInstanceKind::Table(t_idx) => {
                if den != expected_tbl_den[t_idx] {
                    return Err(format!(
                        "Stage A GKR table {t_idx} denominator claim does not match public table"
                    ));
                }
                tbl_mult_claims[t_idx] = -num + inst_mask_term(idx, inst);
            }
            GkrInstanceKind::MaskingInstance => {
                let evals = gkr_mask_evals
                    .ok_or_else(|| "Stage A ZK GKR masking instance missing evals".to_string())?;
                let r_mask =
                    &artifact.ood_point[n_layers - (ZK_MASK_INSTANCE_LOG_ROWS as usize)..];
                let mask_dens = zk_instance_slot0_mask_denominators(policy, tables, challenges);
                let mut den_vec = vec![-challenges.silu_z; ZK_MASK_INSTANCE_ROWS];
                for k in 0..ZK_UNMASKED_GKR_INSTANCES {
                    den_vec[2 * k] = mask_dens[k][0];
                    den_vec[2 * k + 1] = mask_dens[k][1];
                }
                let expected_mask_den = mle_at(&den_vec, r_mask);
                if den != expected_mask_den {
                    return Err(
                        "Stage A ZK GKR masking instance denominator claim mismatch".into(),
                    );
                }
                let eq9 = eq_evals(&r_mask[..(ZK_MASK_INSTANCE_LOG_ROWS as usize) - 1]);
                let mut expected_mask_num = SecureField::zero();
                for k in 0..ZK_UNMASKED_GKR_INSTANCES {
                    expected_mask_num -= eq9[k] * evals[k];
                }
                if num != expected_mask_num {
                    return Err(
                        "Stage A ZK GKR masking instance numerator claim mismatch".into(),
                    );
                }
            }
            GkrInstanceKind::SoftmaxQueriesA
            | GkrInstanceKind::SoftmaxQueriesB
            | GkrInstanceKind::IoRangeQueries => {
                return Err("unexpected Mode 11 GKR instance kind in Mode 9/10 plan".into());
            }
        }
    }

    Ok(GkrLaneLookupBinding {
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
        tbl_mult_claims,
    })
}

fn build_topk_diff_linear_form_with_eq(
    statement: &StageANonlinearBatchStatement,
    geom: &GkrLaneGeometry,
    rt_eq: &[SecureField],
) -> (Vec<SecureField>, SecureField) {
    let shape = statement.shape;
    let two = SecureField::from(BaseField::from_u32_unchecked(2));
    let mut coeffs = vec![SecureField::zero(); geom.rt_rows];
    let mut bias_eval = SecureField::zero();
    let mut slot_of = vec![None; geom.e_pad];

    for t in 0..shape.batch {
        let base = t * geom.e_pad;
        for &e in &statement.routes[t] {
            slot_of[e] = None;
        }
        for (s, &e) in statement.routes[t].iter().enumerate() {
            slot_of[e] = Some(s);
        }
        let e_last = statement.routes[t][shape.top_k - 1];
        for e in 0..shape.experts {
            let w = rt_eq[base + e];
            let two_w = two * w;
            match slot_of[e] {
                Some(s) => {
                    if s + 1 < shape.top_k {
                        let e_next = statement.routes[t][s + 1];
                        coeffs[base + e] += two_w;
                        coeffs[base + e_next] -= two_w;
                        if e > e_next {
                            bias_eval += w;
                        }
                    }
                }
                None => {
                    coeffs[base + e_last] += two_w;
                    coeffs[base + e] -= two_w;
                    if e_last > e {
                        bias_eval += w;
                    }
                }
            }
        }
        for &e in &statement.routes[t] {
            slot_of[e] = None;
        }
    }
    (coeffs, bias_eval)
}

#[derive(Clone, Debug)]
pub struct StageAZkLaneAux {
    pub gkr_mask_evals: [SecureField; ZK_UNMASKED_GKR_INSTANCES],
    pub gkr_libra_sums: Vec<SecureField>,
    pub gkr_libra_evals: Vec<SecureField>,
    pub r4_cfs_sum: SecureField,
    pub r4_cfs_eval: SecureField,
    pub wit_cfs_sum: SecureField,
    pub wit_cfs_eval: SecureField,
    pub cfs_pad_claim: SecureField,
    pub cfs_binding_sumcheck: SumcheckProof,
    pub blind_col_opening: SecureField,
}

#[inline]
pub(crate) fn cfs_deg3_accumulate_weights(
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

#[allow(clippy::too_many_arguments)]
fn build_cfs_binding_weights(
    gkr_layer_ood_points: &[Vec<SecureField>],
    r_rt: &[SecureField],
    r_wit_full: &[SecureField],
    r_cfs_pad: &[SecureField],
    gamma_libra_sums: &[SecureField],
    gamma_libra_evals: &[SecureField],
    gamma_r4_cfs: &[SecureField],
    gamma_wit_cfs: &[SecureField],
    gamma_cfs_pad: SecureField,
) -> Vec<SecureField> {
    let mut w_cfs = vec![SecureField::zero(); ZK_CFS_TABLE_ROWS];
    let mut gkr_off = 0usize;
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
    cfs_deg3_accumulate_weights(&mut w_cfs, 1536, r_rt, gamma_r4_cfs[0], gamma_r4_cfs[1]);
    cfs_deg3_accumulate_weights(
        &mut w_cfs,
        1664,
        r_wit_full,
        gamma_wit_cfs[0],
        gamma_wit_cfs[1],
    );
    let pad_eq = eq_evals(r_cfs_pad);
    for (v, &e) in pad_eq.iter().enumerate() {
        w_cfs[2048 + v] += gamma_cfs_pad * e;
    }
    w_cfs
}

#[derive(Clone, Debug)]
pub struct StageAGkrLaneBatchProof {
    pub witness_commitment: [u8; 32],
    pub router: StageARegisteredRouterProof,
    pub regular_gate_up_claims: Vec<SecureField>,
    pub shared_gate_up_claims: Vec<SecureField>,
    pub regular_gate_up_sumcheck: SumcheckProof,
    pub regular_gate_weight_openings: Vec<SecureField>,
    pub regular_up_weight_openings: Vec<SecureField>,
    pub shared_gate_up_sumcheck: SumcheckProof,
    pub shared_gate_weight_opening: SecureField,
    pub shared_up_weight_opening: SecureField,
    pub regular_down_claim: SecureField,
    pub shared_down_claim: SecureField,
    pub regular_grouped_claim: SecureField,
    pub shared_grouped_claim: SecureField,
    pub regular_down_weight_openings: Vec<SecureField>,
    pub regular_weight_opening: SecureField,
    pub shared_weight_opening: SecureField,
    pub regular_down_sumcheck: SumcheckProof,
    pub shared_down_sumcheck: SumcheckProof,
    pub gkr_proof: GkrBatchProof,
    pub r4_routing_sumcheck: SumcheckProof,
    pub r_witness_sumcheck: SumcheckProof,
    pub witness_openings: [SecureField; WITNESS_GROUP_COLS],
    pub legacy_11col: bool,
    pub zk_aux: Option<StageAZkLaneAux>,
    pub zk_full_aux: Option<crate::stage_a_zk_full::StageAZkFullAux>,
    pub cprs_router_proof: Vec<u8>,
    pub cprs_regular_proof: Vec<u8>,
    pub cprs_shared_proof: Vec<u8>,
    pub cprs_witness_proof: Vec<u8>,
}

fn mix_gkr_lane_statement_with_mode(
    channel: &mut Production100Channel,
    statement: &StageANonlinearBatchStatement,
    context: &StageARegisteredContext,
    profile: &SecurityProfile,
    chain: &StageAChainLink,
    witness_commitment: &[u8; 32],
    mode_tag: u32,
) {
    channel.mix_u64(GKR_LANE_MODE_DOMAIN_TAG);
    channel.mix_u64(u64::from(mode_tag));
    let profile_id = profile.id().as_bytes();
    channel.mix_u64(profile_id.len() as u64);
    for chunk in profile_id.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        channel.mix_u64(u64::from_le_bytes(word));
    }
    channel.mix_u64(u64::from(chain.layer));
    mix_digest(channel, &chain.prev_receipt_digest);
    for value in [
        statement.shape.batch,
        statement.shape.dimension,
        statement.shape.expert_width,
        statement.shape.experts,
        statement.shape.top_k,
        statement.shape.shared_factor,
    ] {
        channel.mix_u64(value as u64);
    }
    mix_digest(channel, &statement.policy_digest);
    mix_digest(channel, &statement.model_digest);
    mix_digest(channel, &statement.model_registry_manifest_digest);
    mix_digest(channel, &statement.table_digest);
    mix_digest(channel, &statement.input_digest);
    mix_digest(channel, &context.scheme_digest);
    mix_digest(channel, &context.router_commitment);
    mix_digest(channel, &context.shared_commitment);
    mix_digest(channel, &context.registered_expert_root);
    for opening in &context.active_expert_openings {
        channel.mix_u64(opening.expert_id as u64);
        mix_digest(channel, &opening.group_commitment);
    }
    if !statement.routes.is_empty() {
        for token in 0..statement.shape.batch {
            channel.mix_u64(token as u64);
            for &expert in &statement.routes[token] {
                channel.mix_u64(expert as u64);
            }
            for &weight in &statement.route_weights_q15[token] {
                channel.mix_u64(u64::from(weight));
            }
        }
    }
    mix_digest(channel, &gkr_lane_output_digest(&statement.output_q8));
    mix_digest(channel, witness_commitment);
}

fn gkr_lane_output_digest(output_q8: &[Vec<i16>]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    hasher.update(b"stwo-ml:stage-a-gkr-lane:outputs:v1");
    hasher.update((output_q8.len() as u64).to_le_bytes());
    for row in output_q8 {
        hasher.update((row.len() as u64).to_le_bytes());
        hash_i16_le_slice(&mut hasher, row);
    }
    hasher.finalize().into()
}

fn verify_gkr_router_sumcheck(
    xs: &[Vec<i16>],
    experts: usize,
    proof: &StageARegisteredRouterProof,
    channel: &mut Production100Channel,
) -> Result<(Vec<SecureField>, Vec<SecureField>), String> {
    let tokens = xs.len();
    let dimension = xs.first().map_or(0, Vec::len);
    if !tokens.is_power_of_two() || !dimension.is_power_of_two() || !experts.is_power_of_two() {
        return Err("Stage A GKR router dimensions must be powers of two".into());
    }
    let log_tokens = tokens.ilog2() as usize;
    let log_dimension = dimension.ilog2() as usize;
    let log_experts = experts.ilog2() as usize;
    let r_tokens = channel.draw_secure_felts(log_tokens);
    let r_experts = channel.draw_secure_felts(log_experts);
    let mut score_point = r_tokens.clone();
    score_point.extend_from_slice(&r_experts);
    channel.mix_felts(&[proof.gate_claim]);
    let _lambda = channel.draw_secure_felt();
    let (s_dimension, final_eval) = partially_verify(proof.gate_claim, &proof.gate_proof, channel)?;
    if s_dimension.len() != log_dimension {
        return Err("Stage A GKR router sumcheck assignment length mismatch".into());
    }
    let r_tok_eq = eq_evals(&r_tokens);
    let s_eq_i64: Vec<[i64; 4]> = eq_evals(&s_dimension)
        .into_iter()
        .map(coeff_coords_i64)
        .collect();
    let mut x_opening = SecureField::zero();
    for (row, &tok_w) in xs.iter().zip(&r_tok_eq) {
        let mut acc = [0i64; 4];
        for (&v, c) in row.iter().zip(&s_eq_i64) {
            let v = i64::from(v);
            acc[0] += c[0] * v;
            acc[1] += c[1] * v;
            acc[2] += c[2] * v;
            acc[3] += c[3] * v;
        }
        x_opening += tok_w * acc_to_secure(acc);
    }
    if final_eval != x_opening * proof.wg_opening {
        return Err("Stage A GKR router matmul sumcheck reduction mismatch".into());
    }
    channel.mix_felts(&[proof.wg_opening]);
    let mut router_weight_point = s_dimension;
    router_weight_point.extend_from_slice(&r_experts);
    Ok((router_weight_point, score_point))
}

fn eval_public_routing_at_r_rt_with_eq(
    statement: &StageANonlinearBatchStatement,
    geom: &GkrLaneGeometry,
    rt_eq: &[SecureField],
) -> (SecureField, SecureField) {
    let mut act_eval = SecureField::zero();
    let mut rw_eval = SecureField::zero();
    for t in 0..statement.shape.batch {
        let base = t * geom.e_pad;
        for (s, &e) in statement.routes[t].iter().enumerate() {
            let w = rt_eq[base + e];
            act_eval += w;
            rw_eval += w * BaseField::from_u32_unchecked(u32::from(
                statement.route_weights_q15[t][s],
            ));
        }
    }
    (act_eval, rw_eval)
}

fn compute_r5_rhs(
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
    down_token_weights: &[SecureField],
    out_eq: &[SecureField],
    regular_down_claim: SecureField,
    shared_down_claim: SecureField,
) -> (SecureField, SecureField) {
    let shape = statement.shape;
    let m_out = SecureField::from(BaseField::from_u32_unchecked(policy.output.multiplier as u32));
    let two_15 = SecureField::from(BaseField::from_u32_unchecked(1u32 << 15));
    let two_s_out = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.output.shift));
    let half_s_out =
        SecureField::from(BaseField::from_u32_unchecked(1u32 << (policy.output.shift - 1)));
    let w_sum: SecureField = down_token_weights.iter().copied().sum();
    let out_eq_i64: Vec<[i64; 4]> = out_eq[..shape.dimension]
        .iter()
        .copied()
        .map(coeff_coords_i64)
        .collect();
    let mut q_out = SecureField::zero();
    let mut p_out = SecureField::zero();
    for t in 0..shape.batch {
        let tw = down_token_weights[t];
        let mut q_acc = [0i64; 4];
        let mut p_acc = [0i64; 4];
        for (&val_i16, c) in statement.output_q8[t].iter().zip(&out_eq_i64) {
            let val = i64::from(val_i16);
            q_acc[0] += c[0] * val;
            q_acc[1] += c[1] * val;
            q_acc[2] += c[2] * val;
            q_acc[3] += c[3] * val;
            let mask = -(val & 1);
            p_acc[0] += c[0] & mask;
            p_acc[1] += c[1] & mask;
            p_acc[2] += c[2] & mask;
            p_acc[3] += c[3] & mask;
        }
        q_out += tw * acc_to_secure(q_acc);
        p_out += tw * acc_to_secure(p_acc);
    }
    let acc_claim = regular_down_claim + two_15 * shared_down_claim;
    match policy.rounding {
        StageARoundingMode::TiesToEven => {
            let r5a = m_out * acc_claim - two_s_out * q_out + half_s_out * w_sum - p_out;
            let r5b = two_s_out * w_sum
                - SecureField::from(BaseField::from_u32_unchecked(2)) * p_out;
            (r5a, r5b)
        }
        StageARoundingMode::HalfTowardPositiveInfinity => {
            let r5a = m_out * acc_claim - two_s_out * q_out + half_s_out * w_sum;
            let r5b =
                SecureField::from(BaseField::from_u32_unchecked((1u32 << policy.output.shift) - 1))
                    * w_sum;
            (r5a, r5b)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_nl_public_weights_at_r_wit(
    statement: &StageANonlinearBatchStatement,
    grouped: &FusedGroupedContext,
    route_active_idx: &[usize],
    geom: &GkrLaneGeometry,
    r_wit: &[SecureField],
    gu_token_weights: &[SecureField],
    gu_expert_weights: &[SecureField],
    reg_proj_eq: &[SecureField],
    sh_proj_eq: &[SecureField],
    down_token_weights: &[SecureField],
    gamma_reg_down: SecureField,
    gamma_sh_down: SecureField,
    expert_point: &[SecureField],
    hid_eq: &[SecureField],
    sh_down_eq: &[SecureField],
) -> (SecureField, SecureField) {
    let shape = statement.shape;
    let log_nl_total = geom.nl_total_rows.ilog2() as usize;
    let log_batch = shape.batch.ilog2() as usize;
    let prefix_bits = geom.n_wit - log_nl_total;
    let pref_eq = eq(&r_wit[..prefix_bits], &slot_bits_felt(0, prefix_bits));
    let r_tok = &r_wit[prefix_bits..prefix_bits + log_batch];
    let r_within = &r_wit[prefix_bits + log_batch..];
    let tok_eq = eq_evals(r_tok);
    let within_eq = eq_evals(r_within);

    let mut s_gu = vec![SecureField::zero(); shape.top_k];
    let mut s_down = vec![SecureField::zero(); shape.top_k];
    for s in 0..shape.top_k {
        let base = s * geom.m_pad;
        s_gu[s] = dot(
            &within_eq[base..base + shape.expert_width],
            &reg_proj_eq[..shape.expert_width],
        );
        s_down[s] = dot(
            &within_eq[base..base + shape.expert_width],
            &hid_eq[..shape.expert_width],
        );
    }
    let sh_base = shape.top_k * geom.m_pad;
    let sh_width = shape.shared_width();
    let s_gu_sh = dot(
        &within_eq[sh_base..sh_base + sh_width],
        &sh_proj_eq[..sh_width],
    );
    let s_down_sh = dot(
        &within_eq[sh_base..sh_base + sh_width],
        &sh_down_eq[..sh_width],
    );

    let active_len = grouped.active_regular_experts.len();
    let exp_eq: Vec<SecureField> = (0..active_len)
        .map(|a| eq_index(expert_point, a))
        .collect();

    let mut w_raw_eval = SecureField::zero();
    let mut w_down_eval = SecureField::zero();
    for t in 0..shape.batch {
        let teq = tok_eq[t];
        let mut gu_tok_sum = gu_expert_weights[active_len] * s_gu_sh;
        let mut down_tok_sum = gamma_sh_down * s_down_sh;
        let route_base = t * shape.top_k;
        for s in 0..shape.top_k {
            let a = route_active_idx[route_base + s];
            gu_tok_sum += gu_expert_weights[a] * s_gu[s];
            let rw = BaseField::from_u32_unchecked(u32::from(statement.route_weights_q15[t][s]));
            down_tok_sum += gamma_reg_down * (exp_eq[a] * rw) * s_down[s];
        }
        w_raw_eval += teq * gu_token_weights[t] * gu_tok_sum;
        w_down_eval += teq * down_token_weights[t] * down_tok_sum;
    }
    (pref_eq * w_raw_eval, pref_eq * w_down_eval)
}

#[allow(clippy::too_many_arguments)]
fn eval_w_out_limbs_at_r_wit(
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
    geom: &GkrLaneGeometry,
    binding: &GkrLaneLookupBinding,
    r_wit: &[SecureField],
    tau_wit: SecureField,
    is_zk_mode: bool,
    down_token_weights: &[SecureField],
    output_point: &[SecureField],
    gamma_gkr_out: SecureField,
    gamma_r5a: SecureField,
    gamma_r5b: SecureField,
    out_limb_openings: &[SecureField],
) -> SecureField {
    let sel_wit = if is_zk_mode {
        SecureField::one() - tau_wit
    } else {
        SecureField::one()
    };
    let eq_tau_out = if is_zk_mode {
        eq(&[binding.tau_out], &[tau_wit])
    } else {
        SecureField::one()
    };
    let out_widths = out_limb_widths(policy.output.shift);
    let limb_shifts = [0u32, out_widths[0] as u32, (out_widths[0] + out_widths[1]) as u32];
    let log_out_rows = geom.out_rows.ilog2() as usize;
    let log_batch = statement.shape.batch.ilog2() as usize;
    let out_pref_bits = geom.n_wit - log_out_rows;
    let out_pref_eq = eq(&r_wit[..out_pref_bits], &slot_bits_felt(0, out_pref_bits));
    let r_out_sub = &r_wit[out_pref_bits..];
    let r_out_tok = &r_out_sub[..log_batch];
    let r_out_dim = &r_out_sub[log_batch..];
    let out_tok_eq = eq_evals(r_out_tok);
    let r5_tok_factor: SecureField = down_token_weights
        .iter()
        .zip(&out_tok_eq)
        .map(|(&w, &e)| w * e)
        .sum();
    let r5_dim_factor = eq(r_out_dim, output_point);
    let r5_sub_eval = r5_tok_factor * r5_dim_factor;
    let gkr_out_sub_eq = eq(r_out_sub, &binding.r_gkr_out);
    let gkr_eval = out_pref_eq * gkr_out_sub_eq;
    let r5_eval = out_pref_eq * r5_sub_eval;

    let mut total = SecureField::zero();
    for k in 0..6 {
        let shift_scale =
            SecureField::from(BaseField::from_u32_unchecked(1u32 << limb_shifts[k % 3]));
        let r5_coeff = if k < 3 {
            (gamma_r5a + gamma_r5b) * shift_scale
        } else {
            gamma_r5b * shift_scale
        };
        let w_k = eq_tau_out * gamma_gkr_out * binding.out_limb_weights[k] * gkr_eval
            + sel_wit * r5_coeff * r5_eval;
        total += w_k * out_limb_openings[k];
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn eval_w_c10_at_r_wit(
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
    geom: &GkrLaneGeometry,
    binding: &GkrLaneLookupBinding,
    r_wit: &[SecureField],
    tau_wit: SecureField,
    is_zk_mode: bool,
    down_token_weights: &[SecureField],
    output_point: &[SecureField],
    r_rt_init: &[SecureField],
    r_rt: &[SecureField],
    topk_coeffs: &[SecureField],
    gamma_gkr_out: SecureField,
    gamma_gkr_rt: SecureField,
    gamma_gkr_tbl: &[SecureField],
    gamma_rt_rel1: SecureField,
    gamma_r4_act: SecureField,
    gamma_r4_rw: SecureField,
    gamma_topk: SecureField,
    gamma_r5a: SecureField,
    gamma_r5b: SecureField,
    r_gkr_last: SecureField,
    gamma_gkr_mask: &[SecureField],
) -> SecureField {
    let mut w_c10_eval = SecureField::zero();
    let sel_wit = if is_zk_mode {
        SecureField::one() - tau_wit
    } else {
        SecureField::one()
    };
    let eq_tau_rt = if is_zk_mode {
        eq(&[binding.tau_rt], &[tau_wit])
    } else {
        SecureField::one()
    };
    let sel_tbl = if is_zk_mode {
        sel_wit * (SecureField::one() - binding.tau_tbl)
    } else {
        SecureField::one()
    };

    if geom.legacy_11col {
        let out_widths = out_limb_widths(policy.output.shift);
        let limb_shifts = [0u32, out_widths[0] as u32, (out_widths[0] + out_widths[1]) as u32];
        let log_out_rows = geom.out_rows.ilog2() as usize;
        let log_batch = statement.shape.batch.ilog2() as usize;
        let out_pref_bits = geom.n_wit - log_out_rows;
        let r_out_sub = &r_wit[out_pref_bits..];
        let r_out_tok = &r_out_sub[..log_batch];
        let r_out_dim = &r_out_sub[log_batch..];
        let out_tok_eq = eq_evals(r_out_tok);
        let r5_tok_factor: SecureField = down_token_weights
            .iter()
            .zip(&out_tok_eq)
            .map(|(&w, &e)| w * e)
            .sum();
        let r5_dim_factor = eq(r_out_dim, output_point);
        let r5_sub_eval = r5_tok_factor * r5_dim_factor;
        let gkr_out_sub_eq = eq(r_out_sub, &binding.r_gkr_out);
        for k in 0..6 {
            let offset = geom.out_subcube_offset(k);
            let pref_idx = offset >> log_out_rows;
            let pref_eq = eq(&r_wit[..out_pref_bits], &slot_bits_felt(pref_idx, out_pref_bits));
            let gkr_eval = pref_eq * gkr_out_sub_eq;
            let r5_eval = pref_eq * r5_sub_eval;
            let shift_scale =
                SecureField::from(BaseField::from_u32_unchecked(1u32 << limb_shifts[k % 3]));
            let r5_coeff = if k < 3 {
                (gamma_r5a + gamma_r5b) * shift_scale
            } else {
                gamma_r5b * shift_scale
            };
            w_c10_eval += gamma_gkr_out * binding.out_limb_weights[k] * gkr_eval
                + sel_wit * r5_coeff * r5_eval;
        }
    }

    let log_rt_rows = geom.rt_rows.ilog2() as usize;
    let rt_pref_bits = geom.n_wit - log_rt_rows;
    let gkr_rt_sub_eq = eq(&r_wit[rt_pref_bits..], &binding.r_gkr_rt);
    for u in 0..7 {
        if !binding.rt_sub_weights[u].is_zero() {
            let offset = geom.rt_subcube_offset(u);
            let pref_idx = offset >> log_rt_rows;
            let pref_eq = eq(&r_wit[..rt_pref_bits], &slot_bits_felt(pref_idx, rt_pref_bits));
            w_c10_eval +=
                eq_tau_rt * gamma_gkr_rt * binding.rt_sub_weights[u] * (pref_eq * gkr_rt_sub_eq);
        }
    }

    let tbl_sub_bits = binding.r_gkr_tbl.len();
    let tbl_pref_bits = geom.n_wit - tbl_sub_bits;
    let gkr_tbl_sub_eq = eq(&r_wit[tbl_pref_bits..], &binding.r_gkr_tbl);
    for v in 0..GKR_TABLES {
        let offset = geom.tbl_subcube_offset(v);
        let pref_idx = offset >> tbl_sub_bits;
        let pref_eq = eq(&r_wit[..tbl_pref_bits], &slot_bits_felt(pref_idx, tbl_pref_bits));
        w_c10_eval += sel_tbl * gamma_gkr_tbl[v] * (pref_eq * gkr_tbl_sub_eq);
    }

    if is_zk_mode {
        let mask_sub_bits = 9usize;
        let mask_pref_bits = geom.n_wit - mask_sub_bits;
        let mask_pref_idx = geom.tbl_subcube_offset(0) >> mask_sub_bits;
        let mask_pref_eq =
            tau_wit * eq(&r_wit[..mask_pref_bits], &slot_bits_felt(mask_pref_idx, mask_pref_bits));
        let mask_sub_eq = eq_evals(&r_wit[mask_pref_bits..]);
        let beta = [
            SecureField::from_u32_unchecked(1, 0, 0, 0),
            SecureField::from_u32_unchecked(0, 1, 0, 0),
            SecureField::from_u32_unchecked(0, 0, 1, 0),
            SecureField::from_u32_unchecked(0, 0, 0, 1),
        ];
        let one_minus_r_last = SecureField::one() - r_gkr_last;
        let mut mask_eval = SecureField::zero();
        for (k, &gamma_m) in gamma_gkr_mask.iter().enumerate() {
            let g0 = gamma_m * one_minus_r_last;
            let g1 = gamma_m * r_gkr_last;
            for c in 0..4 {
                mask_eval +=
                    (g0 * mask_sub_eq[8 * k + c] + g1 * mask_sub_eq[8 * k + 4 + c]) * beta[c];
            }
        }
        w_c10_eval += mask_pref_eq * mask_eval;
    }

    let mut r4_eval = SecureField::zero();
    let two_s_rt = SecureField::from(BaseField::from_u32_unchecked(1u32 << policy.router.shift));
    let rt_init_sub_eq = eq(&r_wit[rt_pref_bits..], r_rt_init);
    let rt_r4_sub_eq = eq(&r_wit[rt_pref_bits..], r_rt);
    let rt_pref_eq = |u: usize| -> SecureField {
        let pref_idx = geom.rt_subcube_offset(u) >> log_rt_rows;
        eq(&r_wit[..rt_pref_bits], &slot_bits_felt(pref_idx, rt_pref_bits))
    };
    let rt0_pref_eq = rt_pref_eq(0);
    r4_eval += gamma_rt_rel1 * two_s_rt * (rt0_pref_eq * rt_init_sub_eq);
    r4_eval += gamma_rt_rel1 * (rt_pref_eq(1) * rt_init_sub_eq);

    r4_eval += gamma_r4_act * (rt_pref_eq(3) * rt_r4_sub_eq);
    r4_eval += gamma_r4_rw * (rt_pref_eq(4) * rt_r4_sub_eq);

    let rt_sub_eq = eq_evals(&r_wit[rt_pref_bits..]);
    let topk_c0_eval = dot(topk_coeffs, &rt_sub_eq);
    r4_eval -= gamma_topk * rt0_pref_eq * topk_c0_eval;
    r4_eval += gamma_topk * (rt_pref_eq(5) * rt_r4_sub_eq);
    r4_eval += gamma_topk
        * SecureField::from(BaseField::from_u32_unchecked(2048))
        * (rt_pref_eq(6) * rt_r4_sub_eq);
    w_c10_eval += sel_wit * r4_eval;

    w_c10_eval
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_against_anchor(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
) -> Result<StageARegisteredVerifierRun, String> {
    verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        None,
        STAGE_A_RECEIPT_MODE_GKR_LANE,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
) -> Result<StageARegisteredVerifierRun, String> {
    verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        stage_b,
        STAGE_A_RECEIPT_MODE_GKR_LANE,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_against_anchor_mode(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    mode_tag: u32,
) -> Result<StageARegisteredVerifierRun, String> {
    verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        None,
        mode_tag,
    )
}

#[derive(Clone, Debug)]
pub struct StageAGkrLaneDeferredCprs {
    pub binding_seed: [u8; 32],
    pub active_experts: Vec<usize>,
    pub witness_commitment: [u8; 32],
    pub witness_params: Params,
    pub wit_points: Vec<Vec<crate::cprs::QM31>>,
    pub wit_claims: Vec<crate::cprs::QM31>,
    pub d_wit: [u8; 32],
    pub router_commitment: [u8; 32],
    pub router_params: Params,
    pub r_router_cprs: Vec<crate::cprs::QM31>,
    pub wg_opening: crate::cprs::QM31,
    pub r_reg_gu_cprs: Vec<crate::cprs::QM31>,
    pub r_reg_down_cprs: Vec<crate::cprs::QM31>,
    pub reg_commitments: Vec<[u8; 32]>,
    pub reg_params: Params,
    pub reg_points: Vec<Vec<crate::cprs::QM31>>,
    pub reg_claims: Vec<crate::cprs::QM31>,
    pub shared_commitment: [u8; 32],
    pub shared_params: Params,
    pub sh_points: Vec<Vec<crate::cprs::QM31>>,
    pub sh_claims: [crate::cprs::QM31; 3],
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
    mode_tag: u32,
) -> Result<StageARegisteredVerifierRun, String> {
    let (run, _) = verify_stage_a_gkr_lane_core(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        stage_b,
        mode_tag,
        false,
        None,
        None,
    )?;
    Ok(run)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_defer_cprs(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    mode_tag: u32,
) -> Result<(StageARegisteredVerifierRun, StageAGkrLaneDeferredCprs), String> {
    verify_stage_a_gkr_lane_batch_defer_cprs_with_f(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        mode_tag,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_defer_cprs_with_f(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    mode_tag: u32,
    witness_f_override: Option<usize>,
) -> Result<(StageARegisteredVerifierRun, StageAGkrLaneDeferredCprs), String> {
    verify_stage_a_gkr_lane_batch_defer_cprs_with_f_and_cont(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        mode_tag,
        witness_f_override,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_gkr_lane_batch_defer_cprs_with_f_and_cont(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    mode_tag: u32,
    witness_f_override: Option<usize>,
    r_cont: Option<&[SecureField]>,
) -> Result<(StageARegisteredVerifierRun, StageAGkrLaneDeferredCprs), String> {
    let (run, deferred_opt) = verify_stage_a_gkr_lane_core(
        anchor,
        context,
        inputs,
        policy,
        profile,
        statement,
        proof,
        chain,
        None,
        mode_tag,
        true,
        witness_f_override,
        r_cont,
    )?;
    Ok((
        run,
        deferred_opt.expect("defer_cprs=true returns deferred CPRS verifier claims"),
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_stage_a_gkr_lane_core(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAGkrLaneBatchProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
    mode_tag: u32,
    defer_cprs: bool,
    witness_f_override: Option<usize>,
    r_cont: Option<&[SecureField]>,
) -> Result<(StageARegisteredVerifierRun, Option<StageAGkrLaneDeferredCprs>), String> {
    if mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        return zk_full::verify_stage_a_zk_full_core(
            anchor,
            context,
            inputs,
            policy,
            profile,
            statement,
            proof,
            chain,
            stage_b,
            defer_cprs,
            witness_f_override,
            r_cont,
        );
    }
    if mode_tag != STAGE_A_RECEIPT_MODE_GKR_LANE && mode_tag != STAGE_A_RECEIPT_MODE_ZK_GKR_LANE {
        return Err(format!("unsupported Stage A GKR lane mode tag {mode_tag}"));
    }
    if r_cont.is_some() || proof.zk_full_aux.is_some() {
        return Err("Stage A Mode 9/10 rejects Mode 11 zk_full_aux / external r_cont".into());
    }
    if &context.anchor_digest() != anchor {
        return Err("Stage A GKR lane context does not match the public model anchor".into());
    }
    if defer_cprs
        && (!proof.cprs_router_proof.is_empty()
            || !proof.cprs_regular_proof.is_empty()
            || !proof.cprs_shared_proof.is_empty()
            || !proof.cprs_witness_proof.is_empty())
    {
        return Err(
            "deferred Stage A GKR lane verification expects empty per-layer CPRS proofs".into(),
        );
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let verify_trace = std::env::var_os("STWO_ML_GKR_VERIFY_TRACE").is_some();
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let mut lap_mark = std::time::Instant::now();
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let lap = |label: &str, mark: &mut std::time::Instant| {
        if verify_trace {
            eprintln!(
                "[gkr_verify] {label} {:.3} ms",
                mark.elapsed().as_secs_f64() * 1_000.0
            );
            *mark = std::time::Instant::now();
        }
    };
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let mut lap_mark = ();
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let lap = |_label: &str, _mark: &mut ()| {};

    policy.validate()?;
    validate_fused_arithmetic_domain(statement.shape, inputs, policy)?;
    validate_statement_shape(statement)?;
    if statement.policy_id != policy.id()
        || statement.policy_digest != policy.digest()?
        || statement.table_digest != public_tables_for_policy(policy)?.digest
        || statement.input_digest != stage_a_input_digest(inputs)
    {
        return Err("Stage A GKR lane statement metadata mismatch".into());
    }
    if context.shape != statement.shape
        || context.model_digest != statement.model_digest
        || context.manifest_digest != statement.model_registry_manifest_digest
    {
        return Err("Stage A GKR lane context metadata mismatch".into());
    }
    let is_zk_mode = mode_tag == STAGE_A_RECEIPT_MODE_ZK_GKR_LANE;
    let zk_aux_ref = if is_zk_mode {
        Some(
            proof
                .zk_aux
                .as_ref()
                .ok_or_else(|| "Stage A ZK GKR lane proof missing zk_aux".to_string())?,
        )
    } else {
        if proof.zk_aux.is_some() {
            return Err("Stage A Mode 9 proof carries unexpected zk_aux".into());
        }
        None
    };
    if proof.legacy_11col && (is_zk_mode || stage_b.is_some() || defer_cprs) {
        return Err("Legacy 11-column Stage A GKR lane layout is only valid for standalone Mode 9".into());
    }
    let geom = if proof.legacy_11col {
        GkrLaneGeometry::for_shape_legacy_11col(statement.shape)?
    } else {
        GkrLaneGeometry::for_shape(statement.shape)?
    };
    let ext_n_wit = if is_zk_mode { geom.n_wit + 1 } else { geom.n_wit };
    let m_ext = ext_n_wit - 2;
    let witness_cols = if proof.legacy_11col {
        WITNESS_GROUP_COLS_LEGACY
    } else {
        WITNESS_GROUP_COLS
    };
    let tables = GkrLanePublicTables::cached(policy)?;
    let grouped_context = weight_free_grouped_context(statement);
    let active_experts = &grouped_context.active_regular_experts;
    let mut route_active_idx = Vec::with_capacity(statement.shape.batch * statement.shape.top_k);
    for token in 0..statement.shape.batch {
        for &expert in &statement.routes[token] {
            let active_position = active_experts
                .binary_search(&expert)
                .map_err(|_| "Stage A routed expert is absent from the active set".to_string())?;
            route_active_idx.push(active_position);
        }
    }
    let cprs_params = context.validate_for_active_experts_any_f(active_experts, profile)?;
    let eff_wit_f = witness_f_override
        .or_else(|| (defer_cprs && cprs_params.router.f() != 4).then_some(cprs_params.router.f()));
    let witness_params = witness_cprs_params_for_mode_cols_and_f(
        ext_n_wit,
        profile,
        mode_tag,
        witness_cols,
        eff_wit_f,
    )?;
    lap("context_and_tables", &mut lap_mark);

    let active_len = active_experts.len();
    if proof.regular_gate_up_claims.len() != 2 * active_len
        || proof.shared_gate_up_claims.len() != 2
        || proof.regular_gate_weight_openings.len() != active_len
        || proof.regular_up_weight_openings.len() != active_len
        || proof.regular_down_weight_openings.len() != active_len
    {
        return Err("Stage A GKR lane claim or weight opening count mismatch".into());
    }

    let channel = &mut Production100Channel::default();
    channel.enter_scope(TranscriptScope::Statement);
    mix_gkr_lane_statement_with_mode(
        channel,
        statement,
        context,
        profile,
        chain,
        &proof.witness_commitment,
        mode_tag,
    );

    // 1. Router matmul sum-check.
    channel.enter_scope(TranscriptScope::RouterBinding);
    let (router_weight_point, r_rt_init) = verify_gkr_router_sumcheck(
        inputs,
        statement.shape.experts,
        &proof.router,
        channel,
    )?;
    lap("router_sumcheck", &mut lap_mark);

    // 2. Gate/Up matmul sum-checks.
    channel.enter_scope(TranscriptScope::BatchAggregation);
    let gu_token_weights = channel.draw_secure_felts(statement.shape.batch);
    let regular_projection_point =
        channel.draw_secure_felts(grouped_context.hidden_padded.ilog2() as usize);
    let shared_projection_point =
        channel.draw_secure_felts(grouped_context.shared_padded.ilog2() as usize);
    let reg_proj_eq = eq_evals(&regular_projection_point);
    let sh_proj_eq = eq_evals(&shared_projection_point);

    let dimension_rounds = statement.shape.dimension.next_power_of_two().ilog2() as usize;
    channel.enter_scope(TranscriptScope::RegularGateUp);
    channel.mix_felts(&proof.regular_gate_up_claims);
    let reg_gu_batch_weights = channel.draw_secure_felts(2 * active_len);
    let mut reg_gu_expected_claim = SecureField::zero();
    for a in 0..active_len {
        reg_gu_expected_claim += reg_gu_batch_weights[a] * proof.regular_gate_up_claims[a]
            + reg_gu_batch_weights[active_len + a] * proof.regular_gate_up_claims[active_len + a];
    }
    let (regular_input_point, reg_gu_final) =
        partially_verify(reg_gu_expected_claim, &proof.regular_gate_up_sumcheck, channel)?;
    if regular_input_point.len() != dimension_rounds {
        return Err("Stage A GKR regular gate/up sumcheck assignment length mismatch".into());
    }
    channel.mix_felts(&proof.regular_gate_weight_openings);
    channel.mix_felts(&proof.regular_up_weight_openings);
    let reg_w_eval: Vec<SecureField> = (0..active_len)
        .map(|a| {
            reg_gu_batch_weights[a] * proof.regular_gate_weight_openings[a]
                + reg_gu_batch_weights[active_len + a] * proof.regular_up_weight_openings[a]
        })
        .collect();
    let reg_in_eq_i64: Vec<[i64; 4]> = eq_evals(&regular_input_point)
        .into_iter()
        .map(coeff_coords_i64)
        .collect();
    let mut expected_reg_gu_final = SecureField::zero();
    for (t, row) in inputs.iter().enumerate() {
        let mut acc = [0i64; 4];
        for (&v, c) in row.iter().zip(&reg_in_eq_i64) {
            let v = i64::from(v);
            acc[0] += c[0] * v;
            acc[1] += c[1] * v;
            acc[2] += c[2] * v;
            acc[3] += c[3] * v;
        }
        let tok_dot = acc_to_secure(acc);
        let mut tok_w_sum = SecureField::zero();
        let route_base = t * statement.shape.top_k;
        for s in 0..statement.shape.top_k {
            tok_w_sum += reg_w_eval[route_active_idx[route_base + s]];
        }
        expected_reg_gu_final += gu_token_weights[t] * tok_dot * tok_w_sum;
    }
    if reg_gu_final != expected_reg_gu_final {
        return Err("Stage A GKR regular gate/up sumcheck final evaluation mismatch".into());
    }

    channel.enter_scope(TranscriptScope::SharedGateUp);
    channel.mix_felts(&proof.shared_gate_up_claims);
    let sh_gu_batch_weights = channel.draw_secure_felts(2);
    let sh_gu_expected_claim = sh_gu_batch_weights[0] * proof.shared_gate_up_claims[0]
        + sh_gu_batch_weights[1] * proof.shared_gate_up_claims[1];
    let (shared_input_point, sh_gu_final) =
        partially_verify(sh_gu_expected_claim, &proof.shared_gate_up_sumcheck, channel)?;
    if shared_input_point.len() != dimension_rounds {
        return Err("Stage A GKR shared gate/up sumcheck assignment length mismatch".into());
    }
    channel.mix_felts(&[
        proof.shared_gate_weight_opening,
        proof.shared_up_weight_opening,
    ]);
    let sh_in_eq_i64: Vec<[i64; 4]> = eq_evals(&shared_input_point)
        .into_iter()
        .map(coeff_coords_i64)
        .collect();
    let mut sh_in_eval = SecureField::zero();
    for (t, row) in inputs.iter().enumerate() {
        let mut acc = [0i64; 4];
        for (&v, c) in row.iter().zip(&sh_in_eq_i64) {
            let v = i64::from(v);
            acc[0] += c[0] * v;
            acc[1] += c[1] * v;
            acc[2] += c[2] * v;
            acc[3] += c[3] * v;
        }
        sh_in_eval += gu_token_weights[t] * acc_to_secure(acc);
    }
    let sh_w_eval = sh_gu_batch_weights[0] * proof.shared_gate_weight_opening
        + sh_gu_batch_weights[1] * proof.shared_up_weight_opening;
    if sh_gu_final != sh_in_eval * sh_w_eval {
        return Err("Stage A GKR shared gate/up sumcheck final evaluation mismatch".into());
    }
    lap("gate_up_sumchecks", &mut lap_mark);

    // 3. Down matmul sum-checks.
    channel.enter_scope(TranscriptScope::DownProjection);
    let down_token_weights = channel.draw_secure_felts(statement.shape.batch);
    let output_padded = grouped_context.output_padded;
    let output_point = channel.draw_secure_felts(output_padded.ilog2() as usize);
    let out_eq = eq_evals(&output_point);
    channel.mix_felts(&[proof.regular_down_claim, proof.shared_down_claim]);

    let active_bits = grouped_context.active_regular_padded.ilog2() as usize;
    let regular_down_rounds = active_bits + grouped_context.hidden_padded.ilog2() as usize;
    channel.enter_scope(TranscriptScope::RegularDown);
    let _regular_down_lambda = channel.draw_secure_felt();
    let (regular_down_assignment, regular_down_final) =
        partially_verify(proof.regular_down_claim, &proof.regular_down_sumcheck, channel)?;
    if regular_down_assignment.len() != regular_down_rounds {
        return Err("Stage A GKR regular down sumcheck assignment length mismatch".into());
    }
    let (expert_point, hidden_point) = regular_down_assignment.split_at(active_bits);
    let hid_eq = eq_evals(hidden_point);
    channel.mix_felts(&proof.regular_down_weight_openings);
    let expected_regular_weight_opening: SecureField = proof
        .regular_down_weight_openings
        .iter()
        .enumerate()
        .map(|(active, &val)| eq_index(expert_point, active) * val)
        .sum();
    if proof.regular_weight_opening != expected_regular_weight_opening
        || regular_down_final != proof.regular_grouped_claim * proof.regular_weight_opening
    {
        return Err("Stage A GKR regular down-projection reduction mismatch".into());
    }
    channel.mix_felts(&[proof.regular_grouped_claim, proof.regular_weight_opening]);

    let shared_down_rounds = grouped_context.shared_padded.ilog2() as usize;
    channel.enter_scope(TranscriptScope::SharedDown);
    let _shared_down_lambda = channel.draw_secure_felt();
    let (shared_down_assignment, shared_down_final) =
        partially_verify(proof.shared_down_claim, &proof.shared_down_sumcheck, channel)?;
    if shared_down_assignment.len() != shared_down_rounds
        || shared_down_final != proof.shared_grouped_claim * proof.shared_weight_opening
    {
        return Err("Stage A GKR shared down-projection reduction mismatch".into());
    }
    let sh_down_eq = eq_evals(&shared_down_assignment);
    channel.mix_felts(&[proof.shared_grouped_claim, proof.shared_weight_opening]);
    lap("down_sumchecks", &mut lap_mark);

    // 4. Verify batch-unified LogUp-GKR (`R3`).
    channel.enter_scope(TranscriptScope::Lookup);
    let lookup_challenges = GkrLaneLookupChallenges::draw(channel);
    let r_cfs_anchor = if is_zk_mode {
        channel.draw_secure_felts(m_ext - ZK_CFS_TABLE_LOG_ROWS)
    } else {
        Vec::new()
    };
    let plan = gkr_lane_plan_for_mode(&geom, mode_tag);
    if proof.gkr_proof.output_claims_by_instance.len() != plan.len()
        || proof.gkr_proof.layer_masks_by_instance.len() != plan.len()
    {
        return Err("Stage A GKR lane proof instance count mismatch".into());
    }
    for (idx, inst) in plan.iter().enumerate() {
        if proof.gkr_proof.layer_masks_by_instance[idx].len() != inst.n_variables()
            || proof.gkr_proof.output_claims_by_instance[idx].len() != 2
        {
            return Err("Stage A GKR lane proof layer mask or output claim shape mismatch".into());
        }
    }
    let gates = vec![Gate::LogUp; plan.len()];
    let (gkr_artifact, gkr_layer_ood_points) = partially_verify_batch_with_libra_masks(
        gates,
        &proof.gkr_proof,
        zk_aux_ref.map(|z| (z.gkr_libra_sums.as_slice(), z.gkr_libra_evals.as_slice())),
        channel,
    )
    .map_err(|e| format!("Stage A GKR lane batch proof rejected: {e}"))?;
    let r_gkr_last = if let Some(z) = zk_aux_ref {
        channel.mix_felts(&z.gkr_mask_evals);
        *gkr_artifact.ood_point.last().expect("ood_point is non-empty")
    } else {
        SecureField::zero()
    };
    lap("gkr_partially_verify_batch", &mut lap_mark);
    let mut gkr_frac_sum = SecureField::zero();
    for claims in &proof.gkr_proof.output_claims_by_instance {
        let (num, den) = (claims[0], claims[1]);
        if den.is_zero() {
            return Err("Stage A GKR lane output denominator is zero".into());
        }
        gkr_frac_sum += num / den;
    }
    if !gkr_frac_sum.is_zero() {
        return Err("Stage A GKR lane LogUp fractions do not cancel to zero".into());
    }
    let binding = derive_gkr_lane_binding_for_mode(
        &gkr_artifact,
        &geom,
        policy,
        &tables,
        &lookup_challenges,
        channel,
        mode_tag,
        zk_aux_ref.map(|z| &z.gkr_mask_evals),
    )?;
    lap("gkr_binding", &mut lap_mark);

    // 5. Verify `r4_routing_sumcheck` and `r_witness_sumcheck`.
    let r4_gammas = channel.draw_secure_felts(2);
    let (r4_init_claim, rho_r4) = if let Some(z) = zk_aux_ref {
        channel.mix_felts(&[z.r4_cfs_sum]);
        let rho = channel.draw_secure_felt();
        (rho * z.r4_cfs_sum, rho)
    } else {
        (SecureField::zero(), SecureField::zero())
    };
    let (r_rt, r4_final) =
        partially_verify(r4_init_claim, &proof.r4_routing_sumcheck, channel)?;
    if let Some(z) = zk_aux_ref {
        channel.mix_felts(&[z.r4_cfs_eval]);
    }
    if r_rt.len() != geom.rt_rows.ilog2() as usize {
        return Err("Stage A GKR R4 routing sumcheck assignment length mismatch".into());
    }
    let rt_eq = eq_evals(&r_rt);
    let (expected_act_at_r_rt, expected_rw_at_r_rt) =
        eval_public_routing_at_r_rt_with_eq(statement, &geom, &rt_eq);
    let expected_r4_final = eq(&r_rt_init, &r_rt)
        * (r4_gammas[0] * expected_act_at_r_rt + r4_gammas[1] * expected_rw_at_r_rt)
        * (SecureField::one() - expected_act_at_r_rt)
        + rho_r4 * zk_aux_ref.map_or(SecureField::zero(), |z| z.r4_cfs_eval);
    if r4_final != expected_r4_final {
        return Err("Stage A GKR R4 routing sumcheck final evaluation mismatch".into());
    }
    lap("r4_routing", &mut lap_mark);

    let r_prod = channel.draw_secure_felts(geom.n_wit);
    let gamma_prod = channel.draw_secure_felt();
    let gamma_gkr_nl = channel.draw_secure_felt();
    let gu_expert_weights = channel.draw_secure_felts(active_len + 1);
    let gamma_gu_rel1 = channel.draw_secure_felts(2);
    let gamma_down_rel = channel.draw_secure_felts(2);
    let gamma_gkr_out = channel.draw_secure_felt();
    let gamma_gkr_rt = channel.draw_secure_felt();
    let gamma_gkr_tbl = channel.draw_secure_felts(GKR_TABLES);
    let gamma_rt_rel1 = channel.draw_secure_felt();
    let gamma_r4_lin = channel.draw_secure_felts(3);
    let gamma_r5 = channel.draw_secure_felts(2);
    let gamma_gkr_mask = if is_zk_mode {
        channel.draw_secure_felts(ZK_UNMASKED_GKR_INSTANCES)
    } else {
        Vec::new()
    };

    let m_gate = SecureField::from(BaseField::from_u32_unchecked(policy.gate.multiplier as u32));
    let m_up = SecureField::from(BaseField::from_u32_unchecked(policy.up.multiplier as u32));
    let m_prod =
        SecureField::from(BaseField::from_u32_unchecked(policy.product.multiplier as u32));
    let m_rt = SecureField::from(BaseField::from_u32_unchecked(policy.router.multiplier as u32));
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
    let c2048 = SecureField::from(BaseField::from_u32_unchecked(2048));

    let mut gu_gate_sum = gu_expert_weights[active_len] * proof.shared_gate_up_claims[0];
    let mut gu_up_sum = gu_expert_weights[active_len] * proof.shared_gate_up_claims[1];
    for a in 0..active_len {
        gu_gate_sum += gu_expert_weights[a] * proof.regular_gate_up_claims[a];
        gu_up_sum += gu_expert_weights[a] * proof.regular_gate_up_claims[active_len + a];
    }
    let gu_rel1_claim =
        gamma_gu_rel1[0] * m_gate * gu_gate_sum + gamma_gu_rel1[1] * m_up * gu_up_sum;
    let down_rel_claim = gamma_down_rel[0] * proof.regular_grouped_claim
        + gamma_down_rel[1] * proof.shared_grouped_claim;
    let rt_rel1_claim = gamma_rt_rel1
        * (m_rt * proof.router.gate_claim + c2048 * two_s_rt + half_s_rt);
    let (topk_coeffs, topk_bias_eval) =
        build_topk_diff_linear_form_with_eq(statement, &geom, &rt_eq);
    let r4_lin_claim = gamma_r4_lin[0] * expected_act_at_r_rt
        + gamma_r4_lin[1] * expected_rw_at_r_rt
        - gamma_r4_lin[2] * topk_bias_eval;
    let (r5a_rhs, r5b_rhs) = compute_r5_rhs(
        statement,
        policy,
        &down_token_weights,
        &out_eq,
        proof.regular_down_claim,
        proof.shared_down_claim,
    );
    let r5_claim = gamma_r5[0] * r5a_rhs + gamma_r5[1] * r5b_rhs;
    let gkr_tbl_claim: SecureField = gamma_gkr_tbl
        .iter()
        .zip(&binding.tbl_mult_claims)
        .map(|(&g, &c)| g * c)
        .sum();
    let gkr_mask_claim: SecureField = if let Some(z) = zk_aux_ref {
        gamma_gkr_mask
            .iter()
            .zip(&z.gkr_mask_evals)
            .map(|(&g, &e)| g * e)
            .sum()
    } else {
        SecureField::zero()
    };
    let r_witness_claim = gamma_gkr_nl * binding.gkr_nl_claim
        + gu_rel1_claim
        + down_rel_claim
        + gamma_gkr_out * binding.gkr_out_claim
        + gamma_gkr_rt * binding.gkr_rt_claim
        + gkr_tbl_claim
        + gkr_mask_claim
        + rt_rel1_claim
        + r4_lin_claim
        + r5_claim;
    lap("witness_claims", &mut lap_mark);

    let (wit_init_claim, rho_wit) = if let Some(z) = zk_aux_ref {
        channel.mix_felts(&[z.wit_cfs_sum]);
        let rho = channel.draw_secure_felt();
        (r_witness_claim + rho * z.wit_cfs_sum, rho)
    } else {
        (r_witness_claim, SecureField::zero())
    };
    let (r_wit_full, r_wit_final) = partially_verify_with_lo_basis(
        wit_init_claim,
        &proof.r_witness_sumcheck,
        channel,
        is_zk_mode,
    )?;
    if let Some(z) = zk_aux_ref {
        channel.mix_felts(&[z.wit_cfs_eval]);
    }
    lap("witness_partially_verify", &mut lap_mark);
    if r_wit_full.len() != ext_n_wit {
        return Err("Stage A GKR witness sumcheck assignment length mismatch".into());
    }
    let (tau_wit, r_wit) = if is_zk_mode {
        (r_wit_full[0], &r_wit_full[1..])
    } else {
        (SecureField::zero(), &r_wit_full[..])
    };
    if is_zk_mode
        && (tau_wit.is_zero()
            || binding.tau_nl.is_zero()
            || binding.tau_out.is_zero()
            || binding.tau_rt.is_zero()
            || binding.tau_tbl.is_zero())
    {
        return Err("Stage A ZK GKR lane rejects degenerate tau == 0 challenge".into());
    }
    let sel_wit = if is_zk_mode {
        SecureField::one() - tau_wit
    } else {
        SecureField::one()
    };
    let eq_tau_nl = if is_zk_mode {
        eq(&[binding.tau_nl], &[tau_wit])
    } else {
        SecureField::one()
    };
    channel.mix_felts(&proof.witness_openings[..witness_cols]);

    let c = &proof.witness_openings;
    let eq_prod_wit = sel_wit * eq(&r_prod, r_wit);
    let deg3_term = gamma_prod * eq_prod_wit * m_prod * (c[2] - c2048) * (c[1] - c2048);
    let pair0_term =
        eq_prod_wit * (-gamma_prod * (two_s_prod * (c[3] - c2048) + (c[6] - half_s_prod)));
    let eq_nl_wit = eq_tau_nl * eq(&binding.r_nl, r_wit);
    let mut nl_dot = SecureField::zero();
    for j in 0..10 {
        nl_dot += binding.nl_col_weights[j] * c[j];
    }
    let pair1_term = eq_nl_wit * gamma_gkr_nl * nl_dot;
    let (w_raw_at_r_wit_base, w_down_at_r_wit_base) = eval_nl_public_weights_at_r_wit(
        statement,
        &grouped_context,
        &route_active_idx,
        &geom,
        r_wit,
        &gu_token_weights,
        &gu_expert_weights,
        &reg_proj_eq,
        &sh_proj_eq,
        &down_token_weights,
        gamma_down_rel[0],
        gamma_down_rel[1],
        expert_point,
        &hid_eq,
        &sh_down_eq,
    );
    let w_raw_at_r_wit = sel_wit * w_raw_at_r_wit_base;
    let w_down_at_r_wit = sel_wit * w_down_at_r_wit_base;
    lap("nl_public_weights_at_r_wit", &mut lap_mark);
    let pair2_y = gamma_gu_rel1[0] * (two_s_gate * (c[0] - c2048) + (c[4] - half_s_gate))
        + gamma_gu_rel1[1] * (two_s_up * (c[1] - c2048) + (c[5] - half_s_up));
    let pair2_term = w_raw_at_r_wit * pair2_y;
    let pair3_term = w_down_at_r_wit * (c[3] - c2048);
    let w_c10_at_r_wit = eval_w_c10_at_r_wit(
        statement,
        policy,
        &geom,
        &binding,
        r_wit,
        tau_wit,
        is_zk_mode,
        &down_token_weights,
        &output_point,
        &r_rt_init,
        &r_rt,
        &topk_coeffs,
        gamma_gkr_out,
        gamma_gkr_rt,
        &gamma_gkr_tbl,
        gamma_rt_rel1,
        gamma_r4_lin[0],
        gamma_r4_lin[1],
        gamma_r4_lin[2],
        gamma_r5[0],
        gamma_r5[1],
        r_gkr_last,
        &gamma_gkr_mask,
    );
    let out_limbs_term = if geom.legacy_11col {
        SecureField::zero()
    } else {
        eval_w_out_limbs_at_r_wit(
            statement,
            policy,
            &geom,
            &binding,
            r_wit,
            tau_wit,
            is_zk_mode,
            &down_token_weights,
            &output_point,
            gamma_gkr_out,
            gamma_r5[0],
            gamma_r5[1],
            &c[COL_WIT_OUT_LIMB_0..COL_WIT_OUT_LIMB_0 + 6],
        )
    };
    lap("w_c10_at_r_wit", &mut lap_mark);
    let pair4_term = w_c10_at_r_wit * c[10];
    let expected_r_wit_final = deg3_term
        + pair0_term
        + pair1_term
        + pair2_term
        + pair3_term
        + pair4_term
        + out_limbs_term
        + rho_wit * zk_aux_ref.map_or(SecureField::zero(), |z| z.wit_cfs_eval);
    if r_wit_final != expected_r_wit_final {
        return Err("Stage A GKR witness sumcheck final evaluation mismatch".into());
    }

    let r_cfs_sub = if let Some(z) = zk_aux_ref {
        let n_gkr_layers = plan
            .iter()
            .map(GkrPlannedInstance::n_variables)
            .max()
            .unwrap_or(0);
        let r_cfs_pad = channel.draw_secure_felts(ZK_CFS_TABLE_LOG_ROWS - 1);
        channel.mix_felts(&[z.cfs_pad_claim]);
        let gamma_libra_sums = channel.draw_secure_felts(n_gkr_layers);
        let gamma_libra_evals = channel.draw_secure_felts(n_gkr_layers);
        let gamma_r4_cfs = channel.draw_secure_felts(2);
        let gamma_wit_cfs = channel.draw_secure_felts(2);
        let gamma_cfs_pad = channel.draw_secure_felt();
        let w_cfs = build_cfs_binding_weights(
            &gkr_layer_ood_points,
            &r_rt,
            &r_wit_full,
            &r_cfs_pad,
            &gamma_libra_sums,
            &gamma_libra_evals,
            &gamma_r4_cfs,
            &gamma_wit_cfs,
            gamma_cfs_pad,
        );
        let mut cfs_binding_claim = gamma_r4_cfs[0] * z.r4_cfs_sum
            + gamma_r4_cfs[1] * z.r4_cfs_eval
            + gamma_wit_cfs[0] * z.wit_cfs_sum
            + gamma_wit_cfs[1] * z.wit_cfs_eval
            + gamma_cfs_pad * z.cfs_pad_claim;
        for layer in 0..n_gkr_layers {
            cfs_binding_claim += gamma_libra_sums[layer] * z.gkr_libra_sums[layer]
                + gamma_libra_evals[layer] * z.gkr_libra_evals[layer];
        }
        let (r_cfs_sub_pt, cfs_final) =
            partially_verify(cfs_binding_claim, &z.cfs_binding_sumcheck, channel)?;
        if r_cfs_sub_pt.len() != ZK_CFS_TABLE_LOG_ROWS {
            return Err("Stage A ZK GKR CFS binding sumcheck assignment length mismatch".into());
        }
        channel.mix_felts(&[z.blind_col_opening]);
        let expected_cfs_final = mle_at(&w_cfs, &r_cfs_sub_pt) * z.blind_col_opening;
        if cfs_final != expected_cfs_final {
            return Err("Stage A ZK GKR CFS binding sumcheck final evaluation mismatch".into());
        }
        r_cfs_sub_pt
    } else {
        Vec::new()
    };

    // 6. Verify CPRS openings (`witness`, `router`, `regular[0..A]`, `shared`).
    let binding_seed = draw_cprs_binding_seed(channel);
    let d_wit = cprs_witness_outer_digest_for_mode_and_legacy(
        &binding_seed,
        statement,
        &proof.witness_commitment,
        &r_wit_full,
        if is_zk_mode {
            Some(r_cfs_sub.as_slice())
        } else {
            None
        },
        proof.legacy_11col,
    );
    let (wit_points, wit_claims) = if let Some(z) = zk_aux_ref {
        let omega_scale = zk_lo_basis_omega_scale();
        let r_wit_hi_cprs = stwo_point_to_cprs(&r_wit_full[..m_ext]);
        let mut r_cfs_hi = r_cfs_anchor;
        r_cfs_hi.extend_from_slice(&r_cfs_sub);
        let r_cfs_hi_cprs = stwo_point_to_cprs(&r_cfs_hi);
        let mut pts = vec![r_wit_hi_cprs; WITNESS_GROUP_COLS];
        pts.push(r_cfs_hi_cprs);
        let mut cls: Vec<_> = proof
            .witness_openings
            .iter()
            .map(|&v| to_cprs_qm31(omega_scale * v))
            .collect();
        cls.push(to_cprs_qm31(omega_scale * z.blind_col_opening));
        (pts, cls)
    } else {
        let r_wit_cprs = stwo_point_to_cprs(&r_wit_full);
        let cls: Vec<_> = proof.witness_openings[..witness_cols]
            .iter()
            .copied()
            .map(to_cprs_qm31)
            .collect();
        let pts = vec![r_wit_cprs; witness_cols];
        (pts, cls)
    };

    let r_router_cprs = stwo_point_to_cprs(&router_weight_point);
    let mut r_reg_gu_stwo = regular_input_point;
    r_reg_gu_stwo.extend_from_slice(&regular_projection_point);
    let mut r_reg_down_stwo = hidden_point.to_vec();
    r_reg_down_stwo.extend_from_slice(&output_point);
    let r_reg_gu_cprs = stwo_point_to_cprs(&r_reg_gu_stwo);
    let r_reg_down_cprs = stwo_point_to_cprs(&r_reg_down_stwo);
    let mut r_sh_gu_stwo = shared_input_point;
    r_sh_gu_stwo.extend_from_slice(&shared_projection_point);
    let mut r_sh_down_stwo = shared_down_assignment;
    r_sh_down_stwo.extend_from_slice(&output_point);
    let r_sh_gu_cprs = stwo_point_to_cprs(&r_sh_gu_stwo);
    let r_sh_down_cprs = stwo_point_to_cprs(&r_sh_down_stwo);

    if defer_cprs {
        let mut reg_commitments = Vec::with_capacity(active_len);
        let mut reg_points = Vec::with_capacity(3 * active_len);
        let mut reg_claims = Vec::with_capacity(3 * active_len);
        for active in 0..active_len {
            reg_commitments.push(context.active_expert_openings[active].group_commitment);
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_down_cprs.clone());
            reg_claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_up_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_down_weight_openings[active]));
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
        let deferred = StageAGkrLaneDeferredCprs {
            binding_seed,
            active_experts: active_experts.clone(),
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
        };
        return Ok((
            StageARegisteredVerifierRun {
                verifier_context_bytes: context.footprint_bytes(),
                transcript_census: channel.census(),
                transcript_trace: channel.trace(),
            },
            Some(deferred),
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
        .map_err(|e| format!("Stage A+B GKR witness CPRS verification failed: {e}"))?;
    } else {
        verify_strict_context(
            &[proof.witness_commitment],
            &wit_points,
            &wit_claims,
            &d_wit,
            &witness_params,
            &proof.cprs_witness_proof,
        )
        .map_err(|e| format!("Stage A GKR witness CPRS verification failed: {e}"))?;
    }

    let is_unified = proof.cprs_regular_proof.is_empty() && proof.cprs_shared_proof.is_empty();
    if stage_b.is_some() && !is_unified {
        return Err(
            "Stage A GKR + Stage B unified CPRS mode requires cprs_regular_proof and cprs_shared_proof to be empty"
                .into(),
        );
    }
    if is_unified {
        let d_layer = cprs_layer_outer_digest(&binding_seed, statement, active_experts, stage_b);

        let extra_b_groups = stage_b.map_or(0, |b| 1 + b.chunk_openings.len());
        let total_groups = extra_b_groups + 1 + active_len + 1;
        let mut commitments = Vec::with_capacity(total_groups);
        let mut group_params_list = Vec::with_capacity(total_groups);
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
            for &c in &b_deferred.leaf_claims {
                claims.push(to_cprs_qm31(c));
            }
            for op in &b_deferred.chunk_openings {
                let chunk_commit = b_deferred
                    .chunk_commitments
                    .get(op.chunk)
                    .copied()
                    .ok_or_else(|| {
                        "Stage B cache opening names an unknown chunk commitment".to_string()
                    })?;
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

        for active in 0..active_len {
            commitments.push(context.active_expert_openings[active].group_commitment);
            group_params_list.push(cprs_params.regular.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_gu_cprs.clone());
            points.push(r_reg_down_cprs.clone());
            claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[active]));
            claims.push(to_cprs_qm31(proof.regular_up_weight_openings[active]));
            claims.push(to_cprs_qm31(proof.regular_down_weight_openings[active]));
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
        .map_err(|e| format!("Stage A GKR unified CPRS opening rejected: {e}"))?;
    } else {
        if proof.cprs_regular_proof.is_empty() || proof.cprs_shared_proof.is_empty() {
            return Err(
                "Stage A GKR separate CPRS mode requires non-empty cprs_regular_proof and cprs_shared_proof"
                    .into(),
            );
        }
        let d_router = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
            CPRS_CLASS_TAG_ROUTER,
            &ROUTER_SLOT_ROLES,
        );
        let d_reg = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
            CPRS_CLASS_TAG_REGULAR,
            &REGULAR_SLOT_ROLES,
        );
        let d_sh = cprs_outer_digest(
            &binding_seed,
            &statement.model_digest,
            &statement.policy_digest,
            &statement.table_digest,
            &statement.input_digest,
            statement.shape.batch,
            active_experts,
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
        .map_err(|e| format!("Stage A GKR router CPRS verification failed: {e}"))?;

        let reg_commitments: Vec<[u8; 32]> = context
            .active_expert_openings
            .iter()
            .map(|op| op.group_commitment)
            .collect();
        let mut reg_points = Vec::with_capacity(3 * active_len);
        let mut reg_claims = Vec::with_capacity(3 * active_len);
        for active in 0..active_len {
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_gu_cprs.clone());
            reg_points.push(r_reg_down_cprs.clone());
            reg_claims.push(to_cprs_qm31(proof.regular_gate_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_up_weight_openings[active]));
            reg_claims.push(to_cprs_qm31(proof.regular_down_weight_openings[active]));
        }
        verify_strict_context(
            &reg_commitments,
            &reg_points,
            &reg_claims,
            &d_reg,
            &cprs_params.regular,
            &proof.cprs_regular_proof,
        )
        .map_err(|e| format!("Stage A GKR regular CPRS verification failed: {e}"))?;

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
        .map_err(|e| format!("Stage A GKR shared CPRS verification failed: {e}"))?;
    }
    lap("cprs_openings", &mut lap_mark);

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

pub fn put_gkr_batch_proof(out: &mut Vec<u8>, proof: &GkrBatchProof) -> Result<(), String> {
    put_u64(out, proof.sumcheck_proofs.len())?;
    for sumcheck in &proof.sumcheck_proofs {
        put_sumcheck(out, sumcheck)?;
    }
    put_u64(out, proof.layer_masks_by_instance.len())?;
    for masks in &proof.layer_masks_by_instance {
        put_u64(out, masks.len())?;
        for mask in masks {
            let flat = mask
                .columns()
                .iter()
                .flat_map(|[a, b]| [*a, *b])
                .collect::<Vec<_>>();
            put_secure_fields(out, &flat)?;
        }
    }
    put_u64(out, proof.output_claims_by_instance.len())?;
    for claims in &proof.output_claims_by_instance {
        put_secure_fields(out, claims)?;
    }
    Ok(())
}

pub fn read_gkr_batch_proof(
    reader: &mut ReceiptReader<'_>,
    layers_by_instance: &[usize],
) -> Result<GkrBatchProof, String> {
    let n_layers = layers_by_instance.iter().copied().max().unwrap_or(0);
    let sumcheck_count = reader.read_len(n_layers, "Stage A GKR lane sumcheck count")?;
    if sumcheck_count != n_layers {
        return Err("Stage A GKR lane sumcheck count mismatch".into());
    }
    let sumcheck_proofs = (0..n_layers)
        .map(|layer| reader.read_sumcheck(&format!("Stage A GKR lane layer {layer} sumcheck")))
        .collect::<Result<Vec<_>, String>>()?;
    let instance_count =
        reader.read_len(layers_by_instance.len(), "Stage A GKR lane instance count")?;
    if instance_count != layers_by_instance.len() {
        return Err("Stage A GKR lane instance count mismatch".into());
    }
    let mut layer_masks_by_instance = Vec::with_capacity(instance_count);
    for &expected in layers_by_instance {
        let mask_count = reader.read_len(expected, "Stage A GKR lane mask count")?;
        if mask_count != expected {
            return Err("Stage A GKR lane mask count mismatch".into());
        }
        let mut masks = Vec::with_capacity(mask_count);
        for _ in 0..mask_count {
            let flat = reader.read_secure_fields_exact(4, "Stage A GKR lane mask")?;
            masks.push(GkrMask::new(vec![[flat[0], flat[1]], [flat[2], flat[3]]]));
        }
        layer_masks_by_instance.push(masks);
    }
    let claim_count = reader.read_len(instance_count, "Stage A GKR lane output claim count")?;
    if claim_count != instance_count {
        return Err("Stage A GKR lane output claim count mismatch".into());
    }
    let output_claims_by_instance = (0..claim_count)
        .map(|_| reader.read_secure_fields_exact(2, "Stage A GKR lane output claim"))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(GkrBatchProof {
        sumcheck_proofs,
        layer_masks_by_instance,
        output_claims_by_instance,
    })
}

pub fn stage_a_gkr_lane_receipt_bytes(proof: &StageAGkrLaneBatchProof) -> Result<Vec<u8>, String> {
    stage_a_gkr_lane_receipt_bytes_with_mode(proof, STAGE_A_RECEIPT_MODE_GKR_LANE)
}

pub fn stage_a_gkr_lane_receipt_bytes_with_mode(
    proof: &StageAGkrLaneBatchProof,
    mode_tag: u32,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(STAGE_A_REGISTERED_RECEIPT_MAGIC);
    put_u32(&mut out, mode_tag);
    out.extend_from_slice(&proof.witness_commitment);
    put_secure_field(&mut out, proof.router.gate_claim);
    put_sumcheck(&mut out, &proof.router.gate_proof)?;
    put_secure_field(&mut out, proof.router.wg_opening);
    put_secure_fields(&mut out, &proof.regular_gate_up_claims)?;
    put_secure_fields(&mut out, &proof.shared_gate_up_claims)?;
    put_sumcheck(&mut out, &proof.regular_gate_up_sumcheck)?;
    put_secure_fields(&mut out, &proof.regular_gate_weight_openings)?;
    put_secure_fields(&mut out, &proof.regular_up_weight_openings)?;
    put_sumcheck(&mut out, &proof.shared_gate_up_sumcheck)?;
    put_secure_field(&mut out, proof.shared_gate_weight_opening);
    put_secure_field(&mut out, proof.shared_up_weight_opening);
    put_secure_field(&mut out, proof.regular_down_claim);
    put_secure_field(&mut out, proof.shared_down_claim);
    put_secure_field(&mut out, proof.regular_grouped_claim);
    put_secure_field(&mut out, proof.shared_grouped_claim);
    put_secure_fields(&mut out, &proof.regular_down_weight_openings)?;
    put_secure_field(&mut out, proof.regular_weight_opening);
    put_secure_field(&mut out, proof.shared_weight_opening);
    put_sumcheck(&mut out, &proof.regular_down_sumcheck)?;
    put_sumcheck(&mut out, &proof.shared_down_sumcheck)?;
    put_gkr_batch_proof(&mut out, &proof.gkr_proof)?;
    put_sumcheck(&mut out, &proof.r4_routing_sumcheck)?;
    put_sumcheck(&mut out, &proof.r_witness_sumcheck)?;
    let wit_cols = if proof.legacy_11col {
        WITNESS_GROUP_COLS_LEGACY
    } else {
        WITNESS_GROUP_COLS
    };
    put_secure_fields(&mut out, &proof.witness_openings[..wit_cols])?;
    match mode_tag {
        STAGE_A_RECEIPT_MODE_GKR_LANE => {
            if proof.zk_aux.is_some() || proof.zk_full_aux.is_some() {
                return Err("Stage A Mode 9 proof carries unexpected ZK auxiliary payload".into());
            }
        }
        STAGE_A_RECEIPT_MODE_ZK_GKR_LANE => {
            if proof.zk_full_aux.is_some() {
                return Err("Stage A Mode 10 proof carries unexpected Mode 11 zk_full_aux".into());
            }
            let z = proof
                .zk_aux
                .as_ref()
                .ok_or_else(|| "Stage A ZK GKR lane proof missing zk_aux".to_string())?;
            put_secure_fields(&mut out, &z.gkr_mask_evals)?;
            put_secure_fields(&mut out, &z.gkr_libra_sums)?;
            put_secure_fields(&mut out, &z.gkr_libra_evals)?;
            put_secure_field(&mut out, z.r4_cfs_sum);
            put_secure_field(&mut out, z.r4_cfs_eval);
            put_secure_field(&mut out, z.wit_cfs_sum);
            put_secure_field(&mut out, z.wit_cfs_eval);
            put_secure_field(&mut out, z.cfs_pad_claim);
            put_sumcheck(&mut out, &z.cfs_binding_sumcheck)?;
            put_secure_field(&mut out, z.blind_col_opening);
        }
        STAGE_A_RECEIPT_MODE_ZK_FULL => {
            if proof.zk_aux.is_some() {
                return Err("Stage A Mode 11 proof carries unexpected Mode 10 zk_aux".into());
            }
            let zf = proof
                .zk_full_aux
                .as_ref()
                .ok_or_else(|| "Stage A Mode 11 proof missing zk_full_aux".to_string())?;
            zk_full::put_stage_a_zk_full_aux(&mut out, zf)?;
        }
        other => {
            return Err(format!(
                "unsupported Stage A GKR lane receipt mode tag {other}"
            ));
        }
    }
    put_bytes(&mut out, &proof.cprs_router_proof)?;
    put_bytes(&mut out, &proof.cprs_regular_proof)?;
    put_bytes(&mut out, &proof.cprs_shared_proof)?;
    put_bytes(&mut out, &proof.cprs_witness_proof)?;
    Ok(out)
}

pub fn decode_stage_a_gkr_lane_receipt(
    bytes: &[u8],
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
) -> Result<StageAGkrLaneBatchProof, String> {
    decode_stage_a_gkr_lane_receipt_with_mode(
        bytes,
        statement,
        policy,
        STAGE_A_RECEIPT_MODE_GKR_LANE,
    )
}

pub fn decode_stage_a_gkr_lane_receipt_with_mode(
    bytes: &[u8],
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
    expected_mode_tag: u32,
) -> Result<StageAGkrLaneBatchProof, String> {
    if expected_mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        zk_full::validate_zk_full_statement_shape(statement)?;
    } else {
        validate_statement_shape(statement)?;
    }
    if statement.policy_id != policy.id() {
        return Err("Stage A GKR lane receipt policy does not match the statement".into());
    }
    let active = if expected_mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        statement.shape.experts
    } else {
        statement
            .routes
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    };
    let geom = if expected_mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        zk_full::gkr_lane_geometry_for_zk_full(statement.shape)?
    } else {
        GkrLaneGeometry::for_shape(statement.shape)?
    };
    let plan = if expected_mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL {
        zk_full::gkr_lane_plan_for_zk_full(&geom)
    } else {
        gkr_lane_plan_for_mode(&geom, expected_mode_tag)
    };
    let layers_by_instance: Vec<usize> = plan.iter().map(GkrPlannedInstance::n_variables).collect();

    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(STAGE_A_REGISTERED_RECEIPT_MAGIC, "Stage A receipt magic")?;
    let mode_tag = reader.read_u32("Stage A receipt mode tag")?;
    if mode_tag != expected_mode_tag {
        return Err(format!(
            "unsupported Stage A GKR lane receipt mode tag {mode_tag}; expected {expected_mode_tag}"
        ));
    }
    let mut witness_commitment = [0u8; 32];
    for b in &mut witness_commitment {
        *b = reader.read_u32_byte("witness commitment byte")?;
    }
    let gate_claim = reader.read_secure_field("router gate claim")?;
    let gate_proof = reader.read_sumcheck("router gate sumcheck")?;
    let wg_opening = reader.read_secure_field("router wg opening")?;
    let regular_gate_up_claims =
        reader.read_secure_fields_exact(2 * active, "regular gate/up claims")?;
    let shared_gate_up_claims = reader.read_secure_fields_exact(2, "shared gate/up claims")?;
    let regular_gate_up_sumcheck = reader.read_sumcheck("regular gate/up sumcheck")?;
    let regular_gate_weight_openings =
        reader.read_secure_fields_exact(active, "regular gate weight openings")?;
    let regular_up_weight_openings =
        reader.read_secure_fields_exact(active, "regular up weight openings")?;
    let shared_gate_up_sumcheck = reader.read_sumcheck("shared gate/up sumcheck")?;
    let shared_gate_weight_opening = reader.read_secure_field("shared gate weight opening")?;
    let shared_up_weight_opening = reader.read_secure_field("shared up weight opening")?;
    let regular_down_claim = reader.read_secure_field("regular down claim")?;
    let shared_down_claim = reader.read_secure_field("shared down claim")?;
    let regular_grouped_claim = reader.read_secure_field("regular grouped claim")?;
    let shared_grouped_claim = reader.read_secure_field("shared grouped claim")?;
    let regular_down_weight_openings =
        reader.read_secure_fields_exact(active, "regular down weight openings")?;
    let regular_weight_opening = reader.read_secure_field("regular weight opening")?;
    let shared_weight_opening = reader.read_secure_field("shared weight opening")?;
    let regular_down_sumcheck = reader.read_sumcheck("regular down sumcheck")?;
    let shared_down_sumcheck = reader.read_sumcheck("shared down sumcheck")?;
    let saved_reader = reader.clone();
    let (gkr_proof, legacy_11col) = match read_gkr_batch_proof(&mut reader, &layers_by_instance) {
        Ok(p) => (p, false),
        Err(e) if expected_mode_tag == STAGE_A_RECEIPT_MODE_GKR_LANE => {
            let geom_legacy = GkrLaneGeometry::for_shape_legacy_11col(statement.shape)?;
            let plan_legacy = gkr_lane_plan_for_mode(&geom_legacy, expected_mode_tag);
            let layers_legacy: Vec<usize> = plan_legacy
                .iter()
                .map(GkrPlannedInstance::n_variables)
                .collect();
            let mut legacy_reader = saved_reader;
            match read_gkr_batch_proof(&mut legacy_reader, &layers_legacy) {
                Ok(p) => {
                    reader = legacy_reader;
                    (p, true)
                }
                Err(_) => return Err(e),
            }
        }
        Err(e) => return Err(e),
    };
    let r4_routing_sumcheck = reader.read_sumcheck("r4 routing sumcheck")?;
    let r_witness_sumcheck = reader.read_sumcheck("r_witness sumcheck")?;
    let wit_cols = if legacy_11col {
        WITNESS_GROUP_COLS_LEGACY
    } else {
        WITNESS_GROUP_COLS
    };
    let witness_openings_vec = reader.read_secure_fields_exact(wit_cols, "witness openings")?;
    let mut witness_openings = [SecureField::zero(); WITNESS_GROUP_COLS];
    witness_openings[..wit_cols].copy_from_slice(&witness_openings_vec);
    let (zk_aux, zk_full_aux) = match expected_mode_tag {
        STAGE_A_RECEIPT_MODE_GKR_LANE => (None, None),
        STAGE_A_RECEIPT_MODE_ZK_GKR_LANE => {
            let n_gkr_layers = layers_by_instance.iter().copied().max().unwrap_or(0);
            let mask_evals_vec =
                reader.read_secure_fields_exact(ZK_UNMASKED_GKR_INSTANCES, "zk gkr mask evals")?;
            let gkr_mask_evals: [SecureField; ZK_UNMASKED_GKR_INSTANCES] = mask_evals_vec
                .try_into()
                .map_err(|_| "zk gkr mask evals count mismatch".to_string())?;
            let gkr_libra_sums =
                reader.read_secure_fields_exact(n_gkr_layers, "zk gkr libra sums")?;
            let gkr_libra_evals =
                reader.read_secure_fields_exact(n_gkr_layers, "zk gkr libra evals")?;
            let r4_cfs_sum = reader.read_secure_field("zk r4 cfs sum")?;
            let r4_cfs_eval = reader.read_secure_field("zk r4 cfs eval")?;
            let wit_cfs_sum = reader.read_secure_field("zk wit cfs sum")?;
            let wit_cfs_eval = reader.read_secure_field("zk wit cfs eval")?;
            let cfs_pad_claim = reader.read_secure_field("zk cfs pad claim")?;
            let cfs_binding_sumcheck = reader.read_sumcheck("zk cfs binding sumcheck")?;
            let blind_col_opening = reader.read_secure_field("zk blind col opening")?;
            (
                Some(StageAZkLaneAux {
                    gkr_mask_evals,
                    gkr_libra_sums,
                    gkr_libra_evals,
                    r4_cfs_sum,
                    r4_cfs_eval,
                    wit_cfs_sum,
                    wit_cfs_eval,
                    cfs_pad_claim,
                    cfs_binding_sumcheck,
                    blind_col_opening,
                }),
                None,
            )
        }
        STAGE_A_RECEIPT_MODE_ZK_FULL => {
            let n_gkr_layers = layers_by_instance.iter().copied().max().unwrap_or(0);
            let zf = zk_full::read_stage_a_zk_full_aux(&mut reader, n_gkr_layers)?;
            (None, Some(zf))
        }
        other => {
            return Err(format!(
                "unsupported Stage A GKR lane receipt mode tag {other}"
            ));
        }
    };
    let cprs_router_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "cprs router proof")?
        .to_vec();
    let cprs_regular_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "cprs regular proof")?
        .to_vec();
    let cprs_shared_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "cprs shared proof")?
        .to_vec();
    let cprs_witness_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "cprs witness proof")?
        .to_vec();
    reader.finish()?;

    let proof = StageAGkrLaneBatchProof {
        witness_commitment,
        router: StageARegisteredRouterProof {
            gate_claim,
            gate_proof,
            wg_opening,
        },
        regular_gate_up_claims,
        shared_gate_up_claims,
        regular_gate_up_sumcheck,
        regular_gate_weight_openings,
        regular_up_weight_openings,
        shared_gate_up_sumcheck,
        shared_gate_weight_opening,
        shared_up_weight_opening,
        regular_down_claim,
        shared_down_claim,
        regular_grouped_claim,
        shared_grouped_claim,
        regular_down_weight_openings,
        regular_weight_opening,
        shared_weight_opening,
        regular_down_sumcheck,
        shared_down_sumcheck,
        gkr_proof,
        r4_routing_sumcheck,
        r_witness_sumcheck,
        witness_openings,
        legacy_11col,
        zk_aux,
        zk_full_aux,
        cprs_router_proof,
        cprs_regular_proof,
        cprs_shared_proof,
        cprs_witness_proof,
    };
    if stage_a_gkr_lane_receipt_bytes_with_mode(&proof, expected_mode_tag)? != bytes {
        return Err("non-canonical Stage A GKR lane receipt encoding".into());
    }
    Ok(proof)
}

#[derive(Clone, Debug)]
pub enum StageAReceiptProof {
    PublicWitness(StageAPublicWitnessBatchProof),
    GkrLane(StageAGkrLaneBatchProof),
    ZkGkrLane(StageAGkrLaneBatchProof),
    ZkFull(StageAGkrLaneBatchProof),
}

impl StageAReceiptProof {
    pub fn mode_id(&self) -> &'static str {
        match self {
            Self::PublicWitness(_) => "public_witness",
            Self::GkrLane(_) => "gkr_lane",
            Self::ZkGkrLane(_) => "zk_gkr_lane",
            Self::ZkFull(_) => "zk_full",
        }
    }
}

pub fn decode_stage_a_receipt_bytes(
    bytes: &[u8],
    statement: &StageANonlinearBatchStatement,
    policy: StageANumericPolicy,
) -> Result<StageAReceiptProof, String> {
    if bytes.len() < 12 || &bytes[..8] != STAGE_A_REGISTERED_RECEIPT_MAGIC {
        return Err("Stage A receipt missing magic header".into());
    }
    let tag = u32::from_le_bytes(bytes[8..12].try_into().expect("4-byte mode tag"));
    match tag {
        STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS | STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8 => {
            decode_stage_a_public_witness_receipt(bytes, statement, policy)
                .map(StageAReceiptProof::PublicWitness)
        }
        STAGE_A_RECEIPT_MODE_GKR_LANE => {
            decode_stage_a_gkr_lane_receipt(bytes, statement, policy)
                .map(StageAReceiptProof::GkrLane)
        }
        STAGE_A_RECEIPT_MODE_ZK_GKR_LANE => decode_stage_a_gkr_lane_receipt_with_mode(
            bytes,
            statement,
            policy,
            STAGE_A_RECEIPT_MODE_ZK_GKR_LANE,
        )
        .map(StageAReceiptProof::ZkGkrLane),
        STAGE_A_RECEIPT_MODE_ZK_FULL => decode_stage_a_gkr_lane_receipt_with_mode(
            bytes,
            statement,
            policy,
            STAGE_A_RECEIPT_MODE_ZK_FULL,
        )
        .map(StageAReceiptProof::ZkFull),
        other => Err(format!(
            "unsupported Stage A receipt mode tag {other}; expected {STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS}, {STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8}, {STAGE_A_RECEIPT_MODE_GKR_LANE}, {STAGE_A_RECEIPT_MODE_ZK_GKR_LANE}, or {STAGE_A_RECEIPT_MODE_ZK_FULL}"
        )),
    }
}

pub fn encode_stage_a_receipt_bytes_matching_header(
    proof: &StageAReceiptProof,
    reference_bytes: &[u8],
) -> Result<Vec<u8>, String> {
    let tag = if reference_bytes.len() >= 12 {
        Some(u32::from_le_bytes(
            reference_bytes[8..12].try_into().expect("4-byte mode tag"),
        ))
    } else {
        None
    };
    match (proof, tag) {
        (
            StageAReceiptProof::PublicWitness(pw),
            Some(STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS),
        ) => stage_a_public_witness_receipt_bytes_with_version(
            pw,
            STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS,
        ),
        (StageAReceiptProof::PublicWitness(pw), _) => {
            stage_a_public_witness_receipt_bytes_with_version(
                pw,
                STAGE_A_RECEIPT_MODE_PUBLIC_WITNESS_V8,
            )
        }
        (StageAReceiptProof::GkrLane(gkr), _) => stage_a_gkr_lane_receipt_bytes(gkr),
        (StageAReceiptProof::ZkGkrLane(gkr), _) => {
            stage_a_gkr_lane_receipt_bytes_with_mode(gkr, STAGE_A_RECEIPT_MODE_ZK_GKR_LANE)
        }
        (StageAReceiptProof::ZkFull(gkr), _) => {
            stage_a_gkr_lane_receipt_bytes_with_mode(gkr, STAGE_A_RECEIPT_MODE_ZK_FULL)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_receipt_against_anchor(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAReceiptProof,
    chain: &StageAChainLink,
) -> Result<StageARegisteredVerifierRun, String> {
    verify_stage_a_receipt_against_anchor_with_stage_b(
        anchor, context, inputs, policy, profile, statement, proof, chain, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_stage_a_receipt_against_anchor_with_stage_b(
    anchor: &[u8; 32],
    context: &StageARegisteredContext,
    inputs: &[Vec<i16>],
    policy: StageANumericPolicy,
    profile: &SecurityProfile,
    statement: &StageANonlinearBatchStatement,
    proof: &StageAReceiptProof,
    chain: &StageAChainLink,
    stage_b: Option<&StageBDeferredCprsClaims>,
) -> Result<StageARegisteredVerifierRun, String> {
    match proof {
        StageAReceiptProof::PublicWitness(pw) => {
            verify_stage_a_public_witness_batch_against_anchor_with_stage_b(
                anchor, context, inputs, policy, profile, statement, pw, chain, stage_b,
            )
        }
        StageAReceiptProof::GkrLane(gkr) => verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b(
            anchor, context, inputs, policy, profile, statement, gkr, chain, stage_b,
        ),
        StageAReceiptProof::ZkGkrLane(gkr) => {
            verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
                anchor,
                context,
                inputs,
                policy,
                profile,
                statement,
                gkr,
                chain,
                stage_b,
                STAGE_A_RECEIPT_MODE_ZK_GKR_LANE,
            )
        }
        StageAReceiptProof::ZkFull(gkr) => {
            verify_stage_a_gkr_lane_batch_against_anchor_with_stage_b_mode(
                anchor,
                context,
                inputs,
                policy,
                profile,
                statement,
                gkr,
                chain,
                stage_b,
                STAGE_A_RECEIPT_MODE_ZK_FULL,
            )
        }
    }
}

pub const CROSS_LAYER_GKR_BLOCK_MAGIC: &[u8; 8] = b"STWOBLK1";
pub const CROSS_LAYER_GKR_BLOCK_VERSION: u32 = 1;
const CROSS_LAYER_BLOCK_DOMAIN_TAG: u64 = 0x5354_574f_424c_4b31;

#[derive(Clone, Debug)]
pub struct CrossLayerGkrBlockProof {
    pub mode_tag: u32,
    pub cprs_weight_f: u8,
    pub cprs_witness_f: u8,
    pub stacked_expert_pack: u32,
    pub stage_b_proofs: Vec<crate::stage_b::StageBProof>,
    pub stage_a_proofs: Vec<StageAGkrLaneBatchProof>,
    pub layer_inactive_expert_openings: Vec<Vec<[SecureField; 3]>>,
    pub block_cprs_weight_proof: Vec<u8>,
    pub block_cprs_witness_proof: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CrossLayerGkrBlockVerifierRun {
    pub layers: usize,
    pub batch: usize,
}

fn cross_layer_step_chain_bytes(b_wire: Option<&[u8]>, a_wire: &[u8]) -> Vec<u8> {
    if let Some(bw) = b_wire {
        let mut out = Vec::with_capacity(16 + bw.len() + a_wire.len());
        out.extend_from_slice(b"STWOLYR1");
        out.extend_from_slice(&(bw.len() as u64).to_le_bytes());
        out.extend_from_slice(bw);
        out.extend_from_slice(a_wire);
        out
    } else {
        a_wire.to_vec()
    }
}

fn eval_i16_matrix_mle_with_eq(
    rows: &[Vec<i16>],
    eq_tok: &[SecureField],
    eq_dim_i64: &[[i64; 4]],
) -> SecureField {
    let mut total = SecureField::zero();
    for (row, &w_t) in rows.iter().zip(eq_tok) {
        let mut acc = [0i64; 4];
        for (&v, c) in row.iter().zip(eq_dim_i64) {
            let v = i64::from(v);
            acc[0] += c[0] * v;
            acc[1] += c[1] * v;
            acc[2] += c[2] * v;
            acc[3] += c[3] * v;
        }
        total += w_t * acc_to_secure(acc);
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn absorb_cross_layer_block_header_and_layers(
    channel: &mut Production100Channel,
    mode_tag: u32,
    num_layers: usize,
    has_stage_b: bool,
    cprs_weight_f: u8,
    cprs_witness_f: u8,
    stacked_expert_pack: u32,
    initial_chain: &StageAChainLink,
    stage_a_anchors: &[[u8; 32]],
    stage_a_statements: &[StageANonlinearBatchStatement],
    stage_a_deferred: &[StageAGkrLaneDeferredCprs],
    stacked_bank_commitments: Option<&[[u8; 32]]>,
    layer_inactive_expert_openings: &[Vec<[SecureField; 3]>],
    stage_b_deferred: Option<&[StageBDeferredCprsClaims]>,
) {
    channel.mix_u64(CROSS_LAYER_BLOCK_DOMAIN_TAG);
    channel.mix_u32s(&[
        CROSS_LAYER_GKR_BLOCK_VERSION,
        mode_tag,
        num_layers as u32,
        u32::from(has_stage_b),
        initial_chain.layer,
        u32::from(cprs_weight_f),
        u32::from(cprs_witness_f),
        stacked_expert_pack,
    ]);
    mix_digest(channel, &initial_chain.prev_receipt_digest);
    for l in 0..num_layers {
        mix_digest(channel, &stage_a_anchors[l]);
        mix_digest(channel, &stage_a_deferred[l].binding_seed);
        mix_digest(channel, &stage_a_deferred[l].d_wit);
        mix_digest(channel, &stage_a_deferred[l].witness_commitment);
        mix_digest(channel, &stage_a_statements[l].input_digest);
        mix_digest(channel, &stage_a_input_digest(&stage_a_statements[l].output_q8));
        if stacked_expert_pack > 0 {
            if let Some(bank_comms) = stacked_bank_commitments {
                mix_digest(channel, &bank_comms[l]);
            }
            let inact = &layer_inactive_expert_openings[l];
            channel.mix_u32s(&[inact.len() as u32]);
            if !inact.is_empty() {
                let mut flat = Vec::with_capacity(3 * inact.len());
                for &[g, u, d] in inact {
                    flat.push(g);
                    flat.push(u);
                    flat.push(d);
                }
                channel.mix_felts(&flat);
            }
        }
        if let Some(b_slice) = stage_b_deferred {
            let bd = &b_slice[l];
            mix_digest(channel, &bd.statement_digest);
            mix_digest(channel, &bd.attention_commitment);
            if let Some(ref wb) = bd.witness_opening {
                mix_digest(channel, &wb.witness_commitment);
            }
        }
    }
}

fn cprs_cross_layer_outer_digest(
    domain: &[u8],
    block_seed: &[u8; 32],
    mode_tag: u32,
    num_layers: usize,
    has_stage_b: bool,
    cprs_weight_f: u8,
    cprs_witness_f: u8,
    stacked_expert_pack: u32,
) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(domain);
    h.update(block_seed);
    h.update(mode_tag.to_le_bytes());
    h.update((num_layers as u64).to_le_bytes());
    h.update([u8::from(has_stage_b), cprs_weight_f, cprs_witness_f]);
    h.update(stacked_expert_pack.to_le_bytes());
    let mut out: [u8; 32] = h.finalize().into();
    if out == [0u8; 32] {
        out[0] = 1;
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn verify_cross_layer_gkr_block(
    stage_a_anchors: &[[u8; 32]],
    stage_a_contexts: &[StageARegisteredContext],
    stacked_bank_commitments: Option<&[[u8; 32]]>,
    layer_inputs_override: Option<&[Vec<Vec<i16>>]>,
    stage_a_policy: StageANumericPolicy,
    profile: &SecurityProfile,
    stage_a_statements: &[StageANonlinearBatchStatement],
    stage_b: Option<(
        &[[u8; 32]],
        crate::stage_b::AttentionPolicy,
        &[crate::stage_b::StageBStatement],
    )>,
    proof: &CrossLayerGkrBlockProof,
    initial_chain: &StageAChainLink,
) -> Result<CrossLayerGkrBlockVerifierRun, String> {
    let single_policy = stage_b.map(|(_, p, _)| p);
    let stage_b_slice = stage_b
        .zip(single_policy.as_ref())
        .map(|((comms, _, stmts), p_ref)| (comms, std::slice::from_ref(p_ref), stmts));
    verify_cross_layer_gkr_block_with_policies(
        stage_a_anchors,
        stage_a_contexts,
        stacked_bank_commitments,
        layer_inputs_override,
        stage_a_policy,
        profile,
        stage_a_statements,
        stage_b_slice,
        proof,
        initial_chain,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_cross_layer_gkr_block_with_policies(
    stage_a_anchors: &[[u8; 32]],
    stage_a_contexts: &[StageARegisteredContext],
    stacked_bank_commitments: Option<&[[u8; 32]]>,
    layer_inputs_override: Option<&[Vec<Vec<i16>>]>,
    stage_a_policy: StageANumericPolicy,
    profile: &SecurityProfile,
    stage_a_statements: &[StageANonlinearBatchStatement],
    stage_b: Option<(
        &[[u8; 32]],
        &[crate::stage_b::AttentionPolicy],
        &[crate::stage_b::StageBStatement],
    )>,
    proof: &CrossLayerGkrBlockProof,
    initial_chain: &StageAChainLink,
) -> Result<CrossLayerGkrBlockVerifierRun, String> {
    let k = stage_a_statements.len();
    if k == 0
        || stage_a_anchors.len() != k
        || stage_a_contexts.len() != k
        || proof.stage_a_proofs.len() != k
    {
        return Err("CrossLayerGkrBlockProof Stage A layer count mismatch".into());
    }
    if !(1..=4).contains(&proof.cprs_weight_f) || !(1..=4).contains(&proof.cprs_witness_f) {
        return Err("CrossLayerGkrBlockProof CPRS arity f out of range 1..=4".into());
    }
    if proof.mode_tag != STAGE_A_RECEIPT_MODE_GKR_LANE
        && proof.mode_tag != STAGE_A_RECEIPT_MODE_ZK_GKR_LANE
        && proof.mode_tag != STAGE_A_RECEIPT_MODE_ZK_FULL
    {
        return Err(format!(
            "unsupported CrossLayerGkrBlockProof mode tag {}",
            proof.mode_tag
        ));
    }
    if proof.stacked_expert_pack > 0 {
        if !proof.stacked_expert_pack.is_power_of_two() {
            return Err("CrossLayerGkrBlockProof stacked_expert_pack must be a power of two".into());
        }
        let bank_comms = stacked_bank_commitments.ok_or_else(|| {
            "CrossLayerGkrBlockProof with stacked_expert_pack > 0 requires stacked_bank_commitments"
                .to_string()
        })?;
        if bank_comms.len() != k || proof.layer_inactive_expert_openings.len() != k {
            return Err("CrossLayerGkrBlockProof stacked bank layer count mismatch".into());
        }
    } else {
        if stacked_bank_commitments.is_some() {
            return Err(
                "CrossLayerGkrBlockProof with stacked_expert_pack == 0 rejects stacked_bank_commitments"
                    .into(),
            );
        }
        if !proof.layer_inactive_expert_openings.is_empty() {
            return Err(
                "CrossLayerGkrBlockProof with stacked_expert_pack == 0 carries inactive openings"
                    .into(),
            );
        }
    }

    let has_stage_b = stage_b.is_some();
    if has_stage_b {
        if layer_inputs_override.is_some() {
            return Err("CrossLayerGkrBlockProof with Stage B derives Stage A inputs from Stage B".into());
        }
        let (attn_comms, attn_policies, b_stmts) = stage_b.unwrap();
        if attn_comms.len() != k || b_stmts.len() != k || proof.stage_b_proofs.len() != k {
            return Err("CrossLayerGkrBlockProof Stage B layer count mismatch".into());
        }
        if attn_policies.len() != 1 && attn_policies.len() != k {
            return Err("CrossLayerGkrBlockProof Stage B policy slice must have length 1 or K".into());
        }
    } else {
        if !proof.stage_b_proofs.is_empty() {
            return Err("Stage-A-only CrossLayerGkrBlockProof carries unexpected Stage B proofs".into());
        }
        let ov = layer_inputs_override
            .ok_or_else(|| "Stage-A-only CrossLayerGkrBlockProof requires layer_inputs_override".to_string())?;
        if ov.len() != k {
            return Err("layer_inputs_override length mismatch".into());
        }
    }
    let is_zk_full = proof.mode_tag == STAGE_A_RECEIPT_MODE_ZK_FULL;

    let batch = stage_a_statements[0].shape.batch;
    let dim = stage_a_statements[0].shape.dimension;
    for s in stage_a_statements {
        if s.shape.batch != batch || s.shape.dimension != dim {
            return Err("all layers in a CrossLayerGkrBlockProof must share batch and dimension".into());
        }
    }
    let weight_f_ov = Some(proof.cprs_weight_f as usize);
    let witness_f_ov = Some(proof.cprs_witness_f as usize);
    let r_cont_opt = if is_zk_full {
        Some(zk_full::draw_zk_block_cont_point(
            initial_chain,
            stage_a_anchors,
            stage_a_statements,
            stage_b.map(|(c, _, s)| (c, s)),
            proof,
        )?)
    } else {
        None
    };

    let mut stage_b_deferred_vec = Vec::with_capacity(if has_stage_b { k } else { 0 });
    let mut stage_a_deferred_vec = Vec::with_capacity(k);
    let mut stage_b_a_rows: Vec<Vec<Vec<i16>>> = Vec::with_capacity(if has_stage_b { k } else { 0 });
    let mut stage_b_h_in_rows: Vec<Vec<Vec<i16>>> = Vec::with_capacity(if has_stage_b { k } else { 0 });

    let mut chain_l = initial_chain.clone();
    for l in 0..k {
        let (derived_x_opt, b_wire_opt) = if let Some((attn_comms, attn_policies, b_stmts)) = stage_b {
            let b_stmt = &b_stmts[l];
            let layer_attn_policy = if attn_policies.len() == 1 {
                attn_policies[0]
            } else {
                attn_policies[l]
            };
            if b_stmt.attention_commitment != attn_comms[l] {
                return Err(format!(
                    "layer {l} Stage B attention_commitment does not match expected anchor"
                ));
            }
            if b_stmt.binding != chain_l.prev_receipt_digest {
                return Err(format!(
                    "layer {l} Stage B statement binding does not match cross-layer chain link"
                ));
            }
            if b_stmt.shape.dimension != dim || b_stmt.tokens.len() != batch {
                return Err(format!("layer {l} Stage B shape mismatch with Stage A"));
            }
            if is_zk_full
                && !proof.stage_b_proofs[l]
                    .gkr_witness
                    .as_ref()
                    .is_some_and(|g| g.zk_aux.is_some())
            {
                return Err(format!(
                    "layer {l} Mode 11 cross-layer block requires Stage B v3 ZK proof"
                ));
            }
            let leaf_params = crate::stage_b::group_params_with_f(
                2 * b_stmt.shape.dimension.ilog2() as usize,
                4,
                weight_f_ov,
                profile,
            )?;
            let (b_run, b_deferred) =
                crate::stage_b::verify_stage_b_defer_cprs_with_cont(
                    b_stmt,
                    layer_attn_policy,
                    profile,
                    &leaf_params,
                    &proof.stage_b_proofs[l],
                    witness_f_ov,
                    r_cont_opt.as_deref(),
                )?;
            let b_wire = crate::stage_b::stage_b_receipt_bytes(&proof.stage_b_proofs[l])?;
            let h_in_l: Vec<Vec<i16>> = b_stmt.tokens.iter().map(|t| t.step.h_in.clone()).collect();
            let mut a_l = Vec::with_capacity(batch);
            let mut x_l = Vec::with_capacity(batch);
            for r in b_run.recomputed {
                a_l.push(r.a);
                x_l.push(r.x);
            }
            if is_zk_full && x_l.first().is_some_and(Vec::is_empty) {
                x_l.clear();
            }
            stage_b_h_in_rows.push(h_in_l);
            stage_b_a_rows.push(a_l);
            stage_b_deferred_vec.push(b_deferred);
            (Some(x_l), Some(b_wire))
        } else {
            (None, None)
        };

        let inputs_ref: &[Vec<i16>] = if let Some(ref x_l) = derived_x_opt {
            x_l.as_slice()
        } else {
            layer_inputs_override.unwrap()[l].as_slice()
        };

        let (_a_run, a_deferred) = verify_stage_a_gkr_lane_batch_defer_cprs_with_f_and_cont(
            &stage_a_anchors[l],
            &stage_a_contexts[l],
            inputs_ref,
            stage_a_policy,
            profile,
            &stage_a_statements[l],
            &proof.stage_a_proofs[l],
            &chain_l,
            proof.mode_tag,
            witness_f_ov,
            r_cont_opt.as_deref(),
        )?;
        if a_deferred.router_params.f() != proof.cprs_weight_f as usize {
            return Err(format!(
                "layer {l} Stage A context CPRS f={} does not match block proof cprs_weight_f={}",
                a_deferred.router_params.f(),
                proof.cprs_weight_f
            ));
        }
        if proof.stacked_expert_pack > 0 {
            let expected_inact =
                stage_a_statements[l].shape.experts - a_deferred.active_experts.len();
            if proof.layer_inactive_expert_openings[l].len() != expected_inact {
                return Err(format!(
                    "layer {l} inactive expert opening count {} does not match expected {}",
                    proof.layer_inactive_expert_openings[l].len(),
                    expected_inact
                ));
            }
        }

        if l + 1 < k {
            let a_wire =
                stage_a_gkr_lane_receipt_bytes_with_mode(&proof.stage_a_proofs[l], proof.mode_tag)?;
            let step_bytes = cross_layer_step_chain_bytes(b_wire_opt.as_deref(), &a_wire);
            let next_layer = initial_chain
                .layer
                .checked_add((l + 1) as u32)
                .ok_or_else(|| "cross-layer block initial_chain.layer + layer_idx overflows u32".to_string())?;
            chain_l = StageAChainLink::next(next_layer, &step_bytes);
        }
        stage_a_deferred_vec.push(a_deferred);
    }

    let mut block_channel = Production100Channel::default();
    absorb_cross_layer_block_header_and_layers(
        &mut block_channel,
        proof.mode_tag,
        k,
        has_stage_b,
        proof.cprs_weight_f,
        proof.cprs_witness_f,
        proof.stacked_expert_pack,
        initial_chain,
        stage_a_anchors,
        stage_a_statements,
        &stage_a_deferred_vec,
        stacked_bank_commitments,
        &proof.layer_inactive_expert_openings,
        if has_stage_b {
            Some(stage_b_deferred_vec.as_slice())
        } else {
            None
        },
    );

    if is_zk_full {
        zk_full::verify_zk_full_block_continuity(
            &mut block_channel,
            k,
            has_stage_b,
            layer_inputs_override,
            stage_a_statements,
            r_cont_opt.as_deref().expect("is_zk_full draws r_cont"),
            proof,
        )?;
    } else if k >= 2 {
        let log_batch = if batch <= 1 {
            0
        } else {
            batch.next_power_of_two().ilog2() as usize
        };
        let log_dim = dim.next_power_of_two().ilog2() as usize;
        let r_cont_tok = block_channel.draw_secure_felts(log_batch);
        let r_cont_dim = block_channel.draw_secure_felts(log_dim);
        let rho_cont = block_channel.draw_secure_felt();
        let eq_tok = if log_batch == 0 {
            vec![SecureField::one()]
        } else {
            eq_evals(&r_cont_tok)
        };
        let eq_dim_i64 = eq_evals_i64(&r_cont_dim);

        let mut lhs_total = SecureField::zero();
        let mut rhs_total = SecureField::zero();
        let mut rho_pow = SecureField::one();
        for l in 0..(k - 1) {
            if has_stage_b {
                let expected_next = crate::stage_b::residual_out(
                    &stage_b_a_rows[l],
                    &stage_a_statements[l].output_q8,
                )?;
                if expected_next != stage_b_h_in_rows[l + 1] {
                    return Err(format!(
                        "Cross-layer residual continuity mismatch between layers {l} and {}",
                        l + 1
                    ));
                }
                lhs_total +=
                    rho_pow * eval_i16_matrix_mle_with_eq(&expected_next, &eq_tok, &eq_dim_i64);
                rhs_total += rho_pow
                    * eval_i16_matrix_mle_with_eq(&stage_b_h_in_rows[l + 1], &eq_tok, &eq_dim_i64);
            } else {
                lhs_total += rho_pow
                    * eval_i16_matrix_mle_with_eq(
                        &stage_a_statements[l].output_q8,
                        &eq_tok,
                        &eq_dim_i64,
                    );
                rhs_total += rho_pow
                    * eval_i16_matrix_mle_with_eq(
                        &layer_inputs_override.unwrap()[l + 1],
                        &eq_tok,
                        &eq_dim_i64,
                    );
            }
            rho_pow *= rho_cont;
        }
        if has_stage_b && lhs_total != rhs_total {
            return Err("Cross-layer randomized continuity identity failed".into());
        }
        block_channel.mix_felts(&[lhs_total, rhs_total]);
    }
    let mut layer_r_pack: Vec<Vec<SecureField>> =
        Vec::with_capacity(if proof.stacked_expert_pack > 0 { k } else { 0 });
    if proof.stacked_expert_pack > 0 {
        let log_pack = proof.stacked_expert_pack.ilog2() as usize;
        for _ in 0..k {
            let rp = if log_pack > 0 {
                block_channel.draw_secure_felts(log_pack)
            } else {
                Vec::new()
            };
            layer_r_pack.push(rp);
        }
    }

    let block_binding_seed = draw_cprs_binding_seed(&mut block_channel);
    let d_block_witness = cprs_cross_layer_outer_digest(
        b"stwo-ml:cross-layer:cprs-witness-outer:v1",
        &block_binding_seed,
        proof.mode_tag,
        k,
        has_stage_b,
        proof.cprs_weight_f,
        proof.cprs_witness_f,
        proof.stacked_expert_pack,
    );
    let d_block_weights = cprs_cross_layer_outer_digest(
        b"stwo-ml:cross-layer:cprs-weights-outer:v1",
        &block_binding_seed,
        proof.mode_tag,
        k,
        has_stage_b,
        proof.cprs_weight_f,
        proof.cprs_witness_f,
        proof.stacked_expert_pack,
    );

    let mut wit_commitments = Vec::with_capacity(2 * k);
    let mut wit_params_list = Vec::with_capacity(2 * k);
    let mut wit_points = Vec::new();
    let mut wit_claims = Vec::new();
    for l in 0..k {
        if has_stage_b {
            if let Some(ref w_b) = stage_b_deferred_vec[l].witness_opening {
                wit_commitments.push(w_b.witness_commitment);
                wit_params_list.push(w_b.witness_params.clone());
                let (b_pts, b_cls) = w_b.cprs_points_and_claims();
                wit_points.extend(b_pts);
                wit_claims.extend(b_cls);
            }
        }
        let ad = &stage_a_deferred_vec[l];
        wit_commitments.push(ad.witness_commitment);
        wit_params_list.push(ad.witness_params.clone());
        wit_points.extend_from_slice(&ad.wit_points);
        wit_claims.extend_from_slice(&ad.wit_claims);
    }
    verify_heterogeneous_strict_context(
        &wit_commitments,
        &wit_params_list,
        &wit_points,
        &wit_claims,
        &d_block_witness,
        &proof.block_cprs_witness_proof,
    )
    .map_err(|e| format!("Cross-layer block witness CPRS opening rejected: {e}"))?;

    let mut w_commitments = Vec::new();
    let mut w_params_list = Vec::new();
    let mut w_points = Vec::new();
    let mut w_claims = Vec::new();
    for l in 0..k {
        if has_stage_b {
            let bd = &stage_b_deferred_vec[l];
            w_commitments.push(bd.attention_commitment);
            w_params_list.push(bd.leaf_params.clone());
            let qkv_cprs = stwo_point_to_cprs(&bd.qkv_point);
            let o_cprs = stwo_point_to_cprs(&bd.o_point);
            w_points.push(qkv_cprs.clone());
            w_points.push(qkv_cprs.clone());
            w_points.push(qkv_cprs);
            w_points.push(o_cprs);
            for &c in &bd.leaf_claims {
                w_claims.push(to_cprs_qm31(c));
            }
            for op in &bd.chunk_openings {
                let chunk_commit = bd.chunk_commitments.get(op.chunk).copied().ok_or_else(|| {
                    "Stage B cache opening names an unknown chunk commitment".to_string()
                })?;
                w_commitments.push(chunk_commit);
                w_params_list.push(bd.chunk_params.clone());
                w_points.push(stwo_point_to_cprs(&op.k_point));
                w_points.push(stwo_point_to_cprs(&op.v_point));
                w_claims.push(to_cprs_qm31(op.k_value));
                w_claims.push(to_cprs_qm31(op.v_value));
            }
        }
        let ad = &stage_a_deferred_vec[l];
        w_commitments.push(ad.router_commitment);
        w_params_list.push(ad.router_params.clone());
        w_points.push(ad.r_router_cprs.clone());
        w_claims.push(ad.wg_opening);

        if proof.stacked_expert_pack > 0 {
            let pack = proof.stacked_expert_pack as usize;
            let shape_l = stage_a_statements[l].shape;
            let geom_l = StageARegisteredGeometry::for_shape(shape_l)?;
            let bank_prm = stacked_expert_bank_params(shape_l, pack, weight_f_ov, profile)?;
            let num_shards = geom_l.experts_padded / pack;
            w_commitments.push(stacked_bank_commitments.unwrap()[l]);
            w_params_list.push(bank_prm);

            let r_pack_l = &layer_r_pack[l];
            let eq_pack: Vec<crate::cprs::QM31> = if r_pack_l.is_empty() {
                vec![to_cprs_qm31(SecureField::one())]
            } else {
                eq_evals(r_pack_l).into_iter().map(to_cprs_qm31).collect()
            };
            let r_pack_cprs = stwo_point_to_cprs(r_pack_l);
            let mut pt_gu = ad.r_reg_gu_cprs.clone();
            pt_gu.extend_from_slice(&r_pack_cprs);
            let mut pt_down = ad.r_reg_down_cprs.clone();
            pt_down.extend_from_slice(&r_pack_cprs);

            let inact_slice = &proof.layer_inactive_expert_openings[l];
            let mut inact_cursor = 0usize;
            let mut full_claims = vec![[crate::cprs::QM31::ZERO; 3]; geom_l.experts_padded];
            for e in 0..shape_l.experts {
                if let Ok(a_idx) = ad.active_experts.binary_search(&e) {
                    full_claims[e] = [
                        ad.reg_claims[3 * a_idx],
                        ad.reg_claims[3 * a_idx + 1],
                        ad.reg_claims[3 * a_idx + 2],
                    ];
                } else {
                    let op = inact_slice[inact_cursor];
                    inact_cursor += 1;
                    full_claims[e] = [
                        to_cprs_qm31(op[0]),
                        to_cprs_qm31(op[1]),
                        to_cprs_qm31(op[2]),
                    ];
                }
            }
            debug_assert_eq!(inact_cursor, inact_slice.len());
            for s in 0..num_shards {
                let mut acc_g = crate::cprs::QM31::ZERO;
                let mut acc_u = crate::cprs::QM31::ZERO;
                let mut acc_d = crate::cprs::QM31::ZERO;
                for u in 0..pack {
                    let w = eq_pack[u];
                    let [eg, eu, ed] = full_claims[s * pack + u];
                    acc_g = acc_g.add(w.mul(eg));
                    acc_u = acc_u.add(w.mul(eu));
                    acc_d = acc_d.add(w.mul(ed));
                }
                w_points.push(pt_gu.clone());
                w_points.push(pt_gu.clone());
                w_points.push(pt_down.clone());
                w_claims.push(acc_g);
                w_claims.push(acc_u);
                w_claims.push(acc_d);
            }
        } else {
            for &rc in &ad.reg_commitments {
                w_commitments.push(rc);
                w_params_list.push(ad.reg_params.clone());
            }
            w_points.extend_from_slice(&ad.reg_points);
            w_claims.extend_from_slice(&ad.reg_claims);
        }

        w_commitments.push(ad.shared_commitment);
        w_params_list.push(ad.shared_params.clone());
        w_points.extend_from_slice(&ad.sh_points);
        w_claims.extend_from_slice(&ad.sh_claims);
    }
    verify_heterogeneous_strict_context(
        &w_commitments,
        &w_params_list,
        &w_points,
        &w_claims,
        &d_block_weights,
        &proof.block_cprs_weight_proof,
    )
    .map_err(|e| format!("Cross-layer block weight CPRS opening rejected: {e}"))?;

    Ok(CrossLayerGkrBlockVerifierRun { layers: k, batch })
}

pub fn cross_layer_gkr_block_bytes(proof: &CrossLayerGkrBlockProof) -> Result<Vec<u8>, String> {
    let k = proof.stage_a_proofs.len();
    if k == 0 {
        return Err("CrossLayerGkrBlockProof cannot encode 0 layers".into());
    }
    if !(1..=4).contains(&proof.cprs_weight_f) || !(1..=4).contains(&proof.cprs_witness_f) {
        return Err("CrossLayerGkrBlockProof CPRS arity f out of range 1..=4".into());
    }
    if proof.mode_tag != STAGE_A_RECEIPT_MODE_GKR_LANE
        && proof.mode_tag != STAGE_A_RECEIPT_MODE_ZK_GKR_LANE
        && proof.mode_tag != STAGE_A_RECEIPT_MODE_ZK_FULL
    {
        return Err(format!(
            "unsupported CrossLayerGkrBlockProof mode tag {}",
            proof.mode_tag
        ));
    }
    let has_stage_b = !proof.stage_b_proofs.is_empty();
    if has_stage_b && proof.stage_b_proofs.len() != k {
        return Err("CrossLayerGkrBlockProof stage_b_proofs length mismatch".into());
    }
    if proof.stacked_expert_pack > 0 {
        if !proof.stacked_expert_pack.is_power_of_two()
            || proof.layer_inactive_expert_openings.len() != k
        {
            return Err("CrossLayerGkrBlockProof invalid stacked_expert_pack metadata".into());
        }
    } else if !proof.layer_inactive_expert_openings.is_empty() {
        return Err("CrossLayerGkrBlockProof unstacked proof carries inactive openings".into());
    }
    let mut out = Vec::new();
    out.extend_from_slice(CROSS_LAYER_GKR_BLOCK_MAGIC);
    put_u32(&mut out, CROSS_LAYER_GKR_BLOCK_VERSION);
    put_u32(&mut out, proof.mode_tag);
    put_u32(&mut out, k as u32);
    put_u32(&mut out, u32::from(has_stage_b));
    out.push(proof.cprs_weight_f);
    out.push(proof.cprs_witness_f);
    put_u32(&mut out, proof.stacked_expert_pack);
    if has_stage_b {
        for bp in &proof.stage_b_proofs {
            if !bp.attention_leaf_proof.is_empty()
                || !bp.cache_chunk_proof.is_empty()
                || bp
                    .gkr_witness
                    .as_ref()
                    .is_some_and(|g| !g.cprs_witness_proof.is_empty())
            {
                return Err(
                    "CrossLayerGkrBlockProof Stage B proof carries non-empty per-layer CPRS bytes"
                        .into(),
                );
            }
            let bw = crate::stage_b::stage_b_receipt_bytes(bp)?;
            put_bytes(&mut out, &bw)?;
        }
    }
    for ap in &proof.stage_a_proofs {
        if !ap.cprs_router_proof.is_empty()
            || !ap.cprs_regular_proof.is_empty()
            || !ap.cprs_shared_proof.is_empty()
            || !ap.cprs_witness_proof.is_empty()
        {
            return Err(
                "CrossLayerGkrBlockProof Stage A proof carries non-empty per-layer CPRS bytes"
                    .into(),
            );
        }
        let aw = stage_a_gkr_lane_receipt_bytes_with_mode(ap, proof.mode_tag)?;
        put_bytes(&mut out, &aw)?;
    }
    if proof.stacked_expert_pack > 0 {
        for inact in &proof.layer_inactive_expert_openings {
            put_u32(&mut out, inact.len() as u32);
            for &[g, u, d] in inact {
                put_secure_field(&mut out, g);
                put_secure_field(&mut out, u);
                put_secure_field(&mut out, d);
            }
        }
    }
    put_bytes(&mut out, &proof.block_cprs_weight_proof)?;
    put_bytes(&mut out, &proof.block_cprs_witness_proof)?;
    Ok(out)
}

pub fn decode_cross_layer_gkr_block(
    bytes: &[u8],
    stage_a_statements: &[StageANonlinearBatchStatement],
    stage_a_policy: StageANumericPolicy,
    stage_b_statements: Option<&[crate::stage_b::StageBStatement]>,
) -> Result<CrossLayerGkrBlockProof, String> {
    let mut reader = ReceiptReader::new(bytes)?;
    reader.expect_raw(CROSS_LAYER_GKR_BLOCK_MAGIC, "cross-layer block magic")?;
    let version = reader.read_u32("cross-layer block version")?;
    if version != CROSS_LAYER_GKR_BLOCK_VERSION {
        return Err(format!("unsupported cross-layer block version {version}"));
    }
    let mode_tag = reader.read_u32("cross-layer block mode tag")?;
    if mode_tag != STAGE_A_RECEIPT_MODE_GKR_LANE
        && mode_tag != STAGE_A_RECEIPT_MODE_ZK_GKR_LANE
        && mode_tag != STAGE_A_RECEIPT_MODE_ZK_FULL
    {
        return Err(format!(
            "unsupported cross-layer block mode tag {mode_tag}"
        ));
    }
    let k = reader.read_u32("cross-layer block layer count")? as usize;
    let has_b_flag = reader.read_u32("cross-layer block has_stage_b flag")?;
    if has_b_flag > 1 {
        return Err("non-boolean has_stage_b flag in cross-layer block".into());
    }
    let cprs_weight_f = reader.read_u32_byte("cross-layer block cprs_weight_f")?;
    let cprs_witness_f = reader.read_u32_byte("cross-layer block cprs_witness_f")?;
    if !(1..=4).contains(&cprs_weight_f) || !(1..=4).contains(&cprs_witness_f) {
        return Err("cross-layer block CPRS arity f out of range 1..=4".into());
    }
    let stacked_expert_pack = reader.read_u32("cross-layer block stacked_expert_pack")?;
    if stacked_expert_pack > 0 && !stacked_expert_pack.is_power_of_two() {
        return Err("cross-layer block stacked_expert_pack is not a power of two".into());
    }
    let has_stage_b = has_b_flag == 1;
    if k == 0 || stage_a_statements.len() != k {
        return Err("cross-layer block layer count does not match Stage A statements".into());
    }
    if has_stage_b != stage_b_statements.is_some() {
        return Err("cross-layer block has_stage_b flag does not match verifier statements".into());
    }
    let mut stage_b_proofs = Vec::with_capacity(if has_stage_b { k } else { 0 });
    if let Some(b_stmts) = stage_b_statements {
        if b_stmts.len() != k {
            return Err("cross-layer block Stage B statement count mismatch".into());
        }
        for b_stmt in b_stmts {
            let bw = reader.read_bytes(MAX_WIRE_RECEIPT_BYTES, "stage_b wire")?;
            let bp = crate::stage_b::decode_stage_b_receipt_bytes(bw, b_stmt)?;
            stage_b_proofs.push(bp);
        }
    }
    let mut stage_a_proofs = Vec::with_capacity(k);
    for a_stmt in stage_a_statements {
        let aw = reader.read_bytes(MAX_WIRE_RECEIPT_BYTES, "stage_a wire")?;
        let ap = decode_stage_a_gkr_lane_receipt_with_mode(aw, a_stmt, stage_a_policy, mode_tag)?;
        stage_a_proofs.push(ap);
    }
    let mut layer_inactive_expert_openings =
        Vec::with_capacity(if stacked_expert_pack > 0 { k } else { 0 });
    if stacked_expert_pack > 0 {
        for a_stmt in stage_a_statements {
            let count = reader.read_u32("layer inactive expert count")? as usize;
            if count > a_stmt.shape.experts {
                return Err("layer inactive expert count exceeds shape.experts".into());
            }
            let mut inact = Vec::with_capacity(count);
            for _ in 0..count {
                let g = reader.read_secure_field("inactive gate opening")?;
                let u = reader.read_secure_field("inactive up opening")?;
                let d = reader.read_secure_field("inactive down opening")?;
                inact.push([g, u, d]);
            }
            layer_inactive_expert_openings.push(inact);
        }
    }
    let block_cprs_weight_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "block weight CPRS")?
        .to_vec();
    let block_cprs_witness_proof = reader
        .read_bytes(MAX_WIRE_RECEIPT_BYTES, "block witness CPRS")?
        .to_vec();
    reader.finish()?;

    let proof = CrossLayerGkrBlockProof {
        mode_tag,
        cprs_weight_f,
        cprs_witness_f,
        stacked_expert_pack,
        stage_b_proofs,
        stage_a_proofs,
        layer_inactive_expert_openings,
        block_cprs_weight_proof,
        block_cprs_witness_proof,
    };
    if cross_layer_gkr_block_bytes(&proof)? != bytes {
        return Err("non-canonical CrossLayerGkrBlockProof encoding".into());
    }
    Ok(proof)
}

#[path = "stage_a_zk_full.rs"]
pub mod zk_full;
