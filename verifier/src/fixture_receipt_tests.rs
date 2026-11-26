//! The real receipt of `tests/data/multi_token_model/` (embedding, one layer and the head of a toy model, `(L, K) =
//! (2, 2)`, mode 9, `dev13`) verified through the library, and every outer constraint removed one at a time on the
//! verifier's side: each removal must make this honest receipt fail. This detects a removal (the composition's random
//! combination changes whatever the constraint does); it is not a soundness test. The paired tests of the prover (not released)
//! show that each constraint is the one rejecting its forgery. The native checks are
//! tied to their own rejection reasons below, so that removing one changes the reason a tampered fixture is refused.

use crate::blake2s_air::{outer_ids, outer_skip};
use crate::standalone_audit::{
    decode_standalone_block_receipt, verify_standalone_block_stark_against_anchor, StandaloneBlockStarkProof,
    StandaloneChainLink,
};
use crate::{decode_standalone_model_gen_manifest, verify_standalone_model_command, StandaloneModelCommandArgs};
use num_traits::One;
use stwo::core::fields::qm31::SecureField;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/multi_token_model")
}

/// `(model_anchor, initial_chain, tokens, gen_tokens)` of `statement.txt`.
fn statement() -> [String; 4] {
    let text = std::fs::read_to_string(fixture().join("statement.txt")).expect("statement");
    ["model_anchor", "initial_chain", "tokens", "gen_tokens"].map(|name| {
        text.lines().find_map(|l| l.strip_prefix(name).and_then(|r| r.strip_prefix(' '))).expect(name).to_string()
    })
}

/// Verify the fixture bundle (receipts read from `dir`) against the recorded statement.
fn verify_in(dir: &Path) -> Result<(), String> {
    let manifest = std::fs::read(fixture().join("model_manifest.mdl2")).expect("manifest");
    verify_manifest(&manifest, dir)
}

/// Verify `manifest` (receipts read from `dir`) against the recorded statement, with the caller's tokens taken from
/// the manifest.
fn verify_manifest(manifest: &[u8], dir: &Path) -> Result<(), String> {
    let [anchor, init, _, _] = statement();
    let m = decode_standalone_model_gen_manifest(manifest)?;
    let join = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let (tokens, gen_toks) = (join(&m.prompt_tokens), join(&m.gen_tokens));
    let args = StandaloneModelCommandArgs {
        model_anchor: Some(&anchor),
        initial_chain: Some(&init),
        tokens: Some(&tokens),
        out_token: None,
        gen_tokens: Some(&gen_toks),
        profile: Some("dev13"),
        bench_runs: 0,
    };
    verify_standalone_model_command(manifest, dir, &args).map(|_| ())
}

/// Every AIR constraint id of the outer circuit (`A01`..`A21`, `B01`..`B15`, `C01`..`C07`).
fn air_ids() -> Vec<(&'static str, u32)> {
    use outer_ids::*;
    vec![
        ("A01", A01), ("A02", A02), ("A03", A03), ("A04", A04), ("A05", A05), ("A06", A06), ("A07", A07),
        ("A08", A08), ("A09", A09), ("A10", A10), ("A11", A11), ("A12", A12), ("A13", A13), ("A14", A14),
        ("A15", A15), ("A16", A16), ("A17", A17), ("A18", A18), ("A19", A19), ("A20", A20), ("A21", A21),
        ("B01", B01), ("B02", B02), ("B03", B03), ("B04", B04), ("B05", B05), ("B06", B06), ("B07", B07),
        ("B08", B08), ("B09", B09), ("B10", B10), ("B11", B11), ("B12", B12), ("B13", B13), ("B14", B14),
        ("B15", B15), ("C01", C01), ("C02", C02), ("C03", C03), ("C04", C04), ("C05", C05), ("C06", C06),
        ("C07", C07),
    ]
}

#[test]
fn the_real_receipt_verifies_and_each_constraint_removal_breaks_it() {
    verify_in(&fixture()).expect("the honest fixture must verify");
    let ids = air_ids();
    assert_eq!(ids.len(), 43);
    for (name, cid) in ids {
        let _off = outer_skip::scope(&[cid]);
        let err = verify_in(&fixture()).expect_err(name);
        assert!(err.contains("composition polynomial OODS value does not match"), "{name}: {err}");
    }
    verify_in(&fixture()).expect("the mask is restored");
}

/// The block receipt of the fixture, its registered anchor and the initial link, from the manifest.
fn block() -> (StandaloneBlockStarkProof, [u8; 32], StandaloneChainLink) {
    let m = decode_standalone_model_gen_manifest(&std::fs::read(fixture().join("model_manifest.mdl2")).expect("manifest"))
        .expect("decode manifest");
    let proof = decode_standalone_block_receipt(&std::fs::read(fixture().join("receipt_00.rec2")).expect("receipt"))
        .expect("decode receipt");
    (proof, m.receipts[0].standalone_anchor, m.initial_chain)
}

/// Each native check names its own rejection of a tampered fixture, so a removal of the check changes the reason
/// (`D16` the global LogUp sum, `D17` the block anchor, the zero lookup nonce of a profile without lookup work, the
/// multi-token vocabulary bound). `D15`'s work branch, `D18` and `D19` (single token) need receipts this crate does not
/// hold; their paired tests are in the prover, which is not released.
#[test]
fn the_real_receipt_names_the_native_check_that_rejects_each_tamper() {
    let (proof, anchor, init) = block();
    verify_standalone_block_stark_against_anchor(&anchor, &init, &proof).expect("the honest block receipt");
    let reject = |p: &StandaloneBlockStarkProof, a: &[u8; 32], want: &str| {
        let err = verify_standalone_block_stark_against_anchor(a, &init, p).expect_err(want);
        assert!(err.contains(want), "want {want:?}: {err}");
    };
    let mut sums = proof.clone();
    sums.claimed_sums.audit_sum += SecureField::one();
    reject(&sums, &anchor, "LogUp sum non-zero");
    let mut other = anchor;
    other[0] ^= 1;
    reject(&proof, &other, "32B anchor mismatch");
    let mut nonce = proof.clone();
    nonce.lookup_pow_nonce = 1;
    reject(&nonce, &anchor, "lookup nonce must be zero");
    let text = std::fs::read_to_string(fixture().join("model_manifest.mdl2")).expect("manifest");
    let oov = text.replacen("tokens 2,11\n", "tokens 2,13\n", 1);
    assert_ne!(oov, text);
    let err = verify_manifest(oov.as_bytes(), &fixture()).expect_err("a prompt token outside the vocabulary");
    assert!(err.contains("prompt token 13 >= vocab"), "{err}");
}

#[test]
fn the_real_receipt_rejects_flipped_bytes() {
    let receipt = std::fs::read(fixture().join("receipt_00.rec2")).expect("receipt");
    let dir = std::env::temp_dir().join(format!("stwo_ml_fixture_flip_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    // A byte of a length prefix (refused when decoding) and two bytes of proof data (refused by the STARK
    // verification), each with its reason; the second data byte is the first one after three quarters of the receipt
    // whose flip still decodes.
    let flipped = |at: usize| {
        let mut bad = receipt.clone();
        bad[at] ^= 0x01;
        bad
    };
    let late = (3 * receipt.len() / 4..receipt.len())
        .find(|&at| decode_standalone_block_receipt(&flipped(at)).is_ok())
        .expect("a data byte");
    let stark = "standalone model: receipt 0: standalone STARK verification rejected";
    let cases = [(receipt.len() / 4, "receipt 0 decode failed"), (receipt.len() / 2, stark), (late, stark)];
    for (at, want) in cases {
        std::fs::write(dir.join("receipt_00.rec2"), flipped(at)).expect("write");
        let err = verify_in(&dir).expect_err("a flipped byte must be rejected");
        assert!(err.contains(want), "a flipped byte at {at}: want {want:?}, got {err}");
    }
    std::fs::write(dir.join("receipt_00.rec2"), &receipt).expect("write");
    verify_in(&dir).expect("the unflipped copy verifies");
    std::fs::remove_dir_all(&dir).expect("clean up");
}
