//! Ummon browser verifier: a thin wrapper over `ummon-verifier` that checks a multi-token `STWOMDL2` bundle held
//! in memory, for `wasm32-unknown-unknown`.
//!
//! It runs the steps of `ummon-verify model` on a multi-token bundle (`verify_standalone_model_gen_command` in
//! `verifier/src/lib.rs`) in the same order: the caller's model anchor, initial link, prompt tokens and generated
//! tokens are compared with the manifest before any receipt is decoded; the receipts are size-capped, decoded and
//! held to the caller's profile; single-token end stages are refused; then
//! `verify_standalone_model_gen_chain_against_anchor` verifies the caller's statement. The only difference is where
//! the receipts come from: the files the caller hands over, looked up by the names the manifest gives, instead of a
//! directory. No check of the verifier is changed or skipped.

use ummon_verifier::{
    compute_standalone_model_anchor, decode_standalone_block_receipt, decode_standalone_model_gen_manifest, hex32,
    parse_chain_link_spec, parse_hex32, verify_standalone_model_gen_chain_against_anchor,
    StandaloneBlockStarkProof, StandaloneChainLink, StandaloneChainManifestEntry, StandaloneChainedBlock,
    StandaloneModelGenManifest, MAX_CHAIN_TOTAL_RECEIPT_BYTES, MAX_WIRE_RECEIPT_BYTES, STANDALONE_MODEL_MANIFEST_MAGIC,
    STANDALONE_PROFILE_IDS,
};

/// The statement a bundle is checked against. Every field comes from the caller, never from the bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub model_anchor: [u8; 32],
    pub initial_chain: StandaloneChainLink,
    pub prompt_tokens: Vec<u32>,
    pub gen_tokens: Vec<u32>,
    pub profile: String,
}

/// One file the caller hands over: its name (no directory) and its bytes.
#[derive(Clone, Copy, Debug)]
pub struct BundleFile<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
}

/// What an accepted bundle establishes, with the statement it was checked against.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub statement: Statement,
    pub final_chain: StandaloneChainLink,
    pub receipts: usize,
    pub layers: usize,
    pub receipt_bytes: usize,
}

const STATEMENT_KEYS: [&str; 5] = ["model_anchor", "initial_chain", "tokens", "gen_tokens", "profile"];

/// Parse a statement written as five `key value` lines, each value in the form of the `model` command's flag:
/// `model_anchor <hex32>`, `initial_chain <layer:hex32>`, `tokens <csv>` (the prompt), `gen_tokens <csv>`,
/// `profile <id>`. Each key exactly once; blank lines are ignored. Unlike the command, the profile is required.
pub fn parse_statement(text: &str) -> Result<Statement, String> {
    let mut values: [Option<&str>; 5] = [None; 5];
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (key, value) = line
            .split_once(char::is_whitespace)
            .ok_or_else(|| format!("statement line '{line}' has no value"))?;
        let i = STATEMENT_KEYS
            .iter()
            .position(|k| *k == key)
            .ok_or_else(|| format!("unknown statement key '{key}'"))?;
        if values[i].replace(value.trim()).is_some() {
            return Err(format!("statement key '{key}' given twice"));
        }
    }
    let get = |i: usize| values[i].ok_or_else(|| format!("statement key '{}' missing", STATEMENT_KEYS[i]));
    let (layer, prev_receipt_digest) = parse_chain_link_spec(get(1)?)?;
    let profile = get(4)?.to_string();
    if !STANDALONE_PROFILE_IDS.contains(&profile.as_str()) {
        return Err(format!("profile '{profile}' is not a profile id of STWOREC2"));
    }
    Ok(Statement {
        model_anchor: parse_hex32(get(0)?)?,
        initial_chain: StandaloneChainLink { layer, prev_receipt_digest },
        prompt_tokens: parse_tokens(get(2)?, "tokens")?,
        gen_tokens: parse_tokens(get(3)?, "gen_tokens")?,
        profile,
    })
}

fn parse_tokens(csv: &str, key: &str) -> Result<Vec<u32>, String> {
    csv.split(',')
        .map(|x| x.trim().parse::<u32>().map_err(|_| format!("invalid {key} entry '{x}'")))
        .collect()
}

/// Split the files the caller packed as repeated `[u32 LE name length][name][u32 LE byte length][bytes]`.
pub fn split_files(mut buf: &[u8]) -> Result<Vec<BundleFile<'_>>, String> {
    let mut files = Vec::new();
    while !buf.is_empty() {
        let name = take_field(&mut buf, "file name")?;
        let bytes = take_field(&mut buf, "file bytes")?;
        let name = std::str::from_utf8(name).map_err(|_| "a file name is not UTF-8".to_string())?;
        files.push(BundleFile { name, bytes });
    }
    Ok(files)
}

fn take_field<'a>(buf: &mut &'a [u8], what: &str) -> Result<&'a [u8], String> {
    let (len, rest) = buf.split_first_chunk::<4>().ok_or_else(|| format!("truncated {what} length"))?;
    let len = usize::try_from(u32::from_le_bytes(*len)).map_err(|_| format!("{what} length does not fit"))?;
    if rest.len() < len {
        return Err(format!("truncated {what}"));
    }
    let (field, tail) = rest.split_at(len);
    *buf = tail;
    Ok(field)
}

/// Parse the statement and the packed files, then verify the bundle: the one call the wasm export makes.
pub fn verify_request(statement: &[u8], packed_files: &[u8]) -> Result<Verdict, String> {
    let text = std::str::from_utf8(statement).map_err(|_| "the statement is not UTF-8".to_string())?;
    let statement = parse_statement(text)?;
    verify_bundle(&statement, &split_files(packed_files)?)
}

/// Verify a multi-token `STWOMDL2` bundle against the caller's statement, in the order of the `model` command.
pub fn verify_bundle(st: &Statement, files: &[BundleFile<'_>]) -> Result<Verdict, String> {
    let manifest = decode_standalone_model_gen_manifest(find_manifest(files)?)?;
    check_statement_against_manifest(st, &manifest)?;
    let (proofs, receipt_bytes) = decode_receipts(&manifest.receipts, files)?;
    if let Some((k, proof)) = proofs.iter().enumerate().find(|(_, p)| p.statement.profile_id != st.profile) {
        return Err(format!(
            "receipt {k} is proved under profile '{}', not the profile '{}' the caller requires",
            proof.statement.profile_id, st.profile
        ));
    }
    if let Some(k) = proofs.iter().position(|p| p.statement.ends.is_some_and(|e| !e.multi_token)) {
        return Err(format!("receipt {k} has a single-token end stage; this checks multi-token bundles only"));
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
    let v = verify_standalone_model_gen_chain_against_anchor(
        &st.model_anchor,
        &st.initial_chain,
        &chained,
        &st.prompt_tokens,
        &st.gen_tokens,
    )?;
    Ok(Verdict {
        statement: st.clone(),
        final_chain: v.final_chain,
        receipts: proofs.len(),
        layers: proofs.iter().map(|p| p.statement.num_layers).sum(),
        receipt_bytes,
    })
}

/// The one file that starts with the `STWOMDL2` magic.
fn find_manifest<'a>(files: &[BundleFile<'a>]) -> Result<&'a [u8], String> {
    let mut manifests = files.iter().filter(|f| f.bytes.starts_with(STANDALONE_MODEL_MANIFEST_MAGIC));
    let manifest = manifests.next().ok_or("no STWOMDL2 manifest among the files")?;
    if manifests.next().is_some() {
        return Err("more than one STWOMDL2 manifest among the files".into());
    }
    Ok(manifest.bytes)
}

/// The checks the `model` command makes before it reads a receipt: the manifest must carry the caller's statement.
fn check_statement_against_manifest(st: &Statement, manifest: &StandaloneModelGenManifest) -> Result<(), String> {
    if st.model_anchor == [0u8; 32] {
        return Err("expected_model_anchor must not be all-zero".into());
    }
    let anchors: Vec<[u8; 32]> = manifest.receipts.iter().map(|r| r.standalone_anchor).collect();
    if compute_standalone_model_anchor(&anchors) != st.model_anchor {
        return Err("standalone model: 32B model anchor mismatch (a receipt anchor, the receipt count or the order differs from registration)".into());
    }
    if manifest.initial_chain != st.initial_chain {
        return Err("manifest initial_chain disagrees with the statement's initial_chain".into());
    }
    if manifest.prompt_tokens != st.prompt_tokens {
        return Err("manifest tokens disagree with the statement's tokens".into());
    }
    if manifest.gen_tokens != st.gen_tokens {
        return Err("manifest gen_tokens disagree with the statement's gen_tokens".into());
    }
    Ok(())
}

/// Find each receipt the manifest names among the files (exactly one file of that name), cap its size and the
/// total as the command does, and decode it.
fn decode_receipts(
    entries: &[StandaloneChainManifestEntry],
    files: &[BundleFile<'_>],
) -> Result<(Vec<StandaloneBlockStarkProof>, usize), String> {
    let mut proofs = Vec::with_capacity(entries.len());
    let mut total = 0usize;
    for (i, entry) in entries.iter().enumerate() {
        if entries[..i].iter().any(|e| e.receipt_path == entry.receipt_path) {
            return Err(format!("duplicate receipt_path '{}' in manifest", entry.receipt_path));
        }
        let mut named = files.iter().filter(|f| f.name == entry.receipt_path);
        let file = named
            .next()
            .ok_or_else(|| format!("receipt {} '{}' is not among the files", entry.block_index, entry.receipt_path))?;
        if named.next().is_some() {
            return Err(format!("more than one file is named '{}'", entry.receipt_path));
        }
        if file.bytes.len() > MAX_WIRE_RECEIPT_BYTES {
            return Err(format!("receipt {} exceeds {MAX_WIRE_RECEIPT_BYTES} bytes", entry.block_index));
        }
        total = match total.checked_add(file.bytes.len()) {
            Some(t) if t <= MAX_CHAIN_TOTAL_RECEIPT_BYTES => t,
            _ => return Err(format!("cumulative receipt size exceeds {MAX_CHAIN_TOTAL_RECEIPT_BYTES} bytes")),
        };
        let proof = decode_standalone_block_receipt(file.bytes)
            .map_err(|r| format!("receipt {} decode failed: {r}", entry.block_index))?;
        proofs.push(proof);
    }
    Ok((proofs, total))
}

fn csv(tokens: &[u32]) -> String {
    tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

/// The report the page shows: `key value` lines, `result accept` or `result reject` first.
pub fn report_text(result: &Result<Verdict, String>) -> String {
    match result {
        Err(reason) => format!("result reject\nreason {reason}\n"),
        Ok(v) => {
            let st = &v.statement;
            format!(
                "result accept\nmodel_anchor {}\ninitial_chain {}:{}\ntokens {}\ngen_tokens {}\nprofile {}\n\
                 final_chain {}:{}\nreceipts {}\nlayers {}\nreceipt_bytes {}\n",
                hex32(&st.model_anchor),
                st.initial_chain.layer,
                hex32(&st.initial_chain.prev_receipt_digest),
                csv(&st.prompt_tokens),
                csv(&st.gen_tokens),
                st.profile,
                v.final_chain.layer,
                hex32(&v.final_chain.prev_receipt_digest),
                v.receipts,
                v.layers,
                v.receipt_bytes,
            )
        }
    }
}

/// C-ABI exports for a page: `ummon_alloc` / `ummon_free` for input buffers, `ummon_verify`, and the report of the
/// last call at `ummon_result_ptr` / `ummon_result_len`.
#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use std::cell::RefCell;

    thread_local! {
        static REPORT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    /// A zeroed buffer of `len` bytes for the caller to fill; release it with `ummon_free(ptr, len)`.
    #[no_mangle]
    pub extern "C" fn ummon_alloc(len: usize) -> *mut u8 {
        Box::into_raw(vec![0u8; len].into_boxed_slice()).cast::<u8>()
    }

    /// # Safety
    /// `ptr` and `len` must come from one `ummon_alloc(len)` call, released once.
    #[no_mangle]
    pub unsafe extern "C" fn ummon_free(ptr: *mut u8, len: usize) {
        if !ptr.is_null() {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
        }
    }

    /// Verify the packed files against the statement: `1` accept, `0` reject. The report is then readable at
    /// `ummon_result_ptr()` for `ummon_result_len()` bytes until the next call.
    ///
    /// # Safety
    /// Both buffers must come from `ummon_alloc` with these lengths.
    #[no_mangle]
    pub unsafe extern "C" fn ummon_verify(
        statement_ptr: *const u8,
        statement_len: usize,
        files_ptr: *const u8,
        files_len: usize,
    ) -> i32 {
        let statement = std::slice::from_raw_parts(statement_ptr, statement_len);
        let files = std::slice::from_raw_parts(files_ptr, files_len);
        let result = super::verify_request(statement, files);
        REPORT.with(|r| *r.borrow_mut() = super::report_text(&result).into_bytes());
        i32::from(result.is_ok())
    }

    #[no_mangle]
    pub extern "C" fn ummon_result_ptr() -> *const u8 {
        REPORT.with(|r| r.borrow().as_ptr())
    }

    #[no_mangle]
    pub extern "C" fn ummon_result_len() -> usize {
        REPORT.with(|r| r.borrow().len())
    }
}
