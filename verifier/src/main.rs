use std::collections::HashSet;
use sha2::{Digest, Sha256};
use ummon_verifier::{
    decode_mla_verification_bundle, decode_spot_check_verification_bundle,
    decode_swa_verification_bundle, hex32, parse_chain_link_spec, parse_hex32,
    read_capped_regular, require_standalone_receipt_profile, run_timed_verifies, verify_cross_layer_block_bundle_bytes,
    verify_mla_verification_bundle, verify_recursive_audit_receipt_bytes,
    verify_spot_check_verification_bundle, verify_standalone_block_receipt_bytes_with_io,
    verify_standalone_chain_command, verify_standalone_model_command, verify_swa_verification_bundle,
    verify_whole_model_committed_receipt_bytes, verify_whole_model_receipt_bytes, StageAChainLink,
    StandaloneChainCommandArgs, StandaloneChainLink, StandaloneModelCommandArgs, CROSS_LAYER_BLOCK_BUNDLE_MAGIC,
    compute_cross_layer_block_anchors_digest,
    MAX_WIRE_RECEIPT_BYTES, MLA_BUNDLE_MAGIC, RECURSIVE_AUDIT_MAGIC, STANDALONE_BLOCK_MAGIC,
    STANDALONE_CHAIN_MANIFEST_MAGIC, STANDALONE_MODEL_MANIFEST_MAGIC, SWA_BUNDLE_MAGIC,
    WHOLE_MODEL_MAGIC,
};

const VALUE_FLAGS: &[&str] = &[
    "--anchor",
    "--chain-anchor",
    "--wiring-root",
    "--initial-chain",
    "--final-chain",
    "--public-io",
    "--prompt-tokens",
    "--emitted-tokens",
    "--prompt-commitment",
    "--emitted-commitment",
    "--model-anchor",
    "--tokens",
    "--out-token",
    "--gen-tokens",
    "--profile",
    "--bench-runs",
];

/// The environment variable each value flag falls back to (`flag_value`). A variable set for a flag the
/// command does not check is refused, never ignored.
const FLAG_ENV: &[(&str, &str)] = &[
    ("--anchor", "STWO_ML_ANCHOR"),
    ("--chain-anchor", "STWO_ML_CHAIN_ANCHOR"),
    ("--wiring-root", "STWO_ML_WIRING_ROOT"),
    ("--initial-chain", "STWO_ML_INITIAL_CHAIN"),
    ("--final-chain", "STWO_ML_FINAL_CHAIN"),
    ("--public-io", "STWO_ML_PUBLIC_IO"),
    ("--prompt-tokens", "STWO_ML_PROMPT_TOKENS"),
    ("--emitted-tokens", "STWO_ML_EMITTED_TOKENS"),
    ("--prompt-commitment", "STWO_ML_PROMPT_COMMITMENT"),
    ("--emitted-commitment", "STWO_ML_EMITTED_COMMITMENT"),
    ("--model-anchor", "STWO_ML_MODEL_ANCHOR"),
    ("--tokens", "STWO_ML_TOKENS"),
    ("--out-token", "STWO_ML_OUT_TOKEN"),
    ("--gen-tokens", "STWO_ML_GEN_TOKENS"),
    ("--profile", "STWO_ML_PROFILE"),
    ("--bench-runs", "STWO_ML_VERIFY_BENCH_RUNS"),
];

#[inline]
fn build_git_sha() -> &'static str {
    env!("UMMON_BUILD_GIT_SHA")
}

#[inline]
fn build_git_dirty() -> bool {
    matches!(env!("UMMON_BUILD_GIT_DIRTY"), "true" | "1")
}

/// Format an arbitrary UTF-8 string as an RFC 8259 JSON string literal (`"..."`), escaping
/// control characters `< 0x20` as `\u00XX` rather than Rust's `{:?}` `\u{...}`.
fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Real OS-kernel process peak resident set size (`MiB`) across the lifetime of this standalone
/// verifier subprocess (`VmHWM` from `/proc/self/status` on Linux; `getrusage(RUSAGE_SELF)` on macOS).
fn process_peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for key in ["VmHWM:", "VmRSS:"] {
                if let Some(line) = status.lines().find(|l| l.starts_with(key)) {
                    if let Some(kib_str) = line[key.len()..].split_whitespace().next() {
                        if let Ok(kib) = kib_str.parse::<u64>() {
                            return (kib as f64) / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct Rusage {
            ru_utime: [i64; 2],
            ru_stime: [i64; 2],
            ru_maxrss: i64,
            _rest: [i64; 13],
        }
        extern "C" {
            fn getrusage(who: i32, r_usage: *mut Rusage) -> i32;
        }
        let mut usage = Rusage {
            ru_utime: [0; 2],
            ru_stime: [0; 2],
            ru_maxrss: 0,
            _rest: [0; 13],
        };
        let rc = unsafe { getrusage(0, &mut usage as *mut Rusage) };
        if rc == 0 && usage.ru_maxrss > 0 {
            (usage.ru_maxrss as f64) / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0.0
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    hex32(&digest)
}

fn validate_cli_flags(args: &[String]) -> Result<(), String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--json" || a == "--help" || a == "-h" {
            i += 1;
            continue;
        }
        if let Some((k, v)) = a.split_once('=') {
            if VALUE_FLAGS.contains(&k) {
                if v.is_empty() {
                    return Err(format!("CLI flag '{k}' requires a non-empty value"));
                }
                if !seen.insert(k.to_string()) {
                    return Err(format!("duplicate CLI flag '{k}'"));
                }
                i += 1;
                continue;
            }
        }
        if VALUE_FLAGS.contains(&a) {
            if !seen.insert(a.to_string()) {
                return Err(format!("duplicate CLI flag '{a}'"));
            }
            let next = args
                .get(i + 1)
                .ok_or_else(|| format!("CLI flag '{a}' missing required value"))?;
            if next.starts_with("--") {
                return Err(format!("CLI flag '{a}' missing value before '{next}'"));
            }
            i += 2;
            continue;
        }
        if a.starts_with('-') {
            return Err(format!("unknown CLI flag '{a}'"));
        }
        i += 1;
    }
    Ok(())
}

fn ensure_only_allowed_cli_flags(
    args: &[String],
    format_name: &str,
    allowed_value_flags: &[&str],
) -> Result<(), String> {
    for (flag, var) in FLAG_ENV {
        let set = std::env::var(var).map(|v| !v.trim().is_empty()).unwrap_or(false);
        if set && !allowed_value_flags.contains(flag) {
            return Err(format!(
                "environment variable {var} sets '{flag}', which {format_name} does not verify; unset it"
            ));
        }
    }
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--json" || a == "--help" || a == "-h" {
            i += 1;
            continue;
        }
        if let Some((k, _)) = a.split_once('=') {
            if VALUE_FLAGS.contains(&k) {
                if !allowed_value_flags.contains(&k) {
                    return Err(format!(
                        "CLI flag '{k}' is not verified by {format_name}; pass the matching subcommand or remove '{k}'"
                    ));
                }
                i += 1;
                continue;
            }
        }
        if VALUE_FLAGS.contains(&a) {
            if !allowed_value_flags.contains(&a) {
                return Err(format!(
                    "CLI flag '{a}' is not verified by {format_name}; pass the matching subcommand or remove '{a}'"
                ));
            }
            i += 2;
            continue;
        }
        i += 1;
    }
    Ok(())
}

fn flag_value(args: &[String], name: &str, env_key: &str) -> Option<String> {
    let eq_prefix = format!("{name}=");
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        if a == name {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = a.strip_prefix(&eq_prefix) {
            return Some(rest.to_string());
        }
        if VALUE_FLAGS.contains(&a) {
            i += 2;
        } else {
            i += 1;
        }
    }
    std::env::var(env_key).ok().filter(|v| !v.trim().is_empty())
}

fn parse_bench_runs(args: &[String]) -> Result<usize, String> {
    match flag_value(args, "--bench-runs", "STWO_ML_VERIFY_BENCH_RUNS") {
        Some(s) => {
            let n = s
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid --bench-runs integer '{s}'"))?;
            if n == 0 {
                return Err("--bench-runs must be >= 1".into());
            }
            Ok(n.min(101))
        }
        None => Ok(1),
    }
}

fn parse_u32_csv(raw: &str) -> Result<Vec<u32>, String> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }
    raw.split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| format!("invalid u32 token '{part}'"))
        })
        .collect()
}

fn positional_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if VALUE_FLAGS.contains(&a.as_str()) {
            skip_next = true;
            continue;
        }
        if !a.starts_with('-') {
            out.push(a.clone());
        }
    }
    out
}

fn expected_magic_for_subcommand(sub: &str) -> Option<&'static [u8; 8]> {
    match sub {
        "mdl1" | ".mdl1" => Some(WHOLE_MODEL_MAGIC),
        "rec2" | ".rec2" => Some(STANDALONE_BLOCK_MAGIC),
        "chn2" | ".chn2" | "chain" => Some(STANDALONE_CHAIN_MANIFEST_MAGIC),
        "mdl2" | ".mdl2" | "model" => Some(STANDALONE_MODEL_MANIFEST_MAGIC),
        "rec1" | ".rec1" => Some(RECURSIVE_AUDIT_MAGIC),
        "blk1" | ".blk1" | "blkb" | ".blkb" => Some(CROSS_LAYER_BLOCK_BUNDLE_MAGIC),
        "mlab" | ".mlab" | "mla1" | ".mla1" => Some(MLA_BUNDLE_MAGIC),
        "swab" | ".swab" | "swa1" | ".swa1" => Some(SWA_BUNDLE_MAGIC),
        "bcn4" | ".bcn4" => Some(b"STWOBCN4"),
        _ => None,
    }
}

const USAGE: &str = "usage: ummon-verify [mdl1|rec2|chn2|chain|model|rec1|blk1|mlab|swab|bcn4] <receipt_or_bundle> [--anchor <hex32>] [--chain-anchor <hex32>] [--model-anchor <hex32>] [--tokens <csv>] [--out-token <u32>] [--gen-tokens <csv>] [--profile <id>] [--wiring-root <hex32>] [--initial-chain <layer:hex32>] [--final-chain <layer:hex32>] [--public-io <hex32>] [--prompt-tokens <csv>] [--emitted-tokens <csv>] [--prompt-commitment <hex32>] [--emitted-commitment <hex32>] [--bench-runs <N>] [--json]\n  --profile checks the profile id of every receipt; the inner parameters (CPRS query count, query and fold\n  grinding bits) are fixed by the anchor, not by the profile id (8 and 16 bits of fold grinding both carry\n  production_hardened).";

fn emit_reject(event: &str, path: &str, len: usize, sha: &str, reason: &str) -> ! {
    let rss_mb = process_peak_rss_mb();
    println!(
        "{{\"event\":{},\"git_sha\":{},\"git_dirty\":{},\"file\":{},\"bytes\":{},\"sha256\":{},\"accepted\":false,\"verifier_peak_rss_mb\":{:.3},\"reason\":{}}}",
        json_quote(event),
        json_quote(build_git_sha()),
        build_git_dirty(),
        json_quote(path),
        len,
        json_quote(sha),
        rss_mb,
        json_quote(reason)
    );
    std::process::exit(1);
}

/// The caller's `--anchor` (or `STWO_ML_ANCHOR`) for a format that carries its own anchor: required, so
/// `accepted: true` never rests on an anchor the prover supplied.
/// `file` is the path, the byte length and the SHA-256 the refusal reports.
fn required_caller_anchor(args: &[String], event: &str, file: (&str, usize, &str), what: &str) -> [u8; 32] {
    let (path, len, sha) = file;
    match flag_value(args, "--anchor", "STWO_ML_ANCHOR") {
        Some(s) => match parse_hex32(&s) {
            Ok(a) => a,
            Err(r) => emit_reject(event, path, len, sha, &r),
        },
        None => emit_reject(
            event,
            path,
            len,
            sha,
            &format!("missing --anchor <hex32> (or STWO_ML_ANCHOR): {what}; the bundle's own anchor is the prover's"),
        ),
    }
}

fn provenance_tail(bench_runs: usize, cold_ms: f64, warm_ms: f64, batch: usize) -> String {
    let rss_mb = process_peak_rss_mb();
    let b = batch.max(1) as f64;
    format!(
        "\"git_sha\":{},\"git_dirty\":{},\"bench_runs\":{},\"cold_verify_ms\":{:.3},\"warm_verify_median_ms\":{:.3},\"verify_ms\":{:.3},\"verify_ms_per_token\":{:.4},\"verifier_peak_rss_mb\":{:.3}",
        json_quote(build_git_sha()),
        build_git_dirty(),
        bench_runs,
        cold_ms,
        warm_ms,
        warm_ms,
        warm_ms / b,
        rss_mb
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Help only on its own: anywhere else, even as the value of a flag the prover supplies, `-h` would end a
    // verification command with the usage and exit status 0 before anything was checked.
    if args.iter().any(|a| a == "--help" || a == "-h") {
        if args.len() == 1 {
            println!("{USAGE}");
            return;
        }
        emit_reject(
            "stwo_ml_verifier_cli_error",
            "",
            0,
            &"0".repeat(64),
            "--help / -h takes no other argument; a verification command line must not contain it",
        );
    }
    let pos = positional_args(&args);
    let (expected_magic, path) = match pos.as_slice() {
        [sub, file] if expected_magic_for_subcommand(sub).is_some() => {
            (expected_magic_for_subcommand(sub), file.clone())
        }
        [file] => (None, file.clone()),
        [] => {
            let env_file = std::env::var("STWO_ML_BEACON_VERIFY_BUNDLE")
                .ok()
                .or_else(|| std::env::var("STWO_ML_VERIFY_FILE").ok())
                .unwrap_or_else(|| {
                    eprintln!("{USAGE}");
                    std::process::exit(2);
                });
            (None, env_file)
        }
        extra => {
            emit_reject(
                "stwo_ml_verifier_cli_error",
                extra.first().map(|s| s.as_str()).unwrap_or(""),
                0,
                &"0".repeat(64),
                &format!("unexpected extra positional arguments: {extra:?}"),
            );
        }
    };
    if let Err(flag_err) = validate_cli_flags(&args) {
        emit_reject("stwo_ml_verifier_cli_error", &path, 0, &"0".repeat(64), &flag_err);
    }
    let bench_runs = match parse_bench_runs(&args) {
        Ok(n) => n,
        Err(r) => emit_reject("stwo_ml_verifier_cli_error", &path, 0, &"0".repeat(64), &r),
    };
    let bytes = match read_capped_regular(std::path::Path::new(&path), MAX_WIRE_RECEIPT_BYTES) {
        Ok(b) => b,
        Err(e) => {
            emit_reject(
                "stwo_ml_verifier_io_error",
                &path,
                0,
                &"0".repeat(64),
                &format!("failed to read verification file {path}: {e}"),
            );
        }
    };
    let file_sha256 = sha256_hex(&bytes);
    let magic = bytes.get(..8).unwrap_or(b"");
    if let Some(exp) = expected_magic {
        if magic != exp {
            emit_reject(
                "stwo_ml_verifier_subcommand_mismatch",
                &path,
                bytes.len(),
                &file_sha256,
                &format!(
                    "subcommand expected magic {} but file starts with {:?}",
                    String::from_utf8_lossy(exp),
                    String::from_utf8_lossy(magic)
                ),
            );
        }
    }

    // 1. Whole-Model Single-Anchor Receipt (`STWOMDL1`, `.mdl1`)
    if magic == WHOLE_MODEL_MAGIC {
        let wire_stmt = match ummon_verifier::decode_whole_model_receipt(&bytes) {
            Ok(p) => p.statement,
            Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
        };
        let allowed_mdl_flags: &[&str] = if wire_stmt.tokens_committed {
            &["--anchor", "--prompt-commitment", "--emitted-commitment", "--bench-runs"]
        } else {
            &["--anchor", "--prompt-tokens", "--emitted-tokens", "--bench-runs"]
        };
        if let Err(r) = ensure_only_allowed_cli_flags(&args, "STWOMDL1", allowed_mdl_flags) {
            emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r);
        }
        let anchor = match flag_value(&args, "--anchor", "STWO_ML_ANCHOR")
            .ok_or_else(|| "missing --anchor <hex32> (or STWO_ML_ANCHOR) for STWOMDL1".to_string())
            .and_then(|s| parse_hex32(&s))
        {
            Ok(a) => a,
            Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
        };
        if wire_stmt.tokens_committed {
            let prompt_com = match flag_value(&args, "--prompt-commitment", "STWO_ML_PROMPT_COMMITMENT")
                .ok_or_else(|| {
                    "missing --prompt-commitment <hex32> (or STWO_ML_PROMPT_COMMITMENT) for committed-tokens STWOMDL1".to_string()
                })
                .and_then(|s| parse_hex32(&s))
            {
                Ok(c) => c,
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            };
            let emitted_com = match flag_value(&args, "--emitted-commitment", "STWO_ML_EMITTED_COMMITMENT")
                .ok_or_else(|| {
                    "missing --emitted-commitment <hex32> (or STWO_ML_EMITTED_COMMITMENT) for committed-tokens STWOMDL1".to_string()
                })
                .and_then(|s| parse_hex32(&s))
            {
                Ok(c) => c,
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            };
            match run_timed_verifies(bench_runs, || {
                let (stmt, run) =
                    verify_whole_model_committed_receipt_bytes(&anchor, &prompt_com, &emitted_com, &bytes)?;
                let ms = run.verify_ms;
                Ok(((stmt, run), ms))
            }) {
                Ok(((stmt, _), cold_ms, warm_ms)) => {
                    let tail = provenance_tail(bench_runs, cold_ms, warm_ms, stmt.batch);
                    println!(
                        "{{\"event\":\"stwo_ml_verifier_whole_model\",\"magic\":\"STWOMDL1\",\"verification_model\":\"standalone_anchor_only\",\"boundary_coverage\":\"full_prompt_to_emitted_tokens\",\"boundary_privacy\":\"salted_blake2s_committed_tokens\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"full_stack_anchor\":\"{}\",\"security_profile\":{},\"mode_tag\":{},\"zk_trace_blinded\":{},\"tokens_committed\":true,\"batch\":{},\"num_blocks\":{},\"layers_per_block\":{},\"total_layers\":{},\"vocab_size\":{},\"d_model\":{},\"prompt_commitment\":\"{}\",\"emitted_commitment\":\"{}\",\"accepted\":true,{tail}}}",
                        json_quote(&path),
                        bytes.len(),
                        file_sha256,
                        hex32(&anchor),
                        json_quote(&stmt.profile_id),
                        stmt.mode_tag,
                        stmt.zk_trace_blinded,
                        stmt.batch,
                        stmt.num_blocks,
                        stmt.layers_per_block,
                        stmt.num_blocks * stmt.layers_per_block,
                        stmt.vocab_size,
                        stmt.d_model,
                        hex32(&stmt.prompt_commitment),
                        hex32(&stmt.emitted_commitment),
                    );
                    return;
                }
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            }
        } else {
            let prompts = match flag_value(&args, "--prompt-tokens", "STWO_ML_PROMPT_TOKENS")
                .ok_or_else(|| {
                    "missing --prompt-tokens <csv> (or STWO_ML_PROMPT_TOKENS) for public-tokens STWOMDL1".to_string()
                })
                .and_then(|s| parse_u32_csv(&s))
            {
                Ok(v) => v,
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            };
            let emitted = match flag_value(&args, "--emitted-tokens", "STWO_ML_EMITTED_TOKENS")
                .ok_or_else(|| {
                    "missing --emitted-tokens <csv> (or STWO_ML_EMITTED_TOKENS) for public-tokens STWOMDL1".to_string()
                })
                .and_then(|s| parse_u32_csv(&s))
            {
                Ok(v) => v,
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            };
            match run_timed_verifies(bench_runs, || {
                let (stmt, run) = verify_whole_model_receipt_bytes(&anchor, &prompts, &emitted, &bytes)?;
                let ms = run.verify_ms;
                Ok(((stmt, run), ms))
            }) {
                Ok(((stmt, _), cold_ms, warm_ms)) => {
                    let tail = provenance_tail(bench_runs, cold_ms, warm_ms, stmt.batch);
                    println!(
                        "{{\"event\":\"stwo_ml_verifier_whole_model\",\"magic\":\"STWOMDL1\",\"verification_model\":\"standalone_anchor_only\",\"boundary_coverage\":\"full_prompt_to_emitted_tokens\",\"boundary_privacy\":\"public_tokens\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"full_stack_anchor\":\"{}\",\"security_profile\":{},\"mode_tag\":{},\"zk_trace_blinded\":{},\"tokens_committed\":false,\"batch\":{},\"num_blocks\":{},\"layers_per_block\":{},\"total_layers\":{},\"vocab_size\":{},\"d_model\":{},\"prompt_token_ids\":{:?},\"emitted_token_ids\":{:?},\"accepted\":true,{tail}}}",
                        json_quote(&path),
                        bytes.len(),
                        file_sha256,
                        hex32(&anchor),
                        json_quote(&stmt.profile_id),
                        stmt.mode_tag,
                        stmt.zk_trace_blinded,
                        stmt.batch,
                        stmt.num_blocks,
                        stmt.layers_per_block,
                        stmt.num_blocks * stmt.layers_per_block,
                        stmt.vocab_size,
                        stmt.d_model,
                        stmt.prompt_token_ids,
                        stmt.emitted_token_ids,
                    );
                    return;
                }
                Err(r) => emit_reject("stwo_ml_verifier_whole_model", &path, bytes.len(), &file_sha256, &r),
            }
        }
    }

    // 2. Standalone Single-Block Receipt (`STWOREC2`, `.rec2`, zero `wiring_root`)
    if magic == STANDALONE_BLOCK_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(
            &args,
            "STWOREC2",
            &["--anchor", "--initial-chain", "--final-chain", "--public-io", "--profile", "--bench-runs"],
        ) {
            emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r);
        }
        let anchor = match flag_value(&args, "--anchor", "STWO_ML_ANCHOR")
            .ok_or_else(|| "missing --anchor <hex32> (or STWO_ML_ANCHOR) for STWOREC2".to_string())
            .and_then(|s| parse_hex32(&s))
        {
            Ok(a) => a,
            Err(r) => emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r),
        };
        let (init_layer, init_prev) = match flag_value(&args, "--initial-chain", "STWO_ML_INITIAL_CHAIN")
            .ok_or_else(|| "missing --initial-chain <layer:hex32> for STWOREC2".to_string())
            .and_then(|s| parse_chain_link_spec(&s))
        {
            Ok(p) => p,
            Err(r) => emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r),
        };
        let final_link = match flag_value(&args, "--final-chain", "STWO_ML_FINAL_CHAIN") {
            Some(s) => match parse_chain_link_spec(&s) {
                Ok((l, d)) => Some(StandaloneChainLink {
                    layer: l,
                    prev_receipt_digest: d,
                }),
                Err(r) => emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r),
            },
            None => None,
        };
        let pub_io = match flag_value(&args, "--public-io", "STWO_ML_PUBLIC_IO") {
            Some(s) => match parse_hex32(&s) {
                Ok(d) => Some(d),
                Err(r) => emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r),
            },
            None => None,
        };
        let link = StandaloneChainLink {
            layer: init_layer,
            prev_receipt_digest: init_prev,
        };
        if let Some(p) = flag_value(&args, "--profile", "STWO_ML_PROFILE") {
            if let Err(r) = require_standalone_receipt_profile(&bytes, &p) {
                emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r);
            }
        }
        match run_timed_verifies(bench_runs, || {
            let (stmt, run) = verify_standalone_block_receipt_bytes_with_io(
                &anchor,
                &link,
                final_link.as_ref(),
                pub_io.as_ref(),
                &bytes,
            )?;
            let ms = run.verify_ms;
            Ok(((stmt, run), ms))
        }) {
            Ok(((stmt, _), cold_ms, warm_ms)) => {
                let tail = provenance_tail(bench_runs, cold_ms, warm_ms, stmt.batch);
                println!(
                    "{{\"event\":\"stwo_ml_verifier_standalone_block\",\"magic\":\"STWOREC2\",\"verification_model\":\"standalone_anchor_only\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"standalone_anchor\":\"{}\",\"security_profile\":{},\"mode_tag\":{},\"zk_trace_blinded\":{},\"batch\":{},\"num_layers\":{},\"initial_layer\":{},\"final_layer\":{},\"accepted\":true,{tail}}}",
                    json_quote(&path),
                    bytes.len(),
                    file_sha256,
                    hex32(&anchor),
                    json_quote(&stmt.profile_id),
                    stmt.mode_tag,
                    stmt.zk_trace_blinded,
                    stmt.batch,
                    stmt.num_layers,
                    stmt.initial_chain.layer,
                    stmt.final_chain.layer,
                );
                return;
            }
            Err(r) => emit_reject("stwo_ml_verifier_standalone_block", &path, bytes.len(), &file_sha256, &r),
        }
    }

    // 2b. Standalone Chained Multi-Block Manifest (`STWOCHN2`, `.chn2` / `chain`)
    if magic == STANDALONE_CHAIN_MANIFEST_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(
            &args,
            "STWOCHN2",
            &["--chain-anchor", "--initial-chain", "--final-chain", "--public-io", "--profile", "--bench-runs"],
        ) {
            emit_reject(
                "stwo_ml_verifier_standalone_block_chain",
                &path,
                bytes.len(),
                &file_sha256,
                &r,
            );
        }
        let chain_anchor_flag = flag_value(&args, "--chain-anchor", "STWO_ML_CHAIN_ANCHOR");
        let initial_chain_flag = flag_value(&args, "--initial-chain", "STWO_ML_INITIAL_CHAIN");
        let final_chain_flag = flag_value(&args, "--final-chain", "STWO_ML_FINAL_CHAIN");
        let public_io_flag = flag_value(&args, "--public-io", "STWO_ML_PUBLIC_IO");
        let profile_flag = flag_value(&args, "--profile", "STWO_ML_PROFILE");
        let base_dir = std::path::Path::new(&path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        let report = match verify_standalone_chain_command(
            &bytes,
            base_dir,
            &StandaloneChainCommandArgs {
                chain_anchor: chain_anchor_flag.as_deref(),
                initial_chain: initial_chain_flag.as_deref(),
                final_chain: final_chain_flag.as_deref(),
                public_io: public_io_flag.as_deref(),
                profile: profile_flag.as_deref(),
                bench_runs,
            },
        ) {
            Ok(r) => r,
            Err(r) => emit_reject(
                "stwo_ml_verifier_standalone_block_chain",
                &path,
                bytes.len(),
                &file_sha256,
                &r,
            ),
        };
        let init_spec = format!(
            "{}:{}",
            report.initial_chain.layer,
            hex32(&report.initial_chain.prev_receipt_digest)
        );
        let final_spec = format!(
            "{}:{}",
            report.verdict.final_chain.layer,
            hex32(&report.verdict.final_chain.prev_receipt_digest)
        );
        let tail = provenance_tail(bench_runs, report.cold_ms, report.warm_ms, report.batch);
        println!(
            "{{\"event\":\"stwo_ml_verifier_standalone_block_chain\",\"magic\":\"STWOCHN2\",\"receipt_magic\":\"STWOREC2\",\"verification_model\":\"standalone_anchor_chain_v10\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"chain_anchor\":\"{}\",\"receipts_sha256\":\"{}\",\"total_receipt_bytes\":{},\"security_profile\":{},\"mode_tag\":{},\"zk_trace_blinded\":{},\"batch\":{},\"num_blocks\":{},\"total_layers\":{},\"initial_chain\":\"{}\",\"final_chain\":\"{}\",\"com_h_in\":\"{}\",\"com_h_out\":\"{}\",\"per_block_receipt_bytes\":{:?},\"per_block_verify_ms\":{:?},\"accepted\":true,{tail}}}",
            json_quote(&path),
            bytes.len(),
            file_sha256,
            hex32(&report.chain_anchor),
            hex32(&report.receipts_sha256),
            report.total_receipt_bytes,
            json_quote(&report.profile_id),
            report.mode_tag,
            report.zk_trace_blinded,
            report.batch,
            report.num_blocks,
            report.total_layers,
            init_spec,
            final_spec,
            hex32(&report.verdict.com_h_in),
            hex32(&report.verdict.com_h_out),
            report.per_block_receipt_bytes,
            report.per_block_verify_ms,
        );
        return;
    }

    // 2c. Model: embedding, blocks and head (`STWOMDL2`, `model`)
    if magic == STANDALONE_MODEL_MANIFEST_MAGIC {
        let event = "stwo_ml_verifier_standalone_model";
        if let Err(r) = ensure_only_allowed_cli_flags(
            &args,
            "STWOMDL2",
            &["--model-anchor", "--initial-chain", "--tokens", "--out-token", "--gen-tokens", "--profile", "--bench-runs"],
        ) {
            emit_reject(event, &path, bytes.len(), &file_sha256, &r);
        }
        let model_anchor_flag = flag_value(&args, "--model-anchor", "STWO_ML_MODEL_ANCHOR");
        let initial_chain_flag = flag_value(&args, "--initial-chain", "STWO_ML_INITIAL_CHAIN");
        let tokens_flag = flag_value(&args, "--tokens", "STWO_ML_TOKENS");
        let out_token_flag = flag_value(&args, "--out-token", "STWO_ML_OUT_TOKEN");
        let gen_tokens_flag = flag_value(&args, "--gen-tokens", "STWO_ML_GEN_TOKENS");
        let profile_flag = flag_value(&args, "--profile", "STWO_ML_PROFILE");
        let base_dir = std::path::Path::new(&path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        let report = match verify_standalone_model_command(
            &bytes,
            base_dir,
            &StandaloneModelCommandArgs {
                model_anchor: model_anchor_flag.as_deref(),
                initial_chain: initial_chain_flag.as_deref(),
                tokens: tokens_flag.as_deref(),
                out_token: out_token_flag.as_deref(),
                gen_tokens: gen_tokens_flag.as_deref(),
                profile: profile_flag.as_deref(),
                bench_runs,
            },
        ) {
            Ok(r) => r,
            Err(r) => emit_reject(event, &path, bytes.len(), &file_sha256, &r),
        };
        let tail = provenance_tail(bench_runs, report.cold_ms, report.warm_ms, report.batch);
        let tokens: Vec<String> = report.verdict.tokens.iter().map(|t| t.to_string()).collect();
        // A multi-token bundle reports its generated tokens and no single output token.
        let (verification_model, out_token, gen_fields) = match &report.gen_tokens {
            None => ("standalone_model_v10", report.verdict.out_token.to_string(), String::new()),
            Some(g) => (
                "standalone_model_multi_token_v1",
                "null".to_string(),
                format!(
                    "\"multi_token\":true,\"prompt_len\":{},\"gen_k\":{},\"gen_tokens\":[{}],",
                    report.verdict.tokens.len(),
                    g.len(),
                    g.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
                ),
            ),
        };
        println!(
            "{{\"event\":\"{event}\",\"magic\":\"STWOMDL2\",\"receipt_magic\":\"STWOREC2\",\"verification_model\":\"{verification_model}\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"model_anchor\":\"{}\",\"tokens\":[{}],\"out_token\":{},{gen_fields}\"vocab_size\":{},\"receipts_sha256\":\"{}\",\"total_receipt_bytes\":{},\"security_profile\":{},\"mode_tag\":{},\"zk_trace_blinded\":{},\"batch\":{},\"num_receipts\":{},\"total_layers\":{},\"initial_chain\":\"{}:{}\",\"final_chain\":\"{}:{}\",\"per_receipt_bytes\":{:?},\"per_receipt_verify_ms\":{:?},\"accepted\":true,{tail}}}",
            json_quote(&path),
            bytes.len(),
            file_sha256,
            hex32(&report.verdict.model_anchor),
            tokens.join(","),
            out_token,
            report.vocab_size,
            hex32(&report.receipts_sha256),
            report.total_receipt_bytes,
            json_quote(&report.profile_id),
            report.mode_tag,
            report.zk_trace_blinded,
            report.batch,
            report.num_receipts,
            report.total_layers,
            report.initial_chain.layer,
            hex32(&report.initial_chain.prev_receipt_digest),
            report.verdict.final_chain.layer,
            hex32(&report.verdict.final_chain.prev_receipt_digest),
            report.per_receipt_bytes,
            report.per_receipt_verify_ms,
        );
        return;
    }

    // 3. Split-Anchor-and-Wiring-Root Receipt (`STWOREC1`, `.rec1`)
    if magic == RECURSIVE_AUDIT_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(
            &args,
            "STWOREC1",
            &[
                "--anchor",
                "--wiring-root",
                "--initial-chain",
                "--final-chain",
                "--public-io",
                "--bench-runs",
            ],
        ) {
            emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r);
        }
        let anchor = match flag_value(&args, "--anchor", "STWO_ML_ANCHOR")
            .ok_or_else(|| "missing --anchor <hex32> for STWOREC1".to_string())
            .and_then(|s| parse_hex32(&s))
        {
            Ok(a) => a,
            Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
        };
        let wiring_root = match flag_value(&args, "--wiring-root", "STWO_ML_WIRING_ROOT")
            .ok_or_else(|| "missing --wiring-root <hex32> for STWOREC1".to_string())
            .and_then(|s| parse_hex32(&s))
        {
            Ok(w) => w,
            Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
        };
        let (init_layer, init_prev) = match flag_value(&args, "--initial-chain", "STWO_ML_INITIAL_CHAIN")
            .ok_or_else(|| "missing --initial-chain <layer:hex32> for STWOREC1".to_string())
            .and_then(|s| parse_chain_link_spec(&s))
        {
            Ok(p) => p,
            Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
        };
        let expected_final = match flag_value(&args, "--final-chain", "STWO_ML_FINAL_CHAIN") {
            Some(s) => match parse_chain_link_spec(&s) {
                Ok((l, d)) => Some(StageAChainLink {
                    layer: l,
                    prev_receipt_digest: d,
                }),
                Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
            },
            None => None,
        };
        let expected_pub_io = match flag_value(&args, "--public-io", "STWO_ML_PUBLIC_IO") {
            Some(s) => match parse_hex32(&s) {
                Ok(d) => Some(d),
                Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
            },
            None => None,
        };
        let link = StageAChainLink {
            layer: init_layer,
            prev_receipt_digest: init_prev,
        };
        match run_timed_verifies(bench_runs, || {
            let (stmt, run) =
                verify_recursive_audit_receipt_bytes(&anchor, &wiring_root, &link, &bytes)?;
            let ms = run.verify_ms;
            Ok(((stmt, run), ms))
        }) {
            Ok(((stmt, _), cold_ms, warm_ms)) => {
                if let Some(exp_f) = expected_final {
                    if stmt.final_chain != exp_f {
                        emit_reject(
                            "stwo_ml_verifier_recursive_audit",
                            &path,
                            bytes.len(),
                            &file_sha256,
                            "STWOREC1 final_chain disagrees with --final-chain",
                        );
                    }
                }
                if let Some(exp_io) = expected_pub_io {
                    if stmt.public_io_digest != exp_io {
                        emit_reject(
                            "stwo_ml_verifier_recursive_audit",
                            &path,
                            bytes.len(),
                            &file_sha256,
                            "STWOREC1 public_io_digest disagrees with --public-io",
                        );
                    }
                }
                let tail = provenance_tail(bench_runs, cold_ms, warm_ms, stmt.batch);
                println!(
                    "{{\"event\":\"stwo_ml_verifier_recursive_audit\",\"magic\":\"STWOREC1\",\"verification_model\":\"two_party_challenger_attested_stark_leg\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"block_anchor\":\"{}\",\"wiring_root\":\"{}\",\"security_profile\":{},\"mode_tag\":{},\"batch\":{},\"num_layers\":{},\"accepted\":true,{tail}}}",
                    json_quote(&path),
                    bytes.len(),
                    file_sha256,
                    hex32(&anchor),
                    hex32(&wiring_root),
                    json_quote(&stmt.profile_id),
                    stmt.mode_tag,
                    stmt.batch,
                    stmt.num_layers,
                );
                return;
            }
            Err(r) => emit_reject("stwo_ml_verifier_recursive_audit", &path, bytes.len(), &file_sha256, &r),
        }
    }

    // 4. Cross-Layer Block Verification Bundle (`STWOBLKB`, `.blkb` wrapping `STWOBLK1`)
    if magic == CROSS_LAYER_BLOCK_BUNDLE_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(&args, "STWOBLKB", &["--anchor", "--initial-chain", "--bench-runs"]) {
            emit_reject("stwo_ml_verifier_cross_layer_block", &path, bytes.len(), &file_sha256, &r);
        }
        let blkb_anchor = required_caller_anchor(
            &args,
            "stwo_ml_verifier_cross_layer_block",
            (&path, bytes.len(), &file_sha256),
            "STWOBLKB is pinned by the digest of its per-layer anchors (compute_cross_layer_block_anchors_digest)",
        );
        let expected_init = match flag_value(&args, "--initial-chain", "STWO_ML_INITIAL_CHAIN") {
            Some(s) => match parse_chain_link_spec(&s) {
                Ok((l, d)) => Some(StageAChainLink {
                    layer: l,
                    prev_receipt_digest: d,
                }),
                Err(r) => emit_reject("stwo_ml_verifier_cross_layer_block", &path, bytes.len(), &file_sha256, &r),
            },
            None => None,
        };
        match run_timed_verifies(bench_runs, || {
            let t0 = std::time::Instant::now();
            let (b, run) = verify_cross_layer_block_bundle_bytes(&bytes)?;
            let verify_ms = t0.elapsed().as_secs_f64() * 1_000.0;
            Ok(((b, run), verify_ms))
        }) {
            Ok(((b, run), cold_ms, warm_ms)) => {
                let anchors_digest = compute_cross_layer_block_anchors_digest(&b.expected_anchors);
                if anchors_digest != blkb_anchor {
                    emit_reject(
                        "stwo_ml_verifier_cross_layer_block",
                        &path,
                        bytes.len(),
                        &file_sha256,
                        "STWOBLKB per-layer anchors do not hash to --anchor",
                    );
                }
                if let Some(exp_i) = expected_init {
                    if b.initial_chain != exp_i {
                        emit_reject(
                            "stwo_ml_verifier_cross_layer_block",
                            &path,
                            bytes.len(),
                            &file_sha256,
                            "STWOBLKB initial_chain disagrees with --initial-chain",
                        );
                    }
                }
                let final_layer = match b.initial_chain.layer.checked_add(run.layers as u32) {
                    Some(fl) => fl,
                    None => emit_reject(
                        "stwo_ml_verifier_cross_layer_block",
                        &path,
                        bytes.len(),
                        &file_sha256,
                        "STWOBLKB initial_chain.layer + num_layers overflows u32",
                    ),
                };
                let tail = provenance_tail(bench_runs, cold_ms, warm_ms, run.batch);
                let anchors_hex: Vec<String> = b.expected_anchors.iter().map(hex32).collect();
                println!(
                    "{{\"event\":\"stwo_ml_verifier_cross_layer_block\",\"magic\":\"STWOBLK1\",\"bundle_magic\":\"STWOBLKB\",\"verification_model\":\"standalone_anchor_only\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"receipt_bytes\":{},\"expected_anchors\":{:?},\"layer_anchors_digest\":\"{}\",\"security_profile\":\"{}\",\"num_layers\":{},\"batch\":{},\"final_layer\":{},\"accepted\":true,{tail}}}",
                    json_quote(&path),
                    bytes.len(),
                    file_sha256,
                    b.block_receipt.len(),
                    anchors_hex,
                    hex32(&anchors_digest),
                    b.profile.id(),
                    run.layers,
                    run.batch,
                    final_layer,
                );
                return;
            }
            Err(r) => emit_reject("stwo_ml_verifier_cross_layer_block", &path, bytes.len(), &file_sha256, &r),
        }
    }

    // 5. Stage B MLA Standalone Anchor Bundle (`STWOMLAB`, `.mlab` wrapping `STWOMLA1`)
    if magic == MLA_BUNDLE_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(&args, "STWOMLAB", &["--anchor", "--bench-runs"]) {
            emit_reject("stwo_ml_verifier_mla_bundle", &path, bytes.len(), &file_sha256, &r);
        }
        let cli_anchor = Some(required_caller_anchor(
            &args,
            "stwo_ml_verifier_mla_bundle",
            (&path, bytes.len(), &file_sha256),
            "STWOMLAB is checked against the anchor the caller registered",
        ));
        let decoded = match decode_mla_verification_bundle(&bytes) {
            Ok(b) => b,
            Err(r) => emit_reject("stwo_ml_verifier_mla_bundle", &path, bytes.len(), &file_sha256, &r),
        };
        if let Some(exp) = cli_anchor {
            if exp != decoded.expected_anchor {
                emit_reject(
                    "stwo_ml_verifier_mla_bundle",
                    &path,
                    bytes.len(),
                    &file_sha256,
                    "CLI --anchor does not match STWOMLAB expected_anchor",
                );
            }
        }
        match run_timed_verifies(bench_runs, || {
            let (batch, ms) = verify_mla_verification_bundle(&decoded)?;
            Ok((batch, ms))
        }) {
            Ok((batch, cold_ms, warm_ms)) => {
                let tail = provenance_tail(bench_runs, cold_ms, warm_ms, batch);
                println!(
                    "{{\"event\":\"stwo_ml_verifier_mla_bundle\",\"magic\":\"STWOMLA1\",\"bundle_magic\":\"STWOMLAB\",\"verification_model\":\"standalone_anchor_only\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"receipt_bytes\":{},\"standalone_mla_anchor\":\"{}\",\"security_profile\":\"{}\",\"batch\":{},\"heads\":{},\"kv_lora_rank\":{},\"accepted\":true,{tail}}}",
                    json_quote(&path),
                    bytes.len(),
                    file_sha256,
                    decoded.receipt_bytes.len(),
                    hex32(&decoded.expected_anchor),
                    decoded.profile.id(),
                    batch,
                    decoded.statement.shape.heads,
                    decoded.statement.shape.kv_lora_rank,
                );
                return;
            }
            Err(r) => emit_reject("stwo_ml_verifier_mla_bundle", &path, bytes.len(), &file_sha256, &r),
        }
    }

    // 6. Stage B SWA + Attention-Sinks Standalone Anchor Bundle (`STWOSWAB`, `.swab` wrapping `STWOSWA1`)
    if magic == SWA_BUNDLE_MAGIC {
        if let Err(r) = ensure_only_allowed_cli_flags(&args, "STWOSWAB", &["--anchor", "--bench-runs"]) {
            emit_reject("stwo_ml_verifier_swa_bundle", &path, bytes.len(), &file_sha256, &r);
        }
        let cli_anchor = Some(required_caller_anchor(
            &args,
            "stwo_ml_verifier_swa_bundle",
            (&path, bytes.len(), &file_sha256),
            "STWOSWAB is checked against the anchor the caller registered",
        ));
        let decoded = match decode_swa_verification_bundle(&bytes) {
            Ok(b) => b,
            Err(r) => emit_reject("stwo_ml_verifier_swa_bundle", &path, bytes.len(), &file_sha256, &r),
        };
        if let Some(exp) = cli_anchor {
            if exp != decoded.expected_anchor {
                emit_reject(
                    "stwo_ml_verifier_swa_bundle",
                    &path,
                    bytes.len(),
                    &file_sha256,
                    "CLI --anchor does not match STWOSWAB expected_anchor",
                );
            }
        }
        match run_timed_verifies(bench_runs, || {
            let (batch, ms) = verify_swa_verification_bundle(&decoded)?;
            Ok((batch, ms))
        }) {
            Ok((batch, cold_ms, warm_ms)) => {
                let tail = provenance_tail(bench_runs, cold_ms, warm_ms, batch);
                println!(
                    "{{\"event\":\"stwo_ml_verifier_swa_bundle\",\"magic\":\"STWOSWA1\",\"bundle_magic\":\"STWOSWAB\",\"verification_model\":\"standalone_anchor_only\",\"file\":{},\"bytes\":{},\"sha256\":\"{}\",\"receipt_bytes\":{},\"standalone_swa_anchor\":\"{}\",\"security_profile\":\"{}\",\"batch\":{},\"heads\":{},\"kv_heads\":{},\"accepted\":true,{tail}}}",
                    json_quote(&path),
                    bytes.len(),
                    file_sha256,
                    decoded.receipt_bytes.len(),
                    hex32(&decoded.expected_anchor),
                    decoded.profile.id(),
                    batch,
                    decoded.statement.shape.heads,
                    decoded.statement.shape.kv_heads,
                );
                return;
            }
            Err(r) => emit_reject("stwo_ml_verifier_swa_bundle", &path, bytes.len(), &file_sha256, &r),
        }
    }

    // 7. Default / Backward-Compatible Spot-Check Bundle (`STWOBCN4` / `STWOBCN3`, `.bcn4`, one layer in 48 per call)
    if let Err(r) = ensure_only_allowed_cli_flags(&args, "STWOBCN4", &["--anchor", "--bench-runs"]) {
        emit_reject("stwo_ml_verifier_bundle", &path, bytes.len(), &file_sha256, &r);
    }
    let bcn_cli_anchor = Some(required_caller_anchor(
        &args,
        "stwo_ml_verifier_bundle",
        (&path, bytes.len(), &file_sha256),
        "STWOBCN4 is checked against the layer anchor the caller registered",
    ));
    let bundle = match decode_spot_check_verification_bundle(&bytes) {
        Ok(b) => b,
        Err(reason) => {
            emit_reject(
                "stwo_ml_verifier_bundle",
                &path,
                bytes.len(),
                &file_sha256,
                &reason,
            );
        }
    };
    if let Some(exp_anchor) = bcn_cli_anchor {
        if exp_anchor != bundle.anchor {
            emit_reject(
                "stwo_ml_verifier_bundle",
                &path,
                bytes.len(),
                &file_sha256,
                "CLI --anchor does not match STWOBCN4 bundle layer anchor",
            );
        }
    }
    let shape = bundle.statement.shape;
    let head = format!(
        concat!(
            "{{\"event\":\"stwo_ml_verifier_bundle\",\"bundle\":{},\"bundle_bytes\":{},\"bundle_sha256\":\"{}\",",
            "\"receipt_bytes\":{},\"attention_receipt_bytes\":{},\"leaf_version\":4,\"boundaries_present\":{},",
            "\"security_profile\":\"{}\",\"policy\":\"{}\",\"attention_policy\":\"{}\",",
            "\"beacon_block_number\":{},\"beacon_block_hash\":\"0x{}\",\"response_root\":\"{}\",",
            "\"layers\":{},\"sample_den\":{},\"batch\":{},\"d\":{},\"m\":{},\"E\":{},\"k\":{},\"shared_factor\":{},",
            "\"attention_heads\":{},\"attention_kv_heads\":{},\"layer_anchor\":\"{}\",\"model_anchor\":\"{}\""
        ),
        json_quote(&path),
        bytes.len(),
        file_sha256,
        bundle.receipt.len(),
        bundle.attention_receipt.len(),
        bundle.boundaries.is_some(),
        bundle.profile.id(),
        bundle.policy.id(),
        bundle.attention_policy.id(),
        bundle.beacon.block_number,
        hex32(&bundle.beacon.block_hash),
        hex32(&bundle.response_root),
        bundle.beacon_policy.layers,
        bundle.beacon_policy.sample_den,
        shape.batch,
        shape.dimension,
        shape.expert_width,
        shape.experts,
        shape.top_k,
        shape.shared_factor,
        bundle.attention.shape.heads,
        bundle.attention.shape.kv_heads,
        hex32(&bundle.anchor),
        hex32(&bundle.context.anchor_digest()),
    );
    match run_timed_verifies(bench_runs, || {
        let verdict = verify_spot_check_verification_bundle(&bundle)?;
        let ms = verdict.verify_ms;
        Ok((verdict, ms))
    }) {
        Ok((verdict, cold_ms, warm_ms)) => {
            let tail = provenance_tail(bench_runs, cold_ms, warm_ms, shape.batch);
            let emitted_next_token_json = match verdict.emitted_next_token {
                Some(tok) => tok.to_string(),
                None => "null".to_string(),
            };
            println!(
                "{head},\"stage_a_mode\":\"{}\",\"challenged_layer\":{},\"continuity_opened\":{},\"emitted_next_token\":{},\"accepted\":true,\"decode_ms\":{:.3},{tail}}}",
                verdict.stage_a_mode,
                verdict.layer,
                bundle.continuity.is_some(),
                emitted_next_token_json,
                verdict.decode_ms,
            );
        }
        Err(reason) => {
            let rss_mb = process_peak_rss_mb();
            println!(
                "{head},\"git_sha\":{},\"git_dirty\":{},\"verifier_peak_rss_mb\":{:.3},\"accepted\":false,\"reason\":{}}}",
                json_quote(build_git_sha()),
                build_git_dirty(),
                rss_mb,
                json_quote(&reason)
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;
    use ummon_verifier::resolve_contained_receipt_path;

    #[test]
    fn json_quote_emits_rfc8259_escapes_without_rust_debug_unicode() {
        assert_eq!(
            json_quote("bad\x1b[31m\"path\\line\n\t\x00"),
            "\"bad\\u001b[31m\\\"path\\\\line\\n\\t\\u0000\""
        );
    }

    #[test]
    fn validate_cli_flags_rejects_unknown_duplicate_and_missing_value_flags() {
        let s = |xs: &[&str]| -> Vec<String> { xs.iter().map(|x| x.to_string()).collect() };
        assert!(validate_cli_flags(&s(&[
            "chn2",
            "chain.chn2",
            "--chain-anchor",
            &"11".repeat(32),
            "--initial-chain",
            &format!("0:{}", "22".repeat(32)),
            "--json",
        ]))
        .is_ok());

        // Unknown flag:
        assert!(validate_cli_flags(&s(&["chn2", "chain.chn2", "--bogus-flag"])).is_err());
        // Duplicate flag (`--chain-anchor` passed twice or `--chain-anchor=...` + `--chain-anchor`):
        assert!(validate_cli_flags(&s(&[
            "chn2",
            "chain.chn2",
            "--chain-anchor",
            &"11".repeat(32),
            "--chain-anchor",
            &"22".repeat(32),
        ]))
        .is_err());
        assert!(validate_cli_flags(&s(&[
            "chn2",
            "chain.chn2",
            &format!("--chain-anchor={}", "11".repeat(32)),
            "--chain-anchor",
            &"22".repeat(32),
        ]))
        .is_err());
        // Missing value at end or before next flag:
        assert!(validate_cli_flags(&s(&["chn2", "chain.chn2", "--chain-anchor"])).is_err());
        assert!(validate_cli_flags(&s(&[
            "chn2",
            "chain.chn2",
            "--chain-anchor",
            "--initial-chain",
            &format!("0:{}", "22".repeat(32)),
        ]))
        .is_err());

        // Per-format CLI flag allowlist blocks omitted-subcommand bundle substitution:
        assert!(ensure_only_allowed_cli_flags(
            &s(&["bundle.blkb", "--anchor", &"11".repeat(32)]),
            "STWOBLKB",
            &["--initial-chain", "--bench-runs"]
        )
        .is_err());
        assert!(ensure_only_allowed_cli_flags(
            &s(&["bundle.bcn4", "--prompt-tokens", "1,2"]),
            "STWOBCN4",
            &["--anchor", "--bench-runs"]
        )
        .is_err());
        assert!(ensure_only_allowed_cli_flags(
            &s(&["committed.mdl1", "--anchor", &"11".repeat(32), "--prompt-tokens", "1,2"]),
            "STWOMDL1",
            &["--anchor", "--prompt-commitment", "--emitted-commitment", "--bench-runs"]
        )
        .is_err());
        assert!(ensure_only_allowed_cli_flags(
            &s(&["public.mdl1", "--anchor", &"11".repeat(32), "--prompt-commitment", &"22".repeat(32)]),
            "STWOMDL1",
            &["--anchor", "--prompt-tokens", "--emitted-tokens", "--bench-runs"]
        )
        .is_err());
        assert!(ensure_only_allowed_cli_flags(
            &s(&["chain.chn2", "--chain-anchor", &"11".repeat(32), "--initial-chain", &format!("0:{}", "22".repeat(32)), "--json"]),
            "STWOCHN2",
            &["--chain-anchor", "--initial-chain", "--final-chain", "--public-io", "--bench-runs"]
        )
        .is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn read_capped_regular_and_contained_path_reject_symlinks_and_devices() {
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!(
            "stwo_ml_verify_symlink_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let real_file = dir.join("real.rec2");
        std::fs::write(&real_file, b"STWOREC2").unwrap();
        assert_eq!(read_capped_regular(&real_file, 64).unwrap(), b"STWOREC2");

        let sym_file = dir.join("link.rec2");
        symlink(&real_file, &sym_file).unwrap();
        assert!(read_capped_regular(&sym_file, 64).is_err());
        assert!(resolve_contained_receipt_path(&dir, "link.rec2").is_err());

        let outside = std::env::temp_dir();
        let sym_dir = dir.join("escape");
        symlink(&outside, &sym_dir).unwrap();
        assert!(resolve_contained_receipt_path(&dir, "escape/anything").is_err());

        assert!(read_capped_regular(std::path::Path::new("/dev/zero"), 64).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
