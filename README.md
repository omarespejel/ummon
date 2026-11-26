# Ummon verifier

This repository lets you check the four proofs of the paper *Ummon: Verifying a 400-Billion-Parameter Language Model
in Milliseconds* by Omar U. Espejel on your own machine. It holds the verifier, a WebAssembly build of it with a
local web page, the four statements, the SHA-256 of every proof file, and a script that checks all of it; the Lean 4
development that machine-checks the interactive soundness of one opening of CPRS, the paper's weight commitment; and
the paper's supplement with the data files and plot scripts of its figures. It does not hold the prover.

Licence: Apache-2.0 (`LICENSE`); copyright 2026 Omar U. Espejel. See "Licence" at the end for the third-party parts.
Repository: https://github.com/omarespejel/ummon. The four proofs are an asset of its release `v1.0`.

## What a proof shows

Each proof states that the integer model registered under a 32-byte model anchor, given 16 prompt tokens, generates
16 stated tokens by greedy decoding, through all 48 layers. The model is an integer model derived from the released
weights of Llama 4 Maverick (400 billion parameters). A proof is a bundle of three receipts (16 layers each, about
1.55 MB each) and a manifest of 915 to 937 bytes.

You give the verifier the statement: the model anchor, the initial chain link, the prompt tokens, the generated
tokens and the security profile name. None of these comes from the bundle. The verifier compares your tokens with
the manifest's, checks every receipt, and checks that the receipts chain from the first layer to the last under your
anchor. It holds no weights: everything about the model is fixed by the anchor.

## What a proof does not show

These limits are the paper's (its box "What this paper does not show", and Sections 3, 5 and 7); read them before
relying on a proof.

- Soundness assumes honest registration from the stated weights, and a verifier must trust the anchor's source: an
  anchor taken from the prover says nothing about which model ran. Only recomputing the anchor from the original
  weights, the calibration file and the paper's code shows that the registered model is a quantised Llama 4
  Maverick. So far only the paper's author has computed it; the exporter, the calibration file and the prover are not
  in this release, so this repository cannot recompute it.
- Agreement with the float model is measured, not proven: 95.1% of top tokens on 30 held-out texts, without the
  16-bit integer head. On two of the four proved statements the float model's own greedy output differs from the
  proof, at near-ties.
- A proof does not show that the prompt is authentic or came from a given user, that the model is good at any task,
  or what the float model would generate. It covers greedy decoding only, and the prompt and the generated tokens must
  fit in 32 positions.
- The security estimate, 84.1 bits (100.5 with proof-of-work credit), is computed from the measured proofs' counts,
  with three inputs counted from the earlier version's code, and rests on six unproven assumptions; no cryptographer
  has reviewed it.
- Each of the four statements was proved once. The prover, not in this repository, needs 63 to 80 minutes and 339 to
  340 GB of host memory on one H100.
- No zero knowledge: the prompt and the generated tokens are public.

## What is here

| Path | What it is |
|---|---|
| `verifier/` | The verifier library, crate `ummon-verifier`, and its command `ummon-verify` (Rust). |
| `web/` | A thin wrapper that builds the same verifier for `wasm32-unknown-unknown`, and a local page that uses it. |
| `docs/evidence/` | Two lookup tables of an earlier receipt format that the verifier compiles in (`verifier/src/stage_a.rs`), and the record of a clean build of `formal_cprs/` (`lean_clean_build_20261126T194919Z/`). |
| `formal_cprs/` | The Lean 4 development of one CPRS opening: sources, `lakefile.toml`, `lean-toolchain`, `lake-manifest.json`, `scripts/check.sh`, and 39 vendored, unmodified ArkLib modules with ArkLib's licence. |
| `statements/` | The four statements (`proof1_prose.txt` ... `proof4_french.txt`) and the expected results (`expected.txt`). |
| `SHA256SUMS` | SHA-256 of the 16 proof files. |
| `VERIFY.sh` | Checks the hashes, then runs the verifier on each proof against its statement. |
| `get_stwo.sh` | Fetches the verifier's stwo dependency into `./stwo` (see below). |
| `third_party/stwo-69e6f9a0.bundle` | Our 23 commits on top of the upstream stwo library, as a git bundle. |
| `rust-toolchain.toml` | The pinned Rust toolchain, `nightly-2026-01-15`. |
| `LICENSE` | The Apache License, Version 2.0. |
| `paper/ummon.pdf`, `paper/ummon_extended_abstract.pdf` | The paper (25 pages) and its two-page extended abstract. |
| `paper/supplement.md` | The paper's supplement: text and tables moved out of the paper when it was shortened. |
| `paper/figures/` | The data file and plot script of Figure 1 (`fig1_verify_vs_size.csv`, `fig1_plot.py`) and of Appendix D's crossovers between proving once and checking many times (`fig2_prove_once_verify_n.csv`, `fig2_plot.py`). |

Not here: the prover, the model export and calibration, the fixtures of the verifier's fixture-based tests, the
measurement scripts and logs, the paper's claims ledger and the evidence it cites, and the working notes of
`formal_cprs/` (its README, claim map, notes on the chain theorem and check logs). The paper's "private source
repository" (Appendix F, Appendix G) is the repository that holds the prover and the calibration file; it is not this
one. The verifier still compiles the code of earlier receipt formats (`.rec1`, `.blk1`, `.mla1`, `.swa1`, `.bcn4`) that the paper does not
use; the four proofs use only the `model` command.

The verifier depends on the stwo Circle STARK library: upstream commit
`f0d79b0fad440dcb0aaf1e20470fdbb37993ea2a` of https://github.com/starkware-libs/stwo plus 23 commits of ours (22
files, 3,229 lines added and 219 removed, mostly in the verifier, the commitment scheme and FRI). `get_stwo.sh` fetches
that upstream commit, applies the bundle and checks that the result is commit `69e6f9a03199d0a17add4855e0da5f3c1a48c785`
with tree `4f3b119deb0bebd94f32a8b7079069a6f7fc2e03`. `git -C stwo log` then shows each of our commits.

Our stwo changes ship only as this bundle for now; a public fork of stwo with the same 23 commits may follow.

This repository and its release ship source only: no built binary or WebAssembly module. You build both yourself
(sections 2 and 5).

## Requirements

- git, a POSIX shell, `sha256sum` or `shasum`.
- Rust through rustup, with the toolchain that `rust-toolchain.toml` pins:
  `rustup toolchain install nightly-2026-01-15 --profile minimal --target wasm32-unknown-unknown`.
- Network access for the first build (the upstream stwo commit and the crates.io dependencies pinned by the
  `Cargo.lock` files), and about 2 GB of disk for the builds. No GPU.
- For the web page: Python 3 or any static file server, and a browser. For the command-line check of the
  WebAssembly build: Node.js.

## 1. Get the stwo dependency

```bash
./get_stwo.sh
```

## 2. Build the verifier

```bash
cargo build --release --locked --manifest-path verifier/Cargo.toml --bin ummon-verify
```

The binary is `verifier/target/release/ummon-verify`.

`cargo test --locked --manifest-path verifier/Cargo.toml` runs the verifier crate's 23 unit tests: statement bounds
and canonicality, model and chain anchors, manifests, security profiles, the Poseidon2 permutation against a known
answer, and the command line's flag, path and JSON handling. The tests that need the source repository's fixtures
(earlier `.bcn4` bundles and a toy model bundle) sit behind the feature `fixture-tests`, off by default, and are not
built here.

## 3. Get the proofs

The four proofs are one archive, `ummon-proofs-v1.tar` (18,580,480 bytes), attached to the release `v1.0` of this
repository. Download and unpack it here:

```bash
curl -L -o ummon-proofs-v1.tar https://github.com/omarespejel/ummon/releases/download/v1.0/ummon-proofs-v1.tar
tar -xf ummon-proofs-v1.tar
```

The archive's SHA-256 is `6bfad160518a73b1f0871153c54d1c33734118631c4e254054cd31dd0a910fbf`
(`shasum -a 256 ummon-proofs-v1.tar`). It unpacks into `proofs/` (16 files, 18,565,739 bytes):

```text
proofs/proof1_prose/     model_manifest.mdl2  receipt_00.rec2  receipt_01.rec2  receipt_02.rec2
proofs/proof2_python/    (same four names)
proofs/proof3_dialogue/  (same four names)
proofs/proof4_french/    (same four names)
```

`SHA256SUMS` in this repository fixes every file, so the download host need not be trusted; `VERIFY.sh` checks them
first.

## 4. Verify the four proofs

```bash
./VERIFY.sh proofs
```

It checks the 16 hashes, runs the verifier on each proof with the statement in `statements/`, and compares the
verifier's report with `statements/expected.txt`. The expected output ends with:

```text
== 2. The verifier on each proof, against the statement in statements/
proof1_prose: accepted; 3 receipts, 48 layers, 4643682 receipt bytes, final chain 48:fb098795e93429814cfbe459020fa302339bf988dd1070ffd1d3b6a2cae4129f
proof2_python: accepted; 3 receipts, 48 layers, 4642210 receipt bytes, final chain 48:f54092755756e5b14fd769ced72dcefbb87ddcbeaa07ecfbc06090254819943b
proof3_dialogue: accepted; 3 receipts, 48 layers, 4654514 receipt bytes, final chain 48:fb90be339cdcfe0658ba88de8338a77d757503d42158660bc1e6f33c8ad92ae1
proof4_french: accepted; 3 receipts, 48 layers, 4621618 receipt bytes, final chain 48:4128921bdd66680ef5db8e727cf780f4a301f26751ed4e415d5c1b412f857889
== RESULT: all four proofs accepted
```

### Each proof by hand

The command for the first proof is the one in the paper's Appendix G:

```bash
verifier/target/release/ummon-verify model proofs/proof1_prose/model_manifest.mdl2 \
  --model-anchor 5f1141e860a60512243a9e0848ee0bea9b9d5b7f03196ae1e1f59b2ba01f1f68 \
  --initial-chain 0:2424242424242424242424242424242424242424242424242424242424242424 \
  --tokens 200000,64,2752,16980,341,7550,2895,290,30404,75041,436,290,113841,262,6060,38794 \
  --gen-tokens 436,1085,4635,323,12824,1648,290,1453,26,589,30404,948,19560,75041,436,290 \
  --profile production_hardened
```

```bash
verifier/target/release/ummon-verify model proofs/proof2_python/model_manifest.mdl2 \
  --model-anchor 5f1141e860a60512243a9e0848ee0bea9b9d5b7f03196ae1e1f59b2ba01f1f68 \
  --initial-chain 0:2424242424242424242424242424242424242424242424242424242424242424 \
  --tokens 200000,2986,30967,1613,55082,22727,2946,1619,432,52071,10692,600,277,10938,45,8649 \
  --gen-tokens 8098,12462,511,3794,159809,290,5281,12948,1792,5990,1660,73441,277,1013,3251,3996 \
  --profile production_hardened
```

```bash
verifier/target/release/ummon-verify model proofs/proof3_dialogue/model_manifest.mdl2 \
  --model-anchor 5f1141e860a60512243a9e0848ee0bea9b9d5b7f03196ae1e1f59b2ba01f1f68 \
  --initial-chain 0:2424242424242424242424242424242424242424242424242424242424242424 \
  --tokens 200000,592,264,38,589,51680,5522,290,2018,323,290,24619,373,116009,3357,26 \
  --gen-tokens 372,3865,675,1657,328,14155,536,26,1102,54,26416,38,372,16795,26,589 \
  --profile production_hardened
```

```bash
verifier/target/release/ummon-verify model proofs/proof4_french/model_manifest.mdl2 \
  --model-anchor 5f1141e860a60512243a9e0848ee0bea9b9d5b7f03196ae1e1f59b2ba01f1f68 \
  --initial-chain 0:2424242424242424242424242424242424242424242424242424242424242424 \
  --tokens 200000,2703,9394,336,18545,39720,472,33821,518,103216,64580,14167,26,22269,532,115515 \
  --gen-tokens 17048,1997,768,31709,145285,24,1579,307,36219,262,472,9385,11227,320,54300,3236 \
  --profile production_hardened
```

Each exits with status 0 and prints one JSON line that contains `"accepted":true`,
`"verification_model":"standalone_model_multi_token_v1"`, `"num_receipts":3`, `"total_layers":48`, and the receipt
bytes and final chain link of `statements/expected.txt`. Change any token, the anchor, or one byte of a receipt, and
it exits with status 1 and a reason. Always pass `--profile production_hardened`: the model anchor already fixes each
receipt's profile, because the profile id is hashed into the receipt's anchor and the verifier derives the outer
proof's parameters from it, so the flag checks that the anchor was registered under this profile; the other profile ids
exist for development and give lower security. The verifier also reads `STWO_ML_*` environment variables; leave them
unset (`VERIFY.sh` runs it in a clean environment).

## 5. Check a proof in a browser

Build the WebAssembly module and put it next to the page:

```bash
cargo build --release --locked --manifest-path web/Cargo.toml --target wasm32-unknown-unknown
cp web/target/wasm32-unknown-unknown/release/ummon_web_verifier.wasm web/
```

Serve `web/` on the loopback address only, and open http://127.0.0.1:8765/ :

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory web
```

Choose the four files of one proof, paste that proof's statement from `statements/` into the statement box (the
first proof's is already there), and press Verify. The page loads the module from its own origin and sends nothing
anywhere. The same module also runs under Node:

```bash
node web/node_check.mjs web/ummon_web_verifier.wasm proofs/proof1_prose statements/proof1_prose.txt 0
```

It prints `result accept`, the statement, `final_chain`, `receipts 3`, `layers 48` and `receipt_bytes`.

## 6. The Lean development of one CPRS opening

**What it proves** (paper, Section 5.3 and Appendix A). Lean 4 machine-checks the interactive soundness of one CPRS
opening under honest registration, for a generic finite field extension and committed matrices of equal length. Let
`K` be a finite field with a subfield `F` of index `d`, an `F`-automorphism of `K` of order `d` and an `F`-basis of
`K`; take `L` registered matrices of `2^m` rows each, packed `d` base-field values to a symbol, a rate exponent
`s >= 1` and a domain of order `N = 2^(m+s)`, commitments after any chosen fold rounds, and `Q` uniform queries. For
a false claim, every deterministic adaptive prover is accepted with probability at most

    ((dL - 1) + 2m + [L >= 2] + N - 2^s) / |K| + ((1 + 2^-s) / 2)^Q

over exactly uniform challenges, with the Iverson bracket `[L >= 2]`. The theorems are `Cprs.Chain.chain_soundness`
and `Cprs.Chain.chain_soundness_moore` (`formal_cprs/CprsFormal/Chain.lean`, `ChainMoore.lean`). The proof uses only
Lean's standard axioms and ArkLib's proof of the correlated-agreement theorem it rests on (BCIKS20, Theorem 4.1),
vendored unmodified in `formal_cprs/vendor/` from ArkLib commit `ace55c3e`.

**What it does not cover.** It is a proof of a model. The model reads each registered word as its committed
polynomial's exact evaluation, with no Merkle tree. The field is abstract; that `QM31` (the
verifier's field), its basis and its Frobenius meet the hypotheses is tested, not proved. It does not cover the
Fiat-Shamir step, proof of work, Merkle binding, openings of data the prover commits such as the witness, the
combination of groups of different lengths that the production verifier performs, or the Rust and in-circuit
verifiers. That the code matches the model is tested, not proved.

**Build it.** Install elan (https://github.com/leanprover/elan); it installs the Lean version that
`formal_cprs/lean-toolchain` pins (`v4.34.0`). Budget about 10 GB of disk. Then:

```bash
cd formal_cprs
lake update
lake exe cache get
bash scripts/check.sh
```

`lake update` fetches the 13 dependencies at the commits pinned in `lakefile.toml`; it should leave
`lake-manifest.json` unchanged (`git diff --exit-code lake-manifest.json`). `lake exe cache get` fetches Mathlib's
compiled files. `scripts/check.sh` fails on any `sorry`, `admit`, `native_decide`, `unsafe` or `axiom` in the
sources, runs `lake build`, and checks that every theorem in the namespace `Cprs` (at least 624) depends on no axiom
beyond `propext`, `Classical.choice` and `Quot.sound`. It ends with `PASS`. Two vendored ArkLib files contain unrelated
`sorry`s that no theorem here uses; the axiom sweep would report `sorryAx` otherwise.

**The clean build on record** (`docs/evidence/lean_clean_build_20261126T194919Z/`). On 2026-11-26 these sources
were built in a cloud container with 16 CPU cores and 32 GiB of memory (x86_64 Linux) that started with only git and
curl. elan 4.4.0 installed Lean 4.34.0 (commit `293d5d0c`) with Lake 5.0.0. `lake update` took 144.8 s and left the
manifest byte for byte as committed; `lake exe cache get` 5.0 s; `scripts/check.sh` passed in 194.9 s (9,093 build
jobs, 624 theorems swept, none with an axiom beyond the three standard ones); the whole run took 381.2 s and 8.8 GiB
of disk (8.4 GiB of build output and 433 MiB of Mathlib cache). `axioms.log` prints the axioms of nine main theorems,
as `docs/evidence/lean_print_axioms.lean` asks, and `result.json` lists the SHA-256 of the 69 files the build read;
its `producer_git_sha` names the commit of the private source repository that launched the build. To confirm that
`formal_cprs/` holds exactly those files:

```bash
cd formal_cprs
sed -n '/"sources_sha256"/,/}/s/^ *"\([^"]*\)": "\([0-9a-f]*\)".*/\2  \1/p' \
  ../docs/evidence/lean_clean_build_20261126T194919Z/result.json | shasum -a 256 -c
```

## 7. The paper's supplement and figures

`paper/supplement.md` holds, as the paper's text left them, the paragraphs and tables moved out of the paper when it
was shortened; its section and table numbers are those of the longer draft. The two plot scripts use the Python
standard library only:

```bash
python3 paper/figures/fig1_plot.py
python3 paper/figures/fig2_plot.py
```

The first writes `paper/figures/fig1_verify_vs_size.svg`, Figure 1, from `fig1_verify_vs_size.csv`. The second prints
the crossovers of Appendix D (374 checks as run, 54 to 215 with FlashProve's measured commitment times counted) and
writes a plot of them, `paper/figures/fig2_prove_once_verify_n.svg`, from `fig2_prove_once_verify_n.csv`. The data
files' `source` columns name the paper's sections and internal records the figures came from; those records are not
in this repository.

## Timings

The paper reports 47 ms (median 46.63 ms of 30 runs) and 22.9 MB of peak memory from process start to exit for the
first proof on one laptop core (Apple M5 Max, on mains power; Section 6.2), and, in Chromium with the files already in
memory, 31.1 ms at the median for the verify call (Section 6.3). Your times will differ with your machine.

## Names

Some hash domain labels inside the verifier begin with `SPARSEPROVE_`, the project's earlier working name (for
example `SPARSEPROVE_STWOREC2_ANCHOR_V1`). They are hashed into every anchor and digest of the published proofs, so
they cannot change without invalidating the proofs; they are kept as historical names.

## Licence

Copyright 2026 Omar U. Espejel. Licensed under the Apache License, Version 2.0 (`LICENSE`).

Third-party parts keep their own terms: `formal_cprs/vendor/` holds 39 modules of ArkLib, unmodified, copyright the
ArkLib Contributors, under Apache-2.0 (`formal_cprs/vendor/ArkLib_LICENSE`); `third_party/stwo-69e6f9a0.bundle` holds
our commits to the stwo library, which is Apache-2.0 (its `LICENSE` arrives with `./get_stwo.sh`). The Rust and Lean
dependencies are fetched by cargo and lake, each with its own licence.
