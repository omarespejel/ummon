# Ummon: supplement to the preprint

This supplement holds what the preprint's PDF no longer carries: paragraphs and tables moved out of the body
and the appendices when the paper was shortened. Section, table and appendix numbers inside it are those of the longer
draft it came from, so Tables 3 to 9 and F1, and the full Table D1, appear here and not in the paper. Every figure is
stated as it was in the paper, with the same sources. In this released copy, eight passages that cited unreleased material are reworded.

# Part 1. Moved from the body

## From the abstract and Figure 1

Nothing moved from the abstract proper.

### Figure 1 caption

Systems slower than the axis are listed beside it.

The figures come from different hardware, models and statements; only the comparison with FlashProve, which we ran
ourselves, is a measured ratio, and we print no ratio for the others.

## From Section 1

One core of an Apple M5 Max laptop checks one in 47 ms from process start to exit, with 23 MB of memory and no
GPU.[^units]

Of the systems in Table H1, ours has the lowest whole-model verification time for any model above one billion
parameters, and our model is as large as any (Figure 1).

Proving is where we pay: 63 to 80 minutes on one H100 for 16 tokens, against 133 minutes for the earlier version's one
token (Section 6.7).

This is a systems contribution. The building blocks are published (Section 8); we combine them at the size of a
400-billion-parameter mixture-of-experts model.

### Table 2 row removed later on 2026-11-26, to keep the table under a quarter page

| Code public | verifier^i^ | yes |

^i^ Not the prover (Appendix G).

## From Section 3

From **Acceptance.**

Since attention is causal, this is greedy decoding of `K` tokens.

One registration serves every split of the window.

From **The anchor.**

The policy bytes hold the shapes, the configuration and the versions; for the first and the last receipt they also say
which end of the model it holds and whether the model emits one token or several.

Under the anchor sit the 8-bit weights, the per-row exponent tables, the norm weights, the shifts fixed by calibration,
the nonlinear, rotary-position, embedding and output-head tables, and the circuit layout. The model anchor,
`Blake2s(MODEL_ANCHOR_V1 || n || anchor_0 || ... || anchor_{n-1})`, fixes the number and the order of the receipts.

From **A trust assumption on the anchor.**

LogUp's soundness needs the last of these, and the integer relations need the others.

From **Who trusts what.**

A dishonest registrant can register a circuit in which false statements have accepting proofs: with a lookup count at
or above `p`, for example, `p` copies of a wrong entry cancel in LogUp. Section 5's bounds therefore hold against a
dishonest prover only when the registration was honest.

From **Recomputing the anchor.**

A recomputation needs the weights (about 800 GB in 16-bit form), the calibration file, the exporter and the prover at
pinned commits, and the registration parameters. Our register-only run printed the anchors the proving runs later
registered; that is our code on our export, not an independent recomputation (Appendix F).

From **What a proof does not show.**

It does not show that the prompt tokens are authentic or came from a given user, that the model is good at any task,
what the float model would generate after the same prompt, decoding other than greedy, or anything after the last
generated token.

## From Section 4

### 4.1 Integer arithmetic of a block

Sites whose outputs can saturate carry explicit overflow terms.

Registration refuses shifts outside windows that keep every relation below `2^30`. A narrow site with a `b`-bit output
needs `b + s + e_j <= 30` for its shift `s` and each row exponent `e_j` (Section 4.3), and a wide site needs
`s + e_j <= 20`.

The row maximum is witnessed as an upper bound on every causal key and as attained on one of them.

**SwiGLU and other tables.** SiLU on 16-bit inputs uses an 8,192-entry table, exact piecewise. Registration generates
the rotary-position tables.

### 4.2 Expert routing

We follow Llama 4's reference routing: the router's sigmoid weight scales the expert's input, not its output. In an MoE
layer, with `x2` the normalised input:

    r[b, e] = 16-bit router logit of token b for expert e, for all 128 experts
    e*[b]   = the smallest e with the largest r[b, e]
    w[b]    = SIGMOID_2048[clamp(r[b, e*], -1024, 1023) + 1024],  in [0, 256]
    x_rt    = floor((w[b] * x2[b] + 128) / 256)
    h_out   = h_mid + routed_expert_{e*}(x_rt) + shared_expert(x2)

It reads the routed projections and their per-row exponents from the committed weights of all 128 experts through that
column, and a test shows that a token read with another expert's rows is refused.

### 4.3 Per-row power-of-two weight scales

A weight matrix `W`, whose rows are outputs, is quantised with one scale per tensor, `sigma`, and one exponent
`e_j >= 0` per row `j`:

    Wq[j, k] = clip(rint(W[j, k] * sigma * 2^(e_j)), -127, 127)
    acc[b, j] = sum_k x[b, k] * Wq[j, k]
    y[b, j]   = floor((acc[b, j] + 2^(s_j - 1)) / 2^(s_j)),    s_j = s + e_j

### 4.4 CPRS: a weight commitment verified inside the outer proof, for a weightless verifier

Four consecutive Mersenne-31 weights form one symbol of `QM31`, whose multiplicative group has a subgroup of order
`2^33`, since `2^33` divides `p^4 - 1`, so a plain Reed-Solomon code over `QM31` on a power-of-two domain encodes the
packed table.

Every challenge and every folded value lies in `QM31`, the outer STARK's own challenge field.

The code has rate 1/4, and an opening checks 139 queries on codewords of `2^30` symbols for the weights and `2^21` to
`2^31` for the witness.

Each receipt's weights form one committed group of 199 to 203 slots of `2^30` entries, and the witness forms separate
groups.

### 4.6 The outer Circle STARK

The outer proof has four traces: the in-circuit verifier, a Blake2s message scheduler, the Blake2s rounds and a bridge
between them, of log sizes 21, 18, 22 and 21 in the measured receipts.

We build the outer proof with S-two's `stwo` library, version 2.2.0, plus 23 commits of ours (Appendix F). The prover
compiles the verifier's own constraint files, so the two cannot disagree on a constraint.

### 4.7 Folding the ends and chaining the receipts

The first receipt also proves the embedding: the token ids select rows of a registered table stored as two 8-bit limbs
with one fixed scale, `E = E_a + 255 * E_b`.

A receipt with an end has a public input/output digest, a Blake2s hash under its own domain. It covers `L`, `K` and the
tokens the receipt binds (the prompt and the fed-back tokens, or the generated tokens), or the boundary commitment at
its other end. The circuit computes it and absorbs `L`, `K` and the tokens before any challenge, and the verifier
recomputes it from the caller's tokens and the boundary commitments.

Consecutive receipts share a boundary commitment: the output commitment of receipt `k` must equal the input commitment
of receipt `k + 1`.

## From Section 5

### 5.1 Parameters and the bound

The union bound sums the error of every round: the end stages, the stacked witness and weight folds, both commitment
openings, every LogUp-GKR round, the outer STARK's rounds and the two boundary hashes of the chain.

None uses the Johnson regime or a conjecture up to capacity.

### 5.2 Readings of the same rounds

A union bound over the rounds of a Fiat-Shamir-compiled protocol is sound only with
round-by-round soundness of each part, which assumption 4 of Section 5.3 takes without proof.

Every reading bounds the chance that one attempt succeeds. Against a prover that makes `Q` queries to the hash,
round-by-round soundness gives about `Q` times the weakest round's error, so `2^k` queries gain about `k` bits on that
round; without it, Fiat-Shamir on a multi-round protocol can lose far more [53].
Here it is the outer LogUp: `2^-87` per receipt, or `2^-103` per hash evaluation once its 16 bits of proof of work are
counted (reading (c) of Table A2).

### 5.3 Unproven assumptions

Every figure of this section rests on six unproven assumptions: the five our accounting script lists, and one that
wrapping needs.

3. The fold theorem of CPRS (Appendix A) bounds the interactive fold phase of one opening at the unique-decoding
   radius, relative to the proximity gaps of [22]; its proof follows Haböck's analysis of Basefold [45]. Its proof is in our design notes and is not reproduced here. It does
   not cover the batching of witness groups of different lengths into one opening, the Fiat-Shamir compilation or proof
   of work, and it has no version for the Johnson regime.

6. Wrapping needs the outer Circle STARK to be knowledge sound in the random-oracle model: its plain soundness says only
   that an accepting inner proof exists, which is not enough for a non-interactive inner argument. We assume it.

### 5.4 Corrections and open issues

None of the
fixes changed a receipt byte, an anchor or the transcript, and the verifiers built with them accept proofs made before
them unchanged.

### 5.5 Testing

FlashProve's test suite gives its verifier no proof made by a cheating program [1].

The outer circuit, the in-circuit check of CPRS, the lookups and the verifier have 63 listed constraints and checks.

Four runs that removed checks from the source caught all 29 removals, and the verifier rejected every tampered input on
our four bundles.

### 5.6 Open questions for the cryptographic review

8. Is the outer Circle STARK knowledge sound in the random-oracle model with our 8-to-1 folds and periodicity sample,
   and is replacing the oracle by Blake2s inside the outer circuit acceptable here?

## From Section 6

### 6.1 Setup

On the laptop, an Apple M5 Max on mains power, FlashProve's verifier and ours on all four bundles ran in one session,
each from spawn to exit, with 3 warm-up and 30 timed runs and the load and power state recorded before and after each
block.

### 6.2 The verifier's cost

A laptop verifies the first proof in 47 ms from process start to exit, with 23 MB of peak memory, and the cloud host in
37 ms inside the process (Table 3).

**Table 3. Verifier cost of the first proof, 4,643,682 bytes in three receipts. Whole process: median of 30 laptop
runs; in process: median of 31 warm runs after a cold one, laptop, and of 21, cloud.**

| Measure | Laptop | Cloud host |
|---|---:|---:|
| Whole process (ms) | 46.63 | –^a^ |
| In process (ms) | 20.25 | 37.4 |
| Peak memory (MB) | 22.9 | 16.0 |

^a^ Not measured.

### 6.3 Where the proof can be checked

In one session, with the files already in memory, the verify call took 31.1 ms at the median (30.4 ms at the minimum) in
Chromium 156, 33.3 ms under Node 24 and 25.6 ms for the same wrapper built natively: WebAssembly took 1.22 and 1.30
times as long as native.

### 6.4 Receipt size against tokens

**Table 4. One receipt at a fixed product of tokens and layers, with synthetic weights at the
real shape, the earlier receipt format, and receipts of 1 to 8 layers without the model's ends.**

| Tokens | Layers | Receipt bytes | Verify, cloud host, warm | Prove | GPU memory peak |
|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 1,508,808 | 13.9 ms | 617.3 s | 85.0 GB |
| 64 | 4 | 1,506,216 | 14.0 ms | 316.3 s | 76.6 GB |
| 128 | 2 | 1,471,864 | 13.3 ms | 224.0 s | 50.5 GB |
| 256 | 1 | 1,465,176 | 13.8 ms | 184.9 s | 33.0 GB |

Also in the paper as Table 4, without the Prove and GPU columns.

They measure size, time and memory, not accuracy or security, and the share of MoE layers differs between rows (Appendix
F).

### 6.6 Comparison with FlashProve

On this pair its proof was 18,790,612 bytes, 4.04 to 4.07 times ours.

On the cloud hosts its verifier took 21.86 s and 3.53 GB as one process with its one-thread setting; ours took 30.6 to
42.1 ms and 15.3 to 16.0 MB inside the process, which is not the same measure.

Two other sessions gave 250.8 (its one-token request) and 262.2 (on battery) (Appendix D).

We have no receipt shape for a 1,024-token window: at 16 layers per receipt the lookup count reaches `p` before 96
tokens (Section 4.5), and our longest receipt covers 256 tokens of one layer, on synthetic weights (Table 4).

Our earlier version and FlashProve proved statements of the same form, one token after a 32-token prompt, on different
weights (Appendices D, E).

### 6.7 Where the prover's time goes

**Table 5. Prover wall time by phase over the three receipts, for the first proof and the earlier version, one run
each.**

| Phase | Ours (s) | Ours (%) | Earlier (s) | Earlier (%) |
|---|---:|---:|---:|---:|
| Load | 347.3 | 7.6 | 623.8 | 7.8 |
| Register^†^ | 701.7 | 15.3 | 1,525.6 | 19.2 |
| Witness commit | 126.6 | 2.8 | 91.5 | 1.1 |
| Lookups | 356.0 | 7.8 | 434.4 | 5.5 |
| Sum-checks^a^ | 1,248.3 | 27.3 | 1,026.2 | 12.9 |
| Weight fold | 620.8 | 13.6 | 2,918.6 | 36.7 |
| Witness folds | 305.4 | 6.7 | 259.2 | 3.3 |
| Weight opening | 660.1 | 14.4 | 785.3 | 9.9 |
| Outer STARK | 204.6 | 4.5 | 290.4 | 3.6 |

^†^ Once per model; our runner repeats it. ^a^ With the witness opening. Walls 4,574.5 and 7,959.8 s (3.8, 4.8 s
untimed).

Also in the paper as Table 5, for the first proof only.

The rewritten weight fold saved 2,298 s, two thirds of the 3,385 s saved; registration fell from 1,526 to 702 s, and we
have not found why. The prover peaked at 339 GB of host memory, 22 GB below its request; the GPU's 85 GB is its buffer
cache filling the card. Appendix F gives per-receipt times, our predictions and the cloud bills.

Across the four proofs the receipts and the verifier's cost hardly moved, and the prover's time did (Table 6). The
receipts are 4,621,618 to 4,654,514 bytes, within 0.5% of the first proof's, and the cloud hosts verified each in 30.6
to 42.1 ms. The prover took 3,779.8 to 4,778.4 s, or 63 to 80 minutes, and peaked at 339 to 340 GB of host memory. The
work was the same: every count the prover prints, 71 per receipt, was equal between the first and the second proof. The
machines were not, and the phases moved unevenly. On the second, our build took 13% less time and the per-layer proving
and lookups 37.5% less (Appendix F), while the weight fold's round on the host processor took 8.8% more. Registration
did the same work in every run, giving the same anchors, yet took 701.7 to 1,176.2 s. The logs do not show the machines'
processors or memory, so why one was faster is not explained. These are single runs (Appendix F).

**Table 6. The four proofs, one run each, each on a different H100 machine.**

| Measure | Prose | Python | Dialogue | French |
|---|---:|---:|---:|---:|
| Receipts (B) | 4,643,682 | 4,642,210 | 4,654,514 | 4,621,618 |
| Verify, cloud (ms) | 37.4 | 30.6 | 33.2 | 42.1 |
| Load (s) | 347.3 | 354.0 | 487.7 | 487.4 |
| Register (s) | 701.7 | 783.1 | 1,004.0 | 1,176.2 |
| Prove (s) | 3,522.4 | 2,640.6 | 3,139.1 | 3,111.2 |
| Wall^a^ (s) | 4,574.5 | 3,779.8 | 4,634.0 | 4,778.4 |
| Host (GB) | 338.9 | 339.0 | 339.1 | 339.7 |
| GPU (GB) | 85.0 | 85.0 | 85.0 | 84.7 |

^a^ Includes 2.2 to 3.5 s outside the three timers. Verification: warm median of 21 runs in one process on the host
that proved it.

Also in the paper as Table 3, with laptop verification and the float model's agreement in place of cloud verification,
load and GPU memory.

### 6.8 Fidelity to the float model

Chinese texts fare worst: 13.28% and 12.50%, and 8.20% for the third, tied with two English texts.

**Table 7. Top-token disagreement with the float model on 7,680 held-out positions. Our Python integer reference of
the 48 layers, read through the model's float output head.**

| Integer model | Top token differs |
|---|---:|
| One scale per tensor (earlier)^a^ | 889 (34.73%) |
| Per-row scales, output and down projections | 582 (7.58%) |
| Per-row scales, all matrices and router (ours) | 379 (4.93%) |
| The same, router in two limbs | 350 (4.56%) |
| Control: float weights, integer activations | 316 (4.11%) |

^a^ The first ten texts only, 2,560 positions.

Also in the paper as Table 6, with FlashProve's rates on the same texts and without the output-and-down row.

**Table 8. Top-token disagreement with the 16-bit weights, step by step, as each side measures it; not like for like.**

| Step | Ours^a^ | FlashProve, reported^b^ |
|---|---:|---:|
| Weights alone | –^c^ | 6.65% (5,643) |
| Integer arithmetic | 4.11% (316) | 3.11% (2,638) |
| Whole integer model | 4.93% (379) | 6.81% (5,779) |

Superseded in the paper by the run of FlashProve's released integer model on our 30 held-out texts (Section 6.8).

^a^ 30 held-out texts, 7,680 positions, against float64. ^b^ Its documentation: 216 WikiText-2 texts, 84,893
positions, against float32, from 4-bit files refit to 13 bits, by programs not in its repository [1]. ^c^ Not run.

**Table 9. The float model on the four proved statements, given each proof's earlier tokens.**

| Statement | Agree (of 16) | First difference^a^ | Smallest margin^b^ |
|---|---:|---:|---:|
| Prose | 16 | none | 0.045 |
| Python code | 16 | none | 0.156 |
| Dialogue | 15 | 4 | 0.011 |
| French | 14 | 1 | 0.025 |

^a^ Generated token, counted from 1; the French statement differs again at 12. ^b^ The float model's top probability
minus its second, over the 16 positions.

Its agreement column is also in the paper, as the last row of Table 3.


### 6.1 Setup (cut while splicing)

Our Python reference computed each statement's 16 tokens beforehand, correcting a draft at its first
mismatch (13, 13, 13 and 16 passes), and the prover fails unless its head emits them.

### Cut later on 2026-11-26, to make room for the like-for-like fidelity run

From 6.1: The prover's wall time covers loading, registration, proving and its own check of each receipt.

From 6.3: In Chromium it accepted the proof and rejected tampered inputs, and under Node it matched the native verifier
on the honest bundle and eight tampered ones.

From 6.6: Its prover's lead is largest per token: 145.7 s for its 1,001-token request, 1,600 to 2,100 times faster than
ours per token.

From 6.7: Against the earlier version the wall time fell from 7,960 to 4,575 s while the output grew from one token to
16 (Appendix F).

From 6.5: The gain comes from fewer receipts, since each costs about the same: a synthetic 8-layer receipt is
1,508,808 bytes and 13.9 ms, a 16-layer receipt of the first proof 1,544,608 to 1,550,569 bytes and 12.44 to 12.48 ms.

From 6.6: Two other sessions that day gave 250.8 (its one-token request) and 262.2 (on battery) (Appendix D).

From 6.6: Under its one-thread setting it used 1.31 cores, ours 0.81.

From Section 7: The prover is slow: 63 to 80 minutes and 339 to 340 GB of host memory on one H100 for 16 tokens, near
the cloud platform's largest memory request.

## From Section 7

Recomputing the anchor and publishing the calibration answer these; no third party has done either.

At 16 layers per receipt the lookup bound keeps a receipt under 96 tokens, while FlashProve proves a 1,024-token window
and Attestable reports support for "up to 16K tokens in the context window".

FlashProve's documentation puts its own prover at 20 to 93 times the cost of serving Maverick's FP8 release with stock
vLLM [1]; we have not measured ours against serving.

## From Section 8

NanoZK's paper also gives 3.4 s and 22 ms for its 12-layer model of width up to 256; we use the 41 s and 4 s it reports
as measured, and that model is far below one billion parameters. Our 4.6 MB proof is among the smaller ones but not the
smallest: zkLLM reports 188 kB for LLaMa-2-13B, zkGPT a proof its table gives as 101K for GPT-2 and zkTransformer
4.31 MB for GPT-2.

**Proofs of inference.**

Attestable [2] proves `y = F(W, x, r)` for committed weights; for models of 8 to 70 billion parameters it reports proofs
of 5.2 to 5.8 MB checked in 205 to 251 ms, as "alpha results" with "100 bits of security".

**Foundations.**

Lasso [35] and its successor Shout [36] handle lookups into very large tables, by decomposing them and by one-hot
addressing; our tables are small, so we register them whole and use LogUp-GKR.

## From Appendix A

### Parameters (prose after Table A1)

Each receipt's weight group has 203, 199 and 203 slots of `2^30` entries, against a limit of 255, and the weight fold
reduces 822, 816 and 855 claims on 288, 285 and 321 regions of those slots. The in-circuit verifier's largest row count,
2,023,242, is 73,910 rows (3.5%) under `2^21` (Section 4.6).

### Readings (prose before Table A2)

For the first proof, (a) recomputed by hand gives the script's 85.59 bits per receipt and 84.10 for the chain.

Of two ways of summing, we keep the lower result: for the earlier version, 96.10 against 96.13 per receipt, and 94.54
for the chain in both.

It holds only under that model, and we give it for comparison.

### The strictest reading and the shrinking-domain form (prose after Table A2)

With the shrinking-domain form of the fold theorem of CPRS in place of the form the accounting cites, (a) and (b) become
86.01 and 99.54 bits per receipt, and 84.42 and 98.14 for the chain, for the earlier version, and 86.00 and 101.95, and
84.42 and 100.93, for the first proof.

### Inputs (prose after Table A3)

Our accounting of the measured proofs (not released) reads every input the prover
prints from the four proofs' logs, equal in all four, and gives these readings unchanged; its inputs table names where
each count came from.

### How the slots are combined

Charged that way for the weights alone, reading (a) would fall to 84.92 bits for the worst receipt and 83.40 for the
chain, and reading (b) to 100.72 and 99.55.

### The fold theorem of CPRS

A written proof, through four lemmas and relative to that theorem, is in our design notes (not released); it was checked by machine
at toy sizes, and no human cryptographer has read it, so we do not present it as established.

# Part 2. Moved from the appendices

## From Appendix B: Removal tests of the inner proof

**Removal tests of the inner proof.** Parts of the inner proof have removal tests, in which a constraint is removed in a
scratch copy and a test must fail: the lookup tree (six removals, each of which let a forgery through and each caught),
the end stages (35 of 35 source mutations caught) and the per-row exponents. Four constraints of the per-row exponents
had no guarding test; they have had one since the commit that made the earlier version's proof. At that commit the outer
circuit had no such coverage (Fixed issues, below). A removed outer constraint would then have gone unnoticed by our
tests; that was a gap in evidence, not a known attack.

## From Appendix B: Necessity tests

Later commits, test code only, strengthen some pairs. Four checks that had a pair at one site now have one at another.
They are the proof-of-work nonce before an inner fold (`D01`), fold consistency between inner oracles (`D08`), the last
layer's sum-check (`D12`) and the bound on prompt tokens (`D19`). The pair for `D05B`, the base-field check of the
commitment's coordinates, showed only that the check is live, with a forgery that kept the claim true; its new pair
carries a false claim. `D01` and `D08` are also paired on a commitment instance of two groups, and the commitment pairs
otherwise use one group. `B13` was paired with a wrong inverse. It is now paired with a real out-of-range word: a search
found, in 1.7 s, the lowest 8-byte counter whose Blake2s-256 digest has a word of at least `2p`, 10,806,609. The test
hashes that message as a free witness and squeezes its digest. With `B13` dropped the circuit accepts it; with `B13`
kept our prover refuses it and the verifier rejects it. In four runs that removed checks from the source (8, 10, 3 and 8
removals), every removal was caught.

## From Appendix B: Fuzzing of the verifier's input handling

**Fuzzing of the verifier's input handling.** An in-process fuzzer (not released) mutated
three bundles of an earlier receipt format (version 10), of 3, 8 and 1 receipts, for 600, 300 and 300 s. It first swept
every header field at boundary values, the structure of each decoded STARK proof and the manifest. It then made random
bit flips, truncations, appended bytes, copied windows and huge 64-bit values. Of 39,974 mutants, 37,714 were refused
and 2,260 left the bytes unchanged and were accepted. None was accepted with other bytes, none panicked, none took over
5 s or 1 GiB, and the largest heap was 17.9 MB. Through the command-line binary, the verifier refused at once a FIFO or
a directory given as a receipt. It also refused at once a receipt or a manifest one byte over 64 MiB, a length of `2^63`
and a manifest of `2^20` tokens. In a legacy bundle format, 12 of 300 mutants were accepted with changed router outputs,
which that format did not bind; the verifier now refuses such a bundle when it carries them (Fixed issues, below). The
fuzzer has not run on the receipt format of the first proof.

## From Appendix B: Validation of the code of the four proofs

The later commit `b48f17bf`, with the later paired tests, passed 1,103 prover tests and 48 verifier tests, none failed.

Our release-candidate notes (not released) list every fix with its commit and every open issue with the reason it is open.

## From Appendix B: Fixed issues

**Fixed issues.** Our code had these issues, now fixed. Most were in the code that made the earlier version's proof
(Section 5.4).

- Called without `--tokens` and `--out-token`, the command line checked the bundle's own statement, so a caller who left
  them out could accept a false one. It now requires both.
- The caller could not require a security profile. `--profile` now exists on the model, single-receipt and chain
  commands, and the name fixes the outer parameters only (Section 3).
- At the verifier's own log-size limits, though not at the measured sizes, the outer XOR lookup count could reach `p`.
  The verifier now refuses those log sizes.
- A legacy bundle format did not bind its router outputs, and fuzzing found accepted changes to them. The verifier now
  refuses such a bundle when it carries router outputs.
- The documented procedure took the anchor from the prover, and environment variables for unchecked flags were ignored.
  The procedure and the benchmark now take the anchor from the registration, and such variables are refused.
- The verifier accepted a chain starting at any layer. A chain must now start at layer 0.
- The statement check accepted an unblinded trace in the measured profile, end stages in another statement type, and
  receipts of one model with different commitment folds or dimensions. Each is now refused, with a test.
- Our first security accounting took the weakest round as the level, which overstated it. The union bound now leads, and
  weakest-round figures appear for comparison only (Appendix A).
- Our soundness note claimed zero knowledge, which its written argument does not support. It no longer does, and we
  claim none.
- 10 of the 22 transcript absorbs that soundness rests on could be removed with every test passing. A golden transcript
  test now catches them.
- Most single-constraint removals in the outer circuit and the in-circuit check of CPRS went unnoticed by our tests:
  51 of 62 in one count and 27 of 28 in another. One removal cut the check of CPRS from 139 queries to 1. Necessity
  tests now cover them, and the verifier crate now tests a real receipt (Section 5.5).
- Four in-circuit checks of CPRS and one base-field check could be deleted with no test noticing. Necessity tests
  now cover them (Section 5.5).
- Before the multi-token code was merged, 23 of its constraint removals went unnoticed by tests. A test now notices
  each.

## From Appendix C: The first proof

The prover's tests include multi-token tamper cases on a three-receipt toy bundle, each refused: a changed generated
token, the same tokens read as another split of `L` and `K`, a truncated or extended list of generated tokens, the
receipts of another split, a head receipt of another split chained in, swapped receipts, a single-token head receipt,
and the single-token output flag.

## From Appendix C: The earlier version

**The earlier version.** The verifier built for its run rejected each of 13 tampered inputs with the reason one would
expect: a changed prompt or output token, on the command line or in the manifest; a wrong model anchor; one flipped byte
in the first or the last receipt; receipts reordered in the manifest or their files swapped; and a receipt of the
earlier format in place of a current one, with and without its manifest line. The verifier built from the first release
candidate (it reports commit `251c0e9f`, not dirty) accepted the bundle under the caller-pinned command of Appendix G
with `--profile production_hardened` and rejected all 13 tampered inputs with the same reasons; so did the verifiers
built at later commits, up to the code of the first proof. Every file of the bundle had the same SHA-256 before and
after, and the model anchor, the receipts digest and the final chain equal the record of the run.

## From Appendix D: The runs

Its documentation describes those tokens as each the model's first choice after the tokens before it.

For this request the weight commitment was made again, because the earlier runs did not keep its secret key.

## From Appendix D: Table D1

**Table D1. Every measurement of Section 6.6, by us, one column per run. The statements and weights differ (see below);
a dash marks a figure not taken.**

| Measure | Ours, earlier | FlashProve, 1 token | Ours, first proof | FlashProve, 16 tokens |
|---|---:|---:|---:|---:|
| Proof (bytes) | 4,632,530 | 18,772,884 | 4,643,682 | 18,790,612 |
| Prove, wall (s) | 7,959.8^a^ | 98.7^b^ | 4,574.5^a^ | 63.4^b^ |
| One-time per model (s) | 1,525.6^c^ | 1,941.1^d^ | 1,180.5^e^ | 1,942.0^d^ |
| Prover host memory (GB) | 292.2 | 261.0 | 338.9 | 261.0 |
| Prover GPU memory (GB) | 85.0^f^ | 73.7 | 85.0^f^ | 73.7 |
| Laptop, whole process (ms)^l^ | – | – | 46.63 | 12,123.5 |
| Laptop, in process (ms) | 19.29^h^ | – | 20.25^h^ | – |
| Laptop memory (MB)^l^ | – | – | 22.9 | 4,009.5 |
| Laptop, 2026-11-23, whole process (ms) | 76.8^g^ | 10,490 | – | – |
| Laptop, 2026-11-23, memory (MB) | 26.1 | 3,990 | – | – |
| Laptop, earlier session, whole process (ms)^m^ | – | 11,504.6 | 45.88 | – |
| Laptop, earlier session, memory (MB)^m^ | – | 4,047.9 | 22.8 | – |
| Laptop, on battery, whole process (ms)^k^ | – | – | 65.05 | 17,057.1 |
| Laptop, on battery, memory (MB)^k^ | – | – | 27.9 | 4,066.9 |
| Cloud verify (ms) | 42.4^i^ | 24,530^j^ | 37.4^i^ | 21,860^j^ |
| Cloud verifier memory (MB) | 15.2 | 3,540 | 16.0 | 3,530 |

^a^ Loading, registration, proving and unattributed time: 623.8, 1,525.6, 5,808.0 and 2.4 s for the earlier version;
347.3, 701.7, 3,522.4 and 3.2 s for the first proof. ^b^ Per request, start-up included. ^c^ Registration inside the
proving call; no separate run. ^d^ Its weight commitment in the same run, made again for the 16-token request, at 266.0
and 265.9 GB; 1,911.0 to 3,155.9 s over four runs. ^e^ A register-only run at 237.8 GB; the proving run repeats
registration (701.7 s). ^f^ The cache fills the card. ^g^ Median of 5 inside `time -l`, which adds about 35 ms. ^h^
Warm: the earlier version's the median of 5 round medians over 405 runs, cold 19.98 ms; the first proof's the median of
31 warm runs in the session of note l, cold 20.14 ms. ^i^ Warm median in process. ^j^ Whole process with `-j 1`; the
16-token run used 24.26 s of CPU, 1.11 cores. ^k^ One session on 2026-11-25, 22:07 to 22:16, on battery power for all
but its last 83 s; median of 30. Ratios 262.2 at the median, 195.9 at the minimum, 145.9 in memory; its verifier used
1.30 cores at the median (1.24 to 1.35), ours 0.84. ^l^ The paper's laptop session: 2026-11-25, 23:13 to 23:20, on mains
power throughout, median of 30. Ratios 260.0 at the median, 282.7 at the minimum, 175 in memory; cores 1.31 (1.30 to
1.33) and 0.81 (0.68 to 0.89). ^m^ 2026-11-25, 01:10 to 01:23, on mains power; its one-token request.

## From Appendix D: The runs' statements and claims

**The runs' statements and claims.** Our earlier version proves that the registered integer model maps 32 tokens to one
output token; FlashProve's one-token request covers position 32 in its fixed 1,024-token window. Each of our proofs
covers 16 tokens generated after a 16-token prompt in a 32-token window; FlashProve's 16-token request covers positions
24 to 39, each the model's own first choice. Ours use real weights, our integer export of the 16-bit release, and
FlashProve's runs synthetic weights from its generator at Maverick's shape. For the chain, our accounting gives 84.14
bits without proof of work and 94.54 with it for the earlier version, and 84.10 and 100.49 for the measured proofs,
computed from their counts, under the same six assumptions. FlashProve's verifier prints a least round of 96.23 bits
without proof of work and 128.00 with up to 32 bits of it counted, and its documentation says neither is a security
level. Its authors claim zero knowledge and publish its code under the Apache 2.0 licence; we claim no zero knowledge,
and we release the verifier and the proofs (Appendix G).

The whole-process figures of the earlier version and FlashProve were taken in one session of five rounds with the order
rotated, its verifier with its one-thread setting (`-j 1`) and ours with its default settings, which use one thread. Our
76.8 ms includes about 35 ms of the timing wrapper, so the ratio of 137 understates. On the wrapper's own real time,
whose 0.01 s resolution is coarse for ours, the ratio is 262, and we take the true ratio to lie between the two. On the
wrapper's own real time the two took 40 ms and 10.48 s. Three laptop sessions on 2026-11-25 timed both verifiers by
spawn to exit. At 01:10, on mains power, its verifier took 250.8 times as long as ours at the median and 260.3 times at
the minimum (10,901.5 against 41.89 ms), on its one-token request. At 22:07, on battery power, it took 262.2 times as
long on the 16-token pair, and both sides ran about 40% slower. At 23:13, on mains power, it took 260.0 times as long on
that pair, and 282.7 times at the minimum; the paper's laptop figures are this session's. Ours agrees within 2% between
the two mains-power sessions, 45.88 and 46.63 ms; an earlier session on 2026-11-25, on battery power between 12% and 2%
charge, timed ours at 62.45 ms. FlashProve's verifier spent about the same on one covered token as on 1,001: its stored
1,001-token proof took 10.64 s and 4.02 GB, 1.4% longer at the median, and 11,504.2 ms (minimum 11,106.4 ms) and 4.03 GB
in the session of 2026-11-25. It reports no in-process time, so we do not set ours against its whole process.

## From Appendix D: What each side documents

**What each side documents.** Ummon's weights are 8-bit integers with per-row power-of-two exponents and a 16-bit head,
its activations 12 to 16 bits with a 24-bit residual stream, and its window 32 positions. FlashProve's reported results
use 13-bit integers refit from a third party's 4-bit files, and its documentation records one run each from 8-bit and
16-bit files. Its activations are 12 bits with sign for norm outputs, queries, keys and values, with 16-bit up
projections and a residual stream below `2^39`, and its window is 1,024 tokens [1]. It states its security as the whole
bound in one number, 126.21 bits with proof of work counted and 96.23 without, per hash computed, and about
`q/2^124 + 2^-101` for its Lean verifier. It claims zero knowledge, proved in Lean for its protocol. Its verifier checks
the weight openings itself, against a key file and a leaf file per commitment; ours checks none (Section 4.6). Per
covered token, on its 1,001-token request, its laptop verification took 11.5 ms and its proof 18.8 kB, against our 2.87
ms in the same session and 290 kB per generated token, and its prover peaked at 261.0 GB of host memory against our
338.9 GB.

## From Appendix D: Other figures

**Other figures.** A new verifier of ours downloads the bundle, 4,644,609 bytes, and takes the 32-byte anchor and the
tokens from the caller. FlashProve's needs, once per commitment, a key file and a leaf file of 6.24 MB (documented) and
three configuration files of 0.23 MB in its stored full-size example, about 25.2 MB with its first proof. Its proofs
were 4.04 to 4.07 times the size of ours on the 16-token pair. At the minimum of 30 runs on the 16-token pair its
verifier took 11.68 s and ours 41.33 ms. Its public 4-bit files are the publisher's `UD-Q4_K_XL` directory of 232 GB.
Its documentation reports 1,851 proofs on them, a steady 111 tokens per second at 24 requests per batch, and one run at
8 bits, not on its released programs, that peaked at 458 GB to prove and 493 GB to commit [1].

## From Appendix D: Proving once and checking many times

For one check, FlashProve's total is 75.5 s and ours 4,574.6 s, 60.6 times as long.

Our prover's wall time includes loading the model and registration; FlashProve's excludes a one-time weight commitment
of 1,911.0 to 3,155.9 s. The counterpart of that commitment is our register-only run, which for the measured model took
1,180.5 s and 237.8 GB of host memory. For the earlier version, loading took 623.8 s and registration 1,525.6 s, and its
proving phase alone, 5,808.0 s, is 57 to 59 times FlashProve's per-request time. If a FlashProve request also paid for a
commitment (2,039.8 s with the 1,941.1 s of the same run), the earlier version would take 3.9 times as long.

For the first proof, the proving phase alone, 3,522.4 s, is 56 times FlashProve's 63.4 s on the 16-token request and 35
to 36 times its one-token requests. With that run's commitment counted for the one request, 2,005.4 s, our whole wall
time takes 2.3 times as long and our proving phase alone 1.8 times. Per token, our prover spent 285.9 s per generated
token. FlashProve's 16-token request spent 3.96 s per covered token, 72 times less, and its 1,001-token request 0.146 s,
about 1,960 times less, a rough estimate across different statements.

## From Appendix D: What each number includes

What each number includes:

- Our prove times include loading the model, registration (which reads no prompt and could run once per model) and our
  prover's own check of each receipt. Its prove time includes its start-up; its weight commitment is a separate step,
  run once per commitment.
- On the laptop, the earlier version's verifier and FlashProve's ran in one session, five rounds interleaved with the
  order rotated, each subject once per round, and every call accepted. The whole-process figures are like for like: one
  process, start to exit, on the same machine; its verifier with its one-thread setting (`-j 1`), ours with its default
  settings. The caller's clock wraps each process in `/usr/bin/time -l`, so it includes starting and ending that
  wrapper, about 35 ms above `time -l`'s own real time for ours; `time -l` real has a resolution of 0.01 s, too coarse
  for ours.
- Our in-process times exclude process start and reading the files (Section 6.1). We have no in-process figure for its
  verifier, so we do not set our in-process figure against its whole process.
- On the cloud hosts, our figures are the warm medians of repeated verifications inside one process (the cold figures
  there were 43.7 ms for the earlier version and 37.5 ms for the first proof); its figure is the wall time of one
  process with its one-thread setting, in the container that proved it. The two ran on Modal H100 containers of the same
  kind but on different machines, in different calls on different days; ours had 48 visible host cores, its container
  24. The first proof's cloud figures come from a third call, on another day. Its verifier took 10.4 to 11.3 s on the
  laptop against 24.53 s on its cloud host.
- The windows differ: its proof of the matching request covers one token but proves a 1,024-token window, whose
  positions 33 to 1,023 hold filler. A second request in the same run, with its stored window unchanged, gave 18,782,108
  bytes, 101.6 s and 24.30 s. Its own stored request, covering 1,001 tokens and proved again by its released programs,
  gave 18,774,932 bytes, 145.7 s and 20.74 s: its proof size does not depend on how many tokens are covered. Our proofs
  differ from every request we ran: each covers 16 generated tokens after a 16-token prompt in a 32-token window.
- Its run uses synthetic weights, ours real weights. Its proof on synthetic weights, 18,772,884 bytes, is close to the
  18.78 MB average that its documentation gives and to the ~19 MB that its research page reports for the public 4-bit
  weights. Its proof claims zero knowledge; ours does not. The two state their security in different terms.

## From Appendix E: Proving

**Proving.** The earlier version proved one token after the full 32-token prompt, on 2026-11-22, on the same kind of
host with a request of 343.6 GB. Its prover was built from commit `9857392b`, whose prover and verifier sources equal
those of commit `e3e2ac3d`. A proof of the same statement with the earlier arithmetic (one scale per tensor, 2026-11-21)
is 4,637,842 bytes and verified in 37.4 ms on the cloud host and in 18.83 ms in process on the laptop, in another
session at a one-minute load of 3.0. The earlier version read the same exported model files, with the same raw tensors
by the hash of their manifest and the same calibration file.

## From Appendix E: Laptop timing

**Laptop timing.** The earlier version's proof was timed in another session, on 2026-11-23, with the same lock, a
one-minute load of 4.6 to 6.5 and the verifier of commit `251c0e9f`. Its whole-process times were taken on the caller's
clock around `/usr/bin/time -l`, which added about 35 ms for our verifier, so they are not comparable with this
session's. Its in-process times come from five processes, each with a cold run and 81 warm runs; the verifier reports no
per-run minimum. FlashProve's verifier was timed in that session on a statement of the same form, on its synthetic
weights, one token after a 32-token prompt. It took 137 to 262 times as long as ours, with 138 to 153 times the memory,
and its proof was 4.05 times the size. Its prover was 78 to 81 times faster per request (Appendix D).

## From Appendix E: Verifier cost

**Verifier cost.** Its 4,632,530-byte proof, made on 2026-11-22, verified on the laptop in 76.8 ms over 5 runs inside
`/usr/bin/time -l`, which adds about 35 ms, and in 19.29 ms warm (median of 5 round medians, 405 runs) and 19.98 ms cold
in process, with 26.1 MB of peak memory in the five timed processes and 28.9 MB in the in-process runs. On the cloud
host it took 43.7 ms cold and 42.4 ms warm in process, 43.1 ms in the prover's own check, and 15.2 MB. Its receipts are
1,539,849, 1,554,512 and 1,538,169 bytes and verified in 14.09, 14.20 and 14.06 ms each on the cloud host; with the same
outer trace sizes, its proof is 5,312 bytes (0.11%) smaller than the earlier arithmetic's. In process, the first proof
took longer on the laptop than the earlier proof, in sessions two days apart.

## From Appendix E: Trace sizes and digests

**Trace sizes and digests.** The in-circuit verifier of its three receipts has 2,004,507 to 2,027,033 rows, under `2^21`
= 2,097,152; the largest is 70,119 rows (3.3%) under it. Its single-token receipts use a public input/output digest of
the same kind as the multi-token ones, under another domain.

## From Appendix E: Fidelity on its prompt

**Fidelity on its prompt.** On the earlier version's 32-token prompt, the exporter's final hidden state has a cosine of
0.999572 with the exporter's float32 reference, against 0.987097 for the earlier arithmetic. We did not check whether
the float model's top token after that prompt is also 290.

## From Appendix E: Verification

**Verification.** The earlier version's proof verifies with the single-token form of the command of Appendix G:

    ummon-verify model <bundle>/model_manifest.mdl2 \
      --model-anchor e817f6b11145dec5945e089d65422cacf9aa8971a597a475a6e9eb276723aea5 \
      --initial-chain 0:2424242424242424242424242424242424242424242424242424242424242424 \
      --tokens 200000,64,2752,16980,341,7550,2895,290,30404,75041,436,290,113841,262,6060,38794,436,290,4635,323,290,12824,511,23946,536,26,352,2432,7144,725,12294,436 \
      --out-token 290

Expected output: exit status 0 and a JSON verdict with `"accepted":true`, 3 receipts, 48 layers, 4,632,530 receipt
bytes and the final chain link `48:b6c2a63b5bd234272853516c27cc69e1750bdbfe9318a815721d791629fa41fb`. The verifiers
built from the first release candidate onwards, up to the code of the first proof, also accept it with
`--profile production_hardened` (Section 6.9).

Its anchor was pinned from the registration run of its proof. On 2026-11-25 we ran the register-only command at full
size on the cloud host, with code equal to commit `24f1aae6`. That commit registers with the earlier version's 8 bits of
proof of work before the inner folds. It read the exported model files and the calibration file that version's proof
read. It printed the model anchor of the command above, and its three receipt anchors equal the receipt lines of the
bundle's manifest. The run took 1,012.4 s, with 237.8 GB of host memory and 33.0 GB of GPU memory at peak. It is a
second run of our own registration code on our own export, with the weights' source, revision and calibration hash
recorded. It is not a recomputation by a third party or by independent code.

## From Appendix F: The S-two copy

**The S-two copy.** Our pinned copy of `stwo`, commit `69e6f9a0`, is upstream commit `f0d79b0f` plus 23 commits of ours
(Section 4.6). By their commit messages, seven of the 23 concern an example component for multilinear evaluations: the
four of our unmerged pull request starkware-libs/stwo#1438, two prover speed-ups and one that exposes constraint
degrees. The standalone verifier depends only on the library's core crate and constraint framework, not on the examples
crate. The other 16 concern the commitment scheme and the verifier. They harden the out-of-domain and per-round
Fiat-Shamir challenges, enforce the periodicity rule of Section 5.3 for every sampled short column, and avoid a
degenerate conjugate line. They also reject malformed or mismatched proof structures, salts for disabled query grinding
and zero-query FRI configurations; five of the 16 add only tests. Everything else in the copy is upstream code at
`f0d79b0f`.

## From Appendix F: Prover memory

**Prover memory.** The measured run registered each codeword without holding its group's table and re-derived the
queried leaves from coefficients (`STWO_ML_CPRS_CODEWORD_FREE=1`), kept Merkle levels above 32 MiB in a temporary file,
and ran the commitment's GPU kernels on the 8-bit weights in place (`STWO_ML_CPRS_GPU=1`). Three prover changes, none of
which alters a receipt byte or an anchor, lowered a 16-layer receipt's memory. The prover frees the weights after the
inner proof, since the outer STARK reads none of them. It streams the LogUp-GKR input layer in chunks, and the outer
prover drops its copy of the base trace once the interaction trace is built. The first proof's last receipt, with its
larger witness group, peaked at 337.0 GB (338.9 GB for the process), under a request of 360.8 GB, the largest the cloud
platform accepts for one function. With every weight alive, a 16-layer receipt was predicted to need 413.4 GB of host
memory. The weights of a 16-layer receipt with both ends, 226.3 GB, are freed after the inner proof; with the two other
changes the predicted peak fell to 282.4 GB. Streaming the LogUp-GKR input layer takes the tree from 92 to 32 bytes per
input entry: it builds the layer above it from the witness in chunks of 4,096 entries and drops each stored layer once
read. The earlier version's receipts peaked at 276.7, 287.1 and 292.1 GB in the full run (292.2 GB for the process), and
at 281.3 GB in the dry run, under a request of 343.6 GB. In the first proof the first two receipts peaked at 276.7 and
287.0 GB, and the last at 337.0 GB (sampled). Before the run we predicted about 310.3 to 312.5 GB for the last receipt,
whose witness is one group of `2^31` symbols; a dry run of that receipt alone reached 321.9 GB.

## From Appendix F: The weight fold and the lookup pool

**The weight fold and the lookup pool.** After the earlier version we rewrote the prover's weight fold to read each
committed region once for all its claims and, on the GPU, to copy each batch of slots once per region. Tests against a
verbatim copy of the old fold show the same transcript values, receipt bytes and anchors. Our earlier design committed a
lookup pool, which a census of the code at the real shape (2 layers, under an earlier security profile) found to be
87.4% of the committed field entries; the published protocol commits only the multiplicities.

## From Appendix F: Prover time per receipt

**Prover time per receipt.** Proving took 1,158.9, 1,164.8 and 1,198.6 s per receipt in the first proof, against
1,985.6, 1,861.6 and 1,960.8 s in the earlier version. Per generated token the prover spent 27.8 times less than the
earlier version's. The rewritten weight fold saved 2,297.8 s of the 3,385.3 s saved. The outer STARK took 28 to 31% less
time per receipt, and the last receipt's witness commitment, which holds the multi-token head, about twice as long (66.6
s against 30.3 s).

## From Appendix F: The four proofs

**The four proofs.** The four calls ran prover code whose sources equal those of `582c7abd`, launched from `ee1ad727`,
`43b27a0c`, `2573d471` and `ac898234`, each on one H100 80GB HBM3 with driver 620.70.06. The GPUs' bus addresses
differed (`CA:00.0`, `00:0E.0`, `04:00.0` and `00:0D.0`), so each ran on a different machine. Each proved the same
registered model, and its receipt anchors equal those of the register-only run. The second to fourth prompts are the
first 16 tokens of the held-out texts `long_03_python`, `long_04_dialogue` and `long_07_french`. Their generated tokens
take 16, 14 and 16 distinct values, and 13, 1 and 0 of them equal the text's own next tokens. Proving took 877.4, 813.5
and 949.7 s per receipt in the second proof, 1,027.8, 982.1 and 1,129.3 s in the third, and 1,016.8, 930.8 and 1,163.6 s
in the fourth. Between the first and the second proof all 71 counts the prover prints per receipt were equal. In the
second, per-layer proving and LogUp-GKR, up to the prover's `after_layers_and_logup` checkpoint (not a Table 5 column),
took 1,035.3 s against 1,657.1 s; the step Table 5 calls witness folds, 158.7 s against 305.4 s; the weight fold 606.5 s
against 620.8 s, of which its round `k_hi`, on the host processor, 147.6 s against 135.6 s; and registration 783.1 s
against 701.7 s. The checkpoint split and Table 5 group the same proving time differently: for the first proof, the
1,657.1 s before the checkpoint and the 277.5 s after the weight opening, to the end of the receipt, make 1,934.6 s, and
Table 5's witness commitment, lookups, sum-checks and outer STARK make 1,935.5 s. On its machine the tests took 898.6 s
against 1,455.8 s and the prover's build 282 s against 323 s, while the fixture loads took the same time. Neither run's
logs show the machine's processor or memory.

## From Appendix F: Receipt shapes

**Receipt shapes.** The rows of Table 4 have different shares of MoE layers: 4 of 8, 2 of 4, 1 of 2 and 0 of 1. If one
MoE layer at 256 tokens fits in one receipt, 48 layers at 256 tokens need at most 48 receipts, which we estimate,
without a measurement, at about 72 MB and 670 ms of verification on the cloud host.

## From Appendix F: Cloud time

**Cloud time.** We predicted 7,000 to 8,600 s for the whole cloud call of the first proof, which took 7,002.9 s
in all. The other three calls took 5,526.7, 6,799.0 and 6,847.9 s.
For the earlier version we predicted 126 to 137 minutes and measured 161.6. Both include
the build and the tests. A run of the earlier version's statement with the earlier arithmetic (one scale per tensor,
2026-11-21) took 99.2 minutes on another host. The two dry runs' single receipts took 2,029 s and 2,012 s, so the
per-row scales' cost on proving is not isolated.

## From Appendix F: Browser build

**Browser build.** We compiled the verifier to WebAssembly through a thin wrapper in a separate crate, with no other
dependency; the verifier's sources are those of commit `3c7ba795`. The wrapper takes the bundle's files and the caller's
statement from memory and runs the steps of the command-line verifier in the same order. The build uses the toolchain
the repository pins:

    cargo +nightly-2026-01-15 build --manifest-path web/Cargo.toml --release --locked --target wasm32-unknown-unknown

The module is 2,146,731 bytes, 649,692 bytes after `gzip -9`, and imports nothing. Under Node 24.18.0 it gave the same
verdict as the same wrapper built natively and as the command-line verifier on nine cases. The honest bundle was
accepted with the final chain link of Appendix G. Eight tampered inputs were rejected: one flipped byte in each receipt
(two in the first), a changed prompt or generated token in the statement, a changed generated token in both the
statement and the manifest, and another model anchor. A flipped length field fails its conversion on the 32-bit target
before the range check that reports it natively; both reject. A build with overflow checks on also accepts the honest
bundle. In Chromium 156, with the page served on the local machine and the files chosen through its file input, the
honest bundle was accepted and three tampered inputs were rejected.

## From Appendix F: Browser timing

**Browser timing.** The browser, Node and the native wrapper ran back to back in one session on 2026-11-26, from
00:15:38 to 00:15:56, on mains power, with the laptop's lock held and the same `.wasm` and the same four files. Each
runner made a fresh instance, one first call, 3 warm-up calls and 30 timed calls, all accepted; the median is the 16th
of the 30 sorted times. The verify call took 31.1 ms at the median in Chromium 156.0.8078.100 (30.4 ms at the minimum,
55.0 ms for the first call), 33.31 ms under Node 24.18.0 (32.57 ms) and 25.57 ms for the same wrapper built natively for
the laptop (24.96 ms). WebAssembly took 1.22 times as long as native in the browser and 1.30 times under Node, by the
median and by the minimum alike. The time covers the verify call on files already in memory: the statement checks, the
model anchor, decoding the manifest and the three receipts, and the chain. It does not cover reading or fetching the
files, copying them into the module's memory or instantiating the module. The browser's clock is coarsened to 0.1 ms
because the page is not cross-origin isolated. The WebAssembly memory was 17,039,360 bytes. The one-minute load was 3.66
before the browser's runs and 3.93 after the last; a virtual machine of another project used about one core throughout.
Earlier timings at a load of 5 to 7 are not used. FlashProve's verifier peaked at 4.01 GB, or 3.73 GiB, on the laptop.
Only Chromium was tried, and we host no page (Appendix G).

## From Appendix F: Weights and calibration

**Weights and calibration.** We exported the weights from a third-party re-upload of the release at its `main` revision,
which resolved to commit `86c5ecb6`, and recorded per-tensor SHA-256 hashes for 504 block tensors and 3 tensors of the
model's ends. The calibration file, 52,856 bytes, is in our private source repository, not in the artifact. The numpy
version of the export is unknown. On our existing export, the register-only run of the measured model took 1,180.5 s,
with 237.8 GB of host memory and 33.0 GB of GPU memory at peak; for the earlier version's model it took 1,012.4 s, with
the same peaks to 0.1 GB. The verifier also checks the earlier version's single-token statement; the anchor tells the
two forms apart, and each refuses the other's arguments.

## From Appendix F: Builds

**Builds.** The prover of the first proof was built from commit `ee1ad727`, whose prover and verifier sources equal
those of `582c7abd`, and ran on Modal; the host's CPU model was not recorded. It read the weights of Section 3 and the
calibration file with SHA-256 `e48382544ce6987549b66ca2f3adf566c2dda30a795f358b97e8f413303595b2`. The statement was
fixed on 2026-11-23. Our Python reference decoded greedily in 13 passes, after checking that it reproduces the export's
own final state on the earlier version's prompt. On the laptop, the verifier built at `3c7ba795` accepted the bundle
under the command of Appendix G, as did the standalone verifier built from the run's commit on the cloud host.

## From Appendix F: Verifier timing details

**Verifier timing details.** The receipts are 1,548,505, 1,544,608 and 1,550,569 bytes. On the laptop the whole process
took 41.33 ms at the minimum of 30 runs; in process the cold run took 20.14 ms and the receipts 6.73, 6.71 and 6.75 ms.
On the cloud host the cold run took 37.5 ms, the receipts 12.45, 12.44 and 12.48 ms, and the prover's own check 38.2 ms.

## From Appendix F: Laptop sessions

**Laptop sessions.** The laptop is an Apple M5 Max (18 cores: 6 performance, 12 efficiency) with 64 GiB (68.7 GB) of
memory, shared with other jobs. The paper's laptop figures come from one session on 2026-11-25, 23:13 to 23:20, with the
laptop's heavy-work lock held. It ran on mains power for every block, with no thermal or performance warning and power
mode 0, as `pmset` recorded before and after each block. The one-minute load was 3.69 before and 5.60 after. Another
project's virtual machine ran throughout, at about 1.05 cores and 9.4 to 10.0 GB. Our verifier was built at commit
`d364785a` (Appendix B) and ran the caller-pinned command of Appendix G on each bundle; FlashProve's ran with the
arguments of our comparison record. For ours, one more process per bundle gave the in-process figures on the verifier's
own clock, which starts after the receipt files are read: one cold run and 31 warm, median. Two other sessions that day
give the same ratio (Appendix D). One, on mains power at a one-minute load of 4.76 to 6.92, measured ours at 45.88 ms on
the first bundle, within 2% of this session. The other, on battery power, was about 40% slower on both sides. An earlier
session that day, at 00:43 on battery power between 12% and 2% charge by the macOS power log, measured ours at 62.45 ms
and, on the verifier's own clock, 28.44 ms warm (median of 31) and 35.03 ms cold. The earlier version's proof was timed
on 2026-11-23, with FlashProve's verifier (Appendix E).

## From Appendix F: Fidelity measures

The first ten texts gave 5.43% (139 of 2,560) and the other twenty 4.69% (240 of 5,120), a difference within chance
(permutation p = 0.54); a bootstrap over texts gives 3.93% to 6.05% for the 30.

The texts are those of the first ten and twenty more of the same ten kinds, two of each, fixed by hash in a commit
before any model ran on them.

## From Appendix F: The integer arithmetic on the four inputs

So on these inputs the integer arithmetic of the fidelity procedure, computed by our accuracy script, reaches the state
that the prover proved and that our multi-token reference computed from the exported block files.

## From Appendix F: Table F1

**Table F1. The float model on the first statement. Row `15 + k` predicts generated token `k`, given the prompt and the
generated tokens before it. Margins are the top token's probability minus the second's, and the same in logits.**

| `k` | Row | Token the proof states | Float reference's top token | Float margin (probability; logit) | Integer reference through the float head: margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 436 | 436 | 0.8326; 3.075 | 0.8549; 3.321 |
| 1 | 16 | 1,085 | 1,085 | 0.5565; 1.277 | 0.5784; 1.342 |
| 2 | 17 | 4,635 | 4,635 | 0.9858; 5.376 | 0.9871; 5.509 |
| 3 | 18 | 323 | 323 | 0.6921; 1.796 | 0.6994; 1.829 |
| 4 | 19 | 12,824 | 12,824 | 0.0668; 0.135 | 0.0679; 0.137 |
| 5 | 20 | 1,648 | 1,648 | 0.8600; 3.314 | 0.8479; 3.252 |
| 6 | 21 | 290 | 290 | 0.0454; 0.094 | 0.0145; 0.030 |
| 7 | 22 | 1,453 | 1,453 | 0.9842; 5.006 | 0.9759; 4.520 |
| 8 | 23 | 26 | 26 | 0.6564; 2.089 | 0.6882; 2.148 |
| 9 | 24 | 589 | 589 | 0.0729; 0.360 | 0.0633; 0.292 |
| 10 | 25 | 30,404 | 30,404 | 0.1834; 0.800 | 0.1677; 0.736 |
| 11 | 26 | 948 | 948 | 0.0808; 0.267 | 0.0649; 0.214 |
| 12 | 27 | 19,560 | 19,560 | 0.3668; 1.034 | 0.3925; 1.119 |
| 13 | 28 | 75,041 | 75,041 | 0.2580; 0.935 | 0.2478; 0.892 |
| 14 | 29 | 436 | 436 | 0.4089; 1.452 | 0.4173; 1.438 |
| 15 | 30 | 290 | 290 | 0.9082; 4.030 | 0.9115; 4.138 |

## From Appendix G: Open science: artifact and evidence

The verifier depends on our copy of the stwo library, upstream commit `f0d79b0f` plus 23 commits of ours that change 22
files by 3,229 added and 219 removed lines, most of them in the verifier, the commitment scheme and FRI (Appendix F).

To recompute the anchor, once the exporter, the prover and the calibration file are released, export the integer model
from the pinned weights with the calibration file, run the register-only command, and compare the printed model anchor
with the one above. Table 1 gives that command's cost on our existing export, not counting the export itself; at toy and
medium shapes the command matches the prover's anchors.

To check the proof in a browser, build the module as in Appendix F, serve `web/` on the loopback address only (for
example `python3 -m http.server 8765 --bind 127.0.0.1 --directory web`), open the page, choose the manifest and the
receipts, and press Verify. The page makes no request beyond its own origin. We host no public page; the wrapper and the
page are in the release.

A full reproduction needs the weights, the calibration file, the exporter, the prover and the registration parameters,
and 63 to 80 minutes of prover time per proof on one H100 with 339 to 340 GB of host memory.

Each fact and number of this paper has a row in our claims ledger, which names its source file and commit and
labels it measured, derived, predicted or designed; the ledger and the evidence it cites are not released.


## From Appendix G: The Lean build

This paragraph describes the build of 2026-11-25, before the comments of `formal_cprs/` were reworded for the
release. The artifact carries the record of the rebuild of 2026-11-26 (`docs/evidence/lean_clean_build_20261126T194919Z/`),
whose figures Appendix G of the paper gives.

**The Lean build.** On 2026-11-25 the Lean development `formal_cprs/`, at the git tree it has at `8d78a86a`
(`fa6e28c2`), was built in a cloud container with 16 CPU cores and 32 GiB that started with only git and curl. It ran
the folder's own recipe: install elan with no default toolchain, then `lake update`, `lake exe cache get`,
`bash scripts/check.sh` and a file that prints the axioms of nine main theorems. elan 4.4.0 installed Lean 4.34.0
(commit `293d5d0c`, x86_64 Linux) from the pinned `lean-toolchain`, with Lake 5.0.0. `lake update` cloned the 13
dependencies at their pinned commits in 157.3 s, and the regenerated `lake-manifest.json` equals the committed one byte
for byte. Mathlib and its own dependencies came from Mathlib's binary cache as compiled files. ArkLib (39 modules), our
development (24), VCVio (27), ToMathlib (27), PolyFun (16), Cslib (6) and ToCslib (2) were built from source. The check
script, unchanged, passed in 273.3 s: 9,093 build jobs and 624 theorems swept, none with an axiom beyond Lean's three
standard ones, and the nine main theorems depend on `propext`, `Classical.choice` and `Quot.sound` only. The SHA-256
hashes of the 69 source files the build reads equal those of the committed folder. The whole run took 461.3 s and 8.8
GiB of disk. The artifact holds the 69 files that build read, with the same SHA-256 hashes (our 24 source files, the 39
vendored ArkLib modules and their root file, the pinned manifest, lakefile and toolchain file, and the two check
scripts), ArkLib's licence and the build's record. In `formal_cprs/`, with elan installed, `lake update`,
`lake exe cache get` and `bash scripts/check.sh` rebuild and check it; the last ends with `PASS`.

## From Appendix H: Proofs of inference

**Proofs of inference.** FlashProve [1] is compared with ours in Section 6.6. Its research page reports 110 tokens per
second and says that "each proof checks one token"; its performance guide counts "each request's 1,024 prompt tokens" in
that rate. With every token of a stored reply checked, under its looser clipped-gap policy, the guide gives "39.1 ± 1.0
counting reply tokens alone". Attestable [2] proves `y = F(W, x, r)` for committed weights `W`. Its figures for four
models, at a context of 4K tokens and batch size 4, are in a table image of its post, which labels the first model Gemma
31B. For Gemma 4 31B it reports 77.69 tokens per second. They come from a reference CPU verifier, and it states "100
bits of security" for what it calls "alpha results". DeepProve [3] proves all generated tokens of a prompt for GPT-2 and
Gemma 3. Its paper reports "approximately 174 and 86 tokens per minute" and verification in 1 to 3.7 s. Its repository's
README reports, for 512-token sequences on a 24-core CPU server, 67 and 27 tokens per minute and 1.3 and 4.3 s of
verification. Its proofs there are 10.7 and 27 MiB (11.2 and 28.3 MB). zkLLM [7] proves "the entire inference process in
under 15 minutes" for models of up to 13 billion parameters.

zkTransformer [5] proves inference "end-to-end, from input token IDs to output token IDs" with "KZH-3 over BN254"; we
did not read its measured figures. zkAgent [4] proves a whole agent run, tool observations included, in 93.18 s on a
coding-assistant execution. NanoZK [6] proves layers separately, linked by a SHA-256 commitment chain, hides weights and
activations, and reports its verification time for an MLP sub-circuit on a CPU. zkPyTorch [13] compiles PyTorch models
into zero-knowledge circuits and reports "150 seconds per token for Llama-3 inference".

Hallmark [23] proves that a quantized TinyLlama-1.1B answers one fixed prompt with eight tokens through the end of the
sequence. Its statement has the form of ours, with greedy tokens fed back inside one causal block of 32 positions, and a
Circle STARK prover proves it as 23 linked segment proofs. Its README reports a 53.94 s proving session on one H200, and
it claims no zero knowledge.

Five more systems report whole-model figures. VerInf [27] "certifies an observed output stream against a committed
integer model and bounds the unexplained information", interactively, over a 1,000-token forward pass of the full
48-layer Llama 4 Maverick. Its README reports 14.3 hours of proving on one DGX Spark, a 93.6 GB proof and 17.7 hours of
verification on 20 CPU cores. SASH [28] proves every layer of Llama-3.1 8B from the embedding to the argmax, for a
4,096-token prompt and a 512-token answer. It reports 1.25 positions per second on one H100, 18.5 GiB (19.9 GB) of
proofs and 119 s to check them on CPU cores. It states that the proofs are "not yet zero-knowledge in the formal sense".
zkGPT [29] proves GPT-2 inference in 21.8 s on 32 threads, with a proof its table gives as 101K, checked in 0.35 s.
ZKTorch [30] proves whole models on a one-token input: LLaMA-2-7B in 2,645.50 s, with a 22.85 MB proof checked in 100.14
s. Sable [42], concurrent with this work, proves all 64 layers of Qwen2.5-32B in 92.5 s for 4,096 tokens and 398 s for
16,384 on one RTX PRO 6000 Blackwell GPU. Each sequence is a random prompt followed by 16 greedily generated tokens. Its
proofs are 106 to 477 MiB (111 to 500 MB), and a verifier with a compiled core checks one in 19 to 56 s on one core of a
Xeon 6776P.

## From Appendix H: Measured by a public benchmark

**Measured by a public benchmark.** Inference Proofs [48] runs systems itself, with each prover on one H200 and each
verifier on 8 CPUs without a GPU. Its request is a 512-token prompt and a 512-token greedy reply, every reply token
checked. It ran FlashProve at the commit we cite, on Unsloth's `UD-Q4_K_XL` GGUF files of Maverick, 4.84 bits per active
weight. It ran SASH at a commit of an unmerged pull request to its repository, on Llama-3.1 8B quantized to W8A8. It
labels every figure as run by one operator and not yet rerun externally. Table H1 lists its figures beside each system's
own.

## From Appendix H: Verification without proofs

**Verification without proofs.** DiFR [14] verifies inference outputs by comparing generated tokens with a trusted
reference implementation run with the same random seed. It reports detecting 4-bit quantization "with AUC > 0.999 within
300 output tokens"; it needs a trusted reference implementation.

## From Appendix H: Hash-based multilinear commitments

**Hash-based multilinear commitments.** Basefold [15] generalises FRI to foldable linear codes and interleaves that
proof of proximity with the sum-check; our opening follows this structure, with a Reed-Solomon code over the degree-4
extension of Mersenne-31. Diamond and Posen [17] introduce ring-switching, a sum-check-based reduction from a commitment
over a small field to one over its extension, on a characteristic-2 adaptation of Basefold. Our packing of four
Mersenne-31 weights into one extension symbol is an instance of it, written as an exact identity at four
Frobenius-conjugate points. WHIR [16], a proof of proximity for constrained Reed-Solomon codes, is presented by its
authors as a replacement for FRI, STIR and Basefold with faster verification. DeepFold [18] moves the FRI-based
multilinear commitment to the list-decoding regime, with fewer query repetitions than Basefold, and adds a batch
evaluation for sizes that are not powers of two. Ours stays in the unique-decoding regime (Section 5.3). Ligero [21]
uses interleaved Reed-Solomon codes with communication proportional to the square root of the circuit size. Brakedown
[19] uses a linear-time encodable code over any large enough field, and Orion [20] a linear-time prover with
polylogarithmic proofs by composition. We have measured none of these at our size. Our opening is checked inside the
outer STARK, so only its cost in the outer circuit matters.

## From Appendix H: Table H1

The figures in Table H1 come from different hardware, models and statements; only the comparison with FlashProve, which
we ran ourselves, is a measured ratio, and we print no ratio for the others.

| System | Model | Verify (s) | Proof (MB) | Hardware | Source |
|---|---|---:|---:|---|---|
| zkPyTorch [13] | VGG-16, Llama-3 | – | – | – | R |
