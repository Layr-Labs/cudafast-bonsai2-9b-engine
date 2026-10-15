# Bonsai 2 9B CUDA — participant contract

This document states the terms that bind a submission to track
`bonsai2-9b-cuda-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/bonsai2_9b_cuda_v1_track.json` is the track contract fixture.
Both carry values, paths, commands and pins only. This document explains them.
It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/bonsai2_9b_cuda_v1_track.json` and `benchmark.json`.
3. This document.
4. `README.md`, `TASK.md` and `AGENTS.md`.

If a configuration file disagrees with this document on a plain value, the
configuration file wins, and CI (`tools/check-doc-consistency.py`) is meant to
catch the drift first. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary. `./tools/fetch-benchd.sh`
resolves it from the bench repository's release channel for this track, and
verifies it against the channel's `benchd.manifest.json` (`{branch,
source_commit, sha256, bytes}`). A submission cannot change what measures it.
The channel is the `main` branch of the public bench repository
`Layr-Labs/benchd`. The fetch needs no token.

## 2. What the track measures

The track measures single-stream prefill and decode speed of the target on a
DGX Spark (GB10, sm_121, 128 GB unified memory).

The **target** is one pinned weight file from
`prism-ml/Ternary-Bonsai-2-9B-gguf` (`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`
@ `9e08fac9`): Bonsai 2 9B, built from `Qwen/Qwen3.5-9B`. The trunk is PQ2_0
(ternary) with prism.hadamard v1 transforms folded into the weights. The file
holds one MTP layer in Q8_0, which uses the shared `output.weight` as its head.
The tokenizer is the Qwen 3.5 tokenizer.
Shape metadata (layers, vocabulary, hidden size) is read from the file at
runtime — submissions must not hard-code the target's constants.

The engine is the PrismML `llama.cpp` fork vendored at `llama.cpp/`. The
resident is `harness/llama_resident/` (`llama-resident` binary). Speculation
uses llama.cpp `draft-mtp` over the native MTP tensors in the weight file, or
llama.cpp `draft-dflash` with the pinned DFlash2 drafter file
(`Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf`, same repository and revision).

| Frozen runtime choice | Value |
|---|---|
| Weight packing | as shipped in the pinned file (no re-quantization) |
| Hadamard/sign transforms | required by the model; may be fused, not removed |
| KV cache in the resident | f16 |
| MTP head | in-file tensors only (`source: pinned`) |
| DFlash2 drafter | the pinned file only, staged beside the target (`dflash_drafter`) |
| Batch size | 1 |
| Context | 8192 (fixture `engine_env`) |

`benchmark.json` `scoring.kvBackend` and the fixture `kv_backend` are
`contiguous`: the resident keeps one contiguous f16 KV cache per stream.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority: `llama.cpp`, `harness` and
`mtp-head.manifest.json`. A submission archive has REPLACE semantics over those
paths.

| Path | Contents |
|---|---|
| `llama.cpp/` | vendored PrismML fork (`llama.cpp/PRISM_PIN.md` records the pin) |
| `harness/` | `protocol-adapter/` (Rust), `llama_resident/` (C++), optional `kernels/` |
| `mtp-head.manifest.json` | the speculative declaration (section 4) |

### 3.1 Optional paths

`mtp-head.manifest.json` is optional. An archive that omits it runs serial.

### 3.2 The byte budget

The static review (`.github/scripts/submission-static-review-checks.sh`) caps
the editable surface, and `benchmark.json` `editableSurfaceByteBudget` declares
the caps:

| Cap | Value | Derivation |
|---|---|---|
| `maxTotalBytes` | 36,047,190 | the surface at rest, 34,998,614, plus 1 MiB |
| `maxFileBytes` | 2,558,837 | the largest file at rest, 1,510,261, plus 1 MiB |
| `maxGrowthBytes` | 36,047,190 | equal to the total: the vendoring commit is the growth |

The cap exists so that source cannot hide lookup tables. It also refuses
symlinks and special files anywhere under an editable path.

### 3.3 What you may not edit

Everything else: `tools/`, `fixtures/`, `setup.sh`, `benchmark.sh`,
`benchmark.json`, `correctness_prompts/`, `.github/`, the docs, the pinned
snapshot, the pinned engine image and benchd.

### 3.4 The target's formats are frozen

Do not re-quantize, re-represent, or change the packing of any weight tensor.
Do not remove or approximate Hadamard/sign transforms the model requires (you
may fuse them). Keep KV f16, the in-file MTP head, the pinned DFlash2 drafter,
batch 1 and context 8192.
This holds even when the result passes the correctness gate.

- **What is legal:** faster kernels, fusion, CUDA graphs, scheduling and memory
  layout that compute the same mathematics.
- **A load-time repack is legal** only when it is input-independent and preserves
  every value exactly (for example a layout swizzle).

### 3.5 Editing the engine

`llama.cpp/` is the full vendored fork. `tools/engine/build.sh` builds
`llama-resident` inside `bonsai9b-engine:v1` with CMake/Ninja
(`harness/llama_resident/CMakeLists.txt`, `-DLLAMA_CPP_DIR`, sm_121). The
static review refuses network fetches in editable CMake (`FetchContent`,
`ExternalProject`, `file(DOWNLOAD)`, network `execute_process`).

Optional CUDA/C++ in `harness/kernels/` runs in the same image profile when
`harness/kernels/build.sh` exists.

The resident runs inside `bonsai9b-engine:v1` with `--network none`, the weight
file and checkout read-only, and a fresh per-leg cache directory.

benchd seals a harness hash over `llama.cpp` and `harness`.

**Reproducibility:** serial mode must be byte-deterministic run to run.
Speculative mode may diverge from serial on near-ties on GB10; the tolerance
gate and replay handle that (`docs/engineering/engine-probe.md`). Check with
`tools/determinism-probe.py` at serial and at your depth.

### 3.6 Where your code runs

On a ranked box every piece of participant code runs as an unprivileged
sandbox user, not as the runner. That covers the adapter build and its
`build.rs`, `harness/kernels/build.sh`, the resident build, and
the adapter at run time. That user has no network, not even loopback. It can
read the checkout (read-only) and write only its own per-job directories. It
cannot reach the goldens, the trusted tools, docker or anything else on the
box.

- **Cargo dependencies come from crates.io only.** Every source in `Cargo.lock`
  must be the crates.io index. A `path` dependency must stay inside `harness/`.
  `[patch]`, `[replace]`, git sources and other registries are refused. The
  trusted side vendors the crates first, and then the build runs offline, so
  a `build.rs` that fetches anything fails.
- **What comes back is the score, nothing else.** The adapter's stderr is kept
  private and never returned, and job logs are not shown to participants. The
  score record carries no text your engine supplied. An engine error appears
  as a failure class plus the sha256 and byte count of its message. Identity
  strings and acceptance lengths are summarised, not copied.
- **Reported digests must be well-formed.** `head_provenance_sha256` and
  `runner_manifest_sha256` must be 64 lowercase hex characters, or they seal
  as `invalid`.
- **Nothing persists between jobs or legs.** Temp files, shared memory, IPC
  objects and message queues the sandbox user leaves are wiped, every process
  it started is killed after its leg, and it cannot schedule jobs. Do not rely
  on anything surviving from a previous leg or job; section 8 forbids the
  reuse anyway.

## 4. The drafters

The track has two speculative arms. Both are pinned; no submission carries
drafter weights, and participants cannot supply or swap drafters (section 4.2).

- **MTP** (`spec.mode` `mtp`, the default). The head is the native MTP /
  `nextn` tensors inside the pinned GGUF weight file, run by llama.cpp
  `draft-mtp`. The fixture's `mtp_head` block records their size and permitted
  depths, 1 to 6.
- **DFlash** (`spec.mode` `dflash`). Prism's DFlash2 block drafter, the pinned
  file `Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf` (`target.files`), run by
  llama.cpp `draft-dflash`. It reads the inputs of eight target layers and
  drafts a block of 16 positions: the anchor and up to 15 drafts. The
  fixture's `dflash_drafter` block records its block size and permitted
  depths, 1 to 15. Its trunk is the target's: the drafter was trained for
  `Ternary-Bonsai-2-9B-PQ2_0.gguf`, whose tensors are byte-identical to the
  trunk of the target file.

The fixture's `allowed_modes` (`serial`, `mtp`, `dflash`) lists the modes a
declaration may select.

### 4.1 What you may declare

```json
{
  "version": 1,
  "source": "pinned",
  "max_bytes": 3221225472,
  "spec": { "enabled": true, "mode": "dflash", "num_speculative_tokens": 7 }
}
```

`tools/spec-declaration.sh` is the one trusted reader of the declaration.

- **Allowed keys.** It accepts `version`, `source`, `max_bytes`, `bytes`,
  `sha256` and `spec`, and refuses any other key. In `spec` it accepts
  `enabled`, `mode` and `num_speculative_tokens`.
- **Mode.** `mode` is `mtp` (the default when absent) or `dflash`, and must be
  in the fixture's `allowed_modes`.
- **Source.** It refuses any `source` except `"pinned"`, which rules out
  `"remote"` and `"in_branch"`.
- **Depth.** It refuses an enabled depth outside the mode's permitted depths:
  `mtp_head.permitted_draft_depths` (1 to 6) or
  `dflash_drafter.permitted_draft_depths` (1 to 15).
- **Serial.** A declaration that is absent, disabled, or has 0 tokens derives
  serial.
- **Recorded, not enforced.** `max_bytes`, `bytes` and `sha256` are recorded
  only. The fixture's `declaration_max_bytes` is 3 GiB, so the pinned head
  fits the cap.

The shipped declaration is disabled: the reference tree runs serial, and the
reference scored against itself is a null control of about 1.000.

### 4.2 What you may not do

- Replace the in-file MTP head, or load any drafter other than the pinned MTP
  tensors and the pinned DFlash2 drafter (no DSpark, Eagle, n-gram, or
  participant weights).
- Re-quantize a drafter: `mtp_head.requantization_permitted` and
  `dflash_drafter.requantization_permitted` are `false`.
- Ship head weights, or fetch anything at run time; the resident has no
  network.

### 4.3 What the size cap does and does not do

The cap is a declaration limit only. `mtp-head.manifest.json` records whether
speculation is enabled, the mode and the draft depth; it does not load weights.
The MTP tensors always come from the pinned GGUF on disk, and the DFlash2
drafter from the pinned drafter file on disk.

### 4.4 What a drafter does in a run

At depth d, `llama-resident` runs the same greedy speculative loop as
`llama.cpp/examples/speculative-simple` (`common_speculative`, `draft-mtp` or
`draft-dflash`, n-max = d): each cycle proposes d drafts (MTP may propose fewer
near the end of the context), verifies them in a batched forward, and commits
only tokens that match the target's greedy choice plus the bonus token. Serial
mode steps one token per cycle. The target decides every emitted token. The
DFlash drafter decodes its whole block in one forward per cycle; `tools/serve-up.sh
--boot --spec dflash` passes it to the resident as `--draft-model`.

The adapter reports each cycle as `acceptance_lengths` and reports drafted,
accepted and committed totals via `spec_counters` (`drafts`, `hits`, `quenches`,
`disagreements`). benchd seals `effective_spec_mode`, `effective_spec_depth`,
`spec_rounds`, `spec_drafted_total`, `spec_accepted_total` and
`spec_acceptance_rate`. The same verify and counter rules apply to both arms.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain(role) = control_seconds_per_token / candidate_seconds_per_token  (per pair)
```

The run is PAIRED on one box, and the pairs run one after the other, never
overlapping:

- **Pairs.** A ranked job runs `official_pairs` = 3 pairs in measurement order
  on one box. Each pair is one SERIAL CONTROL leg on the organizer-staged
  reference tree at `baseline_reference_commit`, then one CANDIDATE leg on the
  submission at its declared depth, single-stream. Pair k measures
  `live_goldens[(k - 1) mod 3]`, and both legs of a pair measure that prompt.
- **Mean combine.** The fixture declares `official_pair_combine: "mean"`. For
  each pair, benchd forms prefill and decode speedups against that pair's own
  control leg and a pair composite with `score_prefill_weight` 0.25 and
  `score_decode_weight` 0.75. It walks the pairs in order and applies the same
  timed gate to each pair against its own control; the first pair that fails
  refuses the run. After every pair passes, the sealed composite is the
  arithmetic mean of the three pair composites. Sealed speedups and per-token
  times are the arithmetic means of the matching per-pair figures — not the
  ratio of mean times, and not the composite of mean speedups.

No baseline is stored anywhere; the control leg measures it next to the
candidate.

**What decode means.** Decode seconds per token is the decode window: the time
of `free_decode_run(N)` divided by N. The seed prefill of the prompt is not
part of it. Prefill is its own phase, timed on its own. `decode_gain` is the
control leg's decode window per token divided by the candidate leg's. The
calibration band and the floors use the same figure. benchd seals the seed
prefill window as a report-only figure under
`*_seed_prefill_window_seconds_per_token`.

#### 5.1.1 Where the prefill window is, and why you cannot move work out of it

benchd splits its own clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

What the engine owes:

1. `free_decode_begin` runs the FULL 512-token seed prefill with the requested
   spec resolved, and replies only after that work has completed. The reply is
   the greedy seed token and the echoed `effective_spec`.
2. `free_decode_run` never re-prefills. It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.

**Moving work between windows is a scoring defect, not an optimisation.**
Deferred seed work lands in the decode window and makes decode slower. Decode
work done early, inside `free_decode_begin`, leaves the decode window and is
not scored, which breaks obligations 1 and 2 above.

The seed token's index: `seed_token` is `expected_decode_seed_token`;
`free_decode_run` returns the N tokens after it, so `tokens[0]` = `expected_decode_tokens[0]`.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 512 |
| Unmeasured prefill warm-up passes before the timed prefill | 1 (same prompt, same phase, computed in full) |
| Timed prefill passes per phase | 5; the phase's prefill time is their median |
| Checked decode steps | 128 |
| Golden shape | 512 `prompt_tokens` and an answer of at most 1024 `expected_tokens` per case (the answer ends at the first end-of-turn token), plus the free-run `benchmark` block |
| Streams per window | 1 |
| Timed prompts per leg | 1; pair k measures `live_goldens[(k - 1) mod 3]` (section 5.2.1) |
| Prompts in the pinned correctness pool | 3 |
| Pairs per ranked job | 3 |
| Legs per ranked job | 6 |

Every timed phase waits behind benchd's GPU cool-down gate. The ranked job
waits for the box to go quiescent first, and refuses to run with no
temperature reader.

### 5.2.1 Every job measures every live golden

The fixture's `live_goldens` lists three hidden coding prompts. Each one is a
user message rendered through the pinned `chat_template.jinja` with thinking
off, and is exactly `seed_tokens` long. Every job measures every live golden:
pair k measures `live_goldens[(k - 1) mod 3]`, and both legs of a pair measure
the same prompt, so the ratio compares like with like. The score is the mean
of the per-pair composites. `metrics.paired_legs` identifies each pair's
prompt by its sha256.

Why: one pair per prompt makes the score depend on all three prompts, and the
mean gives each prompt the same weight.
Every live golden carries its own serial golden and one timed oracle per draft
depth, and the run refuses a declared depth that has no oracle.

The per-box calibration file carries one band per live golden, and benchd
gates each control leg with the band of the prompt that leg measured.

### 5.3 The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 3 |
| `minPairsPerCohort` | 3 |
| `decodeSpeedupFloor` | 0.95 |
| `prefillSpeedupFloor` | 0.95 |
| `kvBackend` | `contiguous` |

The speedup floors apply per pair in the mean-combine gate above.
`iterate --mode official` does not enforce a decode speedup ceiling. It enforces
the speedup floors (0.95 prefill and decode, active) and the fixture's optional
band-down fast-side checks (disabled here).

### 5.4 Token fidelity

In the timed window, benchd allows at most `timed_token_tolerance_per_thousand`
(40) mismatches per thousand of the 129 emitted tokens (seed output plus 128
decode steps), and each mismatch must be the reference's second choice within
`timed_token_near_tie_relative_gap` (0.01); see 5.4.1. The correctness suite
below still uses its own token budget against the serial tape. Every per-depth golden carries the SAME serial `cases[]`
(`tools/record-goldens.sh` splices them), so fidelity is always measured
against serial. The gate accepts similar output. It does not certify lossless
output.

### 5.4.1 The timed window (replay after divergence)

On the ranked paired path, benchd compares the candidate's timed stream to the
depth oracle first. When tokens differ, it **teacher-forces** the candidate's
stream through the reference serial engine and applies the fixture's replay
policy:

- `timed_token_tolerance_per_thousand` (**40** on this track): at most that many
  replay mismatches per thousand timed positions.
- `timed_token_near_tie_relative_gap` (**0.01**): a counted mismatch must be the
  reference's **second choice**, with top-1→top-2 relative gap `≤ 0.01`; any
  other different token refuses the run (`TIMED-DIVERGENCE-NOT-A-NEAR-TIE`).

Measurements and rationale:
[`engineering/replay-policy-measurements.md`](engineering/replay-policy-measurements.md).
benchd v2 does **not** read retired Nemotron keys (`timed_replay_max_logit_gap`,
`timed_replay_off_argmax_per_thousand`).

The local modes (`tools/local-baseline.sh`, `--local-iterate`) still require
**exact** timed-window match (no replay).

### 5.5 Arming

`official_scoring_enabled` in the fixture is the single arm authority. benchd
refuses to seal an official artifact while it is `false` or absent, and so does
`tools/ranked-box-preflight.sh`.

The fixture is armed only after four steps are done:

1. every golden is recorded on a Spark and pinned (`golden_pins_provisional`
   is false; `tools/lint-benchmark-manifest.py` refuses
   `official_scoring_enabled` while that flag is true, and when it is false
   refuses placeholder digests in `timed_prompt_pool`,
   `hidden_correctness_golden`, and `speculative_oracles` — a 64-hex `sha256`
   that is all one repeated digit or `deadbeef` repeated eight times;
   only `tools/ranked-box-preflight.sh` scans the contract file for the
   substring `PENDING-ORGANIZER`, for example
   `BONSAI9B-CUDA-PENDING-ORGANIZER`, and refuses before pin verification);
2. `baseline_reference_commit` names the promoted reference;
3. every box is calibrated;
4. the null control, the reference scored against itself, lands at about 1.000.

The control leg is serial whatever the candidate declares. benchd boots each
leg's resident itself, from that leg's own tree, with
`tools/serve-up.sh --boot --spec serial|mtp|dflash --draft-len N --socket-out FILE`,
and stops it with `--stop --socket PATH`.

A control leg that fails (`SERIAL-CONTROL-LEG-FAILED`) is never your fault,
and the router records it as `control_failed`, an infrastructure outcome, not
a failed submission. The measurement driver keeps one retry budget per job.
It re-runs the whole measurement, both legs of every pair, exactly once when
either (1) the reference's own token divergence
(`SERIAL-CONTROL-LEG-FAILED: benchmark prefill|free-run decode seed|free-run
decode token mismatch`, in the timed or the warm-up leg), or (2) the pair-1
candidate resident fails to boot before any candidate timing is sealed
(`LEG-SERVE-BOOT-FAILED: the candidate leg's ...` with an empty
`metrics.paired_legs` audit trail). A pair-2 candidate boot failure is never
retried: pair 1 is already in `paired_legs`, so a second attempt would be a
best-of-two on your engine. Whichever cause fires first consumes the budget;
a control divergence on attempt 1 followed by a pair-1 boot failure on attempt
2 is final. In every retriable case the driver wipes the sandbox, including
all five per-job directories, before the second run. The second attempt alone
is the result; the first attempt's timings are discarded, never compared. A
second control-leg token divergence exits 75. Every other failure is final,
including every other control-leg failure (spawn, session, protocol, cool
gate, memory), because pair 2's control leg runs after your leg. The log
records `control_leg_attempts=1` or `2`; the sealed `score.json` does not
change.

#### 5.5.1 The per-box calibration

Each ranked box carries a calibration file, named by
`MLXFAST_BASELINE_CALIBRATION` and written by `tools/calibrate-box.sh` (3
passes, one resident boot per pass). It is a HEALTH BAND for the control leg,
never a denominator. A control leg outside the band fails the run by name. A box
whose passes spread by more than 1% gets no band.

`tools/ranked-box-preflight.sh` section 8b refuses a calibration file in any of
these cases:

- its `version` is not 3. Versions 1 and 2 recorded decode as the whole window
  (seed prefill plus decode, over N), and the refusal says to recalibrate;
- it names another track, box or reference commit, or a prompt other than the
  fixture's `calibration_prompt`;
- it is dated before the reference commit or in the future;
- its band cannot fail.

The band's widths come from the fixture. The control leg's per-token time may
run at most `prefill_band_up_tolerance` = 5% (prefill) and
`decode_band_up_tolerance` = 3% (decode) slower than the box's calibration. The
fast-side tolerances (`prefill_band_down_tolerance` and
`decode_band_down_tolerance`, 5% each) are declared but disabled
(`prefill_band_down_enabled` and `decode_band_down_enabled` are `false`).

A failure of the trusted control leg (outside its band, or off its golden,
`SERIAL-CONTROL-LEG-FAILED`) is an infrastructure fault, not the submission's:
the control leg runs the organizer's reference tree, which the submission
cannot change.


### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| `correctness_prompts/public_longcopy_gate_english_4096.txt` | git | Yes: the public prompt text |
| `correctness_prompts/public-longcopy-gate-english-512.golden.json` | git | Yes: for local serial runs |
| `correctness_prompts/public-longcopy-gate-english-512.mtp<N>.golden.json`, one per depth 1 to 6 | git | Yes: for local runs at MTP depth N |
| `correctness_prompts/public-longcopy-gate-english-512.dflash<N>.golden.json`, one per depth 1 to 15 | git | Yes: for local runs at DFlash depth N |
| `timed_prompt_pool[]`, 8 tapes | organizer storage, staged on the box | No |
| `speculative_oracles{}`, 21 per-depth oracles per live prompt (`mtp1`..`mtp6`, `dflash1`..`dflash15`) | organizer storage, staged on the box | No |
| `hidden_correctness_golden` | the default live golden's serial tape, pinned by digest | No |

The organizer stages the set on each box, and the preflight checks every
file's byte count and sha256 against its pin before a ranked run. The fixture's
`r2_path` fields name the files; they are not fetched at run time.

### 5.7 Pool refresh and final ranking

- **Per job.** Each ranked job measures every live golden (section 5.2.1).
- **Refresh.** The live goldens are replaced by prompts no submission has seen
  every 3 to 4 days during the contest.
- **Submissions** are unlimited.
- **Final ranking.** At the close, each team's best submission is taken: the
  mean contest score counts every contest run of that submission that scored or
  failed because of the submission (a team failure counts as 0.0; a
  post-setup reference-leg or band failure counts as 0.0; other infrastructure
  runs are excluded). Ties go to the most recent submission. The top N teams
  (default 10) are selected. Each selected submission is re-measured K times
  (default 5), one run per slot, across different boxes on a held-out pool
  that was never staged during the contest. Each slot is its first run that
  ends with a score or a team failure; infrastructure outcomes and off-pool
  prompts are re-run, not scored. In the final re-score only, the first candidate
  resident boot failure in a slot may be re-dispatched once; a second boot
  failure in the same slot counts as 0.0. A run that fails because of the
  submission, or a reference-leg or band failure that persists after re-runs,
  counts as 0.0 for that slot. The final ranking is the mean of the K slot values (the score, or
  0.0 for a team failure or penalized post-setup infrastructure slot). A team
  with any slot still pending or unresolved is incomplete and is not ranked
  above a complete team. Contest scores decide only who is re-measured.

## 6. Running the benchmark

`benchmarkCommand` runs `tools/bonsai2-9b-yukon-run.sh`. That script selects
one of two paths:

| Condition | Path |
|---|---|
| `GITHUB_ACTIONS` is `true`, or one or more of `MLXFAST_BONSAI9B_GOLDEN_DIR`, `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION` is set | Ranked. The script replaces itself with `tools/bonsai2-9b-measure-and-score.sh`. |
| None of these | Local. The script runs `tools/local-baseline.sh`, which runs `./benchmark.sh --local-iterate` over the public golden at the declared depth. The result has `"score": null`. |

The ranked workflow calls `tools/bonsai2-9b-measure-and-score.sh` directly.
Both scripts are trusted-side tooling.

The ranked script execs:

```text
benchd iterate --mode official \
  --contract fixtures/bonsai2_9b_cuda_v1_track.json \
  --engine .build/release/mlxfast-runtime-worker \
  --weights $MLXFAST_TARGET_SNAPSHOT_DIR \
  --golden $MLXFAST_BONSAI9B_GOLDEN_DIR/<drawn prompt>.golden.json \
  --golden-sha256 <pin> --golden-bytes <pin> \
  --control-golden <serial live golden> --control-golden-sha256 <pin> --control-golden-bytes <pin> \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  --live-prompt <drawn prompt> \
  [--mtp-depth <declared depth> | --candidate-spec '{"mode":"dflash","dflash":{"depth":<declared depth>}}'] [--box $RUNNER_NAME] \
  --score-path score.json
```

benchd seals `score.json`, its `.sha256` sidecar and the integrity sidecar
itself. The script converts nothing.
Before any participant code runs, the job draws a 64-hex nonce into
runner-private storage. After sandbox cleanup, **Reveal the sealed score**
prints that nonce once, the sealed `score.json` if it exists, and an end line
that binds the nonce to the file digest and length. The intake service parses
that block only ([submission-restriction-spec.md](submission-restriction-spec.md) §10); arbitrary text in the same log does not count.

When the reveal step prints a sealed `score.json`, the ranked workflow then
uploads that same file as the one artifact `benchmark.json` `scoreArtifact`
names (`benchmark-results-<run_id>/score.json`). This is the score Yukon reads.
No other file is uploaded (`tools/ci-workflow-egress-scan.sh`).

The rest of the command surface:

- **`preSubmitCommand`** runs the same script with `--preflight-only`, which
  checks the arm gate and the golden pin without measuring.
- **The ranked pipeline** is `.github/workflows/benchmark.yml`, dispatched
  only by `workflow_dispatch`. It runs on
  `[self-hosted, Linux, ARM64, bonsai2-9b-cuda-v1]`, one
  ranked run per box at a time.
- **Sockets.** Nothing in the job may export `DS4_RESIDENT_SOCKET` or
  `BENCH_WORKER_RESIDENT_SOCKET` (benchd's names), and nothing may preset
  `RESIDENT_*`, `VLLM_*`, `HF_*`, `TRANSFORMERS_*`, `TORCHINDUCTOR_*` or
  `TRITON_*`. The preflight refuses each of them by name.

## 7. The pinned artifacts

Plain values live in `fixtures/bonsai2_9b_cuda_v1_track.json` and `benchmark.json`;
this table is the participant-facing summary. Field names match the fixture
where they exist.

| Artifact | Identity |
|---|---|
| Weight file | one GGUF from `prism-ml/Ternary-Bonsai-2-9B-gguf` (`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` @ revision `5b2abb02362efbed46bbe204014b02837778b2c2`, sha256 `c79fcdfaf6b7bf47089b50dadcfd11f9ce7d3e96adf3bbefde7912e5ef69e22b`, 2675410720 bytes) |
| MTP head | in-file `mtp` / `nextn` tensors in the weight file (`mtp_head` in the fixture) |
| DFlash2 drafter | `Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf` @ revision `5b2abb02362efbed46bbe204014b02837778b2c2` (its own `revision` in `target.files`), sha256 `95149abdd584f7a69ea5ead38790d5481454428d2c1616527ff0d45422a762b4`, 1574233856 bytes (`dflash_drafter` and `target.files` in the fixture) |
| Engine source | PrismML `llama.cpp` @ `4fbda1292562cc1f23904d5867fb79603737547a` (`llama.cpp/PRISM_PIN.md`) |
| Engine image | `bonsai9b-engine:v1` built from `nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92` (`docs/design.md` §3); image id per ranked box under `/srv/bonsai9b/engine-image.id`; on a participant machine `./setup.sh` builds the image and writes `.build/engine-image.id` |
| Resident binary | `llama-resident` from `harness/llama_resident/` (organizer-staged reference hash in box calibration) |
| Driver floor | `580.159.03` |
| benchd | the channel manifest for this track (`./tools/fetch-benchd.sh`) |

## 8. Prohibited techniques

A submission that uses any of these is disqualified, whether or not a gate
catches it:

- **Reuse across requests.** Any cache, memo or reuse keyed on a request's
  input tokens, or carried between requests or phases, whose only possible hit
  is the harness repeating a computation. That includes engine prefix caching,
  KV cache reuse, or hybrid GDN / recurrent state kept across a `sync` or
  `invalidate` boundary (`docs/design.md` §4: each connection clears memory and
  bookkeeping), or answering from a recorded stream. benchd's phases replay the
  same prompt; nothing may survive from one phase to the next. Bit-identical
  output does not make it legitimate.
  It also covers the prefill warm-up pass: the timed prefill follows one
  unmeasured prefill of the SAME prompt in the SAME phase, and each pass must
  be computed in full. Carrying anything from the warm-up pass into the timed
  pass is this offence.
- **Hardcoded content.** Hidden prompts, token ids or answers.
- **Timing and protocol tricks.** Timing shortcuts, protocol injection, moving
  work across the prefill/decode boundary (section 5.1.1), network access, and
  filesystem exfiltration.
- **Format changes.** Any change to a frozen format (section 3.4) or to a
  pinned drafter (section 4.2).
- **Out-of-surface edits.** Any change outside `editablePaths`.

Input-independent work stays legal: weight repacks, precomputed shape-keyed
tables, CUDA graph capture at boot, and KV and state reuse WITHIN one request.
Keep every change prompt-independent and model-general. The hidden prompts
differ from the public one.

## 9. Submitting

Use the Yukon CLI for every submission operation. A submission archive packages
only `editablePaths`, and it rejects symlinks and generated artifacts. Keep
`harness/protocol-adapter/target/` and `__pycache__/` out of it, since they
count against the byte cap. The ranked run is the only gate that ranks a
submission.

## 10. License

The weight file is from Prism ML (built from Qwen/Qwen3.5-9B) under the Apache
License 2.0; see the pinned repository for its terms. This repository
distributes no model weights. The vendored engine (`llama.cpp/`) is MIT
(`llama.cpp/LICENSE`). The Engine Protocol adapter and its Rust dependencies
are MIT OR Apache-2.0 as listed in `THIRD_PARTY_NOTICES.md`.

## 11. Speculative decoding and the serial stream

11.1 The target decides every token. A speculative cycle commits only drafts
equal to the target's own greedy choice, plus the target's bonus token.

11.2 Greedy verify (MTP or DFlash) runs a batched forward over d + 1 positions per cycle;
the serial path decodes one token at a time. On GB10 with the pinned GGUF
packing, those paths are deterministic run to run but are not always
byte-identical: batched verify can flip a near-tie argmax that serial decode
would have taken (`docs/engineering/engine-probe.md`). The tolerance gate and replay rules
in section 5.4.1 are set from that near-tie behaviour,
not from a separate FP8 activation-quant path.

11.3 A logits gap changes a token only where the top logits are close (section
5.4.1).

11.4 Four consequences follow for the timed oracles:
(a) the serial tape is the fidelity oracle for every depth (section 5.4);
(b) the timed free-run oracle is authored under the same spec the leg replays,
one golden per depth, so a leg is never checked against another depth's
trajectory;
(c) a free-run trajectory can fork from the serial one at a near-tie, which is
why each depth has its own timed oracle and why the gate prices divergence
rather than forbidding it;
(d) your engine can fork from the reference's trajectory at a near-tie, which is
why the timed window replays a divergence through the reference
(section 5.4.1) rather than failing it.
