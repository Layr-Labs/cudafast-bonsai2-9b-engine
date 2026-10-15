> Nemotron engineering log kept for reference during the Bonsai 2 9B port.

# Engineering log — Nemotron 3.5 Lightning

The organizer's engineering record for this track: how it was derived, what was
measured on the Spark cluster, and what was changed as a result.

This repository keeps the rule set of
`Layr-Labs/cudafast-qwen38-125b-a6b-engine` @
`5707d4f23362a7208483028e6d34519267266729`. It swaps the engine and the model.

## 1. What carried over unchanged

- **Scoring:** composite `prefill^0.25 * decode^0.75`, 2 pairs, per-box paired
  baseline, floors 0.95 and ceiling 5.0, batch size 1, 128 decode steps, the
  10% token-tolerance gate, and depths 1 to 6. (The seed is 4096 tokens here,
  not 1024; see section 7.7.)
- **The surface:** Engine Protocol v1 and the adapter's verb logic, counters
  and fail-closed checks (`harness/protocol-adapter/src/{protocol,adapter,engine,mock,tests,resident}.rs`).
- **The resident:** the resident socket wire and the per-leg
  `serve-up.sh --boot/--stop` contract with benchd.
- **The pipeline:** the trusted pipeline: preflight, measure-and-score,
  calibrate-box, stage-baseline-workspace, fetch-benchd, fetch-goldens, the
  static review and the manifest lint.
- **The job:** the ranked job's shape, holding no credential, taking the GPU
  lock and waiting for quiescence.

## 2. What changed, and why

| Area | Reference | This track | Why |
|---|---|---|---|
| Engine | vendored ds4 (C/CUDA, GGUF only) | vendored vLLM v0.27.1 Python source over the pinned image's compiled build | ds4 has no Mamba-2, no ModelOpt NVFP4 and no `nemotron_h`; vLLM runs all three and is NVIDIA's recommended stack on GB10 |
| Weights | a GGUF conversion | the official ModelOpt safetensors, 60 files pinned | traceable to NVIDIA's checkpoint; no conversion step to trust |
| MTP head | a separate Q8_0 GGUF | `mtp.*` inside shard 52, BF16 | that is how NVIDIA ships it |
| Resident | `ds4-resident` (C) | `harness/vllm_resident` (Python), same wire | the Rust client is unchanged |
| Backend Rust | `ds4_backend.rs` + FFI | `session_backend.rs`, resident only | the in-process FFI path had no vLLM equivalent |
| Isolation | host process | pinned container, `--network none`, read-only snapshot and tree, fresh caches per leg | Python engine code can reach anything the host allows |
| Editable paths | `ds4`, `harness`, `mtp-head`, `mtp-head.manifest.json` | `vllm`, `harness`, `mtp-head.manifest.json` | `mtp-head/` held no weights and was never meant to be editable |
| Custom kernels | inside `ds4/` | `harness/kernels/` | inside benchd's harness-hash roster, so kernel edits are in the sealed provenance |
| KV backend | `contiguous` | `vllm-paged` | vLLM is paged; the benchd branch must accept the value |
| Build cache | `tools/ds4/build-cache.sh` | none | the build is cargo plus a file copy, and a cache would only add risk |
| Box runner | macOS/launchd | Linux/systemd | the boxes are DGX Sparks |
| Launch declaration | MTP depth 1 | serial | the reference's own contract calls the launch candidate serial; the null control must be about 1.000 |

## 3. Reference defects fixed here

These were in the reference and must not come back. `tools/check-doc-consistency.py`
holds the first three in CI.

1. **Editable paths.** TASK.md said there were 3 editable paths and that the
   engine was read-only, while `benchmark.json` listed 4, including the engine.
2. **Head re-quantization.** TASK.md said the head could be re-quantized, and
   the contract and `mtp-head/README.md` said it could not.
3. **Depth coverage.** The preflight and `serve_configuration.depths` covered
   only depths 1 to 3, while 1 to 6 were permitted and had oracles.
4. **Category.** `category: "swift"`.
5. **The launch candidate.** The contract called it serial, but the shipped
   declaration enabled MTP.
6. **Stale docs.** The contract described a `results.json` emitter; benchd
   seals `score.json` itself.
7. **A phantom sealed field.** The docs cited `effective_mean_draft_len`, which
   benchd does not seal.
8. **Section references.** Code comments cited contract section 11.4(c), which
   did not exist.
9. **Stray markers.** The adapter's `lib.rs` carried stray `__REDRAW_MARK_*`
   constants.
10. **Macintosh leftovers.** macOS runner docs and a Seatbelt profile on a Linux
    track.
11. **Silent socket-path overflow.** A long workspace path failed only at the
    resident's bind(). `serve-up.sh` now refuses by name.
12. **Declaration cap.** The pinned head (2.67 GB) was larger than the 2 GiB
    declaration cap. The cap is now 3 GiB, and the head's own size is recorded
    in the fixture.

## 4. Decisions a reviewer should know about

- **Activations are part of the frozen format.** The checkpoint declares its
  experts W4A16 (`hf_quant_config.json`: `W4A16_NVFP4`), so FP4 activation
  quantization is a format change and is prohibited. On GB10, vLLM runs these
  layers through Marlin W4A16 kernels; faster W4A16 kernels are the intended
  kernel lever.
- **The MTP head may not be re-quantized** (`mtp_head.requantization_permitted:
  false`). It is BF16, about 2.5 GiB. A re-quantized head is a large decode
  lever, and allowing it would be a deliberate rules change.
- **Prefix caching is off in the reference engine configuration,** and reuse
  across phases is prohibited by rule (contract section 8). As on the reference
  track, the engine is editable, so this is a rule plus review, not a runtime
  gate.
- **benchd's env var names are kept:** `MLXFAST_QWEN_MTP_TRACK_ID` and
  `DS4_RESIDENT_SOCKET`. benchd reads and injects them under those names.

## 5. Open items (external)

1. **benchd — this track runs its OWN benchd.** It is the GumbiiDigital fork
   `GumbiiDigital/mlxfast-bench`, branch `gumbii/nemotron35-own`: upstream
   `main` @ `a0c95cd` plus this track's contract-oracle entry and two
   behavioural changes (section 7.7): the 60 °C GB10 cool gate and the
   fixture-declared median of several timed prefill passes.
   `tools/fetch-benchd.sh` pins dist commit `1dfdf28` (source `ad17f32`,
   linux-aarch64 only, rebuilt with no cache to the same bytes) and refuses any
   other `source_commit`. The fork's workspace suite passes (1207 tests). The
   bench-runner "Text file busy" test flake is upstream's; it passes
   single-threaded. The earlier upstream PR branch
   (`track/nemotron35-lightning-contract-fixture`) is superseded.
2. **Goldens — recorded, pinned and staged.**
   The 56 live-pool goldens (8 prompts, each a serial golden plus six
   per-depth oracles) and the public golden are recorded, pinned and staged on
   the boxes (sections 7.4, 7.7 and 7.10). R2 is not used: the live pool is
   kept on the jump box and the held-out pool in the organizer archive (see
   docs/pool-refresh-and-final-rescore.md, "Where the goldens live").
3. **`baseline_reference_commit` — pinned.** The repository is published
   (private) as `GumbiiDigital/cudafast-nemotron35-lightning-30b-a3b-engine`.
   The fixture pins the reference tree at `3a9a41c`, whose engine paths
   (harness, vllm, serve-up, the MTP manifest) are identical to the tree the
   round-3 calibrations measured (`4e4c0be`).
4. **Timing noise — addressed; see section 7.7 for the calibration.** The
   seed is 4096 tokens, and the prefill phase is one warm-up pass followed by
   the median of five timed passes. The gate is 60 °C.
5. **GB10 temperature — resolved.** `nvidia-smi` reports a GPU temperature on
   all eight Sparks, and the cool gate is 60 °C (section 7.5).
6. **Logit ties — resolved by the timed reference replay (section 7.8).**

## 6. Verification on the Spark cluster (before arming)

1. **Bring-up, per box:**
   - `tools/box-runner/install.sh` and the systemd unit;
   - `docker pull` of the image by digest;
   - `./tools/fetch-benchd.sh && ./setup.sh`, with every pin passing.
2. **Engine sanity:** boot a serial resident and check that the greedy output
   on the public prompt is coherent and agrees with an HF transformers
   reference. Boot at depths 1 to 6 and check that every depth runs and
   `spec_acceptance_rate` is sane.
3. **Protocol:**
   - `tools/three-flows-dry-run.sh` against the real resident
     (`SERVE_UP_LAUNCHER` unset);
   - `./tools/bonsai2-9b-measure-and-score.sh --preflight-only`.
4. **Determinism:**
   - serial twice gives identical tokens;
   - serial against mtp1 to mtp6 stays inside the 10% budget;
   - the same holds across boxes.
5. **Calibration:** `tools/calibrate-box.sh` on every box, 4 passes, a CV of 1%
   or less.
6. **Null control:** the reference against itself scores about 1.000 on every
   box. The same tree with MTP depth 3 declared scores above 1.
7. **Negative tests:** each of these must be refused:
   - an edit outside `editablePaths` (static review);
   - an over-budget surface;
   - depth 7;
   - `"source": "remote"`;
   - a resident that tries the network;
   - a preset `VLLM_*`, `HF_*` or `RESIDENT_*`;
   - a candidate that fails the gate or regresses more than 5%.
8. **End to end:** a `workflow_dispatch` of `benchmark.yml` on one runner, then
   on the fleet, with `score.json` sealed and printed to the job log. Arm only
   then.

## 7. Findings

### 7.1 Engine bring-up on GB10 (2026-09-26, Sparks 5–8)

- **First boot.** The first real `VllmBackend` boot worked as written: about
  220 s for load, FlashInfer autotune and CUDA graph capture, and about 290 s
  with MTP. The resident loads 20.1 GiB of weights and gives the KV cache
  32 GiB inside the 0.45 reservation.
- **Greedy output.** Greedy output on the public long-copy prompt is coherent.
  The model copies the passage.
- **MTP acceptance.** It is high on this prompt. The mean committed tokens per
  verify cycle were 1.97, 2.91, 3.66, 4.57, 5.57 and 6.40 for depths 1 to 6.

### 7.2 Determinism: the stock kernels are not reproducible (fixed)

- **The symptom.** With vLLM's default kernels, the same 1024-token prefill
  gave different logits run to run, within one resident and across boots. The
  seed logits swung by up to 0.6. Greedy streams forked a few tokens in, which
  would make the serial control leg fail its own golden at random.
- **Bisection.** Per-layer output hashes across identical prefills located
  the first difference at `model.layers.29.mixer.experts`. Every earlier layer
  was bit-identical, including 13 earlier MoE layers. The router output and the
  shared experts were also identical, and so were the routing decisions
  (expert ids and weights).
- **The cause.** The Marlin NVFP4 MoE gives different outputs when it is
  replayed on one fixed captured input. The checkpoint's FP8-quantized
  activations then amplify a last-bit difference into a changed argmax.
- **What did not help:** eager mode (no compile, no CUDA graphs), Triton
  attention, moving the FP8 GEMMs off FlashInfer, and the humming MoE without
  its batch-invariant flag. The global `VLLM_BATCH_INVARIANT` cannot start at
  all: it forces the W4A16 linears onto CUTLASS W4A4.
- **The fix, now the reference engine:**
  - the humming W4A16 MoE (`moe_backend="humming"`) with its batch-invariant
    kernels, via a vendored-vLLM patch that adds `VLLM_HUMMING_BATCH_INVARIANT`;
  - the Mamba FP8 projections on CUTLASS (`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`);
  - a second vendored patch that gives humming a 273 GB/s bandwidth fallback,
    because GB10's NVML exposes no memory clock.
- **Two more sources of box-to-box drift, both pinned:**
  - *The KV cache size.* vLLM profiles it at boot, and the profile differs box
    to box (the CUDA graph memory estimate varies). A different cache geometry
    changed the numerics. The fixture now pins
    `RESIDENT_KV_CACHE_MEMORY_BYTES` = 4 GiB, which `backends.py` passes as
    `kv_cache_memory_bytes`.
  - *Triton autotune of the Mamba-2 SSD kernels.* It picked different tile
    configs on different boots. A third vendored patch,
    `layers/mamba/ops/ssd_pinned_configs.py`, keeps exactly one config for
    each of the five SSD kernels and refuses to start if that config is
    missing. `VLLM_MAMBA_SSD_AUTOTUNE=1` restores autotune for experiments.
- **Result.** On all four Sparks, 10 of 10 serial runs and 10 of 10 MTP-3 runs
  give one stream. The stream is bit-identical ACROSS the four boxes, and the
  MTP-3 stream equals the serial stream exactly (0 of 129 tokens differ).
- **The cost.** Serial decode is about 4.5% slower than the nondeterministic
  Marlin path (78.1 against 81.9 tok/s).
- **What it means for participants.** A faster kernel that breaks
  reproducibility fails the gate at random. `tools/determinism-probe.py`
  checks it on the box.

### 7.3 Timing noise

- **The cold first prefill.** The first 1024-token prefill after a boot was
  about 2.8× slower than the rest, from lazy JIT and autotune. benchd times the
  first prefill of every leg and the fixture declares 0 warm-up runs, so the
  resident now runs an input-independent warm-up before it reports ready: three
  rounds of a synthetic 1024-token prefill plus 64 decode steps.
- **Back-to-back runs in one resident** (client timing, 10–12 runs, three boxes):

  | Phase | Speed | CV |
  |---|---|---|
  | decode | 81.5–81.9 tok/s | 0.40–0.61% |
  | prefill | 5.2–5.4k tok/s | 2.0–2.7% |

- **What drives the prefill spread (profiled).** A torch.profiler run of one
  in-process prefill shows it is GPU-bound. At 1024 tokens, 196 ms of the
  201 ms wall is kernel time, and the humming W4A16 MoE alone is 112 ms. At 64
  tokens the MoE still takes 53 ms, because nearly every expert is touched.
  So a prefill streams most of the NVFP4 expert set from memory: a ~80 ms
  floor at 273 GB/s, plus ~0.12 ms per token. Capturing piecewise CUDA graphs
  up to 1024 tokens changed nothing, which confirms that launch overhead is
  not the cost.
- **Idle makes it worse.** In one resident, back-to-back 1024-token prefills
  run at 203–209 ms. After an idle of 60–240 s they run at 199–239 ms, while
  decode is unchanged (12.4–13.2 ms/token). The likely mechanism is clock
  ramp-up (DVFS) after idle: GB10's NVML exposes no memory clock to confirm
  it. benchd's order for every timed window is an ungated warm-up leg, then
  the cool gate (minutes of idle), then ONE timed prefill. So every measured
  prefill is a post-idle prefill.
- **benchd calibration, 4 passes, first rounds (2026-09-26):**

  | Box | Passes done | Prefill s/tok per pass | Prefill CV | Decode CV |
  |---|---|---|---|---|
  | Spark box 4 | 4/4 | 204.3, 208.0, 195.4, 207.9 µs | 2.9% | ~0.4% |
  | Spark box 8 | 4/4 | 194.8, 214.4, 201.7, 214.6 µs | 4.7% | ~0.8% |
  | Spark box 8 (round 1) | 4/4 | last pass 197.9 µs | 8.4% | — |
  | Spark box 3 | 1/4 | 182.8 µs | — | — |
  | Spark boxes 1, 2, 7 | 0/4 | cool gate refused pass 1 (section 7.5) | — | — |

  Decode meets the fixed 1% maximum. Prefill does not, on any box. benchd
  refuses the calibration (`CALIBRATION-CV-EXCEEDED`), as it should.
- **Options considered (the gate was not loosened):**
  1. A longer timed prefill, for example a 4096-token seed. The fixed cost
     and the ramp are then a smaller share of the window. This changes
     `seed_tokens` from the reference rule set's 1024, and every golden would
     be re-recorded.
  2. A benchd change to time the prefill as the median of several prefills
     inside the window. This belongs to benchd.
  3. A per-phase engine-side "re-warm" before the timed prefill. It is NOT
     recommended: it runs work outside the timed window to beat the cool
     gate's intent, and participants would copy it.

### 7.4 Goldens (2026-09-26)

- **What was recorded.** `tools/record-goldens.sh --apply` ran independently on
  Spark boxes 5 and 6 and wrote 15 files: the 8 timed-pool goldens, the live
  golden at depths 1 to 6 (`botany.mtpN`), and the public golden. All 15 are
  **byte-identical across the two boxes**, and both boxes pinned identical
  fixtures. The set is staged in `~/nemotron35/goldens` on all eight Sparks.
- **Depths 1–3 reproduce serial exactly.** Their goldens are the same bytes as
  the serial `botany` golden.
- **Depths 4–6 diverge from serial.** They diverge early in the decode
  window, and most of the 128 decode tokens differ from serial after that. The divergence
  is deterministic: it is the same on both boxes, and depths 4, 5 and 6 share
  one file. A verify pass of 5 or more tokens evidently takes a kernel path
  whose numerics differ from the 1-token step. The per-depth goldens exist to
  absorb exactly this, so each depth is gated against its own golden. It is
  still an open engine-side improvement: a participant who makes the
  wide-verify path agree with serial loses nothing.

### 7.5 The cool gate on GB10: an idle resident holds the die near 50 °C

- **The symptom.** benchd runs a cool gate (≤ 50 °C on CUDA) before every timed
  phase, in calibration and in the ranked path alike. It aborts when the GPU
  is above 50 °C and has not dropped 0.25 °C in 90 s, after 180 s of waiting.
  In the first two calibration rounds, 4 of 7 boxes were refused on pass 1:
  `GPU is hot and not cooling down`, stalled at 51–52 °C.
- **The cause is the platform, not the engine.** On driver 580.159.03, ANY
  CUDA context holds the GB10 in P0, at about 10.7 W and 2.4 GHz, even with
  zero utilization. A one-element torch tensor does it. The bare box idles in
  P8 at 3.8–4.0 W and 45–46 °C. With a context the die plateaus at 47–49 °C
  on a quiet box, and at 51–52 °C after the boot's heat soak with neighbours
  busy. The loaded vLLM resident's CPU side is idle during the gate (below
  0.5% per thread).
- **Upstream sees it too.** benchd's CUDA runbook records the same thing on the
  Qwen fleet: "the loaded resident held the die at 51 to 52 C". The runbook's
  claim that the ranked path has no gate is out of date: `official_cool_gate`
  gates every ranked phase.
- **What was not done.** The gate was not loosened, clocks were not
  locked, and persistence mode was not changed. The gate and its temperature are benchd's, and clock or
  driver settings are box configuration. Whether a box passes is decided by
  its cooling and ambient. The resolution needs a benchd change, or better
  airflow on the boxes that stall.

### 7.6 Experiment: gate temperature × prefill length (2026-09-26, all 8 Sparks)

- **Method.** `gateprobe.py` (scratch, not in the tree) repeats benchd's timed
  window against one booted serial resident per box:
  1. an ungated warm-up (a prefill of length L, then 128 decode steps);
  2. a wait until the GPU is at or below T °C, with benchd's stall rule
     tracked (180 s floor, 90 s without a 0.25 °C drop);
  3. one timed prefill of L tokens and 128 timed decode steps.

  It runs 4 reps per condition per box: the same sample count as a
  calibration. The limits: timing is client-side, and one resident per box
  means no boot-to-boot variation. So this ranks the options; it does not
  replace a benchd calibration.
- **Results** (per-box CV across 4 reps; a box counts as passing when prefill
  and decode are both ≤ 1%):

  | Gate | Prefill | Prefill CV median / max | Decode CV max | Boxes passing | Gate refusals | Mean prefill |
  |---|---|---|---|---|---|---|
  | 50 °C | 1024 | 2.82% / 4.84% | 1.00% | 0/8 | 18/32 | 208 ms |
  | 50 °C | 2048 | 1.75% / 2.54% | 0.71% | 2/8 | 22/32 | 305 ms |
  | 50 °C | 4096 | 0.89% / 1.69% | 1.00% | 5/8 | 20/32 | 501 ms |
  | 60 °C | 1024 | 2.05% / 2.45% | 0.75% | 1/8 | 0/32 | 203 ms |
  | 60 °C | 2048 | 1.18% / 2.42% | 0.80% | 1/8 | 0/32 | 306 ms |
  | 60 °C | 4096 | 0.68% / 1.97% | 0.64% | 6/8 | 0/32 | 508 ms |

- **Reading.**
  - A 60 °C gate removes every refusal: every wait was 0 s, starting at
    56–57 °C. At 50 °C only Spark boxes 4 and 8 never refused.
  - The longer prefill is what brings the prefill CV down.
  - 4096 at 60 °C is the best combination. Spark box 3 (1.26%) and Spark box 4
    (1.97%) were still above 1%, and a 4-sample CV is itself noisy.
  - Decode is under about 1% everywhere, and 4096-token context costs about
    0.2 ms per decode step.
- **Not done in this experiment.** A real benchd calibration at 60 °C: the
  gate temperature is a compiled benchd constant. Section 7.7 does it with a
  rebuilt benchd.

### 7.7 The fix: a track-specific benchd, a 4096-token seed, one warm-up and the median of five (2026-09-27)

- **What changed, and where.**

  | Change | Where | Why |
  |---|---|---|
  | GB10 cool gate 50 °C → 60 °C | benchd fork `gumbii/nemotron35-own` | A loaded resident holds the die at 49–52 °C in P0 (section 7.5). `GPU T.Limit Temp` is a margin (~96 °C limit); thermal-slowdown counters stayed at 0 µs. |
  | `seed_tokens` 1024 → 4096 | fixture | The ~80 ms bandwidth floor is a smaller share of a longer prefill (section 7.3). |
  | `official_prefill_warmup_runs` 0 → 1 | fixture; the adapter accepts a repeated `prefill` in one prefill phase | An unmeasured pass after the gate idle brings the clocks back up before timing. |
  | `official_prefill_timed_runs` = 5 (new, optional field; median) | benchd fork plus fixture | It damps per-sample noise and drops one slow pass. |

- **The benchd build.** `scripts/build-dist-linux.sh` builds it on a GB10.
  Rebuilding with `BENCHD_DIST_NO_CACHE=1` gives the same bytes. Dist
  `1dfdf28`, source `ad17f32`, benchd sha256 `f95d5949…`. The workspace suite
  passes (1207).
- **Goldens.** All 15, at 4096 tokens, were recorded independently on Spark boxes 5
  and 6 and are byte-identical. The public prompt was extended to 4201
  tokens, and the private excerpts to about 5000 words (the old excerpts are
  byte-identical prefixes).
- **Calibration** (`benchd calibrate-baseline`, 4 passes, a fresh resident
  per pass):

  | Round | Setup | Boxes under 1% on both axes | Prefill CV |
  |---|---|---|---|
  | 1 | 1024 seed, 50 °C gate | 0 of 2 that finished (4 of 7 refused by the gate) | 2.9–8.4% |
  | 2 | 4096 seed, 1 warm-up, 60 °C, 1 timed pass | 3 of 6 | 0.37–2.70% |
  | 3 | 4096 seed, 1 warm-up, 60 °C, **median of 5** | **8 of 8** | **0.17–0.89%** |

  In round 3, decode CV was 0.25–0.84%, with no gate refusals. The reference
  measures prefill at 126–127 µs/token (about 0.52 s per 4096-token prefill)
  and decode at 16.6–16.9 ms/token (benchd-timed, 4096-token context).
- **Status of those files.** The round-3 calibrations
  (`~/nemotron35/cal4096.json` on each box) were measured against a reference
  clone at `4e4c0be`. They become the official per-box calibrations once the
  published reference commit is pinned. Rerun `tools/calibrate-box.sh` then;
  the numbers should not move.

### 7.8 Logit ties make the timed window's exact-match check brittle (resolved: reference replay)

- **What was found.**
  - The target's logits come out of the NVFP4 `lm_head` in BF16. Near the top
    of the distribution their resolution is 1/16 to 1/8, so the top two
    tokens often tie exactly. At a tie the lowest token id wins.
  - Along the reference's own 128-step continuation, every pool prompt has
    at least one exact tie, and several have four or more.
  - With the 4096 seed, every MTP depth splits from serial at an early tie in
    the calibration prompt's continuation. That is why the per-depth goldens differ from serial. The divergence
    is deterministic and byte-identical across boxes.
- **Why it matters.** benchd checks the TIMED free-run decode with an EXACT
  match on every committed token: a mismatch is a hard failure, and the 10%
  tolerance applies only to the correctness gate. So a kernel change that
  moves the last bit of the top logits at a tie position fails the timed
  window, even if it is a legitimate speed-up. The engine is reproducible, so
  the reference itself is safe. Participants, though, are held to bit-exact
  logits at those positions.
- **The decision: reference replay.** A precomputed tie list was
  tried first and does not scale. One prompt's main stream has 16 near-ties
  within one BF16 step. The tree of reference continuations grew 16 → 101 → 283
  branches by depth 3 and still did not cover the reference's own MTP streams.
- **What was built** (benchd fork source `196e586`, dist `cefe79f`):
  - The fixture declares `timed_replay_max_logit_gap` 2.0 and
    `timed_replay_off_argmax_per_thousand` 100.
  - A candidate's timed divergence (prefill token, seed or committed token) is
    deferred.
  - After the pairs, benchd boots the reference once and teacher-forces each
    deferred stream through it, then applies contract section 5.4.1.
  - Control legs and calibration never defer. The single-leg and non-official
    paths refuse a deferred divergence by name.
  - The golden format is unchanged, so no re-recording was needed.
- **How the numbers were set.** The reference's own MTP-3 and MTP-6 streams on
  all 8 pool prompts were replayed through its serial engine. The worst
  in-top-8 gap was 1.625 logits, there were at most 9 off-argmax positions in
  129, and one token fell outside the top 8. The verify pass quantizes FP8
  activations over the whole batch, so it moves logits by more than the last
  bit. The policy leaves margin over all of it.
- **Tests.**
  - In bench-core: the policy, the BF16 step, and the judgement.
  - In bench-runner: deferral at the prefill, seed and token positions, the
    exact match when deferral is off, and the refusal on the non-replay path.
  - In benchd: replay accept, gap refusal, budget refusal, and prefix positions
    skipped.
  - The workspace passes 1219 tests.

### 7.9 Eight-hour load test on all 8 Sparks (2026-09-27)

- **Setup.**
  - Window 03:30–11:30 EDT, all 8 Sparks, one JIT runner per box
    (`tools/box-runner/`, stable name `nemotron35-box-<host>`).
  - The organizer's router (`tools/router/router.py`) stands in for the
    intake service: it keeps only `editablePaths`, overlays them onto the
    armed base branch, pushes `sub/<id>` and dispatches `benchmark.yml`.
  - `tools/router/loadgen.py`, seed 20260927, keeps 6 submissions queued: null
    40%, MTP 1–6 20%, Marlin 8%, CUDA graphs 7%, garbage 10%, negatives 15%.
- **Throughput.**
  - 191 submitted, 186 finished, 5 expired unrun at the window's end.
  - 21–24 jobs per box. A ranked job takes 23 min (median), 29 min (p90).
  - About 23 jobs an hour across the fleet.
  - No infrastructure failure after the fixes below (from 04:30 on).
- **Outcomes against expectations.**

  | Variant | Outcome |
  |---|---|
  | null | 68 pass, 2 refused by the +2% acceptance band |
  | MTP 1–6 | 41 below the 0.95 floor, 1 band drift, 1 boot refused |
  | Marlin | 17 below the 0.95 floor (prefill about 0.91) |
  | CUDA graphs to 4096 | 13 pass, 1 boot refused (memory plan) |
  | garbage | 12 `TIMED-REPLAY-REJECTED`, 2 stopped first by the acceptance band |
  | 3 MB file | 10 refused by the static review |
  | depth 7 | 3 refused by the preflight |
  | trusted paths smuggled | 11 pass (the router dropped them), 1 band drift |
  | engine sets a `VLLM_*` variable | 3 pass (review-only, recorded) |

- **Fairness: the null score.**
  - 68 passes: mean 0.9959, sd 0.0058, range 0.983 to 1.009.
  - The candidate leg reads about 0.4% slow. The spread is about ±1%.
  - Per-box means run from 0.993 (Spark box 3) to 1.000 (Spark box 2).
- **MTP on the 4096 botany prompt.** Decode speed-up by depth, against serial:
  about 1.03 at depth 1, 0.94 at depth 2, 0.97 at depth 3, 1.03 at depth 4,
  0.98 at depth 5 and 0.96 at depth 6. Prefill costs 8% at depth 1 and up to
  18% at depth 6. No depth clears the 0.95 prefill floor.
- **Findings.**
  1. **The +2% acceptance band refuses unchanged code.** 2 of 70 null runs
     were refused, with decode 2.2% slower than the reference. With an sd of
     0.6% plus heat, a 2% band is about 3 sd. Options: a 3% band, or one retry
     on a band miss.
  2. **Spark box 6 drifted slow.** It caused 3 of the band refusals:
     its decode ran 2–3% slower than its calibration. It was power-drained
     and re-calibrated afterwards.
  3. **Memory plan after a heavy job.** One control leg's resident refused to
     boot on Spark box 4: the memory plan came out short, likely because memory
     was not yet returned from the previous job. It happened once.
- **Bugs found and fixed during the test** (all committed):
  - The supervisor did not export `BOX_RUNNER_APP_ENV` to the mint helper.
  - The JIT config file sat in a root-owned directory the runner could not
    clean; it now goes over stdin.
  - The default `work_folder` did not exist; it is now `_work`.
  - `fetch-benchd.sh` chmod-ed the read-only staged pair.
  - The idle-recycle `timeout` killed every job at 60 minutes.
  - `SERVE_UP_SOCKET_DIR=/tmp` is hidden by the container's own `/tmp`, so
    the job hung until the 3600 s health timeout. `serve-up.sh` now refuses it.
  - A cancelled job left its resident container on the GPU. The supervisor
    now removes `serve-up-*` containers on every path that ends a job.
  - The router read no box name, and did not parse the log score.
- **No hosted runners, no artifacts.** The pipeline uses no GitHub-hosted
  runner and uploads no artifact: the surface check runs on a box runner, and
  the score reaches the dispatcher through the job log
  (`tools/router/README.md`). `ci.yml` is manual-only.
- **Participant code ran as `runner` (resolved).** During this test untrusted
  build and engine code ran as the box's `runner` user, which is
  root-equivalent through `docker`. Participant code now runs as an
  unprivileged sandbox user; see `docs/sandbox-design.md`.

### 7.10 Rotation pool goes live: oracles, cross-prompt calibration, held-out pool (2026-09-28)

- **Oracles.** The 42 per-depth oracles (MTP-1 to MTP-6) for the seven other
  pool prompts were recorded on Spark boxes 5 and 6 independently. All 56
  files per box (8 serial goldens, 48 oracles) are byte-identical across
  the two boxes. The 14 already-pinned goldens re-recorded to their exact
  pins.
- **One transient.** Spark box 5 failed once while loading weights, with
  `CUDA error: operation not permitted` in `nvfp4_marlin_process_scales`.
  A resume (`--only … --steps 2,3,4,5,6`) on the same box then recorded
  every file identically.
- **Does one band per box hold for every prompt?** A full calibration was
  run on one non-botany prompt per box, on Spark boxes 1, 2, 3, 4, 7 and 8.
  - Decode lands within 0.43% of botany on every box (−0.43% to 0.00%).
  - Prefill lands within 0.64% on five boxes, and at −1.22% on Spark box 3
    (the medicine prompt).
  - Each score is a ratio against a control leg run on the same prompt in
    the same job, and the prefill band is ±5%. So one botany band per box
    stays valid, and no per-prompt bands are needed.
  - The first Spark box 3 attempt was refused on its CV (decode 1.10%, above
    the 1% maximum). A rerun passed at CV 0.58%.
- **Held-out final pool.** 8 stitched prompts × 7 goldens, each agreed byte
  for byte by two boxes (Spark boxes 1, 2, 3, 4, 7, 8). The pool is kept only in
  `.mlxfast-private/final/` on the organizer workstation, and the box-side
  recordings were deleted.
- **Recorder socket path.** It must stay short: a per-job path under the
  refresh folder came to 108 bytes, over the 107-byte AF_UNIX limit.

### 7.11 Rotation smoke test and 3-hour soak (2026-09-28)

- **Smoke test.** 2 unchanged engines passed (1.006 and 1.002) and the
  garbage engine was replay-rejected. The three jobs drew three different
  prompts (republic, travel, essays).
- **Soak, 06:47–09:47 plus drain.** 81 submissions on 8 prompts:

  | Group | Submissions | Outcome |
  |---|---|---|
  | Unchanged engines | 31 | 30 passed, mean 0.9959, range 0.982–1.009 |
  | Garbage | 10 | all replay-rejected |
  | Depth 7 | 8 | all refused by preflight |
  | 3 MB file | 2 | both refused by the surface check |
  | Edits to trusted paths | 2 | dropped by the router; both passed (about 1.00) |
  | Engine sets `VLLM_*` | 5 | all passed (0.995; observe-only) |
  | CUDA graphs | 3 | all passed (1.007) |
  | Marlin | 2 | floor failed |
  | Reference MTP | 14 | floor failed |
  | Still queued at the stop | 6 | expired, never dispatched |

  There were no infrastructure failures and no watchdog cancels.
- **One reference-leg divergence.**
  - Job `20260928-071724-55b135` on Spark box 8 (republic) failed with
    `SERIAL-CONTROL-LEG-FAILED`: the trusted serial control returned token
    1799 where the golden has 1278, at decode step 47.
  - Three later serial recordings of republic on the same box all
    reproduced the pin (`ed507fd…`) byte for byte. Republic's control leg
    passed on the other boxes.
  - So this was a one-off, not a box or golden defect: 1 in about 100
    control legs over the soak and the smoke test.
  - A control-leg failure is never the submission's fault. Such a job
    is re-run once automatically rather than scored as a failure: the
    measure script re-runs the whole measurement once, only when the trusted
    engine's own tokens diverged, after clearing every per-job directory
    (contract section 5.5). Any other control-leg failure, or a second
    divergence, ends as an infrastructure outcome (`control_failed`).

### 7.12 Adversarial review of the whole setup, and its fixes (2026-09-30 to 10-02)

An adversarial review covered five surfaces: the sandbox and container, the
scoring path and workflow, intake and the router, the box runner and
calibration, and benchd. No path let a submission forge a passing score or
read hidden data. Three paths let a team make bad runs disappear, so only
good draws reached the ranking. All three are fixed and tested on the boxes.

- **Fake score lines in the job log.**
  - The attack: participant output reaches the job log through `build.rs`
    warnings, the kernels build, and the resident log that `serve-up.sh`
    prints when a boot fails. A second copy of benchd's score marker made the
    router discard the run as infrastructure.
  - The fix: commit-reveal.
    - A per-job nonce is drawn into `RUNNER_TEMP` before any participant code
      runs, and the score is revealed after sandbox cleanup in a block tagged
      with it.
    - The parser takes only the last tagged END line, with a sha256 match
      (`docs/submission-restriction-spec.md` §10).
    - A run whose code ran but sealed no valid score is `unsealed`, the
      team's failure.
  - On the boxes: the fake marker and a complete fake block, printed as raw
    log lines, were ignored and the real score was kept.
- **`/dev/shm` between legs.**
  - The attack: `/dev/shm` is a 61 GB tmpfs in unified memory, shared through
    `--ipc host`, and it was wiped only at job start and end. A slow candidate
    could fill it so pair 2's reference leg could not boot: a final
    control-leg failure, recorded as infrastructure.
  - The fix: the candidate leg's `serve-up.sh --stop` kills the sandbox
    user's processes, runs the wipe and waits for memory. A wipe also runs
    before the measurement, and the adapter runs under `prlimit` caps.
  - On a box: a 50 GiB flood by `bonsaisub` took available memory from 106.8 to
    57.9 GiB, below the reference leg's 64.8 GiB. The real stop path removed
    it and restored 108.7 GiB in 4.7 s.
- **The final mean.**
  - The attack: `rescore_report` averaged only passed runs, so an engine could
    fail on purpose on slow draws.
  - The fix: exactly K slots per team.
    - A failure caused by the submission counts as 0.
    - Infrastructure failures are re-run up to a cap.
    - Reference-leg or band failures that persist after re-runs count as 0.
    - Contest selection counts failures the same way.
- **Sealed refusal text.** benchd's out-of-band refusal no longer seals
  control-leg timings, calibration numbers or the box name in the text. A
  refused run seals no timings at all: benchd source `9403e9f`, dist
  `a8fafeb`.
- **The egress rule did not survive reboots.**
  - What happened: a wifi-controller firmware update power-cycled all 8
    boxes. The rule that rejects all `bonsaisub` traffic had been persisted only
    through `netfilter-persistent`, which the boxes lack, so it was gone.
  - Impact: the ranked preflight probes egress and refused every job, so it
    failed closed.
  - The fix: `bonsai9b-sandbox-egress.service` runs
    `/etc/bonsai9b-sandbox/egress.sh` at every boot, which always puts the
    rule first in OUTPUT. The box runner `Requires` the unit and checks the
    rule in `ExecStartPre`.
  - Verified on all 8 boxes. The `bonsaisub` uid differs per box (981, 995,
    996), so checks match the rule by name.
- **Fairness after all fixes.**
  - 40 unchanged engines across all 8 boxes: 39 passed, mean 0.9956, sd
    0.0049. That matches the 0.9959 measured before the fixes.
  - An earlier sample of 7 runs at 0.989 was noise.
  - The `/dev/shm` wipe before each reference leg adds a few seconds of idle.
    The 60 C cool gate does not equalise start temperatures, but the soak
    shows no bias from it.
- **Candidate boot failures from GPU transients.**
  - What happened: one unchanged engine failed to boot with `CUDA error: an
    illegal instruction`, the same family as the Spark box 5 transient in 7.10.
  - Rule: a pair-1 candidate boot failure (no completed pair is sealed, so
    `paired_legs` is absent) shares the job's single re-run.
  - Rule: in the final re-score, each slot may be re-dispatched once after a
    `boot_failed`; a second one counts as 0.
  - A pair-2 boot failure is never re-run in the job: the engine has seen its
    own timing.
- **Router PATH.** After a `gh` upgrade, the router found no `gh` and
  dispatched nothing for 4 hours. Its PATH now uses mise's stable shims.
