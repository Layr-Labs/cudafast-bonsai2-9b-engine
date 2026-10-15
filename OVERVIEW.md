# Bonsai 2 9B CUDA challenge — overview

One-page map for co-organizers and participants. Each section states **why** it
matters; details live in the linked documents.

## What this project is

- **Task.** Speed up pinned Bonsai prefill and decode on a DGX Spark (GB10),
  single stream, without breaking correctness rules.
  **Why:** Ranked score is paired speedup on the same box and prompt; faster
  compute must not trade away verifiable outputs.
- **Target.** `prism-ml/Ternary-Bonsai-2-9B-gguf`
  `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`: Bonsai 2 9B, a ternary PQ2_0 trunk
  with prism.hadamard v1 transforms and one in-file MTP layer in Q8_0.
  **Why:** The ranked score measures the real model, pinned by sha256 and bytes.
- **Track id.** `bonsai2-9b-cuda-v1` (fourth GitHub runner label).
  **Why:** Workflow dispatch and calibration are keyed to this label.
- **Editable.** `llama.cpp/`, `harness/`, `mtp-head.manifest.json` (draft depth
  1–6). All other paths are trusted and replaced at intake.
  **Why:** Sealed harness hash and sandbox contain participant surface area.
- **Frozen formats.** GGUF packing, in-file MTP tensors, KV f16, batch 1,
  context 8192 ([contract §3.4](docs/participant-contract.md)).
  **Why:** Scoring compares against a fixed reference model, not a re-quantized
  variant.
- **Start here.** [TASK.md](TASK.md) (rules), [README.md](README.md) (setup),
  [docs/participant-contract.md](docs/participant-contract.md) (rationale).

## How a submission flows

1. **Intake** (`tools/router/`): keep editable paths only, overlay on armed
   base, push `sub/<id>`, dispatch `.github/workflows/benchmark.yml`.
   **Why:** Trusted tree + participant diff is the only artifact that runs.
2. **Checks:** manifest lint, static review, box preflight, sandboxed build.
   **Why:** Refuse bad pins, leaks, and out-of-contract edits before GPU time.
3. **Measurement:** benchd runs **3 official pairs** (serial control leg, then
   candidate at declared depth). Pair k measures live golden (k - 1) mod 3, so
   every job measures all three hidden coding prompts, one pair per prompt.
   **Why:** Paired legs cancel thermal drift; three prompts stabilize variance.
4. **Result:** nonce-bound score block on the job log; the job uploads one
   artifact, the sealed `score.json`, which Yukon reads.
   **Why:** Score is commit-revealed, and the one artifact holds no secret.

## How scoring works

- **Formula.** `composite = prefill_gain^0.25 × decode_gain^0.75`.
  **Why:** Decode is weighted higher for this product shape.
- **Decode.** Decode time per token is the decode window: the decode run's
  time divided by its N tokens, with no seed prefill in it. Prefill is its
  own phase.
- **Pairs.** `official_pairs: 3`, `official_pair_combine: mean` — each pair
  gated against its own control; sealed score is the mean of pair composites.
  **Why:** One bad pair fails the job; the mean is not a ratio of pooled times.
- **Floors.** 0.95 prefill and decode speedup per pair (official iterate).
  **Why:** Blocks “fast but wrong” or noise wins.
- **Timed window.** Ranked replay policy: `timed_token_tolerance_per_thousand`
  **40**, `timed_token_near_tie_relative_gap` **0.01** (measured on the Bonsai 2
  27B rehearsal: 24 honest MTP legs and one wrong control; see
  `docs/engineering/replay-policy-measurements.md`). On the earlier stand-in
  Qwen3.5-9B weights it was checked once in official mode (one honest MTP
  depth-3 run, one wrong control); the 24-leg measurement was not repeated.
  **Why:** In the measured rehearsal streams, honest MTP streams diverged from
  the serial oracle only on reference near-ties, and the wrong control failed
  with margin.
- **Calibration.** Control legs outside the box band (e.g. +5% prefill, +3%
  decode vs calibration) are refused.
  **Why:** A sick box must not accept or reject submissions unfairly.

## What prevents cheating

- **Sandbox user `bonsaisub`:** no network, read-only checkout, writable job
  dirs only ([docs/sandbox-design.md](docs/sandbox-design.md)).
  **Why:** Participant code cannot exfiltrate or read hidden prompts from disk.
- **Container profile:** engine image `--network none`, read-only mounts,
  dropped caps for ranked builds.
  **Why:** Even with docker on the job account, the scored path is constrained.
- **Sealed score + harness hash:** benchd seals `score.json` and computes the
  harness hash (including `llama.cpp/`). Static review checks the editable
  surface.
  **Why:** Tampering after measurement does not change the logged seal.
- **Hidden prompts:** three hidden coding prompts (`live_goldens`), replaced
  by a pool refresh; the final ranking will use a held-out pool (not yet authored for this track)
  ([docs/pool-refresh-and-final-rescore.md](docs/pool-refresh-and-final-rescore.md)).
  **Why:** Memorizing one prompt does not generalize.

## How it was tested (measured numbers)

Sources: [docs/engineering/engine-probe.md](docs/engineering/engine-probe.md),
[docs/engineering/resident-validation.md](docs/engineering/resident-validation.md).

| Measurement | Result |
|---|---|
| llama-bench tg128, PQ2_0, Bonsai 2 27B, GB10 | 30.9 ± 0.3 tok/s |
| llama-bench tg128, PTQ1_0, same | 37.1 ± 0.1 tok/s |
| llama-bench pp512, PQ2_0 / PTQ1_0 | 1042 ± 40 / 987 ± 8 tok/s |
| Plain greedy decode determinism (3×256 tok, 5 runs, 2 boxes) | byte-identical token ids |
| Community MTP draft-mtp, code prompt | 30.2 → 54.3 tok/s, 71% accept |
| Serial goldens, 8 fresh prompts, recorded on two Sparks | byte-identical across boxes |
| Resident vs llama-server (`--ctx-checkpoints 0`), 7 prompts × 128 tok | byte-identical token ids |
| Resident teacher-forcing / golden `top_logits` replay | bit-identical logits to the recording |
| benchd calibration, serial control leg, 4 passes (one Spark) | CV 0.23% decode, 0.28% prefill |
| Resident e2e (`test_adapter_e2e.sh`), depths 0 / 1 / 2 / 3 | 27.8 / 40.4 / 50.2 / 50.4 tok/s; deterministic per depth |

## Current state

| Area | Status |
|---|---|
| C++ resident + adapter wire | Landed; MTP partial accept, teacher forcing, tests |
| Engine image `bonsai9b-engine:v1` | Dockerfile + sandboxed `tools/engine/build.sh` |
| Fixture / lint / doc consistency | Bonsai 2 9B target pins; pool-9/10/11 goldens and per-depth oracles recorded on Spark box 2 and pinned (sha256 + bytes; files in R2); armed (`official_scoring_enabled: true`) after the calibration of Spark box 2 |
| Ranked runner | Spark box 2 (runner label `bonsai2-9b-cuda-v1`); the box setup is in `Layr-Labs/cuda-setup-scripts` |
| benchd | public `Layr-Labs/benchd` `main` channel, dist commit and source commit pinned in `tools/fetch-benchd.sh` |

## Before launch

| Item | Owner |
|---|---|
| Golden/oracle pins + `official_scoring_enabled` | Organizer |
| Per-box calibration files under `/srv/bonsai9b/calibration/` | Organizer |
| `bonsai9b-box-runner` installed and enabled on contest Sparks | Organizer |
| 9B weight format + MTP head confirmation | PrismML |
| Public repo / runner labels live on GitHub | Organizer |

## Document map

| Question | Document |
|---|---|
| Overview (this page) | `OVERVIEW.md` |
| Rules and commands | `TASK.md`, `benchmark.json` |
| Why each rule exists | `docs/participant-contract.md` |
| Resident wire | `docs/resident.md` |
| Sandbox and egress | `docs/sandbox-design.md` |
| Submission intake | `docs/submission-restriction-spec.md` |
| Pool refresh / final score | `docs/pool-refresh-and-final-rescore.md` |
| Track design (names, staging paths) | `docs/design.md` |
| Engineering measurements | `docs/engineering/README.md` |
| Nemotron port log (reference) | `docs/bonsai9b-port-notes.md` |
| CI scope | `docs/ci-coverage.md` |
