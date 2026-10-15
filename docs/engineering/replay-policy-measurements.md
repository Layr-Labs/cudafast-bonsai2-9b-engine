# Timed-window replay policy: measurements

This document records the measurements behind the two replay keys in
`fixtures/bonsai2_9b_cuda_v1_track.json`. Prompt names, token values and raw
results are not in this repository.

## Policy (pinned benchd)

From `contract.rs` and `official.rs` of the benchd that `tools/fetch-benchd.sh`
pins:

- `timed_token_tolerance_per_thousand` (**N**): after the reference replay, a
  leg passes when `mismatches × 1000 ≤ N × timed_positions`.
- `timed_token_near_tie_relative_gap` (**G**): a mismatch counts toward **N**
  only if the candidate token is the reference's **second choice** and the
  top-1 to top-2 relative gap is `≤ G`. Any other mismatch refuses the run with
  `TIMED-DIVERGENCE-NOT-A-NEAR-TIE` before the budget rule.

The local modes do not replay. They demand an exact match.

## Bonsai 2 27B rehearsal (Spark box 8)

Setup: the Bonsai 2 27B rehearsal weights, eight long-copy rehearsal prompts,
serial goldens, the cuda-engine and llama-resident timed path of benchd's
candidate leg, 129 timed positions per leg (decode seed and 128 steps). Each
mismatch was teacher-forced on a serial reference to get the top-1 to top-2
gap.

| Run | Legs | Result |
|---|---:|---|
| Honest candidate, MTP depth 1 to 3 | 24 | 4 of 8 prompts exact; worst 2 mismatches (15.50 per thousand); worst gap 0.003843; every mismatch was the reference's second choice |
| Serial null control | 2 | 0 mismatches |
| Wrong candidate (second choice at every 10th step), MTP depth 1 | 1 | 35 mismatches (271.32 per thousand), 20 of them not the second choice |

Depth did not change the honest stream on this build: MTP 1, 2 and 3 gave the
same tokens for each prompt.

## Official-mode checks

- Spark box 8, 2026-10-09, benchd `mlxfast-bench` `e7cd2e9`, the 27B rehearsal, one
  prompt, 3 pairs per run, box calibration from `tools/calibrate-box.sh`:
  - a wrong serial engine (second choice at every 10th step) was refused with
    `TIMED-DIVERGENCE-NOT-A-NEAR-TIE` (relative gap 0.067, limit 0.01);
  - a null run (the same engine on both legs) passed: score 0.995, 0 token
    mismatches in every pair.
- Spark box 5, 2026-10-10, the benchd that main pinned then (bench
  `f03f8c4`), the Qwen3.5-9B stand-in, the hidden timed pool, the reference
  tree at the pinned `baseline_reference_commit`, a box calibration, and every
  ranked step of `.github/workflows/benchmark.yml`, 3 pairs per run:
  - an honest candidate at MTP depth 3 passed: score 1.531, 0 token mismatches
    in every pair;
  - a wrong serial engine (second choice at every 10th step) was refused with
    `TIMED-DIVERGENCE-NOT-A-NEAR-TIE` (relative gap 0.19, limit 0.01).

## Decision

| Key | Value | Reason |
|-----|------:|--------|
| `timed_token_tolerance_per_thousand` | **40** | The honest worst case is 15.50 per thousand (2 of 129). 20 allows only 2 mismatches per window, so one more near-tie flip fails an honest leg. 40 allows 5 per window and is about 7 times below the wrong control (271.32). |
| `timed_token_near_tie_relative_gap` | **0.01** | The honest worst gap is 0.003843, and every honest mismatch is the second choice. The wrong controls had gaps of 0.067 and more, or were not the second choice. |

## Open items

- The 24-leg measurement was not repeated on the Qwen3.5-9B stand-in.
- Measure again on the Bonsai 2 9B CUDA track weights before launch.
