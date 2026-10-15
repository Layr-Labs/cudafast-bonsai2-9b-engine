# The task — Bonsai 2 9B CUDA on DGX Spark

Make the target model run faster on a DGX Spark (GB10, sm_121), single stream.

The **target** is one pinned weight file from
`prism-ml/Ternary-Bonsai-2-9B-gguf` (`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`
@ `9e08fac9`): Bonsai 2 9B, built from `Qwen/Qwen3.5-9B`. The trunk is PQ2_0
(ternary, 2.13 bits per weight) with prism.hadamard v1 transforms folded into
the weights. The file holds one MTP layer in Q8_0, which uses the shared
`output.weight` as its head. The tokenizer is the Qwen 3.5 tokenizer.
It runs on the vendored PrismML `llama.cpp` fork with `--spec-type draft-mtp`.
A second pinned file from the same revision,
`Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf`, is Prism's DFlash2 block drafter for
the same trunk; it runs with `--spec-type draft-dflash`.
Layer counts, vocabulary and hidden size come from the file at runtime.

The engine is the PrismML `llama.cpp` fork vendored at `llama.cpp/`. The
scored binary is the Engine Protocol v1 adapter (`harness/protocol-adapter`).
It attaches to one C++ resident per leg (`harness/llama_resident/`,
`llama-resident`), built inside the pinned `bonsai9b-engine:v1` CUDA devel
image with the same container profile as the prior Nemotron track.

The ranked track is `bonsai2-9b-cuda-v1`. Read [README.md](README.md) for setup.
Read [`docs/participant-contract.md`](docs/participant-contract.md) for the
reasons behind every rule.

## What you may change

`benchmark.json` `editablePaths` is the authority. Its editable paths are
`llama.cpp`, `harness` and `mtp-head.manifest.json`. A submission archive
replaces those paths and nothing else.

| Path | What it is |
|---|---|
| `llama.cpp/` | The vendored PrismML fork: ggml CUDA kernels, llama runtime, the speculative loops (`draft-mtp`, `draft-dflash`), scheduling and fusion you need to go faster. |
| `harness/` | The Rust protocol adapter, the C++ resident (`llama_resident/`), and optional custom CUDA/C++ under `harness/kernels/`. A `harness/kernels/build.sh` runs in the pinned image at setup when present. |
| `mtp-head.manifest.json` | The speculative-decoding declaration: `spec.enabled`, `spec.mode` (`mtp` or `dflash`) and `spec.num_speculative_tokens`. |

The rule behind the list is simple. Code that **proposes** tokens or
**computes** the forward pass is editable. Code that **verifies**, **measures**
or **ledgers** stays trusted: `tools/`, `fixtures/`, `setup.sh`,
`benchmark.sh`, `.github/`, the pinned weight file, the pinned engine image and
benchd.

## What stays fixed

> **WARNING — the target's numerical formats are frozen.**
> The weight file's packing and transforms are frozen: no re-quantization, no
> re-representation. Hadamard/sign transforms the model needs may be fused for
> speed, not removed or approximated. The KV cache in the resident is **f16**.
> Batch size is **1**; context is **8192** (from the fixture).
>
> Do not change what the checkpoint represents. You may fuse, kernelize, graph
> and schedule so the same mathematics run faster.

> **WARNING — the drafters are pinned files.**
> The native MTP / nextn tensors live inside the pinned target file. The DFlash2
> drafter is the second pinned file, which the box stages beside the target.
> You may not ship drafter weights or change them. `mtp-head.manifest.json`
> accepts `"source": "pinned"` only.

- **The scored batch size is 1.** It is not a tunable.
- **There are two speculative arms.** `spec.mode` `mtp` (the default) is the
  native MTP layer via llama.cpp `draft-mtp`. `spec.mode` `dflash` is the
  pinned DFlash2 block drafter via llama.cpp `draft-dflash`. Other drafters are
  not part of this track.
- **The draft depth is yours to choose.** For `mtp` it is from 1 to 6; the
  non-editable envelope refuses 7 and above. For `dflash` it is 1 to 15 (the
  drafter's block of 16 holds the anchor and 15 drafts). A disabled declaration
  or depth 0 runs serial. Each run seals `effective_spec_mode` and
  `effective_spec_depth`, along with the per-cycle `acceptance_lengths` and
  `spec_acceptance_rate`.
- **The engine configuration comes from the fixture:**
  `serve_configuration.engine_env` sets context 8192, batch/ubatch limits and
  the memory plan inputs. The resident runs with no network, with the weight
  file and your tree mounted read-only, and with a fresh per-leg cache
  directory.
- **The KV backend is `contiguous`** (`benchmark.json` and the fixture
  `kv_backend`): the resident keeps one contiguous f16 KV cache per stream.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary from the
public bench repository. It needs no token.

```bash
./setup.sh
```

This command builds the adapter and the engine image from
`tools/engine/Dockerfile` (on a ranked box it uses the organizer-staged image
`bonsai9b-engine:v1`). It builds `llama-resident`
inside that image, builds `harness/kernels/` when present, and
downloads or verifies the pinned weight file (and the pinned DFlash2 drafter
when your declaration sets `spec.mode` `dflash`). It needs cargo, docker with the
NVIDIA container runtime, cmake/ninja inside the engine image, jq, python3,
curl and sha256sum.

```bash
./tools/local-baseline.sh
```

This command runs one local leg through benchd on the public golden.

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
```

This command runs the adapter's tests with no GPU. Resident tests live under
`harness/llama_resident/test/` on a GPU box with the target GGUF.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain(role) = control_seconds_per_token / candidate_seconds_per_token  (per pair)
```

Each pair is serial-anchored against its own control leg, so a faster candidate
scores above 1 on that pair. With mean combine, the sealed score is the mean of
the per-pair composites.

- **Pairs.** A ranked run measures 3 pairs (`official_pairs` = 3) on one box in one job.
  Each pair is one serial-control leg on the organizer-staged reference tree,
  then one candidate leg at your declared depth.
- **Workload.** Every leg is single-stream: a 512-token prompt (one unmeasured
  warm-up prefill, then five timed prefills of the same prompt, scored by their
  median) and a 128-step decode. The prompts are the fixture's `live_goldens`:
  three hidden coding prompts. Each one is a user message in the model's own
  chat template, with thinking off.
- **Every job measures every live golden.** Pair k measures prompt
  (k - 1) mod 3. Both legs of a pair measure the same prompt, so the ratio
  stays comparable.
- **Decode.** Decode time per token is the decode window: the time of
  `free_decode_run(N)` divided by N. The seed prefill is not part of it.
  Prefill is its own phase.
- **Mean combine.** The fixture sets `official_pair_combine: "mean"`. Each pair
  gets its own prefill and decode speedups against that pair's control leg and
  its own composite. benchd walks the pairs in measurement order and refuses
  on the first pair that fails the timed gate for that pair alone. After every
  pair passes, the sealed score is the arithmetic mean of the three pair
  composites; sealed speedups and per-token times are the arithmetic means of
  the per-pair figures (not the ratio of mean times, not the composite of mean
  speedups).
- **Bounds.** Both speedup floors are 0.95, enforced per pair in that gate.
  `iterate --mode official` does not apply a decode speedup ceiling.
- **Timed window.** Ranked official iterate replays timed divergences through the
  reference; the fixture pins `timed_token_tolerance_per_thousand` **40** and
  `timed_token_near_tie_relative_gap` **0.01** (see
  `docs/engineering/replay-policy-measurements.md`). Local modes still require
  exact timed-window match.

## Ranking

- **Each ranked job measures every live golden** (one pair per prompt).
- **The live goldens are replaced every 3 to 4 days** by prompts no submission
  has seen.
- **Submissions are unlimited.**
- **The final ranking is a re-measurement.** At the close, each team's best
  submission is taken: the mean contest score counts every run that scored or
  failed because of the submission (a team failure counts as 0.0; a post-setup
  reference-leg or band failure counts as 0.0; other infrastructure runs are
  excluded). The top N teams (default 10) are selected. Each selected
  submission is re-measured K times (default 5), one run per slot, across
  different boxes on a held-out pool that was never staged during the contest.
  Each slot is its first run that ends with a score or a team failure;
  infrastructure failures are re-run. In the final re-score only, one
  re-dispatch is allowed after the first candidate boot failure per slot. A run
  that fails because of the submission,
  or a reference-leg or band failure that persists after re-runs, counts as 0.0
  for that slot. The final ranking is the mean of the K slot values (the score,
  or 0.0 for a team failure or penalized post-setup infrastructure slot).

[The contract, section 5.7](docs/participant-contract.md#57-pool-refresh-and-final-ranking)
states the same rules.

## Local runs are directional

A local run uses your machine, not a ranked Spark. Treat a local score as a
smoke signal, not as a prediction. The ranked box run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, engine configuration, depths, scoring semantics | `fixtures/bonsai2_9b_cuda_v1_track.json` |
| Why each rule says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
CI (`tools/check-doc-consistency.py`) fails when this document drifts from
`benchmark.json` or the fixture.
