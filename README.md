# Bonsai 2 9B CUDA on DGX Spark: inference speed challenge

A ranked engineering challenge: make the pinned Bonsai target prefill and decode
faster on a DGX Spark, single stream, without changing what the model computes
beyond the ranked timed-window rule: at most 40 mismatches per thousand tokens, each the reference's second choice within a relative gap of 0.01.

**[Overview](OVERVIEW.md)** · **[Task and rules](TASK.md)** ·
**[Participant contract](docs/participant-contract.md)** ·
**[Sandbox design](docs/sandbox-design.md)** ·
**[Pool refresh and final ranking](docs/pool-refresh-and-final-rescore.md)**

## At a glance

| | |
|---|---|
| **Target** | `prism-ml/Ternary-Bonsai-2-9B-gguf` `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` (Bonsai 2 9B: PQ2_0 ternary trunk with prism.hadamard v1, one MTP layer in Q8_0), pinned by sha256 and bytes |
| **Hardware** | NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory) |
| **Engine** | PrismML `llama.cpp` vendored at `llama.cpp/`; teams edit `llama.cpp/`, `harness/` and the speculation arm and depth |
| **Drafters** | the in-file MTP layer (`draft-mtp`, depth 1 to 6) and Prism's DFlash2 block drafter `Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf` from the same revision (`draft-dflash`, depth 1 to 15), both pinned |
| **Score** | `prefill_gain^0.25 × decode_gain^0.75`, each gain measured against a trusted reference run on the same box, same prompt, same job |
| **Correctness** | within the token budget: **ranked** runs replay timed-window divergences (near-tie second-choice tokens within `timed_token_tolerance_per_thousand` pass); **local** modes require an exact match to the per-depth oracle with no replay |
| **Integrity** | sandboxed, network-less execution; hidden prompts rotated every 3–4 days; final ranking re-measured on a held-out pool |
| **Track** | `bonsai2-9b-cuda-v1` |

[OVERVIEW.md](OVERVIEW.md) explains each of these in one page: how scoring
works, what prevents cheating, and how the system was tested.

## What runs

```text
benchd (pinned prebuilt, trusted)
  │  spawns per phase, Engine Protocol v1 (NDJSON over stdio)
  ▼
.build/release/mlxfast-runtime-worker      ← harness/protocol-adapter (Rust, editable)
  │  NDJSON over a Unix socket
  ▼
llama-resident, one per leg                 ← harness/llama_resident (C++, editable)
  │  runs inside bonsai9b-engine:v1: --network none, tree and weights read-only
  ▼
llama.cpp + ggml (CUDA)                     ← vendored at llama.cpp/ (editable)
  ▼
the pinned weight file                      ← trusted, verified by bytes + sha256
```

- **The resident loads the model once per leg.** benchd spawns a fresh adapter
  for each phase (warmup, timed prefill, timed decode, correctness), and each
  adapter connects instead of loading.
- **benchd boots each leg's resident itself,** through that leg's own
  `tools/serve-up.sh --boot`, and stops it with `--stop`.
- **The adapter reports raw counters only.** benchd does all timing and
  scoring.

## Setup

The toolchain needed:

- cargo, with Rust 1.98.1 (`rustup toolchain install 1.98.1`)
- docker with the NVIDIA container runtime
- jq, python3, curl, sha256sum

A DGX Spark has all of them after the standard DGX OS setup plus rustup. The
host needs no nvcc. The engine image carries the CUDA build toolchain: CUDA
13.0 (nvcc 13.0.88), gcc 13.3, cmake 3.28.3 and ninja 1.11.1.
[The ranked box toolchain](#the-ranked-box-toolchain) gives the exact versions.

```bash
./tools/fetch-benchd.sh
```

This command resolves the pinned benchd from its release channel and verifies
the bytes against the channel manifest. The channel is the public bench
repository `Layr-Labs/benchd`. The fetch needs no token.

```bash
./setup.sh
```

This command runs `tools/engine/build.sh` and verifies the weight file. The
engine build:

1. builds the adapter and stages it at `.build/release/mlxfast-runtime-worker`;
2. builds the engine image from `tools/engine/Dockerfile` (base digest pinned
   in the fixture) when no image Id is staged, and writes its Id to
   `.build/engine-image.id`. A ranked box uses the organizer-staged image
   `bonsai9b-engine:v1` and never builds it;
3. builds `llama-resident` with CMake/Ninja inside that image;
4. builds `harness/kernels/` in the image when `harness/kernels/build.sh`
   exists.

On a ranked box the adapter build (and any `build.rs`) and the kernels/resident
build run as an unprivileged sandbox user with no network. Cargo dependencies
must come from crates.io and are vendored before an offline build.
[The contract, section 3.6](docs/participant-contract.md#36-where-your-code-runs)
lists the rules.

The weight file is pinned by absolute path on a box (`MLXFAST_TARGET_GGUF`).
On a ranked box it is pre-staged under `/srv/bonsai9b/weights/` and
`setup.sh` only verifies it. With `MLXFAST_TARGET_GGUF` unset, `setup.sh`
downloads the file from the pinned Hugging Face revision into
`reference_weights/`. While the repository is private, give a Hugging Face
token that can read it in `HF_TOKEN`. When `HF_TOKEN` is unset, `setup.sh`
reads the token from the Hugging Face token file of the user
(`${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}`, the file that
`huggingface-cli login` writes). `setup.sh` sends the token only to
huggingface.co and does not log or write it. A verified file needs no token.

```bash
./tools/local-baseline.sh
```

This command boots a resident for your declared depth. It then runs benchd's
local-iterate leg over the public golden of that depth, which the organizer
recorded on a Spark: `correctness_prompts/public-longcopy-gate-english-512.golden.json`
for serial, `correctness_prompts/public-longcopy-gate-english-512.mtp<N>.golden.json`
for MTP depth N and `...dflash<N>.golden.json` for DFlash depth N. A depth-N run can take another token path than the serial run,
so each depth has its own golden. A local run seals `score: null`: only the
paired ranked path produces a score.

```bash
yukon setup
```

This command runs `setupCommand`: `./tools/fetch-benchd.sh`, then
`./setup.sh`.

```bash
yukon run
```

This command runs `benchmarkCommand`. Off a ranked box it runs
`./tools/local-baseline.sh` and seals `score.json` with `"score": null`. Only
the ranked runner gives a score.

## Iterating without a GPU

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
```

These commands test the adapter against its mock engine. Resident unit tests
and `test_adapter_e2e.sh` (depths 0–3 on a box) live under
`harness/llama_resident/test/`.

## Repository structure

| Path | Trust | What it is |
|---|---|---|
| `llama.cpp/` | editable | PrismML fork (pin in `llama.cpp/PRISM_PIN.md`) |
| `harness/protocol-adapter/` | editable | Engine Protocol v1 adapter (Rust) |
| `harness/llama_resident/` | editable | C++ resident: socket wire + llama.cpp linkage |
| `harness/kernels/` | editable, optional | custom CUDA/C++ built by `harness/kernels/build.sh` |
| `mtp-head.manifest.json` | editable | the speculative-decoding declaration |
| `fixtures/bonsai2_9b_cuda_v1_track.json` | trusted | the track contract: pins, engine configuration, depths, scoring semantics |
| `benchmark.json` | trusted | editable paths, commands, byte budget, scoring constants |
| `setup.sh`, `benchmark.sh`, `tools/` | trusted | build, serve, preflight, measure, calibrate, stage, organizer tools |
| `correctness_prompts/` | trusted | the public prompt text and the public golden |
| `.github/` | trusted | CI (`ci.yml`) and the ranked pipeline (`benchmark.yml`) |
| `docs/` | — | contract, resident wire, sandbox, submission rules, pool refresh, engineering notes |

## The pinned artifacts

| Artifact | Pin |
|---|---|
| Weights | `prism-ml/Ternary-Bonsai-2-9B-gguf` file pinned in the fixture (sha256 + bytes) |
| Engine source | PrismML `llama.cpp` @ commit in `llama.cpp/PRISM_PIN.md` |
| Engine image | `bonsai9b-engine:v1` built from pinned `nvidia/cuda` devel digest (see `tools/engine/Dockerfile`) |
| benchd | the channel manifest `tools/fetch-benchd.sh` resolves |
| Goldens | `timed_prompt_pool`, `hidden_correctness_golden` and `speculative_oracles` in the fixture (staged on each box; see [OVERVIEW.md](OVERVIEW.md)) |

## The ranked run

`.github/workflows/benchmark.yml` is dispatched once per submission. It runs
two jobs:

1. **A `surface-check` on a Spark runner** (no GPU, no credential). It runs
   the manifest lint and the static review of the editable surface.
2. **A `ranked` job on a self-hosted Spark** labelled
   `[self-hosted, Linux, ARM64, bonsai2-9b-cuda-v1]`. It runs preflight,
   setup, measurement under the GPU lock, and a commit-reveal score block on
   the job log ([submission restriction
   spec](docs/submission-restriction-spec.md) §10). The fixture declares
   `live_goldens`, so nothing is drawn: every job measures every live golden,
   one pair per prompt, and `score.json` identifies a golden only by its
   sha256.

Participant code runs as `bonsaisub` with no network, reading the checkout
read-only. The adapter's stderr is not returned, and nothing persists between
jobs or legs.

### The ranked box toolchain

| | |
|---|---|
| **Hardware** | NVIDIA DGX Spark (GB10, aarch64) |
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.17.0-1026-nvidia |
| **Driver** | NVIDIA 580.159.03 (the fixture's `driver_min`), CUDA driver API 13.0; Docker 29.2.1, NVIDIA Container Toolkit 1.19.1 |
| **Engine image base** | `nvidia/cuda@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92` (Ubuntu 24.04.3 LTS) |
| **CUDA toolkit in the image** | CUDA 13.0: nvcc V13.0.88, `cuda-nvcc-13-0` and `cuda-cudart-dev-13-0` 13.0.88-1 |
| **Compilers in the image** | gcc/g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04), cmake 3.28.3, ninja 1.11.1; no Rust, Python or PyTorch |
| **CUDA arch** | `sm_121` (the fixture's `serve_configuration.toolchain.cuda_arch`); `tools/engine/build.sh` passes `-DCMAKE_CUDA_ARCHITECTURES=121` |
| **Rust on the host** | 1.98.1 (`1.98.1-aarch64-unknown-linux-gnu`), offline `cargo build --release --locked` on vendored crates.io dependencies |

The host has no nvcc. Every CUDA compile runs inside `bonsai9b-engine:v1`.
That covers `llama-resident` and, when present, `harness/kernels/`. The
adapter (`harness/protocol-adapter`) builds on the host.

The hosted CI does not compile CUDA. Its `llama-resident-cmake` job builds
`llama_resident` for the CPU only and runs ctest on `ubuntu-24.04-arm`. Its
adapter job runs cargo test, clippy and fmt on `ubuntu-latest`. A green hosted
CI does not prove that your CUDA code compiles for `sm_121` with nvcc 13.0.

The ranked job builds your candidate in its step "Setup (engine + pinned
checkpoint)". A compile failure there fails the submission before any scoring.

Build-check inside the same image before you submit:

1. Build the image with `tools/engine/build-image.sh`. It uses the pinned base
   digest.
2. Run `./setup.sh`. It runs `tools/engine/build.sh`, which compiles
   `llama-resident` and the kernels inside the image with the same nvcc, gcc,
   cmake and ninja as the ranked box.
3. Build the adapter with Rust 1.98.1.

## Organizer runbook

1. **Stage each Spark** — box runner, `setup-sandbox-user.sh`, pull/build
   `bonsai9b-engine:v1`, record image id, stage weights and goldens under
   `/srv/bonsai9b/`.
2. **Calibrate** every box; arm scoring when the null control lands near 1.0.
3. **Run the contest** with `tools/router/` for intake and the final re-score.

[docs/engineering/](docs/engineering/README.md) records GB10 probes and fleet
posture; [docs/bonsai9b-port-notes.md](docs/bonsai9b-port-notes.md) keeps the
Nemotron port log for reference.

## Documentation

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, engine configuration, scoring semantics | `fixtures/bonsai2_9b_cuda_v1_track.json` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |
| What CI covers | `docs/ci-coverage.md` |
| How participant code is sandboxed on a box | `docs/sandbox-design.md` |
| What a submission may contain | `docs/submission-restriction-spec.md` |
| Pool refresh and final re-score | `docs/pool-refresh-and-final-rescore.md` |
| Engineering measurements | `docs/engineering/README.md` |
| Nemotron port log (reference) | `docs/bonsai9b-port-notes.md` |
| What a measured run executes | benchd via `tools/fetch-benchd.sh` |
