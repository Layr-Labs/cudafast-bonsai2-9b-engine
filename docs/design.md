# Bonsai 2 9B CUDA track — design and shared contract

Status: living document for organizers and implementers. Engine measurements:
[engineering/engine-probe.md](engineering/engine-probe.md). Port inventory:
[engineering/nemotron-harness-inventory.md](engineering/nemotron-harness-inventory.md).

## 1. What we build

The Nemotron CUDA track trust model (scoring, sandbox, rotation, sealed score)
with the engine swapped:

| | Nemotron track | This track |
|---|---|---|
| Engine | vLLM (Python), `vllm/` | PrismML llama.cpp fork, `llama.cpp/` |
| Resident | `harness/vllm_resident/` in pinned image | `harness/llama_resident/` (C++, libllama), same container hardening |
| Weights | safetensors tree | one GGUF file, sha256 + bytes |
| Speculation | vLLM MTP | llama.cpp `draft-mtp` (in-file MTP layer) |
| Harness hash | harness + vllm | harness + `llama.cpp` (benchd channel) |

Target: `prism-ml/Ternary-Bonsai-2-9B-gguf`
(`Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`), Bonsai 2 9B. The trunk is PQ2_0
with prism.hadamard v1 transforms; the MTP layer is Q8_0 and uses the shared
`output.weight` head. Code must not hard-code layer counts (read shapes from
the GGUF).

## 2. Names (use exactly)

| Thing | Value |
|---|---|
| Track id / 4th runner label | `bonsai2-9b-cuda-v1` |
| `benchmark.json` name | `cudafast-bonsai2-9b` |
| Fixture | `fixtures/bonsai2_9b_cuda_v1_track.json` |
| Measure script | `tools/bonsai2-9b-measure-and-score.sh` |
| Editable paths | `llama.cpp`, `harness`, `mtp-head.manifest.json` |
| Staging root | `/srv/bonsai9b/` |
| Golden dir env | `MLXFAST_BONSAI9B_GOLDEN_DIR` |
| Weights env | `MLXFAST_TARGET_GGUF` |
| Sandbox user / group / home | `bonsaisub` / `bonsaishare` / `/var/lib/bonsaisub` |
| GPU lock | `/tmp/mtplx-gpu-exclusive.lock` |
| Resident backend string | `llamacpp-resident` |
| Ranked runner box | Spark box 2 (runner label `bonsai2-9b-cuda-v1`; box setup in `Layr-Labs/cuda-setup-scripts`) |

Do not modify `/srv/nemotron35/`, `nemosub`, or Nemotron systemd units on shared
hosts.

## 3. Engine pin and build

- Fork: PrismML `llama.cpp` (pin in `llama.cpp/PRISM_PIN.md`).
- Image: `bonsai9b-engine:v1` from pinned `nvidia/cuda` devel digest
  (`tools/engine/Dockerfile`). Image id recorded per box in
  `/srv/bonsai9b/engine-image.id`. On a participant machine
  `tools/engine/build.sh` builds the image and writes `.build/engine-image.id`.
- Build: `tools/engine/build.sh` inside the image as `bonsaisub`, CMake/Ninja
  `llama-resident` linked to vendored `llama.cpp` with `GGML_CUDA=ON`,
  `CMAKE_CUDA_ARCHITECTURES=121`.
- Reference tree: organizer baseline at `/srv/bonsai9b/baseline/`; goldens
  recorded on two boxes must be byte-identical.

## 4. Resident (`harness/llama_resident/`)

Wire: [resident.md](resident.md) (NDJSON over AF_UNIX). Backend string
`llamacpp-resident`. Speculative loop aligned with llama-server /
speculative-simple ([engineering/resident-vs-server.md](engineering/resident-vs-server.md)).

Tests: unit tests with `FakeEngine`; on-box
`harness/llama_resident/test/test_adapter_e2e.sh` at depths 0–3;
`oracle_serial_check.py` vs llama-server for serial greedy decode.

## 5. Scoring and correctness

Matches fixture: **3** official pairs, mean combine,
`composite = prefill_gain^0.25 × decode_gain^0.75`, floors 0.95, 512-token
seed (1 warm-up + 5 timed prefills, median), 128 decode steps, three live
goldens. Timed window: ranked replay policy (`timed_token_tolerance_per_thousand`
40, `timed_token_near_tie_relative_gap` 0.01); see
[engineering/replay-policy-measurements.md](engineering/replay-policy-measurements.md).

## 6. Organizer milestones

1. Resident + engine build + `serve-up.sh` path (done in tree).
2. Fixture pins, lint, workflow, doc consistency (in progress).
3. benchd from the public `Layr-Labs/benchd` `main` channel, with `llama.cpp` in the
   harness hash (bench pull request 400; pin in `tools/fetch-benchd.sh`).
4. Staging on contest Sparks, record goldens, calibrate, enable official scoring.
5. Swap 9B pins when weights ship (PrismML).
