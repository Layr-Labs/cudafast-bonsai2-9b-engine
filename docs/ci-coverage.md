# What CI covers

`.github/workflows/ci.yml` runs on manual dispatch only (`workflow_dispatch`):
its jobs need GitHub-hosted runners, so run the same checks locally before a
push (`AGENTS.md`, "Checks before you submit"). It holds no credential.

## `lint` (x86 ubuntu)

| Step | What it proves |
|---|---|
| egress scan | no workflow references a secret, inherits one, asks for write permission, or uploads anything but the ranked `score.json` (the ranked workflow uploads that one file, after sandbox cleanup) |
| actionlint | the workflows parse and type-check (pinned binary, sha256-verified) |
| shellcheck | every tracked `*.sh` outside `llama.cpp/` passes at warning severity |
| manifest lint | `benchmark.json` against the fixture: schema, paths, commands, scoring constants, byte-budget fallbacks, per-leg serve verbs |
| benchd platform / recorder | `tools/fetch-benchd.sh` refuses a wrong platform or an unverified second binary |
| adapter | `cargo fmt`, `clippy -D warnings` and `cargo test` (mock and scripted sessions) |
| docs agree with the manifest | `tools/check-doc-consistency.py` |
| participant engine edit | the editable list, benchd's harness-hash roster (`llama.cpp` + `harness`), and staging rules |
| serve-up boot/stop, memory plan | the real `serve-up.sh` and the real resident (fake backend) |
| facade, local baseline, setup onboarding | the entry points, through a real resident boot |
| yukon run | `benchmarkCommand` reaches the ranked script on a ranked box and `tools/local-baseline.sh` off one |
| spec declaration | the declaration envelope |
| ranked preflight | every environment refusal, and the section 8 paired-baseline refusals on a scaffolded box |
| calibrator, stager, golden fetcher, box runner | their mechanics, with stub collaborators |

## `resident` (x86 ubuntu)

The C++ resident unit tests (`cargo`/CMake in CI where enabled) and on-box
`test_adapter_e2e.sh` (real `cuda-engine` + resident at depths 0–3).

## `engine-build` (arm64 ubuntu, no GPU)

Verifies the pinned `bonsai9b-engine:v1` image, the CMake build of
`llama-resident`, and static review rules on editable CMake.

## What CI cannot cover

Anything that needs a GPU:

- the engine's numerics;
- speed;
- the goldens;
- calibration;
- the ranked run.

On-box verification and measurements:
[docs/engineering/README.md](engineering/README.md) and
`docs/bonsai9b-port-notes.md` (Nemotron reference log).
`tools/three-flows-dry-run.sh` runs the real benchd binaries over the fake
resident. It is not in CI, because it downloads benchd.
