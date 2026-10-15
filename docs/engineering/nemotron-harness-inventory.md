# Nemotron harness inventory (llama.cpp / Bonsai 2 9B port)

Source of truth (read-only): the Nemotron 3.5 Lightning CUDA reference engine
repository (vLLM track). Scope: every `git ls-files` entry **outside** `vllm/`
(the vendored vLLM tree is **DROP** here and **REPLACE** with vendored
`llama.cpp/` in the Bonsai repo).

Legend — **class**: `KEEP` copy as-is; `RENAME` strings/paths only; `ADAPT` logic tied to Nemotron/vLLM/docker/safetensors; `REPLACE` rewrite for llama.cpp; `DROP` not needed.

---

## Repository root

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `AGENTS.md` | 110 | ADAPT | Participant map: `vllm/` → `llama.cpp/`; model files (`nemotron_h.py`, MoE/Mamba); resident path; determinism notes (§59 → Bonsai/Qwen3.5). |
| `benchmark.json` | 64 | ADAPT | `name`, `trackId`, `description`, `editablePaths` (`llama.cpp`, `harness`, spec manifest), `contractPath`, `scoring.kvBackend`, `leaderboard.namespace`, `staticReviewTrackId`, byte budgets; commands may rename `nemotron35-measure-and-score.sh`. |
| `benchmark.sh` | 48 | RENAME | Facade → new measure script name if renamed. |
| `LICENSE` | 29 | KEEP | |
| `OVERVIEW.md` | 186 | ADAPT | Model, engine, track id, `/srv/nemotron35` staging paths, editable paths, NVFP4/MTP prose → Bonsai/GGUF/llama.cpp. |
| `README.md` | 236 | ADAPT | Setup (docker/vLLM overlay → native or CUDA devel image + llama build); editable table; image digest pins. |
| `TASK.md` | 178 | ADAPT | Target weights format (safetensors/NVFP4 → GGUF ternary); engine; KV backend; `python3 -m unittest … vllm_resident` → C++ resident tests. |
| `THIRD_PARTY_NOTICES.md` | 228 | ADAPT | vLLM notice → llama.cpp / PrismML; container pin section. |
| `setup.sh` | 249 | ADAPT | Steps 1–2: drop `.build/vllm-overlay` checks (125–126); snapshot = GGUF file(s) not `*.safetensors` tree (43–101, 185–230); `DEFAULT_SNAPSHOT_DIR` path/name (49). |
| `mtp-head.manifest.json` | 9 | ADAPT | Spec declaration for llama.cpp `--spec-type draft-mtp` (or serial-only until MTP proven); keep schema, change semantics/docs. |

## `correctness_prompts/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `correctness_prompts/public_longcopy_gate_english_4096.txt` | 190 | KEEP | Prompt text engine-agnostic. |
| `correctness_prompts/public-longcopy-gate-english-4096.golden.json` | 13467 | REPLACE | Re-record with Bonsai reference resident (`tools/record-goldens.sh`); model provenance fields in golden. |

## `docs/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `docs/ci-coverage.md` | 53 | ADAPT | Drop `stage-vllm-overlay` / `nemotron_h` import checks (36–37); add llama.cpp build smoke if any. |
| `docs/nemotron35-port-notes.md` | 635 | RENAME+ADAPT | Rename to bonsai port log; GB10 determinism section becomes Bonsai-specific; keep harness findings that transfer. |
| `docs/participant-contract.md` | 616 | ADAPT | §3 formats (NVFP4 → ternary GGUF); §4 MTP/vLLM → llama speculation; §5.4 replay (keep thresholds, cite fixture lines 40–41); `kvBackend`; staging env names. |
| `docs/pool-refresh-and-final-rescore.md` | 214 | RENAME | Track id, R2 prefix, golden paths; procedure KEEP. |
| `docs/resident.md` | 145 | ADAPT | `harness/vllm_resident/resident.py` → C++ resident; drop vLLM `LLMEngine` (37–38); keep wire table (49–60); memory plan may use GGUF size not safetensors sum (111–118). |
| `docs/sandbox-design.md` | 279 | RENAME | User names if parallel fleet (`bonsaisub` etc.); mechanics KEEP. |
| `docs/submission-restriction-spec.md` | 427 | RENAME | Track/workflow names; intake overlay paths (`vllm` → `llama.cpp`). |

## `fixtures/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `fixtures/nemotron35_lightning_30b_a3b_track.json` | 884 | REPLACE | New track id; `target.files` GGUF pins; `golden_model_type`; `serve_configuration` (`engine`, container or native toolchain, `engine_env`, `engine_args`, depths); all R2 paths; calibration/oracle pins; `vocab_size`/architecture fields. |

## `reference_weights/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `reference_weights/.gitkeep` | 1 | KEEP | Placeholder for local GGUF download dir. |

## `.github/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `.gitattributes` | 2 | KEEP | |
| `.gitignore` | 85 | ADAPT | Ignore `.build/llama-*` instead of/in addition to vllm overlay paths if present. |
| `.github/actionlint.yaml` | 11 | KEEP | |
| `.github/scripts/hardened-git.sh` | 47 | KEEP | |
| `.github/scripts/submission-static-review-checks.sh` | 510 | ADAPT | Track id strings; editable path roots (`vllm` → `llama.cpp`); harness-hash roster in benchd (grep `vllm`). |
| `.github/workflows/benchmark.yml` | 748 | ADAPT | Workflow name; `runs-on` label (4th label = track id, 156–164, 204); reference workspace checks for `.build/vllm-overlay` (341–346) → llama build artifacts; comments vLLM (256–260); `ENGINE_BUILD_OFFLINE` / docker pull assumptions. |
| `.github/workflows/ci.yml` | 280 | ADAPT | Engine overlay / nemotron model import steps if any; track fixture filename. |

## `harness/` (no `vllm/`)

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `harness/README.md` | 26 | ADAPT | Resident implementation language/path. |
| `harness/protocol-adapter/Cargo.toml` | 21 | KEEP | |
| `harness/protocol-adapter/Cargo.lock` | 107 | KEEP | |
| `harness/protocol-adapter/reference/engine-protocol-v1.schema.json` | 154 | KEEP | Wire to benchd unchanged unless protocol version bump. |
| `harness/protocol-adapter/reference/PROVENANCE.md` | 41 | KEEP | |
| `harness/protocol-adapter/src/lib.rs` | 66 | KEEP | |
| `harness/protocol-adapter/src/protocol.rs` | 392 | KEEP | `TOP_LOGITS_K=8` (49–50) is what benchd expects. |
| `harness/protocol-adapter/src/engine.rs` | 222 | KEEP | Documents `top_logits` on correctness steps (20–25). |
| `harness/protocol-adapter/src/session_backend.rs` | 488 | KEEP | `step()` = `eval` + `top_logits(8)` (210–217); spec path uses resident verbs. |
| `harness/protocol-adapter/src/adapter.rs` | 713 | KEEP | Correctness responses attach `top_logits` (572–582); spec counters forwarded (465–516). |
| `harness/protocol-adapter/src/resident.rs` | 487 | ADAPT | `RESIDENT_BACKEND` string `"vllm-resident"` (68, 114–117) → e.g. `llamacpp-resident`; module comments `resident.py` (3); wire unchanged. |
| `harness/protocol-adapter/src/mock.rs` | 358 | KEEP | |
| `harness/protocol-adapter/src/bin/cuda-engine.rs` | 87 | KEEP | |
| `harness/protocol-adapter/src/tests.rs` | 2526 | ADAPT | Tests referencing `vllm-resident` backend string if asserted. |
| `harness/vllm_resident/__init__.py` | 0 | DROP | Replaced by C++ resident. |
| `harness/vllm_resident/backends.py` | 304 | DROP | vLLM `LLMEngine` backend. |
| `harness/vllm_resident/resident.py` | 342 | REPLACE | Port wire + bookkeeping to C++ linked to libllama (same verbs as `docs/resident.md`). |
| `harness/vllm_resident/test_resident.py` | 179 | REPLACE | Python tests → C++ or integration tests against fake backend. |
| `harness/vllm_resident/test_adapter_e2e.py` | 112 | ADAPT | Path to resident binary; backend prefix (77). |

## `tools/` (selected — all tracked scripts)

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `tools/author-prompts.py` | 92 | KEEP | |
| `tools/baseline-calibration-v1-view.sh` | 41 | RENAME | Fixture path if renamed. |
| `tools/benchmark.sh` | 495 | RENAME+ADAPT | Track/fixture references. |
| `tools/box-runner/*` | (see below) | RENAME | Parallel fleet names (see §2). |
| `tools/calibrate-box.sh` | 373 | ADAPT | Uses reference tree `serve-up.sh` (29–34); env/weights paths; script name optional rename. |
| `tools/check-doc-consistency.py` | 132 | ADAPT | Fixture filename, TASK/OVERVIEW engine claims, `docs/nemotron35-port-notes.md` (105). |
| `tools/ci-workflow-egress-scan.sh` | 189 | KEEP | |
| `tools/determinism-probe.py` | 40 | ADAPT | Example paths (`~/nemotron35/repo/…` 21). |
| `tools/download-r2-object.sh` | 393 | RENAME | R2 key prefix = track id. |
| `tools/draw-live-golden.sh` | 59 | KEEP | |
| `tools/engine/build.sh` | 212 | REPLACE | Remove docker image pull + `stage-vllm-overlay.sh` (158–168); build llama.cpp + resident; optional kernels → native compile or minimal container; sandbox path KEEP (18–34). |
| `tools/fetch-benchd.sh` | 551 | KEEP | Channel/manifest may get new track id entry when forked. |
| `tools/fetch-goldens.sh` | 467 | RENAME | `CONTRACT` path (63), R2 prefix. |
| `tools/lint-benchmark-manifest.py` | 1198 | ADAPT | Track-specific command targets and contract path. |
| `tools/local-baseline.sh` | 62 | ADAPT | `serve-up` / weights env. |
| `tools/nemotron35-measure-and-score.sh` | 750 | RENAME+ADAPT | Golden dir env `MLXFAST_NEMOTRON35_GOLDEN_DIR` (377+); `--mtp-depth` (521–525); leg `serve-up` invocations (548+); baseline workspace paths. |
| `tools/pool-refresh.sh` | 369 | RENAME | Fixture/track strings. |
| `tools/ranked-box-preflight.sh` | 755 | ADAPT | **Critical:** vLLM vendor checks (490–505), `weight_owner`/`vllm-resident` (544–549), `.build/vllm-overlay` on reference ws (611–623); → llama.cpp vendor pin + resident binary + build artifacts. |
| `tools/record-goldens.sh` | 192 | ADAPT | `boot serial\|mtp` (129–161) unchanged interface; weights dir = GGUF; fixture filename (52). |
| `tools/run-engine-sandboxed.sh` | 96 | ADAPT | Staged binary paths. |
| `tools/sandbox-wipe.sh` | 69 | KEEP | |
| `tools/serve-up.sh` | 624 | REPLACE | Docker+Python resident (519–545): → launch C++ resident (host or CUDA devel container); memory plan `*.safetensors` (445–446) → GGUF size; `PYTHONPATH`/overlay (515) → `LLAMA_*` or binary `RPATH`; socket names `vllm-resident` (92–94); identity `engine:"vllm"` (586–587); probe still NDJSON `hello` (400–430). |
| `tools/spec-declaration.sh` | 174 | ADAPT | MTP manifest semantics for llama.cpp draft depth. |
| `tools/stage-baseline-workspace.sh` | 177 | ADAPT | Reference tree build expects overlay (grep vllm). |
| `tools/stage-cuda-engine.sh` | 151 | KEEP | Staging adapter binary name unchanged for benchd. |
| `tools/stage-vllm-overlay.sh` | 57 | DROP | Replaced by llama.cpp install/stage script. |
| `tools/stitch-prompts.py` | 198 | KEEP | |
| `tools/three-flows-dry-run.sh` | 148 | ADAPT | Resident backend env. |
| `tools/wait-for-memory.sh` | 90 | ADAPT | May read fixture `engine_env` for reservation fraction. |
| `tools/router/install-service.sh` | 114 | RENAME | systemd unit names. |
| `tools/router/loadgen.py` | 209 | RENAME | Track/workflow ids in defaults. |
| `tools/router/README.md` | 165 | RENAME | |
| `tools/router/router.py` | 980 | ADAPT | `editablePaths` overlay (`vllm` → `llama.cpp`); workflow dispatch. |
| `tools/router/systemd/nemotron35-router.service` | 15 | RENAME | Unit file name + Description. |
| `tools/router/systemd/nemotron35-watch.service` | 15 | RENAME | |
| `tools/router/test_router.py` | 719 | RENAME | Track strings in tests. |
| `tools/router/test_services.py` | 69 | RENAME | |
| `tools/router/watch.sh` | 36 | RENAME | |
| `tools/test-benchmark-facade-env.sh` | 256 | ADAPT | Env contract for measure script. |
| `tools/test-calibrate-box.sh` | 377 | ADAPT | Mock serve-up / baseline paths. |
| `tools/test-engine-build-sandbox.sh` | 307 | ADAPT | Stubs `stage-vllm-overlay.sh` (55); vLLM image ref (130–132). |
| `tools/test-fetch-benchd-platform.sh` | 212 | KEEP | |
| `tools/test-fetch-benchd-recorder.sh` | 312 | KEEP | |
| `tools/test-fetch-goldens.sh` | 453 | RENAME | Fixture path. |
| `tools/test-live-golden-rotation.sh` | 404 | ADAPT | `MLXFAST_NEMOTRON35_GOLDEN_DIR`, mtp oracle paths. |
| `tools/test-local-baseline.sh` | 123 | ADAPT | serve-up / public golden. |
| `tools/test-participant-engine-edit.sh` | 210 | ADAPT | vLLM overlay reachability → llama.cpp edit path. |
| `tools/test-pool-refresh.sh` | 209 | RENAME | Fixture filename. |
| `tools/test-ranked-box-preflight-env.sh` | 555 | ADAPT | Full preflight scaffold incl. vLLM overlay (258–259, 388–389). |
| `tools/test-run-engine-sandboxed.sh` | 284 | KEEP | |
| `tools/test-sandbox-wipe.sh` | 156 | KEEP | |
| `tools/test-serve-up-boot-stop.sh` | 549 | ADAPT | Fake resident; log glob `vllm-resident.*.log` (133). |
| `tools/test-serve-up-plan-memory.sh` | 137 | ADAPT | safetensors sizing → GGUF. |
| `tools/test-setup-onboarding.sh` | 167 | ADAPT | setup.sh / overlay expectations. |
| `tools/test-spec-declaration.sh` | 130 | KEEP | |
| `tools/test-stage-baseline-workspace.sh` | 118 | ADAPT | Reference `.build/vllm-overlay`. |
| `tools/test_stitch_prompts.py` | 113 | KEEP | |
| `tools/test-wait-for-memory.sh` | 123 | ADAPT | Fixture memory fraction. |
| `tools/testlib/synthetic-window.sh` | 28 | KEEP | |

### `tools/box-runner/`

| path | lines | class | what must change |
|------|------:|-------|------------------|
| `tools/box-runner/README.md` | 272 | RENAME | Track label, unit names, `/srv/…` paths. |
| `tools/box-runner/box-runner.env.example` | 71 | RENAME | `MLXFAST_*` golden dir, snapshot path under `/srv/nemotron35`. |
| `tools/box-runner/gh-app-mint.sh` | 228 | KEEP | |
| `tools/box-runner/install.sh` | 212 | RENAME | Service template name. |
| `tools/box-runner/lib/pidtree.sh` | 179 | KEEP | |
| `tools/box-runner/nemotron35-box-runner.service.template` | 37 | RENAME | Unit name, `Requires=nemotron35-sandbox-egress.service` (16–17). |
| `tools/box-runner/setup-sandbox-user.sh` | 571 | RENAME | `nemosub`, `/var/lib/nemotron35/sandbox`, `/etc/nemotron35-sandbox/*`, systemd units (63–64), `DEFAULT_SERVE_LOG` (72). |
| `tools/box-runner/supervisor.sh` | 378 | RENAME | `RUNNER_LABELS` track id. |
| `tools/box-runner/test/*` | ~900 | RENAME | Track strings. |

---

## 1. Where the vLLM resident behaviour is assumed (file:line)

Wire contract: `docs/resident.md` (49–60); implementation reference `harness/vllm_resident/resident.py` (must become C++ with identical semantics).

| Area | Location | Assumption |
|------|----------|------------|
| **Boot / teardown** | `tools/serve-up.sh` | `--boot --spec serial\|mtp --draft-len N` (167–178); writes socket to `--socket-out` (603–616); `--stop --socket` (180–200, 488–491); health via NDJSON `hello` (400–430, 556–576); depth/ctx must match flags (571–576). |
| | `tools/serve-up.sh` | Docker launches `python3 …/harness/vllm_resident/resident.py` (519–542); env `RESIDENT_*`, `PYTHONPATH=${OVERLAY}` (504–516). |
| | `tools/nemotron35-measure-and-score.sh` | benchd boots per-leg `serve-up` from each workspace (548–558, 551 area); refuses preset `DS4_RESIDENT_SOCKET` / `BENCH_WORKER_RESIDENT_SOCKET` (369 area). |
| | `tools/calibrate-box.sh` | benchd calls reference tree `serve-up.sh --boot --spec serial --draft-len 0` each pass (29–34). |
| | `tools/record-goldens.sh` | `boot serial 0` / `boot mtp d` (129–161); `BENCH_WORKER_RESIDENT_SOCKET` for recorder (134–138). |
| **Rust client** | `harness/protocol-adapter/src/resident.rs` | Connect + verbs: `hello`, `invalidate`, `sync`, `eval`, `argmax`, `top_logits`, `eval_speculative`, `spec_run`, `spec_counters`, `bye` (338+); `RESIDENT_BACKEND` sealed by benchd (68, 96–117). |
| | `harness/protocol-adapter/src/session_backend.rs` | Timed/correctness stepping calls `eval` + `top_logits(8)` (210–217); free decode uses `spec_run` when resident offers it. |
| **Preflight** | `tools/ranked-box-preflight.sh` | Requires `harness/vllm_resident/resident.py` (548–549); `serve_configuration.weight_owner == vllm-resident` (544–545); reference ws must have `.build/vllm-overlay/vllm/__init__.py` (611–623). |
| **Sandbox** | `.github/workflows/benchmark.yml` | Candidate leg: `SERVE_UP_CONTAINER_USER=nemosub`, resident in docker (see ranked job env steps ~400+); `tools/run-engine-sandboxed.sh` wraps adapter only. |
| **Tests** | `tools/test-serve-up-boot-stop.sh` | Host launcher + fake backend; expects `vllm-resident.*.log` (133); mtp depth in identity (170–181). |
| | `harness/vllm_resident/test_adapter_e2e.py` | `hello["backend"].startswith("vllm-resident load_epoch=")` (77). |
| **Goldens** | `tools/record-goldens.sh` | Same adapter→resident path as ranked leg (7–18, 133–138). |
| **Memory gating** | `tools/serve-up.sh` | Requires Nemotron snapshot files `config.json`, `model.safetensors.index.json`, `hf_quant_config.json` (295–297); `plan_memory`: sum `*.safetensors` (445–446); `RESIDENT_GPU_MEMORY_UTILIZATION` from fixture `engine_env` (452–457). |

---

## 2. On-box paths, users, units, firewall (parallel Bonsai fleet must not collide)

| Kind | Nemotron name / path | Where defined |
|------|----------------------|---------------|
| **Staging root** | `/srv/nemotron35/` (goldens, calibration, benchd, snapshot) | `OVERVIEW.md:155`; `tools/box-runner/box-runner.env.example`; fleet ops |
| **Goldens dir** | `MLXFAST_NEMOTRON35_GOLDEN_DIR` → typically `/srv/nemotron35/goldens` | `tools/nemotron35-measure-and-score.sh:377–386`; `benchmark.yml:236–240` |
| **Weights snapshot** | `MLXFAST_TARGET_SNAPSHOT_DIR` (safetensors tree on box) | `benchmark.yml:234`; `setup.sh:15–16` |
| **Reference tree** | `MLXFAST_BASELINE_WORKSPACE` | `benchmark.yml:241–242`; `ranked-box-preflight.sh:26–28` |
| **Calibration file** | `MLXFAST_BASELINE_CALIBRATION` (per-box JSON) | `benchmark.yml:244–245`; `calibrate-box.sh:54–56` |
| **benchd** | `BENCHD_BIN_DIR` (e.g. `/srv/nemotron35/benchd-bin`) | `benchmark.yml:235`; `record-goldens.sh:71–74` |
| **Sandbox user** | `nemosub` (uid varies per box) | `setup-sandbox-user.sh:50–53`; `benchmark.yml:379` |
| **Share group** | `nemoshare` | `setup-sandbox-user.sh:52` |
| **Sandbox home** | `/var/lib/nemosub` | `setup-sandbox-user.sh:53` |
| **Sandbox job root** | `/var/lib/nemotron35/sandbox` + `$SANDBOX_JOB` | `setup-sandbox-user.sh:54`; `serve-up.sh:76` |
| **Serve log dir** | `/var/lib/nemotron35/serve-log` (default) | `setup-sandbox-user.sh:72`; `SERVE_UP_LOG_DIR` in `serve-up.sh:58–59` |
| **Sudoers** | `/etc/sudoers.d/nemotron35-sandbox` | `setup-sandbox-user.sh:55–56` |
| **Wipe list** | `/etc/nemotron35-sandbox/wipe-dirs` | `setup-sandbox-user.sh:66–67` |
| **systemd (sandbox)** | `nemotron35-sandbox-egress.service`, `nemotron35-sandbox-acl.service` | `setup-sandbox-user.sh:63–64`; `nemotron35-box-runner.service.template:16–17` |
| **systemd (runner)** | `nemotron35-box-runner.service` | `nemotron35-box-runner.service.template:1–37` |
| **systemd (router)** | `nemotron35-router.service`, `nemotron35-watch.service` | `tools/router/systemd/` |
| **GPU lock** | `/tmp/mtplx-gpu-exclusive.lock` | `benchmark.yml:67–70`; `serve-up.sh:46–48` |
| **Runner labels** | `self-hosted`, `Linux`, `ARM64`, `nemotron3.5-lightning-30b-a3b-cuda-v1` | `benchmark.yml:156–164, 204` |
| **Container name** | `serve-up-<pid>-<time>` (docker rm on stop) | `tools/serve-up.sh:250–254, 533` |
| **Socket paths** | Unsandboxed: `vllm-resident.<tag>.sock`; sandboxed: `<run-tag>/vllm-resident.sock` under `$SANDBOX_ROOT/$SANDBOX_JOB/sock/` | `serve-up.sh:92–94, 76–77` |
| **iptables** | `OUTPUT -m owner --uid-owner nemosub -j REJECT` (v4+v6) | `setup-sandbox-user.sh:59–61`; `nemotron35-box-runner.service.template:21–22` |
| **Rust toolchain** | `/opt/rust/{cargo,rustup}` | `setup-sandbox-user.sh:14–16, 57` |

**Bonsai parallel setup:** use a distinct prefix everywhere (e.g. `/srv/bonsai9b/`, `bonsaisub`, `bonsai9b-box-runner`, track label `bonsai2-9b-cuda-v1`, `MLXFAST_BONSAI9B_GOLDEN_DIR`) so Nemotron staging on the same Sparks stays intact.

---

## 3. Docker image use today vs native llama.cpp build

**Today (vLLM track)**

1. **Pull by digest** — `tools/engine/build.sh:46–48, 158–165` reads `serve_configuration.container_image` + `container_digest` from fixture; ranked box sets `ENGINE_BUILD_OFFLINE=1` (workflow) so image must be pre-pulled.
2. **Overlay** — `tools/stage-vllm-overlay.sh` copies container’s installed `vllm` then overlays git `vllm/vllm/`; referenced in `setup.sh:125–126`, preflight `611–623`.
3. **Resident container** — `serve-up.sh:519–542`: `docker run --gpus all --network none --read-only`, bind-mount checkout + weights + socket dir, `entrypoint python3` → `harness/vllm_resident/resident.py`, writable `/tmp` tmpfs and per-leg cache under socket leg dir for sandbox (`528–531`).
4. **Optional kernels** — `engine/build.sh:174–197`: `docker run … bash ./build.sh` in `harness/kernels` with `PYTHONPATH=overlay`.

**Native llama.cpp direction (Bonsai)**

| Need | vLLM docker | Native replacement |
|------|-----------|-------------------|
| Compiled engine | Prebuilt vLLM CUDA wheels in image | Build PrismML `llama.cpp` fork on box or in **pinned CUDA devel image**; stage `llama-cli` / custom `llama-resident` binary to `.build/release/` or dedicated path benchd invokes |
| Resident process | Python in container | **C++ resident** binary linked to `libllama`, same NDJSON socket (replace `resident.py`) |
| Participant edit surface | `vllm/vllm` overlay | Vendored `llama.cpp/` source; sandbox build as `bonsaisub`: read-only checkout, write only `$SANDBOX_ROOT/$SANDBOX_JOB/{target,vendor,kernels-out}` per `engine/build.sh:58–64, 136–156` |
| Weights | Dir of `.safetensors` | Single or multi-file **GGUF**; `setup.sh` verify loop over fixture `.target.files` |
| Memory plan | `find … *.safetensors` | Sum GGUF bytes (or mmap size); tune `engine_env` fractions for unified GB10 memory |
| `SERVE_UP_LAUNCHER=host` | Used only in tests (`serve-up.sh:66–68`) | Likely **default for ranked** if resident is native binary; keep docker option for hermetic CUDA toolchain if needed |
| Kernels | Python/Triton in container | Optional `harness/kernels/build.sh` → compile `.so` with `cmake`+CUDA on host or one-shot devel container **without** replacing resident |

---

## 4. Token-tolerance gate, near-tie replay, and required logits

**Policy (benchd + fixture, not in this repo’s Rust/Python):**

- Per-stream **token-tolerance budget 10%** against serial tape — `docs/participant-contract.md:345–349`; fixture `token_tolerance_budget: 0.1` (`fixtures/nemotron35_lightning_30b_a3b_track.json:47`).
- **Timed window:** on divergence, reference engine **teacher-forced replay** of candidate stream — `docs/participant-contract.md:351–375`.
- Thresholds: `timed_replay_max_logit_gap: 2.0`, `timed_replay_off_argmax_per_thousand: 100` — fixture `:40–41`; explained contract `:364–369`.

**What the resident must supply (adapter → benchd):**

1. **`eval` {token}** — Teacher-forces one token; advances state; returns frontier argmax in reply (`docs/resident.md:54–55`; `resident.py:213–214`, `122–125`).
2. **`top_logits` {k}** — **Raw logits** at current frontier, **highest first** (`docs/resident.md:56`; `resident.py:162–166, 217–220`). Adapter requests **k=8** (`protocol.rs:49–50`; `session_backend.rs:210–217`).
3. **Correctness phases** — Adapter attaches `top_logits` array (8 entries: `token` + `logit` f64) on `correctness_begin` / `correctness_step` responses (`adapter.rs:572–582`; `engine.rs:20–25`).
4. **Speculative path** — `spec_run`, `eval_speculative`, `spec_counters` for timed decode; `verify_replay_disagreements` forwarded audit-only (`resident.md:91–98`; `adapter.rs:465–469`). vLLM resident reports 0 disagreements (`resident.md:91–92`).

**Logit semantics:** Fixture pins vLLM `logprobs_mode: raw_logits`, `max_logprobs: 8` (`serve_configuration.engine_args` in fixture). llama.cpp resident must expose **the same raw logit values** the reference uses for greedy decode and replay (BF16 tie behaviour documented contract `:355–358`). Greedy argmax from those logits must match `eval`/`sync` frontier tokens.

**Local vs ranked:** `docs/participant-contract.md:379–381` — local modes exact-match only; ranked replay needs reference resident boot (~4 min) when timed stream diverges.

---

## 5. GitHub workflow, runner, intake dependencies

| Component | Role | Key files / lines |
|-----------|------|-------------------|
| **Trigger** | `workflow_dispatch` only (Yukon/intake dispatches per submission) | `benchmark.yml:145–152` |
| **Job 1 `surface-check`** | Manifest lint + static review on box runner, no GPU | `benchmark.yml:162–194`; `tools/lint-benchmark-manifest.py`; `.github/scripts/submission-static-review-checks.sh` |
| **Job 2 `ranked`** | Full measure; needs `surface-check` | `benchmark.yml:198–749` |
| **Runner coupling** | Fourth label = track id; must match `tools/box-runner` `RUNNER_LABELS` | `benchmark.yml:56–65, 201–204` |
| **Runner env (no secrets)** | Snapshot, benchd, goldens, baseline ws, calibration | `benchmark.yml:230–307`; refuse if missing |
| **GPU lock** | `flock /tmp/mtplx-gpu-exclusive.lock` for timed step | `benchmark.yml:67–75` (see workflow run step) |
| **Sandbox** | `SANDBOX_USER=nemosub`, wipe, ACLs, build as sandbox user | `benchmark.yml:361–397+`; `tools/sandbox-wipe.sh`; `tools/engine/build.sh` |
| **Preflight** | Before setup/measure | `tools/ranked-box-preflight.sh` (entire file); invoked from measure script / workflow |
| **Setup** | `fetch-benchd.sh && setup.sh` per `benchmark.json:24–27` | Engine build + weight verify |
| **Measure** | `MLXFAST_SCORE_PATH=score.json ./tools/nemotron35-measure-and-score.sh` | `benchmark.json:34–37`; exec pinned `benchd iterate --mode official` (see measure script ~500+) |
| **Score output** | Sealed `score.json`; **no artifact upload** — intake parses **nonce-tagged log block** | `benchmark.yml:8–14`; `docs/submission-restriction-spec.md` §10 |
| **Intake service** | Overlay only `editablePaths`; dispatch workflow at commit; parse log score | `docs/submission-restriction-spec.md:55–70, §4–5`; `OVERVIEW.md:25–42` |
| **Router (organizer)** | Optional self-hosted submit path | `tools/router/router.py`, `tools/router/README.md` |
| **CI (non-ranked)** | `ci.yml` — doc consistency, shellcheck, overlay smoke | `docs/ci-coverage.md` |
| **benchd channel** | `tools/fetch-benchd.sh` — aarch64-linux-gnu binary + manifest pin | Preflight verifies staged copy |

**Intake must not rely on:** repo secrets, R2 credentials in job env (preflight forbids, `ranked-box-preflight.sh:91–95`), or GitHub artifact for score (`submission-restriction-spec.md:51`).

---

## vllm/ tree (not inventoried row-by-row)

| class | note |
|-------|------|
| **DROP** from Bonsai repo as Nemotron participant surface | Entire `vllm/` git subtree |
| **REPLACE** | Vendored `llama.cpp/` (PrismML fork) as sole editable engine source |

---

*Generated for Bonsai 2 9B challenge port planning. Nemotron repo remains read-only.*
