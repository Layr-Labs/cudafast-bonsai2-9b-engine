# Contributor guide — Bonsai 2 9B CUDA (llama.cpp) engine

Working contract for changes in this repository.

Read [README.md](README.md) first and [TASK.md](TASK.md) second. This file adds
the operational rules for iterating here. The ranked track is
`bonsai2-9b-cuda-v1`.

## Goal

Make the target prefill and decode faster on a DGX Spark, single stream, without
changing what it computes beyond the ranked timed-window rule: at most 40 mismatches per thousand tokens, each the reference's second choice within a relative gap of 0.01.

## Authorities

| Question | File |
|---|---|
| What you may edit | `benchmark.json` `editablePaths`: `llama.cpp`, `harness`, `mtp-head.manifest.json` |
| Pins, engine configuration, depths, scoring | `fixtures/bonsai2_9b_cuda_v1_track.json` |
| Why the rules say what they say | `docs/participant-contract.md` |
| The resident wire | `docs/resident.md` |
| Shared design | `docs/design.md` |

## Where to work

| Lever | Where |
|---|---|
| Speculation arm and depth | `mtp-head.manifest.json` `spec`: `mode` `mtp` (1 to 6) or `dflash` (1 to 15) |
| ggml/llama CUDA kernels, graphs, fusion | `llama.cpp/ggml/`, `llama.cpp/src/` |
| Speculative loops (draft-mtp, draft-dflash) | `llama.cpp/examples/speculative-simple/`, common speculative helpers |
| Resident wire and bookkeeping | `harness/llama_resident/` |
| Adapter / benchd boundary | `harness/protocol-adapter/src/` |
| New CUDA kernels (optional) | `harness/kernels/` + `harness/kernels/build.sh` |
| Engine image and sandboxed build | `tools/engine/` (organizer + ranked build path) |

## Rules

- **Never edit outside `editablePaths`.** That covers `tools/`, `fixtures/`,
  `setup.sh`, `benchmark.sh`, `benchmark.json`, `.github/` and
  `correctness_prompts/`, all trusted. A submission that touches them is
  refused.
- **Never change a numerical format.** Section 3.4 of the contract lists them.
  Keep the weight file packing frozen, KV f16, in-file MTP head, the pinned
  DFlash2 drafter, batch 1 and context 8192. Hadamard/sign transforms may be
  fused, not removed.
- **Never reuse work across requests or phases.** That rules out prefix
  caching, memoized prompts and state kept across `sync`. The resident's
  `invalidate` must leave the engine with nothing from the previous phase.
  The timed prefill follows one unmeasured warm-up prefill of the same prompt
  in the same phase; compute both in full and carry nothing between them.
- **Never move prefill work out of `free_decode_begin`** (contract section 5.1.1).
- **A timed divergence is replayed, not forgiven.** If your timed stream leaves
  the golden at a tie, benchd replays it through the reference: every token
  must be one the reference nearly chose (contract section 5.4.1). The local
  modes cannot replay, so they still demand an exact match.
- **Keep the engine reproducible.** Run `tools/determinism-probe.py` against a
  booted resident; it must report one stream at serial and at your depth.
- **Keep the resident wire intact.** The Rust client in
  `harness/protocol-adapter/src/resident.rs` and the C++ resident must agree.
  `test_adapter_e2e.sh` proves it at depths 0–3 on the target GGUF.
- **Keep `IMPLEMENTED_DEPTH` in the resident and `ENGINE_IMPLEMENTED_DEPTH` in
  `session_backend.rs` equal.** Tests on both sides check it.
- **Report raw counters only.** The adapter never reports a rate, a time or a
  ratio. benchd does all measurement.
- **Keep `harness/protocol-adapter/target/` and `__pycache__/` out of a
  submission.** They count against the byte cap.

## Checks before you submit

```bash
cargo fmt --manifest-path harness/protocol-adapter/Cargo.toml -- --check
cargo clippy --manifest-path harness/protocol-adapter/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
python3 tools/lint-benchmark-manifest.py
cmake -S harness/llama_resident -B /tmp/llama-resident-cpu -DCMAKE_BUILD_TYPE=Release -DLLAMA_RESIDENT_CUDA=OFF
cmake --build /tmp/llama-resident-cpu -j "$(nproc)"
ctest --test-dir /tmp/llama-resident-cpu --output-on-failure --no-tests=error
python3 -m unittest discover -s harness/llama_resident/test -p 'test_oracle_stop.py'
```

These commands check the adapter, the manifest, the resident C++ tests and the oracle
stop rule with no GPU.

```bash
./setup.sh && ./tools/local-baseline.sh
```

This command runs one real leg. It needs a Spark or another sm_121 box with the
pinned image.

## Environment rules on a ranked box

Never export the following in a ranked job; `tools/ranked-box-preflight.sh`
refuses each of them by name:

- **Resident sockets:** `DS4_RESIDENT_SOCKET` and
  `BENCH_WORKER_RESIDENT_SOCKET`. benchd injects them per leg, and an inherited
  one is a third resident.
- **Engine configuration:** anything prefixed `RESIDENT_`, `VLLM_`, `HF_`,
  `TRANSFORMERS_`, `TORCHINDUCTOR_` or `TRITON_`. The fixture's `engine_env`
  is the only engine configuration.
- **Test seams:** `SERVE_UP_LAUNCHER` and `STAGE_VLLM_OVERLAY_BASE`.
- **Measurement skips:** `BENCHD`, `MLXFAST_SKIP_*` and
  `MLXFAST_LOCAL_COOL_GATE`.

## Notes

Keep your own notes in an untracked local file such as `notes.local.md`.
Never commit anything outside `editablePaths`: that includes `docs/`.
