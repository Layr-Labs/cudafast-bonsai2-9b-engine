# harness/

The participant-editable engine adapter.

| Path | What it is |
|---|---|
| `protocol-adapter/` | The Engine Protocol v1 adapter, in Rust. benchd spawns it once per phase as `.build/release/mlxfast-runtime-worker` and speaks NDJSON to it over stdio. It attaches to the leg's resident through the socket benchd injects (`BENCH_WORKER_RESIDENT_SOCKET`, or the legacy `DS4_RESIDENT_SOCKET`). With no socket, it serves a deterministic mock and says so. |
| `llama_resident/` | The resident for this track: `llama-resident` (C++) links vendored libllama, owns the socket wire and frontier bookkeeping. `engine_fake.*` supports GPU-free tests; `engine_llama.*` is the scored engine. Built from `harness/llama_resident/CMakeLists.txt` (see `docs/design.md` section 3). `tools/serve-up.sh` boots it in the pinned CUDA image. |
| `kernels/` (optional) | Custom CUDA/C++ kernels. `kernels/build.sh` runs in the pinned image at setup with `KERNELS_OUT` and `TORCH_CUDA_ARCH_LIST` set. Its output is importable from the resident. |

The adapter reports raw counters only: tokens, `acceptance_lengths`, and the
drafted, accepted and committed totals. benchd times every verb itself.
`docs/resident.md` documents the socket wire, and the protocol schema is
`protocol-adapter/reference/engine-protocol-v1.schema.json`.

## Tests (no GPU)

```bash
cargo test --manifest-path harness/protocol-adapter/Cargo.toml
cargo build --release --manifest-path harness/protocol-adapter/Cargo.toml
cmake -S harness/llama_resident -B /tmp/llama-resident-build -DLLAMA_RESIDENT_CUDA=OFF
cmake --build /tmp/llama-resident-build --target llama-resident-tests
/tmp/llama-resident-build/llama-resident-tests
```

On a Spark box with the target GGUF and CUDA image build:

```bash
harness/llama_resident/test/test_adapter_e2e.sh
```
