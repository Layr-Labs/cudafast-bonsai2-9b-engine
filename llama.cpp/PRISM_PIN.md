# PrismML llama.cpp vendor pin (Bonsai 9B)

| Field | Value |
|---|---|
| Upstream | https://github.com/PrismML-Eng/llama.cpp |
| Commit | `4fbda1292562cc1f23904d5867fb79603737547a` |
| Date (author) | 2026-10-07 11:57:44 -0700 |
| Subject | arm: NEON + i8mm vec_dot for PTQ1_0 (#290) |

## Kept paths (copied into this repo)

- `CMakeLists.txt`, `CMakePresets.json` (if present)
- `cmake/`
- `ggml/`
- `src/`
- `include/`
- `common/`
- `vendor/`
- `examples/CMakeLists.txt`, `examples/speculative-simple/`
- `tools/CMakeLists.txt`, `tools/llama-bench/`
- `LICENSE`, `AUTHORS` (if present)

## Dropped (not vendored)

Everything else in upstream at this commit, including at least:

- `models/`, `docs/`, `tests/`, `gguf-py/`, `.github/`
- `tools/server/` and other tools subtrees (batched-bench, cli, quantize, mtmd, …)
- Other `examples/*` except `speculative-simple/`
- `app/`, `pocs/`, `scripts/`, `requirements/`, `pyproject.toml`, etc.

## Files modified vs upstream (trim / Bonsai hook)

| File | Why |
|---|---|
| `CMakeLists.txt` | Drop `pocs/` when examples build; guard `app/` if directory absent (trimmed tree). No resident hook (resident builds from `harness/llama_resident/`). |
| `ggml/src/CMakeLists.txt` | Only `ggml_add_backend(CUDA)`; CPU via `ggml-cpu`. |
| `vendor/CMakeLists.txt` | Drop unused `miniaudio` (not linked by `common`). |
| `examples/CMakeLists.txt` | Only `add_subdirectory(speculative-simple)` so configure does not require other example dirs. |
| `tools/CMakeLists.txt` | Only `add_subdirectory(llama-bench)` so configure does not require other tool dirs. |
| `common/speculative.cpp` | Draft-MTP sampler: greedy (temp 0, top-1) with fixed seed for reproducible verify batches. |
| `common/sampling.cpp` | Upstream `527ea4d9` (common : take the top-k straight from the logits when top-k starts the chain), applied unchanged. The greedy MTP draft sampler then reads the top logit without building a token array for the whole vocabulary. |

## Local verify (CPU)

```bash
cmake -S llama.cpp -B /tmp/bonsai-cfg-check \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /tmp/bonsai-cfg-check -j8 --target llama common llama-speculative-simple llama-bench
```

Succeeded on 2026-10-08 (x86_64, GCC 16.2.1, CMake 4.4.4).
