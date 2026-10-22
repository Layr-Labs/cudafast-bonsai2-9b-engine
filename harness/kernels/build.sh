#!/usr/bin/env bash
# Official tools/engine/build.sh runs this allowed hook in its pinned GPU
# container, outside the timed benchmark and without network access.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${KERNELS_OUT:?official engine build must provide KERNELS_OUT}"
mkdir -p -- "${KERNELS_OUT}"

nvcc --version
nvcc -O2 -std=c++17 -arch=sm_121a \
  -I "${ROOT_DIR}/llama.cpp/ggml/src/ggml-cuda" \
  "${ROOT_DIR}/harness/llama_resident/test/topk_select_equivalence.cu" \
  -o "${KERNELS_OUT}/topk-select-equivalence"

# Fail setup on any mismatch, runtime error, inactive dispatch, or timeout.
timeout 600 "${KERNELS_OUT}/topk-select-equivalence"
