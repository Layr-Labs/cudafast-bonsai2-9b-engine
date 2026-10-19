#!/usr/bin/env bash
# Run inside the existing offline engine image on a Spark; /src = candidate.
set -euo pipefail
G=/src/llama.cpp/ggml
T=/src/harness/llama_resident/test/dod_qkv
mkdir -p /out
nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -use_fast_math \
  -extended-lambda -lineinfo -Xptxas=-v -Xcompiler=-ffunction-sections,-fdata-sections \
  -Xlinker=--gc-sections -I"$G/include" -I"$G/src" -I"$G/src/ggml-cuda" \
  "$T/equiv.cu" -o /out/equiv 2>/out/ptxas.log
/out/equiv "$@"
