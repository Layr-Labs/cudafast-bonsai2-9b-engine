#!/usr/bin/env bash
# Run the public local fixture through the normal benchmark entry point.
# benchd continues to own correctness, cooling, timing, and sealed artifacts.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/local-baseline.sh

Run the Bonsai 2 9B CUDA local baseline after ./setup.sh. No organizer goldens,
reference workspace, or calibration file are required. This measures local
correctness and timing on one leg; it does not produce a ranked score.

Optional environment overrides (relative paths resolve from the checkout):
  MLXFAST_ENGINE_BIN                .build/release/mlxfast-runtime-worker
  MLXFAST_CORRECTNESS_GOLDEN_PATH    the public golden of the declared depth:
                                    correctness_prompts/public-longcopy-gate-english-512.golden.json
                                    (serial), else ...-512.mtp<N>.golden.json
                                    (mtp-head.manifest.json declares depth N)
  MLXFAST_TARGET_GGUF               the pinned GGUF (from ./setup.sh), else
                                    reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf
  MLXFAST_SCORE_PATH                score.local-iterate.json

The existing benchmark entry point verifies benchd, boots one resident engine
through tools/serve-up.sh, and enables the cool gate. It runs one leg: the
paired ranked path (tools/bonsai2-9b-measure-and-score.sh) is not involved.
USAGE
}

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
  fi
  usage >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${REPO_ROOT}"

export MLXFAST_ENGINE_BIN="${MLXFAST_ENGINE_BIN:-.build/release/mlxfast-runtime-worker}"
# The free run must match its golden exactly, and a depth-N run can take
# another path than the serial one. So each declared depth has its own public
# golden (recorded on a Spark by tools/record-goldens.sh), as each ranked
# prompt has its own per-depth oracle.
if [[ -z "${MLXFAST_CORRECTNESS_GOLDEN_PATH:-}" ]]; then
  spec="$("${REPO_ROOT}/tools/spec-declaration.sh" describe)" || exit 1
  if [[ "${spec}" == serial ]]; then
    MLXFAST_CORRECTNESS_GOLDEN_PATH="correctness_prompts/public-longcopy-gate-english-512.golden.json"
  else
    MLXFAST_CORRECTNESS_GOLDEN_PATH="correctness_prompts/public-longcopy-gate-english-512.${spec}.golden.json"
  fi
fi
export MLXFAST_CORRECTNESS_GOLDEN_PATH
DEFAULT_GGUF="${REPO_ROOT}/reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf"
export MLXFAST_TARGET_GGUF="${MLXFAST_TARGET_GGUF:-${MLXFAST_WEIGHTS_PATH:-${DEFAULT_GGUF}}}"
export MLXFAST_WEIGHTS_PATH="${MLXFAST_WEIGHTS_PATH:-${MLXFAST_TARGET_GGUF}}"
export MLXFAST_SCORE_PATH="${MLXFAST_SCORE_PATH:-score.local-iterate.json}"

if [[ ! -x "${MLXFAST_ENGINE_BIN}" ]]; then
  echo "local-baseline.sh: engine not executable: ${MLXFAST_ENGINE_BIN}" >&2
  echo "local-baseline.sh: run ./setup.sh to build and stage the engine, or set MLXFAST_ENGINE_BIN" >&2
  exit 1
fi
if [[ ! -f "${MLXFAST_TARGET_GGUF}" ]]; then
  echo "local-baseline.sh: target GGUF not found: ${MLXFAST_TARGET_GGUF}" >&2
  echo "local-baseline.sh: run ./setup.sh to download and verify the GGUF, or set MLXFAST_TARGET_GGUF" >&2
  exit 1
fi

{
  echo "local-baseline.sh: public local baseline (unranked)"
  echo "local-baseline.sh: golden=${MLXFAST_CORRECTNESS_GOLDEN_PATH}"
  echo "local-baseline.sh: gguf=${MLXFAST_TARGET_GGUF}"
  echo "local-baseline.sh: result=${MLXFAST_SCORE_PATH}; score=null is expected without ranked paired scoring"
} >&2

exec "${REPO_ROOT}/benchmark.sh" --local-iterate
