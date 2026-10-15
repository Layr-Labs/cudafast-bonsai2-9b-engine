#!/usr/bin/env bash
# Exercise the real local entry point, root proxy, and facade with a stub benchd.
# No model, GPU, toolchain, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Canonical (pwd -P): on macOS mktemp answers under /var, a symlink to
# /private/var, and the helper records its cwd resolved.
WORK="$(cd "$(mktemp -d)" && pwd -P)"
SOCKDIR="$(mktemp -d /tmp/localbase.XXXXXX)"
trap 'rm -rf "${WORK}" "${SOCKDIR}"' EXIT
# shellcheck source=tools/testlib/synthetic-window.sh
. "${REPO_ROOT}/tools/testlib/synthetic-window.sh"
synthetic_window_env
synthetic_window_meminfo "${WORK}/meminfo"
export SERVE_UP_MEMINFO="${WORK}/meminfo" SERVE_UP_SOCKET_DIR="${SOCKDIR}"
TEST_ROOT="${WORK}/checkout with spaces"
mkdir -p "${TEST_ROOT}/tools" "${TEST_ROOT}/fixtures" "${TEST_ROOT}/correctness_prompts" \
  "${TEST_ROOT}/.build/release" "${TEST_ROOT}/harness"
mkdir -p "${TEST_ROOT}/tools/testlib/wire_resident"
cp "${REPO_ROOT}/tools/testlib/wire_resident/"*.py "${TEST_ROOT}/tools/testlib/wire_resident/"
printf '#!/bin/sh\n' > "${TEST_ROOT}/.build/release/llama-resident"
chmod +x "${TEST_ROOT}/.build/release/llama-resident"
GGUF_DEFAULT="${TEST_ROOT}/reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf"
mkdir -p "$(dirname "${GGUF_DEFAULT}")"
synthetic_window_gguf "${GGUF_DEFAULT}"
cp "${REPO_ROOT}/benchmark.sh" "${TEST_ROOT}/"
mkdir -p "${TEST_ROOT}/tools/engine"
cp "${REPO_ROOT}/tools/engine/"{Dockerfile,read-engine-image-id.sh} "${TEST_ROOT}/tools/engine/"
cp "${REPO_ROOT}/tools/"{local-baseline,benchmark,spec-declaration,serve-up}.sh "${TEST_ROOT}/tools/"
cp "${REPO_ROOT}/benchmark.json" "${REPO_ROOT}/mtp-head.manifest.json" "${TEST_ROOT}/"
cp "${REPO_ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json" "${TEST_ROOT}/fixtures/"
# The public golden is recorded on a Spark; the stub benchd never reads it.
echo '{}' > "${TEST_ROOT}/correctness_prompts/public-longcopy-gate-english-512.golden.json"
for d in 1 2 3 4 5 6; do echo '{}' > "${TEST_ROOT}/correctness_prompts/public-longcopy-gate-english-512.mtp${d}.golden.json"; done
# The cases below run serial unless they say otherwise.
SERIAL_DECL='{"version":1,"source":"pinned","spec":{"enabled":false,"num_speculative_tokens":0}}'
printf '%s\n' "${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"

cat > "${WORK}/benchd" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${2:-}" == "--help" ]]; then
  echo '--mtp-depth'
  exit 0
fi
printf '%s\n' "$PWD" > "${CAPTURE}.cwd"
printf '%s\n' "$@" > "${CAPTURE}.argv"
printf '%s\n' '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}'
exit "${STUB_EXIT:-0}"
STUB
chmod +x "${WORK}/benchd"
cp "${WORK}/benchd" "${TEST_ROOT}/.build/release/mlxfast-runtime-worker"

# Every case boots a REAL resident through serve-up.sh on the fake backend
# (host launcher, synthetic snapshot), so the dispatch is the one a real local
# run takes; the stub benchd stands in for the scorer.
run_case() {
  local name="$1"
  shift
  rm -f "${WORK}/${name}.argv"
  rc=0
  (
    cd "${WORK}"
    env -u MLXFAST_ENGINE_BIN -u MLXFAST_CORRECTNESS_GOLDEN_PATH \
      -u MLXFAST_WEIGHTS_PATH -u MLXFAST_SCORE_PATH -u MLXFAST_TARGET_GGUF \
      -u MLXFAST_CORRECTNESS_GOLDEN_SHA256 -u MLXFAST_CORRECTNESS_GOLDEN_BYTES \
      -u MLXFAST_QWEN_MTP_TRACK_ID \
      -u SPEC_DECLARATION_MANIFEST -u SPEC_DECLARATION_CONTRACT \
      BENCHD="${WORK}/benchd" CAPTURE="${WORK}/${name}" "$@"
  ) > "${WORK}/${name}.stdout" 2> "${WORK}/${name}.stderr" || rc=$?
}

assert_arg() {
  local name="$1" flag="$2" value="$3"
  awk -v flag="${flag}" -v value="${value}" '
    previous == flag && $0 == value { found=1 }
    { previous=$0 }
    END { exit !found }
  ' "${WORK}/${name}.argv"
}

run_case defaults "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
[[ "$(cat "${WORK}/defaults.cwd")" == "${TEST_ROOT}" ]]
assert_arg defaults --mode local-iterate
assert_arg defaults --engine .build/release/mlxfast-runtime-worker
assert_arg defaults --weights "${TEST_ROOT}/reference_weights/Ternary-Bonsai-2-9B-gguf"
assert_arg defaults --golden correctness_prompts/public-longcopy-gate-english-512.golden.json
assert_arg defaults --score-path score.local-iterate.json
grep -Fxq -- '--cool-gate' "${WORK}/defaults.argv"
grep -q 'public local baseline (unranked)' "${WORK}/defaults.stderr"
grep -q 'score=null is expected' "${WORK}/defaults.stderr"
[[ "$(cat "${WORK}/defaults.stdout")" == '{"score":null,"passed":true,"metrics":{"decode_seconds_per_token":0.05}}' ]]

# A declared depth picks the public golden of that depth.
printf '%s\n' '{"version":1,"source":"pinned","spec":{"enabled":true,"num_speculative_tokens":3}}' > "${TEST_ROOT}/mtp-head.manifest.json"
run_case depth3 "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg depth3 --golden correctness_prompts/public-longcopy-gate-english-512.mtp3.golden.json
assert_arg depth3 --mtp-depth 3
printf '%s\n' "${SERIAL_DECL}" > "${TEST_ROOT}/mtp-head.manifest.json"

synthetic_window_gguf "${TEST_ROOT}/box.gguf"
run_case snapshot_env MLXFAST_TARGET_GGUF="${TEST_ROOT}/box.gguf" "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg snapshot_env --weights "${TEST_ROOT}"

cp "${WORK}/benchd" "${TEST_ROOT}/custom engine"
cp "${TEST_ROOT}/correctness_prompts/public-longcopy-gate-english-512.golden.json" "${TEST_ROOT}/custom golden.json"
synthetic_window_gguf "${TEST_ROOT}/custom.gguf"
run_case overrides MLXFAST_ENGINE_BIN='custom engine' \
  MLXFAST_CORRECTNESS_GOLDEN_PATH='custom golden.json' \
  MLXFAST_TARGET_GGUF="${TEST_ROOT}/custom.gguf" \
  MLXFAST_SCORE_PATH='custom result.json' \
  "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 0 ]]
assert_arg overrides --engine 'custom engine'
assert_arg overrides --golden 'custom golden.json'
assert_arg overrides --weights "${TEST_ROOT}"
assert_arg overrides --score-path 'custom result.json'

run_case failed STUB_EXIT=1 "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 ]]
run_case help "${TEST_ROOT}/tools/local-baseline.sh" --help
[[ "${rc}" == 0 && ! -f "${WORK}/help.argv" ]]
run_case wrong_mode "${TEST_ROOT}/tools/local-baseline.sh" --official
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_mode.argv" ]]
run_case missing_golden MLXFAST_CORRECTNESS_GOLDEN_PATH=absent.json "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_golden.argv" ]]
grep -q 'correctness golden not found' "${WORK}/missing_golden.stderr"
run_case wrong_pin MLXFAST_CORRECTNESS_GOLDEN_SHA256=bad "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/wrong_pin.argv" ]]
grep -q 'sha256 mismatch' "${WORK}/wrong_pin.stderr"
run_case missing_engine MLXFAST_ENGINE_BIN=absent-engine "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_engine.argv" ]]
grep -q './setup.sh' "${WORK}/missing_engine.stderr"
run_case missing_snapshot MLXFAST_TARGET_GGUF=absent.gguf "${TEST_ROOT}/tools/local-baseline.sh"
[[ "${rc}" == 1 && ! -f "${WORK}/missing_snapshot.argv" ]]
grep -q 'target GGUF not found' "${WORK}/missing_snapshot.stderr"

echo 'test-local-baseline.sh: all 10 cases passed'
