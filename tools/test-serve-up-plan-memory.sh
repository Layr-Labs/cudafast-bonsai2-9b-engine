#!/usr/bin/env bash
# test-serve-up-plan-memory.sh -- prove the memory plan in tools/serve-up.sh.
#
# THE PLAN. GB10 memory is unified: serve-up.sh refuses BEFORE the load unless
# the GGUF, the fixture KV budget, runtime headroom and host overhead fit in
# MemAvailable right now.
#
# WHAT THIS PROVES
#   1. the free-memory boundary, to the kilobyte: the plan ACCEPTS exactly the
#      required amount and REFUSES one kilobyte below it
#   2. weights that do not fit the reservation REFUSE by name
#   3. a meminfo with no MemAvailable REFUSES by name (never a blind load)
#   4. the refusal has NO environment bypass
#   5. the plan prints its table before deciding
#
# Hermetic: the real serve-up.sh, the real resident on its fake backend via the
# host launcher, a synthetic snapshot and a synthetic meminfo.
#
# Usage: tools/test-serve-up-plan-memory.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SERVE_UP="${ROOT_DIR}/tools/serve-up.sh"
FIXTURE="${ROOT_DIR}/fixtures/bonsai2_9b_cuda_v1_track.json"
# shellcheck source=tools/testlib/synthetic-window.sh
. "${ROOT_DIR}/tools/testlib/synthetic-window.sh"
synthetic_window_env

# serve-up.sh's own constant, restated on purpose: if it moves, case 1 goes red.
HOST_OVERHEAD_GIB=8

WORK="$(mktemp -d)"
SOCKDIR="$(mktemp -d /tmp/planmem.XXXXXX)"
trap 'rm -rf "${WORK}" "${SOCKDIR}"' EXIT

fails=0
pass() { printf 'test-serve-up-plan-memory: PASS -- %s\n' "$*"; }
fail() { printf 'test-serve-up-plan-memory: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

GGUF="${WORK}/w/rehearsal.gguf"
synthetic_window_gguf "${GGUF}" || { echo "cannot write the synthetic gguf" >&2; exit 1; }
KV_BYTES="$(jq -r '.serve_configuration.engine_env.RESIDENT_KV_CACHE_MEMORY_BYTES' "${FIXTURE}")"
MIN_HEADROOM_GIB=4
TOTAL_KB=125000000

# The required free bytes, computed the way serve-up.sh computes them.
required_kb="$(python3 - "${GGUF}" "${KV_BYTES}" "${HOST_OVERHEAD_GIB}" "${MIN_HEADROOM_GIB}" <<'PY'
import os, sys
gguf, kv, overhead, headroom = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
body = os.path.getsize(gguf)
want = body + kv + (headroom + overhead) * 1024**3
print(-(-want // 1024))
PY
)"

boot() { # boot MEMINFO [env...] -- --boot a serial leg; prints output, returns rc
  local meminfo="$1"; shift
  env "$@" MLXFAST_TARGET_GGUF="${GGUF}" SERVE_UP_LOG_DIR="${WORK}/log" \
      SERVE_UP_SOCKET_DIR="${SOCKDIR}" SERVE_UP_MEMINFO="${meminfo}" \
      SERVE_UP_HEALTH_TIMEOUT_S=60 \
      "${SERVE_UP}" --boot --spec serial --draft-len 0 --socket-out "${WORK}/sock.txt" 2>&1
}
stop_booted() {
  [[ -s "${WORK}/sock.txt" ]] && "${SERVE_UP}" --stop --socket "$(head -1 "${WORK}/sock.txt")" >/dev/null 2>&1
  rm -f "${WORK}/sock.txt"
}

# --- 1. the boundary --------------------------------------------------------
synthetic_window_meminfo "${WORK}/exact" "${TOTAL_KB}" "${required_kb}"
out="$(boot "${WORK}/exact")"; rc=$?
if [[ ${rc} -eq 0 ]] && grep -q 'memory plan ACCEPTS' <<<"${out}"; then
  pass "1a: exactly the required free memory is accepted"
else
  fail "1a: the exact boundary was refused (rc=${rc}): $(tail -1 <<<"${out}")"
fi
stop_booted
synthetic_window_meminfo "${WORK}/short" "${TOTAL_KB}" "$(( required_kb - 1 ))"
out="$(boot "${WORK}/short")"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'memory plan REFUSES the load:' <<<"${out}"; then
  pass "1b: one kilobyte short is refused by name"
else
  fail "1b: one kilobyte short was not refused (rc=${rc})"
fi
stop_booted

# --- 2. weights that do not fit after host overhead -----------------------
BIG="${WORK}/big.gguf"
synthetic_window_gguf "${BIG}"
truncate -s 8G "${BIG}"
synthetic_window_meminfo "${WORK}/tiny" 16000000 16000000
out="$(env MLXFAST_TARGET_GGUF="${BIG}" SERVE_UP_LOG_DIR="${WORK}/log" SERVE_UP_SOCKET_DIR="${SOCKDIR}" \
      SERVE_UP_MEMINFO="${WORK}/tiny" "${SERVE_UP}" --boot --spec serial --draft-len 0 \
      --socket-out "${WORK}/sock.txt" 2>&1)"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'do not fit' <<<"${out}"; then
  pass "2: oversized weights are refused by name"
else
  fail "2: oversized weights were not refused (rc=${rc}): $(tail -1 <<<"${out}")"
fi
stop_booted

# --- 3. no MemAvailable -----------------------------------------------------
printf 'MemTotal: %s kB\n' "${TOTAL_KB}" > "${WORK}/nomem"
out="$(boot "${WORK}/nomem")"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'refusing to start a load blind' <<<"${out}"; then
  pass "3: a meminfo with no MemAvailable is refused by name"
else
  fail "3: a blind load was not refused (rc=${rc})"
fi
stop_booted

# --- 4. no environment bypass ------------------------------------------------
out="$(boot "${WORK}/short" SERVE_UP_MEM_HEADROOM_GB=0 SERVE_UP_SKIP_MEMORY_PLAN=1 RESIDENT_GPU_MEMORY_UTILIZATION=0.01)"; rc=$?
if [[ ${rc} -ne 0 ]] && grep -q 'memory plan REFUSES' <<<"${out}"; then
  pass "4: no environment variable relaxes the plan"
else
  fail "4: an environment variable relaxed the plan (rc=${rc})"
fi
stop_booted

# --- 5. the table is printed before the decision -----------------------------
out="$(boot "${WORK}/short")"
if grep -q 'memory plan for this leg' <<<"${out}" && grep -q 'KV cache budget' <<<"${out}"; then
  pass "5: the plan table is printed with the refusal"
else
  fail "5: the plan table is missing"
fi
stop_booted

if [[ ${fails} -eq 0 ]]; then
  echo "test-serve-up-plan-memory: all cases passed"
  exit 0
fi
echo "test-serve-up-plan-memory: ${fails} case(s) failed" >&2
exit 1
