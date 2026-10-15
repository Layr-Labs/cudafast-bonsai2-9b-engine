#!/usr/bin/env bash
# The vendoring proof: a participant may EDIT THE ENGINE, and the gates agree.
#
# The engine is the PrismML llama.cpp fork, VENDORED at llama.cpp/ and declared
# in benchmark.json editablePaths, plus harness/ (resident + optional kernels).
# This suite keeps that true. It asserts:
#
#   1. THE ENGINE IS IN THE SURFACE. benchmark.json editablePaths carries llama.cpp
#      and harness, and nothing else but the MTP declaration.
#   2. THE HARNESS HASH RECORDS ENGINE EDITS. benchd seals a hash over a FIXED
#      root set that includes llama.cpp/ and harness/, so an engine edit is visible
#      in the sealed provenance; a docs-only edit outside the roster is not.
#   3. AN ENGINE EDIT IS IN THE VENDORED TREE. Participant changes under llama.cpp/
#      are what the ranked build compiles; there is no overlay layer.
#
# THE ROSTER BELOW IS A HAND-COPY of benchd's HARNESS_HASH_ROOTS
# (crates/bench-core/src/harness_hash.rs). benchd owns the real one; if the two
# drift, this suite is testing a hash nobody computes.
#
# Hermetic: no weights, no GPU, no docker, no network.
#
# Usage: tools/test-participant-engine-edit.sh [-v]
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

EXPECTED_MIN_ASSERTIONS=8
PASSED=0; FAILED=0; FAILURES=()
pass() { PASSED=$((PASSED+1)); [[ "${VERBOSE}" == "1" ]] && echo "ok    $1"; return 0; }
fail() { FAILED=$((FAILED+1)); FAILURES+=("$1"); echo "FAIL  $1" >&2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/participant-edit.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

harness_hash() {
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
ROOTS = ["Package.swift", "Sources", "Tests", "benchmark.json", "benchmark.sh",
         "setup.sh", "tools", "README.md", "TASK.md", "harness", "vllm", "llama.cpp"]
paths = []
for r in ROOTS:
    p = os.path.join(root, r)
    if not os.path.exists(p):
        continue
    if os.path.isdir(p):
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith('.')
                           and not os.path.islink(os.path.join(dirpath, d))]
            for fn in filenames:
                if fn.startswith('.'):
                    continue
                fp = os.path.join(dirpath, fn)
                if os.path.islink(fp) or not os.path.isfile(fp):
                    continue
                paths.append(fp)
    else:
        paths.append(p)
h = hashlib.sha256()
for p in sorted(paths):
    try:
        with open(p, 'rb') as fh:
            data = fh.read()
    except OSError:
        continue
    h.update(p.encode()); h.update(data)
print(h.hexdigest())
PY
}

SB="${WORK}/repo"
mkdir -p "${SB}/llama.cpp" "${SB}/harness/kernels" "${SB}/tools" "${SB}/docs"
cp "${REPO_ROOT}/benchmark.json" "${SB}/benchmark.json"
printf 'pin\n' > "${SB}/llama.cpp/PRISM_PIN.md"
printf 'kernel\n' > "${SB}/harness/kernels/k.cu"
printf 'tool\n' > "${SB}/tools/thing.sh"
printf 'notes\n' > "${SB}/docs/notes.md"
git -C "${SB}" init --quiet
git -C "${SB}" add -A
git -C "${SB}" -c user.email=t@t -c user.name=t commit --quiet -m base

if jq -e '.editablePaths == ["llama.cpp","harness","mtp-head.manifest.json"]' "${SB}/benchmark.json" >/dev/null; then
  pass "benchmark.json editablePaths is exactly llama.cpp, harness, mtp-head.manifest.json"
else
  fail "benchmark.json editablePaths is exactly llama.cpp, harness, mtp-head.manifest.json (got $(jq -c .editablePaths "${SB}/benchmark.json"))"
fi

H0="$(harness_hash "${SB}")"
if [[ "${H0}" =~ ^[0-9a-f]{64}$ ]]; then
  pass "the harness hash is a sha256 over the roster"
else
  fail "the harness hash is a sha256 over the roster (got '${H0}')"
fi
for probe in llama.cpp/PRISM_PIN.md harness/kernels/k.cu tools/thing.sh; do
  printf '# participant tuning\n' >> "${SB}/${probe}"
  if [[ "$(harness_hash "${SB}")" != "${H0}" ]]; then
    pass "an edit to ${probe} CHANGES the harness hash"
  else
    fail "an edit to ${probe} CHANGES the harness hash"
  fi
  git -C "${SB}" checkout --quiet -- "${probe}"
done
printf 'more notes\n' >> "${SB}/docs/notes.md"
if [[ "$(harness_hash "${SB}")" == "${H0}" ]]; then
  pass "an edit outside the roster (docs/) leaves the harness hash unchanged"
else
  fail "an edit outside the roster (docs/) leaves the harness hash unchanged"
fi
git -C "${SB}" checkout --quiet -- docs/notes.md
if [[ "$(harness_hash "${SB}")" == "${H0}" ]]; then
  pass "reverting every edit returns the harness hash to its original value"
else
  fail "reverting every edit returns the harness hash to its original value"
fi

benchd_bin="${BENCHD:-${REPO_ROOT}/benchd-bin/benchd}"
if [[ -x "${benchd_bin}" ]]; then
  H_BENCHD="$(cd "${SB}" && "${benchd_bin}" harness-hash 2>/dev/null | head -1)"
  if [[ "${H_BENCHD}" == "${H0}" ]]; then
    pass "benchd harness-hash equals the transcribed roster's hash"
  else
    fail "benchd harness-hash equals the transcribed roster's hash (benchd ${H_BENCHD:0:12}, transcription ${H0:0:12})"
  fi
else
  echo "test-participant-engine-edit: NOTE -- no benchd binary at ${benchd_bin}; the transcription was not cross-checked against it" >&2
fi

BENCHD_ROSTER_SRC="${BENCHD_SRC_DIR:-}/crates/bench-core/src/harness_hash.rs"
if [[ -n "${BENCHD_SRC_DIR:-}" && -f "${BENCHD_ROSTER_SRC}" ]]; then
  want="$(sed -n '/pub const HARNESS_HASH_ROOTS/,/\];/p' "${BENCHD_ROSTER_SRC}" \
    | grep -oE '"[^"]+"' | tr -d '"' | tr '\n' ' ')"
  have="Package.swift Sources Tests benchmark.json benchmark.sh setup.sh tools README.md TASK.md harness vllm llama.cpp "
  if [[ "${want}" == "${have}" ]]; then
    pass "the transcribed roster matches benchd's HARNESS_HASH_ROOTS"
  else
    fail "the transcribed roster matches benchd's HARNESS_HASH_ROOTS (benchd: ${want})"
  fi
  if [[ "${want}" == *"llama.cpp"* && "${want}" == *"harness"* ]]; then
    pass "benchd's roster includes llama.cpp and harness (engine edits are in the sealed provenance)"
  else
    fail "benchd's roster no longer includes llama.cpp and harness; this suite's premise has changed"
  fi
else
  echo "test-participant-engine-edit: NOTE -- BENCHD_SRC_DIR unset, the roster transcription was not cross-checked" >&2
fi

if [[ -f "${REPO_ROOT}/llama.cpp/PRISM_PIN.md" ]]; then
  cp "${REPO_ROOT}/llama.cpp/PRISM_PIN.md" "${WORK}/PRISM_PIN.md"
  printf '\n# participant probe\n' >> "${WORK}/PRISM_PIN.md"
  if grep -q 'participant probe' "${WORK}/PRISM_PIN.md"; then
    pass "a llama.cpp/ edit is carried in the vendored engine tree the build compiles"
  else
    fail "a llama.cpp/ edit is carried in the vendored engine tree the build compiles"
  fi
else
  fail "llama.cpp/PRISM_PIN.md is missing from this repository"
fi

echo "participant-engine-edit: ${PASSED} passed, ${FAILED} failed"
if [[ "${FAILED}" -ne 0 ]]; then
  printf '  - %s\n' "${FAILURES[@]}" >&2
  exit 1
fi
if [[ "${PASSED}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]]; then
  echo "participant-engine-edit: ran only ${PASSED} assertions, expected at least ${EXPECTED_MIN_ASSERTIONS} -- the suite shrank" >&2
  exit 1
fi
exit 0
