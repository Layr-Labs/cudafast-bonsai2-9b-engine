#!/usr/bin/env bash
# test-pool-refresh.sh -- tools/pool-refresh.sh against stubbed boxes.
#
# ssh/scp/sudo/install/flock are stubs: each "box" is a directory with a home
# (holding a stub tools/record-goldens.sh in checkout/repo) and a goldens
# directory. The script runs from a throwaway repo whose fixture it may edit,
# with a local bare origin. No network, no GPU, no real box.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
failures=0
fail() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }
ok() { echo "ok    $*"; }

FAKE="${WORK}/boxes" STUB="${WORK}/stub" ROOT="${WORK}/root" PRIV="${WORK}/private"
mkdir -p "${FAKE}" "${STUB}" "${ROOT}/tools" "${ROOT}/fixtures" "${PRIV}"
export FAKE

cat > "${STUB}/ssh" <<'EOF'
#!/usr/bin/env bash
while [[ "$1" == -* ]]; do [[ "$1" == -o ]] && shift; shift; done
box="$1"; shift
echo "ssh ${box} $*" >> "${FAKE}/calls.log"
cd "${FAKE}/${box}/home" && HOME="${FAKE}/${box}/home" BOX="${box}" bash -c "$*"
EOF
cat > "${STUB}/scp" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -q ]] && shift
args=()
for a in "$@"; do
  if [[ "${a}" =~ ^([a-z0-9-]+):(.*)$ ]]; then a="${FAKE}/${BASH_REMATCH[1]}/home/${BASH_REMATCH[2]}"; fi
  args+=("${a}")
done
cp "${args[@]}"
EOF
cat > "${STUB}/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == -n ]] && shift
exec "$@"
EOF
cat > "${STUB}/install" <<'EOF'
#!/usr/bin/env bash
files=()
while [[ $# -gt 0 ]]; do case "$1" in -m|-o|-g) shift 2 ;; *) files+=("$1"); shift ;; esac; done
cp "${files[@]}"
EOF
cat > "${STUB}/flock" <<'EOF'
#!/usr/bin/env bash
shift 3; exec "$@"
EOF
# record-goldens stub: deterministic content per (name, step); a box named in
# ${FAKE}/mismatch writes different bytes; ${FAKE}/fail-<box>-<steps> makes one job fail.
RECORD_STUB="${WORK}/record-goldens.sh"
cat > "${RECORD_STUB}" <<'EOF'
#!/usr/bin/env bash
set -e
only=() steps=""
while [[ $# -gt 0 ]]; do
  case "$1" in --prompts) P="$2";; --out) O="$2";; --only) only+=("$2");; --steps) steps="$2";; esac; shift 2
done
echo "record ${BOX} ${steps} ${only[*]}" >> "${FAKE}/calls.log"
[[ ! -e "${FAKE}/fail-${BOX}-${steps}" ]] || { echo "boom" >&2; exit 1; }
extra=""; grep -qx "${BOX}" "${FAKE}/mismatch" 2>/dev/null && extra="${BOX}"
for n in "${only[@]}"; do
  [[ -f "${P}/${n}.tokens.json" ]]
  if [[ "${steps}" == serial ]]; then echo "{\"serial\":\"${n}${extra}\"}" > "${O}/${n}.golden.json"
  else [[ -f "${O}/${n}.golden.json" ]]; echo "{\"${steps}\":\"${n}${extra}\"}" > "${O}/${n}.${steps}.golden.json"; fi
done
EOF
cat > "${WORK}/fake-stitch" <<'EOF'
#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do case "$1" in --out) out="$2";; --label) label="$2";; esac; shift; done
mkdir -p "${out}"
for n in 1 2 3 4 5 6 7 8; do echo "[${n}]" > "${out}/${label}-${n}.tokens.json"; done
EOF
chmod +x "${STUB}"/* "${RECORD_STUB}" "${WORK}/fake-stitch"

BOXES="b1 b2 b3"
for b in ${BOXES}; do
  mkdir -p "${FAKE}/${b}/home/checkout/repo/tools" "${FAKE}/${b}/home/goldens"
  cp "${RECORD_STUB}" "${FAKE}/${b}/home/checkout/repo/tools/record-goldens.sh"
  cp /dev/null "${FAKE}/${b}/home/goldens/.keep"
done

ln -s "${REPO_ROOT}/tools/pool-refresh.sh" "${ROOT}/tools/pool-refresh.sh"
cp "${REPO_ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json" "${ROOT}/fixtures/"
git -C "${ROOT}" init -q -b main
git -C "${ROOT}" add -A
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qm base
git clone -q --bare "${ROOT}" "${WORK}/origin.git"
git -C "${ROOT}" remote add origin "${WORK}/origin.git"
git -C "${ROOT}" fetch -q origin
# The legacy pins must be "staged" on each box for verify.
for b in ${BOXES}; do
  jq -r '[.timed_prompt_pool[], (.speculative_oracles[] | .[])] | .[].r2_path | split("/")[-1]' \
    "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json" | sort -u | while read -r f; do echo legacy > "${FAKE}/${b}/home/goldens/${f}"; done
done
# ...with the sha256 the verify step compares: rewrite the fixture pins to match.
LEGACY_SHA="$(echo legacy | sha256sum | cut -d' ' -f1)"
jq --arg s "${LEGACY_SHA}" '.timed_prompt_pool[].sha256 = $s | .speculative_oracles[][].sha256 = $s
    | .hidden_correctness_golden.sha256 = $s' "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json" > "${WORK}/fx"
mv "${WORK}/fx" "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json"
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "legacy pins" && git -C "${ROOT}" push -q origin HEAD:main

FX="${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json"
# A generation is 8 prompts, and every job measures every live golden, so the
# fixture this suite pins into runs 8 pairs.
jq '.official_pairs = 8' "${FX}" > "${WORK}/fx" && mv "${WORK}/fx" "${FX}"
SHIPPED_POOL="$(jq '.timed_prompt_pool | length' "${FX}")"
SHIPPED_LIVE="$(jq -c '.live_goldens' "${FX}")"
refresh() {
  env PATH="${STUB}:${PATH}" REFRESH_BOXES="${BOXES}" REFRESH_BOX_REPO=checkout/repo \
    REFRESH_PRIVATE="${PRIV}" REFRESH_BOX_GOLDENS='$HOME/goldens' REFRESH_PYTHON="${WORK}/fake-stitch" \
    "${ROOT}/tools/pool-refresh.sh" --corpus "${WORK}" --snapshot "${WORK}" "$@" 2>&1
}
nf() { local c=0 f; for f in $1; do [[ -e "${f}" ]] && c=$((c + 1)); done; echo "${c}"; }  # files matching a glob
records() { grep -c "^record " "${FAKE}/calls.log" 2>/dev/null || echo 0; }

# --- required env ---------------------------------------------------------------
req_out="$(env -u REFRESH_BOXES -u REFRESH_BOX_REPO "${REPO_ROOT}/tools/pool-refresh.sh" --label g1 --dry-run 2>&1 || true)"
if [[ "${req_out}" == *"REFRESH_BOXES is required"* ]]; then
  ok "REFRESH_BOXES is required"
else
  fail "pool-refresh.sh must require REFRESH_BOXES (got: ${req_out})"
fi

# --- dry run: prints, changes nothing --------------------------------------------
out="$(refresh --label g1 --dry-run)"
if grep -q "+ ssh" <<<"${out}" && [[ ! -s "${PRIV}/refresh/g1/state" ]] && [[ "$(records)" == 0 ]]; then
  ok "dry-run prints the commands and records nothing"
else
  fail "dry-run changed state: $(tail -3 <<<"${out}")"
fi

# --- a crash mid-record resumes without re-recording finished jobs ---------------
touch "${FAKE}/fail-b2-mtp3"
out="$(refresh --label g1)" && fail "a failed box job did not fail the run"
grep -q "mtp3@b2 on b2 failed" <<<"${out}" && ok "a failed box job is named" || fail "no failed-job message: $(tail -3 <<<"${out}")"
before="$(records)"
rm "${FAKE}/fail-b2-mtp3"
out="$(refresh --label g1)" || fail "resume failed: $(tail -5 <<<"${out}")"
after="$(records)"
# 3 serial + 42 oracle jobs total (mtp1..6 and dflash1..15, two boxes each);
# the resume re-runs only the failed one (plus any job its box had not reached yet).
[[ $((after - before)) -lt 42 ]] && ok "resume re-ran $((after - before)) job(s), not the whole record step" \
  || fail "resume re-ran $((after - before)) jobs"
[[ "$(jq length "${PRIV}/refresh/g1/pins.json")" == 176 ]] && ok "176 goldens pinned (8 prompts x serial+mtp1..6+dflash1..15)" || fail "pins.json incomplete"
for b in ${BOXES}; do
  [[ "$(nf "${FAKE}/${b}/home/goldens/g1-*")" == 176 ]] || fail "${b}: g1 not staged"
done
ok "all boxes staged"
grep -c " g1 " "${PRIV}/refresh/staging.log" | grep -qx 528 && ok "staging log records every install (176 x 3 boxes)" || fail "staging log count"
if [[ "$(jq -c '.live_goldens' "${FX}")" == '["g1-1","g1-2","g1-3","g1-4","g1-5","g1-6","g1-7","g1-8"]' ]] \
  && [[ "$(jq -c '[has("live_golden_rotation", "live_golden", "calibration_prompt")]' "${FX}")" == '[false,false,false]' ]] \
  && [[ "$(jq '.timed_prompt_pool | length' "${FX}")" == $((SHIPPED_POOL + 8)) ]] \
  && [[ "$(jq '.speculative_oracles["g1-8"] | length' "${FX}")" == 21 ]] \
  && [[ "$(jq -r '.hidden_correctness_golden.sha256' "${FX}")" == "$(jq -r '.["g1-1.golden.json"].sha256' "${PRIV}/refresh/g1/pins.json")" ]]; then
  ok "pin: live_goldens, hidden golden and oracles point at g1; no rotation keys"
else
  fail "pin: fixture not as expected"
fi
grep -q "git -C .* push origin HEAD:main" <<<"${out}" && [[ "$(git -C "${ROOT}" rev-list --count HEAD)" == 2 ]] \
  && ok "pin prints the commit/push commands and commits nothing" || fail "pin committed or printed nothing"
# every golden agreed across two boxes: the serial of g1-1 came from b1 and b2
grep -q "^record b1 serial .*g1-1" "${FAKE}/calls.log" && grep -q "^record b2 serial .*g1-1" "${FAKE}/calls.log" \
  && ok "each serial golden recorded on two boxes" || fail "serial not recorded twice"

# --- verify ---------------------------------------------------------------------
out="$(refresh --label g1 --step verify)" && fail "verify passed before the push" \
  || { grep -q "push the pin commit first" <<<"${out}" && ok "verify refuses before the pin is on origin/main" || fail "verify: $(tail -2 <<<"${out}")"; }
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "pin g1" && git -C "${ROOT}" push -q origin HEAD:main
echo x > "${FAKE}/b3/home/goldens/stray.golden.json"
out="$(refresh --label g1 --step verify)" && fail "verify passed with a stray golden" \
  || { grep -q "b3 holds unpinned goldens: stray.golden.json" <<<"${out}" && ok "verify refuses an unpinned golden by name" || fail "verify: $(tail -2 <<<"${out}")"; }
rm "${FAKE}/b3/home/goldens/stray.golden.json"
echo x > "${FAKE}/b3/home/goldens/g9-1.golden.json"
out="$(refresh --label g1 --step verify)" && ok "verify passes (a later generation staged ahead is tolerated)" || fail "verify: $(tail -3 <<<"${out}")"
rm "${FAKE}/b3/home/goldens/g9-1.golden.json"

# --- byte-identical check ---------------------------------------------------------
echo b2 > "${FAKE}/mismatch"
out="$(refresh --label g2)" && fail "a mismatching box was accepted"
grep -q "g2-1.golden.json differs between serial@b1" <<<"${out}" && [[ ! -f "${PRIV}/refresh/g2/pins.json" ]] \
  && [[ "$(nf "${FAKE}/b1/home/goldens/g2-*")" == 0 ]] \
  && ok "a byte mismatch between two boxes refuses before anything is pinned or staged" \
  || fail "mismatch: $(tail -3 <<<"${out}")"
rm "${FAKE}/mismatch"

# --- final stays off the boxes -------------------------------------------------------
out="$(refresh --label final)" || fail "final author+record failed: $(tail -3 <<<"${out}")"
[[ -f "${PRIV}/final/pins.json" ]] && [[ "$(nf "${FAKE}/*/home/goldens/final-*")" == 0 ]] \
  && ok "final: authored and recorded, nothing staged" || fail "final touched a box"
out="$(refresh --label final --step stage)" && fail "final was staged"
grep -q "REFUSING to stage the held-out final" <<<"${out}" && ok "final stage is refused" || fail "final stage: ${out}"
out="$(refresh --label final --step pin)" && fail "final was pinned"
grep -q "REFUSING to pin the held-out final" <<<"${out}" && ok "final pin is refused" || fail "final pin: ${out}"
out="$(refresh --label final --step stage --final-after-freeze "${WORK}/nope")" && fail "final staged without a freeze"
touch "${WORK}/FROZEN"
out="$(refresh --label final --step stage --final-after-freeze "${WORK}/FROZEN")" || fail "final stage after freeze: $(tail -3 <<<"${out}")"
grep -q " final b1 final-1.golden.json " "${PRIV}/refresh/staging.log" && ok "final staged after the freeze, and logged" || fail "final not logged"

# --- retire ------------------------------------------------------------------------
out="$(refresh --label g1 --step retire)" && fail "retired the live generation"
grep -q "still the live goldens" <<<"${out}" && ok "retire refuses the live generation" || fail "retire: ${out}"
jq --argjson l "${SHIPPED_LIVE}" '.live_goldens = $l' "${FX}" > "${WORK}/fx" && mv "${WORK}/fx" "${FX}"
out="$(refresh --label g1 --step retire)" || fail "retire unpin: $(tail -3 <<<"${out}")"
[[ "$(jq '.timed_prompt_pool | length' "${FX}")" == "${SHIPPED_POOL}" && "$(jq 'has("speculative_oracles") and (.speculative_oracles | has("g1-1") | not)' "${FX}")" == true ]] \
  && [[ "$(nf "${FAKE}/b1/home/goldens/g1-*")" -gt 0 ]] && ok "retire phase 1 unpins g1 and leaves the boxes alone" || fail "retire phase 1"
out="$(refresh --label g1 --step retire)" && fail "retire removed goldens before the unpin was pushed"
git -C "${ROOT}" -c user.name=t -c user.email=t@t commit -qam "retire g1" && git -C "${ROOT}" push -q origin HEAD:main
out="$(refresh --label g1 --step retire)" || fail "retire phase 2: $(tail -3 <<<"${out}")"
[[ "$(nf "${FAKE}/*/home/goldens/g1-*")" == 0 && "$(nf "${PRIV}/refresh/g1/goldens/*")" == 176 ]] \
  && ok "retire phase 2 removes g1 from every box; the private archive stays" || fail "retire phase 2 left files"

if [[ "${failures}" -eq 0 ]]; then
  echo "test-pool-refresh.sh: all cases passed"
  exit 0
fi
echo "test-pool-refresh.sh: ${failures} case(s) failed" >&2
exit 1
