#!/usr/bin/env bash
# setup.sh -- build the engine and verify the pinned target GGUF.
#
# benchmark.json's setupCommand runs `./tools/fetch-benchd.sh && ./setup.sh`;
# benchmark.yml calls it on the ranked box before the GPU-locked measurement.
#
# Steps:
#   1. tools/engine/build.sh -- the protocol adapter, the organizer-built engine
#      image (bonsai9b-engine:v1), llama-resident, and optional kernels/.
#   2. verify the pinned target GGUF named in fixtures/bonsai2_9b_cuda_v1_track.json
#      (byte count first, then sha256), and, when mtp-head.manifest.json
#      declares spec.mode dflash, the pinned DFlash drafter (dflash_drafter.file)
#      the same way. With MLXFAST_TARGET_GGUF set (a ranked box,
#      organizer-staged) the files must already be there, the drafter beside
#      the target. With it UNSET they live under reference_weights/ in this
#      checkout and setup downloads them from the pinned Hugging Face revision
#      when missing.
#
# Environment:
#   MLXFAST_TARGET_GGUF           absolute path to the one GGUF file. Set:
#                                 verify only. Unset: reference_weights/... under
#                                 this checkout, which setup fills.
#   MLXFAST_TARGET_BASE_URL       where a missing GGUF is fetched on the default
#                                 path (default: the pinned HF repo at its revision,
#                                 or at the file's own target.files revision).
#   HF_TOKEN                      optional Hugging Face token. Set it only when the
#                                 pinned repository is private. setup sends it as a
#                                 bearer header on the default HF path only, on
#                                 curl's stdin; it is not logged and not written.
#                                 Unset: setup reads the first line of the token
#                                 file of this user, ${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
#                                 (huggingface-cli login writes it). No file: no token.
#   MLXFAST_SKIP_ENGINE_BUILD=1   skip step 1 (a pre-built engine is staged)
#   MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1 / SKIP_MODEL_DOWNLOAD=1
#                                 skip step 2 (no scored run is possible)
#   MLXFAST_SKIP_WEIGHTS_SHA256=1 verify byte counts only
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'setup.sh: %s\n' "$*"; }
die() { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }

# HF_TOKEN unset: the token file of this user, one line, trimmed.
HF_TOKEN_FILE="${HF_TOKEN_PATH:-${HOME:-}/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -r "${HF_TOKEN_FILE}" ]]; then
  IFS= read -r HF_TOKEN < "${HF_TOKEN_FILE}" || true
  HF_TOKEN="${HF_TOKEN//[[:space:]]/}"
fi

CONTRACT="${ROOT_DIR}/fixtures/bonsai2_9b_cuda_v1_track.json"
command -v jq >/dev/null 2>&1 || die "jq is required"

# The public model repository the target was pinned from, at that revision
# (the fixture's upstream_model_id / upstream_revision). Its files are flat at
# the repository root and the local copy keeps them flat, under
# reference_weights/<repository name>.
UPSTREAM_MODEL_ID="$(jq -r '.target.upstream_model_id // empty' "${CONTRACT}")"
UPSTREAM_REVISION="$(jq -r '.target.upstream_revision // empty' "${CONTRACT}")"
[[ -n "${UPSTREAM_MODEL_ID}" && -n "${UPSTREAM_REVISION}" ]] \
  || die "the fixture names no target.upstream_model_id / upstream_revision"
DEFAULT_GGUF_DIR="${ROOT_DIR}/reference_weights/${UPSTREAM_MODEL_ID##*/}"

gguf_pin_from_fixture() {
  if jq -e '.target.gguf.path' "${CONTRACT}" >/dev/null 2>&1; then
    jq -r '[.target.gguf.path, (.target.gguf.bytes|tostring), .target.gguf.sha256] | @tsv' "${CONTRACT}"
    return 0
  fi
  local line
  line="$(jq -r '.target.files[] | select(.path | endswith(".gguf")) | [.path, (.bytes|tostring), .sha256] | @tsv' "${CONTRACT}" | head -1)"
  [[ -n "${line}" ]] || die "the fixture pins no target GGUF"
  printf '%s\n' "${line}"
}

source_path_for() {
  printf '%s\n' "$1"
}

resolve_target_source() {
  TARGET_BASE_URL="${MLXFAST_TARGET_BASE_URL:-https://huggingface.co/${UPSTREAM_MODEL_ID}/resolve/${UPSTREAM_REVISION}}"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

file_identity() {
  # size, mtime and inode: what the verified marker records per file.
  stat -c '%s %Y %i' "$1" 2>/dev/null || stat -f '%z %m %i' "$1"
}

download_pinned_file() {
  # download_pinned_file REL WANT_BYTES WANT_SHA -- fetch one pinned file into
  # the snapshot, resumable, and keep it only when both pins match.
  local rel="$1" want_bytes="$2" want_sha="$3"
  local dest="${SNAPSHOT_DIR}/${rel}" partial="${SNAPSHOT_DIR}/${rel}.partial"
  local url got_bytes got_sha rev
  url="${TARGET_BASE_URL}/$(source_path_for "${rel}")"
  # A file with its own "revision" in target.files comes from that revision
  # of the same repository (the upstream replaced it after upstream_revision).
  rev="$(jq -r --arg p "${rel}" 'first((.target.files // [])[] | select(.path == $p) | .revision // empty)' "${CONTRACT}")"
  if [[ -n "${rev}" && -z "${MLXFAST_TARGET_BASE_URL:-}" ]]; then
    url="https://huggingface.co/${UPSTREAM_MODEL_ID}/resolve/${rev}/$(source_path_for "${rel}")"
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to download ${rel}"
  log "downloading ${rel} (${want_bytes} bytes) from ${url}"
  # The token goes on curl's stdin (-K -), not on its argv. curl does not send
  # it to the host that a redirect names.
  if [[ -n "${HF_TOKEN:-}" && -z "${MLXFAST_TARGET_BASE_URL:-}" ]]; then
    printf 'header = "Authorization: Bearer %s"\n' "${HF_TOKEN}" \
      | curl -K - -fL --retry 5 --retry-delay 5 --retry-all-errors -C - -o "${partial}" "${url}"
  else
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors -C - -o "${partial}" "${url}"
  fi || die "download failed for ${rel}; rerun ./setup.sh to resume from ${partial}"
  got_bytes="$(wc -c < "${partial}" | tr -d '[:space:]')"
  if [[ "${got_bytes}" != "${want_bytes}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} is ${got_bytes} bytes, pinned ${want_bytes}; discarded"
  fi
  got_sha="$(sha256_of "${partial}")"
  if [[ "${got_sha}" != "${want_sha}" ]]; then
    rm -f "${partial}"
    die "downloaded ${rel} has sha256 ${got_sha}, pinned ${want_sha}; discarded"
  fi
  mv -f "${partial}" "${dest}"
  log "downloaded and verified ${rel}"
}

gguf_marker_path() {
  local key
  key="$(printf '%s %s %s' "${GGUF_REL}" "${GGUF_BYTES}" "${GGUF_SHA}" | sha256_of /dev/stdin | cut -c 1-16)"
  printf '%s/.verified-gguf-%s\n' "${GGUF_DIR}" "${key}"
}

# --- 1. engine build --------------------------------------------------------
if [[ "${MLXFAST_SKIP_ENGINE_BUILD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_ENGINE_BUILD=1: not building the engine; a pre-built engine must be staged"
  [[ -x "${ROOT_DIR}/.build/release/mlxfast-runtime-worker" ]] \
    || die "no staged engine at .build/release/mlxfast-runtime-worker"
  [[ -x "${ROOT_DIR}/.build/release/llama-resident" ]] \
    || die "no staged llama-resident at .build/release/llama-resident"
else
  log "building the engine (tools/engine/build.sh)"
  "${ROOT_DIR}/tools/engine/build.sh"
fi

# --- 2. target GGUF and DFlash drafter verification -------------------------
# The GGUFs the fixture pins in target.files: the target, then the DFlash
# drafter (dflash_drafter.file), which a dflash leg loads from beside it.
stage_gguf() { # stage_gguf REL BYTES SHA256 LABEL [STAGED_PATH]
  local GGUF_REL="$1" GGUF_BYTES="$2" GGUF_SHA="$3" label="$4" staged="${5:-}"
  local local_gguf GGUF_PATH marker verified_by_marker identity got_bytes got_sha
  [[ -n "${GGUF_REL}" && -n "${GGUF_BYTES}" && -n "${GGUF_SHA}" ]] \
    || die "the contract pins no ${label}"
  if [[ -n "${MLXFAST_TARGET_GGUF:-}" ]]; then
    local_gguf=0
    GGUF_PATH="${staged:-$(dirname -- "${MLXFAST_TARGET_GGUF}")/${GGUF_REL}}"
    [[ -f "${GGUF_PATH}" ]] || die "the ${label} is not staged: ${GGUF_PATH}"
  else
    local_gguf=1
    GGUF_PATH="${GGUF_DIR}/${GGUF_REL}"
    log "MLXFAST_TARGET_GGUF is unset; using ${GGUF_PATH}"
    if [[ ! -f "${GGUF_PATH}" ]] || [[ "$(wc -c < "${GGUF_PATH}" | tr -d '[:space:]')" != "${GGUF_BYTES}" ]]; then
      download_pinned_file "${GGUF_REL}" "${GGUF_BYTES}" "${GGUF_SHA}"
    fi
  fi
  marker=""
  [[ "${local_gguf}" == "1" ]] && marker="$(gguf_marker_path)"
  verified_by_marker=0
  if [[ "${local_gguf}" == "1" && "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && -f "${marker}" ]]; then
    identity="$(printf '%s %s\n' "${GGUF_REL}" "$(file_identity "${GGUF_PATH}")")"
    [[ "${identity}" == "$(cat "${marker}")" ]] && verified_by_marker=1
  fi
  log "verifying the pinned ${label} at ${GGUF_PATH}"
  got_bytes="$(wc -c < "${GGUF_PATH}" | tr -d '[:space:]')"
  [[ "${got_bytes}" == "${GGUF_BYTES}" ]] \
    || die "byte-count mismatch for ${GGUF_REL}: staged ${got_bytes}, pinned ${GGUF_BYTES}"
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" != "1" && "${verified_by_marker}" != "1" ]]; then
    got_sha="$(sha256_of "${GGUF_PATH}")"
    [[ "${got_sha}" == "${GGUF_SHA}" ]] \
      || die "sha256 mismatch for ${GGUF_REL}: staged ${got_sha}, pinned ${GGUF_SHA}"
    if [[ "${local_gguf}" == "1" ]]; then
      printf '%s %s\n' "${GGUF_REL}" "$(file_identity "${GGUF_PATH}")" > "${marker}.tmp.$$"
      mv -f "${marker}.tmp.$$" "${marker}"
      log "wrote the verification marker $(basename "${marker}")"
    fi
  fi
  if [[ "${MLXFAST_SKIP_WEIGHTS_SHA256:-0}" == "1" ]]; then
    log "${label} verified by byte count (sha256 skipped)"
  elif [[ "${verified_by_marker}" == "1" ]]; then
    log "${label} verified by byte count and the marker of the last full sha256 pass"
  else
    log "${label} verified by byte count and sha256"
  fi
  STAGED_GGUF_PATH="${GGUF_PATH}"
}

if [[ "${MLXFAST_SKIP_WEIGHTS_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  log "MLXFAST_SKIP_WEIGHTS_DOWNLOAD/SKIP_MODEL_DOWNLOAD set: not verifying the GGUF (no scored run is possible)"
else
  if [[ -n "${MLXFAST_TARGET_GGUF:-}" ]]; then
    [[ -f "${MLXFAST_TARGET_GGUF}" ]] || die "MLXFAST_TARGET_GGUF is not a file: ${MLXFAST_TARGET_GGUF}"
  else
    GGUF_DIR="${DEFAULT_GGUF_DIR}"
    mkdir -p "${GGUF_DIR}" || die "cannot create ${GGUF_DIR}"
    setup_lock="${GGUF_DIR}/.setup-lock"
    mkdir "${setup_lock}" 2>/dev/null \
      || die "another setup is filling ${GGUF_DIR}, or left ${setup_lock}; remove it when no setup is running"
    trap 'rmdir "${setup_lock}" 2>/dev/null || true' EXIT
    resolve_target_source
    SNAPSHOT_DIR="${GGUF_DIR}"
  fi
  IFS=$'\t' read -r rel bytes sha < <(gguf_pin_from_fixture)
  stage_gguf "${rel}" "${bytes}" "${sha}" "target GGUF" "${MLXFAST_TARGET_GGUF:-}"
  target_path="${STAGED_GGUF_PATH}"
  # The drafter only when the declaration runs it (mtp-head.manifest.json
  # spec.mode dflash): an mtp or serial run never loads it.
  drafter="$(jq -r '.dflash_drafter.file // empty' "${CONTRACT}")"
  if [[ -n "${drafter}" && "$("${ROOT_DIR}/tools/spec-declaration.sh" mode)" == "dflash" ]]; then
    IFS=$'\t' read -r rel bytes sha < <(jq -r --arg f "${drafter}" 'first(.target.files[] | select(.path == $f)) | [.path, (.bytes|tostring), .sha256] | @tsv' "${CONTRACT}")
    stage_gguf "${rel}" "${bytes}" "${sha}" "DFlash drafter"
  fi
  log "target GGUF ready at ${target_path}; export MLXFAST_TARGET_GGUF for serve-up"
  log "local runs: tools/local-baseline.sh"
fi

log "setup complete: engine staged for benchd and the target GGUF verified per the steps above"
