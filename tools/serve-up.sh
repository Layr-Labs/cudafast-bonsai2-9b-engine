#!/usr/bin/env bash
# serve-up.sh -- boot the ONE llama.cpp resident for a benchmark leg.
#
# THREE FORMS. The first is the PER-LEG pair the paired ranked path uses; the
# third is the wrapper, which local drivers use.
#
#   tools/serve-up.sh --boot --spec serial|mtp|dflash --draft-len N --socket-out FILE
#   tools/serve-up.sh --stop --socket PATH
#   tools/serve-up.sh CMD [ARGS...]
#
# --boot boots ONE resident FROM THE TREE THIS SCRIPT LIVES IN, waits for the
# healthy hello, writes the resident's socket path as the FIRST LINE of FILE,
# and EXITS 0 WITH THE RESIDENT STILL RUNNING. --stop tears that resident down
# and is idempotent: a second stop, or a stop of a resident that already died,
# is a no-op that exits 0. A caller uses --stop on success and on failure alike.
#
# WHY THE PAIR EXISTS. The ranked run is PAIRED, and each leg has its own tree:
# the serial control leg runs the REFERENCE tree's engine, the candidate leg
# runs the submission's. benchd runs each leg's serve-up FROM THAT LEG'S
# WORKSPACE, and injects the socket it reads back into that leg's worker spawns
# as DS4_RESIDENT_SOCKET and BENCH_WORKER_RESIDENT_SOCKET (benchd's names).
#
# LEG 1 IS ALWAYS `--spec serial --draft-len 0`. The control leg is serial
# whatever the candidate declares, and in --boot mode the FLAGS are the
# authority: a SERVE_UP_SPECULATIVE in the environment that disagrees is
# refused rather than honoured.
#
# ONE RESIDENT AT A TIME. The memory plan below refuses unless the resident's
# reservation is free RIGHT NOW, so the two legs are sequential: boot leg 1,
# measure it, stop it, and only then boot leg 2.
#
# WEIGHTS LOAD ONCE PER LEG. benchd spawns a fresh `cuda-engine` per phase;
# each connects to this resident (.build/release/llama-resident) and loads
# nothing. The native MTP layer is part of the same GGUF, so an mtp leg loads
# no second file. A dflash leg also loads the pinned DFlash drafter
# (fixture dflash_drafter.file), which the box stages beside the target GGUF.
#
# THE RESIDENT RUNS IN THE PINNED CONTAINER (bonsai9b-engine:v1) with NO
# NETWORK (--network none), the GGUF and the leg's tree mounted READ-ONLY, and
# the staged llama-resident binary from this tree. Its only writable mounts are
# the socket directory and a per-boot run directory holding HOME, so no cache
# survives from one leg into the next.
#
# GPU LOCK -- NOT TAKEN HERE. The CALLER owns the box GPU window
# (.github/workflows/benchmark.yml holds the box-wide flock on
# /tmp/mtplx-gpu-exclusive.lock for the whole measurement) and this script
# boots the resident inside it.
#
# Environment:
#   MLXFAST_TARGET_GGUF       absolute path to the one pinned GGUF (required on
#                             the ranked box). SERVE_UP_GGUF is honoured too.
#   SERVE_UP_WEIGHTS_DIR      legacy alias: when it names a regular file, it is
#                             treated as the GGUF path (tests only).
#   SERVE_UP_SPECULATIVE      0 = serial serve; 1 = speculative serve (SERVE_UP_SPEC_MODE)
#                             at SERVE_UP_SPEC_DRAFT_LEN drafts. DERIVED by the caller
#                             from mtp-head.manifest.json via tools/spec-declaration.sh.
#                             Default 0 -- the stock serial launch reference.
#   SERVE_UP_SPEC_MODE        mtp (default) or dflash when SPECULATIVE=1, from
#                             tools/spec-declaration.sh mode.
#   SERVE_UP_SPEC_DRAFT_LEN   declared num_speculative_tokens (default 1 when SPECULATIVE=1)
#   SERVE_UP_LOG_DIR          where the run's identity, ready file and resident
#                             log are kept (default <repo>/.build/resident)
#   SERVE_UP_SOCKET_DIR       where the resident's Unix socket is bound (default
#                             ${TMPDIR:-/tmp}/bonsai9b-serve-up-<uid>: the
#                             container's own /tmp hides a bind of /tmp itself,
#                             not of a directory in it). SEPARATE from the log dir because a
#                             Unix socket path holds only 107 bytes on Linux and
#                             the ranked job's workspace path is already long.
#   SERVE_UP_HEALTH_TIMEOUT_S ceiling on the load + first healthy hello (default 3600)
#   SERVE_UP_MEMINFO          where memory is read from (default /proc/meminfo)
#   SERVE_UP_LAUNCHER         TEST SEAM: `docker` (default, the only ranked value)
#                             or `host`, which runs the wire-resident test shim
#                             when RESIDENT_BACKEND=fake (off-box tests only).
#                             tools/ranked-box-preflight.sh refuses it set.
#   SERVE_UP_CONTAINER_USER   the SANDBOX user (docs/sandbox-design.md 6a: bonsaisub).
#   SERVE_UP_SANDBOX_TREE     the ranked workflow's form of the above: the
#                             candidate checkout. Only the tree at this path
#                             takes SANDBOX_USER as SERVE_UP_CONTAINER_USER.
#   SHARE_GROUP               its share group (default bonsaishare; 6b).
#   SERVE_UP_CONTAINER_PREFIX the resident container name is this prefix and the
#                             run tag (default serve-up-, so serve-up-<pid>-<time>).
#                             A second runner on one box sets its own prefix, so
#                             a cleanup by name prefix never removes the other
#                             runner's residents (docs/sandbox-design.md 6b).
#                             Set by the workflow for the CANDIDATE leg only; unset,
#                             nothing below changes. Set, it needs SANDBOX_ROOT and
#                             SANDBOX_JOB and the docker launcher; the container runs
#                             as that user's uid and SHARE_GROUP's gid, and its ONLY
#                             writable mount is a fresh per-leg directory
#                             $SANDBOX_ROOT/$SANDBOX_JOB/sock/<run tag>/ (2770) holding
#                             the socket, HOME and the caches. The resident log and
#                             the identity file stay in SERVE_UP_LOG_DIR, which that
#                             user cannot write. No ready file and no .pid sidecar are
#                             written: the only place the sandbox user could see them
#                             is one it can also rewrite. The health probe runs as
#                             that user (the resident chmods its socket 0600), and
#                             --stop stops the container BY NAME, never by a pid.
#                             The container also gets --security-opt
#                             no-new-privileges and --cap-drop ALL
#                             (SERVE_UP_ALLOW_NEW_PRIVS=1 / SERVE_UP_NO_CAP_DROP=1
#                             turn each off if a GPU runtime needs it). On --stop,
#                             after the container is gone, every remaining process
#                             of that user is killed and tools/sandbox-wipe.sh runs
#                             (SANDBOX_USER is SERVE_UP_CONTAINER_USER).
#
# SANDBOXED SOCKETS ARE RECOGNISED BY THEIR SHAPE, <run tag>/llamacpp-resident.sock,
# so --stop never reads a sidecar beside one, even when it runs without the
# variable. The unsandboxed shape is llamacpp-resident.<run tag>.sock.
#
# Everything the ENGINE is configured with -- context, batch shape, memory
# fraction, offline flags -- comes from the fixture's
# serve_configuration.engine_env, never from this script's environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
log() { printf 'serve-up.sh: %s\n' "$*" >&2; }
die() { printf 'serve-up.sh: %s\n' "$*" >&2; exit 1; }
CONTAINER_PREFIX="${SERVE_UP_CONTAINER_PREFIX:-serve-up-}"

# THE SANDBOXED TREE IS NAMED, NOT INHERITED. benchd boots BOTH legs of a paired
# run with one environment. Only the CANDIDATE's resident runs as the sandbox
# user: the serial-control leg's worker is benchd's own and runs as the runner,
# so it cannot open a sandboxed resident's 0600 socket. The ranked workflow names
# the candidate checkout in SERVE_UP_SANDBOX_TREE. The tree at that path takes
# SANDBOX_USER as SERVE_UP_CONTAINER_USER; every other tree (the reference tree)
# runs its resident as the runner, as calibrate-box.sh measures it.
if [[ -z "${SERVE_UP_CONTAINER_USER:-}" && -n "${SERVE_UP_SANDBOX_TREE:-}" ]]; then
  sandbox_tree="$(cd -- "${SERVE_UP_SANDBOX_TREE}" >/dev/null 2>&1 && pwd -P)" \
    || die "SERVE_UP_SANDBOX_TREE '${SERVE_UP_SANDBOX_TREE}' is not a directory"
  if [[ "${sandbox_tree}" == "${SCRIPT_DIR}" ]]; then
    [[ -n "${SANDBOX_USER:-}" ]] || die "SERVE_UP_SANDBOX_TREE names this tree, but SANDBOX_USER is unset"
    SERVE_UP_CONTAINER_USER="${SANDBOX_USER}"
  fi
fi

# After a sandboxed candidate leg's container is stopped: kill stragglers, wipe
# shared dirs, then wait for memory (docs/sandbox-design.md).
sandbox_leg_teardown_wipe() {
  local user="$1"
  local left=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sudo -n -u "${user}" -- kill -KILL -1 >/dev/null 2>&1 || true
    left="$(pgrep -u "${user}" 2>/dev/null | tr '\n' ' ' || true)"
    [[ -z "${left// /}" ]] && break
    sleep 0.5
  done
  if [[ -n "${left// /}" ]]; then
    die "${user} processes survived kill -KILL -1: ${left}"
  fi
  if ! SANDBOX_USER="${user}" "${SCRIPT_DIR}/tools/sandbox-wipe.sh"; then
    die "sandbox wipe after candidate-leg --stop failed"
  fi
  if ! bash "${SCRIPT_DIR}/tools/wait-for-memory.sh"; then
    die "wait-for-memory after candidate-leg --stop failed"
  fi
}

sandbox_stop_socket_is_sandboxed() {
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || return 1
  # Prefix match on job-scoped SANDBOX_ROOT/SANDBOX_JOB from the trusted workflow; not path canonicalization.
  [[ "${STOP_SOCKET}" == "${SANDBOX_ROOT%/}/${SANDBOX_JOB}/sock/"* ]]
}

sandbox_stop_wipe_user() {
  if [[ -n "${SERVE_UP_CONTAINER_USER:-}" ]]; then
    printf '%s' "${SERVE_UP_CONTAINER_USER}"
    return 0
  fi
  if [[ -n "${SANDBOX_USER:-}" ]]; then
    printf '%s' "${SANDBOX_USER}"
    return 0
  fi
  die "sandbox --stop wipe requires SERVE_UP_CONTAINER_USER or SANDBOX_USER (refusing to guess)"
}

usage() {
  die "usage: MLXFAST_TARGET_GGUF=<file> tools/serve-up.sh --boot --spec serial|mtp|dflash --draft-len N --socket-out FILE
       tools/serve-up.sh --stop --socket PATH
       MLXFAST_TARGET_GGUF=<file> tools/serve-up.sh CMD [ARGS...]"
}

[[ $# -ge 1 ]] || usage

MODE="wrapper"
BOOT_SPEC=""
BOOT_DRAFT_LEN=""
BOOT_SOCKET_OUT=""
STOP_SOCKET=""

# A FLAG WITH NO VALUE IS REFUSED BY NAME. `--spec` at the end of the argv used
# to take "${2:-}" -- the empty string -- and then `shift 2` ran off the end,
# which under `set -u` ended the script with a bare exit 1 and no message. A
# caller then saw a failed boot and nothing saying why. The value is required
# here, before the shift, so the refusal names the flag that is missing one.
need_value() { # need_value FLAG VALUE_OR_EMPTY
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

case "$1" in
  --boot)
    MODE="boot"; shift
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --spec)       need_value "$@"; BOOT_SPEC="$2"; shift 2 ;;
        --draft-len)  need_value "$@"; BOOT_DRAFT_LEN="$2"; shift 2 ;;
        --socket-out) need_value "$@"; BOOT_SOCKET_OUT="$2"; shift 2 ;;
        *) die "--boot: unknown argument '$1'" ;;
      esac
    done
    ;;
  --stop)
    MODE="stop"; shift
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --socket) need_value "$@"; STOP_SOCKET="$2"; shift 2 ;;
        *) die "--stop: unknown argument '$1'" ;;
      esac
    done
    ;;
esac

# --- --stop: idempotent teardown of one booted resident ----------------------
# It needs NOTHING the boot needed -- no weights, no engine header, no memory
# plan. Everything it must know is beside the socket, so it runs before any of
# that is resolved and cannot fail for a reason unrelated to stopping.
#
# THE PID AND THE READY FILE COME FROM THE SIDECAR the boot wrote
# (<socket>.pid: the pid on line 1, the ready file on line 2), never from a
# guess about what is listening on the socket. A stop that had to infer its
# target could kill the wrong process on a shared box.
if [[ "${MODE}" == "stop" ]]; then
  [[ -n "${STOP_SOCKET}" ]] || die "--stop requires --socket PATH"
  # A SANDBOXED LEG: everything beside its socket is writable by the sandbox
  # user, so nothing there is read. The container name comes from the run tag
  # in the path, checked against the shape --boot writes, and docker stops it:
  # SIGTERM, then SIGKILL after 10s, the same budget as the pid path below.
  if [[ -n "${SERVE_UP_CONTAINER_USER:-}" || "$(basename -- "${STOP_SOCKET}")" == "llamacpp-resident.sock" ]]; then
    stop_tag="$(basename -- "$(dirname -- "${STOP_SOCKET}")")"
    stop_container=""
    [[ "$(basename -- "${STOP_SOCKET}")" != "llamacpp-resident.sock" || ! "${stop_tag}" =~ ^[0-9]+-[0-9]{8}T[0-9]{6}Z$ ]] \
      || stop_container="${CONTAINER_PREFIX}${stop_tag}"
    if [[ -n "${stop_container}" ]] && docker stop -t 10 "${stop_container}" >/dev/null 2>&1; then
      log "resident container ${stop_container} stopped"
    else
      log "no running resident for ${STOP_SOCKET}; nothing to stop"
    fi
    [[ -z "${stop_container}" ]] || docker rm -f "${stop_container}" >/dev/null 2>&1 || true
    rm -f "${STOP_SOCKET}" 2>/dev/null || true
    if [[ -n "${SERVE_UP_CONTAINER_USER:-}" ]] || sandbox_stop_socket_is_sandboxed; then
      sandbox_leg_teardown_wipe "$(sandbox_stop_wipe_user)"
    fi
    exit 0
  fi
  stop_pidfile="${STOP_SOCKET}.pid"
  stop_pid=""
  stop_ready=""
  stop_container=""
  if [[ -f "${stop_pidfile}" ]]; then
    stop_pid="$(sed -n 1p "${stop_pidfile}" | tr -d '[:space:]')"
    stop_ready="$(sed -n 2p "${stop_pidfile}")"
    stop_container="$(sed -n 3p "${stop_pidfile}" | tr -d '[:space:]')"
  fi
  if [[ -n "${stop_pid}" ]] && kill -0 "${stop_pid}" 2>/dev/null; then
    kill -TERM "${stop_pid}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${stop_pid}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${stop_pid}" 2>/dev/null; then
      log "resident pid ${stop_pid} did not stop on SIGTERM; killing it"
      kill -KILL "${stop_pid}" 2>/dev/null || true
    fi
    log "resident pid ${stop_pid} stopped"
  else
    # IDEMPOTENT BY DESIGN. A second --stop, or a --stop of a resident that
    # already exited, is a no-op and exits 0: the caller runs it on success and
    # on failure alike, and a teardown that refused the second call would turn
    # a clean failure path into a second failure.
    log "no running resident for ${STOP_SOCKET}; nothing to stop"
  fi
  # The container is named in the sidecar; remove it even if its docker client
  # is already gone, so a killed client can never leave the engine holding the
  # GPU. The name is ours (<prefix><pid>-<time>), never a guess.
  if [[ "${stop_container}" =~ ^"${CONTAINER_PREFIX}"[0-9]+-[0-9TZ]+$ ]] && command -v docker >/dev/null 2>&1; then
    docker rm -f "${stop_container}" >/dev/null 2>&1 || true
  fi
  rm -f "${STOP_SOCKET}" "${stop_pidfile}" 2>/dev/null || true
  [[ -z "${stop_ready}" ]] || rm -f "${stop_ready}" 2>/dev/null || true
  exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 is required (it speaks the resident's health probe)"
command -v jq >/dev/null 2>&1 || die "jq is required (it reads the track fixture)"

# --- --boot: the flags are the authority over the spec -----------------------
if [[ "${MODE}" == "boot" ]]; then
  [[ -n "${BOOT_SOCKET_OUT}" ]] || die "--boot requires --socket-out FILE"
  case "${BOOT_SPEC}" in
    serial)
      # A serial boot with a draft length is a contradiction, not a default to
      # silently ignore: the caller believes it asked for a drafter.
      [[ -z "${BOOT_DRAFT_LEN}" || "${BOOT_DRAFT_LEN}" == "0" ]]         || die "--boot --spec serial takes --draft-len 0 (got '${BOOT_DRAFT_LEN}'); the serial control leg arms no drafter"
      boot_speculative=0
      ;;
    mtp|dflash)
      [[ "${BOOT_DRAFT_LEN}" =~ ^[1-9][0-9]*$ ]]         || die "--boot --spec ${BOOT_SPEC} requires --draft-len N with N >= 1 (got '${BOOT_DRAFT_LEN}')"
      boot_speculative=1
      ;;
    "") die "--boot requires --spec serial|mtp|dflash" ;;
    *)  die "--boot --spec must be serial, mtp or dflash (got '${BOOT_SPEC}')" ;;
  esac
  # THE ENVIRONMENT MAY NOT DISAGREE WITH THE FLAG. On the ranked path leg 1 is
  # serial whatever the candidate declares, so an inherited SERVE_UP_SPECULATIVE
  # that says otherwise is a refusal, never an override.
  if [[ -n "${SERVE_UP_SPECULATIVE:-}" && "${SERVE_UP_SPECULATIVE}" != "${boot_speculative}" ]]; then
    die "--boot --spec ${BOOT_SPEC} derives SERVE_UP_SPECULATIVE=${boot_speculative} but the environment carries ${SERVE_UP_SPECULATIVE}; in --boot mode the flag is the authority and a disagreeing environment is refused"
  fi
  if [[ "${boot_speculative}" == "1" && -n "${SERVE_UP_SPEC_MODE:-}" && "${SERVE_UP_SPEC_MODE}" != "${BOOT_SPEC}" ]]; then
    die "--boot --spec ${BOOT_SPEC} but the environment carries SERVE_UP_SPEC_MODE=${SERVE_UP_SPEC_MODE}; in --boot mode the flag is the authority and a disagreeing environment is refused"
  fi
  SERVE_UP_SPECULATIVE="${boot_speculative}"
  if [[ "${boot_speculative}" == "1" ]]; then
    SERVE_UP_SPEC_MODE="${BOOT_SPEC}"
    SERVE_UP_SPEC_DRAFT_LEN="${BOOT_DRAFT_LEN}"
  fi
fi
TRACK_FIXTURE="${SCRIPT_DIR}/fixtures/bonsai2_9b_cuda_v1_track.json"
[[ -f "${TRACK_FIXTURE}" ]] || die "the track fixture is missing: ${TRACK_FIXTURE}"

SERVE_UP_GGUF="${MLXFAST_TARGET_GGUF:-${SERVE_UP_GGUF:-}}"
if [[ -z "${SERVE_UP_GGUF}" && -n "${SERVE_UP_WEIGHTS_DIR:-}" && -f "${SERVE_UP_WEIGHTS_DIR}" ]]; then
  SERVE_UP_GGUF="${SERVE_UP_WEIGHTS_DIR}"
fi
[[ -n "${SERVE_UP_GGUF}" ]] || die "set MLXFAST_TARGET_GGUF (or SERVE_UP_GGUF) to the pinned target GGUF"
[[ -f "${SERVE_UP_GGUF}" ]] || die "target GGUF is missing: '${SERVE_UP_GGUF}' (run ./setup.sh)"
SERVE_UP_GGUF="$(readlink -f "${SERVE_UP_GGUF}")"

SERVE_UP_SPECULATIVE="${SERVE_UP_SPECULATIVE:-0}"
case "${SERVE_UP_SPECULATIVE}" in 0|1) ;; *) die "SERVE_UP_SPECULATIVE must be 0 or 1 (got '${SERVE_UP_SPECULATIVE}')" ;; esac
DRAFT_GGUF=""
if [[ "${SERVE_UP_SPECULATIVE}" == "1" ]]; then
  SERVE_UP_SPEC_MODE="${SERVE_UP_SPEC_MODE:-mtp}"
  case "${SERVE_UP_SPEC_MODE}" in
    mtp)    envelope_key="mtp_head" ;;
    dflash) envelope_key="dflash_drafter" ;;
    *) die "SERVE_UP_SPEC_MODE must be mtp or dflash (got '${SERVE_UP_SPEC_MODE}')" ;;
  esac
  SERVE_UP_SPEC_DRAFT_LEN="${SERVE_UP_SPEC_DRAFT_LEN:-1}"
  # The fixture's permitted_draft_depths for this mode is the envelope.
  [[ "${SERVE_UP_SPEC_DRAFT_LEN}" =~ ^[1-9][0-9]*$ ]] \
    && jq -e --arg k "${envelope_key}" --argjson n "${SERVE_UP_SPEC_DRAFT_LEN}" \
         '.[$k].permitted_draft_depths | index($n) != null' "${TRACK_FIXTURE}" >/dev/null \
    || die "SERVE_UP_SPEC_DRAFT_LEN=${SERVE_UP_SPEC_DRAFT_LEN} is not in ${envelope_key}.permitted_draft_depths ($(jq -c --arg k "${envelope_key}" '.[$k].permitted_draft_depths' "${TRACK_FIXTURE}"))"
  SPEC_DEPTH="${SERVE_UP_SPEC_DRAFT_LEN}"
  SPEC_LABEL="${SERVE_UP_SPEC_MODE}${SPEC_DEPTH}"
  if [[ "${SERVE_UP_SPEC_MODE}" == "dflash" ]]; then
    # The drafter is a separate pinned file that the box stages beside the target GGUF.
    draft_name="$(jq -r '.dflash_drafter.file // empty' "${TRACK_FIXTURE}")"
    [[ -n "${draft_name}" ]] || die "the fixture names no dflash_drafter.file"
    DRAFT_GGUF="$(dirname -- "${SERVE_UP_GGUF}")/${draft_name}"
    [[ -f "${DRAFT_GGUF}" ]] || die "the DFlash drafter is missing: '${DRAFT_GGUF}' (run ./setup.sh)"
    DRAFT_GGUF="$(readlink -f "${DRAFT_GGUF}")"
  fi
else
  SERVE_UP_SPEC_MODE="serial"
  SPEC_DEPTH=0
  SPEC_LABEL="serial"
fi

SERVE_UP_LOG_DIR="${SERVE_UP_LOG_DIR:-${SCRIPT_DIR}/.build/resident}"
SERVE_UP_SOCKET_DIR="${SERVE_UP_SOCKET_DIR:-${TMPDIR:-/tmp}/bonsai9b-serve-up-$(id -u)}"
SERVE_UP_HEALTH_TIMEOUT_S="${SERVE_UP_HEALTH_TIMEOUT_S:-3600}"
SERVE_UP_MEMINFO="${SERVE_UP_MEMINFO:-/proc/meminfo}"
SERVE_UP_LAUNCHER="${SERVE_UP_LAUNCHER:-docker}"
case "${SERVE_UP_LAUNCHER}" in docker|host) ;; *) die "SERVE_UP_LAUNCHER must be docker or host (got '${SERVE_UP_LAUNCHER}')" ;; esac
SANDBOX_USER_ID=""
if [[ -n "${SERVE_UP_CONTAINER_USER:-}" ]]; then
  [[ "${SERVE_UP_LAUNCHER}" == "docker" ]] || die "SERVE_UP_CONTAINER_USER runs the resident in the container; it cannot be combined with SERVE_UP_LAUNCHER=${SERVE_UP_LAUNCHER}"
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] || die "SERVE_UP_CONTAINER_USER needs SANDBOX_ROOT and SANDBOX_JOB (the per-job socket directory)"
  # The per-job socket directory is tools/engine/build.sh's (2750, group SHARE_GROUP); a
  # missing one is refused rather than created here with the wrong owner.
  SERVE_UP_SOCKET_DIR="${SANDBOX_ROOT}/${SANDBOX_JOB}/sock"
  [[ -d "${SERVE_UP_SOCKET_DIR}" ]] || die "the per-job socket directory ${SERVE_UP_SOCKET_DIR} is missing (tools/engine/build.sh creates it)"
  sandbox_uid="$(id -u "${SERVE_UP_CONTAINER_USER}")" || die "no such user: ${SERVE_UP_CONTAINER_USER}"
  sandbox_gid="$(getent group "${SHARE_GROUP:-bonsaishare}" | cut -d: -f3)"
  [[ -n "${sandbox_gid}" ]] || die "no ${SHARE_GROUP:-bonsaishare} group on this box (tools/box-runner/setup-sandbox-user.sh creates it)"
  SANDBOX_USER_ID="${sandbox_uid}:${sandbox_gid}"
fi
mkdir -p "${SERVE_UP_LOG_DIR}" "${SERVE_UP_SOCKET_DIR}"
SERVE_UP_LOG_DIR="$(cd -- "${SERVE_UP_LOG_DIR}" && pwd -P)"
SERVE_UP_SOCKET_DIR="$(cd -- "${SERVE_UP_SOCKET_DIR}" && pwd -P)"
# The container mounts its own --tmpfs /tmp, which hides a bind of the host's
# /tmp: the socket would never appear here and the health wait would run out.
[[ "${SERVE_UP_LAUNCHER}" != "docker" || "${SERVE_UP_SOCKET_DIR}" != "/tmp" ]] \
  || die "SERVE_UP_SOCKET_DIR (or TMPDIR) resolves to /tmp, which the container's own /tmp hides; set SERVE_UP_SOCKET_DIR to another short directory"

# --- the fixture is the authority for the engine configuration ---------------
ENGINE_PIN="$(jq -r '.serve_configuration.engine_pin // empty' "${TRACK_FIXTURE}")"
BASE_DIGEST="$(jq -r '.serve_configuration.base_image_digest // empty' "${TRACK_FIXTURE}")"
[[ "${BASE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "the fixture pins no serve_configuration.base_image_digest (the nvidia/cuda devel base)"
dockerfile_base="$(sed -n 's/^FROM nvidia\/cuda@\(sha256:[0-9a-f]\{64\}\).*/\1/p' "${SCRIPT_DIR}/tools/engine/Dockerfile" | head -1)"
[[ "${dockerfile_base}" == "${BASE_DIGEST}" ]] \
  || die "tools/engine/Dockerfile base (${dockerfile_base:-absent}) disagrees with fixture base_image_digest ${BASE_DIGEST}"
# shellcheck source=tools/engine/read-engine-image-id.sh
. "${SCRIPT_DIR}/tools/engine/read-engine-image-id.sh"
ENGINE_IMAGE_ID=""
# engine_env: KEY=VALUE lines, exported to the resident verbatim.
mapfile -t ENGINE_ENV < <(jq -r '.serve_configuration.engine_env | to_entries[] | "\(.key)=\(.value)"' "${TRACK_FIXTURE}")
[[ ${#ENGINE_ENV[@]} -gt 0 ]] || die "the fixture declares no serve_configuration.engine_env"
engine_env_value() { # engine_env_value KEY
  local kv
  for kv in "${ENGINE_ENV[@]}"; do [[ "${kv%%=*}" == "$1" ]] && { printf '%s\n' "${kv#*=}"; return 0; }; done
  die "the fixture's engine_env has no $1"
}
CTX_SIZE="$(engine_env_value RESIDENT_CTX_SIZE)"
KV_BYTES="$(engine_env_value RESIDENT_KV_CACHE_MEMORY_BYTES)"

RESIDENT_BIN="${SCRIPT_DIR}/.build/release/llama-resident"
FAKE_RESIDENT="${SCRIPT_DIR}/tools/testlib/wire_resident/resident.py"
if [[ "${SERVE_UP_LAUNCHER}" == "docker" ]]; then
  command -v docker >/dev/null 2>&1 || die "docker is required to run the pinned engine container"
  # A ranked job (RUNNER_NAME set) must say which tree is sandboxed: the
  # candidate's container user, or SERVE_UP_SANDBOX_TREE naming the candidate
  # checkout. The reference tree's serve in that job runs as the runner by
  # design (the control leg's worker must open its socket), so a named
  # SERVE_UP_SANDBOX_TREE that is another tree is not a refusal.
  [[ -n "${RUNNER_NAME:-}" && -z "${SERVE_UP_CONTAINER_USER:-}" && -z "${SERVE_UP_SANDBOX_TREE:-}" ]] \
    && die "ranked docker serve requires SERVE_UP_CONTAINER_USER or SERVE_UP_SANDBOX_TREE (refusing an unsandboxed candidate)"
  [[ -x "${RESIDENT_BIN}" ]] || die "no staged llama-resident at ${RESIDENT_BIN} (run ./setup.sh)"
  ENGINE_IMAGE_ID="$(read_engine_image_id)" \
    || die "no engine image Id on this box (run ./setup.sh; on a ranked box the organizer writes /srv/bonsai9b/engine-image.id)"
  verify_engine_image_id_present "${ENGINE_IMAGE_ID}" \
    || die "the staged engine image ${ENGINE_IMAGE_ID} is not present (run ./setup.sh)"
elif [[ "${RESIDENT_BACKEND:-}" == "fake" ]]; then
  [[ -f "${FAKE_RESIDENT}" ]] || die "no test wire resident at ${FAKE_RESIDENT}"
else
  [[ -x "${RESIDENT_BIN}" ]] || die "no llama-resident at ${RESIDENT_BIN} (run ./setup.sh)"
fi

RUN_TAG="$$-$(date -u +%Y%m%dT%H%M%SZ)"
CONTAINER_NAME="${CONTAINER_PREFIX}${RUN_TAG}"
SOCKET_PATH="${SERVE_UP_SOCKET_DIR}/llamacpp-resident.${RUN_TAG}.sock"
READY_FILE="${SERVE_UP_LOG_DIR}/llamacpp-resident.${RUN_TAG}.ready"
# A fresh run directory per boot: HOME and every compile cache live here, so a
# leg never inherits another leg's torch.compile / Triton / vLLM cache.
RUN_DIR="${SERVE_UP_LOG_DIR}/run.${RUN_TAG}"
if [[ -n "${SANDBOX_USER_ID}" ]]; then
  SOCKET_LEG_DIR="${SERVE_UP_SOCKET_DIR}/${RUN_TAG}"
  SOCKET_PATH="${SOCKET_LEG_DIR}/llamacpp-resident.sock"
  READY_FILE=""
  RUN_DIR="${SOCKET_LEG_DIR}/run"
fi
# A Unix socket path holds at most 107 bytes on Linux; refuse here, by name,
# rather than let the resident die at bind() after the memory plan.
(( ${#SOCKET_PATH} <= 107 )) \
  || die "the resident socket path is ${#SOCKET_PATH} bytes, over the 107-byte AF_UNIX limit: ${SOCKET_PATH} (set SERVE_UP_SOCKET_DIR to a shorter directory)"
RESIDENT_LOG="${SERVE_UP_LOG_DIR}/llamacpp-resident.${RUN_TAG}.log"
IDENTITY_FILE="${SERVE_UP_LOG_DIR}/serve-identity.json"
if [[ -n "${SANDBOX_USER_ID}" ]]; then
  # A plain mkdir, not -p: a directory the sandbox user planted at this name
  # first is refused, never adopted.
  mkdir -m 2770 "${SOCKET_LEG_DIR}" || die "cannot create the per-leg socket directory ${SOCKET_LEG_DIR}"
  mkdir -m 2770 "${RUN_DIR}" "${RUN_DIR}/home" "${RUN_DIR}/cache"
else
  mkdir -p "${RUN_DIR}/home" "${RUN_DIR}/cache"
fi

driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
pin_tag="${ENGINE_PIN:0:12}"
[[ -n "${pin_tag}" ]] || pin_tag="llama.cpp"
img_tag="${ENGINE_IMAGE_ID:7:12}"
[[ -n "${img_tag}" ]] || img_tag="unknown"
ENGINE_IDENT="llama.cpp@${pin_tag} image=${img_tag} driver=${driver_ver:-unknown} ${SPEC_LABEL}"

# --- the health probe: a real hello over the real socket ---------------------
# A sandboxed resident's socket is 0600 and owned by the sandbox user, so only
# that user can speak to it: the probe drops to it (argv, never a shell string).
probe_resident() {
  local run=(python3)
  [[ -z "${SANDBOX_USER_ID}" ]] || run=(sudo -n -u "${SERVE_UP_CONTAINER_USER}" -- python3)
  "${run[@]}" - "$1" <<'PY'
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sys.argv[1])
    s.sendall(b'{"op":"hello"}\n')
    line = b""
    while not line.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        line += chunk
    s.sendall(b'{"op":"bye"}\n')
    s.recv(65536)  # the bye's acknowledgement, so the resident logs a clean close
    s.close()
    reply = json.loads(line.decode())
except Exception as err:                       # noqa: BLE001 - any failure is unhealthy
    print(f"probe failed: {err}", file=sys.stderr)
    sys.exit(1)
if reply.get("ok") is not True:
    print(f"probe refused: {reply}", file=sys.stderr)
    sys.exit(1)
print(json.dumps({k: reply.get(k) for k in
                  ("vocab_size", "eos_token", "mtp_armed", "spec_mode", "draft_tokens", "ctx_size",
                   "load_epoch", "ident")}))
PY
}

# --- the memory plan: refuse BEFORE the load, never during it ----------------
# GB10 memory is unified. llama.cpp has no memory fraction: require the GGUF
# (and the DFlash drafter on a dflash leg), the fixture's KV budget for ctx,
# 4 GiB headroom and 8 GiB host overhead to fit in MemAvailable right now.
gib() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }
HOST_OVERHEAD_GIB=8
MIN_HEADROOM_GIB=4

plan_memory() {
  local body_bytes avail_kb avail_bytes kv_bytes want_bytes
  body_bytes="$(stat -c '%s' "${SERVE_UP_GGUF}")"
  (( body_bytes > 0 )) || die "memory plan REFUSES the load: empty GGUF ${SERVE_UP_GGUF}"
  [[ -z "${DRAFT_GGUF}" ]] || body_bytes=$(( body_bytes + $(stat -c '%s' "${DRAFT_GGUF}") ))
  [[ -r "${SERVE_UP_MEMINFO}" ]] || die "cannot read ${SERVE_UP_MEMINFO}; refusing to start a load blind"
  avail_kb="$(awk '/^MemAvailable:/ {print $2; exit}' "${SERVE_UP_MEMINFO}")"
  [[ -n "${avail_kb}" ]] || die "${SERVE_UP_MEMINFO} lacks MemAvailable; refusing to start a load blind"
  avail_bytes=$(( avail_kb * 1024 ))
  kv_bytes="${KV_BYTES}"
  want_bytes=$(( body_bytes + kv_bytes + (MIN_HEADROOM_GIB + HOST_OVERHEAD_GIB) * 1073741824 ))
  {
    printf 'serve-up.sh: memory plan for this leg (GiB)\n'
    printf '  %-26s %9s\n' "weights (gguf + drafter)" "$(gib "${body_bytes}")"
    printf '  %-26s %9s   fixture engine_env\n' "KV cache budget" "$(gib "${kv_bytes}")"
    printf '  %-26s %9s\n' "runtime headroom" "${MIN_HEADROOM_GIB}.00"
    printf '  %-26s %9s\n' "host process overhead" "${HOST_OVERHEAD_GIB}.00"
    printf '  %-26s %9s\n' "required free" "$(gib "${want_bytes}")"
    printf '  %-26s %9s   %s\n' "available" "$(gib "${avail_bytes}")" "${SERVE_UP_MEMINFO}"
  } >&2
  (( body_bytes + kv_bytes + MIN_HEADROOM_GIB * 1073741824 <= avail_bytes - HOST_OVERHEAD_GIB * 1073741824 )) \
    || die "memory plan REFUSES the load: the weights ($(gib "${body_bytes}") GiB) plus KV ($(gib "${kv_bytes}") GiB) and ${MIN_HEADROOM_GIB} GiB headroom do not fit after host overhead. Nothing has been loaded."
  (( avail_bytes >= want_bytes )) \
    || die "memory plan REFUSES the load: it needs $(gib "${want_bytes}") GiB free and ${SERVE_UP_MEMINFO} reports $(gib "${avail_bytes}") GiB. Nothing has been loaded."
  log "memory plan ACCEPTS the load: $(gib "${want_bytes}") GiB required, $(gib "${avail_bytes}") GiB available"
}

# --- teardown ----------------------------------------------------------------
RESIDENT_PID=""
KEEP_RESIDENT=0
teardown() {
  if [[ "${KEEP_RESIDENT}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${RESIDENT_PID}" ]] && kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    kill -TERM "${RESIDENT_PID}" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "${RESIDENT_PID}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${RESIDENT_PID}" 2>/dev/null; then
      log "resident did not stop on SIGTERM; killing pid ${RESIDENT_PID}"
      kill -KILL "${RESIDENT_PID}" 2>/dev/null || true
    fi
    log "resident torn down; its log is ${RESIDENT_LOG}"
  fi
  if [[ "${SERVE_UP_LAUNCHER}" == "docker" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  rm -f "${SOCKET_PATH}" "${READY_FILE}" "${SOCKET_PATH}.pid" 2>/dev/null || true
}
trap teardown EXIT
trap 'exit 130' INT TERM

# --- boot the one resident ---------------------------------------------------
plan_memory
rm -f "${SOCKET_PATH}" "${READY_FILE}" 2>/dev/null || true
# The resident log is runner's alone (0600, created before anything writes it):
# a candidate resident's log can carry the hidden prompts, and the log dir
# outlives the job. The identity file is 0640. Neither is ever other-readable.
( umask 077 && : > "${RESIDENT_LOG}" ) || die "cannot create the resident log ${RESIDENT_LOG}"
( umask 027 && : >> "${IDENTITY_FILE}" && chmod 0640 "${IDENTITY_FILE}" ) || die "cannot create ${IDENTITY_FILE}"
RESIDENT_SPEC_ARG="${SERVE_UP_SPEC_MODE}"
RESIDENT_DRAFT_LEN="${SPEC_DEPTH}"
DRAFT_ARGS=()
DRAFT_MOUNT=()
if [[ -n "${DRAFT_GGUF}" ]]; then
  DRAFT_ARGS=(--draft-model "${DRAFT_GGUF}")
  DRAFT_MOUNT=(--mount "type=bind,src=${DRAFT_GGUF},dst=${DRAFT_GGUF},readonly")
fi
RESIDENT_ENV=(
  "RESIDENT_SOCKET=${SOCKET_PATH}"
  "RESIDENT_READY_FILE=${READY_FILE}"
  "RESIDENT_MODEL_DIR=${SERVE_UP_GGUF}"
  "RESIDENT_SPEC_DEPTH=${SPEC_DEPTH}"
  "RESIDENT_SPEC_MODE=${SERVE_UP_SPEC_MODE}"
  "RESIDENT_DRAFT_MODEL=${DRAFT_GGUF}"
  "RESIDENT_IDENT=${ENGINE_IDENT}"
  "RESIDENT_CTX_SIZE=${CTX_SIZE}"
  "HOME=${RUN_DIR}/home"
  "XDG_CACHE_HOME=${RUN_DIR}/cache"
  "${ENGINE_ENV[@]}"
)
log "booting the resident: ${SERVE_UP_GGUF} (${SPEC_LABEL}) via ${SERVE_UP_LAUNCHER}; ONE load for the whole leg"
if [[ "${SERVE_UP_LAUNCHER}" == "docker" ]]; then
  container_user="$(id -u):$(id -g)"
  rw_mounts=(--mount "type=bind,src=${SERVE_UP_SOCKET_DIR},dst=${SERVE_UP_SOCKET_DIR}"
             --mount "type=bind,src=${SERVE_UP_LOG_DIR},dst=${SERVE_UP_LOG_DIR}")
  hardening=()
  if [[ -n "${SANDBOX_USER_ID}" ]]; then
    container_user="${SANDBOX_USER_ID}"
    rw_mounts=(--mount "type=bind,src=${SOCKET_LEG_DIR},dst=${SOCKET_LEG_DIR}")
  fi
  [[ "${SERVE_UP_ALLOW_NEW_PRIVS:-0}" == 1 ]] || hardening+=(--security-opt no-new-privileges)
  [[ "${SERVE_UP_NO_CAP_DROP:-0}" == 1 ]] || hardening+=(--cap-drop ALL)
  docker run --rm --init --name "${CONTAINER_NAME}" \
    --gpus all --ipc host --network none \
    --user "${container_user}" "${hardening[@]}" \
    --read-only --tmpfs /tmp:rw,exec,size=8g \
    --mount "type=bind,src=${SCRIPT_DIR},dst=${SCRIPT_DIR},readonly" \
    --mount "type=bind,src=${SERVE_UP_GGUF},dst=${SERVE_UP_GGUF},readonly" \
    ${DRAFT_MOUNT[@]+"${DRAFT_MOUNT[@]}"} \
    "${rw_mounts[@]}" \
    --entrypoint "${RESIDENT_BIN}" "${ENGINE_IMAGE_ID}" \
    --model "${SERVE_UP_GGUF}" --socket "${SOCKET_PATH}" --ctx "${CTX_SIZE}" \
    --spec "${RESIDENT_SPEC_ARG}" --draft-len "${RESIDENT_DRAFT_LEN}" ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
    --ident "${ENGINE_IDENT}" >"${RESIDENT_LOG}" 2>&1 &
elif [[ "${RESIDENT_BACKEND:-}" == "fake" ]]; then
  env RESIDENT_BACKEND=fake "${RESIDENT_ENV[@]}" python3 "${FAKE_RESIDENT}" >"${RESIDENT_LOG}" 2>&1 &
else
  env "${RESIDENT_ENV[@]}" "${RESIDENT_BIN}" \
    --model "${SERVE_UP_GGUF}" --socket "${SOCKET_PATH}" --ctx "${CTX_SIZE}" \
    --spec "${RESIDENT_SPEC_ARG}" --draft-len "${RESIDENT_DRAFT_LEN}" ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
    --ident "${ENGINE_IDENT}" >"${RESIDENT_LOG}" 2>&1 &
fi
RESIDENT_PID=$!

start="$(date +%s)"
while :; do
  if ! kill -0 "${RESIDENT_PID}" 2>/dev/null; then
    log "the resident exited before it was healthy; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  # Sandboxed, there is no ready file (see the header); the bound socket is the hint.
  boot_ready=0
  [[ -n "${READY_FILE}" && -f "${READY_FILE}" ]] && boot_ready=1
  [[ -S "${SOCKET_PATH}" ]] && boot_ready=1
  if [[ "${boot_ready}" == "1" ]] && probe_resident "${SOCKET_PATH}" >/dev/null 2>&1; then
    break
  fi
  now="$(date +%s)"
  if (( now - start >= SERVE_UP_HEALTH_TIMEOUT_S )); then
    log "the resident was not healthy within ${SERVE_UP_HEALTH_TIMEOUT_S}s; its log:"
    tail -40 "${RESIDENT_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
log "resident healthy on ${SOCKET_PATH} after $(( $(date +%s) - start ))s; every phase now connects instead of loading"

HELLO_JSON="$(probe_resident "${SOCKET_PATH}")" || die "the resident stopped answering before the run started"
hello_depth="$(printf '%s' "${HELLO_JSON}" | jq -r '.draft_tokens - 1')"
[[ "${hello_depth}" == "${SPEC_DEPTH}" ]] \
  || die "the resident reports depth ${hello_depth} but this leg declared ${SPEC_DEPTH}"
hello_mode="$(printf '%s' "${HELLO_JSON}" | jq -r '.spec_mode')"
[[ "${hello_mode}" == "${SERVE_UP_SPEC_MODE}" ]] \
  || die "the resident reports spec mode ${hello_mode} but this leg declared ${SERVE_UP_SPEC_MODE}"
hello_ctx="$(printf '%s' "${HELLO_JSON}" | jq -r '.ctx_size')"
[[ "${hello_ctx}" == "${CTX_SIZE}" ]] \
  || die "the resident reports context ${hello_ctx} but the fixture pins ${CTX_SIZE}"

# --- the leg's identity ------------------------------------------------------
jq -n \
  --arg socket "${SOCKET_PATH}" --arg pid "${RESIDENT_PID}" --arg container "${CONTAINER_NAME}" \
  --arg launcher "${SERVE_UP_LAUNCHER}" --arg image "${ENGINE_IMAGE_ID}" --arg pin "${ENGINE_PIN}" \
  --arg driver "${driver_ver:-unknown}" --arg ident "${ENGINE_IDENT}" \
  --arg weights "${SERVE_UP_GGUF}" --arg drafter "${DRAFT_GGUF}" --arg spec "${SPEC_LABEL}" \
  --argjson depth "${SPEC_DEPTH}" --argjson hello "${HELLO_JSON}" \
  --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{engine:"llama.cpp", weight_owner:"llamacpp-resident", resident_socket:$socket, resident_pid:$pid,
    container:$container, launcher:$launcher, image:$image, engine_pin:$pin, driver_version:$driver,
    engine_ident:$ident, weights_dir:$weights, draft_model:$drafter, spec_config:$spec, declared_draft_len:$depth,
    hello:$hello, started:$started}' > "${IDENTITY_FILE}"

export SERVE_IDENTITY_FILE="${IDENTITY_FILE}"
export SERVE_IDENTITY_WEIGHT_OWNER="llamacpp-resident"
export SERVE_IDENTITY_SPEC_CONFIG="${SPEC_LABEL}"
export SERVE_IDENTITY_SPEC_SPECULATIVE="${SERVE_UP_SPECULATIVE}"
export SERVE_IDENTITY_SPEC_DRAFT_LEN="${SPEC_DEPTH}"
export SERVE_IDENTITY_RESIDENT_PID="${RESIDENT_PID}"
export DS4_RESIDENT_SOCKET="${SOCKET_PATH}"
export BENCH_WORKER_RESIDENT_SOCKET="${SOCKET_PATH}"

# --- --boot: publish the socket and hand the resident over -------------------
# The sidecar beside the socket carries what --stop needs and nothing else: the
# pid on line 1, the ready file on line 2, the container name on line 3 (empty
# for the host launcher). It is written BEFORE the socket path is published.
if [[ "${MODE}" == "boot" ]]; then
  # Sandboxed, --stop needs no sidecar: the container name is in the socket path.
  if [[ -z "${SANDBOX_USER_ID}" ]]; then
    container_line=""
    [[ "${SERVE_UP_LAUNCHER}" != "docker" ]] || container_line="${CONTAINER_NAME}"
    printf '%s\n%s\n%s\n' "${RESIDENT_PID}" "${READY_FILE}" "${container_line}" > "${SOCKET_PATH}.pid"
  fi
  socket_out_dir="$(dirname -- "${BOOT_SOCKET_OUT}")"
  [[ "${socket_out_dir}" == "." ]] || mkdir -p "${socket_out_dir}"
  printf '%s\n' "${SOCKET_PATH}" > "${BOOT_SOCKET_OUT}"
  KEEP_RESIDENT=1
  log "resident handed over on ${SOCKET_PATH} (${SPEC_LABEL}); stop it with: tools/serve-up.sh --stop --socket ${SOCKET_PATH}"
  exit 0
fi

log "running: $*"
set +e
"$@"
rc=$?
set -e
exit "${rc}"
