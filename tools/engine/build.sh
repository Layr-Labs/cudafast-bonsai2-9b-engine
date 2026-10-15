#!/usr/bin/env bash
# tools/engine/build.sh -- build everything the scored engine needs, from THIS
# tree, and stage it where benchd and tools/serve-up.sh look.
#
#   1. the Engine Protocol v1 adapter (harness/protocol-adapter, cargo), staged
#      as .build/release/mlxfast-runtime-worker by tools/stage-cuda-engine.sh;
#   2. the pinned engine container bonsai9b-engine:v1 (present locally;
#      ENGINE_BUILD_OFFLINE=1 refuses if it is missing; jobs never pull). With
#      no image Id and no sandbox it builds the image (a participant machine);
#   3. llama-resident: cmake/ninja inside that image, staged at
#      .build/release/llama-resident;
#   4. the participant's optional harness/kernels/: when harness/kernels/build.sh
#      exists it runs INSIDE the pinned container with no network and writes
#      .build/kernels.
#
# SANDBOX (docs/sandbox-design.md sections 5 and 6a). With SANDBOX_USER set
# participant code never runs as the caller (cargo vendoring + offline build,
# then llama-resident compile in docker as SANDBOX_USER:SHARE_GROUP, default
# bonsaishare, optional kernels). SHARE_GROUP and RUST_DIR are
# tools/box-runner/setup-sandbox-user.sh's names.
#
# Needs cargo, docker (with the NVIDIA runtime) and jq.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'engine/build.sh: %s\n' "$*"; }
die() { printf 'engine/build.sh: %s\n' "$*" >&2; exit 1; }

FIXTURE="${ROOT_DIR}/fixtures/bonsai2_9b_cuda_v1_track.json"
# shellcheck source=tools/engine/read-engine-image-id.sh
. "${ROOT_DIR}/tools/engine/read-engine-image-id.sh"
for tool in cargo docker jq; do command -v "${tool}" >/dev/null 2>&1 || die "${tool} is required"; done
ARCH="$(jq -r '.serve_configuration.toolchain.cuda_arch' "${FIXTURE}")"
LLAMA_CPP_DIR="${ROOT_DIR}/llama.cpp"
RESIDENT_SRC="${ROOT_DIR}/harness/llama_resident"
STAGED_RESIDENT="${ROOT_DIR}/.build/release/llama-resident"

MANIFEST="${ROOT_DIR}/harness/protocol-adapter/Cargo.toml"
SANDBOX_USER="${SANDBOX_USER:-}"
if [[ -n "${SANDBOX_USER}" ]]; then
  [[ -n "${SANDBOX_ROOT:-}" && -n "${SANDBOX_JOB:-}" ]] \
    || die "SANDBOX_USER is set but SANDBOX_ROOT or SANDBOX_JOB is not"
  sudo -n -u "${SANDBOX_USER}" true \
    || die "SANDBOX_USER=${SANDBOX_USER} is set but 'sudo -n -u ${SANDBOX_USER} true' fails; run tools/box-runner/setup-sandbox-user.sh"
  command -v python3 >/dev/null 2>&1 || die "python3 is required to check the Cargo inputs before vendoring"
  JOB="${SANDBOX_ROOT}/${SANDBOX_JOB}"
  [[ ! -e "${JOB}" ]] || die "sandbox job directory ${JOB} already exists"
  mkdir -m 2750 "${JOB}" "${JOB}/vendor" "${JOB}/sock"
  mkdir -m 2770 "${JOB}/cargo-home" "${JOB}/target" "${JOB}/kernels-out" "${JOB}/adapter-home" "${JOB}/cmake-out"
fi

vendor_inputs_ok() {
  python3 - "${ROOT_DIR}" <<'PY' || die "vendor refused (see above); nothing was vendored"
import os, sys, tomllib
root = sys.argv[1]
harness = os.path.join(root, "harness")
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
bad = [] if not os.path.islink(harness) else ["harness/ is a symlink"]
lock = os.path.join(harness, "protocol-adapter", "Cargo.lock")
try:
    with open(lock, "rb") as f:
        pkgs = tomllib.load(f).get("package", [])
except (OSError, tomllib.TOMLDecodeError) as e:
    pkgs = []; bad.append(f"Cargo.lock unreadable: {e}")
for p in pkgs:
    if "source" in p and p["source"] != CRATES_IO:
        bad.append(f"Cargo.lock: {p.get('name')} has source {p['source']!r}, not crates.io")
def walk(node, rel, where, top):
    if isinstance(node, list):
        for v in node: walk(v, rel, where, False)
        return
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if top and k in ("patch", "replace"):
            bad.append(f"{where}: has a [{k}] section")
        if k in ("registry", "registry-index") and v != "crates-io":
            bad.append(f"{where}: {k} = {v!r} is not crates-io")
        paths = [v] if k in ("path", "workspace", "build") and isinstance(v, str) else \
                [x for x in v if isinstance(x, str)] if k == "members" and isinstance(v, list) else []
        for p in paths:
            real = os.path.realpath(os.path.join(rel, p))
            if not real.startswith(harness + os.sep):
                bad.append(f"{where}: {k} = {p!r} resolves outside harness/ ({real})")
        walk(v, rel, where, False)
for d, dirs, files in os.walk(harness):
    dirs[:] = [x for x in dirs if x != "target"]
    if "Cargo.toml" in files:
        t = os.path.join(d, "Cargo.toml")
        where = os.path.relpath(t, root)
        try:
            with open(t, "rb") as f:
                walk(tomllib.load(f), d, where, True)
        except (OSError, tomllib.TOMLDecodeError) as e:
            bad.append(f"{where}: unreadable: {e}")
for b in bad:
    print(f"engine/build.sh: vendor refused: {b}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
}

sandbox_quiet() {
  sudo -n -u "${SANDBOX_USER}" -- kill -KILL -1 2>/dev/null || true
  local rc=0
  pgrep -u "${SANDBOX_USER}" >/dev/null || rc=$?
  [[ ${rc} -eq 1 ]] || die "sandbox user ${SANDBOX_USER} still has processes after the build (pgrep rc=${rc})"
}

stage_llama_resident() {
  local src="$1"
  if [[ -n "${SANDBOX_USER}" ]]; then
    if [[ -L "${src}" || ! -f "${src}" || "$(stat -c %U -- "${src}")" != "${SANDBOX_USER}" ]]; then
      die "refusing ${src}: not a regular file owned by ${SANDBOX_USER}"
    fi
  fi
  mkdir -p "$(dirname "${STAGED_RESIDENT}")"
  rm -f "${STAGED_RESIDENT}"
  cp --no-dereference --preserve=mode "${src}" "${STAGED_RESIDENT}"
  chmod 0755 "${STAGED_RESIDENT}"
  if [[ -L "${STAGED_RESIDENT}" || ! -f "${STAGED_RESIDENT}" ]]; then
    rm -f "${STAGED_RESIDENT}"
    die "refusing the staged llama-resident: not a regular file after copy"
  fi
}

build_llama_resident_in_image() {
  local build_dir="$1" run_user="$2"
  [[ -f "${RESIDENT_SRC}/CMakeLists.txt" ]] \
    || die "harness/llama_resident/CMakeLists.txt is missing"
  [[ -d "${LLAMA_CPP_DIR}" ]] || die "llama.cpp/ is missing"
  local cmake_cmd
  cmake_cmd="$(printf 'cmake -G Ninja -S %q -B %q -DLLAMA_CPP_DIR=%q -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release && ninja -C %q llama-resident' \
    "${RESIDENT_SRC}" "${build_dir}" "${LLAMA_CPP_DIR}" "${build_dir}")"
  # nvcc writes its intermediates to /tmp; the read-only root needs a tmpfs there.
  local run_args=(--rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges
    --tmpfs "/tmp:rw,exec,size=16g"
    --user "${run_user}"
    --mount "type=bind,src=${ROOT_DIR},dst=${ROOT_DIR},readonly"
    --mount "type=bind,src=${build_dir},dst=${build_dir}"
    --workdir "${build_dir}" --entrypoint bash "${engine_image_id}" -lc "${cmake_cmd}")
  log "building llama-resident inside ${engine_image_id} (no network)"
  docker run "${run_args[@]}"
  [[ -x "${build_dir}/bin/llama-resident" ]] \
    || die "llama-resident build did not produce ${build_dir}/bin/llama-resident"
  stage_llama_resident "${build_dir}/bin/llama-resident"
}

# 1. adapter
log "building the protocol adapter"
if [[ -z "${SANDBOX_USER}" ]]; then
  ENGINE_CUDA_ARCH="${ARCH}" cargo build --release --locked \
    --manifest-path "${MANIFEST}"
  "${ROOT_DIR}/tools/stage-cuda-engine.sh"
else
  vendor_inputs_ok
  log "vendoring crates into ${JOB}/vendor"
  cargo vendor --locked --manifest-path "${MANIFEST}" "${JOB}/vendor" \
    > "${JOB}/cargo-home/config.toml"
  odd="$(find "${JOB}/vendor" -type l -o ! -type f ! -type d | head -5)"
  [[ -z "${odd}" ]] || die "vendor refused: the vendored tree holds a symlink or special file: ${odd//$'\n'/; }"
  chmod -R g+rX,o-rwx "${JOB}/vendor"
  log "building as ${SANDBOX_USER}, offline"
  sudo -n -u "${SANDBOX_USER}" -- prlimit --nproc="${CARGO_NPROC_LIMIT:-4096}" \
    --as="${CARGO_AS_LIMIT:-68719476736}" env PATH="${RUST_DIR:-/opt/rust}/cargo/bin:/usr/bin:/bin" \
    RUSTUP_HOME="${RUST_DIR:-/opt/rust}/rustup" CARGO_HOME="${JOB}/cargo-home" \
    CARGO_TARGET_DIR="${JOB}/target" ENGINE_CUDA_ARCH="${ARCH}" \
    cargo build --release --locked --offline --manifest-path "${MANIFEST}"
  sandbox_quiet
  CUDA_ENGINE_EXECUTABLE="${JOB}/target/release/cuda-engine" \
    "${ROOT_DIR}/tools/stage-cuda-engine.sh"
fi

# 2. engine image (run by image Id only). A ranked box has the organizer's Id
# file and builds offline, so a missing Id or image there stops the build. On a
# participant machine nothing has built the image yet: build it here and write
# its Id to .build/engine-image.id, where read_engine_image_id finds it. The
# build sets no tag: on a box that also runs the ranked track, the tag
# bonsai9b-engine:v1 names the organizer's image, and a containerd image store
# drops that image when the tag moves.
if ! engine_image_id="$(read_engine_image_id)" \
    || ! verify_engine_image_id_present "${engine_image_id}"; then
  [[ "${ENGINE_BUILD_OFFLINE:-0}" != "1" && -z "${SANDBOX_USER}" ]] \
    || die "no engine image (organizer writes /srv/bonsai9b/engine-image.id and builds the image it names; tests may set BONSAI9B_ENGINE_IMAGE_ID)"
  log "building the engine image from tools/engine/Dockerfile"
  mkdir -p "${ROOT_DIR}/.build"
  docker build -f "${ROOT_DIR}/tools/engine/Dockerfile" \
    --iidfile "${ROOT_DIR}/.build/engine-image.id" "${ROOT_DIR}/tools/engine"
  engine_image_id="$(read_engine_image_id)" \
    || die "docker build wrote no readable engine image Id"
  verify_engine_image_id_present "${engine_image_id}" \
    || die "the engine image ${engine_image_id} is not present after the build"
fi
log "engine image present: ${engine_image_id}"

# 3. llama-resident
if [[ -z "${SANDBOX_USER}" ]]; then
  LOCAL_BUILD="${ROOT_DIR}/.build/engine-cmake"
  rm -rf "${LOCAL_BUILD}"
  mkdir -p "${LOCAL_BUILD}"
  build_llama_resident_in_image "${LOCAL_BUILD}" "$(id -u):$(id -g)"
else
  SHARE_GID="$(getent group "${SHARE_GROUP:-bonsaishare}" | cut -d: -f3)"
  [[ -n "${SHARE_GID}" ]] || die "group ${SHARE_GROUP:-bonsaishare} does not exist"
  build_llama_resident_in_image "${JOB}/cmake-out" "$(id -u "${SANDBOX_USER}"):${SHARE_GID}"
  sandbox_quiet
fi

# 4. optional participant kernels
KERNELS_SRC="${ROOT_DIR}/harness/kernels"
rm -rf "${ROOT_DIR}/.build/kernels"
mkdir -p "${ROOT_DIR}/.build/kernels"
if [[ -f "${KERNELS_SRC}/build.sh" ]]; then
  log "building harness/kernels/ inside the pinned container (no network)"
  if [[ -z "${SANDBOX_USER}" ]]; then
    KERNELS_OUT="${ROOT_DIR}/.build/kernels"
    RUN_ARGS=(--user "$(id -u):$(id -g)" --mount "type=bind,src=${ROOT_DIR},dst=${ROOT_DIR}")
  else
    KERNELS_OUT="${JOB}/kernels-out"
    SHARE_GID="$(getent group "${SHARE_GROUP:-bonsaishare}" | cut -d: -f3)"
    [[ -n "${SHARE_GID}" ]] || die "group ${SHARE_GROUP:-bonsaishare} does not exist"
    RUN_ARGS=(--user "$(id -u "${SANDBOX_USER}"):${SHARE_GID}"
      --mount "type=bind,src=${ROOT_DIR},dst=${ROOT_DIR},readonly"
      --mount "type=bind,src=${KERNELS_OUT},dst=${KERNELS_OUT}"
      --cpus "${KERNELS_CPUS:-16}" --memory "${KERNELS_MEMORY:-64g}"
      --pids-limit "${KERNELS_PIDS_LIMIT:-4096}"
      --cap-drop ALL --security-opt no-new-privileges)
  fi
  docker run --rm --gpus all --network none "${RUN_ARGS[@]}" \
    --env HOME=/tmp --env "KERNELS_OUT=${KERNELS_OUT}" \
    --env "TORCH_CUDA_ARCH_LIST=$(printf '%s' "${ARCH#sm_}" | sed 's/^\(.*\)\(.\)$/\1.\2/')a" \
    --workdir "${KERNELS_SRC}" --entrypoint bash "${engine_image_id}" ./build.sh
  if [[ -n "${SANDBOX_USER}" ]]; then
    sandbox_quiet
    cp -rP --preserve=mode "${KERNELS_OUT}/." "${ROOT_DIR}/.build/kernels/"
    odd="$(find "${ROOT_DIR}/.build/kernels" -type l -o ! -type f ! -type d | head -5)"
    if [[ -n "${odd}" ]]; then
      rm -rf "${ROOT_DIR}/.build/kernels"
      die "kernels refused: harness/kernels/build.sh produced a symlink or special file: ${odd//$'\n'/; }"
    fi
  fi
else
  log "no harness/kernels/build.sh; nothing to build"
fi
log "engine built and staged"
