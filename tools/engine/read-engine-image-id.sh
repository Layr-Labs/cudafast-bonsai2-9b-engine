#!/usr/bin/env bash
# Resolve the engine image Id (sha256:…) for docker run/inspect.
# Never use the bonsai9b-engine:v1 tag at runtime.
#
# The Id comes from the first of:
#   1. BONSAI9B_ENGINE_IMAGE_ID (tests);
#   2. the file BONSAI9B_ENGINE_IMAGE_ID_FILE names (a ranked box);
#   3. /srv/bonsai9b/engine-image.id (a ranked box; the organizer writes it);
#   4. .build/engine-image.id in this checkout (a participant machine;
#      tools/engine/build.sh writes it). --require-file skips this step.
#
# Usage: source this file, then call read_engine_image_id [--require-file]
set -euo pipefail

ENGINE_IMAGE_ID_BOX_FILE="/srv/bonsai9b/engine-image.id"
ENGINE_IMAGE_ID_CHECKOUT_FILE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)/.build/engine-image.id"

read_engine_image_id() {
  local require_file=0
  [[ "${1:-}" == --require-file ]] && require_file=1
  local id file
  id="$(tr -d '[:space:]' <<<"${BONSAI9B_ENGINE_IMAGE_ID:-}")"
  if [[ -n "${id}" ]]; then
    [[ "${id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
      || { echo "read-engine-image-id: BONSAI9B_ENGINE_IMAGE_ID is not sha256:… (${id})" >&2; return 1; }
    printf '%s\n' "${id}"
    return 0
  fi
  file="${BONSAI9B_ENGINE_IMAGE_ID_FILE:-${ENGINE_IMAGE_ID_BOX_FILE}}"
  if [[ -z "${BONSAI9B_ENGINE_IMAGE_ID_FILE:-}" && ! -f "${file}" && "${require_file}" == 0 ]]; then
    file="${ENGINE_IMAGE_ID_CHECKOUT_FILE}"
  fi
  if [[ ! -f "${file}" ]]; then
    [[ "${require_file}" == 1 ]] \
      && { echo "read-engine-image-id: ${file} is missing (organizer must stage the engine image Id)" >&2; return 1; }
    echo "read-engine-image-id: ${file} is missing" >&2
    return 1
  fi
  id="$(tr -d '[:space:]' < "${file}")"
  [[ -n "${id}" ]] \
    || { echo "read-engine-image-id: ${file} is empty" >&2; return 1; }
  [[ "${id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || { echo "read-engine-image-id: ${file} does not hold a sha256 image Id (${id})" >&2; return 1; }
  printf '%s\n' "${id}"
}

verify_engine_image_id_present() {
  local want="$1"
  command -v docker >/dev/null 2>&1 || { echo "read-engine-image-id: docker is required" >&2; return 1; }
  local got
  got="$(docker image inspect --format '{{.Id}}' "${want}" 2>/dev/null || true)"
  [[ -n "${got}" ]] \
    || { echo "read-engine-image-id: engine image ${want} is not present on this box" >&2; return 1; }
  [[ "${got}" == "${want}" ]] \
    || { echo "read-engine-image-id: docker image inspect Id (${got}) disagrees with ${want}" >&2; return 1; }
}
