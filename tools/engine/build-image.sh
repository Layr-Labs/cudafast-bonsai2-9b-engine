#!/usr/bin/env bash
# Organizer-only: build bonsai9b-engine:v1 and write its image ID to OUT_FILE.
# Never called from a ranked job. A participant machine gets an untagged image
# from tools/engine/build.sh.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." >/dev/null && pwd -P)"
OUT_FILE="${1:-}"
[[ -n "${OUT_FILE}" ]] || { echo "usage: tools/engine/build-image.sh OUT_FILE" >&2; exit 2; }

TAG="bonsai9b-engine:v1"
docker build -f "${ROOT_DIR}/tools/engine/Dockerfile" -t "${TAG}" "${ROOT_DIR}/tools/engine"
docker image inspect --format '{{.Id}}' "${TAG}" > "${OUT_FILE}"
printf 'tools/engine/build-image.sh: wrote %s (%s)\n' "${OUT_FILE}" "$(cat "${OUT_FILE}")"
