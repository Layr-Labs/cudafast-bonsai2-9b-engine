#!/usr/bin/env bash
# setup.sh step 2 against a tiny pinned GGUF served from a file:// base.
# No GPU, toolchain, checkpoint, credentials, or network is used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'rm -rf "${WORK}"' EXIT

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1; else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}

ROOT="${WORK}/checkout with spaces"
mkdir -p "${ROOT}/fixtures" "${ROOT}/tools" "${ROOT}/.build/release"
cp "${REPO_ROOT}/setup.sh" "${ROOT}/"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/.build/release/mlxfast-runtime-worker"
printf '#!/usr/bin/env bash\nexit 0\n' > "${ROOT}/.build/release/llama-resident"
chmod +x "${ROOT}/.build/release/"*

SOURCE="${WORK}/source"
mkdir -p "${SOURCE}"
GGUF_NAME="rehearsal.gguf"
head -c $((RANDOM + 1000)) /dev/urandom > "${SOURCE}/${GGUF_NAME}"
GGUF_BYTES="$(wc -c < "${SOURCE}/${GGUF_NAME}" | tr -d ' ')"
GGUF_SHA="$(sha256_of "${SOURCE}/${GGUF_NAME}")"
write_fixture() {
  jq -n --arg name "${GGUF_NAME}" --argjson b "${GGUF_BYTES}" --arg s "${GGUF_SHA}" \
    '{target:{upstream_model_id:"example/model", upstream_revision:"0000000", gguf:{path:$name, bytes:$b, sha256:$s}}}' \
    > "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json"
}
write_fixture

run_setup() {
  local name="$1"; shift
  rc=0
  (
    cd "${ROOT}"
    env -u MLXFAST_TARGET_GGUF -u MLXFAST_TARGET_BASE_URL \
      MLXFAST_SKIP_ENGINE_BUILD=1 MLXFAST_TARGET_BASE_URL="file://${SOURCE}" "$@" ./setup.sh
  ) > "${WORK}/${name}.log" 2>&1 || rc=$?
}
downloads() { grep -c 'downloading ' "${WORK}/$1.log" || true; }
DEFAULT_DIR="${ROOT}/reference_weights/model"
DEFAULT_GGUF="${DEFAULT_DIR}/${GGUF_NAME}"

run_setup fresh
[[ "${rc}" == 0 ]] || { cat "${WORK}/fresh.log"; exit 1; }
grep -q 'MLXFAST_TARGET_GGUF is unset; using' "${WORK}/fresh.log"
[[ "$(downloads fresh)" == 1 ]]
cmp -s "${DEFAULT_GGUF}" "${SOURCE}/${GGUF_NAME}"
grep -q 'by byte count and sha256' "${WORK}/fresh.log"
grep -q 'wrote the verification marker' "${WORK}/fresh.log"
marker="$(ls "${DEFAULT_DIR}"/.verified-gguf-*)"
[[ "$(wc -l < "${marker}" | tr -d ' ')" == 1 ]]

run_setup rerun MLXFAST_TARGET_BASE_URL="file://${WORK}/nowhere"
[[ "${rc}" == 0 ]] || { cat "${WORK}/rerun.log"; exit 1; }
[[ "$(downloads rerun)" == 0 ]]
grep -q 'verified by byte count and the marker' "${WORK}/rerun.log"

truncate -s 10 "${DEFAULT_GGUF}"
run_setup truncated
[[ "${rc}" == 0 ]] || { cat "${WORK}/truncated.log"; exit 1; }
[[ "$(downloads truncated)" == 1 ]]
cmp -s "${DEFAULT_GGUF}" "${SOURCE}/${GGUF_NAME}"

head -c "${GGUF_BYTES}" /dev/urandom > "${DEFAULT_GGUF}.tmp" && mv "${DEFAULT_GGUF}.tmp" "${DEFAULT_GGUF}"
run_setup tampered
[[ "${rc}" == 1 ]]
grep -q "sha256 mismatch for ${GGUF_NAME}" "${WORK}/tampered.log"
rm -f "${DEFAULT_GGUF}"

cp "${SOURCE}/${GGUF_NAME}" "${WORK}/good.bak"
head -c "${GGUF_BYTES}" /dev/urandom > "${SOURCE}/${GGUF_NAME}"
run_setup corrupt_source
[[ "${rc}" == 1 ]]
grep -q "downloaded ${GGUF_NAME} has sha256 .*; discarded" "${WORK}/corrupt_source.log"
cp "${WORK}/good.bak" "${SOURCE}/${GGUF_NAME}"

run_setup restore
[[ "${rc}" == 0 ]] || { cat "${WORK}/restore.log"; exit 1; }
new_sha="$(printf '0%.0s' {1..64})"
jq --arg s "${new_sha}" '.target.gguf.sha256 = $s' "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json" > "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json.tmp"
mv "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json.tmp" "${ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json"
run_setup repinned
[[ "${rc}" == 1 ]]
grep -q "sha256 mismatch for ${GGUF_NAME}" "${WORK}/repinned.log"
write_fixture

BOX_GGUF="${WORK}/box.gguf"
cp "${SOURCE}/${GGUF_NAME}" "${BOX_GGUF}"
for pass in box box_again; do
  run_setup "${pass}" MLXFAST_TARGET_GGUF="${BOX_GGUF}"
  [[ "${rc}" == 0 ]] || { cat "${WORK}/${pass}.log"; exit 1; }
  [[ "$(downloads "${pass}")" == 0 ]]
  grep -q 'by byte count and sha256' "${WORK}/${pass}.log"
  [[ -z "$(ls "$(dirname "${BOX_GGUF}")"/.verified-gguf-* 2>/dev/null)" ]]
done

run_setup box_missing MLXFAST_TARGET_GGUF="${WORK}/no-such.gguf"
[[ "${rc}" == 1 ]]
grep -q 'MLXFAST_TARGET_GGUF is not a file' "${WORK}/box_missing.log"

truncate -s 10 "${BOX_GGUF}"
run_setup box_short MLXFAST_TARGET_GGUF="${BOX_GGUF}"
[[ "${rc}" == 1 ]]
grep -q "byte-count mismatch for ${GGUF_NAME}" "${WORK}/box_short.log"

mkdir -p "${DEFAULT_DIR}/.setup-lock"
run_setup locked
[[ "${rc}" == 1 ]]
grep -q 'another setup is filling' "${WORK}/locked.log"
rmdir "${DEFAULT_DIR}/.setup-lock"

rm -rf "${DEFAULT_DIR}"
run_setup skipped MLXFAST_SKIP_WEIGHTS_DOWNLOAD=1
[[ "${rc}" == 0 ]]
[[ "$(downloads skipped)" == 0 && ! -d "${DEFAULT_DIR}" ]]
grep -q 'not verifying the GGUF' "${WORK}/skipped.log"

# HF_TOKEN: on the default HF path only, on curl's stdin, never on its argv,
# in the log or in a file. HF_TOKEN unset: the first line of the user's token
# file (HF_TOKEN_PATH, else $HOME/.cache/huggingface/token), trimmed. A curl
# stub records each call and serves the file.
STUB="${WORK}/stub"
mkdir -p "${STUB}"
cat > "${STUB}/curl" <<'STUB_EOF'
#!/usr/bin/env bash
out="" k=no
for ((i = 1; i <= $#; i++)); do
  [[ "${!i}" == -o ]] && { j=$((i + 1)); out="${!j}"; }
  [[ "${!i}" == -K ]] && k=yes
done
cfg=""; [[ "${k}" == yes ]] && cfg="$(cat)"
argv_tok=no; [[ "$*" == *"${STUB_TOKEN}"* ]] && argv_tok=yes
auth=no; [[ "${cfg}" == "header = \"Authorization: Bearer ${STUB_TOKEN}\"" ]] && auth=yes
printf 'K=%s AUTH=%s ARGV_TOKEN=%s URL=%s\n' "${k}" "${auth}" "${argv_tok}" "${*: -1}" >> "${STUB_CALLS}"
cp "${STUB_SOURCE}" "${out}"
STUB_EOF
chmod +x "${STUB}/curl"
token="hf-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')"
HOME_T="${WORK}/home"
mkdir -p "${HOME_T}"
run_stub() { # run_stub NAME ENV... -- setup.sh with the curl stub, the default base unless ENV sets one
  local name="$1"; shift
  rm -rf "${DEFAULT_DIR}"; : > "${WORK}/${name}.calls"; rc=0
  ( cd "${ROOT}" && env -u MLXFAST_TARGET_GGUF -u MLXFAST_TARGET_BASE_URL -u HF_TOKEN -u HF_TOKEN_PATH HOME="${HOME_T}" PATH="${STUB}:${PATH}" \
      STUB_TOKEN="${token}" STUB_CALLS="${WORK}/${name}.calls" STUB_SOURCE="${SOURCE}/${GGUF_NAME}" \
      MLXFAST_SKIP_ENGINE_BUILD=1 "$@" ./setup.sh ) > "${WORK}/${name}.log" 2>&1 || rc=$?
}
run_stub hf_token HF_TOKEN="${token}"
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_token.log"; exit 1; }
grep -qx "K=yes AUTH=yes ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_token.calls"
if grep -rqsF -- "${token}" "${WORK}/hf_token.log" "${ROOT}"; then echo 'FAIL: the HF token is in the log or the checkout' >&2; exit 1; fi
run_stub hf_public
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_public.log"; exit 1; }
grep -qx "K=no AUTH=no ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_public.calls"
run_stub hf_other_base HF_TOKEN="${token}" MLXFAST_TARGET_BASE_URL="https://mirror.example/x"
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_other_base.log"; exit 1; }
grep -qx "K=no AUTH=no ARGV_TOKEN=no URL=https://mirror.example/x/${GGUF_NAME}" "${WORK}/hf_other_base.calls"
mkdir -p "${HOME_T}/.cache/huggingface"
printf '  %s \r\nsecond-line\n' "${token}" > "${HOME_T}/.cache/huggingface/token"
run_stub hf_token_file
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_token_file.log"; exit 1; }
grep -qx "K=yes AUTH=yes ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_token_file.calls"
if grep -rqsF -- "${token}" "${WORK}/hf_token_file.log" "${ROOT}"; then echo 'FAIL: the HF token from the file is in the log or the checkout' >&2; exit 1; fi
printf 'hf-file-token-not-used\n' > "${HOME_T}/.cache/huggingface/token"
run_stub hf_env_wins HF_TOKEN="${token}"
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_env_wins.log"; exit 1; }
grep -qx "K=yes AUTH=yes ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_env_wins.calls"
rm -f "${HOME_T}/.cache/huggingface/token"
printf '%s\n' "${token}" > "${WORK}/token-elsewhere"
run_stub hf_token_path HF_TOKEN_PATH="${WORK}/token-elsewhere"
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_token_path.log"; exit 1; }
grep -qx "K=yes AUTH=yes ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_token_path.calls"
run_stub hf_no_file
[[ "${rc}" == 0 ]] || { cat "${WORK}/hf_no_file.log"; exit 1; }
grep -qx "K=no AUTH=no ARGV_TOKEN=no URL=https://huggingface.co/example/model/resolve/0000000/${GGUF_NAME}" "${WORK}/hf_no_file.calls"

echo 'test-setup-onboarding.sh: all cases passed'
