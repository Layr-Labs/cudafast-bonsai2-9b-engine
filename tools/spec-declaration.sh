#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the serve-boot speculative config
# from it. tools/serve-up.sh, .github/workflows/benchmark.yml and
# tools/ranked-box-preflight.sh ALL resolve the serve spec THROUGH this one
# script, so the fail-closed validation and the derivation can never drift
# between the arm gate and the serve that actually boots.
#
# NOT AN EDITABLE PATH. benchmark.json editablePaths is
# {llama.cpp, harness, mtp-head.manifest.json}. The DECLARATION
# (mtp-head.manifest.json) is editable -- a submission EXPRESSES a value there --
# but this DERIVATION is trusted, so a submission cannot rewrite how the value is
# interpreted or relax the envelope.
#
# WHY THIS EXISTS. The drafter and its depth are fixed when the resident boots
# (`--spec mtp|dflash --draft-len`), so they cannot ride the Engine Protocol
# wire per request: the leg's resident is booted at the declared mode and depth
# (tools/serve-up.sh).
# This script is the declaration -> serve-config bridge. The echo half is in the
# adapter (harness/protocol-adapter: `resolve_spec` -> `effective_spec`), which
# resolves a requested depth against the envelope and echoes what ran; benchd
# seals that echo.
#
# THE DECLARATION is the serve knob, in mtp-head.manifest.json:
#
#   "spec": { "enabled": <bool>, "mode": "mtp" | "dflash", "num_speculative_tokens": <int> }
#
# ABSENT `spec` block, `enabled: false`, or `num_speculative_tokens: 0` => SERIAL
# (RESIDENT_SPEC_DEPTH=0), the launch-reference behaviour. `enabled: true` with N
# in the mode's permitted draft depths => the resident boots that drafter at N
# speculative tokens: `mtp` (the default when `mode` is absent) is the target
# GGUF's own MTP layer (llama.cpp draft-mtp, mtp_head.permitted_draft_depths);
# `dflash` is the pinned DFlash block drafter (llama.cpp draft-dflash,
# dflash_drafter.permitted_draft_depths). The mode must be in the fixture's
# allowed_modes.
#
# THE REST OF THE DECLARATION is validated here too, because this is the one
# trusted reader of the file (issue #24 work item A). docs/participant-contract.md
# 4.1 promises that `pinned` is the only accepted source and that there is no
# `arm` key -- the drafter is selected by `spec.mode`. Only the `spec`
# block was ever read, so `"source": "remote"` and an unknown top-level key both
# passed a `validate`. `max_bytes`, `bytes` and `sha256` are recorded, not read.
#
# VALIDATION is fail-closed. A REFUSAL (exit 1, message on stderr) for:
#   * a declaration that is not a JSON object, or carries an unknown top-level
#     key (allowed: version, source, max_bytes, bytes, sha256, spec);
#   * a `source` other than "pinned" -- "remote" and "in_branch" by name;
#   * a `spec` block that is not an object, or carries an unknown key;
#   * `enabled` that is not a boolean, or `num_speculative_tokens` not an integer;
#   * a `mode` that is not "mtp" or "dflash", or not in the fixture's allowed_modes;
#   * a value outside the structural range 0..SPEC_MAX_TOKENS;
#   * an ENABLED depth that is not in the mode's permitted_draft_depths
#     (fixtures/bonsai2_9b_cuda_v1_track.json `mtp_head.permitted_draft_depths`
#     or `dflash_drafter.permitted_draft_depths`, the envelopes
#     harness/protocol-adapter/src/session_backend.rs mirrors). An out-of-envelope
#     depth is REFUSED, never clamped -- so the serve that boots and the
#     `effective_spec` the adapter seals can never disagree, matching the
#     sibling's resolveDepth discipline.
#
# Usage:
#   tools/spec-declaration.sh speculative   # 0 (serial) or 1 (mtp or dflash)
#   tools/spec-declaration.sh mode          # "serial", "mtp" or "dflash"
#   tools/spec-declaration.sh draft-len     # num_speculative_tokens (0 = serial)
#   tools/spec-declaration.sh describe      # "serial", "mtp<N>" or "dflash<N>" (identity label)
#   tools/spec-declaration.sh validate      # exit 0 if valid, else refuse
# Every subcommand validates the declaration first and refuses on an invalid one.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/bonsai2_9b_cuda_v1_track.json}"

# The structural sanity ceiling for a declared draft length: the DFlash
# drafter's block of 16 holds the anchor and at most 15 drafts. The mode's
# permitted_draft_depths is the tighter, AUTHORITATIVE set enforced below when
# the value is enabled; this is the outer type/range guard around it.
SPEC_MAX_TOKENS=15


fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist, so the stock-repo derivation
# never depends on a spec block being present.
enabled="false"
mode="mtp"
raw_tokens="0"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"

  # --- the whole declaration, not just the spec block ------------------------
  # Everything the file promises is checked, in the same fail-closed style as
  # the spec block below. An ABSENT key is fine throughout: an absent
  # declaration, an absent source and an absent spec block are all the default
  # (organizer-pinned head, serial), which is what most submissions carry.
  [[ "$(jq -r 'type' "${MANIFEST}")" == "object" ]] \
    || fail "mtp-head.manifest.json must be a JSON object with keys {version, source, max_bytes, bytes, sha256, spec}"

  # A typo'd or invented top-level key must not read as its default, exactly as
  # for the spec block. `arm` is the one worth naming: this track has a single
  # speculative arm, so a declaration that tries to select one is stating
  # something the runner will not honour, and silence there is the wrong answer.
  unknown_top="$(jq -r 'keys[] | select(. != "version" and . != "source" and . != "max_bytes" and . != "bytes" and . != "sha256" and . != "spec")' "${MANIFEST}")"
  [[ -z "${unknown_top}" ]] \
    || fail "the declaration carries unknown top-level key(s): $(printf '%s' "${unknown_top}" | tr '\n' ' '); allowed keys are version, source, max_bytes, bytes, sha256, spec"

  if [[ "$(jq -r 'has("source")' "${MANIFEST}")" == "true" ]]; then
    declared_source="$(jq -r '.source | if type == "string" then . else tojson end' "${MANIFEST}")"
    [[ "${declared_source}" == "pinned" ]] \
      || fail "source \"${declared_source}\" is not accepted; the head is the organizer-staged pinned head, so the only accepted source is \"pinned\""
  fi

  # max_bytes, bytes and sha256 are recorded, not read: nothing bounds a load
  # by them, and the staged head is verified by setup.sh against the fixture pin.

  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {enabled, mode, num_speculative_tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal, matching the sibling's rejectUnknownSpecKeys.
    unknown="$(jq -r '.spec | keys[] | select(. != "enabled" and . != "mode" and . != "num_speculative_tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are enabled, mode, num_speculative_tokens"
    if [[ "$(jq -r '.spec | has("mode")' "${MANIFEST}")" == "true" ]]; then
      mode="$(jq -r '.spec.mode | if type == "string" then . else tojson end' "${MANIFEST}")"
      [[ "${mode}" == "mtp" || "${mode}" == "dflash" ]] \
        || fail "spec.mode \"${mode}\" is not a drafter this track runs; it must be \"mtp\" or \"dflash\""
    fi
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 must be a CONTRACT-permitted depth.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  # The mode must be one the track admits, and the depth one of that mode's
  # permitted_draft_depths (the fixture's authority, the same envelopes
  # harness/protocol-adapter/src/session_backend.rs mirrors). Enforce both so
  # the serve can never boot a drafter or a depth the scored envelope would
  # refuse.
  [[ "$(jq -r --arg m "${mode}" '(.allowed_modes // []) | index($m) != null' "${CONTRACT}")" == "true" ]] \
    || fail "spec.mode \"${mode}\" is not in the contract's allowed_modes ($(jq -c '.allowed_modes' "${CONTRACT}"))"
  envelope="mtp_head"
  [[ "${mode}" == "mtp" ]] || envelope="dflash_drafter"
  jq -e --arg k "${envelope}" '.[$k].permitted_draft_depths | arrays' >/dev/null 2>&1 < "${CONTRACT}" \
    || fail "the contract declares no ${envelope}.permitted_draft_depths for mode ${mode}"
  if [[ "$(jq -r --arg k "${envelope}" --argjson n "${raw_tokens}" '(.[$k].permitted_draft_depths | index($n)) != null' "${CONTRACT}")" != "true" ]]; then
    permitted="$(jq -r --arg k "${envelope}" '.[$k].permitted_draft_depths | map(tostring) | join(", ")' "${CONTRACT}")"
    fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted ${mode} draft depth (${envelope}.permitted_draft_depths: ${permitted})"
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  mode)        if [[ "${spec}" == "1" ]]; then echo "${mode}"; else echo "serial"; fi ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "${mode}${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|mode|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
