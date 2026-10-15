#!/usr/bin/env bash
# synthetic-window.sh -- a stand-in target GGUF for off-box tests.
#
# Sourced, not run. It writes a tiny GGUF-shaped file so the real scripts can
# be driven with no checkpoint. The resident those tests boot runs
# RESIDENT_BACKEND=fake through SERVE_UP_LAUNCHER=host, so nothing reads the
# file's contents.

# synthetic_window_gguf FILE -- write a stand-in GGUF at FILE.
synthetic_window_gguf() {
  local file="$1"
  mkdir -p "$(dirname -- "${file}")" || return 1
  head -c 4096 /dev/zero > "${file}"
}

# synthetic_window_weights DIR -- legacy name; writes DIR/rehearsal.gguf.
synthetic_window_weights() {
  synthetic_window_gguf "$1/rehearsal.gguf"
}

# synthetic_window_meminfo FILE [TOTAL_KB] [AVAIL_KB] -- a meminfo the plan reads.
synthetic_window_meminfo() {
  printf 'MemTotal:       %s kB\nMemAvailable:   %s kB\n' "${2:-125000000}" "${3:-120000000}" > "$1"
}

# synthetic_window_env -- the launcher seam every off-box serve uses.
synthetic_window_env() {
  export SERVE_UP_LAUNCHER=host RESIDENT_BACKEND=fake
}
