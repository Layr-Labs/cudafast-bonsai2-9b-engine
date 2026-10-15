#!/usr/bin/env bash
# install-service.sh -- install the router and fleet-watcher user units
# (systemd/bonsai9b-router.service, bonsai9b-watch.service) from this checkout.
#
# Usage:
#   tools/router/install-service.sh                     # render; enable router; enable watch only when ~/.config/bonsai9b/watch.env sets WATCH_HOSTS
#   tools/router/install-service.sh --uninstall         # stop and remove both
#   tools/router/install-service.sh --render DIRECTORY  # render only; needs no systemd session
#   tools/router/install-service.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
# Must equal the unit's EnvironmentFile=%h/.config/bonsai9b/watch.env (not XDG_CONFIG_HOME).
WATCH_ENV="$HOME/.config/bonsai9b/watch.env"
UNITS=(bonsai9b-router.service bonsai9b-watch.service)

watch_configured() {
  # The value must name at least one host after the quotes are removed.
  [[ -f "${WATCH_ENV}" ]] || return 1
  local v
  v="$(sed -nE 's/^[[:space:]]*WATCH_HOSTS=(.*)$/\1/p' "${WATCH_ENV}" | tail -n 1 | tr -d "\"'" | tr -d '[:space:]')"
  [[ -n "${v}" ]]
}

usage() {
  echo "Usage: $0 [--uninstall | --render DIRECTORY | --help]"
}

render() {
  python3 - "$SCRIPT_DIR" "$REPO_DIR" "$1" <<'PY'
import os
from pathlib import Path
import re
import sys

source, repo, output = sys.argv[1:]

def escape(value, command=False):
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise SystemExit("Service paths and PATH must not contain control characters")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return value.replace("$", "$$") if command else value

import shutil
# A minimal PATH: where gh and python3 live, plus the system dirs. The session
# PATH carries short-lived tool dirs that must not be baked into a unit.
dirs = []
for tool in ("gh", "python3", "git"):
    found = shutil.which(tool)
    if not found:
        raise SystemExit(f"{tool} is not on PATH; the router needs it")
    d = os.path.dirname(found)
    if d not in dirs:
        dirs.append(d)
for d in ("/usr/local/bin", "/usr/bin", "/bin"):
    if d not in dirs:
        dirs.append(d)
values = {"@REPO@": escape(repo), "@EXEC_REPO@": escape(repo, command=True),
          "@PATH@": escape(":".join(dirs))}
destination = Path(output)
destination.mkdir(parents=True, exist_ok=True)
for name in ("bonsai9b-router.service", "bonsai9b-watch.service"):
    text = (Path(source) / "systemd" / name).read_text()
    # One pass avoids interpreting placeholder-like text inside a path.
    text = re.sub(r"@(?:EXEC_REPO|REPO|PATH)@", lambda m: values[m[0]], text)
    (destination / name).write_text(text)
PY
}

stop_loaded() {
  local unit loaded
  for unit in "${UNITS[@]}"; do
    loaded="$(systemctl --user list-units --all --plain --no-legend "$unit")"
    if [[ -n "$loaded" ]]; then
      systemctl --user stop "$unit" || true
      # Failed transient units may remain loaded after stop; release them too.
      loaded="$(systemctl --user list-units --all --plain --no-legend "$unit")"
      if [[ -n "$loaded" ]]; then
        systemctl --user reset-failed "$unit"
      fi
    fi
  done
}

case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --render)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    render "$2"
    exit 0
    ;;
  --uninstall)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    stop_loaded
    for unit in "${UNITS[@]}"; do
      if [[ -e "$UNIT_DIR/$unit" || -L "$UNIT_DIR/$unit" ]]; then
        systemctl --user disable "$unit"
        rm -- "$UNIT_DIR/$unit"
      fi
    done
    systemctl --user daemon-reload
    echo "Services removed; ~/bonsai9b-router state and reports retained."
    exit 0
    ;;
  "") [[ $# -eq 0 ]] || { usage >&2; exit 2; } ;;
  *) usage >&2; exit 2 ;;
esac

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf -- "$STAGING_DIR"' EXIT
render "$STAGING_DIR"
mkdir -p -- "$UNIT_DIR"
# Stop transient units before loading their persistent replacements. Also
# restart installed units on updates so the new paths and PATH take effect.
stop_loaded
for unit in "${UNITS[@]}"; do
  install -m 0644 -- "$STAGING_DIR/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload
systemctl --user enable --now bonsai9b-router.service
if watch_configured; then
  systemctl --user enable --now bonsai9b-watch.service
  echo "Installed and started the router and fleet watcher. For startup at boot without login, run:"
else
  if systemctl --user is-enabled bonsai9b-watch.service &>/dev/null; then
    systemctl --user disable --now bonsai9b-watch.service || true
  fi
  echo "Installed and started the router only."
  echo "Fleet watcher not enabled: create ${WATCH_ENV} with WATCH_HOSTS (see tools/router/systemd/watch.env.example), then re-run this script."
  echo "For startup at boot without login, run:"
fi
printf '  loginctl enable-linger %q\n' "${USER:-$(id -un)}"
