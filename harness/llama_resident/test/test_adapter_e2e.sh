#!/usr/bin/env bash
# On-box e2e: real cuda-engine against llama-resident on the target GGUF.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/.build/llama-resident}"
ADAPTER="${ADAPTER:-$ROOT/harness/protocol-adapter/target/release/cuda-engine}"
RESIDENT_BIN="${RESIDENT_BIN:-$BUILD_DIR/bin/llama-resident}"
MODEL="${MLXFAST_TARGET_GGUF:-}"
CTX="${RESIDENT_CTX:-8192}"
DECODE_COUNT="${DECODE_COUNT:-128}"
GOLDEN_LEN="${GOLDEN_LEN:-256}"
DEPTHS="${DEPTHS:-0 1 2 3}"
# The drafter of every depth above 0: mtp (the in-file MTP layer) or dflash
# (the pinned DFlash2 drafter, beside the target GGUF).
SPEC_MODE="${SPEC_MODE:-mtp}"

if [[ ! -x "$ADAPTER" ]]; then
  echo "missing cuda-engine at $ADAPTER (cargo build --release in harness/protocol-adapter)" >&2
  exit 1
fi
if [[ ! -x "$RESIDENT_BIN" ]]; then
  echo "missing llama-resident at $RESIDENT_BIN" >&2
  exit 1
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "set MLXFAST_TARGET_GGUF to the target GGUF path" >&2
  exit 1
fi

export RESIDENT_BIN MODEL MLXFAST_TARGET_GGUF="$MODEL"
DRAFT_ARGS=()
case "$SPEC_MODE" in
  mtp) ;;
  dflash)
    DRAFT_MODEL="$(dirname "$MODEL")/$(jq -r '.dflash_drafter.file' "$ROOT/fixtures/bonsai2_9b_cuda_v1_track.json")"
    [[ -f "$DRAFT_MODEL" ]] || { echo "no DFlash drafter at $DRAFT_MODEL" >&2; exit 1; }
    DRAFT_ARGS=(--draft-model "$DRAFT_MODEL")
    ;;
  *) echo "SPEC_MODE must be mtp or dflash (got '$SPEC_MODE')" >&2; exit 1 ;;
esac

PROMPT_JSON='['"$(python3 -c 'print(",".join(str(i) for i in range(100,164)))')"']'

run_phase() {
  local sock="$1"
  local depth="$2"
  local count="$3"
  local spec_json
  if [[ "$depth" == 0 ]]; then
    spec_json='{"mode":"serial"}'
  else
    spec_json="{\"mode\":\"$SPEC_MODE\",\"$SPEC_MODE\":{\"depth\":$depth}}"
  fi
  export BENCH_WORKER_RESIDENT_SOCKET="$sock"
  unset DS4_RESIDENT_SOCKET
  local stdin
  stdin=$(cat <<EOF
{"id":1,"kind":"free_decode_begin","seed_tokens":$PROMPT_JSON,"spec":$spec_json}
{"id":2,"kind":"free_decode_run","count":$count}
EOF
)
  python3 - "$stdin" "$ADAPTER" <<'PY'
import json, subprocess, sys
stdin, adapter = sys.argv[1], sys.argv[2]
out = subprocess.run([adapter], input=stdin, capture_output=True, text=True, timeout=600)
if out.returncode != 0:
    sys.stderr.write(out.stderr)
    sys.exit(out.returncode)
lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
print(json.dumps(lines))
PY
}

boot_resident() {
  local depth="$1"
  local dir="$2"
  local sock="$dir/r.sock"
  local ready="$dir/ready"
  local spec=serial
  local draft=0
  local draft_args=()
  if [[ "$depth" != 0 ]]; then
    spec="$SPEC_MODE"
    draft="$depth"
    draft_args=("${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"}")
  fi
  rm -f "$sock" "$ready"
  export RESIDENT_READY_FILE="$ready"
  "$RESIDENT_BIN" \
    --model "$MODEL" \
    --socket "$sock" \
    --ctx "$CTX" \
    --spec "$spec" \
    --draft-len "$draft" \
    "${draft_args[@]+"${draft_args[@]}"}" \
    --ident "e2e-depth-$depth" \
    >"$dir/resident-$depth.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do
    [[ -f "$ready" ]] && break
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "resident exited during boot (depth=$depth)" >&2
      tail -30 "$dir/resident-$depth.log" >&2 || true
      wait "$pid" || true
      exit 1
    fi
    sleep 0.5
  done
  [[ -f "$ready" ]] || { echo "resident not ready depth=$depth" >&2; kill "$pid"; exit 1; }
  echo "$pid" >"$dir/pid"
  echo "$sock"
}

stop_resident() {
  local dir="$1"
  if [[ -f "$dir/pid" ]]; then
    kill "$(cat "$dir/pid")" 2>/dev/null || true
    wait "$(cat "$dir/pid")" 2>/dev/null || true
    rm -f "$dir/pid"
  fi
}

golden_replay_serial() {
  local sock="$1"
  python3 - "$sock" "$PROMPT_JSON" "$GOLDEN_LEN" <<'PY'
import json, socket, sys

sock_path, prompt_json, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = json.loads(prompt_json)

def call(s, op, **kw):
    req = {"op": op, **kw}
    s.sendall((json.dumps(req) + "\n").encode())
    return json.loads(s.makefile("rb").readline())

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
assert call(s, "hello")["ok"]
seed = call(s, "sync", tokens=prompt)["token"]
tape = []
logits_tape = []
t = seed
for _ in range(n):
    top = call(s, "top_logits", k=8)
    tape.append(t)
    logits_tape.append((top["ids"], top["logits"]))
    t = call(s, "eval", token=t)["token"]
s.close()

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect(sock_path)
assert call(s, "hello")["ok"]
call(s, "sync", tokens=prompt)
for i, tok in enumerate(tape):
    if i > 0:
        call(s, "eval", token=tape[i - 1])
    top = call(s, "top_logits", k=8)
    want_ids, want_logits = logits_tape[i]
    if top["ids"] != want_ids or top["logits"] != want_logits:
        raise SystemExit(f"golden replay mismatch at step {i}")
call(s, "bye")
s.close()
print("golden_replay_ok")
PY
}

WORKDIR="$(mktemp -d)"
trap 'stop_resident "$WORKDIR" 2>/dev/null; rm -rf "$WORKDIR"' EXIT

echo "=== adapter e2e (model=$MODEL, drafter $SPEC_MODE) ==="
for depth in $DEPTHS; do
  echo "--- depth $depth ---"
  SOCK="$(boot_resident "$depth" "$WORKDIR")"
  t0=$(date +%s.%N)
  run1="$(run_phase "$SOCK" "$depth" "$DECODE_COUNT")"
  t1=$(date +%s.%N)
  run2="$(run_phase "$SOCK" "$depth" "$DECODE_COUNT")"
  python3 - "$run1" "$run2" "$depth" "$DECODE_COUNT" "$t0" "$t1" <<'PY'
import json, sys
run1, run2, depth, count, t0, t1 = sys.argv[1:7]
a, b = json.loads(run1), json.loads(run2)
hello, begin, run = a
assert hello.get("backend", "").startswith("llamacpp-resident load_epoch="), hello
assert begin["ok"] and run["ok"]
assert run["committed_total"] == int(count)
assert run["committed_total"] == sum(run["acceptance_lengths"])
assert run["committed_total"] == len(run["tokens"])
assert run["tokens"] == b[2]["tokens"], "determinism failed"
elapsed = float(t1) - float(t0)
tps = int(count) / elapsed if elapsed > 0 else 0.0
print(f"depth={depth} decode_tok_s={tps:.2f} committed={run['committed_total']} drafted={run.get('drafted_total', 0)}")
PY
  stop_resident "$WORKDIR"
done

echo "--- golden replay (serial) ---"
SOCK="$(boot_resident 0 "$WORKDIR")"
golden_replay_serial "$SOCK"
stop_resident "$WORKDIR"

echo "--- teacher forcing replay (serial) ---"
python3 "$ROOT/harness/llama_resident/test/test_teacher_forcing_e2e.py"

echo "=== all depths passed ==="
