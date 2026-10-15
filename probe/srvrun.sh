#!/bin/sh
# usage: srvrun.sh TAG MODEL N EXTRA_JSON -- server args...
# Run from any checkout: resolves the repo root from this script's location.
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "${ROOT}" || exit 1
mkdir -p logs
TAG=$1 M=$2 N=$3 X=$4
shift 5
flock /tmp/mtplx-gpu-exclusive.lock sh -c '
llama.cpp/build/bin/llama-server -m models/'"$M"' -ngl 99 -fa on -c 8192 -np 1 --port 18080 --host 127.0.0.1 "$@" > logs/'"$TAG"'.server.log 2>&1 &
P=$!
for i in $(seq 120); do curl -sf localhost:18080/health >/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 1; done
DUMP=logs/'"$TAG"' python3 probe/determinism.py http://127.0.0.1:18080 '"$N"' '"'$X'"'; echo probe_exit=$?
kill $P; wait $P 2>/dev/null' sh "$@"
