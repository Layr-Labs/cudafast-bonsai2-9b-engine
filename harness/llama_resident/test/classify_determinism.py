#!/usr/bin/env python3
"""Classify MTP determinism: same-process vs fresh-process (depth 2)."""
import json
import os
import subprocess
import sys
import tempfile
import time

MODEL = os.path.expanduser(
    os.environ.get(
        "MLXFAST_TARGET_GGUF",
        "reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
    )
)
RESIDENT = os.path.expanduser(os.environ["RESIDENT_BIN"])
ADAPTER = os.path.expanduser(os.environ["ADAPTER"])
DEPTH = int(os.environ.get("CLASSIFY_DEPTH", "2"))
COUNT = int(os.environ.get("CLASSIFY_COUNT", "64"))
N = int(os.environ.get("CLASSIFY_N", "3"))
PROMPT = list(range(100, 164))
READY_WAIT_S = int(os.environ.get("CLASSIFY_READY_WAIT_S", "900"))


def stdin_body():
    spec = json.dumps({"mode": "mtp", "mtp": {"depth": DEPTH}})
    return (
        '{"id":1,"kind":"free_decode_begin","seed_tokens":'
        + json.dumps(PROMPT)
        + ',"spec":'
        + spec
        + "}\n"
        '{"id":2,"kind":"free_decode_run","count":'
        + str(COUNT)
        + "}\n"
    )


def run_adapter(sock: str):
    env = os.environ.copy()
    env["BENCH_WORKER_RESIDENT_SOCKET"] = sock
    env.pop("DS4_RESIDENT_SOCKET", None)
    out = subprocess.run(
        [ADAPTER],
        input=stdin_body(),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-3000:])
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    return lines[2]["tokens"]


def boot(workdir: str):
    sock = f"{workdir}/r.sock"
    ready = f"{workdir}/ready"
    os.makedirs(workdir, exist_ok=True)
    for path in (sock, ready):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    env = os.environ.copy()
    env["RESIDENT_READY_FILE"] = ready
    log_path = f"{workdir}/resident.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [
                RESIDENT,
                "--model",
                MODEL,
                "--socket",
                sock,
                "--ctx",
                "8192",
                "--spec",
                "mtp",
                "--draft-len",
                str(DEPTH),
                "--ident",
                "classify",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + READY_WAIT_S
    while time.time() < deadline:
        if os.path.isfile(ready):
            return proc, sock
        if os.path.exists(sock):
            try:
                import socket as pysock

                s = pysock.socket(pysock.AF_UNIX, pysock.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(sock)
                s.close()
                return proc, sock
            except OSError:
                pass
        if proc.poll() is not None:
            with open(log_path) as log:
                raise RuntimeError(log.read()[-4000:])
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"resident not ready after {READY_WAIT_S}s")


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        m = min(len(a), len(b))
        return m, a[m] if m < len(a) else None, b[m] if m < len(b) else None
    return None


def classify_runs(runs):
    ref = runs[0]
    for idx, tokens in enumerate(runs[1:], start=2):
        diff = first_diff(ref, tokens)
        if diff:
            return False, idx, diff
    return True, None, None


def main():
    root = tempfile.mkdtemp(prefix="classify-")
    print(f"workdir={root}", file=sys.stderr)
    proc, sock = boot(root)
    same = [run_adapter(sock) for _ in range(N)]
    proc.kill()
    proc.wait()
    ok_same, run_i, diff = classify_runs(same)
    print("=== SAME PROCESS N=%d ===" % N)
    print("all_match:", ok_same)
    if not ok_same:
        print(f"run {run_i} differs at token index {diff[0]}: {diff[1]} vs {diff[2]}")

    fresh = []
    for i in range(N):
        d = f"{root}/fresh{i}"
        p, s = boot(d)
        fresh.append(run_adapter(s))
        p.kill()
        p.wait()
    ok_fresh, run_i, diff = classify_runs(fresh)
    print("=== FRESH PROCESS N=%d ===" % N)
    print("all_match:", ok_fresh)
    if not ok_fresh:
        print(f"run {run_i} differs at token index {diff[0]}: {diff[1]} vs {diff[2]}")

    if ok_fresh and not ok_same:
        print("CLASSIFICATION: fresh-deterministic, same-process-NOT => state leak between requests")
    elif not ok_fresh and not ok_same:
        print("CLASSIFICATION: both nondeterministic => loop differs per run")
    elif ok_fresh and ok_same:
        print("CLASSIFICATION: both deterministic")
    else:
        print("CLASSIFICATION: same-process ok but fresh not (unexpected)")


if __name__ == "__main__":
    main()
