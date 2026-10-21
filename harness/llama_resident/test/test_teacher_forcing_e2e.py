#!/usr/bin/env python3
"""E2e: replay a recorded serial eval tape (including forced tokens) and match top_logits.

Run under flock /tmp/mtplx-gpu-exclusive.lock on ranked boxes.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = os.path.expanduser(
    os.environ.get(
        "MLXFAST_TARGET_GGUF",
        "reference_weights/Ternary-Bonsai-2-9B-gguf/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
    )
)
RESIDENT_BIN = os.path.expanduser(
    os.environ.get(
        "RESIDENT_BIN",
        str(ROOT / ".build/llama-resident/bin/llama-resident"),
    )
)
PROMPT = json.loads(
    os.environ.get("TEACHER_PROMPT_JSON", "[" + ",".join(str(i) for i in range(100, 164)) + "]")
)
STEPS = int(os.environ.get("TEACHER_STEPS", "64"))
CTX = int(os.environ.get("RESIDENT_CTX", "8192"))


def call(s: socket.socket, op: str, **kw) -> dict:
    s.sendall((json.dumps({"op": op, **kw}) + "\n").encode())
    return json.loads(s.makefile("rb").readline())


def boot(sock: str, ready: str, log: str) -> subprocess.Popen[str]:
    for p in (sock, ready, log):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    os.environ["RESIDENT_READY_FILE"] = ready
    log_fp = open(log, "w")
    proc = subprocess.Popen(
        [
            RESIDENT_BIN,
            "--model",
            MODEL,
            "--socket",
            sock,
            "--ctx",
            str(CTX),
            "--spec",
            "serial",
            "--ident",
            "teacher-forcing-e2e",
        ],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(900):
        if os.path.exists(ready):
            return proc
        if proc.poll() is not None:
            tail = Path(log).read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"resident exited during boot:\n{tail}")
        time.sleep(0.5)
    proc.kill()
    tail = Path(log).read_text(encoding="utf-8", errors="replace")[-4000:]
    raise RuntimeError(f"resident not ready:\n{tail}")


def record(sock: str) -> tuple[list[int], list[tuple[list[int], list[float]]]]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    assert call(s, "hello")["ok"]
    frontier = call(s, "sync", tokens=PROMPT)["token"]
    fed: list[int] = []
    logits_tape: list[tuple[list[int], list[float]]] = []
    for step in range(STEPS):
        top = call(s, "top_logits", k=8)
        logits_tape.append((top["ids"], top["logits"]))
        if step % 11 == 5:
            token = (frontier + 17) % 500 + 50
        else:
            token = frontier
        fed.append(token)
        frontier = call(s, "eval", token=token)["token"]
    call(s, "bye")
    s.close()
    return fed, logits_tape


def replay(sock: str, fed: list[int], want_logits: list[tuple[list[int], list[float]]]) -> None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    assert call(s, "hello")["ok"]
    call(s, "sync", tokens=PROMPT)
    for i, token in enumerate(fed):
        top = call(s, "top_logits", k=8)
        want_ids, want_log = want_logits[i]
        if top["ids"] != want_ids or top["logits"] != want_log:
            raise SystemExit(f"logits mismatch at step {i}")
        call(s, "eval", token=token)
    call(s, "bye")
    s.close()


def main() -> int:
    if not os.path.isfile(MODEL):
        print(f"missing model: {MODEL}", file=sys.stderr)
        return 2
    if not os.access(RESIDENT_BIN, os.X_OK):
        print(f"missing resident: {RESIDENT_BIN}", file=sys.stderr)
        return 2

    work = Path(os.environ.get("TEACHER_WORKDIR", "/tmp/teacher-forcing-e2e"))
    work.mkdir(parents=True, exist_ok=True)
    sock = str(work / "r.sock")
    ready = str(work / "ready")

    log = str(work / "resident.log")
    proc = boot(sock, ready, log)
    try:
        fed, tape = record(sock)
        replay(sock, fed, tape)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("teacher_forcing_e2e_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
