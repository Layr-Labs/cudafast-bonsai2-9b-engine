#!/usr/bin/env python3
"""Compare depth-0 resident greedy decode to llama-server on fixed token prompts.

Requires a running or auto-started llama-server (same GGUF, -ngl 99 -fa on -c 8192 -np 1)
and a built llama-resident. Tokenizes with POST /tokenize; oracle via POST /completion
with prompt as token ids (temperature 0, top_k 1, cache_prompt false).

Usage:
  MLXFAST_TARGET_GGUF=... LLAMA_SERVER=... RESIDENT_BIN=... \\
    python3 harness/llama_resident/test/oracle_serial_check.py

Env:
  N_PREDICT     generated tokens to compare (default 128)
  PROMPT_LEN    target prompt token count (default 4096)
  SERVER_URL    if set, do not start server (default starts on 127.0.0.1:18081)
  SKIP_GUTENBERG  if 1, only use the gate prompt + bundled extras under test/oracle_texts/
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEXT_DIR = Path(__file__).resolve().parent / "oracle_texts"
GATE_PATH = ROOT / "correctness_prompts" / "public_longcopy_gate_english_4096.txt"

N_PREDICT = int(os.environ.get("N_PREDICT", "128"))
# llama-server /completion field stop_type when generation ends on an EOS token.
EOS_STOP_TYPE = "eos"
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "4096"))
MODEL = os.path.expanduser(
    os.environ.get(
        "MLXFAST_TARGET_GGUF",
        "",
    )
)
RESIDENT_BIN = os.path.expanduser(
    os.environ.get(
        "RESIDENT_BIN",
        str(ROOT / ".build/llama-resident/bin/llama-resident"),
    )
)
LLAMA_SERVER = os.path.expanduser(
    os.environ.get("LLAMA_SERVER", str(ROOT / ".build/llama-server/bin/llama-server"))
)
SERVER_URL = os.environ.get("SERVER_URL", "")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "18081"))
CTX = int(os.environ.get("RESIDENT_CTX", "8192"))
# Match llama-server common default (common.h) unless the box overrides via env.
RESIDENT_UBATCH = os.environ.get("RESIDENT_UBATCH", "512")
SERVER_UBATCH = os.environ.get("SERVER_UBATCH", RESIDENT_UBATCH)

GUTENBERG = [
    (
        "pride",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
    ),
    (
        "alice",
        "https://www.gutenberg.org/files/11/11-0.txt",
    ),
    (
        "sherlock",
        "https://www.gutenberg.org/files/1661/1661-0.txt",
    ),
    (
        "moby",
        "https://www.gutenberg.org/files/2701/2701-0.txt",
    ),
    (
        "frankenstein",
        "https://www.gutenberg.org/files/84/84-0.txt",
    ),
    (
        "dracula",
        "https://www.gutenberg.org/files/345/345-0.txt",
    ),
]


def http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_health(base: str, proc: subprocess.Popen[str] | None, seconds: int = 180) -> None:
    for _ in range(seconds):
        try:
            http_json(base + "/health")
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
            time.sleep(1)
    raise RuntimeError("llama-server health check timed out")


def start_server() -> tuple[str, subprocess.Popen[str] | None, str | None]:
    if SERVER_URL:
        return SERVER_URL.rstrip("/"), None, None
    work = tempfile.mkdtemp(prefix="oracle-srv-")
    log = f"{work}/server.log"
    cmd = [
        LLAMA_SERVER,
        "-m",
        MODEL,
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        str(CTX),
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
    ]
    cmd.extend(["-ub", SERVER_UBATCH])
    # llama-server's default context checkpoints (hybrid/recurrent models) split prompt processing at
    # checkpoint boundaries, which shifts the numerics; the resident prefills plainly. With them off the
    # two agree bit for bit (gate pos 0 log-prob gap 2.1654 on both; 7/7 prompts x 128 tokens, Spark box 8).
    cmd.extend(["--ctx-checkpoints", "0"])
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{SERVER_PORT}"
    wait_health(base, proc)
    return base, proc, work


def tokenize(base: str, content: str, parse_special: bool = True) -> list[int]:
    r = http_json(
        base + "/tokenize",
        {"content": content, "add_special": False, "parse_special": parse_special},
    )
    return list(r["tokens"])


def fit_tokens(base: str, text: str, n: int) -> list[int]:
    toks = tokenize(base, text)
    if len(toks) >= n:
        return toks[:n]
    # grow by re-tokenizing with padding text (rare for Gutenberg chunks)
    pad = text + "\n" + (" The end. " * ((n - len(toks)) // 2 + 1))
    toks = tokenize(base, pad)
    if len(toks) < n:
        raise RuntimeError(f"could not reach {n} tokens (got {len(toks)})")
    return toks[:n]


def server_greedy(base: str, prompt_tokens: list[int], n_predict: int) -> tuple[list[int], str | None]:
    body = {
        "prompt": prompt_tokens,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,
        "return_tokens": True,
    }
    r = http_json(base + "/completion", body)
    stop_type = r["stop_type"] if "stop_type" in r else None
    return list(r["tokens"]), stop_type


def resident_for_compare(
    oracle: list[int],
    resident: list[int],
    stop_type: str | None,
    n_predict: int,
) -> tuple[list[int], str | None]:
    if len(oracle) >= n_predict:
        return resident, None
    if len(oracle) == 0:
        return resident, "oracle returned empty token list"
    if stop_type is None:
        return resident, "oracle stopped early but completion response missing stop_type"
    if stop_type != EOS_STOP_TYPE:
        return resident, (
            f"oracle stopped early at {len(oracle)} tokens "
            f"(stop_type={stop_type!r}, expected {EOS_STOP_TYPE!r} to trim)"
        )
    return resident[: len(oracle)], None


def resident_greedy(prompt_tokens: list[int], n_predict: int) -> list[int]:
    work = tempfile.mkdtemp(prefix="oracle-res-")
    sock = f"{work}/r.sock"
    ready = f"{work}/ready"
    os.environ["RESIDENT_READY_FILE"] = ready
    res_cmd = [
        RESIDENT_BIN,
        "--model",
        MODEL,
        "--socket",
        sock,
        "--ctx",
        str(CTX),
        "--spec",
        "serial",
        "--draft-len",
        "0",
    ]
    res_cmd.extend(["--n-ubatch", RESIDENT_UBATCH])
    proc = subprocess.Popen(
        res_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(300):
            if os.path.exists(ready):
                break
            if proc.poll() is not None:
                raise RuntimeError("llama-resident exited during boot")
            time.sleep(0.2)

        def call(s: socket.socket, op: str, **kw: object) -> dict:
            s.sendall((json.dumps({"op": op, **kw}) + "\n").encode())
            return json.loads(s.makefile("rb").readline())

        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock)
        assert call(s, "hello")["ok"]
        seed = call(s, "sync", tokens=prompt_tokens)["token"]
        out = [seed]
        t = seed
        for _ in range(n_predict - 1):
            t = call(s, "eval", token=t)["token"]
            out.append(t)
        s.close()
        return out
    finally:
        proc.kill()
        proc.wait()


def load_prompt_sources() -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = [("gate", GATE_PATH.read_text(encoding="utf-8"))]
    if os.environ.get("SKIP_GUTENBERG") == "1":
        return sources
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    seen = {name for name, _ in sources}
    for name, url in GUTENBERG:
        path = TEXT_DIR / f"{name}.txt"
        if not path.exists():
            subprocess.run(
                ["curl", "-fsSL", "-o", str(path), url],
                check=True,
                timeout=120,
            )
        if name in seen:
            continue
        seen.add(name)
        sources.append((name, path.read_text(encoding="utf-8", errors="replace")))
    return sources


def main() -> int:
    if not os.path.isfile(MODEL):
        print(f"missing model: {MODEL}", file=sys.stderr)
        return 2
    if not os.access(RESIDENT_BIN, os.X_OK):
        print(f"missing resident: {RESIDENT_BIN}", file=sys.stderr)
        return 2
    if not SERVER_URL and not os.access(LLAMA_SERVER, os.X_OK):
        print(f"missing llama-server: {LLAMA_SERVER}", file=sys.stderr)
        return 2

    base, srv_proc, _srv_work = start_server()
    rows: list[tuple[str, int | None, int | None, int | None]] = []
    bad = 0
    try:
        sources = load_prompt_sources()
        if len(sources) < 6:
            print(f"need >=6 prompt sources, got {len(sources)}", file=sys.stderr)
            return 2

        print(f"oracle_serial_check model={MODEL} n_predict={N_PREDICT} prompt_len={PROMPT_LEN}")
        print(f"{'prompt':<14} {'len':>5} {'first_diff':>10} {'oracle':>8} {'resident':>8}")
        for name, text in sources:
            prompt = fit_tokens(base, text, PROMPT_LEN)
            oracle, stop_type = server_greedy(base, prompt, N_PREDICT)
            resident = resident_greedy(prompt, N_PREDICT)
            # Trim the resident only when the server ended on EOS (stop_type); other early
            # stops are compared fail-closed. Do not send ignore_eos: it biases EOS to -inf.
            resident, stop_err = resident_for_compare(oracle, resident, stop_type, N_PREDICT)
            if stop_err is not None:
                bad += 1
                rows.append((name, -1, None, None))
                print(f"{name:<14} {len(prompt):>5} {'early_stop':>10} {'-':>8} {'-':>8}")
                print(f"{name}: {stop_err}", file=sys.stderr)
                continue
            diff = None
            for i, (a, b) in enumerate(zip(oracle, resident)):
                if a != b:
                    diff = i
                    break
            if len(oracle) != len(resident):
                diff = diff if diff is not None else min(len(oracle), len(resident))
            if diff is not None:
                bad += 1
                o_tok = oracle[diff] if diff < len(oracle) else None
                r_tok = resident[diff] if diff < len(resident) else None
            else:
                o_tok = r_tok = None
            rows.append((name, diff, o_tok, r_tok))
            fd = diff if diff is not None else "ok"
            print(f"{name:<14} {len(prompt):>5} {str(fd):>10} {str(o_tok):>8} {str(r_tok):>8}")
    finally:
        if srv_proc is not None:
            srv_proc.kill()
            srv_proc.wait()

    if bad:
        print(f"FAIL: {bad} prompt(s) diverged", file=sys.stderr)
        return 1
    print("PASS: all prompts match oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
