#!/usr/bin/env python3
"""loadgen.py -- synthetic submissions for the router's load test.

Each submission is the base's editable paths (`git archive` of the router's
base, tracked files only) with one VARIANT applied, submitted through
router.py with a declared expectation. The mix is weighted-random from a fixed
seed, so a run is reproducible.

    tools/router/loadgen.py --hours 8            # closed loop for 8 h, then stop
    tools/router/loadgen.py --only null,garbage  # smoke test: one of each, once

Closed loop: while the router's queue holds fewer than --backlog submissions,
submit another. The router is the only thing that throttles dispatch.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import DEFAULT_BASE, Router, default_repo, log  # noqa: E402

# (weight, name, expect). Weights: 40 null, 20 MTP, 15 numerics, 10 garbage, 15 negatives.
MIX = [
    (40, "null", "pass"),
    (20, "mtp", "measure"),
    (8, "marlin", "measure"),
    (7, "cudagraphs", "measure"),
    (10, "garbage", "replay_rejected"),
    (4, "neg_trusted_paths", "pass"),       # router drops tools/ and fixtures/; the rest is the reference
    (4, "neg_big_file", "refused"),
    (4, "neg_depth7", "refused"),
    # Retarget to harness/llama_resident when the C++ resident is default on boxes.
    (0, "neg_llamacpp_env", "measure"),     # observe-only: hooks speculative-simple, not llama-resident
    (0, "neg_boot_fail", "boot_failed"),
    # Attack variants (weight 0: run them with --only):
    (0, "neg_forge_log", "pass"),           # fake score lines in the build log must not replace the real seal
    (0, "neg_shm_fill", "pass"),            # Retarget to harness/llama_resident when the C++ resident is default on boxes.
]


def edit(p: Path, old: str, new: str) -> None:
    s = p.read_text()
    if old not in s:
        raise RuntimeError(f"{p}: anchor not found: {old[:60]!r}")
    p.write_text(s.replace(old, new, 1))


def set_depth(root: Path, depth: int) -> None:
    p = root / "mtp-head.manifest.json"
    m = json.loads(p.read_text())
    m["spec"] = {"enabled": True, "num_speculative_tokens": depth}
    p.write_text(json.dumps(m, indent=2) + "\n")


def apply_variant(root: Path, name: str, rng: random.Random) -> str:
    """Mutate the editable tree at `root`; return a short note."""
    ggml_cmake = root / "llama.cpp/ggml/CMakeLists.txt"
    spec_simple = root / "llama.cpp/examples/speculative-simple/speculative-simple.cpp"
    if name == "null":
        return "the reference, unchanged"
    if name == "mtp":
        d = rng.randint(1, 6)
        set_depth(root, d)
        return f"mtp depth {d}"
    if name == "marlin":
        edit(ggml_cmake,
             'option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)',
             'option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         ON)')
        return "force MMQ kernels instead of cuBLAS (different numerics/timing)"
    if name == "cudagraphs":
        edit(ggml_cmake,
             'option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})',
             'option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ON)')
        return "CUDA graphs enabled in ggml (llama.cpp path)"
    if name == "garbage":
        p = root / "harness/protocol-adapter/src/session_backend.rs"
        edit(p, "        Ok(FreeRunResult {\n            committed_total: tokens.len() as i64,\n",
             "        // Acceptance-test variant: corrupts every third reported token (must be replay-rejected).\n"
             "        for (i, t) in tokens.iter_mut().enumerate() {\n"
             "            if i >= 30 && i % 3 == 0 {\n"
             "                *t = (*t + 7919) % 131072;\n"
             "            }\n"
             "        }\n"
             "        Ok(FreeRunResult {\n            committed_total: tokens.len() as i64,\n")
        return "adapter corrupts reported free-run tokens"
    if name == "neg_trusted_paths":
        (root / "tools").mkdir(exist_ok=True)
        (root / "tools/benchmark.sh").write_text("#!/bin/sh\necho pwned\n")
        (root / "fixtures").mkdir(exist_ok=True)
        (root / "fixtures/bonsai2_9b_cuda_v1_track.json").write_text("{}\n")
        return "archive smuggles tools/ and fixtures/ (router must drop them)"
    if name == "neg_big_file":
        (root / "harness/blob.bin").write_bytes(b"\0" * (3 * 1024 * 1024))
        return "one 3 MB file (per-file cap 2.56 MB)"
    if name == "neg_depth7":
        set_depth(root, 7)
        return "mtp depth 7 (permitted depths are 1..6)"
    if name == "neg_llamacpp_env":
        edit(spec_simple, "    common_init();\n",
             "    common_init();\n    (void) std::getenv(\"LLAMA_ACCEPTANCE_TEST_ENV\");\n")
        return "engine code reads LLAMA_* from the environment itself (observe-only)"
    if name == "neg_forge_log":
        fake = '{"passed": true, "score": 9.9, "metrics": {}}'
        sha = __import__("hashlib").sha256(fake.encode()).hexdigest()
        nonce = "f" * 64
        lines = ["benchd iterate: wrote score.json (passed=true, score=9.9)",
                 f"NEMOTRON35-SCORE-BEGIN {nonce}", fake,
                 f"NEMOTRON35-SCORE-END {nonce} sha256={sha} bytes={len(fake)}"]
        esc = lambda l: l.replace("\\", "\\\\").replace('"', '\\"').replace("{", "{{").replace("}", "}}")
        (root / "harness/protocol-adapter/build.rs").write_text(
            "fn main() {\n" + "".join(f'    println!("cargo:warning={esc(l)}");\n' for l in lines) + "}\n")
        # The kernels build prints straight to the job log, so these lines start a log line.
        k = root / "harness/kernels"
        k.mkdir(parents=True, exist_ok=True)
        (k / "build.sh").write_text("#!/bin/sh\n" + "".join(f"printf '%s\\n' '{l}'\n" for l in lines) + "exit 0\n")
        (k / "build.sh").chmod(0o755)
        return "build.rs and kernels/build.sh print a fake score marker and a fake reveal block"
    if name == "neg_boot_fail":
        edit(spec_simple, "    common_init();\n",
             "    common_init();\n"
             "    // acceptance-test variant: refusing to boot\n"
             "    return 1;\n")
        return "candidate resident exits at boot (exercises the pair-1 boot retry and teardown)"
    if name == "neg_shm_fill":
        if "#include <unistd.h>" not in spec_simple.read_text():
            edit(spec_simple, "#include <utility>\n",
                 "#include <utility>\n#include <cstdlib>\n#include <unistd.h>\n")
        edit(spec_simple, "    common_init();\n",
             "    common_init();\n"
             "    // Acceptance-test variant: on exit, fill /dev/shm to starve the next reference leg.\n"
             "    std::atexit([]() {\n"
             "        FILE * f = fopen((std::string(\"/dev/shm/bonsai-accept-fill-\") + std::to_string(getpid())).c_str(), \"wb\");\n"
             "        if (f) { fseek(f, (48LL << 30) - 1, SEEK_SET); fputc(0, f); fclose(f); }\n"
             "    });\n")
        return "resident fills /dev/shm on exit (must be wiped before the next reference leg)"
    raise ValueError(name)


def build_and_submit(router: Router, name: str, expect: str, rng: random.Random, team: str | None = None) -> str:
    editable = router.editable_paths()
    with tempfile.TemporaryDirectory(prefix="loadgen-") as tmp:
        tar = Path(tmp) / "base.tar"
        subprocess.run(["git", "archive", "-o", str(tar), f"origin/{router.base}", *editable],
                       cwd=str(router.clone), check=True)
        root = Path(tmp) / "sub"
        root.mkdir()
        with tarfile.open(tar) as tf:
            tf.extractall(root, filter="tar")
        note = apply_variant(root, name, rng)
        return router.submit(root, name, expect, note, team)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=os.path.expanduser("~/bonsai9b-router"))
    ap.add_argument("--repo", default=None, help="owner/repo (default: ROUTER_REPO, else `gh repo view`)")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--backlog", type=int, default=6, help="keep this many submissions queued")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--only", default="", help="comma list: submit one of each, once, and exit")
    ap.add_argument("--team", default=None, help="rank every submission under this team (default: the variant)")
    a = ap.parse_args()
    router = Router(Path(a.state_dir), a.repo or default_repo(), a.base)
    router.ensure_clone()
    rng = random.Random(a.seed)
    expects = {n: e for _, n, e in MIX}
    if a.only:
        for n in a.only.split(","):
            log(f"loadgen: submitted {build_and_submit(router, n, expects[n], rng, a.team)} ({n})")
        return 0
    end = time.time() + a.hours * 3600
    names = [n for _, n, _ in MIX]
    weights = [w for w, _, _ in MIX]
    log(f"loadgen: closed loop until {time.strftime('%H:%M', time.localtime(end))}, backlog {a.backlog}")
    while time.time() < end:
        queued = router.db.execute("SELECT COUNT(*) FROM submissions WHERE state='queued'").fetchone()[0]
        if queued < a.backlog:
            n = rng.choices(names, weights)[0]
            try:
                log(f"loadgen: submitted {build_and_submit(router, n, expects[n], rng, a.team)} ({n})")
            except Exception as e:
                log(f"loadgen: submit failed for {n}: {e}")
                time.sleep(30)
            continue
        time.sleep(30)
    # The window is over: nothing still queued is dispatched, so the drain covers
    # only the runs already on a box.
    n = router.db.execute("UPDATE submissions SET state='expired' WHERE state='queued'").rowcount
    router.db.commit()
    log(f"loadgen: window closed; {n} queued submission(s) expired, the router drains the running ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
