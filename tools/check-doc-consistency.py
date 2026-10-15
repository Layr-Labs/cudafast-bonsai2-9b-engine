#!/usr/bin/env python3
"""check-doc-consistency.py -- the docs must say what the manifest says.

The reference track shipped a TASK.md that listed three editable paths and
called the engine read-only while benchmark.json listed four, including the
engine; it disagreed with its own contract about re-quantizing the head, and
it pinned three draft depths while permitting six. This check is the tripwire
for that class of drift. It reads the values from benchmark.json and the track
fixture and requires every participant-facing document to state them, and it
refuses the retired track's vocabulary outside the provenance documents.

Exit 0 when consistent, 1 with every finding listed otherwise.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FINDINGS: list[str] = []

RETIRED_REPLAY_KEYS = (
    "timed_replay_max_logit_gap",
    "timed_replay_off_argmax_per_thousand",
)


def need(cond: bool, msg: str) -> None:
    if not cond:
        FINDINGS.append(msg)


def main() -> int:
    manifest = json.loads((ROOT / "benchmark.json").read_text())
    fixture = json.loads((ROOT / manifest["contractPath"]).read_text())
    docs = {
        name: (ROOT / name).read_text()
        for name in ("TASK.md", "README.md", "AGENTS.md", "docs/participant-contract.md")
    }

    editable = manifest["editablePaths"]
    depths = fixture["mtp_head"]["permitted_draft_depths"]
    scoring = manifest["scoring"]
    track = manifest["trackId"]

    # 1. Every participant document names the track and every editable path.
    for name, text in docs.items():
        need(track in text, f"{name}: does not name the track id {track}")
    for name in ("TASK.md", "AGENTS.md", "docs/participant-contract.md"):
        for path in editable:
            need(f"`{path}`" in docs[name], f"{name}: does not name editable path `{path}`")
    # 1b. The count, where a document states one, is the manifest's.
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    for name, text in docs.items():
        for m in re.finditer(r"(one|two|three|four|five|\d) (?:editable paths|entries)", text):
            n = m.group(1)
            need(n in (str(len(editable)), words.get(len(editable), "")),
                 f"{name}: says '{m.group(0)}' but editablePaths has {len(editable)}")

    # 2. The depth envelope.
    lo, hi = min(depths), max(depths)
    need(depths == list(range(lo, hi + 1)), "fixture: permitted_draft_depths is not contiguous")
    need(f"from {lo} to {hi}" in docs["TASK.md"], f"TASK.md: does not state depths from {lo} to {hi}")
    need(f"{lo} to {hi}" in docs["docs/participant-contract.md"],
         f"docs/participant-contract.md: does not state depths {lo} to {hi}")
    labels = [f"mtp{d}" for d in depths]
    if "dflash" in fixture["allowed_modes"]:
        drafter = fixture["dflash_drafter"]
        dflash_depths = drafter["permitted_draft_depths"]
        need(dflash_depths == list(range(1, drafter["block_size"])),
             "fixture: dflash_drafter.permitted_draft_depths is not 1 to block_size - 1")
        need(any(f["path"] == drafter["file"] for f in fixture["target"]["files"]),
             f"fixture: dflash_drafter.file {drafter['file']} has no target.files pin")
        dlo, dhi = min(dflash_depths), max(dflash_depths)
        for name in ("TASK.md", "docs/participant-contract.md"):
            need("dflash" in docs[name] and f"{dlo} to {dhi}" in docs[name],
                 f"{name}: does not state the dflash drafter and its depths {dlo} to {dhi}")
        labels += [f"dflash{d}" for d in dflash_depths]
    serve_depths = sorted(fixture["serve_configuration"]["depths"])
    need(serve_depths == sorted(["serial"] + labels),
         f"fixture: serve_configuration.depths {serve_depths} do not cover serial + every permitted depth")
    rotation_pool = fixture.get("live_goldens") or fixture.get("live_golden_rotation", {}).get("pool", [fixture.get("live_golden")])
    for prompt in rotation_pool:
        need(sorted(fixture.get("speculative_oracles", {}).get(prompt, {})) == sorted(labels),
             f"fixture: speculative_oracles.{prompt} does not carry exactly one oracle per permitted depth")
    # 2b. The scored prompt is drawn per job; the participant documents say so.
    if "live_golden_rotation" in fixture:
        for name in ("TASK.md", "README.md", "docs/participant-contract.md"):
            need("drawn at random" in docs[name],
                 f"{name}: does not say the scored prompt is drawn at random per job (live_golden_rotation)")

    # 2c. Every job measures every live golden; the participant documents say so.
    if "live_goldens" in fixture:
        for name in ("TASK.md", "docs/participant-contract.md"):
            need("every live golden" in docs[name],
                 f"{name}: does not say every job measures every live golden (live_goldens)")
            need(f"{fixture['seed_tokens']}-token" in docs[name],
                 f"{name}: does not state the {fixture['seed_tokens']}-token prompt (seed_tokens)")

    # 3. Scoring constants appear in TASK.md and the contract.
    tol = fixture.get("timed_token_tolerance_per_thousand")
    for name in ("TASK.md", "docs/participant-contract.md"):
        text = docs[name]
        need(f"prefill_gain ^ {scoring['scoredExponents']['prefillGainExponent']}" in text
             and f"decode_gain ^ {scoring['scoredExponents']['decodeGainExponent']}" in text,
             f"{name}: the composite formula disagrees with scoredExponents")
        need(str(scoring["prefillSpeedupFloor"]) in text, f"{name}: missing the floor {scoring['prefillSpeedupFloor']}")
        need(f"`{scoring['kvBackend']}`" in text, f"{name}: missing the kv backend `{scoring['kvBackend']}`")
        need(f"{fixture['official_pairs']} pairs" in text, f"{name}: does not state {fixture['official_pairs']} pairs")
        need("official_pair_combine" in text and "mean" in text,
             f"{name}: does not state official_pair_combine mean scoring")
        if scoring.get("decodeSpeedupCeiling") is not None:
            need(False, f"benchmark.json scoring.decodeSpeedupCeiling must not be set (no official ceiling gate)")
        if tol is None:
            need("exact match" in text.lower() or "exact-match" in text.lower(),
                 f"{name}: fixture declares no timed_token_tolerance_per_thousand; must state timed exact match")
            for key in RETIRED_REPLAY_KEYS:
                if key in fixture and key not in fixture.get("_pending_step5_replay_policy", {}):
                    need(False, f"fixture: active {key} without timed_token_tolerance_per_thousand")
                if f"`{key}`" in text:
                    need("_pending_step5_replay_policy" in text,
                         f"{name}: cites active `{key}` but fixture has no timed_token_tolerance_per_thousand")
        else:
            need(str(tol) in text,
                 f"{name}: fixture timed_token_tolerance_per_thousand={tol} must appear in the doc")
    need(fixture["kv_backend"] == scoring["kvBackend"], "fixture kv_backend disagrees with benchmark.json")
    for key in RETIRED_REPLAY_KEYS:
        need(key not in fixture, f"fixture: {key} must live only under _pending_step5_replay_policy")

    # 4. The head rules agree everywhere.
    requant = fixture["mtp_head"]["requantization_permitted"]
    for name, text in docs.items():
        if not requant:
            need(not re.search(r"may re-quantize (it|the head|either)", text, re.I),
                 f"{name}: permits re-quantizing the head; the fixture forbids it")
    need(fixture["mtp_head"]["bytes"] <= fixture["mtp_head"]["declaration_max_bytes"],
         "fixture: the pinned head is larger than the declaration cap")
    decl = json.loads((ROOT / "mtp-head.manifest.json").read_text())
    need(decl.get("max_bytes", 0) <= fixture["mtp_head"]["declaration_max_bytes"],
         "mtp-head.manifest.json: max_bytes exceeds the fixture's declaration cap")

    # 5. The retired ds4 / qwen3.8 leaderboard vocabulary stays out of the
    #    participant-facing documents (retired Nemotron/Qwen leaderboard strings).
    retired = re.compile(
        r"\b(ds4|Gemma|macmon|libds4qwen|qwen3\.8-125b|qwen-3\.8-mtp-challenge)\b",
        re.I,
    )
    benchd_names = re.compile(
        r"DS4_RESIDENT_SOCKET|MLXFAST_QWEN_MTP_TRACK_ID|qwen3\.8-125b-a6b(-cuda)?-v1|"
        r"qwen-mtp-ranked-benchmark|qwen signer|qwen/bench|retired qwen|ds4-era"
    )
    for rel in ("benchmark.json", *docs):
        text = (ROOT / rel).read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            scrubbed = benchd_names.sub("", line)
            m = retired.search(scrubbed)
            if m:
                FINDINGS.append(f"{rel}:{lineno}: retired-track term '{m.group(0)}'")

    if FINDINGS:
        print(f"check-doc-consistency: {len(FINDINGS)} finding(s)")
        for f in FINDINGS:
            print(f"  - {f}")
        return 1
    print("check-doc-consistency: the docs agree with benchmark.json and the fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
