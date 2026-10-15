# Resident validation (rehearsal GGUF)

## Automated checks

| Check | What it proves |
|---|---|
| `harness/llama_resident/test/test_session.cpp` | Bookkeeping with `FakeEngine`: spec vs serial streams, teacher forcing, wire |
| `harness/llama_resident/test/test_adapter_e2e.sh` | Real `cuda-engine` + `llama-resident` at depths 0–3: run-to-run token determinism, counter arithmetic, golden `top_logits` replay |
| `harness/llama_resident/test/oracle_serial_check.py` | Depth-0 greedy decode matches `llama-server` on fixed prompts (4096 seed + 128 decode) |
| `harness/llama_resident/test/test_teacher_forcing_e2e.py` | Serial `eval` replay matches golden `top_logits` bit-for-bit |
| `tools/test-serve-up-boot-stop.sh` | `serve-up.sh` boot/stop and identity checks |

`test_adapter_e2e.sh` prints `decode_tok_s` per depth (two back-to-back
`free_decode_run` phases in one resident). **No fleet-wide e2e throughput sheet
is pinned yet**; use [engine-probe.md](engine-probe.md) llama-bench tg128 as the
serial decode baseline on GB10 (30.9 tok/s PQ2_0, 37.1 tok/s PTQ1_0).

## Determinism and oracle (engine probe + resident)

- Plain llama-server greedy decode: byte-identical across runs and boxes
  ([engine-probe.md](engine-probe.md)).
- Resident serial path is checked against server in `oracle_serial_check.py` on
  bring-up boxes.
- Speculative paths: do **not** assert serial tape == spec tape on GPU; batched
  verify can flip near-ties ([engine-probe.md](engine-probe.md),
  [resident-vs-server.md](resident-vs-server.md)).

## Goldens

Rehearsal pool goldens are recorded with `tools/record-goldens.sh` and pinned in
the fixture. Target state: byte-identical serial + per-depth oracles across two
recording boxes (same procedure as the Nemotron track). Fixture flag
`golden_pins_provisional` remains true until pins are finalized for ranked
scoring.

## Measured results (rehearsal model Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0, DGX Spark GB10)

| Check | Result |
|---|---|
| `test_adapter_e2e.sh`, depths 0 / 1 / 2 / 3, 128 decode tokens | 27.8 / 40.4 / 50.2 / 50.4 tok/s; each depth deterministic run to run; counters consistent |
| Serial golden replay and teacher-forcing replay | `top_logits` bit-identical to the recording |
| `oracle_serial_check.py` (llama-server with `--ctx-checkpoints 0`, 7 prompts × 128 tokens) | 7/7 identical. With the server's default context checkpoints, the server splits prompt processing and near-tie tokens can differ. |
| Serial goldens for 8 fresh 4096-token prompts, recorded on two Sparks | byte-identical |
| benchd `calibrate-baseline`, serial control leg, 4 passes, one Spark | decode CV 0.23%, prefill CV 0.28% |

The resident build from the trimmed vendored tree in the engine image and a native build from the full upstream tree give bit-identical logits.
