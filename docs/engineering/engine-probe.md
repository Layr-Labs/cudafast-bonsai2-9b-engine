# Engine probe: PrismML llama.cpp on DGX Spark (GB10)

Measurement date: 2026-10-08. Boxes: Spark box 1, Spark box 3 (GB10 sm_121, driver
580.159.03, CUDA 13.0.88). GPU runs held `/tmp/mtplx-gpu-exclusive.lock`.
Probe scripts: `probe/determinism.py`, `probe/srvrun.sh`.

## Builds

| Box | Ref | Notes |
|---|---|---|
| Spark box 1 | tag `prism-b10754-2459f68` | built from source; no arm64 CUDA prebuilt |
| Spark box 3 | `prism` HEAD `4fbda1292` (includes GDN state fix) | |

`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`; the fork rewrites arch to
`121a`. Build ~3 minutes with `-j16`.

## Speed (llama-bench, batch 1, `-fa 1 -ngl 99`, 5 reps), Bonsai 2 27B

| Packing | Size | pp512 tok/s | tg128 tok/s |
|---|---|---|---|
| PQ2_0 | 6.70 GiB | 1042 ± 40 | 30.9 ± 0.3 |
| PTQ1_0 | 5.53 GiB | 987 ± 8 (HEAD: 1011) | **37.1 ± 0.1** (HEAD: 37.4) |

On GB10, **PTQ1_0 decodes ~20% faster** (bandwidth-bound). PQ2_0 prefills ~5%
faster. Recommend freezing **PTQ1_0** for decode-weighted scoring; re-check on
9B weights.

## Determinism (greedy, llama-server, `-np 1 -c 8192`, `cache_prompt:false`, 3 prompts × 256 tok)

- Plain decode: **byte-identical** token ids across 5 runs, across restart,
  across Spark box 1 vs Spark box 3, and across release tag vs HEAD. Suitable for
  golden recording.
- DSpark speculation: byte-identical run-to-run, but **not identical to plain
  decode** on 1 of 3 prompts (prose diverges at token 55/256). Code and math
  prompts match. Batched verify numerics can flip argmax vs batch-1 decode.
- Correctness under speculation cannot be “token-identical to non-speculative
  reference” without a frozen spec config in the reference or a logit tolerance
  gate.

## Speculative decoding (fork capabilities)

`--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-*`.

- **MTP:** Bonsai 2 27B stock file has no MTP layers; community
  `ProCreations/Ternary-Bonsai-2-27B-MTP` enables `draft-mtp` (see table
  below). Final 9B MTP head TBD with PrismML.
- **DSpark** on Ternary-Bonsai-27B PQ2_0 + converted drafter, `--spec-type
  draft-dspark --spec-draft-n-max 4`:

| Prompt | plain tok/s | DSpark tok/s | accept | speedup |
|---|---|---|---|---|
| code (quicksort) | 30.8 | 82–91 | 80% | ~2.9× |
| prose (sky) | 30.8 | 59.4 | 45% | ~1.9× |
| math | 30.8 | 87.4 | 79% | ~2.8× |

- Community MTP on Bonsai 2 27B PQ2_0, `draft-mtp`:

| Prompt | plain PQ2_0 | MTP n-max 2 (accept) | MTP n-max 3 |
|---|---|---|---|
| code | 30.2 | 54.3 (71%) | 54.2 |
| prose | 30.0 | 51.4 (64%) | 48.0 |
| math | 30.0 | 61.8 (88%) | 65.0 |

MTP output is deterministic run-to-run but differs from plain decode on 2 of 3
prompts (near-tie flips). PQ2_0 vs PTQ1_0 plain outputs also match on only 2
of 3 prompts: frozen packing is part of the reference.

## Cross-check (MLX fleet, 2026-10-08)

Stock Qwen3.5-9B + native MTP head (Q8_0), n-max 3: 53.0 → 102.9 tok/s
(1.95×), ids identical to plain. Interleaved ABAB measurement recommended on
hot GPUs; GB10 ranked path uses calibration bands for the same reason.
