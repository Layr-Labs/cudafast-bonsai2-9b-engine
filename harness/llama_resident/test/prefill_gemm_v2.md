# D1 prefill GEMM v2: paired B matrix loads

`GGML_CUDA_PREFILL_GEMM_V2=1` enables adjacent-column B pairing in the D1
plane gate/up consumer (F16 and F32 SwiGLU) and the stream-K J128 TMAX
consumer (plain and residual ADD). The default is **off**. Other consumers
keep their existing specializations.

Each warp and K128 block previously issued eight A `ldmatrix.x4`, sixteen B
`ldmatrix.x2`, and 32 integer MMAs. Pairing replaces two adjacent B `.x2`
loads with one `.x4`: **24 matrix loads become 16**, with unchanged operand
bytes and MMA count. Lanes 0–15 provide the original column tile's two
matrix addresses; lanes 16–31 provide those of the next column tile. The
four output registers are the original B fragments in order. See NVIDIA's
[ldmatrix specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix).

This targets matrix-load issue/dependency overhead in the measured
long-scoreboard/wait profile. The second column fragment becomes available
before its MMAs without another load instruction. It adds two live B
registers at source level. Shared memory remains 50,176 bytes, with the
same eight warps, double buffers, TMA copies and barriers. The intended
residency is still two CTAs per SM. Check compiler allocation: spilling
could erase the gain, as it did in the earlier D4 pairing experiment.

Every output element sees the same four integer MMA updates in sub-block
order, then the **unchanged** two-FFMA D1 epilogue for each K128 block in K
order. Scale loads, stream-K segment bounds and partial layout, fixup
arithmetic, residual ADD, SwiGLU and F16 rounding/clamping are unchanged.

A screening estimate is 2–4% less time in the two GEMM families, or
`51 ms * 0.02–0.04 = 1.02–2.04 ms` per 75 ms prefill. With unchanged decode,
that is approximately **+0.34–0.69% composite**. These are assumptions to
test on GB10, not measurements; the same shared-memory traffic and barriers
remain.

## Build and run on GB10

Use the pinned CUDA 13 engine image/toolchain, with a writable build
directory **outside the source tree**. From the repository root:

```bash
cmake -S harness/llama_resident -B /tmp/prefill-gemm-v2 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_RESIDENT_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build /tmp/prefill-gemm-v2 --target prefill-gemm-v2-equivalence -j 8
ctest --test-dir /tmp/prefill-gemm-v2 -R '^prefill-gemm-v2-' \
  --output-on-failure --no-tests=error
/tmp/prefill-gemm-v2/prefill-gemm-v2-equivalence --enabled --launches 40
/tmp/prefill-gemm-v2/prefill-gemm-v2-equivalence --enabled --cold-l2 --launches 40
```

The test always runs explicit baseline **and** v2 templates in the same
process. `--enabled` / `--disabled` additionally check the real environment
dispatch by graph function identity; no flag checks the absent/default-off
case. All three use production-shaped D1 inputs at T=512.

There are seven cases: plane 24576x4096 in F16 and F32, J128 8192x4096,
4096x4096, 4096x12288, and the latter two with fused ADD. J128 uses the
production 48 stream-K partitions / 96 CTAs and the production fixup.
Every case compares guarded output words and all stream-K scratch words
over A/B/zero/A graph replays. A small GPU witness first checks all B fragment
registers from `.x4` against the original two `.x2` instructions.

Timing alternates AB/BA order and prints medians over 24 launches per
variant by default (minimum 20). Events are recorded inside the replayed
graphs, excluding host enqueue gaps. J128 includes its fixup. Cold mode rotates
four independently populated, registered weight copies, and touches an
eviction buffer of max(64 MiB, four times device L2) before **each** timed
operation. Uploads, fills, packing, eviction and verification are outside
the CUDA events.

Expect two `GEMM_V2_RESOURCE` lines, one `GEMM_V2_EXACT` line with
`mismatch_words=0`, and one `GEMM_V2_BENCH` line per case. Resource output
includes registers, local bytes and CTAs/SM; inspect for zero local bytes
and two CTAs/SM before promotion. Timing fields are `baseline_us`, `v2_us`,
`speedup`, `saved_us`. Success ends with
`PREFILL_GEMM_V2_EQUIVALENCE_PASS shapes=7 mismatch_words=0`; any mismatch
exits nonzero. CUDA compilation, equality and speed still require the GB10
run. This test does not claim full-model token equivalence or a ranked gain.
