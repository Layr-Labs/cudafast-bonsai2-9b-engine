#pragma once

// Optional Q8 decode ring, split into two 64-row CTAs for each original
// 128-row virtual CTA. K partitions, per-row arithmetic and fixup addresses
// match the original ring exactly. The additional layout copy only permutes
// bytes; the original TM128 copy remains available to every fallback path.
// Validated in decode-tm64-benchmark; PTQ keeps the original kernel.

// Identical to ggml_cuda_dl_work, with explicit virtual launch coordinates.
template <int G, int BPI>
static __host__ __device__ void ggml_cuda_dl_tm64_work(int & it, int & st, int & rem, int & tile_inc,
        int nty, int nkb, int mode, int b, int grid) {
    const int nst = nkb/G;
    int begin, end;
    if (mode == 1) {
        begin = int(int64_t(b)*nty/grid)*nst;
        end = int(int64_t(b+1)*nty/grid)*nst;
    } else {
        const int64_t work = int64_t(nty)*nkb;
        int b0 = int(int64_t(b)*work/grid), b1 = int(int64_t(b+1)*work/grid);
        b0 -= (b0%nkb)%BPI;
        b1 -= (b1%nkb)%BPI;
        begin = b0/G; end = b1/G;
    }
    it = begin/nst; st = begin-it*nst; rem = end-begin; tile_inc = 1;
}

// For SPS=2 every segment and tile boundary is even, so the two consecutive
// K steps of a stage never straddle two TM64 tiles. Checked by the host audit.
static __host__ __device__ int64_t ggml_cuda_dl_tm64_stage_index(int virtual_step, int nst, int half) {
    return (int64_t(virtual_step/nst)*2+half)*nst+virtual_step%nst;
}

template <int FMT, int S, int SPS>
__launch_bounds__(9*32, 2)
static __global__ void mul_mat_q_q8_j16_dl_ring_tm64(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    static_assert(FMT == GGML_CUDA_DL_FMT_Q8 && S == 2 && SPS == 1, "Q8 TM64 only");
    using F = ggml_cuda_dl_fmt<FMT>;
    constexpr int GB = F::GB;
    constexpr int G  = F::G;
    constexpr int I  = 64;
    constexpr int J  = 16;
    constexpr int XB = I*GB;                                       // weights of one CTA step
    constexpr int YB = J*(int) sizeof(block_q8_1_mmq);             // activation slot of one step
    constexpr int SB = SPS*(XB + YB);
    extern __shared__ __align__(128) char ggml_cuda_dl_ring_smem_buf[];
    char     * smem = ggml_cuda_dl_ring_smem_buf;
    uint64_t * full = (uint64_t *) (smem + S*SB);
    uint64_t * empty = full + S;

    const int nst = nkb / G;
    const int nty = nrows_x / 128;
    const int half = blockIdx.x % 2;
    const int virtual_block = blockIdx.x / 2;
    const int virtual_grid = gridDim.x / 2;
    int it, st, rem, tile_inc;
    ggml_cuda_dl_tm64_work<G, F::BPI>(it, st, rem, tile_inc, nty, nkb, mode,
                                virtual_block, virtual_grid);
    const int total   = rem;
    const int nstages = (total + SPS - 1) / SPS;
    const int gs0     = it*nst + st;
    const int tid     = threadIdx.y*32 + threadIdx.x;
    const uint32_t yb_bytes = (uint32_t) (ncols_y*(int) sizeof(block_q8_1_mmq));

    if (tid == 256) {
#pragma unroll
        for (int k = 0; k < S; ++k) {
            ggml_cuda_dl_mbar_init(&full[k], 1);
            ggml_cuda_dl_mbar_init(&empty[k], 8);
        }
        ggml_cuda_dl_mbar_init_fence();
    }
    __syncthreads();
    // Weights of the first S stages before the grid dependency wait (they do not depend on the previous kernel).
    // Weights are read once: evict-first in L2; the activations keep the default policy (every CTA reads them).
    const uint64_t wpol = ggml_cuda_dl_policy_evict_first();
    if (tid == 256) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            ggml_cuda_dl_mbar_expect_tx(&full[k], (uint32_t) n*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + k*SB, x + ggml_cuda_dl_tm64_stage_index(gs0 + k*SPS, nst, half)*XB, (uint32_t) n*XB, &full[k], wpol);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    // Dedicated ninth producer warp; eight consumer warps keep original rows.
    // Empty parity releases a slot only after all consumers snapshot operands.
    if (threadIdx.y == 8) {
        if (threadIdx.x == 0) {
            for (int k = 0; k < nstages; ++k) {
                const int slot = k % S;
                const int nn = min(SPS, total - k*SPS);
                const int gsk = gs0 + k*SPS;
                if (k >= S) {
                    ggml_cuda_dl_mbar_wait(&empty[slot], ((k/S - 1) & 1));
                    ggml_cuda_dl_mbar_expect_tx(&full[slot], (uint32_t) nn*(XB + yb_bytes));
                    ggml_cuda_dl_bulk_g2s_hint(smem + slot*SB,
                        x + ggml_cuda_dl_tm64_stage_index(gsk, nst, half)*XB,
                        (uint32_t) nn*XB, &full[slot], wpol);
                }
                for (int j = 0; j < nn; ++j) {
                    ggml_cuda_dl_bulk_g2s(smem + slot*SB + SPS*XB + j*YB,
                        (const char *) y + (int64_t) ((gsk + j) % nst)*yb_bytes,
                        yb_bytes, &full[slot]);
                }
            }
        }
        return;
    }
    const int w = threadIdx.y;
    const int g = threadIdx.x / 4;
    const int c = threadIdx.x % 4;
    float sum[1][4] = {{0.0f}};
#pragma unroll 1
    for (int k = 0; k < nstages; ++k) {
        const int s = k % S;
        const int n = min(SPS, total - k*SPS);
        ggml_cuda_dl_mbar_wait(&full[s], (uint32_t) ((k / S) & 1));
#pragma unroll
        for (int j = 0; j < SPS; ++j) {
            if (j < n) {
                const char * sx = smem + s*SB + j*XB + (w*8 + g)*GB;
                const char * sy = smem + s*SB + SPS*XB + j*YB + g*(int) sizeof(block_q8_1_mmq);
                ggml_cuda_dl_q8_xstage xs;
                ggml_cuda_dl_q8_ystage ys;
                ggml_cuda_dl_q8_load_x(xs, sx, c);
                ggml_cuda_dl_q8_load_y(ys, sy, c);
                __syncwarp();
                if (threadIdx.x == 0) {
                    const uint32_t addr = ggml_cuda_dl_smem_u32(&empty[s]);
                    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
                }
                ggml_cuda_dl_q8_compute(sum[0], xs, ys, c);
                ++st;
                const bool last = k*SPS + j == total - 1;
                if (st == nst || last) { // end of a segment: the stores of the DL kernels
                    if (st == nst) {
                        const int row = it*128 + half*64 + w*8;
                        if (row < nrows_lo) {
                            ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride_col_dst, ncols_dst - 1);
                        } else {
                            ggml_cuda_pq2_j16_store<1>(sum, dst_hi, row - nrows_lo, stride_col_dst, ncols_dst - 1);
                        }
                        st = 0;
                        it += tile_inc;
                    } else {
                        ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + (int64_t) virtual_block*(J*128), half*64 + w*8, 128, J - 1);
                    }
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        sum[0][l] = 0.0f;
                    }
                }
            }
        }
    }
}
