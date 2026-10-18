// Frozen promoted03f1054 Q8 loader, ring step and producer.
static __device__ __forceinline__ void ggml_cuda_dl_q8_load_x_reference(
        ggml_cuda_dl_q8_xstage & st, const char * __restrict__ xb, const int c) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const char     * qsi = xb + s*(int) sizeof(block_q8_0) + sizeof(ggml_half) + 8*c;
        const int        mis = (int) (reinterpret_cast<uintptr_t>(qsi) & 3); // 0 or 2
        const uint32_t * qwi = reinterpret_cast<const uint32_t *>(qsi - mis);
        const uint32_t   w0  = qwi[0];
        const uint32_t   w1  = qwi[1];
        const uint32_t   w2  = qwi[1 + mis/2]; // w1 again when aligned
        st.q[s][0] = (int) __funnelshift_r(w0, w1, 8*mis);
        st.q[s][1] = (int) __funnelshift_r(w1, w2, 8*mis);
    }
    const uint32_t d0 = *(const uint16_t *) (xb + 0*sizeof(block_q8_0));
    const uint32_t d1 = *(const uint16_t *) (xb + 1*sizeof(block_q8_0));
    const uint32_t d2 = *(const uint16_t *) (xb + 2*sizeof(block_q8_0));
    const uint32_t d3 = *(const uint16_t *) (xb + 3*sizeof(block_q8_0));
    st.d[0] = (int) (d0 | (d1 << 16));
    st.d[1] = (int) (d2 | (d3 << 16));
}

template <int FMT>
static __device__ __forceinline__ void ggml_cuda_dl_ring_step_reference(float (&sum)[1][4], const char * sx, const char * sy, const int c) {
    if constexpr (FMT == GGML_CUDA_DL_FMT_PQ2) {
        ggml_cuda_pq2_j16_stage<1> st;
        ggml_cuda_pq2_j16_load<1>(st, sx, (int) sizeof(block_pq2_0), sy, c);
        ggml_cuda_pq2_j16_compute<1>(sum, st, c);
    } else if constexpr (FMT == GGML_CUDA_DL_FMT_PTQ1 || FMT == GGML_CUDA_DL_FMT_PTQ1S) {
        ggml_cuda_ptq1_j16_stage st;
        ggml_cuda_ptq1_j16_load(st, sx, sy, c);
        ggml_cuda_ptq1_j16_compute<FMT == GGML_CUDA_DL_FMT_PTQ1S>(sum, st, c);
    } else {
        ggml_cuda_dl_q8_xstage xs;
        ggml_cuda_dl_q8_ystage ys;
        ggml_cuda_dl_q8_load_x_reference(xs, sx, c);
        ggml_cuda_dl_q8_load_y(ys, sy, c);
        ggml_cuda_dl_q8_compute(sum[0], xs, ys, c);
    }
}


template <int FMT, int S, int SPS>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_j16_dl_ring_reference(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    using F = ggml_cuda_dl_fmt<FMT>;
    constexpr int GB = F::GB;
    constexpr int G  = F::G;
    constexpr int I  = 128;
    constexpr int J  = 16;
    constexpr int XB = I*GB;                                       // weights of one CTA step
    constexpr int YB = J*(int) sizeof(block_q8_1_mmq);             // activation slot of one step
    constexpr int SB = ggml_cuda_dl_ring_stage_bytes<FMT, SPS>();
    extern __shared__ __align__(128) char ggml_cuda_dl_ring_smem_buf[];
    char     * smem = ggml_cuda_dl_ring_smem_buf;
    uint64_t * full = (uint64_t *) (smem + S*SB);

    const int nst = nkb / G;
    const int nty = nrows_x / I;
    int it, st, rem, tile_inc;
    ggml_cuda_dl_work<G, F::BPI>(it, st, rem, tile_inc, nty, nkb, mode);
    const int total   = rem;
    const int nstages = (total + SPS - 1) / SPS;
    const int gs0     = it*nst + st;
    const int tid     = threadIdx.y*32 + threadIdx.x;
    const uint32_t yb_bytes = (uint32_t) (ncols_y*(int) sizeof(block_q8_1_mmq));

    if (tid == 0) {
#pragma unroll
        for (int k = 0; k < S; ++k) {
            ggml_cuda_dl_mbar_init(&full[k], 1);
        }
        ggml_cuda_dl_mbar_init_fence();
    }
    __syncthreads();
    // Weights of the first S stages before the grid dependency wait (they do not depend on the previous kernel).
    // Weights are read once: evict-first in L2; the activations keep the default policy (every CTA reads them).
    const uint64_t wpol = ggml_cuda_dl_policy_evict_first();
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            ggml_cuda_dl_mbar_expect_tx(&full[k], (uint32_t) n*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + k*SB, x + (int64_t) (gs0 + k*SPS)*XB, (uint32_t) n*XB, &full[k], wpol);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            for (int j = 0; j < n; ++j) {
                const int stj = (gs0 + k*SPS + j) % nst;
                ggml_cuda_dl_bulk_g2s(smem + k*SB + SPS*XB + j*YB, (const char *) y + (int64_t) stj*yb_bytes, yb_bytes, &full[k]);
            }
        }
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
                ggml_cuda_dl_ring_step_reference<FMT>(sum, sx, sy, c);
                ++st;
                const bool last = k*SPS + j == total - 1;
                if (st == nst || last) { // end of a segment: the stores of the DL kernels
                    if (st == nst) {
                        const int row = it*I + w*8;
                        if (row < nrows_lo) {
                            ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride_col_dst, ncols_dst - 1);
                        } else {
                            ggml_cuda_pq2_j16_store<1>(sum, dst_hi, row - nrows_lo, stride_col_dst, ncols_dst - 1);
                        }
                        st = 0;
                        it += tile_inc;
                    } else {
                        ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + (int64_t) blockIdx.x*(J*I), w*8, I, J - 1);
                    }
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        sum[0][l] = 0.0f;
                    }
                }
            }
        }
        // Only reused stages need all warps to finish before the next bulk copy.
        // This condition is uniform across the CTA; no stage is read after its last use.
        if (k + S < nstages) {
            __syncthreads();
        }
        if (tid == 0 && k + S < nstages) {
            const int kk = k + S;
            const int nn = min(SPS, total - kk*SPS);
            const int gsk = gs0 + kk*SPS;
            ggml_cuda_dl_mbar_expect_tx(&full[s], (uint32_t) nn*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + s*SB, x + (int64_t) gsk*XB, (uint32_t) nn*XB, &full[s], wpol);
            for (int j = 0; j < nn; ++j) {
                ggml_cuda_dl_bulk_g2s(smem + s*SB + SPS*XB + j*YB, (const char *) y + (int64_t) ((gsk + j) % nst)*yb_bytes, yb_bytes, &full[s]);
            }
        }
    }
}
