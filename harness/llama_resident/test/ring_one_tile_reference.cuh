template <int G, int blocks_per_iter>
static __device__ __forceinline__ void ggml_cuda_dl_work_reference(int & it, int & st, int & rem, int & tile_inc,
                                                         const int nty, const int nkb, const int mode) {
    const int b   = blockIdx.x;
    const int nst = nkb / G;
    if (mode == 2) {
        it = b; st = 0; tile_inc = gridDim.x;
        rem = b < nty ? ((nty - 1 - b)/gridDim.x + 1)*nst : 0;
        return;
    }
    int gs, gs_stop;
    if (mode == 1) {
        gs      = (int) ((int64_t) b      *nty / gridDim.x) * nst;
        gs_stop = (int) ((int64_t) (b + 1)*nty / gridDim.x) * nst;
    } else {
        const int64_t W = (int64_t) nty*nkb;
        int kbc      = (int) ((int64_t) b      *W / gridDim.x);
        int kbc_stop = (int) ((int64_t) (b + 1)*W / gridDim.x);
        kbc      -= (kbc      % nkb) % blocks_per_iter;
        kbc_stop -= (kbc_stop % nkb) % blocks_per_iter;
        gs      = kbc / G;
        gs_stop = kbc_stop / G;
    }
    it = gs / nst; st = gs - it*nst; rem = gs_stop - gs; tile_inc = 1;
}

#pragma once

// GB10 (sm_121) J=16 operand code for PTQ1_0 groups of the decode layout copy (mmq-j16-dl.cuh).
//
// For a PQ2_0 weight whose 2-bit codes q are all in {0, 1, 2} (value (q - 1)*d), the decode layout copy
// holds, per row and K step of 128 values, the PTQ1_0 block (block_ptq1_0, ggml-common.h: 24 + 2 trit
// bytes, then the fp16 d) of the same codes and the same d bits, written with the fork's PTQ1_0 trit codec
// (quantize_row_ptq1_0_ref: base-3 digits, the first in the most significant position, byte =
// ceil(q*256/243), stages of 16 and 8 bytes plus qh). It is value-exact: dequantize_row_ptq1_0 of the
// group equals dequantize_row_pq2_0 of the original block for every value. A tensor with any code 3
// (+2d, not a trit) keeps the 34-byte PQ2_0 groups.
//
// The operands reproduce the int32 dot products of the PQ2_0 kernels exactly. mma m16n8k32 with A = the
// 16 activation columns (s8) and B = 8 weight rows. Inside a 32-value sub-block the fragment ints are in
// natural order (logical int c = physical int c, logical 4 + c = physical 4 + c) for both operands, so
// every mma sums the same 32 products as mul_mat_q<GGML_TYPE_PQ2_0, 16>. B holds the trits t = q in
// {0, 1, 2} as u8 and the accumulator starts at MAGIC - sum(a), from an s8 mma of the same A with B = -1.
// Integer arithmetic is exact, so the result is MAGIC + sum(a*(t - 1)): the accumulator bits of the
// PQ2_0 kernels. The epilogue is the PQ2_0 one: for every sub-block in increasing K,
// sum += cf*dA*dB with cf = (float) idot exactly, dA = (float) d (same fp16 bits), dB = the D4 scale.
//
// Trit decode of a word of trit bytes b: the fork's decoder is t_n(b) = (((b*3^n) mod 256)*3) >> 8. With
// H_k(b) = floor(b*3^k / 256), t_n = H_{n+1} - 3*H_n. b*3^k fits a 16-bit lane for k <= 5, so two
// bytes per register give H_k of four bytes with two multiplies and one byte permute, and t_n of four
// bytes with one IMAD (3*H_n <= 240 and the difference is in {0, 1, 2}: no carry between bytes).

#include "mmq-pq2-j16.cuh"

static_assert(sizeof(block_ptq1_0) == 28, "PTQ1_0 block is 7 words");

static __device__ __forceinline__ void ggml_cuda_ptq1_reference_j16_mma_u8(int (&c)[4], const int a0, const int a1, const int a2,
                                                                const int a3, const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// H_k of the 4 bytes of a word (byte order kept): lo = bytes 0, 2 as 16-bit lanes, hi = bytes 1, 3.
static __device__ __forceinline__ uint32_t ggml_cuda_ptq1_reference_h(const uint32_t lo, const uint32_t hi, const uint32_t m) {
    return __byte_perm(lo*m, hi*m, 0x7351);
}

// Operands of one K step for lane (g, c): activation ints c and 4 + c of every sub-block of columns g and
// g + 8, their D4 scales, and the words c, 4 + (c & 1) and 6 (qh0, qh1, d) of the PTQ1_0 group of row g.
struct ggml_cuda_ptq1_reference_j16_stage {
    int      ya[4][2][2]; // [sub-block][column g + 8h][int c, int 4 + c]
    float4   yd[2];
    uint32_t wa, wb, w6;
};

static __device__ __forceinline__ void ggml_cuda_ptq1_reference_j16_load(
        ggml_cuda_ptq1_reference_j16_stage & st, const char * __restrict__ xr, const char * yb, const int c) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const char * p = yb + h*8*sizeof(block_q8_1_mmq) + 4*sizeof(float) + 32*s + 4*c;
            st.ya[s][h][0] = *(const int *) (p);
            st.ya[s][h][1] = *(const int *) (p + 16);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        st.yd[h] = *(const float4 *) (yb + h*8*sizeof(block_q8_1_mmq));
    }
    const uint32_t * xw = (const uint32_t *) xr;
    st.wa = xw[c];
    st.wb = xw[4 + (c & 1)];
    st.w6 = xw[6];
}

// B fragments (b[s][0] = logical int c, b[s][1] = logical int 4 + c of sub-block s) of weight row g, as trits.
// Value v of the block: v < 80: qs byte v % 16, trit v / 16; 80 <= v < 120: qs byte 16 + (v - 80) % 8,
// trit (v - 80) / 8; v >= 120: qh byte (v - 120) % 2, trit (v - 120) / 2.
//   word c:           t0..t4 = values 16n + 4c.. -> (s0: c, 4+c), (s1: c, 4+c), (s2: c)
//   word 4 + (c & 1): trit n = values 80 + 8n + 4(c & 1).. -> s2 int 4+c: n = c/2, s3 int c: n = 2 + c/2,
//                     s3 int 4+c (c < 2): n = 4
//   qh (c >= 2):      s3 int 4+c = values 120 + 4(c - 2).. = (qh0, qh1) trits a, a + 1 with a = 2(c & 1)
static __device__ __forceinline__ void ggml_cuda_ptq1_reference_j16_decode(uint32_t (&b)[4][2], const ggml_cuda_ptq1_reference_j16_stage & st, const int c) {
    {
        const uint32_t lo = __byte_perm(st.wa, 0, 0x4240);
        const uint32_t hi = __byte_perm(st.wa, 0, 0x4341);
        const uint32_t H1 = ggml_cuda_ptq1_reference_h(lo, hi, 3u);
        const uint32_t H2 = ggml_cuda_ptq1_reference_h(lo, hi, 9u);
        const uint32_t H3 = ggml_cuda_ptq1_reference_h(lo, hi, 27u);
        const uint32_t H4 = ggml_cuda_ptq1_reference_h(lo, hi, 81u);
        const uint32_t H5 = ggml_cuda_ptq1_reference_h(lo, hi, 243u);
        b[0][0] = H1;
        b[0][1] = H2 - 3u*H1;
        b[1][0] = H3 - 3u*H2;
        b[1][1] = H4 - 3u*H3;
        b[2][0] = H5 - 3u*H4;
    }
    uint32_t qh;
    {
        const uint32_t v  = __byte_perm(st.w6, 0, 0x4140); // lanes (qh0, qh1)
        const uint32_t m  = (c & 1) ? 9u : 1u;              // 3^a, a = 2(c & 1)
        const uint32_t L0 = v*m;
        const uint32_t L1 = L0*3u;
        const uint32_t L2 = L0*9u;
        const uint32_t Q0 = __byte_perm(L0, L1, 0x7531);    // (H_a(qh0), H_a(qh1), H_a+1(qh0), H_a+1(qh1))
        const uint32_t Q1 = __byte_perm(L1, L2, 0x7531);
        qh = Q1 - 3u*Q0;
    }
    {
        const uint32_t lo = __byte_perm(st.wb, 0, 0x4240);
        const uint32_t hi = __byte_perm(st.wb, 0, 0x4341);
        const uint32_t H1 = ggml_cuda_ptq1_reference_h(lo, hi, 3u);
        const uint32_t H2 = ggml_cuda_ptq1_reference_h(lo, hi, 9u);
        const uint32_t H3 = ggml_cuda_ptq1_reference_h(lo, hi, 27u);
        const uint32_t H4 = ggml_cuda_ptq1_reference_h(lo, hi, 81u);
        const uint32_t H5 = ggml_cuda_ptq1_reference_h(lo, hi, 243u);
        const bool lo_c = c < 2;
        b[2][1] = lo_c ? H1 : H2 - 3u*H1;
        b[3][0] = lo_c ? H3 - 3u*H2 : H4 - 3u*H3;
        b[3][1] = lo_c ? H5 - 3u*H4 : qh;
    }
}

// SGN: the trits {0, 1, 2} become signed bytes {-1, 0, 1} (per byte (t + 0x7F) ^ 0x80, no carry since t <= 2) and
// one s8.s8 mma accumulates sum(a*(t - 1)) onto MAGIC, the integer the base forms as MAGIC - sum(a) + sum(a*t)
// with two mma; int32 accumulation is exact, so C and every float below are the same bits.
template <bool SGN = false>
static __device__ __forceinline__ void ggml_cuda_ptq1_reference_j16_compute(float (&sum)[1][4], const ggml_cuda_ptq1_reference_j16_stage & st, const int c) {
    // dA of the C rows 2c and 2c+1 of the 8-row tile: the scale of row r was loaded by lanes 4r..4r+3.
    const int xd   = (int) (st.w6 >> 16);
    const int d_lo = __shfl_sync(0xFFFFFFFF, xd, 8*c + 0);
    const int d_hi = __shfl_sync(0xFFFFFFFF, xd, 8*c + 4);
    const float dA[2] = {
        __half2float(__ushort_as_half((unsigned short) d_lo)),
        __half2float(__ushort_as_half((unsigned short) d_hi)),
    };
    const float dB[2][4] = {
        {st.yd[0].x, st.yd[0].y, st.yd[0].z, st.yd[0].w},
        {st.yd[1].x, st.yd[1].y, st.yd[1].z, st.yd[1].w},
    };
    uint32_t b[4][2];
    ggml_cuda_ptq1_reference_j16_decode(b, st, c);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
        if constexpr (SGN) {
            const int b0 = (int) ((b[s][0] + 0x7F7F7F7Fu) ^ 0x80808080u);
            const int b1 = (int) ((b[s][1] + 0x7F7F7F7Fu) ^ 0x80808080u);
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], b0, b1);    // MAGIC + sum(a*(t-1))
        } else {
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], -1, -1);       // MAGIC - sum(a)
            ggml_cuda_ptq1_reference_j16_mma_u8(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], (int) b[s][0], (int) b[s][1]);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float cf = __int_as_float(C[l]) - 12582912.0f; // == (float) idot, exactly
            sum[0][l] += cf*dA[l % 2]*dB[l / 2][s];
        }
    }
}


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
        ggml_cuda_ptq1_reference_j16_stage st;
        ggml_cuda_ptq1_reference_j16_load(st, sx, sy, c);
        ggml_cuda_ptq1_reference_j16_compute<FMT == GGML_CUDA_DL_FMT_PTQ1S>(sum, st, c);
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
    ggml_cuda_dl_work_reference<G, F::BPI>(it, st, rem, tile_inc, nty, nkb, mode);
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
