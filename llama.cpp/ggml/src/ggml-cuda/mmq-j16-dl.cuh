#pragma once

#include <memory>

// GB10 (sm_121) decode layout ("DL") of the J=16 MMQ weights, plus the streaming J=16 kernels that read it.
//
// The decode layout is a load-time copy of a PQ2_0 / Q8_0 weight matrix whose 34-byte blocks are only
// permuted: every block is copied byte for byte, nothing is re-encoded, split or re-scaled. A "group" is
// the bytes of one weight row for one K step of 128 values: one PQ2_0 block (34 B) or the four
// consecutive Q8_0 blocks of that step (136 B, kept in their original order). With nst = K/128 steps:
//   original layout: group (row i, step st) at index  i*nst + st
//   TM (tile-major): ((i/128)*nst + st)*128 + i%128                 [tile][step][row]
//   WM (warp-major): (((i/128)*16 + (i%128)/8)*nst + st)*8 + i%8    [tile][8-row group][step][row]
// In TM the 128 rows of one tile and one step are 128 consecutive groups (4352 B for PQ2_0, 17408 B for
// Q8_0), so one CTA step reads one contiguous run and the whole K range of a CTA (stream-k included) is
// one contiguous region.
//
// The layout, the repack and the address/prefetch code are independent of the block size: they are
// parameterized by the group size GB (bytes of one row and one K step: G blocks of BB bytes, e.g. PQ2_0
// 1 x 34 B, Q8_0 4 x 34 B, PTQ1_0 1 x 28 B), so another 128-value-per-step block format reuses them.
//
// The kernels are mul_mat_q_pq2_j16 (mmq-pq2-j16.cuh) and its Q8_0 twin: the same 16-warp CTA, the
// same per-lane operand registers (same load expressions relative to the group), the same stream-k
// partition and tmp_fixup layout, and the same per-element accumulation (sum += cf*dA*dB for every
// 32-value sub-block in increasing K). Only the weight addresses and the L2 prefetch pattern differ,
// so the results are bit-identical to mul_mat_q<type, 16, false> (+ mul_mat_q_stream_k_fixup).
//
// PTQ1_0 groups (C51): a PQ2_0 weight whose 2-bit codes are all in {0, 1, 2} gets, in the same TM order,
// 28-byte PTQ1_0 groups instead of its 34-byte PQ2_0 blocks: each block is re-encoded at load with the
// fork's PTQ1_0 trit codec from the same codes and the same fp16 d bits, which preserves every value
// exactly (mmq-ptq1-j16.cuh), and verified by decoding it back. mul_mat_q_ptq1_j16_dl reads them with the
// PTQ1_0 operand code of mmq-ptq1-j16.cuh (same int32 dot products, same epilogue and order) and the
// work partition, stores and prefetch of mul_mat_q_pq2_j16_dl. A tensor with a code 3 keeps PQ2_0 groups.

#include "mmq-pq2-j16.cuh"
#include "mmq-ptq1-j16.cuh"

#define GGML_CUDA_DL_TM 0
#define GGML_CUDA_DL_WM 1
#define GGML_CUDA_DL_TM64 2

#define GGML_CUDA_DL_H_SECTOR 0 // per-warp 32-byte L2 sector hints of its step region (CH must be 1)
#define GGML_CUDA_DL_H_WBULK  1 // per-warp cp.async.bulk.prefetch.L2 of its step region (CH > 1: WM only)
#define GGML_CUDA_DL_H_CBULK  2 // per-CTA cp.async.bulk.prefetch.L2 of the CTA step region (TM only)

static __device__ __forceinline__ void ggml_cuda_dl_bulk_prefetch(const char * p, const uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(__cvta_generic_to_global(p)), "r"(bytes));
#else
    GGML_UNUSED_VARS(p, bytes);
#endif
}

// L2 prefetch stream of one warp (or of the CTA, CBULK). It walks the CTA's own sequence of steps (the
// steps of its segments in order, across tile boundaries) HD steps ahead of the loads, so a new segment
// starts with its first steps already prefetched. No architectural effect.
template <int LAYOUT, int HMODE, int GB, int CH>
struct ggml_cuda_dl_hint {
    static_assert(HMODE != GGML_CUDA_DL_H_SECTOR || CH == 1, "sector hints are per step");
    static_assert(HMODE != GGML_CUDA_DL_H_CBULK || LAYOUT == GGML_CUDA_DL_TM, "CTA prefetch needs TM");
    static_assert(!(HMODE == GGML_CUDA_DL_H_WBULK && LAYOUT == GGML_CUDA_DL_TM) || CH == 1, "TM warp steps are not contiguous");
    static constexpr int step_bytes = LAYOUT == GGML_CUDA_DL_TM ? 128*GB : 8*GB;

    const char * p;    // region of the step cnt (warp: 8 rows; CBULK: 128 rows)
    int cnt;           // steps of the CTA's sequence passed so far
    int total;         // steps in the CTA's sequence
    int next;          // first step not yet prefetched
    int st;            // step inside the tile
    int nst;
    int64_t tile_jump; // bytes from the end of a tile's run to the start of the next tile's run (0: contiguous)

    // Sequence starts at (tile it0, step st0) and visits tiles it0, it0 + tile_inc, ... ; total steps.
    __device__ __forceinline__ void init(const char * x, const int it0, const int st0, const int total0, const int nst0,
                                         const int w, const int tile_inc) {
        cnt = 0; total = total0; next = 0; st = st0; nst = nst0;
        if constexpr (LAYOUT == GGML_CUDA_DL_TM) {
            p = x + (((int64_t) it0*nst0 + st0)*128 + (HMODE == GGML_CUDA_DL_H_CBULK ? 0 : w*8))*GB;
            tile_jump = (int64_t) (tile_inc - 1)*nst0*128*GB;
        } else {
            p = x + ((((int64_t) it0*16 + w)*nst0 + st0)*8)*GB;
            tile_jump = ((int64_t) 16*tile_inc - 1)*nst0*8*GB;
        }
    }
    __device__ __forceinline__ void issue(const int n) const {
        if constexpr (HMODE == GGML_CUDA_DL_H_SECTOR) {
            // The warp's 8*GB bytes: 9 sectors for PQ2_0 (272 B at 16-B alignment); 8*GB/32 when 8*GB is a multiple
            // of 32 (Q8_0 1088 B, PTQ1_0 224 B: the region is then sector aligned); otherwise one spare sector.
            constexpr int nsec = GB == 34 ? 9 : ((8*GB) % 32 == 0 ? (8*GB)/32 : (8*GB + 31)/32 + 1);
            const int lane = threadIdx.x;
            if (lane < nsec) {
                ggml_cuda_pq2_j16_hint(p + 32*lane);
            }
            if constexpr (nsec > 32) {
                if (lane < nsec - 32) {
                    ggml_cuda_pq2_j16_hint(p + 1024 + 32*lane);
                }
            }
            GGML_UNUSED(n);
        } else if constexpr (HMODE == GGML_CUDA_DL_H_WBULK) {
            if (threadIdx.x == 0) {
                ggml_cuda_dl_bulk_prefetch(p, (uint32_t) (n*8*GB));
            }
        } else {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                ggml_cuda_dl_bulk_prefetch(p, (uint32_t) (n*128*GB));
            }
        }
    }
    __device__ __forceinline__ void advance() {
        ++cnt;
        p += step_bytes;
        if (++st == nst) {
            st = 0;
            p += tile_jump;
        }
    }
    // Steps of one bulk prefetch: contiguous in memory (inside one tile unless TM tiles follow each other).
    __device__ __forceinline__ int chunk(const int n0) const {
        int n = min(n0, total - cnt);
        if (LAYOUT == GGML_CUDA_DL_WM || tile_jump != 0) {
            n = min(n, nst - st);
        }
        return n;
    }
    __device__ __forceinline__ void step() {
        if (cnt < total && cnt >= next) {
            const int n = CH == 1 ? 1 : chunk(CH);
            issue(n);
            next = cnt + n;
        }
        advance();
    }
    // Steps [0, HD) of the sequence, issued before the grid dependency wait (weights only).
    template <int HD>
    __device__ __forceinline__ void prologue() {
        if constexpr (CH == 1 || HMODE == GGML_CUDA_DL_H_SECTOR) {
#pragma unroll
            for (int d = 0; d < HD; ++d) {
                step();
            }
        } else {
            if (cnt < total) {
                const int n = chunk(HD);
                issue(n);
                next = cnt + n;
            }
#pragma unroll
            for (int d = 0; d < HD; ++d) {
                advance();
            }
        }
    }
};

// Work of CTA blockIdx.x: a sequence of segments starting at (tile it, step st), total steps rem, visiting
// tiles it, it + tile_inc, ... (G = 34-byte blocks per step).
//   mode 0: the stream-k partition of mul_mat_q<type, 16, false> with gridDim.x CUDA blocks (gridDim.x ==
//           number of tiles is pure tiling); contiguous range of (tile, step).
//   mode 1: contiguous whole tiles per CTA.  mode 2: tiles b, b + gridDim.x, ... (whole tiles).
// In modes 1 and 2 every tile is one full-K segment of one CTA, exactly as in pure tiling.
template <int G, int blocks_per_iter>
static __device__ __forceinline__ void ggml_cuda_dl_work(int & it, int & st, int & rem, int & tile_inc,
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

// Address of the group (row 8*w + g of tile it, step st).
template <int LAYOUT, int GB>
static __device__ __forceinline__ const char * ggml_cuda_dl_group(
        const char * x, const int it, const int st, const int nst, const int w, const int g) {
    if constexpr (LAYOUT == GGML_CUDA_DL_TM) {
        return x + (((int64_t) it*nst + st)*128 + w*8 + g)*GB;
    } else {
        return x + ((((int64_t) it*16 + w)*nst + st)*8 + g)*GB;
    }
}

// ---------------------------------------------------------------------------------------------
// PQ2_0

template <int LAYOUT, int HMODE, int CH>
static __device__ __forceinline__ void ggml_cuda_dl_pq2_segment(
        float (&sum)[1][4], const char * __restrict__ xr, const char * yb, const int y_step_bytes, const int ns,
        ggml_cuda_dl_hint<LAYOUT, HMODE, (int) sizeof(block_pq2_0), CH> & hs, const int c) {
    constexpr int GB = sizeof(block_pq2_0);
    constexpr int xs = LAYOUT == GGML_CUDA_DL_TM ? 128*GB : 8*GB;
    ggml_cuda_pq2_j16_stage<1> st0;
    ggml_cuda_pq2_j16_stage<1> st1;
    if (ns > 0) {
        ggml_cuda_pq2_j16_load<1>(st0, xr, GB, yb, c);
    }
    for (int k = 0; k < ns; k += 2) {
        hs.step();
        if (k + 1 < ns) {
            ggml_cuda_pq2_j16_load<1>(st1, xr + xs, GB, yb + y_step_bytes, c);
        }
        ggml_cuda_pq2_j16_compute<1>(sum, st0, c);
        if (k + 1 >= ns) {
            break;
        }
        hs.step();
        if (k + 2 < ns) {
            ggml_cuda_pq2_j16_load<1>(st0, xr + 2*xs, GB, yb + 2*y_step_bytes, c);
        }
        ggml_cuda_pq2_j16_compute<1>(sum, st1, c);
        xr += 2*xs;
        yb += 2*y_step_bytes;
    }
}

// grid = (CTAs, 1), block = (32, 16). x is the decode layout copy (LAYOUT) of a PQ2_0 matrix with
// nrows_x rows (multiple of 128) and nkb blocks per row. PDL as mul_mat_q_pq2_j16.
// Stacking: rows [0, nrows_lo) go to dst, rows [nrows_lo, nrows_x) to dst_hi (same column stride), so
// two matrices with the same input (ffn_gate | ffn_up) stored back to back in the decode layout (TM is
// tile-contiguous: the stacked copy is the two copies concatenated) run as one pure-tiling launch. Only
// for pure tiling (mode 1 or 2): every tile is one full-K segment of one CTA, as in the two launches.
// Unstacked: nrows_lo = nrows_x.
template <int LAYOUT, int HMODE, int HD, int CH>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_pq2_j16_dl(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    constexpr int GB = sizeof(block_pq2_0);
    constexpr int I  = 128;
    constexpr int J  = 16;
    const int nst = nkb;
    const int nty = nrows_x / I;
    int it, st, rem, tile_inc;
    ggml_cuda_dl_work<1, MMQ_ITER_K / QK_PQ2_0>(it, st, rem, tile_inc, nty, nkb, mode);

    const int w = threadIdx.y;
    const int g = threadIdx.x / 4;
    const int c = threadIdx.x % 4;

    ggml_cuda_dl_hint<LAYOUT, HMODE, GB, CH> hs;
    hs.init(x, it, st, rem, nst, w, tile_inc);
    hs.template prologue<HD>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    const int y_step_bytes = ncols_y*(int) sizeof(block_q8_1_mmq);
    while (rem > 0) {
        const int ns = min(nst - st, rem);
        const char * xr = ggml_cuda_dl_group<LAYOUT, GB>(x, it, st, nst, w, g);
        const char * yb = (const char *) y + (int64_t) st*y_step_bytes + g*(int) sizeof(block_q8_1_mmq);
        float sum[1][4] = {{0.0f}};
        ggml_cuda_dl_pq2_segment<LAYOUT, HMODE, CH>(sum, xr, yb, y_step_bytes, ns, hs, c);
        if (st + ns == nst) {
            const int row = it*I + w*8;
            if (row < nrows_lo) {
                ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride_col_dst, ncols_dst - 1);
            } else {
                ggml_cuda_pq2_j16_store<1>(sum, dst_hi, row - nrows_lo, stride_col_dst, ncols_dst - 1);
            }
        } else {
            ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + (int64_t) blockIdx.x*(J*I), w*8, I, J - 1);
        }
        rem -= ns;
        st = 0;
        it += tile_inc;
    }
}

// ---------------------------------------------------------------------------------------------
// PTQ1_0 groups of a PQ2_0 weight (C51)

template <int LAYOUT, int HMODE, int CH>
static __device__ __forceinline__ void ggml_cuda_dl_ptq1_segment(
        float (&sum)[1][4], const char * __restrict__ xr, const char * yb, const int y_step_bytes, const int ns,
        ggml_cuda_dl_hint<LAYOUT, HMODE, (int) sizeof(block_ptq1_0), CH> & hs, const int c) {
    constexpr int GB = sizeof(block_ptq1_0);
    constexpr int xs = LAYOUT == GGML_CUDA_DL_TM ? 128*GB : 8*GB;
    ggml_cuda_ptq1_j16_stage st0;
    ggml_cuda_ptq1_j16_stage st1;
    if (ns > 0) {
        ggml_cuda_ptq1_j16_load(st0, xr, yb, c);
    }
    for (int k = 0; k < ns; k += 2) {
        hs.step();
        if (k + 1 < ns) {
            ggml_cuda_ptq1_j16_load(st1, xr + xs, yb + y_step_bytes, c);
        }
        ggml_cuda_ptq1_j16_compute(sum, st0, c);
        if (k + 1 >= ns) {
            break;
        }
        hs.step();
        if (k + 2 < ns) {
            ggml_cuda_ptq1_j16_load(st0, xr + 2*xs, yb + 2*y_step_bytes, c);
        }
        ggml_cuda_ptq1_j16_compute(sum, st1, c);
        xr += 2*xs;
        yb += 2*y_step_bytes;
    }
}

// mul_mat_q_pq2_j16_dl for a copy with PTQ1_0 groups: same arguments, partition, stacking and PDL.
template <int LAYOUT, int HMODE, int HD, int CH>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_ptq1_j16_dl(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    constexpr int GB = sizeof(block_ptq1_0);
    constexpr int I  = 128;
    constexpr int J  = 16;
    const int nst = nkb;
    const int nty = nrows_x / I;
    int it, st, rem, tile_inc;
    ggml_cuda_dl_work<1, MMQ_ITER_K / QK_PQ2_0>(it, st, rem, tile_inc, nty, nkb, mode);

    const int w = threadIdx.y;
    const int g = threadIdx.x / 4;
    const int c = threadIdx.x % 4;

    ggml_cuda_dl_hint<LAYOUT, HMODE, GB, CH> hs;
    hs.init(x, it, st, rem, nst, w, tile_inc);
    hs.template prologue<HD>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    const int y_step_bytes = ncols_y*(int) sizeof(block_q8_1_mmq);
    while (rem > 0) {
        const int ns = min(nst - st, rem);
        const char * xr = ggml_cuda_dl_group<LAYOUT, GB>(x, it, st, nst, w, g);
        const char * yb = (const char *) y + (int64_t) st*y_step_bytes + g*(int) sizeof(block_q8_1_mmq);
        float sum[1][4] = {{0.0f}};
        ggml_cuda_dl_ptq1_segment<LAYOUT, HMODE, CH>(sum, xr, yb, y_step_bytes, ns, hs, c);
        if (st + ns == nst) {
            const int row = it*I + w*8;
            if (row < nrows_lo) {
                ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride_col_dst, ncols_dst - 1);
            } else {
                ggml_cuda_pq2_j16_store<1>(sum, dst_hi, row - nrows_lo, stride_col_dst, ncols_dst - 1);
            }
        } else {
            ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + (int64_t) blockIdx.x*(J*I), w*8, I, J - 1);
        }
        rem -= ns;
        st = 0;
        it += tile_inc;
    }
}

// ---------------------------------------------------------------------------------------------
// Q8_0 (the C39 operand code: one step = 4 Q8_0 blocks of the warp's 8 rows, lane row g, ints 2c, 2c+1)

struct ggml_cuda_dl_q8_xstage {
    int q[4][2]; // ints 2c, 2c+1 of block s
    int d[2];    // half bits of row g: blocks 0,1 in d[0], blocks 2,3 in d[1]
};

// The 8 qs bytes of (block, c) sit at a 2-byte aligned address: aligned 32-bit words + funnel shift
// (the values equal the int8 bytes in memory order, as get_int_b2).
static __device__ __forceinline__ void ggml_cuda_dl_q8_load_x(
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

struct ggml_cuda_dl_q8_ystage {
    int2   ya[4][2]; // activation ints (2c, 2c+1) of sub-block s, column g + 8h
    float4 yd[2];    // D4 scales of column g + 8h
};

static __device__ __forceinline__ void ggml_cuda_dl_q8_load_y(ggml_cuda_dl_q8_ystage & st, const char * yb, const int c) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            st.ya[s][h] = *(const int2 *) (yb + h*8*sizeof(block_q8_1_mmq) + 4*sizeof(float) + 32*s + 8*c);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        st.yd[h] = *(const float4 *) (yb + h*8*sizeof(block_q8_1_mmq));
    }
}

static __device__ __forceinline__ void ggml_cuda_dl_q8_compute(
        float (&sum)[4], const ggml_cuda_dl_q8_xstage & xs, const ggml_cuda_dl_q8_ystage & ys, const int c) {
    // dA of the C rows 2c and 2c+1 (scale of row r loaded by lanes 4r..4r+3), per block s.
    float dA[4][2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int d_lo = __shfl_sync(0xFFFFFFFF, xs.d[p], 8*c + 0);
        const int d_hi = __shfl_sync(0xFFFFFFFF, xs.d[p], 8*c + 4);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            dA[2*p + k][0] = __half2float(__ushort_as_half((unsigned short) (d_lo >> (16*k))));
            dA[2*p + k][1] = __half2float(__ushort_as_half((unsigned short) (d_hi >> (16*k))));
        }
    }
    const float dB[2][4] = {
        {ys.yd[0].x, ys.yd[0].y, ys.yd[0].z, ys.yd[0].w},
        {ys.yd[1].x, ys.yd[1].y, ys.yd[1].z, ys.yd[1].w},
    };
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
        ggml_cuda_pq2_j16_mma(C, ys.ya[s][0].x, ys.ya[s][1].x, ys.ya[s][0].y, ys.ya[s][1].y, xs.q[s][0], xs.q[s][1]);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float cf = __int_as_float(C[l]) - 12582912.0f; // == (float) idot, exactly (|idot| <= 32*127*127 < 2^22)
            sum[l] += cf*dA[s][l % 2]*dB[l / 2][s];
        }
    }
}

template <int LAYOUT, int HMODE, int CH>
static __device__ __forceinline__ void ggml_cuda_dl_q8_segment(
        float (&sum)[4], const char * __restrict__ xr, const char * yb, const int y_step_bytes, const int ns,
        ggml_cuda_dl_hint<LAYOUT, HMODE, 4*(int) sizeof(block_q8_0), CH> & hs, const int c) {
    constexpr int GB = 4*sizeof(block_q8_0);
    constexpr int xs = LAYOUT == GGML_CUDA_DL_TM ? 128*GB : 8*GB;
    ggml_cuda_dl_q8_xstage x0, x1;
    ggml_cuda_dl_q8_ystage y0, y1;
    if (ns > 0) {
        ggml_cuda_dl_q8_load_x(x0, xr, c);
        ggml_cuda_dl_q8_load_y(y0, yb, c);
    }
    for (int k = 0; k < ns; k += 2) {
        hs.step();
        if (k + 1 < ns) {
            ggml_cuda_dl_q8_load_x(x1, xr + xs, c);
            ggml_cuda_dl_q8_load_y(y1, yb + y_step_bytes, c);
        }
        ggml_cuda_dl_q8_compute(sum, x0, y0, c);
        if (k + 1 >= ns) {
            break;
        }
        hs.step();
        if (k + 2 < ns) {
            ggml_cuda_dl_q8_load_x(x0, xr + 2*xs, c);
            ggml_cuda_dl_q8_load_y(y0, yb + 2*y_step_bytes, c);
        }
        ggml_cuda_dl_q8_compute(sum, x1, y1, c);
        xr += 2*xs;
        yb += 2*y_step_bytes;
    }
}

// grid = (CTAs, 1), block = (32, 16). x is the decode layout copy of a Q8_0 matrix (nkb Q8_0 blocks per row).
template <int LAYOUT, int HMODE, int HD, int CH>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_q8_0_j16_dl(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    constexpr int GB = 4*sizeof(block_q8_0);
    constexpr int I  = 128;
    constexpr int J  = 16;
    const int nst = nkb / 4;
    const int nty = nrows_x / I;
    int it, st, rem, tile_inc;
    ggml_cuda_dl_work<4, MMQ_ITER_K / QK8_0>(it, st, rem, tile_inc, nty, nkb, mode);

    const int w = threadIdx.y;
    const int g = threadIdx.x / 4;
    const int c = threadIdx.x % 4;

    ggml_cuda_dl_hint<LAYOUT, HMODE, GB, CH> hs;
    hs.init(x, it, st, rem, nst, w, tile_inc);
    hs.template prologue<HD>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    const int y_step_bytes = ncols_y*(int) sizeof(block_q8_1_mmq);
    while (rem > 0) {
        const int ns = min(nst - st, rem);
        const char * xr = ggml_cuda_dl_group<LAYOUT, GB>(x, it, st, nst, w, g);
        const char * yb = (const char *) y + (int64_t) st*y_step_bytes + g*(int) sizeof(block_q8_1_mmq);
        float sum[1][4] = {{0.0f}};
        ggml_cuda_dl_q8_segment<LAYOUT, HMODE, CH>(sum[0], xr, yb, y_step_bytes, ns, hs, c);
        if (st + ns == nst) {
            const int row = it*I + w*8;
            if (row < nrows_lo) {
                ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride_col_dst, ncols_dst - 1);
            } else {
                ggml_cuda_pq2_j16_store<1>(sum, dst_hi, row - nrows_lo, stride_col_dst, ncols_dst - 1);
            }
        } else {
            ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + (int64_t) blockIdx.x*(J*I), w*8, I, J - 1);
        }
        rem -= ns;
        st = 0;
        it += tile_inc;
    }
}

// ---------------------------------------------------------------------------------------------
// Repack: dst[decode index] = src[original index], one thread per block of BB bytes (BB even), copied as
// halfwords. G = blocks per group (PQ2_0: 1 x 34 B, Q8_0: 4 x 34 B, PTQ1_0: 1 x 28 B). The source is a
// contiguous [row][block] matrix with nkb blocks per row (nkb % G == 0, nrows % 128 == 0).
// Original [row][block] index of the block at decode layout index d.
template <int LAYOUT, int G>
static __device__ __forceinline__ int64_t ggml_cuda_dl_src_index(const int64_t d, const int nkb) {
    const int nst = nkb / G;
    const int s   = (int) (d % G);
    int64_t   q   = d / G;
    int64_t row;
    int st;
    if constexpr (LAYOUT == GGML_CUDA_DL_TM || LAYOUT == GGML_CUDA_DL_TM64) {
        constexpr int rows = LAYOUT == GGML_CUDA_DL_TM64 ? 64 : 128;
        const int r = (int) (q % rows); q /= rows;
        st = (int) (q % nst);
        const int64_t it = q / nst;
        row = it*rows + r;
    } else {
        const int r8 = (int) (q % 8); q /= 8;
        st = (int) (q % nst); q /= nst;
        const int w = (int) (q % 16);
        const int64_t it = q / 16;
        row = it*128 + w*8 + r8;
    }
    return (row*nst + st)*G + s;
}

template <int LAYOUT, int G, int BB>
static __global__ void ggml_cuda_dl_repack_kernel(const char * __restrict__ src, char * __restrict__ dst, const int nrows, const int nkb) {
    static_assert(BB % 2 == 0, "block size");
    const int64_t d    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t nblk = (int64_t) nrows*nkb;
    if (d >= nblk) {
        return;
    }
    const int64_t sidx = ggml_cuda_dl_src_index<LAYOUT, G>(d, nkb);
    const uint16_t * ps = (const uint16_t *) (src + sidx*BB);
    uint16_t       * pd = (uint16_t       *) (dst + d*BB);
#pragma unroll
    for (int k = 0; k < BB/2; ++k) {
        pd[k] = ps[k];
    }
}

// Repack of a PQ2_0 matrix into PTQ1_0 groups (C51): group d = the PTQ1_0 encoding of the original block
// (ggml_cuda_ptq1_encode_block: fork codec, same codes, same d bits, decoded back and compared). *fail != 0
// if any block has a code 3 or does not round-trip; the caller then keeps PQ2_0 groups.
template <int LAYOUT>
static __global__ void ggml_cuda_dl_repack_ptq1_kernel(const char * __restrict__ src, char * __restrict__ dst, const int nrows,
                                                       const int nkb, unsigned int * fail) {
    const int64_t d    = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t nblk = (int64_t) nrows*nkb;
    if (d >= nblk) {
        return;
    }
    const int64_t sidx = ggml_cuda_dl_src_index<LAYOUT, 1>(d, nkb);
    uint32_t o[7];
    if (!ggml_cuda_ptq1_encode_block((const uint16_t *) (src + sidx*(int64_t) sizeof(block_pq2_0)), o)) {
        atomicOr(fail, 1u);
    }
    uint32_t * pd = (uint32_t *) (dst + d*(int64_t) sizeof(block_ptq1_0));
#pragma unroll
    for (int k = 0; k < 7; ++k) {
        pd[k] = o[k];
    }
}

static void ggml_cuda_dl_repack_ptq1(const char * src, char * dst, const int nrows, const int nkb, const int layout,
                                     unsigned int * fail, cudaStream_t stream) {
    const int64_t nblk = (int64_t) nrows*nkb;
    const dim3 grid((unsigned) ((nblk + 255) / 256));
    if (layout == GGML_CUDA_DL_TM) {
        ggml_cuda_dl_repack_ptq1_kernel<GGML_CUDA_DL_TM><<<grid, 256, 0, stream>>>(src, dst, nrows, nkb, fail);
    } else {
        ggml_cuda_dl_repack_ptq1_kernel<GGML_CUDA_DL_WM><<<grid, 256, 0, stream>>>(src, dst, nrows, nkb, fail);
    }
}

template <int G, int BB>
static void ggml_cuda_dl_repack(const char * src, char * dst, const int nrows, const int nkb, const int layout, cudaStream_t stream) {
    const int64_t nblk = (int64_t) nrows*nkb;
    const dim3 grid((unsigned) ((nblk + 255) / 256));
    if (layout == GGML_CUDA_DL_TM) {
        ggml_cuda_dl_repack_kernel<GGML_CUDA_DL_TM, G, BB><<<grid, 256, 0, stream>>>(src, dst, nrows, nkb);
    } else if (layout == GGML_CUDA_DL_TM64) {
        ggml_cuda_dl_repack_kernel<GGML_CUDA_DL_TM64, G, BB><<<grid, 256, 0, stream>>>(src, dst, nrows, nkb);
    } else {
        ggml_cuda_dl_repack_kernel<GGML_CUDA_DL_WM, G, BB><<<grid, 256, 0, stream>>>(src, dst, nrows, nkb);
    }
}

// Preserve row words0..5, pair qh tails then FP16 scales inside the same
// eight existing word6 slots. No added bytes or numerical conversion.
static __global__ void ggml_cuda_dl_ptq_pair_tail_kernel(
        const uint32_t * src, uint32_t * dst, const int64_t words) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < words) {
        const int q = int(i % 56);
        if (q % 7 != 6) {
            dst[i] = src[i];
        } else {
            const int row = q/7;
            const int p = row&3;
            const int shift = row < 4 ? 0 : 16;
            const int64_t b = (i/56)*56 + 14*p + 6;
            dst[i] = ((src[b] >> shift)&0xffffu) | ((src[b+7] >> shift) << 16);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bulk-copy ring (C64, TM layout). One thread streams the CTA's step regions (the 128 rows x GB bytes of one K step,
// contiguous in TM) and the step's activations (ncols_y q8_1 blocks, contiguous) into a ring of S shared-memory
// stages with cp.async.bulk, completing on one mbarrier per stage; the warps read their operands from the stage
// with the same load expressions as the DL kernels. Same work partition, segments, per-element accumulation (the
// same compute calls in increasing K) and stores: the results are bit-identical. The weight copies carry an L2
// evict-first policy (C73; a cache hint only, no effect on values).

#define GGML_CUDA_DL_FMT_PQ2  0
#define GGML_CUDA_DL_FMT_PTQ1 1
#define GGML_CUDA_DL_FMT_Q8   2
#define GGML_CUDA_DL_FMT_PTQ1S 3 // PTQ1_0 with one signed mma per sub-block (ggml_cuda_ptq1_j16_compute<true>)

template <int FMT> struct ggml_cuda_dl_fmt;
template <> struct ggml_cuda_dl_fmt<GGML_CUDA_DL_FMT_PQ2>  { static constexpr int GB = sizeof(block_pq2_0);   static constexpr int G = 1; static constexpr int BPI = MMQ_ITER_K / QK_PQ2_0; };
template <> struct ggml_cuda_dl_fmt<GGML_CUDA_DL_FMT_PTQ1> { static constexpr int GB = sizeof(block_ptq1_0);  static constexpr int G = 1; static constexpr int BPI = MMQ_ITER_K / QK_PQ2_0; };
template <> struct ggml_cuda_dl_fmt<GGML_CUDA_DL_FMT_PTQ1S> : ggml_cuda_dl_fmt<GGML_CUDA_DL_FMT_PTQ1> {};
template <> struct ggml_cuda_dl_fmt<GGML_CUDA_DL_FMT_Q8>   { static constexpr int GB = 4*sizeof(block_q8_0);  static constexpr int G = 4; static constexpr int BPI = MMQ_ITER_K / QK8_0;   };

static __device__ __forceinline__ uint32_t ggml_cuda_dl_smem_u32(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

static __device__ __forceinline__ void ggml_cuda_dl_mbar_init(uint64_t * bar, const uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(ggml_cuda_dl_smem_u32(bar)), "r"(count) : "memory");
#else
    GGML_UNUSED_VARS(bar, count);
#endif
}

static __device__ __forceinline__ void ggml_cuda_dl_mbar_init_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

static __device__ __forceinline__ void ggml_cuda_dl_mbar_expect_tx(uint64_t * bar, const uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(ggml_cuda_dl_smem_u32(bar)), "r"(bytes) : "memory");
#else
    GGML_UNUSED_VARS(bar, bytes);
#endif
}

static __device__ __forceinline__ void ggml_cuda_dl_mbar_wait(uint64_t * bar, const uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    uint32_t done = 0;
    do {
        asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                     : "=r"(done) : "r"(ggml_cuda_dl_smem_u32(bar)), "r"(parity) : "memory");
    } while (!done);
#else
    GGML_UNUSED_VARS(bar, parity);
#endif
}

// bytes (multiple of 16) from global src (16-byte aligned) to shared dst (16-byte aligned), completing on bar
static __device__ __forceinline__ void ggml_cuda_dl_bulk_g2s(void * dst, const void * src, const uint32_t bytes, uint64_t * bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(ggml_cuda_dl_smem_u32(dst)), "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(ggml_cuda_dl_smem_u32(bar))
                 : "memory");
#else
    GGML_UNUSED_VARS(dst, src, bytes, bar);
#endif
}

// The same copy with an L2 cache policy (createpolicy): the ring streams every weight byte exactly once per launch,
// so its lines are allocated evict-first (C73). The L2 then keeps the data the other kernels reuse (activations, KV,
// recurrent state, outputs and partials) instead of filling with weights that are never read again.
static __device__ __forceinline__ void ggml_cuda_dl_bulk_g2s_hint(void * dst, const void * src, const uint32_t bytes, uint64_t * bar,
                                                                  const uint64_t policy) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 :: "r"(ggml_cuda_dl_smem_u32(dst)), "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(ggml_cuda_dl_smem_u32(bar)),
                    "l"(policy)
                 : "memory");
#else
    GGML_UNUSED_VARS(dst, src, bytes, bar, policy);
#endif
}

static __device__ __forceinline__ uint64_t ggml_cuda_dl_policy_evict_first() {
    uint64_t policy = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
#endif
    return policy;
}

// One stage = SPS consecutive K steps of the CTA: their weights (one contiguous TM run) and their activations.
template <int FMT, int SPS>
static __host__ __device__ constexpr int ggml_cuda_dl_ring_stage_bytes() {
    return SPS*(128*ggml_cuda_dl_fmt<FMT>::GB + 16*(int) sizeof(block_q8_1_mmq));
}

template <int FMT, int SPS>
static __host__ __device__ constexpr size_t ggml_cuda_dl_ring_smem(const int S) {
    return (size_t) S*ggml_cuda_dl_ring_stage_bytes<FMT, SPS>() + (size_t) S*sizeof(uint64_t);
}

// Signed PTQ1 FFN variant: keep only one sub-block of activation words and
// scales live. The MMA operands and FP32 accumulation order are unchanged.
static __device__ __forceinline__ void ggml_cuda_ptq1_j16_jit_step(
        float (&sum)[1][4], const char * sx, const char * sy, const int c) {
    ggml_cuda_ptq1_j16_stage st;
    const uint32_t * xw = (const uint32_t *) sx;
    st.wa = xw[c];
    st.wb = xw[4 + (c & 1)];
    st.w6 = xw[6];
    // The original decoder reads only these three initialized weight words.
    uint32_t b[4][2];
    ggml_cuda_ptq1_j16_decode(b, st, c);
    const int xd = int(st.w6 >> 16);
    const int d_lo = __shfl_sync(0xFFFFFFFF, xd, 8*c + 0);
    const int d_hi = __shfl_sync(0xFFFFFFFF, xd, 8*c + 4);
    const float dA[2] = {
        __half2float(__ushort_as_half((unsigned short) d_lo)),
        __half2float(__ushort_as_half((unsigned short) d_hi)),
    };
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        // Volatile shared reads prevent the compiler from reconstructing the
        // original all-four-subblock operand stage. Four words and two scales
        // serve this MMA; values, signed mapping and FP32 expressions are exact.
        const char * p0 = sy + 4*sizeof(float) + 32*s + 4*c;
        const char * p1 = p0 + 8*sizeof(block_q8_1_mmq);
        const int a0 = *(const volatile int *) p0;
        const int a1 = *(const volatile int *) p1;
        const int a2 = *(const volatile int *) (p0 + 16);
        const int a3 = *(const volatile int *) (p1 + 16);
        const float dB[2] = {
            ((const volatile float *) sy)[s],
            ((const volatile float *) (sy + 8*sizeof(block_q8_1_mmq)))[s],
        };
        int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC,
                    GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
        const int b0 = int((b[s][0] + 0x7F7F7F7Fu) ^ 0x80808080u);
        const int b1 = int((b[s][1] + 0x7F7F7F7Fu) ^ 0x80808080u);
        ggml_cuda_pq2_j16_mma(C, a0, a1, a2, a3, b0, b1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float cf = __int_as_float(C[l]) - 12582912.0f;
            sum[0][l] += cf*dA[l % 2]*dB[l / 2];
        }
    }
}

// Operands of one step of lane (g, c) of warp w from a stage, accumulated into sum (the DL kernels' calls).
template <int FMT, bool PAIR_TAIL = false>
static __device__ __forceinline__ void ggml_cuda_dl_ring_step(float (&sum)[1][4], const char * sx, const char * sy, const int c) {
    if constexpr (FMT == GGML_CUDA_DL_FMT_PQ2) {
        ggml_cuda_pq2_j16_stage<1> st;
        ggml_cuda_pq2_j16_load<1>(st, sx, (int) sizeof(block_pq2_0), sy, c);
        ggml_cuda_pq2_j16_compute<1>(sum, st, c);
    } else if constexpr (FMT == GGML_CUDA_DL_FMT_PTQ1 || FMT == GGML_CUDA_DL_FMT_PTQ1S) {
        ggml_cuda_ptq1_j16_stage st;
        ggml_cuda_ptq1_j16_load<PAIR_TAIL>(st, sx, sy, c);
        ggml_cuda_ptq1_j16_compute<FMT == GGML_CUDA_DL_FMT_PTQ1S, PAIR_TAIL>(sum, st, c);
    } else {
        ggml_cuda_dl_q8_xstage xs;
        ggml_cuda_dl_q8_ystage ys;
        ggml_cuda_dl_q8_load_x(xs, sx, c);
        ggml_cuda_dl_q8_load_y(ys, sy, c);
        ggml_cuda_dl_q8_compute(sum[0], xs, ys, c);
    }
}

// Same arguments as mul_mat_q_*_j16_dl (TM copy; mode 0 stream-k or mode 1 contiguous whole tiles), dynamic
// shared memory ggml_cuda_dl_ring_smem<FMT, SPS>(S). Stage layout: weights of the SPS steps, then SPS x 16
// activation blocks.
template <int FMT, int S, int SPS, bool JIT = false, int MIN_CTAS = 1, bool PURE = false, bool PAIR_TAIL = false>
__launch_bounds__(16*32, MIN_CTAS)
static __global__ void mul_mat_q_j16_dl_ring(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int ncols_y, const int stride_col_dst, const int ncols_dst,
        const int mode, float * dst_hi, const int nrows_lo) {
    static_assert(!JIT || FMT == GGML_CUDA_DL_FMT_PTQ1S, "JIT operands require signed PTQ1");
    static_assert(!PAIR_TAIL || (!JIT && FMT == GGML_CUDA_DL_FMT_PTQ1S), "paired tail signed PTQ ring only");
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
    ggml_cuda_dl_work<G, F::BPI>(it, st, rem, tile_inc, nty, nkb, PURE ? 1 : mode);
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
                if constexpr (JIT) {
                    ggml_cuda_ptq1_j16_jit_step(sum, sx, sy, c);
                } else {
                    ggml_cuda_dl_ring_step<FMT, PAIR_TAIL>(sum, sx, sy, c);
                }
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

// ---------------------------------------------------------------------------------------------
#include "mmq-j16-dl-tm64.cuh"
#include "mmq-j16-fixup-pair.cuh"

// Host side (included by mmq.cuh after mmq_args and mmq-pq2-j16.cuh). GB10 only.
//
// The decode layout copies are made once at model load (ggml_backend_cuda_register_decode_layout,
// ggml-cuda.cu -> ggml_cuda_dl_register, mmq.cu) for every eligible PQ2_0 / Q8_0 weight; the original
// tensors stay in place and every other path (prefill MMQ, MMVQ, get_rows, ...) keeps reading them.
// launch_mul_mat_q<type, 16, false> looks the weight up by its data pointer and, when the call is the
// full matrix or a view of its leading rows (no channels, ids), runs the streaming kernel on the copy.
// TM keeps the tiles in row order, so the first nrows_x/128 tiles of the copy are exactly the view's
// rows: a leading-row view (e.g. the DFlash2 drafter's range of the LM head) reads a prefix of the copy.

struct ggml_cuda_dl_entry {
    const char * src;     // original weight data (lookup key)
    char       * dl;      // decode layout (TM) copy
    int64_t      nrows;
    int64_t      ncols;
    ggml_type    type;
    ggml_type    dl_type; // group format of the copy: type, or GGML_TYPE_PTQ1_0 for a PQ2_0 weight (C51)
    char       * dl_tm64 = nullptr; // optional Q8 copy; dl always retains TM128 for fallback
    // DOD QKV entries keep independent lookup keys and point into one allocation.
    // Shared ownership also covers buffers being unregistered in separate calls.
    std::shared_ptr<char> dl_owner;
    char * dl_ptq_pair_tail = nullptr; // optional exact non-JIT head operands
};

// Only measured Q8 FFN and encoder shapes. This is independent of input data;
// smaller attention matrices and every PTQ matrix keep their current kernels.
static bool ggml_cuda_dl_q8_tm64_shape(int64_t rows, int64_t cols) {
    return (rows == 12288 && cols == 4096) ||
           (rows == 4096 && (cols == 12288 || cols == 32768));
}

static bool ggml_cuda_dl_q8_tm64_use(const mmq_args & args, const ggml_cuda_dl_entry & e) {
    return args.type_x == GGML_TYPE_Q8_0 && e.type == GGML_TYPE_Q8_0 && e.dl_type == GGML_TYPE_Q8_0 &&
           e.dl_tm64 != nullptr && args.nrows_x == e.nrows && args.ncols_x == e.ncols &&
           args.ncols_dst == 16 && args.ncols_y == 16 && args.ncols_max == 16 && args.dst_hi == nullptr &&
           ggml_cuda_dl_q8_tm64_shape(args.nrows_x, args.ncols_x);
}

// mmq.cu
const ggml_cuda_dl_entry * ggml_cuda_dl_find(const void * src);
size_t ggml_cuda_dl_register(const void * src, int64_t nrows, int64_t ncols, ggml_type type);
void   ggml_cuda_dl_unregister_range(const void * base, size_t size);

static bool ggml_cuda_mmq_dl_supported(const mmq_args & args, const ggml_cuda_dl_entry & e) {
    const int64_t qk = e.type == GGML_TYPE_PQ2_0 ? QK_PQ2_0 : QK8_0;
    return args.type_x == e.type && args.ids_dst == nullptr && args.expert_bounds == nullptr && args.y_scale == nullptr &&
        args.nchannels_x == 1 && args.nchannels_y == 1 && args.nsamples_x == 1 && args.nsamples_y == 1 &&
        args.nrows_x <= e.nrows && args.ncols_x == e.ncols && args.stride_row_x == e.ncols/qk &&
        args.nrows_x % 128 == 0 && args.ncols_x % MMQ_ITER_K == 0 &&
        args.ncols_dst >= 1 && args.ncols_dst <= 16 && args.ncols_max == args.ncols_dst &&
        args.nrows_dst % 2 == 0 && (uintptr_t) args.dst % 8 == 0 && (uintptr_t) args.y % 16 == 0 &&
        (args.ncols_x/QK8_1_MMQ + 1)*args.ncols_y*(int64_t) sizeof(block_q8_1_mmq) < (int64_t(1) << 31);
}

template <typename Kernel>
static void ggml_cuda_mmq_dl_launch(Kernel kernel, bool * attr_set, const mmq_args & args, const ggml_cuda_dl_entry & e,
                                    const int grid, const int mode, const int nkb, float * tmp_fixup, cudaStream_t stream) {
    // Every warp re-reads the activations through L1: ask for the largest L1 (no shared memory is used).
    const int device = ggml_cuda_get_device();
    if (!attr_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxL1));
        attr_set[device] = true;
    }
    // Programmatic stream serialization (PDL), as launch_mul_mat_q does on GB10.
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(dim3(grid, 1, 1), dim3(32, 16, 1), 0, stream),
        (const char *) e.dl, args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst, mode, args.dst_hi ? args.dst_hi : args.dst,
        (int) (args.dst_hi ? args.nrows_lo : args.nrows_x));
}

template <int FMT, int S, int SPS, bool JIT = false, int MIN_CTAS = 1, bool PURE = false, bool PAIR_TAIL = false>
static void ggml_cuda_mmq_dl_launch_ring(const mmq_args & args, const ggml_cuda_dl_entry & e, const int grid, const int mode,
                                         const int nkb, float * tmp_fixup, cudaStream_t stream) {
    const auto kernel = mul_mat_q_j16_dl_ring<FMT, S, SPS, JIT, MIN_CTAS, PURE, PAIR_TAIL>;
    constexpr size_t smem = ggml_cuda_dl_ring_smem<FMT, SPS>(S);
    static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
    const int device = ggml_cuda_get_device();
    if (!attr[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared));
        attr[device] = true;
    }
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(dim3(grid, 1, 1), dim3(32, 16, 1), smem, stream),
        (const char *) (PAIR_TAIL ? e.dl_ptq_pair_tail : e.dl), args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst, mode, args.dst_hi ? args.dst_hi : args.dst,
        (int) (args.dst_hi ? args.nrows_lo : args.nrows_x));
}

static int ggml_cuda_dl_env_int(const char * name, const int dflt) {
    const char * v = getenv(name);
    return v ? atoi(v) : dflt;
}

// Only the measured pure-tile FFN shapes use the two-CTA operand schedule.
// The original TM128 copy, output ownership and K order are retained.
static bool ggml_cuda_dl_ptq_ffn_jit_use(const mmq_args & args, const ggml_cuda_dl_entry & e,
                                       const int grid, const int nsm) {
    static const bool enabled = ggml_cuda_dl_env_int("GGML_CUDA_PTQ_FFN_JIT", 1) != 0;
    const bool single = args.nrows_x == 12288 && args.dst_hi == nullptr && args.nrows_dst == 12288;
    const bool stacked = args.nrows_x == 24576 && args.dst_hi != nullptr &&
                         args.nrows_lo == 12288 && args.nrows_dst == 12288;
    return enabled && (single || stacked) && args.ncols_x == 4096 &&
           args.nrows_x == e.nrows && args.ncols_x == e.ncols && e.dl != nullptr &&
           args.type_x == GGML_TYPE_PQ2_0 && e.type == GGML_TYPE_PQ2_0 && e.dl_type == GGML_TYPE_PTQ1_0 &&
           args.ncols_y == 16 && args.ncols_dst == 16 && args.ncols_max == 16 &&
           grid == args.nrows_x/128 && grid <= 4*nsm && ggml_cuda_mmq_dl_supported(args, e);
}

static void ggml_cuda_mmq_dl_launch_q8_tm64(const mmq_args & args, const ggml_cuda_dl_entry & e,
                                           const int grid, const int mode, const int nkb,
                                           float * tmp_fixup, cudaStream_t stream) {
    const auto kernel = mul_mat_q_q8_j16_dl_ring_tm64<GGML_CUDA_DL_FMT_Q8, 2, 1>;
    constexpr size_t smem = 2*(64*4*sizeof(block_q8_0) + 16*sizeof(block_q8_1_mmq)) + 2*sizeof(uint64_t);
    static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
    const int device = ggml_cuda_get_device();
    if (!attr[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared));
        attr[device] = true;
    }
    // The doubled physical grid retains the original virtual K partitions and
    // 128-row fixup allocation. Each pair writes disjoint halves of each tile.
    ggml_cuda_kernel_launch(kernel,
        ggml_cuda_kernel_launch_params(dim3(2*grid, 1, 1), dim3(32, 8, 1), smem, stream),
        (const char *) e.dl_tm64, args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst, mode, args.dst, (int) args.nrows_x);
}

// grid: the mul_mat_q grid rule (ggml_cuda_mmq_pq2_j16_grid). Stream-k shapes keep that partition and the
// unchanged mul_mat_q_stream_k_fixup; pure-tiling shapes compute every tile in one CTA (mode 1, one tile
// per CTA). Prefetch per shape class from the microbenchmark (c44 NOTES): PQ2_0 pure tiling: CTA bulk
// prefetch (2 steps ahead; 16 steps in chunks of 8 for many waves, the LM head); PQ2_0 stream-k: CTA bulk
// 2 ahead from 8 MB, warp sector hints 2 ahead below; Q8_0: warp sector hints 2 ahead.
template <ggml_type type>
static void ggml_cuda_mmq_dl_run(const mmq_args & args, const ggml_cuda_dl_entry & e, const int grid, const int nsm,
                                 float * tmp_fixup, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_Q8_0, "PQ2_0 / Q8_0 only");
    constexpr int TM = GGML_CUDA_DL_TM;
    const int nkb    = (int) (args.ncols_x / ggml_cuda_type_traits<type>::qk);
    const int ntiles = (int) (args.nrows_x / 128);
    const bool tiling = grid == ntiles;
    if constexpr (type == GGML_TYPE_PQ2_0) {
        const int64_t bytes = args.nrows_x*nkb*(int64_t) sizeof(block_pq2_0);
        // PTQ1_0 matrices of at least 2048 rows: bulk-copy ring of 2-step stages (c64); GGML_CUDA_DL_RING_PT=0 off.
        static const bool ring_pt = ggml_cuda_dl_env_int("GGML_CUDA_DL_RING_PT", 1) != 0;
        // one signed mma per sub-block instead of two (same integers, ggml_cuda_ptq1_j16_compute);
        // GGML_CUDA_PTQ1_SIGNED=0 keeps the two.
        static const bool ptq1_sgn = ggml_cuda_dl_env_int("GGML_CUDA_PTQ1_SIGNED", 1) != 0;
        if (e.dl_type == GGML_TYPE_PTQ1_0 && ring_pt && args.nrows_x >= 2048 && ptq1_sgn) {
            constexpr int FMT = GGML_CUDA_DL_FMT_PTQ1S;
            if (args.nrows_x == 248320 && e.dl_ptq_pair_tail != nullptr && args.nrows_x == e.nrows && args.ncols_x == e.ncols &&
                    tiling && ntiles > 4*nsm) {
                ggml_cuda_mmq_dl_launch_ring<FMT, 2, 2, false, 1, false, true>(args, e, grid, 1, nkb, tmp_fixup, stream);
            } else if (ggml_cuda_dl_ptq_ffn_jit_use(args, e, grid, nsm)) {
                ggml_cuda_mmq_dl_launch_ring<FMT, 3, 2, true, 2, true>(args, e, grid, 1, nkb, tmp_fixup, stream);
            } else if (tiling && ntiles > 4*nsm) {
                ggml_cuda_mmq_dl_launch_ring<FMT, 2, 2>(args, e, grid, 1, nkb, tmp_fixup, stream);
            } else {
                ggml_cuda_mmq_dl_launch_ring<FMT, 3, 2>(args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
            }
        } else if (e.dl_type == GGML_TYPE_PTQ1_0 && ring_pt && args.nrows_x >= 2048) {
            constexpr int FMT = GGML_CUDA_DL_FMT_PTQ1;
            if (tiling && ntiles > 4*nsm) {
                ggml_cuda_mmq_dl_launch_ring<FMT, 2, 2>(args, e, grid, 1, nkb, tmp_fixup, stream);
            } else {
                ggml_cuda_mmq_dl_launch_ring<FMT, 3, 2>(args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
            }
        } else if (e.dl_type == GGML_TYPE_PTQ1_0) {
            // PTQ1_0 groups (C51). Prefetch from the c51 microbenchmark: CTA bulk prefetch, 16 steps in chunks
            // of 8 for many waves of pure tiling (the LM head), 3 steps ahead for every other shape.
            if (tiling && ntiles > 4*nsm) {
                static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
                ggml_cuda_mmq_dl_launch(mul_mat_q_ptq1_j16_dl<TM, GGML_CUDA_DL_H_CBULK, 16, 8>, attr, args, e, grid, 1, nkb, tmp_fixup, stream);
            } else {
                static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
                ggml_cuda_mmq_dl_launch(mul_mat_q_ptq1_j16_dl<TM, GGML_CUDA_DL_H_CBULK, 3, 1>, attr, args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
            }
        } else if (tiling && ntiles > 4*nsm) {
            static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
            ggml_cuda_mmq_dl_launch(mul_mat_q_pq2_j16_dl<TM, GGML_CUDA_DL_H_CBULK, 16, 8>, attr, args, e, grid, 1, nkb, tmp_fixup, stream);
        } else if (tiling || bytes >= (int64_t(8) << 20)) {
            static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
            ggml_cuda_mmq_dl_launch(mul_mat_q_pq2_j16_dl<TM, GGML_CUDA_DL_H_CBULK, 2, 1>, attr, args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
        } else {
            static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
            ggml_cuda_mmq_dl_launch(mul_mat_q_pq2_j16_dl<TM, GGML_CUDA_DL_H_SECTOR, 2, 1>, attr, args, e, grid, 0, nkb, tmp_fixup, stream);
        }
    } else {
        // Q8_0 matrices of at least 4096 rows: bulk-copy ring, 2 stages of one step (c64 microbenchmark);
        // GGML_CUDA_DL_RING=0 keeps the warp-load kernel.
        static const bool ring     = ggml_cuda_dl_env_int("GGML_CUDA_DL_RING", 1) != 0;
        static const int  ring_min = ggml_cuda_dl_env_int("GGML_CUDA_DL_RING_MIN_ROWS", 4096);
        if (ring && args.nrows_x >= ring_min) {
            if (ggml_cuda_dl_q8_tm64_use(args, e)) {
                ggml_cuda_mmq_dl_launch_q8_tm64(args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
            } else {
                ggml_cuda_mmq_dl_launch_ring<GGML_CUDA_DL_FMT_Q8, 2, 1>(args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
            }
        } else {
            static bool attr[GGML_CUDA_MAX_DEVICES] = { false };
            ggml_cuda_mmq_dl_launch(mul_mat_q_q8_0_j16_dl<TM, GGML_CUDA_DL_H_SECTOR, 2, 1>, attr, args, e, grid, tiling ? 1 : 0, nkb, tmp_fixup, stream);
        }
    }
    if (!tiling) {
        if constexpr (type == GGML_TYPE_Q8_0) {
            const int BPI=ggml_cuda_mmq_get_K_vram(type,16,false,GGML_CUDA_CC_DGX_SPARK)/QK8_0;
            if (grid==48 && args.nrows_x==4096 && nkb>=16*BPI && nkb%BPI==0) {
                ggml_cuda_kernel_launch(mul_mat_q_q8_dl_fixup_pair,
                    ggml_cuda_kernel_launch_params(dim3(32,4),dim3(32,4),0,stream),
                    args.dst,(const float *)tmp_fixup,(int)args.nrows_dst,(int)args.ncols_dst);
                return;
            }
        }
        const uint3 blocks_per_ne00_fd = init_fastdiv_values(nkb);
        const uint3 one_fd             = init_fastdiv_values(1);
        // Compile the48-SM partition divisor into the DL-only fixup; generic MMQ launches stay unchanged.
        const auto fixup = grid == 48
            ? (args.nrows_x <= 1024 ? mul_mat_q_stream_k_fixup<type, 16, false, 48, true>
                                   : mul_mat_q_stream_k_fixup<type, 16, false, 48>)
            : mul_mat_q_stream_k_fixup<type, 16, false>;
        ggml_cuda_kernel_launch(fixup,
            ggml_cuda_kernel_launch_params(dim3(grid, 128/32, 1), dim3(32, 4, 1), 0, stream),
            nullptr, nullptr, args.dst, tmp_fixup, blocks_per_ne00_fd, (int) args.nrows_x, (int) args.ncols_dst,
            (int) args.nrows_dst, one_fd, (int) args.stride_channel_dst, one_fd, (int) args.stride_sample_dst, one_fd);
    }
}


// D1: the same signed-PTQ1 ring arithmetic as mmq-j16-dl.cuh. The combined
// mode partitions all 80 tiles over 48 CTAs (fp32 reassociation). GROUPED
// preserves the separate 64/8/8-tile partitions, each over 48 CTAs.
struct ggml_cuda_dod_qkv_args {
    const char * x;
    const int * y;
    float * dst;
    float * dst_hi;
    float * dst_hi2;
    int stride[3];
    int ncols;
};

static_assert(ggml_cuda_mmq_get_config_gb10(GGML_TYPE_PQ2_0, 16, false).I == 128, "D1 uses the original 128-row tiles");
static_assert(ggml_cuda_mmq_get_config_gb10(GGML_TYPE_PQ2_0, 16, false).nthreads/2 == 128, "D1 fixup uses four warps");
static_assert(ggml_cuda_mmq_get_config_gb10(GGML_TYPE_PQ2_0, 16, false).K_vram == MMQ_ITER_K, "D1 keeps the original K alignment");

static __host__ __device__ __forceinline__ int ggml_cuda_dod_qkv_start(int b, int rows, int nkb) {
    int kb = int(int64_t(b)*(rows/128)*nkb/48);
    kb -= (kb % nkb) % (MMQ_ITER_K/QK_PQ2_0);
    return kb;
}

static __device__ __forceinline__ float * ggml_cuda_dod_qkv_dst(
        const ggml_cuda_dod_qkv_args & a, int & row, int & stride) {
    if (row < 8192) {
        stride = a.stride[0];
        return a.dst;
    }
    if (row < 9216) {
        row -= 8192;
        stride = a.stride[1];
        return a.dst_hi;
    }
    row -= 9216;
    stride = a.stride[2];
    return a.dst_hi2;
}

template <bool GROUPED>
__launch_bounds__(16*32, 1)
static __global__ void mul_mat_q_dod_qkv_ring(ggml_cuda_dod_qkv_args a, float * tmp_fixup) {
    constexpr int GB = sizeof(block_ptq1_0), I = 128, J = 16, S = 3, SPS = 2;
    constexpr int K = 4096, nkb = K/QK_PQ2_0, XB = I*GB;
    constexpr int YB = J*int(sizeof(block_q8_1_mmq));
    constexpr int SB = ggml_cuda_dl_ring_stage_bytes<GGML_CUDA_DL_FMT_PTQ1S, SPS>();
    const int problem = GROUPED ? int(blockIdx.x)/48 : 0;
    const int b = GROUPED ? int(blockIdx.x)%48 : int(blockIdx.x);
    const int rows = GROUPED ? (problem == 0 ? 8192 : 1024) : 10240;
    const int row_base = GROUPED ? (problem == 0 ? 0 : 8192 + (problem - 1)*1024) : 0;
    const int begin = ggml_cuda_dod_qkv_start(b, rows, nkb);
    const int total = ggml_cuda_dod_qkv_start(b + 1, rows, nkb) - begin;
    int it = begin/nkb, st = begin % nkb;
    const int nstages = (total + SPS - 1)/SPS;
    const char * x = a.x + int64_t(row_base)*nkb*GB;
    const int * y = a.y;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const uint32_t yb_bytes = uint32_t(a.ncols*int(sizeof(block_q8_1_mmq)));
    extern __shared__ __align__(128) char ggml_cuda_dod_qkv_smem[];
    char * smem = ggml_cuda_dod_qkv_smem;
    uint64_t * full = (uint64_t *)(smem + S*SB);
    if (tid == 0) {
#pragma unroll
        for (int k = 0; k < S; ++k) { ggml_cuda_dl_mbar_init(&full[k], 1); }
        ggml_cuda_dl_mbar_init_fence();
    }
    __syncthreads();
    const uint64_t wpol = ggml_cuda_dl_policy_evict_first();
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            ggml_cuda_dl_mbar_expect_tx(&full[k], uint32_t(n)*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + k*SB, x + int64_t(begin + k*SPS)*XB, uint32_t(n)*XB, &full[k], wpol);
        }
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    if (tid == 0) {
        for (int k = 0; k < S && k < nstages; ++k) {
            const int n = min(SPS, total - k*SPS);
            for (int j = 0; j < n; ++j) {
                const int stj = (begin + k*SPS + j) % nkb;
                ggml_cuda_dl_bulk_g2s(smem + k*SB + SPS*XB + j*YB, (const char *)y + int64_t(stj)*yb_bytes, yb_bytes, &full[k]);
            }
        }
    }
    const int w = threadIdx.y, g = threadIdx.x/4, c = threadIdx.x%4;
    float sum[1][4] = {{0.0f}};
#pragma unroll 1
    for (int k = 0; k < nstages; ++k) {
        const int s = k%S, n = min(SPS, total - k*SPS);
        ggml_cuda_dl_mbar_wait(&full[s], uint32_t((k/S)&1));
#pragma unroll
        for (int j = 0; j < SPS; ++j) {
            if (j < n) {
                const char * sx = smem + s*SB + j*XB + (w*8 + g)*GB;
                const char * sy = smem + s*SB + SPS*XB + j*YB + g*int(sizeof(block_q8_1_mmq));
                ggml_cuda_dl_ring_step<GGML_CUDA_DL_FMT_PTQ1S>(sum, sx, sy, c);
                ++st;
                if (st == nkb || k*SPS + j == total - 1) {
                    if (st == nkb) {
                        int row = row_base + it*I + w*8, stride;
                        float * dst = ggml_cuda_dod_qkv_dst(a, row, stride);
                        ggml_cuda_pq2_j16_store<1>(sum, dst, row, stride, a.ncols - 1);
                        st = 0;
                        ++it;
                    } else {
                        ggml_cuda_pq2_j16_store<1>(sum, tmp_fixup + int64_t(blockIdx.x)*(J*I), w*8, I, J - 1);
                    }
#pragma unroll
                    for (int l = 0; l < 4; ++l) { sum[0][l] = 0.0f; }
                }
            }
        }
        if (k + S < nstages) { __syncthreads(); }
        if (tid == 0 && k + S < nstages) {
            const int kk = k + S, nn = min(SPS, total - kk*SPS), gsk = begin + kk*SPS;
            ggml_cuda_dl_mbar_expect_tx(&full[s], uint32_t(nn)*(XB + yb_bytes));
            ggml_cuda_dl_bulk_g2s_hint(smem + s*SB, x + int64_t(gsk)*XB, uint32_t(nn)*XB, &full[s], wpol);
            for (int j = 0; j < nn; ++j) {
                ggml_cuda_dl_bulk_g2s(smem + s*SB + SPS*XB + j*YB,
                    (const char *)y + int64_t((gsk + j)%nkb)*yb_bytes, yb_bytes, &full[s]);
            }
        }
    }
}

// Same descending CTA-partial addition and final dst += sum as the original
// mul_mat_q_stream_k_fixup<PQ2_0,16,false,48>. One launch for all destinations.
template <bool GROUPED>
__launch_bounds__(128, 1)
static __global__ void mul_mat_q_dod_qkv_fixup(ggml_cuda_dod_qkv_args a, const float * partials) {
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    constexpr int nkb = 4096/QK_PQ2_0;
    const int problem = GROUPED ? int(blockIdx.x)/48 : 0;
    const int b = GROUPED ? int(blockIdx.x)%48 : int(blockIdx.x);
    const int rows = GROUPED ? (problem == 0 ? 8192 : 1024) : 10240;
    const int row_base = GROUPED ? (problem == 0 ? 0 : 8192 + (problem - 1)*1024) : 0;
    const int begin = ggml_cuda_dod_qkv_start(b, rows, nkb);
    const int end = ggml_cuda_dod_qkv_start(b + 1, rows, nkb);
    if (begin == end || begin%nkb == 0 || (begin/nkb == end/nkb && end%nkb != 0)) { return; }
    float sum[4] = {0.0f};
    int stop = begin;
    for (int prev = b - 1; prev >= 0; --prev) {
        const int start = ggml_cuda_dod_qkv_start(prev, rows, nkb);
        if (start != stop) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum[j] += partials[int64_t((GROUPED ? problem*48 : 0) + prev)*2048 +
                                   (4*j + threadIdx.y)*128 + blockIdx.y*32 + threadIdx.x];
            }
            if (start%nkb == 0 || start/nkb < begin/nkb) { break; }
        }
        stop = start;
    }
    int row = row_base + (begin/nkb)*128 + blockIdx.y*32 + threadIdx.x, stride;
    float * dst = ggml_cuda_dod_qkv_dst(a, row, stride);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = 4*j + threadIdx.y;
        if (col < a.ncols) { dst[col*stride + row] += sum[j]; }
    }
}

// Compact only the live grouped-QKV fixup CTAs. Geometry is fixed by
// ggml_cuda_dod_qkv_args: Q=8192 rows, K/V=1024 rows, K=4096,48 partitions each.
__launch_bounds__(128, 1)
static __global__ void mul_mat_q_dod_qkv_fixup_compact(ggml_cuda_dod_qkv_args a, const float * partials) {
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
    constexpr int nkb=4096/QK_PQ2_0;
    static_assert(nkb==32 && MMQ_ITER_K/QK_PQ2_0==2,"compact fixed partition");
    const int compact=blockIdx.x;
    const bool q=compact<32;
    const int problem=q?0:1+(compact-32)/8;
    const int local=q?compact:(compact-32)%8;
    const int b=q?3*(local/2)+1+local%2:6*local+5;
    const int tile=q?(4*b)/3:local;
    float sum[4]={0.0f};
    const int count=q?1:5;
#pragma unroll
    for (int off=1;off<=5;++off) {
        if (off<=count) {
#pragma unroll
            for (int j=0;j<4;++j) {
                sum[j]+=partials[int64_t(problem*48+b-off)*2048+
                    (4*j+threadIdx.y)*128+blockIdx.y*32+threadIdx.x];
            }
        }
    }
    const int base=q?0:8192+(problem-1)*1024;
    int row=base+tile*128+blockIdx.y*32+threadIdx.x,stride;
    float * dst=ggml_cuda_dod_qkv_dst(a,row,stride);
#pragma unroll
    for (int j=0;j<4;++j) {
        const int col=4*j+threadIdx.y;
        if (col<a.ncols) dst[col*stride+row]+=sum[j];
    }
}

template <bool GROUPED>
static void ggml_cuda_dod_qkv_launch(ggml_cuda_dod_qkv_args a, float * partials, cudaStream_t stream) {
    constexpr int grid = GROUPED ? 144 : 48;
    constexpr size_t smem = ggml_cuda_dl_ring_smem<GGML_CUDA_DL_FMT_PTQ1S, 2>(3);
    const auto kernel = mul_mat_q_dod_qkv_ring<GROUPED>;
    static bool attr[GGML_CUDA_MAX_DEVICES] = {false};
    const int device = ggml_cuda_get_device();
    if (!attr[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, int(cudaSharedmemCarveoutMaxShared)));
        attr[device] = true;
    }
    ggml_cuda_kernel_launch(kernel, ggml_cuda_kernel_launch_params(dim3(grid), dim3(32,16), smem, stream), a, partials);
    if constexpr (GROUPED) {
        ggml_cuda_kernel_launch(mul_mat_q_dod_qkv_fixup_compact,
            ggml_cuda_kernel_launch_params(dim3(48,4),dim3(32,4),0,stream),a,partials);
    } else {
        ggml_cuda_kernel_launch(mul_mat_q_dod_qkv_fixup<false>,
            ggml_cuda_kernel_launch_params(dim3(grid,4),dim3(32,4),0,stream),a,partials);
    }
    CUDA_CHECK(cudaGetLastError());
}
