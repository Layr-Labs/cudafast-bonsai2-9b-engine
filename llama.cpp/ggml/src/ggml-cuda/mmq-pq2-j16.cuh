#pragma once

// GB10 (sm_121) J=16 PQ2_0 x q8_1 matmul without the X shared-memory tile.
//
// Bit-exact replacement for mul_mat_q<GGML_TYPE_PQ2_0, 16, false> (+ its stream-k fixup):
//  * same work partition: the grid is (old grid, row slices). CTA (b, q) runs every segment that
//    stream-k block b of the old kernel runs, for the rows of slice q of each 128-row tile, and
//    writes the same destination (dst for a segment that ends its tile, tmp_fixup[b] for the
//    trailing one); the unchanged mul_mat_q_stream_k_fixup completes the stream-k tiles.
//  * same accumulation per output element: for every 32-value sub-block in increasing K order,
//    sum += (float) idot * dA * dB with dA = (float) x->d and dB = the q8_1 D4 scale. idot is the
//    exact int32 dot product of the same 32 products (the K order inside a sub-block is permuted
//    identically for both operands). The int -> float conversion starts the mma accumulator at
//    0x4B400000 (= 12582912.0f) and subtracts 12582912.0f: |idot| <= 32*2*127 < 2^22, so the result
//    is exactly (float) idot.
// Each warp loads its own PQ2 bytes from global memory and unpacks them straight into mma
// fragments (m16n8k32 with A = the 16 activation columns, B = 8 weight rows); the activations
// (L1/L2 resident) are read from global memory by every warp. No shared memory, no barriers.

#include "common.cuh"

#include <cstdint>

#define GGML_CUDA_PQ2_J16_MAGIC 0x4B400000 // 12582912.0f = 1.5*2^23

static __device__ __forceinline__ void ggml_cuda_pq2_j16_mma(int (&c)[4], const int a0, const int a1, const int a2,
                                                             const int a3, const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// Same 2-bit -> int8 codec expressions as ggml_cuda_mmq_load_tiles_pq2_0 (q = 8 packed values).
static __device__ __forceinline__ void ggml_cuda_pq2_j16_unpack(const int q, int & qx, int & qy) {
    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    qx = __byte_perm(qe, qo, 0x5140);
    qy = __byte_perm(qe, qo, 0x7362);
}

// Fragment mapping (lane = 4*g + c). Inside a 32-value sub-block, the mma's logical int c of
// both operands is physical int 2c and its logical int 4 + c is physical int 2c + 1, so:
//   A (activations): column g (+8): the two ints 2c, 2c+1 of the sub-block = one 8-byte load;
//   B (weights):     row g of the 8-row tile: bytes 2c, 2c+1 of the sub-block's 8 bytes = one
//                    16-bit load, unpacked into the 4+4 values of those two ints;
//   C: c0 = (col g, row 2c), c1 = (col g, row 2c+1), c2 = (col g+8, row 2c), c3 = (col g+8, row 2c+1).
template <int NT>
struct ggml_cuda_pq2_j16_stage {
    int2   ya[4][2];  // activation ints (2c, 2c+1) of sub-block s, column g + 8h
    float4 yd[2];     // activation D4 scales of column g + 8h
    int    xq[NT][4]; // weight bytes (2c, 2c+1) of sub-block s, row 8t + g
    int    xd;        // weight scale bits of row g (t = 0 in the low half, t = 1 in the high half)
};

template <int NT>
static __device__ __forceinline__ void ggml_cuda_pq2_j16_load(
        ggml_cuda_pq2_j16_stage<NT> & st, const char * __restrict__ xr, const int stride_row_bytes,
        const char * yb, const int c) {
    static_assert(NT == 1 || NT == 2, "NT");
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
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int16_t * q = (const int16_t *) (xr + t*8*stride_row_bytes + sizeof(ggml_half)) + c;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            st.xq[t][s] = q[4*s];
        }
    }
    if constexpr (NT == 1) {
        st.xd = *(const uint16_t *) xr;
    } else {
        const uint32_t d0 = *(const uint16_t *) xr;
        const uint32_t d1 = *(const uint16_t *) (xr + 8*stride_row_bytes);
        st.xd = d0 | (d1 << 16);
    }
}

template <int NT>
static __device__ __forceinline__ void ggml_cuda_pq2_j16_compute(
        float (&sum)[NT][4], const ggml_cuda_pq2_j16_stage<NT> & st, const int c) {
    // dA of the C rows 2c and 2c+1 of each 8-row tile: the scale of row r was loaded by lanes 4r..4r+3.
    const int d_lo = __shfl_sync(0xFFFFFFFF, st.xd, 8*c + 0);
    const int d_hi = __shfl_sync(0xFFFFFFFF, st.xd, 8*c + 4);
    float dA[NT][2];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        dA[t][0] = __half2float(__ushort_as_half((unsigned short) (d_lo >> (16*t))));
        dA[t][1] = __half2float(__ushort_as_half((unsigned short) (d_hi >> (16*t))));
    }
    const float dB[2][4] = {
        {st.yd[0].x, st.yd[0].y, st.yd[0].z, st.yd[0].w},
        {st.yd[1].x, st.yd[1].y, st.yd[1].z, st.yd[1].w},
    };
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            int b0, b1;
            ggml_cuda_pq2_j16_unpack(st.xq[t][s], b0, b1);
            int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0].x, st.ya[s][1].x, st.ya[s][0].y, st.ya[s][1].y, b0, b1);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float cf = __int_as_float(C[l]) - 12582912.0f; // == (float) idot, exactly
                sum[t][l] += cf*dA[t][l % 2]*dB[l / 2][s];
            }
        }
    }
}

// One output segment of the warp: rows [row0, row0 + 8*NT), K blocks [kb_start, kb_stop) of 128
// values, all 16 columns. Accumulates into sum (zeroed by the caller). Register double buffering of
// the operands of the next block; L2 hints for the weight rows HINT_D blocks ahead (one 32-byte
// sector hint per row and sector of a block; no architectural effect).
// The activations y (written by the previous kernel) are only read after the grid dependency wait of
// the kernel; no y/dst pointer is __restrict__, so no such access can move above it.
static __device__ __forceinline__ void ggml_cuda_pq2_j16_hint(const char * p) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
#else
    GGML_UNUSED(p);
#endif
}

// L2 hints of blocks [kb_first, kb_first + n) of the warp's weight rows (lane mapping as below).
template <int NT>
static __device__ __forceinline__ void ggml_cuda_pq2_j16_hint_rows(
        const char * __restrict__ x, const int row0, const int stride_row_x, const int kb_first, const int n,
        const int kb_stop) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
    const int stride_row_bytes = stride_row_x*(int) sizeof(block_pq2_0);
    if (NT == 2 || c < 2) {
        const char * xh = x + (int64_t) (row0 + g + (NT == 2 ? (c/2)*8 : 0))*stride_row_bytes +
            (int64_t) kb_first*(int) sizeof(block_pq2_0) + (c % 2)*32;
        for (int d = 0; d < n && kb_first + d < kb_stop; ++d) {
            ggml_cuda_pq2_j16_hint(xh + d*(int) sizeof(block_pq2_0));
        }
    }
}

template <int NT, int HINT_D>
static __device__ __forceinline__ void ggml_cuda_pq2_j16_segment(
        float (&sum)[NT][4], const char * __restrict__ x, const int * y, const int row0,
        const int stride_row_x, const int ncols_y, const int kb_start, const int kb_stop, const bool prologue_hinted) {
    static_assert(HINT_D >= 1, "HINT_D");
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;

    const int stride_row_bytes = stride_row_x*(int) sizeof(block_pq2_0);
    const char * xr = x + (int64_t) (row0 + g)*stride_row_bytes + (int64_t) kb_start*(int) sizeof(block_pq2_0);
    const int y_kb_bytes = ncols_y*(int) sizeof(block_q8_1_mmq);
    const char * yb = (const char *) y + (int64_t) kb_start*y_kb_bytes + g*(int) sizeof(block_q8_1_mmq);

    // hint lane (g, c): row 8*(c/2) + g of the warp (NT == 2) or row g (NT == 1, c < 2), sector c%2.
    const bool hint_lane = NT == 2 || c < 2;
    const char * xh = xr + (NT == 2 ? (c/2)*8*stride_row_bytes : 0) + (c % 2)*32;
    auto hint = [&](const char * p) {
        ggml_cuda_pq2_j16_hint(p);
    };
    if (!prologue_hinted) {
#pragma unroll
        for (int d = 1; d < HINT_D; ++d) {
            if (hint_lane && kb_start + d < kb_stop) {
                hint(xh + d*(int) sizeof(block_pq2_0));
            }
        }
    }

    ggml_cuda_pq2_j16_stage<NT> st0;
    ggml_cuda_pq2_j16_stage<NT> st1;
    int kb = kb_start;
    if (kb < kb_stop) {
        ggml_cuda_pq2_j16_load<NT>(st0, xr, stride_row_bytes, yb, c);
    }
    while (kb < kb_stop) {
        if (hint_lane && kb + HINT_D < kb_stop) {
            hint(xh + HINT_D*(int) sizeof(block_pq2_0));
        }
        if (kb + 1 < kb_stop) {
            ggml_cuda_pq2_j16_load<NT>(st1, xr + sizeof(block_pq2_0), stride_row_bytes, yb + y_kb_bytes, c);
        }
        ggml_cuda_pq2_j16_compute<NT>(sum, st0, c);
        if (kb + 1 >= kb_stop) {
            break;
        }
        if (hint_lane && kb + 1 + HINT_D < kb_stop) {
            hint(xh + (HINT_D + 1)*(int) sizeof(block_pq2_0));
        }
        if (kb + 2 < kb_stop) {
            ggml_cuda_pq2_j16_load<NT>(st0, xr + 2*sizeof(block_pq2_0), stride_row_bytes, yb + 2*y_kb_bytes, c);
        }
        ggml_cuda_pq2_j16_compute<NT>(sum, st1, c);
        kb += 2;
        xr += 2*sizeof(block_pq2_0);
        xh += 2*sizeof(block_pq2_0);
        yb += 2*y_kb_bytes;
    }
}

template <int NT>
static __device__ __forceinline__ void ggml_cuda_pq2_j16_store(
        const float (&sum)[NT][4], float * out, const int row_in_out, const int stride_col, const int j_max) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int j = g + 8*h;
            if (j > j_max) {
                continue;
            }
            const int i = row_in_out + 8*t + 2*c;
            *(float2 *) (out + (int64_t) j*stride_col + i) = make_float2(sum[t][2*h + 0], sum[t][2*h + 1]);
        }
    }
}

// grid = (old grid, 128 / (NW*8*NT)), block = (32, NW). The K partition is the one of
// mul_mat_q<GGML_TYPE_PQ2_0, 16, false> with gridDim.x blocks: gridDim.x == number of 128-row tiles
// is pure tiling (one full-K segment per tile), fewer blocks is stream-k.
// Launched with programmatic stream serialization on GB10: L2 hints for the first weight blocks
// (weights only, no memory read), then the grid dependency wait before y is read or dst/tmp_fixup
// written; y/dst/tmp_fixup are not __restrict__ so nothing moves above the wait.
template <int NW, int NT, int MINB, int HINT_D>
__launch_bounds__(NW*32, MINB)
static __global__ void mul_mat_q_pq2_j16(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int ncols_dst) {
    constexpr int I          = 128; // tile height of the old kernel
    constexpr int J          = 16;
    constexpr int rows_slice = NW*8*NT;
    static_assert(I % rows_slice == 0, "slice");
    constexpr int blocks_per_iter = MMQ_ITER_K / QK_PQ2_0;

    // Same index math as mul_mat_q (ntx = nchannels = nsamples = 1).
    const int nty = (nrows_x + I - 1) / I;
    const int64_t W = (int64_t) nty*nkb;
    const int b = blockIdx.x;
    int kbc      = (int) ((int64_t) b      *W / gridDim.x);
    int kbc_stop = (int) ((int64_t) (b + 1)*W / gridDim.x);
    kbc      -= (kbc      % nkb) % blocks_per_iter;
    kbc_stop -= (kbc_stop % nkb) % blocks_per_iter;

    const int row_in_tile = blockIdx.y*rows_slice + threadIdx.y*8*NT;
    const int j_max = ncols_dst - 1;

    int kb0_start = kbc % nkb;
    int kb0_stop  = min(nkb, kb0_start + kbc_stop - kbc);

    if (kbc < kbc_stop) {
        ggml_cuda_pq2_j16_hint_rows<NT>(x, (kbc / nkb)*I + row_in_tile, stride_row_x, kb0_start, HINT_D, kb0_stop);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    bool hinted = true; // the first segment's prologue hints were issued above

    while (kbc < kbc_stop && kb0_stop == nkb) {
        const int it = kbc / nkb;
        float sum[NT][4] = {{0.0f}};
        ggml_cuda_pq2_j16_segment<NT, HINT_D>(sum, x, y, it*I + row_in_tile, stride_row_x, ncols_y, kb0_start, kb0_stop, hinted);
        hinted = false;
        ggml_cuda_pq2_j16_store<NT>(sum, dst, it*I + row_in_tile, stride_col_dst, j_max);

        kbc += nkb;
        kbc -= kbc % nkb;
        kb0_start = 0;
        kb0_stop  = min(nkb, kbc_stop - kbc);
    }
    if (kbc >= kbc_stop) {
        return;
    }
    // Trailing partial tile: partial sums to the fixup buffer of block b (layout [j][i], stride I).
    const int it = kbc / nkb;
    float sum[NT][4] = {{0.0f}};
    ggml_cuda_pq2_j16_segment<NT, HINT_D>(sum, x, y, it*I + row_in_tile, stride_row_x, ncols_y, kb0_start, kb0_stop, hinted);
    ggml_cuda_pq2_j16_store<NT>(sum, tmp_fixup + (int64_t) b*(J*I), row_in_tile, I, J - 1);
}

// ---------------------------------------------------------------------------------------------
// Host side (included by mmq.cuh after mmq_args). Used on GGML_CUDA_CC_DGX_SPARK only, for
// mul_mat_q<GGML_TYPE_PQ2_0, 16, false> without ids/channels/samples.

#define GGML_CUDA_PQ2_J16_NW   16 // warps per CTA (8 weight rows per warp: one CTA per 128-row tile)
#define GGML_CUDA_PQ2_J16_NT    1
#define GGML_CUDA_PQ2_J16_MINB  1
#define GGML_CUDA_PQ2_J16_HD    3

static bool ggml_cuda_mmq_pq2_j16_supported(const mmq_args & args) {
    return args.type_x == GGML_TYPE_PQ2_0 && args.ids_dst == nullptr && args.expert_bounds == nullptr &&
        args.y_scale == nullptr &&
        args.nchannels_x == 1 && args.nchannels_y == 1 && args.nsamples_x == 1 && args.nsamples_y == 1 &&
        args.nrows_x % 128 == 0 && args.ncols_x % (2*QK_PQ2_0) == 0 &&
        args.ncols_dst >= 1 && args.ncols_dst <= 16 && args.ncols_max == args.ncols_dst &&
        args.nrows_dst % 2 == 0 && (uintptr_t) args.dst % 8 == 0 && (uintptr_t) args.y % 16 == 0 &&
        args.nrows_x*args.stride_row_x*(int64_t) sizeof(block_pq2_0) < (int64_t(1) << 31) &&
        (args.ncols_x/QK_PQ2_0 + 1)*args.ncols_y*(int64_t) sizeof(block_q8_1_mmq) < (int64_t(1) << 31);
}

// Same grid rule as launch_mul_mat_q for the PQ2_0 J=16 (stream-k) config on NVIDIA.
static int ggml_cuda_mmq_pq2_j16_grid(const mmq_args & args, const int nsm) {
    const int ntiles = (int) ((args.nrows_x + 127) / 128);
    const int nwaves = (ntiles + nsm - 1) / nsm;
    const int eff    = 100*ntiles / (nsm*nwaves);
    return eff >= 90 ? ntiles : nsm;
}

static bool ggml_cuda_mmq_pq2_j16_needs_fixup(const mmq_args & args, const int grid) {
    const int ntiles = (int) ((args.nrows_x + 127) / 128);
    return ntiles % grid != 0;
}

// Shapes where this kernel beats mul_mat_q in the decode graph: the stream-k partition (the grid is
// not one CUDA block per 128-row tile) with at least 2048 rows. Pure tiling (e.g. 12288 rows, the LM
// head) and small matrices stay on mul_mat_q. Decided from the same values launch_mul_mat_q uses.
static bool ggml_cuda_mmq_pq2_j16_use(const mmq_args & args, const int nsm) {
    if (!ggml_cuda_mmq_pq2_j16_supported(args) || args.nrows_x < 2048) {
        return false;
    }
    const int ntiles = (int) ((args.nrows_x + 127) / 128);
    return ggml_cuda_mmq_pq2_j16_grid(args, nsm) != ntiles;
}

// Template so that only the PQ2_0 MMQ translation unit instantiates the kernels.
template <ggml_type type>
static void ggml_cuda_mmq_pq2_j16_run(const mmq_args & args, const int grid, float * tmp_fixup, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_PQ2_0, "PQ2_0 only");
    constexpr int NW = GGML_CUDA_PQ2_J16_NW;
    constexpr int NT = GGML_CUDA_PQ2_J16_NT;
    const int nkb = (int) (args.ncols_x / QK_PQ2_0);
    const dim3 block_nums(grid, 128 / (NW*8*NT), 1);
    const dim3 block_dims(32, NW, 1);
    const auto kernel = mul_mat_q_pq2_j16<NW, NT, GGML_CUDA_PQ2_J16_MINB, GGML_CUDA_PQ2_J16_HD>;
    // Every warp re-reads the activations through L1: ask for the largest L1 (no shared memory is used).
    static bool carveout_set[GGML_CUDA_MAX_DEVICES] = { false };
    const int device = ggml_cuda_get_device();
    if (!carveout_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxL1));
        carveout_set[device] = true;
    }
    // Programmatic stream serialization (PDL), as launch_mul_mat_q does on GB10.
    ggml_cuda_kernel_launch(kernel,
        ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream),
        args.x, args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.stride_row_x, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst);
    if (ggml_cuda_mmq_pq2_j16_needs_fixup(args, grid)) {
        const uint3 blocks_per_ne00_fd = init_fastdiv_values(nkb);
        const uint3 one_fd             = init_fastdiv_values(1);
        ggml_cuda_kernel_launch(mul_mat_q_stream_k_fixup<type, 16, false>,
            ggml_cuda_kernel_launch_params(dim3(grid, 128/32, 1), dim3(32, 4, 1), 0, stream),
            nullptr, nullptr, args.dst, tmp_fixup, blocks_per_ne00_fd, (int) args.nrows_x, (int) args.ncols_dst,
            (int) args.nrows_dst, one_fd, (int) args.stride_channel_dst, one_fd, (int) args.stride_sample_dst, one_fd,
            ggml_cuda_mmq_partition_init((uint64_t) (args.nrows_x/128)*nkb, grid));
    }
}
