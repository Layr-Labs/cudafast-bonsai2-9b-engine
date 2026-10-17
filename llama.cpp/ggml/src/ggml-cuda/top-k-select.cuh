#pragma once

// Streaming top-k for wide rows with a small k, e.g. the DFlash2 selector that
// takes the top 16 of a 248k-entry vocabulary row for every block position.
//
// The result is the same as taking the first k entries of the stable
// descending cub radix sort used by the argsort fallback: keys are ordered by
// cub's float bit twiddle, -0.0 is treated as +0.0, and equal keys keep
// ascending column order. Each candidate is packed into one 64-bit value whose
// unsigned order is exactly that total order, so selection is a sequence of
// strict "largest value below the previous winner" block reductions.
//
// Pass 1 splits every row into chunks and writes the k best packed values of
// each chunk. Pass 2 selects the k best of those candidates per row and writes
// the column indices in descending order.

#include <cstdint>
#include <cuda_runtime.h>

#define TOPK_SELECT_NT          256
#define TOPK_SELECT_PER_THREAD  32
#define TOPK_SELECT_CHUNK       (TOPK_SELECT_NT * TOPK_SELECT_PER_THREAD)
#define TOPK_SELECT_MAX_K       32
#define TOPK_SELECT_MAX_ROWS    65535

static __device__ __forceinline__ uint64_t topk_select_pack(const float v, const int col) {
    uint32_t u = __float_as_uint(v);
    u = u == 0x80000000u ? 0u : u;                                 // -0.0 sorts as +0.0
    const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // cub float twiddle
    return ((uint64_t) ord << 32) | (uint64_t) (~(uint32_t) col);     // ties: lower column first
}

// Block-wide maximum. `red` needs TOPK_SELECT_NT/32 + 1 entries; every thread
// returns the same value and `red` is reusable after return.
static __device__ __forceinline__ uint64_t topk_select_block_max(uint64_t v, uint64_t * red) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint64_t w = __shfl_xor_sync(0xffffffff, v, offset);
        v = w > v ? w : v;
    }
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < TOPK_SELECT_NT / 32 ? red[lane] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const uint64_t w = __shfl_xor_sync(0xffffffff, v, offset);
            v = w > v ? w : v;
        }
        if (lane == 0) {
            red[TOPK_SELECT_NT / 32] = v;
        }
    }
    __syncthreads();
    v = red[TOPK_SELECT_NT / 32];
    __syncthreads();
    return v;
}

// grid (nchunks, nrows); part holds nrows * nchunks * k packed values.
// A chunk with fewer than k columns pads its tail with 0, which ranks below
// every packed column because the low word ~col is at least 2^31.
static __global__ void __launch_bounds__(TOPK_SELECT_NT)
topk_select_partial(const float * __restrict__ src, uint64_t * __restrict__ part, const int ncols, const int k) {
    __shared__ uint64_t red[TOPK_SELECT_NT / 32 + 1];

    const int     row = blockIdx.y;
    const int     beg = blockIdx.x * TOPK_SELECT_CHUNK;
    const float * x   = src + (int64_t) row * ncols;

    uint64_t v[TOPK_SELECT_PER_THREAD];
#pragma unroll
    for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
        const int col = beg + j * TOPK_SELECT_NT + threadIdx.x;
        v[j] = col < ncols ? topk_select_pack(x[col], col) : 0;
    }

    uint64_t * out = part + ((int64_t) row * gridDim.x + blockIdx.x) * k;
    uint64_t best = 0;
#pragma unroll
    for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
        best = v[j] > best ? v[j] : best;
    }
    for (int r = 0; r < k; ++r) {
        const uint64_t winner = topk_select_block_max(best, red);
        if (threadIdx.x == 0) {
            out[r] = winner;
        }
        // Every non-padding key contains a unique column ID. Only the owner
        // of the winner needs a new local maximum. Other maxima remain valid.
        // Multiple owners of zero padding all correctly remain zero.
        if (r + 1 < k && best == winner) {
            best = 0;
#pragma unroll
            for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
                const uint64_t c = v[j];
                if (c < winner && c > best) {
                    best = c;
                }
            }
        }
    }
}

// grid (nrows); writes k column indices per row in descending key order.
static __global__ void __launch_bounds__(TOPK_SELECT_NT)
topk_select_merge(const uint64_t * __restrict__ part, int * __restrict__ dst, const int nchunks, const int k) {
    __shared__ uint64_t red[TOPK_SELECT_NT / 32 + 1];

    const int        row = blockIdx.x;
    const int        n   = nchunks * k;
    const uint64_t * p   = part + (int64_t) row * n;

    uint64_t prev = 0;
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
        for (int i = threadIdx.x; i < n; i += TOPK_SELECT_NT) {
            const uint64_t c = p[i];
            if ((r == 0 || c < prev) && c > best) {
                best = c;
            }
        }
        prev = topk_select_block_max(best, red);
        if (threadIdx.x == 0) {
            dst[(int64_t) row * k + r] = (int) ~(uint32_t) prev;
        }
    }
}

static inline bool topk_select_supported(const int64_t ncols, const int64_t nrows, const int64_t k) {
    return k >= 1 && k <= TOPK_SELECT_MAX_K && ncols >= k && ncols <= INT32_MAX - TOPK_SELECT_CHUNK &&
           nrows >= 1 && nrows <= TOPK_SELECT_MAX_ROWS;
}

static inline size_t topk_select_scratch_elems(const int ncols, const int nrows, const int k) {
    const int nchunks = (ncols + TOPK_SELECT_CHUNK - 1) / TOPK_SELECT_CHUNK;
    return (size_t) nrows * nchunks * k;
}

// src: nrows contiguous rows of ncols floats; dst: nrows * k indices.
static inline void topk_select_launch(const float * src, int * dst, uint64_t * scratch, const int ncols,
                                      const int nrows, const int k, cudaStream_t stream) {
    const int nchunks = (ncols + TOPK_SELECT_CHUNK - 1) / TOPK_SELECT_CHUNK;
    topk_select_partial<<<dim3(nchunks, nrows, 1), TOPK_SELECT_NT, 0, stream>>>(src, scratch, ncols, k);
    topk_select_merge<<<dim3(nrows, 1, 1), TOPK_SELECT_NT, 0, stream>>>(scratch, dst, nchunks, k);
}
