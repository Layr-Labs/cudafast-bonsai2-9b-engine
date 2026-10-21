#pragma once
#include "top-k-select.cuh"

// Exact tile sufficient statistic: every global top-eight key belongs to its
// tile's local top eight. Logits stores and MMA accumulation stay unchanged.
static __device__ __forceinline__ void ggml_cuda_head_cert_store(
        const float (&sum)[1][4], uint64_t * lists, uint64_t (*keys)[129], const int tile, const int nt) {
    const int lane = threadIdx.x, w = threadIdx.y, g = lane/4, c = lane%4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int pos = g + 8*h, col = w*8 + 2*c;
        keys[pos][col]   = topk_select_pack(sum[0][2*h],   tile*128 + col);
        keys[pos][col+1] = topk_select_pack(sum[0][2*h+1], tile*128 + col+1);
    }
    __syncthreads();
    // A warp now owns one output position, with four keys per lane.
    uint64_t v[4], best = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = keys[w][lane + 32*j];
        best = v[j] > best ? v[j] : best;
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const uint64_t winner = topk_select_warp_max(best);
        if (lane == 0) lists[((int64_t) w*nt + tile)*8 + r] = winner;
        if (r < 7 && best == winner) {
            best = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) best = v[j] < winner && v[j] > best ? v[j] : best;
        }
    }
}

// One warp merges 64 sorted tile lists using two independent list heads/lane.
// Each chunk covers exactly the same 8192 vocabulary columns as the fallback.
static __global__ void __launch_bounds__(32)
topk_tile_partial(uint64_t * part, const int nt, const uint64_t * lists) {
    topk_select_grid_dependency();
    const int lane = threadIdx.x, row = blockIdx.y, first = blockIdx.x*64;
    const int t0 = first + lane, t1 = t0 + 32;
    const uint64_t * p = lists + (int64_t) row*nt*8;
    int i0 = 0, i1 = 0;
    uint64_t h0 = t0 < nt ? p[t0*8] : 0, h1 = t1 < nt ? p[t1*8] : 0;
    uint64_t * out = part + ((int64_t) row*gridDim.x + blockIdx.x)*8;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const uint64_t winner = topk_select_warp_max(h0 > h1 ? h0 : h1);
        if (lane == 0) out[r] = winner;
        if (r < 7) {
            if (h0 == winner) { ++i0; h0 = i0 < 8 ? p[t0*8 + i0] : 0; }
            if (h1 == winner) { ++i1; h1 = i1 < 8 ? p[t1*8 + i1] : 0; }
        }
    }
}
