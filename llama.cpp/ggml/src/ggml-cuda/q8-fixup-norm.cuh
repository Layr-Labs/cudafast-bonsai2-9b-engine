#pragma once
#include "common.cuh"

// Same 32-tile/48-partition pair geometry as mul_mat_q_q8_dl_fixup_pair.
// Full normalization rows own their fixup; the original producer K chain and
// both rounded additions (+0 partial, then dst) are retained.
static __global__ void q8_fixup_rms_norm_f32(float * x, const float * partials,
        const float * weight, float * dst, const float eps) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x, tid = threadIdx.x;
    x += row*4096; dst += row*4096;
    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = tid; col < 4096; col += 1024) {
        const int tile = col/128, pred = 3*(tile/2) + tile%2;
        const float p = __fadd_rn(0.0f, partials[pred*(16*128) + row*128 + col%128]);
        const float xi = __fadd_rn(x[col], p);
        x[col] = xi;
        tmp += xi*xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, 1024>(tmp, s_sum);
    const float scale = rsqrtf(tmp/4096 + eps);
    for (int col = tid; col < 4096; col += 1024) {
        dst[col] = scale*x[col]*weight[col];
    }
}
