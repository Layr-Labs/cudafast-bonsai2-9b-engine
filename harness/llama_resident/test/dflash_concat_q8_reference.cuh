#pragma once
// Frozen b9549c7 quantizer; independent of the added two-source input branch.
#include "common.cuh"
#include "mmq.cuh"

template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool swiglu = false, bool rms_scale = false>
static __global__ void dflash_concat_q8_reference(
        const float * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used,
        const float * gate = nullptr,
        const float * norm_weight = nullptr,
        const float * norm_scale = nullptr) {
    // quantize_mmq_q8_1_cuda launches this kernel with programmatic stream serialization on GB10; nothing
    // is read before ggml_cuda_pdl_sync() below, and the pointer arguments are not __restrict__ so no
    // access moves above it.
    ggml_cuda_pdl_lc();

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (swiglu) {
        const float4 gi = i0 < ne00 ? ((const float4 *) gate)[(base_idx + i00)/4] :
                                      make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x *= ggml_cuda_op_silu_single(gi.x);
        xi.y *= ggml_cuda_op_silu_single(gi.y);
        xi.z *= ggml_cuda_op_silu_single(gi.z);
        xi.w *= ggml_cuda_op_silu_single(gi.w);
    }
    if constexpr (rms_scale) {
        const int64_t row = ((int64_t) blockIdx.z)*ne1 + blockIdx.x;
        const float scale = norm_scale[row];
        const float4 wi = i0 < ne00 ? ((const float4 *) norm_weight)[i00/4] : make_float4(0, 0, 0, 0);
        // Preserve the operation order of the unfused RMS_NORM -> MUL graph:
        // (scale * x) * weight.  Reassociating this as x * (scale * weight)
        // can change the last bit and, in turn, a greedy decoding trajectory.
        xi.x = (scale*xi.x)*wi.x;
        xi.y = (scale*xi.y)*wi.y;
        xi.z = (scale*xi.z)*wi.z;
        xi.w = (scale*xi.w)*wi.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}
