#pragma once

#include "common.cuh"

// DFlash2 block conv (ggml_dflash2_conv): one kernel for the drafter's dynamic depthwise block conv.
// Header-only (static), included by ggml-cuda.cu.

// out[h, t] = sum_k (dynamic[t][side, k, g(h)] + base[side, k, h]) * x[h, t - k], x = 0 before the block start.
// Bit-identical to the unfused chain the drafter graph used (repeat/add for the weight, fill/concat for the
// shifted input, one mul per tap and one add per extra tap): every add and mul is a separately rounded f32
// operation in the same order, so no FMA contraction (explicit _rn intrinsics). An optional residual is added
// last, as the add after the conv did.
static __global__ void dflash2_conv_f32(
        const float * __restrict__ x, const char * __restrict__ dyn, const float * __restrict__ base,
        const float * __restrict__ resid, float * __restrict__ dst, const int n_embd, const int kernel, const int block_size, const int group_size,
        const int n_groups, const int64_t dyn_nb1, const int side) {
    const int h = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;

    // programmatic stream serialization on GB10: x and dynamic come from earlier kernels of the stream
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();

    if (h >= n_embd) {
        return;
    }

    const float * c  = (const float *) (dyn + (int64_t) t*dyn_nb1) + (int64_t) side*kernel*n_groups + h/group_size;
    const float * b  = base + (int64_t) side*kernel*n_embd + h;
    const int     tb = t % block_size;

    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        const float w    = __fadd_rn(c[(int64_t) k*n_groups], b[(int64_t) k*n_embd]);
        const float v    = tb >= k ? x[(int64_t) (t - k)*n_embd + h] : 0.0f;
        const float term = __fmul_rn(w, v);
        acc = k == 0 ? term : __fadd_rn(acc, term);
    }
    if (resid != nullptr) {
        acc = __fadd_rn(acc, resid[(int64_t) t*n_embd + h]);
    }
    dst[(int64_t) t*n_embd + h] = acc;
}

static void ggml_cuda_op_dflash2_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * dynamic = dst->src[1];
    const ggml_tensor * base    = dst->src[2];
    const ggml_tensor * resid   = dst->src[3];

    GGML_ASSERT(resid == nullptr || (resid->type == GGML_TYPE_F32 && ggml_is_contiguous(resid)));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dynamic->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(base) && ggml_is_contiguous(dst));
    GGML_ASSERT(dynamic->nb[0] == sizeof(float));

    const int side       = ggml_get_op_params_i32(dst, 0);
    const int block_size = ggml_get_op_params_i32(dst, 1);
    const int group_size = ggml_get_op_params_i32(dst, 2);

    const int n_embd   = (int) x->ne[0];
    const int n_tokens = (int) x->ne[1];
    const int kernel   = (int) base->ne[1];
    const int n_groups = n_embd / group_size;

    constexpr int block = 256;
    const int     cc    = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_cuda_kernel_launch_params launch_params(dim3((n_embd + block - 1)/block, n_tokens, 1), dim3(block, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch_gb10(cc, dflash2_conv_f32, launch_params,
            (const float *) x->data, (const char *) dynamic->data, (const float *) base->data,
            resid ? (const float *) resid->data : (const float *) nullptr, (float *) dst->data,
            n_embd, kernel, block_size, group_size, n_groups, (int64_t) dynamic->nb[1], side);
}
