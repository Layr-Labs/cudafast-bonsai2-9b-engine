#pragma once

// Contiguous F32 only; one thread retains the original scalar operation on
// four independent elements. Loads precede stores, including an in-place dst.
template <float (*op)(float)>
static __global__ void unary_gated_vector4_kernel(
        const float * x, const float * g, float * dst, const int64_t vectors) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= vectors) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float4 a = ((const float4 *) x)[i];
    const float4 b = ((const float4 *) g)[i];
    ((float4 *) dst)[i] = make_float4(op(a.x)*b.x, op(a.y)*b.y, op(a.z)*b.z, op(a.w)*b.w);
}
