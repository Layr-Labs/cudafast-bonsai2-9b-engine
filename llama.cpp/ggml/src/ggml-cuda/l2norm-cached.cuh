#pragma once

// Same one-warp reduction tree as l2_norm_f32<32> for128 columns. Values
// remain in registers between sum-of-squares and scale; no arithmetic changes.
static __global__ void l2_norm_f32_cached_128(
        const float * x, float * dst, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int row=blockIdx.x, channel=blockIdx.y, sample=blockIdx.z;
    const int tid=threadIdx.x;
    x += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*gridDim.y + channel)*gridDim.x + row)*128;
    float v[4];
    float tmp=0.0f;
    ggml_cuda_pdl_sync();
#pragma unroll
    for(int i=0;i<4;++i) {
        v[i]=x[tid+32*i];
        tmp += v[i]*v[i];
    }
    extern __shared__ float s_sum[];
    tmp=block_reduce<block_reduce_method::SUM,32>(tmp,s_sum);
    ggml_cuda_pdl_lc();
    const float scale=rsqrtf(fmaxf(tmp,eps*eps));
#pragma unroll
    for(int i=0;i<4;++i) dst[tid+32*i]=scale*v[i];
}
