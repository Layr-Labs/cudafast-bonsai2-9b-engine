#pragma once
// Geometry proof: 32 tiles / 48 partitions = 2/3 tile each. For nkb >=
// 16*BPI and nkb%BPI==0, starts at b=3q are 2q whole tiles; starts at
// b=3q+1,3q+2 lie strictly inside tiles 2q and 2q+1. Their owners each
// reduce exactly one predecessor (b-1). The +0 partial accumulation and
// final dst addition are both retained, including signed-zero behavior.
__launch_bounds__(128,1)
static __global__ void mul_mat_q_q8_dl_fixup_pair(float * dst,const float * partials,int stride,int cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    const int owner=3*(int(blockIdx.x)/2)+1+(int(blockIdx.x)%2);
    const int pred=owner-1;
    const int row=32*blockIdx.y+threadIdx.x;
#pragma unroll
    for(int j=0;j<4;++j) {
        const int col=threadIdx.y+4*j;
        if(col<cols) {
            const float v=partials[pred*(16*128)+col*128+row];
            const float sum=0.0f+v;
            dst[col*stride+int(blockIdx.x)*128+row]+=sum;
        }
    }
}

// Small Q8 DL48 projections: equal groups of producer partitions per tile.
// The guard ensures every group member has nonempty aligned work.
template <int NTILES>
__launch_bounds__(128, 1)
static __global__ void mul_mat_q_q8_small_fixup_compact(
        float * dst, const float * partials, int stride, int cols) {
    static_assert(NTILES == 1 || NTILES == 2 || NTILES == 4 || NTILES == 8, "small DL48 tiles");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    constexpr int PARTS = 48/NTILES;
    const int tile = blockIdx.x;
    const int owner = PARTS*(tile+1)-1;
    const int row = 32*blockIdx.y + threadIdx.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = threadIdx.y + 4*j;
        if (col < cols) {
            float sum = 0.0f;
#pragma unroll
            for (int pred = 1; pred < PARTS; ++pred) {
                sum += partials[(owner-pred)*(16*128) + col*128 + row];
            }
            dst[col*stride + tile*128 + row] += sum;
        }
    }
}
