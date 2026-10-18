#pragma once

// Decode-layout single-matrix fixup: one tile owner instead of every stream-K
// partition. Descending partial accumulation is exactly the generic kernel's order.
template<ggml_type TYPE>
static __global__ void __launch_bounds__(128)
ggml_cuda_j16_tile_fixup(float * dst, const float * partial, const int ncols,
        const int stride, const uint3 nkb_fd, const uint3 tiles_fd, const uint3 grid_fd) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif
    constexpr int I=128,J=16;
    constexpr int BPI=ggml_cuda_mmq_get_K_vram(TYPE,J,false)/ggml_cuda_type_traits<TYPE>::qk;
    const int tile=blockIdx.x, nkb=nkb_fd.z, ntiles=tiles_fd.z, work=ntiles*nkb;
    // This is the last partition with an unrounded start below this tile's end.
    const int owner=fastdiv((tile+1)*grid_fd.z-1,tiles_fd);
    int start=fastdiv(owner*work,grid_fd);
    start-=fastmodulo(start,nkb_fd)%BPI;
    if(start<=tile*nkb) return; // owner started at this tile's boundary or earlier: no fixup
    const int row=blockIdx.y*32+threadIdx.x;
    float sum[4]={0.0f,0.0f,0.0f,0.0f};
    int previous=owner-1,end=start;
    while(true) {
        int begin=fastdiv(previous*work,grid_fd);
        begin-=fastmodulo(begin,nkb_fd)%BPI;
        if(begin==end) {--previous;end=begin;continue;}
#pragma unroll
        for(int l=0;l<4;++l) {
            const int col=4*l+threadIdx.y;
            sum[l]+=partial[int64_t(previous)*(J*I)+col*I+row];
        }
        if(begin<=tile*nkb) break;
        --previous;end=begin;
    }
#pragma unroll
    for(int l=0;l<4;++l) {
        const int col=4*l+threadIdx.y;
        if(col<ncols) dst[int64_t(col)*stride+int64_t(tile)*I+row]+=sum[l];
    }
}
