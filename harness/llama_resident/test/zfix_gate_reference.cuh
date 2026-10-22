// Frozen759 per-head RMS/SiLU/FWHT/D4 consumer.
template <bool SIGNED, bool D1 = false>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_gnorm_reference(const float * x, const float * z, const int64_t z_stride,
        const float * w, const int ncols, const float eps, const uint3 ne1, const int64_t g1, const int64_t g2,
        const int64_t g3, const float * signs, block_q8_1_mmq * y, const int64_t rows) {
    constexpr int N=4096, NT=256, NE=N/NT, HD=128, HW=HD/WARP_SIZE;
    __shared__ float shared[N];
    __shared__ float s_sum[NE][NT/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE, a0=tid%HD;
    const int64_t row=blockIdx.y;
    float v[NE];
    int64_t head[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#pragma unroll
    for(int i=0;i<NE;++i) {
        const uint2 a21=fast_div_modulo((uint32_t)(i*(NT/HD)+tid/HD), ne1);
        head[i]=row*g3+(int64_t)a21.y*g1+(int64_t)a21.x*g2;
        v[i]=x[head[i]*HD+a0];
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[i][warp]=tmp;
    }
    __syncthreads();
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        if(lane<HW) tmp=s_sum[i][(warp/HW)*HW+lane];
        tmp=warp_reduce_sum(tmp);
        const float mean=tmp/ncols;
        const float scale=rsqrtf(mean+eps);
        const float xn=scale*v[i]*w[a0];
        const float g=ggml_cuda_op_silu_single(z[head[i]*z_stride+a0])*xn;
        v[i]=g*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[i*NT+tid];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    if constexpr(D1)fwht_quantize_mmq_d1_store<N,NT>(v,y,0,row,rows,tid,lane);else fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
