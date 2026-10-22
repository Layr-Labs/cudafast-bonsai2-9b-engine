// Frozen759 RMS/FWHT/D1 producer.
template <bool SIGNED, bool D1 = false>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_rms_reference(const float * x, const int64_t stride_row, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, int64_t rows) {
    constexpr int N=4096, NT=256, NE=N/NT, NR=1024, NJ=NR/NT;
    __shared__ float shared[N];
    __shared__ float s_sum[NR/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    x+=row*stride_row;
#pragma unroll
    for(int i=0;i<NE;++i) v[i]=x[i*NT+tid];
#pragma unroll
    for(int j=0;j<NJ;++j) {
        float tmp=0.0f;
#pragma unroll
        for(int i=j;i<NE;i+=NJ) tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[lane]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int col=i*NT+tid;
        const float xn=scale*v[i]*w[col];
        if(norm) norm[row*N+col]=xn;
        v[i]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[col];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    if constexpr(D1)fwht_quantize_mmq_d1_store<N,NT>(v,y,0,row,rows,tid,lane);else fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
