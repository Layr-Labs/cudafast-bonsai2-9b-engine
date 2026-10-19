template <int N, int NT>
__device__ __forceinline__ void frozen227_block_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <bool SIGNED>
__launch_bounds__(256, 1)
static __global__ void frozen227_rms(const float * x, const int64_t stride_row, const float * w,
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
    frozen227_block_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}

template <bool SIGNED>
__launch_bounds__(256, 1)
static __global__ void frozen227_gnorm(const float * x, const float * z, const int64_t z_stride,
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
    frozen227_block_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
