template <int N, int NT>
__device__ __forceinline__ void frozen221_butterfly(float * reg, float * s, const int tid, const int lane) {
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

template <typename T, bool SIGNED, bool SWIGLU>
__launch_bounds__(256, 1)
static __global__ void frozen221_quantize(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
                                         int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
#pragma unroll
        for(int i=0;i<NE;++i) {
            if constexpr(SWIGLU) {
                const int64_t col=base+i*NT+tid;
                v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+col]))*float(g[row*o1+col])))*(1.0f/64.0f);
            } else {
                v[i]=float(x[row*k+base+i*NT+tid])*(1.0f/64.0f);
            }
            if constexpr(SIGNED) v[i]*=signs[base+i*NT+tid];
        }
        frozen221_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
    fwht_quantize_mmq_d4_store<N,NT>(v,y,base,row,rows,tid,lane);
    GGML_UNUSED(kp); GGML_UNUSED(g); GGML_UNUSED(o0); GGML_UNUSED(o1);
}
