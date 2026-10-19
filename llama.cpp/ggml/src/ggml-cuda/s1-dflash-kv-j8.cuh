// S1 drafter K/V full-K kernel; production selects only the measured J16 path.
#pragma once
#include "common.cuh"

#ifndef GGML_CUDA_S1_DFLASH_KV_J8_DEFAULT
#define GGML_CUDA_S1_DFLASH_KV_J8_DEFAULT 2
#endif
#ifndef GGML_CUDA_S1_DFLASH_KV_TRACE_DEFAULT
#define GGML_CUDA_S1_DFLASH_KV_TRACE_DEFAULT 0
#endif

// S1: only the authenticated DFlash [context;noise] concat's two ordinary
// Q8 K/V consumers may use this grouped full-K route. False launches nothing.
static inline bool ggml_cuda_s1_dflash_context_input(const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_NONE || t->view_src) return false;
    if (!strcmp(t->name,"dflash_ctx_rows")) return (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
    // ggml-backend.cpp names scheduler input copies BACKEND#name#copy_id.
    // A single-copy transfer can omit INPUT on the copy, so recognize that
    // exact wrapper too, without treating arbitrary same-shape tensors as draft.
    const char * marker=strchr(t->name,'#');
    static const char tag[]="dflash_ctx_rows#";
    if (!marker || marker==t->name || strncmp(marker+1,tag,sizeof(tag)-1)) return false;
    const char * id=marker+sizeof(tag);
    if (!*id) return false;
    for (;*id;++id) if (*id<'0' || *id>'9') return false;
    return true;
}
bool ggml_cuda_s1_dflash_kv_enabled();
bool ggml_cuda_s1_dflash_kv_trace();
bool ggml_cuda_s1_dflash_kv_j16(ggml_backend_cuda_context & ctx, const ggml_tensor * cat,
    const ggml_tensor * first, const ggml_tensor * second, const void * q8);

#ifdef GGML_CUDA_S1_DFLASH_KV_IMPLEMENTATION
#include "mmq.cuh"

// bit 0 retains the measured J8 probe; bit 1 selects its single J16 redesign.
// J8 stays a test diagnostic. Production accepts exactly mode 2; 0 restores
// the parent projections. No activation format or cache consumer is changed.
template <int J>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q8_0, J, false), 1)
static __global__ void ggml_cuda_s1_kv_full_k(const char * wk, const char * wv, const int * y, float * dk, float * dv) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    static_assert(J == 8 || J == 16, "bounded drafter prototype");
    constexpr int I = ggml_cuda_mmq_get_I(GGML_TYPE_Q8_0, J, false);
    static_assert(I == 128, "one CTA must cover one head");
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    extern __shared__ int ids[];
    const int tid = threadIdx.x + 32*threadIdx.y;
    if (tid < J) ids[tid] = tid;
    __syncthreads();
    const int it = blockIdx.x, jt = blockIdx.y;
    const char * w = blockIdx.z ? wv : wk;
    float * dst = blockIdx.z ? dv : dk;
    // Existing integer MMA, D4 scales, weight unpack and FP32 accumulation.
    // Full K in increasing order, no partial buffer and no stream-K fixup.
    mul_mat_q_process_tile<GGML_TYPE_Q8_0, J, false, false>(
        w, it*I*128, y + jt*J*(sizeof(block_q8_1_mmq)/sizeof(int)), ids,
        dst + jt*J*1024 + it*I, nullptr, nullptr, 128, 32, 1024,
        1024-it*I-1, J-1, 0, 128);
#else
    GGML_UNUSED(wk); GGML_UNUSED(wv); GGML_UNUSED(y); GGML_UNUSED(dk); GGML_UNUSED(dv);
#endif
}

template <int J>
static inline void ggml_cuda_s1_kv_full_k_launch(int cc, const mmq_args & k, const mmq_args & v, cudaStream_t stream) {
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, false, cc);
    const int smem = mmq_get_nbytes_shared(config, cc);
    CUDA_SET_SHARED_MEMORY_LIMIT(ggml_cuda_s1_kv_full_k<J>, smem);
    ggml_cuda_kernel_launch(ggml_cuda_s1_kv_full_k<J>,
        ggml_cuda_kernel_launch_params(dim3(8,32/J,2),dim3(32,config.nthreads/32,1),smem,stream),
        k.x, v.x, k.y, k.dst, v.dst);
}

static inline bool ggml_cuda_s1_kv_j8_launch(ggml_backend_cuda_context & ctx, const mmq_args & k,
                                           const mmq_args & v, cudaStream_t stream) {
    const char * value = getenv("GGML_CUDA_S1_DFLASH_KV_J8");
    const int mode = (value ? atoi(value) : GGML_CUDA_S1_DFLASH_KV_J8_DEFAULT) & 3;
    if (!mode) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    auto eligible = [](const mmq_args & a) {
        return a.type_x == GGML_TYPE_Q8_0 && a.nrows_x == 1024 && a.ncols_x == 4096 &&
            a.ncols_dst == 32 && a.ncols_y == 32 && a.ncols_max == 32 &&
            a.nrows_dst == 1024 && a.stride_row_x == 128 &&
            a.nchannels_x == 1 && a.nchannels_y == 1 && a.nsamples_x == 1 && a.nsamples_y == 1 &&
            !a.ids_dst && !a.expert_bounds && !a.y_scale && !a.add_res && !a.add_dst && !a.dst_hi;
    };
    if (cc != GGML_CUDA_CC_DGX_SPARK || !eligible(k) || !eligible(v) || k.y != v.y) return false;
    if (mode & 2) ggml_cuda_s1_kv_full_k_launch<16>(cc,k,v,stream);
    else ggml_cuda_s1_kv_full_k_launch<8>(cc,k,v,stream);
    GGML_UNUSED(ctx);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // GGML_CUDA_S1_DFLASH_KV_IMPLEMENTATION
