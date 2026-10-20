#pragma once

// Lossless, input-independent copies of the registered gate/up stack. One
// buffer holds the same 32 code bytes and two fp16 scale bytes per PQ2 block.
// The map key is the actual stacked-weight address, never a tensor name.
struct ggml_cuda_prefill_plane_entry {
    unsigned char * data;
    int64_t nrows, ncols;
    size_t bytes;
    int device;
};
const ggml_cuda_prefill_plane_entry * ggml_cuda_prefill_plane_find(const void * src);

static bool ggml_cuda_prefill_plane_enabled() {
    static const bool enabled = ggml_cuda_dl_env_int("GGML_CUDA_PREFILL_PLANE", 1) != 0;
    return enabled;
}

// D4 alias route: reuse each thread's consumed compact-code words for scales.
// Enabled by default; an explicit zero retains the original kernel.
static bool ggml_cuda_prefill_plane_d4_alias_enabled() {
    static const bool enabled = ggml_cuda_dl_env_int("GGML_CUDA_PREFILL_PLANE_D4_ALIAS", 1) != 0;
    return enabled;
}

// Reduce D4-gather bank conflicts by permuting only producer ownership.
// Canonical weights/shared layout and warp compute mapping remain unchanged.
static bool ggml_cuda_prefill_plane_owner_enabled() {
    static const bool enabled = ggml_cuda_dl_env_int("GGML_CUDA_PREFILL_PLANE_OWNER", 1) != 0;
    return enabled;
}

static __global__ void ggml_cuda_prefill_plane_pack(const block_pq2_0 * original,
        unsigned char * dst,int rows,int nkb) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=rows*nkb) return;
    const int row=index/nkb,kb=index%nkb;
    const int64_t dest=((int64_t)(row/64)*nkb+kb)*64+row%64;
    // Inverse: row=(dest/(64*nkb))*64+dest%64, kb=(dest/64)%nkb.
    // Thus every original block has one unique destination.
    unsigned char * codes=dst+dest*32;
    uint16_t * scale=((uint16_t *)(dst+(int64_t)rows*nkb*32))+dest;
    const auto & block=original[index];
    *scale=__half_as_ushort(block.d);
#pragma unroll
    for (int i=0;i<32;++i) codes[i]=block.qs[i];
}

static __device__ __forceinline__ void ggml_cuda_prefill_plane_bulk(
        uint32_t dst_y,uint32_t dst_codes,const char * y,const unsigned char * codes,uint32_t bar) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(bar),"r"(128*144+64*32) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(dst_y),"l"(y),"r"(128*144),"r"(bar) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(dst_codes),"l"(codes),"r"(64*32),"r"(bar) : "memory");
}

static __device__ __forceinline__ uint32_t ggml_cuda_prefill_plane_lds_u32(uint32_t addr) {
    uint32_t result;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(addr));
    return result;
}

// Each thread owns code[8*tid..8*tid+7], which is also D4[2*tid..2*tid+1].
// PTX-local code words keep all address inputs live until the final output moves.
// Read both code words before overwriting them; the scale copies are bitwise.
static __device__ __forceinline__ void ggml_cuda_prefill_plane_d4_alias_load(
        uint32_t code, uint32_t y, uint32_t & w0, uint32_t & w1) {
    asm volatile("{\n"
        " .reg .b32 code0, code1, d0, d1;\n"
        " ld.shared.u32 code0, [%2];\n"
        " ld.shared.u32 code1, [%2+4];\n"
        " ld.shared.u32 d0, [%3];\n"
        " ld.shared.u32 d1, [%3+144];\n"
        " st.shared.v2.u32 [%2], {d0, d1};\n"
        " mov.u32 %0, code0;\n"
        " mov.u32 %1, code1;\n"
        "}\n"
        : "=r"(w0), "=r"(w1) : "r"(code), "r"(y) : "memory");
}

// Exact D1 alias specialization: every owner reads its own original code
// words; only owners0..63 copy scale0's 128 fields into the first512bytes.
// One PTX block keeps code reads before their possible overwrite. Predicate
// only the header traffic, avoiding two copies of the code-load instruction.
static __device__ __forceinline__ void ggml_cuda_prefill_plane_d1_alias_load(
        uint32_t code, uint32_t y, uint32_t owner, uint32_t & w0, uint32_t & w1) {
    asm volatile("{\n"
        " .reg .b32 code0, code1, d0, d1;\n"
        " .reg .pred copy;\n"
        " ld.shared.u32 code0, [%2];\n"
        " ld.shared.u32 code1, [%2+4];\n"
        " setp.lt.u32 copy, %4, 64;\n"
        " @copy ld.shared.u32 d0, [%3];\n"
        " @copy ld.shared.u32 d1, [%3+144];\n"
        " @copy st.shared.v2.u32 [%2], {d0, d1};\n"
        " mov.u32 %0, code0;\n"
        " mov.u32 %1, code1;\n"
        "}\n"
        : "=r"(w0), "=r"(w1) : "r"(code), "r"(y), "r"(owner) : "memory");
}

// Original 64x128 macro tile, eight compute warps and two CTAs/SM. Both Y and
// compact 2-bit code tiles use TMA double buffers. The only global X loads in
// the loop are 64 fp16 scales; one next-scale register stays live across MMA.
// The 50176-byte footprint equals the existing stream-K kernel, with barriers
// living in padding of the decoded X tile instead of enlarging shared memory.
template <bool ADD,bool GLU,bool D4_ALIAS=false,bool OWNER=false,bool D1=false>
static __global__ void __launch_bounds__(256,2) mul_mat_q_pq2_plane(
        const unsigned char * codes,const uint16_t * scales,const int * y,float * dst,
        int nkb,int ncols_y,int stride_col_dst,int ntx,const float * residual) {
    constexpr int YBYTES=128*144,XBYTES=64*144,CBYTES=64*32;
    extern __shared__ int plane_smem[];
    const uint32_t smem=(uint32_t)__cvta_generic_to_shared(plane_smem);
    const uint32_t xtile=smem+2*YBYTES, ctiles=xtile+XBYTES, bars=xtile+136;
    const int tid=threadIdx.y*32+threadIdx.x;
    const int owner=OWNER?(tid^(((tid>>2)&3)<<6)):tid,quarter=owner%4,row=owner/4;
    const int rowtile=blockIdx.x/ntx,col0=(blockIdx.x%ntx)*128;
    const int wr=threadIdx.y/4,wc=threadIdx.y%4;
    const unsigned char * cp=codes+(int64_t)rowtile*nkb*CBYTES;
    const uint16_t * sp=scales+(int64_t)rowtile*nkb*64+row;
    const char * yp=(const char *)y+(int64_t)col0*144;
    const int64_t ystep=(int64_t)ncols_y*144;
    ggml_cuda_pdl_sync();
    if (tid==0) {
        ggml_cuda_pq2_j128_mbar_init(bars);
        ggml_cuda_pq2_j128_mbar_init(bars+144);
        ggml_cuda_pq2_j128_mbar_fence_init();
    }
    __syncthreads();
    if (tid==0) {
        ggml_cuda_prefill_plane_bulk(smem,ctiles,yp,cp,bars);
        if (nkb>1) ggml_cuda_prefill_plane_bulk(smem+YBYTES,ctiles+CBYTES,yp+ystep,cp+CBYTES,bars+144);
    }
    uint32_t scale=quarter==0?sp[0]:0;
    float acc[2][4][4]={};
    for (int kb=0;kb<nkb;++kb) {
        const int buf=kb&1;
        ggml_cuda_pq2_j128_mbar_wait(bars+buf*144,(kb/2)&1);
        const uint32_t ci=ctiles+buf*CBYTES+row*32+quarter*8;
        ggml_cuda_pq2_j128_xraw raw;
        if constexpr (D4_ALIAS) {
            // Each thread overwrites only its own two compact-code words after
            // loading both. The existing CTA barrier publishes the complete D4 table.
            const uint32_t yi=smem+buf*YBYTES+(2*owner%128)*144+(owner/64)*4;
            if constexpr(D1) ggml_cuda_prefill_plane_d1_alias_load(ci,yi,owner,raw.w0,raw.w1);
            else ggml_cuda_prefill_plane_d4_alias_load(ci,yi,raw.w0,raw.w1);
        } else {
            raw.w0=ggml_cuda_prefill_plane_lds_u32(ci); raw.w1=ggml_cuda_prefill_plane_lds_u32(ci+4);
        }
        raw.w2=raw.w1; raw.sh=0; raw.d=scale;
        ggml_cuda_pq2_j128_xstore(raw,xtile+row*144+quarter*32,quarter);
        if (quarter==0 && kb+1<nkb) scale=sp[(kb+1)*64];
        __syncthreads();
        if constexpr (D4_ALIAS) {
            if constexpr(D1) ggml_cuda_pq2_j128_d1_compute<true>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144,ctiles+buf*CBYTES+wc*32*4);
            else ggml_cuda_pq2_ws_compute_compact<true>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144,ctiles+buf*CBYTES+wc*32*4);
        } else {
            if constexpr(D1) ggml_cuda_pq2_j128_d1_compute<false>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144);
            else ggml_cuda_pq2_ws_compute_compact<false>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144);
        }
        if (kb+1<nkb) {
            __syncthreads();
            if (tid==0 && kb+2<nkb) {
                ggml_cuda_prefill_plane_bulk(smem+buf*YBYTES,ctiles+buf*CBYTES,yp+(kb+2)*ystep,cp+(int64_t)(kb+2)*CBYTES,bars+buf*144);
            }
        }
    }
    ggml_cuda_pdl_lc();
    if constexpr (GLU) {
        ggml_cuda_pq2_j128_store_glu(acc,dst+(int64_t)col0*stride_col_dst+rowtile*32,
                                   stride_col_dst,wr*32,wc*32,ncols_y-col0-1);
    } else if constexpr (ADD) {
        const int64_t off=(int64_t)col0*stride_col_dst+rowtile*64;
        ggml_cuda_pq2_j128_store_add(acc,dst+off,residual+off,stride_col_dst,wr*32,wc*32,ncols_y-col0-1);
    } else {
        ggml_cuda_pq2_j128_store(acc,dst+(int64_t)col0*stride_col_dst+rowtile*64,
                               stride_col_dst,wr*32,wc*32,ncols_y-col0-1);
    }
}

// Only complete, contiguous GLU tiles are routed here. Unsupported shapes,
// disabled mode or a missing load-time copy retain the existing J128 path.
template <ggml_type type>
static bool ggml_cuda_prefill_plane_glu_run(const mmq_args & args, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_PQ2_0, "PQ2_0 only");
    if (!ggml_cuda_prefill_plane_enabled() || !ggml_cuda_mmq_pq2_j128_supported(args) ||
            args.ncols_max < 128 || args.ncols_max%128 != 0 || args.ncols_y != args.ncols_max ||
            args.nrows_x <= 0 || args.nrows_dst != args.nrows_x/2 ||
            args.stride_row_x != args.ncols_x/QK_PQ2_0 || args.add_res || args.add_dst || args.dst_hi) {
        return false;
    }
    const int device=ggml_cuda_get_device();
    if (ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_DGX_SPARK) return false;
    const auto * e=ggml_cuda_prefill_plane_find(args.x);
    if (!e || e->device != device || e->ncols != args.ncols_x || e->nrows != args.nrows_x) return false;
    constexpr int SHARED=2*128*144+64*144+2*64*32;
    const bool alias=ggml_cuda_prefill_plane_d4_alias_enabled();
    const bool owner=alias && ggml_cuda_prefill_plane_owner_enabled();
    const bool d1=args.d1_y && owner; // other explicit plane modes retain the compatible original consumer
    const auto kernel=d1 ? mul_mat_q_pq2_plane<false,true,true,true,true> :
                      (owner?mul_mat_q_pq2_plane<false,true,true,true>:
                      alias?mul_mat_q_pq2_plane<false,true,true,false>:mul_mat_q_pq2_plane<false,true,false,false>);
    static bool attr[2][2][2][GGML_CUDA_MAX_DEVICES]={};
    if (!attr[d1][alias][owner][device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
        CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,(int)cudaSharedmemCarveoutMaxShared));
        attr[d1][alias][owner][device]=true;
    }
    static bool logged=false;
    if(d1 && !logged){GGML_LOG_INFO("PREFILL_D1_CONSUMER planeGU rows=%lld\n",(long long)args.ncols_y);logged=true;}
    const int nkb=args.ncols_x/QK_PQ2_0,ntx=args.ncols_max/128;
    const uint16_t * scales=(const uint16_t *)(e->data+(int64_t)e->nrows*nkb*32);
    ggml_cuda_kernel_launch(kernel,
        ggml_cuda_kernel_launch_params(dim3((args.nrows_x/64)*ntx),dim3(32,8),SHARED,stream),
        e->data,scales,args.y,args.dst,nkb,(int)args.ncols_y,(int)args.nrows_dst,ntx,(const float *)nullptr);
    return true;
}
