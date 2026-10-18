#pragma once

// Frozen b9549c7 original plane. Only this symbol is renamed; all shared
// helpers are independently source-pinned. Production does not include this.
template <bool ADD,bool GLU,bool D4_ALIAS=false>
static __global__ void __launch_bounds__(256,2) prefill_plane_owner_b954_reference(
        const unsigned char * codes,const uint16_t * scales,const int * y,float * dst,
        int nkb,int ncols_y,int stride_col_dst,int ntx,const float * residual) {
    constexpr int YBYTES=128*144,XBYTES=64*144,CBYTES=64*32;
    extern __shared__ int plane_smem[];
    const uint32_t smem=(uint32_t)__cvta_generic_to_shared(plane_smem);
    const uint32_t xtile=smem+2*YBYTES, ctiles=xtile+XBYTES, bars=xtile+136;
    const int tid=threadIdx.y*32+threadIdx.x,quarter=tid%4,row=tid/4;
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
            const uint32_t yi=smem+buf*YBYTES+(2*tid%128)*144+(tid/64)*4;
            ggml_cuda_prefill_plane_d4_alias_load(ci,yi,raw.w0,raw.w1);
        } else {
            raw.w0=ggml_cuda_prefill_plane_lds_u32(ci); raw.w1=ggml_cuda_prefill_plane_lds_u32(ci+4);
        }
        raw.w2=raw.w1; raw.sh=0; raw.d=scale;
        ggml_cuda_pq2_j128_xstore(raw,xtile+row*144+quarter*32,quarter);
        if (quarter==0 && kb+1<nkb) scale=sp[(kb+1)*64];
        __syncthreads();
        if constexpr (D4_ALIAS) {
            ggml_cuda_pq2_ws_compute_compact<true>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144,ctiles+buf*CBYTES+wc*32*4);
        } else {
            ggml_cuda_pq2_ws_compute_compact<false>(acc,xtile+wr*32*144,smem+buf*YBYTES+wc*32*144);
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
