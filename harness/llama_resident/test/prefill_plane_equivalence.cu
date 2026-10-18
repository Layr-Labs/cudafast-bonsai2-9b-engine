// Exercise the production load-time cache, exact GLU route, fallback guards,
// and teardown through either original source weight. No model is loaded.
#include "mmq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));
        std::exit(2);
    }
}
static void require(bool value,const char * message) {
    if (!value) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}

int main(int argc,char ** argv) {
    const bool disabled=argc==2 && std::strcmp(argv[1],"--disabled")==0;
    require(argc==1 || disabled,"arguments");
    setenv("GGML_CUDA_PREFILL_PLANE",disabled?"0":"1",1);
    ggml_cuda_set_device(0);
    require(ggml_cuda_info().devices[0].cc==GGML_CUDA_CC_DGX_SPARK,"GB10 required");

    constexpr int F=256,K=256,N=128,NKB=K/QK_PQ2_0;
    constexpr size_t WBYTES=F*NKB*sizeof(block_pq2_0);
    std::mt19937 rng(1729);
    std::vector<block_pq2_0> gate(F*NKB),up(F*NKB);
    for (auto * weight : {&gate,&up}) for (auto & b : *weight) {
        const uint16_t bits=ggml_fp32_to_fp16(float(1+rng()%100)/1000.0f);
        std::memcpy(&b.d,&bits,2);
        for (auto & q:b.qs) q=uint8_t(rng());
    }
    // Packing must preserve negative zero and subnormal scale bits too.
    const uint16_t edge_bits[]={0x0000,0x8000,0x0001,0x03ff,0x0400,0x8400};
    for (size_t i=0;i<sizeof(edge_bits)/sizeof(edge_bits[0]);++i)
        std::memcpy(&gate[i].d,&edge_bits[i],2);
    std::vector<block_q8_1_mmq> y(NKB*N);
    for (auto & b:y) {
        for (auto & d:b.d4) d=float(1+rng()%100)/2000.0f;
        for (auto & q:b.qs) q=int(rng()%256)-128;
    }
    block_pq2_0 * dg=nullptr,* du=nullptr;
    int * dy=nullptr;
    float * old=nullptr,* next=nullptr;
    check(cudaMalloc(&dg,WBYTES)); check(cudaMalloc(&du,WBYTES));
    check(cudaMalloc(&dy,y.size()*sizeof(y[0])));
    check(cudaMalloc(&old,F*N*sizeof(float))); check(cudaMalloc(&next,F*N*sizeof(float)));
    check(cudaMemcpy(du,up.data(),WBYTES,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy,y.data(),y.size()*sizeof(y[0]),cudaMemcpyHostToDevice));

    for (int cycle=0;cycle<2;++cycle) {
        gate.back().qs[0]^=0x5a; // a new registration must see new source bytes
        check(cudaMemcpy(dg,gate.data(),WBYTES,cudaMemcpyHostToDevice));
        const size_t allocated=ggml_cuda_glu_pair_register(dg,du,F,K,GGML_TYPE_PQ2_0);
        require(allocated==(disabled?2:4)*WBYTES,"layout allocation accounting");
        const auto * pair=ggml_cuda_glu_pair_find(dg);
        require(pair && pair->up==du,"pair registration");
        const void * stacked=pair->stacked;
        const auto * packed=ggml_cuda_prefill_plane_find(stacked);
        require(disabled?packed==nullptr:packed!=nullptr,"mode controls plane cache");
        require(ggml_cuda_glu_pair_register(dg,du,F,K,GGML_TYPE_PQ2_0)==0,"duplicate registration");

        if (packed) {
            std::vector<unsigned char> copy(packed->bytes);
            check(cudaMemcpy(copy.data(),packed->data,copy.size(),cudaMemcpyDeviceToHost));
            for (int r=0;r<2*F;++r) for (int kb=0;kb<NKB;++kb) {
                const auto & src=(r%16<8?gate:up)[((r/16)*8+r%8)*NKB+kb];
                const size_t p=((r/64)*NKB+kb)*64+r%64;
                require(std::memcmp(copy.data()+p*32,src.qs,32)==0,"code bytes");
                require(std::memcmp(copy.data()+2*F*NKB*32+p*2,&src.d,2)==0,"scale bits");
            }
        }
        std::vector<block_pq2_0> unchanged(F*NKB);
        check(cudaMemcpy(unchanged.data(),dg,WBYTES,cudaMemcpyDeviceToHost));
        require(std::memcmp(unchanged.data(),gate.data(),WBYTES)==0,"original gate unchanged");
        check(cudaMemcpy(unchanged.data(),du,WBYTES,cudaMemcpyDeviceToHost));
        require(std::memcmp(unchanged.data(),up.data(),WBYTES)==0,"original up unchanged");

        mmq_args args={};
        args.x=(const char *)stacked; args.type_x=GGML_TYPE_PQ2_0; args.y=dy; args.dst=next;
        args.ncols_x=K; args.nrows_x=2*F; args.ncols_dst=N; args.ncols_y=N; args.ncols_max=N;
        args.stride_row_x=NKB; args.nrows_dst=F;
        args.nchannels_x=args.nchannels_y=args.nsamples_x=args.nsamples_y=1;
        const bool routed=ggml_cuda_prefill_plane_glu_run<GGML_TYPE_PQ2_0>(args,nullptr);
        require(routed!=disabled,"production route");
        if (routed) {
            check(cudaFuncSetAttribute(mul_mat_q_pq2_j128<true,false,true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,ggml_cuda_pq2_j128_smem()));
            mul_mat_q_pq2_j128<true,false,true><<<(2*F/64),dim3(32,8),ggml_cuda_pq2_j128_smem()>>>(
                args.x,dy,old,nullptr,2*F,NKB,NKB,N,F,N,1,nullptr,nullptr);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            std::vector<float> a(F*N),b(F*N);
            check(cudaMemcpy(a.data(),old,a.size()*sizeof(float),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),next,b.size()*sizeof(float),cudaMemcpyDeviceToHost));
            require(std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,"bitwise GLU output");
        }
        for (int guard=0;guard<4;++guard) {
            auto bad=args;
            if (guard==0) bad.ncols_max=127;
            if (guard==1) bad.stride_row_x=NKB+1;
            if (guard==2) bad.nrows_x-=128;
            if (guard==3) bad.x=(const char *)dg;
            require(!ggml_cuda_prefill_plane_glu_run<GGML_TYPE_PQ2_0>(bad,nullptr),"unsupported shape fallback");
        }
        ggml_cuda_dl_unregister_range(cycle==0?(void *)dg:(void *)du,WBYTES);
        require(!ggml_cuda_glu_pair_find(dg),"pair teardown through either source");
        require(!ggml_cuda_prefill_plane_find(stacked),"plane teardown before address reuse");
    }
    check(cudaFree(next)); check(cudaFree(old)); check(cudaFree(dy)); check(cudaFree(du)); check(cudaFree(dg));
    std::printf("PASS: plane %s; exact bytes, route, guards, duplicate registration, both teardown sources\n",
                disabled?"disabled":"enabled");
    return 0;
}
