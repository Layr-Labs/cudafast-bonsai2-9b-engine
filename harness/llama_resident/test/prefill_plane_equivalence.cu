// Exercise the production load-time cache, exact GLU route, fallback guards,
// and teardown through either original source weight. No model is loaded.
#include "mmq.cuh"
#include "prefill_plane_accepted_478_reference.cuh"
#include <cmath>
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

static size_t output_mismatches(const std::vector<float> & reference,const std::vector<float> & actual,
        size_t guard,int features,const char * label,int cycle,int replay) {
    require(reference.size()==actual.size(),"comparison shape");
    size_t count=0,first=reference.size();
    uint32_t first_ref=0,first_actual=0;
    for(size_t i=0;i<reference.size();++i) {
        uint32_t r,a; std::memcpy(&r,&reference[i],4); std::memcpy(&a,&actual[i],4);
        if(r==a) continue;
        if(count++==0) { first=i; first_ref=r; first_actual=a; }
    }
    if(count) {
        const bool interior=first>=guard && first<reference.size()-guard;
        const size_t index=interior?first-guard:0;
        std::fprintf(stderr,"PLANE_MISMATCH label=%s cycle=%d replay=%d words=%zu first_word=%zu first_byte=%zu "
                    "interior=%d column=%lld feature=%lld ref_bits=%08x actual_bits=%08x\n",
                    label,cycle,replay,count,first,first*4,int(interior),
                    interior?(long long)(index/features):-1LL,interior?(long long)(index%features):-1LL,
                    first_ref,first_actual);
    }
    return count;
}

template<bool COMPACT> static void diagnose_j128(const mmq_args & args,int nkb,float * dst,float * storage,
        size_t out_alloc,size_t guard,int features,cudaStream_t stream,const std::vector<float> & reference) {
    const auto kernel=mul_mat_q_pq2_j128<true,false,true,COMPACT>;
    check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ggml_cuda_pq2_j128_smem()));
    check(cudaMemsetAsync(storage,0xff,out_alloc,stream));
    ggml_cuda_kernel_launch(kernel,
        ggml_cuda_kernel_launch_params(dim3((2*features/64)*(args.ncols_y/128)),dim3(32,8),
                                       ggml_cuda_pq2_j128_smem(),stream),
        args.x,args.y,dst,(float *)nullptr,2*features,nkb,nkb,(int)args.ncols_y,features,
        (int)args.ncols_y,(int)args.ncols_y/128,(const float *)nullptr,(float *)nullptr);
    check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
    std::vector<float> result(reference.size());
    check(cudaMemcpy(result.data(),storage,out_alloc,cudaMemcpyDeviceToHost));
    const char * label=COMPACT?"accepted_plane_vs_J128_compact":"accepted_plane_vs_J128_unrolled";
    const size_t count=output_mismatches(reference,result,guard,features,label,0,0);
    // This is a diagnostic of two pre-existing kernels. The alias regression
    // gate below remains an exact comparison against accepted478's plane path.
    std::printf("PLANE_J128_DIAGNOSTIC compact=%d mismatch_words=%zu total_words=%zu\n",
                int(COMPACT),count,result.size());
}

int main(int argc,char ** argv) {
    bool disabled=false,alias_requested=false,alias_off=false,small=false;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--disabled")==0) disabled=true;
        else if(std::strcmp(argv[i],"--alias")==0) alias_requested=true;
        else if(std::strcmp(argv[i],"--alias-off")==0) alias_off=true;
        else if(std::strcmp(argv[i],"--small")==0) small=true;
        else require(false,"arguments");
    }
    require(!(alias_requested && alias_off),"unambiguous alias mode");
    setenv("GGML_CUDA_PREFILL_PLANE",disabled?"0":"1",1);
    if(alias_requested || alias_off) setenv("GGML_CUDA_PREFILL_PLANE_D4_ALIAS",alias_requested?"1":"0",1);
    else {
        unsetenv("GGML_CUDA_PREFILL_PLANE_D4_ALIAS"); // exercise the real source default
        require(std::getenv("GGML_CUDA_PREFILL_PLANE_D4_ALIAS")==nullptr,"alias environment genuinely absent");
    }
    const bool alias=!alias_off;
    require(ggml_cuda_prefill_plane_d4_alias_enabled()==alias,"alias explicit one / explicit zero / default ON");
    ggml_cuda_set_device(0);
    require(ggml_cuda_info().devices[0].cc==GGML_CUDA_CC_DGX_SPARK,"GB10 required");

    const bool actual=!small; // no arguments: absent environment at the actual model shape
    const int F=actual?12288:256,K=actual?4096:256,N=actual?512:128,NKB=K/QK_PQ2_0;
    std::printf("PLANE_MODE alias_env=%s expected_alias=%d F=%d K=%d N=%d plane_disabled=%d\n",
                alias_requested?"1":alias_off?"0":"absent",int(alias),F,K,N,int(disabled));
    const size_t WBYTES=size_t(F)*NKB*sizeof(block_pq2_0);
    constexpr int GUARD=64;
    const size_t OUT=size_t(F)*N,OUT_ALLOC=(OUT+2*GUARD)*sizeof(float);
    cudaStream_t stream=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
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
    float * old=nullptr,* next=nullptr,* old_storage=nullptr,* next_storage=nullptr;
    check(cudaMalloc(&dg,WBYTES)); check(cudaMalloc(&du,WBYTES));
    check(cudaMalloc(&dy,y.size()*sizeof(y[0])));
    check(cudaMalloc(&old_storage,OUT_ALLOC)); check(cudaMalloc(&next_storage,OUT_ALLOC));
    old=old_storage+GUARD; next=next_storage+GUARD;
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
        check(cudaDeviceSynchronize()); // warm attributes and finish the initial eager route
        cudaGraph_t graph=nullptr;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        require(ggml_cuda_prefill_plane_glu_run<GGML_TYPE_PQ2_0>(args,stream)==routed,"captured production route");
        check(cudaStreamEndCapture(stream,&graph));
        size_t nodes=0; check(cudaGraphGetNodes(graph,nullptr,&nodes));
        require(nodes==size_t(routed?1:0),"exact production graph node count");
        if (routed) {
            cudaGraphNode_t node=nullptr; check(cudaGraphGetNodes(graph,&node,&nodes));
            cudaGraphNodeType kind; check(cudaGraphNodeGetType(node,&kind));
            require(kind==cudaGraphNodeTypeKernel,"production graph kernel");
            cudaKernelNodeParams p{}; check(cudaGraphKernelNodeGetParams(node,&p));
            const void * expected=alias?(const void *)mul_mat_q_pq2_plane<false,true,true>:
                                        (const void *)mul_mat_q_pq2_plane<false,true,false>;
            require(p.func==expected,"production alias/original function identity");
            require(p.gridDim.x==unsigned((2*F/64)*(N/128)) && p.gridDim.y==1 && p.gridDim.z==1 &&
                    p.blockDim.x==32 && p.blockDim.y==8 && p.blockDim.z==1 && p.sharedMemBytes==50176,
                    "unchanged production launch geometry");
            const void * cp=nullptr,* sp=nullptr,* yp=nullptr,* dp=nullptr,* residual=nullptr;
            int dims[4]{};
            std::memcpy(&cp,p.kernelParams[0],sizeof(cp)); std::memcpy(&sp,p.kernelParams[1],sizeof(sp));
            std::memcpy(&yp,p.kernelParams[2],sizeof(yp)); std::memcpy(&dp,p.kernelParams[3],sizeof(dp));
            for(int i=0;i<4;++i) std::memcpy(dims+i,p.kernelParams[4+i],sizeof(int));
            std::memcpy(&residual,p.kernelParams[8],sizeof(residual));
            require(cp==packed->data && sp==packed->data+size_t(2*F)*NKB*32 && yp==dy && dp==next &&
                    dims[0]==NKB && dims[1]==N && dims[2]==F && dims[3]==N/128 && !residual,
                    "all nine production kernel arguments");
            cudaFuncAttributes resource{}; check(cudaFuncGetAttributes(&resource,expected));
            int ctas=0; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas,expected,256,50176));
            require(resource.numRegs<=128 && resource.localSizeBytes==0 && ctas>=2,"production resource ceiling");
            std::printf("PLANE_ROUTE alias=%d F=%d K=%d N=%d regs=%d local=%zu CTA=%d\n",
                        int(alias),F,K,N,resource.numRegs,resource.localSizeBytes,ctas);
            cudaGraphExec_t exec=nullptr; check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            const auto reference=prefill_plane_accepted_478_reference<false,true>;
            check(cudaFuncSetAttribute(reference,cudaFuncAttributeMaxDynamicSharedMemorySize,50176));
            check(cudaFuncSetAttribute(reference,cudaFuncAttributePreferredSharedMemoryCarveout,
                                      (int)cudaSharedmemCarveoutMaxShared));
            std::vector<float> a(OUT+2*GUARD),b(a.size()),first(OUT);
            const unsigned seeds[]={101,202,101,101};
            for(int replay=0;replay<4;++replay) {
                rng.seed(seeds[replay]);
                for(auto & block:y) {
                    for(auto & d:block.d4) d=float(1+rng()%100)/2000.0f;
                    for(auto & q:block.qs) q=replay==2?0:int(rng()%256)-128;
                }
                // The graph stream is nonblocking. Default-stream cudaMemset
                // may return before its large fill completes and would race
                // graph stores; enqueue all replay inputs/poisons on this stream.
                check(cudaMemcpyAsync(dy,y.data(),y.size()*sizeof(y[0]),cudaMemcpyHostToDevice,stream));
                check(cudaMemsetAsync(old_storage,0xff,OUT_ALLOC,stream));
                check(cudaMemsetAsync(next_storage,0xff,OUT_ALLOC,stream));
                check(cudaGraphLaunch(exec,stream)); check(cudaStreamSynchronize(stream));
                ggml_cuda_kernel_launch(reference,
                    ggml_cuda_kernel_launch_params(dim3((2*F/64)*(N/128)),dim3(32,8),50176,stream),
                    (const unsigned char *)packed->data,(const uint16_t *)(packed->data+size_t(2*F)*NKB*32),
                    (const int *)dy,old,NKB,N,F,N/128,(const float *)nullptr);
                check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
                check(cudaMemcpy(a.data(),old_storage,OUT_ALLOC,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(),next_storage,OUT_ALLOC,cudaMemcpyDeviceToHost));
                const size_t mismatches=output_mismatches(a,b,GUARD,F,"accepted_plane_vs_production",cycle,replay);
                if(cycle==0 && replay==0) {
                    diagnose_j128<false>(args,NKB,old,old_storage,OUT_ALLOC,GUARD,F,stream,a);
                    diagnose_j128<true>(args,NKB,old,old_storage,OUT_ALLOC,GUARD,F,stream,a);
                }
                require(mismatches==0,"all output/guard bytes versus accepted478 plane after changed-input graph");
                for(size_t i=0;i<OUT;++i) require(std::isfinite(b[GUARD+i]),"every output written and finite");
                for(int i=0;i<GUARD;++i) for(size_t j:{size_t(i),size_t(GUARD)+OUT+i}) {
                    uint32_t bits; std::memcpy(&bits,&b[j],4); require(bits==0xffffffffu,"output canaries");
                }
                if(replay==0) std::memcpy(first.data(),b.data()+GUARD,OUT*sizeof(float));
                if(replay==3) require(std::memcmp(first.data(),b.data()+GUARD,OUT*sizeof(float))==0,"A/B/zero/A deterministic replay");
                std::vector<block_q8_1_mmq> unchanged_y(y.size());
                check(cudaMemcpy(unchanged_y.data(),dy,y.size()*sizeof(y[0]),cudaMemcpyDeviceToHost));
                require(std::memcmp(unchanged_y.data(),y.data(),y.size()*sizeof(y[0]))==0,"activation bytes unchanged");
            }
            check(cudaGraphExecDestroy(exec));
        }
        check(cudaGraphDestroy(graph)); // all graphs die before registration/storage teardown
        for (int guard=0;guard<10;++guard) {
            auto bad=args;
            if (guard==0) bad.ncols_max=127;
            if (guard==1) bad.stride_row_x=NKB+1;
            if (guard==2) bad.nrows_x-=128;
            if (guard==3) bad.x=(const char *)dg;
            if (guard==4) bad.ncols_y-=1;
            if (guard==5) bad.ncols_x-=128;
            if (guard==6) bad.add_res=old;
            if (guard==7) bad.add_dst=old;
            if (guard==8) bad.dst_hi=old;
            if (guard==9) bad.y=(const int *)((const char *)dy+4);
            require(!ggml_cuda_prefill_plane_glu_run<GGML_TYPE_PQ2_0>(bad,nullptr),"unsupported shape fallback");
        }
        ggml_cuda_dl_unregister_range(cycle==0?(void *)dg:(void *)du,WBYTES);
        require(!ggml_cuda_glu_pair_find(dg),"pair teardown through either source");
        require(!ggml_cuda_prefill_plane_find(stacked),"plane teardown before address reuse");
    }
    check(cudaStreamDestroy(stream));
    check(cudaFree(next_storage)); check(cudaFree(old_storage)); check(cudaFree(dy)); check(cudaFree(du)); check(cudaFree(dg));
    std::printf("PASS: plane %s alias=%d; exact bytes, route, 10 guards, changed graphs, duplicate registration, both teardown sources\n",
                disabled?"disabled":"enabled",int(alias));
    return 0;
}
