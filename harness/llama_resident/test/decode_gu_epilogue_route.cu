// Exact production registry/route gate for the optional decode SwiGLU epilogue.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
template <typename T> struct allocation {
    T * data = nullptr;
    explicit allocation(size_t count) { check(cudaMalloc(&data, count*sizeof(T))); }
    ~allocation() { cudaFree(data); }
    allocation(const allocation &) = delete;
};
struct guarded {
    static constexpr size_t guard = 32;
    size_t count;
    allocation<float> storage;
    explicit guarded(size_t count) : count(count), storage(count + 2*guard) {}
    float * data() const { return storage.data + guard; }
    void poison(cudaStream_t stream) { check(cudaMemsetAsync(storage.data, 0x7e, (count + 2*guard)*sizeof(float), stream)); }
    std::vector<float> read() const {
        std::vector<float> all(count + 2*guard);
        check(cudaMemcpy(all.data(), storage.data, all.size()*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i) {
            uint32_t before, after;
            std::memcpy(&before, &all[i], 4); std::memcpy(&after, &all[guard + count + i], 4);
            require(before == 0x7e7e7e7e && after == 0x7e7e7e7e, "output/fixup guard");
        }
        return {all.begin() + guard, all.begin() + guard + count};
    }
};
static uint32_t next_random(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static void fill_weights(std::vector<block_pq2_0> & weights, uint32_t & random) {
    uint8_t trits[81];
    for (int i = 0; i < 81; ++i) {
        int value = i; trits[i] = 0;
        for (int j = 0; j < 4; ++j) { trits[i] |= uint8_t(value%3) << (2*j); value /= 3; }
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        float scale = float(1 + next_random(random)%100)/1000;
        if (i%127 == 0) scale = 0;
        if (i%131 == 0) scale = -scale;
        if (i%137 == 0) scale = 3e-7f;
        const ggml_fp16_t bits = ggml_fp32_to_fp16(scale);
        std::memcpy(&weights[i].d, &bits, sizeof(bits));
        for (auto & q : weights[i].qs) q = trits[next_random(random)%81];
    }
}
static void fill_activation(std::vector<block_q8_1_mmq> & blocks, uint32_t & random, int phase) {
    for (size_t i = 0; i < blocks.size(); ++i) {
        for (auto & d : blocks[i].d4) d = i%127 == 0 ? 0 : float(1 + next_random(random)%100)/(phase ? 3000 : 2000);
        for (auto & q : blocks[i].qs) q = int(next_random(random)%256) - 128;
    }
}
static void same(const std::vector<float> & expected, const std::vector<float> & got,
                 int rows, int cols, int k, const char * stage, bool partial = false) {
    require(expected.size() == got.size(), "comparison size");
    for (size_t i = 0; i < got.size(); ++i) {
        // Only live columns of stream-K partial tiles are defined.
        if (partial && (i/128)%16 >= size_t(cols)) continue;
        if (std::isfinite(got[i]) && std::memcmp(&expected[i], &got[i], sizeof(float)) == 0) continue;
        std::fprintf(stderr, "M=%d N=%d K=%d stage=%s index=%zu expected=%a actual=%a\n",
                     rows, cols, k, stage, i, expected[i], got[i]);
        require(false, "bitwise equivalence");
    }
}
static __global__ void glu_reference(const float * pair, float * out, int count) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x)
        out[i]=ggml_cuda_op_silu_single(pair[i])*pair[count+i];
}
static void inspect(cudaGraph_t graph) {
    size_t count=0; check(cudaGraphGetNodes(graph,nullptr,&count));
    require(count==2,"production GEMM and plain FWHT only");
    std::vector<cudaGraphNode_t> nodes(count); check(cudaGraphGetNodes(graph,nodes.data(),&count));
    int found=0;
    for (auto node:nodes) {
        cudaGraphNodeType type; check(cudaGraphNodeGetType(node,&type)); require(type==cudaGraphNodeTypeKernel,"kernel node");
        cudaKernelNodeParams p={}; check(cudaGraphKernelNodeGetParams(node,&p));
        const char * name=nullptr; check(cudaFuncGetName(&name,p.func));
        if (std::strstr(name,"mul_mat_q_ptq_gu_ring")) {
            ++found; require(p.gridDim.x==192 && p.blockDim.x==32 && p.blockDim.y==16,"pure production geometry");
            cudaFuncAttributes a; check(cudaFuncGetAttributes(&a,p.func)); int active;
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,p.func,512,p.sharedMemBytes));
            require(a.numRegs<=64 && a.localSizeBytes==0 && active>=2,"retained two-CTA resources");
            std::printf("resource regs=%d local=%zu active=%d\n",a.numRegs,a.localSizeBytes,active);
        }
    }
    require(found==1,"actual linked production epilogue selected");
}
int main(int argc,char ** argv) {
    const bool disabled=argc==2 && !std::strcmp(argv[1],"--disabled");
    const bool ring_off=argc==2 && !std::strcmp(argv[1],"--ring-off");
    const bool jit_off=argc==2 && !std::strcmp(argv[1],"--jit-off");
    const bool unsigned_mma=argc==2 && !std::strcmp(argv[1],"--unsigned");
    require(argc==1 || disabled || ring_off || jit_off || unsigned_mma,"supported mode");
    const bool selected=!disabled && !ring_off && !jit_off && !unsigned_mma;
    setenv("GGML_CUDA_DECODE_GU_EPILOGUE",disabled?"0":"1",1);
    setenv("GGML_CUDA_PTQ_FFN_JIT",jit_off?"0":"1",1);
    setenv("GGML_CUDA_DL_RING_PT",ring_off?"0":"1",1);
    setenv("GGML_CUDA_PTQ1_SIGNED",unsigned_mma?"0":"1",1);
    setenv("GGML_CUDA_DECODE_LAYOUT","1",1); setenv("GGML_CUDA_DECODE_PTQ1","1",1);
    setenv("GGML_CUDA_PREFILL_PLANE","0",1); setenv("GGML_CUDA_PDL","1",1);
    ggml_backend_t backend=ggml_backend_cuda_init(0); require(backend!=nullptr,"CUDA backend");
    ggml_backend_cuda_context cuda(0); ggml_cuda_set_device(0);
    auto registry=ggml_backend_reg_by_name("CUDA");
    auto reg=(size_t (*)(const ggml_tensor *))ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_register_decode_layout");
    auto reg_pair=(size_t (*)(const ggml_tensor *,const ggml_tensor *))ggml_backend_reg_get_proc_address(registry,"ggml_backend_cuda_register_prefill_glu_pair");
    require(reg && reg_pair,"production registration APIs");
    constexpr int F=12288,K=4096,N=16,STEPS=K/128;
    constexpr size_t PACKED=size_t(F)*STEPS*sizeof(block_ptq1_0), VALUES=size_t(F)*N;
    ggml_init_params ip={ggml_tensor_overhead()*8,nullptr,true}; auto * meta=ggml_init(ip);
    auto * w=ggml_new_tensor_2d(meta,GGML_TYPE_PQ2_0,K,F);
    auto * u=ggml_new_tensor_2d(meta,GGML_TYPE_PQ2_0,K,F);
    auto weights_buffer=ggml_backend_alloc_ctx_tensors(meta,backend); require(weights_buffer!=nullptr,"weights allocation");
    ggml_backend_buffer_set_usage(weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * x=ggml_new_tensor_2d(meta,GGML_TYPE_F32,K,N);
    auto * out=ggml_new_tensor_2d(meta,GGML_TYPE_F32,F,N);
    guarded result(VALUES),baseline(2*VALUES),expected_glu(VALUES);
    guarded expected_q(size_t(F/128)*N*sizeof(block_q8_1_mmq)/4),actual_q(expected_q.count);
    out->data=result.data();
    allocation<int> q8(size_t(STEPS)*N*sizeof(block_q8_1_mmq)/4);
    allocation<float> signs(F);
    uint32_t random=0x9f2a48;
    std::vector<float> hs(F); for(auto & s:hs)s=next_random(random)&1?1.0f:-1.0f;
    check(cudaMemcpy(signs.data,hs.data(),F*4,cudaMemcpyHostToDevice));
    std::vector<block_pq2_0> hw(size_t(F)*STEPS),hu(hw.size()); fill_weights(hw,random);fill_weights(hu,random);
    ggml_backend_tensor_set(u,hu.data(),0,ggml_nbytes(u));
    int comparisons=0;
    for(int cycle=0;cycle<2;++cycle) {
        if(cycle)hw[0].qs[0]=uint8_t((hw[0].qs[0]&~3u)|(((hw[0].qs[0]&3u)+1)%3));
        ggml_backend_tensor_set(w,hw.data(),0,ggml_nbytes(w));
        require(reg(w)==PACKED && reg(u)==PACKED,"original PTQ registration");
        require(reg_pair(w,u)==2*(ggml_nbytes(w)+PACKED)+(disabled?0:2*PACKED),"copy accounting");
        const auto * pair=ggml_cuda_glu_pair_find(w->data);
        require(pair && pair->up==u->data && pair->dl_stacked && bool(pair->dl_glu)==!disabled,"registered pair mode");
        require(reg_pair(w,u)==0,"idempotent pair registration");
        if(pair->dl_glu) {
            std::vector<uint32_t> source(2*PACKED/4),packed(source.size());
            check(cudaMemcpy(source.data(),pair->dl_stacked,2*PACKED,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(packed.data(),pair->dl_glu,2*PACKED,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<packed.size();++i) {
                const size_t block=i/7; const int row=int(block/(128*STEPS))*128+int(block%128);
                const int kb=int(block/128)%STEPS,old_row=row/2+(row&1)*F;
                const size_t old_block=(size_t(old_row/128)*STEPS+kb)*128+old_row%128;
                require(packed[i]==source[7*old_block+i%7],"every code/scale word exact permutation");
            }
        }
        auto launch=[&](){return ggml_cuda_mul_mat_q_glu(cuda,w,x,out,pair->stacked,q8.data);};
        for(int n:{1,7,15}) {
            x->ne[1]=out->ne[1]=n; require(!launch(),"non-N16 fallback");
        }
        x->ne[1]=out->ne[1]=N;
        out->nb[1]+=4;require(!launch(),"strided output fallback");out->nb[1]-=4;
        w->nb[1]+=34;require(!launch(),"strided weight fallback");w->nb[1]-=34;
        w->ne[1]-=128;out->ne[0]-=128;require(!launch(),"leading-row view fallback");w->ne[1]+=128;out->ne[0]+=128;
        require(!ggml_cuda_mul_mat_q_glu(cuda,w,x,out,pair->stacked,(char *)q8.data+4),"unaligned D4 fallback");
        if(selected) {
            auto e=*ggml_cuda_dl_find(w->data);e.dl=pair->dl_stacked;e.nrows=2*F;
            mmq_args args={};args.type_x=GGML_TYPE_PQ2_0;args.x=(const char *)w->data;args.y=q8.data;
            args.dst=baseline.data();args.dst_hi=baseline.data()+VALUES;args.nrows_lo=F;
            args.ncols_x=K;args.nrows_x=2*F;args.stride_row_x=STEPS;args.nrows_dst=F;
            args.ncols_dst=args.ncols_y=args.ncols_max=N;args.nchannels_x=args.nchannels_y=args.nsamples_x=args.nsamples_y=1;
            args.stride_channel_dst=args.stride_sample_dst=VALUES;
            cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(cuda.stream(),cudaStreamCaptureModeGlobal));
            require(launch(),"enabled production route");
            fwht_quantize_mmq_d4_cuda(result.data(),GGML_TYPE_F32,signs.data,actual_q.data(),F,F,N,cuda.stream());
            check(cudaStreamEndCapture(cuda.stream(),&graph));inspect(graph);
            check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            for(int phase:{0,1,1,0,2,0}) {
                std::vector<block_q8_1_mmq> y(size_t(STEPS)*N);fill_activation(y,random,phase);
                if(phase==2)for(auto & b:y)for(auto & q:b.qs)q=0;
                check(cudaMemcpyAsync(q8.data,y.data(),y.size()*sizeof(y[0]),cudaMemcpyHostToDevice,cuda.stream()));
                baseline.poison(cuda.stream());expected_glu.poison(cuda.stream());expected_q.poison(cuda.stream());
                ggml_cuda_mmq_dl_run<GGML_TYPE_PQ2_0>(args,e,192,48,nullptr,cuda.stream());
                glu_reference<<<512,256,0,cuda.stream()>>>(baseline.data(),expected_glu.data(),int(VALUES));
                fwht_quantize_mmq_d4_cuda(baseline.data(),GGML_TYPE_F32,signs.data,expected_q.data(),F,F,N,cuda.stream(),baseline.data()+VALUES,F,F);
                result.poison(cuda.stream());actual_q.poison(cuda.stream());
                check(cudaGraphLaunch(executable,cuda.stream()));check(cudaStreamSynchronize(cuda.stream()));
                same(expected_glu.read(),result.read(),F,N,K,"exact production GLU");
                const auto eq=expected_q.read(),aq=actual_q.read();
                require(std::memcmp(eq.data(),aq.data(),eq.size()*4)==0,"exact production D4 bytes");
                baseline.read();comparisons+=2;
            }
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
        } else require(!launch(),"disabled semantic guard fallback");
        ggml_cuda_dl_unregister_range(cycle?u->data:w->data,ggml_nbytes(w));
        require(!ggml_cuda_glu_pair_find(w->data),"teardown through either original source");
        ggml_cuda_dl_unregister_range(cycle?w->data:u->data,ggml_nbytes(w));
    }
    ggml_backend_buffer_free(weights_buffer);ggml_free(meta);ggml_backend_free(backend);
    std::printf("DECODE_GU_EPILOGUE_ROUTE_PASS selected=%d comparisons=%d registrations=2 guards=1\n",int(selected),comparisons);
}
