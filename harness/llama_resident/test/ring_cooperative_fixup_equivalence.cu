// Frozen promoted producer + generic fixup versus cooperative fused producer/fixup.
// nvcc -O3 -std=c++17 -arch=sm_121 -use_fast_math -I llama.cpp/ggml/include \
// -I llama.cpp/ggml/src -I llama.cpp/ggml/src/ggml-cuda this-file.cu -o ring-cursor-test
#include "mmq.cuh"
#include "ring_cooperative_fixup_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s line %d\n",#x,cudaGetErrorString(e),__LINE__); std::exit(2); } } while(0)

template<int FMT,int S,int SPS,bool OLD>
static void launch(const char *x,const int *y,float *d,float *p,int rows,int k,int grid,int mode,int cols,cudaStream_t stream=0) {
    constexpr auto kernel=OLD ? mul_mat_q_j16_dl_ring_reference<FMT,S,SPS> : mul_mat_q_j16_dl_ring<FMT,S,SPS,true>;
    constexpr size_t smem=ggml_cuda_dl_ring_smem<FMT,SPS>(S);
    static bool attr=false;
    if(!attr) { CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)); CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared));attr=true; }
    constexpr int QK=FMT==GGML_CUDA_DL_FMT_Q8 ? 32 : 128;
    if constexpr(OLD) {
        kernel<<<grid,dim3(32,16),smem,stream>>>(x,y,d,p,rows,k/QK,16,rows,cols,mode,nullptr,rows);
        const uint3 fd=init_fastdiv_values(k/QK), one=init_fastdiv_values(1);
        constexpr ggml_type type=FMT==GGML_CUDA_DL_FMT_Q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_PQ2_0;
        mul_mat_q_stream_k_fixup<type,16,false><<<dim3(grid,4),dim3(32,4),0,stream>>>(nullptr,nullptr,d,p,fd,rows,cols,rows,one,0,one,0,one);
    } else {
        cudaLaunchAttribute attrs[2]={};
        attrs[0].id=cudaLaunchAttributeCooperative;attrs[0].val.cooperative=1;
        // Exercise the same attribute combination as the production GB10 route.
        attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attrs[1].val.programmaticStreamSerializationAllowed=1;
        cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(32,16);cfg.dynamicSmemBytes=smem;cfg.stream=stream;cfg.attrs=attrs;cfg.numAttrs=2;
        CHECK(cudaLaunchKernelEx(&cfg,kernel,x,y,d,p,rows,k/QK,16,rows,cols,mode,(float*)nullptr,rows));
    }
    CHECK(cudaGetLastError());
}
template<class F> static float measure(F f) {
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    for(int i=0;i<32;++i)f();std::vector<float> times;
    for(int r=0;r<3;++r) { CHECK(cudaEventRecord(a));for(int i=0;i<50;++i)f();CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*20); }
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));std::sort(times.begin(),times.end());return times[1];
}
struct Shape { int rows,k; };
template<int FMT,int S,int SPS> static int run() {
    std::mt19937 rng(20261018+FMT);int cases=0;
    constexpr bool q8=FMT==GGML_CUDA_DL_FMT_Q8;
    constexpr int qk=q8?32:128,gb=q8?sizeof(block_q8_0):sizeof(block_ptq1_0);
    for(auto sh : {Shape{128,128},Shape{128,256},Shape{128,384},Shape{128,512},Shape{128,640},Shape{4096,384},Shape{4096,640},Shape{1024,1024},Shape{2048,2048},Shape{4096,4096},Shape{12288,4096},Shape{4096,12288}}) {
        const int rows=sh.rows,k=sh.k;std::vector<char> hx(size_t(rows)*(k/qk)*gb);
        for(size_t i=0;i<hx.size();i+=gb) {
            const float scale=rng()%11==0?0.0f:float(int(rng()%63)-31)/128.0f;
            if constexpr(q8) { const half h=__float2half(scale);std::memcpy(hx.data()+i,&h,2);for(int j=0;j<32;++j)hx[i+2+j]=char(int(rng()%255)-127); }
            else { block_ptq1_0 b;for(auto &v:b.qs)v=rng()%243;for(auto &v:b.qh)v=rng()%81;b.d=__float2half(scale);std::memcpy(hx.data()+i,&b,gb); }
        }
        std::vector<block_q8_1_mmq> hy(size_t(k/128)*16);
        for(auto &b:hy) { for(auto &d:b.d4)d=rng()%9==0?0.0f:float(int(rng()%127)-63)/256.0f;for(auto &v:b.qs)v=int(rng()%255)-127; }
        char *x;int *y;float *a,*b,*pa,*pb;
        const size_t bytes=size_t(rows)*16*sizeof(float),pbytes=size_t(96)*128*16*sizeof(float);
        CHECK(cudaMalloc(&x,hx.size()));CHECK(cudaMalloc(&y,hy.size()*sizeof(hy[0])));
        CHECK(cudaMalloc(&a,bytes));CHECK(cudaMalloc(&b,bytes));CHECK(cudaMalloc(&pa,pbytes));CHECK(cudaMalloc(&pb,pbytes));
        CHECK(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice));
        std::vector<uint32_t> ha(bytes/4),hb(ha.size()),hpa(pbytes/4),hpb(hpa.size());
        for(int mode : {0})for(int grid : {1,7,47,48})for(int cols : {1,4,7,16}) {
            CHECK(cudaMemset(a,0xA5,bytes));CHECK(cudaMemset(b,0xA5,bytes));CHECK(cudaMemset(pa,0xA5,pbytes));CHECK(cudaMemset(pb,0xA5,pbytes));
            launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,grid,mode,cols);launch<FMT,S,SPS,false>(x,y,b,pb,rows,k,grid,mode,cols);CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hpa.data(),pa,pbytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,pbytes,cudaMemcpyDeviceToHost));
            if(ha!=hb || hpa!=hpb) { fprintf(stderr,"FAIL fmt=%d S=%d SPS=%d rows=%d k=%d mode=%d grid=%d cols=%d\n",FMT,S,SPS,rows,k,mode,grid,cols);std::exit(1); }
            ++cases;
        }
        if(k>=2048) {
            for(int mode : {0}) {
                std::vector<float> gains, old_times, new_times;
                for (int pair=0; pair<5; ++pair) {
                    float old_us, new_us;
                    if (pair&1) {new_us=measure([&]{launch<FMT,S,SPS,false>(x,y,b,pb,rows,k,48,mode,16);});old_us=measure([&]{launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,48,mode,16);});}
                    else {old_us=measure([&]{launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,48,mode,16);});new_us=measure([&]{launch<FMT,S,SPS,false>(x,y,b,pb,rows,k,48,mode,16);});}
                    gains.push_back(old_us/new_us);old_times.push_back(old_us);new_times.push_back(new_us);
                }
                std::sort(gains.begin(),gains.end());std::sort(old_times.begin(),old_times.end());std::sort(new_times.begin(),new_times.end());
                const float old_us=old_times[2],new_us=new_times[2];
                printf("fmt=%d S=%d SPS=%d rows=%d k=%d mode=%d old_us=%.4f new_us=%.4f gain=%.6f\n",FMT,S,SPS,rows,k,mode,old_us,new_us,gains[2]);
            }
        }
        cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
        CHECK(cudaMemset(a,0xA5,bytes));CHECK(cudaMemset(b,0xA5,bytes));CHECK(cudaMemset(pa,0xA5,pbytes));CHECK(cudaMemset(pb,0xA5,pbytes));
        launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,48,0,16);CHECK(cudaDeviceSynchronize());
        CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        launch<FMT,S,SPS,false>(x,y,b,pb,rows,k,48,0,16,stream);
        CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
        CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hpa.data(),pa,pbytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,pbytes,cudaMemcpyDeviceToHost));
        if(ha!=hb || hpa!=hpb) {fprintf(stderr,"FAIL graph fmt=%d rows=%d k=%d\n",FMT,rows,k);std::exit(1);}++cases;
        CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFree(x));CHECK(cudaFree(y));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(pa));CHECK(cudaFree(pb));
    }
    return cases;
}
int main() {
    int cases=run<GGML_CUDA_DL_FMT_Q8,2,1>()+run<GGML_CUDA_DL_FMT_PTQ1S,3,2>()+run<GGML_CUDA_DL_FMT_PTQ1S,2,2>();
    printf("RING_COOPERATIVE_FIXUP_PASS cases=%d exact_output_and_partial_bits=true\n",cases);
}
