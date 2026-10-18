// Actual packed production kernel versus frozen scalar frontier, including guards.
#include "fattn-common.cuh"
#include "fattn_uniform_v4_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"CUDA line%d: %s\n",__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)

static float measure(cudaGraphExec_t graph,cudaStream_t s,cudaEvent_t a,cudaEvent_t b) {
    CHECK(cudaEventRecord(a,s));
    for(int i=0;i<40;++i) CHECK(cudaGraphLaunch(graph,s));
    CHECK(cudaEventRecord(b,s)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); return ms*25.0f/32.0f;
}

template<int D,int C1,int C2>
static int test(cudaStream_t s,cudaEvent_t a,cudaEvent_t b) {
    std::mt19937 rng(950033+D+C1+C2); std::uniform_real_distribution<float> val(-3.0f,3.0f);
    int cases=0;
    for(int tokens:{1,17,512}) for(int gqa:{1,4,5}) for(int kv:{1,2})
    for(int sequences:{1,2}) for(int bpt:{2,3,7}) for(int mode:{0,1,2,3}) {
        int heads=kv*gqa, nx=(tokens+C1-1)/C1, nz=(gqa+C2-1)/C2;
        int tiles=sequences*kv*nz*nx, blocks=tiles*bpt;
        size_t n=size_t(sequences)*tokens*heads*D;
        size_t meta=size_t(blocks)*2*C1*C2; // float2 metadata, two planes
        size_t storage=meta*2+size_t(blocks)*C1*C2*D; // float units
        std::vector<float> init(n+64), fix(storage+64);
        for(size_t i=0;i<n;++i) init[i]=val(rng);
        for(size_t i=0;i<storage;++i) fix[i]=val(rng);
        for(size_t i=n;i<init.size();++i) init[i]=9876.5f;
        for(size_t i=storage;i<fix.size();++i) fix[i]=4321.25f;
        float2 *m=reinterpret_cast<float2*>(fix.data());
        for(size_t i=0;i<meta;++i) {
            m[i].x=val(rng); m[i].y=std::abs(val(rng))+0.5f;
            if(mode==1) m[i].x=float(int(i%7)-3)*20.0f;
            if(mode==2) m[i].x=(i&1)?std::nextafter(-20.0f,-INFINITY):0.0f;
            if(mode==3) m[i].x=(i&1)?std::nextafter(-20.0f,INFINITY):0.0f;
        }
        float *out[2],*dfix;
        CHECK(cudaMalloc(&dfix,fix.size()*4)); CHECK(cudaMemcpyAsync(dfix,fix.data(),fix.size()*4,cudaMemcpyHostToDevice,s));
        for(int impl=0;impl<2;++impl) { CHECK(cudaMalloc(&out[impl],init.size()*4)); CHECK(cudaMemcpyAsync(out[impl],init.data(),init.size()*4,cudaMemcpyHostToDevice,s)); }
        uint3 fd0=init_fastdiv_values(nx*nz*kv),fd1=init_fastdiv_values(nx*nz),fd2=init_fastdiv_values(nx);
        dim3 grid(tiles,C1,C2);
        reference_flash_attn_stream_k_fixup_uniform<D,C1,C2><<<grid,D,0,s>>>(out[0],(const float2*)dfix,tokens,heads,kv,blocks,gqa,bpt,fd0,fd1,fd2);
        flash_attn_stream_k_fixup_uniform_v4<D,C1,C2><<<grid,D/4,0,s>>>(out[1],(const float2*)dfix,tokens,heads,kv,blocks,gqa,bpt,fd0,fd1,fd2);
        CHECK(cudaGetLastError()); CHECK(cudaStreamSynchronize(s));
        std::vector<float> got[2];
        for(int impl=0;impl<2;++impl) { got[impl].resize(init.size()); CHECK(cudaMemcpy(got[impl].data(),out[impl],init.size()*4,cudaMemcpyDeviceToHost)); }
        if(std::memcmp(got[0].data(),got[1].data(),init.size()*4)!=0) {
            for(size_t i=0;i<init.size();++i) if(std::memcmp(&got[0][i],&got[1][i],4)!=0) {
                fprintf(stderr,"MISMATCH D%d C%d,%d tokens%d gqa%d kv%d seq%d bpt%d mode%d at%zu ref%a new%a\n",D,C1,C2,tokens,gqa,kv,sequences,bpt,mode,i,got[0][i],got[1][i]); break;
            } exit(1);
        }
        for(size_t i=n;i<init.size();++i) if(got[0][i]!=9876.5f) { fprintf(stderr,"OUTPUT_GUARD\n"); exit(1); }
        ++cases;
        if(mode==0&&kv==1&&sequences==1&&gqa==4&&(tokens==1||tokens==512)) {
            cudaGraphExec_t graphs[2];
            for(int impl=0;impl<2;++impl) {
                CHECK(cudaMemcpyAsync(out[impl],init.data(),init.size()*4,cudaMemcpyHostToDevice,s));
                cudaGraph_t graph; CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
                for(int node=0;node<32;++node) {
                    if(impl==0) reference_flash_attn_stream_k_fixup_uniform<D,C1,C2><<<grid,D,0,s>>>(out[impl],(const float2*)dfix,tokens,heads,kv,blocks,gqa,bpt,fd0,fd1,fd2);
                    else flash_attn_stream_k_fixup_uniform_v4<D,C1,C2><<<grid,D/4,0,s>>>(out[impl],(const float2*)dfix,tokens,heads,kv,blocks,gqa,bpt,fd0,fd1,fd2);
                }
                CHECK(cudaStreamEndCapture(s,&graph)); CHECK(cudaGraphInstantiate(&graphs[impl],graph,0)); CHECK(cudaGraphDestroy(graph));
                for(int warmup=0;warmup<10;++warmup) CHECK(cudaGraphLaunch(graphs[impl],s));
            }
            CHECK(cudaStreamSynchronize(s)); std::vector<float>ratio,baseline,candidate;
            for(int pair=0;pair<11;++pair) {
                float t[2]; for(int j=0;j<2;++j) { int impl=(pair+j)&1; t[impl]=measure(graphs[impl],s,a,b); }
                baseline.push_back(t[0]); candidate.push_back(t[1]); ratio.push_back(t[0]/t[1]);
            }
            std::sort(ratio.begin(),ratio.end()); std::sort(baseline.begin(),baseline.end()); std::sort(candidate.begin(),candidate.end());
            printf("FATTN_V4_TIMING D=%d C1=%d C2=%d tokens=%d bpt=%d baseline_us=%.4f candidate_us=%.4f ratio=%.6f min=%.6f max=%.6f\n",D,C1,C2,tokens,bpt,baseline[5],candidate[5],ratio[5],ratio.front(),ratio.back()); fflush(stdout);
            for(int impl=0;impl<2;++impl) CHECK(cudaGraphExecDestroy(graphs[impl]));
        }
        for(int impl=0;impl<2;++impl) CHECK(cudaFree(out[impl])); CHECK(cudaFree(dfix));
    }
    return cases;
}
int main() {
    cudaStream_t s;cudaEvent_t a,b;CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    int cases=test<128,16,4>(s,a,b)+test<256,16,4>(s,a,b);
    printf("FATTN_UNIFORM_V4_PASS exact_output_cases=%d guards_intact=true\n",cases);
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));CHECK(cudaStreamDestroy(s));
}
