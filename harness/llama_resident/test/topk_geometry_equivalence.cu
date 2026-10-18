#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "topk_geometry_reference.cuh"
#include "../../../llama.cpp/ggml/src/ggml-cuda/top-k-select.cuh"
#define CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess) { fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e_)); exit(2); } } while(0)

template<class K, class... A> void pdl(K kernel, dim3 grid, dim3 block, cudaStream_t stream, A... args) {
    cudaLaunchAttribute attr{};
    attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim=grid; cfg.blockDim=block; cfg.stream=stream;
    cfg.attrs=&attr; cfg.numAttrs=1;
    CHECK(cudaLaunchKernelEx(&cfg,kernel,args...));
}
static void launch(int impl, bool programmatic, const float *x, int *out, uint64_t *scratch,
                   int cols, int rows, int k, cudaStream_t stream) {
    if (!programmatic) {
        if(impl==0) ref_topk_select_launch(x,out,scratch,cols,rows,k,stream);
        else topk_select_launch(x,out,scratch,cols,rows,k,stream);
        return;
    }
    const int chunks=topk_select_nchunks(cols);
    if(impl==0) {
        pdl(ref_topk_select_partial,dim3(chunks,rows),dim3(REF_TOPK_SELECT_NT),stream,x,scratch,cols,k);
        if(chunks<=REF_TOPK_SELECT_MERGE_WARP) pdl(ref_topk_select_merge_warp,dim3(rows),dim3(32),stream,scratch,out,chunks,k);
        else pdl(ref_topk_select_merge,dim3(rows),dim3(REF_TOPK_SELECT_NT),stream,scratch,out,chunks,k);
    } else {
        pdl(topk_select_partial,dim3(chunks,rows),dim3(TOPK_SELECT_NT),stream,x,scratch,cols,k);
        if(chunks<=TOPK_SELECT_MERGE_WARP) pdl(topk_select_merge_warp,dim3(rows),dim3(32),stream,scratch,out,chunks,k);
        else pdl(topk_select_merge,dim3(rows),dim3(TOPK_SELECT_NT),stream,scratch,out,chunks,k);
    }
}
static uint64_t pack(float f, int col) {
    uint32_t u; memcpy(&u,&f,4); if(u==0x80000000u) u=0;
    const uint32_t ord=(u&0x80000000u)?~u:(u|0x80000000u);
    return (uint64_t(ord)<<32)|uint32_t(~uint32_t(col));
}
static float measure(cudaGraphExec_t graph,cudaStream_t s,cudaEvent_t a,cudaEvent_t b) {
    CHECK(cudaEventRecord(a,s));
    for(int i=0;i<100;++i) CHECK(cudaGraphLaunch(graph,s));
    CHECK(cudaEventRecord(b,s)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); return ms*10;
}
int main() {
    static_assert(TOPK_SELECT_CHUNK==REF_TOPK_SELECT_CHUNK,"fixed partitioning");
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::mt19937 rng(870031); int cases=0;
    for(int cols:{17,1025,8193,248320,262145}) for(int rows:{1,3,16}) for(int k:{1,7,8,9,15,16,32}) {
        if(k>cols) continue;
        float *x; int *out[2]; uint64_t *scratch[2];
        const size_t count=size_t(cols)*rows;
        CHECK(cudaMalloc(&x,count*4));
        for(int i=0;i<2;++i) {
            CHECK(cudaMalloc(&out[i],size_t(rows)*k*4));
            CHECK(cudaMalloc(&scratch[i],topk_select_scratch_elems(cols,rows,k)*8));
        }
        std::vector<float> host(count); std::vector<int> want(size_t(rows)*k),got(want.size());
        for(int pattern=0;pattern<5;++pattern) {
            std::normal_distribution<float> nd(0,5);
            for(size_t i=0;i<count;++i) {
                if(pattern==0) host[i]=nd(rng);
                else if(pattern==1) host[i]=float(int(rng()%5)-2);
                else if(pattern==2) { uint32_t u=(rng()&1)?0x80000000u:0; memcpy(&host[i],&u,4); }
                else if(pattern==3) { uint32_t u=rng(); memcpy(&host[i],&u,4); }
                else host[i]=-float(i%cols);
            }
            for(int r=0;r<rows;++r) {
                std::vector<uint64_t> keys(cols);
                for(int c=0;c<cols;++c) keys[c]=pack(host[size_t(r)*cols+c],c);
                std::partial_sort(keys.begin(),keys.begin()+k,keys.end(),std::greater<uint64_t>());
                for(int j=0;j<k;++j) want[size_t(r)*k+j]=int(~uint32_t(keys[j]));
            }
            CHECK(cudaMemcpyAsync(x,host.data(),count*4,cudaMemcpyHostToDevice,stream));
            for(bool programmatic:{false,true}) for(int impl=0;impl<2;++impl) {
                launch(impl,programmatic,x,out[impl],scratch[impl],cols,rows,k,stream);
                CHECK(cudaMemcpyAsync(got.data(),out[impl],got.size()*4,cudaMemcpyDeviceToHost,stream));
                CHECK(cudaStreamSynchronize(stream));
                if(got!=want) { fprintf(stderr,"FAIL cols=%d rows=%d k=%d pattern=%d pdl=%d impl=%d\n",cols,rows,k,pattern,int(programmatic),impl); return 1; }
                if(impl==1) ++cases;
            }
        }
        if(cols==248320 && (rows==1||rows==16) && (k==8||k==16)) {
            cudaGraphExec_t graphs[2]; cudaEvent_t a,b; CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
            for(int i=0;i<2;++i) {
                cudaGraph_t g; CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
                launch(i,true,x,out[i],scratch[i],cols,rows,k,stream);
                CHECK(cudaStreamEndCapture(stream,&g)); CHECK(cudaGraphInstantiate(&graphs[i],g,0));CHECK(cudaGraphDestroy(g));
                for(int j=0;j<5;++j) CHECK(cudaGraphLaunch(graphs[i],stream));
            }
            CHECK(cudaStreamSynchronize(stream)); std::vector<float> ratios,ts[2];
            for(int pair=0;pair<21;++pair) {
                float t[2]; for(int q=0;q<2;++q) {int i=(pair+q)%2;t[i]=measure(graphs[i],stream,a,b);ts[i].push_back(t[i]);}
                ratios.push_back(t[0]/t[1]);
            }
            for(int i=0;i<2;++i) { std::sort(ts[i].begin(),ts[i].end());CHECK(cudaGraphExecDestroy(graphs[i])); }
            std::sort(ratios.begin(),ratios.end());
            printf("TOPK_GEOMETRY_TIMING cols=%d rows=%d k=%d baseline_us=%.4f candidate_us=%.4f paired_ratio=%.6f min=%.6f max=%.6f\n",cols,rows,k,ts[0][10],ts[1][10],ratios[10],ratios.front(),ratios.back());fflush(stdout);
            CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
        }
        for(int i=0;i<2;++i) {CHECK(cudaFree(out[i]));CHECK(cudaFree(scratch[i]));} CHECK(cudaFree(x));
    }
    CHECK(cudaStreamDestroy(stream));
    printf("TOPK_GEOMETRY_PASS exact_cases=%d reference_and_candidate_match_cpu_total_order=1 ordinary_and_pdl=1\n",cases);
    return 0;
}
