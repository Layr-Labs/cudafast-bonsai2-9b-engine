// Standalone exact Q8 decode-layout fixup comparison on GB10.
#include "mmq.cuh"
#include "fixup_early_columns_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
static cudaStream_t test_stream;
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
template<ggml_type TYPE> static void old_fixup(float *d,float *p,int rows,int cols,int stride,int nkb,int grid) {
    const uint3 one=init_fastdiv_values(1),kb=init_fastdiv_values(nkb);
    auto kernel = grid == 48 ? mul_mat_q_stream_k_fixup_early_columns_reference<TYPE,16,false,48> : mul_mat_q_stream_k_fixup_early_columns_reference<TYPE,16,false>;
    kernel<<<dim3(grid,4),dim3(32,4),0,test_stream>>>(
        nullptr,nullptr,d,p,kb,rows,cols,stride,one,stride*cols,one,stride*cols,one);
    CHECK(cudaGetLastError());
}
template<ggml_type TYPE> static void new_fixup(float *d,float *p,int rows,int cols,int stride,int nkb,int grid) {
    const uint3 one=init_fastdiv_values(1),kb=init_fastdiv_values(nkb);
    auto kernel = grid == 48
        ? (cols == 1 ? mul_mat_q_stream_k_fixup<TYPE,16,false,48,true> : mul_mat_q_stream_k_fixup<TYPE,16,false,48>)
        : mul_mat_q_stream_k_fixup<TYPE,16,false>;
    kernel<<<dim3(grid,4),dim3(32,4),0,test_stream>>>(
        nullptr,nullptr,d,p,kb,rows,cols,stride,one,stride*cols,one,stride*cols,one);
    CHECK(cudaGetLastError());
}
template<class F> static cudaGraphExec_t capture(F f) {
    cudaGraph_t graph; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<128;++i)f();
    CHECK(cudaStreamEndCapture(test_stream,&graph));
    CHECK(cudaGraphInstantiate(&exec,graph,0)); CHECK(cudaGraphDestroy(graph)); return exec;
}
static float measure(cudaGraphExec_t graph) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for(int i=0;i<5;++i)CHECK(cudaGraphLaunch(graph,test_stream));
    CHECK(cudaEventRecord(a,test_stream));
    for(int i=0;i<32;++i)CHECK(cudaGraphLaunch(graph,test_stream));
    CHECK(cudaEventRecord(b,test_stream)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); return ms*1000/(128*32);
}
template<ggml_type TYPE> static int run() {
    std::mt19937 rng(20261017); cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,0));int cases=0;
    for(int rows:{128,256,384,512,1024,2048,4096,8192,12288}) for(int k:{128,256,384,512,640,2048,4096,4224,12288,29826048,29826176})
    for(int grid:{prop.multiProcessorCount,2*prop.multiProcessorCount,97})
    for(int cols:{1,2,3,4,7,16}) for(int pad:{0,32}) {
        const int stride=rows+pad,nkb=k/(TYPE==GGML_TYPE_Q8_0?32:128);const size_t n=size_t(stride)*cols,np=size_t(grid)*16*128;
        std::vector<float> hd(n),hp(np);for(auto &v:hd)v=float(int(rng()%2049)-1024)*0.03125f;
        for(auto &v:hp) { v=float(int(rng()%2049)-1024)*0.125f; if(rng()%7==0)v*=32768.0f; }
        float *a,*b,*p;CHECK(cudaMalloc(&a,n*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMalloc(&p,np*4));
        CHECK(cudaMemcpy(a,hd.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,hd.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p,hp.data(),np*4,cudaMemcpyHostToDevice));
        old_fixup<TYPE>(a,p,rows,cols,stride,nkb,grid);new_fixup<TYPE>(b,p,rows,cols,stride,nkb,grid);CHECK(cudaDeviceSynchronize());
        std::vector<float> ha(n),hb(n);CHECK(cudaMemcpy(ha.data(),a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,n*4,cudaMemcpyDeviceToHost));
        if(memcmp(ha.data(),hb.data(),n*4)) { fprintf(stderr,"FAIL rows=%d k=%d grid=%d cols=%d pad=%d\n",rows,k,grid,cols,pad);exit(3); }++cases;
        if((cols==1 || cols==3 || cols==16) && !pad && grid==prop.multiProcessorCount && k==4096 && !getenv("GGML_TEST_EXACT_ONLY")) {
            auto old_graph=capture([&]{old_fixup<TYPE>(a,p,rows,cols,stride,nkb,grid);});
            auto new_graph=capture([&]{new_fixup<TYPE>(b,p,rows,cols,stride,nkb,grid);});
            std::vector<float> gains;
            for(int pair=0;pair<5;++pair) {
                float old_us,new_us;
                if(pair&1) { new_us=measure(new_graph); old_us=measure(old_graph); }
                else { old_us=measure(old_graph); new_us=measure(new_graph); }
                gains.push_back(old_us/new_us);
                printf("type=%d rows=%d k=%d grid=%d cols=%d pair=%d old_us=%.6f new_us=%.6f gain=%.6f\n",int(TYPE),rows,k,grid,cols,pair,old_us,new_us,old_us/new_us); fflush(stdout);
            }
            std::sort(gains.begin(),gains.end());
            printf("FIXUP_SHAPE_MEDIAN type=%d rows=%d cols=%d gain=%.6f\n",int(TYPE),rows,cols,gains[2]);
            CHECK(cudaGraphExecDestroy(old_graph)); CHECK(cudaGraphExecDestroy(new_graph));
        }
        CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(p));
    }
    printf("FIXUP_EARLY_COLUMNS_TYPE_PASS cases=%d exact_bits=true\n",cases);
    return cases;
}

int main() {
    CHECK(cudaStreamCreate(&test_stream));
    int cases=run<GGML_TYPE_Q8_0>()+run<GGML_TYPE_PQ2_0>();
    CHECK(cudaStreamDestroy(test_stream));
    printf("FIXUP_EARLY_COLUMNS_PASS cases=%d exact_output_bits=true\n",cases);
}
