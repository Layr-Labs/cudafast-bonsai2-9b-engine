// Standalone exact Q8 decode-layout fixup comparison on GB10.
#include "mmq.cuh"
#include "fixup_static_grid_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); exit(2); } } while(0)
template<ggml_type TYPE> static void old_fixup(float *d,float *p,int rows,int cols,int stride,int nkb,int grid) {
    const uint3 one=init_fastdiv_values(1),kb=init_fastdiv_values(nkb);
    mul_mat_q_stream_k_fixup_reference<TYPE,16,false><<<dim3(grid,4),dim3(32,4)>>>(
        nullptr,nullptr,d,p,kb,rows,cols,stride,one,stride*cols,one,stride*cols,one);
    CHECK(cudaGetLastError());
}
template<ggml_type TYPE> static void new_fixup(float *d,float *p,int rows,int cols,int stride,int nkb,int grid) {
    const uint3 one=init_fastdiv_values(1),kb=init_fastdiv_values(nkb);
    auto kernel = grid == 48 ? mul_mat_q_stream_k_fixup<TYPE,16,false,48> : mul_mat_q_stream_k_fixup<TYPE,16,false>;
    kernel<<<dim3(grid,4),dim3(32,4)>>>(
        nullptr,nullptr,d,p,kb,rows,cols,stride,one,stride*cols,one,stride*cols,one);
    CHECK(cudaGetLastError());
}
template<class F> static float measure(F f) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for(int i=0;i<10;++i)f(); std::vector<float> times;
    for(int r=0;r<3;++r) { CHECK(cudaEventRecord(a)); for(int i=0;i<100;++i)f(); CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b)); float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); times.push_back(ms*10); }
    CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b)); std::sort(times.begin(),times.end());return times[1];
}
template<ggml_type TYPE> static int run() {
    std::mt19937 rng(20261017); cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,0));int cases=0;
    for(int rows:{128,1024,2048,4096,12288}) for(int k:{256,2048,4096,12288})
    for(int grid:{prop.multiProcessorCount,2*prop.multiProcessorCount,97})
    for(int cols:{1,7,16}) for(int pad:{0,32}) {
        const int stride=rows+pad,nkb=k/(TYPE==GGML_TYPE_Q8_0?32:128);const size_t n=size_t(stride)*cols,np=size_t(grid)*16*128;
        std::vector<float> hd(n),hp(np);for(auto &v:hd)v=float(int(rng()%2049)-1024)*0.03125f;
        for(auto &v:hp) { v=float(int(rng()%2049)-1024)*0.125f; if(rng()%7==0)v*=32768.0f; }
        float *a,*b,*p;CHECK(cudaMalloc(&a,n*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMalloc(&p,np*4));
        CHECK(cudaMemcpy(a,hd.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,hd.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p,hp.data(),np*4,cudaMemcpyHostToDevice));
        old_fixup<TYPE>(a,p,rows,cols,stride,nkb,grid);new_fixup<TYPE>(b,p,rows,cols,stride,nkb,grid);CHECK(cudaDeviceSynchronize());
        std::vector<float> ha(n),hb(n);CHECK(cudaMemcpy(ha.data(),a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,n*4,cudaMemcpyDeviceToHost));
        if(memcmp(ha.data(),hb.data(),n*4)) { fprintf(stderr,"FAIL rows=%d k=%d grid=%d cols=%d pad=%d\n",rows,k,grid,cols,pad);return 1; }++cases;
        if(cols==16 && !pad && grid==prop.multiProcessorCount && k==4096) {
            float old_us=measure([&]{old_fixup<TYPE>(a,p,rows,cols,stride,nkb,grid);});
            float new_us=measure([&]{new_fixup<TYPE>(b,p,rows,cols,stride,nkb,grid);});
            printf("rows=%d k=%d grid=%d old_us=%.4f new_us=%.4f gain=%.5f\n",rows,k,grid,old_us,new_us,old_us/new_us);fflush(stdout);
        }
        CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(p));
    }
    printf("FIXUP_STATIC_GRID_TYPE_PASS cases=%d exact_bits=true\n",cases);
    return cases;
}

int main() {
    const int q8=run<GGML_TYPE_Q8_0>();
    if(q8!=360) return 1;
    const int pq=run<GGML_TYPE_PQ2_0>();
    if(pq!=360) return 1;
    printf("FIXUP_STATIC_GRID_PASS cases=%d exact_output_bits=true\n",q8+pq);
    return 0;
}
