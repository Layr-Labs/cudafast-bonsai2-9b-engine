#pragma once
#include "common.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define S1_CHECK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    fprintf(stderr,"CUDA line=%d %s\n",__LINE__,cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <typename T> struct s1_buffer {
    T * p = nullptr;
    size_t n;
    explicit s1_buffer(size_t n_) : n(n_) { S1_CHECK(cudaMalloc(&p,n*sizeof(T))); }
    ~s1_buffer() { cudaFree(p); }
    void upload(const std::vector<T> & h) { S1_CHECK(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> read() const {
        std::vector<T> h(n); S1_CHECK(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return h;
    }
    s1_buffer(const s1_buffer &) = delete;
};

struct s1_error {
    double abs = 0, rel = 0;
    size_t bad = 0, nonfinite = 0, different = 0;
    void add(float a, float b, double atol, double rtol) {
        if (memcmp(&a,&b,4)) ++different;
        if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; ++bad; return; }
        const double e = std::abs(double(a)-double(b));
        abs = std::max(abs,e); rel = std::max(rel,e/std::max(1e-6,std::abs(double(a))));
        if (e > atol+rtol*std::abs(double(a))) ++bad;
    }
};

template <typename F> static cudaGraphExec_t s1_capture(cudaStream_t s, F fn) {
    cudaGraph_t graph; cudaGraphExec_t exec;
    S1_CHECK(cudaStreamSynchronize(s));
    S1_CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal)); fn();
    S1_CHECK(cudaStreamEndCapture(s,&graph));
    S1_CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    S1_CHECK(cudaGraphDestroy(graph)); return exec;
}

static float s1_time(cudaGraphExec_t graph, cudaStream_t s, int replays = 16) {
    cudaEvent_t a,b; S1_CHECK(cudaEventCreate(&a)); S1_CHECK(cudaEventCreate(&b));
    S1_CHECK(cudaEventRecord(a,s));
    for (int i=0;i<replays;++i) S1_CHECK(cudaGraphLaunch(graph,s));
    S1_CHECK(cudaEventRecord(b,s)); S1_CHECK(cudaEventSynchronize(b));
    float ms; S1_CHECK(cudaEventElapsedTime(&ms,a,b));
    S1_CHECK(cudaEventDestroy(a)); S1_CHECK(cudaEventDestroy(b)); return ms*1000/replays;
}

// An actual dependency producer, not an independent eviction kernel. Inputs can
// change at fixed addresses between graph replays.
template <typename T> static __global__ void s1_produce(const T * seed, T * dst, size_t n) {
    ggml_cuda_pdl_lc(); ggml_cuda_pdl_sync();
    for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=size_t(gridDim.x)*blockDim.x) dst[i]=seed[i];
}
template <typename T> static void s1_copy(const T * seed, T * dst, size_t n, cudaStream_t s) {
    ggml_cuda_kernel_launch(s1_produce<T>,ggml_cuda_kernel_launch_params(dim3(48),dim3(256),0,s),seed,dst,n);
}
