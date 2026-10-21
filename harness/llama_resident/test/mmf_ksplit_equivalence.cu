// Bit equivalence of the GB10 K-split f32 mul_mat_f path against the single-block kernel.
// Build in the pinned image from the checkout root:
//   nvcc -O2 -std=c++17 -arch=sm_121 -use_fast_math -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//        -I llama.cpp/ggml/src/ggml-cuda harness/llama_resident/test/mmf_ksplit_equivalence.cu -o mmf-ksplit-equivalence
#include "mmf.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 2; } } while (0)

static float pick(std::mt19937 & rng, int mode) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    switch (mode) {
        case 0: return u(rng);
        case 1: return u(rng) * std::ldexp(1.0f, (int) (rng() % 40) - 20);
        case 2: return (rng() % 7 == 0) ? 0.0f : ((rng() % 2) ? -0.0f : u(rng) * 1e-3f);
        default: return std::round(u(rng) * 8.0f) / 8.0f;
    }
}

int main() {
    constexpr int ncols = 4096, cols = mmf_ksplit_cols, nwarps = mmf_ksplit_nwarps, rpb = MMF_ROWS_PER_BLOCK;
    int checks = 0, mismatches = 0;
    for (int nrb : {1, 2, 4}) {
        const int nrows = nrb * rpb;
        for (int mode = 0; mode < 4; ++mode) {
            for (int seed = 0; seed < 4; ++seed) {
                std::mt19937 rng(1234 + 97*seed + 13*mode + nrb);
                std::vector<float> x(size_t(nrows)*ncols), y(size_t(cols)*ncols);
                for (auto & v : x) v = pick(rng, mode);
                for (auto & v : y) v = pick(rng, mode);
                float *dx, *dy, *d_ref, *d_new, *d_part;
                CHECK(cudaMalloc(&dx, x.size()*4)); CHECK(cudaMalloc(&dy, y.size()*4));
                CHECK(cudaMalloc(&d_ref, size_t(nrows)*cols*4)); CHECK(cudaMalloc(&d_new, size_t(nrows)*cols*4));
                CHECK(cudaMalloc(&d_part, size_t(nrb)*GGML_PAD(cols, 8)*nwarps*rpb*4));
                CHECK(cudaMemcpy(dx, x.data(), x.size()*4, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(dy, y.data(), y.size()*4, cudaMemcpyHostToDevice));
                CHECK(cudaMemset(d_ref, 0xff, size_t(nrows)*cols*4)); CHECK(cudaMemset(d_new, 0xee, size_t(nrows)*cols*4));
                const int smem_ref = std::max(nwarps*16*(32 + mmf_get_padding(GGML_CUDA_CC_DGX_SPARK))*4,
                                              GGML_PAD(cols, 8)*(nwarps*rpb + mmf_get_padding(GGML_CUDA_CC_DGX_SPARK))*4);
                mul_mat_f<float, rpb, cols, nwarps, false><<<dim3(nrb, 1, 1), dim3(32, nwarps, 1), smem_ref>>>(
                    dx, dy, nullptr, d_ref, ncols, cols, 1, ncols, ncols, nrows, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
                mul_mat_f_ksplit_partial<<<nrb*nwarps*(rpb/16)*(cols/8), dim3(32, 1, 1), 16*(32 + mmf_get_padding(GGML_CUDA_CC_DGX_SPARK))*4>>>(
                    dx, dy, d_part, ncols, ncols, ncols);
                mul_mat_f_ksplit_reduce<<<nrb, dim3(rpb, cols, 1)>>>(d_part, d_new, nrows);
                CHECK(cudaDeviceSynchronize());
                std::vector<uint32_t> a(size_t(nrows)*cols), b(size_t(nrows)*cols);
                CHECK(cudaMemcpy(a.data(), d_ref, a.size()*4, cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(b.data(), d_new, b.size()*4, cudaMemcpyDeviceToHost));
                for (size_t k = 0; k < a.size(); ++k) {
                    ++checks;
                    if (a[k] != b[k]) {
                        if (mismatches < 10) fprintf(stderr, "mismatch nrb=%d mode=%d seed=%d idx=%zu %08x %08x\n", nrb, mode, seed, k, a[k], b[k]);
                        ++mismatches;
                    }
                }
                cudaFree(dx); cudaFree(dy); cudaFree(d_ref); cudaFree(d_new); cudaFree(d_part);
            }
        }
    }
    printf("mmf ksplit equivalence: %d outputs compared, %d mismatches\n", checks, mismatches);
    return mismatches == 0 ? 0 : 1;
}
