// GB10 signed PTQ decode-layout ring: byte exactness and stage-depth microbenchmark.
// nvcc -O3 -std=c++17 -arch=sm_121 -use_fast_math \
//   -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//   -I llama.cpp/ggml/src/ggml-cuda \
//   harness/llama_resident/test/ptq_ring_stage_equivalence.cu -o ptq-ring-stage-test
#include "mmq.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s at %d\n", #x, cudaGetErrorString(e), __LINE__); std::exit(2); } } while (0)

struct Shape { int rows, k; };

template<int S, int SPS>
static void ring(const char *x, const int *y, float *out, int rows, int k, int grid=0, int mode=1, float *tmp=nullptr, int cols=16) {
    constexpr auto kernel = mul_mat_q_j16_dl_ring<GGML_CUDA_DL_FMT_PTQ1S, S, SPS>;
    constexpr size_t smem = ggml_cuda_dl_ring_smem<GGML_CUDA_DL_FMT_PTQ1S, SPS>(S);
    static bool attr = false;
    if (!attr) {
        CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        attr = true;
    }
    kernel<<<grid ? grid : rows/128, dim3(32,16), smem>>>(x,y,out,tmp,rows,k/128,16,rows,cols,mode,nullptr,rows);
    CHECK(cudaGetLastError());
}

static void plain(const char *x, const int *y, float *out, int rows, int k) {
    ring<3,2>(x,y,out,rows,k);
    CHECK(cudaGetLastError());
}

template<class F>
static float measure(F launch) {
    cudaEvent_t begin,end;
    CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
    for (int i=0;i<8;++i) launch();
    std::vector<float> samples;
    for (int round=0;round<3;++round) {
        CHECK(cudaEventRecord(begin));
        for (int i=0;i<80;++i) launch();
        CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
        float ms; CHECK(cudaEventElapsedTime(&ms,begin,end)); samples.push_back(ms*1000/80);
    }
    CHECK(cudaEventDestroy(begin)); CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(),samples.end()); return samples[1];
}

int main() {
    std::mt19937 rng(20261017);
    int cases=0;
    for (Shape shape : {Shape{1024,4096},Shape{2048,4096},Shape{4096,4096},
                        Shape{12288,4096},Shape{4096,12288},Shape{4096,2048}}) {
        const int rows=shape.rows, k=shape.k;
        for (int mode=0;mode<2;++mode) {
            std::vector<char> hx(size_t(rows)*(k/128)*sizeof(block_ptq1_0));
            for (size_t i=0;i<hx.size();i+=sizeof(block_ptq1_0)) {
                const float scale = mode && rng()%7==0 ? 0.0f : float(int(rng()%63)-31)/128.0f;
                block_ptq1_0 block;
                for (auto &q:block.qs) q=rng()%243;
                for (auto &q:block.qh) q=rng()%81;
                block.d=__float2half(scale); std::memcpy(hx.data()+i,&block,sizeof(block));
            }
            std::vector<block_q8_1_mmq> hy(size_t(k/128)*16);
            for (auto &b:hy) {
                for (auto &d:b.d4) d=mode && rng()%9==0 ? 0.0f : float(int(rng()%127)-63)/256.0f;
                for (auto &q:b.qs) q=int(rng()%255)-127;
            }
            char *x; int *y; float *out; const size_t bytes=size_t(rows)*16*sizeof(float);
            CHECK(cudaMalloc(&x,hx.size())); CHECK(cudaMalloc(&y,hy.size()*sizeof(hy[0])));
            CHECK(cudaMalloc(&out,bytes));
            CHECK(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice));
            std::vector<uint32_t> ref(size_t(rows)*16), got(ref.size());
            plain(x,y,out,rows,k); CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(ref.data(),out,bytes,cudaMemcpyDeviceToHost));
            for (int stage : {32,22,24}) {
                CHECK(cudaMemset(out,0xFF,bytes));
                if(stage==32) ring<3,2>(x,y,out,rows,k);
                if(stage==22) ring<2,2>(x,y,out,rows,k);
                if(stage==24) ring<2,4>(x,y,out,rows,k);
                CHECK(cudaDeviceSynchronize()); CHECK(cudaMemcpy(got.data(),out,bytes,cudaMemcpyDeviceToHost));
                for(size_t i=0;i<ref.size();++i) if(ref[i]!=got[i]) {
                    fprintf(stderr,"FAIL rows=%d k=%d mode=%d S=%d index=%zu %08x != %08x\n",
                        rows,k,mode,stage,i,ref[i],got[i]); return 1;
                }
                ++cases;
            }
            // Compare completed tiles and trailing partial buffers under the production stream-K partition.
            cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,0));
            for (int grid : {prop.multiProcessorCount,2*prop.multiProcessorCount}) {
                float *ref_dst,*ref_tmp,*new_tmp; const size_t tmp_bytes=size_t(grid)*16*128*sizeof(float);
                CHECK(cudaMalloc(&ref_dst,bytes)); CHECK(cudaMalloc(&ref_tmp,tmp_bytes)); CHECK(cudaMalloc(&new_tmp,tmp_bytes));
                for (int cols : {1,7,16}) for (int candidate_steps : {2,4}) {
                    CHECK(cudaMemset(ref_dst,0,bytes)); CHECK(cudaMemset(out,0,bytes));
                    CHECK(cudaMemset(ref_tmp,0,tmp_bytes)); CHECK(cudaMemset(new_tmp,0,tmp_bytes));
                    ring<3,2>(x,y,ref_dst,rows,k,grid,0,ref_tmp,cols);
                    if(candidate_steps==2) ring<2,2>(x,y,out,rows,k,grid,0,new_tmp,cols);
                    else ring<2,4>(x,y,out,rows,k,grid,0,new_tmp,cols);
                    CHECK(cudaDeviceSynchronize());
                    std::vector<unsigned char> a(bytes+tmp_bytes),b(bytes+tmp_bytes);
                    CHECK(cudaMemcpy(a.data(),ref_dst,bytes,cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(b.data(),out,bytes,cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(a.data()+bytes,ref_tmp,tmp_bytes,cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(b.data()+bytes,new_tmp,tmp_bytes,cudaMemcpyDeviceToHost));
                    if(a!=b) { fprintf(stderr,"stream-K mismatch rows=%d k=%d grid=%d cols=%d mode=%d\n",rows,k,grid,cols,mode); return 1; }
                    ++cases;
                }
                CHECK(cudaFree(ref_dst)); CHECK(cudaFree(ref_tmp)); CHECK(cudaFree(new_tmp));
            }
            if(mode==0) {
                const float us_plain=measure([&]{plain(x,y,out,rows,k);});
                const float us3=measure([&]{ring<3,2>(x,y,out,rows,k);});
                const float us2=measure([&]{ring<2,2>(x,y,out,rows,k);});
                const float us4=measure([&]{ring<2,4>(x,y,out,rows,k);});
                printf("rows=%d k=%d plain_us=%.4f S3P2_us=%.4f S2P2_us=%.4f S2P4_us=%.4f S2P2_gain=%.5f S2P4_gain=%.5f\n",
                    rows,k,us_plain,us3,us2,us4,us3/us2,us3/us4); fflush(stdout);
            }
            CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(out));
        }
    }
    printf("PTQ_RING_STAGE_PASS cases=%d exact_output_and_stream_k_partial_bits=true\n",cases);
    return 0;
}
