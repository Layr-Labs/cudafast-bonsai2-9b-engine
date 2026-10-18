// Same-executable A/B of promoted239 TM64 versus operand-native Q8 groups.
// nvcc -O3 -std=c++17 -arch=sm_121 -use_fast_math -I llama.cpp/ggml/include \
// -I llama.cpp/ggml/src -I llama.cpp/ggml/src/ggml-cuda this-file.cu -o ring-tail-test
#include "mmq.cuh"
#include "q8_native_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s line %d\n",#x,cudaGetErrorString(e),__LINE__); std::exit(2); } } while(0)

static cudaStream_t test_stream;
template<int FMT,int S,int SPS,bool OLD>
static void launch(const char *x,const int *y,float *d,float *p,int rows,int k,int grid,int mode,int cols) {
    constexpr auto kernel=OLD ? reference239_q8_j16_dl_ring_tm64<FMT,S,SPS> : mul_mat_q_q8_j16_dl_ring_tm64<FMT,S,SPS>;
    constexpr int ROWS = 64;
    constexpr size_t smem=S*SPS*(ROWS*ggml_cuda_dl_fmt<FMT>::GB+16*sizeof(block_q8_1_mmq)) + S*sizeof(uint64_t);
    static bool attr=false;
    if(!attr) { CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));attr=true; }
    constexpr int QK=FMT==GGML_CUDA_DL_FMT_Q8 ? 32 : 128;
    kernel<<<grid*(128/ROWS),dim3(32,ROWS/8),smem,test_stream>>>(x,y,d,p,rows,k/QK,16,rows,cols,mode,nullptr,rows);
    CHECK(cudaGetLastError());
}
template<class F> static float measure(F f) {
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    for(int i=0;i<32;++i)f();std::vector<float> times;
    for(int r=0;r<3;++r) { CHECK(cudaEventRecord(a,test_stream));for(int i=0;i<200;++i)f();CHECK(cudaEventRecord(b,test_stream));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*5); }
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));std::sort(times.begin(),times.end());return times[1];
}
template<class F> static cudaGraphExec_t graph_for(F f) {
    cudaGraph_t g; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeGlobal));
    for (int i=0;i<32;++i) f();
    CHECK(cudaStreamEndCapture(test_stream,&g));
    CHECK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
    CHECK(cudaGraphDestroy(g)); return exec;
}
static float graph_time(cudaGraphExec_t g) {
    cudaEvent_t a,b; CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
    CHECK(cudaEventRecord(a,test_stream));
    for(int i=0;i<8;++i) CHECK(cudaGraphLaunch(g,test_stream));
    CHECK(cudaEventRecord(b,test_stream));CHECK(cudaEventSynchronize(b));
    float ms;CHECK(cudaEventElapsedTime(&ms,a,b));
    CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));return ms*1000/(8*32);
}
struct Shape { int rows,k; };
template<int FMT,int S,int SPS> static int run() {
    std::mt19937 rng(20261018+FMT);int cases=0;
    constexpr bool q8=FMT==GGML_CUDA_DL_FMT_Q8;
    constexpr int qk=q8?32:128,gb=q8?sizeof(block_q8_0):sizeof(block_ptq1_0);
    for(auto sh : {Shape{128,256},Shape{128,512},Shape{1024,1024},Shape{2048,2048},Shape{4096,4096},Shape{12288,4096},Shape{4096,12288},Shape{4096,32768}}) {
        const int rows=sh.rows,k=sh.k;std::vector<char> hx(size_t(rows)*(k/qk)*gb);
        for(size_t i=0;i<hx.size();i+=gb) {
            const float scale=rng()%11==0?0.0f:float(int(rng()%63)-31)/128.0f;
            if constexpr(q8) { const half h=__float2half(scale);std::memcpy(hx.data()+i,&h,2);for(int j=0;j<32;++j)hx[i+2+j]=char(int(rng()%255)-127); }
            else { block_ptq1_0 b;for(auto &v:b.qs)v=rng()%243;for(auto &v:b.qh)v=rng()%81;b.d=__float2half(scale);std::memcpy(hx.data()+i,&b,gb); }
        }
        std::vector<block_q8_1_mmq> hy(size_t(k/128)*16);
        for(auto &b:hy) { for(auto &d:b.d4)d=rng()%9==0?0.0f:float(int(rng()%127)-63)/256.0f;for(auto &v:b.qs)v=int(rng()%255)-127; }
        std::vector<char> hsource(hx.size()), hnative(hx.size());
        const int nkb = k/32, nst = nkb/4;
        for (int row = 0; row < rows; ++row) for (int kb = 0; kb < nkb; ++kb) {
            const size_t group = (size_t(row/64)*nst+kb/4)*64+row%64;
            const size_t old = (group*4+kb%4)*34;
            std::memcpy(hsource.data()+(size_t(row)*nkb+kb)*34,hx.data()+old,34);
            std::memcpy(hnative.data()+group*136+(kb%4)*32,hx.data()+old+2,32);
            std::memcpy(hnative.data()+group*136+128+(kb%4)*2,hx.data()+old,2);
        }
        char *x,*source,*native;int *y;float *a,*b,*pa,*pb;
        const size_t bytes=size_t(rows)*16*sizeof(float),pbytes=size_t(96)*128*16*sizeof(float);
        CHECK(cudaMalloc(&x,hx.size()));CHECK(cudaMalloc(&y,hy.size()*sizeof(hy[0])));
        CHECK(cudaMalloc(&source,hsource.size()));CHECK(cudaMalloc(&native,hnative.size()));
        CHECK(cudaMemcpy(source,hsource.data(),hsource.size(),cudaMemcpyHostToDevice));
        ggml_cuda_dl_repack<4,sizeof(block_q8_0)>(source,native,rows,nkb,GGML_CUDA_DL_TM64,test_stream);
        CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(test_stream));
        std::vector<char> actual(hnative.size());
        CHECK(cudaMemcpy(actual.data(),native,actual.size(),cudaMemcpyDeviceToHost));
        if(actual!=hnative) {fprintf(stderr,"REPACK_FAIL rows=%d k=%d\n",rows,k);std::exit(1);}
        // Invert native group bytes independently: every source byte is retained.
        std::vector<char> inverse(hx.size());
        for (size_t g=0;g<actual.size()/136;++g) for(int s=0;s<4;++s) {
            std::memcpy(inverse.data()+g*136+s*34,actual.data()+g*136+128+s*2,2);
            std::memcpy(inverse.data()+g*136+s*34+2,actual.data()+g*136+s*32,32);
        }
        if(inverse!=hx) {fprintf(stderr,"INVERSE_FAIL\n");std::exit(1);}
        CHECK(cudaMalloc(&a,bytes));CHECK(cudaMalloc(&b,bytes));CHECK(cudaMalloc(&pa,pbytes));CHECK(cudaMalloc(&pb,pbytes));
        CHECK(cudaMemcpy(x,hx.data(),hx.size(),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice));
        std::vector<uint32_t> ha(bytes/4),hb(ha.size()),hpa(pbytes/4),hpb(hpa.size());
        for(int mode : {0,1,2})for(int grid : {48,96})for(int cols : {1,7,16}) {
            // All sentinel writes precede the nonblocking-stream kernels explicitly.
            CHECK(cudaMemsetAsync(a,0xA5,bytes,test_stream));CHECK(cudaMemsetAsync(b,0xA5,bytes,test_stream));CHECK(cudaMemsetAsync(pa,0xA5,pbytes,test_stream));CHECK(cudaMemsetAsync(pb,0xA5,pbytes,test_stream));
            launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,grid,mode,cols);launch<FMT,S,SPS,false>(native,y,b,pb,rows,k,grid,mode,cols);CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(hpa.data(),pa,pbytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,pbytes,cudaMemcpyDeviceToHost));
            if(ha!=hb || hpa!=hpb) { fprintf(stderr,"FAIL fmt=%d S=%d SPS=%d rows=%d k=%d mode=%d grid=%d cols=%d\n",FMT,S,SPS,rows,k,mode,grid,cols);std::exit(1); }
            ++cases;
        }
        if((rows==12288&&k==4096)||(rows==4096&&(k==12288||k==32768))) {
            for(int mode : {0,1}) {
                auto ga=graph_for([&]{launch<FMT,S,SPS,true>(x,y,a,pa,rows,k,48,mode,16);});
                auto gb=graph_for([&]{launch<FMT,S,SPS,false>(native,y,b,pb,rows,k,48,mode,16);});
                for(int r=0;r<10;++r) {
                    float old_us,new_us;
                    if(r%2==0) {old_us=graph_time(ga);new_us=graph_time(gb);}
                    else {new_us=graph_time(gb);old_us=graph_time(ga);}
                    if(r>=2) printf("GRAPH rows=%d k=%d mode=%d round=%d old_us=%.4f new_us=%.4f gain=%.6f\n",rows,k,mode,r,old_us,new_us,old_us/new_us);
                }
                CHECK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));
                CHECK(cudaMemcpy(hpa.data(),pa,pbytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,pbytes,cudaMemcpyDeviceToHost));
                if(ha!=hb||hpa!=hpb) {fprintf(stderr,"GRAPH_FAIL\n");std::exit(1);}
                CHECK(cudaGraphExecDestroy(ga));CHECK(cudaGraphExecDestroy(gb));
            }
        }
        CHECK(cudaFree(source));CHECK(cudaFree(native));CHECK(cudaFree(x));CHECK(cudaFree(y));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(pa));CHECK(cudaFree(pb));
    }
    return cases;
}
int main() {
    CHECK(cudaStreamCreateWithFlags(&test_stream,cudaStreamNonBlocking));
    int cases=run<GGML_CUDA_DL_FMT_Q8,2,1>();
    CHECK(cudaStreamDestroy(test_stream));
    printf("Q8_NATIVE_PASS cases=%d exact_output_and_partial_bits=true\n",cases);
}
