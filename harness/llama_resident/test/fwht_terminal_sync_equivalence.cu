// Frozen4403d91 versus terminal barrier specialization: output bits and races.
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA line%d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);} } while(0)
template <int N, int NT>
__device__ __forceinline__ void frozen_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <int N, int NT, bool TERMINAL = false>
__device__ __forceinline__ void candidate_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        // Keep read-before-overwrite protection between shared stages. A
        // terminal caller never reuses s after the last cross-warp read.
        if (!TERMINAL || 2*h < NT) {
            __syncthreads();
        }
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}


template<int NT, bool TERMINAL> __global__ void transform(const float *x,float *y) {
    constexpr int N=4096, NE=N/NT;
    __shared__ float s[N];
    const int t=threadIdx.x;
    const int row=blockIdx.x;
    float v[NE];
#pragma unroll
    for(int j=0;j<NE;++j) v[j]=x[row*N+j*NT+t];
    if constexpr(TERMINAL) candidate_butterfly<N,NT,true>(v,s,t,t%32);
    else frozen_butterfly<N,NT>(v,s,t,t%32);
#pragma unroll
    for(int j=0;j<NE;++j) y[row*N+j*NT+t]=v[j];
}
template<int NT> void test(bool race_only,int &cases) {
    std::mt19937 rng(160001+NT);
    for(int rows:{1,16}) for(int mode=0;mode<(race_only?1:9);++mode) {
        const size_t n=size_t(rows)*4096, bytes=n*sizeof(float), guard=256;
        std::vector<float> x(n);
        for(size_t i=0;i<n;++i) {
            uint32_t bits=rng();
            if(mode==0)x[i]=float(int(bits%65535)-32767)/1024.0f;
            if(mode==1)x[i]=(i&1)?-0.0f:0.0f;
            if(mode==2)x[i]=std::numeric_limits<float>::denorm_min()*float(int(i%31)-15);
            if(mode==3)x[i]=(i&1)?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity();
            if(mode==4)x[i]=std::numeric_limits<float>::quiet_NaN();
            if(mode==5){x[i]=float(int(i%101)-50);if(i%97==0)x[i]=std::numeric_limits<float>::infinity();if(i%101==0)x[i]=std::numeric_limits<float>::quiet_NaN();}
            if(mode==6)x[i]=(i&1)?-std::numeric_limits<float>::max():std::numeric_limits<float>::max();
            if(mode==7)x[i]=std::numeric_limits<float>::min()*float(int(i%31)-15);
            if(mode==8)std::memcpy(&x[i],&bits,4);
        }
        float *dx,*dy[2];CHECK(cudaMalloc(&dx,bytes));CHECK(cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice));
        for(int j=0;j<2;++j){CHECK(cudaMalloc(&dy[j],bytes+guard));CHECK(cudaMemset(dy[j],0xa5,bytes+guard));}
        cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        const int launches=race_only?1:8;
        for(int i=0;i<launches;++i){transform<NT,false><<<rows,NT,0,stream>>>(dx,dy[0]);transform<NT,true><<<rows,NT,0,stream>>>(dx,dy[1]);}
        CHECK(cudaGetLastError());CHECK(cudaStreamSynchronize(stream));
        std::vector<unsigned char>a(bytes+guard),b(bytes+guard);CHECK(cudaMemcpy(a.data(),dy[0],bytes+guard,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),dy[1],bytes+guard,cudaMemcpyDeviceToHost));
        if(a!=b){std::fprintf(stderr,"FWHT mismatch NT%d rows%d mode%d\n",NT,rows,mode);std::exit(1);}
        for(size_t i=bytes;i<bytes+guard;++i)if(a[i]!=0xa5||b[i]!=0xa5){std::fprintf(stderr,"guard changed\n");std::exit(1);}
        CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(dx));CHECK(cudaFree(dy[0]));CHECK(cudaFree(dy[1]));++cases;
    }
}
int main(int argc,char **argv){bool race=argc==2&&std::strcmp(argv[1],"--race")==0;int cases=0;test<256>(race,cases);test<512>(race,cases);test<1024>(race,cases);std::printf("FWHT_TERMINAL_SYNC_PASS cases=%d race=%d bits_exact=true guards_intact=true\n",cases,race);}
