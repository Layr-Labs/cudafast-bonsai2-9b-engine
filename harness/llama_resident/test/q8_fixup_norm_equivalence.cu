// nvcc -std=c++17 -O3 -arch=sm_121a -I.../ggml/src -I.../ggml/include -I.../ggml/src/ggml-cuda ...
// Reference RMS copied verbatim from leader norm.cu, symbol renamed only.
#include "q8-fixup-norm.cuh"
#include "mmq-j16-fixup-pair.cuh"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#define C(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);return 2;} } while(0)
template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void reference_rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

int main() {
    cudaStream_t stream; C(cudaStreamCreate(&stream));
    constexpr int n=16*4096, np=48*16*128, guard=64;
    float *a,*b,*o,*r,*p,*w;
    C(cudaMalloc(&a,(n+guard)*4)); C(cudaMalloc(&b,(n+guard)*4));
    C(cudaMalloc(&o,(n+guard)*4)); C(cudaMalloc(&r,(n+guard)*4));
    C(cudaMalloc(&p,(np+guard)*4)); C(cudaMalloc(&w,4096*4));
    std::vector<float> h(n+guard), hp(np+guard), hw(4096), av(n+guard), bv(n+guard), rv(n+guard), ov(n+guard), pv(np+guard);
    unsigned rng=37417;
    auto next=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
    int cases=0,replays=0;
    for (bool inplace : {false,true}) for (float eps : {1e-6f,1e-5f}) {
        cudaGraph_t g; cudaGraphExec_t exec;
        C(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        q8_fixup_rms_norm_f32<<<16,1024,32*sizeof(float),stream>>>(b,p,w,inplace?b:o,eps);
        C(cudaStreamEndCapture(stream,&g)); C(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0)); C(cudaGraphDestroy(g));
        for(int mode=0;mode<20;++mode) {
            for(int i=0;i<n;++i) {
                float v=float(int(next()%200001)-100000)/7919;
                if(mode==1)v=0;
                if(mode==2)v=-0.0f;
                if(mode==3)v=(i%2?1:-1)*1e-20f;
                if(mode==4)v=(i%2?1:-1)*1e12f;
                if(mode==5)v=(i%2?1:-1)*1e-40f;
                if(mode==6)v=1;
                h[i]=v;
            }
            for(int i=0;i<np;++i) {
                float v=float(int(next()%200001)-100000)/33331;
                if(mode==1 || mode==2)v=-0.0f;
                if(mode==3)v=(i%3?1:-1)*1e-20f;
                if(mode==4)v=(i%2?1:-1)*1e12f;
                if(mode==5)v=(i%2?1:-1)*1e-40f;
                if(mode==6)v=-1;
                hp[i]=v;
            }
            for(int i=0;i<4096;++i)hw[i]=float(int(next()%20001)-10000)/1000;
            for(int i=0;i<guard;++i){h[n+i]=12345;hp[np+i]=34567;}
            C(cudaMemcpyAsync(a,h.data(),h.size()*4,cudaMemcpyHostToDevice,stream));
            C(cudaMemcpyAsync(b,h.data(),h.size()*4,cudaMemcpyHostToDevice,stream));
            C(cudaMemcpyAsync(p,hp.data(),hp.size()*4,cudaMemcpyHostToDevice,stream));
            C(cudaMemcpyAsync(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice,stream));
            C(cudaMemsetAsync(o,0x7e,(n+guard)*4,stream));C(cudaMemsetAsync(r,0x7e,(n+guard)*4,stream));
            mul_mat_q_q8_dl_fixup_pair<<<dim3(32,4),dim3(32,4),0,stream>>>(a,p,4096,16);
            reference_rms_norm_f32<1024,true><<<16,1024,32*sizeof(float),stream>>>(a,inplace?a:r,4096,4096,16*4096,16*4096,eps,
                w,0,0,0,init_fastdiv_values(4096),init_fastdiv_values(1),init_fastdiv_values(1),init_fastdiv_values(1));
            C(cudaGraphLaunch(exec,stream));++replays;
            C(cudaMemcpyAsync(av.data(),a,av.size()*4,cudaMemcpyDeviceToHost,stream));
            C(cudaMemcpyAsync(bv.data(),b,bv.size()*4,cudaMemcpyDeviceToHost,stream));
            C(cudaMemcpyAsync(rv.data(),r,rv.size()*4,cudaMemcpyDeviceToHost,stream));
            C(cudaMemcpyAsync(ov.data(),o,ov.size()*4,cudaMemcpyDeviceToHost,stream));
            C(cudaMemcpyAsync(pv.data(),p,pv.size()*4,cudaMemcpyDeviceToHost,stream));
            C(cudaStreamSynchronize(stream));
            auto check=[&](const std::vector<float>& x,const std::vector<float>& y,const char *what){
                if(std::memcmp(x.data(),y.data(),x.size()*4))for(size_t k=0;k<x.size();++k)if(std::memcmp(&x[k],&y[k],4)){
                    fprintf(stderr,"FAIL %s mode=%d inplace=%d eps=%g idx=%zu ref=%a got=%a\n",what,mode,inplace,eps,k,x[k],y[k]);return false;}return true;};
            if(!check(av,bv,"materialized-or-inplace") || !check(rv,ov,"normalized") || !check(hp,pv,"partials-guards"))return 1;
            ++cases;
        }
        C(cudaGraphExecDestroy(exec));
    }
    C(cudaFree(a));C(cudaFree(b));C(cudaFree(o));C(cudaFree(r));C(cudaFree(p));C(cudaFree(w));C(cudaStreamDestroy(stream));
    printf("Q8_FIXUP_NORM_PASS bitwise_cases=%d changed_input_graph_replays=%d rows=%d guards_exact=true inplace_and_outofplace=true\n",cases,replays,cases*16);
}
