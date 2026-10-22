// Generated from production quantize.cu; compare to frozen759 producer.
#include "common.cuh"
#include "fwht.cuh"
#include "quantize.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line%d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
template<int N,int NT>static __device__ __forceinline__ void fwht_quantize_mmq_d1_store(
 const float*v,block_q8_1_mmq*y,int64_t base,int64_t row,int64_t rows,int tid,int lane){
 static_assert(NT%128==0,"whole128 activation groups");
 constexpr int NE=N/NT;
 __shared__ float maxima[N/32];
 const int warp=tid/32,group=warp&~3;
#pragma unroll
 for(int i=0;i<NE;++i){const float m=warp_reduce_max<32>(fabsf(v[i]));if(lane==0)maxima[warp*NE+i]=m;}
 __syncthreads();
 const int64_t col0=base+tid;const int offset=col0%128;y+=(col0/128)*rows+row;
 const int64_t stride=(NT/128)*rows;
#pragma unroll
 for(int i=0;i<NE;++i){
  const float amax=fmaxf(fmaxf(maxima[group*NE+i],maxima[(group+1)*NE+i]),fmaxf(maxima[(group+2)*NE+i],maxima[(group+3)*NE+i]));
  const float inv=amax>0.0f?127.0f/amax:0.0f;
  y[i*stride].qs[offset]=int8_t(roundf(v[i]*inv));
  if(lane==0)y[i*stride].d4[offset/32]=amax>0.0f?1.0f/inv:0.0f;
 }
 __syncthreads();
}

// MMQ D4 quantization of transformed rows held as v[i] = element base + i*NT + tid of row `row`.
template <int N, int NT>
static __device__ __forceinline__ void fwht_quantize_mmq_d4_store(const float * v, block_q8_1_mmq * y,
        const int64_t base, const int64_t row, const int64_t rows, const int tid, const int lane) {
    static_assert(NT % QK8_1_MMQ == 0, "D4 block stride must be integral");
    // Every caller's base is zero or a multiple of N=4096. NT advances
    // complete quantization blocks, leaving the intra-block offset fixed.
    const int64_t col0 = base + tid;
    const int offset = col0 % QK8_1_MMQ;
    y += (col0/QK8_1_MMQ)*rows + row;
    const int64_t stride = (NT/QK8_1_MMQ)*rows;
#pragma unroll
    for(int i=0;i<N/NT;++i) {
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        y[i*stride].qs[offset]=q;
        if(lane==0) y[i*stride].d4[offset/32]=1.0f/d_inv;
    }
}

// C6a: preserve standalone FWHT butterfly and MMQ scale/rounding; only stores differ.
// SWIGLU: the input is a SwiGLU no other node reads. Each element is formed with the expression of
// unary_gated_op_kernel<op_silu, T> (x activated, g the multiplier, row strides o0/o1 in elements),
// so it is the value that kernel would have stored and the plain variant would load here.
template <bool SIGNED, bool D1 = false>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_rms(const float * x, const int64_t stride_row, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, int64_t rows) {
    constexpr int N=4096, NT=256, NE=N/NT, NR=1024, NJ=NR/NT;
    __shared__ float shared[N];
    __shared__ float s_sum[NR/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    x+=row*stride_row;
#pragma unroll
    for(int i=0;i<NE;++i) v[i]=x[i*NT+tid];
#pragma unroll
    for(int j=0;j<NJ;++j) {
        float tmp=0.0f;
#pragma unroll
        for(int i=j;i<NE;i+=NJ) tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[lane]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int col=i*NT+tid;
        const float xn=scale*v[i]*w[col];
        if(norm) norm[row*N+col]=xn;
        v[i]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[col];
    }
    // Prefill D1: transpose warp ownership, keep stages32/64/128 in registers, restore it.
    // Arithmetic stage and operand order is identical; D4 retains the old schedule.
    if constexpr(D1) ggml_cuda_fwht_register_transpose_butterfly<N,NT>(v,shared,tid,lane);
    else ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    if constexpr(D1)fwht_quantize_mmq_d1_store<N,NT>(v,y,0,row,rows,tid,lane);else fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}

#include "rms_transpose_reference.cuh"
template<bool SIGNED> void run(int rows, int pad, int mode, int &cases) {
 constexpr int K=4096;const int stride=K+pad;constexpr size_t GUARD=128;
 const size_t nx=size_t(stride)*rows, nf=size_t(K)*rows*sizeof(float), nq=size_t(K)*rows/128*sizeof(block_q8_1_mmq);
 float *x,*w,*sign,*na,*nb;block_q8_1_mmq *qa,*qb;
 CHECK(cudaMalloc(&x,nx*4));CHECK(cudaMalloc(&w,K*4));CHECK(cudaMalloc(&sign,K*4));
 CHECK(cudaMalloc(&na,nf+GUARD));CHECK(cudaMalloc(&nb,nf+GUARD));CHECK(cudaMalloc(&qa,nq+GUARD));CHECK(cudaMalloc(&qb,nq+GUARD));
 std::mt19937 gen(47000+rows+mode);std::uniform_real_distribution<float> rnd(-1,1);
 std::vector<float> hx(nx),hw(K),hs(K);std::vector<unsigned char>a(std::max(nf,nq)+GUARD),b(a.size());
 auto fill=[&](){for(size_t i=0;i<nx;++i){float v=rnd(gen);if(mode==1)v*=std::ldexp(1.0f,int(gen()%41)-20);if(mode==2)v=(i%5==0)?-0.0f:(i%7==0)?1e-40f:v;hx[i]=v;}for(int i=0;i<K;++i){hw[i]=rnd(gen);hs[i]=(gen()&1)?1.f:-1.f;}CHECK(cudaMemcpy(x,hx.data(),nx*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(w,hw.data(),K*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(sign,hs.data(),K*4,cudaMemcpyHostToDevice));};
 auto poison=[&](){CHECK(cudaMemset(na,0xa5,nf+GUARD));CHECK(cudaMemset(nb,0xa5,nf+GUARD));CHECK(cudaMemset(qa,0xa5,nq+GUARD));CHECK(cudaMemset(qb,0xa5,nq+GUARD));CHECK(cudaDeviceSynchronize());};
 auto launch=[&](cudaStream_t stream){
   fwht_quantize_mmq_d4_rms_reference<SIGNED,true><<<dim3(1,rows,1),256,0,stream>>>(x,stride,w,K,1e-6f,sign,na,qa,rows);
   fwht_quantize_mmq_d4_rms<SIGNED,true><<<dim3(1,rows,1),256,0,stream>>>(x,stride,w,K,1e-6f,sign,nb,qb,rows);
   CHECK(cudaGetLastError());
 };
 auto compare=[&](){CHECK(cudaDeviceSynchronize());for(int q=0;q<2;++q){size_t n=q?nq:nf;CHECK(cudaMemcpy(a.data(),q?(void*)qa:(void*)na,n+GUARD,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),q?(void*)qb:(void*)nb,n+GUARD,cudaMemcpyDeviceToHost));if(memcmp(a.data(),b.data(),n+GUARD)){for(size_t i=0;i<n+GUARD;++i)if(a[i]!=b[i]){fprintf(stderr,"FAIL rows=%d pad=%d mode=%d signed=%d plane=%d byte=%zu old=%u new=%u\n",rows,pad,mode,SIGNED,q,i,a[i],b[i]);std::exit(1);}}for(size_t i=n;i<n+GUARD;++i)if(a[i]!=0xa5){fprintf(stderr,"GUARD_FAIL\n");std::exit(3);}}++cases;};
 fill();poison();launch(nullptr);compare();
 cudaStream_t st;CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));cudaGraph_t graph;cudaGraphExec_t exec;
 CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));launch(st);CHECK(cudaStreamEndCapture(st,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int r=0;r<3;++r){fill();poison();CHECK(cudaGraphLaunch(exec,st));CHECK(cudaStreamSynchronize(st));compare();}
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(st));
 CHECK(cudaFree(x));CHECK(cudaFree(w));CHECK(cudaFree(sign));CHECK(cudaFree(na));CHECK(cudaFree(nb));CHECK(cudaFree(qa));CHECK(cudaFree(qb));
}
int main(){int cases=0;for(int rows:{128,512})for(int pad:{0,96})for(int mode:{0,1,2}){run<false>(rows,pad,mode,cases);run<true>(rows,pad,mode,cases);}printf("RMS_TRANSPOSE_EXACT_PASS cases=%d quant_bytes=true norm_bits=true guards=true changed_input_graph_replays=72\n",cases);}
