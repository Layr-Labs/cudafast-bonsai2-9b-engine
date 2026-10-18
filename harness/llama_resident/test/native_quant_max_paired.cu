// Frozen frontier versus candidate fused FWHT quantizer in one executable.
#include "common.cuh"
#include "fwht.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
// MMQ D4 quantization of transformed rows held as v[i] = element base + i*NT + tid of row `row`.
template <int N, int NT>
static __device__ __forceinline__ void ref_fwht_quantize_mmq_d4_store(const float * v, block_q8_1_mmq * y,
        const int64_t base, const int64_t row, const int64_t rows, const int tid, const int lane) {
#pragma unroll
    for(int i=0;i<N/NT;++i) {
        const int64_t col=base+i*NT+tid;
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

// C6a: preserve standalone FWHT butterfly and MMQ scale/rounding; only stores differ.
// SWIGLU: the input is a SwiGLU no other node reads. Each element is formed with the expression of
// unary_gated_op_kernel<op_silu, T> (x activated, g the multiplier, row strides o0/o1 in elements),
// so it is the value that kernel would have stored and the plain variant would load here.
template <typename T, bool SIGNED, bool SWIGLU>
__launch_bounds__(256, 1)
static __global__ void ref_fwht_quantize_mmq_d4(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
                                         int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
#pragma unroll
        for(int i=0;i<NE;++i) {
            if constexpr(SWIGLU) {
                const int64_t col=base+i*NT+tid;
                v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+col]))*float(g[row*o1+col])))*(1.0f/64.0f);
            } else {
                v[i]=float(x[row*k+base+i*NT+tid])*(1.0f/64.0f);
            }
            if constexpr(SIGNED) v[i]*=signs[base+i*NT+tid];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
    ref_fwht_quantize_mmq_d4_store<N,NT>(v,y,base,row,rows,tid,lane);
    GGML_UNUSED(kp); GGML_UNUSED(g); GGML_UNUSED(o0); GGML_UNUSED(o1);
}

// MMQ D4 quantization of transformed rows held as v[i] = element base + i*NT + tid of row `row`.
template <int N, int NT, bool NATIVE_MAX = false>
static __device__ __forceinline__ void candidate_fwht_quantize_mmq_d4_store(const float * v, block_q8_1_mmq * y,
        const int64_t base, const int64_t row, const int64_t rows, const int tid, const int lane) {
#pragma unroll
    for(int i=0;i<N/NT;++i) {
        const int64_t col=base+i*NT+tid;
        const float magnitude = fabsf(v[i]);
        float amax;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        if constexpr (NATIVE_MAX) {
            // Positive normal IEEE magnitudes are ordered exactly as unsigned bits.
            // Preserve the existing fmax reduction for NaNs and denormal-only groups.
            const uint32_t max_bits = __reduce_max_sync(0xffffffffu, __float_as_uint(magnitude));
            amax = max_bits >= 0x00800000u && max_bits <= 0x7f800000u ?
                __uint_as_float(max_bits) : warp_reduce_max<32>(magnitude);
        } else
#endif
        {
            amax = warp_reduce_max<32>(magnitude);
        }
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

// C6a: preserve standalone FWHT butterfly and MMQ scale/rounding; only stores differ.
// SWIGLU: the input is a SwiGLU no other node reads. Each element is formed with the expression of
// unary_gated_op_kernel<op_silu, T> (x activated, g the multiplier, row strides o0/o1 in elements),
// so it is the value that kernel would have stored and the plain variant would load here.
template <typename T, bool SIGNED, bool SWIGLU, bool NATIVE_MAX = false>
__launch_bounds__(256, 1)
static __global__ void candidate_fwht_quantize_mmq_d4(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
                                         int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
#pragma unroll
        for(int i=0;i<NE;++i) {
            if constexpr(SWIGLU) {
                const int64_t col=base+i*NT+tid;
                v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+col]))*float(g[row*o1+col])))*(1.0f/64.0f);
            } else {
                v[i]=float(x[row*k+base+i*NT+tid])*(1.0f/64.0f);
            }
            if constexpr(SIGNED) v[i]*=signs[base+i*NT+tid];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
    candidate_fwht_quantize_mmq_d4_store<N,NT,NATIVE_MAX>(v,y,base,row,rows,tid,lane);
    GGML_UNUSED(kp); GGML_UNUSED(g); GGML_UNUSED(o0); GGML_UNUSED(o1);
}

static float measure(cudaGraphExec_t graph,cudaStream_t s,cudaEvent_t a,cudaEvent_t b) {
 CHECK(cudaEventRecord(a,s)); for(int i=0;i<100;++i) CHECK(cudaGraphLaunch(graph,s));
 CHECK(cudaEventRecord(b,s));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));return ms*10.0f;
}
int main() {
 cudaStream_t s;cudaEvent_t a,b;CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
 std::mt19937 rng(910031);std::normal_distribution<float> d(0,3);int cases=0;
 for(int k:{4096,8192,12288}) for(int rows:{1,16,512}) for(int sign:{0,1}) for(int mode=0;mode<9;++mode) {
  int kp=k+4096;size_t n=size_t(k)*rows;size_t bytes=size_t(kp)*rows/QK8_1_MMQ*sizeof(block_q8_1_mmq);
  std::vector<float>x(n),h(k);for(auto&v:x)v=d(rng);for(auto&v:h)v=(rng()&1)?1:-1;
  for(size_t i=0;i<n;++i) {
   if(mode==1)x[i]=(i&1)?-0.0f:0.0f;
   if(mode==2)x[i]=std::numeric_limits<float>::denorm_min()*float(int(i%31)-15);
   if(mode==3)x[i]=(i&1)?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity();
   if(mode==4)x[i]=std::numeric_limits<float>::quiet_NaN();
   if(mode==5 && i%97==0)x[i]=std::numeric_limits<float>::infinity();
   if(mode==5 && i%101==0)x[i]=std::numeric_limits<float>::quiet_NaN();
   if(mode==6)x[i]=(i&1)?-std::numeric_limits<float>::max():std::numeric_limits<float>::max();
   if(mode==7)x[i]=std::numeric_limits<float>::min()*float(int(i%31)-15);
   if(mode==8){uint32_t bits=rng();std::memcpy(&x[i],&bits,4);}
  }
  float *dx,*ds;block_q8_1_mmq *y[2];CHECK(cudaMalloc(&dx,n*4));CHECK(cudaMalloc(&ds,k*4));CHECK(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,h.data(),k*4,cudaMemcpyHostToDevice));
  cudaGraphExec_t graphs[2];
  for(int impl=0;impl<2;++impl){CHECK(cudaMalloc(&y[impl],bytes));CHECK(cudaMemset(y[impl],0xa5,bytes));cudaGraph_t g;CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
   for(int node=0;node<(mode==0?32:1);++node) {
   if(impl==0){if(sign)ref_fwht_quantize_mmq_d4<float,true,false><<<dim3(kp/4096,rows),256,0,s>>>(dx,nullptr,ds,y[impl],k,kp,rows,k,k);else ref_fwht_quantize_mmq_d4<float,false,false><<<dim3(kp/4096,rows),256,0,s>>>(dx,nullptr,nullptr,y[impl],k,kp,rows,k,k);}
   else {if(sign)candidate_fwht_quantize_mmq_d4<float,true,false,true><<<dim3(kp/4096,rows),256,0,s>>>(dx,nullptr,ds,y[impl],k,kp,rows,k,k);else candidate_fwht_quantize_mmq_d4<float,false,false,true><<<dim3(kp/4096,rows),256,0,s>>>(dx,nullptr,nullptr,y[impl],k,kp,rows,k,k);}
   }
   CHECK(cudaStreamEndCapture(s,&g));CHECK(cudaGraphInstantiate(&graphs[impl],g,0));CHECK(cudaGraphDestroy(g));for(int i=0;i<10;++i)CHECK(cudaGraphLaunch(graphs[impl],s));}
  CHECK(cudaStreamSynchronize(s));std::vector<unsigned char>got[2];for(int i=0;i<2;++i){got[i].resize(bytes);CHECK(cudaMemcpy(got[i].data(),y[i],bytes,cudaMemcpyDeviceToHost));}if(got[0]!=got[1]){fprintf(stderr,"MISMATCH k%d rows%d sign%d\n",k,rows,sign);return 1;}++cases;
  std::vector<float>ratio,bt,ct;for(int pair=0;pair<(mode==0?15:0);++pair){float t[2];for(int j=0;j<2;++j){int impl=(pair+j)&1;t[impl]=measure(graphs[impl],s,a,b)/32.0f;}bt.push_back(t[0]);ct.push_back(t[1]);ratio.push_back(t[0]/t[1]);}
  if(mode==0) { std::sort(ratio.begin(),ratio.end());std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());printf("QUANT_MAX_TIMING k=%d rows=%d signed=%d baseline_us=%.4f candidate_us=%.4f ratio=%.6f min=%.6f max=%.6f\n",k,rows,sign,bt[7],ct[7],ratio[7],ratio.front(),ratio.back());fflush(stdout); }
  for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(graphs[i]));CHECK(cudaFree(y[i]));}CHECK(cudaFree(dx));CHECK(cudaFree(ds));
 }
 printf("QUANT_MAX_PAIRED_PASS byte_exact_cases=%d\n",cases);
}
