// Compare original signed integer indexing and exact host-precomputed fastdiv.
#include "common.cuh"
#include "fwht.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CHECK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
static __device__ __forceinline__ float fwht_gate_op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
template <typename T, bool SIGNED, int DIVMODE>
__launch_bounds__(256, 1)
static __global__ void paired_fwht_quantize_mmq_d4_glu_perm(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
        int64_t k, int64_t rows, int64_t o0, int64_t o1, int n0, int n1, int64_t s1, int64_t s2,
        const uint3 n0_fd, const uint3 n1_fd) {
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
            const int64_t col=base+i*NT+tid;
            int i0, i1, i2;
            if constexpr (DIVMODE>0) {
                // Host guard proves col fits the original nonnegative int domain.
                const uint2 q0=DIVMODE==2
                    ? make_uint2((uint32_t)col>>n0_fd.y,(uint32_t)col&(n0_fd.z-1))
                    : fast_div_modulo((uint32_t)col,n0_fd);
                const uint2 q1=DIVMODE==2
                    ? make_uint2(q0.x>>n1_fd.y,q0.x&(n1_fd.z-1))
                    : fast_div_modulo(q0.x,n1_fd);
                i0=q0.y; i1=q1.y; i2=q1.x;
            } else {
                const int c=int(col), q=c/n0;
                i0=c-q*n0; i2=q/n1; i1=q-i2*n1;
            }
            const int64_t src=i0+s1*i1+s2*i2;
            v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+src]))*float(g[row*o1+src])))*(1.0f/64.0f);
            if constexpr(SIGNED) v[i]*=signs[col];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int64_t col=base+i*NT+tid;
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

template <bool SIGNED, bool STRIDED>
__launch_bounds__(256, 1)
static __global__ void paired_fwht_quantize_mmq_d4_sigmoid_gate(const float * gate, const float * attn, const float * signs,
        block_q8_1_mmq * y, int64_t k, int64_t rows, int n0, int64_t g1, int64_t gt, int64_t at) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
        // n0==NT: fixed in-row input and one exact g1 stride per register.
        const float * gate_base=gate;
        if constexpr (STRIDED) gate_base += row*gt+(base/NT)*g1+tid;
#pragma unroll
        for(int i=0;i<NE;++i) {
            const int64_t col=base+i*NT+tid;
            float gate_value;
            if constexpr (STRIDED) {
                gate_value=gate_base[i*g1];
            } else {
                const int c=int(col), i1=c/n0, i0=c-i1*n0;
                gate_value=gate[row*gt+i1*g1+i0];
            }
            v[i]=float(fwht_gate_op_sigmoid(float(gate_value))*float(attn[row*at+col]))*(1.0f/64.0f);
            if constexpr(SIGNED) v[i]*=signs[col];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int64_t col=base+i*NT+tid;
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

static float measure(cudaGraphExec_t graph,cudaStream_t s,cudaEvent_t a,cudaEvent_t b) {
 CHECK(cudaEventRecord(a,s)); for(int i=0;i<100;++i) CHECK(cudaGraphLaunch(graph,s));
 CHECK(cudaEventRecord(b,s));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));return ms*10.0f/32.0f;
}
int main() {
 cudaStream_t s;cudaEvent_t a,b;CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
 std::mt19937 rng(165);std::normal_distribution<float>d(0,3);int cases=0;
 for(int kind:{1})for(int k:{4096,8192})for(int rows:{1,16,512})for(bool sign:{false,true}) {
  int kp=k+4096;size_t n=size_t(k)*rows;size_t bytes=size_t(kp)*rows/QK8_1_MMQ*sizeof(block_q8_1_mmq);
  std::vector<float>x(n),g(n*2),h(k);for(auto&v:x)v=d(rng);for(auto&v:g)v=d(rng);for(auto&v:h)v=(rng()&1)?1:-1;
  float *dx,*dg,*ds;block_q8_1_mmq*y[2];CHECK(cudaMalloc(&dx,n*4));CHECK(cudaMalloc(&dg,n*8));CHECK(cudaMalloc(&ds,k*4));CHECK(cudaMemcpy(dx,x.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dg,g.data(),n*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,h.data(),k*4,cudaMemcpyHostToDevice));
  cudaGraphExec_t graphs[2];
  const ggml_cuda_kernel_launch_params lp(dim3(kp/4096,rows,1),dim3(256,1,1),0,s);
  const uint3 fd0=init_fastdiv_values(256),fd1=init_fastdiv_values(2);
  for(int impl=0;impl<2;++impl) {CHECK(cudaMalloc(&y[impl],bytes));CHECK(cudaMemsetAsync(y[impl],0xa5,bytes,s));CHECK(cudaStreamSynchronize(s));cudaGraph_t graph;CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
   for(int node=0;node<32;++node) {
#define DO_GLU(S,F) ggml_cuda_kernel_launch(paired_fwht_quantize_mmq_d4_glu_perm<float,S,F>,lp,dx,dg,sign?ds:nullptr,y[impl],(int64_t)k,(int64_t)rows,(int64_t)k,(int64_t)k,128,2,(int64_t)(k/2),(int64_t)128,fd0,fd1)
#define DO_GATE(S,F) ggml_cuda_kernel_launch(paired_fwht_quantize_mmq_d4_sigmoid_gate<S,F>,lp,dg,dx,sign?ds:nullptr,y[impl],(int64_t)k,(int64_t)rows,256,(int64_t)512,(int64_t)(2*k),(int64_t)k)
    if(kind==0){if(impl){if(sign){DO_GLU(true,2);}else{DO_GLU(false,2);}}else{if(sign){DO_GLU(true,1);}else{DO_GLU(false,1);}}}
    else{if(impl){if(sign){DO_GATE(true,true);}else{DO_GATE(false,true);}}else{if(sign){DO_GATE(true,false);}else{DO_GATE(false,false);}}}
#undef DO_GLU
#undef DO_GATE
   }
   CHECK(cudaStreamEndCapture(s,&graph));CHECK(cudaGraphInstantiate(&graphs[impl],graph,0));CHECK(cudaGraphDestroy(graph));for(int i=0;i<10;++i)CHECK(cudaGraphLaunch(graphs[impl],s));
  }
  CHECK(cudaStreamSynchronize(s));std::vector<unsigned char>got[2];for(int i=0;i<2;++i){got[i].resize(bytes);CHECK(cudaMemcpy(got[i].data(),y[i],bytes,cudaMemcpyDeviceToHost));}if(got[0]!=got[1]){fprintf(stderr,"MISMATCH kind%d k%d rows%d sign%d\n",kind,k,rows,sign);return 1;}++cases;
  std::vector<float>ratio,bt,ct;for(int pair=0;pair<15;++pair){float t[2];for(int j=0;j<2;++j){int impl=(pair+j)&1;t[impl]=measure(graphs[impl],s,a,b);}bt.push_back(t[0]);ct.push_back(t[1]);ratio.push_back(t[0]/t[1]);}
  std::sort(ratio.begin(),ratio.end());std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());printf("GATED_ONLY_STRIDE_TIMING kind=%d k=%d rows=%d signed=%d baseline_us=%.4f candidate_us=%.4f ratio=%.6f min=%.6f max=%.6f\n",kind,k,rows,sign,bt[7],ct[7],ratio[7],ratio.front(),ratio.back());fflush(stdout);
  for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(graphs[i]));CHECK(cudaFree(y[i]));}CHECK(cudaFree(dx));CHECK(cudaFree(dg));CHECK(cudaFree(ds));
 }
 printf("GATED_ONLY_STRIDE_PAIRED_PASS byte_exact_cases=%d\n",cases);
}
