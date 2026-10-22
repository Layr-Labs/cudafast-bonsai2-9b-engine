#include "quantize.cuh"
#include "fwht.cuh"
#include "unary.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line=%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
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

template <bool SIGNED, bool D1 = false, bool ZFIX = false>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_gnorm(const float * x, const float * z, const int64_t z_stride,
        const float * w, const int ncols, const float eps, const uint3 ne1, const int64_t g1, const int64_t g2,
        const int64_t g3, const float * signs, block_q8_1_mmq * y, const int64_t rows, const float * zpartials = nullptr) {
    constexpr int N=4096, NT=256, NE=N/NT, HD=128, HW=HD/WARP_SIZE;
    __shared__ float shared[N];
    __shared__ float s_sum[NE][NT/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE, a0=tid%HD;
    const int64_t row=blockIdx.y;
    float v[NE];
    int64_t head[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#pragma unroll
    for(int i=0;i<NE;++i) {
        const uint2 a21=fast_div_modulo((uint32_t)(i*(NT/HD)+tid/HD), ne1);
        head[i]=row*g3+(int64_t)a21.y*g1+(int64_t)a21.x*g2;
        v[i]=x[head[i]*HD+a0];
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[i][warp]=tmp;
    }
    __syncthreads();
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        if(lane<HW) tmp=s_sum[i][(warp/HW)*HW+lane];
        tmp=warp_reduce_sum(tmp);
        const float mean=tmp/ncols;
        const float scale=rsqrtf(mean+eps);
        const float xn=scale*v[i]*w[a0];
        const int64_t zi = head[i]*z_stride+a0;
        float zv = z[zi];
        if constexpr (ZFIX) {
            // Exact4096x16/grid48 original descending fixup: one predecessor per tile.
            const int tile = (zi%4096)/128;
            const int pred = tile + tile/2;
            const float ps = __fadd_rn(0.0f,zpartials[pred*2048+(zi/4096)*128+zi%128]);
            zv = __fadd_rn(zv,ps);
        }
        const float g=ggml_cuda_op_silu_single(zv)*xn;
        v[i]=g*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[i*NT+tid];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    if constexpr(D1)fwht_quantize_mmq_d1_store<N,NT>(v,y,0,row,rows,tid,lane);else fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}

#include "zfix_gate_reference.cuh"
// Exact narrow specialization of original grid48/32tile fixup: +0 then one predecessor, then tail+sum.
__global__ void reference_fixup(float *z,const float *p){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4096*16)return;int tile=(i%4096)/128;z[i]+=0.0f+p[(tile+tile/2)*2048+(i/4096)*128+i%128];}
void run(int ne1,int g1,int g2,int g3,int mode,int &cases){
 constexpr int K=4096,ROWS=16,N=K*ROWS,GUARD=128;size_t nq=N/128*sizeof(block_q8_1_mmq);size_t np=48*2048;
 float *x,*tail,*z,*p,*w,*sign;block_q8_1_mmq *qa,*qb;
 CHECK(cudaMalloc(&x,N*4));CHECK(cudaMalloc(&tail,N*4));CHECK(cudaMalloc(&z,N*4));CHECK(cudaMalloc(&p,np*4));CHECK(cudaMalloc(&w,128*4));CHECK(cudaMalloc(&sign,K*4));CHECK(cudaMalloc(&qa,nq+GUARD));CHECK(cudaMalloc(&qb,nq+GUARD));
 std::vector<float>hx(N),ht(N),hp(np),hw(128),hs(K);std::vector<unsigned char>a(nq+GUARD),b(a.size());std::mt19937 gen(48000+mode+g1);std::uniform_real_distribution<float>rnd(-2.f,2.f);
 auto fill=[&](){for(int i=0;i<N;++i){hx[i]=rnd(gen);ht[i]=rnd(gen);if(mode==1)ht[i]*=1e-15f;if(mode==2)ht[i]=(i%4==0)?-0.f:(i%7==0)?1e-40f:ht[i];}for(auto &v:hp)v=rnd(gen);if(mode==1)for(auto &v:hp)v*=1e-15f;if(mode==2)for(size_t i=0;i<np;++i)if(i%4==0)hp[i]=-0.f;for(auto &v:hw)v=rnd(gen);for(auto &v:hs)v=(gen()&1)?1.f:-1.f;CHECK(cudaMemcpy(x,hx.data(),N*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(tail,ht.data(),N*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(p,hp.data(),np*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(w,hw.data(),128*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(sign,hs.data(),K*4,cudaMemcpyHostToDevice));};
 auto launch=[&](cudaStream_t st){CHECK(cudaMemcpyAsync(z,tail,N*4,cudaMemcpyDeviceToDevice,st));reference_fixup<<<N/256,256,0,st>>>(z,p);fwht_quantize_mmq_d4_gnorm_reference<true,false><<<dim3(1,ROWS),256,0,st>>>(x,z,128,w,128,1e-6f,init_fastdiv_values(ne1),g1,g2,g3,sign,qa,ROWS);fwht_quantize_mmq_d4_gnorm<true,false,true><<<dim3(1,ROWS),256,0,st>>>(x,tail,128,w,128,1e-6f,init_fastdiv_values(ne1),g1,g2,g3,sign,qb,ROWS,p);};
 auto check=[&](){CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(a.data(),qa,a.size(),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),qb,b.size(),cudaMemcpyDeviceToHost));for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]||(i>=nq&&a[i]!=0xa5)){fprintf(stderr,"ZFIX_MISMATCH ne1=%d g=%d,%d,%d mode=%d byte=%zu old=%u new=%u\n",ne1,g1,g2,g3,mode,i,a[i],b[i]);exit(1);}++cases;};
 fill();CHECK(cudaMemset(qa,0xa5,a.size()));CHECK(cudaMemset(qb,0xa5,b.size()));CHECK(cudaDeviceSynchronize());launch(0);check();
 cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaGraph_t graph;cudaGraphExec_t exec;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(stream);CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));for(int replay=0;replay<3;++replay){fill();CHECK(cudaMemset(qa,0xa5,a.size()));CHECK(cudaMemset(qb,0xa5,b.size()));CHECK(cudaDeviceSynchronize());CHECK(cudaGraphLaunch(exec,stream));check();}CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));for(auto v:{(void*)x,(void*)tail,(void*)z,(void*)p,(void*)w,(void*)sign,(void*)qa,(void*)qb})CHECK(cudaFree(v));
}
int main(){int cases=0;for(int mode:{0,1,2}){run(32,1,32,32,mode,cases);run(16,2,1,32,mode,cases);run(32,16,512,1,mode,cases);}printf("ZFIX_GATE_EXACT_PASS cases=%d quant_bytes=true guards=true changed_input_graph_replays=27\n",cases);}
