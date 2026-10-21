// Same executable: immutable leader ring vs aligned internal Q8 SoA candidate.
#include "mmq.cuh"
#include "q8_soa_reference.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA_FAIL %s line=%d\n",cudaGetErrorString(e),__LINE__);exit(2);} } while(0)
static unsigned rng=1234567;
static unsigned next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static cudaStream_t stream;
constexpr size_t smem=2*(64*4*sizeof(block_q8_0)+16*sizeof(block_q8_1_mmq))+2*sizeof(uint64_t);
static void launch(bool ref,const char *x,const int *y,float *d,float *p,int rows,int k,int grid,int mode,int cols,int stride) {
 if(ref) reference_q8_j16_dl_ring_tm64<2,2,1><<<2*grid,dim3(32,8),smem,stream>>>(x,y,d,p,rows,k/32,16,stride,cols,mode,nullptr,rows);
 else mul_mat_q_q8_j16_dl_ring_tm64<2,2,1><<<2*grid,dim3(32,8),smem,stream>>>(x,y,d,p,rows,k/32,16,stride,cols,mode,nullptr,rows);
 CHECK(cudaGetLastError());
}
struct Shape{int rows,k;};
int main(){
 CHECK(cudaStreamCreate(&stream));int cases=0,replays=0;
 for(auto sh:{Shape{128,128},Shape{128,256},Shape{128,384},Shape{1024,1024},Shape{4096,4096},Shape{4096,12288},Shape{4096,32768},Shape{12288,4096}}) {
  const int rows=sh.rows,k=sh.k,stride=rows+32,guard=64;
  std::vector<block_q8_0> weights(size_t(rows)*k/32);
  const unsigned short scales[]={0,0x8000,0x2c00,0x3000,0xb000,0x0400};
  for(auto &b:weights){auto h=scales[next()%6];std::memcpy(&b.d,&h,2);for(auto &q:b.qs)q=int(next()%256)-128;}
  char *w,*x,*xr;int *y;float *a,*b,*pa,*pb;
  const size_t nb=size_t(stride)*16+guard,np=size_t(96)*128*16+guard;
  CHECK(cudaMalloc(&w,weights.size()*sizeof(weights[0])));CHECK(cudaMalloc(&x,weights.size()*sizeof(weights[0])));CHECK(cudaMalloc(&xr,weights.size()*sizeof(weights[0])));
  CHECK(cudaMemcpyAsync(w,weights.data(),weights.size()*sizeof(weights[0]),cudaMemcpyHostToDevice,stream));
  const int64_t blocks=weights.size();
  ggml_cuda_dl_repack_kernel<GGML_CUDA_DL_TM64,4,34><<<(blocks+255)/256,256,0,stream>>>(w,x,rows,k/32);CHECK(cudaGetLastError());
  // CPU independently constructs original TM64 AoS, never candidate repacker.
   std::vector<block_q8_0> refpack(weights.size());
   for(size_t d=0;d<weights.size();++d){
     size_t z=d/4; const size_t sub=d%4, row64=z%64; z/=64;
     const size_t stage=z%(k/128), tile=z/(k/128);
     refpack[d]=weights[((tile*64+row64)*(k/128)+stage)*4+sub];
   }
   CHECK(cudaMemcpyAsync(xr,refpack.data(),weights.size()*sizeof(weights[0]),cudaMemcpyHostToDevice,stream));
   std::vector<block_q8_1_mmq> hy(size_t(k/128)*16);
  CHECK(cudaMalloc(&y,hy.size()*sizeof(hy[0])));
  CHECK(cudaMalloc(&a,nb*4));CHECK(cudaMalloc(&b,nb*4));CHECK(cudaMalloc(&pa,np*4));CHECK(cudaMalloc(&pb,np*4));
  std::vector<unsigned> ha(nb),hb(nb),hpa(np),hpb(np);
  for(int mode:{0,1,2})for(int grid:{48,96})for(int cols:{1,7,16}){
   cudaGraph_t graph;cudaGraphExec_t exec;
   CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
   launch(false,x,y,b,pb,rows,k,grid,mode,cols,stride);
   CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));CHECK(cudaGraphDestroy(graph));
   for(int v=0;v<4;++v){
    for(auto &q:hy){for(auto &d:q.d4)d=(v==1?0.0f:float(int(next()%127)-63)/256);for(auto &z:q.qs)z=int(next()%256)-128;}
    CHECK(cudaMemcpyAsync(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemsetAsync(a,0x7e,nb*4,stream));CHECK(cudaMemsetAsync(b,0x7e,nb*4,stream));
    CHECK(cudaMemsetAsync(pa,0x7e,np*4,stream));CHECK(cudaMemsetAsync(pb,0x7e,np*4,stream));
    launch(true,xr,y,a,pa,rows,k,grid,mode,cols,stride);
    if(v==0)launch(false,x,y,b,pb,rows,k,grid,mode,cols,stride);else{CHECK(cudaGraphLaunch(exec,stream));++replays;}
    CHECK(cudaMemcpyAsync(ha.data(),a,nb*4,cudaMemcpyDeviceToHost,stream));CHECK(cudaMemcpyAsync(hb.data(),b,nb*4,cudaMemcpyDeviceToHost,stream));
    CHECK(cudaMemcpyAsync(hpa.data(),pa,np*4,cudaMemcpyDeviceToHost,stream));CHECK(cudaMemcpyAsync(hpb.data(),pb,np*4,cudaMemcpyDeviceToHost,stream));CHECK(cudaStreamSynchronize(stream));
    if(ha!=hb || hpa!=hpb){fprintf(stderr,"SOA_FAIL rows=%d k=%d mode=%d grid=%d cols=%d input=%d output_equal=%d partial_equal=%d\n",rows,k,mode,grid,cols,v,ha==hb,hpa==hpb);return 1;}
    ++cases;
   }
   CHECK(cudaGraphExecDestroy(exec));
  }
  CHECK(cudaFree(w));CHECK(cudaFree(x));CHECK(cudaFree(xr));CHECK(cudaFree(y));CHECK(cudaFree(a));CHECK(cudaFree(b));CHECK(cudaFree(pa));CHECK(cudaFree(pb));
 }
 cudaFuncAttributes olda,newa;CHECK(cudaFuncGetAttributes(&olda,reference_q8_j16_dl_ring_tm64<2,2,1>));CHECK(cudaFuncGetAttributes(&newa,mul_mat_q_q8_j16_dl_ring_tm64<2,2,1>));
 printf("RESOURCE reference_regs=%d candidate_regs=%d reference_local=%zu candidate_local=%zu\n",olda.numRegs,newa.numRegs,olda.localSizeBytes,newa.localSizeBytes);
 CHECK(cudaStreamDestroy(stream));printf("Q8_SOA_PASS bitwise_cases=%d changed_input_graph_replays=%d outputs_partials_guards_exact=true\n",cases,replays);
}
