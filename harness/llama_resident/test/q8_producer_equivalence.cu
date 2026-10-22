// Flight065: frozen biased Q8 ring vs exact unbiased Q8 ring, all output/partial/guard bytes.
#include "mmq.cuh"
#include "q8_producer_reference.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(...) do {auto e=(__VA_ARGS__);if(e!=cudaSuccess){fprintf(stderr,"%s line %d: %s\n",#__VA_ARGS__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
static constexpr size_t sm1=2*(64*4*sizeof(block_q8_0)+16*sizeof(block_q8_1_mmq))+16;
static constexpr size_t sm2=sm1+16;
static void launch(bool ref,const char*x,const int*y,float*d,float*p,int rows,int nkb,int cols,int stride,int mode,int grid,cudaStream_t s){
 if(ref)flight065_reference_tm64<GGML_CUDA_DL_FMT_Q8,2,1><<<2*grid,dim3(32,8),sm1,s>>>(x,y,d,p,rows,nkb,16,stride,cols,mode,d,rows);
 else mul_mat_q_q8_j16_dl_ring_tm64<GGML_CUDA_DL_FMT_Q8,2,1><<<2*grid,dim3(32,9),sm2,s>>>(x,y,d,p,rows,nkb,16,stride,cols,mode,d,rows);
 CHECK(cudaGetLastError());
}
int main(){
 CHECK(cudaFuncSetAttribute(flight065_reference_tm64<GGML_CUDA_DL_FMT_Q8,2,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm1));
 CHECK(cudaFuncSetAttribute(mul_mat_q_q8_j16_dl_ring_tm64<GGML_CUDA_DL_FMT_Q8,2,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm2));
 cudaStream_t s;CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));std::mt19937 rng(20261022);int cases=0,replays=0;
 for(auto shape:std::vector<std::pair<int,int>>{{128,256},{128,512},{1024,1024},{4096,4096},{4096,12288},{4096,32768},{12288,4096}})
 for(int cols:{1,7,16})for(int mode:{0,1,2})for(int grid:{48,96}){
  const int rows=shape.first,nkb=shape.second/32,stride=rows+32;
  const size_t nx=size_t(rows)*nkb*sizeof(block_q8_0),ny=size_t(nkb/4)*16*sizeof(block_q8_1_mmq),nd=size_t(stride)*cols+64,np=size_t(grid)*16*128+64;
  std::vector<unsigned short> hx(nx/2),hy(ny/2);char*x;int*y;float*a,*b,*pa,*pb;
  CHECK(cudaMalloc(&x,nx));CHECK(cudaMalloc(&y,ny));CHECK(cudaMalloc(&a,nd*4));CHECK(cudaMalloc(&b,nd*4));CHECK(cudaMalloc(&pa,np*4));CHECK(cudaMalloc(&pb,np*4));
  cudaGraph_t ga,gb;cudaGraphExec_t ea,eb;
  CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));launch(true,x,y,a,pa,rows,nkb,cols,stride,mode,grid,s);CHECK(cudaStreamEndCapture(s,&ga));CHECK(cudaGraphInstantiate(&ea,ga,nullptr,nullptr,0));
  CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));launch(false,x,y,b,pb,rows,nkb,cols,stride,mode,grid,s);CHECK(cudaStreamEndCapture(s,&gb));CHECK(cudaGraphInstantiate(&eb,gb,nullptr,nullptr,0));
  for(int rep=0;rep<3;++rep){
   // Every potential half scale remains finite; all other bits are arbitrary operands.
   for(auto&v:hx)v=unsigned(rng())&0xbbff;for(auto&v:hy)v=unsigned(rng())&0xbbff;
   CHECK(cudaMemcpyAsync(x,hx.data(),nx,cudaMemcpyHostToDevice,s));CHECK(cudaMemcpyAsync(y,hy.data(),ny,cudaMemcpyHostToDevice,s));
   for(auto p:{a,b})CHECK(cudaMemsetAsync(p,0xa5,nd*4,s));for(auto p:{pa,pb})CHECK(cudaMemsetAsync(p,0x5a,np*4,s));
   if(rep==0){launch(true,x,y,a,pa,rows,nkb,cols,stride,mode,grid,s);launch(false,x,y,b,pb,rows,nkb,cols,stride,mode,grid,s);}else{CHECK(cudaGraphLaunch(ea,s));CHECK(cudaGraphLaunch(eb,s));++replays;}
   CHECK(cudaStreamSynchronize(s));std::vector<unsigned char> aa(nd*4),bb(nd*4),ap(np*4),bp(np*4);
   CHECK(cudaMemcpy(aa.data(),a,nd*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bb.data(),b,nd*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ap.data(),pa,np*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bp.data(),pb,np*4,cudaMemcpyDeviceToHost));
   if(aa!=bb||ap!=bp){fprintf(stderr,"FAIL rows=%d K=%d cols=%d mode=%d grid=%d rep=%d output=%d partial=%d\n",rows,shape.second,cols,mode,grid,rep,int(aa!=bb),int(ap!=bp));return 1;}++cases;
  }
  printf("ok rows=%d K=%d cols=%d mode=%d grid=%d\n",rows,shape.second,cols,mode,grid);fflush(stdout);
  CHECK(cudaGraphExecDestroy(ea));CHECK(cudaGraphExecDestroy(eb));CHECK(cudaGraphDestroy(ga));CHECK(cudaGraphDestroy(gb));for(auto p:{a,b,pa,pb})CHECK(cudaFree(p));CHECK(cudaFree(x));CHECK(cudaFree(y));
 }
 CHECK(cudaStreamDestroy(s));printf("PASS Flight065 cases=%d changed_input_graph_replays=%d all_output_partial_guard_bytes_exact=true\n",cases,replays);
}
