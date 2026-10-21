// Frozen promoted241 GU versus guarded one-tile final epilogue.
#include "mmq.cuh"
#include "ptq_single_tile_reference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s line%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<bool OLD, bool ONE_TILE> static void launch_impl(const char*x,const int*y,float*d,float*p,int rows,int k,int grid,int cols,cudaStream_t stream){
 constexpr auto fn=OLD?reference248_mul_mat_q_ptq_gu_ring<GGML_CUDA_DL_FMT_PTQ1S,3,2,true,2,true>:mul_mat_q_ptq_gu_ring<GGML_CUDA_DL_FMT_PTQ1S,3,2,true,2,true,ONE_TILE>;
 constexpr size_t sm=ggml_cuda_dl_ring_smem<GGML_CUDA_DL_FMT_PTQ1S,2>(3);
 static bool attr=false;if(!attr){CK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));CK(cudaFuncSetAttribute(fn,cudaFuncAttributePreferredSharedMemoryCarveout,100));attr=true;}
 fn<<<grid,dim3(32,16),sm,stream>>>(x,y,d,p,rows,k/128,16,rows/2,cols,1,d,rows/2);CK(cudaGetLastError());}
template<bool OLD> static void launch(const char*x,const int*y,float*d,float*p,int rows,int k,int grid,int cols,cudaStream_t stream=0){
 if(grid==rows/128)launch_impl<OLD,true>(x,y,d,p,rows,k,grid,cols,stream);
 else launch_impl<OLD,false>(x,y,d,p,rows,k,grid,cols,stream);
}
template<class F> static float measure(F f){cudaStream_t st;CK(cudaStreamCreate(&st));cudaGraph_t g;cudaGraphExec_t ex;
 CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));for(int i=0;i<8;++i)f(st);CK(cudaStreamEndCapture(st,&g));CK(cudaGraphInstantiate(&ex,g,0));
 CK(cudaGraphLaunch(ex,st));cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,st));for(int i=0;i<8;++i)CK(cudaGraphLaunch(ex,st));CK(cudaEventRecord(b,st));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));
 CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaGraphExecDestroy(ex));CK(cudaGraphDestroy(g));CK(cudaStreamDestroy(st));return ms*1000/64;}
struct Shape{int rows,k;};
int main(int argc,char**argv){bool exact=argc>1;std::mt19937 rng(20261019);int cases=0;
 for(int family=0; family<3; ++family)for(auto sh:{Shape{128,128},Shape{256,256},Shape{384,384},Shape{1024,640},Shape{4096,2048},Shape{24576,4096},Shape{24576,12288}}){int rows=sh.rows,k=sh.k;std::vector<block_ptq1_0> hx(size_t(rows)*k/128);std::vector<block_q8_1_mmq>hy(size_t(k/128)*16);
 for(auto &b:hx){for(auto &v:b.qs)v=rng()%243;for(auto &v:b.qh)v=rng()%81;b.d=__float2half(rng()%11==0?0.0f:float(int(rng()%31)-15)/256);}
 if(family==1)for(auto &b:hx){for(auto &v:b.qs)v=0;for(auto &v:b.qh)v=0;}
  for(auto &b:hy){for(auto &d:b.d4)d=float(int(rng()%127)-63)/256;for(auto &v:b.qs)v=int(rng()%256)-128;}
 if(family==2)for(auto &b:hy){for(auto &v:b.qs)v=-128;}
  char*x;int*y;float*a,*b,*pa,*pb;size_t words=size_t(rows/2)*16+64,bytes=words*4,pbytes=size_t(rows/128+1)*128*16*4;
 CK(cudaMalloc(&x,hx.size()*sizeof(hx[0])));CK(cudaMalloc(&y,hy.size()*sizeof(hy[0])));CK(cudaMalloc(&a,bytes));CK(cudaMalloc(&b,bytes));CK(cudaMalloc(&pa,pbytes));CK(cudaMalloc(&pb,pbytes));CK(cudaMemcpy(x,hx.data(),hx.size()*sizeof(hx[0]),cudaMemcpyHostToDevice));CK(cudaMemcpy(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice));
 std::vector<uint32_t> ha(words),hb(words),hpa(pbytes/4),hpb(pbytes/4);
 for(int grid:{rows/128,std::max(1,rows/256),std::max(1,rows/384)})for(int cols:{1,7,16}){CK(cudaMemset(a,0xa5,bytes));CK(cudaMemset(b,0xa5,bytes));CK(cudaMemset(pa,0xa5,pbytes));CK(cudaMemset(pb,0xa5,pbytes));launch<true>(x,y,a,pa,rows,k,grid,cols);launch<false>(x,y,b,pb,rows,k,grid,cols);CK(cudaDeviceSynchronize());CK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hpa.data(),pa,pbytes,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hpb.data(),pb,pbytes,cudaMemcpyDeviceToHost));if(ha!=hb||hpa!=hpb){fprintf(stderr,"FAIL rows=%d k=%d grid=%d cols=%d\n",rows,k,grid,cols);return 1;}++cases;}
 if(!exact&&family==0&&k>=2048){for(int pair=0;pair<9;++pair){float old_us,new_us;if(pair&1){new_us=measure([&](cudaStream_t st){launch<false>(x,y,b,pb,rows,k,rows/128,16,st);});old_us=measure([&](cudaStream_t st){launch<true>(x,y,a,pa,rows,k,rows/128,16,st);});}else{old_us=measure([&](cudaStream_t st){launch<true>(x,y,a,pa,rows,k,rows/128,16,st);});new_us=measure([&](cudaStream_t st){launch<false>(x,y,b,pb,rows,k,rows/128,16,st);});}printf("GRAPH rows=%d k=%d pair=%d old_us=%.4f new_us=%.4f gain=%.6f\n",rows,k,pair,old_us,new_us,old_us/new_us);fflush(stdout);}}
 CK(cudaFree(x));CK(cudaFree(y));CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(pa));CK(cudaFree(pb));}
 printf("PTQ_SINGLE_TILE_PASS cases=%d output_and_guards_bitexact=true\n",cases);}
