// Exact frozen split-K producer + pair-fixup versus local split reset.
#include "mmq.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#define CHECK(x) do { auto status=(x); if(status!=cudaSuccess){fprintf(stderr,"CUDA_FAIL %s %s line=%d\n",#x,cudaGetErrorString(status),__LINE__);exit(2);} } while(0)
static unsigned rng=1234567;
static unsigned next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static cudaStream_t stream;
constexpr size_t smem=2*(64*4*sizeof(block_q8_0)+16*sizeof(block_q8_1_mmq))+2*sizeof(uint64_t);
static void launch(bool owned,const char *w,const int *y,float *d,float *tmp,int rows,int k,int stride){
 if(owned) mul_mat_q_q8_j16_pairsplit_tm64<2,2,1><<<rows/64,dim3(32,8),smem,stream>>>(w,y,d,tmp,rows,k/32,16,stride,16,1,d,rows);
 else { mul_mat_q_q8_j16_dl_ring_tm64<2,2,1><<<96,dim3(32,8),smem,stream>>>(w,y,d,tmp,rows,k/32,16,stride,16,0,d,rows); mul_mat_q_q8_dl_fixup_pair<<<dim3(32,4),dim3(32,4),0,stream>>>(d,tmp,stride,16); }
 CHECK(cudaGetLastError());
}
int main(){
 CHECK(cudaStreamCreate(&stream));int cases=0,replays=0;
 for(int k:{12288,32768}) {
  const int rows=4096;std::vector<block_q8_0>w(size_t(rows)*k/32);
  const unsigned short scales[]={0,0x2c00,0x3000,0xb000,0x0400};
  for(auto &b:w){auto h=scales[next()%5];std::memcpy(&b.d,&h,2);for(auto &q:b.qs)q=int(next()%256)-128;}
  char *dw,*dl;int *dy;float *dr,*dc,*tmp;
  CHECK(cudaMalloc(&dw,w.size()*sizeof(w[0])));CHECK(cudaMalloc(&dl,w.size()*sizeof(w[0])));
  CHECK(cudaMemcpyAsync(dw,w.data(),w.size()*sizeof(w[0]),cudaMemcpyHostToDevice,stream));
  const int64_t blocks=int64_t(rows)*k/32;
  ggml_cuda_dl_repack_kernel<GGML_CUDA_DL_TM64,4,34><<<(blocks+255)/256,256,0,stream>>>(dw,dl,rows,k/32);CHECK(cudaGetLastError());
  std::vector<block_q8_1_mmq> y(size_t(k/128)*16);CHECK(cudaMalloc(&dy,y.size()*sizeof(y[0])));
  CHECK(cudaMalloc(&dr,size_t(16)*4128*sizeof(float)));CHECK(cudaMalloc(&dc,size_t(16)*4128*sizeof(float)));CHECK(cudaMalloc(&tmp,size_t(48)*16*128*sizeof(float)));
  for(int stride:{4096,4100,4128}) {
   cudaGraph_t g;cudaGraphExec_t e;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(true,dl,dy,dc,tmp,rows,k,stride);CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));CHECK(cudaGraphDestroy(g));
   for(int v=0;v<3;++v){
    for(auto &b:y){for(auto &d:b.d4)d=float(1+next()%100)/2000;for(auto &q:b.qs)q=int(next()%256)-128;}
    CHECK(cudaMemcpyAsync(dy,y.data(),y.size()*sizeof(y[0]),cudaMemcpyHostToDevice,stream));
    CHECK(cudaMemsetAsync(dr,0x7e,size_t(16)*4128*sizeof(float),stream));CHECK(cudaMemsetAsync(dc,0x7e,size_t(16)*4128*sizeof(float),stream));
    launch(false,dl,dy,dr,tmp,rows,k,stride);if(v==0)launch(true,dl,dy,dc,tmp,rows,k,stride);else{CHECK(cudaGraphLaunch(e,stream));++replays;}
    std::vector<float>a(size_t(16)*4128),b(a.size());CHECK(cudaMemcpyAsync(a.data(),dr,a.size()*4,cudaMemcpyDeviceToHost,stream));CHECK(cudaMemcpyAsync(b.data(),dc,b.size()*4,cudaMemcpyDeviceToHost,stream));CHECK(cudaStreamSynchronize(stream));
    if(std::memcmp(a.data(),b.data(),a.size()*4)){for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],4)){fprintf(stderr,"PAIRSPLIT_Q8_FAIL k=%d stride=%d input=%d index=%zu ref=%a got=%a\n",k,stride,v,i,a[i],b[i]);return 1;}}
    for(int col=0;col<16;++col)for(int row=0;row<rows;++row)if(!std::isfinite(b[col*stride+row])){fprintf(stderr,"NONFINITE\n");return 1;}
    ++cases;
   }
   CHECK(cudaGraphExecDestroy(e));
  }
  CHECK(cudaFree(dw));CHECK(cudaFree(dl));CHECK(cudaFree(dy));CHECK(cudaFree(dr));CHECK(cudaFree(dc));CHECK(cudaFree(tmp));
 }
 CHECK(cudaStreamDestroy(stream));printf("PAIRSPLIT_Q8_KCHAIN_PASS bitwise_original_splitK_cases=%d changed_input_graph_replays=%d full_output_guards_exact=true\n",cases,replays);
}
