// Independent original grouped-QKV partition walk versus compact fixed map.
#include "mmq.cuh"
#include "qkv_compact_fixup_reference.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s: %s\n",__LINE__,#x,cudaGetErrorString(e));exit(2);}}while(0)
static cudaStream_t stream;
template<bool OLD> static void launch(ggml_cuda_dod_qkv_args a,const float*p){
 if constexpr(OLD) reference203_qkv_fixup<true><<<dim3(144,4),dim3(32,4),0,stream>>>(a,p);
 else mul_mat_q_dod_qkv_fixup_compact<<<dim3(48,4),dim3(32,4),0,stream>>>(a,p);
 CK(cudaGetLastError());
}
template<bool OLD> static cudaGraphExec_t graph(ggml_cuda_dod_qkv_args a,const float*p){
 cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 for(int i=0;i<16;++i)launch<OLD>(a,p);CK(cudaStreamEndCapture(stream,&g));CK(cudaGraphInstantiate(&ex,g,0));CK(cudaGraphDestroy(g));return ex;
}
static float time_graph(cudaGraphExec_t g){
 cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));
 for(int i=0;i<8;++i)CK(cudaGraphLaunch(g,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));
 float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/128;
}
int main(){
 CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::mt19937 rng(20261019);int cases=0;
 const size_t pwords=144*2048;std::vector<float> hp(pwords);float *p;CK(cudaMalloc(&p,pwords*4));
 for(int padding:{0,7,64}){
  int stride[3]={8192+padding,1024+padding,1024+padding};size_t words[3];
  float *a[3],*b[3];std::vector<float> initial[3];std::vector<unsigned> ha[3],hb[3];
  for(int i=0;i<3;++i){words[i]=size_t(stride[i])*16+64;initial[i].resize(words[i]);ha[i].resize(words[i]);hb[i].resize(words[i]);CK(cudaMalloc(&a[i],words[i]*4));CK(cudaMalloc(&b[i],words[i]*4));}
  ggml_cuda_dod_qkv_args aa{},bb{};aa.dst=a[0];aa.dst_hi=a[1];aa.dst_hi2=a[2];bb.dst=b[0];bb.dst_hi=b[1];bb.dst_hi2=b[2];
  for(int i=0;i<3;++i)aa.stride[i]=bb.stride[i]=stride[i];
  for(int seed=0;seed<8;++seed){
   for(auto &v:hp)v=seed==0?0.0f:float(int(rng()%65537)-32768)/float(1u<<(rng()%24));
   CK(cudaMemcpyAsync(p,hp.data(),pwords*4,cudaMemcpyHostToDevice,stream));
   for(int i=0;i<3;++i)for(auto&v:initial[i])v=float(int(rng()%65537)-32768)/1024;
   for(int cols=1;cols<=16;++cols){
    aa.ncols=bb.ncols=cols;
    for(int i=0;i<3;++i){CK(cudaMemcpyAsync(a[i],initial[i].data(),words[i]*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(b[i],initial[i].data(),words[i]*4,cudaMemcpyHostToDevice,stream));}
    launch<true>(aa,p);launch<false>(bb,p);CK(cudaStreamSynchronize(stream));
    for(int i=0;i<3;++i){CK(cudaMemcpy(ha[i].data(),a[i],words[i]*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb[i].data(),b[i],words[i]*4,cudaMemcpyDeviceToHost));if(ha[i]!=hb[i]){fprintf(stderr,"FAIL pad=%d seed=%d cols=%d dst=%d\n",padding,seed,cols,i);exit(1);}}
    ++cases;
   }
  }
  aa.ncols=bb.ncols=16;auto ga=graph<true>(aa,p),gb=graph<false>(bb,p);
  for(int round=0;round<12;++round){float old_us,new_us;if(round%2==0){old_us=time_graph(ga);new_us=time_graph(gb);}else{new_us=time_graph(gb);old_us=time_graph(ga);}if(round>=2)printf("GRAPH pad=%d round=%d old_us=%.4f new_us=%.4f gain=%.6f\n",padding,round,old_us,new_us,old_us/new_us);}
  CK(cudaGraphExecDestroy(ga));CK(cudaGraphExecDestroy(gb));for(int i=0;i<3;++i){CK(cudaFree(a[i]));CK(cudaFree(b[i]));}
 }
 CK(cudaFree(p));CK(cudaStreamDestroy(stream));printf("QKV_COMPACT_FIXUP_PASS cases=%d full_destination_and_guard_bits=true\n",cases);
}
