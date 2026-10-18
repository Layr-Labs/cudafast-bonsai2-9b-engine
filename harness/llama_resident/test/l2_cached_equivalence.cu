#include "common.cuh"
#include "l2norm-cached.cuh"
#include "l2_cached_reference.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do {auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s: %s at%d\n",#x,cudaGetErrorString(err),__LINE__);exit(2);}}while(0)
static cudaStream_t st;
static cudaGraphExec_t graph_for(bool old,float*x,float*y,int rows,int channels,int samples,int64_t sr,int64_t sc,int64_t ss,float eps) {
 cudaGraph_t g;cudaGraphExec_t exec;
 CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
 for(int i=0;i<64;++i) {
  if(old) reference168_l2_norm_f32<32><<<dim3(rows,channels,samples),32,0,st>>>(x,y,128,sr,sc,ss,eps);
  else l2_norm_f32_cached_128<<<dim3(rows,channels,samples),32,0,st>>>(x,y,sr,sc,ss,eps);
 }
 CHECK(cudaStreamEndCapture(st,&g));CHECK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));CHECK(cudaGraphDestroy(g));return exec;
}
static float timing(cudaGraphExec_t g) {
 cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a,st));
 for(int i=0;i<4;++i)CHECK(cudaGraphLaunch(g,st));
 CHECK(cudaEventRecord(b,st));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));return ms*1000/256;
}
int main() {
 CHECK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));std::mt19937 rng(20261018);int cases=0;
 const float special[]={0,-0.0f,FLT_MIN,-FLT_MIN,1e-40f,-1e-40f,1e-20f,-1e-20f,FLT_MAX,-FLT_MAX,1e19f,-1e19f,INFINITY,-INFINITY,NAN,-NAN};
 for(int rows:{1,3,16,512})for(int channels:{1,8})for(int samples:{1,2})for(bool padded:{false,true})for(float eps:{0.0f,1e-6f})for(int mode:{0,1,2,3}){
  int64_t sr=128+(padded?17:0),sc=rows*sr+(padded?31:0),ss=channels*sc+(padded?51:0);
  size_t nx=samples*ss,ny=size_t(rows)*channels*samples*128;
  std::vector<float> h(nx);for(auto&f:h)f=mode==0?float(int(rng()%4096)-2048)/128.f:mode==1?special[rng()%16]:mode==2?0.0f:std::ldexp(float(int(rng()%1024)-512),int(rng()%81)-40);
  float*x,*a,*b;CHECK(cudaMalloc(&x,nx*4));CHECK(cudaMalloc(&a,(ny+32)*4));CHECK(cudaMalloc(&b,(ny+32)*4));CHECK(cudaMemcpy(x,h.data(),nx*4,cudaMemcpyHostToDevice));
  CHECK(cudaMemsetAsync(a,0xA5,(ny+32)*4,st));CHECK(cudaMemsetAsync(b,0xA5,(ny+32)*4,st));
  reference168_l2_norm_f32<32><<<dim3(rows,channels,samples),32,0,st>>>(x,a,128,sr,sc,ss,eps);
  l2_norm_f32_cached_128<<<dim3(rows,channels,samples),32,0,st>>>(x,b,sr,sc,ss,eps);CHECK(cudaStreamSynchronize(st));
  std::vector<uint32_t> ha(ny+32),hb(ny+32);CHECK(cudaMemcpy(ha.data(),a,(ny+32)*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,(ny+32)*4,cudaMemcpyDeviceToHost));
  if(ha!=hb){fprintf(stderr,"FAIL rows%d channels%d samples%d padded%d eps%g mode%d\n",rows,channels,samples,padded,eps,mode);for(size_t i=0,n=0;i<ha.size()&&n<8;++i)if(ha[i]!=hb[i]){fprintf(stderr,"%zu %08x %08x\n",i,ha[i],hb[i]);++n;}return 1;}++cases;
  if(mode==0&&!padded&&eps>0&&samples==1){
   auto ga=graph_for(true,x,a,rows,channels,samples,sr,sc,ss,eps),gb=graph_for(false,x,b,rows,channels,samples,sr,sc,ss,eps);
   for(int r=0;r<8;++r){float u,v;if(r%2==0){u=timing(ga);v=timing(gb);}else{v=timing(gb);u=timing(ga);}if(r>=2)printf("GRAPH rows=%d channels=%d round=%d old_us=%.4f new_us=%.4f gain=%.6f\n",rows,channels,r,u,v,u/v);}
   CHECK(cudaMemcpy(ha.data(),a,(ny+32)*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,(ny+32)*4,cudaMemcpyDeviceToHost));if(ha!=hb){fprintf(stderr,"GRAPH_FAIL\n");return 1;}
   CHECK(cudaGraphExecDestroy(ga));CHECK(cudaGraphExecDestroy(gb));
  }
  CHECK(cudaFree(x));CHECK(cudaFree(a));CHECK(cudaFree(b));
 }
 CHECK(cudaStreamDestroy(st));printf("L2_CACHED_PASS cases=%d exact_output_and_guards=true\n",cases);
}
