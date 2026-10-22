// Promoted759 producer vs resident-Q activation plane; includes production dispatch.
#include "mmq.cuh"
#include "q_activation_plane_reference.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s: %s\n",__LINE__,#x,cudaGetErrorString(e));exit(2);} }while(0)
int ggml_cuda_get_device(){int d;CK(cudaGetDevice(&d));return d;}
void ggml_cuda_error(const char*s,const char*f,const char*p,int l,const char*m){fprintf(stderr,"%s %s %s:%d %s\n",s,f,p,l,m);exit(3);}
extern "C" void ggml_abort(const char*p,int l,const char*,...){fprintf(stderr,"abort %s:%d\n",p,l);exit(3);}
const ggml_cuda_device_info & ggml_cuda_info(){static auto i=[] {ggml_cuda_device_info v={};cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));v.device_count=v.physical_device_count=1;v.devices[0].cc=p.major*100+p.minor*10;v.devices[0].nsm=p.multiProcessorCount;v.devices[0].smpb=p.sharedMemPerBlock;v.devices[0].smpbo=p.sharedMemPerBlockOptin;v.devices[0].warp_size=p.warpSize;return v;}();return i;}
template<class T> static T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
static cudaStream_t stream;
static constexpr int ROWS[3]={8192,1024,1024};
template<bool OLD> static void launch(ggml_cuda_dod_qkv_args a,float*t){
 if constexpr(OLD){
  constexpr size_t sm=ggml_cuda_dl_ring_smem<GGML_CUDA_DL_FMT_PTQ1S,2>(3);
  CK(cudaFuncSetAttribute(mul_mat_q_dod_qkv_reference<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(sm)));
  ggml_cuda_kernel_launch(mul_mat_q_dod_qkv_reference<true>,ggml_cuda_kernel_launch_params(dim3(144),dim3(32,16),sm,stream),a,t);
  ggml_cuda_kernel_launch(mul_mat_q_dod_qkv_fixup_compact,ggml_cuda_kernel_launch_params(dim3(48,4),dim3(32,4),0,stream),a,t);
 }else ggml_cuda_dod_qkv_launch<true>(a,t,stream);
 CK(cudaGetLastError());
}
static int cases=0,replays=0;
static void compare(ggml_cuda_dod_qkv_args a,ggml_cuda_dod_qkv_args b,const std::vector<size_t>& words,int seed,int n){
 float *aa[]={a.dst,a.dst_hi,a.dst_hi2},*bb[]={b.dst,b.dst_hi,b.dst_hi2};
 for(int i=0;i<3;++i){std::vector<unsigned> x(words[i]),y(words[i]);CK(cudaMemcpy(x.data(),aa[i],words[i]*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),bb[i],words[i]*4,cudaMemcpyDeviceToHost));
  for(size_t j=0;j<words[i];++j){if(x[j]!=y[j]){fprintf(stderr,"FAIL seed=%d n=%d seg=%d word=%zu %08x %08x\n",seed,n,i,j,x[j],y[j]);exit(1);}size_t col=j/a.stride[i],row=j%a.stride[i];if((col>=size_t(n)||row>=size_t(ROWS[i]))&&x[j]!=0xa5a5a5a5u){fprintf(stderr,"GUARD FAIL\n");exit(1);}}
 }
 ++cases;
}
int main(){
 CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::mt19937 rng(20261022);
 const size_t xb=10240*32*sizeof(block_ptq1_0);std::vector<unsigned char> hx(xb);auto*x=alloc<char>(xb);auto*y=alloc<block_q8_1_mmq>(32*16+16);
 auto*ta=alloc<float>(144*2048),*tb=alloc<float>(144*2048);
 for(int pad:{0,8,64}){
  ggml_cuda_dod_qkv_args a{},b{};a.x=b.x=x;a.y=b.y=(int*)y;
  std::vector<size_t> words(3);float*aa[3],*bb[3];for(int i=0;i<3;++i){a.stride[i]=b.stride[i]=ROWS[i]+pad+8*i;words[i]=size_t(a.stride[i])*18+64;aa[i]=alloc<float>(words[i]);bb[i]=alloc<float>(words[i]);}
  a.dst=aa[0];a.dst_hi=aa[1];a.dst_hi2=aa[2];b.dst=bb[0];b.dst_hi=bb[1];b.dst_hi2=bb[2];
  for(int seed=0;seed<3;++seed){
   // All byte patterns in packed PTQ payload; finite half scales including signed zero.
   for(size_t i=0;i<xb;i+=sizeof(block_ptq1_0)){for(size_t j=0;j<sizeof(block_ptq1_0);++j)hx[i+j]=rng()&255;half d=__float2half(seed==0?-0.0f:float(int(rng()%201)-100)*.0003f);memcpy(hx.data()+i,&d,2);}
   CK(cudaMemcpyAsync(x,hx.data(),xb,cudaMemcpyHostToDevice,stream));
   for(int n:{1,8,15,16}){
    a.ncols=b.ncols=n;std::vector<block_q8_1_mmq> hy(32*n+16);
    for(auto &v:hy){for(int i=0;i<4;++i)v.d4[i]=seed==0?0.0f:float(int(rng()%201)-100)*.003f;for(auto&q:v.qs)q=int8_t(int(rng()%255)-127);}
    CK(cudaMemcpyAsync(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice,stream));
    for(int i=0;i<3;++i){CK(cudaMemsetAsync(aa[i],0xa5,words[i]*4,stream));CK(cudaMemsetAsync(bb[i],0xa5,words[i]*4,stream));}
    launch<true>(a,ta);launch<false>(b,tb);CK(cudaStreamSynchronize(stream));compare(a,b,words,seed,n);
   }
  }
  a.ncols=b.ncols=16;cudaGraph_t graph;cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch<true>(a,ta);launch<false>(b,tb);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));CK(cudaGraphDestroy(graph));
  for(int r=0;r<6;++r){std::vector<block_q8_1_mmq> hy(32*16+16);for(auto&v:hy){for(auto&d:v.d4)d=float(int(rng()%201)-100)*.001f;for(auto&q:v.qs)q=int8_t(int(rng()%255)-127);}CK(cudaMemcpyAsync(y,hy.data(),hy.size()*sizeof(hy[0]),cudaMemcpyHostToDevice,stream));for(int i=0;i<3;++i){CK(cudaMemsetAsync(aa[i],0xa5,words[i]*4,stream));CK(cudaMemsetAsync(bb[i],0xa5,words[i]*4,stream));}CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));compare(a,b,words,10+r,16);++replays;}
  CK(cudaGraphExecDestroy(exec));for(int i=0;i<3;++i){CK(cudaFree(aa[i]));CK(cudaFree(bb[i]));}
 }
 CK(cudaFree(x));CK(cudaFree(y));CK(cudaFree(ta));CK(cudaFree(tb));CK(cudaStreamDestroy(stream));printf("Q_ACTIVATION_PLANE_PASS exact_cases=%d changed_input_graph_replays=%d full_destinations_and_guards=true\n",cases,replays);
}
