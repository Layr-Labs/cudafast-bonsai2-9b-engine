#include "gated_delta_net_kernel.cuh"
#include "gdn_launch_overlap_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA_FAIL %d %s\n",__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
template<class Kernel,class...Args>static void pdl_launch(dim3 grid,dim3 block,Kernel kernel,Args...args){cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cudaLaunchConfig_t config{};config.gridDim=grid;config.blockDim=block;config.attrs=&attr;config.numAttrs=1;CHECK(cudaLaunchKernelEx(&config,kernel,args...));}
constexpr int S=128,H=32,HK=16,K=20,MAX_T=16;static int T=16;constexpr int64_t D=int64_t(S)*S*H;
struct Data{float *q,*k,*v,*g,*b,*bias,*ra,*state,*dst;};
static float* alloc(const std::vector<float>&h){float*p;CHECK(cudaMalloc(&p,h.size()*4));CHECK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
__global__ void consume(const float *src,float *dst,int count){ggml_cuda_pdl_lc();int i=int(blockIdx.x)*blockDim.x+threadIdx.x;ggml_cuda_pdl_sync();if(i<count)dst[i]=src[i];}
static std::vector<float> values(size_t n,std::mt19937&r,float scale){std::vector<float> h(n);for(auto &v:h)v=float(int(r()%2001)-1000)*scale;return h;}
static void same(float*a,float*b,size_t n,const char*label){std::vector<unsigned> x(n),y(n);CHECK(cudaMemcpy(x.data(),a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(y.data(),b,n*4,cudaMemcpyDeviceToHost));if(memcmp(x.data(),y.data(),n*4)){for(size_t i=0;i<n;++i)if(x[i]!=y[i]){fprintf(stderr,"GDN_LAUNCH_OVERLAP_FAIL %s index=%zu old=%08x new=%08x\n",label,i,x[i],y[i]);break;}std::exit(1);}}
template<bool RAW,bool PRE,bool LAZY,bool OLD> static void launch(Data d,int lazy){
 const dim3 grid(H,1,8),block(32,4);auto kernel=OLD?gated_delta_net_l8v_reference<RAW,PRE,LAZY>:gated_delta_net_l8v_cuda<RAW,PRE,LAZY>;
 pdl_launch(grid,block,kernel, d.q,d.k,d.v,d.g,d.b,d.bias,d.ra,d.state,d.dst,d.state,H,T,1,S,S*HK,S*HK*T,S,S*H,S*H*T,1,H,H*T,init_fastdiv_values(HK),init_fastdiv_values(1),1.0f/sqrtf(float(S)),D,K,nullptr,D,lazy);CHECK(cudaGetLastError());
}
template<class F>static float measure(F f){cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));for(int i=0;i<3;++i)f();std::vector<float> t;for(int i=0;i<3;++i){CHECK(cudaEventRecord(a));for(int j=0;j<20;++j)f();CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));t.push_back(ms*50);}CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));std::sort(t.begin(),t.end());return t[1];}
template<bool RAW,bool PRE,bool LAZY>static int run(){
 std::mt19937 r(20261018);Data a,b;a.q=alloc(values(S*HK*T,r,.0005f));a.k=alloc(values(S*HK*T,r,.0005f));a.v=alloc(values(S*H*T,r,.0005f));
 auto hg=values(H*T,r,.0001f);for(auto&v:hg)v=PRE?.97f:-fabsf(v);a.g=alloc(hg);a.b=alloc(values(H*T,r,.0005f));a.bias=alloc(std::vector<float>(H,.1f));a.ra=alloc(std::vector<float>(H,-1.0f));b=a;
 auto initial=values(D*K,r,.0001f);int cases=0;
 for(int evict:{0,8}){a.state=alloc(initial);b.state=alloc(initial);a.dst=alloc(std::vector<float>(S*H*T,123.0f));b.dst=alloc(std::vector<float>(S*H*T,123.0f));
  for(int round=0;round<3;++round){launch<RAW,PRE,LAZY,true>(a,(LAZY?1:0)|evict);launch<RAW,PRE,LAZY,false>(b,(LAZY?1:0)|evict);float *copy;CHECK(cudaMalloc(&copy,S*H*T*4));pdl_launch(dim3((S*H*T+255)/256),dim3(256),consume,b.dst,copy,S*H*T);CHECK(cudaDeviceSynchronize());same(a.dst,copy,S*H*T,"PDL-consumer-output");CHECK(cudaFree(copy));same(a.dst,b.dst,S*H*T,"output");same(a.state,b.state,D*K,"complete-state");++cases;}
  float x=measure([&]{launch<RAW,PRE,LAZY,true>(a,(LAZY?1:0)|evict);}),y=measure([&]{launch<RAW,PRE,LAZY,false>(b,(LAZY?1:0)|evict);});printf("gdn tokens=%d raw=%d pre=%d lazy=%d evict=%d old_us=%.4f new_us=%.4f gain=%.5f\n",T,RAW,PRE,LAZY,evict,x,y,x/y);fflush(stdout);
  CHECK(cudaFree(a.state));CHECK(cudaFree(b.state));CHECK(cudaFree(a.dst));CHECK(cudaFree(b.dst));
 }
 for(float*p:{a.q,a.k,a.v,a.g,a.b,a.bias,a.ra})CHECK(cudaFree(p));return cases;
}
int main(){int n=0;for(T=8;T<=MAX_T;++T)n+=run<true,false,true>()+run<true,false,false>()+run<false,false,true>()+run<false,false,false>()+run<false,true,true>()+run<false,true,false>();printf("GDN_LAUNCH_OVERLAP_PASS cases=%d exact_output_and_state_bits=true\n",n);}
