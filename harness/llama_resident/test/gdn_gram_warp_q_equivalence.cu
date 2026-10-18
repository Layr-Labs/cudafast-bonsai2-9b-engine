// Same-executable bit-exact preparation workspace comparison against promoted239e6e8.
#include "gated_delta_net_kernel.cuh"
#include "gdn_gram_warp_q_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int ggml_cuda_get_device() {
    int device = 0;
    const cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(2); }
    return device;
}
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "%s in %s at %s:%d: %s\n", stmt, func, file, line, msg);
    std::exit(2);
}
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    std::abort();
}
#define CHECK(x) do { auto err_=(x);if(err_!=cudaSuccess){fprintf(stderr,"%s: %s line %d\n",#x,cudaGetErrorString(err_),__LINE__);std::exit(2);}}while(0)
template<class T> T *alloc(size_t n) { T*p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p; }
template<class T> T *upload(const std::vector<T>&v) {T*p=alloc<T>(v.size());CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<bool RAW,bool PRE,bool OLD> void launch(const float*q,const float*k,const float*g,const float*b,const float*dt,const float*ra,gdn_compact32*w,int H,int HK,int T,int64_t sq1,int64_t sq2,int64_t sb1,int64_t sb2,cudaStream_t stream) {
    const dim3 grid(H,(T+31)/32);
    if constexpr(OLD) reference239_gdn_compact32_prepare<RAW,PRE><<<grid,256,0,stream>>>(q,k,g,b,dt,ra,w,H,T,sq1,sq2,sb1,sb2,init_fastdiv_values(HK));
    else gdn_compact32_prepare<RAW,PRE><<<grid,256,0,stream>>>(q,k,g,b,dt,ra,w,H,T,sq1,sq2,sb1,sb2,init_fastdiv_values(HK));
    CHECK(cudaGetLastError());
}
template<class F> float measure(F f,cudaStream_t stream){
    for(int i=0;i<4;++i)f();cudaGraph_t g;cudaGraphExec_t e;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int i=0;i<8;++i)f();CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a,stream));for(int i=0;i<12;++i)CHECK(cudaGraphLaunch(e,stream));CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));CHECK(cudaGraphExecDestroy(e));CHECK(cudaGraphDestroy(g));return ms*1000/(8*12);
}
template<bool RAW,bool PRE> int run(int T,int H,int HK,int layout,cudaStream_t stream) {
    std::mt19937 rng(20261018+T+H+layout);std::normal_distribution<float> nd(0,.05f);
    std::vector<float> q(size_t(T)*HK*128),k(q.size()),g(size_t(T)*H),b(g.size()),dt(H),ra(H);
    for(auto&v:q)v=nd(rng);for(auto&v:k)v=nd(rng);
    for(size_t i=0;i<g.size();++i){g[i]=PRE?(i%37==0?0.0f:.93f+nd(rng)):(RAW?nd(rng):-fabsf(nd(rng)));b[i]=RAW?nd(rng):.3f+nd(rng);}
    for(int i=0;i<H;++i){dt[i]=nd(rng);ra[i]=-.2f-fabsf(nd(rng));}
    float*dq=upload(q),*dk=upload(k),*dg=upload(g),*db=upload(b),*ddt=upload(dt),*dra=upload(ra);
    const int64_t sq1=layout?T*128:128,sq2=layout?128:HK*128,sb1=layout?T:1,sb2=layout?1:H;
    const size_t bytes=size_t(H)*((T+31)/32)*sizeof(gdn_compact32);auto*a=alloc<gdn_compact32>(bytes/sizeof(gdn_compact32));auto*c=alloc<gdn_compact32>(bytes/sizeof(gdn_compact32));
    CHECK(cudaMemsetAsync(a,0xA5,bytes,stream));CHECK(cudaMemsetAsync(c,0xA5,bytes,stream));
    auto old=[&]{launch<RAW,PRE,true>(dq,dk,dg,db,ddt,dra,a,H,HK,T,sq1,sq2,sb1,sb2,stream);};
    auto now=[&]{launch<RAW,PRE,false>(dq,dk,dg,db,ddt,dra,c,H,HK,T,sq1,sq2,sb1,sb2,stream);};
    old();now();CHECK(cudaStreamSynchronize(stream));std::vector<unsigned>ha(bytes/4),hc(bytes/4);CHECK(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hc.data(),c,bytes,cudaMemcpyDeviceToHost));
    if(ha!=hc){for(size_t i=0;i<ha.size();++i)if(ha[i]!=hc[i]){fprintf(stderr,"FAIL raw=%d pre=%d T=%d H=%d HK=%d layout=%d word=%zu old=%08x new=%08x\n",RAW,PRE,T,H,HK,layout,i,ha[i],hc[i]);break;}std::exit(1);}
    if(T>=128 && H==32 && layout==0)for(int round=0;round<5;++round){float ot,nt;if(round%2){nt=measure(now,stream);ot=measure(old,stream);}else{ot=measure(old,stream);nt=measure(now,stream);}printf("GRAPH raw=%d pre=%d T=%d H=%d round=%d old_us=%.4f new_us=%.4f gain=%.6f\n",RAW,PRE,T,H,round,ot,nt,ot/nt);}
    for(auto*p:{dq,dk,dg,db,ddt,dra})CHECK(cudaFree(p));CHECK(cudaFree(a));CHECK(cudaFree(c));return 1;
}
int main(){cudaStream_t s;CHECK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));int n=0;
    for(int T:{1,16,31,32,33,47,63,64,65,127,128,255,496,512,1024})for(int H:{1,3,32})for(int layout:{0,1}){
        n+=run<false,false>(T,H,std::max(1,H/2),layout,s);n+=run<true,false>(T,H,std::max(1,H/2),layout,s);n+=run<false,true>(T,H,std::max(1,H/2),layout,s);
    }CHECK(cudaStreamDestroy(s));printf("GDN_GRAM_WARP_Q_PASS cases=%d exact_workspace_bits=true\n",n);
}
