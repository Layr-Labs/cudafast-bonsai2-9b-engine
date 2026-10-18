// Round coefficient transport once instead of eight times; compare full carry pipeline.
#include "gated_delta_net_kernel.cuh"
#include "gdn-half-prefill.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#define CK(x) do {auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"%s: %s line%d\n",#x,cudaGetErrorString(err),__LINE__);exit(2);}}while(0)
using gdn_half_c16::Frame;using gdn_half_c16::Coeff;using gdn_half_c16::HalfCoeff;
static cudaStream_t stream;
template<class T> T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
static void upload(float*p,const std::vector<float>&v){CK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
template<class F>static cudaGraphExec_t graph(F f){cudaGraph_t g;cudaGraphExec_t ex;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<4;++i)f();CK(cudaStreamEndCapture(stream,&g));CK(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));CK(cudaGraphDestroy(g));return ex;}
static float timed(cudaGraphExec_t g){cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a,stream));for(int i=0;i<4;++i)CK(cudaGraphLaunch(g,stream));CK(cudaEventRecord(b,stream));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));return ms*1000/16;}
static void one(int nt,bool packed,bool raw,unsigned seed,bool bench,bool sanitize){
 constexpr int H=32,HK=16,D=128,C=16;const int chunks=(nt+15)/16;const size_t state_n=H*D*D,out_n=size_t(nt+16)*H*D;
 std::mt19937 rng(seed);auto val=[&](){return (int(rng()%2001)-1000)*0.0001f;};
 std::vector<float>q(size_t(chunks*C)*HK*D),k(q.size()),v(size_t(chunks*C)*H*D),g(size_t(chunks*C)*H),beta(g.size()),bias(H),a(H),initial(state_n);
 for(auto*p:{&q,&k,&v,&initial})for(auto&x:*p)x=val();
 for(auto&x:g)x=raw?val():0.98f+val()*0.1f;for(auto&x:beta)x=raw?val():0.4f+val();for(auto&x:a)x=-0.1f;
 float *dq=alloc<float>(q.size()),*dk=alloc<float>(k.size()),*dv=alloc<float>(v.size()),*dg=alloc<float>(g.size()),*db=alloc<float>(beta.size()),*dbias=alloc<float>(H),*da=alloc<float>(H),*di=alloc<float>(state_n);
 upload(dq,q);upload(dk,k);upload(dv,v);upload(dg,g);upload(db,beta);upload(dbias,bias);upload(da,a);upload(di,initial);
 Frame f{dq,dk,dv,dg,db,dbias,da,nt,packed,packed?int64_t(chunks*C*D):D,packed?D:int64_t(HK*D),D,H*D};
 Coeff*coef=alloc<Coeff>(size_t(chunks)*H);__half*pk=alloc<__half>(size_t(chunks)*HK*2*C*D);
 HalfCoeff*paired=alloc<HalfCoeff>(size_t(chunks)*H);__half*pk2=alloc<__half>(size_t(chunks)*HK*2*C*D);
 if(raw)gdn_half_prefill::prepare_packed<true><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,coef,pk);
 else gdn_half_prefill::prepare_packed<false><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,coef,pk);
 if(raw)gdn_half_prefill::prepare_packed<true,false,HalfCoeff><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,paired,pk2);
 else gdn_half_prefill::prepare_packed<false,false,HalfCoeff><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,paired,pk2);
 CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));
 auto eqbuf=[&](const void*a,const void*b,size_t n){std::vector<unsigned char>x(n),y(n);CK(cudaMemcpy(x.data(),a,n,cudaMemcpyDeviceToHost));CK(cudaMemcpy(y.data(),b,n,cudaMemcpyDeviceToHost));if(x!=y){for(size_t i=0;i<n;++i)if(x[i]!=y[i]){fprintf(stderr,"PREP_FAIL nt=%d packed=%d raw=%d seed=%u byte=%zu old=%u new=%u\n",nt,packed,raw,seed,i,x[i],y[i]);break;}exit(1);}};
 std::vector<Coeff>ca(size_t(chunks)*H);std::vector<HalfCoeff>cb(ca.size());CK(cudaMemcpy(ca.data(),coef,ca.size()*sizeof(Coeff),cudaMemcpyDeviceToHost));CK(cudaMemcpy(cb.data(),paired,cb.size()*sizeof(HalfCoeff),cudaMemcpyDeviceToHost));
 for(size_t c=0;c<ca.size();++c){
  for(int e=0;e<C*C;++e)if(__half_as_ushort(__float2half_rn(ca[c].a[e]))!=__half_as_ushort(cb[c].a[e])||__half_as_ushort(__float2half_rn(ca[c].m[e]))!=__half_as_ushort(cb[c].m[e])){fprintf(stderr,"COEFF_FAIL c=%zu e=%d\n",c,e);exit(1);}
  if(memcmp(ca[c].p,cb[c].p,C*sizeof(float))||memcmp(ca[c].last,cb[c].last,C*sizeof(float))){fprintf(stderr,"DECAY_FAIL c=%zu\n",c);exit(1);}
 }eqbuf(pk,pk2,size_t(chunks)*HK*2*C*D*sizeof(__half));
 float *o[2],*s[2],*bd[2];for(int i=0;i<2;++i){o[i]=alloc<float>(out_n);s[i]=alloc<float>(state_n+32);bd[i]=alloc<float>(state_n+32);CK(cudaMemsetAsync(o[i],0xa5,out_n*4,stream));CK(cudaMemsetAsync(s[i],0xa5,(state_n+32)*4,stream));CK(cudaMemsetAsync(bd[i],0xa5,(state_n+32)*4,stream));}
 auto old=[&](){gdn_half_prefill::carry_packed<true,false><<<dim3(H,8),dim3(32,4),0,stream>>>(f,coef,pk,di,o[0],s[0],bd[0],1.0f/sqrtf(128.0f));};
 auto cand=[&](){gdn_half_prefill::carry_packed<true,false,HalfCoeff><<<dim3(H,8),dim3(32,4),0,stream>>>(f,paired,pk2,di,o[1],s[1],bd[1],1.0f/sqrtf(128.0f));};
 if(!sanitize){old();cand();}CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));
 auto eq=[&](){for(int which=0;which<3;++which){float**p=which==0?o:which==1?s:bd;size_t n=which==0?out_n:state_n+32;std::vector<unsigned>aa(n),bb(n);CK(cudaMemcpy(aa.data(),p[0],n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(bb.data(),p[1],n*4,cudaMemcpyDeviceToHost));for(size_t i=0;i<n;++i)if(aa[i]!=bb[i]){fprintf(stderr,"FAIL nt=%d packed=%d raw=%d seed=%u buffer=%d word=%zu old=%08x new=%08x\n",nt,packed,raw,seed,which,i,aa[i],bb[i]);exit(1);}for(size_t i=which==0?size_t(nt)*H*D:state_n;i<n;++i)if(aa[i]!=0xa5a5a5a5u){fprintf(stderr,"GUARD_FAIL\n");exit(1);}}};if(!sanitize)eq();
 if(bench){auto prep_old=[&](){gdn_half_prefill::prepare_packed<false><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,coef,pk);};auto prep_new=[&](){gdn_half_prefill::prepare_packed<false,false,HalfCoeff><<<dim3(H,chunks),dim3(32,4),0,stream>>>(f,paired,pk2);};auto pipe_old=[&](){prep_old();old();};auto pipe_new=[&](){prep_new();cand();};auto ga=graph(pipe_old),gb=graph(pipe_new);for(int r=0;r<10;++r){float aa,bb;if(r%2){bb=timed(gb);aa=timed(ga);}else{aa=timed(ga);bb=timed(gb);}if(r>=2)printf("GRAPH nt=%d packed=%d raw=%d round=%d old_us=%.4f new_us=%.4f gain=%.6f\n",nt,packed,raw,r,aa,bb,aa/bb);}eq();CK(cudaGraphExecDestroy(ga));CK(cudaGraphExecDestroy(gb));}
 for(float*p:{dq,dk,dv,dg,db,dbias,da,di,o[0],o[1],s[0],s[1],bd[0],bd[1]})CK(cudaFree(p));CK(cudaFree(coef));CK(cudaFree(pk));CK(cudaFree(paired));CK(cudaFree(pk2));
}
int main(int argc,char**){CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));int cases=0;for(int nt:{16,32,48,64,128,496,512})for(bool packed:{false,true})for(bool raw:{false,true})for(unsigned seed:{17u,37u,97u}){if(argc>1&&(seed!=17u||nt!=48))continue;one(nt,packed,raw,seed,argc==1&&seed==17u&&nt>=128&&!raw,argc>1);++cases;}CK(cudaStreamDestroy(stream));if(argc>1)printf("GDN_HALF_COEFF_TRANSPORT_SANITIZER_CASES=%d\n",cases);else printf("GDN_HALF_COEFF_TRANSPORT_PASS cases=%d exact_output_state_boundary=true\n",cases);}
