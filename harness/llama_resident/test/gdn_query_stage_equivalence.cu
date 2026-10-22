// Same executable, independent leader kernel versus CTA-shared query staging.
#include "gated_delta_net_kernel.cuh"
#include "gated_delta_net_dedup.cuh"
#include "gdn_query_stage_reference.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s at %d\n",cudaGetErrorString(e),__LINE__);exit(2);}}while(0)
static unsigned rng=0x712453;
static unsigned next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void fill(std::vector<float>&x,float s){for(auto &v:x)v=float(int(next()%2049)-1024)*s;}
constexpr int H=2,S=128,K=16,STRIDE=H*S*S,GUARD=64;
static int cases=0,replays=0;

template<bool RAW,bool PRE,bool LAZY,bool G1>
static void run(int nt){
 const size_t ns=size_t(STRIDE)*K+2*GUARD,no=size_t(nt)*H*S+2*GUARD;
 std::vector<float> hq(size_t(nt)*H*S),hk(hq.size()),hv(hq.size()),hg(size_t(nt)*H),hb(hg.size()),hs(ns),outa(no),outb(no),sta(ns),stb(ns);
 fill(hq,1.0f/8192);fill(hk,1.0f/8192);fill(hv,1.0f/1024);fill(hg,1.0f/1024);fill(hb,1.0f/1024);fill(hs,1.0f/8192);
 // Exercise stable activated inputs and the softplus threshold in the raw path.
 for(size_t i=0;i<hg.size();++i)hg[i]=RAW?(i%3==0?20.125f:-float(i%11)*0.125f):(PRE?0.95f:-0.02f);
 for(int i=0;i<GUARD;++i)hs[i]=hs[ns-1-i]=12345.0f;
 const float bias[H]={0.125f,-0.25f},alpha[H]={-0.015625f,-0.03125f};
 float *q,*k,*v,*g,*b,*rb,*ra,*a,*c,*oa,*ob;
 uint32_t *ca,*cb;cudaStream_t stream;CK(cudaStreamCreate(&stream));
 auto alloc=[](float**p,size_t n){CK(cudaMalloc(p,n*4));};
 alloc(&q,hq.size());alloc(&k,hk.size());alloc(&v,hv.size());alloc(&g,hg.size());alloc(&b,hb.size());alloc(&rb,H);alloc(&ra,H);alloc(&a,ns);alloc(&c,ns);alloc(&oa,no);alloc(&ob,no);
 CK(cudaMalloc(&ca,H*4));CK(cudaMalloc(&cb,H*4));
 auto cp=[&](float*d,const std::vector<float>&h){CK(cudaMemcpyAsync(d,h.data(),h.size()*4,cudaMemcpyHostToDevice,stream));};
 cp(q,hq);cp(k,hk);cp(v,hv);cp(g,hg);cp(b,hb);cp(a,hs);cp(c,hs);
 CK(cudaMemcpyAsync(rb,bias,sizeof(bias),cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(ra,alpha,sizeof(alpha),cudaMemcpyHostToDevice,stream));
 std::vector<uint32_t> counters(H,GDN_DEDUP_REHYDRATE);
 CK(cudaMemcpyAsync(ca,counters.data(),H*4,cudaMemcpyHostToDevice,stream));CK(cudaMemcpyAsync(cb,counters.data(),H*4,cudaMemcpyHostToDevice,stream));
 const auto magic=init_fastdiv_values(H),seqmagic=init_fastdiv_values(1);
 int row=1;
 auto launch=[&](bool ref){
  float*st=(ref?a:c)+GUARD;float*out=(ref?oa:ob)+GUARD;uint32_t*counter=ref?ca:cb;
  const auto kernel=ref?gated_delta_net_dedup_l8v_reference<RAW,PRE,LAZY,true,G1>:gated_delta_net_dedup_l8v_cuda<RAW,PRE,LAZY,true,G1>;
  kernel<<<dim3(H,1,8),dim3(32,4),0,stream>>>(q,k,v,g,b,rb,ra,st+row*STRIDE,out,st,H,nt,1,S,H*S,nt*H*S,S,H*S,nt*H*S,1,H,nt*H,magic,seqmagic,1.0f/128,STRIDE,K,nullptr,STRIDE,1,counter);
  CK(cudaGetLastError());
 };
 for(int step=0;step<5;++step){
  row=step==0?1:(step==1?0:(step==2?1:(step==3?3:0)));
  fill(hq,1.0f/8192);fill(hk,1.0f/8192);fill(hv,1.0f/1024);cp(q,hq);cp(k,hk);cp(v,hv);
  CK(cudaMemsetAsync(oa,0x7e,no*4,stream));CK(cudaMemsetAsync(ob,0x7e,no*4,stream));
  launch(true);
  if(step==0)launch(false);else{
   cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(false);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));CK(cudaGraphLaunch(exec,stream));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));++replays;
  }
  CK(cudaMemcpyAsync(sta.data(),a,ns*4,cudaMemcpyDeviceToHost,stream));CK(cudaMemcpyAsync(stb.data(),c,ns*4,cudaMemcpyDeviceToHost,stream));CK(cudaMemcpyAsync(outa.data(),oa,no*4,cudaMemcpyDeviceToHost,stream));CK(cudaMemcpyAsync(outb.data(),ob,no*4,cudaMemcpyDeviceToHost,stream));CK(cudaStreamSynchronize(stream));
  if(memcmp(sta.data(),stb.data(),ns*4)||memcmp(outa.data(),outb.data(),no*4)){
   for(size_t i=0;i<ns;++i)if(memcmp(&sta[i],&stb[i],4)){fprintf(stderr,"STATE raw%d pre%d lazy%d g1%d nt%d step%d i%zu %a %a\n",RAW,PRE,LAZY,G1,nt,step,i,sta[i],stb[i]);break;}
   for(size_t i=0;i<no;++i)if(memcmp(&outa[i],&outb[i],4)){fprintf(stderr,"OUT raw%d pre%d lazy%d g1%d nt%d step%d i%zu %a %a\n",RAW,PRE,LAZY,G1,nt,step,i,outa[i],outb[i]);break;}exit(3);
  }
  ++cases;
 }
 for(auto p:{q,k,v,g,b,rb,ra,a,c,oa,ob})CK(cudaFree(p));CK(cudaFree(ca));CK(cudaFree(cb));CK(cudaStreamDestroy(stream));
}
int main(){
 for(int nt:{1,4,8,15,16}){
  run<true,false,true,true>(nt);run<true,false,true,false>(nt);run<false,false,true,true>(nt);run<false,true,true,true>(nt);run<false,true,false,true>(nt);
 }
 printf("GDN_QUERY_STAGE_PASS exact_cases=%d changed_input_graph_replays=%d outputs_full_state_guards_exact=true\n",cases,replays);
}
