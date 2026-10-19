// Test-only real launch_fattn path: unchanged 58f vs quarter-query/one-stage MMA.
// Predeclared numerical bound: max error / max(1,max|reference|) and relative
// L2 <= 2e-3; old self control and changed-input graph repeats stay bit exact.
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "fattn-mma-f16.cuh"
#include "fattn_lowstage_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void ck(cudaError_t e) {
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}
}
static void need(bool b,const char *s){if(!b){fprintf(stderr,"FAIL %s\n",s);exit(1);}}
template<class T> struct Buffer {
    T *p; explicit Buffer(size_t n){ck(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
};
struct Graph {
    cudaGraph_t g{}; cudaGraphExec_t e{};
    ~Graph(){if(e)cudaGraphExecDestroy(e);if(g)cudaGraphDestroy(g);}
};
template<int D> static void launch(bool candidate,ggml_backend_cuda_context &c,ggml_tensor *o) {
    if(candidate)ggml_cuda_fattn_lowstage::ggml_cuda_flash_attn_ext_mma_f16_case<D,D,8,2>(c,o);
    else ggml_cuda_flash_attn_ext_mma_f16_case<D,D,16,4>(c,o);
}
template<int D> static void capture(Graph &g,bool mode,ggml_backend_cuda_context &c,ggml_tensor *o,int repeats) {
    ck(cudaStreamBeginCapture(c.stream(),cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<repeats;++i)launch<D>(mode,c,o);
    ck(cudaStreamEndCapture(c.stream(),&g.g));
    ck(cudaGraphInstantiate(&g.e,g.g,nullptr,nullptr,0));
}
struct Resource {int regs,local,cta; unsigned shared,grid;};
static Resource resource(Graph &g) {
    size_t n=0;ck(cudaGraphGetNodes(g.g,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);
    ck(cudaGraphGetNodes(g.g,nodes.data(),&n));Resource r{};int found=0;
    for(auto node:nodes){cudaGraphNodeType t;ck(cudaGraphNodeGetType(node,&t));
        if(t!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams p{};ck(cudaGraphKernelNodeGetParams(node,&p));
        if(!p.sharedMemBytes)continue;
        cudaFuncAttributes a{};ck(cudaFuncGetAttributes(&a,p.func));int cta;
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cta,p.func,p.blockDim.x*p.blockDim.y*p.blockDim.z,p.sharedMemBytes));
        need(p.blockDim.x*p.blockDim.y*p.blockDim.z==128,"CTA threads");
        r={a.numRegs,int(a.localSizeBytes),cta,p.sharedMemBytes,p.gridDim.x};++found;
    }
    need(found==1,"one FA kernel plus unchanged global fixup");return r;
}
static float timed(Graph &g,cudaStream_t s,int repeats,void *cold) {
    // Cache eviction is before the event and before each single invocation.
    if(cold)ck(cudaMemsetAsync(cold,0x3c,72*1024*1024,s));
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    ck(cudaEventRecord(a,s));ck(cudaGraphLaunch(g.e,s));ck(cudaEventRecord(b,s));
    ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return ms*1000/repeats;
}
template<int D> static void shape(ggml_backend_cuda_context &c,int nq,int nk,bool causal) {
    constexpr int hkv=D==256?4:8,hq=4*hkv,guard=64;
    constexpr uint32_t sentinel=0x47123456;
    auto *meta=ggml_init({1024*1024,nullptr,true});need(meta,"tensor metadata");
    auto *q=ggml_new_tensor_4d(meta,GGML_TYPE_F32,D,nq,hq,1);
    auto *k=ggml_new_tensor_4d(meta,GGML_TYPE_F16,D,nk,hkv,1);
    auto *v=ggml_new_tensor_4d(meta,GGML_TYPE_F16,D,nk,hkv,1);
    auto *m=ggml_new_tensor_4d(meta,GGML_TYPE_F16,nk,GGML_PAD(nq,64),1,1);
    auto *o=ggml_flash_attn_ext(meta,q,k,v,m,1.0f/sqrtf(D),0,0);
    const size_t outputs=ggml_nelements(o),nqv=ggml_nelements(q),nkv=ggml_nelements(k);
    Buffer<float> dq(nqv);Buffer<ggml_fp16_t> dk(nkv),dv(nkv),dm(ggml_nelements(m));
    Buffer<uint32_t> out(outputs+2*guard);Buffer<unsigned char> cold(72*1024*1024);
    q->data=dq.p;k->data=dk.p;v->data=dv.p;m->data=dm.p;o->data=out.p+guard;
    std::vector<float> hqv(nqv),fkv(nkv);std::vector<ggml_fp16_t> hkvv(nkv),mask(ggml_nelements(m));
    int past=std::min(512,nk-nq);
    for(int j=0;j<m->ne[1];++j)for(int i=0;i<nk;++i)
        mask[size_t(j)*nk+i]=ggml_fp32_to_fp16(i<past+(causal?std::min(j+1,nq):nq)?0.0f:-INFINITY);
    ck(cudaMemcpy(dm.p,mask.data(),mask.size()*2,cudaMemcpyHostToDevice));
    std::vector<uint32_t> poison(outputs+2*guard,sentinel);
    auto reset=[&]{ck(cudaMemcpyAsync(out.p,poison.data(),poison.size()*4,cudaMemcpyHostToDevice,c.stream()));};
    auto upload=[&](int phase){std::mt19937 rng(20261019+phase);std::normal_distribution<float> d(0,0.5f);
        for(auto &x:hqv)x=phase==2?0:d(rng);
        ck(cudaMemcpyAsync(dq.p,hqv.data(),hqv.size()*4,cudaMemcpyHostToDevice,c.stream()));
        for(auto ptr:{dk.p,dv.p}){for(auto &x:fkv)x=phase==2?0:d(rng);
            ggml_fp32_to_fp16_row(fkv.data(),hkvv.data(),fkv.size());
            ck(cudaMemcpyAsync(ptr,hkvv.data(),hkvv.size()*2,cudaMemcpyHostToDevice,c.stream()));
            ck(cudaStreamSynchronize(c.stream()));}};
    auto read=[&]{ck(cudaStreamSynchronize(c.stream()));std::vector<uint32_t> r(poison.size());
        ck(cudaMemcpy(r.data(),out.p,r.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<guard;++i)need(r[i]==sentinel&&r[guard+outputs+i]==sentinel,"output guards");
        return r;};
    upload(0);reset();launch<D>(false,c,o);launch<D>(true,c,o);ck(cudaStreamSynchronize(c.stream()));
    Graph g[2],warm[2];for(int mode=0;mode<2;++mode){capture<D>(g[mode],mode,c,o,1);capture<D>(warm[mode],mode,c,o,20);}
    auto a=resource(g[0]),b=resource(g[1]);
    printf("RESOURCE D=%d nq=%d nk=%d causal=%d old_regs=%d new_regs=%d old_local=%d new_local=%d old_shared=%u new_shared=%u old_cta=%d new_cta=%d old_grid=%u new_grid=%u\n",
        D,nq,nk,causal,a.regs,b.regs,a.local,b.local,a.shared,b.shared,a.cta,b.cta,a.grid,b.grid);fflush(stdout);
    need(b.regs<=128&&b.local<=a.local&&b.shared<=24576&&b.cta==4,"predeclared four-CTA resource gate");
    need(b.grid%(4*hkv)==0 && a.grid%hkv==0,"uniform KV partitions");
    printf("PARTITIONS old=%u new=%u query_tile_old=16 query_tile_new=8 group_old=4 group_new=2\n",a.grid/hkv,b.grid/(4*hkv));
    double maxerr=0,maxrel=0;int bitdiff=0;
    std::vector<uint32_t> first[2];
    for(int phase:{0,1,0,2}){upload(phase);std::vector<uint32_t> results[2];
        for(int mode=0;mode<2;++mode){reset();ck(cudaGraphLaunch(g[mode].e,c.stream()));results[mode]=read();
            reset();ck(cudaGraphLaunch(g[mode].e,c.stream()));need(results[mode]==read(),"exact self replay");
            if(phase==0){if(first[mode].empty())first[mode]=results[mode];else need(first[mode]==results[mode],"A/B/A deterministic replay");}}
        double largest=1,err=0,sq=0,esq=0;
        for(size_t j=guard;j<guard+outputs;++j){float x,y;memcpy(&x,&results[0][j],4);memcpy(&y,&results[1][j],4);
            need(std::isfinite(x)&&std::isfinite(y),"finite full outputs");
            largest=std::max(largest,double(fabsf(x)));err=std::max(err,double(fabsf(x-y)));
            sq+=double(x)*x;esq+=double(x-y)*(x-y);bitdiff+=results[0][j]!=results[1][j];}
        maxerr=std::max(maxerr,err/largest);maxrel=std::max(maxrel,sqrt(esq/std::max(sq,1e-30)));
        need(err/largest<=2e-3&&sqrt(esq/std::max(sq,1e-30))<=2e-3,"predeclared numerical gate");}
    printf("NUMERIC D=%d nq=%d nk=%d causal=%d scaledmax=%.9g relL2=%.9g bitdiff=%d\n",D,nq,nk,causal,maxerr,maxrel,bitdiff);
    upload(0);
    for(bool evict:{false,true}){auto *tg=evict?g:warm;
        for(int r=0;r<10;++r){double oldus=0,newus=0;
            for(int mode: {0,1,1,0}){float us=timed(tg[mode],c.stream(),evict?1:20,evict?cold.p:nullptr);(mode?newus:oldus)+=us;}
            if(r>=2)printf("TIME D=%d nq=%d nk=%d causal=%d cold=%d round=%d old_us=%.6f new_us=%.6f gain=%.6f\n",D,nq,nk,causal,evict,r,oldus/2,newus/2,oldus/newus);}}
    ck(cudaStreamSynchronize(c.stream()));ggml_free(meta);
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);auto backend=ggml_backend_cuda_init(0);need(backend,"CUDA backend");
    ggml_backend_cuda_context c(0);ggml_cuda_set_device(0);
    for(int nk:{512,768,1024})for(bool causal:{false,true}){
        shape<128>(c,16,nk,causal);shape<256>(c,16,nk,causal);}
    puts("FATTN_LOWSTAGE_PASS cases=12");return 0;}
