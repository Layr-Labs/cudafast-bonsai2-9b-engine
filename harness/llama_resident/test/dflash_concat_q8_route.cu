// Actual backend route + same-address connected CONCAT/quantize/K/V gate.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "common.cuh"
#include "concat.cuh"
#include "quantize.cuh"
#include "dflash_concat_q8_reference.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void need(bool ok,const char * why) {if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
static void check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
static void equal(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,const char* why) {
    need(a.size()==b.size(),"same byte count");
    for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]){std::fprintf(stderr,"%s byte=%zu old=%02x new=%02x\n",why,i,a[i],b[i]);need(false,why);}
}
static std::vector<uint8_t> read(const void* p,size_t n) {
    std::vector<uint8_t> v(n);check(cudaMemcpy(v.data(),p,n,cudaMemcpyDeviceToHost));return v;
}
static std::vector<uint8_t> read(const ggml_tensor* t) {
    std::vector<uint8_t> v(ggml_nbytes(t));ggml_backend_tensor_get(t,v.data(),0,v.size());return v;
}
struct Device {
    uint8_t* raw=nullptr;size_t bytes;
    static constexpr size_t guard=256;
    explicit Device(size_t n):bytes(n){check(cudaMalloc((void**)&raw,bytes+2*guard));poison();check(cudaStreamSynchronize(nullptr));}
    ~Device(){if(raw)check(cudaFree(raw));}
    void* data(){return raw+guard;}
    void poison(cudaStream_t stream=nullptr){check(cudaMemsetAsync(raw,0xa5,bytes+2*guard,stream));}
    void guards(){auto a=read(raw,guard),b=read(raw+guard+bytes,guard);for(auto x:a)need(x==0xa5,"leading guard");for(auto x:b)need(x==0xa5,"trailing guard");}
};
struct Backend {
    ggml_backend_t b=ggml_backend_cuda_init(0);
    Backend(){need(b,"CUDA backend");}~Backend(){ggml_backend_synchronize(b);ggml_backend_free(b);}
    ggml_backend_cuda_context& cuda(){return *static_cast<ggml_backend_cuda_context*>(b->context);}
};
struct Spec {int k=4096,na=16,nb=16,m=1024;bool strided=false,unaligned=false,extra=false,view=false,observable=false;};
struct Inputs {
    static constexpr int guard=16;
    ggml_context* ctx;ggml_backend_buffer_t buffer;
    ggml_tensor * ar,*br,*a,*b,*cat,*w0,*w1,*o0,*o1;Spec s;int stride;
    Inputs(Backend& be,Spec s):s(s),stride(s.k+(s.strided?4:0)) {
        ctx=ggml_init({ggml_tensor_overhead()*32,nullptr,true});
        const int pad=s.unaligned?1:0;
        ar=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(stride)*s.na+2*guard+pad);
        br=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,int64_t(stride)*s.nb+2*guard+pad);
        a=ggml_view_2d(ctx,ar,s.k,s.na,stride*4,(guard+pad)*4);
        b=ggml_view_2d(ctx,br,s.k,s.nb,stride*4,(guard+pad)*4);
        cat=ggml_concat(ctx,a,b,1);
        w0=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,s.k,s.m);
        w1=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,s.k,s.m);
        o0=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s.m,s.na+s.nb);
        o1=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s.m,s.na+s.nb);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,be.b);need(buffer,"inputs allocation");
        ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::mt19937 rng(181022);
        for(auto* w:{w0,w1}){
            std::vector<uint8_t> bytes(ggml_nbytes(w));
            for(size_t i=0;i<bytes.size()/sizeof(block_q8_0);++i){
                float scale=float(1+rng()%31)/8192;
                if(i%127==0)scale=0;if(i%131==0)scale=-scale;if(i%137==0)scale=3e-7f;
                auto d=ggml_fp32_to_fp16(scale);std::memcpy(bytes.data()+i*34,&d,2);
                for(int j=0;j<32;++j)bytes[i*34+2+j]=uint8_t(rng());
            }
            ggml_backend_tensor_set(w,bytes.data(),0,bytes.size());
        }
    }
    ~Inputs(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void fill(int phase) {
        std::mt19937 rng(1729+phase);
        for(auto* t:{ar,br}){
            std::vector<uint32_t> v(ggml_nelements(t),0x4a5a39c7);
            const int n=t==ar?s.na:s.nb;const int offset=guard+(s.unaligned?1:0);
            for(int r=0;r<n;++r)for(int k=0;k<s.k;++k){
                float x=phase==2?0.0f:float(int(rng()%2049)-1024)/1024;
                if(phase==3)x*=1e-30f;if((r*s.k+k)%97==0)x=-0.0f;
                std::memcpy(&v[offset+r*stride+k],&x,4);
            }
            ggml_backend_tensor_set(t,v.data(),0,v.size()*4);
        }
    }
};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Graph()=default;
    explicit Graph(cudaGraph_t source){check(cudaGraphClone(&graph,source));instantiate();}
    ~Graph(){if(exec)check(cudaGraphExecDestroy(exec));if(graph)check(cudaGraphDestroy(graph));}
    void instantiate(){check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));}
    void launch(cudaStream_t stream){check(cudaGraphLaunch(exec,stream));}
};
static cudaGraph_t captured(Backend& be,ggml_cgraph* graph) {
#ifdef USE_CUDA_GRAPH
    need(graph->n_nodes>0,"graph nodes");auto* n=graph->nodes[0];uint64_t key=uintptr_t(n);
    key=key*0x9E3779B97F4A7C15ull^uint64_t(graph->n_nodes);
    for(int i=0;i<GGML_MAX_DIMS;++i)key=key*0x9E3779B97F4A7C15ull^uint64_t(n->ne[i]);
    auto found=be.cuda().cuda_graphs.find((void*)uintptr_t(key));need(found!=be.cuda().cuda_graphs.end(),"exact graph key");
    const auto& c=*found->second;need(c.graph&&c.instance&&c.warmup_complete,"captured backend graph");
    need(c.node_props.size()==size_t(graph->n_nodes),"capture size");
    for(int i=0;i<graph->n_nodes;++i){ggml_cuda_graph::node_properties p={};
        std::memcpy(&p.node,graph->nodes[i],sizeof(ggml_tensor));
        for(int j=0;j<GGML_MAX_SRC;++j)if(graph->nodes[i]->src[j]){
            p.node_src_data_ptrs[j]=graph->nodes[i]->src[j]->data;
            std::memcpy(p.node_src_ne[j],graph->nodes[i]->src[j]->ne,sizeof(p.node_src_ne[j]));
            std::memcpy(p.node_src_nb[j],graph->nodes[i]->src[j]->nb,sizeof(p.node_src_nb[j]));
        }
        need(!std::memcmp(&c.node_props[i],&p,sizeof(p)),"exact graph properties");
    }
    return c.graph;
#else
    need(false,"CUDA graph build required");return nullptr;
#endif
}
static void inspect_mmq_consumers(cudaGraph_t graph,void* packed,void* output0,void* output1) {
    size_t n=0;check(cudaGraphGetNodes(graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(graph,nodes.data(),&n));
    int seen[2]={0,0};
    for(auto node:nodes){
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));const char* name=nullptr;check(cudaFuncGetName(&name,p.func));
        if(!std::strstr(name,"mul_mat_q")||std::strstr(name,"fixup")||std::strstr(name,"quantize"))continue;
        // These N32 Q8 readers use the ordinary mul_mat_q signature (x,y,ids,
        // expert_bounds,dst,...), never the N<=16 DL kernel specialization.
        void* dst=*(void**)p.kernelParams[4];int index=dst==output0?0:dst==output1?1:-1;
        need(index>=0,"only the two expected MMQ outputs");
        need(*(void**)p.kernelParams[1]==packed,"both MMQs consume the actual quantizer output");++seen[index];
    }
    need(seen[0]==1&&seen[1]==1,"exactly one unchanged MMQ per K/V output");
    size_t edges=0;check(cudaGraphGetEdges(graph,nullptr,nullptr,nullptr,&edges));
    std::vector<cudaGraphNode_t> from(edges),to(edges);std::vector<cudaGraphEdgeData> data(edges);
    check(cudaGraphGetEdges(graph,from.data(),to.data(),data.data(),&edges));
    bool pdl=false;for(auto edge:data)pdl|=edge.type==cudaGraphDependencyTypeProgrammatic;
    need(pdl,"actual programmatic-dependency capture edges");
}
struct Route {
    Backend& be;Inputs& in;ggml_context* ctx;ggml_cgraph* graph;ggml_gallocr_t alloc;
    ggml_tensor *a,*b,*cat,*o0,*o1,*extra=nullptr;
    Route(Backend& be,Inputs& in,bool original):be(be),in(in){
        ctx=ggml_init({ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true});graph=ggml_new_graph(ctx);
        const int off=(in.guard+(in.s.unaligned?1:0))*4;
        a=ggml_view_2d(ctx,in.ar,in.s.k,in.s.na,in.stride*4,off);
        b=ggml_view_2d(ctx,in.br,in.s.k,in.s.nb,in.stride*4,off);
        ggml_build_forward_expand(graph,a);ggml_build_forward_expand(graph,b);
        cat=ggml_concat(ctx,a,b,1);if(original||in.s.observable)ggml_set_output(cat);
        o0=ggml_mul_mat(ctx,in.w0,cat);o1=ggml_mul_mat(ctx,in.w1,cat);
        ggml_set_output(o0);ggml_set_output(o1);ggml_build_forward_expand(graph,o0);ggml_build_forward_expand(graph,o1);
        if(in.s.extra)extra=ggml_scale(ctx,cat,0.5f);
        if(in.s.view)extra=ggml_cont(ctx,ggml_view_2d(ctx,cat,in.s.k,in.s.na,in.s.k*4,0));
        if(extra){ggml_set_output(extra);ggml_build_forward_expand(graph,extra);}
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.b));need(alloc&&ggml_gallocr_alloc_graph(alloc,graph),"route graph allocation");
    }
    ~Route(){ggml_gallocr_free(alloc);ggml_free(ctx);}
    void run(){need(ggml_backend_graph_compute(be.b,graph)==GGML_STATUS_SUCCESS,"backend graph compute");ggml_backend_synchronize(be.b);}
    void* inspect(bool fused,void* fused_func){
        size_t n=0;auto g=captured(be,graph);check(cudaGraphGetNodes(g,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(g,nodes.data(),&n));
        int qcount=0,ccount=0;void* out=nullptr;
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));const char* name=nullptr;check(cudaFuncGetName(&name,p.func));
            if(std::strstr(name,"concat_"))++ccount;
            if(!std::strstr(name,"quantize_mmq_q8_1"))continue;
            ++qcount;need((p.func==fused_func)==fused,"actual linked fused function identity");out=*(void**)p.kernelParams[2];
            need(p.gridDim.x==unsigned(in.s.na+in.s.nb)&&p.gridDim.y==unsigned(in.s.k/512)&&p.blockDim.x==128,"quantizer geometry");
            need(*(int64_t*)p.kernelParams[3]==in.s.k&&*(int*)p.kernelParams[8]==in.s.na+in.s.nb,"quantizer full output pitch");
            if(fused)need(*(void**)p.kernelParams[0]==a->data&&*(void**)p.kernelParams[11]==b->data&&*(int*)p.kernelParams[10]==16,"original slab pointers and split");
            else need(*(void**)p.kernelParams[0]==cat->data,"original materialized concat read");
        }
        need(qcount==1,"K/V share exactly one quantization");need(ccount==(fused?0:1),"concat elided only in eligible route");
        need(out,"captured packed activation");inspect_mmq_consumers(g,out,o0->data,o1->data);return out;
    }
};
static void frozen_quant(const float* x,void* y,int k,int n,cudaStream_t stream){
    const int64_t width=k,extent=int64_t(k)*n;
    mmq_launch_kernel(GGML_CUDA_CC_DGX_SPARK,dflash_concat_q8_reference<MMQ_Q8_1_DS_LAYOUT_D4,false>,dim3(n,k/512),dim3(128),0,stream,
        x,(const int32_t*)nullptr,y,width,width,extent,extent,width,n,1,0,(const float*)nullptr,(const float*)nullptr,(const float*)nullptr);
}
static void* fused_function(Backend& be,Inputs& in,Device& out){
    auto stream=be.cuda().stream();quantize_mmq_q8_1_concat_cuda((float*)in.a->data,(float*)in.b->data,out.data(),4096,16,16,stream);check(cudaStreamSynchronize(stream));
    Graph g;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    quantize_mmq_q8_1_concat_cuda((float*)in.a->data,(float*)in.b->data,out.data(),4096,16,16,stream);
    check(cudaStreamEndCapture(stream,&g.graph));size_t n=0;check(cudaGraphGetNodes(g.graph,nullptr,&n));need(n==1,"one linked fused quantizer");
    cudaGraphNode_t node;check(cudaGraphGetNodes(g.graph,&node,&n));cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));return p.func;
}
static uint32_t hash32(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
static __global__ void trash_init(uint32_t* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x){uint32_t x=uint32_t(i);x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;p[i]=x;}}
static __global__ void trash_add(uint32_t* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]+=1;}
static float median(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static void direct_gate(Backend& be,Inputs& in,void* fused_func){
    auto& ctx=be.cuda();auto stream=ctx.stream();const size_t defined=size_t(4096/128)*32*sizeof(block_q8_1_mmq);
    Device packed(ggml_cuda_mul_mat_q_q8_size(in.w0,in.cat)),oracle(packed.bytes),trash(72*1024*1024);
    Graph arms[3];
    auto launch=[&](int arm){
        if(arm<2){ggml_cuda_op_concat(ctx,in.cat);quantize_mmq_q8_1_cuda((float*)in.cat->data,nullptr,packed.data(),GGML_TYPE_Q8_0,4096,4096,4096*32,4096*32,4096,32,1,1,stream);}
        else quantize_mmq_q8_1_concat_cuda((float*)in.a->data,(float*)in.b->data,packed.data(),4096,16,16,stream);
        ggml_cuda_mul_mat_q(ctx,in.w0,in.cat,nullptr,in.o0,nullptr,nullptr,nullptr,packed.data(),false);
        ggml_cuda_mul_mat_q(ctx,in.w1,in.cat,nullptr,in.o1,nullptr,nullptr,nullptr,packed.data(),false);
    };
    in.fill(0);for(int arm=0;arm<3;++arm)launch(arm);check(cudaStreamSynchronize(stream));
    for(int arm=0;arm<3;++arm){check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(arm);check(cudaStreamEndCapture(stream,&arms[arm].graph));arms[arm].instantiate();
        size_t n=0;check(cudaGraphGetNodes(arms[arm].graph,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(arms[arm].graph,nodes.data(),&n));int quant=0;
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));const char* name=nullptr;check(cudaFuncGetName(&name,p.func));
            if(std::strstr(name,"quantize_mmq_q8_1")){
                ++quant;need(*(void**)p.kernelParams[2]==packed.data(),"all arms same quant address");need((p.func==fused_func)==(arm==2),"timed real kernel route");
                if(arm!=1){cudaFuncAttributes attributes={};check(cudaFuncGetAttributes(&attributes,p.func));int active=0;
                    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,p.func,int(p.blockDim.x*p.blockDim.y*p.blockDim.z),p.sharedMemBytes));
                    std::printf("DFLASH_CONCAT_Q8_RESOURCE arm=%s registers=%d local_bytes=%zu static_shared=%zu dynamic_shared=%u active_ctas=%d block=%u/%u/%u grid=%u/%u/%u\n",
                        arm==0?"original":"fused",attributes.numRegs,attributes.localSizeBytes,attributes.sharedSizeBytes,p.sharedMemBytes,active,
                        p.blockDim.x,p.blockDim.y,p.blockDim.z,p.gridDim.x,p.gridDim.y,p.gridDim.z);
                }
            }
        }need(quant==1,"one quant per timed chain");inspect_mmq_consumers(arms[arm].graph,packed.data(),in.o0->data,in.o1->data);
    }
    for(int phase:{0,1,1,0,2,3,0}){
        in.fill(phase);auto a=read(in.ar),b=read(in.br);packed.poison(stream);arms[0].launch(stream);check(cudaStreamSynchronize(stream));
        auto q=read(packed.data(),packed.bytes),o0=read(in.o0),o1=read(in.o1);
        frozen_quant((float*)in.cat->data,oracle.data(),4096,32,stream);check(cudaStreamSynchronize(stream));equal(read(oracle.data(),defined),std::vector<uint8_t>(q.begin(),q.begin()+defined),"frozen original quantizer");
        for(int arm:{1,2}){packed.poison(stream);arms[arm].launch(stream);check(cudaStreamSynchronize(stream));equal(q,read(packed.data(),packed.bytes),"all D4 bytes and untouched padding");equal(o0,read(in.o0),"complete K output");equal(o1,read(in.o1),"complete V output");}
        equal(a,read(in.ar),"first slab immutable");equal(b,read(in.br),"second slab immutable");packed.guards();oracle.guards();
    }
    in.fill(0);const size_t nw=trash.bytes/4;trash_init<<<256,256,0,stream>>>((uint32_t*)trash.data(),nw);check(cudaStreamSynchronize(stream));
    cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));unsigned evictions=0;
    auto sample=[&](int arm,bool cold){if(cold){trash_add<<<256,256,0,stream>>>((uint32_t*)trash.data(),nw);++evictions;}else arms[arm].launch(stream);
        check(cudaEventRecord(start,stream));arms[arm].launch(stream);check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,start,end));return ms*1000;};
    for(int other:{1,2})for(bool cold:{false,true}){
        std::vector<float> old,newer,saved;
        for(int q=0;q<7;++q){int order[4]={0,other,other,0};if(q&1){order[0]=other;order[1]=0;order[2]=0;order[3]=other;}
            float a=0,b=0,raw[4]={};for(int j=0;j<4;++j){const int v=order[j];raw[j]=sample(v,cold);if(v==0)a+=raw[j];else b+=raw[j];}
            a*=0.5f;b*=0.5f;old.push_back(a);newer.push_back(b);saved.push_back(a-b);
            std::printf("DFLASH_CONCAT_Q8_QUARTET arm=%s cache=%s quartet=%d order=%d/%d/%d/%d raw_us=%.6f/%.6f/%.6f/%.6f old_us=%.6f new_us=%.6f saved_us=%.6f\n",
                other==1?"self":"fused",cold?"evicted":"warm",q,order[0],order[1],order[2],order[3],raw[0],raw[1],raw[2],raw[3],a,b,a-b);
        }
        std::printf("DFLASH_CONCAT_Q8_TIMING arm=%s cache=%s old_us=%.3f new_us=%.3f paired_saved_us=%.3f positive=%d/7 same_addresses=true complete_KV_chain=true\n",other==1?"self":"fused",cold?"evicted":"warm",median(old),median(newer),median(saved),int(std::count_if(saved.begin(),saved.end(),[](float x){return x>0;})));
    }
    check(cudaEventDestroy(end));check(cudaEventDestroy(start));auto bytes=read(trash.data(),trash.bytes);
    for(size_t i=0;i<nw;++i){uint32_t x;std::memcpy(&x,bytes.data()+4*i,4);need(x==hash32(uint32_t(i))+evictions,"full nonuniform eviction coverage");}
    need(evictions==56,"balanced cold launch count");trash.guards();packed.guards();oracle.guards();
    std::printf("DFLASH_CONCAT_Q8_DIRECT_PASS changed_cases=7 controls=original/self/fused evictions=%u\n",evictions);
}
int main(int argc,char** argv){
    const bool explicit_enabled=argc==2&&!std::strcmp(argv[1],"--enabled"),disabled=argc==2&&!std::strcmp(argv[1],"--disabled");
    need(argc==1||explicit_enabled||disabled,"mode");if(explicit_enabled)setenv("GGML_CUDA_DFLASH_CONCAT_Q8","1",1);else if(disabled)setenv("GGML_CUDA_DFLASH_CONCAT_Q8","0",1);else unsetenv("GGML_CUDA_DFLASH_CONCAT_Q8");
    const bool enabled=!disabled;
    if(argc==1)need(getenv("GGML_CUDA_DFLASH_CONCAT_Q8")==nullptr,"default route genuinely has absent environment");
    setenv("GGML_CUDA_PDL","1",1);unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    void* func=nullptr;{
        Backend be;Inputs in(be,{});in.fill(0);Device out(32*32*sizeof(block_q8_1_mmq));func=fused_function(be,in,out);
        if(explicit_enabled)direct_gate(be,in,func);
    }
    std::vector<Spec> specs={{},{4096,16,15},{2048,16,16},{4096,16,16,128},
        {4096,16,16,1024,true},{4096,16,16,1024,false,true},{4096,16,16,1024,false,false,true},
        {4096,16,16,1024,false,false,false,true},{4096,16,16,1024,false,false,false,false,true}};
    int cases=0;
    for(size_t i=0;i<specs.size();++i){Backend be;Inputs in(be,specs[i]);Route old(be,in,true),now(be,in,false);in.fill(0);
        for(int r=0;r<3;++r){old.run();now.run();}
        const bool fused=enabled&&i==0;old.inspect(false,func);now.inspect(fused,func);
        Graph clone(captured(be,now.graph));Device oracle(size_t(in.s.k/128)*(in.s.na+in.s.nb)*sizeof(block_q8_1_mmq));
        for(int phase:{0,1,1,0,2,3,0}){in.fill(phase);auto a=read(in.ar),b=read(in.br);old.run();auto o0=read(old.o0),o1=read(old.o1);auto x=old.extra?read(old.extra):std::vector<uint8_t>();
            auto ref=read(old.inspect(false,func),oracle.bytes);frozen_quant((float*)old.cat->data,oracle.data(),in.s.k,in.s.na+in.s.nb,be.cuda().stream());check(cudaStreamSynchronize(be.cuda().stream()));equal(ref,read(oracle.data(),oracle.bytes),"backend original vs independent frozen quantizer");
            clone.launch(be.cuda().stream());check(cudaStreamSynchronize(be.cuda().stream()));equal(o0,read(now.o0),"backend K bytes");equal(o1,read(now.o1),"backend V bytes");if(now.extra)equal(x,read(now.extra),"extra observer fallback");equal(ref,read(now.inspect(fused,func),oracle.bytes),"backend complete Q8 bytes");
            equal(a,read(in.ar),"backend first slab immutable");equal(b,read(in.br),"backend second slab immutable");oracle.guards();++cases;
        }
        std::printf("DFLASH_CONCAT_Q8_ROUTE case=%zu fused=%d source_pitch=%d shape=%d/%d+%d/%d\n",i,int(fused),in.stride,in.s.k,in.s.na,in.s.nb,in.s.m);
    }
    std::printf("DFLASH_CONCAT_Q8_PASS byte_cases=%d mode=%s\n",cases,explicit_enabled?"enabled":disabled?"explicit0":"absent");
}
