// Actual backend gather/scatter/PDL/graph integration for the GD2L writer.
// Coherent lazy caches and full eager imports are supported. Artificial mixtures
// of stale lazy headers and independent eager columns are outside this contract.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"
#include "gdn-dedup.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>
namespace {
constexpr int D=128,H=32,HK=16,K=16,TMAX=512,G=64;
constexpr size_t ROW=size_t(D)*D*H,CACHE=ROW*(K+1);
constexpr uint32_t CANARY=0x46a98700u,GD1=0x4744314cu,GD2=0x4744324cu,REHYDRATE=0x80000000u;
int checks=0;
void need(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL %s\n",why);std::exit(1);}}
void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
void equal(const char* label,const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){need(a.size()==b.size(),"same comparison extent");for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]){std::fprintf(stderr,"FAIL %s at=%zu %08x/%08x\n",label,i,a[i],b[i]);std::exit(1);}}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
std::vector<uint32_t> read(ggml_tensor* t){std::vector<uint32_t> b(ggml_nelements(t));ggml_backend_tensor_get(t,b.data(),0,b.size()*4);return b;}
struct Backend {
    ggml_backend_t b=ggml_backend_cuda_init(0);
    Backend(){need(b,"backend init");}~Backend(){ggml_backend_synchronize(b);ggml_backend_free(b);}
    ggml_backend_cuda_context& cuda(){return *static_cast<ggml_backend_cuda_context*>(b->context);}
};
// Match ggml_cuda_graph_get_key exactly. Allocator addresses can be reused by
// temporary fixtures; an output-pointer scan could select an unrelated graph.
cudaGraph_t captured(Backend& backend, ggml_cgraph* graph) {
#ifdef USE_CUDA_GRAPH
    need(graph->n_nodes > 0, "capture graph has nodes");
    const ggml_tensor* node0 = graph->nodes[0];
    uint64_t key = uint64_t(uintptr_t(node0));
    key = key*0x9E3779B97F4A7C15ull ^ uint64_t(graph->n_nodes);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) key = key*0x9E3779B97F4A7C15ull ^ uint64_t(node0->ne[i]);
    const auto found = backend.cuda().cuda_graphs.find(reinterpret_cast<const void*>(uintptr_t(key)));
    need(found != backend.cuda().cuda_graphs.end(), "exact backend capture key exists");
    const auto& capture = *found->second;
    need(capture.graph && capture.instance && capture.warmup_complete, "exact backend capture ready");
    need(capture.node_props.size() == size_t(graph->n_nodes), "exact backend capture node count");
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_cuda_graph::node_properties prop = {};
        std::memcpy(&prop.node, graph->nodes[i], sizeof(ggml_tensor));
        for (int j = 0; j < GGML_MAX_SRC; ++j) if (graph->nodes[i]->src[j]) {
            prop.node_src_data_ptrs[j] = graph->nodes[i]->src[j]->data;
            std::memcpy(prop.node_src_ne[j], graph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
            std::memcpy(prop.node_src_nb[j], graph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
        }
        need(std::memcmp(&capture.node_props[i], &prop, sizeof(prop)) == 0, "exact backend capture properties");
    }
    return capture.graph;
#else
    need(false, "CUDA graphs required"); return nullptr;
#endif
}
struct Inputs {
    ggml_context * ctx=ggml_init({ggml_tensor_overhead()*16,nullptr,true});ggml_backend_buffer_t buffer;
    ggml_tensor *q,*k,*v,*g,*beta,*bias,*a,*ids;
    Inputs(Backend& b){q=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,D*HK*TMAX);k=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,D*HK*TMAX);v=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,D*H*TMAX);g=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H*TMAX);beta=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H*TMAX);bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H);a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H);ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);buffer=ggml_backend_alloc_ctx_tensors(ctx,b.b);need(buffer,"inputs allocation");}
    ~Inputs(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void fill(unsigned seed,int mode){std::mt19937 rng(seed);std::uniform_real_distribution<float> u(-1,1);auto put=[&](ggml_tensor* t,float scale,float offset){std::vector<float>x(ggml_nelements(t));for(auto& z:x)z=scale*u(rng)+offset;ggml_backend_tensor_set(t,x.data(),0,x.size()*4);};put(q,.09f,0);put(k,.09f,0);put(v,.8f,0);put(g,mode==1?.8f:.025f,mode==1?-2.f:-.10f);put(beta,mode==1?.8f:.15f,mode==1?0.f:.5f);put(bias,.1f,-.2f);put(a,.03f,-.1f);}
    void row(int rb){ggml_backend_tensor_set(ids,&rb,0,4);}
};
struct State {
    ggml_context* ctx=ggml_init({ggml_tensor_overhead()*4,nullptr,true});ggml_tensor* raw=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,CACHE+2*G);ggml_backend_buffer_t buffer;
    State(Backend& b){buffer=ggml_backend_alloc_ctx_tensors(ctx,b.b);need(buffer,"state allocation");}
    ~State(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    float* data(){return static_cast<float*>(raw->data)+G;}
    std::vector<uint32_t> get(){auto b=read(raw);for(int i=0;i<G;++i)need(b[i]==CANARY&&b[G+CACHE+i]==CANARY,"cache guards");return b;}
    void put(const std::vector<uint32_t>& b){need(b.size()==CACHE+2*G,"state extent");ggml_backend_tensor_set(raw,b.data(),0,b.size()*4);}
    static std::vector<uint32_t> initial(unsigned seed){std::mt19937 rng(seed);std::uniform_real_distribution<float> u(-.02f,.02f);std::vector<uint32_t>b(CACHE+2*G,CANARY);for(size_t i=G;i<G+CACHE;++i)b[i]=bits(u(rng));return b;}
};
struct Graph {
    Backend& backend;Inputs& inputs;State& state;int t,mode;bool candidate,zero,early,eager;
    ggml_context* ctx=ggml_init({ggml_tensor_overhead()*160+ggml_graph_overhead(),nullptr,true});ggml_cgraph* graph=ggml_new_graph(ctx);ggml_gallocr_t allocator;
    ggml_tensor *out,*gdn,*cache,*gather,*exposed=nullptr;
    Graph(Backend& b,Inputs& in,State& st,int t,int mode,bool candidate,bool zero=false,bool early=false,bool eager=false):backend(b),inputs(in),state(st),t(t),mode(mode),candidate(candidate),zero(zero),early(early),eager(eager){
        auto view=[&](ggml_tensor* raw,int d,int h){return ggml_reshape_4d(ctx,ggml_view_1d(ctx,raw,int64_t(d)*h*t,0),d,h,t,1);};
        // Distinct first VIEW per graph protects actual backend capture keys.
        // SCALE is an early-PDL-triggering producer, so the GDN wait is exercised.
        auto q=ggml_scale(ctx,view(in.q,D,HK),1.001f);auto k=view(in.k,D,HK);auto v=view(in.v,D,H);auto g=view(in.g,1,H);auto beta=view(in.beta,1,H);
        cache=ggml_view_2d(ctx,st.raw,ROW,K+1,ROW*4,G*4);
        if(early){exposed=ggml_scale(ctx,ggml_view_1d(ctx,cache,ROW,10*ROW*4),.5f);ggml_set_output(exposed);ggml_build_forward_expand(graph,exposed);}
        if(zero)ggml_build_forward_expand(graph,ggml_scale_inplace(ctx,ggml_view_1d(ctx,cache,ROW,0),0.f));
        gather=ggml_get_rows(ctx,cache,in.ids);if(eager)ggml_set_output(gather);
        auto s=ggml_reshape_4d(ctx,gather,D,D,H,1);gdn=ggml_gated_delta_net(ctx,q,k,v,g,beta,s,K);if(mode==1)ggml_gated_delta_net_set_raw_gates(gdn,in.bias,in.a);
        const int w=std::min(t,K);auto tail=ggml_view_3d(ctx,gdn,ROW,1,w,ROW*4,ROW*4,size_t(D)*H*t*4);auto dst=ggml_view_3d(ctx,cache,ROW,1,w,ROW*4,ROW*4,0);ggml_build_forward_expand(graph,ggml_cpy(ctx,tail,dst));
        out=ggml_cont(ctx,ggml_view_1d(ctx,gdn,int64_t(D)*H*t,0));ggml_set_output(out);ggml_build_forward_expand(graph,out);allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b.b));need(ggml_gallocr_alloc_graph(allocator,graph),"graph allocation");
    }
    ~Graph(){ggml_backend_synchronize(backend.b);ggml_gallocr_free(allocator);ggml_free(ctx);}
    bool n1() const { const char * e=std::getenv("GGML_CUDA_GDN_CHUNKED");return t>=64&&(!e||std::atoi(e)!=0); }
    void flag(){setenv("GGML_CUDA_GDN_DEDUP",candidate?"1":"0",1);}
    void run(){flag();need(ggml_backend_graph_compute_async(backend.b,graph)==GGML_STATUS_SUCCESS,"graph compute");ggml_backend_synchronize(backend.b);}
    uint32_t* counter(){flag();return ggml_cuda_gdn_dedup_counter(backend.cuda(),state.data(),ROW,H,K);}
    std::vector<uint32_t> counts(){auto* p=counter();std::vector<uint32_t>b(H,REHYDRATE);if(p)check(cudaMemcpy(b.data(),p,H*4,cudaMemcpyDeviceToHost));return b;}
    cudaGraph_t captured(){return ::captured(backend,graph);}
    void inspect(bool wanted){auto gr=captured();size_t n=0;check(cudaGraphGetNodes(gr,nullptr,&n));std::vector<cudaGraphNode_t> nodes(n);check(cudaGraphGetNodes(gr,nodes.data(),&n));int gdns=0,prepares=0,scans=0,resets=0,half_prepares=0,half_scans=0,materializers=0;for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));const char* name=nullptr;check(cudaFuncGetName(&name,p.func));prepares+=std::strstr(name,"gdn_compact32_prepare")||std::strstr(name,"gdn_chunk32_prepare");scans+=std::strstr(name,"gdn_compact32_scan")||std::strstr(name,"gdn_chunk32_scan");half_prepares+=std::strstr(name,"prepare_packed")!=nullptr;half_scans+=std::strstr(name,"carry_packed")!=nullptr;materializers+=std::strstr(name,"gdn_half_initial_state")!=nullptr;resets+=std::strstr(name,"rehydrate")!=nullptr;if(!std::strstr(name,"gated_delta_net_"))continue;++gdns;const bool dedup=std::strstr(name,"gated_delta_net_dedup_l8v_cuda");need(dedup==wanted,"captured positive/negative GD2 dispatch identity");need(p.gridDim.x==H&&p.gridDim.y==1&&p.gridDim.z==8&&p.blockDim.x==32&&p.blockDim.y==4,"actual GDN geometry");need(*static_cast<float**>(p.kernelParams[9])==state.data(),"actual fused cache destination");if(dedup)need(*static_cast<uint32_t**>(p.kernelParams[30])==counter(),"actual stable counter pointer");}
        need(gdns==1,"one native recurrence or N1 tail");const char* half_env=std::getenv("GGML_CUDA_GDN_HALF_PREFILL");const bool half=n1()&&(!half_env||std::atoi(half_env)!=0);need(prepares==int(n1()&&!half)&&scans==int(n1()&&!half),"captured C32 prepare/scan selection");need(half_prepares==int(half)&&half_scans==int(half)&&materializers==int(half),"captured C16 prepare/carry/materializer selection");if(n1()&&counter())need(resets>0,"captured N1 cache invalidation");
#ifdef GGML_CUDA_USE_PDL
        size_t edges=0;check(cudaGraphGetEdges(gr,nullptr,nullptr,nullptr,&edges));std::vector<cudaGraphNode_t> from(edges),to(edges);std::vector<cudaGraphEdgeData> data(edges);check(cudaGraphGetEdges(gr,from.data(),to.data(),data.data(),&edges));bool pdl=false;for(auto e:data)pdl|=e.type==cudaGraphDependencyTypeProgrammatic;need(pdl,"actual programmatic-dependency capture edges");
#else
        need(false,"PDL enabled build required");
#endif
    }
};
struct Before{std::vector<uint32_t> cache,counter;uint32_t* counter_pointer=nullptr;};
size_t rec(int z,int h){return G+size_t(2+z)*ROW+size_t(h)*D*D;}
Before before(Graph& g,uint32_t* owner_counter=nullptr){
    auto* local=g.counter();
    need(!local||!owner_counter||local==owner_counter,"existing context counter agrees with live owner");
    auto* p=local?local:owner_counter;
    std::vector<uint32_t> count(H,REHYDRATE);
    if(p)check(cudaMemcpy(count.data(),p,H*4,cudaMemcpyDeviceToHost));
    // A newly constructed context has no local registry mapping yet. If another
    // live owner already holds this cache, first registration must reuse its
    // current generation; absence from the new context does not mean rehydrate.
    return {g.state.get(),std::move(count),p};
}
// N1 marks every long compatible cache write; the next GD2 verify must rehydrate.
void compare(Graph& reference,Graph& got,const Before& old,const char* label,bool writer,bool eager=false,bool reset=false){
    equal(label,read(reference.out),read(got.out));const auto ref=reference.state.get(),actual=got.state.get();auto expected=old.cache;const int w=std::min(got.t,K);const bool verify=got.t>=8&&got.t<=16;
    if(eager){equal(label,ref,actual);++checks;return;}
    std::copy_n(ref.begin()+G+ROW,ROW,expected.begin()+G+ROW);
    if(!verify||got.zero)std::copy_n(ref.begin()+G,ROW,expected.begin()+G);
    else for(int h=0;h<H;++h)for(int c=0;c<D;c+=4)for(int j=0;j<2;++j){size_t at=G+size_t(h)*D*D+c*D+j;expected[at]=ref[at];}
    auto now=got.counts();
    auto* actual_counter=got.counter();
    if(old.counter_pointer)need(actual_counter==old.counter_pointer,"registration preserves existing counter allocation");
    for(int h=0;h<H;++h){const uint32_t c=old.counter[h],gen=(c&~REHYDRATE)/8u;const bool v2=writer&&!(c&REHYDRATE);
        const uint32_t want=(reset||got.n1())?REHYDRATE:writer?((c&REHYDRATE)?0u:c+8u):c;
        if(actual_counter&&now[h]!=want){
            const auto* node=got.gdn;
            const bool allowed=got.backend.cuda().gdn_dedup_nodes.count(node)!=0;
            std::fprintf(stderr,"FAIL counter ownership case=%s head=%d before=%08x got=%08x want=%08x before_ptr=%p actual_ptr=%p writer=%d reset=%d allowed=%d dispatches=%llu record=%08x gen=%08x\n",
                label,h,c,now[h],want,static_cast<void*>(old.counter_pointer),static_cast<void*>(actual_counter),int(writer),int(reset),int(allowed),
                static_cast<unsigned long long>(got.backend.cuda().gdn_dedup_launches),actual[rec(0,h)],actual[rec(0,h)+3]);
            std::exit(1);
        }
        for(int z=0;z<8;++z){const size_t at=rec(z,h);need(ref[at]==GD1&&ref[at+1]==uint32_t(w),"original lazy record formed");std::copy_n(ref.begin()+at,8,expected.begin()+at);expected[at]=v2?GD2:GD1;expected[at+3]=v2?gen:0;
            if(v2)for(int n=0;n<w;++n)std::copy_n(ref.begin()+at+8+n*146+128,16,expected.begin()+at+8+n*16);
            else std::copy_n(ref.begin()+at+8,w*146,expected.begin()+at+8);
        }
        if(v2){const size_t at=G+size_t(10+(gen&1))*ROW+size_t(h)*D*D,src=rec(0,h);expected[at]=GD2;expected[at+1]=w;expected[at+2]=0;expected[at+3]=gen;for(int n=0;n<w;++n){std::copy_n(ref.begin()+src+8+n*146,128,expected.begin()+at+8+n*130);std::copy_n(ref.begin()+src+8+n*146+144,2,expected.begin()+at+8+n*130+128);}}
    }
    equal(label,expected,actual);++checks;
}
void step(Graph& ref,Graph& got,unsigned seed,int rb,const char* label,bool writer=true,bool eager=false,bool reset=false,uint32_t* owner_counter=nullptr){
    std::printf("case %s T%d raw%d rb%d writer%d\n",label,got.t,got.mode,rb,int(writer));std::fflush(stdout);ref.inputs.fill(seed,got.mode);ref.inputs.row(rb);auto old=before(got,owner_counter);ref.run();got.run();compare(ref,got,old,label,writer,eager,reset);
    if(owner_counter){const auto now=got.counts();std::printf("counter-owner case=%s before0=%08x after0=%08x same_pointer=%d dispatches=%llu\n",label,old.counter[0],now[0],int(got.counter()==owner_counter),static_cast<unsigned long long>(got.backend.cuda().gdn_dedup_launches));}
}
void seed(Graph& ref,Graph& got,unsigned n){auto init=State::initial(n);ref.state.put(init);got.state.put(init);for(int i=0;i<3;++i)step(ref,got,n+1+i,0,"seed-captured-GD2");}
struct Clone {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;Clone(cudaGraph_t src){check(cudaGraphClone(&graph,src));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));}~Clone(){cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}void run(Backend& b){check(cudaGraphLaunch(exec,b.cuda().stream()));ggml_backend_synchronize(b.b);}};
struct Writer {
    Backend& backend;ggml_context* ctx=ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead(),nullptr,true});ggml_cgraph* graph=ggml_new_graph(ctx);ggml_tensor* copy;
    Writer(Backend& b,State& from,State& to):backend(b){auto* src=ggml_view_1d(ctx,from.raw,CACHE+2*G,0);copy=ggml_cpy(ctx,src,to.raw);ggml_build_forward_expand(graph,copy);}
    ~Writer(){ggml_backend_synchronize(backend.b);ggml_free(ctx);}
    void run(){need(ggml_backend_graph_compute_async(backend.b,graph)==GGML_STATUS_SUCCESS,"writer graph compute");ggml_backend_synchronize(backend.b);}
    cudaGraph_t captured(){return ::captured(backend,graph);}
    void reset_nodes(){size_t n=0;auto gr=captured();check(cudaGraphGetNodes(gr,nullptr,&n));std::vector<cudaGraphNode_t>nodes(n);check(cudaGraphGetNodes(gr,nodes.data(),&n));int resets=0;for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type!=cudaGraphNodeTypeKernel)continue;cudaKernelNodeParams p={};check(cudaGraphKernelNodeGetParams(node,&p));const char* name=nullptr;check(cudaFuncGetName(&name,p.func));resets+=std::strstr(name,"rehydrate")!=nullptr;}need(resets>=2,"writer captures reset brackets");}
};
void writer_lifecycle(){
    Backend control,writer;auto owner=std::make_unique<Backend>();Inputs in(control);State rs(control),gs(writer),source(writer);
    const auto init=State::initial(1701);rs.put(init);gs.put(init);source.put(init);
    Writer import(writer,source,gs);setenv("GGML_CUDA_GDN_DEDUP","0",1);for(int i=0;i<3;++i)import.run();import.captured();
    Graph ref(control,in,rs,16,1,false);auto got=std::make_unique<Graph>(*owner,in,gs,16,1,true);seed(ref,*got,1710);
    const auto sr=rs.get(),sg=gs.get();source.put(sg);uint32_t* counter=got->counter();need(counter,"owned counter before writer recapture");
    // The writer was already captured before the registry existed. Generation
    // change must prevent its old executable from replaying without reset.
    for(int i=0;i<3;++i){rs.put(sr);import.run();for(auto c:got->counts())need(c==REHYDRATE,"old writer graph obtains rehydrate");step(ref,*got,1720+i,8,"writer-before-registration-recapture");}
    import.reset_nodes();need(!writer.cuda().gdn_dedup_owners.empty(),"writer-only graph retains reset allocation");Clone clone(import.captured());
    got.reset();owner.reset(); // only the writer graph/clone owns this counter now
    std::vector<uint32_t> count(H);check(cudaMemcpy(count.data(),counter,H*4,cudaMemcpyDeviceToHost));
    clone.run(writer);check(cudaMemcpy(count.data(),counter,H*4,cudaMemcpyDeviceToHost));for(auto c:count)need(c==REHYDRATE,"writer clone survives original owner teardown");
    Backend new_owner;Graph next(new_owner,in,gs,16,1,true);rs.put(sr);step(ref,next,1730,8,"new-owner-after-writer-only-lifetime",true,false,false,counter);need(next.counter()==counter,"same cache retains counter across owner handoff");
    // Exact production buffer-free notification, while retaining physical test
    // storage so address reuse is deterministic. This tests the stale captured
    // pointer mechanism without depending on cudaMalloc returning the same VA.
    ggml_cuda_gdn_dedup_forget(gs.raw->data,ggml_nbytes(gs.raw));gs.put(sg);rs.put(sr);for(auto c:next.counts())need(c==REHYDRATE,"same-VA import still reaches captured counter");step(ref,next,1731,8,"same-address-teardown-notification-import");
}
void main_cases(){
    Backend control,candidate;Inputs in(control);State rs(control),gs(candidate);
    for(int mode=0;mode<2;++mode){
        Graph ref(control,in,rs,16,mode,false),got(candidate,in,gs,16,mode,true);
        for(int rb=0;rb<=16;++rb){seed(ref,got,100+rb);step(ref,got,300+rb,rb,"valid-GD2-all-rb");}
        ref.inspect(false);got.inspect(true);need(candidate.cuda().gdn_dedup_launches>0,"positive production GD2 launch count");
        Clone clone(got.captured());
        for(int i=0;i<12;++i){in.fill(400+i,mode);in.row(i%17);auto old=before(got);ref.run();if(i&1)clone.run(candidate);else got.run();compare(ref,got,old,"changed-input-original-clone-alternation",true);}
        // One context may build another graph over the same cache; every graph
        // must embed the same counter, not a fresh generation allocation.
        Graph other(candidate,in,gs,16,mode,true);for(int i=0;i<3;++i)step(ref,other,440+i,8,"same-context-independent-graph");need(other.counter()==got.counter(),"same-context graph counter identity");other.inspect(true);
        {Backend second;Graph foreign(second,in,gs,16,mode,true);for(int i=0;i<3;++i)step(ref,foreign,450+i,8,"serialized-second-context",true,false,false,got.counter());need(foreign.counter()==got.counter(),"two live contexts share cache counter");foreign.inspect(true);}
        step(ref,got,460,15,"surviving-owner-after-other-context-destruction");
        auto eager_seed=State::initial(470);rs.put(eager_seed);gs.put(eager_seed);
        std::vector<uint32_t> wrap(H,0x7ffffff8u);check(cudaMemcpy(got.counter(),wrap.data(),H*4,cudaMemcpyHostToDevice));
        step(ref,got,471,0,"last-owned-counter-generation");step(ref,got,472,8,"owned-counter-wrap-rehydrate");step(ref,got,473,0,"owned-counter-wrap-restart");

        for(int t:{1,4,64,512}){Graph short_ref(control,in,rs,t,mode,false),short_got(candidate,in,gs,t,mode,true);for(int rb:{0,8,16}){seed(ref,got,500+t+rb);step(short_ref,short_got,600+t+rb,rb,"GD2-original-compatible-token-shape",false);step(ref,got,650+t+rb,0,"compatible-GD1-back-to-GD2");}if(t>=64)short_got.inspect(false);}
        // Imported GD2 has unrelated generation. Every public write path must
        // request the one-pass GD1 compatibility writer, including graph replay.
        seed(ref,got,700);const auto saved_ref=rs.get(),saved_got=gs.get();State src(candidate);src.put(saved_got);
        for(int path=0;path<6;++path){rs.put(saved_ref);switch(path){
            case 0:gs.put(saved_got);break;
            case 1:ggml_backend_tensor_set_async(candidate.b,gs.raw,saved_got.data(),0,saved_got.size()*4);ggml_backend_synchronize(candidate.b);break;
            case 2:ggml_backend_tensor_set_2d(gs.raw,saved_got.data(),0,4,saved_got.size(),4,4);break;
            case 3:ggml_backend_tensor_set_2d_async(candidate.b,gs.raw,saved_got.data(),0,4,saved_got.size(),4,4);ggml_backend_synchronize(candidate.b);break;
            case 4:ggml_backend_tensor_copy(src.raw,gs.raw);break;
            case 5:ggml_backend_tensor_copy_async(control.b,candidate.b,src.raw,gs.raw);ggml_backend_synchronize(candidate.b);break;
        }for(auto c:got.counts())need(c==REHYDRATE,"public import hook rehydrates every head");step(ref,got,710+path,8,"public-import-rehydrate");step(ref,got,720+path,0,"public-import-next-generation");}
        // Full eager imports and byte-wise memset clear marker-bearing row0.
        const auto init=State::initial(800);rs.put(init);gs.put(init);step(ref,got,801,0,"full-eager-import");
        for(int rb:{2,4,8,10,11,15}){rs.put(init);gs.put(init);Graph eager_ref(control,in,rs,16,mode,false,false,false,true);step(eager_ref,got,810+rb,rb,"aliased-eager-import-safe-fallback",true,true);}
        seed(ref,got,900);ggml_backend_tensor_memset(rs.raw,0,G*4,ROW*4);ggml_backend_tensor_memset(gs.raw,0,G*4,ROW*4);step(ref,got,901,0,"public-memset-cleared-row");
        // Whole-buffer and sparse-range reset APIs, with guards repaired outside
        // the registered cache interval. Both must invalidate captured counters.
        auto repair=[&](State& state){std::vector<uint32_t> guard(G,CANARY);ggml_backend_tensor_set(state.raw,guard.data(),0,G*4);ggml_backend_tensor_set(state.raw,guard.data(),(G+CACHE)*4,G*4);};
        ggml_backend_buffer_clear(rs.buffer,0);ggml_backend_buffer_clear(gs.buffer,0);repair(rs);repair(gs);
        step(ref,got,910,0,"whole-buffer-clear-rehydrate");
        auto clear=(bool (*)(ggml_backend_buffer_t,const size_t*,int))ggml_backend_reg_get_proc_address(ggml_backend_reg_by_name("CUDA"),"ggml_backend_cuda_clear_ranges");need(clear,"sparse-clear API");
        const size_t ranges[4]={G*4,ROW*4,0,1};need(clear(rs.buffer,ranges,1)&&clear(gs.buffer,ranges,1),"sparse clear accepted");ggml_backend_synchronize(control.b);ggml_backend_synchronize(candidate.b);
        step(ref,got,911,0,"sparse-range-clear-rehydrate");

        // Zero graph captures reset nodes on both sides and uses original GD1.
        for(int t:{16,512}){seed(ref,got,950+t);Graph zr(control,in,rs,t,mode,false,true),zg(candidate,in,gs,t,mode,true,true);for(int i=0;i<3;++i)step(zr,zg,960+t+i,0,"deferred-zero-graph",false,false,true);zg.inspect(false);Clone zclone(zg.captured());in.fill(990+t,mode);in.row(0);auto old=before(zg);zr.run();zclone.run(candidate);compare(zr,zg,old,"deferred-zero-clone",false,false,true);step(ref,got,1001+t,0,"zero-to-verify-rehydrate");}
        // An extra reader of an internal slot declines the new writer. Start
        // from full eager state; arbitrary external reads of private lazy bytes
        // are not a supported logical-state API in either representation.
        rs.put(init);gs.put(init);Graph er(control,in,rs,16,mode,false,false,true),eg(candidate,in,gs,16,mode,true,false,true);for(int i=0;i<3;++i){step(er,eg,1100+i,0,"extra-early-cache-reader-fallback",false);equal("early-reader-output",read(er.exposed),read(eg.exposed));}eg.inspect(false);
    }
}
}
int main(){unsetenv("GGML_CUDA_GDN_DEDUP");need(ggml_cuda_gdn_dedup_enabled(),"default-on flag");setenv("GGML_CUDA_GDN_DEDUP","0",1);need(!ggml_cuda_gdn_dedup_enabled(),"explicit opt-out");setenv("GGML_CUDA_PDL","1",1);setenv("GGML_CUDA_GDN_L8","1",1);setenv("GGML_CUDA_GDN_LAZY_ROLLBACK","1",1);setenv("GGML_CUDA_GDN_GATHER_FUSION","1",1);unsetenv("GGML_CUDA_DISABLE_GRAPHS");writer_lifecycle();main_cases();std::printf("GD2L production route PASS checks=%d all_cache_words=1 import_hooks=1 same_context=1 cross_context=1 pdl=1 graph_clone=1 deferred_zero=1 early_reader=1\n",checks);}
