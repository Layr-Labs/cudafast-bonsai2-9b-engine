// The candidate is the actual production backend capture, with no kernel replacement.
// Its independent reference is the frozen b9549c7 kernel inserted into a clone
// with every pointer, launch attribute and full/PDL edge preserved.
#define main gdn_quotient_unused_dedup_main
#include "gdn_dedup_route.cu"
#undef main
#include "gdn_quotient_reference.cuh"
#include <limits>

namespace {
bool quotient_enabled=true;
struct DeviceWords {
    uint32_t* p=nullptr; size_t n;
    explicit DeviceWords(size_t n):n(n){check(cudaMalloc(&p,n*4));}
    ~DeviceWords(){cudaFree(p);}
    DeviceWords(const DeviceWords&)=delete;
    DeviceWords& operator=(const DeviceWords&)=delete;
};
struct Saved {
    DeviceWords cache{CACHE+2*G},counter{H};
    void take(Graph& g) {
        auto stream=g.backend.cuda().stream();
        check(cudaMemcpyAsync(cache.p,g.state.raw->data,cache.n*4,cudaMemcpyDeviceToDevice,stream));
        check(cudaMemcpyAsync(counter.p,g.counter(),H*4,cudaMemcpyDeviceToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    void restore(Graph& g) {
        auto stream=g.backend.cuda().stream();
        check(cudaMemcpyAsync(g.state.raw->data,cache.p,cache.n*4,cudaMemcpyDeviceToDevice,stream));
        check(cudaMemcpyAsync(g.counter(),counter.p,H*4,cudaMemcpyDeviceToDevice,stream));
        check(cudaMemsetAsync(g.gdn->data,0xff,ggml_nbytes(g.gdn),stream));
        check(cudaMemsetAsync(g.out->data,0xff,ggml_nbytes(g.out),stream));
    }
};
struct Observation {
    std::vector<uint32_t> cache,counter,out,scratch;
    explicit Observation(Graph& g):cache(g.state.get()),counter(g.counts()),out(read(g.out)),scratch(read(g.gdn)){}
    void equals(const Observation& got,const char* label) const {
        equal(label,cache,got.cache); equal(label,counter,got.counter);
        equal(label,out,got.out); equal(label,scratch,got.scratch); ++checks;
    }
};
void edge_identity(cudaGraph_t original,cudaGraph_t clone) {
    size_t n=0,m=0;
    check(cudaGraphGetEdges(original,nullptr,nullptr,nullptr,&n));
    check(cudaGraphGetEdges(clone,nullptr,nullptr,nullptr,&m)); need(n==m,"cloned edge count");
    std::vector<cudaGraphNode_t>a(n),b(n),x(n),y(n);
    std::vector<cudaGraphEdgeData>e(n),f(n);
    check(cudaGraphGetEdges(original,a.data(),b.data(),e.data(),&n));
    check(cudaGraphGetEdges(clone,x.data(),y.data(),f.data(),&m));
    unsigned pdl=0;
    for(size_t i=0;i<n;++i){cudaGraphNode_t ca,cb;
        check(cudaGraphNodeFindInClone(&ca,a[i],clone)); check(cudaGraphNodeFindInClone(&cb,b[i],clone));
        bool found=false;
        for(size_t j=0;j<m;++j)if(x[j]==ca&&y[j]==cb&&std::memcmp(&e[i],&f[j],sizeof(e[i]))==0)found=true;
        need(found,"all original full/PDL edge descriptors unchanged"); pdl+=e[i].type==cudaGraphDependencyTypeProgrammatic;
    }
    need(pdl>0,"actual backend PDL edges retained");
}
struct Arm {
    cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr; void* func=nullptr;
    Arm(Graph& owner,bool frozen_reference) {
        const auto src=owner.captured(); check(cudaGraphClone(&graph,src));
        size_t n=0; check(cudaGraphGetNodes(graph,nullptr,&n)); std::vector<cudaGraphNode_t>nodes(n);
        check(cudaGraphGetNodes(graph,nodes.data(),&n)); int changed=0;
        for(auto node:nodes){cudaGraphNodeType type; check(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams p{}; check(cudaGraphKernelNodeGetParams(node,&p));
            const char* name=nullptr; check(cudaFuncGetName(&name,p.func));
            if(!std::strstr(name,"gated_delta_net_dedup_l8v_cuda"))continue;
            need(std::strncmp(name,"_Z30gated_delta_net_dedup_l8v_cuda",std::strlen("_Z30gated_delta_net_dedup_l8v_cuda"))==0,"actual backend production symbol, no test replacement");
            ++changed;
            const char* signature=owner.mode
                ? (quotient_enabled?"ILb1ELb0ELb1ELb1E":"ILb1ELb0ELb1ELb0E")
                : (quotient_enabled?"ILb0ELb0ELb1ELb1E":"ILb0ELb0ELb1ELb0E");
            need(std::strstr(name,signature),"actual RAW/PRE/LAZY/QUOTIENT specialization");
            need(p.gridDim.x==H&&p.gridDim.y==1&&p.gridDim.z==8&&p.blockDim.x==32&&p.blockDim.y==4&&p.blockDim.z==1&&p.sharedMemBytes==0,"unchanged actual GDN geometry/static shared");
            need(*static_cast<float**>(p.kernelParams[7])==owner.state.data()&&*static_cast<float**>(p.kernelParams[9])==owner.state.data(),"actual state/current pointers");
            need(*static_cast<float**>(p.kernelParams[8])==owner.gdn->data&&*static_cast<uint32_t**>(p.kernelParams[30])==owner.counter(),"actual output/counter pointers");
            need(*static_cast<int64_t*>(p.kernelParams[10])==H&&*static_cast<int64_t*>(p.kernelParams[11])==owner.t&&*static_cast<int64_t*>(p.kernelParams[12])==1,"actual H/T/seq shape");
            need(*static_cast<int64_t*>(p.kernelParams[25])==int64_t(ROW)&&*static_cast<int*>(p.kernelParams[26])==K&&*static_cast<int32_t**>(p.kernelParams[27])==owner.inputs.ids->data&&*static_cast<int64_t*>(p.kernelParams[28])==int64_t(ROW),"actual rollback/row arguments");
            const auto production_func=p.func;
            if(frozen_reference){
                const size_t sizes[31]={8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,sizeof(uint3),sizeof(uint3),4,8,4,8,8,4,8};
                std::array<std::array<unsigned char,16>,31> values{};std::array<void*,31> args{};
                for(int j=0;j<31;++j){std::memcpy(values[j].data(),p.kernelParams[j],sizes[j]);args[j]=values[j].data();}
                p.func=owner.mode?(void*)gdn_quotient_reference::gated_delta_net_dedup_l8v_cuda<true,false,true>:(void*)gdn_quotient_reference::gated_delta_net_dedup_l8v_cuda<false,false,true>;
                need(p.extra==nullptr,"ordinary captured kernel argument ABI");p.kernelParams=args.data();
                check(cudaGraphKernelNodeSetParams(node,&p));
                cudaKernelNodeParams actual{};check(cudaGraphKernelNodeGetParams(node,&actual));
                need(actual.func==p.func,"frozen reference captured function identity");
                // Every argument remains the original captured value. The runtime
                // copies argument values when setting the node parameters.
                for(int j=0;j<31;++j)need(std::memcmp(actual.kernelParams[j],p.kernelParams[j],sizes[j])==0,"all 31 actual argument bytes preserved");
            } else {
                cudaKernelNodeParams actual{};check(cudaGraphKernelNodeGetParams(node,&actual));
                need(actual.func==production_func,"candidate keeps actual backend dispatch unmodified");
            }
            func=p.func;
        }
        need(changed==1,"one and only one GDN function in complete graph"); edge_identity(src,graph);
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Arm(){cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    void run(Graph& owner){check(cudaGraphLaunch(exec,owner.backend.cuda().stream()));}
};
void resources(void* fn,const char* arm) {
    cudaFuncAttributes a{}; check(cudaFuncGetAttributes(&a,fn)); int ctas=0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas,fn,128,0));
    std::printf("GDN_QUOTIENT_RESOURCE arm=%s regs=%d local=%zu shared=%zu active_ctas=%d\n",arm,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,ctas);
    need(a.localSizeBytes==0,"quotient experiment must remain spill-free");
}
void raw_reset(Graph& g,unsigned seed,uint32_t counter=REHYDRATE) {
    // Direct same-stream fixture restore avoids adding import work to any arm;
    // explicit counter values model the production import/wrap protocol.
    const auto initial=State::initial(seed); std::vector<uint32_t>count(H,counter);
    auto st=g.backend.cuda().stream();
    check(cudaMemcpyAsync(g.state.raw->data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,st));
    check(cudaMemcpyAsync(g.counter(),count.data(),count.size()*4,cudaMemcpyHostToDevice,st));
    check(cudaStreamSynchronize(st));
}
void seed_lazy(Graph& g,Arm& original,Saved& saved,unsigned seed,bool gd1=false) {
    raw_reset(g,seed); g.inputs.fill(seed+1,g.mode); g.inputs.row(0);
    original.run(g); if(!gd1)original.run(g);
    saved.take(g);
}
void exact_case(Graph& g,Arm& original,Arm& candidate,Saved& saved,const char* label) {
    saved.restore(g);original.run(g);check(cudaStreamSynchronize(g.backend.cuda().stream()));const Observation want(g);
    saved.restore(g);candidate.run(g);check(cudaStreamSynchronize(g.backend.cuda().stream()));want.equals(Observation(g),label);
}
void correctness(int mode,int t) {
    Backend backend;Inputs in(backend);State state(backend);Graph g(backend,in,state,t,mode,true);
    state.put(State::initial(11));in.fill(12,mode);in.row(0);for(int i=0;i<3;++i)g.run();g.inspect(true);
    Arm original(g,true),self(g,true),candidate(g,false);Saved saved;
    resources(original.func,mode?"raw-original":"pre-original");resources(candidate.func,mode?"raw-candidate":"pre-candidate");
    for(int rb=0;rb<=t;++rb){
        seed_lazy(g,original,saved,101+rb);std::vector<uint32_t>first;
        for(int phase=0;phase<4;++phase){in.fill(phase==1?4321:1234,mode);in.row(rb);
            if(phase==2){auto st=backend.cuda().stream();for(auto* x:{in.q,in.k,in.v,in.g,in.beta})check(cudaMemsetAsync(x->data,0,ggml_nbytes(x),st));check(cudaStreamSynchronize(st));}
            exact_case(g,original,candidate,saved,"all-rb A/B/zero/A complete output/cache/counter/scratch");
            const auto out=read(g.out);if(phase==0)first=out;if(phase==3)equal("changed-input replay determinism",first,out);
        }
    }
    seed_lazy(g,original,saved,701,true);in.fill(702,mode);in.row(std::min(8,t));exact_case(g,original,candidate,saved,"GD1 input-to-GD2 compatibility");
    for(int rb:{0,1,2,4,8,10,11,15,16}){raw_reset(g,801);saved.take(g);in.fill(802,mode);in.row(rb);exact_case(g,original,candidate,saved,"uniform eager/import-safe fallback");}
    seed_lazy(g,original,saved,901);auto corrupt=state.get();
    for(int h=0;h<H;++h)for(int z=0;z<8;++z)corrupt[rec(z,h)]=0;
    check(cudaMemcpyAsync(state.raw->data,corrupt.data(),corrupt.size()*4,cudaMemcpyHostToDevice,backend.cuda().stream()));saved.take(g);
    in.fill(902,mode);in.row(14);exact_case(g,original,candidate,saved,"invalid-record eager row14 fallback");
    raw_reset(g,1001,0x7ffffff8u);saved.take(g);in.fill(1002,mode);in.row(0);
    exact_case(g,original,candidate,saved,"last owned generation");saved.take(g);in.row(8);exact_case(g,original,candidate,saved,"counter wrap/rehydrate");
    // An evolving chain tests the same history allocations/generations across
    // many launches. Snapshot before each pair; leave the candidate's exact
    // result as the next launch's input.
    seed_lazy(g,original,saved,1101);
    for(int round=0;round<32;++round){saved.take(g);in.fill(1200+round,mode);in.row(round%(t+1));exact_case(g,original,candidate,saved,"32-round evolving exact cache/counter chain");}
    saved.take(g);exact_case(g,original,self,saved,"identical captured-graph self control");
    std::printf("GDN_QUOTIENT_CORRECT raw=%d tokens=%d cumulative_checks=%d all_cache_words=1 all_scratch_words=1 same_addresses=1 pdl_edges=1\n",mode,t,checks);
}
__host__ __device__ uint32_t hash32(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
__global__ void init_evict(uint32_t*p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=hash32(uint32_t(i));}
__global__ void run_evict(uint32_t*p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]+=1u;}
float median(std::vector<float> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Events {cudaEvent_t a{},b{};Events(){check(cudaEventCreate(&a));check(cudaEventCreate(&b));}~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}};
void timing() {
    Backend backend;Inputs in(backend);State state(backend);Graph g(backend,in,state,16,1,true);
    state.put(State::initial(21));in.fill(22,1);in.row(0);for(int i=0;i<3;++i)g.run();g.inspect(true);
    Arm original(g,true),self(g,true),candidate(g,false);Saved saved;Events events;
    DeviceWords evict(72*1024*1024/4);uint32_t evictions=0;
    auto st=backend.cuda().stream();init_evict<<<256,256,0,st>>>(evict.p,evict.n);check(cudaGetLastError());check(cudaStreamSynchronize(st));
    auto sample=[&](Arm& arm,bool cold){saved.restore(g);if(cold){run_evict<<<256,256,0,st>>>(evict.p,evict.n);++evictions;}else for(int i=0;i<8;++i)arm.run(g);
        const int repeats=cold?1:64;check(cudaEventRecord(events.a,st));for(int i=0;i<repeats;++i)arm.run(g);check(cudaEventRecord(events.b,st));check(cudaEventSynchronize(events.b));float ms=0;check(cudaEventElapsedTime(&ms,events.a,events.b));return 1000*ms/repeats;};
    for(int rb:{0,8,15,16}){
        seed_lazy(g,original,saved,2001+rb);in.fill(2011+rb,1);in.row(rb);
        for(bool cold:{false,true}){
            std::array<std::vector<float>,2>a,b,d;
            constexpr int QUARTETS=21;
            for(int q=0;q<QUARTETS;++q)for(int offset=0;offset<2;++offset){
                const int family=(q+offset)%2;const bool isself=family==0;
                auto& next=isself?self:candidate;float x[4];
                for(int j=0;j<4;++j){const bool use_next=(j==1||j==2)^(q&1);x[j]=sample(use_next?next:original,cold);(use_next?b[family]:a[family]).push_back(x[j]);}
                const float diff=(q&1)?(x[1]+x[2]-x[0]-x[3])*.5f:(x[0]+x[3]-x[1]-x[2])*.5f;d[family].push_back(diff);
                std::printf("GDN_QUOTIENT_RAW rb=%d cold=%d self=%d quartet=%d us=%.6f,%.6f,%.6f,%.6f saved=%.6f\n",rb,cold,isself,q,x[0],x[1],x[2],x[3],diff);
            }
            for(int family=0;family<2;++family){
                auto absd=d[family];for(auto& x:absd)x=std::fabs(x);
                std::printf("GDN_QUOTIENT_TIME rb=%d cold=%d self=%d original_us=%.6f next_us=%.6f paired_saved_us=%.6f max_abs_pair_us=%.6f positive=%zu/%d warm_steady_replays=64 cold_snapshot_replays=1\n",rb,cold,family==0,median(a[family]),median(b[family]),median(d[family]),*std::max_element(absd.begin(),absd.end()),size_t(std::count_if(d[family].begin(),d[family].end(),[](float x){return x>0;})),QUARTETS);
            }
        }
        // Paired measurements evolve state in warm repeats; compare that exact
        // repeat count too, not only the one-launch cold snapshots.
        saved.restore(g);for(int i=0;i<72;++i)original.run(g);check(cudaStreamSynchronize(st));const Observation want(g);
        saved.restore(g);for(int i=0;i<72;++i)candidate.run(g);check(cudaStreamSynchronize(st));want.equals(Observation(g),"timed warm steady-chain exactness");
    }
    std::vector<uint32_t>e(evict.n);check(cudaMemcpyAsync(e.data(),evict.p,evict.n*4,cudaMemcpyDeviceToHost,st));check(cudaStreamSynchronize(st));
    for(size_t i=0;i<e.size();++i)need(e[i]==hash32(uint32_t(i))+evictions,"every random eviction word touched once per call");
    std::printf("GDN_QUOTIENT_EVICTION PASS words=%zu calls=%u\n",evict.n,evictions);
}
}
int main(int argc,char** argv){
    bool run_timing=false;const char* selection="absent-on";
    unsetenv("GGML_CUDA_GDN_QUOTIENT_REUSE");
    for(int i=1;i<argc;++i){
        if(std::strcmp(argv[i],"--enabled")==0){setenv("GGML_CUDA_GDN_QUOTIENT_REUSE","1",1);quotient_enabled=true;selection="explicit-on";}
        else if(std::strcmp(argv[i],"--disabled")==0){setenv("GGML_CUDA_GDN_QUOTIENT_REUSE","0",1);quotient_enabled=false;selection="explicit-off";}
        else if(std::strcmp(argv[i],"--timing")==0)run_timing=true;
        else need(false,"known quotient route option");
    }
    if(std::strcmp(selection,"absent-on")==0)need(getenv("GGML_CUDA_GDN_QUOTIENT_REUSE")==nullptr,"default route genuinely has absent environment");
    std::printf("GDN_QUOTIENT_DISPATCH selection=%s expected_reuse=%d\n",selection,int(quotient_enabled));
    setenv("GGML_CUDA_PDL","1",1);setenv("GGML_CUDA_GDN_L8","1",1);setenv("GGML_CUDA_GDN_LAZY_ROLLBACK","1",1);setenv("GGML_CUDA_GDN_GATHER_FUSION","1",1);unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    for(int mode:{0,1})for(int t:{8,16})correctness(mode,t);if(run_timing)timing();
    std::printf("GDN_QUOTIENT_ROUTE PASS exact_checks=%d full_backend_graph=1 actual_production_dispatch=1 frozen_b954_reference=1\n",checks);
}
