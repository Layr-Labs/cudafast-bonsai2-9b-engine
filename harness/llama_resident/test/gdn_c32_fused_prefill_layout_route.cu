// Numerical C32 is compared only against the same arithmetic through natural
// and packed real producers. No tolerance is substituted for cache bit checks.
#define main unused_gdn_prefill_layout_main
#include "gdn_prefill_layout_equivalence.cu"
#undef main

namespace {
void assert_c32_capture(Pipeline& p) {
#ifdef USE_CUDA_GRAPH
    const auto* node0=p.graph->nodes[0];
    uint64_t key=uint64_t(uintptr_t(node0));
    key=key*0x9E3779B97F4A7C15ull^uint64_t(p.graph->n_nodes);
    for(int i=0;i<GGML_MAX_DIMS;++i)key=key*0x9E3779B97F4A7C15ull^uint64_t(node0->ne[i]);
    const auto it=p.cuda().cuda_graphs.find(reinterpret_cast<const void*>(uintptr_t(key)));
    require(it!=p.cuda().cuda_graphs.end(),"exact C32 capture key");
    const auto& capture=*it->second;
    require(capture.graph&&capture.instance&&capture.warmup_complete,"C32 capture ready");
    size_t count=0;check(cudaGraphGetNodes(capture.graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(capture.graph,nodes.data(),&count));
    int prepares=0,carries=0,materializers=0,tails=0,old_half=0;
    for(auto node:nodes) {
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type!=cudaGraphNodeTypeKernel)continue;
        cudaKernelNodeParams params={};check(cudaGraphKernelNodeGetParams(node,&params));
        const char* name=nullptr;check(cudaFuncGetName(&name,params.func));
        const bool c32=std::strstr(name,"gdn_half_c32");
        prepares+=c32&&std::strstr(name,"prepare");
        carries+=c32&&std::strstr(name,"carry");
        materializers+=std::strstr(name,"gdn_half_initial_state")!=nullptr;
        old_half+=std::strstr(name,"prepare_packed")||std::strstr(name,"carry_packed");
        tails+=std::strstr(name,"gated_delta_net_")!=nullptr;
        if(c32)require(params.blockDim.x==32&&params.blockDim.y==8,"C32 block geometry");
    }
    const int wanted=p.s.t>=64;
    require(prepares==wanted&&carries==wanted&&materializers==0,"actual C32 prepare/carry/materializer");
    require(tails==1&&old_half==0,"original scalar tail and no C16 kernel");
#else
    require(false,"CUDA graphs required");
#endif
}
}

int main(int argc,char** argv) {
    setenv("GGML_CUDA_GDN_FUSED_INITIAL","1",1);
    const bool sanitize=argc==2&&std::strcmp(argv[1],"--sanitize")==0;
    setenv("GGML_CUDA_GDN_HALF_C32","1",1);
    unsetenv("GGML_CUDA_GDN_HALF_PREFILL");
    int checks=0;
    // T80 exercises a complete C32 prefix; other lengths exercise identity padding.
    for(int tokens:{64,80,128,512})for(bool raw:{false,true})for(bool zero:{false,true}) {
        if(sanitize&&(tokens!=64||raw||zero))continue;
        Shape s{32,tokens,K,raw,zero,false,MATCH,"C32-natural-packed"};
        auto ps=pipelines(s);
        for(int generation=0;generation<(sanitize?1:3);++generation) {
            Result reference;
            for(int mode=0;mode<2;++mode) {
                auto& p=*ps[mode];p.upload(701+generation*37);for(int warm=0;warm<3;++warm)p.run();p.assert_route();assert_c32_capture(p);
                auto result=p.fetch();
                if(mode==0)reference=std::move(result);
                else {compare(s,p.mode,generation,0,reference,result);++checks;}
            }
        }
    }
    if(!sanitize)for(int tokens:{16,512}) {
        Shape seed{32,tokens,K,true,false,false,MATCH,"C32-lazy-seed"};
        auto seeds=pipelines(seed);std::array<Result,2> saved;
        for(int mode=0;mode<2;++mode) {
            seeds[mode]->upload(911);
            // The native T16 seed must remain the original first GD1 import;
            // repeated native executions legitimately advance it to GD2.
            for(int warm=0;warm<(tokens>=64?3:1);++warm)seeds[mode]->run();
            seeds[mode]->assert_route();if(tokens>=64)assert_c32_capture(*seeds[mode]);
            saved[mode]=seeds[mode]->fetch();
        }
        Shape next{32,64,K,true,false,false,MATCH,"C32-lazy-rollback"};
        auto ps=pipelines(next);
        for(int rollback=0;rollback<=K;++rollback) {
            Result reference;
            for(int mode=0;mode<2;++mode) {
                auto& p=*ps[mode];p.upload(1301+rollback,&saved[mode],rollback);for(int warm=0;warm<3;++warm)p.run();p.assert_route();assert_c32_capture(p);
                auto result=p.fetch();
                if(mode==0)reference=std::move(result);
                else {compare(next,p.mode,rollback,rollback,reference,result);++checks;}
            }
        }
    }
    if(sanitize)std::printf("GDN_C32_PREFILL_SANITIZE_PASS checks=%d natural_packed=1 T64=1\n",checks);
    else std::printf("GDN_C32_FUSED_PREFILL_LAYOUT_PASS checks=%d actual_capture=1 natural_packed=1 raw_normal=1 zero=1 all_rb=0..16\n",checks);
}
