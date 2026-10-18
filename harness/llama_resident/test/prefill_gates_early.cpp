// Identical gate matmuls in two topological orders. The graph allocator must
// retain early gate outputs across scratch-heavy intervening work. Repeated
// A/B/B/A inputs exercise the backend's ordinary CUDA graph capture/replay.
#include "llama-prefill-gates.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef PREFILL_GATES_TEST_CUDA
#include "ggml-backend-impl.h"
#include "common.cuh"
#endif

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void require(bool ok,const char * message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
static bool overlaps(const ggml_tensor * a,const ggml_tensor * b) {
    const uintptr_t p=(uintptr_t)a->data,q=(uintptr_t)b->data;
    return p<q+ggml_nbytes(b) && q<p+ggml_nbytes(a);
}
struct graph_case {
    ggml_context * ctx=nullptr;
    ggml_cgraph * graph=nullptr;
    ggml_gallocr_t alloc=nullptr;
    ggml_tensor * output=nullptr,*beta=nullptr,*alpha=nullptr;
    bool scheduled=false;
    graph_case(ggml_backend_t backend,ggml_tensor * input,ggml_tensor * wb,ggml_tensor * wa,
               ggml_tensor * norm_weight,bool early) {
        ctx=ggml_init({ggml_tensor_overhead()*128+ggml_graph_overhead(),nullptr,true});
        require(ctx!=nullptr,"graph context");
        graph=ggml_new_graph(ctx);
        ggml_tensor * norm=ggml_mul(ctx,ggml_rms_norm(ctx,input,1e-6f),norm_weight);
        ggml_build_forward_expand(graph,norm); // Same layer boundary as qwen35.
        std::vector<ggml_tensor *> scratch;
        scratch.push_back(ggml_concat(ctx,norm,norm,0));
        for (int i=0;i<4;++i) scratch.push_back(ggml_scale(ctx,scratch.back(),1.0f+0.125f*i));
        beta=ggml_mul_mat(ctx,wb,norm); alpha=ggml_mul_mat(ctx,wa,norm);
        const int64_t rows=input->ne[1];
        const int before=ggml_graph_n_nodes(graph);
        require(!llm_schedule_prefill_gates(graph,norm,beta,alpha,rows,2,true),"batch guard");
        require(!llm_schedule_prefill_gates(graph,norm,beta,alpha,rows,1,false),"default-off guard");
        ggml_tensor * foreign_alpha=ggml_mul_mat(ctx,wa,input);
        require(!llm_schedule_prefill_gates(graph,norm,beta,foreign_alpha,rows,1,true),"same-input guard");
        require(!llm_schedule_prefill_gates(graph,norm,beta,ggml_scale(ctx,alpha,1.0f),rows,1,true),
                "unmodified-matmul guard");
        require(ggml_graph_n_nodes(graph)==before,"fallback leaves graph untouched");
        scheduled=llm_schedule_prefill_gates(graph,norm,beta,alpha,rows,1,early);
        require(scheduled==(early && rows>16),"eligible prefill only");
        // Expose both projections as distinct output columns; a cancellation
        // in alpha+beta cannot conceal a bit mismatch in either GEMM.
        ggml_tensor * gates=ggml_concat(ctx,beta,alpha,0);
        ggml_tensor * view=ggml_view_2d(ctx,scratch.back(),32,rows,scratch.back()->nb[1],0);
        output=ggml_concat(ctx,view,gates,0);
        ggml_set_output(output);
        ggml_build_forward_expand(graph,output);
        int ib=-1,ia=-1,is=-1;
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            const auto * t=ggml_graph_node(graph,i);
            if (t==beta) ib=i;
            if (t==alpha) ia=i;
            if (t==scratch.front()) is=i;
        }
        require(ib>=0 && ia>ib && is>=0,"original beta-before-alpha order");
        require(scheduled?ia<is:is<ib,"selected topological position");
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(alloc && ggml_gallocr_alloc_graph(alloc,graph),"graph allocation");
        if (scheduled) for (const auto * t:scratch)
            require(!overlaps(t,beta) && !overlaps(t,alpha),"live early output aliases intervening scratch");
    }
    ~graph_case() { ggml_gallocr_free(alloc); ggml_free(ctx); }
};
static std::vector<float> read(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor,result.data(),0,result.size()*sizeof(float));
    return result;
}
static void exact(const std::vector<float> & a,const std::vector<float> & b) {
    require(a.size()==b.size(),"output size");
    for (size_t i=0;i<a.size();++i) if (std::memcmp(&a[i],&b[i],sizeof(float))) {
        std::fprintf(stderr,"output[%zu] baseline=%a early=%a\n",i,a[i],b[i]);
        require(false,"unchanged matmul arithmetic");
    }
}
int main() {
#ifdef PREFILL_GATES_TEST_CUDA
    constexpr bool cuda=true;
    require(getenv("GGML_CUDA_DISABLE_GRAPHS")==nullptr,"CUDA graph replay enabled");
#else
    constexpr bool cuda=false;
#endif
    ggml_backend_t backend=ggml_backend_init_by_name(cuda?"CUDA0":"CPU",nullptr);
    require(backend!=nullptr,"backend available");
    if (!cuda) ggml_backend_cpu_set_n_threads(backend,2);
    std::mt19937 rng(20261018);
    std::uniform_real_distribution<float> random(-1.0f,1.0f);
    for (int rows:{cuda?512:32,16,7}) {
        ggml_context * inputs=ggml_init({ggml_tensor_overhead()*8,nullptr,true});
        ggml_tensor * x=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,4096,rows);
        ggml_tensor * wb=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,4096,32);
        ggml_tensor * wa=ggml_new_tensor_2d(inputs,GGML_TYPE_F32,4096,32);
        ggml_tensor * nw=ggml_new_tensor_1d(inputs,GGML_TYPE_F32,4096);
        ggml_backend_buffer_t buffer=ggml_backend_alloc_ctx_tensors(inputs,backend);
        require(buffer!=nullptr,"input/weight allocation");
        for (auto * tensor:{wb,wa,nw}) {
            std::vector<float> values(ggml_nelements(tensor));
            for (float & v:values) v=random(rng)*0.125f;
            ggml_backend_tensor_set(tensor,values.data(),0,values.size()*sizeof(float));
        }
        {
            graph_case baseline(backend,x,wb,wa,nw,false),early(backend,x,wb,wa,nw,true);
            std::vector<float> a(ggml_nelements(x)),b(a.size()),previous;
            for (auto * values:{&a,&b}) for (float & v:*values) v=random(rng);
            for (int replay=0;replay<8;++replay) {
                const auto & current=replay%4==1 || replay%4==2?b:a;
                ggml_backend_tensor_set(x,current.data(),0,current.size()*sizeof(float));
                require(ggml_backend_graph_compute(backend,baseline.graph)==GGML_STATUS_SUCCESS,"baseline compute");
                const auto expected=read(baseline.output);
                require(ggml_backend_graph_compute(backend,early.graph)==GGML_STATUS_SUCCESS,"early compute");
                exact(expected,read(early.output));
                exact(current,read(x));
                if (replay==1) require(std::memcmp(previous.data(),expected.data(),expected.size()*sizeof(float)),
                                      "changed input changes output");
                previous=expected;
            }
#ifdef PREFILL_GATES_TEST_CUDA
#ifdef USE_CUDA_GRAPH
            auto * context=(ggml_backend_cuda_context *)backend->context;
            for (const auto * which:{&baseline,&early}) {
                bool captured=false;
                for (const auto & entry:context->cuda_graphs) {
                    const auto & cached=*entry.second;
                    captured|=cached.instance && cached.warmup_complete && !cached.node_props.empty() &&
                              cached.node_props.back().node.data==which->output->data &&
                              cached.node_props.size()==size_t(ggml_graph_n_nodes(which->graph));
                }
                require(captured,"actual CUDA graph capture/replay for this graph");
            }
#else
            require(false,"build must enable CUDA graphs");
#endif
#endif
            std::printf("PASS backend=%s rows=%d scheduled=%d exact_gate_outputs=1 lifetime_aliases=0 changed_input_replays=8\n",
                        cuda?"CUDA":"CPU",rows,early.scheduled);
        }
        ggml_backend_buffer_free(buffer); ggml_free(inputs);
    }
    ggml_backend_free(backend);
}
