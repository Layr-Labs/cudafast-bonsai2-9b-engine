// Compare production conv -> prepare -> carry -> scalar-tail graphs. Copies in
// a diagnostic clone observe live intermediates before the allocator reuses them;
// --timing measures the original, uninstrumented graphs, including all work.
#define main unused_conv_c16_layout_main
#include "gdn_prefill_layout_equivalence.cu"
#undef main

namespace {
constexpr int QKC = 2*16*D;

cudaGraph_t capture(Pipeline & p) {
#ifdef USE_CUDA_GRAPH
    const auto * first = p.graph->nodes[0];
    uint64_t key = uint64_t(uintptr_t(first));
    key = key*0x9E3779B97F4A7C15ull ^ uint64_t(p.graph->n_nodes);
    for (int i=0;i<GGML_MAX_DIMS;++i) key = key*0x9E3779B97F4A7C15ull ^ uint64_t(first->ne[i]);
    const auto it = p.cuda().cuda_graphs.find(reinterpret_cast<const void *>(uintptr_t(key)));
    require(it != p.cuda().cuda_graphs.end(), "exact graph capture key");
    require(it->second->graph && it->second->instance && it->second->warmup_complete, "ready CUDA graph");
    return it->second->graph;
#else
    require(false, "CUDA graphs required");
    return nullptr;
#endif
}

void route(Pipeline & p, int expected) {
    const auto it = p.cuda().gdn_prefill_qk_layout.find(p.gdn);
    const int actual = it == p.cuda().gdn_prefill_qk_layout.end() ? 0 : it->second.second;
    if (actual != expected) {
        std::fprintf(stderr,"FAIL conv-c16 route case=%s T=%d expected=%d actual=%d\n",
                     p.s.name,p.s.t,expected,actual);
        std::exit(1);
    }
    capture(p);
}

void warm(Pipeline & p, bool enabled) {
    if (enabled) unsetenv("GGML_CUDA_CONV_C16_PACK");
    else setenv("GGML_CUDA_CONV_C16_PACK","0",1);
    for (int i=0;i<3;++i) { p.upload(97+i); p.run(); }
}

void upload(Pipeline & p, unsigned seed) {
    p.upload(seed);
    // Exercise the norm-epsilon branch inside the half prefix, including K.
    const std::array<float,D> zero = {};
    for (int t=8;t<24;++t) for (int channel : {0,16*D})
        ggml_backend_tensor_set(p.x,zero.data(),(size_t(t)*p.channels+channel)*sizeof(float),zero.size()*sizeof(float));
}

struct Probe {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    uint16_t * packed = nullptr;
    uint32_t * l2 = nullptr, * y = nullptr;
    size_t packed_bytes, l2_bytes, y_bytes;

    Probe(Pipeline & p, bool direct) {
        packed_bytes = size_t(p.s.t-16)*QKC*sizeof(uint16_t);
        l2_bytes = ggml_nbytes(p.l2);
        const auto * silu = p.l2->src[0]->view_src;
        require(silu && silu->op == GGML_OP_UNARY, "production SILU buffer");
        y_bytes = ggml_nbytes(silu);
        check(cudaMallocHost(&packed,packed_bytes));
        check(cudaMallocHost(&l2,l2_bytes));
        check(cudaMallocHost(&y,y_bytes));
        check(cudaGraphClone(&graph,capture(p)));
        size_t count = 0;
        check(cudaGraphGetNodes(graph,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count);
        check(cudaGraphGetNodes(graph,nodes.data(),&count));
        cudaGraphNode_t prepare = nullptr;
        void * source = nullptr;
        int prepares = 0, carries = 0, producers = 0, tails = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            check(cudaGraphNodeGetType(node,&type));
            if (type != cudaGraphNodeTypeKernel) continue;
            // The v1 getter refuses PDL launch attributes on CUDA 13; CMake runs this test with GGML_CUDA_PDL=0.
            cudaKernelNodeParams args = {};
            check(cudaGraphKernelNodeGetParams(node,&args));
            const char * name = nullptr;
            check(cudaFuncGetName(&name,args.func));
            producers += std::strstr(name,"ssm_conv_front_f32") != nullptr;
            carries += std::strstr(name,"carry_packed") != nullptr;
            tails += std::strstr(name,"gated_delta_net_l8_cuda") != nullptr;
            if (!std::strstr(name,"prepare_packed")) continue;
            prepare = node;
            std::memcpy(&source,args.kernelParams[2],sizeof(source));
            ++prepares;
        }
        require(producers == 1 && prepares == 1 && carries == 1 && tails == 1, "real fused C16 pipeline with scalar tail");
        require((source == p.l2->data) == direct, "producer/consumer share L2 half slabs only on direct route");
        size_t children_count = 0;
        check(cudaGraphNodeGetDependentNodes(prepare,nullptr,nullptr,&children_count));
        require(children_count != 0, "preparation has a downstream carry dependency");
        std::vector<cudaGraphNode_t> children(children_count);
        check(cudaGraphNodeGetDependentNodes(prepare,children.data(),nullptr,&children_count));
        cudaGraphNode_t copy_packed, copy_l2, copy_y;
        check(cudaGraphAddMemcpyNode1D(&copy_packed,graph,&prepare,1,packed,source,packed_bytes,cudaMemcpyDeviceToHost));
        check(cudaGraphAddMemcpyNode1D(&copy_l2,graph,&copy_packed,1,l2,p.l2->data,l2_bytes,cudaMemcpyDeviceToHost));
        check(cudaGraphAddMemcpyNode1D(&copy_y,graph,&copy_l2,1,y,silu->data,y_bytes,cudaMemcpyDeviceToHost));
        // In particular, carry cannot finish and release L2/V before these
        // copies complete. No reliance on a freed scratch buffer's contents.
        for (auto child : children) check(cudaGraphAddDependencies(graph,&copy_y,&child,nullptr,1));
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Probe() {
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
        check(cudaFreeHost(y)); check(cudaFreeHost(l2)); check(cudaFreeHost(packed));
    }
    void run(Pipeline & p) {
        check(cudaGraphLaunch(exec,p.cuda().stream()));
        ggml_backend_synchronize(p.backend);
    }
};

void packed_equal(const Pipeline & p, const Probe & ref, const Probe & got) {
    require(ref.packed_bytes == got.packed_bytes, "packed sizes");
    for (size_t i=0;i<ref.packed_bytes/2;++i) if (ref.packed[i] != got.packed[i]) {
        std::fprintf(stderr,"FAIL packed Q/K T=%d half=%zu expected=%04x got=%04x\n",
                     p.s.t,i,unsigned(ref.packed[i]),unsigned(got.packed[i]));
        std::exit(1);
    }
    // Old layout 2 is head-major, new layout 3's final16 is token-major.
    // Both retain the original lane permutation, so compare physical columns.
    for (int t=p.s.t-16;t<p.s.t;++t) for (int h=0;h<32;++h) for (int d=0;d<D;++d)
        require(ref.l2[(size_t(h)*p.s.t+t)*D+d] == got.l2[size_t(t)*QKC+h*D+d], "exact F32 final16 Q/K");
    for (int t=0;t<p.s.t;++t) for (int d=QKC;d<p.channels;++d)
        require(ref.y[size_t(t)*p.channels+d] == got.y[size_t(t)*p.channels+d], "exact F32 V");
}

void equivalence() {
    int checks = 0;
    for (int tokens : {64,128,512}) for (bool raw : {false,true}) for (int variant : {0,1,2,3}) {
        Shape s{32,tokens,K,raw,bool(variant&1),bool(variant&2),MATCH,"conv-c16"};
        Pipeline reference(s,2), candidate(s,2);
        warm(reference,false); route(reference,2);
        warm(candidate,true); route(candidate,3);
        Probe ref(reference,false), got(candidate,true);
        for (int generation=0;generation<3;++generation) {
            const unsigned seed = 701+37*generation;
            upload(reference,seed); ref.run(reference);
            upload(candidate,seed); got.run(candidate);
            packed_equal(candidate,ref,got);
            const auto before = candidate.fetch();
            const auto control = reference.fetch();
            equal("exact conv snapshots",control.conv,before.conv,true);
            for (uint32_t word : before.out) require((word & 0x7f800000u) != 0x7f800000u, "finite C16 output");
            upload(candidate,seed); got.run(candidate);
            compare(s,3,generation,0,before,candidate.fetch());
            ++checks;
        }
    }
    // Each negative case must retain the preexisting layout and dispatcher.
    for (auto fallback : {L2_OUTPUT,Q_OUTPUT,EXTRA_Q_USER,NO_CONV_FUSION}) {
        Pipeline p({32,512,K,true,false,false,fallback,"exposed-fallback"},2);
        warm(p,true); route(p,0);
    }
    for (Shape s : {Shape{4,512,K,true,false,false,MATCH,"head-fallback"},
                    Shape{32,65,K,true,false,false,MATCH,"length-fallback"},
                    Shape{32,128,8,true,false,false,MATCH,"K8-fallback"},
                    Shape{32,512,1,true,false,false,MATCH,"K1-fallback"},
                    Shape{32,16,K,true,false,false,MATCH,"decode-fallback"}}) {
        Pipeline p(s,2); warm(p,true); p.assert_route();
    }
    {
        Pipeline p({32,512,K,true,false,false,MATCH,"cache-fallback"},2);
        ggml_set_output(p.gdn); // disables the fused cache scatter, leaving a valid F32 layout-2 consumer
        warm(p,true); route(p,2);
    }
    std::printf("CONV_C16_PASS checks=%d packed_QK_bytes_exact=1 F32_tail_exact=1 V32_exact=1 changed_replays=3 alias_zero=1\n",checks);
}

void timing() {
    Shape s{32,512,K,true,true,true,MATCH,"conv-c16-timing"};
    Pipeline reference(s,2), candidate(s,2);
    warm(reference,false); route(reference,2);
    warm(candidate,true); route(candidate,3);
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::array<std::vector<float>,2> times;
    for (int round=0;round<7;++round) for (int step=0;step<2;++step) {
        const int which = (round+step)%2;
        auto & p = which ? candidate : reference;
        setenv("GGML_CUDA_CONV_C16_PACK",which?"1":"0",1);
        p.upload(1991);
        check(cudaEventRecord(begin,p.cuda().stream()));
        for (int iteration=0;iteration<20;++iteration) p.run(false);
        check(cudaEventRecord(end,p.cuda().stream())); check(cudaEventSynchronize(end));
        float ms = 0; check(cudaEventElapsedTime(&ms,begin,end));
        times[which].push_back(ms*1000/20);
    }
    for (auto & t : times) std::sort(t.begin(),t.end());
    std::printf("CONV_C16_TIMING T=512 V=F32 reference_us=%.3f candidate_us=%.3f saved_us=%.3f bar_us=40\n",
                times[0][3],times[1][3],times[0][3]-times[1][3]);
    check(cudaEventDestroy(end)); check(cudaEventDestroy(begin));
}
} // namespace

int main(int argc,char ** argv) {
    if (argc == 2 && std::strcmp(argv[1],"--timing") == 0) { timing(); return 0; }
    if (argc == 2) {
        int expected = 2;
        if (std::strcmp(argv[1],"--disabled") == 0) setenv("GGML_CUDA_CONV_C16_PACK","0",1);
        else if (std::strcmp(argv[1],"--half-off") == 0) setenv("GGML_CUDA_GDN_HALF_PREFILL","0",1);
        else if (std::strcmp(argv[1],"--chunk-off") == 0) setenv("GGML_CUDA_GDN_CHUNKED","0",1);
        else if (std::strcmp(argv[1],"--l8-off") == 0) { setenv("GGML_CUDA_GDN_L8","0",1); expected=0; }
        else { std::fprintf(stderr,"unknown argument: %s\n",argv[1]); return 2; }
        Pipeline p({32,512,K,true,false,false,MATCH,"switch-fallback"},2);
        for (int i=0;i<3;++i) { p.upload(97+i); p.run(); }
        route(p,expected);
        std::printf("CONV_C16_FALLBACK_PASS %s\n",argv[1]); return 0;
    }
    equivalence();
}
