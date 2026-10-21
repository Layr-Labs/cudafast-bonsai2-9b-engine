// Opt-in conv-front producer + long-GDN layout gate. No transpose or CPU packing is timed.
// Every run uses the real graph recognizer, producer, GDN dispatch and cache fusions.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "common.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int D = 128, K = 16, GUARD = 64;
constexpr uint32_t SENTINEL = 0x46a98700;
void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); }
}
void require(bool ok, const char * why) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); std::exit(1); }
}
enum Fallback { MATCH, L2_OUTPUT, Q_OUTPUT, EXTRA_Q_USER, NO_CONV_FUSION };
struct Shape {
    int h = 4, t = 64, slots = K;
    bool raw = false, zero = false, alias = false;
    Fallback fallback = MATCH;
    const char * name = "normal";
};
struct Result {
    std::vector<uint32_t> out, conv, state, exposed;
};
struct Pipeline {
    Shape s;
    int mode, hk, channels, qkc;
    int64_t state_row, conv_row;
    ggml_backend_t backend;
    ggml_context * input_ctx;
    ggml_context * graph_ctx;
    ggml_backend_buffer_t input_buffer;
    ggml_gallocr_t allocator;
    ggml_cgraph * graph;
    ggml_tensor * x, * w, * g, * beta, * bias, * a, * ids;
    ggml_tensor * conv_raw, * state_raw, * conv_cache, * state_cache;
    ggml_tensor * out, * gdn, * l2, * exposed = nullptr;
    ggml_backend_cuda_context & cuda() { return *static_cast<ggml_backend_cuda_context *>(backend->context); }
    Pipeline(Shape shape, int layout) : s(shape), mode(layout), hk(s.h/2), channels(2*D*hk+D*s.h),
        qkc(2*D*hk), state_row(int64_t(D)*D*s.h), conv_row(3*channels) {
        backend = ggml_backend_cuda_init(0);
        require(backend != nullptr, "CUDA backend initialization");
        input_ctx = ggml_init({ggml_tensor_overhead()*32, nullptr, true});
        graph_ctx = ggml_init({ggml_tensor_overhead()*512+ggml_graph_overhead(), nullptr, true});
        require(input_ctx && graph_ctx, "ggml contexts");
        x = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, channels, s.t);
        w = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, 4, channels);
            g = ggml_new_tensor_4d(input_ctx, GGML_TYPE_F32, 1, s.h, s.t, 1);
            beta = ggml_new_tensor_4d(input_ctx, GGML_TYPE_F32, 1, s.h, s.t, 1);
        bias = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, s.h);
        a = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, s.h);
        ids = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, 1);
        conv_raw = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, conv_row*(K+1)+2*GUARD);
        state_raw = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, state_row*(K+1)+2*GUARD);
        input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx, backend);
        require(input_buffer != nullptr, "input allocation");
        ggml_backend_buffer_clear(input_buffer, 0);
        auto * c = graph_ctx;
        graph = ggml_new_graph(c);
        conv_cache = ggml_view_2d(c, conv_raw, conv_row, K+1, conv_row*sizeof(float), GUARD*sizeof(float));
        state_cache = ggml_view_2d(c, state_raw, state_row, K+1, state_row*sizeof(float), GUARD*sizeof(float));
        ggml_tensor * qkv = s.alias ? ggml_scale(c, x, 1.0f) : x;
        if (s.zero) ggml_build_forward_expand(graph, ggml_scale_inplace(c, ggml_view_1d(c, conv_cache, conv_row, 0), 0.0f));
        auto * cs = ggml_reshape_3d(c, ggml_get_rows(c, conv_cache, ids), 3, channels, 1);
        auto * xt = ggml_transpose(c, ggml_reshape_3d(c, qkv, channels, s.t, 1));
        auto * cat = ggml_concat(c, cs, xt, 0);
        if (s.fallback == NO_CONV_FUSION) ggml_set_output(cat);
        for (int i=1; i<=K; ++i) {
            const int t0 = std::max(0, s.t-K+i);
            auto * src = ggml_view_3d(c, cat, 3, channels, 1, cat->nb[1], cat->nb[2], t0*sizeof(float));
            auto * dst = ggml_view_2d(c, conv_cache, conv_row, 1, conv_cache->nb[1], (K-i)*conv_cache->nb[1]);
            ggml_build_forward_expand(graph, ggml_cpy(c, src, dst));
        }
        auto * y = ggml_silu(c, ggml_ssm_conv(c, cat, w));
        auto * qk = ggml_view_4d(c, y, D, 2*hk, s.t, 1, D*sizeof(float), y->nb[1], y->nb[1]*s.t, 0);
        l2 = ggml_l2_norm(c, qk, 1e-6f);
        auto * q = ggml_view_4d(c, l2, D, hk, s.t, 1, l2->nb[1], l2->nb[2], l2->nb[3], 0);
        auto * k = ggml_view_4d(c, l2, D, hk, s.t, 1, l2->nb[1], l2->nb[2], l2->nb[3], hk*l2->nb[1]);
        auto * v = ggml_view_4d(c, y, D, s.h, s.t, 1, D*sizeof(float), y->nb[1], y->nb[1]*s.t, qkc*sizeof(float));
        if (s.fallback == L2_OUTPUT) { exposed=l2; ggml_set_output(l2); }
        if (s.fallback == Q_OUTPUT) { exposed=q; ggml_set_output(q); }
        if (s.zero) ggml_build_forward_expand(graph, ggml_scale_inplace(c, ggml_view_1d(c, state_cache, state_row, 0), 0.0f));
        auto * state = ggml_reshape_4d(c, ggml_get_rows(c, state_cache, ids), D, D, s.h, 1);
        gdn = ggml_gated_delta_net(c, q, k, v, g, beta, state, s.slots);
        if (s.raw) ggml_gated_delta_net_set_raw_gates(gdn, bias, a);
        const int n_written = std::min(s.t, s.slots);
        auto * tail = ggml_view_3d(c, gdn, state_row, 1, n_written, state_row*sizeof(float), state_row*sizeof(float),
                                   size_t(D)*s.h*s.t*sizeof(float));
        auto * cache_dst = ggml_view_3d(c, state_cache, state_row, 1, n_written, state_row*sizeof(float), state_row*sizeof(float), 0);
        ggml_build_forward_expand(graph, ggml_cpy(c, tail, cache_dst));
        out = ggml_view_4d(c, gdn, D, s.h, s.t, 1, D*sizeof(float), D*s.h*sizeof(float), size_t(D)*s.h*s.t*sizeof(float), 0);
        // Marking a VIEW as output propagates OUTPUT to GDN and disables its cache scatter fusion.
        // Keep the real lazy path; this one final attention copy is included honestly in all timings.
        out = ggml_cont(c, out);
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        if (s.fallback == EXTRA_Q_USER) {
            exposed = ggml_scale(c, ggml_cont(c, q), 0.5f);
            ggml_set_output(exposed);
            ggml_build_forward_expand(graph, exposed);
        }
        if (exposed) ggml_build_forward_expand(graph, exposed);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(ggml_gallocr_alloc_graph(allocator, graph), "graph allocation");
    }
    ~Pipeline() {
        ggml_backend_synchronize(backend);
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(input_buffer);
        ggml_backend_free(backend);
        ggml_free(graph_ctx);
        ggml_free(input_ctx);
    }
    void set_mode() {
        const char * text = mode == 2 ? "2" : "0";
        setenv("GGML_CUDA_GDN_PREFILL_QK_LAYOUT", text, 1);
    }
    void upload(unsigned seed, const Result * prior = nullptr, int rb = 0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f,1.0f);
        auto fill = [&](ggml_tensor * tensor, float scale, float offset) {
            std::vector<float> data(ggml_nelements(tensor));
            for (size_t i=0;i<data.size();++i) data[i]=scale*dist(rng)+offset;
            if (tensor == x) {
                for (size_t i=0;i<data.size();i+=97) data[i]=(i/97)%2 ? 0.0f : -0.0f;
                // A full head's last 8 tokens are zero: include the norm-epsilon branch after the conv window.
                for (int t=std::max(0,s.t-8);t<s.t;++t) std::fill_n(data.begin()+size_t(t)*channels,D,0.0f);
            }
            ggml_backend_tensor_set(tensor,data.data(),0,data.size()*sizeof(float));
        };
        fill(x,0.4f,0); fill(w,0.3f,0);
            fill(g,s.raw ? 1.0f : 0.02f,s.raw ? -2.0f : -0.08f);
            fill(beta,s.raw ? 1.0f : 0.2f,s.raw ? 0.0f : 0.5f);
        fill(bias,0.1f,-0.2f); fill(a,0.05f,-0.2f);
        auto cache = [&](ggml_tensor * tensor, const std::vector<uint32_t> * from) {
            if (from) { ggml_backend_tensor_set(tensor,from->data(),0,from->size()*sizeof(uint32_t)); return; }
            std::vector<uint32_t> words(ggml_nelements(tensor),SENTINEL);
            for (size_t i=GUARD;i<words.size()-GUARD;++i) {
                float f=0.03f*dist(rng); std::memcpy(&words[i],&f,sizeof(f));
            }
            ggml_backend_tensor_set(tensor,words.data(),0,words.size()*sizeof(uint32_t));
        };
        cache(conv_raw, prior ? &prior->conv : nullptr); cache(state_raw, prior ? &prior->state : nullptr);
        ggml_backend_tensor_set(ids,&rb,0,sizeof(rb));
    }
    void run(bool sync = true) {
        set_mode();
        require(ggml_backend_graph_compute_async(backend,graph) == GGML_STATUS_SUCCESS,"graph compute");
        if (sync) ggml_backend_synchronize(backend);
    }
    void assert_route() {
        const int expected = s.fallback == MATCH && s.t >= 64 && s.slots > 1 ? mode : 0;
        const auto it = cuda().gdn_prefill_qk_layout.find(gdn);
        const int actual = it == cuda().gdn_prefill_qk_layout.end() ? 0 : it->second.second;
        if (expected != actual) {
            std::fprintf(stderr,"FAIL route case=%s H=%d T=%d request=%d expected=%d got=%d\n",s.name,s.h,s.t,mode,expected,actual);
            std::exit(1);
        }
#ifdef USE_CUDA_GRAPH
        require(cuda().any_cuda_graph_has_instance(), "test must use an instantiated CUDA graph");
#else
        require(false, "test requires production CUDA graph build");
#endif
    }
    Result fetch() {
        auto read = [&](ggml_tensor * tensor) {
            std::vector<uint32_t> words(ggml_nelements(tensor));
            // Q is strided over tokens; contiguous exposed results use a separate CPU row gather below.
            if (tensor == exposed && !ggml_is_contiguous(tensor)) {
                for (int64_t t=0;t<tensor->ne[2];++t) for (int64_t h=0;h<tensor->ne[1];++h)
                    ggml_backend_tensor_get(tensor,words.data()+(t*tensor->ne[1]+h)*D,
                                            t*tensor->nb[2]+h*tensor->nb[1],D*sizeof(float));
            } else ggml_backend_tensor_get(tensor,words.data(),0,words.size()*sizeof(uint32_t));
            return words;
        };
        Result result{read(out),read(conv_raw),read(state_raw),exposed ? read(exposed) : std::vector<uint32_t>{}};
        if (s.slots >= 10) {
            // GD1L/NOF1 constants from gated_delta_net_kernel.cuh. Assert real lazy formation,
            // so eager snapshots cannot accidentally pass a test labelled lazy/NOFINAL.
            for (int z=0;z<8;++z) for (int h=0;h<s.h;++h) {
                const size_t at=GUARD+size_t(2+z)*state_row+size_t(h)*D*D;
                const uint32_t final_flag=s.t>=8 && s.t<=16 ? 0x4e4f4631u : 0u;
                if (result.state[at]!=0x4744314cu || result.state[at+1]!=uint32_t(std::min(s.t,s.slots)) ||
                    result.state[at+2]!=final_flag) {
                    std::fprintf(stderr,"FAIL lazy formation case=%s T=%d layout=%d head=%d record=%d header=%08x/%u/%08x expected_final=%08x\n",
                        s.name,s.t,mode,h,z,result.state[at],result.state[at+1],result.state[at+2],final_flag);
                    std::exit(1);
                }
            }
        }
        return result;
    }
};

void equal(const char * label, const std::vector<uint32_t> & ref, const std::vector<uint32_t> & got, bool guard) {
    require(ref.size() == got.size(),"comparison sizes");
    for (size_t i=0;i<ref.size();++i) {
        const bool g=guard && (i<GUARD || i>=ref.size()-GUARD);
        if (ref[i]!=got[i] || (g && ref[i]!=SENTINEL)) {
            std::fprintf(stderr,"FAIL %s word=%zu ref=%08x candidate=%08x guard=%d\n",label,i,ref[i],got[i],int(g));
            std::exit(1);
        }
    }
}
void compare(const Shape & s, int mode, int replay, int rb, const Result & ref, const Result & got) {
    char label[180];
    auto part = [&](const char * what, const std::vector<uint32_t> & x, const std::vector<uint32_t> & y, bool guard) {
        std::snprintf(label,sizeof(label),"%s H=%d T=%d layout=%d replay=%d rb=%d %s",s.name,s.h,s.t,mode,replay,rb,what);
        equal(label,x,y,guard);
    };
    for (uint32_t word : ref.out) require((word & 0x7f800000u) != 0x7f800000u, "nonfinite output");
    part("output",ref.out,got.out,false); part("conv-cache",ref.conv,got.conv,true);
    part("GDN-cache",ref.state,got.state,true); part("exposed",ref.exposed,got.exposed,false);
}
std::array<std::unique_ptr<Pipeline>,2> pipelines(Shape s) {
    std::array<std::unique_ptr<Pipeline>,2> ps = {
        std::make_unique<Pipeline>(s,0),std::make_unique<Pipeline>(s,2)};
    // First backend call warms up, the second captures. Every checked generation below is a changed-input replay.
    for (auto & p : ps) {
        for (int warm=0;warm<2;++warm) { p->upload(43+warm); p->run(); }
        p->assert_route();
    }
    return ps;
}
void correctness() {
    std::vector<Shape> shapes;
    for (int t : {64,65,127,128,129,511,512,513}) shapes.push_back({4,t,K,false,false,false,MATCH,"normal"});
    shapes.push_back({32,512,K,false,false,false,MATCH,"actual"});
    shapes.push_back({4,65,K,true,false,false,MATCH,"raw"});
    shapes.push_back({4,65,K,false,true,false,MATCH,"deferred-zero"});
    shapes.push_back({4,65,K,false,false,true,MATCH,"allocator-alias"});
    for (auto fallback : {L2_OUTPUT,Q_OUTPUT,EXTRA_Q_USER,NO_CONV_FUSION})
        shapes.push_back({4,64,K,false,false,false,fallback,"fallback"});
    for (int t : {16,33}) shapes.push_back({4,t,K,false,false,false,MATCH,"short-fallback"});
    shapes.push_back({4,64,1,false,false,false,MATCH,"K1-fallback"});
    int checks=0;
    for (Shape s : shapes) {
        auto ps=pipelines(s);
        for (int replay=0;replay<3;++replay) {
            Result ref;
            for (int mode=0;mode<2;++mode) {
                auto & p=*ps[mode]; p.upload(701+replay*37); p.run(); p.assert_route();
                auto got=p.fetch();
                if (!mode) ref=std::move(got); else { compare(s,p.mode,replay,0,ref,got); ++checks; }
            }
        }
        std::printf("exact PASS case=%s H=%d T=%d raw=%d zero=%d alias=%d fallback=%d changed_replays=3\n",
                    s.name,s.h,s.t,int(s.raw),int(s.zero),int(s.alias),int(s.fallback));
        std::fflush(stdout);
    }
    for (int seed_tokens : {16,65}) {
        Shape seed_shape{4,seed_tokens,K,false,false,false,MATCH,seed_tokens == 16 ? "verify-NOFINAL-seed" : "prefill-seed"};
        auto seeds=pipelines(seed_shape);
        std::array<Result,2> saved;
        for (int mode=0;mode<2;++mode) {
            seeds[mode]->upload(911); seeds[mode]->run(); seeds[mode]->assert_route(); saved[mode]=seeds[mode]->fetch();
        }
        for (int mode=1;mode<2;++mode) { compare(seed_shape,seeds[mode]->mode,0,0,saved[0],saved[mode]); ++checks; }
        Shape next{4,64,K,false,false,false,MATCH,seed_tokens == 16 ? "NOFINAL-rollback" : "prefill-rollback"};
        auto ps=pipelines(next);
        for (int rb=0;rb<=K;++rb) {
            Result ref;
            for (int mode=0;mode<2;++mode) {
                auto & p=*ps[mode]; p.upload(1301+rb,&saved[mode],rb); p.run(); p.assert_route();
                auto got=p.fetch();
                if (!mode) ref=std::move(got); else { compare(next,p.mode,rb,rb,ref,got); ++checks; }
            }
        }
    }
    std::printf("exact PASS checks=%d (head-major layout; outputs, all conv/GDN cache words and guards; rb=0..16)\n",checks);
    std::fflush(stdout);
}

} // namespace

int main() {
    // Isolate the exact F32 address permutation from the numerical half-prefix route.
    setenv("GGML_CUDA_CONV_C16_PACK","0",1);
    correctness();
    return 0;
}
