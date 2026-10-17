// Exact check of the CUDA conv front-end fusion (ggml_cuda_try_conv_front): the subgraph build_conv_state and
// qwen35 build -- conv_input = CONCAT(conv_states, TRANSPOSE(qkv)), the K conv-state snapshot CPYs into the cache,
// SSM_CONV and SILU -- run once fused and once unfused (conv_input marked as a graph output, which the fusion
// refuses). The SILU output and the whole cache must match bit for bit. With "gather" the conv states come from a
// GET_ROWS as in the model, and the allocator puts the SILU output where the gathered states and qkv were (shifted
// alias); with "exact" they are an input, and the output lands exactly on qkv; "keep" keeps qkv alive so that the
// output cannot overlap it. The fused run must leave conv_input's memory untouched (it is never materialized), which
// shows that the fusion fired.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int      kConv     = 4;
constexpr int      kK        = 16;  // n_rs_seq + 1 snapshot slots
constexpr int      kRows     = 3;   // conv-state rows of the cache
constexpr uint32_t kSentinel = 0x7fc0beefu;

enum Mode { GATHER, EXACT, KEEP };

struct Case {
    int64_t C;
    int64_t T;
    Mode    mode;
};

struct Result {
    std::vector<float> y;
    std::vector<float> cache;
    bool               cat_untouched = false;
    bool               y_on_qkv      = false;
};

Result run(ggml_backend_t backend, const Case & cs, const std::vector<float> & x, const std::vector<float> & states,
           const std::vector<float> & w, const std::vector<float> & cache0, const int32_t row, const bool reference) {
    const int64_t C = cs.C, T = cs.T;

    ggml_init_params ip_in = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context *   ctx_in = ggml_init(ip_in);
    ggml_tensor *    t_x      = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, C, T);
    ggml_tensor *    t_states = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, (kConv - 1) * C, kRows);
    ggml_tensor *    t_ids    = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, 1);
    ggml_tensor *    t_w      = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, kConv, C);
    ggml_tensor *    t_cache  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, (kConv - 1) * C, kK + 1);
    ggml_tensor *    t_guard  = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, 2 * C * T + 4096);
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    ggml_backend_buffer_clear(buf_in, 0);
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_states, states.data(), 0, states.size() * sizeof(float));
    ggml_backend_tensor_set(t_ids, &row, 0, sizeof(row));
    ggml_backend_tensor_set(t_w, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(t_cache, cache0.data(), 0, cache0.size() * sizeof(float));

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph(ctx);

    ggml_tensor * qkv = ggml_scale(ctx, t_x, 1.0f);  // an allocator-owned qkv, like the projection output
    ggml_tensor * cst = cs.mode == EXACT ? ggml_view_1d(ctx, t_states, (kConv - 1) * C, row * t_states->nb[1]) :
                                           ggml_get_rows(ctx, t_states, t_ids);
    cst               = ggml_reshape_3d(ctx, cst, kConv - 1, C, 1);
    ggml_tensor * qt  = ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, C, T, 1));
    ggml_tensor * cat = ggml_concat(ctx, cst, qt, 0);
    if (reference) {
        ggml_set_output(cat);
    }
    if (T > 32 && T % 32 != 0) {
        // the unfused ssm_conv_long_token_f32 reads up to 31 floats past conv_input's end when T % 32 != 0: keep a
        // tensor behind it (too large for the blocks freed by then)
        ggml_build_forward_expand(gf, cat);
        ggml_tensor * guard = ggml_scale(ctx, t_guard, 1.0f);
        ggml_set_output(guard);
        ggml_build_forward_expand(gf, guard);
    }
    const size_t row_size = ggml_row_size(GGML_TYPE_F32, (kConv - 1) * C);
    for (int64_t t = 1; t <= kK; ++t) {
        const int64_t s_idx  = std::max<int64_t>(0, cat->ne[0] - cst->ne[0] - kK + t);
        const int64_t s_slot = kK - t;
        ggml_tensor * last   = ggml_view_3d(ctx, cat, kConv - 1, C, 1, cat->nb[1], cat->nb[2],
                                            ggml_row_size(cat->type, s_idx));
        ggml_tensor * upd    = ggml_view_2d(ctx, t_cache, (kConv - 1) * C, 1, t_cache->nb[1], s_slot * row_size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
    }
    ggml_tensor * y = ggml_silu(ctx, ggml_ssm_conv(ctx, cat, t_w));
    ggml_set_output(y);
    ggml_tensor * end = cs.mode == KEEP ? ggml_add(ctx, y, qkv) : y;
    ggml_build_forward_expand(gf, end);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        exit(2);
    }
    const size_t                cat_n = (size_t) ggml_nelements(cat);
    const std::vector<uint32_t> sentinel(cat_n, kSentinel);
    ggml_backend_tensor_set(cat, sentinel.data(), 0, cat_n * sizeof(uint32_t));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(2);
    }

    Result r;
    r.y_on_qkv = y->data == qkv->data;
    if (!reference) {
        printf("  offsets vs qkv: states %+lld cat %+lld y %+lld\n", (long long) ((char *) cst->data - (char *) qkv->data),
               (long long) ((char *) cat->data - (char *) qkv->data), (long long) ((char *) y->data - (char *) qkv->data));
    }
    r.y.resize((size_t) ggml_nelements(y));
    r.cache.resize(cache0.size());
    ggml_backend_tensor_get(y, r.y.data(), 0, r.y.size() * sizeof(float));
    ggml_backend_tensor_get(t_cache, r.cache.data(), 0, r.cache.size() * sizeof(float));
    std::vector<uint32_t> cat_bits(cat_n);
    ggml_backend_tensor_get(cat, cat_bits.data(), 0, cat_n * sizeof(uint32_t));
    r.cat_untouched = std::all_of(cat_bits.begin(), cat_bits.end(), [](uint32_t u) { return u == kSentinel; });

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
    return r;
}

bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main() {
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) {
        fprintf(stderr, "no CUDA backend\n");
        return 2;
    }
    std::vector<Case> cases;
    for (const int64_t C : { 8192, 256 }) {
        for (const int64_t T : { 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 512 }) {
            for (const Mode mode : { GATHER, EXACT, KEEP }) {
                cases.push_back({ C, T, mode });
            }
        }
    }
    std::mt19937                    rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int                             failures = 0;
    for (const Case & cs : cases) {
        std::vector<float> x((size_t) (cs.C * cs.T)), states((size_t) ((kConv - 1) * cs.C * kRows)),
            w((size_t) (kConv * cs.C)), cache0((size_t) ((kConv - 1) * cs.C * (kK + 1)));
        for (auto * v : { &x, &states, &w, &cache0 }) {
            for (float & f : *v) {
                f = nd(rng);
            }
        }
        for (size_t i = 0; i < x.size(); i += 97) {
            x[i] = (i / 97) % 2 ? 0.0f : -0.0f;  // signed zeros through the chain
        }
        const int32_t row = 1;
        const Result  ref = run(backend, cs, x, states, w, cache0, row, true);
        const Result  cand = run(backend, cs, x, states, w, cache0, row, false);
        const bool    ok = same_bits(ref.y, cand.y) && same_bits(ref.cache, cand.cache) && cand.cat_untouched &&
                        !ref.cat_untouched;
        printf("C=%-5lld T=%-4lld %s y_on_qkv=%d fused=%d y=%s cache=%s %s\n", (long long) cs.C, (long long) cs.T,
               cs.mode == GATHER ? "gather" : cs.mode == EXACT ? "exact " : "keep  ", (int) cand.y_on_qkv, (int) cand.cat_untouched,
               same_bits(ref.y, cand.y) ? "same" : "DIFF", same_bits(ref.cache, cand.cache) ? "same" : "DIFF",
               ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(backend);
    printf("conv_front_equivalence %s (%d/%zu failed)\n", failures ? "FAILED" : "passed", failures, cases.size());
    return failures ? 1 : 0;
}
