// Exact check of the CUDA conv front-end fusion (ggml_cuda_try_conv_front, ggml_cuda_try_conv_gather_skip): the
// subgraph build_conv_state and qwen35 build -- conv_states = GET_ROWS of a conv-state cache, conv_input =
// CONCAT(conv_states, TRANSPOSE(qkv)), the K conv-state snapshot CPYs into the cache, SSM_CONV, SILU, and the
// L2_NORM of the leading 128-channel heads (q, k) beside a view of the rest (v) -- run once fused and once unfused
// (conv_input marked as a graph output, which the fusion refuses). The cache, the L2_NORM output and the v channels
// (or the whole SILU output without the L2_NORM) must match bit for bit. The L2_NORM is folded from 64 tokens.
// Allocations: "gather" gathers the conv states from a separate cache, and the allocator puts the SILU output where
// the gathered states and qkv were (shifted alias); "slot0" gathers them from the snapshot cache row that snapshot
// slot 0 overwrites ("slot0z": after build_rs's in-place zeroing SCALE of that row); "exact" takes them from an input view (no GET_ROWS), and the output lands exactly on qkv; "keep"
// keeps qkv alive so that the output cannot overlap it. The fused run must leave conv_input's memory untouched (it is
// never materialized), and with the L2_NORM it must not write the SILU output's q/k channels.
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
constexpr int      kRows     = 3;   // conv-state rows of the separate cache
constexpr uint32_t kSentinel = 0x7fc0beefu;
constexpr float    kEps      = 1e-6f;

enum Mode { GATHER, SLOT0, SLOT0Z, EXACT, KEEP };

const char * mode_name(Mode m) {
    return m == GATHER ? "gather" : m == SLOT0 ? "slot0 " : m == SLOT0Z ? "slot0z" : m == EXACT ? "exact " : "keep  ";
}

struct Case {
    int64_t C;
    int64_t T;
    Mode    mode;
    bool    l2;
};

struct Result {
    std::vector<float> out;    // l2: L2_NORM output + v channels; else the SILU output
    std::vector<float> y_qk;   // l2: the SILU output's q/k channels (left unwritten by the fusion)
    std::vector<float> cache;
    bool               cat_untouched = false;
};

int64_t qk_channels(int64_t C) {
    return C >= 512 ? C / 2 : 128;
}

Result run(ggml_backend_t backend, const Case & cs, const std::vector<float> & x, const std::vector<float> & states,
           const std::vector<float> & w, const std::vector<float> & cache0, const bool reference) {
    const int64_t C = cs.C, T = cs.T, QK = qk_channels(C);

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
    const int32_t row = cs.mode == SLOT0 || cs.mode == SLOT0Z ? 0 : 1;
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_states, states.data(), 0, states.size() * sizeof(float));
    ggml_backend_tensor_set(t_ids, &row, 0, sizeof(row));
    ggml_backend_tensor_set(t_w, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(t_cache, cache0.data(), 0, cache0.size() * sizeof(float));

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph(ctx);

    ggml_tensor * qkv = ggml_scale(ctx, t_x, 1.0f);  // an allocator-owned qkv, like the projection output
    if (cs.mode == SLOT0Z) {
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, ggml_view_1d(ctx, t_cache, (kConv - 1) * C, 0), 0.0f));
    }
    ggml_tensor * cst = cs.mode == EXACT ? ggml_view_1d(ctx, t_states, (kConv - 1) * C, row * t_states->nb[1]) :
                        cs.mode == SLOT0 || cs.mode == SLOT0Z ? ggml_get_rows(ctx, t_cache, t_ids) :
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
    ggml_tensor * l2 = nullptr;
    ggml_tensor * v  = nullptr;
    if (cs.l2) {
        ggml_tensor * qk_view = ggml_view_4d(ctx, y, 128, QK / 128, T, 1, 128 * sizeof(float), y->nb[1], y->nb[1] * T, 0);
        l2 = ggml_l2_norm(ctx, qk_view, kEps);
        ggml_set_output(l2);
        ggml_build_forward_expand(gf, l2);
        ggml_tensor * v_view = ggml_view_4d(ctx, y, 128, (C - QK) / 128, T, 1, 128 * sizeof(float), y->nb[1],
                                            y->nb[1] * T, QK * sizeof(float));
        v = ggml_cont(ctx, v_view);
        ggml_set_output(v);
        ggml_build_forward_expand(gf, v);
    } else {
        ggml_set_output(y);
    }
    if (!cs.l2) {
        ggml_build_forward_expand(gf, y);
    }
    if (cs.mode == KEEP) {
        ggml_tensor * end = ggml_scale(ctx, qkv, 1.0f);  // qkv read again at the end: the output cannot overlap it
        ggml_set_output(end);
        ggml_build_forward_expand(gf, end);
    }

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
    if (cs.l2) {
        r.out.resize((size_t) (ggml_nelements(l2) + ggml_nelements(v)));
        ggml_backend_tensor_get(l2, r.out.data(), 0, ggml_nbytes(l2));
        ggml_backend_tensor_get(v, r.out.data() + ggml_nelements(l2), 0, ggml_nbytes(v));
        r.y_qk.resize((size_t) (QK * T));
        for (int64_t t = 0; t < T; ++t) {
            ggml_backend_tensor_get(y, r.y_qk.data() + t * QK, t * y->nb[1], QK * sizeof(float));
        }
    } else {
        r.out.resize((size_t) ggml_nelements(y));
        ggml_backend_tensor_get(y, r.out.data(), 0, ggml_nbytes(y));
    }
    r.cache.resize(cache0.size());
    ggml_backend_tensor_get(t_cache, r.cache.data(), 0, r.cache.size() * sizeof(float));
    // conv_input's memory must still hold the sentinel, except where a tensor allocated after it (once conv_input
    // is dead) was written
    std::vector<uint32_t> cat_bits(cat_n);
    ggml_backend_tensor_get(cat, cat_bits.data(), 0, cat_n * sizeof(uint32_t));
    const char * cat_beg = (const char *) cat->data;
    size_t       checked = 0;
    r.cat_untouched      = true;
    for (size_t i = 0; i < cat_n; ++i) {
        const char * a = cat_beg + i * sizeof(uint32_t);
        bool         reused = false;
        for (const ggml_tensor * t : { l2, v }) {
            reused = reused || (t && a >= (const char *) t->data && a < (const char *) t->data + ggml_nbytes(t));
        }
        if (!reused) {
            ++checked;
            r.cat_untouched = r.cat_untouched && cat_bits[i] == kSentinel;
        }
    }
    r.cat_untouched = r.cat_untouched && checked > 0;
    if (!reference && getenv("CONV_FRONT_TEST_VERBOSE")) {
        printf("  offsets vs qkv: states %+lld cat %+lld y %+lld l2 %+lld v %+lld (cat words checked %zu)\n",
               (long long) ((char *) cst->data - (char *) qkv->data), (long long) ((char *) cat->data - (char *) qkv->data),
               (long long) ((char *) y->data - (char *) qkv->data),
               (long long) (l2 ? (char *) l2->data - (char *) qkv->data : 0),
               (long long) (v ? (char *) v->data - (char *) qkv->data : 0), checked);
    }

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
            for (const Mode mode : { GATHER, SLOT0, SLOT0Z, EXACT, KEEP }) {
                for (const bool l2 : { false, true }) {
                    cases.push_back({ C, T, mode, l2 });
                }
            }
        }
    }
    std::mt19937                    rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int                             failures = 0, l2_fused = 0;
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
        for (int64_t t = 0; t < cs.T; t += 5) {
            for (int64_t c = 0; c < 128 && c < cs.C; ++c) {
                x[t * cs.C + c] = 0.0f;  // an all-zero head row: the l2 scale hits eps
            }
        }
        const Result ref  = run(backend, cs, x, states, w, cache0, true);
        const Result cand = run(backend, cs, x, states, w, cache0, false);
        // with the L2_NORM (folded from 64 tokens), the fusion leaves the SILU output's q/k channels unwritten
        const bool qk_skipped = cs.l2 && !same_bits(ref.y_qk, cand.y_qk);
        const bool ok = same_bits(ref.out, cand.out) && same_bits(ref.cache, cand.cache) && cand.cat_untouched &&
                        !ref.cat_untouched && (!cs.l2 || qk_skipped == (cs.T >= 64));
        l2_fused += qk_skipped ? 1 : 0;
        printf("C=%-5lld T=%-4lld %s l2=%d fused=%d l2_fused=%d out=%s cache=%s %s\n", (long long) cs.C,
               (long long) cs.T, mode_name(cs.mode), (int) cs.l2, (int) cand.cat_untouched, (int) qk_skipped,
               same_bits(ref.out, cand.out) ? "same" : "DIFF", same_bits(ref.cache, cand.cache) ? "same" : "DIFF",
               ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(backend);
    printf("conv_front_equivalence %s (%d/%zu failed, %d with the L2_NORM folded)\n", failures ? "FAILED" : "passed",
           failures, cases.size(), l2_fused);
    return failures ? 1 : 0;
}
