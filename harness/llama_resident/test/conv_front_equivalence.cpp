// Exact check of the CUDA conv front-end fusion (ggml_cuda_try_conv_front, ggml_cuda_try_conv_gather_skip): the
// subgraph build_conv_state and qwen35 build -- conv_states = GET_ROWS of a conv-state cache, conv_input =
// CONCAT(conv_states, TRANSPOSE(qkv)), the K conv-state snapshot CPYs into the cache, SSM_CONV, SILU, and the
// L2_NORM of the leading 128-channel heads (q, k) beside a view of the rest (v) -- run once fused and once unfused
// (conv_input marked as a graph output, which the fusion refuses). The cache, the L2_NORM output and the v channels
// (or the whole SILU output without the L2_NORM) must match bit for bit. The short-batch L2 fold is disabled by
// GGML_CUDA_DOD_CONVL2=0. --split-l2-reference keeps conv_front in the reference, but inserts a value-exact CONT
// before L2_NORM so the latter cannot be folded. --replays N reuses each graph with new inputs and cache row ids.
// --reverse-snapshots changes the copy-node order, checking that chunk ownership does not assume sorted columns.
// --tokens T / --channels C / --cache-mode NAME bound a production-shape equivalence or untimed ncu probe.
// Report max absolute/relative error and an atol/rtol check in addition to the stronger bitwise assertion.
// Allocations: "gather" gathers the conv states from a separate cache, and the allocator puts the SILU output where
// the gathered states and qkv were (shifted alias); "slot0" gathers them from the snapshot cache row that snapshot
// slot 0 overwrites ("slot0z": after build_rs's in-place zeroing SCALE of that row); "exact" takes them from an input view (no GET_ROWS), and the output lands exactly on qkv; "keep"
// keeps qkv alive so that the output cannot overlap it. The fused run must leave conv_input's memory untouched (it is
// never materialized), and with the L2_NORM it must not write the SILU output's q/k channels.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
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
           const std::vector<float> & w, const std::vector<float> & cache0, const bool reference,
           const bool split_l2_reference, const int replays, const bool reverse_snapshots) {
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
    if (reference && !split_l2_reference) {
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
    for (int64_t i = 1; i <= kK; ++i) {
        const int64_t t = reverse_snapshots ? kK + 1 - i : i;
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
    ggml_tensor * l2_copy = nullptr;
    if (cs.l2) {
        ggml_tensor * qk_view = ggml_view_4d(ctx, y, 128, QK / 128, T, 1, 128 * sizeof(float), y->nb[1], y->nb[1] * T, 0);
        if (reference && split_l2_reference) {
            // Retain the conv front-end, including its state/snapshot/alias logic; make only L2 separate.
            l2_copy = ggml_cont(ctx, qk_view);
            ggml_set_output(l2_copy);
        }
        l2 = ggml_l2_norm(ctx, l2_copy ? l2_copy : qk_view, kEps);
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
    Result r;
    r.cat_untouched = true;
    for (int rep = 0; rep < replays; ++rep) {
        // Graph capture/replay must reload the current inputs and the current gathered cache row. Do not carry
        // results from one trial into the next: exercise the existing zero/snapshot logic anew each time.
        std::vector<float> x_step = x;
        if (rep % 2) {
            for (float & f : x_step) f = -f;
        }
        const int32_t step_row = cs.mode == SLOT0 || cs.mode == SLOT0Z ?
                                    (rep % 3 == 0 ? 0 : rep % 3 == 1 ? kK - 1 : kK / 2) : row;
        ggml_backend_tensor_set(t_x, x_step.data(), 0, x_step.size() * sizeof(float));
        ggml_backend_tensor_set(t_states, states.data(), 0, states.size() * sizeof(float));
        ggml_backend_tensor_set(t_cache, cache0.data(), 0, cache0.size() * sizeof(float));
        ggml_backend_tensor_set(t_ids, &step_row, 0, sizeof(step_row));
        ggml_backend_tensor_set(cat, sentinel.data(), 0, cat_n * sizeof(uint32_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            exit(2);
        }

        if (cs.l2) {
            const size_t off = r.out.size(), qk_off = r.y_qk.size();
            r.out.resize(off + (size_t) (ggml_nelements(l2) + ggml_nelements(v)));
            ggml_backend_tensor_get(l2, r.out.data() + off, 0, ggml_nbytes(l2));
            ggml_backend_tensor_get(v, r.out.data() + off + ggml_nelements(l2), 0, ggml_nbytes(v));
            r.y_qk.resize(qk_off + (size_t) (QK * T));
            for (int64_t t = 0; t < T; ++t) {
                ggml_backend_tensor_get(y, r.y_qk.data() + qk_off + t * QK, t * y->nb[1], QK * sizeof(float));
            }
        } else {
            const size_t off = r.out.size();
            r.out.resize(off + (size_t) ggml_nelements(y));
            ggml_backend_tensor_get(y, r.out.data() + off, 0, ggml_nbytes(y));
        }
        const size_t cache_off = r.cache.size();
        r.cache.resize(cache_off + cache0.size());
        ggml_backend_tensor_get(t_cache, r.cache.data() + cache_off, 0, cache0.size() * sizeof(float));
        // conv_input's memory must still hold the sentinel, except where a tensor allocated after it (once conv_input
        // is dead) was written
        std::vector<uint32_t> cat_bits(cat_n);
        ggml_backend_tensor_get(cat, cat_bits.data(), 0, cat_n * sizeof(uint32_t));
        const char * cat_beg = (const char *) cat->data;
        size_t       checked = 0;
        for (size_t i = 0; i < cat_n; ++i) {
            const char * a = cat_beg + i * sizeof(uint32_t);
            bool         reused = false;
            for (const ggml_tensor * t : { l2, v, l2_copy }) {
                reused = reused || (t && a >= (const char *) t->data && a < (const char *) t->data + ggml_nbytes(t));
            }
            if (!reused) {
                ++checked;
                r.cat_untouched = r.cat_untouched && cat_bits[i] == kSentinel;
            }
        }
        r.cat_untouched = r.cat_untouched && checked > 0;
        if (!reference && rep == 0 && getenv("CONV_FRONT_TEST_VERBOSE")) {
            printf("  offsets vs qkv: states %+lld cat %+lld y %+lld l2 %+lld v %+lld (cat words checked %zu)\n",
                   (long long) ((char *) cst->data - (char *) qkv->data), (long long) ((char *) cat->data - (char *) qkv->data),
                   (long long) ((char *) y->data - (char *) qkv->data),
                   (long long) (l2 ? (char *) l2->data - (char *) qkv->data : 0),
                   (long long) (v ? (char *) v->data - (char *) qkv->data : 0), checked);
        }
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

struct Errors {
    double abs = 0.0, rel = 0.0;
    size_t outside_tolerance = 0;
};

Errors errors(const std::vector<float> & ref, const std::vector<float> & cand) {
    Errors e;
    if (ref.size() != cand.size()) { e.outside_tolerance = 1; return e; }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(cand[i])) {
            e.abs = e.rel = HUGE_VAL;
            ++e.outside_tolerance;
            continue;
        }
        const double d = std::abs(double(ref[i]) - double(cand[i]));
        e.abs = std::max(e.abs, d);
        e.rel = std::max(e.rel, d / std::max(1e-30, std::abs(double(ref[i]))));
        e.outside_tolerance += d > 1e-7 + 1e-6 * std::abs(double(ref[i]));
    }
    return e;
}

}  // namespace

int main(int argc, char ** argv) {
    bool split_l2_reference = false, short_only = false, reverse_snapshots = false;
    int replays = 1;
    int filter_t = 0, filter_c = 0, filter_mode = -1;
    const char * mode_args[] = { "gather", "slot0", "slot0z", "exact", "keep" };
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--split-l2-reference")) split_l2_reference = true;
        else if (!std::strcmp(argv[i], "--short-only")) short_only = true;
        else if (!std::strcmp(argv[i], "--reverse-snapshots")) reverse_snapshots = true;
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) filter_t = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--channels") && i + 1 < argc) filter_c = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--cache-mode") && i + 1 < argc) {
            const char * value = argv[++i];
            for (int m = 0; m < 5; ++m) if (!std::strcmp(value, mode_args[m])) filter_mode = m;
            if (filter_mode < 0) { fprintf(stderr, "unknown cache mode: %s\n", value); return 2; }
        }
        else if (!std::strcmp(argv[i], "--replays") && i + 1 < argc) replays = std::atoi(argv[++i]);
        else { fprintf(stderr, "unknown/incomplete option: %s\n", argv[i]); return 2; }
    }
    if (replays < 1 || replays > 4) { fprintf(stderr, "--replays must be 1..4\n"); return 2; }
    const auto enabled = [](const char * name) { const char * v = getenv(name); return !v || std::atoi(v); };
    const bool short_l2 = enabled("GGML_CUDA_DOD_CONVL2");
    const bool any_l2 = enabled("GGML_CUDA_CONV_FRONT_L2");
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) {
        fprintf(stderr, "no CUDA backend\n");
        return 2;
    }
    std::vector<Case> cases;
    for (const int64_t C : { 8192, 256 }) {
        if (filter_c && C != filter_c) continue;
        for (const int64_t T : { 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 512 }) {
            if (filter_t && T != filter_t) continue;
            if (short_only && T >= 64) continue;
            for (const Mode mode : { GATHER, SLOT0, SLOT0Z, EXACT, KEEP }) {
                if (filter_mode >= 0 && mode != filter_mode) continue;
                for (const bool l2 : { false, true }) {
                    if (split_l2_reference && !l2) continue;
                    cases.push_back({ C, T, mode, l2 });
                }
            }
        }
    }
    if (cases.empty()) { fprintf(stderr, "no cases match the filters\n"); ggml_backend_free(backend); return 2; }
    std::mt19937                    rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int                             failures = 0, qk_storage_diff = 0;
    double                          max_abs = 0.0, max_rel = 0.0;
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
        // Tiny conv weights put a whole head under the L2 epsilon floor even with a nonzero history window.
        for (int c = 0; c < 128; ++c) {
            for (int j = 0; j < kConv; ++j) w[c * kConv + j] *= 1e-10f;
        }
        // A truly zero second head (when present), including all history contributions.
        if (qk_channels(cs.C) >= 256) {
            std::fill(w.begin() + 128 * kConv, w.begin() + 256 * kConv, 0.0f);
        }
        const Result ref  = run(backend, cs, x, states, w, cache0, true, split_l2_reference, replays, reverse_snapshots);
        const Result cand = run(backend, cs, x, states, w, cache0, false, split_l2_reference, replays, reverse_snapshots);
        const Errors oe = errors(ref.out, cand.out), ce = errors(ref.cache, cand.cache);
        max_abs = std::max({max_abs, oe.abs, ce.abs});
        max_rel = std::max({max_rel, oe.rel, ce.rel});
        // With L2 folded, conv_front leaves the SILU output's q/k channels unwritten.
        const bool qk_skipped = cs.l2 && !same_bits(ref.y_qk, cand.y_qk);
        const bool expect_l2 = any_l2 && (cs.T >= 64 || short_l2);
        // The split reference has a different allocation because of CONT. Its dead SILU storage may be reused;
        // use the original, allocation-matched reference for the sentinel/route assertion, not the split copy.
        const bool ok = same_bits(ref.out, cand.out) && same_bits(ref.cache, cand.cache) && cand.cat_untouched &&
                        (split_l2_reference || !ref.cat_untouched) &&
                        (split_l2_reference || !cs.l2 || qk_skipped == expect_l2) &&
                        oe.outside_tolerance == 0 && ce.outside_tolerance == 0;
        qk_storage_diff += qk_skipped ? 1 : 0;
        printf("C=%-5lld T=%-4lld %s l2=%d fused=%d qk_storage_diff=%d out=%s cache=%s max_abs=%.3g max_rel=%.3g %s\n", (long long) cs.C,
               (long long) cs.T, mode_name(cs.mode), (int) cs.l2, (int) cand.cat_untouched, (int) qk_skipped,
               same_bits(ref.out, cand.out) ? "same" : "DIFF", same_bits(ref.cache, cand.cache) ? "same" : "DIFF",
               std::max(oe.abs, ce.abs), std::max(oe.rel, ce.rel),
               ok ? "ok" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(backend);
    printf("conv_front_equivalence %s (%d/%zu failed, %d q/k storage differences, route_checked=%d)\n", failures ? "FAILED" : "passed",
           failures, cases.size(), qk_storage_diff, (int) !split_l2_reference);
    printf("max_abs=%.9g max_rel=%.9g atol=1e-7 rtol=1e-6 exact_required=1 split_l2_reference=%d replays=%d\n",
           max_abs, max_rel, (int) split_l2_reference, replays);
    return failures ? 1 : 0;
}
