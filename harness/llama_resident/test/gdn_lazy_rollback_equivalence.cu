// Bit-exact check of the lazy recurrent rollback in gated_delta_net_kernel.cuh against the eager
// per-token snapshots it replaces. For every rollback depth the next ubatch must produce the same
// attention output and the same final state from a lazily written cache as from an eagerly
// written one, and an eagerly written cache must stay readable with lazy enabled.
//
// nvcc -O2 -std=c++17 -arch=sm_121 -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//      -I llama.cpp/ggml/src/ggml-cuda harness/llama_resident/test/gdn_lazy_rollback_equivalence.cu \
//      -o gdn-lazy-rollback-equivalence
#include "gated_delta_net_kernel.cuh"
#ifdef GGML_CUDA_GDN_N1_TEST
#include "gated_delta_net_dedup.cuh"
#endif

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>

#ifdef GGML_CUDA_GDN_N1_TEST
// The PDL launch wrapper and ggml assertions need these tiny host services. Keep the N1
// executable standalone rather than linking another build of the backend.
int ggml_cuda_get_device() {
    int device = 0;
    const cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); std::exit(2); }
    return device;
}
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "%s in %s at %s:%d: %s\n", stmt, func, file, line, msg);
    std::exit(2);
}
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    std::abort();
}
#endif

#define CHECK(x)                                                                                  \
    do {                                                                                          \
        cudaError_t err_ = (x);                                                                   \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(2);                                                                              \
        }                                                                                         \
    } while (0)

static constexpr int S_v = 128;
static constexpr int H   = 32;   // value heads
static constexpr int HK  = 16;   // q/k heads
static constexpr int K   = 16;   // rollback slots (DFlash depth 15 + 1)
static constexpr int64_t D = (int64_t) S_v * S_v * H;  // one cache row (one sequence)

struct Inputs {
    std::vector<float> q, k, v, g, beta;
    int n_tokens;
};

static Inputs make_inputs(int n_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Inputs in;
    in.n_tokens = n_tokens;
    in.q.resize((size_t) S_v * HK * n_tokens);
    in.k.resize((size_t) S_v * HK * n_tokens);
    in.v.resize((size_t) S_v * H * n_tokens);
    in.g.resize((size_t) H * n_tokens);
    in.beta.resize((size_t) H * n_tokens);
    for (auto & x : in.q) x = nd(rng) * 0.09f;
    for (auto & x : in.k) x = nd(rng) * 0.09f;   // roughly l2-normalised rows
    for (auto & x : in.v) x = nd(rng);
    for (auto & x : in.g) x = nd(rng) - 2.0f;    // raw or log-space decay
    for (auto & x : in.beta) x = nd(rng);
    return in;
}

struct Dev {
    float * q, * k, * v, * g, * beta, * dst;
    int layout = 0;
};

static Dev upload(const Inputs & in) {
    Dev d;
    auto up = [](const std::vector<float> & h) {
        float * p;
        CHECK(cudaMalloc(&p, h.size() * sizeof(float)));
        CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        return p;
    };
    d.q = up(in.q); d.k = up(in.k); d.v = up(in.v); d.g = up(in.g); d.beta = up(in.beta);
    CHECK(cudaMalloc(&d.dst, (size_t) S_v * H * in.n_tokens * sizeof(float)));
    CHECK(cudaMemset(d.dst, 0xff, (size_t) S_v * H * in.n_tokens * sizeof(float)));
    return d;
}

static void release(Dev & d) {
    cudaFree(d.q); cudaFree(d.k); cudaFree(d.v); cudaFree(d.g); cudaFree(d.beta); cudaFree(d.dst);
}

static float * g_raw_dt_bias = nullptr;
static float * g_raw_a = nullptr;
static int *   g_ids = nullptr;

// One fused-cache keep call: read row `row` of the cache (slot row for a one-cell cache), write
// the slots. cache holds K slots of one row each, so the slot stride is one row.
template <bool RAW>
static void run(const Dev & d, int n_tokens, float * cache, int row, int lazy) {
    CHECK(cudaMemcpy(g_ids, &row, sizeof(int), cudaMemcpyHostToDevice));
    const dim3 grid(H, 1, S_v / (4 * 4));
    const dim3 block(32, 4, 1);
    const uint3 neqk1_magic = init_fastdiv_values(HK);
    const uint3 rq3_magic   = init_fastdiv_values(1);
    gated_delta_net_cuda<S_v, false, true, RAW, false><<<grid, block>>>(
        d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, /*curr_state*/ cache, d.dst, /*state*/ cache,
        H, n_tokens, 1,
        /*sq*/ S_v, (int64_t) S_v * HK, (int64_t) S_v * HK * n_tokens,
        /*sv*/ S_v, (int64_t) S_v * H, (int64_t) S_v * H * n_tokens,
        /*sb*/ 1, H, (int64_t) H * n_tokens,
        neqk1_magic, rq3_magic, 1.0f / sqrtf((float) S_v), /*slot stride*/ D, K, g_ids, /*row stride*/ D, lazy);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}

static std::vector<uint32_t> fetch(const float * p, size_t n) {
    std::vector<uint32_t> h(n);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static int failures = 0, checks = 0;

static void expect_equal(const char * what, const std::vector<uint32_t> & a, const std::vector<uint32_t> & b) {
    ++checks;
    if (a != b) {
        size_t i = 0;
        while (i < a.size() && a[i] == b[i]) ++i;
        fprintf(stderr, "  MISMATCH %s at %zu: %08x vs %08x\n", what, i, a[i], b[i]);
        ++failures;
    }
}

template <bool RAW>
static void scenario(int n_first, const std::vector<int> & next_lens, bool eager_writer_lazy_reader) {
    const size_t cache_floats = (size_t) D * K;
    float * eager, * lazyc, * e2, * l2;
    CHECK(cudaMalloc(&eager, cache_floats * sizeof(float)));
    CHECK(cudaMalloc(&lazyc, cache_floats * sizeof(float)));
    CHECK(cudaMalloc(&e2, cache_floats * sizeof(float)));
    CHECK(cudaMalloc(&l2, cache_floats * sizeof(float)));

    // initial state in slot 0, garbage elsewhere
    {
        std::mt19937 rng(77 + n_first);
        std::normal_distribution<float> nd(0.0f, 0.3f);
        std::vector<float> h(cache_floats);
        for (auto & x : h) x = nd(rng);
        CHECK(cudaMemcpy(eager, h.data(), cache_floats * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(lazyc, h.data(), cache_floats * sizeof(float), cudaMemcpyHostToDevice));
    }

    const Inputs a = make_inputs(n_first, 1000 + n_first);
    Dev da = upload(a), db = upload(a);
    run<RAW>(da, n_first, eager, 0, 0);
    run<RAW>(db, n_first, eager_writer_lazy_reader ? lazyc : lazyc, 0, eager_writer_lazy_reader ? 0 : 1);
    char what[160];
    snprintf(what, sizeof(what), "RAW=%d first=%d attn", RAW, n_first);
    expect_equal(what, fetch(da.dst, (size_t) S_v * H * n_first), fetch(db.dst, (size_t) S_v * H * n_first));
    snprintf(what, sizeof(what), "RAW=%d first=%d final state", RAW, n_first);
    expect_equal(what, fetch(eager, D), fetch(lazyc, D));
    release(da);
    release(db);

    const int max_rb = std::min(n_first, K) - 1;
    for (int n_next : next_lens) {
        for (int rb = 0; rb <= max_rb; ++rb) {
            CHECK(cudaMemcpy(e2, eager, cache_floats * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(l2, lazyc, cache_floats * sizeof(float), cudaMemcpyDeviceToDevice));
            const Inputs b = make_inputs(n_next, 5000 + 31 * rb + n_next);
            Dev d1 = upload(b), d2 = upload(b);
            run<RAW>(d1, n_next, e2, rb, 0);  // eager reader: slot rb holds the snapshot
            run<RAW>(d2, n_next, l2, rb, 1);  // lazy reader
            snprintf(what, sizeof(what), "RAW=%d first=%d rb=%d next=%d%s attn", RAW, n_first, rb, n_next,
                     eager_writer_lazy_reader ? " eager-written" : "");
            expect_equal(what, fetch(d1.dst, (size_t) S_v * H * n_next), fetch(d2.dst, (size_t) S_v * H * n_next));
            snprintf(what, sizeof(what), "RAW=%d first=%d rb=%d next=%d%s state", RAW, n_first, rb, n_next,
                     eager_writer_lazy_reader ? " eager-written" : "");
            // eager writes the newest snapshot to slot 0; lazy writes the final state to slot 0 too
            expect_equal(what, fetch(e2, D), fetch(l2, D));
            release(d1);
            release(d2);
        }
    }
    printf("%s RAW=%d first=%d next={", failures ? "...." : "ok  ", RAW, n_first);
    for (int n : next_lens) printf(" %d", n);
    printf(" }%s (checks so far %d, failures %d)\n", eager_writer_lazy_reader ? " eager-written cache" : "", checks, failures);
    cudaFree(eager); cudaFree(lazyc); cudaFree(e2); cudaFree(l2);
}

#ifndef GGML_CUDA_GDN_N1_TEST
int main() {
    CHECK(cudaMalloc(&g_ids, sizeof(int)));
    {
        std::mt19937 rng(9);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> b(H), a(H);
        for (auto & x : b) x = nd(rng);
        for (auto & x : a) x = -std::fabs(nd(rng)) - 0.1f;
        CHECK(cudaMalloc(&g_raw_dt_bias, H * sizeof(float)));
        CHECK(cudaMalloc(&g_raw_a, H * sizeof(float)));
        CHECK(cudaMemcpy(g_raw_dt_bias, b.data(), H * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(g_raw_a, a.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    }
    scenario<true>(16, { 16, 5, 1 }, false);   // DFlash15 verify block, then every rollback depth
    scenario<true>(40, { 16 }, false);         // prefill-like ubatch longer than the window
    scenario<true>(7, { 16, 3 }, false);       // ubatch shorter than the window
    scenario<false>(16, { 16 }, false);        // activated gates
    scenario<true>(16, { 16 }, true);          // eagerly written cache read with lazy enabled
    printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
#else
// Build this same standalone TU with -DGGML_CUDA_GDN_N1_TEST for N1's
// tolerance/rollback suite, or run it with --bench for CUDA-graph event timing.
// Without the define the original bit-exact lazy rollback suite is unchanged.
namespace n1 {
static int input_layout = 0; // 0 natural, 1 permuted keys, 2 permuted/head-major

static double atol = 2e-5, rtol = 2e-4;
static int bad = 0;

template <typename T> static T * alloc(size_t count) {
    T * p = nullptr;
    CHECK(cudaMalloc(&p, count*sizeof(T)));
    return p;
}
static std::vector<float> floats(const float * p, size_t n) {
    std::vector<float> h(n);
    CHECK(cudaMemcpy(h.data(), p, n*sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}
static std::vector<float> floats(const std::vector<uint32_t> & source) {
    std::vector<float> out(source.size());
    std::memcpy(out.data(), source.data(), source.size()*sizeof(float));
    return out;
}
static void close(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    double ma = 0, mr = 0;
    size_t failures = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i]-b[i]);
        ma = std::max(ma, e);
        mr = std::max(mr, e/std::max(std::fabs((double) b[i]), 1e-6));
        failures += !std::isfinite(a[i]) || !std::isfinite(b[i]) || e > atol+rtol*std::fabs(b[i]);
    }
    printf("metric=%s max_abs=%.9g max_rel_floor_1e-6=%.9g failed=%zu words=%zu\n", label, ma, mr, failures, a.size());
    if (failures) { ++bad; }
}
static void bits(const char * label, const float * a, const float * b, size_t n) {
    const auto x = fetch(a, n), y = fetch(b, n);
    if (x != y) { fprintf(stderr, "BIT FAIL %s\n", label); ++bad; }
}

// mode 0: activated beta/log gate, 1: raw, 2: activated beta/exp gate,
// 3: production raw-gate precompute followed by G_PRECOMPUTED.
static Inputs inputs(int T, int mode, int profile) {
    Inputs in = make_inputs(T, 4872+T+profile*1709);
    for (auto * data : { &in.q, &in.k }) {
        for (size_t base = 0; base < data->size(); base += 128) {
            double ss = 0;
            for (int r = 0; r < 128; ++r) { ss += (double) (*data)[base+r]*(*data)[base+r]; }
            const float norm = (float) std::sqrt(ss);
            for (int r = 0; r < 128; ++r) { (*data)[base+r] /= norm; }
        }
    }
    const bool raw = mode == 1 || mode == 3;
    for (size_t i = 0; i < in.g.size(); ++i) {
        const float f = (float) ((i*17+7)%97)/97;
        in.g[i] = raw ? -4.0f+f : -0.005f-0.08f*f;
        if (profile == 1) { in.g[i] = raw ? 2.0f+f : -2.0f-5.0f*f; }
        if (profile == 2 && (i/H)%7 == 0) { in.g[i] = raw ? 200.0f : -120.0f; }
        if (!raw) { in.beta[i] = 1.0f/(1.0f+std::exp(-in.beta[i])); }
        if (profile == 2 && i%19 == 0) { in.beta[i] = raw ? -80.0f : 0.0f; }
        if (profile == 2 && i%23 == 0) { in.beta[i] = raw ? 80.0f : 1.0f; }
    }
    return in;
}
static Dev device_inputs(int T, int mode, int profile) {
    Inputs in = inputs(T, mode, profile);
    const int layout = T >= 64 ? input_layout : 0;
    if (layout) {
        for (auto * data : { &in.q, &in.k }) {
            std::vector<float> packed(data->size());
            for (int t = 0; t < T; ++t) for (int h = 0; h < HK; ++h) for (int r = 0; r < 128; ++r) {
                const int pr = (r/32)*32+(r%8)*4+(r%32)/8;
                const size_t at = layout == 2 ? (size_t(h)*T+t)*128+pr : (size_t(t)*HK+h)*128+pr;
                packed[at] = (*data)[(size_t(t)*HK+h)*128+r];
            }
            data->swap(packed);
        }
    }
    Dev d = upload(in);
    d.layout = layout;
    if (mode == 2) { gdn_precompute_exp<<<(H*T+255)/256, 256>>>(d.g, d.g, H*T); }
    if (mode == 3) {
        gdn_precompute_raw_gates<<<(H*T+255)/256, 256>>>(d.g, d.beta, g_raw_dt_bias, g_raw_a, d.g, d.beta, H, H*T);
    }
    CHECK(cudaGetLastError());
    return d;
}
template <bool RAW, bool PRE, bool HEAD_MAJOR>
static void launch_t(const Dev & d, int T, float * cache, gdn_chunk32 * work, bool candidate, bool lazy,
                     bool zero, bool verify, cudaStream_t stream, const cudaEvent_t * phase_events, bool row_ilp,
                     bool prefetch, bool wide, bool compact, bool qstage) {
    const dim3 grid(H, 1, 8), block(32, 4);
    const auto hk = init_fastdiv_values(HK), seq = init_fastdiv_values(1);
    const float scale = 1.0f/std::sqrt(128.0f);
    const int64_t sq1 = d.layout == 2 ? int64_t(T)*128 : 128;
    const int64_t sq2 = d.layout == 2 ? 128 : int64_t(128)*HK, sq3 = int64_t(T)*128*HK;
    const int64_t sv2 = int64_t(128)*H;
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);
    if (candidate) {
        gdn_chunk32_launch<RAW, PRE, HEAD_MAJOR>(d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a,
            cache, d.dst, cache, work, H, T, sq1, sq2, sq3, 128, sv2, sv2*T, 1, H, H*T,
            hk, seq, scale, D, K, g_ids, D, lazy, zero ? cache : nullptr, 0.0f, 0.0f, stream, phase_events, row_ilp, prefetch, wide, compact, qstage);
    } else if (T <= 7) {
        ggml_cuda_kernel_launch(gated_delta_net_cuda<128, false, true, RAW, PRE>, lp,
            d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, cache, d.dst, cache,
            (int64_t) H, (int64_t) T, (int64_t) 1, sq1, sq2, sq3, (int64_t) 128,
            sv2, sv2*T, (int64_t) 1, (int64_t) H, (int64_t) H*T, hk, seq, scale, D, K, g_ids, D, (int) lazy);
    } else if (verify && T == 16) {
#define N1_VERIFY(LAZY_) \
        ggml_cuda_kernel_launch(gated_delta_net_l8v_cuda<RAW, PRE, LAZY_>, lp, \
            d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, cache, d.dst, cache, \
            (int64_t) H, (int64_t) T, (int64_t) 1, sq1, sq2, sq3, (int64_t) 128, \
            sv2, sv2*T, (int64_t) 1, (int64_t) H, (int64_t) H*T, hk, seq, scale, D, K, g_ids, D, (int) lazy)
        if (lazy) { N1_VERIFY(true); } else { N1_VERIFY(false); }
#undef N1_VERIFY
    } else {
        ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<RAW, PRE, HEAD_MAJOR>, lp,
            d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, cache, d.dst, cache,
            (int64_t) H, (int64_t) T, (int64_t) 1, sq1, sq2, sq3, (int64_t) 128,
            sv2, sv2*T, (int64_t) 1, (int64_t) H, (int64_t) H*T, hk, seq, scale, D, K, g_ids, D,
            (int) lazy, zero ? cache : nullptr, 0.0f, 0.0f);
    }
}
static void launch(const Dev & d, int T, int mode, float * cache, gdn_chunk32 * work, bool candidate,
                   bool lazy, bool zero, bool verify = false, cudaStream_t stream = nullptr,
                   const cudaEvent_t * phase_events = nullptr, int scan_variant = -1, int prefetch_variant = -1,
                   int wide_variant = -1, int compact_variant = -1, int qstage_variant = -1) {
    const bool ilp = scan_variant < 0 ? gdn_chunk32_scan_ilp_enabled() : scan_variant != 0;
    const bool pf = prefetch_variant < 0 ? gdn_chunk32_prefetch_enabled() : prefetch_variant != 0;
    const bool wide = wide_variant < 0 ? gdn_chunk32_wide_enabled() : wide_variant != 0;
    const bool compact = compact_variant < 0 ? gdn_chunk32_compact_enabled() : compact_variant != 0;
    const bool qs = qstage_variant < 0 ? gdn_chunk32_qstage_enabled() : qstage_variant != 0;
#define N1_TYPED(RAW_, PRE_, PACK_) \
    launch_t<RAW_, PRE_, PACK_>(d, T, cache, work, candidate, lazy, zero, verify, stream, phase_events, ilp, pf, wide, compact, qs)
    if (d.layout) {
        if (mode == 1) { N1_TYPED(true, false, true); }
        else if (mode >= 2) { N1_TYPED(false, true, true); }
        else { N1_TYPED(false, false, true); }
    } else {
        if (mode == 1) { N1_TYPED(true, false, false); }
        else if (mode >= 2) { N1_TYPED(false, true, false); }
        else { N1_TYPED(false, false, false); }
    }
#undef N1_TYPED
}

template <bool RAW, bool PRE>
static void dedup_verify_t(const Dev & d, float * cache, uint32_t * counter) {
    const auto hk = init_fastdiv_values(HK), seq = init_fastdiv_values(1);
    const int64_t sq2 = int64_t(128)*HK, sv2 = int64_t(128)*H;
    const ggml_cuda_kernel_launch_params lp(dim3(H,1,8), dim3(32,4), 0, nullptr);
    // Non-null selects the real GD2 writer; null tests the materialized tail.
    if (counter) {
    ggml_cuda_kernel_launch(gated_delta_net_dedup_l8v_cuda<RAW, PRE, true>, lp,
        d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, cache, d.dst, cache,
        int64_t(H), int64_t(16), int64_t(1), int64_t(128), sq2, sq2*16,
        int64_t(128), sv2, sv2*16, int64_t(1), int64_t(H), int64_t(H*16),
        hk, seq, 1.0f/std::sqrt(128.0f), D, K, g_ids, D, 1, counter);
    } else {
    ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<RAW, PRE, false, true>, lp,
        d.q, d.k, d.v, d.g, d.beta, g_raw_dt_bias, g_raw_a, cache, d.dst, cache,
        int64_t(H), int64_t(16), int64_t(1), int64_t(128), sq2, sq2*16,
        int64_t(128), sv2, sv2*16, int64_t(1), int64_t(H), int64_t(H*16),
        hk, seq, 1.0f/std::sqrt(128.0f), D, K, g_ids, D, 1, (const float *) nullptr, 0.0f, 0.0f);
    }

}
static void dedup_verify(const Dev & d, int mode, float * cache, uint32_t * counter) {
    if (mode == 1) { dedup_verify_t<true, false>(d, cache, counter); }
    else if (mode >= 2) { dedup_verify_t<false, true>(d, cache, counter); }
    else { dedup_verify_t<false, false>(d, cache, counter); }
}
static void row(int r) { CHECK(cudaMemcpy(g_ids, &r, sizeof(r), cudaMemcpyHostToDevice)); }
static void reset(float * p, bool fresh) {
    CHECK(cudaMemset(p, 0xcd, (size_t) D*K*sizeof(float))); // finite poison, including old headers
    if (!fresh) {
        std::mt19937 rng(71);
        std::normal_distribution<float> nd(0, .03f);
        std::vector<float> init(D);
        for (auto & x : init) { x = nd(rng); }
        CHECK(cudaMemcpy(p, init.data(), D*sizeof(float), cudaMemcpyHostToDevice));
    }
}
static void test(int T, int mode, int profile, bool fresh) {
    printf("case T=%d H=32 HK=16 D=128 mode=%d profile=%d fresh=%d\n", T, mode, profile, fresh);
    float * ref = alloc<float>(D*K), * eager = alloc<float>(D*K), * lazy = alloc<float>(D*K);
    float * a = alloc<float>(D*K), * b = alloc<float>(D*K);
    gdn_chunk32 * work = alloc<gdn_chunk32>(H*((T-K+31)/32));
    Dev dr = device_inputs(T, mode, profile), dc = device_inputs(T, mode, profile);
    row(0); reset(ref, fresh); reset(eager, fresh); reset(lazy, fresh);
    launch(dr, T, mode, ref, work, false, false, fresh);
    launch(dc, T, mode, eager, work, true, false, fresh);
    close("output", floats(dc.dst, (size_t) T*128*H), floats(dr.dst, (size_t) T*128*H));
    close("final_state", floats(eager, D), floats(ref, D));
    close("all_eager_snapshots", floats(eager, D*K), floats(ref, D*K));
    const auto eager_out = fetch(dc.dst, (size_t) T*128*H);
    if (gdn_chunk32_compact_enabled()) {
        reset(b, fresh);
        launch(dc, T, mode, b, work, true, false, fresh, false, nullptr, nullptr, 1, 1, 1, 1,
            gdn_chunk32_qstage_enabled() ? 0 : 1);
        bits("Q staging/direct snapshots", eager, b, D*K);
        if (eager_out != fetch(dc.dst, size_t(T)*128*H)) { ++bad; fprintf(stderr, "BIT FAIL Q staging/direct outputs\n"); }
    }
    if (T == 64 || T == 65 || T == 512) {
        // Compact changes f32 arithmetic. Preserve the old i1-i4 bit tests
        // against a separate noncompact baseline; compare compact by tolerance.
        reset(a, fresh);
        launch(dc, T, mode, a, work, true, false, fresh, false, nullptr, nullptr, -1, -1, -1, 0);
        const auto old_out = fetch(dc.dst, (size_t) T*128*H);
        close("compact/i4 snapshots", floats(eager, D*K), floats(a, D*K));
        close("compact/i4 output", floats(eager_out), floats(old_out));
        reset(b, fresh);
        launch(dc, T, mode, b, work, true, false, fresh, false, nullptr, nullptr, 1,
               gdn_chunk32_prefetch_enabled() ? 0 : 1, -1, 0);
        bits("prefetch/i2 eager snapshots", a, b, D*K);
        if (old_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL prefetch/i2 outputs\n"); }
        reset(b, fresh);
        launch(dc, T, mode, b, work, true, false, fresh, false, nullptr, nullptr,
               gdn_chunk32_scan_ilp_enabled() ? 0 : 1, -1, -1, 0);
        bits("row-ILP/serial eager snapshots", a, b, D*K);
        if (old_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL row-ILP/serial outputs\n"); }
        reset(b, fresh);
        launch(dc, T, mode, b, work, true, false, fresh, false, nullptr, nullptr, 1, 1,
               gdn_chunk32_wide_enabled() ? 0 : 1, 0);
        bits("wide/i3 eager snapshots", a, b, D*K);
        if (old_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL wide/i3 outputs\n"); }
        reset(b, fresh);
        launch(dc, T, mode, b, work, true, false, fresh, false, nullptr, nullptr, 1, 1, 1, 1);
        close("explicit compact snapshots", floats(b, D*K), floats(ref, D*K));
        close("explicit compact output", floats(dc.dst, (size_t) T*128*H), floats(dr.dst, (size_t) T*128*H));
    }
    if (T == 65 && mode == 3 && profile == 0 && !fresh) {
        // A valid f32 row with a non-vector-aligned base must use the old path.
        const size_t nq = (size_t) T*HK*128;
        float * shifted = alloc<float>(nq+1);
        CHECK(cudaMemcpy(shifted+1, dc.q, nq*sizeof(float), cudaMemcpyDeviceToDevice));
        Dev misaligned = dc; misaligned.q = shifted+1;
        reset(a, false); reset(b, false);
        launch(dc, T, mode, a, work, true, false, false, false, nullptr, nullptr, 1, 1, 1, 0);
        const auto old_out = fetch(dc.dst, (size_t) T*128*H);
        launch(misaligned, T, mode, b, work, true, false, false, false, nullptr, nullptr, 1, 1, 1, 1);
        bits("unaligned q fallback snapshots", a, b, D*K);
        if (old_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL unaligned q fallback outputs\n"); }
        cudaFree(shifted);
        const size_t nv = (size_t) T*H*128;
        shifted = alloc<float>(nv+1);
        CHECK(cudaMemcpy(shifted+1, dc.v, nv*sizeof(float), cudaMemcpyDeviceToDevice));
        misaligned = dc; misaligned.v = shifted+1;
        reset(b, false);
        launch(misaligned, T, mode, b, work, true, false, false, false, nullptr, nullptr, 1, 1, 1, 1);
        bits("unaligned v fallback snapshots", a, b, D*K);
        if (old_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL unaligned v fallback outputs\n"); }
        cudaFree(shifted);
    }
    reset(ref, fresh);
    launch(dr, T, mode, ref, work, false, true, fresh);
    launch(dc, T, mode, lazy, work, true, true, fresh);
    close("lazy_window_base", floats(lazy+D, D), floats(ref+D, D));
    bits("candidate eager/lazy final", eager, lazy, D);
    if (eager_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL eager/lazy outputs\n"); }
    if (fresh) {
        // A new fresh-cell prefill must recompute from zero even when old lazy
        // headers/checksums are valid. This also models repeated warm prefills.
        launch(dc, T, mode, lazy, work, true, true, true);
        if (eager_out != fetch(dc.dst, (size_t) T*128*H)) { ++bad; fprintf(stderr, "BIT FAIL fresh-cell reuse\n"); }
        bits("fresh-cell repeat final", eager, lazy, D);
    }
    // Use the frontier's real GD2 writer, not hand-made record bytes. Both
    // parity slots and rb0 NOFINAL/all selected rollback inputs feed N1.
    if ((T == 64 || T == 512) && profile == 0 && !fresh) {
        Dev verify = device_inputs(16, mode, 0);
        uint32_t * counter = alloc<uint32_t>(H);
        float * ea = alloc<float>(D*K), * gd2 = alloc<float>(D*K);
        for (uint32_t gen : { 0u, 1u }) {
            CHECK(cudaMemcpy(ea, eager, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(gd2, eager, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
            std::vector<uint32_t> counts(H, gen*8);
            CHECK(cudaMemcpy(counter, counts.data(), H*sizeof(uint32_t), cudaMemcpyHostToDevice));
            row(0);
            launch(verify, 16, mode, ea, work, false, false, false, true);
            dedup_verify(verify, mode, gd2, counter);
            const auto records = fetch(gd2, D*K);
            for (int h = 0; h < H; ++h) for (int z = 0; z < 8; ++z) {
                const size_t off = size_t(2+z)*D+size_t(h)*128*128;
                if (records[off] != GDN_DEDUP_MAGIC || records[off+3] != gen) {
                    ++bad; fprintf(stderr, "BIT FAIL GD2 writer generation\n");
                }
            }
            for (int rb : { 0, 1, 7, 15 }) {
                CHECK(cudaMemcpy(a, ea, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                CHECK(cudaMemcpy(b, gd2, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                row(rb);
                launch(dc, T, mode, a, work, true, false, false);
                const auto out = fetch(dc.dst, size_t(T)*128*H);
                launch(dc, T, mode, b, work, true, true, false);
                bits("GD2 -> N1 state", a, b, D);
                if (out != fetch(dc.dst, size_t(T)*128*H)) { ++bad; fprintf(stderr, "BIT FAIL GD2 -> N1 output\n"); }
            }
            // Materialized slot0 may contain the same finite bit patterns as
            // NOFINAL. The tail must consume it, not the still-valid old record.
            CHECK(cudaMemcpy(a, ea, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(b, gd2, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
            auto materialized = fetch(ea, D);
            for (int h = 0; h < H; ++h) for (int c = 0; c < 128; c += 4) {
                const size_t at = size_t(h)*128*128+c*128;
                materialized[at+0] = GDN_LAZY_SLOT0_MARK0;
                materialized[at+1] = GDN_LAZY_SLOT0_MARK1;
            }
            CHECK(cudaMemcpy(a, materialized.data(), D*sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(b, materialized.data(), D*sizeof(float), cudaMemcpyHostToDevice));
            row(0);
            launch(verify, 16, mode, a, work, false, false, false);
            const auto tail_out = fetch(verify.dst, size_t(16)*128*H);
            dedup_verify(verify, mode, b, nullptr);
            bits("materialized marker-collision tail state", a, b, D);
            if (tail_out != fetch(verify.dst, size_t(16)*128*H)) { ++bad; fprintf(stderr, "BIT FAIL materialized marker-collision tail output\n"); }
        }
        row(0);
        cudaFree(counter); cudaFree(ea); cudaFree(gd2); release(verify);
    }
    // Every written record payload and metadata field is exact. Checksums depend
    // on the rounded base and are instead validated by every rollback below.
    const auto lr = fetch(lazy, D*K), rr = fetch(ref, D*K);
    for (int z = 0; z < 8; ++z) {
        for (int h = 0; h < H; ++h) {
            const size_t off = (size_t) (2+z)*D+(size_t) h*128*128;
            for (int i = 0; i < GDN_LAZY_HDR+K*146; ++i) {
                if (i >= 3 && i < GDN_LAZY_HDR) { continue; }
                if (lr[off+i] != rr[off+i]) { ++bad; fprintf(stderr, "BIT FAIL record layout/payload\n"); break; }
            }
        }
    }
    // Exercise the real generic decode and l8v verify kernels, all rollback
    // depths, including their NOFINAL slot-0 marker. These comparisons are
    // bit-exact between eager and lazy writers of the SAME rounded boundary.
    if ((T == 64 || T == 512) && profile == 0 && !fresh) {
        for (int rb = 0; rb < K; ++rb) {
            for (int next : { 1, 16 }) {
                CHECK(cudaMemcpy(a, eager, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                CHECK(cudaMemcpy(b, lazy, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                if (next == 1 && (rb == 1 || rb == 7 || rb == 15)) {
                    Dev ca = device_inputs(T, mode, 0), cb = device_inputs(T, mode, 0);
                    row(rb);
                    launch(ca, T, mode, a, work, true, false, false);
                    launch(cb, T, mode, b, work, true, true, false);
                    bits("chunk input rollback output", ca.dst, cb.dst, (size_t) T*128*H);
                    bits("chunk input rollback final", a, b, D);
                    release(ca); release(cb);
                    CHECK(cudaMemcpy(a, eager, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                    CHECK(cudaMemcpy(b, lazy, D*K*sizeof(float), cudaMemcpyDeviceToDevice));
                }
                Dev da = device_inputs(next, mode, 0), db = device_inputs(next, mode, 0);
                row(rb);
                launch(da, next, mode, a, work, false, false, false, true);
                launch(db, next, mode, b, work, false, true, false, true);
                bits("rollback output", da.dst, db.dst, (size_t) next*128*H);
                row(0);
                if (next == 16 && (rb == 0 || rb == 15)) {
                    Dev ca = device_inputs(T, mode, 0), cb = device_inputs(T, mode, 0);
                    launch(ca, T, mode, a, work, true, false, false);
                    launch(cb, T, mode, b, work, true, true, false);
                    bits("chunk input NOFINAL output", ca.dst, cb.dst, (size_t) T*128*H);
                    bits("chunk input NOFINAL final", a, b, D);
                    release(ca); release(cb);
                } else if (next == 16) {
                    Dev ca = device_inputs(1, mode, 0), cb = device_inputs(1, mode, 0);
                    launch(ca, 1, mode, a, work, false, false, false);
                    launch(cb, 1, mode, b, work, false, true, false);
                    bits("verify NOFINAL replay output", ca.dst, cb.dst, 128*H);
                    release(ca); release(cb);
                }
                bits("rollback final", a, b, D);
                release(da); release(db);
            }
        }
    }
    release(dr); release(dc);
    cudaFree(work); cudaFree(ref); cudaFree(eager); cudaFree(lazy); cudaFree(a); cudaFree(b);
}

struct PhaseTimes {
    float prepare = std::numeric_limits<float>::quiet_NaN();
    float scan = std::numeric_limits<float>::quiet_NaN();
    float tail = std::numeric_limits<float>::quiet_NaN();
    float sum = std::numeric_limits<float>::quiet_NaN();
};
static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return .5f*(v[(v.size()-1)/2]+v[v.size()/2]);
}
static PhaseTimes phase_times(const Dev & d, int T, int iters, float * cache, gdn_chunk32 * work,
                             cudaStream_t stream) {
    PhaseTimes result;
    if (!gdn_chunk32_enabled()) { return result; } // disabled fields stay NaN, not a mislabeled full l8 tail
    cudaEvent_t events[4];
    for (auto & event : events) { CHECK(cudaEventCreate(&event)); }
    cudaGraph_t graph; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch(d, T, 3, cache, work, true, true, true, false, stream, events);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int w = 0; w < 5; ++w) { CHECK(cudaGraphLaunch(exec, stream)); }
    CHECK(cudaStreamSynchronize(stream));
    std::vector<float> samples[4];
    for (int i = 0; i < iters; ++i) {
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaStreamSynchronize(stream));
        float sum = 0;
        for (int p = 0; p < 3; ++p) {
            float ms;
            CHECK(cudaEventElapsedTime(&ms, events[p], events[p+1]));
            samples[p].push_back(ms*1000);
            sum += ms*1000;
        }
        samples[3].push_back(sum);
    }
    // Finish the graph before destroying its event nodes or reusing storage.
    CHECK(cudaStreamSynchronize(stream));
    result = { median(samples[0]), median(samples[1]), median(samples[2]), median(samples[3]) };
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
    for (auto event : events) { CHECK(cudaEventDestroy(event)); }
    return result;
}

static void profile_once(int T) {
    // No warm-up, graph, baseline or timing launches: the regex-selected set is
    // exactly prepare, scan<ROW_ILP>, l8(K-token tail), one invocation each.
    // device_inputs also launches raw gate precompute once, excluded by that regex.
    Dev d = device_inputs(T, 3, 0);
    float * cache = alloc<float>(D*K);
    auto * work = alloc<gdn_chunk32>(H*((T-K+31)/32));
    row(0); reset(cache, true);
    CHECK(cudaDeviceSynchronize());
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    printf("NCU_ONCE T=%d prefix=%d C=32 H=32 D=128 scan_ilp=%d prefetch=%d wide=%d compact=%d qstage=%d layout=%d prepare_ctas=%d scan_ctas=%d scan_threads=%d tail_ctas=256\n",
        T, T-K, gdn_chunk32_scan_ilp_enabled(), gdn_chunk32_prefetch_enabled(), gdn_chunk32_wide_enabled(), gdn_chunk32_compact_enabled(), gdn_chunk32_qstage_enabled(), n1::input_layout,
        H*((T-K+31)/32), gdn_chunk32_wide_enabled() ? 128 : 256, gdn_chunk32_wide_enabled() ? 256 : 128);
    fflush(stdout);
    launch(d, T, 3, cache, work, true, true, true, false, stream);
    CHECK(cudaStreamSynchronize(stream));
    // Host-only finite check; no extra diagnostic kernel is launched.
    for (float x : floats(d.dst, (size_t) T*H*128)) { if (!std::isfinite(x)) { ++bad; break; } }
    for (float x : floats(cache, D)) { if (!std::isfinite(x)) { ++bad; break; } }
    CHECK(cudaStreamDestroy(stream));
    cudaFree(work); cudaFree(cache); release(d);
}

static void bench(int T, int iters) {
    Dev d = device_inputs(T, 3, 0);
    float * cache = alloc<float>(D*K); // identical cache VA in all three arms
    auto * work = alloc<gdn_chunk32>(H*((T-K+31)/32));
    cudaGraph_t graph[3]; cudaGraphExec_t exec[3];
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<float> samples[3];
    row(0);
    for (int i = 0; i < 3; ++i) {
        reset(cache, true);
        CHECK(cudaDeviceSynchronize());
        // Instantiate/query PDL metadata once outside capture.
        launch(d, T, 3, cache, work, i != 0, true, true, false, stream, nullptr, -1, -1, -1, -1, i == 2);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(d, T, 3, cache, work, i != 0, true, true, false, stream, nullptr, -1, -1, -1, -1, i == 2);
        CHECK(cudaStreamEndCapture(stream, &graph[i]));
        CHECK(cudaGraphInstantiate(&exec[i], graph[i], nullptr, nullptr, 0));
        for (int w = 0; w < 5; ++w) { CHECK(cudaGraphLaunch(exec[i], stream)); }
        CHECK(cudaStreamSynchronize(stream));
        samples[i] = floats(d.dst, (size_t) T*128*H);
    }
    const int selected = gdn_chunk32_qstage_enabled() ? 2 : 1;
    close("graph_warm_output", samples[selected], samples[0]);
    if (std::memcmp(samples[1].data(), samples[2].data(), samples[1].size()*sizeof(float))) {
        ++bad; fprintf(stderr, "BIT FAIL direct/staged graph output\n");
    }
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin, end; CHECK(cudaEventCreate(&begin)); CHECK(cudaEventCreate(&end));
    std::vector<float> us[3];
    for (int round = 0; round < 12; ++round) {
        for (int leg = 0; leg < 3; ++leg) {
            const int i = (round%3+((round&1) ? 2-leg : leg))%3;
            CHECK(cudaEventRecord(begin, stream));
            for (int n = 0; n < iters; ++n) { CHECK(cudaGraphLaunch(exec[i], stream)); }
            CHECK(cudaEventRecord(end, stream)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, begin, end));
            us[i].push_back(ms*1000/iters);
        }
    }
    const float exact = median(us[0]), direct = median(us[1]), staged = median(us[2]);
    const float chunk = selected == 2 ? staged : direct;
    const auto phases = phase_times(d, T, iters, cache, work, stream);
    close("phase_graph_output", floats(d.dst, (size_t) T*128*H), samples[selected]);
    printf("BENCH T=%d H=32 HK=16 D=128 C=32 K=16 enabled=%d scan_ilp=%d prefetch=%d wide=%d compact=%d qstage=%d layout=%d iters=%d l8_us=%.3f chunk_total_us=%.3f prepare_us=%.3f scan_us=%.3f tail_us=%.3f phase_sum_us=%.3f phase_event_delta_us=%.3f direct_us=%.3f staged_us=%.3f qstage_ratio=%.5f same_cache=1 ratio=%.5f projected_24_layers_ms=%.4f scratch_bytes=%zu active_workspace_bytes=%zu\n",
        T, gdn_chunk32_enabled(), gdn_chunk32_scan_ilp_enabled(), gdn_chunk32_prefetch_enabled(), gdn_chunk32_wide_enabled(), gdn_chunk32_compact_enabled(), gdn_chunk32_qstage_enabled(), n1::input_layout, iters, exact, chunk,
        phases.prepare, phases.scan, phases.tail, phases.sum, phases.sum-chunk, direct, staged, direct/staged, exact/chunk, chunk*.024f,
        sizeof(gdn_chunk32)*H*((T-K+31)/32),
        gdn_chunk32_enabled() ? (gdn_chunk32_compact_enabled() ? sizeof(gdn_compact32) : sizeof(gdn_chunk32))*H*((T-K+31)/32) : size_t(0));
    for (int i = 0; i < 3; ++i) { cudaGraphExecDestroy(exec[i]); cudaGraphDestroy(graph[i]); }
    cudaEventDestroy(begin); cudaEventDestroy(end); cudaStreamDestroy(stream); cudaFree(cache); cudaFree(work); release(d);
}
} // namespace n1

int main(int argc, char ** argv) {
    bool benchmark = false, ncu_friendly = false; int only = 0, iters = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") { benchmark = true; }
        else if (arg == "--ncu-friendly") { ncu_friendly = true; }
        else if (arg == "--layout" && i+1 < argc) { n1::input_layout = std::atoi(argv[++i]); }
        else if (arg == "--tokens" && i+1 < argc) { only = std::atoi(argv[++i]); }
        else if (arg == "--iters" && i+1 < argc) { iters = std::atoi(argv[++i]); }
        else { fprintf(stderr, "usage: %s [--bench | --ncu-friendly] [--tokens 64..1024] [--layout 0|1|2] [--iters N]\n", argv[0]); return 2; }
    }
    if ((only && (only < 64 || only > 1024)) || iters < 1 || n1::input_layout < 0 || n1::input_layout > 2) { return 2; }
    if (ncu_friendly && (benchmark || !gdn_chunk32_enabled())) {
        fprintf(stderr, "--ncu-friendly requires chunking enabled and cannot be combined with --bench\n"); return 2;
    }
    if (ncu_friendly && !only) { only = 512; }
    printf("N1_CONFIG chunked=%d scan_ilp=%d prefetch=%d wide=%d compact=%d qstage=%d layout=%d benchmark=%d ncu_friendly=%d\n",
        gdn_chunk32_enabled(), gdn_chunk32_scan_ilp_enabled(), gdn_chunk32_prefetch_enabled(), gdn_chunk32_wide_enabled(), gdn_chunk32_compact_enabled(), gdn_chunk32_qstage_enabled(), n1::input_layout, benchmark, ncu_friendly);
    CHECK(cudaMalloc(&g_ids, sizeof(int)));
    CHECK(cudaMalloc(&g_raw_dt_bias, H*sizeof(float)));
    CHECK(cudaMalloc(&g_raw_a, H*sizeof(float)));
    std::vector<float> bias(H), a(H, -1.0f);
    for (int h = 0; h < H; ++h) { bias[h] = .01f*h; }
    CHECK(cudaMemcpy(g_raw_dt_bias, bias.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(g_raw_a, a.data(), H*sizeof(float), cudaMemcpyHostToDevice));
    const std::vector<int> lengths = only ? std::vector<int>{ only } : std::vector<int>{ 64, 65, 80, 95, 128, 256, 512, 1024 };
    for (int T : lengths) {
        if (ncu_friendly) { n1::profile_once(T); continue; }
        if (benchmark) { n1::bench(T, iters); continue; }
        for (int mode = 0; mode < 4; ++mode) {
            for (int profile = 0; profile < 3; ++profile) {
                for (bool fresh : { false, true }) { n1::test(T, mode, profile, fresh); }
            }
        }
    }
    CHECK(cudaDeviceSynchronize());
    cudaFree(g_ids); cudaFree(g_raw_dt_bias); cudaFree(g_raw_a);
    if (ncu_friendly) {
        printf("N1 NCU %s failures=%d (one prepare/scan/tail launch, finite check only)\n", n1::bad ? "FAIL" : "DONE", n1::bad);
    } else {
        printf("N1 %s failures=%d (GPU tolerance/layout test, no token-fidelity claim)\n", n1::bad ? "FAIL" : "PASS", n1::bad);
    }
    return n1::bad ? 1 : 0;
}
#endif
