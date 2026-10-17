// Bit-exact check of the lazy recurrent rollback in gated_delta_net_kernel.cuh against the eager
// per-token snapshots it replaces. For every rollback depth the next ubatch must produce the same
// attention output and the same final state from a lazily written cache as from an eagerly
// written one, and an eagerly written cache must stay readable with lazy enabled.
//
// nvcc -O2 -std=c++17 -arch=sm_121 -I llama.cpp/ggml/include -I llama.cpp/ggml/src \
//      -I llama.cpp/ggml/src/ggml-cuda harness/llama_resident/test/gdn_lazy_rollback_equivalence.cu \
//      -o gdn-lazy-rollback-equivalence
#include "gated_delta_net_kernel.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
