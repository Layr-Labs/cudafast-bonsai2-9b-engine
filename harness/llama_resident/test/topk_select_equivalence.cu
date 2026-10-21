// Byte-exact check of the streaming top-k selection (top-k-select.cuh) against
// the argsort fallback it replaces: the first k entries of the stable
// descending cub radix sort (DeviceSegmentedRadixSort, the CUDA graph capture
// route, and DeviceRadixSort for a single row), a host stable sort, and the
// previous (C8) streaming selector (topk_select_c8_reference.cuh).
//
// nvcc -O2 -std=c++17 -arch=sm_121 -I llama.cpp/ggml/src/ggml-cuda \
//      harness/llama_resident/test/topk_select_equivalence.cu -o topk-select-equivalence
#include "top-k-select.cuh"
#include "topk_select_c8_reference.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#define CHECK(x)                                                                              \
    do {                                                                                      \
        cudaError_t err_ = (x);                                                               \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(2);                                                                          \
        }                                                                                     \
    } while (0)

static uint64_t host_pack(const float v, const int col) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    u = u == 0x80000000u ? 0u : u;
    const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t) ord << 32) | (uint64_t) (~(uint32_t) col);
}

static std::vector<int> host_reference(const std::vector<float> & x, int ncols, int nrows, int k) {
    std::vector<int> out((size_t) nrows * k);
    std::vector<int> idx(ncols);
    for (int r = 0; r < nrows; ++r) {
        const float * row = x.data() + (size_t) r * ncols;
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return host_pack(row[a], a) > host_pack(row[b], b); });
        std::copy(idx.begin(), idx.begin() + k, out.begin() + (size_t) r * k);
    }
    return out;
}

// first k of cub's stable descending radix sort, as in argsort_f32_i32_cuda_cub
static std::vector<int> cub_reference(const float * d_x, int ncols, int nrows, int k, bool segmented) {
    const size_t n = (size_t) ncols * nrows;
    float * d_keys_in, * d_keys_out;
    int * d_idx_in, * d_idx_out, * d_offsets;
    CHECK(cudaMalloc(&d_keys_in, n * sizeof(float)));
    CHECK(cudaMalloc(&d_keys_out, n * sizeof(float)));
    CHECK(cudaMalloc(&d_idx_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_idx_out, n * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, (nrows + 1) * sizeof(int)));
    CHECK(cudaMemcpy(d_keys_in, d_x, n * sizeof(float), cudaMemcpyDeviceToDevice));
    std::vector<int> h_idx(n), h_off(nrows + 1);
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            h_idx[(size_t) r * ncols + c] = c;
        }
    }
    for (int r = 0; r <= nrows; ++r) {
        h_off[r] = r * ncols;
    }
    CHECK(cudaMemcpy(d_idx_in, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, h_off.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    size_t temp_bytes = 0;
    if (segmented) {
        CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, temp_bytes, d_keys_in, d_keys_out, d_idx_in,
                                                                 d_idx_out, (int) n, nrows, d_offsets, d_offsets + 1));
    } else {
        CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, d_keys_in, d_keys_out, d_idx_in, d_idx_out,
                                                        (int) n));
    }
    void * d_temp;
    CHECK(cudaMalloc(&d_temp, temp_bytes));
    if (segmented) {
        CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp, temp_bytes, d_keys_in, d_keys_out, d_idx_in,
                                                                 d_idx_out, (int) n, nrows, d_offsets, d_offsets + 1));
    } else {
        CHECK(cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes, d_keys_in, d_keys_out, d_idx_in, d_idx_out,
                                                        (int) n));
    }
    CHECK(cudaDeviceSynchronize());
    std::vector<int> sorted(n);
    CHECK(cudaMemcpy(sorted.data(), d_idx_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> out((size_t) nrows * k);
    for (int r = 0; r < nrows; ++r) {
        std::copy(sorted.begin() + (size_t) r * ncols, sorted.begin() + (size_t) r * ncols + k,
                  out.begin() + (size_t) r * k);
    }
    cudaFree(d_temp);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_idx_in);
    cudaFree(d_idx_out);
    cudaFree(d_offsets);
    return out;
}

static float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static constexpr int kPatterns = 16;

static std::vector<float> make_data(int pattern, int ncols, int nrows, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 5.0f);
    std::uniform_int_distribution<int> level(-2, 1);
    std::uniform_int_distribution<int> pos(0, ncols - 1);
    std::vector<float> x((size_t) ncols * nrows);
    const float inf = std::numeric_limits<float>::infinity();
    // integer levels 0..M-1: about CHUNK/M keys share the top value, so the threshold path's survivor count
    // sweeps across its 256-entry buffer
    const int levels[] = { 16, 24, 32, 40, 48 };
    for (size_t i = 0; i < x.size(); ++i) {
        const int col = (int) (i % ncols);
        switch (pattern) {
            case 0: x[i] = normal(rng); break;                                  // logits-like
            case 1: x[i] = (float) level(rng); break;                           // heavy ties
            case 2: x[i] = (rng() & 1) ? 0.0f : -0.0f; break;                   // signed zeros only
            case 3: x[i] = -std::fabs(normal(rng)) - 1.0f; break;               // negative, zeros injected below
            case 4: x[i] = normal(rng); break;                                  // specials injected below
            case 5: {                                                           // NaN rows, several payloads
                const uint32_t nans[] = { 0x7fc00000u, 0xffc00000u, 0x7fffffffu, 0xffffffffu, 0x7f800001u, 0xff800001u };
                x[i] = bits_to_float(nans[rng() % 6]);
            } break;
            case 6: x[i] = 1.0f; break;                                         // constant rows
            case 7: x[i] = (rng() & 1) ? inf : -inf; break;                     // infinities only
            case 8: x[i] = (float) col; break;                                  // ascending
            case 9: x[i] = -(float) col; break;                                 // descending
            case 10: case 11: case 12: case 13: case 14:
                x[i] = (float) (rng() % levels[pattern - 10]); break;
            default:                                                            // winners in one thread's columns
                x[i] = (col % TOPK_SELECT_NT == 7) ? 100.0f + normal(rng) : normal(rng); break;
        }
    }
    for (int r = 0; r < nrows; ++r) {
        float * row = x.data() + (size_t) r * ncols;
        if (pattern == 3) {
            for (int j = 0; j < 40; ++j) {
                row[pos(rng)] = (j & 1) ? 0.0f : -0.0f;
            }
        }
        if (pattern == 4) {
            const float specials[] = { std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                       std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min() };
            for (int j = 0; j < 24; ++j) {
                row[pos(rng)] = specials[j % 8];
            }
            row[0] = specials[(r * 3) % 8];  // column 0 and the last column are chunk edges
            row[ncols - 1] = specials[(r * 5 + 1) % 8];
        }
    }
    return x;
}

static int compare(const char * what, const std::vector<int> & got, const std::vector<int> & want) {
    if (got == want) {
        return 0;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            fprintf(stderr, "  MISMATCH %s at %zu: got %d want %d\n", what, i, got[i], want[i]);
            break;
        }
    }
    return 1;
}

int main() {
    struct Case { int ncols, nrows, k; };
    const Case cases[] = {
        { 248320, 16, 16 }, { 248320, 1, 16 }, { 248320, 15, 16 }, { 1025, 3, 16 }, { 8192, 4, 32 },
        { 8192, 192, 32 }, { 8193, 2, 32 },    { 24577, 5, 1 },   { 100003, 7, 5 },   { 49153, 2, 17 }, { 32, 3, 32 },
        // verify top-8 rows, 32 / 33 / 37 chunks (warp merge limit), threshold path k edges
        { 248320, 16, 8 },  { 248320, 1, 8 },  { 248320, 4, 1 },   { 262144, 2, 16 }, { 262145, 2, 16 },
        { 300000, 3, 8 },   { 300000, 2, 32 }, { 65536, 3, 15 },   { 16384, 2, 16 },  { 16, 2, 16 },
    };
    int failures = 0, checks = 0;
    for (const Case & cs : cases) {
        for (int pattern = 0; pattern < kPatterns; ++pattern) {
            const std::vector<float> x = make_data(pattern, cs.ncols, cs.nrows, 1234u + pattern * 77u + cs.ncols);
            float * d_x;
            int * d_dst;
            uint64_t * d_scratch;
            CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_dst, (size_t) cs.nrows * cs.k * sizeof(int)));
            CHECK(cudaMalloc(&d_scratch, topk_select_scratch_elems(cs.ncols, cs.nrows, cs.k) * sizeof(uint64_t)));
            CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
            topk_select_launch(d_x, d_dst, d_scratch, cs.ncols, cs.nrows, cs.k, 0);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
            std::vector<int> got((size_t) cs.nrows * cs.k);
            CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(int), cudaMemcpyDeviceToHost));

            std::vector<int> c8((size_t) cs.nrows * cs.k);
            {
                uint64_t * d_scratch_c8;
                CHECK(cudaMalloc(&d_scratch_c8, topk_c8_scratch_elems(cs.ncols, cs.nrows, cs.k) * sizeof(uint64_t)));
                CHECK(cudaMemset(d_dst, 0xff, c8.size() * sizeof(int)));
                topk_c8_launch(d_x, d_dst, d_scratch_c8, cs.ncols, cs.nrows, cs.k, 0);
                CHECK(cudaGetLastError());
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaMemcpy(c8.data(), d_dst, c8.size() * sizeof(int), cudaMemcpyDeviceToHost));
                cudaFree(d_scratch_c8);
            }

            char what[160];
            snprintf(what, sizeof(what), "ncols=%d nrows=%d k=%d pattern=%d", cs.ncols, cs.nrows, cs.k, pattern);
            int bad = 0;
            bad |= compare("c8-selector", got, c8);
            bad |= compare("host", got, host_reference(x, cs.ncols, cs.nrows, cs.k));
            bad |= compare("cub-segmented-radix", got, cub_reference(d_x, cs.ncols, cs.nrows, cs.k, true));
            checks += 3;
            if (cs.nrows == 1) {
                bad |= compare("cub-radix", got, cub_reference(d_x, cs.ncols, cs.nrows, cs.k, false));
                checks += 1;
            }
            printf("%s %s\n", bad ? "FAIL" : "ok  ", what);
            failures += bad;
            cudaFree(d_x);
            cudaFree(d_dst);
            cudaFree(d_scratch);
        }
    }

    // CUDA graph capture and replay with changed inputs in the same buffers.
    // Keep the vocabulary path and cover the production screened-head shape.
    for (const auto cs : { Case{248320, 16, 16}, Case{8192, 192, 32} }) {
        const int ncols = cs.ncols, nrows = cs.nrows, k = cs.k;
        float * d_x;
        int * d_dst;
        uint64_t * d_scratch;
        CHECK(cudaMalloc(&d_x, (size_t) ncols * nrows * sizeof(float)));
        CHECK(cudaMalloc(&d_dst, (size_t) nrows * k * sizeof(int)));
        CHECK(cudaMalloc(&d_scratch, topk_select_scratch_elems(ncols, nrows, k) * sizeof(uint64_t)));
        cudaStream_t stream;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        topk_select_launch(d_x, d_dst, d_scratch, ncols, nrows, k, stream);
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&exec, graph, 0));
        for (int rep = 0; rep < 3; ++rep) {
            const std::vector<float> x = make_data(rep == 1 ? 1 : (rep == 2 ? 4 : 0), ncols, nrows, 999u + rep);
            CHECK(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK(cudaGraphLaunch(exec, stream));
            CHECK(cudaStreamSynchronize(stream));
            std::vector<int> got((size_t) nrows * k);
            CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
            const int bad = compare("graph-replay", got, host_reference(x, ncols, nrows, k));
            printf("%s graph replay %d\n", bad ? "FAIL" : "ok  ", rep);
            failures += bad;
            checks += 1;
        }

        // informative timing only: this selector, the C8 selector, and segmented radix sort + copy
        uint64_t * d_scratch_c8;
        CHECK(cudaMalloc(&d_scratch_c8, topk_c8_scratch_elems(ncols, nrows, k) * sizeof(uint64_t)));
        cudaGraph_t graph_c8;
        cudaGraphExec_t exec_c8;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        topk_c8_launch(d_x, d_dst, d_scratch_c8, ncols, nrows, k, stream);
        CHECK(cudaStreamEndCapture(stream, &graph_c8));
        CHECK(cudaGraphInstantiate(&exec_c8, graph_c8, 0));
        cudaEvent_t e0, e1;
        CHECK(cudaEventCreate(&e0));
        CHECK(cudaEventCreate(&e1));
        const int iters = 50;
        CHECK(cudaEventRecord(e0, stream));
        for (int i = 0; i < iters; ++i) {
            CHECK(cudaGraphLaunch(exec, stream));
        }
        CHECK(cudaEventRecord(e1, stream));
        CHECK(cudaEventSynchronize(e1));
        float ms_select = 0.0f;
        CHECK(cudaEventElapsedTime(&ms_select, e0, e1));
        CHECK(cudaEventRecord(e0, stream));
        for (int i = 0; i < iters; ++i) {
            CHECK(cudaGraphLaunch(exec_c8, stream));
        }
        CHECK(cudaEventRecord(e1, stream));
        CHECK(cudaEventSynchronize(e1));
        float ms_c8 = 0.0f;
        CHECK(cudaEventElapsedTime(&ms_c8, e0, e1));
        printf("info: %dx%d k=%d select %.1f us/call, C8 select %.1f us/call (graph)\n",
               nrows, ncols, k, 1000.0f * ms_select / iters, 1000.0f * ms_c8 / iters);
        cudaGraphExecDestroy(exec_c8);
        cudaGraphDestroy(graph_c8);
        cudaFree(d_scratch_c8);
        CHECK(cudaEventRecord(e0, 0));
        for (int i = 0; i < 5; ++i) {
            (void) cub_reference(d_x, ncols, nrows, k, true);
        }
        CHECK(cudaEventRecord(e1, 0));
        CHECK(cudaEventSynchronize(e1));
        float ms_sort = 0.0f;
        CHECK(cudaEventElapsedTime(&ms_sort, e0, e1));
        printf("info: select %.1f us/call (graph), segmented radix reference incl. setup %.1f us/call\n",
               1000.0f * ms_select / iters, 1000.0f * ms_sort / 5);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_x);
        cudaFree(d_dst);
        cudaFree(d_scratch);
    }

    printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
