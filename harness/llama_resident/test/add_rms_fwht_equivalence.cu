// Verify-batch layer input ADD -> RMS_NORM -> MUL(w) -> signed FWHT -> MMQ D4 quantization: the fused writer
// fwht_quantize_mmq_d4_add_rms against both paths the graph takes without it - ggml_cuda_op_add_rms_norm_fused
// followed by fwht_quantize_mmq_d4_cuda, and the unfused ggml_cuda_op_add, ggml_cuda_op_rms_norm_fused,
// ggml_cuda_op_fwht(_signed) and quantize_mmq_q8_1_cuda. Every byte of the sum, the normalized product and the q8
// rows, plus the guards. Rows 1/2/3/8/16, signs on/off, eps 1e-6 and 0, in-place sum (over a or b) and separate,
// random and adversarial rows (zeros, cancelling sums, subnormals, extremes, inf, nan, overflowing sums,
// adversarial weights), eager and graph replays with changed inputs.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "binbcast.cuh"
#include "fwht.cuh"
#include "norm.cuh"
#include "quantize.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));
        std::exit(2);
    }
}

static const float special[] = {
    0.0f, -0.0f, 1e-45f, -1e-45f, 1e-40f, -1e-40f, FLT_MIN, -FLT_MIN, 1e-30f, -1e-30f, 1e-20f, -1e-20f,
    FLT_MAX, -FLT_MAX, 1e30f, -1e30f, 1e19f, -1e19f, 2e19f, INFINITY, -INFINITY, NAN, -NAN,
    1.0f, -1.0f, 0.5f, 3.0f, -7.0f, 1e-3f, 65504.0f,
};

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261016);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-4.0f, 4.0f);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    constexpr int k = 4096;
    constexpr size_t guard = 64;
    int cases = 0;
    for (int rows : {1, 2, 3, 8, 16}) for (int inplace : {0, 1, 2}) for (bool signs_on : {false, true})
    for (float eps : {1e-6f, 0.0f}) for (int mode : {0, 1, 2, 3}) {
        // mode 0: random rows with per-row magnitudes, 1: every 9th input adversarial, 2: rows of zeros,
        // cancelling, tiny, huge and special values, 3: adversarial weights. inplace 1/2: the sum over a/b.
        const size_t n    = size_t(k) * rows;
        const size_t nb   = n * sizeof(float);
        const size_t qbytes = size_t(k) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);

        std::vector<float> ha(n), hb(n), hw(k), hs(k);
        auto fill = [&]() {
            for (int r = 0; r < rows; ++r) {
                const float mag  = std::exp2(float(int(rng() % 41) - 20));
                const int   kind = mode == 2 ? r % 7 : -1;
                for (int c = 0; c < k; ++c) {
                    float u = normal(rng) * mag, v = normal(rng) * mag;
                    if (mode == 1 && rng() % 9 == 0) u = special[pick(rng)];
                    if (mode == 1 && rng() % 9 == 0) v = special[pick(rng)];
                    if (kind == 0) { u = 0.0f; v = 0.0f; }
                    if (kind == 1) v = -u;
                    if (kind == 2) { u = c % 2 ? 1e-40f : -3e-41f; v = 1e-45f; }
                    if (kind == 3) { u = c % 3 ? 1e20f : -2e19f; v = u; }
                    if (kind == 4) { u = special[pick(rng)]; v = special[pick(rng)]; }
                    if (kind == 5) { u = c == r ? 5.0f : 0.0f; v = 0.0f; }
                    if (kind == 6) { u = c % 64 == 0 ? INFINITY : u; }
                    ha[size_t(r) * k + c] = u;
                    hb[size_t(r) * k + c] = v;
                }
            }
            for (int c = 0; c < k; ++c) {
                hw[c] = mode == 3 && rng() % 5 == 0 ? special[pick(rng)] : uni(rng);
                hs[c] = (rng() & 1) ? 1.0f : -1.0f;
            }
        };
        fill();

        // per path: a, b, sum (a/b/own for in-place), norm, f32 transform, q8
        float * da[3], * db[3], * dsum_own[3], * dnorm[3];
        float * dw = nullptr, * ds = nullptr, * dt = nullptr;
        void * dq[3];
        for (int p = 0; p < 3; ++p) {
            check(cudaMalloc(&da[p], nb + guard));
            check(cudaMalloc(&db[p], nb + guard));
            check(cudaMalloc(&dsum_own[p], nb + guard));
            check(cudaMalloc(&dnorm[p], nb + guard));
            check(cudaMalloc(&dq[p], qbytes + guard));
        }
        check(cudaMalloc(&dw, k * sizeof(float)));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&dt, nb));
        auto sum_ptr = [&](int p) { return inplace == 1 ? da[p] : inplace == 2 ? db[p] : dsum_own[p]; };
        // the kernels launch as programmatic dependents on a non-blocking stream: write inputs and poison the
        // outputs on that stream and let it drain, so no copy is still in flight when a kernel starts
        auto upload = [&]() {
            for (int p = 0; p < 3; ++p) {
                check(cudaMemcpyAsync(da[p], ha.data(), nb, cudaMemcpyHostToDevice, cuda.stream()));
                check(cudaMemcpyAsync(db[p], hb.data(), nb, cudaMemcpyHostToDevice, cuda.stream()));
                check(cudaMemsetAsync((char *) da[p] + nb, 0xa5, guard, cuda.stream()));
                check(cudaMemsetAsync((char *) db[p] + nb, 0xa5, guard, cuda.stream()));
                check(cudaMemsetAsync(dsum_own[p], 0xa5, nb + guard, cuda.stream()));
                check(cudaMemsetAsync(dnorm[p], 0xa5, nb + guard, cuda.stream()));
                check(cudaMemsetAsync(dq[p], 0xa5, qbytes + guard, cuda.stream()));
            }
            check(cudaMemcpyAsync(dw, hw.data(), k * sizeof(float), cudaMemcpyHostToDevice, cuda.stream()));
            check(cudaMemcpyAsync(ds, hs.data(), k * sizeof(float), cudaMemcpyHostToDevice, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        };
        upload();

        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        w->data = dw;
        ggml_tensor * sg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        sg->data = ds;
        ggml_tensor * tr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        tr->data = dt;
        ggml_tensor * add[2], * rms[2], * mul[2];
        for (int p = 0; p < 2; ++p) {
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
            a->data = da[p];
            b->data = db[p];
            add[p] = ggml_add(ctx, a, b);
            add[p]->data = sum_ptr(p);
            rms[p] = ggml_rms_norm(ctx, add[p], eps);
            mul[p] = ggml_mul(ctx, rms[p], w);
            mul[p]->data = dnorm[p];
        }

        std::vector<unsigned char> h[3];
        for (bool capture : {false, true}) {
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            // path 0: the C40g add_rms_norm kernel, then the plain transform writer
            ggml_cuda_op_add_rms_norm_fused(cuda, add[0], rms[0], mul[0]);
            fwht_quantize_mmq_d4_cuda(dnorm[0], GGML_TYPE_F32, signs_on ? ds : nullptr, dq[0], k, k, rows, cuda.stream());
            // path 1: unfused add, norm, standalone transform, quantization
            ggml_cuda_op_add(cuda, add[1]);
            ggml_cuda_op_rms_norm_fused(cuda, rms[1], mul[1]);
            if (!(signs_on ? ggml_cuda_op_fwht_signed(cuda, mul[1], sg, tr) : ggml_cuda_op_fwht(cuda, mul[1], tr))) return 3;
            quantize_mmq_q8_1_cuda(dt, nullptr, dq[1], GGML_TYPE_PQ2_0, k, k, int64_t(k) * rows, int64_t(k) * rows, k, rows, 1, 1, cuda.stream());
            // path 2: the fused writer
            fwht_quantize_mmq_d4_add_rms_cuda(da[2], db[2], sum_ptr(2), dw, eps, signs_on ? ds : nullptr, dnorm[2], dq[2], k, rows, cuda.stream());
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (capture) {
                check(cudaStreamEndCapture(cuda.stream(), &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
                if (capture) {
                    if (replay) fill();
                    upload();
                    check(cudaGraphLaunch(exec, cuda.stream()));
                }
                check(cudaStreamSynchronize(cuda.stream()));
                for (int what = 0; what < 3; ++what) {
                    const size_t bytes = what == 2 ? qbytes : nb;
                    for (int p = 0; p < 3; ++p) {
                        h[p].assign(bytes + guard, 0);
                        const void * src = what == 0 ? (const void *) sum_ptr(p) : what == 1 ? (const void *) dnorm[p] : dq[p];
                        check(cudaMemcpy(h[p].data(), src, bytes + guard, cudaMemcpyDeviceToHost));
                    }
                    for (size_t i = 0; i < bytes + guard; ++i) {
                        const bool bad = h[2][i] != h[0][i] || h[2][i] != h[1][i] || (i >= bytes && h[2][i] != 0xa5);
                        if (bad) {
                            std::fprintf(stderr, "%s mismatch rows=%d inplace=%d signs=%d eps=%g mode=%d graph=%d byte=%zu c40g=%u unfused=%u fused=%u\n",
                                         what == 0 ? "sum" : what == 1 ? "norm" : "q8", rows, inplace, signs_on, eps, mode, capture,
                                         i, h[0][i], h[1][i], h[2][i]);
                            return 1;
                        }
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        for (int p = 0; p < 3; ++p) {
            check(cudaFree(da[p])); check(cudaFree(db[p])); check(cudaFree(dsum_own[p])); check(cudaFree(dnorm[p])); check(cudaFree(dq[p]));
        }
        check(cudaFree(dw)); check(cudaFree(ds)); check(cudaFree(dt));
    }
    ggml_backend_free(backend);
    std::printf("add_rms_fwht_equivalence_ok cases=%d sum_exact=true norm_exact=true q8_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
