// Unfused RMS_NORM*w -> FWHT -> MMQ D4 quantize kernels versus the fused writers: the one that normalizes
// the rows itself and the plain transform writer fed with the unfused norm. Every q8 byte, every byte of the
// normalized product, and the guards. Rows 1/3/16/512, contiguous and padded row stride, signs on/off, eps 1e-6
// and 0, random and adversarial rows (zeros, subnormals, extremes, inf, nan, overflowing sums), eager and graph
// replays with changed inputs.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "fwht.cuh"
#include "norm.cuh"
#include "quantize.cuh"

#include <cfloat>
#include <cstdint>
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
    for (int rows : {1, 3, 16, 512}) for (bool strided : {false, true}) for (bool signs_on : {false, true})
    for (float eps : {1e-6f, 0.0f}) for (int mode : {0, 1, 2, 3}) {
        // mode 0: random rows with per-row magnitudes, 1: every 9th input adversarial, 2: rows of zeros,
        // tiny, huge and special values, 3: adversarial weights
        if (rows == 512 && (strided || eps == 0.0f)) continue;
        const int    row_stride = k + (strided ? 96 : 0);
        const size_t n_x        = size_t(row_stride) * rows;
        const size_t n_out      = size_t(k) * rows;
        const size_t qbytes     = size_t(k) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);

        std::vector<float> hx(n_x), hw(k), hs(k);
        auto fill = [&]() {
            for (int r = 0; r < rows; ++r) {
                const float mag = std::exp2(float(int(rng() % 41) - 20));
                const int   kind = mode == 2 ? r % 6 : -1;
                for (int c = 0; c < row_stride; ++c) {
                    float v = normal(rng) * mag;
                    if (mode == 1 && rng() % 9 == 0) v = special[pick(rng)];
                    if (kind == 0) v = 0.0f;
                    if (kind == 1) v = (c % 2 ? 1e-40f : -3e-41f);
                    if (kind == 2) v = (c % 3 ? 1e20f : -2e19f);
                    if (kind == 3) v = special[pick(rng)];
                    if (kind == 4) v = c == r ? 5.0f : 0.0f;
                    if (kind == 5) v = c % 64 == 0 ? INFINITY : normal(rng);
                    hx[size_t(r) * row_stride + c] = v;
                }
            }
            for (int c = 0; c < k; ++c) {
                hw[c] = mode == 3 && rng() % 5 == 0 ? special[pick(rng)] : uni(rng);
                hs[c] = (rng() & 1) ? 1.0f : -1.0f;
            }
        };
        fill();

        float * dx = nullptr, * dw = nullptr, * ds = nullptr, * dnr = nullptr, * dnf = nullptr;
        void * qr = nullptr, * qf = nullptr, * qp = nullptr;
        float * dt = nullptr;
        check(cudaMalloc(&dx, n_x * sizeof(float)));
        check(cudaMalloc(&dw, k * sizeof(float)));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&dnr, n_out * sizeof(float) + guard));
        check(cudaMalloc(&dnf, n_out * sizeof(float) + guard));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        check(cudaMalloc(&qp, qbytes + guard));
        check(cudaMalloc(&dt, n_out * sizeof(float)));
        auto upload = [&]() {
            check(cudaMemcpy(dx, hx.data(), n_x * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw, hw.data(), k * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, hs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload();

        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_stride, rows);
        base->data = dx;
        ggml_tensor * x   = ggml_view_2d(ctx, base, k, rows, base->nb[1], 0);
        ggml_tensor * w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        w->data = dw;
        ggml_tensor * rms = ggml_rms_norm(ctx, x, eps);
        ggml_tensor * mul = ggml_mul(ctx, rms, w);
        mul->data = dnr;
        ggml_tensor * tr  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        tr->data = dt;
        ggml_tensor * sg  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        sg->data = ds;

        std::vector<unsigned char> ref(qbytes + guard), fused(qbytes + guard), plain(qbytes + guard);
        std::vector<unsigned char> nref(n_out * sizeof(float) + guard), nfused(n_out * sizeof(float) + guard);
        // the kernels launch as programmatic dependents on a non-blocking stream: poison the outputs on
        // that stream and let it drain, so no memset is still in flight when a kernel starts
        auto poison = [&]() {
            check(cudaMemsetAsync(qr, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaMemsetAsync(qf, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaMemsetAsync(qp, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaMemsetAsync(dnr, 0xa5, n_out * sizeof(float) + guard, cuda.stream()));
            check(cudaMemsetAsync(dnf, 0xa5, n_out * sizeof(float) + guard, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        };
        for (bool capture : {false, true}) {
            poison();
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            ggml_cuda_op_rms_norm_fused(cuda, rms, mul);
            if (!(signs_on ? ggml_cuda_op_fwht_signed(cuda, mul, sg, tr) : ggml_cuda_op_fwht(cuda, mul, tr))) return 3;
            quantize_mmq_q8_1_cuda(dt, nullptr, qr, GGML_TYPE_PQ2_0, k, k, int64_t(k) * rows, int64_t(k) * rows, k, rows, 1, 1, cuda.stream());
            fwht_quantize_mmq_d4_cuda(dnr, GGML_TYPE_F32, signs_on ? ds : nullptr, qp, k, k, rows, cuda.stream());
            fwht_quantize_mmq_d4_rms_cuda(dx, row_stride, dw, eps, signs_on ? ds : nullptr, dnf, qf, k, rows, cuda.stream());
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (capture) {
                check(cudaStreamEndCapture(cuda.stream(), &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
                if (capture) {
                    if (replay) {
                        fill();
                        upload();
                        poison();
                    }
                    check(cudaGraphLaunch(exec, cuda.stream()));
                }
                check(cudaStreamSynchronize(cuda.stream()));
                check(cudaMemcpy(ref.data(), qr, qbytes + guard, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(fused.data(), qf, qbytes + guard, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(plain.data(), qp, qbytes + guard, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < qbytes + guard; ++i) {
                    if (ref[i] != plain[i] || (i >= qbytes && plain[i] != 0xa5)) {
                        std::fprintf(stderr, "plain q8 mismatch rows=%d strided=%d signs=%d eps=%g mode=%d graph=%d byte=%zu ref=%u plain=%u\n",
                                     rows, strided, signs_on, eps, mode, capture, i, ref[i], plain[i]);
                        return 1;
                    }
                }
                check(cudaMemcpy(nref.data(), dnr, n_out * sizeof(float) + guard, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(nfused.data(), dnf, n_out * sizeof(float) + guard, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < qbytes + guard; ++i) {
                    if (ref[i] != fused[i] || (i >= qbytes && (ref[i] != 0xa5 || fused[i] != 0xa5))) {
                        std::fprintf(stderr, "q8 mismatch rows=%d strided=%d signs=%d eps=%g mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                     rows, strided, signs_on, eps, mode, capture, i, ref[i], fused[i]);
                        return 1;
                    }
                }
                for (size_t i = 0; i < n_out * sizeof(float) + guard; ++i) {
                    if (nref[i] != nfused[i] || (i >= n_out * sizeof(float) && (nref[i] != 0xa5 || nfused[i] != 0xa5))) {
                        std::fprintf(stderr, "norm mismatch rows=%d strided=%d signs=%d eps=%g mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                     rows, strided, signs_on, eps, mode, capture, i, nref[i], nfused[i]);
                        size_t bad = 0, untouched = 0;
                        for (size_t e = 0; e < n_out; ++e) {
                            uint32_t a, b;
                            memcpy(&a, &nref[e * 4], 4);
                            memcpy(&b, &nfused[e * 4], 4);
                            bad += a != b;
                            untouched += a == 0xa5a5a5a5u;
                            if (a != b && bad <= 4) {
                                const size_t r = e / k, c = e % k;
                                std::fprintf(stderr, "  e=%zu row=%zu col=%zu ref=%08x(%g) fused=%08x(%g) x=%g w=%g\n", e, r, c, a,
                                             *(float *) &a, b, *(float *) &b, hx[r * row_stride + c], hw[c]);
                            }
                        }
                        std::fprintf(stderr, "  mismatching=%zu of %zu, ref untouched=%zu\n", bad, n_out, untouched);
                        return 1;
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dx)); check(cudaFree(dw)); check(cudaFree(ds)); check(cudaFree(dnr)); check(cudaFree(dnf));
        check(cudaFree(qr)); check(cudaFree(qf)); check(cudaFree(qp)); check(cudaFree(dt));
    }
    ggml_backend_free(backend);
    std::printf("rms_fwht_equivalence_ok cases=%d bytes_exact=true norm_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
