// SWIGLU kernel -> FWHT/MMQ D4 writer versus the writer that forms the SWIGLU itself: every byte/guard.
// Split, fused and swapped GLU, padded row stride, f32/f16, signs, transform padding, rows 1/16/512,
// random and adversarial operands (zeros, subnormals, extremes, inf, nan, exp overflow edges),
// eager and graph replays with changed operands.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
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
    FLT_MAX, -FLT_MAX, 1e30f, -1e30f, 3e38f, -3e38f, INFINITY, -INFINITY, NAN, -NAN,
    88.0f, -88.0f, 88.7f, -88.7f, 88.73f, -88.73f, 89.0f, -89.0f, 103.0f, -103.0f, 104.0f, -104.0f,
    126.0f, -126.0f, 127.9f, -127.9f, 150.0f, -150.0f, 1e6f, -1e6f, 17.0f, -17.0f, 1e-7f, -1e-7f,
    0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 0.693147f, -0.693147f, 15.9424f, -15.9424f,
};

enum glu_variant { SPLIT, SPLIT_STRIDED, FUSED, FUSED_SWAPPED };

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261016);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    int cases = 0;
    for (int multiple : {1, 3}) for (int rows : {1, 16, 512}) for (bool signs_on : {false, true})
    for (bool half_on : {false, true}) for (bool padded : {false, true})
    for (glu_variant variant : {SPLIT, SPLIT_STRIDED, FUSED, FUSED_SWAPPED}) for (int mode : {0, 1, 2}) {
        // mode 0: random, 1: random with every 7th operand adversarial, 2: adversarial only
        if (rows == 512 && (padded || half_on || variant == SPLIT_STRIDED)) continue;
        const int    k        = 4096 * multiple;
        const int    kp       = k + (padded ? 4096 : 0);
        const bool   split    = variant == SPLIT || variant == SPLIT_STRIDED;
        const int    row_pad  = variant == SPLIT_STRIDED ? 64 : 0;
        const int    src_ne0  = split ? k : 2 * k;          // elements per operand row
        const int    src_row  = src_ne0 + row_pad;          // operand row stride in elements
        const size_t src_n    = size_t(src_row) * rows;     // elements per operand buffer
        const size_t ts       = half_on ? sizeof(ggml_fp16_t) : sizeof(float);
        const ggml_type type  = half_on ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const size_t qbytes   = size_t(kp) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);
        constexpr size_t guard = 64;

        std::vector<float> h0(src_n), h1(src_n), signs(k);
        std::vector<ggml_fp16_t> h0h(src_n), h1h(src_n);
        auto fill = [&]() {
            for (size_t i = 0; i < src_n; ++i) {
                const bool adv0 = mode == 2 || (mode == 1 && rng() % 7 == 0);
                const bool adv1 = mode == 2 || (mode == 1 && rng() % 7 == 0);
                h0[i] = adv0 ? special[pick(rng)] : dist(rng);
                h1[i] = adv1 ? special[pick(rng)] : dist(rng);
            }
            for (int i = 0; i < k; ++i) signs[i] = (rng() & 1) ? 1.0f : -1.0f;
            ggml_fp32_to_fp16_row(h0.data(), h0h.data(), src_n);
            ggml_fp32_to_fp16_row(h1.data(), h1h.data(), src_n);
        };
        fill();

        void * d0 = nullptr, * d1 = nullptr, * dt = nullptr;
        float * ds = nullptr;
        void * qr = nullptr, * qf = nullptr;
        check(cudaMalloc(&d0, src_n * ts));
        check(cudaMalloc(&d1, src_n * ts));
        check(cudaMalloc(&dt, size_t(k) * rows * ts));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        auto upload = [&]() {
            check(cudaMemcpy(d0, half_on ? (const void *) h0h.data() : (const void *) h0.data(), src_n * ts, cudaMemcpyHostToDevice));
            check(cudaMemcpy(d1, half_on ? (const void *) h1h.data() : (const void *) h1.data(), src_n * ts, cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, signs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload();

        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * base0 = ggml_new_tensor_2d(ctx, type, src_row, rows);
        ggml_tensor * base1 = ggml_new_tensor_2d(ctx, type, src_row, rows);
        base0->data = d0;
        base1->data = d1;
        ggml_tensor * a = ggml_view_2d(ctx, base0, src_ne0, rows, base0->nb[1], 0);
        ggml_tensor * b = ggml_view_2d(ctx, base1, src_ne0, rows, base1->nb[1], 0);
        ggml_tensor * glu = variant == FUSED         ? ggml_swiglu(ctx, a) :
                            variant == FUSED_SWAPPED ? ggml_swiglu_swapped(ctx, a) :
                                                       ggml_swiglu_split(ctx, a, b);
        glu->data = dt;
        if (glu->ne[0] != k || ggml_nrows(glu) != rows) return 3;

        // operand addressing exactly as ggml_cuda_try_fwht_mmq derives it
        const ggml_tensor * src0 = glu->src[0];
        const ggml_tensor * src1 = glu->src[1];
        const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
        const int32_t swapped = ggml_get_op_params_i32(glu, 1);
        const char * p0 = (const char *) src0->data;
        const char * p1 = src1 ? (const char *) src1->data : p0;
        if (!src1) {
            p0 += swapped ? nc * ts : 0;
            p1 += swapped ? 0 : nc * ts;
        }
        const int64_t o0 = src0->nb[1] / ts;
        const int64_t o1 = (src1 ? src1->nb[1] : src0->nb[1]) / ts;

        std::vector<unsigned char> ref(qbytes + guard), fused(qbytes + guard);
        // the kernels launch as programmatic dependents on a non-blocking stream: poison the outputs on
        // that stream and let it drain, so no memset is still in flight when a kernel starts
        auto poison = [&]() {
            check(cudaMemsetAsync(qr, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaMemsetAsync(qf, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        };
        for (bool capture : {false, true}) {
            poison();
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            ggml_cuda_op_swiglu(cuda, glu);
            fwht_quantize_mmq_d4_cuda(dt, type, signs_on ? ds : nullptr, qr, k, kp, rows, cuda.stream());
            fwht_quantize_mmq_d4_cuda(p0, type, signs_on ? ds : nullptr, qf, k, kp, rows, cuda.stream(), p1, o0, o1);
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
                for (size_t i = 0; i < qbytes + guard; ++i) {
                    if (ref[i] != fused[i] || (i >= qbytes && (ref[i] != 0xa5 || fused[i] != 0xa5))) {
                        std::fprintf(stderr, "mismatch k=%d rows=%d signs=%d half=%d pad=%d variant=%d mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                     k, rows, signs_on, half_on, padded, int(variant), mode, capture, i, ref[i], fused[i]);
                        return 1;
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(d0)); check(cudaFree(d1)); check(cudaFree(dt)); check(cudaFree(ds));
        check(cudaFree(qr)); check(cudaFree(qf));
    }
    ggml_backend_free(backend);
    std::printf("swiglu_fwht_equivalence_ok cases=%d bytes_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
