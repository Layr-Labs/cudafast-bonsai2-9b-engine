// GDN output projection input: unfused per-head RMS_NORM*w (128-wide heads) -> SWIGLU(z, .) -> CONT of the
// permuted head-regrouping view -> FWHT -> MMQ D4 quantize kernels versus fwht_quantize_mmq_d4_gnorm, which
// forms the transform input from x and z itself. Every q8 byte and the guards. Rows 1/3/16/512, head groupings
// 2x16 and 4x8, contiguous and padded z rows, signs on/off, eps 1e-6 and 0, random and adversarial inputs (zeros,
// subnormals, extremes, inf, nan, overflowing sums, exp overflow edges), eager and graph replays with changed
// inputs.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "cpy.cuh"
#include "fwht.cuh"
#include "norm.cuh"
#include "quantize.cuh"
#include "unary.cuh"

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
    FLT_MAX, -FLT_MAX, 1e30f, -1e30f, 1e19f, -1e19f, INFINITY, -INFINITY, NAN, -NAN,
    88.7f, -88.7f, 89.0f, -89.0f, 104.0f, -104.0f, 1.0f, -1.0f, 0.5f, 3.0f, -7.0f, 1e-3f, 65504.0f,
};

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261016);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(-4.0f, 4.0f);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    constexpr int hd = 128, nh = 32, k = hd * nh;
    constexpr size_t guard = 64;
    int cases = 0;
    for (int rows : {1, 3, 16, 512}) for (int rep : {2, 4}) for (bool zpad : {false, true})
    for (bool signs_on : {false, true}) for (float eps : {1e-6f, 0.0f}) for (int mode : {0, 1, 2, 3}) {
        // mode 0: random heads with per-head magnitudes, 1: every 9th input adversarial, 2: whole heads of
        // zeros, tiny, huge and special values, 3: adversarial norm weights
        if (rows == 512 && (zpad || eps == 0.0f || rep == 4)) continue;
        const int    zrow   = hd + (zpad ? 32 : 0);
        const size_t n_x    = size_t(k) * rows;
        const size_t n_z    = size_t(zrow) * nh * rows;
        const size_t qbytes = size_t(k) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);

        std::vector<float> hx(n_x), hz(n_z), hw(hd), hs(k);
        auto fill = [&]() {
            for (int h = 0; h < nh * rows; ++h) {
                const float mag  = std::exp2(float(int(rng() % 41) - 20));
                const int   kind = mode == 2 ? h % 6 : -1;
                for (int c = 0; c < hd; ++c) {
                    float v = normal(rng) * mag;
                    if (mode == 1 && rng() % 9 == 0) v = special[pick(rng)];
                    if (kind == 0) v = 0.0f;
                    if (kind == 1) v = (c % 2 ? 1e-40f : -3e-41f);
                    if (kind == 2) v = (c % 3 ? 1e20f : -2e19f);
                    if (kind == 3) v = special[pick(rng)];
                    if (kind == 4) v = c == h % hd ? 5.0f : 0.0f;
                    if (kind == 5) v = c % 64 == 0 ? INFINITY : normal(rng);
                    hx[size_t(h) * hd + c] = v;
                }
                for (int c = 0; c < zrow; ++c) {
                    float u = uni(rng) * 3.0f;
                    if ((mode == 1 && rng() % 9 == 0) || kind == 3) u = special[pick(rng)];
                    hz[size_t(h) * zrow + c] = u;
                }
            }
            for (int c = 0; c < hd; ++c) hw[c] = mode == 3 && rng() % 5 == 0 ? special[pick(rng)] : uni(rng);
            for (int c = 0; c < k; ++c) hs[c] = (rng() & 1) ? 1.0f : -1.0f;
        };
        fill();

        float * dx = nullptr, * dz = nullptr, * dw = nullptr, * ds = nullptr;
        float * dnorm = nullptr, * dglu = nullptr, * dcont = nullptr, * dt = nullptr;
        void * qr = nullptr, * qf = nullptr;
        check(cudaMalloc(&dx, n_x * sizeof(float)));
        check(cudaMalloc(&dz, n_z * sizeof(float)));
        check(cudaMalloc(&dw, hd * sizeof(float)));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&dnorm, n_x * sizeof(float)));
        check(cudaMalloc(&dglu, n_x * sizeof(float)));
        check(cudaMalloc(&dcont, n_x * sizeof(float)));
        check(cudaMalloc(&dt, n_x * sizeof(float)));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        auto upload = [&]() {
            check(cudaMemcpy(dx, hx.data(), n_x * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dz, hz.data(), n_z * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw, hw.data(), hd * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, hs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload();

        // the graph llama_model_qwen35 builds for the GDN output and the ssm_out Hadamard input
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, rows);
        x->data = dx;
        ggml_tensor * zb   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, zrow, nh, rows);
        zb->data = dz;
        ggml_tensor * z    = ggml_view_3d(ctx, zb, hd, nh, rows, zb->nb[1], zb->nb[2], 0);
        ggml_tensor * w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        w->data = dw;
        ggml_tensor * sg   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        sg->data = ds;
        ggml_tensor * rms  = ggml_rms_norm(ctx, x, eps);
        ggml_tensor * mul  = ggml_mul(ctx, rms, w);
        mul->data = dnorm;
        ggml_tensor * glu  = ggml_swiglu_split(ctx, z, mul);
        glu->data = dglu;
        ggml_tensor * r1   = ggml_reshape_2d(ctx, glu, k, rows);
        ggml_tensor * r2   = ggml_reshape_4d(ctx, r1, hd, nh / rep, rep, rows);
        ggml_tensor * perm = ggml_permute(ctx, r2, 0, 2, 1, 3);
        ggml_tensor * cont = ggml_cont(ctx, perm);
        cont->data = dcont;
        ggml_tensor * flat = ggml_reshape_2d(ctx, cont, k, rows);
        ggml_tensor * tr   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        tr->data = dt;
        const int64_t hb = hd * sizeof(float);
        if (perm->data != glu->data || perm->ne[0] != hd || perm->nb[1] % hb || perm->nb[2] % hb || perm->nb[3] % hb) return 3;

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
            ggml_cuda_op_rms_norm_fused(cuda, rms, mul);
            ggml_cuda_op_swiglu(cuda, glu);
            ggml_cuda_cpy(cuda, perm, cont);
            if (!(signs_on ? ggml_cuda_op_fwht_signed(cuda, flat, sg, tr) : ggml_cuda_op_fwht(cuda, flat, tr))) return 3;
            quantize_mmq_q8_1_cuda(dt, nullptr, qr, GGML_TYPE_PQ2_0, k, k, int64_t(k) * rows, int64_t(k) * rows, k, rows, 1, 1, cuda.stream());
            fwht_quantize_mmq_d4_gnorm_cuda(dx, dz, z->nb[1] / sizeof(float), dw, eps, perm->ne[1], perm->nb[1] / hb,
                                            perm->nb[2] / hb, perm->nb[3] / hb, signs_on ? ds : nullptr, qf, rows, cuda.stream());
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
                        std::fprintf(stderr, "q8 mismatch rows=%d rep=%d zpad=%d signs=%d eps=%g mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                     rows, rep, zpad, signs_on, eps, mode, capture, i, ref[i], fused[i]);
                        return 1;
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dx)); check(cudaFree(dz)); check(cudaFree(dw)); check(cudaFree(ds));
        check(cudaFree(dnorm)); check(cudaFree(dglu)); check(cudaFree(dcont)); check(cudaFree(dt));
        check(cudaFree(qr)); check(cudaFree(qf));
    }
    ggml_backend_free(backend);
    std::printf("gnorm_fwht_equivalence_ok cases=%d bytes_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
