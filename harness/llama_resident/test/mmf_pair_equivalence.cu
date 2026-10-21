// GB10 paired k-split matmul (ggml_cuda_mul_mat_f_ksplit_pair: both GDN gate projections in one clustered launch)
// versus two ggml_cuda_mul_mat_f calls (the k-split partial + reduce of each): every output bit. K 4096 and other
// lengths of the k-split route, 32/64/128 rows, a padded activation row stride, random and adversarial operands
// (signed zeros, subnormals, extremes, inf, nan), eager and graph-captured runs. The candidate's outputs are
// poisoned on the kernels' stream first, so a skipped write shows.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "mmf.cuh"

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
    0.0f, -0.0f, 1e-45f, -1e-45f, 1e-40f, -1e-40f, FLT_MIN, -FLT_MIN, 1e-30f, -1e-30f, FLT_MAX, -FLT_MAX,
    1e30f, -1e30f, INFINITY, -INFINITY, NAN, 0.5f, -0.5f, 1.0f, -1.0f, 3.0f, -3.0f,
};

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261016);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    int cases = 0, fused = 0, failures = 0;
    for (int k : {4096, 3072, 5120}) for (int rows : {32, 64, 128}) for (int pad : {0, 64}) for (int mode : {0, 1, 2})
    for (bool capture : {false, true}) {
        // mode 0: random, 1: every 13th operand adversarial, 2: adversarial only
        constexpr int cols = 16;
        const int ky = k + pad;  // activation row stride
        std::vector<float> wa(size_t(k) * rows), wb(size_t(k) * rows), y(size_t(ky) * cols);
        for (auto * v : {&wa, &wb, &y}) {
            for (size_t i = 0; i < v->size(); ++i) {
                const bool adv = mode == 2 || (mode == 1 && i % 13 == 0);
                (*v)[i] = adv ? special[pick(rng)] : dist(rng);
            }
        }
        float *dwa, *dwb, *dy, *ra, *rb, *ca, *cb;
        const size_t out_n = size_t(rows) * cols;
        check(cudaMalloc(&dwa, wa.size() * sizeof(float)));
        check(cudaMalloc(&dwb, wb.size() * sizeof(float)));
        check(cudaMalloc(&dy, y.size() * sizeof(float)));
        for (float ** p : {&ra, &rb, &ca, &cb}) check(cudaMalloc(p, out_n * sizeof(float)));
        check(cudaMemcpy(dwa, wa.data(), wa.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dwb, wb.data(), wb.size() * sizeof(float), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dy, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice));

        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * ta  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        ggml_tensor * tb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        ggml_tensor * ty0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ky, cols);
        ggml_tensor * ty  = ggml_view_2d(ctx, ty0, k, cols, ty0->nb[1], 0);
        ta->data = dwa; tb->data = dwb; ty0->data = dy; ty->data = dy;
        auto out = [&](float * d) {
            ggml_tensor * t = ggml_mul_mat(ctx, ta, ty);
            t->data = d;
            return t;
        };
        ggml_tensor * oa_ref = out(ra);
        ggml_tensor * ob_ref = out(rb);
        ggml_tensor * oa = out(ca);
        ggml_tensor * ob = out(cb);
        oa_ref->src[0] = ta; ob_ref->src[0] = tb; oa->src[0] = ta; ob->src[0] = tb;

        check(cudaMemsetAsync(ra, 0x5a, out_n * sizeof(float), cuda.stream()));
        check(cudaMemsetAsync(rb, 0x5a, out_n * sizeof(float), cuda.stream()));
        ggml_cuda_mul_mat_f(cuda, ta, ty, nullptr, oa_ref);
        ggml_cuda_mul_mat_f(cuda, tb, ty, nullptr, ob_ref);
        // poison on the kernels' stream (they launch as programmatic dependents)
        check(cudaMemsetAsync(ca, 0xa5, out_n * sizeof(float), cuda.stream()));
        check(cudaMemsetAsync(cb, 0xa5, out_n * sizeof(float), cuda.stream()));
        bool ok_launch;
        if (capture) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            ok_launch = ggml_cuda_mul_mat_f_ksplit_pair(cuda, ta, tb, ty, oa, ob);
            check(cudaStreamEndCapture(cuda.stream(), &g));
            check(cudaGraphInstantiate(&ge, g, 0));
            check(cudaGraphLaunch(ge, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
            check(cudaGraphExecDestroy(ge));
            check(cudaGraphDestroy(g));
        } else {
            ok_launch = ggml_cuda_mul_mat_f_ksplit_pair(cuda, ta, tb, ty, oa, ob);
        }
        check(cudaStreamSynchronize(cuda.stream()));
        check(cudaGetLastError());

        std::vector<uint32_t> hra(out_n), hrb(out_n), hca(out_n), hcb(out_n);
        check(cudaMemcpy(hra.data(), ra, out_n * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hrb.data(), rb, out_n * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hca.data(), ca, out_n * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(hcb.data(), cb, out_n * sizeof(float), cudaMemcpyDeviceToHost));
        const bool same = hra == hca && hrb == hcb;
        // only the k-split route (K multiple of the 8-warp split) fuses; otherwise nothing may be launched
        const bool ok = ok_launch ? same : true;
        ++cases;
        fused += ok_launch ? 1 : 0;
        if (!ok) {
            ++failures;
            size_t bad = 0;
            for (size_t i = 0; i < out_n; ++i) bad += (hra[i] != hca[i]) + (hrb[i] != hcb[i]);
            std::printf("FAIL k=%d rows=%d pad=%d mode=%d capture=%d: %zu words differ\n", k, rows, pad, mode,
                        (int) capture, bad);
        }
        ggml_free(ctx);
        for (float * p : {dwa, dwb, dy, ra, rb, ca, cb}) check(cudaFree(p));
    }
    ggml_backend_free(backend);
    std::printf("mmf_pair_equivalence %s: %d cases, %d fused, %d failed\n", failures || fused == 0 ? "FAILED" : "passed",
                cases, fused, failures);
    return failures || fused == 0 ? 1 : 0;
}
