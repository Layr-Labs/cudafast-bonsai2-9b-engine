// GDN out-projection input (C56): SWIGLU kernel -> CONT of the head-permuted view -> FWHT/MMQ D4 writer,
// versus fwht_quantize_mmq_d4_glu_perm reading the GLU operands through the view: every byte and guard.
// Head geometries [hd, nk, rep] -> [hd, rep, nk] (the Qwen3.5 tiled -> grouped order and others, identity
// included), K 4096/8192, rows 1/3/16/512, signs on/off, transform padding, random and adversarial operands
// (zeros, subnormals, extremes, inf, nan, exp overflow edges), eager and graph replays with changed operands.
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "cpy.cuh"
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

struct geometry { int hd, nk, rep; };

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    std::mt19937 rng(20261016);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    const geometry geoms[] = { {128, 16, 2}, {128, 8, 4}, {128, 32, 1}, {64, 32, 2}, {128, 32, 2}, {256, 4, 8}, {96, 32, 4}, {128, 32, 3}, {64, 16, 12}, {4096, 1, 1} };
    int cases = 0;
    for (const geometry & gm : geoms) for (int rows : {1, 3, 16, 512}) for (bool signs_on : {false, true})
    for (bool padded : {false, true}) for (int mode : {0, 1, 2}) {
        // mode 0: random, 1: random with every 7th operand adversarial, 2: adversarial only
        if (rows == 512 && (padded || mode == 1)) continue;
        const int    k       = gm.hd * gm.nk * gm.rep;
        const int    kp      = k + (padded ? 4096 : 0);
        const size_t n_elem  = size_t(k) * rows;
        const size_t qbytes  = size_t(kp) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);
        constexpr size_t guard = 64;

        std::vector<float> hz(n_elem), hn(n_elem), signs(k);
        auto fill = [&]() {
            for (size_t i = 0; i < n_elem; ++i) {
                const bool adv0 = mode == 2 || (mode == 1 && rng() % 7 == 0);
                const bool adv1 = mode == 2 || (mode == 1 && rng() % 7 == 0);
                hz[i] = adv0 ? special[pick(rng)] : dist(rng);
                hn[i] = adv1 ? special[pick(rng)] : dist(rng);
            }
            for (int i = 0; i < k; ++i) signs[i] = (rng() & 1) ? 1.0f : -1.0f;
        };
        fill();

        float * dz = nullptr, * dn = nullptr, * dg = nullptr, * dc = nullptr, * ds = nullptr;
        void * qr = nullptr, * qf = nullptr;
        check(cudaMalloc(&dz, n_elem * sizeof(float)));
        check(cudaMalloc(&dn, n_elem * sizeof(float)));
        check(cudaMalloc(&dg, n_elem * sizeof(float)));
        check(cudaMalloc(&dc, n_elem * sizeof(float)));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        auto upload = [&]() {
            check(cudaMemcpy(dz, hz.data(), n_elem * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dn, hn.data(), n_elem * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, signs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload();

        // the graph's shapes: z and the norm as [hd, nk*rep heads, T], the GLU, then
        // reshape_4d [hd, nk, rep, T] -> permute(0, 2, 1, 3) -> CONT
        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * z    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gm.hd, gm.nk * gm.rep, rows);
        ggml_tensor * nrm  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gm.hd, gm.nk * gm.rep, rows);
        z->data   = dz;
        nrm->data = dn;
        ggml_tensor * glu  = ggml_swiglu_split(ctx, z, nrm);
        glu->data = dg;
        ggml_tensor * r4   = ggml_reshape_4d(ctx, glu, gm.hd, gm.nk, gm.rep, rows);
        ggml_tensor * perm = ggml_permute(ctx, r4, 0, 2, 1, 3);
        perm->data = dg;
        ggml_tensor * cont = ggml_cont(ctx, perm);
        cont->data = dc;

        // view parameters exactly as ggml_cuda_try_fwht_mmq_glu_perm derives them
        if (perm->view_src != glu || perm->nb[0] != sizeof(float) || perm->nb[3] != size_t(k) * sizeof(float) ||
                perm->ne[0] * perm->ne[1] * perm->ne[2] != k || perm->ne[3] != rows) {
            return 3;
        }
        const int     n0 = (int) perm->ne[0];
        const int     n1 = (int) perm->ne[1];
        const int64_t s1 = (int64_t) (perm->nb[1] / sizeof(float));
        const int64_t s2 = (int64_t) (perm->nb[2] / sizeof(float));

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
            ggml_cuda_dup(cuda, cont);
            fwht_quantize_mmq_d4_cuda(dc, GGML_TYPE_F32, signs_on ? ds : nullptr, qr, k, kp, rows, cuda.stream());
            fwht_quantize_mmq_d4_glu_perm_cuda(dz, dn, signs_on ? ds : nullptr, qf, k, kp, rows, k, k, n0, n1, s1, s2,
                                               cuda.stream());
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
                        std::fprintf(stderr, "mismatch hd=%d nk=%d rep=%d rows=%d signs=%d pad=%d mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                     gm.hd, gm.nk, gm.rep, rows, signs_on, padded, mode, capture, i, ref[i], fused[i]);
                        return 1;
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dz)); check(cudaFree(dn)); check(cudaFree(dg)); check(cudaFree(dc)); check(cudaFree(ds));
        check(cudaFree(qr)); check(cudaFree(qf));
    }
    ggml_backend_free(backend);
    std::printf("glu_perm_fwht_equivalence_ok cases=%d bytes_exact=true guards_intact=true changed_input_graph_replays=3\n", cases);
    return 0;
}
