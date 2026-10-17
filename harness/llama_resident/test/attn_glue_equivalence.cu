// C62 attention glue, fused kernels versus the unfused kernels the graph runs, every byte (and guard / untouched
// cache rows):
//  (1) RMS_NORM * w -> MROPE / IMROPE [-> K SET_ROWS (+ V SET_ROWS)]: ggml_cuda_op_rms_norm_fused + ggml_cuda_op_rope
//      + ggml_cuda_op_set_rows (K, V) versus ggml_cuda_op_rms_norm_mul_mrope_fused. Target geometry (256-wide heads,
//      64 rotated dims, IMROPE [11, 11, 10, 0], strided Q view) and drafter geometry (128-wide heads, MROPE [64, 0, 0, 0],
//      YaRN 1/16), plus other sections; Q (16/32 heads), K+V (4/8 heads) into f16 caches; 1/2/16/32 tokens; random and
//      adversarial operands; distinct per-section positions; eager and graph replays with changed inputs.
//  (2) MUL(attn, SIGMOID(CONT(gate view))) -> signed FWHT/MMQ D4: ggml_cuda_dup + ggml_cuda_op_unary_mul +
//      fwht_quantize_mmq_d4_cuda versus fwht_quantize_mmq_d4_sigmoid_gate_cuda; the joint Q|gate view and a contiguous
//      gate; K 4096/8192; rows 1/3/16/512; signs on/off; transform padding.
// The kernels launch as programmatic dependents on a non-blocking stream: outputs are poisoned on that stream and
// the stream drained before every run (no memset in flight when a kernel starts).
#include "ggml-cuda.h"
#include "ggml.h"
#include "common.cuh"
#include "cpy.cuh"
#include "norm.cuh"
#include "quantize.cuh"
#include "rope.cuh"
#include "set-rows.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
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
    88.0f, -88.0f, 88.7f, -88.7f, 89.0f, -89.0f, 103.0f, -103.0f, 104.0f, -104.0f, 126.0f, -126.0f,
    1e6f, -1e6f, 17.0f, -17.0f, 1e-7f, -1e-7f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f,
};

static std::mt19937 rng(20261016);

static float draw(int mode, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::uniform_int_distribution<int> pick(0, int(sizeof(special) / sizeof(special[0])) - 1);
    const bool adv = mode == 2 || (mode == 1 && rng() % 7 == 0);
    return adv ? special[pick(rng)] : dist(rng);
}

struct rope_cfg {
    int hd, n_dims, mode, sections[4], n_ctx_orig;
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
};

static int n_fail = 0;

static bool bytes_equal(const std::vector<unsigned char> & a, const std::vector<unsigned char> & b, const char * what,
                        const char * desc) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (bad++ == 0) first = i;
        }
    }
    if (bad) {
        if (n_fail++ < 12) {
            std::fprintf(stderr, "mismatch %s %s bytes=%zu first=%zu ref=%u fused=%u\n", what, desc, bad, first, a[first], b[first]);
        }
        return false;
    }
    return true;
}

// (1) norm + rope (+ stores)
static int test_mrope(ggml_backend_cuda_context & cuda, int & cases) {
    const rope_cfg cfgs[] = {
        { 256,  64, GGML_ROPE_TYPE_IMROPE, {11, 11, 10, 0}, 262144, 1e7f, 1.0f,      0.0f, 1.0f, 32.0f, 1.0f },  // target
        { 128, 128, GGML_ROPE_TYPE_MROPE,  {64,  0,  0, 0},  16384, 1e7f, 1.0f/16.0f, 1.0f, 1.0f, 32.0f, 1.0f },  // drafter (YaRN)
        { 256,  64, GGML_ROPE_TYPE_MROPE,  {11, 11, 10, 0}, 262144, 1e7f, 1.0f,      0.0f, 1.0f, 32.0f, 1.0f },
        { 128,  96, GGML_ROPE_TYPE_IMROPE, {16, 16, 12, 4},   8192, 1e6f, 0.5f,      0.5f, 0.9f, 32.0f, 1.0f },
        { 128, 128, GGML_ROPE_TYPE_MROPE,  {24, 20, 20, 0},  16384, 1e4f, 1.0f,      0.0f, 1.0f, 32.0f, 1.0f },
    };
    const int n_cache = 1024;
    for (const rope_cfg & rc : cfgs) for (int heads : {4, 16}) for (int T : {1, 2, 16, 32}) for (int store : {0, 1, 2})
    for (bool strided : {false, true}) for (int mode : {0, 1, 2}) {
        if (store > 0 && strided) continue;        // the K input is a contiguous reshape; Q is the strided view
        if (mode == 1 && T == 1) continue;
        const int    hd    = rc.hd;
        const int    rowe  = hd * heads;                 // elements per token row
        const int    xrow  = strided ? 2 * rowe : rowe;  // the Q view of the joint Q|gate projection
        const size_t nx    = size_t(xrow) * T;
        const size_t nv    = size_t(rowe) * T;
        const size_t ncach = size_t(n_cache) * rowe;
        const float  eps   = 1e-6f;

        std::vector<float> hx(nx), hw(hd), hv(nv);
        std::vector<int32_t> hpos(4 * T);
        std::vector<int64_t> hidx(T), hvidx(T);
        auto fill = [&]() {
            for (auto & f : hx) f = draw(mode, -4.0f, 4.0f);
            for (auto & f : hw) f = draw(mode == 2 ? 1 : 0, -2.0f, 2.0f);
            for (auto & f : hv) f = draw(mode, -8.0f, 8.0f);
            for (int s = 0; s < 4; ++s) for (int t = 0; t < T; ++t) hpos[s * T + t] = int32_t(rng() % 8192);
            std::vector<int64_t> perm(n_cache);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int t = 0; t < T; ++t) { hidx[t] = perm[t]; hvidx[t] = perm[n_cache - 1 - t]; }
        };
        fill();

        float * dx, * dw, * dv, * dnorm, * drope_r, * drope_f;
        int32_t * dpos; int64_t * didx, * dvidx;
        ggml_fp16_t * kr, * kf, * vr, * vf;
        check(cudaMalloc(&dx, nx * sizeof(float)));
        check(cudaMalloc(&dw, hd * sizeof(float)));
        check(cudaMalloc(&dv, nv * sizeof(float)));
        check(cudaMalloc(&dnorm, nv * sizeof(float)));
        check(cudaMalloc(&drope_r, nv * sizeof(float)));
        check(cudaMalloc(&drope_f, nv * sizeof(float)));
        check(cudaMalloc(&dpos, 4 * T * sizeof(int32_t)));
        check(cudaMalloc(&didx, T * sizeof(int64_t)));
        check(cudaMalloc(&dvidx, T * sizeof(int64_t)));
        check(cudaMalloc(&kr, ncach * sizeof(ggml_fp16_t)));
        check(cudaMalloc(&kf, ncach * sizeof(ggml_fp16_t)));
        check(cudaMalloc(&vr, ncach * sizeof(ggml_fp16_t)));
        check(cudaMalloc(&vf, ncach * sizeof(ggml_fp16_t)));
        auto upload = [&]() {
            check(cudaMemcpy(dx, hx.data(), nx * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dw, hw.data(), hd * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dv, hv.data(), nv * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dpos, hpos.data(), 4 * T * sizeof(int32_t), cudaMemcpyHostToDevice));
            check(cudaMemcpy(didx, hidx.data(), T * sizeof(int64_t), cudaMemcpyHostToDevice));
            check(cudaMemcpy(dvidx, hvidx.data(), T * sizeof(int64_t), cudaMemcpyHostToDevice));
        };
        upload();

        ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        ggml_tensor * xbase = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, xrow, T);
        xbase->data = dx;
        ggml_tensor * x = ggml_view_3d(ctx, xbase, hd, heads, T, (strided ? 2 : 1) * hd * sizeof(float), xbase->nb[1], 0);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        w->data = dw;
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
        pos->data = dpos;
        ggml_tensor * rms = ggml_rms_norm(ctx, x, eps);
        ggml_tensor * mul = ggml_mul(ctx, rms, w);
        mul->data = dnorm;
        int sections[4] = { rc.sections[0], rc.sections[1], rc.sections[2], rc.sections[3] };
        ggml_tensor * rope = ggml_rope_multi(ctx, mul, pos, nullptr, rc.n_dims, sections, rc.mode, rc.n_ctx_orig,
                                             rc.freq_base, rc.freq_scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        rope->data = drope_r;
        ggml_tensor * kc_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowe, n_cache); kc_r->data = kr;
        ggml_tensor * kc_f = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowe, n_cache); kc_f->data = kf;
        ggml_tensor * vc_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowe, n_cache); vc_r->data = vr;
        ggml_tensor * vc_f = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rowe, n_cache); vc_f->data = vf;
        ggml_tensor * idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T); idx->data = didx;
        ggml_tensor * vidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T); vidx->data = dvidx;
        ggml_tensor * kview = ggml_view_2d(ctx, rope, rowe, T, rope->nb[2], 0);
        ggml_tensor * vsrc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, heads, T); vsrc->data = dv;
        ggml_tensor * vview = ggml_view_2d(ctx, vsrc, rowe, T, vsrc->nb[2], 0);
        ggml_tensor * ks_r = ggml_set_rows(ctx, kc_r, kview, idx);
        ggml_tensor * vs_r = ggml_set_rows(ctx, vc_r, vview, vidx);
        // fused-side graph objects (same shapes, own outputs)
        ggml_tensor * rope_f = ggml_rope_multi(ctx, mul, pos, nullptr, rc.n_dims, sections, rc.mode, rc.n_ctx_orig,
                                               rc.freq_base, rc.freq_scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        rope_f->data = drope_f;
        ggml_tensor * kview_f = ggml_view_2d(ctx, rope_f, rowe, T, rope_f->nb[2], 0);
        ggml_tensor * ks_f = ggml_set_rows(ctx, kc_f, kview_f, idx);
        ggml_tensor * vs_f = ggml_set_rows(ctx, vc_f, vview, vidx);
        if (!ggml_cuda_rms_norm_mul_mrope_supported(rms, mul, rope_f)) {
            std::fprintf(stderr, "unsupported config hd=%d n_dims=%d mode=%d\n", hd, rc.n_dims, rc.mode);
            return 3;
        }

        std::vector<unsigned char> a(nv * sizeof(float)), b(nv * sizeof(float));
        std::vector<unsigned char> ca(ncach * 2), cb(ncach * 2);
        auto poison = [&]() {
            check(cudaMemsetAsync(drope_r, 0xa5, nv * sizeof(float), cuda.stream()));
            check(cudaMemsetAsync(drope_f, 0xa5, nv * sizeof(float), cuda.stream()));
            check(cudaMemsetAsync(dnorm, 0x5a, nv * sizeof(float), cuda.stream()));
            check(cudaMemsetAsync(kr, 0xa5, ncach * 2, cuda.stream()));
            check(cudaMemsetAsync(kf, 0xa5, ncach * 2, cuda.stream()));
            check(cudaMemsetAsync(vr, 0xa5, ncach * 2, cuda.stream()));
            check(cudaMemsetAsync(vf, 0xa5, ncach * 2, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        };
        char desc[256];
        std::snprintf(desc, sizeof(desc), "hd=%d n_dims=%d mode=%d sec=[%d,%d,%d,%d] heads=%d T=%d store=%d strided=%d input=%d",
                      hd, rc.n_dims, rc.mode, sections[0], sections[1], sections[2], sections[3], heads, T, store, strided, mode);
        for (bool capture : {false, true}) {
            poison();
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            ggml_cuda_op_rms_norm_fused(cuda, rms, mul);
            ggml_cuda_op_rope(cuda, rope);
            if (store >= 1) ggml_cuda_op_set_rows(cuda, ks_r);
            if (store >= 2) ggml_cuda_op_set_rows(cuda, vs_r);
            ggml_cuda_op_rms_norm_mul_mrope_fused(cuda, rms, mul, rope_f, store >= 1 ? ks_f : nullptr,
                                                  store >= 2 ? vview : nullptr, store >= 2 ? vs_f : nullptr);
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (capture) {
                check(cudaStreamEndCapture(cuda.stream(), &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
                if (capture) {
                    if (replay) { fill(); upload(); poison(); }
                    check(cudaGraphLaunch(exec, cuda.stream()));
                }
                check(cudaStreamSynchronize(cuda.stream()));
                if (store == 0) {
                    check(cudaMemcpy(a.data(), drope_r, a.size(), cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(b.data(), drope_f, b.size(), cudaMemcpyDeviceToHost));
                    bytes_equal(a, b, "rope", desc);
                } else {
                    check(cudaMemcpy(ca.data(), kr, ca.size(), cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(cb.data(), kf, cb.size(), cudaMemcpyDeviceToHost));
                    bytes_equal(ca, cb, "k_cache", desc);
                    // the fused K store does not write the F32 rope tensor
                    check(cudaMemcpy(b.data(), drope_f, b.size(), cudaMemcpyDeviceToHost));
                    for (unsigned char c : b) if (c != 0xa5) { std::fprintf(stderr, "rope tensor written %s\n", desc); return 1; }
                    check(cudaMemcpy(ca.data(), vr, ca.size(), cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(cb.data(), vf, cb.size(), cudaMemcpyDeviceToHost));
                    if (store == 1) {
                        for (unsigned char c : cb) if (c != 0xa5) { std::fprintf(stderr, "v cache written %s\n", desc); return 1; }
                    } else {
                        bytes_equal(ca, cb, "v_cache", desc);
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dx)); check(cudaFree(dw)); check(cudaFree(dv)); check(cudaFree(dnorm));
        check(cudaFree(drope_r)); check(cudaFree(drope_f)); check(cudaFree(dpos)); check(cudaFree(didx)); check(cudaFree(dvidx));
        check(cudaFree(kr)); check(cudaFree(kf)); check(cudaFree(vr)); check(cudaFree(vf));
    }
    return 0;
}

// (2) sigmoid gate in front of the signed transform
static int test_gate(ggml_backend_cuda_context & cuda, int & cases) {
    for (int multiple : {1, 2}) for (int rows : {1, 3, 16, 512}) for (bool joint : {true, false}) for (bool signs_on : {false, true})
    for (bool padded : {false, true}) for (int mode : {0, 1, 2}) {
        if (rows == 512 && (padded || mode == 1)) continue;
        const int    k      = 4096 * multiple;
        const int    kp     = k + (padded ? 4096 : 0);
        const int    hd     = 256, nh = k / hd;
        const int    qrow   = joint ? 2 * k : k;          // the joint projection row: [q head | gate head] per head
        const size_t nq     = size_t(qrow) * rows;
        const size_t na     = size_t(k) * rows;
        const size_t qbytes = size_t(kp) * rows / QK8_1_MMQ * sizeof(block_q8_1_mmq);
        constexpr size_t guard = 64;

        std::vector<float> hq(nq), ha(na), hs(k);
        auto fill = [&]() {
            for (auto & f : hq) f = draw(mode, -12.0f, 12.0f);
            for (auto & f : ha) f = draw(mode, -8.0f, 8.0f);
            for (auto & f : hs) f = (rng() & 1) ? 1.0f : -1.0f;
        };
        fill();
        float * dq, * da, * ds, * dcont, * dsig, * dmul;
        void * qr, * qf;
        check(cudaMalloc(&dq, nq * sizeof(float)));
        check(cudaMalloc(&da, na * sizeof(float)));
        check(cudaMalloc(&ds, k * sizeof(float)));
        check(cudaMalloc(&dcont, na * sizeof(float)));
        check(cudaMalloc(&dsig, na * sizeof(float)));
        check(cudaMalloc(&dmul, na * sizeof(float)));
        check(cudaMalloc(&qr, qbytes + guard));
        check(cudaMalloc(&qf, qbytes + guard));
        auto upload = [&]() {
            check(cudaMemcpy(dq, hq.data(), nq * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(da, ha.data(), na * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(ds, hs.data(), k * sizeof(float), cudaMemcpyHostToDevice));
        };
        upload();

        ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_tensor * qfull = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, qrow, rows);
        qfull->data = dq;
        ggml_tensor * gview = joint
            ? ggml_view_3d(ctx, qfull, hd, nh, rows, 2 * hd * sizeof(float), qfull->nb[1], hd * sizeof(float))
            : ggml_view_2d(ctx, qfull, k, rows, qfull->nb[1], 0);
        ggml_tensor * cont = ggml_cont_2d(ctx, gview, k, rows);
        cont->data = dcont;
        ggml_tensor * sig = ggml_sigmoid(ctx, cont);
        sig->data = dsig;
        ggml_tensor * attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        attn->data = da;
        ggml_tensor * mulg = ggml_mul(ctx, attn, sig);
        mulg->data = dmul;
        // view parameters exactly as ggml_cuda_try_fwht_mmq_sigmoid_gate derives them
        int64_t n0 = gview->ne[0], g1, gt;
        if (gview->ne[0] * gview->ne[1] == k && gview->ne[2] == rows) {
            g1 = gview->nb[1] / sizeof(float); gt = gview->nb[2] / sizeof(float);
        } else {
            g1 = 0; gt = gview->nb[1] / sizeof(float);
        }

        std::vector<unsigned char> ref(qbytes + guard), fused(qbytes + guard);
        auto poison = [&]() {
            check(cudaMemsetAsync(qr, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaMemsetAsync(qf, 0xa5, qbytes + guard, cuda.stream()));
            check(cudaStreamSynchronize(cuda.stream()));
        };
        for (bool capture : {false, true}) {
            poison();
            if (capture) check(cudaStreamBeginCapture(cuda.stream(), cudaStreamCaptureModeGlobal));
            ggml_cuda_dup(cuda, cont);
            ggml_cuda_op_unary_mul(cuda, sig, mulg);
            fwht_quantize_mmq_d4_cuda(dmul, GGML_TYPE_F32, signs_on ? ds : nullptr, qr, k, kp, rows, cuda.stream());
            fwht_quantize_mmq_d4_sigmoid_gate_cuda((const float *) gview->data, da, signs_on ? ds : nullptr, qf, k, kp, rows,
                                                   (int) n0, g1, gt, k, cuda.stream());
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (capture) {
                check(cudaStreamEndCapture(cuda.stream(), &graph));
                check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
                if (capture) {
                    if (replay) { fill(); upload(); poison(); }
                    check(cudaGraphLaunch(exec, cuda.stream()));
                }
                check(cudaStreamSynchronize(cuda.stream()));
                check(cudaMemcpy(ref.data(), qr, qbytes + guard, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(fused.data(), qf, qbytes + guard, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < qbytes + guard; ++i) {
                    if (ref[i] != fused[i] || (i >= qbytes && (ref[i] != 0xa5 || fused[i] != 0xa5))) {
                        if (n_fail++ < 24) {
                            std::fprintf(stderr, "gate mismatch k=%d rows=%d joint=%d signs=%d pad=%d mode=%d graph=%d byte=%zu ref=%u fused=%u\n",
                                         k, rows, joint, signs_on, padded, mode, capture, i, ref[i], fused[i]);
                        }
                        break;
                    }
                }
                ++cases;
            }
            if (capture) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
        }
        ggml_free(ctx);
        check(cudaFree(dq)); check(cudaFree(da)); check(cudaFree(ds)); check(cudaFree(dcont)); check(cudaFree(dsig));
        check(cudaFree(dmul)); check(cudaFree(qr)); check(cudaFree(qf));
    }
    return 0;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) return 2;
    ggml_backend_cuda_context cuda(0);
    int mrope_cases = 0, gate_cases = 0;
    int rc = test_mrope(cuda, mrope_cases);
    const int mrope_fail = n_fail;
    rc |= test_gate(cuda, gate_cases);
    ggml_backend_free(backend);
    if (rc || n_fail) {
        std::fprintf(stderr, "attn_glue_equivalence_FAILED mrope_fail=%d gate_fail=%d rc=%d\n", mrope_fail, n_fail - mrope_fail, rc);
        return 1;
    }
    std::printf("attn_glue_equivalence_ok mrope_cases=%d gate_cases=%d bytes_exact=true untouched_rows_intact=true changed_input_graph_replays=3\n",
                mrope_cases, gate_cases);
    return 0;
}
