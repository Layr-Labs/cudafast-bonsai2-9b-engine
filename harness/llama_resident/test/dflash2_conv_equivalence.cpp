// Exact check of GGML_OP_DFLASH2_CONV (ggml_dflash2_conv) on the CUDA backend against the unfused chain the
// DFlash2 graph built before (build_dflash2_conv in llama.cpp/src/models/dflash.cpp: reshape/view/cont of the
// dynamic coefficients, repeat over the group, add of the static base, one fill+concat shift per extra tap, one
// mul per tap, one add per extra tap, then the residual add). Both graphs run on CUDA (so ggml-cuda's own fusions
// of the chain apply as in the model); outputs must match bit for bit. Cases: n_embd 4096 and 256, group 16,
// kernel 2 and 3, 16 tokens in 1 block and 32 in 2 blocks, both sides, with and without the residual, random and
// adversarial operands (signed zeros, subnormals, inf, nan, extremes).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    int64_t H, T, block, kernel;
    int     side;
    bool    residual;
    int     mode; // 0 random, 1 random + every 7th adversarial, 2 adversarial only
};

const float special[] = { 0.0f, -0.0f, 1e-45f, -1e-45f, 1e-40f, -1e-40f, FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX,
                          1e30f, -1e30f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f, 0.5f, -2.0f, 1e-7f };

ggml_tensor * chain(ggml_context * ctx0, ggml_tensor * hidden, ggml_tensor * dynamic, ggml_tensor * base, int side,
                    int64_t block_size, int64_t group_size, ggml_tensor * residual) {
    const int64_t hidden_size = hidden->ne[0];
    const int64_t n_tokens    = hidden->ne[1];
    const int64_t n_blocks    = n_tokens / block_size;
    const int64_t kernel_size = base->ne[1];
    const int64_t n_groups    = hidden_size / group_size;

    ggml_tensor * blocks = ggml_reshape_3d(ctx0, hidden, hidden_size, block_size, n_blocks);
    ggml_tensor * coeffs = ggml_reshape_4d(ctx0, dynamic, n_groups, kernel_size, 2, n_tokens);
    ggml_tensor * coeffs_side = ggml_view_3d(ctx0, coeffs, n_groups, kernel_size, n_tokens,
            coeffs->nb[1], coeffs->nb[3], side * coeffs->nb[2]);
    ggml_tensor * coeff_all = ggml_cont(ctx0, coeffs_side);
    coeff_all = ggml_reshape_4d(ctx0, coeff_all, 1, n_groups, kernel_size, n_tokens);
    coeff_all = ggml_repeat_4d(ctx0, coeff_all, group_size, n_groups, kernel_size, n_tokens);
    ggml_tensor * base_side = ggml_reshape_4d(ctx0,
            ggml_view_1d(ctx0, base, hidden_size * kernel_size, side * base->nb[2]),
            group_size, n_groups, kernel_size, 1);
    ggml_tensor * weight_all = ggml_add(ctx0, coeff_all, base_side);

    ggml_tensor * result = nullptr;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
        ggml_tensor * values = blocks;
        if (tap > 0) {
            ggml_tensor * zeros = ggml_fill(ctx0,
                    ggml_new_tensor_3d(ctx0, hidden->type, hidden_size, std::min(tap, block_size), n_blocks), 0.0f);
            if (tap < block_size) {
                ggml_tensor * previous = ggml_view_3d(ctx0, blocks, hidden_size, block_size - tap, n_blocks,
                        blocks->nb[1], blocks->nb[2], 0);
                values = ggml_concat(ctx0, zeros, previous, 1);
            } else {
                values = zeros;
            }
        }
        values = ggml_reshape_2d(ctx0, values, hidden_size, n_tokens);
        ggml_tensor * weight = ggml_reshape_2d(ctx0,
                ggml_cont(ctx0, ggml_view_4d(ctx0, weight_all, group_size, n_groups, 1, n_tokens,
                        weight_all->nb[1], weight_all->nb[2], weight_all->nb[3], tap * weight_all->nb[2])),
                hidden_size, n_tokens);
        ggml_tensor * term = ggml_mul(ctx0, weight, values);
        result = result ? ggml_add(ctx0, result, term) : term;
    }
    return residual ? ggml_add(ctx0, result, residual) : result;
}

std::vector<float> run(ggml_backend_t backend, const Case & cs, const std::vector<float> & x, const std::vector<float> & dyn,
                       const std::vector<float> & base, const std::vector<float> & res, bool fused) {
    const int64_t G = 16;
    ggml_init_params ip_in = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_in = ggml_init(ip_in);
    ggml_tensor * t_x    = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, cs.H, cs.T);
    ggml_tensor * t_dyn  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, (cs.H / G) * cs.kernel * 2, cs.T);
    ggml_tensor * t_base = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, cs.H, cs.kernel, 2);
    ggml_tensor * t_res  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, cs.H, cs.T);
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    ggml_backend_tensor_set(t_x, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(t_dyn, dyn.data(), 0, dyn.size() * sizeof(float));
    ggml_backend_tensor_set(t_base, base.data(), 0, base.size() * sizeof(float));
    ggml_backend_tensor_set(t_res, res.data(), 0, res.size() * sizeof(float));

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph *  gf  = ggml_new_graph(ctx);
    // allocator-owned producers, like the projections in the model
    ggml_tensor * hx = ggml_scale(ctx, t_x, 1.0f);
    ggml_tensor * hd = ggml_scale(ctx, t_dyn, 1.0f);
    ggml_tensor * hr = cs.residual ? ggml_scale(ctx, t_res, 1.0f) : nullptr;
    ggml_tensor * out = fused ? ggml_dflash2_conv(ctx, hx, hd, t_base, cs.side, (int) cs.block, (int) G, hr)
                              : chain(ctx, hx, hd, t_base, cs.side, cs.block, G, hr);
    ggml_tensor * y = ggml_scale(ctx, out, 1.0f); // a consumer, so the op's output is an intermediate
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> h((size_t) (cs.H * cs.T));
    ggml_backend_tensor_get(y, h.data(), 0, h.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
    return h;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) {
        fprintf(stderr, "no CUDA backend\n");
        return 2;
    }
    std::mt19937 rng(20261016);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::uniform_int_distribution<int> pick(0, (int) (sizeof(special) / sizeof(special[0])) - 1);
    int failures = 0, cases = 0;
    for (int64_t H : { 4096, 256 }) for (int64_t kernel : { 2, 3 }) for (int64_t T : { 16, 32 })
    for (int side : { 0, 1 }) for (bool residual : { false, true }) for (int mode : { 0, 1, 2 }) {
        const Case cs = { H, T, 16, kernel, side, residual, mode };
        auto gen = [&](size_t n) {
            std::vector<float> v(n);
            for (float & f : v) {
                const bool adv = mode == 2 || (mode == 1 && rng() % 7 == 0);
                f = adv ? special[pick(rng)] : dist(rng);
            }
            return v;
        };
        const auto x    = gen((size_t) (H * T));
        const auto dyn  = gen((size_t) ((H / 16) * kernel * 2 * T));
        const auto base = gen((size_t) (H * kernel * 2));
        const auto res  = gen((size_t) (H * T));
        const auto ref  = run(backend, cs, x, dyn, base, res, false);
        const auto cand = run(backend, cs, x, dyn, base, res, true);
        size_t diff = 0, nan_pairs = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            uint32_t a, b;
            memcpy(&a, &ref[i], 4);
            memcpy(&b, &cand[i], 4);
            if (a != b) {
                if (std::isnan(ref[i]) && std::isnan(cand[i])) {
                    ++nan_pairs;
                } else {
                    ++diff;
                }
            }
        }
        ++cases;
        if (diff) {
            ++failures;
            printf("FAIL H=%lld K=%lld T=%lld side=%d res=%d mode=%d: %zu differing values\n", (long long) H,
                   (long long) kernel, (long long) T, side, (int) residual, mode, diff);
        } else if (nan_pairs) {
            printf("ok   H=%lld K=%lld T=%lld side=%d res=%d mode=%d (%zu NaN payload differences)\n", (long long) H,
                   (long long) kernel, (long long) T, side, (int) residual, mode, nan_pairs);
        }
    }
    printf("dflash2-conv-equivalence: %d cases, %d failures\n", cases, failures);
    ggml_backend_free(backend);
    return failures ? 1 : 0;
}
