#include "gated_delta_net.cuh"
#include "gated_delta_net_kernel.cuh"
#include "scale.cuh"
#include "state-clear.cuh"

#include <cstdlib>

template <bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * rb_d, const float * ra_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const int32_t * s_ids, int64_t s_row_stride, int lazy, const float * const zero[3], cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int num_warps = 4;
    // Same predicate as the kernel, fed the arch compiled for this cc.
    const int cols_per_warp = GGML_CUDA_CC_IS_NVIDIA(cc) ? gdn_cols_per_warp(ggml_cuda_highest_compiled_arch(cc), S_v, KDA) : 1;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * cols_per_warp - 1) / (num_warps * cols_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
            break;
        }
        case 128: {
            // GB10 ubatches of >= GDN_L8_MIN_TOKENS tokens (prefill, the 16-row verify): the lane-permuted reduction
            // variant (bit-identical, see the kernel).
            // GGML_CUDA_GDN_L8=0 disables it.
            static const bool l8_enabled = getenv("GGML_CUDA_GDN_L8") == nullptr || atoi(getenv("GGML_CUDA_GDN_L8")) != 0;
            // the 16-row verify kernel's state, base and record traffic with the streaming (evict-first) operators;
            // GGML_CUDA_GDN_STATE_EVICT=0 keeps the normal ones (cache policy only, same bits)
            static const bool gdn_state_evict = getenv("GGML_CUDA_GDN_STATE_EVICT") == nullptr || atoi(getenv("GGML_CUDA_GDN_STATE_EVICT")) != 0;
            if constexpr (keep_rs_t && !KDA) {
                // the lazy condition gated_delta_net_cuda evaluates on the device
                const bool lazy_eff = lazy && state_slot_stride > 0 && 2 + (int) grid_dims.z <= K;
                if (l8_enabled && cc == GGML_CUDA_CC_DGX_SPARK && warp_size == 32 && cols_per_warp == 4 &&
                        n_tokens >= GDN_L8V_MIN_TOKENS && n_tokens <= GDN_L8V_MAX_TOKENS && (!lazy_eff || n_tokens <= K)) {
                    if (lazy_eff) {
                        const auto kernel = n_tokens == 16
                            ? gated_delta_net_l8v_cuda<RAW, G_PRECOMPUTED, true, 16>
                            : gated_delta_net_l8v_cuda<RAW, G_PRECOMPUTED, true>;
                        ggml_cuda_kernel_launch(kernel, launch_params,
                            q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride,
                            1 | (gdn_state_evict ? 8 : 0));
                    } else {
                        const auto kernel = n_tokens == 16
                            ? gated_delta_net_l8v_cuda<RAW, G_PRECOMPUTED, false, 16>
                            : gated_delta_net_l8v_cuda<RAW, G_PRECOMPUTED, false>;
                        ggml_cuda_kernel_launch(kernel, launch_params,
                            q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride,
                            gdn_state_evict ? 8 : 0);
                    }
                    break;
                }
                if (l8_enabled && cc == GGML_CUDA_CC_DGX_SPARK && warp_size == 32 && cols_per_warp == 4 && n_tokens >= GDN_L8_MIN_TOKENS) {
                    ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<RAW, G_PRECOMPUTED>, launch_params,
                        q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy,
                        zero[0], zero[0] ? *zero[1] : 0.0f, zero[0] ? *zero[2] : 0.0f);
                    break;
                }
            }
            GGML_ASSERT(zero[0] == nullptr);
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride, lazy);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // fused state gather, registered for this node by this context's graph evaluator (ggml_cuda_try_gdn_gather_skip)
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    const ggml_tensor * zero_node = nullptr;
    if (const ggml_cuda_gated_delta_net_gather * gather = ctx.gdn_gathers().find(dst)) {
        s_d          = gather->base;
        s_ids        = gather->ids;
        s_row_stride = gather->row_stride;
        zero_node    = gather->zero_node;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // Lazy rollback needs the cache on both ends: the fused gather reads the row the rollback
    // selected and the fused scatter writes the slots (see gated_delta_net_kernel.cuh).
    // GGML_CUDA_GDN_LAZY_ROLLBACK=0 restores the per-token snapshots.
    static const bool lazy_disabled = getenv("GGML_CUDA_GDN_LAZY_ROLLBACK") != nullptr &&
                                      atoi(getenv("GGML_CUDA_GDN_LAZY_ROLLBACK")) == 0;
    const int lazy = !lazy_disabled && keep_rs && !kda && cache != nullptr && s_ids != nullptr && n_seqs == 1 ? 1 : 0;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // GB10 per-request state clear (C68): the cache bytes this launch writes (state-clear.cuh), from the same
    // predicates as the kernels. Per (seq, head) row of S_v*S_v floats: lazy: rollback slots 0 (final state) and 1
    // (window base) in full, slots 2 .. 1 + grid_z the record (header + W tokens); eager: slots 0 .. min(T, K) - 1
    // in full; K == 1: slot 0.
    if (cache != nullptr && ggml_cuda_dirty_tracking()) {
        const int     cc_fp      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const int     cpw        = GGML_CUDA_CC_IS_NVIDIA(cc_fp) ? gdn_cols_per_warp(ggml_cuda_highest_compiled_arch(cc_fp), S_v, kda) : 1;
        const int64_t grid_z     = (S_v + 4*cpw - 1) / (4*cpw);
        const size_t  row_bytes  = (size_t) (S_v*S_v)*sizeof(float);
        const size_t  slot_bytes = row_bytes*(size_t) (H*n_seqs);
        const float * base       = cache->data;
        const int64_t ss         = cache->slot_stride;
        if (!keep_rs || ss == 0) {
            ggml_cuda_dirty_mark(base, slot_bytes, stream);
        } else if (lazy && 2 + grid_z <= K) {
            ggml_cuda_dirty_mark(base, slot_bytes, stream);
            ggml_cuda_dirty_mark(base + ss, slot_bytes, stream);
            const int64_t W   = std::min<int64_t>(n_tokens, K);
            const size_t  rec = (size_t) (GDN_LAZY_HDR + W*(S_v + GDN_LAZY_NWARPS*cpw + 2))*sizeof(float);
            for (int64_t z = 0; z < grid_z; ++z) {
                ggml_cuda_dirty_mark_2d(base + (2 + z)*ss, std::min(rec, row_bytes), row_bytes, (size_t) (H*n_seqs), stream);
            }
        } else {
            for (int64_t t = 0; t < std::min<int64_t>(n_tokens, K); ++t) {
                ggml_cuda_dirty_mark(base + t*ss, slot_bytes, stream);
            }
        }
    }

    const bool    raw  = ggml_get_op_params_i32(dst, 1) != 0;
    const float * rb_d = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * ra_d = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!(raw && kda)); // raw gates are defined for the scalar gate only

    // GB10 long-prompt path: exp(g) once per (token, head) instead of once per column-warp.
    // Only for activated gates; with raw gates (#165) the activation happens inside the kernel.
    ggml_cuda_pool_alloc<float> g_exp_alloc(ctx.pool());
    bool g_precomputed = false;
    if (!kda && !raw && S_v == 128 && n_tokens >= 32 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK) {
        const int64_t n_g = ggml_nelements(src_g);
        g_exp_alloc.alloc(n_g);
        const int block = 256;
        const int grid = std::min<int64_t>((n_g + block - 1)/block, 4096);
        gdn_precompute_exp<<<grid, block, 0, stream>>>(g_d, g_exp_alloc.ptr, n_g);
        g_d = g_exp_alloc.ptr;
        g_precomputed = true;
    }

    // GB10 prefill with raw gates on the lane-permuted path: activate the gates once per (token, head) and run the
    // G_PRECOMPUTED kernel on them (gdn_precompute_raw_gates uses the RAW kernel's expressions).
    ggml_cuda_pool_alloc<float> raw_g_alloc(ctx.pool());
    ggml_cuda_pool_alloc<float> raw_b_alloc(ctx.pool());
    bool raw_precomputed = false;
    {
        static const bool l8_enabled = getenv("GGML_CUDA_GDN_L8") == nullptr || atoi(getenv("GGML_CUDA_GDN_L8")) != 0;
        if (raw && keep_rs && l8_enabled && S_v == 128 && n_tokens >= GDN_L8_RAW_PRE_MIN_TOKENS &&
                ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK &&
                ggml_nelements(src_g) == ggml_nelements(src_beta) && src_g->ne[0] == 1 && sb1 == 1 && nbb1 == sizeof(float)) {
            const int64_t n_g = ggml_nelements(src_g);
            raw_g_alloc.alloc(n_g);
            raw_b_alloc.alloc(n_g);
            const int block = 256;
            const int grid  = (int) std::min<int64_t>((n_g + block - 1)/block, 4096);
            gdn_precompute_raw_gates<<<grid, block, 0, stream>>>(g_d, b_d, rb_d, ra_d, raw_g_alloc.ptr, raw_b_alloc.ptr,
                                                                  (int) H, n_g);
            g_d = raw_g_alloc.ptr;
            b_d = raw_b_alloc.ptr;
            raw_precomputed = true;
        }
    }

    // a deferred zeroing SCALE of a cache row: gated_delta_net_l8_cuda applies it on load when it also overwrites that
    // row (slot 0 of the cell it writes, n_seqs == 1); otherwise the SCALE runs now, before the kernel reads the row
    const float * zero[3] = { nullptr, nullptr, nullptr };
    float         zero_sb[2];
    if (zero_node != nullptr) {
        static const bool l8_on = getenv("GGML_CUDA_GDN_L8") == nullptr || atoi(getenv("GGML_CUDA_GDN_L8")) != 0;
        const int  cc_z  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const bool l8    = l8_on && !kda && keep_rs && S_v == 128 && cc_z == GGML_CUDA_CC_DGX_SPARK &&
                           ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size == 32 &&
                           gdn_cols_per_warp(ggml_cuda_highest_compiled_arch(cc_z), S_v, false) == 4 &&
                           n_tokens >= GDN_L8_MIN_TOKENS && n_tokens > GDN_L8V_MAX_TOKENS;
        if (l8 && cache != nullptr && n_seqs == 1 && (const float *) zero_node->data == state_d &&
                ggml_nbytes(zero_node) == (size_t) S_v * S_v * H * sizeof(float)) {
            memcpy(&zero_sb[0], (const float *) zero_node->op_params + 0, sizeof(float));
            memcpy(&zero_sb[1], (const float *) zero_node->op_params + 1, sizeof(float));
            zero[0] = (const float *) zero_node->data;
            zero[1] = &zero_sb[0];
            zero[2] = &zero_sb[1];
        } else {
            ggml_cuda_op_scale(ctx, const_cast<ggml_tensor *>(zero_node));
        }
    }

#define GDN_LAUNCH(KDA_, KEEP_, RAW_, PRE_)                                                       \
    launch_gated_delta_net<KDA_, KEEP_, RAW_, PRE_>(q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                    \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, lazy, zero, stream)

    if (kda) {
        if (keep_rs) { GDN_LAUNCH(true,  true,  false, false); } else { GDN_LAUNCH(true,  false, false, false); }
    } else if (raw_precomputed) {
        GDN_LAUNCH(false, true,  false, true);
    } else if (raw) {
        if (keep_rs) { GDN_LAUNCH(false, true,  true,  false); } else { GDN_LAUNCH(false, false, true,  false); }
    } else {
        if (g_precomputed) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, true);  } else { GDN_LAUNCH(false, false, false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false); } else { GDN_LAUNCH(false, false, false, false); }
        }
    }
#undef GDN_LAUNCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
