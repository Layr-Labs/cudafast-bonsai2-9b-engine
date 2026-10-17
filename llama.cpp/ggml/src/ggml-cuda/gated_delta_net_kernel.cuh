#pragma once

// Gated delta net recurrence kernel; included by gated_delta_net.cu and its standalone test.
#include "common.cuh"


// Host and device must agree on columns/warp. Kernel uses __CUDA_ARCH__; host uses ggml_cuda_highest_compiled_arch(cc). NVIDIA Ampere+ and S_v==128 and !KDA -> 4; HIP/MUSA stay 1.
static constexpr __host__ __device__ int gdn_cols_per_warp(int arch, int S_v, bool KDA) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void) arch; (void) S_v; (void) KDA;
    return 1;
#else
    return GGML_CUDA_CC_IS_NVIDIA(arch) && arch >= GGML_CUDA_CC_AMPERE && S_v == 128 && !KDA ? 4 : 1;
#endif
}

// warp_reduce_sum for N independent values: one butterfly step for all of them at a time, so their
// shuffle latencies overlap. Each value goes through exactly warp_reduce_sum's shuffles and adds.
template <int width, int N>
static __device__ __forceinline__ void gdn_warp_reduce_sum_n(float (&x)[N]) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int c = 0; c < N; ++c) {
            x[c] += __shfl_xor_sync(0xffffffff, x[c], offset, width);
        }
    }
}

// GB10 prefill with raw gates: beta and g activated once per (token, head) with gated_delta_net_cuda's RAW
// expressions (the per-token values it would form in every column warp), ready for G_PRECOMPUTED.
// g and beta are contiguous [1, H, n_tokens, n_seqs], so the head of element i is i % H.
static __global__ void gdn_precompute_raw_gates(const float * g, const float * beta, const float * raw_dt_bias,
                                                const float * raw_a, float * g_exp, float * beta_act, const int H,
                                                const int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x*gridDim.x) {
        const int   h = (int) (i % H);
        const float x = g[i] + raw_dt_bias[h];
        const float g0 = raw_a[h] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        g_exp[i]    = expf(g0);
        beta_act[i] = 1.0f / (1.0f + expf(-beta[i]));
    }
}

static __global__ void gdn_precompute_exp(const float * g, float * g_exp, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x*gridDim.x) {
        g_exp[i] = expf(g[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// Lazy recurrent rollback (keep_rs_t with the fused cache gather and scatter).
//
// The eager scheme writes the full state after each of the last K tokens into rollback slots
// 0..K-1 (2 MiB per slot for 32 heads x 128 x 128 f32), although the next ubatch reads at most
// one of them. The lazy scheme writes the final state to slot 0, the state at the start of the
// rollback window to slot 1, and each block's per-token inputs to slot 2 + blockIdx.z. When the
// next ubatch reads slot r >= 1 (r tokens rolled back), the block reloads slot 1 and replays the
// first W - r window tokens with the same per-column arithmetic as the main loop, which
// reproduces the eager snapshot bit for bit. A record that does not match (magic, window,
// per-warp checksum of the stored base) falls back to reading slot r as written by the eager
// scheme.

#define GDN_LAZY_MAGIC  0x4744314cu   // "GD1L"
#define GDN_LAZY_HDR    8             // magic, W, reserved x2, per-warp base checksum x4
#define GDN_LAZY_NWARPS 4

template <int S_v, int cols_per_warp>
static constexpr __host__ __device__ int gdn_lazy_token_stride() {
    return S_v + GDN_LAZY_NWARPS * cols_per_warp + 2;  // k, v of the block's columns, g, beta
}

static __device__ __forceinline__ uint32_t gdn_lazy_mix(uint32_t h, const uint32_t x) {
    h ^= x + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
}

template <int warp_size>
static __device__ __forceinline__ uint32_t gdn_lazy_warp_hash(uint32_t h) {
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xffffffff, h, offset, warp_size);
        h = gdn_lazy_mix(h ^ (uint32_t) offset, o) ^ gdn_lazy_mix(o ^ (uint32_t) offset, h);
    }
    return h;
}

template <int S_v, int warp_size, int rows_per_lane, int cols_per_warp>
static __device__ __forceinline__ uint32_t gdn_lazy_store_base(
        const float (&s_shard)[cols_per_warp][rows_per_lane], float * base, const int col, const int lane) {
    uint32_t h = 0x811c9dc5u ^ (uint32_t) lane;
#pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            base[(col + c) * S_v + i] = s_shard[c][r];
            h = gdn_lazy_mix(h, __float_as_uint(s_shard[c][r]));
        }
    }
    return gdn_lazy_warp_hash<warp_size>(h);
}

template <int S_v, int warp_size, int rows_per_lane, int cols_per_warp>
static __device__ __forceinline__ void gdn_lazy_store_inputs(
        float * rec, const int tw, const float (&k_reg)[rows_per_lane], const float * v_t, const int col,
        const int lane, const float g_val, const float beta_val) {
    float * tok = rec + GDN_LAZY_HDR + tw * gdn_lazy_token_stride<S_v, cols_per_warp>();
    if (threadIdx.y == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            tok[r * warp_size + lane] = k_reg[r];
        }
        if (lane == 0) {
            tok[S_v + GDN_LAZY_NWARPS * cols_per_warp + 0] = g_val;
            tok[S_v + GDN_LAZY_NWARPS * cols_per_warp + 1] = beta_val;
        }
    }
    if (lane < cols_per_warp) {
        tok[S_v + threadIdx.y * cols_per_warp + lane] = v_t[col + lane];
    }
}

static __device__ __forceinline__ void gdn_lazy_store_header(float * rec, const int W, const uint32_t cs, const int lane) {
    uint32_t * hdr = (uint32_t *) rec;
    if (lane == 0) {
        hdr[4 + threadIdx.y] = cs;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        hdr[1] = (uint32_t) W;
        __threadfence_block();
        hdr[0] = GDN_LAZY_MAGIC;
    }
}

template <int S_v, int warp_size, int rows_per_lane, int cols_per_warp>
static __device__ __forceinline__ bool gdn_lazy_replay(
        float (&s_shard)[cols_per_warp][rows_per_lane], const float * curr_state, const float * state,
        const int64_t slot_stride, const int K, const int col, const int lane) {
    if (slot_stride <= 0) {
        return false;
    }
    const int64_t diff = curr_state - state;  // input row relative to slot 0 of the same cache row
    if (diff <= 0 || diff % slot_stride != 0) {
        return false;
    }
    const int64_t rb = diff / slot_stride;  // rolled-back tokens
    const float * rec = state + (2 + blockIdx.z) * slot_stride;
    const uint32_t * hdr = (const uint32_t *) rec;
    const int W = (int) hdr[1];
    if (hdr[0] != GDN_LAZY_MAGIC || W < 1 || W > K || rb > W || 2 + (int) gridDim.z > K) {
        return false;
    }
    const float * base = state + slot_stride;
    uint32_t h = 0x811c9dc5u ^ (uint32_t) lane;
#pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = base[(col + c) * S_v + i];
            h = gdn_lazy_mix(h, __float_as_uint(s_shard[c][r]));
        }
    }
    if (gdn_lazy_warp_hash<warp_size>(h) != hdr[4 + threadIdx.y]) {
        return false;  // slot 1 no longer holds this record's base
    }
    const int n_replay = W - (int) rb;
    for (int t = 0; t < n_replay; t++) {
        const float * tok = rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, cols_per_warp>();
        const float g_val    = tok[S_v + GDN_LAZY_NWARPS * cols_per_warp + 0];
        const float beta_val = tok[S_v + GDN_LAZY_NWARPS * cols_per_warp + 1];
        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = tok[r * warp_size + lane];
        }
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[c][r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            float delta_col = (tok[S_v + threadIdx.y * cols_per_warp + c] - g_val * kv_col) * beta_val;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
            }
        }
    }
    return true;
}

// RAW: beta and g arrive pre-activation (ggml_gated_delta_net_set_raw_gates); the kernel applies
// sigmoid(beta) and raw_a[h] * softplus(g + raw_dt_bias[h]) with the unary kernels' formulas.
// G_PRECOMPUTED: g already holds exp(g) (GB10 long-prompt path); only used with RAW == false.
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
// keep_rs_t: all 32 x 8 blocks of the verify grid must be resident at once (6 per SM on GB10's 48 SMs)
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, keep_rs_t ? 6 : 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * raw_dt_bias,
                                     const float * raw_a,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride,
                                     int           lazy) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Each warp owns one or more columns, using warp-level primitives to reduce across rows.
    const int      lane     = threadIdx.x;
#if defined(__CUDA_ARCH__)
    constexpr int cols_per_warp = gdn_cols_per_warp(__CUDA_ARCH__, S_v, KDA);
#else
    constexpr int cols_per_warp = 1; // host pass only; never executed
#endif
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // Wait before any s_ids / state load. inp_s_copy is a host upload today, so this is a no-op
    // on that path; ggml_cuda_try_gdn_gather_skip accepts any I32 ids tensor.
    ggml_cuda_pdl_sync();

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // fused gather: read this sequence's live state straight out of the cache row s_ids[sequence]
    const int64_t state_in_offset      = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v)
                                         + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[cols_per_warp][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    // the lazy records use slots 1 .. 1 + gridDim.z, which must exist
    lazy = lazy && state_slot_stride > 0 && 2 + (int) gridDim.z <= K;
    bool replayed = false;
    if constexpr (keep_rs_t && !KDA) {
        if (lazy) {
            replayed = gdn_lazy_replay<S_v, warp_size, rows_per_lane, cols_per_warp>(
                s_shard, curr_state, state, state_slot_stride, K, col, lane);
        }
    }
    if (!replayed) {
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[c][r] = curr_state[(col + c) * S_v + i];
            }
        }
    }
    if constexpr (keep_rs_t && !KDA) {
        if (lazy) {
            // every warp has finished reading the previous lazy record before any warp rewrites it
            __syncthreads();
        }
    }
    [[maybe_unused]] const int lazy_W  = (int) (n_tokens < K ? n_tokens : K);
    [[maybe_unused]] const int lazy_t0 = (int) n_tokens - lazy_W;
    [[maybe_unused]] uint32_t  lazy_cs = 0;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
            float g0 = *g_t;
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

            if constexpr (keep_rs_t) {
                if (lazy && t >= lazy_t0) {
                    if (t == lazy_t0) {
                        lazy_cs = gdn_lazy_store_base<S_v, warp_size, rows_per_lane, cols_per_warp>(
                            s_shard, state + state_slot_stride, col, lane);
                    }
                    gdn_lazy_store_inputs<S_v, warp_size, rows_per_lane, cols_per_warp>(
                        state + (2 + blockIdx.z) * state_slot_stride, t - lazy_t0, k_reg, v_t, col, lane,
                        g_val, beta_val);
                }
            }

            // Each warp owns one or more columns and reuses the common q/k registers. The columns are
            // independent: their v loads are issued up front and their warp reductions run side by side
            // (the stores to dst cannot alias v, but the compiler cannot know that). Per column the
            // operations and their order are those of the one-column-at-a-time loop.
            float v_col[cols_per_warp];
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                v_col[c] = v_t[col + c];
            }
            float kv_col[cols_per_warp];
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                kv_col[c] = kv_shard;
            }
            gdn_warp_reduce_sum_n<warp_size, cols_per_warp>(kv_col);

            float attn_col[cols_per_warp];
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                float delta_col = (v_col[c] - g_val * kv_col[c]) * beta_val;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }
                attn_col[c] = attn_partial;
            }
            gdn_warp_reduce_sum_n<warp_size, cols_per_warp>(attn_col);

            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
                    attn_data[col + c] = attn_col[c] * scale;
                }
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[0][r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[0][r]  = expf(g_t[i]) * s_shard[0][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[0][r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (!lazy && target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }

    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[(col + c) * S_v + i] = s_shard[c][r];
            }
        }
    }

    if constexpr (keep_rs_t && !KDA) {
        if (lazy) {
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    state[(col + c) * S_v + i] = s_shard[c][r];
                }
            }
            gdn_lazy_store_header(state + (2 + blockIdx.z) * state_slot_stride, lazy_W, lazy_cs, lane);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GB10 prefill (n_tokens >= 64): gated_delta_net_cuda<128, false, true, RAW, G_PRECOMPUTED> with a lane-permuted
// column reduction. In gated_delta_net_cuda a warp owns 4 columns and lane L holds rows r*32 + L of each; a
// column's dot products end in warp_reduce_sum (xor 16, 8, 4, 2, 1), 40 shuffles per token. Here the 8 lanes
// of group grp = lane / 8 own column col + grp, and lane n = lane % 8 holds the rows of the old lanes n + 8m
// (m = 0..3). It forms the old lanes' partial sums with the same r-ordered chains, folds them as the first two
// butterfly steps do, (x_n + x_n+16) + (x_n+8 + x_n+24), and finishes with xor 4, 2, 1 inside the group: the same
// leaves in the same tree, so every column value, state element, attention output, lazy record (inputs, base
// and the old per-warp checksum) and snapshot is bit-identical to gated_delta_net_cuda. 6 shuffles per token.

#define GDN_L8_ROWS 16  // rows per lane: 4 old lanes x 4 rows

// old-layout checksum of the 4 columns of this warp, read back from base (gdn_lazy_store_base's order)
static __device__ __forceinline__ uint32_t gdn_l8_base_checksum(const float * base, const int col, const int lane) {
    uint32_t h = 0x811c9dc5u ^ (uint32_t) lane;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            h = gdn_lazy_mix(h, __float_as_uint(base[(col + c) * 128 + r * 32 + lane]));
        }
    }
    return gdn_lazy_warp_hash<32>(h);
}

// one token of the recurrence for this lane's column: s holds rows r*32 + n + 8m as s[m*4 + r]
static __device__ __forceinline__ float gdn_l8_reduce(float (&p)[4]) {
    float x = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

template <bool RAW, bool G_PRECOMPUTED>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_l8_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                        const float * raw_dt_bias, const float * raw_a, const float * curr_state, float * dst,
                        float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
                        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
                        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row_stride,
                        int lazy) {
    constexpr int S_v = 128;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      grp      = lane / 8;
    const int      n        = lane % 8;
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * 4;  // first column of this warp
    const int      my_col   = col + grp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    ggml_cuda_pdl_sync();

    const int64_t state_in_offset  = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v)
                                     + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    float s[GDN_L8_ROWS];  // s[m*4 + r] = S[r*32 + n + 8m][my_col]

    lazy = lazy && state_slot_stride > 0 && 2 + (int) gridDim.z <= K;
    bool replayed = false;
    if (lazy) {
        // gdn_lazy_replay in this layout
        const int64_t diff = curr_state - state;
        if (state_slot_stride > 0 && diff > 0 && diff % state_slot_stride == 0) {
            const int64_t rb  = diff / state_slot_stride;
            const float * rec = state + (2 + blockIdx.z) * state_slot_stride;
            const uint32_t * hdr = (const uint32_t *) rec;
            const int W = (int) hdr[1];
            const float * base = state + state_slot_stride;
            if (hdr[0] == GDN_LAZY_MAGIC && W >= 1 && W <= K && rb <= W && 2 + (int) gridDim.z <= K &&
                    gdn_l8_base_checksum(base, col, lane) == hdr[4 + threadIdx.y]) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        s[m*4 + r] = base[my_col * S_v + r * 32 + n + 8 * m];
                    }
                }
                const int n_replay = W - (int) rb;
                for (int t = 0; t < n_replay; t++) {
                    const float * tok = rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, 4>();
                    const float g_val    = tok[S_v + GDN_LAZY_NWARPS * 4 + 0];
                    const float beta_val = tok[S_v + GDN_LAZY_NWARPS * 4 + 1];
                    float kr[GDN_L8_ROWS];
                    float p[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        p[m] = 0.0f;
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            kr[m*4 + r] = tok[r * 32 + n + 8 * m];
                            p[m] += s[m*4 + r] * kr[m*4 + r];
                        }
                    }
                    const float kv_col    = gdn_l8_reduce(p);
                    const float delta_col = (tok[S_v + threadIdx.y * 4 + grp] - g_val * kv_col) * beta_val;
#pragma unroll
                    for (int j = 0; j < GDN_L8_ROWS; ++j) {
                        s[j] = g_val * s[j] + kr[j] * delta_col;
                    }
                }
                replayed = true;
            }
        }
    }
    if (!replayed) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[m*4 + r] = curr_state[my_col * S_v + r * 32 + n + 8 * m];
            }
        }
    }
    if (lazy) {
        // every warp has finished reading the previous lazy record before any warp rewrites it
        __syncthreads();
    }
    const int lazy_W  = (int) (n_tokens < K ? n_tokens : K);
    const int lazy_t0 = (int) n_tokens - lazy_W;
    uint32_t  lazy_cs = 0;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        float beta_val = beta[gb_offset];
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }
        float kr[GDN_L8_ROWS];
        float qr[GDN_L8_ROWS];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                kr[m*4 + r] = k_t[r * 32 + n + 8 * m];
                qr[m*4 + r] = q_t[r * 32 + n + 8 * m];
            }
        }
        const float v_col = v_t[my_col];

        static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
        float g0 = g[gb_offset];
        if constexpr (RAW) {
            const float x = g0 + raw_dt_bias[h_idx];
            g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        }
        const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

        if (lazy && t >= lazy_t0) {
            float * base = state + state_slot_stride;
            if (t == lazy_t0) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        base[my_col * S_v + r * 32 + n + 8 * m] = s[m*4 + r];
                    }
                }
                __syncwarp();
                lazy_cs = gdn_l8_base_checksum(base, col, lane);
            }
            // gdn_lazy_store_inputs: the head's k row (one group of warp 0), this warp's v columns, g, beta
            float * tok = state + (2 + blockIdx.z) * state_slot_stride + GDN_LAZY_HDR +
                          (t - lazy_t0) * gdn_lazy_token_stride<S_v, 4>();
            if (threadIdx.y == 0 && grp == 0) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        tok[r * 32 + n + 8 * m] = kr[m*4 + r];
                    }
                }
                if (n == 0) {
                    tok[S_v + GDN_LAZY_NWARPS * 4 + 0] = g_val;
                    tok[S_v + GDN_LAZY_NWARPS * 4 + 1] = beta_val;
                }
            }
            if (n == 0) {
                tok[S_v + threadIdx.y * 4 + grp] = v_col;
            }
        }

        float p[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                kv_shard += s[m*4 + r] * kr[m*4 + r];
            }
            p[m] = kv_shard;
        }
        const float kv_col    = gdn_l8_reduce(p);
        const float delta_col = (v_col - g_val * kv_col) * beta_val;

#pragma unroll
        for (int m = 0; m < 4; ++m) {
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[m*4 + r]    = g_val * s[m*4 + r] + kr[m*4 + r] * delta_col;
                attn_partial += s[m*4 + r] * qr[m*4 + r];
            }
            p[m] = attn_partial;
        }
        const float attn_col = gdn_l8_reduce(p);
        if (n == 0) {
            attn_data[my_col] = attn_col * scale;
        }
        attn_data += S_v * H;

        // eager snapshots (slot 0 = most recent state, slot s = s tokens back)
        const int target_slot = (int) n_tokens - 1 - t;
        if (!lazy && target_slot >= 0 && target_slot < K) {
            float * snap = state + target_slot * state_slot_stride;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    snap[my_col * S_v + r * 32 + n + 8 * m] = s[m*4 + r];
                }
            }
        }
    }

    if (lazy) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                state[my_col * S_v + r * 32 + n + 8 * m] = s[m*4 + r];
            }
        }
        gdn_lazy_store_header(state + (2 + blockIdx.z) * state_slot_stride, lazy_W, lazy_cs, lane);
    }
}
