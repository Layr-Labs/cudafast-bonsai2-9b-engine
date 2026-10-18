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
// scheme. A verify-sized ubatch (gated_delta_net_l8v_cuda, whose window is all its tokens) does
// not store the final state either: its record says so (GDN_LAZY_NOFINAL) and it marks slot 0
// (GDN_LAZY_SLOT0_MARK0/1 in the first two elements of each warp's first column, two finite values,
// so that the zeroing SCALE of a fresh cell still gives zeros), and a next ubatch that reads slot 0
// replays all W window tokens from slot 1 instead. Any other write of slot 0 (a final-state store,
// the eager snapshots, a zeroing of the row) removes the mark.

// State traffic with the streaming (evict-first) cache operators when evs: the recurrent state and the lazy records
// are touched once per ubatch, so they leave L2 before the layer's activations. Cache policy only.
static __device__ __forceinline__ float gdn_ld(const bool evs, const float * p) {
    return evs ? __ldcs(p) : *p;
}

static __device__ __forceinline__ void gdn_st(const bool evs, float * p, const float v) {
    if (evs) {
        __stcs(p, v);
    } else {
        *p = v;
    }
}

#define GDN_DEDUP_MAGIC  0x4744324cu  // "GD2L": one K/g/beta record per head, parity slots10/11
#define GDN_DEDUP_REHYDRATE 0x80000000u
#define GDN_LAZY_MAGIC   0x4744314cu  // "GD1L"
#define GDN_LAZY_NOFINAL 0x4e4f4631u  // hdr[2]: slot 0 does not hold the final state ("NOF1")
#define GDN_LAZY_SLOT0_MARK0 0x1e3d5b79u  // ~1.0e-20
#define GDN_LAZY_SLOT0_MARK1 0x9c2b4f8du  // ~-5.7e-22

// slot 0 (first column of the warp at col) carries the mark of a verify ubatch that did not store the final state
static __device__ __forceinline__ bool gdn_lazy_slot0_marked(const float * slot0, const int col, const int S_v) {
    return __float_as_uint(slot0[col * S_v + 0]) == GDN_LAZY_SLOT0_MARK0 &&
           __float_as_uint(slot0[col * S_v + 1]) == GDN_LAZY_SLOT0_MARK1;
}
#define GDN_LAZY_HDR     8            // magic, W, final flag, reserved, per-warp base checksum x4
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

static __device__ __forceinline__ void gdn_lazy_store_header(float * rec, const int W, const uint32_t cs, const int lane,
                                                             const bool final_in_slot0 = true) {
    uint32_t * hdr = (uint32_t *) rec;
    if (lane == 0) {
        hdr[4 + threadIdx.y] = cs;
    }
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        hdr[1] = (uint32_t) W;
        hdr[2] = final_in_slot0 ? 0u : GDN_LAZY_NOFINAL;
        hdr[3] = 0;
        __threadfence_block();
        hdr[0] = GDN_LAZY_MAGIC;
    }
}

// A compatible fallback may consume an imported GD2L cache without a counter:
// it writes GD1L and never changes either shared parity record. Only seq1, the
// original eight logical record slices own this representation. Wider N1
// readers provide that logical geometry explicitly; writers keep the format.
static __device__ __forceinline__ const float * gdn_dedup_read_record(
        const float * state, int64_t stride, int K, const uint32_t * hdr, int record_nz = -1) {
    if (hdr[0] != GDN_DEDUP_MAGIC || K < 12 || gridDim.y != 1 || (record_nz < 0 ? (int) gridDim.z : record_nz) != 8) return nullptr;
    const uint32_t gen = hdr[3];
    const float * rec = state + (10 + (gen & 1u)) * stride;
    const uint32_t * sh = reinterpret_cast<const uint32_t *>(rec);
    return sh[0] == GDN_DEDUP_MAGIC && sh[1] == hdr[1] && sh[3] == gen ? rec : nullptr;
}

template <int S_v, int warp_size, int rows_per_lane, int cols_per_warp>
static __device__ __forceinline__ bool gdn_lazy_replay(
        float (&s_shard)[cols_per_warp][rows_per_lane], const float * curr_state, const float * state,
        const int64_t slot_stride, const int K, const int col, const int lane,
        const int record_z = -1, const int record_warp = -1, const int record_nz = -1) {
    if (slot_stride <= 0) {
        return false;
    }
    const int64_t diff = curr_state - state;  // input row relative to slot 0 of the same cache row
    if (diff < 0 || diff % slot_stride != 0) {
        return false;
    }
    const int64_t rb = diff / slot_stride;  // rolled-back tokens
    const int rz = record_z < 0 ? (int) blockIdx.z : record_z;
    const int rw = record_warp < 0 ? (int) threadIdx.y : record_warp;
    const int nz = record_nz < 0 ? (int) gridDim.z : record_nz;
    const float * rec = state + (2 + rz) * slot_stride;
    const uint32_t * hdr = (const uint32_t *) rec;
    const int W = (int) hdr[1];
    const float * shared_rec = nullptr;
    if constexpr (S_v == 128 && warp_size == 32 && cols_per_warp == 4) {
        shared_rec = gdn_dedup_read_record(state, slot_stride, K, hdr, nz);
    }
    if ((hdr[0] != GDN_LAZY_MAGIC && !shared_rec) || W < 1 || W > K || rb > W || 2 + nz > K ||
            (rb == 0 && (hdr[2] != GDN_LAZY_NOFINAL || !gdn_lazy_slot0_marked(curr_state, col, S_v)))) {
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
    if (gdn_lazy_warp_hash<warp_size>(h) != hdr[4 + rw]) {
        return false;  // slot 1 no longer holds this record's base
    }
    const int n_replay = W - (int) rb;
    for (int t = 0; t < n_replay; t++) {
        const float * tok = shared_rec ? shared_rec + GDN_LAZY_HDR + t * (S_v + 2)
                                      : rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, cols_per_warp>();
        const float * tv = shared_rec ? rec + GDN_LAZY_HDR + t * 16 : tok + S_v;
        const int gb = shared_rec ? S_v : S_v + GDN_LAZY_NWARPS * cols_per_warp;
        const float g_val    = tok[gb + 0];
        const float beta_val = tok[gb + 1];
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

            float delta_col = (tv[rw * cols_per_warp + c] - g_val * kv_col) * beta_val;

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
// GB10, n_tokens >= GDN_L8_MIN_TOKENS: gated_delta_net_cuda<128, false, true, RAW, G_PRECOMPUTED> with a lane-permuted
// column reduction. In gated_delta_net_cuda a warp owns 4 columns and lane L holds rows r*32 + L of each; a
// column's dot products end in warp_reduce_sum (xor 16, 8, 4, 2, 1), 40 shuffles per token. Here the 8 lanes
// of group grp = lane / 8 own column col + grp, and lane n = lane % 8 holds the rows of the old lanes n + 8m
// (m = 0..3). It forms the old lanes' partial sums with the same r-ordered chains, folds them as the first two
// butterfly steps do, (x_n + x_n+16) + (x_n+8 + x_n+24), and finishes with xor 4, 2, 1 inside the group: the same
// leaves in the same tree, so every column value, state element, attention output, lazy record (inputs, base
// and the old per-warp checksum) and snapshot is bit-identical to gated_delta_net_cuda. 6 shuffles per token.

#define GDN_L8_ROWS 16  // rows per lane: 4 old lanes x 4 rows
#define GDN_L8_HAS_ZERO_ROW 1  // gated_delta_net_l8_cuda takes zero_row, zero_scale, zero_bias
#define GDN_L8_STAGE 4  // gated_delta_net_l8_cuda: tokens per shared-memory stage
#define GDN_L8_MIN_TOKENS 17  // ubatches of at least this many tokens use it (prefill)
#define GDN_L8_RAW_PRE_MIN_TOKENS 64  // raw gates are activated in a separate pass from this many tokens on
#define GDN_L8V_MIN_TOKENS 8   // gated_delta_net_l8v_cuda: ubatches of GDN_L8V_MIN_TOKENS ..
#define GDN_L8V_MAX_TOKENS 16  // .. GDN_L8V_MAX_TOKENS tokens (the 16-row verify; at most 32)

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

template <bool RAW, bool G_PRECOMPUTED, bool HEAD_MAJOR = false, bool MATERIALIZED = false>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_l8_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                        const float * raw_dt_bias, const float * raw_a, const float * curr_state, float * dst,
                        float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
                        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
                        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row_stride,
                        int lazy, const float * zero_row, float zero_scale, float zero_bias) {
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

    // [C45 prefetch hook] an L2 prefetch of the next weights goes here (before the wait), as in gated_delta_net_cuda.

    ggml_cuda_pdl_sync();

    const int64_t state_row        = s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v;
    const int64_t state_in_offset  = state_row + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    // the row read is the one a deferred in-place SCALE (zero_row) would have rewritten: apply it on load
    const bool zeroed = zero_row != nullptr && curr_state + state_row == zero_row;
    state      += state_out_offset;
    curr_state += state_in_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    float s[GDN_L8_ROWS];  // s[m*4 + r] = S[r*32 + n + 8m][my_col]

    lazy = lazy && state_slot_stride > 0 && 2 + (int) gridDim.z <= K;
    bool replayed = false;
    if (lazy && !zeroed && !MATERIALIZED) {
        // gdn_lazy_replay in this layout
        const int64_t diff = curr_state - state;
        if (state_slot_stride > 0 && diff >= 0 && diff % state_slot_stride == 0) {
            const int64_t rb  = diff / state_slot_stride;
            const float * rec = state + (2 + blockIdx.z) * state_slot_stride;
            const uint32_t * hdr = (const uint32_t *) rec;
            const int W = (int) hdr[1];
            const float * base = state + state_slot_stride;
            const float * shared_rec = gdn_dedup_read_record(state, state_slot_stride, K, hdr);
            if ((hdr[0] == GDN_LAZY_MAGIC || shared_rec) && W >= 1 && W <= K && rb <= W && 2 + (int) gridDim.z <= K &&
                    (rb > 0 || (hdr[2] == GDN_LAZY_NOFINAL && gdn_lazy_slot0_marked(curr_state, col, S_v))) &&
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
                    const float * tok = shared_rec ? shared_rec + GDN_LAZY_HDR + t * (S_v + 2)
                                                  : rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, 4>();
                    const float * tv = shared_rec ? rec + GDN_LAZY_HDR + t * 16 : tok + S_v;
                    const int gb = shared_rec ? S_v : S_v + GDN_LAZY_NWARPS * 4;
                    const float g_val    = tok[gb + 0];
                    const float beta_val = tok[gb + 1];
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
                    const float delta_col = (tv[threadIdx.y * 4 + grp] - g_val * kv_col) * beta_val;
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
                const float x = curr_state[my_col * S_v + r * 32 + n + 8 * m];
                s[m*4 + r] = zeroed ? zero_scale * x + zero_bias : x;  // scale_f32's expression
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

    // The token inputs are staged in shared memory, GDN_L8_STAGE tokens per stage, the next stage copied (cp.async)
    // while this one is used: the head's k rows in [r][n][m] order, the block's 16 v columns, g and beta.
    // Both paths fill the original shared layout with the same Q/K values.
    constexpr int CH     = GDN_L8_STAGE;
    constexpr int TOK_SM = 2 * S_v + 16 + 4;
    __shared__ __align__(16) float stage[2][CH][TOK_SM];
    {
        const int tid = threadIdx.y * 32 + lane;
        const int pos = ((tid / 32) * 8 + tid % 8) * 4 + (tid % 32) / 8;  // where row tid of k goes
        const float * k_src = k + iq3 * sq3 + iq1 * sq1 + (HEAD_MAJOR ? lane * 4 : tid);
        const float * q_src = q + iq3 * sq3 + iq1 * sq1 + (HEAD_MAJOR ? lane * 4 : tid);
        const float * x_src = tid < 16 ? v + sequence * sv3 + h_idx * sv1 + blockIdx.z * 16 + tid :
                              (tid == 16 ? g : beta) + sequence * sb3 + h_idx * sb1;
        const int64_t x_t   = tid < 16 ? sv2 : sb2;
        const int     n_ch  = (int) ((n_tokens + CH - 1) / CH);
        const auto stage_tokens = [&](const int c) {
            float * buf = &stage[c & 1][0][0];
            if constexpr (HEAD_MAJOR) {
                // Each Q/K row is already permuted to [r][n][m]. Four warps copy four tokens,
                // preserving the old shared layout. Both outer layouts use their explicit sq strides;
                // the opt-in conv-front producer writes this without a separate packing pass.
                static_assert(!HEAD_MAJOR || CH == 4, "one token per copy warp");
                const int tt = threadIdx.y;
                const int64_t t = (int64_t) c * CH + tt;
                if (t < n_tokens) {
                    const unsigned int dk = (unsigned int) __cvta_generic_to_shared(&buf[tt * TOK_SM + lane * 4]);
                    const unsigned int dq = (unsigned int) __cvta_generic_to_shared(&buf[tt * TOK_SM + S_v + lane * 4]);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dk), "l"(k_src + t * sq2));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dq), "l"(q_src + t * sq2));
                }
            }
#pragma unroll
            for (int tt = 0; tt < CH; ++tt) {
                const int64_t t = (int64_t) c * CH + tt;
                if (t < n_tokens) {
                    if constexpr (!HEAD_MAJOR) {
                        const unsigned int dk = (unsigned int) __cvta_generic_to_shared(&buf[tt * TOK_SM + pos]);
                        const unsigned int dq = (unsigned int) __cvta_generic_to_shared(&buf[tt * TOK_SM + S_v + pos]);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dk), "l"(k_src + t * sq2));
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dq), "l"(q_src + t * sq2));
                    }
                    if (tid < 18) {
                        const unsigned int dx = (unsigned int) __cvta_generic_to_shared(&buf[tt * TOK_SM + 2 * S_v + tid]);
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dx), "l"(x_src + t * x_t));
                    }
                }
            }
            asm volatile("cp.async.commit_group;");
        };
        stage_tokens(0);

    for (int t = 0; t < n_tokens; t++) {
        const int tt = t % CH;
        if (tt == 0) {
            // stage t / CH is complete, and every warp is done with the previous one, whose buffer the next takes
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (t / CH + 1 < n_ch) {
                stage_tokens(t / CH + 1);
            }
        }
        const float * tok_sm = stage[(t / CH) & 1][tt];

        float beta_val = tok_sm[2 * S_v + 17];
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }
        float kr[GDN_L8_ROWS];
        float qr[GDN_L8_ROWS];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 k4 = *(const float4 *) &tok_sm[(r * 8 + n) * 4];
            const float4 q4 = *(const float4 *) &tok_sm[S_v + (r * 8 + n) * 4];
            kr[0*4 + r] = k4.x; kr[1*4 + r] = k4.y; kr[2*4 + r] = k4.z; kr[3*4 + r] = k4.w;
            qr[0*4 + r] = q4.x; qr[1*4 + r] = q4.y; qr[2*4 + r] = q4.z; qr[3*4 + r] = q4.w;
        }
        const float v_col = tok_sm[2 * S_v + threadIdx.y * 4 + grp];

        static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
        float g0 = tok_sm[2 * S_v + 16];
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

// GB10 verify-sized ubatches (n_tokens <= GDN_L8V_MAX_TOKENS): gated_delta_net_l8_cuda with all token inputs but q
// staged in shared memory before the recurrence and the gates activated once per token (by lane t), so a step
// waits on no global load of k, v, g or beta. Values and their order are those of gated_delta_net_l8_cuda.
template <bool RAW, bool G_PRECOMPUTED, bool LAZY>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_l8v_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
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

    // [C45 prefetch hook] an L2 prefetch of the next weights goes here (before the wait), as in gated_delta_net_cuda.

    ggml_cuda_pdl_sync();
    // bit 3 of lazy: state, base and record traffic with the streaming (evict-first) cache operators
    const bool evs = (lazy & 8) != 0;

    const int64_t state_in_offset  = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v)
                                     + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;
    attn_data  += (sequence * n_tokens * H + h_idx) * S_v;

    // All n_tokens (<= GDN_L8V_MAX_TOKENS) tokens' k rows (in the order lane n reads them: [r][n][m]), the block's
    // 16 v columns, g and beta are copied to shared memory up front (cp.async, overlapping the state load).
    constexpr int TOK_SM = S_v + 16 + 4;
    __shared__ __align__(16) float stage[GDN_L8V_MAX_TOKENS][TOK_SM];
    {
        const int tid = threadIdx.y * 32 + lane;
        const float * k_base  = k + iq3 * sq3 + iq1 * sq1;
        const float * v_base  = v + sequence * sv3 + h_idx * sv1 + blockIdx.z * 16;
        const int64_t gb_base = sequence * sb3 + h_idx * sb1;
        for (int e = tid; e < (int) n_tokens * TOK_SM; e += 128) {
            const int t = e / TOK_SM;
            const int i = e % TOK_SM;
            const float * src;
            int dst_i = i;
            if (i < S_v) {
                const int r = i / 32, n_ = i % 8, m = (i % 32) / 8;
                src   = k_base + t * sq2 + i;
                dst_i = (r * 8 + n_) * 4 + m;
            } else if (i < S_v + 16) {
                src = v_base + t * sv2 + (i - S_v);
            } else if (i < S_v + 18) {
                src = (i == S_v + 16 ? g : beta) + gb_base + t * sb2;
            } else {
                continue;  // padding
            }
            const unsigned int dst = (unsigned int) __cvta_generic_to_shared(&stage[t][dst_i]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
    }

    float s[GDN_L8_ROWS];  // s[m*4 + r] = S[r*32 + n + 8m][my_col]

    // LAZY: the launcher checked lazy, state_slot_stride > 0, 2 + gridDim.z <= K and n_tokens <= K (so the
    // rollback window starts at token 0)
    lazy = LAZY;
    bool replayed = false;
    if (LAZY) {
        // gdn_lazy_replay in this layout
        const int64_t diff = curr_state - state;
        if (state_slot_stride > 0 && diff >= 0 && diff % state_slot_stride == 0) {
            const int64_t rb  = diff / state_slot_stride;
            const float * rec = state + (2 + blockIdx.z) * state_slot_stride;
            const uint32_t * hdr = (const uint32_t *) rec;
            const int W = (int) hdr[1];
            const float * base = state + state_slot_stride;
            const float * shared_rec = gdn_dedup_read_record(state, state_slot_stride, K, hdr);
            if ((hdr[0] == GDN_LAZY_MAGIC || shared_rec) && W >= 1 && W <= K && rb <= W && 2 + (int) gridDim.z <= K &&
                    (rb > 0 || (hdr[2] == GDN_LAZY_NOFINAL && gdn_lazy_slot0_marked(curr_state, col, S_v))) &&
                    gdn_l8_base_checksum(base, col, lane) == hdr[4 + threadIdx.y]) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        s[m*4 + r] = gdn_ld(evs, &base[my_col * S_v + r * 32 + n + 8 * m]);
                    }
                }
                const int n_replay = W - (int) rb;
                for (int t = 0; t < n_replay; t++) {
                    const float * tok = shared_rec ? shared_rec + GDN_LAZY_HDR + t * (S_v + 2)
                                                  : rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, 4>();
                    const float * tv = shared_rec ? rec + GDN_LAZY_HDR + t * 16 : tok + S_v;
                    const int gb = shared_rec ? S_v : S_v + GDN_LAZY_NWARPS * 4;
                    const float g_val    = gdn_ld(evs, &tok[gb + 0]);
                    const float beta_val = gdn_ld(evs, &tok[gb + 1]);
                    float kr[GDN_L8_ROWS];
                    float p[4];
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        p[m] = 0.0f;
#pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            kr[m*4 + r] = gdn_ld(evs, &tok[r * 32 + n + 8 * m]);
                            p[m] += s[m*4 + r] * kr[m*4 + r];
                        }
                    }
                    const float kv_col    = gdn_l8_reduce(p);
                    const float delta_col = (gdn_ld(evs, &tv[threadIdx.y * 4 + grp]) - g_val * kv_col) * beta_val;
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
                s[m*4 + r] = gdn_ld(evs, &curr_state[my_col * S_v + r * 32 + n + 8 * m]);
            }
        }
    }
    // the staged inputs are complete, and every warp has finished reading the previous lazy record before any
    // warp rewrites it
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // the gates of token t, activated by lane t with the main loop's expressions, are broadcast in its step
    float g_lane = 0.0f, beta_lane = 0.0f;
    if (lane < n_tokens) {
        beta_lane = stage[lane][S_v + 17];
        if constexpr (RAW) {
            beta_lane = 1.0f / (1.0f + expf(-beta_lane));
        }
        float g0 = stage[lane][S_v + 16];
        if constexpr (RAW) {
            const float x = g0 + raw_dt_bias[h_idx];
            g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        }
        g_lane = G_PRECOMPUTED ? g0 : expf(g0);
    }
    const int lazy_W  = (int) (n_tokens < K ? n_tokens : K);
    uint32_t  lazy_cs = 0;
    const int tid     = threadIdx.y * 32 + lane;
    if constexpr (LAZY) {
        // gdn_lazy_store_base at token 0, and gdn_lazy_store_inputs of every token from the staged inputs, before
        // the recurrence: the loop below then stores nothing to global memory
        float * base = state + state_slot_stride;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                gdn_st(evs, &base[my_col * S_v + r * 32 + n + 8 * m], s[m*4 + r]);
            }
        }
        __syncwarp();
        lazy_cs = gdn_l8_base_checksum(base, col, lane);

        float * rec = state + (2 + blockIdx.z) * state_slot_stride + GDN_LAZY_HDR;
        constexpr int stride = gdn_lazy_token_stride<S_v, 4>();
        for (int e = tid; e < (int) n_tokens * (S_v + 16); e += 128) {
            const int t = e / (S_v + 16);
            const int i = e % (S_v + 16);
            gdn_st(evs, &rec[t * stride + i], i < S_v ? stage[t][((i / 32) * 8 + i % 8) * 4 + (i % 32) / 8] : stage[t][i]);
        }
        if (threadIdx.y == 0 && lane < n_tokens) {
            gdn_st(evs, &rec[lane * stride + S_v + GDN_LAZY_NWARPS * 4 + 0], g_lane);
            gdn_st(evs, &rec[lane * stride + S_v + GDN_LAZY_NWARPS * 4 + 1], beta_lane);
        }
    }

    __shared__ float attn_sm[GDN_L8V_MAX_TOKENS][16];  // this block's 16 columns of the attention output
#pragma unroll
    for (int t = 0; t < GDN_L8V_MAX_TOKENS; t++) {
        if (t >= n_tokens) {
            break;
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * tok_sm = stage[t];

        const float beta_val = __shfl_sync(0xffffffff, beta_lane, t, 32);
        const float g_val    = __shfl_sync(0xffffffff, g_lane, t, 32);
        float kr[GDN_L8_ROWS];
        float qr[GDN_L8_ROWS];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 k4 = *(const float4 *) &tok_sm[(r * 8 + n) * 4];
            kr[0*4 + r] = k4.x; kr[1*4 + r] = k4.y; kr[2*4 + r] = k4.z; kr[3*4 + r] = k4.w;
        }
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qr[m*4 + r] = q_t[r * 32 + n + 8 * m];
            }
        }
        const float v_col = tok_sm[S_v + threadIdx.y * 4 + grp];

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
            attn_sm[t][threadIdx.y * 4 + grp] = attn_col * scale;
        }

        if constexpr (!LAZY) {
            // eager snapshots (slot 0 = most recent state, slot s = s tokens back)
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
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
    }
    __syncthreads();
    for (int e = tid; e < (int) n_tokens * 16; e += 128) {
        attn_data[(e / 16) * S_v * H + blockIdx.z * 16 + e % 16] = attn_sm[e / 16][e % 16];
    }

    if (lazy) {
        // the window is every token (n_tokens <= K): slot 0 is rebuilt from slot 1 by a replay when read
        if (lane == 0) {
            ((uint32_t *) state)[col * S_v + 0] = GDN_LAZY_SLOT0_MARK0;
            ((uint32_t *) state)[col * S_v + 1] = GDN_LAZY_SLOT0_MARK1;
        }
        gdn_lazy_store_header(state + (2 + blockIdx.z) * state_slot_stride, lazy_W, lazy_cs, lane,
                              /*final_in_slot0 =*/ false);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// N1: full-f32 chunk/WY prefill. No TF32, reduced-precision operand, or gate quotient.
// For a chunk, G_i=prod(a_0..a_i), R_ij=prod(a_(j+1)..a_i), and
// L_ij=beta_i R_ij dot(k_i,k_j) (j<i). The unit-lower triangular solves give
// W=(I+L)^-1 diag(beta G) K, U=(I+L)^-1 diag(beta) V.
// Then D=U-W S, Y=G Q S+tril(R*(Q K^T)) D,
// S'=G_last S+K^T diag(R_last) D. All products/solves/accumulators are f32.
// The last K tokens run through l8, so its snapshots/records describe an exact
// recurrence from the chunk-rounded boundary, including checksums and slot 0.
static constexpr int GDN_CHUNK = 32;
static bool gdn_chunk32_enabled() {
    static const bool enabled = getenv("GGML_CUDA_GDN_CHUNKED") == nullptr || atoi(getenv("GGML_CUDA_GDN_CHUNKED")) != 0;
    return enabled;
}
static bool gdn_chunk32_scan_ilp_enabled() {
    static const bool enabled = getenv("GGML_CUDA_GDN_SCAN_ILP") == nullptr || atoi(getenv("GGML_CUDA_GDN_SCAN_ILP")) != 0;
    return enabled;
}
static bool gdn_chunk32_prefetch_enabled() {
    static const bool enabled = getenv("GGML_CUDA_GDN_PREFETCH") == nullptr || atoi(getenv("GGML_CUDA_GDN_PREFETCH")) != 0;
    return enabled && gdn_chunk32_scan_ilp_enabled();
}
static bool gdn_chunk32_wide_enabled() {
    static const bool enabled = getenv("GGML_CUDA_GDN_WIDE") == nullptr || atoi(getenv("GGML_CUDA_GDN_WIDE")) != 0;
    return enabled && gdn_chunk32_prefetch_enabled();
}
static bool gdn_chunk32_compact_enabled() {
    static const bool enabled = getenv("GGML_CUDA_GDN_COMPACT") == nullptr || atoi(getenv("GGML_CUDA_GDN_COMPACT")) != 0;
    return enabled && gdn_chunk32_wide_enabled();
}

static bool gdn_chunk32_qstage_enabled() {
    static const bool enabled = !getenv("GGML_CUDA_GDN_QSTAGE") || atoi(getenv("GGML_CUDA_GDN_QSTAGE")) != 0;
    return enabled && gdn_chunk32_compact_enabled();
}

// Explicit 16-byte f32 loads, cached in L1 as well as L2. Volatile inline PTX
// also keeps them after the kernel's PDL dependency synchronization. The host
// checks alignment/strides before selecting this path. No numerical cast.
static __device__ __forceinline__ float4 gdn_chunk32_load4(const float * p) {
    float4 v;
    asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ float gdn_chunk32_load1(const float * p) {
    float v;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ void gdn_chunk32_fma4(float a, const float4 & b, float4 & c) {
    c.x = fmaf(a, b.x, c.x); c.y = fmaf(a, b.y, c.y);
    c.z = fmaf(a, b.z, c.z); c.w = fmaf(a, b.w, c.w);
}
static __device__ __forceinline__ float4 gdn_chunk32_mul4(float a, const float4 & b) {
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}
// Physical key order of the frontier Q/K producer -> natural cache row.
// Replay shards start in natural lane ownership; permute them within a warp
// before the wide shared scatter so the XOR layout keeps its bank behavior.
template <bool HEAD_MAJOR>
static __device__ __forceinline__ int gdn_chunk32_natural_key(int row) {
    return HEAD_MAJOR ? (row/32)*32+(row%32)/4+(row%4)*8 : row;
}
template <bool HEAD_MAJOR>
static __device__ __forceinline__ float gdn_chunk32_replay_lane(float value, int lane) {
    if constexpr (HEAD_MAJOR) { return __shfl_sync(0xffffffffu, value, lane/4+(lane%4)*8); }
    return value;
}
struct gdn_chunk32 {
    float w[GDN_CHUNK][128];
    float u[GDN_CHUNK][128];
    float m[GDN_CHUNK][GDN_CHUNK];
    float prefix[GDN_CHUNK];
    float suffix[GDN_CHUNK];
};
// Wide scan: issue all non-W/Q chunk reads before the projections, then stage
// K/M just before their first consumer barrier. This hides their load latency
// behind W*S and Q*S without another shared buffer or another chunk's state.
struct gdn_chunk32_scan_stage {
    float4 k[4], m, u;
    float suffix[4], prefix, last;
};
// (I+L)^-1 replaces both 32x128 W/U transforms. Raw V is consumed by scan.
// Packed entries use only this prefix of the original scratch allocation; the
// full capacity remains available for alignment/diagnostic fallbacks to i1-i4.
struct gdn_compact32 {
    float inv[GDN_CHUNK][GDN_CHUNK];
    float m[GDN_CHUNK][GDN_CHUNK];
    float prefix[GDN_CHUNK], suffix[GDN_CHUNK], beta[GDN_CHUNK];
};
static_assert(sizeof(gdn_compact32) == 8576, "compact f32 layout");
static_assert(sizeof(gdn_compact32) <= sizeof(gdn_chunk32), "scratch capacity");

template <bool RAW, bool G_PRECOMPUTED, bool PREFETCH, bool QREG>
static __global__ void __launch_bounds__(256, QREG ? 4 : 2) gdn_chunk32_prepare(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * raw_dt_bias, const float * raw_a, gdn_chunk32 * work,
        int H, int prefix_tokens, int64_t sq1, int64_t sq2, int64_t sv1, int64_t sv2,
        int64_t sb1, int64_t sb2, uint3 neqk1_magic) {
    constexpr int C = GDN_CHUNK;
    constexpr int PITCH = PREFETCH ? 132 : 129;
    const int h = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x;
    const int t0 = ch*C;
    const int n = min(C, prefix_tokens-t0);
    const int kh = fastmodulo(h, neqk1_magic);
    gdn_chunk32 & out = work[ch*H+h];
    // Padding avoids 32-way bank conflicts in the pairwise Gram products.
    // The wide path broadcasts each q row from warp registers. Removing its
    // 16 KiB shared tile admits four 256-thread CTAs/SM at <=64 registers.
    __shared__ __align__(16) float qs[QREG ? 1 : C][PITCH], ks[C][PITCH];
    __shared__ float lower[C][C+1];
    __shared__ float decay[C], b[C], G[C];
    ggml_cuda_pdl_sync();
    if constexpr (PREFETCH) {
        for (int i = tid; i < C*32; i += 256) {
            const int t = i/32, r = (i%32)*4;
            float4 q4;
            if constexpr (!QREG) {
                q4 = t < n ? gdn_chunk32_load4(q+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
            }
            const float4 k4 = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
            if constexpr (!QREG) { *(float4 *) &qs[t][r] = q4; }
            *(float4 *) &ks[t][r] = k4;
        }
    } else {
        for (int i = tid; i < C*128; i += 256) {
            const int t = i/128, r = i%128;
            qs[t][r] = t < n ? q[kh*sq1+(t0+t)*sq2+r] : 0.0f;
            ks[t][r] = t < n ? k[kh*sq1+(t0+t)*sq2+r] : 0.0f;
        }
    }
    if (tid < C) {
        float gv = 1.0f, bv = 0.0f;
        if (tid < n) {
            float g0 = g[h*sb1+(t0+tid)*sb2];
            bv = beta[h*sb1+(t0+tid)*sb2];
            static_assert(!(RAW && G_PRECOMPUTED), "raw and precomputed gates are exclusive");
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h];
                g0 = raw_a[h] * (x > 20.0f ? x : logf(1.0f+expf(x)));
                bv = 1.0f/(1.0f+expf(-bv));
            }
            gv = G_PRECOMPUTED ? g0 : expf(g0);
        }
        decay[tid] = gv;
        b[tid] = bv;
    }
    __syncthreads();
    if (tid < C) {
        float p = 1.0f, s = 1.0f;
        for (int j = 0; j <= tid; ++j) { p *= decay[j]; }
        for (int j = tid+1; j < n; ++j) { s *= decay[j]; }
        G[tid] = out.prefix[tid] = p;
        out.suffix[tid] = s;
    }
    for (int ij = tid; ij < C*C; ij += 256) {
        const int i = ij/C, j = ij%C;
        float kk = 0.0f, qk = 0.0f, ratio = 1.0f;
        if constexpr (QREG) {
            // i is warp-uniform, j is the lane. ALL lanes must execute the
            // shuffles, including j>i; only the Gram FMAs are triangular.
            if (i < n) {
                const float * qi = q+kh*sq1+(t0+i)*sq2;
                const float4 q_lane = make_float4(gdn_chunk32_load1(qi+j), gdn_chunk32_load1(qi+j+32),
                    gdn_chunk32_load1(qi+j+64), gdn_chunk32_load1(qi+j+96));
                if (j <= i) { for (int t = j+1; t <= i; ++t) { ratio *= decay[t]; } }
#pragma unroll
                for (int r = 0; r < 128; r += 4) {
                    const float qr = r < 32 ? q_lane.x : r < 64 ? q_lane.y : r < 96 ? q_lane.z : q_lane.w;
                    const float4 q4 = make_float4(__shfl_sync(0xffffffffu, qr, (r%32)+0),
                        __shfl_sync(0xffffffffu, qr, (r%32)+1), __shfl_sync(0xffffffffu, qr, (r%32)+2),
                        __shfl_sync(0xffffffffu, qr, (r%32)+3));
                    if (j <= i) {
                        const float4 ki = *(const float4 *) &ks[i][r];
                        const float4 kj = *(const float4 *) &ks[j][r];
                        kk = fmaf(ki.x, kj.x, kk); qk = fmaf(q4.x, kj.x, qk);
                        kk = fmaf(ki.y, kj.y, kk); qk = fmaf(q4.y, kj.y, qk);
                        kk = fmaf(ki.z, kj.z, kk); qk = fmaf(q4.z, kj.z, qk);
                        kk = fmaf(ki.w, kj.w, kk); qk = fmaf(q4.w, kj.w, qk);
                    }
                }
            }
        } else if (j <= i && i < n) {
            // Pairwise products remain defined even when a gate or G_i is zero.
            for (int t = j+1; t <= i; ++t) { ratio *= decay[t]; }
            if constexpr (PREFETCH) {
#pragma unroll
                for (int r = 0; r < 128; r += 4) {
                    const float4 ki = *(const float4 *) &ks[i][r];
                    const float4 kj = *(const float4 *) &ks[j][r];
                    const float4 qi = *(const float4 *) &qs[i][r];
                    // Keep the original ascending-r accumulation, not a new tree.
                    kk = fmaf(ki.x, kj.x, kk); qk = fmaf(qi.x, kj.x, qk);
                    kk = fmaf(ki.y, kj.y, kk); qk = fmaf(qi.y, kj.y, qk);
                    kk = fmaf(ki.z, kj.z, kk); qk = fmaf(qi.z, kj.z, qk);
                    kk = fmaf(ki.w, kj.w, kk); qk = fmaf(qi.w, kj.w, qk);
                }
            } else {
#pragma unroll
                for (int r = 0; r < 128; ++r) {
                    kk = fmaf(ks[i][r], ks[j][r], kk);
                    qk = fmaf(qs[i][r], ks[j][r], qk);
                }
            }
        }
        lower[i][j] = j < i ? b[i]*ratio*kk : 0.0f;
        out.m[i][j] = ratio*qk;
    }
    __syncthreads();
    // Independent triangular solves for the 128 W and 128 U columns. Unroll
    // both loops so the 32 previous entries stay in registers, not local memory.
    const bool is_w = tid < 128;
    const int r = tid%128;
    float x[C];
    float rhs_pre[C];
    if constexpr (PREFETCH) {
        // Issue all independent RHS reads before the dependent solve and before
        // any W/U store. This avoids one global v load on every triangular row.
        float v_pre[C];
        if (!is_w) {
#pragma unroll
            for (int i = 0; i < C; ++i) {
                v_pre[i] = i < n ? gdn_chunk32_load1(v+h*sv1+(t0+i)*sv2+r) : 0.0f;
            }
        }
#pragma unroll
        for (int i = 0; i < C; ++i) {
            rhs_pre[i] = i < n ? (is_w ? (b[i]*G[i])*ks[i][r] : b[i]*v_pre[i]) : 0.0f;
        }
        asm volatile("" ::: "memory"); // compiler barrier, no GPU synchronization
    }
#pragma unroll
    for (int i = 0; i < C; ++i) {
        float rhs = 0.0f;
        if constexpr (PREFETCH) { rhs = rhs_pre[i]; }
        else if (i < n) {
            rhs = is_w ? (b[i]*G[i])*ks[i][r] : b[i]*v[h*sv1+(t0+i)*sv2+r];
        }
#pragma unroll
        for (int j = 0; j < i; ++j) { rhs = fmaf(-lower[i][j], x[j], rhs); }
        x[i] = rhs;
        if constexpr (!PREFETCH) {
            if (is_w) { out.w[i][r] = rhs; } else { out.u[i][r] = rhs; }
        }
    }
    if constexpr (PREFETCH) {
#pragma unroll
        for (int i = 0; i < C; ++i) {
            if (is_w) { out.w[i][r] = x[i]; } else { out.u[i][r] = x[i]; }
        }
    }
}

template <bool RAW, bool G_PRECOMPUTED>
// Leave more register room than i4's spill-producing four-CTA/64-register cap.
static __global__ void __launch_bounds__(256, 3) gdn_compact32_prepare(
        const float * q, const float * k, const float * g, const float * beta,
        const float * raw_dt_bias, const float * raw_a, gdn_compact32 * work,
        int H, int prefix_tokens, int64_t sq1, int64_t sq2,
        int64_t sb1, int64_t sb2, uint3 neqk1_magic) {
    constexpr int C = GDN_CHUNK;
    constexpr int PITCH = 132;
    const int h = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x;
    const int t0 = ch*C;
    const int n = min(C, prefix_tokens-t0);
    const int kh = fastmodulo(h, neqk1_magic);
    gdn_compact32 & out = work[ch*H+h];
    __shared__ __align__(16) float ks[C][PITCH], qs[8][PITCH];
    __shared__ float lower[C][C+1];
    __shared__ float decay[C], b[C];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < C*32; i += 256) {
        const int t = i/32, r = (i%32)*4;
        const float4 k4 = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
        *(float4 *) &ks[t][r] = k4;
    }
    if (tid < C) {
        float gv = 1.0f, bv = 0.0f;
        if (tid < n) {
            float g0 = g[h*sb1+(t0+tid)*sb2];
            bv = beta[h*sb1+(t0+tid)*sb2];
            static_assert(!(RAW && G_PRECOMPUTED), "raw and precomputed gates are exclusive");
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h];
                g0 = raw_a[h] * (x > 20.0f ? x : logf(1.0f+expf(x)));
                bv = 1.0f/(1.0f+expf(-bv));
            }
            gv = G_PRECOMPUTED ? g0 : expf(g0);
        }
        decay[tid] = gv;
        b[tid] = out.beta[tid] = bv;
    }
    __syncthreads();
    if (tid < C) {
        float p = 1.0f, s = 1.0f;
        for (int j = 0; j <= tid; ++j) { p *= decay[j]; }
        for (int j = tid+1; j < n; ++j) { s *= decay[j]; }
        out.prefix[tid] = p;
        out.suffix[tid] = s;
    }
    for (int ij = tid; ij < C*C; ij += 256) {
        const int i = ij/C, j = ij%C;
        float kk = 0.0f, qk = 0.0f, ratio = 1.0f;
        // i is warp-uniform, j is the lane. ALL lanes must execute the
        // shuffles, including j>i; only the Gram FMAs are triangular.
        if (i < n) {
            const float * qi = q+kh*sq1+(t0+i)*sq2;
            // Each warp stages only its current query row, not the full chunk.
            // The leading fence prevents reuse until every lane finished its previous row.
            __syncwarp();
            *(float4 *) &qs[tid/32][4*j] = gdn_chunk32_load4(qi+4*j);
            __syncwarp();
            if (j <= i) { for (int t = j+1; t <= i; ++t) { ratio *= decay[t]; } }
#pragma unroll
            for (int r = 0; r < 128; r += 4) {
                const float4 q4 = *(const float4 *) &qs[tid/32][r];
                if (j <= i) {
                    const float4 ki = *(const float4 *) &ks[i][r];
                    const float4 kj = *(const float4 *) &ks[j][r];
                    kk = fmaf(ki.x, kj.x, kk); qk = fmaf(q4.x, kj.x, qk);
                    kk = fmaf(ki.y, kj.y, kk); qk = fmaf(q4.y, kj.y, qk);
                    kk = fmaf(ki.z, kj.z, kk); qk = fmaf(q4.z, kj.z, qk);
                    kk = fmaf(ki.w, kj.w, kk); qk = fmaf(q4.w, kj.w, qk);
                }
            }
        }
        lower[i][j] = j < i ? b[i]*ratio*kk : 0.0f;
        out.m[i][j] = ratio*qk;
    }
    __syncthreads();
    // Only 32 inverse columns, rather than 128 W + 128 U columns.
    // Solve the unit-diagonal system directly; there is no division or gate
    // quotient. Each inverse entry uses ascending-j f32 FMAs.
    if (tid < C) {
        float x[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            float rhs = i == tid ? 1.0f : 0.0f;
#pragma unroll
            for (int j = 0; j < i; ++j) { rhs = fmaf(-lower[i][j], x[j], rhs); }
            x[i] = rhs;
        }
#pragma unroll
        for (int i = 0; i < C; ++i) { out.inv[i][tid] = x[i]; }
    }
}

// One token x four value columns per thread. W/Q are reused across four values,
// loaded as float4 with register lookahead; kd and delta use vector shared reads.
// The only inter-chunk dependency is the same state tile as before. First W/Q
// vectors of the next chunk are loaded while the current state update computes.
// WIDE doubles dv/CTA, and uses three-quad lookahead (a four-slot register ring).
template <int DV, bool WIDE>
static __device__ __forceinline__ void gdn_chunk32_tile4(
        const float * q_head, const gdn_chunk32 * work, float * dst, int H, int h,
        int prefix_tokens, int t0, int n, int col0, int tid, int64_t sq2, float scale,
        float (&s)[128][DV], float (&delta)[GDN_CHUNK][DV],
        float (&kd)[GDN_CHUNK][132], float (&m)[GDN_CHUNK][36],
        float4 (&look_w)[WIDE ? 3 : 1], float4 (&look_q)[WIDE ? 3 : 1],
        const gdn_chunk32_scan_stage & stage) {
    constexpr int PIPE = WIDE ? 4 : 2, SEED = PIPE-1;
    const int ti = tid/(DV/4), cv = (tid%(DV/4))*4;
    const gdn_chunk32 & in = work[(t0/GDN_CHUNK)*H+h];
    float4 d = make_float4(0, 0, 0, 0), y = d;
    if (ti < n) {
        float4 w4[PIPE], q4[PIPE];
#pragma unroll
        for (int p = 0; p < SEED; ++p) { w4[p] = look_w[p]; q4[p] = look_q[p]; }
#pragma unroll
        for (int r4 = 0; r4 < 32; ++r4) {
            const int cur = r4%PIPE, future = r4+SEED;
            if (future < 32) {
                w4[future%PIPE] = gdn_chunk32_load4(&in.w[ti][future*4]);
                q4[future%PIPE] = gdn_chunk32_load4(q_head+(t0+ti)*sq2+future*4);
            }
            const float4 s0 = *(const float4 *) &s[r4*4+0][cv ^ (WIDE ? ((r4*4+0)&7)*4 : 0)];
            const float4 s1 = *(const float4 *) &s[r4*4+1][cv ^ (WIDE ? ((r4*4+1)&7)*4 : 0)];
            const float4 s2 = *(const float4 *) &s[r4*4+2][cv ^ (WIDE ? ((r4*4+2)&7)*4 : 0)];
            const float4 s3 = *(const float4 *) &s[r4*4+3][cv ^ (WIDE ? ((r4*4+3)&7)*4 : 0)];
            gdn_chunk32_fma4(w4[cur].x, s0, d); gdn_chunk32_fma4(q4[cur].x, s0, y);
            gdn_chunk32_fma4(w4[cur].y, s1, d); gdn_chunk32_fma4(q4[cur].y, s1, y);
            gdn_chunk32_fma4(w4[cur].z, s2, d); gdn_chunk32_fma4(q4[cur].z, s2, y);
            gdn_chunk32_fma4(w4[cur].w, s3, d); gdn_chunk32_fma4(q4[cur].w, s3, y);
        }
        const float4 u = WIDE ? stage.u : gdn_chunk32_load4(&in.u[ti][col0+cv]);
        d = make_float4(u.x-d.x, u.y-d.y, u.z-d.z, u.w-d.w);
        y = gdn_chunk32_mul4(WIDE ? stage.prefix : in.prefix[ti], y);
    }
    if constexpr (WIDE) {
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = tid/32+p*8, r = (tid%32)*4;
            *(float4 *) &kd[t][r] = gdn_chunk32_mul4(stage.suffix[p], stage.k[p]);
        }
        *(float4 *) &m[ti][cv] = stage.m;
    }
    *(float4 *) &delta[ti][cv] = d;
    __syncthreads();
    if (ti < n) {
        for (int j = 0; j <= ti; ++j) {
            const float4 dj = *(const float4 *) &delta[j][cv];
            gdn_chunk32_fma4(m[ti][j], dj, y);
        }
        *(float4 *) &dst[((int64_t) (t0+ti)*H+h)*128+col0+cv] = gdn_chunk32_mul4(scale, y);
    }
    __syncthreads();
    if (t0+GDN_CHUNK < prefix_tokens && ti < min(GDN_CHUNK, prefix_tokens-t0-GDN_CHUNK)) {
        const gdn_chunk32 & next = work[(t0/GDN_CHUNK+1)*H+h];
#pragma unroll
        for (int p = 0; p < SEED; ++p) {
            look_w[p] = gdn_chunk32_load4(&next.w[ti][p*4]);
            look_q[p] = gdn_chunk32_load4(q_head+(t0+GDN_CHUNK+ti)*sq2+p*4);
        }
    }
    float4 x[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int sc = cv ^ (WIDE ? ((ti*4+r)&7)*4 : 0);
        x[r] = gdn_chunk32_mul4(WIDE ? stage.last : in.prefix[n-1], *(const float4 *) &s[ti*4+r][sc]);
    }
    for (int j = 0; j < n; ++j) {
        const float4 dj = *(const float4 *) &delta[j][cv];
        const float4 kj = *(const float4 *) &kd[j][ti*4];
        gdn_chunk32_fma4(kj.x, dj, x[0]); gdn_chunk32_fma4(kj.y, dj, x[1]);
        gdn_chunk32_fma4(kj.z, dj, x[2]); gdn_chunk32_fma4(kj.w, dj, x[3]);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int sc = cv ^ (WIDE ? ((ti*4+r)&7)*4 : 0);
        *(float4 *) &s[ti*4+r][sc] = x[r];
    }
}

template <bool ROW_ILP, bool PREFETCH, bool WIDE, bool HEAD_MAJOR>
static __global__ void gdn_chunk32_scan(
        const float * q, const float * k, const float * curr_state, float * dst, float * state,
        const gdn_chunk32 * work, int H, int prefix_tokens, int64_t sq1, int64_t sq2,
        uint3 neqk1_magic, float scale, int64_t slot_stride, int K, const int32_t * s_ids,
        int64_t row_stride, int lazy, const float * zero_row, float zero_scale, float zero_bias) {
    constexpr int C = GDN_CHUNK;
    constexpr int KP = PREFETCH ? 132 : 129, MP = PREFETCH ? 36 : C+1;
    constexpr int DV = WIDE ? 32 : 16, NT = WIDE ? 256 : 128;
    static_assert(!WIDE || (PREFETCH && ROW_ILP), "wide scan needs vector/ILP path");
    // Keep the four dv tiles of a head adjacent in the CTA launch order, so
    // shared W/Q/K/M input lines can be reused in L2 by neighboring CTAs.
    const int h = WIDE ? blockIdx.x/4 : blockIdx.x;
    const int slice = WIDE ? blockIdx.x%4 : blockIdx.z;
    const int lane = threadIdx.x, tid = threadIdx.y*32+lane;
    const int col0 = slice*DV, col = col0+threadIdx.y*4;
    const int vc = tid%16, tr = tid/16; // 8 token rows x 16 value columns
    const int kh = fastmodulo(h, neqk1_magic);
    __shared__ __align__(16) float s[128][DV], delta[C][DV], kd[C][KP], m[C][MP];
    ggml_cuda_pdl_sync();
    const int64_t in_row = s_ids ? (int64_t) s_ids[0]*row_stride : 0;
    const bool zeroed = zero_row != nullptr && curr_state+in_row == zero_row;
    curr_state += in_row+(int64_t) h*128*128;
    state += (int64_t) h*128*128;
    lazy = lazy && slot_stride > 0 && 2+(WIDE ? 8 : (int) gridDim.z) <= K;
    float shard[4][4];
    const bool replayed = lazy && !zeroed &&
        gdn_lazy_replay<128, 32, 4, 4>(shard, curr_state, state, slot_stride, K, col, lane,
            WIDE ? 2*slice+(int) threadIdx.y/4 : -1, WIDE ? (int) threadIdx.y%4 : -1, WIDE ? 8 : -1);
    if constexpr (WIDE) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = r*32+lane;
            float4 x;
            x.x = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[0][r], lane) : curr_state[(col+0)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
            x.y = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[1][r], lane) : curr_state[(col+1)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
            x.z = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[2][r], lane) : curr_state[(col+2)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
            x.w = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[3][r], lane) : curr_state[(col+3)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
            if (zeroed) {
                x.x = zero_scale*x.x+zero_bias; x.y = zero_scale*x.y+zero_bias;
                x.z = zero_scale*x.z+zero_bias; x.w = zero_scale*x.w+zero_bias;
            }
            // Each eight-thread, 128-byte vector store visits all 32 banks.
            *(float4 *) &s[row][(threadIdx.y*4) ^ ((row&7)*4)] = x;
        }
    } else {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = r*32+lane;
            float x = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[c][r], lane) : curr_state[(col+c)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
            if (zeroed) { x = zero_scale*x+zero_bias; }
            s[row][threadIdx.y*4+c] = x;
        }
    }
    }
    float4 look_w[WIDE ? 3 : 1], look_q[WIDE ? 3 : 1];
    if constexpr (PREFETCH) {
        const int ti = tid/(DV/4);
        if (ti < min(C, prefix_tokens)) {
#pragma unroll
            for (int p = 0; p < (WIDE ? 3 : 1); ++p) {
                look_w[p] = gdn_chunk32_load4(&work[h].w[ti][p*4]);
                look_q[p] = gdn_chunk32_load4(q+kh*sq1+ti*sq2+p*4);
            }
        }
    }
    __syncthreads();
    for (int t0 = 0; t0 < prefix_tokens; t0 += C) {
        const int n = min(C, prefix_tokens-t0);
        const gdn_chunk32 & in = work[(t0/C)*H+h];
        if constexpr (PREFETCH) {
            gdn_chunk32_scan_stage stage;
            if constexpr (WIDE) {
                // Four independent K/suffix groups per thread, plus M/U/gates,
                // all issued before their arithmetic consumers. The projection
                // needs only S/W/Q; kd and M are first read after its barrier.
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int t = tid/32+p*8, r = (tid%32)*4;
                    stage.k[p] = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
                    stage.suffix[p] = gdn_chunk32_load1(&in.suffix[t]);
                }
                const int ti = tid/8, cv = (tid%8)*4;
                stage.m = gdn_chunk32_load4(&in.m[ti][cv]);
                stage.u = ti < n ? gdn_chunk32_load4(&in.u[ti][col0+cv]) : make_float4(0, 0, 0, 0);
                stage.prefix = gdn_chunk32_load1(&in.prefix[ti]);
                stage.last = gdn_chunk32_load1(&in.prefix[n-1]);
                asm volatile("" ::: "memory");
            } else {
            for (int i = tid; i < C*32; i += NT) {
                const int t = i/32, r = (i%32)*4;
                const float4 k4 = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
                *(float4 *) &kd[t][r] = gdn_chunk32_mul4(in.suffix[t], k4);
            }
            for (int i = tid; i < C*C/4; i += NT) {
                *(float4 *) &m[i/(C/4)][(i%(C/4))*4] = gdn_chunk32_load4(&in.m[i/(C/4)][(i%(C/4))*4]);
            }
            }
            gdn_chunk32_tile4<DV, WIDE>(q+kh*sq1, work, dst, H, h, prefix_tokens, t0, n, col0, tid, sq2, scale,
                s, delta, kd, m, look_w, look_q, stage);
        } else {
        for (int i = tid; i < C*128; i += 128) {
            const int t = i/128, r = i%128;
            kd[t][r] = t < n ? k[kh*sq1+(t0+t)*sq2+r]*in.suffix[t] : 0.0f;
        }
        for (int i = tid; i < C*C; i += 128) { m[i/C][i%C] = in.m[i/C][i%C]; }
        float d[4] = {}, y[4] = {};
#pragma unroll
        for (int r = 0; r < 128; ++r) {
            const float sr = s[r][vc];
#pragma unroll
            for (int z = 0; z < 4; ++z) {
                const int t = tr+z*8;
                if (t < n) {
                    d[z] = fmaf(in.w[t][r], sr, d[z]);
                    y[z] = fmaf(q[kh*sq1+(t0+t)*sq2+r], sr, y[z]);
                }
            }
        }
#pragma unroll
        for (int z = 0; z < 4; ++z) {
            const int t = tr+z*8;
            delta[t][vc] = t < n ? in.u[t][col0+vc]-d[z] : 0.0f;
            y[z] *= in.prefix[t];
        }
        __syncthreads();
#pragma unroll
        for (int z = 0; z < 4; ++z) {
            const int t = tr+z*8;
            if (t < n) {
                for (int j = 0; j <= t; ++j) { y[z] = fmaf(m[t][j], delta[j][vc], y[z]); }
                dst[((int64_t) (t0+t)*H+h)*128+col0+vc] = y[z]*scale;
            }
        }
        // All Q*S reads must finish before any thread changes S.
        __syncthreads();
        if constexpr (ROW_ILP) {
            // Sixteen independent row accumulators, rather than sixteen serial
            // inner loops. Reuse each delta load across those rows. Each state
            // element still sees the same initial multiply and ascending-t FMAs.
            float x[16];
#pragma unroll
            for (int z = 0; z < 16; ++z) { x[z] = in.prefix[n-1]*s[tr+z*8][vc]; }
            for (int t = 0; t < n; ++t) {
                const float dv = delta[t][vc];
#pragma unroll
                for (int z = 0; z < 16; ++z) { x[z] = fmaf(kd[t][tr+z*8], dv, x[z]); }
            }
#pragma unroll
            for (int z = 0; z < 16; ++z) { s[tr+z*8][vc] = x[z]; }
        } else {
#pragma unroll
            for (int z = 0; z < 16; ++z) {
                const int r = tr+z*8;
                float x = in.prefix[n-1]*s[r][vc];
                for (int t = 0; t < n; ++t) { x = fmaf(kd[t][r], delta[t][vc], x); }
                s[r][vc] = x;
            }
        }
        } // PREFETCH == false
        __syncthreads();
    }
    if constexpr (WIDE) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = r*32+lane;
            const float4 x = *(const float4 *) &s[row][(threadIdx.y*4) ^ ((row&7)*4)];
            state[(col+0)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.x; state[(col+1)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.y;
            state[(col+2)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.z; state[(col+3)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.w;
        }
    } else {
#pragma unroll
    for (int z = 0; z < 16; ++z) {
        const int r = tr+z*8;
        state[(col0+vc)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(r)] = s[r][vc];
    }
    }
}

// Compact f32 form: A=diag(beta)*(V-diag(G)*K*S), D=(I+L)^-1*A.
// Y and the carry use the original chunk equations. This adds one triangular
// product but eliminates both W/U arrays and all global W projection loads.
// K is staged raw and reused by both projections and the carry. Direct Q uses
// five chunk barriers; staged Q adds three to reuse the matrix scratch safely.
template <bool HEAD_MAJOR, bool QSTAGE>
static __global__ void __launch_bounds__(256, 2) gdn_compact32_scan(
        const float * q, const float * k, const float * v, const float * curr_state, float * dst, float * state,
        const gdn_compact32 * work, int H, int prefix_tokens, int64_t sq1, int64_t sq2,
        int64_t sv1, int64_t sv2, uint3 neqk1_magic, float scale, int64_t slot_stride, int K,
        const int32_t * s_ids, int64_t row_stride, int lazy, const float * zero_row,
        float zero_scale, float zero_bias) {
    constexpr int C = GDN_CHUNK;
    const int h = blockIdx.x/4, slice = blockIdx.x%4;
    const int lane = threadIdx.x, tid = threadIdx.y*32+lane;
    const int ti = tid/8, cv = (tid%8)*4, col0 = slice*32, col = col0+threadIdx.y*4;
    const int kh = fastmodulo(h, neqk1_magic);
    __shared__ __align__(16) float s[128][32], delta[C][32], ks[C][132];
    // Q is needed only during projection; M/Inv only after it. The same
    // 9 KiB holds either two matrices or a 32-token x 64-key Q tile.
    union reuse_t {
        struct { float m[C][36], inv[C][36]; } matrix;
        float q[C][68];
    };
    static_assert(sizeof(reuse_t) == 2*C*36*sizeof(float), "shared reuse footprint");
    __shared__ __align__(16) reuse_t reuse;
    auto & m = reuse.matrix.m;
    auto & inv = reuse.matrix.inv;
    __shared__ float suffix[C];
    ggml_cuda_pdl_sync();
    const int64_t in_row = s_ids ? (int64_t) s_ids[0]*row_stride : 0;
    const bool zeroed = zero_row != nullptr && curr_state+in_row == zero_row;
    curr_state += in_row+(int64_t) h*128*128;
    state += (int64_t) h*128*128;
    lazy = lazy && slot_stride > 0 && 2+8 <= K;
    float shard[4][4];
    const bool replayed = lazy && !zeroed &&
        gdn_lazy_replay<128, 32, 4, 4>(shard, curr_state, state, slot_stride, K, col, lane,
            2*slice+(int) threadIdx.y/4, (int) threadIdx.y%4, 8);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r*32+lane;
        float4 x;
        x.x = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[0][r], lane) : curr_state[(col+0)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
        x.y = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[1][r], lane) : curr_state[(col+1)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
        x.z = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[2][r], lane) : curr_state[(col+2)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
        x.w = replayed ? gdn_chunk32_replay_lane<HEAD_MAJOR>(shard[3][r], lane) : curr_state[(col+3)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)];
        if (zeroed) {
            x.x = zero_scale*x.x+zero_bias; x.y = zero_scale*x.y+zero_bias;
            x.z = zero_scale*x.z+zero_bias; x.w = zero_scale*x.w+zero_bias;
        }
        *(float4 *) &s[row][(threadIdx.y*4) ^ ((row&7)*4)] = x;
    }
    float4 look_q[QSTAGE ? 2 : 3];
    if constexpr (QSTAGE) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int i = tid+p*256, qt = i/16, qr = (i%16)*4;
            look_q[p] = qt < min(C, prefix_tokens) ? gdn_chunk32_load4(q+kh*sq1+qt*sq2+qr) : make_float4(0, 0, 0, 0);
        }
    } else if (ti < min(C, prefix_tokens)) {
#pragma unroll
        for (int p = 0; p < 3; ++p) { look_q[p] = gdn_chunk32_load4(q+kh*sq1+ti*sq2+p*4); }
    }
    __syncthreads();
    for (int t0 = 0; t0 < prefix_tokens; t0 += C) {
        const int n = min(C, prefix_tokens-t0);
        const gdn_compact32 & in = work[(t0/C)*H+h];
        // Independent global reads, before arithmetic/stores use any of them.
        float4 k_pre[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t = tid/32+p*8, r = lane*4;
            k_pre[p] = t < n ? gdn_chunk32_load4(k+kh*sq1+(t0+t)*sq2+r) : make_float4(0, 0, 0, 0);
        }
        float4 q_next[2];
        if constexpr (QSTAGE) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int i = tid+p*256, qt = i/16, qr = (i%16)*4;
                q_next[p] = qt < n ? gdn_chunk32_load4(q+kh*sq1+(t0+qt)*sq2+64+qr) : make_float4(0, 0, 0, 0);
            }
        }
        const float4 mi = gdn_chunk32_load4(&in.m[ti][cv]);
        const float4 ii = gdn_chunk32_load4(&in.inv[ti][cv]);
        const float4 vi = ti < n ? gdn_chunk32_load4(v+h*sv1+(t0+ti)*sv2+col0+cv) : make_float4(0, 0, 0, 0);
        const float gi = gdn_chunk32_load1(&in.prefix[ti]);
        const float bi = gdn_chunk32_load1(&in.beta[ti]);
        const float last = gdn_chunk32_load1(&in.prefix[n-1]);
        const float sf = tid < C ? gdn_chunk32_load1(&in.suffix[tid]) : 1.0f;
        asm volatile("" ::: "memory");
#pragma unroll
        for (int p = 0; p < 4; ++p) { *(float4 *) &ks[tid/32+p*8][lane*4] = k_pre[p]; }
        if constexpr (QSTAGE) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int i = tid+p*256;
                *(float4 *) &reuse.q[i/16][(i%16)*4] = look_q[p];
            }
        } else {
            *(float4 *) &m[ti][cv] = mi;
            *(float4 *) &inv[ti][cv] = ii;
        }
        if (tid < C) { suffix[tid] = sf; }
        __syncthreads();
        float4 a = make_float4(0, 0, 0, 0), y = a;
        if constexpr (QSTAGE) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (half == 1) {
                    __syncthreads(); // finish every reader of the first Q half
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int i = tid+p*256;
                        *(float4 *) &reuse.q[i/16][(i%16)*4] = q_next[p];
                    }
                    __syncthreads();
                }
                if (ti < n) {
#pragma unroll
                    for (int j4 = 0; j4 < 16; ++j4) {
                        const int r4 = half*16+j4;
                        const float4 qi = *(const float4 *) &reuse.q[ti][j4*4];
                        const float4 ki = *(const float4 *) &ks[ti][r4*4];
                        const float4 s0 = *(const float4 *) &s[r4*4+0][cv ^ (((r4*4+0)&7)*4)];
                        const float4 s1 = *(const float4 *) &s[r4*4+1][cv ^ (((r4*4+1)&7)*4)];
                        const float4 s2 = *(const float4 *) &s[r4*4+2][cv ^ (((r4*4+2)&7)*4)];
                        const float4 s3 = *(const float4 *) &s[r4*4+3][cv ^ (((r4*4+3)&7)*4)];
                        gdn_chunk32_fma4(ki.x, s0, a); gdn_chunk32_fma4(qi.x, s0, y);
                        gdn_chunk32_fma4(ki.y, s1, a); gdn_chunk32_fma4(qi.y, s1, y);
                        gdn_chunk32_fma4(ki.z, s2, a); gdn_chunk32_fma4(qi.z, s2, y);
                        gdn_chunk32_fma4(ki.w, s3, a); gdn_chunk32_fma4(qi.w, s3, y);
                    }
                }
            }
            __syncthreads(); // all Q readers finish before M/Inv overwrite Q
            *(float4 *) &m[ti][cv] = mi;
            *(float4 *) &inv[ti][cv] = ii;
        } else {
            if (ti < n) {
                float4 q4[4];
#pragma unroll
                for (int p = 0; p < 3; ++p) { q4[p] = look_q[p]; }
#pragma unroll
                for (int r4 = 0; r4 < 32; ++r4) {
                    const int cur = r4%4, future = r4+3;
                    if (future < 32) { q4[future%4] = gdn_chunk32_load4(q+kh*sq1+(t0+ti)*sq2+future*4); }
                    const float4 ki = *(const float4 *) &ks[ti][r4*4];
                    const float4 s0 = *(const float4 *) &s[r4*4+0][cv ^ (((r4*4+0)&7)*4)];
                    const float4 s1 = *(const float4 *) &s[r4*4+1][cv ^ (((r4*4+1)&7)*4)];
                    const float4 s2 = *(const float4 *) &s[r4*4+2][cv ^ (((r4*4+2)&7)*4)];
                    const float4 s3 = *(const float4 *) &s[r4*4+3][cv ^ (((r4*4+3)&7)*4)];
                    gdn_chunk32_fma4(ki.x, s0, a); gdn_chunk32_fma4(q4[cur].x, s0, y);
                    gdn_chunk32_fma4(ki.y, s1, a); gdn_chunk32_fma4(q4[cur].y, s1, y);
                    gdn_chunk32_fma4(ki.z, s2, a); gdn_chunk32_fma4(q4[cur].z, s2, y);
                    gdn_chunk32_fma4(ki.w, s3, a); gdn_chunk32_fma4(q4[cur].w, s3, y);
                }
            }
        }
        if (ti < n) {
            a = make_float4(bi*fmaf(-gi, a.x, vi.x), bi*fmaf(-gi, a.y, vi.y),
                bi*fmaf(-gi, a.z, vi.z), bi*fmaf(-gi, a.w, vi.w));
            y = gdn_chunk32_mul4(gi, y);
        }
        *(float4 *) &delta[ti][cv] = a;
        __syncthreads(); // A and all S projection readers are complete.
        float4 d = make_float4(0, 0, 0, 0);
        if (ti < n) {
            for (int j = 0; j <= ti; ++j) { gdn_chunk32_fma4(inv[ti][j], *(const float4 *) &delta[j][cv], d); }
        }
        __syncthreads(); // No reader may still use A when D overwrites it.
        *(float4 *) &delta[ti][cv] = d;
        __syncthreads();
        if (ti < n) {
            for (int j = 0; j <= ti; ++j) { gdn_chunk32_fma4(m[ti][j], *(const float4 *) &delta[j][cv], y); }
            *(float4 *) &dst[((int64_t) (t0+ti)*H+h)*128+col0+cv] = gdn_chunk32_mul4(scale, y);
        }
        if constexpr (QSTAGE) {
            if (t0+C < prefix_tokens) {
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int i = tid+p*256, qt = i/16, qr = (i%16)*4;
                    look_q[p] = qt < min(C, prefix_tokens-t0-C) ?
                        gdn_chunk32_load4(q+kh*sq1+(t0+C+qt)*sq2+qr) : make_float4(0, 0, 0, 0);
                }
            }
        } else if (t0+C < prefix_tokens && ti < min(C, prefix_tokens-t0-C)) {
#pragma unroll
            for (int p = 0; p < 3; ++p) { look_q[p] = gdn_chunk32_load4(q+kh*sq1+(t0+C+ti)*sq2+p*4); }
        }
        float4 x[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { x[r] = gdn_chunk32_mul4(last, *(const float4 *) &s[ti*4+r][cv ^ (((ti*4+r)&7)*4)]); }
        for (int j = 0; j < n; ++j) {
            const float4 dj = gdn_chunk32_mul4(suffix[j], *(const float4 *) &delta[j][cv]);
            const float4 kj = *(const float4 *) &ks[j][ti*4];
            gdn_chunk32_fma4(kj.x, dj, x[0]); gdn_chunk32_fma4(kj.y, dj, x[1]);
            gdn_chunk32_fma4(kj.z, dj, x[2]); gdn_chunk32_fma4(kj.w, dj, x[3]);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { *(float4 *) &s[ti*4+r][cv ^ (((ti*4+r)&7)*4)] = x[r]; }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = r*32+lane;
        const float4 x = *(const float4 *) &s[row][(threadIdx.y*4) ^ ((row&7)*4)];
        state[(col+0)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.x; state[(col+1)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.y;
        state[(col+2)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.z; state[(col+3)*128+gdn_chunk32_natural_key<HEAD_MAJOR>(row)] = x.w;
    }
}

// Eligible only for S_v=128, one sequence, keep_rs, 1<K<T, and NVIDIA.
// Work has H*ceil((T-K)/32) entries. No allocation, host readback or synchronization
// here: the caller owns scratch through its stream/capture lifetime.
template <bool RAW, bool G_PRECOMPUTED, bool HEAD_MAJOR = false>
static void gdn_chunk32_launch(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * raw_dt_bias, const float * raw_a, const float * curr_state, float * dst, float * state,
        gdn_chunk32 * work, int H, int T, int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
        uint3 neqk1_magic, uint3 rq3_magic, float scale, int64_t slot_stride, int K,
        const int32_t * s_ids, int64_t row_stride, int lazy, const float * zero_row,
        float zero_scale, float zero_bias, cudaStream_t stream,
        const cudaEvent_t * phase_events = nullptr, bool row_ilp = gdn_chunk32_scan_ilp_enabled(),
        bool prefetch = gdn_chunk32_prefetch_enabled(), bool wide = gdn_chunk32_wide_enabled(),
        bool compact = gdn_chunk32_compact_enabled(), bool qstage = gdn_chunk32_qstage_enabled()) {
    const ggml_cuda_kernel_launch_params scan(dim3(H, 1, 8), dim3(32, 4), 0, stream);
    if (!gdn_chunk32_enabled()) {
        ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<RAW, G_PRECOMPUTED, HEAD_MAJOR>, scan,
            q, k, v, g, beta, raw_dt_bias, raw_a, curr_state, dst, state,
            (int64_t) H, (int64_t) T, (int64_t) 1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1_magic, rq3_magic, scale, slot_stride, K, s_ids, row_stride, lazy, zero_row, zero_scale, zero_bias);
        return;
    }
    // Global vector reads require aligned rows; unsupported layouts retain i2.
    prefetch = prefetch && row_ilp && ((uintptr_t) q%16 == 0) && ((uintptr_t) k%16 == 0) &&
        ((uintptr_t) dst%16 == 0) && ((uintptr_t) work%16 == 0) && sq1%4 == 0 && sq2%4 == 0;
    wide = wide && prefetch;
    compact = compact && wide && ((uintptr_t) v%16 == 0) && sv1%4 == 0 && sv2%4 == 0;
    const int prefix = T-K;
    const ggml_cuda_kernel_launch_params prep(dim3(H, (prefix+31)/32), dim3(256), 0, stream);
    const ggml_cuda_kernel_launch_params wide_scan(dim3(H*4), dim3(32, 8), 0, stream);
    // Optional standalone instrumentation only. The production caller supplies
    // no events; normal graphs keep the original three kernel nodes.
    if (phase_events) { CUDA_CHECK(cudaEventRecordWithFlags(phase_events[0], stream, cudaEventRecordExternal)); }
#define GDN_CHUNK32_PREP(PREF_, QREG_) \
    ggml_cuda_kernel_launch(gdn_chunk32_prepare<RAW, G_PRECOMPUTED, PREF_, QREG_>, prep, \
        q, k, v, g, beta, raw_dt_bias, raw_a, work, H, prefix, sq1, sq2, sv1, sv2, sb1, sb2, neqk1_magic)
    if (compact) {
        ggml_cuda_kernel_launch(gdn_compact32_prepare<RAW, G_PRECOMPUTED>, prep,
            q, k, g, beta, raw_dt_bias, raw_a, (gdn_compact32 *) work,
            H, prefix, sq1, sq2, sb1, sb2, neqk1_magic);
    }
    else if (wide) { GDN_CHUNK32_PREP(true, true); }
    else if (prefetch) { GDN_CHUNK32_PREP(true, false); }
    else { GDN_CHUNK32_PREP(false, false); }
#undef GDN_CHUNK32_PREP
    if (phase_events) { CUDA_CHECK(cudaEventRecordWithFlags(phase_events[1], stream, cudaEventRecordExternal)); }
#define GDN_CHUNK32_SCAN(ILP_, PREF_, WIDE_) \
    ggml_cuda_kernel_launch(gdn_chunk32_scan<ILP_, PREF_, WIDE_, HEAD_MAJOR>, WIDE_ ? wide_scan : scan, q, k, curr_state, dst, state, work, \
        H, prefix, sq1, sq2, neqk1_magic, scale, slot_stride, K, s_ids, row_stride, lazy, \
        zero_row, zero_scale, zero_bias)
    if (compact) {
#define GDN_COMPACT_SCAN(QSTAGE_) \
        ggml_cuda_kernel_launch(gdn_compact32_scan<HEAD_MAJOR, QSTAGE_>, wide_scan, q, k, v, curr_state, dst, state, \
            (const gdn_compact32 *) work, H, prefix, sq1, sq2, sv1, sv2, neqk1_magic, scale, \
            slot_stride, K, s_ids, row_stride, lazy, zero_row, zero_scale, zero_bias)
        if (qstage) { GDN_COMPACT_SCAN(true); } else { GDN_COMPACT_SCAN(false); }
#undef GDN_COMPACT_SCAN
    }
    else if (wide) { GDN_CHUNK32_SCAN(true, true, true); }
    else if (prefetch) { GDN_CHUNK32_SCAN(true, true, false); }
    else if (row_ilp) { GDN_CHUNK32_SCAN(true, false, false); }
    else { GDN_CHUNK32_SCAN(false, false, false); }
#undef GDN_CHUNK32_SCAN
    if (phase_events) { CUDA_CHECK(cudaEventRecordWithFlags(phase_events[2], stream, cudaEventRecordExternal)); }
    // The prefix just materialized slot0. Do not reinterpret it through the old
    // NOFINAL record, even if two rounded state words happen to equal its mark.
    ggml_cuda_kernel_launch(gated_delta_net_l8_cuda<RAW, G_PRECOMPUTED, HEAD_MAJOR, true>, scan,
        q+prefix*sq2, k+prefix*sq2, v+prefix*sv2, g+prefix*sb2, beta+prefix*sb2,
        raw_dt_bias, raw_a, state, dst+(int64_t) prefix*H*128, state,
        (int64_t) H, (int64_t) K, (int64_t) 1, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        neqk1_magic, rq3_magic, scale, slot_stride, K, (const int32_t *) nullptr, (int64_t) 0,
        lazy, (const float *) nullptr, 0.0f, 0.0f);
    if (phase_events) { CUDA_CHECK(cudaEventRecordWithFlags(phase_events[3], stream, cudaEventRecordExternal)); }
}
#endif
