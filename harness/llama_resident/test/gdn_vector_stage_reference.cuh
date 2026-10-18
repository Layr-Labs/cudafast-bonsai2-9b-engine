// Frozen03f1054 l8v kernel; unchanged helpers come from the production header.
template <bool RAW, bool G_PRECOMPUTED, bool LAZY>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_l8v_reference(const float * q, const float * k, const float * v, const float * g, const float * beta,
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
            if (hdr[0] == GDN_LAZY_MAGIC && W >= 1 && W <= K && rb <= W && 2 + (int) gridDim.z <= K &&
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
                    const float * tok = rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, 4>();
                    const float g_val    = gdn_ld(evs, &tok[S_v + GDN_LAZY_NWARPS * 4 + 0]);
                    const float beta_val = gdn_ld(evs, &tok[S_v + GDN_LAZY_NWARPS * 4 + 1]);
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
                    const float delta_col = (gdn_ld(evs, &tok[S_v + threadIdx.y * 4 + grp]) - g_val * kv_col) * beta_val;
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
