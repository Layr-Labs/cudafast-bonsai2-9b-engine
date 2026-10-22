#pragma once
// Exact l8v recurrence with per-head deduplication of private rollback K/g/beta records.
// Scope: one sequence, coherent private-cache records, serialized launches on
// each cache/counter pair. Full eager imports are also supported. An arbitrary
// mixture of valid lazy records and eager columns is not a coherent cache;
// this test does not claim to repair the baseline's partial-cache alias race.
#include "gated_delta_net_kernel.cuh"

static __device__ __forceinline__ void gdn_dedup_store_header(
        float * rec, float * shared_rec, int W, uint32_t cs, int lane, uint32_t gen) {
    auto * hdr = reinterpret_cast<uint32_t *>(rec);
    if (lane == 0) hdr[4 + threadIdx.y] = cs;
    __syncthreads();
    if (threadIdx.y == 0 && lane == 0) {
        hdr[1] = uint32_t(W);
        hdr[2] = GDN_LAZY_NOFINAL;
        hdr[3] = shared_rec ? gen : 0u;
        if (shared_rec && blockIdx.z == 0) {
            auto * sh = reinterpret_cast<uint32_t *>(shared_rec);
            sh[1] = uint32_t(W); sh[2] = 0; sh[3] = gen;
            __threadfence_block(); sh[0] = GDN_DEDUP_MAGIC;
        }
        __threadfence_block(); hdr[0] = shared_rec ? GDN_DEDUP_MAGIC : GDN_LAZY_MAGIC;
    }
}

// G1 i2: the original FP recurrence with a single-read base/hash path and
// warp-private, two-token cp.async replay staging. No FP arithmetic is reordered.
// A 4x4 register transpose swaps the column register with lane bits 3/4.
// It is self-inverse: natural [column][r] <-> l8 [old lane group][r].
static __device__ __forceinline__ void gdn_g1_transpose(float (&s)[GDN_L8_ROWS]) {
    const int lane = threadIdx.x;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        float x0=s[r], x1=s[4+r], x2=s[8+r], x3=s[12+r];
        float a=__shfl_xor_sync(0xffffffff, (lane&8) ? x0 : x1, 8, 32);
        float b=__shfl_xor_sync(0xffffffff, (lane&8) ? x2 : x3, 8, 32);
        if (lane&8) { x0=a; x2=b; } else { x1=a; x3=b; }
        a=__shfl_xor_sync(0xffffffff, (lane&16) ? x0 : x2, 16, 32);
        b=__shfl_xor_sync(0xffffffff, (lane&16) ? x1 : x3, 16, 32);
        if (lane&16) { x0=a; x1=b; } else { x2=a; x3=b; }
        s[r]=x0; s[4+r]=x1; s[8+r]=x2; s[12+r]=x3;
    }
}

// Read each base word once, hashing it in the original c/r order, then put the
// same bits into the recurrence's register layout. Guard failure still falls
// back to curr_state; the checksum is neither skipped nor weakened.
static __device__ __forceinline__ uint32_t gdn_g1_load_base(
        const float * base, int col, bool evs, float (&s)[GDN_L8_ROWS]) {
    const int lane=threadIdx.x;
    uint32_t h=0x811c9dc5u ^ uint32_t(lane);
#pragma unroll
    for (int c=0; c<4; ++c) {
#pragma unroll
        for (int r=0; r<4; ++r) {
            const float x=gdn_ld(evs, &base[(col+c)*128+r*32+lane]);
            s[c*4+r]=x;
            h=gdn_lazy_mix(h, __float_as_uint(x));
        }
    }
    h=gdn_lazy_warp_hash<32>(h);
    gdn_g1_transpose(s);
    return h;
}

// Hash the actual values being stored, in the old hash order. There is no
// store->syncwarp->global reload. Preserve the l8 registers for the main loop.
static __device__ __forceinline__ uint32_t gdn_g1_store_base(
        float * base, int col, bool evs, const float (&s)[GDN_L8_ROWS]) {
    const int lane=threadIdx.x;
    float old[GDN_L8_ROWS];
#pragma unroll
    for (int j=0; j<GDN_L8_ROWS; ++j) old[j]=s[j];
    gdn_g1_transpose(old);
    uint32_t h=0x811c9dc5u ^ uint32_t(lane);
#pragma unroll
    for (int c=0; c<4; ++c) {
#pragma unroll
        for (int r=0; r<4; ++r) {
            const float x=old[c*4+r];
            gdn_st(evs, &base[(col+c)*128+r*32+lane], x);
            h=gdn_lazy_mix(h, __float_as_uint(x));
        }
    }
    return gdn_lazy_warp_hash<32>(h);
}

static __device__ __forceinline__ void gdn_g1_copy4(float * dst, const float * src) {
    const unsigned int sm=static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm), "l"(src) : "memory");
}

// One warp cooperatively copies its K row, four V columns and two gates.
// GD1 and GD2 remain unchanged, including odd (130-float) GD2 token strides.
// Four-byte copies require no new alignment or repacking of either format.
static __device__ __forceinline__ void gdn_g1_stage_replay(
        float * dst, const float * rec, const float * shared_rec, bool v2, int t) {
    const int lane=threadIdx.x;
    const float * tok=v2 ? shared_rec+GDN_LAZY_HDR+t*130 : rec+GDN_LAZY_HDR+t*146;
    const float * tv=v2 ? rec+GDN_LAZY_HDR+t*16 : tok+128;
#pragma unroll
    for (int r=0; r<4; ++r) {
        const int i=r*32+lane;
        gdn_g1_copy4(&dst[(r*8+lane%8)*4+lane/8], &tok[i]);
    }
    if (lane<4) gdn_g1_copy4(&dst[128+lane], &tv[threadIdx.y*4+lane]);
    if (lane<2) gdn_g1_copy4(&dst[132+lane], &tok[(v2 ? 128 : 144)+lane]);
    asm volatile("cp.async.commit_group;" ::: "memory");
}

static __device__ __forceinline__ bool gdn_g1_replay(
        float (&s)[GDN_L8_ROWS], const float * base, int col, bool evs, uint32_t checksum,
        const float * rec, const float * shared_rec, bool v2, int count) {
    // 4352 B total. Private warp buffers preserve the old per-warp checksum
    // fallback behavior without adding a CTA barrier inside the replay loop.
    __shared__ __align__(16) float buffer[4][2][136];
    const int lane=threadIdx.x, n=lane%8, grp=lane/8;
    if (count>0) gdn_g1_stage_replay(buffer[threadIdx.y][0],rec,shared_rec,v2,0);
    if (count>1) gdn_g1_stage_replay(buffer[threadIdx.y][1],rec,shared_rec,v2,1);
    // Metadata was checked by the caller. Prefetch the first two payloads
    // while the base and its full old checksum are loaded and transposed.
    if (gdn_g1_load_base(base,col,evs,s)!=checksum) {
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncwarp();
        return false;
    }
    if (count==0) return true;
#pragma unroll 1
    for (int t=0; t<count; ++t) {
        // Wait for the current token, leaving only the following token in flight.
        // The earlier current-input copy group is also completed by this wait.
        if (t+1<count) { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
        else { asm volatile("cp.async.wait_all;" ::: "memory"); }
        __syncwarp();
        const float * tok=buffer[threadIdx.y][t&1];
        const float g_val=tok[132], beta_val=tok[133];
        float kr[GDN_L8_ROWS];
        float p[4];
#pragma unroll
        for (int r=0; r<4; ++r) {
            const float4 k4=*(const float4 *)&tok[(r*8+n)*4];
            kr[r]=k4.x; kr[4+r]=k4.y; kr[8+r]=k4.z; kr[12+r]=k4.w;
        }
#pragma unroll
        for (int m=0; m<4; ++m) {
            p[m]=0.0f;
#pragma unroll
            for (int r=0; r<4; ++r) p[m]+=s[m*4+r]*kr[m*4+r];
        }
        const float kv_col=gdn_l8_reduce(p);
        const float delta_col=(tok[128+grp]-g_val*kv_col)*beta_val;
#pragma unroll
        for (int j=0; j<GDN_L8_ROWS; ++j) s[j]=g_val*s[j]+kr[j]*delta_col;
        // All shared reads finish before this buffer is recycled for token t+2.
        __syncwarp();
        if (t+2<count) gdn_g1_stage_replay(buffer[threadIdx.y][t&1],rec,shared_rec,v2,t+2);
    }
    return true;
}

template <bool RAW, bool G_PRECOMPUTED, bool LAZY, bool REUSE_QUOTIENT = false, bool G1 = false>
__global__ void __launch_bounds__(128, 6)
gated_delta_net_dedup_l8v_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                        const float * raw_dt_bias, const float * raw_a, const float * curr_state, float * dst,
                        float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
                        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic,
                        float scale, int64_t state_slot_stride, int K, const int32_t * s_ids, int64_t s_row_stride,
                        int lazy, uint32_t * generation_counter) {
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

    // One ticket per CTA, exactly as the old GD2L protocol. Same-cache launches
    // must be serialized; captured graph clones share this counter allocation.
    // A host cache import sets REHYDRATE on every head before launching. That
    // one pass reads either record format and writes GD1L only. Thus it never
    // overwrites an imported shared K buffer whose parity belongs to another
    // counter. The eighth ticket resets the next launch's counter to zero.
    __shared__ uint32_t ticket;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ticket = generation_counter ? atomicAdd(generation_counter + h_idx, 1u) : GDN_DEDUP_REHYDRATE;
        if (generation_counter && (ticket & GDN_DEDUP_REHYDRATE) && (ticket & 7u) == 7u) {
            atomicExch(generation_counter + h_idx, 0u);
        }
    }
    __syncthreads();
    const uint32_t gen = (ticket & ~GDN_DEDUP_REHYDRATE) / 8u;
    const bool dedup = LAZY && generation_counter && !(ticket & GDN_DEDUP_REHYDRATE) && K >= 12;

    // All n_tokens (<= GDN_L8V_MAX_TOKENS) tokens' k rows (in the order lane n reads them: [r][n][m]), the block's
    // 16 v columns, g and beta are copied to shared memory up front (cp.async, overlapping the state load).
    constexpr int TOK_SM = S_v + 16 + 4;
    __shared__ __align__(16) float stage[GDN_L8V_MAX_TOKENS][TOK_SM];
    {
        const int tid = threadIdx.y * 32 + lane;
        const float * k_base  = k + iq3 * sq3 + iq1 * sq1;
        const float * v_base  = v + sequence * sv3 + h_idx * sv1 + blockIdx.z * 16;
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
            } else {
                continue;  // gates are produced once by warp zero below; trailing padding
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
        int64_t slot_q = 0;
        if constexpr (REUSE_QUOTIENT) {
            if (state_slot_stride > 0 && diff >= 0) {
                slot_q = diff / state_slot_stride;
            }
        }
        // Positive stride and nonnegative diff give slot_q*stride <= diff <= INT64_MAX.
        // The product check is exact and reuses the quotient needed by rollback.
        if (state_slot_stride > 0 && diff >= 0 &&
                (REUSE_QUOTIENT ? slot_q * state_slot_stride == diff : diff % state_slot_stride == 0)) {
            const int64_t rb = REUSE_QUOTIENT ? slot_q : diff / state_slot_stride;
            const float * rec = state + (2 + blockIdx.z) * state_slot_stride;
            const uint32_t * hdr = (const uint32_t *) rec;
            const int W = (int) hdr[1];
            const float * base = state + state_slot_stride;
            const bool v2 = hdr[0] == GDN_DEDUP_MAGIC;
            const uint32_t rgen = hdr[3];
            const float * shared_rec = v2 && K >= 12 ? state + (10 + (rgen & 1)) * state_slot_stride : nullptr;
            const uint32_t * sh = reinterpret_cast<const uint32_t *>(shared_rec);
            const bool v2_ok = shared_rec && sh[0] == GDN_DEDUP_MAGIC && sh[1] == uint32_t(W) && sh[3] == rgen &&
                               (!dedup || rgen + 1u == gen);
            if ((hdr[0] == GDN_LAZY_MAGIC || v2_ok) && W >= 1 && W <= K && rb <= W && 2 + (int) gridDim.z <= K &&
                    (rb > 0 || (hdr[2] == GDN_LAZY_NOFINAL && gdn_lazy_slot0_marked(curr_state, col, S_v)))) {
                if constexpr (G1) {
                    replayed=gdn_g1_replay(s,base,col,evs,hdr[4+threadIdx.y],rec,shared_rec,v2,W-int(rb));
                } else if (gdn_l8_base_checksum(base, col, lane) == hdr[4 + threadIdx.y]) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        s[m*4 + r] = gdn_ld(evs, &base[my_col * S_v + r * 32 + n + 8 * m]);
                    }
                }
                const int n_replay = W - (int) rb;
                for (int t = 0; t < n_replay; t++) {
                    const float * tok = v2 ? shared_rec + GDN_LAZY_HDR + t * (S_v + 2)
                                           : rec + GDN_LAZY_HDR + t * gdn_lazy_token_stride<S_v, 4>();
                    const float * tv = v2 ? rec + GDN_LAZY_HDR + t * 16 : tok + S_v;
                    const int gb = v2 ? S_v : S_v + 16;
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
    // Uniform within each CTA even if a corrupted per-warp checksum declines
    // replay. Eager state above slot1 remains in its column-disjoint layout.
    // the staged inputs are complete, and every warp has finished reading the previous lazy record before any
    // warp rewrites it
    asm volatile("cp.async.wait_all;");
    // One producer per CTA owns each token's gates. The existing CTA barrier
    // publishes activated values to the four V-column warps without a new sync.
    if (threadIdx.y == 0 && lane < n_tokens) {
        const int64_t gate_offset = sequence * sb3 + h_idx * sb1 + lane * sb2;
        float beta_owned = beta[gate_offset];
        if constexpr (RAW) {
            beta_owned = 1.0f / (1.0f + expf(-beta_owned));
        }
        float g_owned = g[gate_offset];
        if constexpr (RAW) {
            const float x = g_owned + raw_dt_bias[h_idx];
            g_owned = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        }
        stage[lane][S_v + 16] = G_PRECOMPUTED ? g_owned : expf(g_owned);
        stage[lane][S_v + 17] = beta_owned;
    }
    const int any_eager = __syncthreads_or(!replayed);
    const int64_t input_diff = curr_state - state;
    lazy = LAZY && (!any_eager || input_diff == 0 || input_diff == state_slot_stride);

    // the gates of token t, activated by lane t with the main loop's expressions, are broadcast in its step
    float g_lane = 0.0f, beta_lane = 0.0f;
    if (lane < n_tokens) {
        beta_lane = stage[lane][S_v + 17];
        g_lane = stage[lane][S_v + 16];
    }
    const int lazy_W  = (int) (n_tokens < K ? n_tokens : K);
    uint32_t  lazy_cs = 0;
    const int tid     = threadIdx.y * 32 + lane;
    if (lazy) {
        // gdn_lazy_store_base at token 0, and gdn_lazy_store_inputs of every token from the staged inputs, before
        // the recurrence: the loop below then stores nothing to global memory
        float * base = state + state_slot_stride;
        if constexpr (G1) {
            lazy_cs=gdn_g1_store_base(base,col,evs,s);
        } else {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                gdn_st(evs, &base[my_col * S_v + r * 32 + n + 8 * m], s[m*4 + r]);
            }
        }
        __syncwarp();
        lazy_cs = gdn_l8_base_checksum(base, col, lane);
        }

        float * rec = state + (2 + blockIdx.z) * state_slot_stride + GDN_LAZY_HDR;
        if (dedup) {
            // The previous generation's K area is read-only for this launch.
            float * shared_rec = state + (10 + (gen & 1)) * state_slot_stride + GDN_LAZY_HDR;
            for (int e = tid; e < int(n_tokens) * 16; e += 128) {
                gdn_st(evs, &rec[e], stage[e / 16][S_v + e % 16]);
            }
            if (blockIdx.z == 0) {
                for (int e = tid; e < int(n_tokens) * S_v; e += 128) {
                    const int t = e / S_v, i = e % S_v;
                    gdn_st(evs, &shared_rec[t * (S_v + 2) + i],
                        stage[t][((i / 32) * 8 + i % 8) * 4 + (i % 32) / 8]);
                }
                if (threadIdx.y == 0 && lane < n_tokens) {
                    gdn_st(evs, &shared_rec[lane * (S_v + 2) + S_v], g_lane);
                    gdn_st(evs, &shared_rec[lane * (S_v + 2) + S_v + 1], beta_lane);
                }
            }
        } else {
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

        if (!lazy) {
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
        gdn_dedup_store_header(state + (2 + blockIdx.z) * state_slot_stride,
                         dedup ? state + (10 + (gen & 1)) * state_slot_stride : nullptr,
                         lazy_W, lazy_cs, lane, gen);
    }
}
