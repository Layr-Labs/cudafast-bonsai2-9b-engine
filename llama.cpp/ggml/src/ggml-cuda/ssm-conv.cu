#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// x(t, c): at t*x_t + c*x_c, or for t >= t_tail in x_tail, a copy of those rows (C floats per row)
struct ssm_conv_front_x {
    const float * x;
    const float * x_tail;
    int64_t       t_tail;
    int64_t       x_t;
    int64_t       x_c;
    int64_t       C;

    __device__ __forceinline__ float operator()(const int64_t t, const int64_t c) const {
        return t >= t_tail ? x_tail[(t - t_tail) * C + c] : x[t * x_t + c * x_c];
    }
};

// conv_input (j, c) for j >= n_prev: x(j - n_prev, c); for j < n_prev: st, the block's staged conv states, element
// (j, c) at st[(c - cs0)*n_prev + j]
static __device__ __forceinline__ float ssm_conv_front_input(const float * st, const int64_t cs0,
        const ssm_conv_front_x & x, const int64_t c, const int64_t j, const int n_prev) {
    return j < n_prev ? st[(c - cs0) * n_prev + j] : x(j - n_prev, c);
}

// Two kinds of blocks.
// - A v block owns 32 channels (from c_v0) for all tokens; warp s of the block computes tokens [s*seg_len,
//   (s+1)*seg_len) of the SILU output y. y may alias x shifted by `hold` token rows (y(t, c) at x(t - hold, c),
//   hold >= 0), so a warp writes no output before it has read the inputs at that place: every read of x by another
//   warp that an output can overwrite -- the 3-column window at a segment start, the snapshot columns -- happens
//   before the second barrier, and the first `hold` outputs of a segment, which land on the previous segment's
//   inputs, are written only after the third one (the end of every read).
// - With qk, the channels [0, c_v0) are 128-channel heads whose SILU output is only read by an L2_NORM: a qk block
//   computes head h, tokens [ch*qk_tt, (ch+1)*qk_tt) into shared memory and writes them l2-normalized to qk (same
//   per-row arithmetic as l2_norm_f32<32>); y is not written for those channels, nor does qk overlap x.
// A block stages the conv states of its channels before the first barrier and writes the snapshots of its channels
// after it, so a snapshot may land on the gathered conv-state row (states = gather_base + gather_ids[0]*gather_row);
// only the block that owns a channel reads its conv states (v blocks; qk blocks of chunk 0). x rows that qk lands on
// are read from a copy (xs.x_tail).
template <bool apply_silu, int n_seg, int unroll, int qk_tt, int qk_layout = 0>
static __global__ void __launch_bounds__(32 * n_seg) ssm_conv_front_f32(const float * states, const int32_t * gather_ids,
        const int64_t gather_row, const float * w, const float * bias, float * y, float * qk,
        const ggml_cuda_ssm_conv_snaps snaps, const int64_t C, const int64_t c_v0, const int n_vblocks,
        const int n_chunks, const int64_t n_t, const int64_t seg_len, const ssm_conv_front_x xs,
        const int64_t w_c, const int64_t y_t, const int hold, const float eps, const float * zero_row,
        const float zero_scale, const float zero_bias) {
    constexpr int d_conv  = 4;
    constexpr int n_prev  = d_conv - 1;
    constexpr int head    = 128;
    constexpr int threads = 32 * n_seg;
    static_assert(n_seg % 4 == 0, "a qk block needs 128-thread groups");
    __shared__ float held[n_seg][GGML_CUDA_SSM_CONV_MAX_HOLD][WARP_SIZE];
    __shared__ float st[head * n_prev];
    __shared__ float buf[qk_tt][head];

    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();

    if (gather_ids != nullptr) {
        states += (int64_t) gather_ids[0] * gather_row;
    }
    // the gathered row is the one a deferred in-place SCALE would have rewritten: apply it (scale_f32's expression)
    const bool zeroed = zero_row != nullptr && states == zero_row;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    // conv output (before SILU) for token t from the window and x(t); the window then moves by one column
    float wr[d_conv];
    float b = 0.0f;
    const auto conv = [&](float * win, const float xt) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_prev; ++j) {
            sumf += win[j] * wr[j];
        }
        sumf += xt * wr[n_prev];
        sumf += b;
#pragma unroll
        for (int j = 0; j < n_prev - 1; ++j) {
            win[j] = win[j + 1];
        }
        win[n_prev - 1] = xt;
        return apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    };

    if ((int) blockIdx.x < n_vblocks) {
        const int     seg = threadIdx.y;
        const int64_t c0  = c_v0 + (int64_t) blockIdx.x * WARP_SIZE;
        const int64_t c   = c0 + threadIdx.x;
        const int64_t t0  = (int64_t) seg * seg_len;
        const int64_t t1  = min(n_t, t0 + seg_len);
        const bool    run = c < C && t0 < t1;

        for (int e = tid; e < WARP_SIZE * n_prev; e += threads) {
            if (c0 + e / n_prev < C) {
                const float v = states[c0 * n_prev + e];
                st[e] = zeroed ? zero_scale * v + zero_bias : v;
            }
        }
        __syncthreads();

        // snapshots of this block's channels; consecutive threads write consecutive floats
        const int per_snap = snaps.width * WARP_SIZE;
        for (int e = tid; e < snaps.n * per_snap; e += threads) {
            const int     k  = e / per_snap;
            const int     r  = e - k * per_snap;
            const int64_t cs = c0 + r / snaps.width;
            const int     j  = r % snaps.width;
            if (cs < C) {
                snaps.dst[k][cs * snaps.width + j] = ssm_conv_front_input(st, c0, xs, cs, snaps.col[k] + j, n_prev);
            }
        }

        float win[n_prev];  // conv_input columns t .. t + n_prev - 1 for output t
        if (run) {
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                wr[j] = w[c * w_c + j];
            }
            b = bias != nullptr ? bias[c] : 0.0f;
#pragma unroll
            for (int j = 0; j < n_prev; ++j) {
                win[j] = ssm_conv_front_input(st, c0, xs, c, t0 + j, n_prev);
            }
        }
        __syncthreads();

        if (run) {
            float * yc = y + c;
            const auto step = [&](const int64_t t, const float xt) {
                const float v = conv(win, xt);
                if (t - t0 < hold) {
                    held[seg][t - t0][threadIdx.x] = v;
                } else {
                    yc[t * y_t] = v;
                }
            };
            int64_t t = t0;
            for (; t + unroll <= t1; t += unroll) {
                float xu[unroll];  // conv_input column t + u + n_prev
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    xu[u] = xs(t + u, c);
                }
#pragma unroll
                for (int u = 0; u < unroll; ++u) {
                    step(t + u, xu[u]);
                }
            }
            for (; t < t1; ++t) {
                step(t, xs(t, c));
            }
        }

        if (hold > 0) {
            __syncthreads();
            if (run) {
                for (int h = 0; h < hold && t0 + h < t1; ++h) {
                    y[(t0 + h) * y_t + c] = held[seg][h][threadIdx.x];
                }
            }
        }
        return;
    }

    // qk block: head h, chunk ch
    constexpr int n_sub = threads / head;
    const int     q     = blockIdx.x - n_vblocks;
    const int     h     = q / n_chunks;
    const int     ch    = q % n_chunks;
    const int64_t c0    = (int64_t) h * head;
    const int     cl    = tid % head;
    const int64_t c     = c0 + cl;
    const int64_t tc0   = (int64_t) ch * qk_tt;
    const int64_t tc1   = min(n_t, tc0 + qk_tt);
    const int64_t sub   = (qk_tt + n_sub - 1) / n_sub;
    const int64_t t0    = tc0 + (tid / head) * sub;
    const int64_t t1    = min(tc1, t0 + sub);

    if (ch == 0) {
        for (int e = tid; e < head * n_prev; e += threads) {
            const float v = states[c0 * n_prev + e];
            st[e] = zeroed ? zero_scale * v + zero_bias : v;
        }
    }
    __syncthreads();
    if (ch == 0) {
        const int per_snap = snaps.width * head;
        for (int e = tid; e < snaps.n * per_snap; e += threads) {
            const int     k  = e / per_snap;
            const int     r  = e - k * per_snap;
            const int64_t cs = c0 + r / snaps.width;
            const int     j  = r % snaps.width;
            snaps.dst[k][cs * snaps.width + j] = ssm_conv_front_input(st, c0, xs, cs, snaps.col[k] + j, n_prev);
        }
    }

    if (t0 < t1) {
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            wr[j] = w[c * w_c + j];
        }
        b = bias != nullptr ? bias[c] : 0.0f;
        float win[n_prev];
#pragma unroll
        for (int j = 0; j < n_prev; ++j) {
            win[j] = ssm_conv_front_input(st, c0, xs, c, t0 + j, n_prev);
        }
        int64_t t = t0;
        for (; t + unroll <= t1; t += unroll) {
            float xu[unroll];
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                xu[u] = xs(t + u, c);
            }
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                buf[t + u - tc0][cl] = conv(win, xu[u]);
            }
        }
        for (; t < t1; ++t) {
            buf[t - tc0][cl] = conv(win, xs(t, c));
        }
    }
    __syncthreads();

    // l2_norm_f32<32> per (token, head) row: lane l adds the squares of columns l, l + 32, l + 64, l + 96 in order,
    // then the warp sums the lanes
    for (int64_t t = tc0 + tid / WARP_SIZE; t < tc1; t += n_seg) {
        float tmp = 0.0f;
#pragma unroll
        for (int i = 0; i < head / WARP_SIZE; ++i) {
            const float xi = buf[t - tc0][i * WARP_SIZE + threadIdx.x];
            tmp += xi * xi;
        }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
        for (int i = 0; i < head / WARP_SIZE; ++i) {
            // The permutation changes addresses only. Each warp still writes the same 32 consecutive words.
            const int col = qk_layout ? i * WARP_SIZE + (threadIdx.x % 8) * 4 + threadIdx.x / 8
                                      : i * WARP_SIZE + threadIdx.x;
            const int64_t row = qk_layout == 2 ? ((int64_t) h * n_t + t) * head : t * c_v0 + c0;
            qk[row + col] = scale * buf[t - tc0][i * WARP_SIZE + threadIdx.x];
        }
    }
}

void ggml_cuda_ssm_conv_front(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_input, const ggml_tensor * ssm,
                              const ggml_tensor * out, const bool apply_silu, const ggml_cuda_ssm_conv_snaps & snaps,
                              const ggml_cuda_ssm_conv_front_ext & ext) {
    const ggml_tensor * states = conv_input->src[0];
    const ggml_tensor * xv     = conv_input->src[1];  // [n_t, C] transposed view of the token-major x
    const ggml_tensor * w      = ssm->src[1];
    GGML_ASSERT(w->ne[0] == 4 && states->ne[0] == 3 && ext.hold >= 0 && ext.hold <= GGML_CUDA_SSM_CONV_MAX_HOLD);

    const int64_t C   = conv_input->ne[1];
    const int64_t n_t = xv->ne[0];
    const int64_t x_t = xv->nb[0] / sizeof(float);
    const int64_t x_c = xv->nb[1] / sizeof(float);
    const int64_t w_c = w->nb[1] / sizeof(float);
    const int64_t y_t = out->nb[1] / sizeof(float);
    const int     cc  = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    cudaStream_t  stream = ctx.stream();
    GGML_ASSERT(ext.qk == nullptr || (ext.qk_channels % 128 == 0 && ext.qk_channels <= C));
    GGML_ASSERT(ext.gather_ids == nullptr || !ext.copy_states);

    // conv_states under the output: read a copy, made before any output is written
    ggml_cuda_pool_alloc<float> states_copy(ctx.pool());
    const float *               states_d = ext.gather_ids ? ext.gather_base : (const float *) states->data;
    if (ext.copy_states) {
        states_copy.alloc(ggml_nelements(states));
        CUDA_CHECK(cudaMemcpyAsync(states_copy.get(), states_d, ggml_nbytes(states), cudaMemcpyDeviceToDevice, stream));
        states_d = states_copy.get();
    }
    // x rows the qk output lands on: read a copy, made before any output is written
    ggml_cuda_pool_alloc<float> x_tail(ctx.pool());
    ssm_conv_front_x            xs = { (const float *) xv->data, nullptr, n_t, x_t, x_c, C };
    if (ext.x_tail_t0 < n_t) {
        GGML_ASSERT(ext.qk != nullptr && ext.x_tail_t0 >= 0 && x_c == 1);
        const int64_t rows = n_t - ext.x_tail_t0;
        x_tail.alloc(rows * C);
        CUDA_CHECK(cudaMemcpy2DAsync(x_tail.get(), C * sizeof(float), xs.x + ext.x_tail_t0 * x_t, x_t * sizeof(float),
                                     C * sizeof(float), rows, cudaMemcpyDeviceToDevice, stream));
        xs.x_tail = x_tail.get();
        xs.t_tail = ext.x_tail_t0;
    }
    const float * w_d  = (const float *) w->data;
    float *       y_d  = (float *) out->data;
    const int64_t c_v0 = ext.qk ? ext.qk_channels : 0;
    const int     n_vb = (int) ((C - c_v0 + WARP_SIZE - 1) / WARP_SIZE);

    constexpr int unroll = 8;
    auto launch = [&](auto silu_c, auto seg_c, auto tt_c, auto layout_c) {
        constexpr bool silu     = decltype(silu_c)::value;
        constexpr int  n_seg    = decltype(seg_c)::value;
        constexpr int  qk_tt    = decltype(tt_c)::value;
        const int64_t  seg_len  = (n_t + n_seg - 1) / n_seg;
        const int      n_chunks = (int) ((n_t + qk_tt - 1) / qk_tt);
        const int      n_qkb    = ext.qk ? (int) (c_v0 / 128) * n_chunks : 0;
        ggml_cuda_kernel_launch_gb10(cc, ssm_conv_front_f32<silu, n_seg, unroll, qk_tt, decltype(layout_c)::value>,
            ggml_cuda_kernel_launch_params(dim3((unsigned int) (n_vb + n_qkb)), dim3(WARP_SIZE, n_seg), 0, stream),
            states_d, ext.gather_ids, ext.gather_row, w_d, (const float *) nullptr, y_d, ext.qk, snaps, C, c_v0,
            n_vb, n_chunks, n_t, seg_len, xs, w_c, y_t, ext.hold, ext.eps, ext.zero_row, ext.zero_scale, ext.zero_bias);
    };
    // Only the prefill SILU/L2 producer has experimental layouts; every unmatched path is the original template.
    if (n_t >= 64 && apply_silu && ext.qk && ext.qk_layout == 2) {
        launch(std::true_type{}, std::integral_constant<int, 8>{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 2>{});
    } else if (n_t >= 64) {
        if (apply_silu) launch(std::true_type{},  std::integral_constant<int, 8>{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 0>{});
        else            launch(std::false_type{}, std::integral_constant<int, 8>{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 0>{});
    } else {
        if (apply_silu) launch(std::true_type{},  std::integral_constant<int, 4>{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 0>{});
        else            launch(std::false_type{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 0>{});
    }
}
