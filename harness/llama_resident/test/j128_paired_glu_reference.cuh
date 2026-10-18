// Frozen verified 067 compute, segment and kernel.
template <bool DT = false>
static __device__ __forceinline__ void j128_paired_reference_compute(float (&acc)[2][4][4], const uint32_t xs, const uint32_t ys,
        const uint32_t dbt = 0) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
    constexpr int XS = 4*ggml_cuda_pq2_j128_xs; // bytes
    constexpr int YS = 4*ggml_cuda_pq2_j128_ys;
    // ldmatrix row addresses of this lane: A (x4): rows lane%16, ints (lane/16)*4; B (x2): columns lane%8,
    // ints 4 + ((lane/8)%2)*4 (the first 4 ints of a column record are the D4 scales).
    const uint32_t xa = xs + (lane % 16)*XS + (lane / 16)*16;
    const uint32_t ya = ys + (lane % 8)*YS + 16 + ((lane / 8) % 2)*16;
    const uint32_t xd = xs + g*XS + 4*32;   // dA of row g
    const uint32_t yd = ys + 2*c*YS;        // D4 scales of column 2c

    float dA[2][2], ndA[2][2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            dA [rt][h] = ggml_cuda_pq2_j128_lds_f32(xd + (rt*16 + 8*h)*XS);
            ndA[rt][h] = -12582912.0f*dA[rt][h];
        }
    }

    // sub-block outer: A of one sub-block live (8 registers), dB per (column tile, sub-block)
#pragma unroll 4
    for (int s = 0; s < 4; ++s) {
        int A[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
            ggml_cuda_pq2_j128_ldsm_x4(A[rt], xa + rt*16*XS + s*32);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            float dB[2];
            if constexpr (DT) {
                // transposed D4 scales: [s][column] floats of the warp's columns, columns 2c, 2c+1 adjacent
                const float2 v = ggml_cuda_pq2_j128_lds_v2f32(dbt + s*512 + (ct*8 + 2*c)*4);
                dB[0] = v.x; dB[1] = v.y;
            } else {
                dB[0] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + 4*s);
                dB[1] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + YS + 4*s);
            }
            int B[2];
            ggml_cuda_pq2_j128_ldsm_x2(B, ya + ct*8*YS + s*32);
#pragma unroll
            for (int rt = 0; rt < 2; ++rt) {
                int C[4];
                ggml_cuda_pq2_j128_mma(C, A[rt], B[0], B[1]);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float t = __fmaf_rn(__int_as_float(C[l]), dA[rt][l/2], ndA[rt][l/2]);
                    acc[rt][ct][l] = __fmaf_rn(t, dB[l%2], acc[rt][ct][l]);
                }
            }
        }
    }
}
// Frozen attack067 bulk plus transpose reference, itself exact against promoted.
template <bool PURE>
static __device__ __forceinline__ void j128_fused_reference_segment(
        float (&acc)[2][4][4], const uint32_t smem, const char * __restrict__ x, const int * y,
        const int row0, const int stride_row_x, const int ncols_y, const int col0,
        const int kb_start, const int kb_stop, uint32_t & ph) {
    constexpr int G  = 64;                      // threads per column group (warps w%4 == wc)
    constexpr int XS = 4*ggml_cuda_pq2_j128_xs; // bytes per X row
    constexpr int YS = 4*ggml_cuda_pq2_j128_ys; // bytes per Y column record
    constexpr uint32_t YBYTES = 128*YS;
    constexpr uint32_t XBYTES = 64*XS;
    constexpr int HINT_D = 4; // L2 hint of the X rows 4 K blocks ahead
    constexpr int BS = (int) sizeof(block_pq2_0);

    const int tid = threadIdx.y*32 + threadIdx.x;
    const int wr  = threadIdx.y / 4;
    const int wc  = threadIdx.y % 4;
    const int m   = (threadIdx.y / 4)*32 + threadIdx.x; // member index in the column group

    // X task of this thread: row tid/4 of the slice, quarter kq of the block
    const int kq = tid % 4;
    const char * xblk = x + ((int64_t) (row0 + tid/4)*stride_row_x + kb_start)*BS;
    const int hint_off = kq == 1 ? BS - 1 : 0; // lanes kq 0/1 hint the first/last byte of the block
    const bool hint_lane = kq < 2;
    const uint32_t xst = smem + 2*YBYTES + (tid/4)*XS + kq*32; // X tile store address

    const int64_t y_kb_bytes = (int64_t) ncols_y*YS;
    const char * yb = (const char *) y + kb_start*y_kb_bytes + (int64_t) (col0 + wc*32)*YS; // column group's records
    const uint32_t ycp = smem + wc*32*YS; // column group's records in Y buffer 0

    const uint32_t xw = smem + 2*YBYTES + wr*32*XS; // warp's X rows
    const uint32_t yw = smem + wc*32*YS;            // warp's Y columns (buffer 0)

    if constexpr (PURE) {
        GGML_UNUSED(ph);
        ggml_cuda_pq2_j128_xraw xr;

        // prologue: block kb_start (the global X words are read before the barrier: registers only)
        ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        __syncthreads(); // the previous segment's last compute no longer reads any tile
        ggml_cuda_pq2_j128_yload<G>(ycp, yb, m);
        ggml_cuda_pq2_j128_cp_commit();
        ggml_cuda_pq2_j128_xstore(xr, xst, kq);
        if (kb_start + 1 < kb_stop) {
            yb   += y_kb_bytes;
            xblk += BS;
            ggml_cuda_pq2_j128_yload<G>(ycp + YBYTES, yb, m);
            ggml_cuda_pq2_j128_cp_commit();
            ggml_cuda_pq2_j128_xload(xr, xblk, kq);
            ggml_cuda_pq2_j128_cp_wait<1>();
        } else {
            ggml_cuda_pq2_j128_cp_wait<0>();
        }
        __syncthreads();
        // invariant at the top of iteration kb: xblk/yb point at block kb + 1 (when it exists)
        for (int kb = kb_start; kb < kb_stop; ++kb) {
            const uint32_t cur = (kb - kb_start) & 1;
            if (hint_lane && kb + HINT_D < kb_stop) {
                ggml_cuda_pq2_j128_hint(xblk + (HINT_D - 1)*BS + hint_off);
            }
            j128_paired_reference_compute<false>(acc, xw, yw + cur*YBYTES);
            if (kb + 1 >= kb_stop) {
                break;
            }
            __syncthreads(); // X tile and Y buffer `cur` no longer read
            ggml_cuda_pq2_j128_xstore(xr, xst, kq);
            if (kb + 2 < kb_stop) {
                yb   += y_kb_bytes;
                xblk += BS;
                ggml_cuda_pq2_j128_yload<G>(ycp + cur*YBYTES, yb, m);
                ggml_cuda_pq2_j128_cp_commit();
                ggml_cuda_pq2_j128_xload(xr, xblk, kq);
                ggml_cuda_pq2_j128_cp_wait<1>();
            } else {
                ggml_cuda_pq2_j128_cp_wait<0>();
            }
            __syncthreads(); // X tile (block kb+1) and Y buffer `cur ^ 1` complete
        }
    } else {
        // Y through one bulk copy (TMA) per K block; after it lands, the D4 headers of the 128 columns are
        // transposed by threads 0..127 into dBT[buf][s][col] (2 KB per buffer) so that a lane reads the
        // scales of its two columns with one 8-byte load. mbarriers live in the padding of X row 0.
        const uint32_t bar  = smem + 2*YBYTES + 136;          // buffer b: X row b bytes 136..143 (pad of the 144-byte rows)
        const uint32_t dbt0 = smem + 2*YBYTES + XBYTES;       // 2 x 2048 bytes
        const char * ysrc = (const char *) y + kb_start*y_kb_bytes + (int64_t) col0*YS;
        auto transpose = [&](const uint32_t buf) { // threads 0..127: column tid of Y buffer buf
            if (tid < 128) {
                const uint4 h = ggml_cuda_pq2_j128_lds_v4(smem + buf*YBYTES + tid*YS);
                const uint32_t d = dbt0 + buf*2048 + tid*4;
                ggml_cuda_pq2_j128_sts_b32(d + 0*512, (int) h.x);
                ggml_cuda_pq2_j128_sts_b32(d + 1*512, (int) h.y);
                ggml_cuda_pq2_j128_sts_b32(d + 2*512, (int) h.z);
                ggml_cuda_pq2_j128_sts_b32(d + 3*512, (int) h.w);
            }
        };
        ggml_cuda_pq2_j128_xraw xr;
        ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        __syncthreads(); // the previous segment's last compute no longer reads any tile
        if (tid == 0) {
            ggml_cuda_pq2_j128_bulk_y(smem, ysrc, YBYTES, bar);
        }
        ggml_cuda_pq2_j128_xstore(xr, xst, kq);
        if (kb_start + 1 < kb_stop) {
            if (tid == 0) {
                ggml_cuda_pq2_j128_bulk_y(smem + YBYTES, ysrc + y_kb_bytes, YBYTES, bar + 144);
            }
            xblk += BS;
            ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        }
        ggml_cuda_pq2_j128_mbar_wait(bar, ph & 1);
        ph ^= 1u;
        transpose(0);
        __syncthreads();
        const uint32_t dbw = dbt0 + wc*32*4; // the warp's columns
        for (int kb = kb_start; kb < kb_stop; ++kb) {
            const uint32_t cur = (kb - kb_start) & 1;
            if (hint_lane && kb + HINT_D < kb_stop) {
                ggml_cuda_pq2_j128_hint(xblk + (HINT_D - 1)*BS + hint_off);
            }
            j128_paired_reference_compute<true>(acc, xw, yw + cur*YBYTES, dbw + cur*2048);
            if (kb + 1 >= kb_stop) {
                break;
            }
            __syncthreads(); // X tile, Y buffer `cur` and dBT `cur` no longer read
            ggml_cuda_pq2_j128_xstore(xr, xst, kq);
            if (kb + 2 < kb_stop) {
                if (tid == 0) {
                    ggml_cuda_pq2_j128_bulk_y(smem + cur*YBYTES, ysrc + (int64_t) (kb - kb_start + 2)*y_kb_bytes, YBYTES, bar + 144*cur);
                }
                xblk += BS;
                ggml_cuda_pq2_j128_xload(xr, xblk, kq);
            }
            ggml_cuda_pq2_j128_mbar_wait(bar + 144*(cur ^ 1), (ph >> (cur ^ 1)) & 1); // Y of block kb+1 landed
            ph ^= 1u << (cur ^ 1);
            transpose(cur ^ 1);
            __syncthreads(); // X tile (block kb+1) and dBT `cur ^ 1` complete
        }
        GGML_UNUSED(ycp); GGML_UNUSED(m); GGML_UNUSED(yb);
    }
}
static __device__ __forceinline__ void j128_fused_reference_res_hint(const float * res, const int stride_col, const int row0,
        const int col0, const int ncols_dst) {
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int j = col0 + tid / 2;
    if (j < ncols_dst) {
        ggml_cuda_pq2_j128_hint((const char *) (res + (int64_t) j*stride_col + row0 + (tid % 2)*32));
    }
}
template <bool PURE, bool ADD, bool GLU = false>
__launch_bounds__(256, 2)
static __global__ void mul_mat_q_pq2_j128_reference(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int ncols_dst, const int ntx, const float * res, float * dst_add) {
    constexpr int I    = 128; // tile of the old kernel
    constexpr int J    = 128;
    constexpr int ROWS = 64;
    constexpr int QS   = I / ROWS;
    constexpr int blocks_per_iter = MMQ_ITER_K / QK_PQ2_0; // segment alignment of the old kernel

    extern __shared__ int smem_pq2_j128[];
    const uint32_t smem = (uint32_t) __cvta_generic_to_shared(smem_pq2_j128);

    const int q = blockIdx.x % QS;
    const int b = blockIdx.x / QS;
    const int nb_old = gridDim.x / QS;

    // Same index math as mul_mat_q (nchannels = nsamples = 1): kbc runs over (it, jt, kb), jt fastest.
    const int nty = nrows_x / I;
    const int64_t W = (int64_t) ntx*nty*nkb;
    int kbc      = (int) ((int64_t) b      *W / nb_old);
    int kbc_stop = (int) ((int64_t) (b + 1)*W / nb_old);
    kbc      -= (kbc      % nkb) % blocks_per_iter;
    kbc_stop -= (kbc_stop % nkb) % blocks_per_iter;

    const int wr = threadIdx.y / 4;
    const int wc = threadIdx.y % 4;
    const int row_in_tile = q*ROWS + wr*32; // the warp's rows in the 128-row tile
    const int col_in_tile = wc*32;

    int kb0_start = kbc % nkb;
    int kb0_stop  = min(nkb, kb0_start + kbc_stop - kbc);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    // Programmatic stream serialization: L2 hints for the first 4 K blocks of this CTA's weight rows
    // (weights only, no memory read), then the grid dependency wait before y is read or dst/tmp_fixup
    // written; y/dst/tmp_fixup are not __restrict__ so nothing moves above the wait.
    if (kbc < kbc_stop) {
        const int tid = threadIdx.y*32 + threadIdx.x;
        const int it  = (kbc / nkb) / ntx;
        const int row = tid / 4;
        const int64_t stride_row_bytes = (int64_t) stride_row_x*(int64_t) sizeof(block_pq2_0);
        const char * p = x + (int64_t) (it*I + q*ROWS + row)*stride_row_bytes + (int64_t) kb0_start*(int64_t) sizeof(block_pq2_0);
        // 4 blocks = 136 bytes from a 2-byte aligned start: at most 6 sectors; 4 lanes x 2 hints
        const int k = tid % 4;
        const int n4 = min(4, kb0_stop - kb0_start)*(int) sizeof(block_pq2_0);
        if (32*k < n4) ggml_cuda_pq2_j128_hint(p + 32*k);
        if (32*(k + 4) < n4 + 32) ggml_cuda_pq2_j128_hint(p + min(32*(k + 4), n4 - 1));
    }
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
#endif

    if constexpr (!ADD) {
        GGML_UNUSED(res); GGML_UNUSED(dst_add);
    }
    uint32_t ph = 0; // stream-k: wait parity per Y buffer (bit b = buffer b)
    { // Both tile modes use the existing bulk-copy segment.
        const uint32_t bar = smem + (uint32_t) (2*128*4*ggml_cuda_pq2_j128_ys) + 136u;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            ggml_cuda_pq2_j128_mbar_init(bar);
            ggml_cuda_pq2_j128_mbar_init(bar + 144);
            ggml_cuda_pq2_j128_mbar_fence_init();
        }
        __syncthreads();
    }
    if constexpr (PURE) {
        const int jt = b % ntx;
        const int it = b / ntx;
        float acc[2][4][4] = {{{0.0f}}};
        if constexpr (ADD) {
            j128_fused_reference_res_hint(res, stride_col_dst, it*I + q*ROWS, jt*J, ncols_dst);
        }
        j128_fused_reference_segment<false>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, 0, nkb, ph);
        if constexpr (ADD) {
            const int64_t off = (int64_t) jt*J*stride_col_dst + it*I;
            ggml_cuda_pq2_j128_store_add(acc, dst_add + off, res + off, stride_col_dst, row_in_tile, col_in_tile,
                                         ncols_dst - jt*J - 1);
        } else if constexpr (GLU) {
            ggml_cuda_pq2_j128_store_glu(acc, dst + (int64_t) jt*J*stride_col_dst + it*(I/2), stride_col_dst, row_in_tile,
                                         col_in_tile, ncols_dst - jt*J - 1);
        } else {
            ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                     ncols_dst - jt*J - 1);
        }
        return;
    }

    // One code path for every segment (a single inlined segment keeps one register allocation): a
    // segment that ends its tile writes dst; the trailing one (which ends inside a tile) writes the
    // partial sums of all 128 columns to the fixup buffer of block b and is necessarily the last.
    while (kbc < kbc_stop) {
        const int t  = kbc / nkb;
        const int jt = t % ntx;
        const int it = t / ntx;
        const bool ends_tile = kb0_stop == nkb;
        const bool whole     = ADD && ends_tile && kb0_start == 0; // completes its tile alone: adds the residual
        float acc[2][4][4] = {{{0.0f}}};
        if (whole) {
            j128_fused_reference_res_hint(res, stride_col_dst, it*I + q*ROWS, jt*J, ncols_dst);
        }
        j128_fused_reference_segment<PURE>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, kb0_start, kb0_stop, ph);
        if (!ends_tile) {
            ggml_cuda_pq2_j128_store(acc, tmp_fixup + (int64_t) b*(J*I), I, row_in_tile, col_in_tile, J - 1);
            return;
        }
        if constexpr (ADD) {
            if (whole) {
                const int64_t off = (int64_t) jt*J*stride_col_dst + it*I;
                ggml_cuda_pq2_j128_store_add(acc, dst_add + off, res + off, stride_col_dst, row_in_tile, col_in_tile,
                                             ncols_dst - jt*J - 1);
            } else {
                ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile,
                                         col_in_tile, ncols_dst - jt*J - 1);
            }
        } else {
            ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                     ncols_dst - jt*J - 1);
        }

        kbc += nkb;
        kbc -= kbc % nkb;
        kb0_start = 0;
        kb0_stop  = min(nkb, kbc_stop - kbc);
    }
}
