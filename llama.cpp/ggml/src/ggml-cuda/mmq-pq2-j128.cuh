#pragma once

// GB10 (sm_121) J=128 PQ2_0 x q8_1 matmul (prefill) at 16 warps per SM.
//
// Bit-exact replacement for mul_mat_q<GGML_TYPE_PQ2_0, 128, false> (+ its stream-k fixup):
//  * same work partition: CTA (b, q) runs every K segment that block b of the old kernel runs (the
//    same kbc arithmetic, including the alignment of segment bounds to the old 256-value K
//    iteration), for row half q (64 rows) of each 128 x 128 output tile. It writes dst for a
//    segment that ends its tile and tmp_fixup[b] (layout [j][i], stride 128) for the trailing one;
//    the unchanged mul_mat_q_stream_k_fixup completes the stream-k tiles. Pure tiling = one block
//    per tile.
//  * same accumulation per output element: sum starts at +0 for every segment and, for every
//    32-value sub-block in increasing K order, sum = fma(RN(idot*dA), dB, sum) with dA = (float)
//    x->d and dB = the q8_1 D4 scale, i.e. the old `sum += C*dA*dB` (I2F, FMUL, FFMA). idot is the
//    exact int32 dot product of the same 32 products (identical int8 tiles, same mma).
//  * 2-FFMA epilogue: the mma accumulator starts at 0x4B400000 (12582912.0f = 1.5*2^23); for
//    |idot| <= 2^22 (here |idot| <= 32*2*128) the int32 result is the float 12582912 + idot
//    exactly. ndA = -12582912*dA is exact (dA: <= 11 significant bits) and
//    fma(D_f, dA, ndA) = RN(idot*dA) = FMUL(I2F(idot), dA) (the product itself is exact, <= 24
//    significant bits). For dA > 0, and for dA == +0 with idot >= 0, the two are bit-identical
//    including the sign of zero (exhaustively tested over all fp16 dA and all idot).
// CTA = 64 rows x 128 columns, 8 warps, 2 CTAs per SM (16 warps, <= 128 registers, 46-50 KB of
// shared memory each); each warp owns a 32 x 32 output block (32 fp32 accumulators per thread).
// X (weights) are unpacked once per CTA into a shared int8 tile (same codec as
// ggml_cuda_mmq_load_tiles_pq2_0) from raw words fetched one K block ahead; the activation tile
// is double-buffered with cp.async.

#include "common.cuh"
// cp-async.cuh (no include guard) is included by mmq.cuh before this header

#include <cstdint>

#define GGML_CUDA_PQ2_J128_MAGIC 0x4B400000 // 12582912.0f = 1.5*2^23

static constexpr int ggml_cuda_pq2_j128_ys = 36; // ints per activation column record: 4 floats d4 + 32 ints qs (= MMQ_TILE_Y_K)
static constexpr int ggml_cuda_pq2_j128_xs = 36; // ints per weight row: 32 ints int8 values + dA + 3 pad (stride % 8 == 4)

static __device__ __forceinline__ void ggml_cuda_pq2_j128_mma(int (&d)[4], const int (&a)[4], const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    const int m = GGML_CUDA_PQ2_J128_MAGIC;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m));
#else
    GGML_UNUSED_VARS(d, a, b0, b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_ldsm_x4(int (&r)[4], const uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(GGML_USE_HIP)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_ldsm_x2(int (&r)[2], const uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(GGML_USE_HIP)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm volatile("cp.async.commit_group;");
#endif
}

template <int N>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_hint(const char * p) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
#else
    GGML_UNUSED(p);
#endif
}

static __device__ __forceinline__ float ggml_cuda_pq2_j128_lds_f32(const uint32_t addr) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
    return v;
}

static __device__ __forceinline__ uint4 ggml_cuda_pq2_j128_lds_v4(const uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_cp4(const uint32_t dst, const void * src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst), "l"(src));
#else
    GGML_UNUSED_VARS(dst, src);
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_sts_v4(const uint32_t addr, const int a, const int b, const int c, const int d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_sts_b32(const uint32_t addr, const int a) {
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(a) : "memory");
}

// Raw words of one thread's X task: row `row` of the slice, 8 code bytes kq*8..kq*8+7 of PQ2 block kb
// (the 32 weights of ints kq*8..kq*8+7), read as three aligned 32-bit words (the block is 2-byte
// aligned, C22c), plus the fp16 scale bits.
struct ggml_cuda_pq2_j128_xraw {
    uint32_t w0, w1, w2;
    uint32_t sh; // funnel shift: 0 or 16
    uint32_t d;
};

static __device__ __forceinline__ void ggml_cuda_pq2_j128_xload(ggml_cuda_pq2_j128_xraw & r, const char * blk, const int kq) {
    const uint8_t  * qsi = (const uint8_t *) blk + sizeof(ggml_half) + kq*8;
    const uint32_t   mis = (uint32_t) (reinterpret_cast<uintptr_t>(qsi) & 3); // 0 or 2
    const uint32_t * qwi = reinterpret_cast<const uint32_t *>(qsi - mis);
    r.w0 = qwi[0];
    r.w1 = qwi[1];
    r.w2 = qwi[1 + mis/2]; // w1 again when aligned: no read past the 8 bytes
    r.sh = 8*mis;
    r.d  = kq == 0 ? (uint32_t) *(const uint16_t *) blk : 0u;
}

// Same 2-bit -> int8 codec expressions as ggml_cuda_mmq_load_tiles_pq2_0; 8 ints to the row's ints
// kq*8..kq*8+7 (addr = their shared address), dA (float) to int 32 of the row (kq == 0).
static __device__ __forceinline__ void ggml_cuda_pq2_j128_xstore(const ggml_cuda_pq2_j128_xraw & r, const uint32_t addr, const int kq) {
    const uint32_t qw[2] = {__funnelshift_r(r.w0, r.w1, r.sh), __funnelshift_r(r.w1, r.w2, r.sh)};
    int v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = j % 2 == 0 ? int(int16_t(qw[j/2] & 0xFFFFu)) : int(qw[j/2]) >> 16;

        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        v[2*j + 0] = __byte_perm(qe, qo, 0x5140);
        v[2*j + 1] = __byte_perm(qe, qo, 0x7362);
    }
    ggml_cuda_pq2_j128_sts_v4(addr,      v[0], v[1], v[2], v[3]);
    ggml_cuda_pq2_j128_sts_v4(addr + 16, v[4], v[5], v[6], v[7]);
    if (kq == 0) {
        ggml_cuda_pq2_j128_sts_b32(addr + 4*32, __float_as_int(__half2float(__ushort_as_half((unsigned short) r.d))));
    }
}

// Activation tile of one K block: the 128 column records of the tile column, 18432 contiguous bytes.
// The 32 records of column group wc (4608 bytes) are copied by the G threads of that group
// (member index m), so a column group only depends on its own copies.
template <int G>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_yload(const uint32_t dst, const char * src, const int m) {
    constexpr int nchunks = 32*ggml_cuda_pq2_j128_ys*(int) sizeof(int)/16; // 288
#pragma unroll
    for (int c0 = 0; c0 < nchunks; c0 += G) {
        const int c = c0 + m;
        if (c0 + G <= nchunks || c < nchunks) {
            cp_async_cg_16<256>(dst + 16*c, src + 16*c);
        }
    }
}

// Named barrier with an immediate id (a register id makes ptxas reserve all 16 barriers per CTA).
// One K block (128 values) of the warp's 32 x 32 output block. acc[rt][ct][l]: rt = 16-row tile,
// ct = 8-column tile, l = mma C element (rows g, g+8 x columns 2c, 2c+1).
// xs/ys: shared addresses of the warp's first X row / first Y column record in the current buffers.
static __device__ __forceinline__ void ggml_cuda_pq2_j128_compute(float (&acc)[2][4][4], const uint32_t xs, const uint32_t ys) {
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
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int A[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
            ggml_cuda_pq2_j128_ldsm_x4(A[rt], xa + rt*16*XS + s*32);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            const float dB[2] = {ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + 4*s), ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + YS + 4*s)};
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

// One output segment of the CTA: rows [row0, row0 + 64) of the tile, its 128 columns from col0, K blocks
// [kb_start, kb_stop). block = (32, 8): warp w owns rows 32*(w/4) .. +31 and columns 32*(w%4) .. +31.
// Shared layout (bytes): Y buffers 0/1 [0, 2*18432), the X tile [36864, 46080), XR: raw X slots
// [46080, 50176). One X tile, two Y buffers: per K block, compute, barrier, unpack the next X block
// and start the copy of the one after, barrier.
// XR = 0: each thread holds the raw X words of the next block in registers (global loads issued one
// block ahead); XR = 1: they are staged in a per-thread shared slot with 4-byte cp.async, committed
// together with the Y copy of the same block.
template <int XR>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_segment(
        float (&acc)[2][4][4], const uint32_t smem, const char * __restrict__ x, const int * y,
        const int row0, const int stride_row_x, const int ncols_y, const int col0,
        const int kb_start, const int kb_stop) {
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

    if constexpr (XR) {
        const uint32_t slot = smem + 2*YBYTES + XBYTES + tid*16;
        auto xcopy = [&](const char * blk) { // the words of ggml_cuda_pq2_j128_xload
            const char * qsi = blk + sizeof(ggml_half) + kq*8;
            const uint32_t mis = (uint32_t) (reinterpret_cast<uintptr_t>(qsi) & 3);
            const char * qw = qsi - mis;
            ggml_cuda_pq2_j128_cp4(slot + 0, qw);
            ggml_cuda_pq2_j128_cp4(slot + 4, qw + 4);
            ggml_cuda_pq2_j128_cp4(slot + 8, qw + 4 + 2*mis); // w1 again when aligned
            if (kq == 0) {
                ggml_cuda_pq2_j128_cp4(slot + 12, (const char *) (reinterpret_cast<uintptr_t>(blk) & ~(uintptr_t) 3));
            }
        };
        auto xdecode = [&](const char * blk) {
            const uint4 v = ggml_cuda_pq2_j128_lds_v4(slot);
            const uint32_t b3 = (uint32_t) (reinterpret_cast<uintptr_t>(blk) & 3); // 0 or 2
            ggml_cuda_pq2_j128_xraw r;
            r.w0 = v.x; r.w1 = v.y; r.w2 = v.z;
            r.sh = 8*((b3 + 2) & 3); // misalignment of the code bytes blk + 2 + kq*8
            r.d  = kq == 0 ? (v.w >> (8*b3)) & 0xFFFFu : 0u;
            ggml_cuda_pq2_j128_xstore(r, xst, kq);
        };
        xcopy(xblk);
        ggml_cuda_pq2_j128_cp_commit();
        __syncthreads(); // the previous segment's last compute no longer reads any tile
        ggml_cuda_pq2_j128_yload<G>(ycp, yb, m);
        ggml_cuda_pq2_j128_cp_commit();
        ggml_cuda_pq2_j128_cp_wait<1>();
        xdecode(xblk);
        if (kb_start + 1 < kb_stop) {
            yb   += y_kb_bytes;
            xblk += BS;
            ggml_cuda_pq2_j128_yload<G>(ycp + YBYTES, yb, m);
            xcopy(xblk);
            ggml_cuda_pq2_j128_cp_commit();
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
            ggml_cuda_pq2_j128_compute(acc, xw, yw + cur*YBYTES);
            if (kb + 1 >= kb_stop) {
                break;
            }
            __syncthreads(); // X tile and Y buffer `cur` no longer read
            ggml_cuda_pq2_j128_cp_wait<0>(); // own copies of block kb+1 (raw X, Y) complete
            xdecode(xblk);
            if (kb + 2 < kb_stop) {
                yb   += y_kb_bytes;
                xblk += BS;
                ggml_cuda_pq2_j128_yload<G>(ycp + cur*YBYTES, yb, m);
                xcopy(xblk);
                ggml_cuda_pq2_j128_cp_commit();
            }
            __syncthreads(); // X tile (block kb+1) and Y buffer `cur ^ 1` complete
        }
    } else {
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
            ggml_cuda_pq2_j128_compute(acc, xw, yw + cur*YBYTES);
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
    }
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_store(
        const float (&acc)[2][4][4], float * out, const int stride_col, const int row_base, const int col_base,
        const int j_max) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int j = col_base + ct*8 + 2*c + l % 2;
                if (j > j_max) {
                    continue;
                }
                const int i = row_base + rt*16 + g + 8*(l / 2);
                out[(int64_t) j*stride_col + i] = acc[rt][ct][l];
            }
        }
    }
}

// grid = old grid * 2 (1D: row half q = blockIdx.x % 2 fastest), block = (32, 8), 2 CTAs per SM.
// PURE: pure tiling (grid == number of tiles), block b is tile b with the full K range.
template <int XR, bool PURE>
__launch_bounds__(256, 2)
static __global__ void mul_mat_q_pq2_j128(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int ncols_dst, const int ntx) {
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

    if constexpr (PURE) {
        const int jt = b % ntx;
        const int it = b / ntx;
        float acc[2][4][4] = {{{0.0f}}};
        ggml_cuda_pq2_j128_segment<XR>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, 0, nkb);
        ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                 ncols_dst - jt*J - 1);
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
        float acc[2][4][4] = {{{0.0f}}};
        ggml_cuda_pq2_j128_segment<XR>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, kb0_start, kb0_stop);
        if (!ends_tile) {
            ggml_cuda_pq2_j128_store(acc, tmp_fixup + (int64_t) b*(J*I), I, row_in_tile, col_in_tile, J - 1);
            return;
        }
        ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                 ncols_dst - jt*J - 1);

        kbc += nkb;
        kbc -= kbc % nkb;
        kb0_start = 0;
        kb0_stop  = min(nkb, kbc_stop - kbc);
    }
}

// ---------------------------------------------------------------------------------------------
// Host side (included by mmq.cuh after mmq_args). Used on GGML_CUDA_CC_DGX_SPARK only, for
// mul_mat_q<GGML_TYPE_PQ2_0, 128, false> without ids/channels/samples.

template <int XR>
static constexpr size_t ggml_cuda_pq2_j128_smem() {
    return (size_t) (2*128*ggml_cuda_pq2_j128_ys + 64*ggml_cuda_pq2_j128_xs)*sizeof(int) + (XR ? (size_t) 256*16 : 0);
}

static bool ggml_cuda_mmq_pq2_j128_supported(const mmq_args & args) {
    return args.type_x == GGML_TYPE_PQ2_0 && args.ids_dst == nullptr && args.expert_bounds == nullptr &&
        args.y_scale == nullptr &&
        args.nchannels_x == 1 && args.nchannels_y == 1 && args.nsamples_x == 1 && args.nsamples_y == 1 &&
        args.nrows_x % 128 == 0 && args.ncols_x % (2*QK_PQ2_0) == 0 &&
        args.ncols_dst >= 1 && args.ncols_max == args.ncols_dst &&
        (uintptr_t) args.y % 16 == 0 &&
        args.nrows_x*args.stride_row_x*(int64_t) sizeof(block_pq2_0) < (int64_t(1) << 31) &&
        (args.ncols_x/QK_PQ2_0 + 1)*args.ncols_y*(int64_t) sizeof(block_q8_1_mmq) < (int64_t(1) << 31);
}

// Same grid rule as launch_mul_mat_q for the PQ2_0 J=128 (stream-k) config on NVIDIA.
static int ggml_cuda_mmq_pq2_j128_grid(const mmq_args & args, const int nsm) {
    const int nty    = (int) ((args.nrows_x + 127) / 128);
    const int ntx    = (int) ((args.ncols_max + 127) / 128);
    const int ntiles = nty*ntx;
    const int nwaves = (ntiles + nsm - 1) / nsm;
    const int eff    = 100*ntiles / (nsm*nwaves);
    return eff >= 90 ? ntiles : nsm;
}

static bool ggml_cuda_mmq_pq2_j128_needs_fixup(const mmq_args & args, const int grid) {
    const int ntiles = (int) ((args.nrows_x + 127) / 128) * (int) ((args.ncols_max + 127) / 128);
    return ntiles % grid != 0;
}

static bool ggml_cuda_mmq_pq2_j128_use(const mmq_args & args, const int nsm) {
    GGML_UNUSED(nsm);
    return ggml_cuda_mmq_pq2_j128_supported(args);
}

template <ggml_type type, int XR, bool PURE>
static void ggml_cuda_mmq_pq2_j128_launch(const mmq_args & args, const int grid, float * tmp_fixup, cudaStream_t stream) {
    constexpr size_t smem = ggml_cuda_pq2_j128_smem<XR>();
    const int nkb = (int) (args.ncols_x / QK_PQ2_0);
    const int ntx = (int) ((args.ncols_max + 127) / 128);
    const auto kernel = mul_mat_q_pq2_j128<XR, PURE>;
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
    const int device = ggml_cuda_get_device();
    if (!attr_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared));
        attr_set[device] = true;
    }
    // Programmatic stream serialization (PDL), as launch_mul_mat_q does on GB10.
    ggml_cuda_kernel_launch(kernel,
        ggml_cuda_kernel_launch_params(dim3(2*grid, 1, 1), dim3(32, 8, 1), smem, stream),
        args.x, args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.stride_row_x, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst, ntx);
    if (!PURE && ggml_cuda_mmq_pq2_j128_needs_fixup(args, grid)) {
        const uint3 blocks_per_ne00_fd = init_fastdiv_values(nkb);
        const uint3 one_fd             = init_fastdiv_values(1);
        const uint3 ntx_fd             = init_fastdiv_values(ntx);
        ggml_cuda_kernel_launch(mul_mat_q_stream_k_fixup<type, 128, false>,
            ggml_cuda_kernel_launch_params(dim3(grid, 128/32, 1), dim3(32, 4, 1), 0, stream),
            nullptr, nullptr, args.dst, tmp_fixup, blocks_per_ne00_fd, (int) args.nrows_x, (int) args.ncols_dst,
            (int) args.nrows_dst, one_fd, (int) args.stride_channel_dst, one_fd, (int) args.stride_sample_dst, ntx_fd);
    }
}

// Template so that only the PQ2_0 MMQ translation unit instantiates the kernels. Pure tiling keeps the
// raw X words in registers; the stream-k partition stages them in shared memory (faster on GB10 for
// each shape, measured).
template <ggml_type type>
static void ggml_cuda_mmq_pq2_j128_run(const mmq_args & args, const int grid, float * tmp_fixup, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_PQ2_0, "PQ2_0 only");
    const int ntiles = (int) ((args.nrows_x + 127) / 128) * (int) ((args.ncols_max + 127) / 128);
    if (grid == ntiles) {
        ggml_cuda_mmq_pq2_j128_launch<type, 0, true>(args, grid, tmp_fixup, stream);
    } else {
        ggml_cuda_mmq_pq2_j128_launch<type, 1, false>(args, grid, tmp_fixup, stream);
    }
}
