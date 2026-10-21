#pragma once

// GB10 (sm_121) J=16 operand code for PTQ1_0 groups of the decode layout copy (mmq-j16-dl.cuh).
//
// For a PQ2_0 weight whose 2-bit codes q are all in {0, 1, 2} (value (q - 1)*d), the decode layout copy
// holds, per row and K step of 128 values, the PTQ1_0 block (block_ptq1_0, ggml-common.h: 24 + 2 trit
// bytes, then the fp16 d) of the same codes and the same d bits, written with the fork's PTQ1_0 trit codec
// (quantize_row_ptq1_0_ref: base-3 digits, the first in the most significant position, byte =
// ceil(q*256/243), stages of 16 and 8 bytes plus qh). It is value-exact: dequantize_row_ptq1_0 of the
// group equals dequantize_row_pq2_0 of the original block for every value. A tensor with any code 3
// (+2d, not a trit) keeps the 34-byte PQ2_0 groups.
//
// The operands reproduce the int32 dot products of the PQ2_0 kernels exactly. mma m16n8k32 with A = the
// 16 activation columns (s8) and B = 8 weight rows. Inside a 32-value sub-block the fragment ints are in
// natural order (logical int c = physical int c, logical 4 + c = physical 4 + c) for both operands, so
// every mma sums the same 32 products as mul_mat_q<GGML_TYPE_PQ2_0, 16>. B holds the trits t = q in
// {0, 1, 2} as u8 and the accumulator starts at MAGIC - sum(a), from an s8 mma of the same A with B = -1.
// Integer arithmetic is exact, so the result is MAGIC + sum(a*(t - 1)): the accumulator bits of the
// PQ2_0 kernels. The epilogue is the PQ2_0 one: for every sub-block in increasing K,
// sum += cf*dA*dB with cf = (float) idot exactly, dA = (float) d (same fp16 bits), dB = the D4 scale.
//
// Trit decode of a word of trit bytes b: the fork's decoder is t_n(b) = (((b*3^n) mod 256)*3) >> 8. With
// H_k(b) = floor(b*3^k / 256), t_n = H_{n+1} - 3*H_n. b*3^k fits a 16-bit lane for k <= 5, so two
// bytes per register give H_k of four bytes with two multiplies and one byte permute, and t_n of four
// bytes with one IMAD (3*H_n <= 240 and the difference is in {0, 1, 2}: no carry between bytes).

#include "mmq-pq2-j16.cuh"

static_assert(sizeof(block_ptq1_0) == 28, "PTQ1_0 block is 7 words");

static __device__ __forceinline__ void ggml_cuda_ptq1_j16_mma_u8(int (&c)[4], const int a0, const int a1, const int a2,
                                                                const int a3, const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// H_k of the 4 bytes of a word (byte order kept): lo = bytes 0, 2 as 16-bit lanes, hi = bytes 1, 3.
static __device__ __forceinline__ uint32_t ggml_cuda_ptq1_h(const uint32_t lo, const uint32_t hi, const uint32_t m) {
    return __byte_perm(lo*m, hi*m, 0x7351);
}

// Operands of one K step for lane (g, c): activation ints c and 4 + c of every sub-block of columns g and
// g + 8, their D4 scales, and the words c, 4 + (c & 1) and 6 (qh0, qh1, d) of the PTQ1_0 group of row g.
struct ggml_cuda_ptq1_j16_stage {
    int      ya[4][2][2]; // [sub-block][column g + 8h][int c, int 4 + c]
    float4   yd[2];
    uint32_t wa, wb, w6;
    uint32_t scale_pair; // initialized only by PAIR_TAIL ring loads
};

template <bool PAIR_TAIL = false>
static __device__ __forceinline__ void ggml_cuda_ptq1_j16_load(
        ggml_cuda_ptq1_j16_stage & st, const char * __restrict__ xr, const char * yb, const int c) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const char * p = yb + h*8*sizeof(block_q8_1_mmq) + 4*sizeof(float) + 32*s + 4*c;
            st.ya[s][h][0] = *(const int *) (p);
            st.ya[s][h][1] = *(const int *) (p + 16);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        st.yd[h] = *(const float4 *) (yb + h*8*sizeof(block_q8_1_mmq));
    }
    const uint32_t * xw = (const uint32_t *) xr;
    st.wa = xw[c];
    st.wb = xw[4 + (c & 1)];
    if constexpr (PAIR_TAIL) {
        const uint32_t * tile = xw - 7*(int(threadIdx.x)/4);
        st.w6 = tile[7*((int(threadIdx.x)/4)/2) + 6] >> (16*((int(threadIdx.x)/4)&1));
        st.scale_pair = tile[7*(4+c) + 6];
    } else {
        st.w6 = xw[6];
    }
}

// B fragments (b[s][0] = logical int c, b[s][1] = logical int 4 + c of sub-block s) of weight row g, as trits.
// Value v of the block: v < 80: qs byte v % 16, trit v / 16; 80 <= v < 120: qs byte 16 + (v - 80) % 8,
// trit (v - 80) / 8; v >= 120: qh byte (v - 120) % 2, trit (v - 120) / 2.
//   word c:           t0..t4 = values 16n + 4c.. -> (s0: c, 4+c), (s1: c, 4+c), (s2: c)
//   word 4 + (c & 1): trit n = values 80 + 8n + 4(c & 1).. -> s2 int 4+c: n = c/2, s3 int c: n = 2 + c/2,
//                     s3 int 4+c (c < 2): n = 4
//   qh (c >= 2):      s3 int 4+c = values 120 + 4(c - 2).. = (qh0, qh1) trits a, a + 1 with a = 2(c & 1)
static __device__ __forceinline__ void ggml_cuda_ptq1_j16_decode(uint32_t (&b)[4][2], const ggml_cuda_ptq1_j16_stage & st, const int c) {
    {
        const uint32_t lo = __byte_perm(st.wa, 0, 0x4240);
        const uint32_t hi = __byte_perm(st.wa, 0, 0x4341);
        const uint32_t H1 = ggml_cuda_ptq1_h(lo, hi, 3u);
        const uint32_t H2 = ggml_cuda_ptq1_h(lo, hi, 9u);
        const uint32_t H3 = ggml_cuda_ptq1_h(lo, hi, 27u);
        const uint32_t H4 = ggml_cuda_ptq1_h(lo, hi, 81u);
        const uint32_t H5 = ggml_cuda_ptq1_h(lo, hi, 243u);
        b[0][0] = H1;
        b[0][1] = H2 - 3u*H1;
        b[1][0] = H3 - 3u*H2;
        b[1][1] = H4 - 3u*H3;
        b[2][0] = H5 - 3u*H4;
    }
    uint32_t qh;
    {
        const uint32_t v  = __byte_perm(st.w6, 0, 0x4140); // lanes (qh0, qh1)
        const uint32_t m  = (c & 1) ? 9u : 1u;              // 3^a, a = 2(c & 1)
        const uint32_t L0 = v*m;
        const uint32_t L1 = L0*3u;
        const uint32_t L2 = L0*9u;
        const uint32_t Q0 = __byte_perm(L0, L1, 0x7531);    // (H_a(qh0), H_a(qh1), H_a+1(qh0), H_a+1(qh1))
        const uint32_t Q1 = __byte_perm(L1, L2, 0x7531);
        qh = Q1 - 3u*Q0;
    }
    {
        const uint32_t lo = __byte_perm(st.wb, 0, 0x4240);
        const uint32_t hi = __byte_perm(st.wb, 0, 0x4341);
        const uint32_t H1 = ggml_cuda_ptq1_h(lo, hi, 3u);
        const uint32_t H2 = ggml_cuda_ptq1_h(lo, hi, 9u);
        const uint32_t H3 = ggml_cuda_ptq1_h(lo, hi, 27u);
        const uint32_t H4 = ggml_cuda_ptq1_h(lo, hi, 81u);
        const uint32_t H5 = ggml_cuda_ptq1_h(lo, hi, 243u);
        const bool lo_c = c < 2;
        b[2][1] = lo_c ? H1 : H2 - 3u*H1;
        b[3][0] = lo_c ? H3 - 3u*H2 : H4 - 3u*H3;
        b[3][1] = lo_c ? H5 - 3u*H4 : qh;
    }
}

// SGN: the trits {0, 1, 2} become signed bytes {-1, 0, 1} (per byte (t + 0x7F) ^ 0x80, no carry since t <= 2) and
// one s8.s8 mma accumulates sum(a*(t - 1)) onto MAGIC, the integer the base forms as MAGIC - sum(a) + sum(a*t)
// with two mma; int32 accumulation is exact, so C and every float below are the same bits.
template <bool SGN = false, bool PAIR_TAIL = false>
static __device__ __forceinline__ void ggml_cuda_ptq1_j16_compute(float (&sum)[1][4], const ggml_cuda_ptq1_j16_stage & st, const int c) {
    // dA of the C rows 2c and 2c+1 of the 8-row tile: the scale of row r was loaded by lanes 4r..4r+3.
    int d_lo, d_hi;
    if constexpr (PAIR_TAIL) {
        d_lo = int(st.scale_pair & 0xffffu);
        d_hi = int(st.scale_pair >> 16);
    } else {
        const int xd = (int) (st.w6 >> 16);
        d_lo = __shfl_sync(0xFFFFFFFF, xd, 8*c + 0);
        d_hi = __shfl_sync(0xFFFFFFFF, xd, 8*c + 4);
    }
    const float dA[2] = {
        __half2float(__ushort_as_half((unsigned short) d_lo)),
        __half2float(__ushort_as_half((unsigned short) d_hi)),
    };
    const float dB[2][4] = {
        {st.yd[0].x, st.yd[0].y, st.yd[0].z, st.yd[0].w},
        {st.yd[1].x, st.yd[1].y, st.yd[1].z, st.yd[1].w},
    };
    uint32_t b[4][2];
    ggml_cuda_ptq1_j16_decode(b, st, c);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int C[4] = {GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC, GGML_CUDA_PQ2_J16_MAGIC};
        if constexpr (SGN) {
            const int b0 = (int) ((b[s][0] + 0x7F7F7F7Fu) ^ 0x80808080u);
            const int b1 = (int) ((b[s][1] + 0x7F7F7F7Fu) ^ 0x80808080u);
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], b0, b1);    // MAGIC + sum(a*(t-1))
        } else {
            ggml_cuda_pq2_j16_mma(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], -1, -1);       // MAGIC - sum(a)
            ggml_cuda_ptq1_j16_mma_u8(C, st.ya[s][0][0], st.ya[s][1][0], st.ya[s][0][1], st.ya[s][1][1], (int) b[s][0], (int) b[s][1]);
        }
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float cf = __int_as_float(C[l]) - 12582912.0f; // == (float) idot, exactly
            sum[0][l] += cf*dA[l % 2]*dB[l / 2][s];
        }
    }
}

// PQ2_0 block (34 bytes, 2-byte aligned) -> PTQ1_0 block (7 words) with the fork's codec and the same d bits.
// Returns false if a code is 3 or the result does not decode back (with the fork's decoder) to the same codes.
static __device__ __forceinline__ bool ggml_cuda_ptq1_encode_block(const uint16_t * __restrict__ s, uint32_t (&o)[7]) {
    const uint16_t d = s[0];
    uint8_t code[QK_PQ2_0];
    bool ok = true;
#pragma unroll
    for (int k = 0; k < QK_PQ2_0/8; ++k) {
        const uint32_t q = s[1 + k];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            code[8*k + j] = (uint8_t) ((q >> (2*j)) & 3);
            ok = ok && code[8*k + j] != 3;
        }
    }
    uint8_t out[sizeof(block_ptq1_0)];
    // qs: stage 16 (bytes 0..15: values m + 16n), stage 8 (bytes 16..23: values 80 + m + 8n), 5 trits per byte
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        uint32_t q = 0;
#pragma unroll
        for (int n = 0; n < 5; ++n) { q = q*3 + code[m + 16*n]; }
        out[m] = (uint8_t) ((q*256 + 242)/243);
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        uint32_t q = 0;
#pragma unroll
        for (int n = 0; n < 5; ++n) { q = q*3 + code[80 + m + 8*n]; }
        out[16 + m] = (uint8_t) ((q*256 + 242)/243);
    }
    // qh: values 120 + h + 2m, 4 trits, then q *= 3 (the first trit in the most significant position)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t q = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) { q = q*3 + code[120 + h + 2*m]; }
        q *= 3;
        out[24 + h] = (uint8_t) ((q*256 + 242)/243);
    }
    out[26] = (uint8_t) (d & 0xFF);
    out[27] = (uint8_t) (d >> 8);
    // round trip with the fork's decoder (dequantize_row_ptq1_0)
#pragma unroll
    for (int v = 0; v < QK_PQ2_0; ++v) {
        int byte, n;
        if (v < 80)       { byte = v % 16;             n = v / 16; }
        else if (v < 120) { byte = 16 + (v - 80) % 8;  n = (v - 80) / 8; }
        else              { byte = 24 + (v - 120) % 2; n = (v - 120) / 2; }
        uint32_t p3 = 1;
        for (int e = 0; e < n; ++e) { p3 *= 3; }
        const uint8_t qq = (uint8_t) (out[byte]*p3);
        ok = ok && (uint8_t) (((uint32_t) qq*3) >> 8) == code[v];
    }
#pragma unroll
    for (int w = 0; w < 7; ++w) {
        o[w] = (uint32_t) out[4*w] | ((uint32_t) out[4*w + 1] << 8) | ((uint32_t) out[4*w + 2] << 16) | ((uint32_t) out[4*w + 3] << 24);
    }
    return ok;
}
