#include "quantize.cuh"
#include "fwht.cuh"
#include "mmvq-ptq1_0.cuh"
#include "unary.cuh"
#include <cstdint>

#if defined(BLACKWELL_MMA_AVAILABLE)
// this maps to 256-bit loads in PTX on supported devices,
// and otherwise falls back to 2 128-bit loads
struct __builtin_align__(32) float8 {
    float x; float y; float z; float w;
    float p; float q; float r; float s;
};

#if CUDART_VERSION >= 12080
static __device__ __forceinline__ float nvfp4_native_scale_error(
        const float vals[QK_NVFP4_SUB], const float inv_col_scale, const float inv_scale, const float scale) {
    const float scale_dequant = 2.0f * scale;
    float err = 0.0f;

#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k += 4) {
        const float v0 = vals[k + 0] * inv_col_scale;
        const float v1 = vals[k + 1] * inv_col_scale;
        const float v2 = vals[k + 2] * inv_col_scale;
        const float v3 = vals[k + 3] * inv_col_scale;

        const __nv_fp4x4_e2m1 q(make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);

        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
        const __half2 h2_lo = static_cast<__half2>(hraw2_lo);
        const __half2 h2_hi = static_cast<__half2>(hraw2_hi);
        const float2 dq_lo = __half22float2(h2_lo);
        const float2 dq_hi = __half22float2(h2_hi);

        const float err0 = fabsf(v0) - fabsf(dq_lo.x) * scale_dequant;
        const float err1 = fabsf(v1) - fabsf(dq_lo.y) * scale_dequant;
        const float err2 = fabsf(v2) - fabsf(dq_hi.x) * scale_dequant;
        const float err3 = fabsf(v3) - fabsf(dq_hi.y) * scale_dequant;

        err = fmaf(err0, err0, err);
        err = fmaf(err1, err1, err);
        err = fmaf(err2, err2, err);
        err = fmaf(err3, err3, err);
    }

    return err;
}
#endif // CUDART_VERSION >= 12080
#endif // defined(BLACKWELL_MMA_AVAILABLE)

// layout (ggml_cuda_q8_1_layout):
//   AOS      plain block_q8_1, float input sum in ds.y
//   SOA_ISUM warp-transposed layout (ggml_cuda_ptq1_q8_word) with the exact integer sum of the
//            quantized values bit-cast into ds.y; the 1-column PTQ1_0 vec-dot folds the digit bias
//            {0,1,2} -> {-1,0,+1} into one subtraction per 32-block instead of a SIMD byte
//            subtract per 4 weights (bit-identical to the biased path)
//   PT       planar-transposed layout consumed by the 2-8 column PTQ1_0 mat-vec path
//            (see mmvq-ptq1_0.cuh); same quantization, same bytes per row as block_q8_1, and the
//            same exact integer sum in ds.y as SOA_ISUM so that kernel can also use unbiased trits
template <ggml_cuda_q8_1_layout layout>
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    if constexpr (layout == GGML_CUDA_Q8_1_PT) {
        const int64_t row_cont = (i3*ne2.z + i2) * ne1 + i1;
        char * ycol = (char *) vy + row_cont * (ne0 * 9 / 8); // same row stride as block_q8_1
        const int64_t nblk = ne0 / QK_PTQ1_0;
        const int64_t kb   = i0 / QK_PTQ1_0;
        const int     e    = i0 % QK_PTQ1_0;
        ycol[((e / 16)*nblk + kb) * 16 + (e % 16)] = q;

        int isum = q;
        isum = warp_reduce_sum<QK8_1>(isum);

        if (iqs > 0) {
            return;
        }

        // |isum| <= 32*127 fits int16; keep the raw bits in the half slot (read back with __half_as_short).
        half2 * ds = (half2 *) (ycol + 8*nblk*16) + kb*4 + e / QK8_1;
        *ds = make_half2(__float2half(d), __short_as_half((short) isum));
        return;
    }
    if constexpr (layout == GGML_CUDA_Q8_1_SOA_ISUM) {
        // Warp-transposed layout (see ggml_cuda_ptq1_q8_word); ne0 is padded to GGML_CUDA_PTQ1_K_PAD.
        int isum = q;
        isum = warp_reduce_sum<QK8_1>(isum);
        int32_t *     yw       = (int32_t *) vy;
        const int64_t col      = i_cont / ne0;
        const int64_t col_base = col * (ne0 / QK8_1) * (int64_t) (sizeof(block_q8_1) / 4);
        const int     ib_col   = (int) (i0 / QK8_1);
        ((int8_t *) (yw + col_base + ggml_cuda_ptq1_q8_word(ib_col, iqs / 4)))[iqs % 4] = q;
        if (iqs > 0) {
            return;
        }
        // |isum| <= 32*127 fits int16; keep the raw bits in the half slot.
        const half2 ds = make_half2(__float2half(d), __short_as_half((short) isum));
        yw[col_base + ggml_cuda_ptq1_q8_word(ib_col, 8)] = *reinterpret_cast<const int32_t *>(&ds);
        return;
    }

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

// Hadamard transform folded into the activation quantizer (see ggml_cuda_fwht_skip): one block per
// (transform width N, row), the same butterfly as fwht_cuda_block, then the same per-32 quantization
// and stores as quantize_q8_1<layout> above. Element i*NT + tid of the transform sits in reg[i], so
// each 32-block kb = i*(NT/32) + tid/32 lies in one warp and is reduced with 32-wide warp reductions.
// Rows of x are contiguous (ne00 elements); row index = ((i3*ne2 + i2)*ne1 + i1) like a contiguous src1.
template <int N, int NT, ggml_cuda_q8_1_layout layout, typename T, bool has_signs>
__launch_bounds__(NT, 1)
static __global__ void fwht_quantize_q8_1(
        const T * x_ptr, const float * signs, void * vy_ptr,
        const int64_t ne00, const int64_t ne0, const uint32_t ne1, const uint3 ne2, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0 && NT % QK8_1 == 0, "bad fused FWHT shape");
    static_assert(warp_size % QK8_1 == 0, "a 32-block must not span two warps");

    __shared__ float s[N];

    ggml_cuda_pdl_lc();

    const int64_t blk = blockIdx.x;                 // transform block within the row
    const int64_t i3  = fastdiv(blockIdx.z, ne2);
    const int64_t i2  = blockIdx.z - i3*ne2.z;
    const int64_t i1  = blockIdx.y;
    const int64_t row = (i3*ne2.z + i2) * ne1 + i1;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    const int64_t base = blk * N;                   // first element of this transform in the row

    float reg[NE];
    ggml_cuda_pdl_sync();
    if (base < ne00) {
        const T *     src       = x_ptr + row * ne00 + base;
        const float * signs_row = has_signs ? signs + base : nullptr; // (r % n_blk) * N with r = row*n_blk + blk
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = (float) src[i * NT + tid] * scale;
            if (has_signs) {
                reg[i] *= signs_row[i * NT + tid];
            }
        }
        ggml_cuda_fwht_block_butterfly<N, NT>(reg, s, tid, lane);
    } else {
        // padding beyond the real row: zero blocks, same as the plain quantizer's x = 0 path
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            reg[i] = 0.0f;
        }
    }

    void * GGML_CUDA_RESTRICT vy = vy_ptr;

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const float xi = reg[i];
        const int64_t i0 = base + i * NT + tid;   // element within the row (i0 % 32 == tid % 32)

        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);

        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

        const int iqs = lane % QK8_1;

        if constexpr (layout == GGML_CUDA_Q8_1_PT) {
            char * ycol = (char *) vy + row * (ne0 * 9 / 8);
            const int64_t nblk = ne0 / QK_PTQ1_0;
            const int64_t kb   = i0 / QK_PTQ1_0;
            const int     e    = i0 % QK_PTQ1_0;
            ycol[((e / 16)*nblk + kb) * 16 + (e % 16)] = q;

            int isum = q;
            isum = warp_reduce_sum<QK8_1>(isum);
            if (iqs == 0) {
                half2 * ds = (half2 *) (ycol + 8*nblk*16) + kb*4 + e / QK8_1;
                *ds = make_half2(__float2half(d), __short_as_half((short) isum));
            }
        } else if constexpr (layout == GGML_CUDA_Q8_1_SOA_ISUM) {
            int isum = q;
            isum = warp_reduce_sum<QK8_1>(isum);
            int32_t *     yw       = (int32_t *) vy;
            const int64_t col_base = row * (ne0 / QK8_1) * (int64_t) (sizeof(block_q8_1) / 4);
            const int     ib_col   = (int) (i0 / QK8_1);
            ((int8_t *) (yw + col_base + ggml_cuda_ptq1_q8_word(ib_col, iqs / 4)))[iqs % 4] = q;
            if (iqs == 0) {
                const half2 ds = make_half2(__float2half(d), __short_as_half((short) isum));
                yw[col_base + ggml_cuda_ptq1_q8_word(ib_col, 8)] = *reinterpret_cast<const int32_t *>(&ds);
            }
        } else {
            block_q8_1 * y = (block_q8_1 *) vy;
            const int64_t ib = (row * ne0 + i0) / QK8_1;
            y[ib].qs[iqs] = q;
            if (iqs == 0) {
                y[ib].ds = make_half2(d, sum);
            }
        }
    }
}

template <int N, typename T>
static void fwht_quantize_launch_layout(
        const T * x, const float * signs, void * vy, const ggml_cuda_q8_1_layout layout,
        const int64_t ne00, const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float scale, cudaStream_t stream) {
    constexpr int NT = N < 256 ? N : 256;
    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);
    const dim3 num_blocks((unsigned) (ne0 / N), (unsigned) ne1, (unsigned) (ne2*ne3));
    const dim3 block_size(NT, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
#define FWHT_Q_LAUNCH(LAYOUT) \
    if (signs) { \
        ggml_cuda_kernel_launch(fwht_quantize_q8_1<N, NT, LAYOUT, T, true>,  lp, x, signs, vy, ne00, ne0, (uint32_t) ne1, ne2_fastdiv, scale); \
    } else { \
        ggml_cuda_kernel_launch(fwht_quantize_q8_1<N, NT, LAYOUT, T, false>, lp, x, signs, vy, ne00, ne0, (uint32_t) ne1, ne2_fastdiv, scale); \
    }
    switch (layout) {
        case GGML_CUDA_Q8_1_PT:       FWHT_Q_LAUNCH(GGML_CUDA_Q8_1_PT);       break;
        case GGML_CUDA_Q8_1_SOA_ISUM: FWHT_Q_LAUNCH(GGML_CUDA_Q8_1_SOA_ISUM); break;
        default:                      FWHT_Q_LAUNCH(GGML_CUDA_Q8_1_AOS);      break;
    }
#undef FWHT_Q_LAUNCH
}

template <typename T>
static void fwht_quantize_launch(
        const T * x, const float * signs, void * vy, const ggml_cuda_q8_1_layout layout, const int n,
        const int64_t ne00, const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float scale, cudaStream_t stream) {
    switch (n) {
        case   64: fwht_quantize_launch_layout<  64, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        case  128: fwht_quantize_launch_layout< 128, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        case  256: fwht_quantize_launch_layout< 256, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        case  512: fwht_quantize_launch_layout< 512, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        case 1024: fwht_quantize_launch_layout<1024, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        case 2048: fwht_quantize_launch_layout<2048, T>(x, signs, vy, layout, ne00, ne0, ne1, ne2, ne3, scale, stream); break;
        default: GGML_ABORT("fused FWHT quantizer: unsupported transform width %d", n);
    }
}

bool ggml_cuda_fwht_quantize_supported(const int n, const int64_t ne00) {
    switch (n) {
        case 64: case 128: case 256: case 512: case 1024: case 2048:
            break;
        default:
            return false;
    }
    return ne00 % n == 0 && ne00 % QK8_1 == 0;
}

void fwht_quantize_row_q8_1_cuda(
        const void * x, const ggml_type x_type, const float * signs, const int n, void * vy,
        const ggml_cuda_q8_1_layout layout, const int64_t ne00, const int64_t ne0, const int64_t ncols, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_fwht_quantize_supported(n, ne00));
    GGML_ASSERT(ne0 % n == 0 && ne0 >= ne00);
    if (layout == GGML_CUDA_Q8_1_PT) {
        GGML_ASSERT(ne0 % QK_PTQ1_0 == 0);
    }
    const float scale = 1.0f / sqrtf((float) n);
    if (x_type == GGML_TYPE_F16) {
        fwht_quantize_launch<half>((const half *) x, signs, vy, layout, n, ne00, ne0, ncols, 1, 1, scale, stream);
    } else {
        GGML_ASSERT(x_type == GGML_TYPE_F32);
        fwht_quantize_launch<float>((const float *) x, signs, vy, layout, n, ne00, ne0, ncols, 1, 1, scale, stream);
    }
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy, float * __restrict__ scale,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int n_expert_used) {
#if defined(BLACKWELL_MMA_AVAILABLE)

    const int64_t blocks_per_col = (ne0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.y % ne2;
        const int64_t i3  = blockIdx.y / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    }
    const float * __restrict__  x_row = x + base_idx;

    float amax = 0.0f;
    if constexpr (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const float * x_base = x_row + i0;
            const float8 v = reinterpret_cast<const float8 *>(x_base)[0];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
            amax = fmaxf(amax, fabsf(v.p));
            amax = fmaxf(amax, fabsf(v.q));
            amax = fmaxf(amax, fabsf(v.r));
            amax = fmaxf(amax, fabsf(v.s));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
    }

    amax = warp_reduce_max<WARP_SIZE>(amax);

    __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();

    if (warp == 0) {
        amax = threadIdx.x < int(CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE) ? warp_amax[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_amax[0] = amax / (6.0f * 448.0f);
            if constexpr (scatter) {
#pragma unroll
                for (int slot = 0; slot < n_expert_used; ++slot) {
                    const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                    scale[i] = warp_amax[0];
                }
            } else {
                scale[blockIdx.y * ne1 + blockIdx.x] = warp_amax[0];
            }
        }
    }
    __syncthreads();

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_FP4_MMQ;
        const int sub = (i0_base % QK_FP4_MMQ) / QK_NVFP4_SUB;

        const float row_scale = warp_amax[0];
        const float inv_col_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;

        float vals[QK_NVFP4_SUB];
        if constexpr (use_aligned_float8) {
            const float * x_base = x_row + i0_base;
            const float8 v0 = i0_base +  7 < ne00 ? reinterpret_cast<const float8 *>(x_base)[0]     : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const float8 v1 = i0_base + 15 < ne00 ? reinterpret_cast<const float8 *>(x_base + 8)[0] : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z; vals[3] = v0.w;
            vals[4] = v0.p; vals[5] = v0.q; vals[6] = v0.r; vals[7] = v0.s;
            vals[8] = v1.x; vals[9] = v1.y; vals[10] = v1.z; vals[11] = v1.w;
            vals[12] = v1.p; vals[13] = v1.q; vals[14] = v1.r; vals[15] = v1.s;
        } else {
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const int64_t i00 = i0_base + k;
                vals[k] = i00 < ne00 ? x_row[i00] : 0.0f;
            }
        }

        uint32_t q0 = 0;
        uint32_t q1 = 0;

        float amax_sub = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
        }

        static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
        const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

        uint8_t fp8_code = (uint8_t) first_fp8_code;
        float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
        float inv_scale_err = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#if CUDART_VERSION >= 12080
        float best_err = nvfp4_native_scale_error(vals, inv_col_scale, inv_scale_err, subblock_scale);
#else
        float best_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals[k] * inv_col_scale;
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, inv_scale_err);
            const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * subblock_scale;
            best_err = fmaf(err_diff, err_diff, best_err);
        }
#endif // CUDART_VERSION >= 12080

#pragma unroll
        for (int i = 1; i < 5; ++i) {
            const int test_code = first_fp8_code + test_offsets[i];
            if (test_code < 0 || test_code > 0x7e) {
                continue;
            }

            const float test_scale = ggml_cuda_ue4m3_to_fp32((uint8_t) test_code);
            const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
#if CUDART_VERSION >= 12080
            const float cur_err = nvfp4_native_scale_error(vals, inv_col_scale, test_inv_scale, test_scale);
#else
            float cur_err = 0.0f;
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const float v = vals[k] * inv_col_scale;
                const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
                const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * test_scale;
                cur_err = fmaf(err_diff, err_diff, cur_err);
            }
#endif // CUDART_VERSION >= 12080

            if (cur_err < best_err) {
                best_err = cur_err;
                fp8_code = (uint8_t) test_code;
                subblock_scale = test_scale;
            }
        }
#if CUDART_VERSION >= 12080
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        const float s = inv_col_scale * inv_scale;

        __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
        __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
        __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
        __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

        const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
        const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
        const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
        const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

        q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q0_hi_c.x)) << 16) | (uint32_t(uint8_t(q0_hi_c.y)) << 24);
        q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q1_hi_c.x)) << 16) | (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 0] * inv_col_scale, inv_scale)) << (8 * k);
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 8] * inv_col_scale, inv_scale)) << (8 * k + 4);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 4] * inv_col_scale, inv_scale)) << (8 * k);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale)) << (8 * k + 4);
        }
#endif // CUDART_VERSION >= 12080

        if constexpr (scatter) {
#pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                block_fp4_mmq * yb = y + (k_block * ne1 + i);
                uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
                yqs[2 * sub + 0] = q0;
                yqs[2 * sub + 1] = q1;
                reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
            }
        } else {
            block_fp4_mmq * yb = y + (blockIdx.y * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x);
            uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
            yqs[2 * sub + 0] = q0;
            yqs[2 * sub + 1] = q1;
            reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used);
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)

}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter>
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2,
                                          const int     n_expert_used) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t block_fp4_mmq_size = QK_FP4_MMQ;
    const int64_t k_block            = warp_start_offset / block_fp4_mmq_size;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;

    ggml_cuda_pdl_sync();
    int64_t base_pos;
    if constexpr (scatter) {
        base_pos = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_pos = i3 * s03 + i2 * s02 + i01 * s01;
    }

    uint8_t scales[2];
    char2   packed[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
        packed[b] = *(char2 *) &fp4_packed;
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        char2 q;
        q.x = (q_hi_0 << 4) | q_lo_0;
        q.y = (q_hi_1 << 4) | q_lo_1;
        packed[b] = q;
#endif // CUDART_VERSION >= 12080
    }

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    if constexpr (scatter) {
#pragma unroll
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            block_fp4_mmq * yb = y + (k_block * ne1 + i);
            char2 * yqs2 = (char2 *) yb->qs;
            if (lane_in_group == 0) {
                yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
                yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
            }
            if (lane_id_32 == 0) {
                yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
            }
        }
    } else {
        const int64_t ib0 = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
        block_fp4_mmq * yb = y + (ib0 + k_block * ne1 + blockIdx.x);
        char2 * yqs2 = (char2 *) yb->qs;
        if (lane_in_group == 0) {
            yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
            yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
        }
        if (lane_id_32 == 0) {
            yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
        }
    }
    GGML_UNUSED(n_expert_used);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool swiglu = false, bool rms_scale = false, bool row_concat = false>
static __global__ void quantize_mmq_q8_1(
        const float * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used,
        const float * gate = nullptr,
        const float * norm_weight = nullptr,
        const float * norm_scale = nullptr) {
    // quantize_mmq_q8_1_cuda launches this kernel with programmatic stream serialization on GB10; nothing
    // is read before ggml_cuda_pdl_sync() below, and the pointer arguments are not __restrict__ so no
    // access moves above it.
    ggml_cuda_pdl_lc();

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    if constexpr (row_concat) {
        static_assert(!scatter && !swiglu && !rms_scale && ds_layout == MMQ_Q8_1_DS_LAYOUT_D4,
                      "two-source concat keeps the plain Q8_0 activation path");
        // Every float4 belongs to one token row. Source selection changes only
        // the address; the reduction, rounding and k-major output layout below
        // remain the original quantizer's, including N=32 row interleaving.
        // In this mutually exclusive mode the otherwise unused gate and
        // n_expert_used arguments hold the second slab and first row count.
        if (blockIdx.x >= (unsigned) n_expert_used) {
            x = gate;
            base_idx -= (int64_t) n_expert_used*s01;
        }
    }
    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (swiglu) {
        const float4 gi = i0 < ne00 ? ((const float4 *) gate)[(base_idx + i00)/4] :
                                      make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        xi.x *= ggml_cuda_op_silu_single(gi.x);
        xi.y *= ggml_cuda_op_silu_single(gi.y);
        xi.z *= ggml_cuda_op_silu_single(gi.z);
        xi.w *= ggml_cuda_op_silu_single(gi.w);
    }
    if constexpr (rms_scale) {
        const int64_t row = ((int64_t) blockIdx.z)*ne1 + blockIdx.x;
        const float scale = norm_scale[row];
        const float4 wi = i0 < ne00 ? ((const float4 *) norm_weight)[i00/4] : make_float4(0, 0, 0, 0);
        // Preserve the operation order of the unfused RMS_NORM -> MUL graph:
        // (scale * x) * weight.  Reassociating this as x * (scale * weight)
        // can change the last bit and, in turn, a greedy decoding trajectory.
        xi.x = (scale*xi.x)*wi.x;
        xi.y = (scale*xi.y)*wi.y;
        xi.z = (scale*xi.z)*wi.z;
        xi.w = (scale*xi.w)*wi.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

void quantize_mmq_q8_1_concat_cuda(
        const float * first, const float * second, void * out,
        int64_t width, int first_rows, int second_rows, cudaStream_t stream) {
    GGML_ASSERT(first && second && out && width > 0 && width % 512 == 0 && first_rows > 0 && second_rows > 0);
    const int rows = first_rows + second_rows;
    const dim3 grid(rows, width/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ), 1);
    const dim3 block(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, stream);
    ggml_cuda_kernel_launch(quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, false, false, true>, lp,
        first, (const int32_t *) nullptr, out, width, width, width*rows, width*rows,
        width, rows, 1, first_rows, second, (const float *) nullptr, (const float *) nullptr);
}

void quantize_mmq_q8_1_swiglu_cuda(
        const float * x, const float * gate, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, gate);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, gate);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, gate);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void quantize_mmq_q8_1_rms_cuda(
        const float * x, const float * weight, const float * scale, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, nullptr, weight, scale);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, nullptr, weight, scale);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0, nullptr, weight, scale);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_cuda_q8_1_layout layout,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
    switch (layout) {
        case GGML_CUDA_Q8_1_PT:
            GGML_ASSERT(ne0 % QK_PTQ1_0 == 0);
            ggml_cuda_kernel_launch(quantize_q8_1<GGML_CUDA_Q8_1_PT>, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
            break;
        case GGML_CUDA_Q8_1_SOA_ISUM:
            GGML_ASSERT(ne0 % GGML_CUDA_PTQ1_K_PAD == 0);
            ggml_cuda_kernel_launch(quantize_q8_1<GGML_CUDA_Q8_1_SOA_ISUM>, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
            break;
        default:
            ggml_cuda_kernel_launch(quantize_q8_1<GGML_CUDA_Q8_1_AOS>, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
            break;
    }
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            mmq_launch_kernel(cc, quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>, num_blocks, block_size, 0, stream,
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0, nullptr, nullptr, nullptr);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            mmq_launch_kernel(cc, quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false>, num_blocks, block_size, 0, stream,
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0, nullptr, nullptr, nullptr);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            mmq_launch_kernel(cc, quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false>, num_blocks, block_size, 0, stream,
                x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0, nullptr, nullptr, nullptr);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// scatter=true reuses the quant kernel: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_q8_1_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// scatter=true reuses the quant kernels: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_fp4_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, float * scale, const ggml_type type_src0, const bool use_aligned_float8,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne0 > 0);
    if (type_src0 == GGML_TYPE_NVFP4) {
        GGML_ASSERT(scale);
        GGML_ASSERT(ne00 % QK_NVFP4 == 0);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks(n_tokens, 1, 1);
        if (use_aligned_float8) {
            quantize_mmq_nvfp4<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, scale, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/nrows_dst, /*ne2=*/1, n_expert_used);
        } else {
            quantize_mmq_nvfp4<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, scale, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/nrows_dst, /*ne2=*/1, n_expert_used);
        }
    } else {
        GGML_ASSERT(type_src0 == GGML_TYPE_MXFP4);
        constexpr int nwarps = 8;
        constexpr int vals_per_block = nwarps * 2 * QK_MXFP4;
        const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
        const dim3 block_size(WARP_SIZE, nwarps, 1);
        const dim3 num_blocks(n_tokens, block_num_y, 1);
        quantize_mmq_mxfp4<true><<<num_blocks, block_size, 0, stream>>>(
            x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
    }
}

void quantize_mmq_fp4_cuda(
        const float * x, const int32_t * ids, void * vy, float * scale, const ggml_type type_src0, const bool use_aligned_float8,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_MXFP4 || type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);

    if (type_src0 == GGML_TYPE_NVFP4) {
        GGML_ASSERT(scale);
        GGML_ASSERT(ne00 % QK_NVFP4 == 0);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks(ne1, ne2 * ne3, 1);
        if (use_aligned_float8) {
            quantize_mmq_nvfp4<false, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
        } else {
            quantize_mmq_nvfp4<false, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
        }
    } else {
        GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

        constexpr int nwarps = 8;
        constexpr int vals_per_warp  = 2 * QK_MXFP4;
        constexpr int vals_per_block = nwarps * vals_per_warp;

        const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
        const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
        const dim3    block_size(WARP_SIZE, nwarps, 1);

        quantize_mmq_mxfp4<false><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
    }
}

// MMQ D4 quantization of transformed rows held as v[i] = element base + i*NT + tid of row `row`.
template <int N, int NT>
static __device__ __forceinline__ void fwht_quantize_mmq_d4_store(const float * v, block_q8_1_mmq * y,
        const int64_t base, const int64_t row, const int64_t rows, const int tid, const int lane) {
    static_assert(NT % QK8_1_MMQ == 0, "D4 block stride must be integral");
    // Every caller's base is zero or a multiple of N=4096. NT advances
    // complete quantization blocks, leaving the intra-block offset fixed.
    const int64_t col0 = base + tid;
    const int offset = col0 % QK8_1_MMQ;
    y += (col0/QK8_1_MMQ)*rows + row;
    const int64_t stride = (NT/QK8_1_MMQ)*rows;
#pragma unroll
    for(int i=0;i<N/NT;++i) {
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        y[i*stride].qs[offset]=q;
        if(lane==0) y[i*stride].d4[offset/32]=1.0f/d_inv;
    }
}

// C6a: preserve standalone FWHT butterfly and MMQ scale/rounding; only stores differ.
// SWIGLU: the input is a SwiGLU no other node reads. Each element is formed with the expression of
// unary_gated_op_kernel<op_silu, T> (x activated, g the multiplier, row strides o0/o1 in elements),
// so it is the value that kernel would have stored and the plain variant would load here.
template <typename T, bool SIGNED, bool SWIGLU>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
                                         int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
#pragma unroll
        for(int i=0;i<NE;++i) {
            if constexpr(SWIGLU) {
                const int64_t col=base+i*NT+tid;
                v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+col]))*float(g[row*o1+col])))*(1.0f/64.0f);
            } else {
                v[i]=float(x[row*k+base+i*NT+tid])*(1.0f/64.0f);
            }
            if constexpr(SIGNED) v[i]*=signs[base+i*NT+tid];
        }
        ggml_cuda_fwht_register_transpose_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
    fwht_quantize_mmq_d4_store<N,NT>(v,y,base,row,rows,tid,lane);
    GGML_UNUSED(kp); GGML_UNUSED(g); GGML_UNUSED(o0); GGML_UNUSED(o1);
}
void fwht_quantize_mmq_d4_cuda(const void * x, ggml_type type, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, cudaStream_t stream, const void * g, int64_t o0, int64_t o1) {
    GGML_ASSERT(k%4096==0 && kp%4096==0 && kp>=k);
    const dim3 grid(kp/4096,rows,1);
    const ggml_cuda_kernel_launch_params lp(grid,dim3(256,1,1),0,stream);
#define C6_LAUNCH(T, GLU) do { if(signs) { ggml_cuda_kernel_launch(fwht_quantize_mmq_d4<T,true,GLU>,lp,(const T*)x,(const T*)g,signs,(block_q8_1_mmq*)y,k,kp,rows,o0,o1); } else { ggml_cuda_kernel_launch(fwht_quantize_mmq_d4<T,false,GLU>,lp,(const T*)x,(const T*)g,signs,(block_q8_1_mmq*)y,k,kp,rows,o0,o1); } } while(0)
#define C6_LAUNCH_T(T) do { if(g) { C6_LAUNCH(T, true); } else { C6_LAUNCH(T, false); } } while(0)
    if(type==GGML_TYPE_F32) { C6_LAUNCH_T(float); } else { GGML_ASSERT(type==GGML_TYPE_F16); C6_LAUNCH_T(half); }
#undef C6_LAUNCH_T
#undef C6_LAUNCH
}

// C56: the transform input is SWIGLU(x, g) read through a permuted view of it (the GDN out-projection's
// head-permute CONT, which is then not materialized). Transform column col = i0 + n0*(i1 + n1*i2) holds the
// GLU element at offset i0 + s1*i1 + s2*i2 of its row; each element is formed with unary_gated_op_kernel
// <op_silu, T>'s expression (x activated, g the multiplier, row strides o0/o1 in elements), so it is the
// value that kernel and the CONT would have stored. The butterfly and the quantization are
// fwht_quantize_mmq_d4's.
template <typename T, bool SIGNED>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_glu_perm(const T * x, const T * g, const float * signs, block_q8_1_mmq * y,
        int64_t k, int64_t rows, int64_t o0, int64_t o1, int n0, int n1, int64_t s1, int64_t s2) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
#pragma unroll
        for(int i=0;i<NE;++i) {
            const int64_t col=base+i*NT+tid;
            const int c=int(col), q=c/n0, i0=c-q*n0, i2=q/n1, i1=q-i2*n1;
            const int64_t src=i0+s1*i1+s2*i2;
            v[i]=float(T(ggml_cuda_op_silu_single(float(x[row*o0+src]))*float(g[row*o1+src])))*(1.0f/64.0f);
            if constexpr(SIGNED) v[i]*=signs[col];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int64_t col=base+i*NT+tid;
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

void fwht_quantize_mmq_d4_glu_perm_cuda(const float * x, const float * g, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, int64_t o0, int64_t o1, int n0, int n1, int64_t s1, int64_t s2,
    cudaStream_t stream) {
    GGML_ASSERT(k%4096==0 && kp%4096==0 && kp>=k && n0>0 && n1>0);
    const dim3 grid(kp/4096,rows,1);
    const ggml_cuda_kernel_launch_params lp(grid,dim3(256,1,1),0,stream);
    if (signs) {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_glu_perm<float,true>,lp,x,g,signs,(block_q8_1_mmq*)y,k,rows,o0,o1,n0,n1,s1,s2);
    } else {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_glu_perm<float,false>,lp,x,g,signs,(block_q8_1_mmq*)y,k,rows,o0,o1,n0,n1,s1,s2);
    }
}

// RMS_NORM * w in front of the signed transform, one block per 4096-wide row. The sum of squares is
// formed in the order of rms_norm_f32<1024, true>: its thread t accumulates columns t, t+1024, t+2048,
// t+3072, here v[j], v[j+4], v[j+8], v[j+12] of thread t%256 (j = t/256), followed by the same xor
// warp reduction, per-warp partials s_sum[t/32] and final warp reduction; mean, scale and the product
// use its expressions. norm, when set, receives the product for the norm's other readers.
template <bool SIGNED>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_rms(const float * x, const int64_t stride_row, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, int64_t rows) {
    constexpr int N=4096, NT=256, NE=N/NT, NR=1024, NJ=NR/NT;
    __shared__ float shared[N];
    __shared__ float s_sum[NR/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    x+=row*stride_row;
#pragma unroll
    for(int i=0;i<NE;++i) v[i]=x[i*NT+tid];
#pragma unroll
    for(int j=0;j<NJ;++j) {
        float tmp=0.0f;
#pragma unroll
        for(int i=j;i<NE;i+=NJ) tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp;
    }
    __syncthreads();
    const float tmp=warp_reduce_sum(s_sum[lane]);
    const float mean=tmp/ncols;
    const float scale=rsqrtf(mean+eps);
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int col=i*NT+tid;
        const float xn=scale*v[i]*w[col];
        if(norm) norm[row*N+col]=xn;
        v[i]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[col];
    }
    ggml_cuda_fwht_block_butterfly<N,NT,true>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
// GDN output projection input, one block per 4096-wide row (token): per-head RMS_NORM(x) * w over 128-wide heads
// (as rms_norm_f32<128, true>), SWIGLU(z, normed) (as unary_gated_op_kernel<op_silu, float>) and the CONT of
// the permuted view that regroups the heads, formed per element: feature f = a0 + 128*(a1 + ne1*a2) of row r reads
// head row r*g3 + a1*g1 + a2*g2, column a0. Then the signed transform as fwht_quantize_mmq_d4. Register i of the
// threads of half tid/128 holds one head, column tid%128 being thread tid%128 of that head's rms block: the same
// square, xor warp reduction, four per-warp partials and final warp reduction (zeros above lane 3).
template <bool SIGNED>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_gnorm(const float * x, const float * z, const int64_t z_stride,
        const float * w, const int ncols, const float eps, const uint3 ne1, const int64_t g1, const int64_t g2,
        const int64_t g3, const float * signs, block_q8_1_mmq * y, const int64_t rows) {
    constexpr int N=4096, NT=256, NE=N/NT, HD=128, HW=HD/WARP_SIZE;
    __shared__ float shared[N];
    __shared__ float s_sum[NE][NT/WARP_SIZE];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE, a0=tid%HD;
    const int64_t row=blockIdx.y;
    float v[NE];
    int64_t head[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#pragma unroll
    for(int i=0;i<NE;++i) {
        const uint2 a21=fast_div_modulo((uint32_t)(i*(NT/HD)+tid/HD), ne1);
        head[i]=row*g3+(int64_t)a21.y*g1+(int64_t)a21.x*g2;
        v[i]=x[head[i]*HD+a0];
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        tmp+=v[i]*v[i];
        tmp=warp_reduce_sum(tmp);
        if(lane==0) s_sum[i][warp]=tmp;
    }
    __syncthreads();
#pragma unroll
    for(int i=0;i<NE;++i) {
        float tmp=0.0f;
        if(lane<HW) tmp=s_sum[i][(warp/HW)*HW+lane];
        tmp=warp_reduce_sum(tmp);
        const float mean=tmp/ncols;
        const float scale=rsqrtf(mean+eps);
        const float xn=scale*v[i]*w[a0];
        const float g=ggml_cuda_op_silu_single(z[head[i]*z_stride+a0])*xn;
        v[i]=g*(1.0f/64.0f);
        if constexpr(SIGNED) v[i]*=signs[i*NT+tid];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
void fwht_quantize_mmq_d4_gnorm_cuda(const float * x, const float * z, int64_t z_stride, const float * w, float eps,
    int64_t ne1, int64_t g1, int64_t g2, int64_t g3, const float * signs, void * y, int64_t rows, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params lp(dim3(1,rows,1),dim3(256,1,1),0,stream);
    const uint3 ne1_fd=init_fastdiv_values(ne1);
    if(signs) {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_gnorm<true>,lp,x,z,z_stride,w,128,eps,ne1_fd,g1,g2,g3,signs,(block_q8_1_mmq*)y,rows);
    } else {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_gnorm<false>,lp,x,z,z_stride,w,128,eps,ne1_fd,g1,g2,g3,signs,(block_q8_1_mmq*)y,rows);
    }
}

// Verify-batch layer input: ADD -> RMS_NORM -> MUL(w) exactly as add_rms_norm_f32<1024> forms it (its thread t
// owns columns t + 1024k, k = 0..3: __fadd_rn sum, FFMA square chain, warp sums s_sum[t/32], final warp sum,
// rsqrtf(tmp/ncols + eps), scale*x*w), then the signed transform through ggml_cuda_fwht_block_butterfly<4096, NT>
// (the ascending stage order and the expressions of every other instantiation) and the D4 quantization. Thread u
// holds columns u + NT*m in v[m] and stands for add_rms_norm's threads u + NT*j, j < 1024/NT (columns with
// m % (1024/NT) == j, in increasing m). The weights and signs are model constants, so they are loaded before the
// wait for the producing kernel. sum receives the residual; norm, when set, the product.
template <int NT, bool SIGNED>
__launch_bounds__(NT, 1)
static __global__ void fwht_quantize_mmq_d4_add_rms(const float * a, const float * b, float * sum, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, const int64_t rows) {
    constexpr int N=4096, NE=N/NT, NR=1024, R=NR/NT;
    static_assert(NT==256 || NT==512 || NT==1024, "add_rms fold block size");
    __shared__ float s_sum[NR/WARP_SIZE];
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE], wv[NE], sv[NE];
#pragma unroll
    for(int m=0;m<NE;++m) {
        wv[m]=w[m*NT+tid];
        sv[m]=SIGNED ? signs[m*NT+tid] : 1.0f;
    }
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    a+=row*N;
    b+=row*N;
    sum+=row*N;
    float tmp[R];
#pragma unroll
    for(int j=0;j<R;++j) tmp[j]=0.0f;
#pragma unroll
    for(int m=0;m<NE;++m) {
        const int col=m*NT+tid;
        const float xi=__fadd_rn(a[col], b[col]);
        sum[col]=xi;
        v[m]=xi;
        tmp[m%R]+=xi*xi;
    }
#pragma unroll
    for(int j=0;j<R;++j) {
        tmp[j]=warp_reduce_sum(tmp[j]);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp[j];
    }
    __syncthreads();
    const float total=warp_reduce_sum(s_sum[lane]);
    const float scale=rsqrtf(total/ncols+eps);
#pragma unroll
    for(int m=0;m<NE;++m) {
        const float xn=scale*v[m]*wv[m];
        if(norm) norm[row*N+m*NT+tid]=xn;
        v[m]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[m]*=sv[m];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
void fwht_quantize_mmq_d4_add_rms_cuda(const float * a, const float * b, float * sum, const float * w, float eps,
    const float * signs, float * norm, void * y, int64_t k, int64_t rows, cudaStream_t stream) {
    GGML_ASSERT(k==4096);
    constexpr int NT=1024; // measured faster in the verify chain than 512 and 256
    const ggml_cuda_kernel_launch_params lp(dim3(1,rows,1),dim3(NT,1,1),0,stream);
    if(signs) {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_add_rms<NT, true>,lp,a,b,sum,w,(int)k,eps,signs,norm,(block_q8_1_mmq*)y,rows);
    } else {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_add_rms<NT, false>,lp,a,b,sum,w,(int)k,eps,signs,norm,(block_q8_1_mmq*)y,rows);
    }
}

// Exact grid48 PTQ stream-K fixup folded into its exclusive ADD/RMS consumer.
// The producer's +0 partial accumulation, suffix addition and RN residual add
// are deliberately separate, matching the original fixup then ADD sequence.
template <int NT, bool SIGNED>
__launch_bounds__(NT, 1)
static __global__ void fwht_quantize_mmq_d4_fixup_add_rms(const float * a, const float * b, float * sum, const float * w,
        const int ncols, const float eps, const float * signs, float * norm, block_q8_1_mmq * y, const int64_t rows, const float * partials) {
    constexpr int N=4096, NE=N/NT, NR=1024, R=NR/NT;
    static_assert(NT==256 || NT==512 || NT==1024, "add_rms fold block size");
    __shared__ float s_sum[NR/WARP_SIZE];
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
    const int64_t row=blockIdx.y;
    float v[NE], wv[NE], sv[NE];
#pragma unroll
    for(int m=0;m<NE;++m) {
        wv[m]=w[m*NT+tid];
        sv[m]=SIGNED ? signs[m*NT+tid] : 1.0f;
    }
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    a+=row*N;
    b+=row*N;
    sum+=row*N;
    // NT is a multiple of 256, so advancing m increases source by
    // 3*NT/256 exactly. Hoist the thread-local base once, leaving the
    // original floating-point fixup operations and order unchanged.
    const int tile0 = tid/128;
    partials += (tile0 + tile0/2)*2048 + row*128 + tid%128;
    constexpr int PARTIAL_STRIDE = (3*NT/256)*2048;
    float tmp[R];
#pragma unroll
    for(int j=0;j<R;++j) tmp[j]=0.0f;
#pragma unroll
    for(int m=0;m<NE;++m) {
        const int col=m*NT+tid;
        // M4096 / grid48 / K4096 or K12288: one original partial per tile.
        // Keep the original fixup's +0 accumulation and suffix += partial_sum.
        float partial_sum=0.0f;
        partial_sum+=partials[m*PARTIAL_STRIDE];
        float corrected=a[col];
        corrected+=partial_sum;
        const float xi=__fadd_rn(corrected, b[col]);
        sum[col]=xi;
        v[m]=xi;
        tmp[m%R]+=xi*xi;
    }
#pragma unroll
    for(int j=0;j<R;++j) {
        tmp[j]=warp_reduce_sum(tmp[j]);
        if(lane==0) s_sum[j*(NT/WARP_SIZE)+warp]=tmp[j];
    }
    __syncthreads();
    const float total=warp_reduce_sum(s_sum[lane]);
    const float scale=rsqrtf(total/ncols+eps);
#pragma unroll
    for(int m=0;m<NE;++m) {
        const float xn=scale*v[m]*wv[m];
        if(norm) norm[row*N+m*NT+tid]=xn;
        v[m]=xn*(1.0f/64.0f);
        if constexpr(SIGNED) v[m]*=sv[m];
    }
    ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    fwht_quantize_mmq_d4_store<N,NT>(v,y,0,row,rows,tid,lane);
}
void fwht_quantize_mmq_d4_fixup_add_rms_cuda(const float * a, const float * b, float * sum,
        const float * w, float eps, const float * signs, float * norm, void * y,
        const float * partials, cudaStream_t stream) {
    constexpr int NT=1024;
    const ggml_cuda_kernel_launch_params lp(dim3(1,16,1),dim3(NT,1,1),0,stream);
    if (signs) {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_fixup_add_rms<NT,true>, lp,
            a,b,sum,w,4096,eps,signs,norm,(block_q8_1_mmq *)y,int64_t(16),partials);
    } else {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_fixup_add_rms<NT,false>, lp,
            a,b,sum,w,4096,eps,signs,norm,(block_q8_1_mmq *)y,int64_t(16),partials);
    }
}

void fwht_quantize_mmq_d4_rms_cuda(const float * x, int64_t stride_row, const float * w, float eps,
    const float * signs, float * norm, void * y, int64_t k, int64_t rows, cudaStream_t stream) {
    GGML_ASSERT(k==4096);
    const dim3 grid(1,rows,1);
    const ggml_cuda_kernel_launch_params lp(grid,dim3(256,1,1),0,stream);
    if(signs) {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_rms<true>,lp,x,stride_row,w,(int)k,eps,signs,norm,(block_q8_1_mmq*)y,rows);
    } else {
        ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_rms<false>,lp,x,stride_row,w,(int)k,eps,signs,norm,(block_q8_1_mmq*)y,rows);
    }
}

// C62: the attention output gate in front of the signed transform: the input is MUL(attn, SIGMOID(CONT(gate view)))
// with the gate read through its strided view of the joint Q|gate projection (the CONT is not materialized). Each
// element is formed with unary_gated_op_kernel<op_sigmoid>'s expression (the gate activated, attn the multiplier),
// so it is the value the CONT + fused sigmoid-mul kernels would have stored. Transform column col = i0 + n0*i1 of
// row r reads gate[r*gt + i1*g1 + i0] and attn[r*at + col]. Butterfly and quantization are fwht_quantize_mmq_d4's.
static __device__ __forceinline__ float fwht_gate_op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <bool SIGNED, bool STRIDED>
__launch_bounds__(256, 1)
static __global__ void fwht_quantize_mmq_d4_sigmoid_gate(const float * gate, const float * attn, const float * signs,
        block_q8_1_mmq * y, int64_t k, int64_t rows, int n0, int64_t g1, int64_t gt, int64_t at) {
    constexpr int N=4096, NT=256, NE=N/NT;
    __shared__ float shared[N];
    const int tid=threadIdx.x, lane=tid%32;
    const int64_t row=blockIdx.y, base=int64_t(blockIdx.x)*N;
    float v[NE];
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if (base<k) {
        // n0==NT: fixed in-row input and one exact g1 stride per register.
        const float * gate_base=gate;
        if constexpr (STRIDED) gate_base += row*gt+(base/NT)*g1+tid;
#pragma unroll
        for(int i=0;i<NE;++i) {
            const int64_t col=base+i*NT+tid;
            float gate_value;
            if constexpr (STRIDED) {
                gate_value=gate_base[i*g1];
            } else {
                const int c=int(col), i1=c/n0, i0=c-i1*n0;
                gate_value=gate[row*gt+i1*g1+i0];
            }
            v[i]=float(fwht_gate_op_sigmoid(float(gate_value))*float(attn[row*at+col]))*(1.0f/64.0f);
            if constexpr(SIGNED) v[i]*=signs[col];
        }
        ggml_cuda_fwht_block_butterfly<N,NT>(v,shared,tid,lane);
    } else {
#pragma unroll
        for(int i=0;i<NE;++i) v[i]=0.0f;
    }
#pragma unroll
    for(int i=0;i<NE;++i) {
        const int64_t col=base+i*NT+tid;
        const float amax=warp_reduce_max<32>(fabsf(v[i]));
        const float d_inv=127.0f/amax;
        const int8_t q=roundf(v[i]*d_inv);
        const int64_t ib=(col/QK8_1_MMQ)*rows+row;
        y[ib].qs[col%QK8_1_MMQ]=q;
        if(lane==0) y[ib].d4[(col%QK8_1_MMQ)/32]=1.0f/d_inv;
    }
}

void fwht_quantize_mmq_d4_sigmoid_gate_cuda(const float * gate, const float * attn, const float * signs, void * y,
    int64_t k, int64_t kp, int64_t rows, int n0, int64_t g1, int64_t gt, int64_t at, cudaStream_t stream) {
    GGML_ASSERT(k%4096==0 && kp%4096==0 && kp>=k && n0>0);
    const dim3 grid(kp/4096,rows,1);
    const ggml_cuda_kernel_launch_params lp(grid,dim3(256,1,1),0,stream);
#define SIGMOID_STRIDE_LAUNCH(S) do { if (signs) { \
    ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_sigmoid_gate<true,S>,lp,gate,attn,signs,(block_q8_1_mmq*)y,k,rows,n0,g1,gt,at); \
    } else { \
    ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_sigmoid_gate<false,S>,lp,gate,attn,signs,(block_q8_1_mmq*)y,k,rows,n0,g1,gt,at); \
    } } while (0)
    if (kp<=INT32_MAX && n0==256) { SIGMOID_STRIDE_LAUNCH(true); }
    else { SIGMOID_STRIDE_LAUNCH(false); }
#undef SIGMOID_STRIDE_LAUNCH
}
