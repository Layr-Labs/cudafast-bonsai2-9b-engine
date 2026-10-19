#include "common.cuh"

// In-register block FWHT over N floats held as NE = N/NT per thread, element i*NT + tid in reg[i].
// Shared by the standalone kernel (fwht_cuda_block) and the fused FWHT + q8_1 quantizer so both
// produce bit-identical transforms. `s` must hold N floats.
template <int N, int NT>
__device__ __forceinline__ void ggml_cuda_fwht_block_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

template <int N, int NT>
__device__ __forceinline__ void ggml_cuda_fwht_register_transpose_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    static_assert(N == 4096 && NT == 256, "register transpose geometry");
    constexpr int NW = NT / warp_size;
    const int warp = tid / warp_size;
#pragma unroll
    for (int j = 0; j < NE; ++j) s[j*NT + tid] = reg[j];
    __syncthreads();
    // In transposed ownership, register bit [0..2] is the old warp
    // index, while current warp selects old register bit [0..2].
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        reg[j] = s[(warp + NW*(j/NW))*NT + (j%NW)*warp_size + lane];
    }
    __syncthreads(); // all original shared reads finish before overwrite
#pragma unroll
    for (int step = 1; step < NW; step *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j+k], y = reg[j+k+step];
                reg[j+k] = x + y;
                reg[j+k+step] = x - y;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        s[(warp + NW*(j/NW))*NT + (j%NW)*warp_size + lane] = reg[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) reg[j] = s[j*NT + tid];
    // Scratch is dead after this helper in its one selected caller.

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Terminal FWHT for a square 32-warp x 32-lane block. The caller must not
// reuse scratch after return without a block barrier: the final read is unbarriered.
template <int N, int NT>
__device__ __forceinline__ void ggml_cuda_fwht_warp_lane_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    static_assert(N == 4096 && NT == 1024, "warp/lane transpose geometry");
    static_assert(warp_size == 32 && NT == warp_size * warp_size, "square warp/lane geometry required");
    const int warp = tid / warp_size;
    // XOR swizzle gives unique banks for both row and column traversals.
#pragma unroll
    for (int j = 0; j < NE; ++j) s[j*NT + warp*warp_size + (lane ^ warp)] = reg[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) reg[j] = s[j*NT + lane*warp_size + (warp ^ lane)];
    __syncthreads();
    // Lanes now own old warp bits, so original h32..512 stages become
    // shuffle h1..16 in the same order and with the same lower/upper values.
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int j = 0; j < NE; ++j) s[j*NT + lane*warp_size + (warp ^ lane)] = reg[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NE; ++j) reg[j] = s[j*NT + warp*warp_size + (lane ^ warp)];
    // Scratch is dead in the selected terminal caller.

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);
