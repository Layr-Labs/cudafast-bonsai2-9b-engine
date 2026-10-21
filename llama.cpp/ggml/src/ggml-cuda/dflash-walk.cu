#include "dflash-walk.cuh"

// One warp runs the small-selector walk: n_steps dependent argmaxes over top_k scores each. It keeps
// std::max_element's rule (replace only on a strict "best < candidate", so the first maximum wins and a
// NaN never replaces the best) and the host walk's float -> int32 truncation, so every id equals the one
// common/speculative.cpp would pick from the same lattice.
static __global__ void dflash_walk_f32(const char * lattice, const int64_t nb1, const int top_k, const int row0,
                                       const int n_steps, int32_t * out) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (top_k >= 8 && top_k <= 32) {
        const int lane = threadIdx.x;
        int predecessor = 0;
        for (int i = 0; i < n_steps; ++i) {
            const float * row = (const float *) (lattice + (int64_t) (row0 + i)*nb1);
            const float * scores = row + top_k + (int64_t) predecessor*top_k;
            const float value = lane < top_k ? scores[lane] : 0.0f;
            uint32_t bits = __float_as_uint(value);
            const uint32_t magnitude = bits & 0x7fffffffu;
            const bool unusual = lane < top_k &&
                ((magnitude != 0 && magnitude < 0x00800000u) || magnitude > 0x7f800000u);
            if (__any_sync(0xffffffffu, unusual)) {
                // Reuse the exact comparator for NaNs and subnormals, including
                // its compilation mode's flush-to-zero behavior and first NaN.
                int best = 0;
                if (lane == 0) {
                    float best_v = scores[0];
                    for (int k = 1; k < top_k; ++k) {
                        const float v = scores[k];
                        if (best_v < v) {
                            best_v = v;
                            best = k;
                        }
                    }
                }
                predecessor = __shfl_sync(0xffffffffu, best, 0);
            } else {
                // Normal/Inf numeric order, with both zero signs tied.
                bits = bits == 0x80000000u ? 0u : bits;
                const uint32_t key = lane < top_k ?
                    ((bits & 0x80000000u) ? ~bits : bits | 0x80000000u) : 0u;
                const uint32_t largest = __reduce_max_sync(0xffffffffu, key);
                const uint32_t first = __reduce_min_sync(0xffffffffu,
                    lane < top_k && key == largest ? (uint32_t) lane : 0xffffffffu);
                predecessor = (int) first;
            }
            if (lane == 0) {
                out[i] = (int32_t) row[predecessor];
            }
        }
        return;
    }
#endif
    // Tiny/larger selectors and other architectures retain the original scalar scan.
    if (threadIdx.x != 0) {
        return;
    }
    int predecessor = 0;
    for (int i = 0; i < n_steps; ++i) {
        const float * row    = (const float *) (lattice + (int64_t) (row0 + i)*nb1);
        const float * scores = row + top_k + (int64_t) predecessor*top_k;
        int   best   = 0;
        float best_v = scores[0];
        for (int k = 1; k < top_k; ++k) {
            const float v = scores[k];
            if (best_v < v) {
                best_v = v;
                best   = k;
            }
        }
        predecessor = best;
        out[i] = (int32_t) row[predecessor];
    }
}

void ggml_cuda_op_dflash_walk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32);

    const int top_k   = ggml_get_op_params_i32(dst, 0);
    const int row0    = ggml_get_op_params_i32(dst, 1);
    const int n_steps = ggml_get_op_params_i32(dst, 2);

    const ggml_cuda_kernel_launch_params launch_params(dim3(1, 1, 1), dim3(WARP_SIZE, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(dflash_walk_f32, launch_params, (const char *) src0->data, (int64_t) src0->nb[1], top_k, row0,
                            n_steps, (int32_t *) dst->data);
}
